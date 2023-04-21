""" Implemented by Simon Mathis, svm34@cam.ac.uk """

from __future__ import annotations

import math
from abc import ABC
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
import torch_geometric
import torch_scatter

s_V = Tuple[torch.Tensor, torch.Tensor]

# GVP GNN base model along the lines of
# https://github.com/drorlab/gvp-pytorch/blob/82af6b22eaf8311c15733117b0071408d24ed877/gvp/atom3d.py#L115

_DEFAULT_V_DIM = (100, 16)
_DEFAULT_E_DIM = (32, 1)

class RES_GVP(nn.Module):
    def __init__(self, example, dropout, **model_args):
        super().__init__()
        ns, _ = _DEFAULT_V_DIM
        self.gvp = GVPNetwork.init_from_example(example, **model_args)
        self.dense = nn.Sequential(
            nn.Linear(ns, 2*ns), nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(2*ns, 20)
        )   
    
    def forward(self, graph):
        out = self.gvp(graph, scatter_mean=False)
        out = self.dense(out)
        return out[graph.ca_idx + graph.ptr[:-1]]

# Relevant papers:
# Learning from Protein Structure with Geometric Vector Perceptrons,
# Equivariant Graph Neural Networks for 3D Macromolecular Structure,
class GVP(nn.Module):
    def __init__(
        self,
        in_dims: Tuple[int, int],
        out_dims: Tuple[int, int],
        activations=(F.relu, torch.sigmoid),
        vector_gate: bool = False,
        eps: float = 1e-4,
        device = torch.device('cpu'),
    ) -> None:
        super().__init__()
        in_scalar, in_vector = in_dims
        out_scalar, out_vector = out_dims
        self.sigma, self.sigma_plus = activations

        if self.sigma is None:
            self.sigma = nn.Identity()
        if self.sigma_plus is None:
            self.sigma_plus = nn.Identity()

        self.h = max(in_vector, out_vector)
        self.W_h = nn.Parameter(torch.empty((self.h, in_vector), device=device))
        self.W_mu = nn.Parameter(torch.empty((out_vector, self.h), device=device))

        self.W_m = nn.Linear(self.h + in_scalar, out_scalar)
        self.v = in_vector
        self.mu = out_vector
        self.n = in_scalar
        self.m = out_scalar
        self.vector_gate = vector_gate

        if vector_gate:
            self.sigma_g = nn.Sigmoid()
            self.W_g = nn.Linear(out_scalar, out_vector)

        self.eps = eps
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.W_h, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.W_mu, a=math.sqrt(5))
        self.W_m.reset_parameters()
        if self.vector_gate:
            self.W_g.reset_parameters()

    def forward(self, x: Union[torch.Tensor, s_V]) -> Union[torch.Tensor, s_V]:
        """Geometric vector perceptron"""
        s, V = (
            x if self.v > 0 else (x, torch.empty((x.shape[0], 0, 3), device=x.device))
        )

        assert (
            s.shape[-1] == self.n
        ), f"{s.shape[-1]} != {self.n} Scalar dimension mismatch"
        assert (
            V.shape[-2] == self.v
        ), f" {V.shape[-2]} != {self.v} Vector dimension mismatch"
        assert V.shape[0] == s.shape[0], "Batch size mismatch"

        V_h = self.W_h @ V
        V_mu = self.W_mu @ V_h
        s_h = torch.clip(torch.norm(V_h, dim=-1), min=self.eps)
        s_hn = torch.cat([s, s_h], dim=-1)
        s_m = self.W_m(s_hn)
        s_dash = self.sigma(s_m)
        if self.vector_gate:
            V_dash = self.sigma_g(self.W_g(self.sigma_plus(s_m)))[..., None] * V_mu
        else:
            v_mu = torch.clip(torch.norm(V_mu, dim=-1, keepdim=True), min=self.eps)
            V_dash = self.sigma_plus(v_mu) * V_mu
        return (s_dash, V_dash) if self.mu > 0 else s_dash


class GVPDropout(nn.Module):
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.dropout_features = nn.Dropout(p)
        self.dropout_vector = nn.Dropout1d(p)

    def forward(self, x: Union[torch.Tensor, s_V]) -> Union[torch.Tensor, s_V]:
        if isinstance(x, torch.Tensor):
            return self.dropout_features(x)

        s, V = x
        s = self.dropout_features(s)
        V = self.dropout_vector(V)
        return s, V


class GVPLayerNorm(nn.Module):
    def __init__(self, dims: Tuple[int, int], eps: float = 0.00001) -> None:
        super().__init__()
        self.eps = math.sqrt(eps)
        self.scalar_size, self.vector_size = dims
        self.feature_layer_norm = nn.LayerNorm(self.scalar_size, eps=eps)

    def forward(self, x: Union[torch.Tensor, s_V]) -> Union[torch.Tensor, s_V]:
        if self.vector_size == 0:
            return self.feature_layer_norm(x)

        s, V = x
        s = self.feature_layer_norm(s)
        norm = torch.clip(
            torch.linalg.vector_norm(V, dim=(-1, -2), keepdim=True)
            / math.sqrt(self.vector_size),
            min=self.eps,
        )

        V = V / norm
        return s, V


class GVPMessagePassing(MessagePassing, ABC):
    def __init__(
        self,
        in_dims: Tuple[int, int],
        out_dims: Tuple[int, int],
        edge_dims: Tuple[int, int],
        hidden_dims: Optional[Tuple[int, int]] = None,
        activations=(F.relu, torch.sigmoid),
        vector_gate: bool = False,
        attention: bool = True,
        aggr: str = "add",
        normalization_factor: float = 1.0,
        device = torch.device('cpu'),
    ):
        super().__init__(aggr)
        if hidden_dims is None:
            hidden_dims = out_dims

        in_scalar, in_vector = in_dims
        hidden_scalar, hidden_vector = hidden_dims

        edge_scalar, edge_vector = edge_dims

        self.out_scalar, self.out_vector = out_dims
        self.in_vector = in_vector
        self.hidden_scalar = hidden_scalar
        self.hidden_vector = hidden_vector
        self.normalization_factor = normalization_factor

        GVP_ = partial(GVP, activations=activations, vector_gate=vector_gate)
        self.edge_gvps = nn.Sequential(
            GVP_(
                (2 * in_scalar + edge_scalar, 2 * in_vector + edge_vector),
                hidden_dims, device=device,
            ),
            GVP_(hidden_dims, hidden_dims, device=device),
            GVP_(hidden_dims, out_dims, activations=(None, None), device=device),
        )

        self.attention = attention
        if attention:
            self.attention_gvp = GVP_(
                out_dims,
                (1, 0),
                activations=(torch.sigmoid, None),
                device=device,
            )

    def forward(self, x: s_V, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> s_V:
        s, V = x
        v_dim = V.shape[-1]
        V = torch.flatten(V, start_dim=-2, end_dim=-1)
        return self.propagate(edge_index, s=s, V=V, edge_attr=edge_attr, v_dim=v_dim)

    def message(self, s_i, s_j, V_i, V_j, edge_attr, v_dim):
        V_i = V_i.view(*V_i.shape[:-1], self.in_vector, v_dim)
        V_j = V_j.view(*V_j.shape[:-1], self.in_vector, v_dim)
        edge_scalar, edge_vector = edge_attr

        s = torch.cat([s_i, s_j, edge_scalar], dim=-1)
        V = torch.cat([V_i, V_j, edge_vector], dim=-2)
        s, V = self.edge_gvps((s, V))

        if self.attention:
            att = self.attention_gvp((s, V))
            s, V = att * s, att[..., None] * V
        return self._combine(s, V)

    def update(self, aggr_out: torch.Tensor) -> s_V:
        s_aggr, V_aggr = self._split(aggr_out, self.out_scalar, self.out_vector)
        if self.aggr == "add" or self.aggr == "sum":
            s_aggr = s_aggr / self.normalization_factor
            V_aggr = V_aggr / self.normalization_factor
        return s_aggr, V_aggr

    def _combine(self, s, V) -> torch.Tensor:
        V = torch.flatten(V, start_dim=-2, end_dim=-1)
        return torch.cat([s, V], dim=-1)

    def _split(self, s_V: torch.Tensor, scalar: int, vector: int) -> s_V:
        s = s_V[..., :scalar]
        V = s_V[..., scalar:]
        V = V.view(*V.shape[:-1], vector, -1)
        return s, V

    def reset_parameters(self):
        for gvp in self.edge_gvps:
            gvp.reset_parameters()
        if self.attention:
            self.attention_gvp.reset_parameters()


class GVPConvLayer(GVPMessagePassing, ABC):
    def __init__(
        self,
        node_dims,
        edge_dims,
        drop_rate=0.1,
        activations=(F.relu, torch.sigmoid),
        vector_gate=False,
        residual=True,
        attention=True,
        aggr: str = "add",
        normalization_factor: float = 1.0,
        device = torch.device('cpu'),
    ):
        super().__init__(
            node_dims,
            node_dims,
            edge_dims,
            hidden_dims=node_dims,
            activations=activations,
            vector_gate=vector_gate,
            attention=attention,
            aggr=aggr,
            normalization_factor=normalization_factor,
            device=device,
        )
        self.residual = residual
        self.drop_rate = drop_rate
        GVP_ = partial(GVP, activations=activations, vector_gate=vector_gate)
        self.norm = nn.ModuleList([GVPLayerNorm(node_dims) for _ in range(2)])
        self.dropout = nn.ModuleList([GVPDropout(drop_rate) for _ in range(2)])

        self.ff_func = nn.Sequential(
            GVP_(node_dims, node_dims, device=device),
            GVP_(node_dims, node_dims, activations=(None, None), device=device),
        )
        self.residual = residual

    def forward(
        self,
        x: Union[s_V, torch.Tensor],
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> s_V:

        s, V = super().forward(x, edge_index, edge_attr)
        if self.residual:
            s, V = self.dropout[0]((s, V))
            s, V = x[0] + s, x[1] + V
            s, V = self.norm[0]((s, V))

        x = (s, V)
        s, V = self.ff_func(x)

        if self.residual:
            s, V = self.dropout[1]((s, V))
            s, V = s + x[0], V + x[1]
            s, V = self.norm[1]((s, V))

        return s, V


class GVPNetwork(nn.Module):
    def __init__(
        self,
        in_dims: Tuple[int, int],
        hidden_dims: Tuple[int, int],
        in_edge_dims: Tuple[int, int],
        n_node_types: int,
        n_layers: int = 3,
        attention: bool = False,
        normalization_factor: float = 100.0,
        aggr: str = "add",
        activations=(F.silu, None),
        vector_gate: bool = True,
        eps=1e-4,
        device = torch.device('cpu'),
    ) -> None:
        super().__init__()
        hidden_edge_dims = _DEFAULT_E_DIM

        self.eps = eps
        self.embed = nn.Embedding(n_node_types, n_node_types)

        in_dims = (in_dims[0] + n_node_types, in_dims[1])
        self.embedding_in = nn.Sequential(
            GVPLayerNorm(in_dims),
            GVP(
                in_dims, hidden_dims, activations=(None, None), vector_gate=vector_gate, 
                device=device
            ),
        )

        out_dims, _ = hidden_dims
        self.embedding_out = nn.Sequential(
            GVPLayerNorm(hidden_dims),
            GVP(
                hidden_dims, (out_dims, 0), activations=activations, vector_gate=vector_gate,
                device=device,
            ),
        )
        self.edge_embedding = nn.Sequential(
            GVPLayerNorm(in_edge_dims),
            GVP(
                in_edge_dims,
                hidden_edge_dims,
                activations=(None, None),
                vector_gate=vector_gate,
                device=device,
            ),
        )

        self.layers = nn.ModuleList(
            [
                GVPConvLayer(
                    hidden_dims,
                    hidden_edge_dims,
                    activations=activations,
                    vector_gate=vector_gate,
                    residual=True,
                    attention=attention,
                    aggr=aggr,
                    normalization_factor=normalization_factor,
                    device=device
                )
                for _ in range(n_layers)
            ]
        )

    @classmethod
    def init_from_example(
        cls, x: torch_geometric.data.Data | torch_geometric.data.batch.Batch, **kwargs
    ):
        n_in_node_feats = (
            x.node_s.shape[-1] if "node_s" in x else 0,
            x.node_v.shape[-2] if "node_v" in x else 0,
        )
        n_in_edge_feats = (
            x.edge_s.shape[-1] if "edge_s" in x else 0,
            x.edge_v.shape[-2] if "edge_v" in x else 0,
        )
        if "n_node_types" not in kwargs.keys():
            kwargs["n_node_types"] = x.node_type.max().item() + 1
            # _logger.warning(
            #     "n_node_types not specified, using number of node types in data: %d",
            #     kwargs["n_node_types"],
            # )
        return cls(in_dims=n_in_node_feats, hidden_dims=_DEFAULT_V_DIM, in_edge_dims=n_in_edge_feats, **kwargs)

    def forward(self,
        batch: torch_geometric.data.Data | torch_geometric.data.batch.Batch,
        scatter_mean: bool = True,
        dense: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass of the GVP-GNN.
        Args:
            batch (torch_geometric.data.Data | torch_geometric.data.batch.Batch):
                A batch of graphs, each of type `torch_geometric.data.Data` with
                node and edge features as returned from `AtomGraphBuilder` or `ResidueGraphBuilder`.
            scatter_mean (bool, optional):
                if `True`, returns the mean of final node embeddings (for each graph), else, returns
                the embedding. Defaults to True.
            dense (bool, optional): if `True`, applies final dense layer to reduce embedding
                      to a single scalar; else, returns the embedding. Defaults to False.
        Returns:
            torch.Tensor: _description_
        """
        if isinstance(batch, torch_geometric.data.Data):
            batch = torch_geometric.data.Batch.from_data_list([batch])
        if "node_s" in batch:
            batch.node_s = torch.cat((self.embed(batch.node_type), batch.node_s), dim=-1)
        else:
            batch.node_s = self.embed(batch.node_type)

        h_V = (batch.node_s, batch.node_v) if "node_v" in batch else batch.node_s
        h_E = (batch.edge_s, batch.edge_v)

        # edge_attr = self.get_edge_attr(edge_index, pos)
        # edge_attr = self.edge_embedding(edge_attr)
        
        h_V = self.embedding_in(h_V)
        h_E = self.edge_embedding(h_E)

        for layer in self.layers:
            h_V = layer(h_V, batch.edge_index, h_E)

        out = self.embedding_out(h_V)

        batch_id = batch.batch
        if scatter_mean:
            out = torch_scatter.scatter_mean(out, batch_id, dim=0)
        if dense:
            out = self.dense(out).squeeze(-1)

        return out