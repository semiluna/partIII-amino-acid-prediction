# Code adapted from https://github.com/Bayer-Group/eqgat/blob/5bbce9bb84fabd986f865cb3d3af8ddbb48c3d37/eqgat/models/eqgat.py"

from typing import Optional, Tuple, Callable, Union
import math

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.init import kaiming_uniform_
from torch.nn.init import zeros_

from torch_geometric.data import Batch
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import reset
from torch_scatter import scatter, scatter_mean
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter.composite import scatter_softmax

class RES_EQGATModel(nn.Module):
    def __init__(
        self,
        example,
        dropout: float,
        num_elements: int = 9,
        out_units: int = 20,
        sdim: int = 100,
        vdim: int = 16,
        n_layers: int = 5,
        r_cutoff: float = 4.5,
        num_radial: int = 32,
        model_type: str = "eqgat",
        graph_level: bool = False,
        use_norm: bool = True,
        aggr: str = "mean",
        graph_pooling: str = "mean",
        cross_ablate: bool = False,
        no_feat_attn: bool = False,
        **other_args
    ):
        super(RES_EQGATModel, self).__init__()

        self.sdim = sdim
        self.vdim = vdim
        self.depth = n_layers
        self.r_cutoff = r_cutoff
        self.num_radial = num_radial
        self.model_type = model_type.lower()
        self.graph_level = graph_level
        self.num_elements = num_elements
        self.out_units = out_units

        self.init_embedding = nn.Embedding(num_embeddings=num_elements, embedding_dim=sdim)
        
        self.gnn = EQGATGNN(
            dims=(sdim, vdim),
            depth=n_layers,
            cutoff=r_cutoff,
            num_radial=num_radial,
            use_norm=use_norm,
            basis="bessel",
            use_mlp_update=True,
            use_cross_product=not cross_ablate,
            no_feat_attn=no_feat_attn,
            vector_aggr=aggr
        )

        self.use_norm = use_norm

        if use_norm:
            self.post_norm = LayerNorm(dims=(sdim, vdim))
        else:
            self.post_norm = None
        self.post_lin = GatedEquivBlock(in_dims=(sdim, vdim),
                                        out_dims=(sdim, None),
                                        hs_dim=sdim, hv_dim=vdim,
                                        use_mlp=False)
        
        self.downstream = nn.Sequential(
            DenseLayer(sdim, sdim, activation=nn.SiLU(), bias=True),
            nn.Dropout(dropout),
            DenseLayer(sdim, out_units, bias=True)
        )

        self.graph_pooling = graph_pooling
        self.apply(reset)
    
    def forward(self, data: Batch) -> Tensor:
        s, pos, batch = data.node_type, data.x, data.batch
        edge_index, d = data.edge_index, data.edge_weights

        s = self.init_embedding(s)

        row, col = edge_index
        rel_pos = pos[row] - pos[col]
        rel_pos = F.normalize(rel_pos, dim=-1, eps=1e-6)
        edge_attr = d, rel_pos

        v = torch.zeros(size=[s.size(0), 3, self.vdim], device=s.device)
        s, v = self.gnn(x=(s, v), edge_index=edge_index, edge_attr=edge_attr, batch=batch)
        if self.use_norm:
            s, v = self.post_norm(x=(s, v), batch=batch)
        s, _ = self.post_lin(x=(s, v))

        if self.graph_level:
            y_pred = scatter(s, index=batch, dim=0, reduce=self.graph_pooling)
        else:
            y_pred = s

        subset_idx = data.ca_idx + data.ptr[:-1]
        y_pred = y_pred[subset_idx]

        y_pred = self.downstream(y_pred)

        return y_pred


class EQGATGNN(nn.Module):
    def __init__(
        self,
        dims: Tuple[int, int] = (128, 32),
        depth: int = 5,
        eps: float = 1e-6,
        cutoff: Optional[float] = 5.0,
        num_radial: Optional[int] = 32,
        use_norm: bool = False,
        basis: str = "bessel",
        use_mlp_update: bool = True,
        use_cross_product: bool = True,
        no_feat_attn: bool = False,
        vector_aggr: str = "mean"
    ):
        super(EQGATGNN, self).__init__()
        self.dims = dims
        self.depth = depth
        self.use_norm = use_norm
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.use_cross_product = use_cross_product
        self.vector_aggr = vector_aggr

        if use_cross_product:
            print("Using Cross Product")
            if no_feat_attn:
                print("Without Feature Attention")
                module = EQGATNoFeatAttnConv
            else:
                print("With Feature Attention")
                module = EQGATConv
        else:
            print("Using No Cross Product with Feature Attention")
            module = EQGATConvNoCross

        for i in range(depth):
            self.convs.append(
                module(
                    in_dims=dims,
                    has_v_in=i > 0,
                    out_dims=dims,
                    cutoff=cutoff,
                    num_radial=num_radial,
                    eps=eps,
                    basis=basis,
                    use_mlp_update=use_mlp_update,
                    vector_aggr=vector_aggr
                )
            )
            if use_norm:
                self.norms.append(
                    LayerNorm(dims=dims, affine=True)
                )
        self.apply(fn=reset)

    def forward(
        self,
        x: Tuple[Tensor, Tensor],
        edge_index: Tensor,
        edge_attr: Tuple[Tensor, Tensor],
        batch: Tensor
    ) -> Tuple[Tensor, Tensor]:

        s, v = x
        for i in range(len(self.convs)):
            s, v = self.convs[i](x=(s, v), edge_index=edge_index, edge_attr=edge_attr)
            if self.use_norm:
                s, v = self.norms[i](x=(s, v), batch=batch)
        return s, v



def degree_normalization(edge_index,
                         num_nodes: Optional[int] = None,
                         flow: str = "source_to_target") -> Tensor:

    if flow not in ["source_to_target", "target_to_source"]:
        print(f"Wrong selected flow {flow}.")
        print("Only 'source_to_target', or 'target_to_source' is possible")
        print("Exiting code")
        exit()

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    edge_weight = torch.ones((edge_index.size(1),), dtype=torch.get_default_dtype(),
                             device=edge_index.device)

    row, col = edge_index[0], edge_index[1]
    idx = col if flow == "source_to_target" else row
    deg = scatter(edge_weight, idx, dim=0, dim_size=num_nodes, reduce="add")
    deg = torch.clamp(deg, min=1.0)
    deg_inv_sqrt = deg.pow_(-0.5)
    norm = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    return norm


def sqrt_normalization(edge_index,
                       num_nodes: Optional[int] = None,
                       flow: str = "source_to_target") -> Tensor:

    if flow not in ["source_to_target", "target_to_source"]:
        print(f"Wrong selected flow {flow}.")
        print("Only 'source_to_target', or 'target_to_source' is possible")
        print("Exiting code")
        exit()

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    edge_weight = torch.ones((edge_index.size(1),), dtype=torch.get_default_dtype(),
                             device=edge_index.device)

    row, col = edge_index[0], edge_index[1]
    idx = col if flow == "source_to_target" else row
    deg = scatter(edge_weight, idx, dim=0, dim_size=num_nodes, reduce="add")
    deg = torch.clamp(deg, min=1.0)
    deg_inv_sqrt = deg.pow_(-0.5)
    norm = edge_weight * deg_inv_sqrt[col]
    return norm


def scatter_normalization(
    x: Tensor,
    index: Tensor,
    dim: int = 0,
    act: Callable = nn.Softplus(),
    eps: float = 1e-6,
    dim_size: Optional[int] = None,
):
    xa = act(x)
    aggr_logits = scatter(src=xa, index=index, dim=dim, reduce="add", dim_size=dim_size)
    aggr_logits = aggr_logits[index]
    xa = xa / (aggr_logits + eps)
    return xa


class EQGATConv(MessagePassing):
    def __init__(
        self,
        in_dims: Tuple[int, Optional[int]],
        out_dims: Tuple[int, Optional[int]],
        num_radial: int,
        cutoff: float,
        eps: float = 1e-6,
        has_v_in: bool = True,
        basis: str = "bessel",
        use_mlp_update: bool = True,
        vector_aggr: str = "mean",
    ):
        super(EQGATConv, self).__init__(
            node_dim=0, aggr=None, flow="source_to_target"
        )

        self.vector_aggr = vector_aggr
        self.in_dims = in_dims
        self.si, self.vi = in_dims
        self.out_dims = out_dims
        self.so, self.vo = out_dims
        self.has_v_in = has_v_in
        if has_v_in:
            self.vector_net = DenseLayer(self.vi, self.vi, bias=False)
            self.v_mul = 3
        else:
            self.v_mul = 1
            self.vector_net = nn.Identity()

        self.cutoff = cutoff
        self.num_radial = num_radial

        if basis == "chebyshev":
            rbf = ChebyshevExpansion
        elif basis == "bessel":
            rbf = BesselExpansion
        elif basis == "gaussian":
            rbf = GaussianExpansion
        else:
            raise ValueError

        self.distance_expansion = rbf(
            max_value=cutoff,
            K=num_radial,
        )
        self.cutoff_fnc = PolynomialCutoff(cutoff, p=6)
        self.edge_net = nn.Sequential(DenseLayer(2 * self.si + self.num_radial,
                                                 self.si,
                                                 bias=True, activation=nn.SiLU()),
                                      DenseLayer(self.si, self.v_mul * self.vi + self.si,
                                                 bias=True)
                                      )
        self.scalar_net = DenseLayer(self.si, self.si, bias=True)
        self.update_net = GatedEquivBlock(in_dims=(self.si, self.vi),
                                          hs_dim=self.si, hv_dim=self.vi,
                                          out_dims=(self.si, self.vi),
                                          norm_eps=eps,
                                          use_mlp=use_mlp_update)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.edge_net)
        reset(self.scalar_net)
        reset(self.update_net)

    def forward(
        self,
        x: Tuple[Tensor, Tensor],
        edge_index: Tensor,
        edge_attr: Tuple[Tensor, Tensor],
    ):

        s, v = x
        d, r = edge_attr

        ms, mv = self.propagate(
            sa=s,
            sb=self.scalar_net(s),
            va=v,
            vb=self.vector_net(v),
            edge_attr=(d, r),
            edge_index=edge_index,
            dim_size=s.size(0),
        )

        s = ms + s
        v = mv + v

        ms, mv = self.update_net(x=(s, v))

        s = ms + s
        v = mv + v

        return s, v

    def aggregate(
        self,
            inputs: Tuple[Tensor, Tensor],
            index: Tensor,
            dim_size: Optional[int] = None
    ) -> Tuple[Tensor, Tensor]:
        s = scatter(inputs[0], index=index, dim=0, reduce="add", dim_size=dim_size)
        v = scatter(inputs[1], index=index, dim=0, reduce=self.vector_aggr, dim_size=dim_size)
        return s, v

    def message(
        self,
        sa_i: Tensor,
        sa_j: Tensor,
        sb_j: Tensor,
        va_i: Tensor,
        va_j: Tensor,
        vb_j: Tensor,
        index: Tensor,
        edge_attr: Tuple[Tensor, Tensor],
        dim_size: Optional[int]
    ) -> Tuple[Tensor, Tensor]:

        d, r = edge_attr

        de = self.distance_expansion(d)
        dc = self.cutoff_fnc(d)
        de = dc.view(-1, 1) * de

        aij = torch.cat([sa_i, sa_j, de], dim=-1)
        aij = self.edge_net(aij)

        if self.has_v_in:
            aij, vij0 = aij.split([self.si, 3*self.vi], dim=-1)
            vij0 = vij0.unsqueeze(1)
            vij0, vij1, vij2 = vij0.chunk(3, dim=-1)
        else:
            aij, vij0 = aij.split([self.si, self.vi], dim=-1)
            vij0 = vij0.unsqueeze(1)

        # feature attention
        aij = scatter_softmax(aij, index=index, dim=0, dim_size=dim_size)
        ns_j = aij * sb_j
        nv0_j = r.unsqueeze(-1) * vij0

        if self.has_v_in:
            nv1_j = vij1 * vb_j
            v_ij_cross = torch.linalg.cross(va_i, vb_j, dim=1)
            nv2_j = vij2 * v_ij_cross
            nv_j = nv0_j + nv1_j + nv2_j
        else:
            nv_j = nv0_j

        return ns_j, nv_j


class EQGATNoFeatAttnConv(MessagePassing):
    def __init__(
        self,
        in_dims: Tuple[int, Optional[int]],
        out_dims: Tuple[int, Optional[int]],
        num_radial: int,
        cutoff: float,
        eps: float = 1e-6,
        has_v_in: bool = True,
        basis: str = "bessel",
        use_mlp_update: bool = True,
        vector_aggr: str = "mean",
    ):
        super(EQGATNoFeatAttnConv, self).__init__(
            node_dim=0, aggr=None, flow="source_to_target"
        )

        self.vector_aggr = vector_aggr
        self.in_dims = in_dims
        self.si, self.vi = in_dims
        self.out_dims = out_dims
        self.so, self.vo = out_dims
        self.has_v_in = has_v_in
        if has_v_in:
            self.vector_net = DenseLayer(self.vi, self.vi, bias=False)
            self.v_mul = 3
        else:
            self.v_mul = 1
            self.vector_net = nn.Identity()

        self.cutoff = cutoff
        self.num_radial = num_radial

        if basis == "chebyshev":
            rbf = ChebyshevExpansion
        elif basis == "bessel":
            rbf = BesselExpansion
        elif basis == "gaussian":
            rbf = GaussianExpansion
        else:
            raise ValueError

        self.distance_expansion = rbf(
            max_value=cutoff,
            K=num_radial,
        )
        self.cutoff_fnc = PolynomialCutoff(cutoff, p=6)
        self.edge_net = nn.Sequential(DenseLayer(2 * self.si + self.num_radial,
                                                 self.si,
                                                 bias=True, activation=nn.SiLU()),
                                      DenseLayer(self.si, self.v_mul * self.vi + 1,
                                                 bias=True)
                                      )
        self.scalar_net = DenseLayer(self.si, self.si, bias=True)
        self.update_net = GatedEquivBlock(in_dims=(self.si, self.vi),
                                          hs_dim=self.si, hv_dim=self.vi,
                                          out_dims=(self.si, self.vi),
                                          norm_eps=eps,
                                          use_mlp=use_mlp_update)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.edge_net)
        reset(self.scalar_net)
        reset(self.update_net)

    def forward(
        self,
        x: Tuple[Tensor, Tensor],
        edge_index: Tensor,
        edge_attr: Tuple[Tensor, Tensor],
    ):

        s, v = x
        d, r = edge_attr

        ms, mv = self.propagate(
            sa=s,
            sb=self.scalar_net(s),
            va=v,
            vb=self.vector_net(v),
            edge_attr=(d, r),
            edge_index=edge_index,
            dim_size=s.size(0),
        )

        s = ms + s
        v = mv + v

        ms, mv = self.update_net(x=(s, v))

        s = ms + s
        v = mv + v

        return s, v

    def aggregate(
        self,
            inputs: Tuple[Tensor, Tensor],
            index: Tensor,
            dim_size: Optional[int] = None
    ) -> Tuple[Tensor, Tensor]:
        s = scatter(inputs[0], index=index, dim=0, reduce="add", dim_size=dim_size)
        v = scatter(inputs[1], index=index, dim=0, reduce=self.vector_aggr, dim_size=dim_size)
        return s, v

    def message(
        self,
        sa_i: Tensor,
        sa_j: Tensor,
        sb_j: Tensor,
        va_i: Tensor,
        va_j: Tensor,
        vb_j: Tensor,
        index: Tensor,
        edge_attr: Tuple[Tensor, Tensor],
        dim_size: Optional[int]
    ) -> Tuple[Tensor, Tensor]:

        d, r = edge_attr

        de = self.distance_expansion(d)
        dc = self.cutoff_fnc(d)
        de = dc.view(-1, 1) * de

        aij = torch.cat([sa_i, sa_j, de], dim=-1)
        aij = self.edge_net(aij)

        if self.has_v_in:
            aij, vij0 = aij.split([1, 3*self.vi], dim=-1)
            vij0 = vij0.unsqueeze(1)
            vij0, vij1, vij2 = vij0.chunk(3, dim=-1)
        else:
            aij, vij0 = aij.split([1, self.vi], dim=-1)
            vij0 = vij0.unsqueeze(1)

        # feature attention
        aij = scatter_softmax(aij, index=index, dim=0, dim_size=dim_size)
        ns_j = aij * sb_j
        nv0_j = r.unsqueeze(-1) * vij0

        if self.has_v_in:
            nv1_j = vij1 * vb_j
            v_ij_cross = torch.linalg.cross(va_i, vb_j, dim=1)
            nv2_j = vij2 * v_ij_cross
            nv_j = nv0_j + nv1_j + nv2_j
        else:
            nv_j = nv0_j

        return ns_j, nv_j


class EQGATConvNoCross(MessagePassing):
    def __init__(
        self,
        in_dims: Tuple[int, Optional[int]],
        out_dims: Tuple[int, Optional[int]],
        num_radial: int,
        cutoff: float,
        eps: float = 1e-6,
        has_v_in: bool = True,
        basis: str = "bessel",
        use_mlp_update: bool = True,
        vector_aggr: str = "mean",
    ):
        super(EQGATConvNoCross, self).__init__(
            node_dim=0, aggr=None, flow="source_to_target"
        )

        self.vector_aggr = vector_aggr
        self.in_dims = in_dims
        self.si, self.vi = in_dims
        self.out_dims = out_dims
        self.so, self.vo = out_dims
        self.has_v_in = has_v_in
        if has_v_in:
            self.vector_net = DenseLayer(self.vi, self.vi, bias=False)
            self.v_mul = 2
        else:
            self.v_mul = 1
            self.vector_net = nn.Identity()

        self.cutoff = cutoff
        self.num_radial = num_radial

        if basis == "chebyshev":
            rbf = ChebyshevExpansion
        elif basis == "bessel":
            rbf = BesselExpansion
        elif basis == "gaussian":
            rbf = GaussianExpansion
        else:
            raise ValueError

        self.distance_expansion = rbf(
            max_value=cutoff,
            K=num_radial,
        )
        self.cutoff_fnc = PolynomialCutoff(cutoff, p=6)
        self.edge_net = nn.Sequential(DenseLayer(2 * self.si + self.num_radial,
                                                 self.si,
                                                 bias=True, activation=nn.SiLU()),
                                      DenseLayer(self.si, self.v_mul * self.vi + self.si,
                                                 bias=True)
                                      )
        self.scalar_net = DenseLayer(self.si, self.si, bias=True)
        self.update_net = GatedEquivBlock(in_dims=(self.si, self.vi),
                                          hs_dim=self.si, hv_dim=self.vi,
                                          out_dims=(self.si, self.vi),
                                          norm_eps=eps,
                                          use_mlp=use_mlp_update)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.edge_net)
        reset(self.scalar_net)
        reset(self.update_net)

    def forward(
        self,
        x: Tuple[Tensor, Tensor],
        edge_index: Tensor,
        edge_attr: Tuple[Tensor, Tensor],
    ):

        s, v = x
        d, r = edge_attr

        ms, mv = self.propagate(
            sa=s,
            sb=self.scalar_net(s),
            vb=self.vector_net(v),
            edge_attr=(d, r),
            edge_index=edge_index,
            dim_size=s.size(0),
        )

        s = ms + s
        v = mv + v

        ms, mv = self.update_net(x=(s, v))

        s = ms + s
        v = mv + v

        return s, v

    def aggregate(
        self,
            inputs: Tuple[Tensor, Tensor],
            index: Tensor,
            dim_size: Optional[int] = None
    ) -> Tuple[Tensor, Tensor]:
        s = scatter(inputs[0], index=index, dim=0, reduce="add", dim_size=dim_size)
        v = scatter(inputs[1], index=index, dim=0, reduce=self.vector_aggr, dim_size=dim_size)
        return s, v

    def message(
        self,
        sa_i: Tensor,
        sa_j: Tensor,
        sb_j: Tensor,
        vb_j: Tensor,
        index: Tensor,
        edge_attr: Tuple[Tensor, Tensor],
        dim_size: Optional[int]
    ) -> Tuple[Tensor, Tensor]:

        d, r = edge_attr

        de = self.distance_expansion(d)
        dc = self.cutoff_fnc(d)
        de = dc.view(-1, 1) * de

        aij = torch.cat([sa_i, sa_j, de], dim=-1)
        aij = self.edge_net(aij)

        if self.has_v_in:
            aij, vij0 = aij.split([self.si, 2*self.vi], dim=-1)
            vij0 = vij0.unsqueeze(1)
            vij0, vij1 = vij0.chunk(2, dim=-1)
        else:
            aij, vij0 = aij.split([self.si, self.vi], dim=-1)
            vij0 = vij0.unsqueeze(1)

        # feature attention
        aij = scatter_softmax(aij, index=index, dim=0, dim_size=dim_size)
        ns_j = aij * sb_j
        nv0_j = r.unsqueeze(-1) * vij0

        if self.has_v_in:
            nv1_j = vij1 * vb_j
            nv_j = nv0_j + nv1_j
        else:
            nv_j = nv0_j

        return ns_j, nv_j


if __name__ == '__main__':
    sdim = 128
    vdim = 32

    module = EQGATConv(in_dims=(sdim, vdim),
                       out_dims=(sdim, vdim),
                       num_radial=32,
                       cutoff=5.0,
                       has_v_in=True)


    print(sum(m.numel() for m in module.parameters() if m.requires_grad))

    from torch_geometric.nn import radius_graph
    s = torch.randn(30, sdim, requires_grad=True)
    v = torch.randn(30, 3, vdim, requires_grad=True)
    pos = torch.empty(30, 3).normal_(mean=0.0, std=3.0)
    edge_index = radius_graph(pos, r=5.0, flow="source_to_target")
    j, i = edge_index
    p_ij = pos[j] - pos[i]
    d_ij = torch.pow(p_ij, 2).sum(-1).sqrt()
    p_ij_n = p_ij / d_ij.unsqueeze(-1)

    os, ov = module(x=(s, v),
                    edge_index=edge_index,
                    edge_attr=(d_ij, p_ij_n)
                    )

    from scipy.spatial.transform import Rotation

    Q = torch.tensor(Rotation.random().as_matrix(), dtype=torch.get_default_dtype())

    vR = torch.einsum('ij, njk -> nik', Q, v)
    p_ij_n_R = torch.einsum('ij, nj -> ni', Q, p_ij_n)

    ovR_ = torch.einsum('ij, njk -> nik', Q, ov)

    osR, ovR = module(x=(s, vR),
                      edge_index=edge_index,
                      edge_attr=(d_ij, p_ij_n_R)
                      )

    print(torch.norm(os-osR, p=2))
    print(torch.norm(ovR_-ovR, p=2))

    module = EQGATConvNoCross(in_dims=(sdim, vdim),
                              out_dims=(sdim, vdim),
                              num_radial=32,
                              cutoff=5.0,
                              has_v_in=True)

    print(sum(m.numel() for m in module.parameters() if m.requires_grad))
    os, ov = module(x=(s, v),
                    edge_index=edge_index,
                    edge_attr=(d_ij, p_ij_n)
                    )
    osR, ovR = module(x=(s, vR),
                      edge_index=edge_index,
                      edge_attr=(d_ij, p_ij_n_R)
                      )
    ovR_ = torch.einsum('ij, njk -> nik', Q, ov)

    print(torch.norm(os - osR, p=2))
    print(torch.norm(ovR_ - ovR, p=2))



class DenseLayer(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: Union[Callable, nn.Module] = None,
        weight_init: Callable = kaiming_uniform_,
        bias_init: Callable = zeros_,
    ):
        self.weight_init = weight_init
        self.bias_init = bias_init
        super(DenseLayer, self).__init__(in_features, out_features, bias)

        self.activation = activation
        if self.activation is None:
            self.activation = nn.Identity()

    def reset_parameters(self):
        # https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py#L106
        self.weight_init(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            self.bias_init(self.bias)

    def forward(self, input: torch.Tensor):
        y = F.linear(input, self.weight, self.bias)
        y = self.activation(y)
        return y


class CosineCutoff(nn.Module):
    def __init__(self, cutoff):
        super(CosineCutoff, self).__init__()
        self.cutoff = cutoff

    @staticmethod
    def cosine_cutoff(r: Tensor, rcut: float) -> Tensor:
        out = 0.5 * (torch.cos((math.pi * r) / rcut) + 1.0)
        out = out * (r < rcut).float()
        return out

    def forward(self, r):
        return self.cosine_cutoff(r, self.cutoff)

    def __repr__(self):
        return f"{self.__class__.__name__}(cutoff={self.cutoff})"


class PolynomialCutoff(nn.Module):
    def __init__(self, cutoff, p: int = 6):
        super(PolynomialCutoff, self).__init__()
        self.cutoff = cutoff
        self.p = p

    @staticmethod
    def polynomial_cutoff(
        r: Tensor,
        rcut: float,
        p: float = 6.0
    ) -> Tensor:
        """
        Polynomial cutoff, as proposed in DimeNet: https://arxiv.org/abs/2003.03123
        """
        if not p >= 2.0:
            print(f"Exponent p={p} has to be >= 2.")
            print("Exiting code.")
            exit()

        rscaled = r / rcut

        out = 1.0
        out = out - (((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(rscaled, p))
        out = out + (p * (p + 2.0) * torch.pow(rscaled, p + 1.0))
        out = out - ((p * (p + 1.0) / 2) * torch.pow(rscaled, p + 2.0))

        return out * (rscaled < 1.0).float()

    def forward(self, r):
        return self.polynomial_cutoff(r=r, rcut=self.cutoff, p=self.p)

    def __repr__(self):
        return f"{self.__class__.__name__}(cutoff={self.cutoff}, p={self.p})"


class ChebyshevExpansion(nn.Module):
    def __init__(self, max_value: float, K: int = 20):
        super(ChebyshevExpansion, self).__init__()
        self.K = K
        self.max_value = max_value
        self.shift_scale = lambda x: 2 * x / max_value - 1.0

    @staticmethod
    def chebyshev_recursion(x, n):
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        if not n > 2:
            print(f"Naural exponent n={n} has to be > 2.")
            print("Exiting code.")
            exit()

        t_n_1 = torch.ones_like(x)
        t_n = x
        ts = [t_n_1, t_n]
        for i in range(n - 2):
            t_n_new = 2 * x * t_n - t_n_1
            t_n_1 = t_n
            t_n = t_n_new
            ts.append(t_n_new)
        return torch.cat(ts, dim=-1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.shift_scale(x)
        x = self.chebyshev_recursion(x, n=self.K)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(K={self.K}, max_value={self.max_value})"


def gaussian_basis_expansion(inputs: Tensor, offsets: Tensor, widths: Tensor):
    coeff = -0.5 / torch.pow(widths, 2)
    diff = inputs[..., None] - offsets
    y = torch.exp(coeff * torch.pow(diff, 2))
    return y


class GaussianExpansion(nn.Module):

    def __init__(
        self, max_value: float, K: int, start: float = 0.0, trainable: bool = False
    ):
        super(GaussianExpansion, self).__init__()
        self.K = K

        offset = torch.linspace(start, max_value, K)
        widths = torch.FloatTensor(
            torch.abs(offset[1] - offset[0]) * torch.ones_like(offset)
        )
        if trainable:
            self.widths = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer("widths", widths)
            self.register_buffer("offsets", offset)

    def forward(self, inputs: Tensor):
        return gaussian_basis_expansion(inputs, self.offsets, self.widths)


class BesselExpansion(nn.Module):
    def __init__(
        self, max_value: float, K: int = 20
    ):
        super(BesselExpansion, self).__init__()
        self.max_value = max_value
        self.K = K
        frequency = math.pi * torch.arange(start=1, end=K + 1)
        self.register_buffer("frequency", frequency)
        self.reset_parameters()

    def reset_parameters(self):
        self.frequency.data = math.pi * torch.arange(start=1, end=self.K + 1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Bessel RBF, as proposed in DimeNet: https://arxiv.org/abs/2003.03123
        """
        ax = x.unsqueeze(-1) / self.max_value
        ax = ax * self.frequency
        sinax = torch.sin(ax)
        norm = torch.where(
            x == 0.0, torch.tensor(1.0, dtype=x.dtype, device=x.device), x
        )
        out = sinax / norm[..., None]
        out *= math.sqrt(2 / self.max_value)
        return out


class GatedEquivBlock(nn.Module):
    def __init__(
        self,
        in_dims: Tuple[int, int],
        out_dims: Tuple[int, Optional[int]],
        hs_dim: Optional[int] = None,
        hv_dim: Optional[int] = None,
        norm_eps: float = 1e-6,
        use_mlp: bool = False
    ):
        super(GatedEquivBlock, self).__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vo = 0 if self.vo is None else self.vo

        self.hs_dim = hs_dim or max(self.si, self.so)
        self.hv_dim = hv_dim or max(self.vi, self.vo)
        self.norm_eps = norm_eps

        self.use_mlp = use_mlp

        self.Wv0 = DenseLayer(self.vi, self.hv_dim + self.vo, bias=False)

        if not use_mlp:
            self.Ws = DenseLayer(self.hv_dim + self.si, self.vo + self.so, bias=True)
        else:
            self.Ws = nn.Sequential(
                DenseLayer(self.hv_dim + self.si, self.si, bias=True, activation=nn.SiLU()),
                DenseLayer(self.si, self.vo + self.so, bias=True)
            )
            self.Wv1 = DenseLayer(self.vo, self.vo, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.Ws)
        reset(self.Wv0)
        if self.use_mlp:
            reset(self.Wv1)


    def forward(self, x: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:

        s, v = x
        vv = self.Wv0(v)

        if self.vo > 0:
            vnorm, v = vv.split([self.hv_dim, self.vo], dim=-1)
        else:
            vnorm = vv

        vnorm = torch.clamp(torch.pow(vnorm, 2).sum(dim=1), min=self.norm_eps).sqrt()
        s = torch.cat([s, vnorm], dim=-1)
        s = self.Ws(s)
        if self.vo > 0:
            gate, s = s.split([self.vo, self.so], dim=-1)
            v = gate.unsqueeze(1) * v
            if self.use_mlp:
                v = self.Wv1(v)

        return s, v


class GatedEquivBlockTP(nn.Module):
    def __init__(
        self,
        in_dims: Tuple[int, int],
        out_dims: Tuple[int, Optional[int]],
        hs_dim: Optional[int] = None,
        hv_dim: Optional[int] = None,
        norm_eps: float = 1e-6,
        use_mlp: bool = False
    ):
        super(GatedEquivBlockTP, self).__init__()
        self.si, self.vi = in_dims
        self.so, self.vo = out_dims
        self.vo = 0 if self.vo is None else self.vo

        self.hs_dim = hs_dim or max(self.si, self.so)
        self.hv_dim = hv_dim or max(self.vi, self.vo)
        self.norm_eps = norm_eps

        self.use_mlp = use_mlp

        self.scalar_mixing = nn.Parameter(torch.zeros(size=(1, 1, self.si, self.vi)))
        self.vector_mixing = nn.Parameter(torch.zeros(size=(1, 1, self.si, self.vi)))

        if not use_mlp:
            self.Ws = DenseLayer(self.si, self.vo + self.so, bias=True)
        else:
            self.Ws = nn.Sequential(
                DenseLayer(self.si, self.si, bias=True, activation=nn.SiLU()),
                DenseLayer(self.si, self.vo + self.so, bias=True)
            )
            self.Wv = DenseLayer(self.vo, self.vo, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        reset(self.Ws)
        if self.use_mlp:
            reset(self.Wv)
        kaiming_uniform_(self.scalar_mixing)
        kaiming_uniform_(self.vector_mixing)

    def forward(self, x: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:

        s, v = x

        # tensor product of l=0 and l=1
        sv = torch.einsum('bk, bdm -> bdkm', s, v)

        # path for scalar features
        # mean aggregate along vector-channels
        s = torch.sum(sv * self.scalar_mixing, dim=-1)
        # make SO(3) invariant
        s = torch.pow(s, 2).sum(dim=1)
        s = torch.clamp(s, min=self.norm_eps).sqrt()
        # feed into invariant MLP / linear net
        s = self.Ws(s)

        if self.vo > 0:
            # path for vector features
            # mean aggregate along scalar-channels
            v = torch.sum(self.vector_mixing * sv, dim=-2)
            gate, s = s.split([self.vo, self.so], dim=-1)
            v = gate.unsqueeze(1) * v
            if self.use_mlp:
                v = self.Wv(v)

        return s, v


class BatchNorm(nn.Module):
    def __init__(self, dims: Tuple[int, Optional[int]], eps: float = 1e-6, affine: bool = True):
        super().__init__()

        self.dims = dims
        self.sdim, self.vdim = dims
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.Tensor(self.sdim))
            self.bias = nn.Parameter(torch.Tensor(self.sdim))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            self.weight.data.fill_(1.0)
            self.bias.data.fill_(0.0)

    def forward(self, x: Tuple[Tensor, Optional[Tensor]], batch: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        s, v = x
        batch_size = int(batch.max()) + 1

        smean = scatter_mean(s, batch, dim=0, dim_size=batch_size)

        if s.device == "cpu":
            smean = smean.index_select(0, batch)
        else:
            smean = torch.gather(smean, dim=0, index=batch.view(-1, 1))

        s = s - smean

        var = scatter_mean(s * s, batch, dim=0, dim_size=batch_size)
        var = torch.clamp(var, min=self.eps)
        prec = torch.pow(torch.sqrt(var), -1)
        if prec.device == "cpu":
            prec = prec.index_select(0, batch)
        else:
            prec = torch.gather(prec, dim=0, index=batch.view(-1, 1))

        sout = s * prec

        if self.weight is not None and self.bias is not None:
            sout = sout * self.weight + self.bias

        if v is not None:
            vmean = torch.pow(v, 2).sum(-1, keepdim=True)
            vmean = scatter_mean(vmean, batch, dim=0, dim_size=batch_size)
            vmean = torch.clamp(vmean, min=self.eps)
            if vmean.device == "cpu":
                vmean = vmean.index_select(0, batch)
            else:
                vmean = torch.gather(vmean, dim=0, index=batch.view(-1, 1, 1))

            vout = v / vmean
        else:
            vout = None

        out = sout, vout

        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}(dims={self.dims}, '
                f'affine={self.affine})')



class LayerNorm(nn.Module):
    def __init__(self, dims: Tuple[int, Optional[int]], eps: float = 1e-6, affine: bool = True):
        super().__init__()

        self.dims = dims
        self.sdim, self.vdim = dims
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.Tensor(self.sdim))
            self.bias = nn.Parameter(torch.Tensor(self.sdim))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            self.weight.data.fill_(1.0)
            self.bias.data.fill_(0.0)

    def forward(self, x: Tuple[Tensor, Optional[Tensor]], batch: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        s, v = x
        batch_size = int(batch.max()) + 1
        smean = s.mean(dim=-1, keepdim=True)
        smean = scatter_mean(smean, batch, dim=0, dim_size=batch_size)

        if s.device == "cpu":
            smean = smean.index_select(0, batch)
        else:
            smean = torch.gather(smean, dim=0, index=batch.view(-1, 1))

        s = s - smean

        var = (s * s).mean(dim=-1, keepdim=True)
        var = scatter_mean(var, batch, dim=0, dim_size=batch_size)
        var = torch.clamp(var, min=self.eps)
        prec = 1 / var

        if prec.device == "cpu":
            prec = prec.index_select(0, batch)
        else:
            prec = torch.gather(prec, dim=0, index=batch.view(-1, 1))

        sout = s * prec

        if self.weight is not None and self.bias is not None:
            sout = sout * self.weight + self.bias

        if v is not None:
            vmean = torch.pow(v, 2).sum(-1, keepdim=True).mean(dim=-1, keepdim=True)
            vmean = scatter_mean(vmean, batch, dim=0, dim_size=batch_size)
            vmean = torch.clamp(vmean, min=self.eps)
            if vmean.device == "cpu":
                vmean = vmean.index_select(0, batch)
            else:
                vmean = torch.gather(vmean, dim=0, index=batch.view(-1, 1, 1))

            vout = v / vmean
        else:
            vout = None

        out = sout, vout

        return out

    def __repr__(self):
        return (f'{self.__class__.__name__}(dims={self.dims}, '
                f'affine={self.affine})')


class ShiftedSoftPlus(nn.Module):
    def __init__(self):
        super(ShiftedSoftPlus, self).__init__()

    def forward(self, x: Tensor):
        return F.softplus(x) - math.log(2.0)



def visualize_basis_expansions():
    import matplotlib.pyplot as plt

    dist = torch.linspace(0, 5.0, 1000)

    K = 32
    gauss_rbf = GaussianExpansion(max_value=5.0, K=K)
    cheb_rbf = ChebyshevExpansion(max_value=5.0, K=K)
    bessel_rbf = BesselExpansion(max_value=5.0, K=K)

    gauss = gauss_rbf(dist)
    cheb = cheb_rbf(dist)
    bessel = bessel_rbf(dist)

    show = "gaussian"

    if show == "gaussian":
        plt.plot(dist, gauss)
    elif show == "cheb":
        plt.plot(dist, cheb)
    elif show == "bessel":
        plt.plot(dist, bessel)
    plt.xlabel("Distance")
    plt.ylabel("RBF Values")
    plt.title(f"{show} RBF")
    plt.show()