import torch
import torch.nn as nn
from torch.nn import functional as F

from torch_geometric.nn import global_add_pool, global_mean_pool
from e3nn import o3 

from utils.blocks import (
    EquivariantProductBasisBlock,
    RadialEmbeddingBlock,
)

from utils.irreps_tools import reshape_irreps
from utils.tfn_layers import TensorProductConvLayer

ATOM_TYPES = 9
AMINO_ACID_TYPES = 20

class RES_MACEModel(nn.Module):
    def __init__(
        self,
        example,
        dropout,
        r_max=10.0,
        num_bessel=8,
        num_polynomial_cutoff=5,
        max_ell=2,
        correlation=3,
        n_layers=5,
        emb_dim=64,
        in_dim=ATOM_TYPES, # 9 ATOM TYPES
        out_dim=AMINO_ACID_TYPES,
        aggr="sum",
        pool="sum",
        residual=True,
        scalar_pred=True
    ):
        super().__init__()
        self.r_max = r_max
        self.emb_dim = emb_dim
        self.num_layers = n_layers
        self.residual = residual
        self.scalar_pred = scalar_pred
        # Embedding
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        sh_irreps = o3.Irreps.spherical_harmonics(max_ell)
        self.spherical_harmonics = o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )

        # Embedding lookup for initial node features
        self.emb_in = torch.nn.Embedding(in_dim, emb_dim)

        self.convs = torch.nn.ModuleList()
        self.prods = torch.nn.ModuleList()
        self.reshapes = torch.nn.ModuleList()
        hidden_irreps = (sh_irreps * emb_dim).sort()[0].simplify()
        irrep_seq = [
            o3.Irreps(f'{emb_dim}x0e'),
            # o3.Irreps(f'{emb_dim}x0e + {emb_dim}x1o + {emb_dim}x2e'),
            # o3.Irreps(f'{emb_dim//2}x0e + {emb_dim//2}x0o + {emb_dim//2}x1e + {emb_dim//2}x1o + {emb_dim//2}x2e + {emb_dim//2}x2o'),
            hidden_irreps
        ]
        for i in range(n_layers):
            in_irreps = irrep_seq[min(i, len(irrep_seq) - 1)]
            out_irreps = irrep_seq[min(i + 1, len(irrep_seq) - 1)]
            conv = TensorProductConvLayer(
                in_irreps=in_irreps,
                out_irreps=out_irreps,
                sh_irreps=sh_irreps,
                edge_feats_dim=self.radial_embedding.out_dim,
                hidden_dim=emb_dim,
                gate=False,
                aggr=aggr,
            )
            self.convs.append(conv)
            self.reshapes.append(reshape_irreps(out_irreps))
            prod = EquivariantProductBasisBlock(
                node_feats_irreps=out_irreps,
                target_irreps=out_irreps,
                correlation=correlation,
                element_dependent=False,
                num_elements=in_dim,
                use_sc=residual
            )
            self.prods.append(prod)

        # Global pooling/readout function
        self.pool = {"mean": global_mean_pool, "sum": global_add_pool}[pool]

        if self.scalar_pred:
            # Predictor MLP
            self.pred = torch.nn.Sequential(
                torch.nn.Linear(emb_dim, emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(emb_dim, out_dim)
            )
        else:
            self.pred = torch.nn.Linear(hidden_irreps.dim, out_dim)
    
    def forward(self, batch):
        h = self.emb_in(batch.node_type)  # (n,) -> (n, d)

        # Edge features
        vectors = batch.x[batch.edge_index[0]] - batch.x[batch.edge_index[1]]  # [n_edges, 3]
        lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]
        edge_attrs = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)
        
        for conv, reshape, prod in zip(self.convs, self.reshapes, self.prods):
            # Message passing layer
            h_update = conv(h, batch.edge_index, edge_attrs, edge_feats)
            # Update node features
            sc = F.pad(h, (0, h_update.shape[-1] - h.shape[-1]))
            h = prod(reshape(h_update), sc, None)

        if self.scalar_pred:
            # Select only scalars for prediction
            h = h[:,:self.emb_dim]

        # NOTE: We remove the read-out function in order to perform self-supervised classification
        out = self.pred(h)
        return out[batch.ca_idx + batch.ptr[:-1]]

        # out = self.pool(h, batch.batch)  # (n, d) -> (batch_size, d)
        # return self.pred(out)  # (batch_size, out_dim)
