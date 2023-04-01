""" Implemented by Simon Mathis, svm34@cam.ac.uk"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_cluster
import torch_geometric
from atom3d.util.formats import bp_to_df, read_any
from Bio.Data.IUPACData import protein_letters_3to1

from proteins import STANDARD_AMINO_ACIDS, STANDARD_ELEMENTS

_aa_alphabet = {aa: i for i, aa in enumerate(STANDARD_AMINO_ACIDS)}
_element_alphabet = {el: i for i, el in enumerate(STANDARD_ELEMENTS)}
_3to1 = lambda aa: protein_letters_3to1[aa.capitalize()]


class ProteinGraphBuilder:
    def __init__(
        self,
        node_alphabet: dict[str, int],
        edge_cutoff: float = 4.5,
        num_rbf: int = 16,
        self_loop: bool = False,
        device: str = "cpu",
        dtype=torch.float32,
    ):
        self.node_alphabet = node_alphabet
        self.edge_cutoff = edge_cutoff
        self.num_rbf = num_rbf
        self.device = device
        self.dtype = dtype
        self.self_loop = self_loop
        # TODO: Allow various radial functions

    def from_file(self, file_path: str, **kwargs) -> torch_geometric.data.Data:
        """
        Read a protein from a file and return a graph representation.

        Args:
            file_path (str): Path to the file to read.
            **kwargs: Additional arguments to pass to `__call__`.

        Returns:
            torch_geometric.data.Data: Graph representation of the protein.
        """

        pdb_structure = read_any(file_path)
        df = bp_to_df(pdb_structure)
        return self(df, **kwargs)


class AtomGraphBuilder(ProteinGraphBuilder):
    """
    Implementation of an ATOM3D Transform which featurizes the atomic
    coordinates in an ATOM3D dataframes into `torch_geometric.data.Data`
    graphs.

    Returned graphs have the following attributes:
    -x          atomic coordinates, shape [n_nodes, 3]
    -node_type  numeric encoding of atomic identity, shape [n_nodes]
    -edge_index edge indices, shape [2, n_edges]
    -edge_s     edge scalar features, shape [n_edges, 16]
    -edge_v     edge scalar features, shape [n_edges, 1, 3]

    Subclasses of AtomGraphBuilder will produce graphs with additional
    attributes for the tasks-specific training labels, in addition
    to the above.

    All subclasses of AtomGraphBuilder directly inherit the AtomGraphBuilder
    constructor.

    Args:
        node_alphabet (dict[str, int]): Dictionary mapping atomic identity
            to a numeric encoding.
        edge_cutoff (float): Cutoff distance for edges.
        **kwargs: Additional arguments to pass to `ProteinGraphBuilder`.

    Returns:
        torch_geometric.data.Data: atom-level graph
    """

    def __init__(
        self, node_alphabet: dict[str, int] = _element_alphabet, edge_cutoff: float = 4.5, **kwargs
    ):
        super().__init__(node_alphabet, edge_cutoff, **kwargs)

    def __call__(self, df: pd.DataFrame) -> torch_geometric.data.Data:
        """
        Featurize an ATOM3D dataframe into an atom-level graph representation.

        Args:
            df (pd.DataFrame): `pandas.DataFrame` of atomic coordinates
                    in the ATOM3D format.

        Returns:
            torch_geometric.data.Data: atom-level graph
        """
        with torch.no_grad():
            coords = torch.as_tensor(
                df[["x", "y", "z"]].to_numpy(), dtype=self.dtype, device=self.device
            )
            atoms = torch.as_tensor(
                df["element"].str.upper().map(self.node_alphabet).to_numpy(),
                dtype=torch.long,
                device=self.device,
            )

            edge_index = torch_cluster.radius_graph(coords, r=self.edge_cutoff, loop=self.self_loop)

            edge_s, edge_v = _edge_features(
                coords,
                edge_index,
                D_max=self.edge_cutoff,
                num_rbf=self.num_rbf,
                device=self.device,
            )

            return torch_geometric.data.Data(
                x=coords,
                node_type=atoms,
                edge_index=edge_index,
                edge_s=edge_s,
                edge_v=edge_v,
            )


class ResidueGraphBuilder(ProteinGraphBuilder):
    """
    Implementation of an ATOM3D Transform which featurizes the atomic
    coordinates in an ATOM3D dataframes into `torch_geometric.data.Data`
    residue-level graphs.

    Returned graphs have the following attributes:
    -x          residue coordinates, shape [n_nodes, 3]
    -node_type  numeric encoding of residue identity, shape [n_nodes]
    -edge_index edge indices, shape [2, n_edges]
    -node_s     node scalar features, shape [n_nodes, 6]
    -node_v     node vector features, shape [n_nodes, 3, 3]
    -edge_s     edge scalar features, shape [n_edges, n_rbf + n_pos_embeddings]
    -edge_v     edge scalar features, shape [n_edges, 1, 3]
    -mask       mask for valid nodes, shape [n_nodes]

    Subclasses of ResidueGraphBuilder will produce graphs with additional
    attributes for the tasks-specific training labels, in addition
    to the above.

    All subclasses of ResidueGraphBuilder directly inherit the ResidueGraphBuilder
    constructor.

    Args:
        node_alphabet (dict[str, int]): Dictionary mapping one-letter residue
            identity to a numeric encoding.
        granularity (Literal["CA", "centroid", "backbone"]): Granularity of the
            residue-level graph. If "CA", only the alpha-carbon atoms are
            included. If "centroid", the centroid of each residue is used as the
            node position. If "backbone", the centroid of the backbone atoms
            (N, CA, C) is used as the node position. Defaults to "CA".
        edge_cutoff (float): Cutoff distance for edges.
        n_pos_embeddings (int): Number of positional embeddings to use. Defaults
            to 16.
        **kwargs: Additional arguments to pass to `ProteinGraphBuilder`.

    Returns:
        torch_geometric.data.Data: atom-level graph
    """

    def __init__(
        self,
        node_alphabet: dict[str, int] = _aa_alphabet,
        granularity: Literal["CA", "centroid", "backbone"] = "CA",
        edge_cutoff: float = 10.0,
        n_pos_embeddings: int = 16,
        **kwargs,
    ):
        super().__init__(node_alphabet, edge_cutoff, **kwargs)
        self.granularity = granularity
        self.n_pos_embeddings = n_pos_embeddings

    def __call__(self, df: pd.DataFrame) -> torch_geometric.data.Data:

        # Get nodes
        coords = torch.as_tensor(self._get_residue_coords(df), device=self.device, dtype=self.dtype)
        node_types = torch.as_tensor(
            self._get_residue_types(df), dtype=torch.long, device=self.device
        )
        mask = torch.isfinite(coords.sum(axis=1))
        coords[~mask] = np.inf

        # Get edges
        edge_index = torch_cluster.radius_graph(coords, r=self.edge_cutoff, loop=self.self_loop)

        # Add node features
        dihedrals = self._dihedrals(df)
        orientations = self._orientations(coords)
        sidechains = self._sidechains(df)
        node_s = dihedrals
        node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)

        # Add edge features
        pos_embeddings = self._positional_embeddings(edge_index)
        E_vectors = coords[edge_index[0]] - coords[edge_index[1]]
        rbf = _rbf(E_vectors.norm(dim=-1), D_count=self.num_rbf, device=self.device)
        edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
        edge_v = _normalize(E_vectors).unsqueeze(-2)

        # Turn NaN to zeros
        node_s, node_v, edge_s, edge_v = map(torch.nan_to_num, (node_s, node_v, edge_s, edge_v))

        return torch_geometric.data.Data(
            x=coords,
            node_type=node_types,
            edge_index=edge_index,
            node_s=node_s,
            node_v=node_v,
            edge_s=edge_s,
            edge_v=edge_v,
            mask=mask,
        )

    def _get_residue_coords(self, df: pd.DataFrame) -> np.ndarray:
        if self.granularity == "CA":
            return df[df["name"] == "CA"][["x", "y", "z"]].to_numpy()
        elif self.granularity == "centroid":
            return df.groupby("residue")[["x", "y", "z"]].mean().to_numpy()
        elif self.granularity == "backbone":
            raise NotImplementedError("Backbone granularity not implemented yet.")
            return df[df["name"].isin(["N", "CA", "C"])][["x", "y", "z"]].to_numpy()
        else:
            raise ValueError(f"Granularity {self.granularity} not recognized.")

    def _get_residue_types(self, df: pd.DataFrame) -> np.ndarray:
        if self.granularity == "CA":
            return df[df["name"] == "CA"]["resname"].map(_3to1).map(self.node_alphabet).to_numpy()
        elif self.granularity == "centroid":
            return (
                df.groupby("residue")["resname"]
                .agg(pd.Series.mode)
                .map(_3to1)
                .map(self.node_alphabet)
                .to_numpy()
            )
        elif self.granularity == "backbone":
            raise NotImplementedError("Backbone granularity not implemented yet.")
            return df[df["name"].isin(["N", "CA", "C"])][["x", "y", "z"]].to_numpy()
        else:
            raise ValueError(f"Granularity {self.granularity} not recognized.")

    def _dihedrals(self, df: pd.DataFrame, eps: float = 1e-7) -> torch.Tensor:
        # From https://github.com/jingraham/neurips19-graph-protein-design
        X = torch.as_tensor(
            df[df["name"].isin(["N", "CA", "C", "O"])][["x", "y", "z"]].values.reshape(-1, 4, 3)
        )

        X = torch.reshape(X[:, :3], [3 * X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = _normalize(dX, dim=-1)
        u_2 = U[:-2]
        u_1 = U[1:-1]
        u_0 = U[2:]

        # Backbone normals
        n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, [1, 2])
        D = torch.reshape(D, [-1, 3])
        # Lift angle representations to the circle
        D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
        return D_features

    def _positional_embeddings(self, edge_index: torch.Tensor) -> torch.Tensor:
        # From https://github.com/jingraham/neurips19-graph-protein-design
        d = edge_index[0] - edge_index[1]

        frequency = torch.exp(
            torch.arange(0, self.n_pos_embeddings, 2, dtype=self.dtype, device=self.device)
            * -(np.log(10000.0) / self.n_pos_embeddings)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

    def _orientations(self, X: torch.Tensor) -> torch.Tensor:
        forward = _normalize(X[1:] - X[:-1])
        backward = _normalize(X[:-1] - X[1:])
        forward = F.pad(forward, [0, 0, 0, 1])
        backward = F.pad(backward, [0, 0, 1, 0])
        return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

    def _sidechains(self, df: pd.DataFrame) -> torch.Tensor:
        # For each structure, X should be a num_residues x 4 x 3 nested
        # list of the positions of the backbone N, C-alpha, C, and O atoms of
        # each residue (in that order).
        X = torch.as_tensor(
            df[df["name"].isin(["N", "CA", "C", "O"])][["x", "y", "z"]].values.reshape(-1, 4, 3)
        )
        n, origin, c = X[:, 0], X[:, 1], X[:, 2]
        c, n = _normalize(c - origin), _normalize(n - origin)
        bisector = _normalize(c + n)
        perp = _normalize(torch.cross(c, n))
        vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
        return vec


def _edge_features(
    coords: torch.Tensor,
    edge_index: torch.Tensor,
    D_max: float = 4.5,
    num_rbf: int = 16,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:

    E_vectors = coords[edge_index[0]] - coords[edge_index[1]]
    rbf = _rbf(E_vectors.norm(dim=-1), D_max=D_max, D_count=num_rbf, device=device)

    edge_s = rbf
    edge_v = _normalize(E_vectors).unsqueeze(-2)

    edge_s, edge_v = map(torch.nan_to_num, (edge_s, edge_v))

    return edge_s, edge_v


def _normalize(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    """
    return torch.nan_to_num(torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _rbf(
    D: torch.Tensor, D_min: float = 0.0, D_max: float = 20.0, D_count: int = 16, device: str = "cpu"
) -> torch.Tensor:
    """
    From https://github.com/jingraham/neurips19-graph-protein-design

    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    """
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-(((D_expand - D_mu) / D_sigma) ** 2))
    return RBF
