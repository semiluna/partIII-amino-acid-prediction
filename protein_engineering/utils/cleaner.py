"""Cleans up a PDB file/CIF file using pdbfixer.
Adatpted and expanded from
https://github.com/deepmind/alphafold/blob/a3941673e90b8d1d75c60b16a4b3707ebf7ba527/alphafold/relax/cleanup.py
"""

import io
import os
import sys
from typing import Optional

import numpy as np
import openmm
from pdbfixer import PDBFixer
from scipy.spatial import KDTree

from simtk.openmm import app
from simtk.openmm.app import element


__all__ = ["clean_protein"]


#### PATCHES ####
# Updated find_nearest_distance with a KDTree for a speedup
def _findNearestDistance(self, context, topology, newAtoms, n_workers: int = 1):
    positions = (
        context.getState(getPositions=True)
        .getPositions(asNumpy=True)
        .value_in_unit(openmm.unit.nanometer)
    )
    atomResidue = [atom.residue for atom in topology.atoms()]
    kdtree = KDTree(
        positions[: len(atomResidue)],
    )
    if not hasattr(self, "yaosen_inresidue"):
        self.yaosen_inresidue = dict()
        for atom in newAtoms:
            self.yaosen_inresidue[atom.index] = np.array([i.index for i in atom.residue.atoms()])
    nearest = sys.float_info.max
    # if nearest < 0.13, then refine, thus we only find the near points, say 2.0.
    for atom in newAtoms:
        query = positions[atom.index]
        neighbor_index = np.array(kdtree.query_ball_point(query, 2.0, workers=n_workers))
        neighbor_index_mask = np.isin(
            neighbor_index, self.yaosen_inresidue[atom.index], invert=True
        )
        neighbor_index = neighbor_index[neighbor_index_mask]
        # only calculate the distance between positions[neighbor_index] and query = positions[atom.index]
        if len(neighbor_index) == 0:
            continue
        dist = np.sqrt(np.min(np.sum((positions[neighbor_index] - query) ** 2, axis=1)))
        if dist < nearest:
            nearest = dist
    return nearest


PDBFixer._findNearestDistance = _findNearestDistance


#### FUNCTIONS ####
def clean_protein(
    *,
    file_path: os.PathLike = None,
    pdb_file: io.TextIOBase = None,
    pdbx_file: io.TextIOBase = None,
    **kwargs,
) -> io.StringIO:
    """Cleans up a PDB files using pdbfixer. Returns a StringIO object formatted as a PDB file.
    Args:
        file_path (str): Path to a PDB/PDBx file. Mutually exclusive with pdb_file and pdbx_file.
        pdb_file (io.TextIOBase): An IO (e.g. StringIO, FileIO) object containing a PDB file.
            Mutually exclusive with file_path and pdbx_file.
        pdbx_file (io.TextIOBase): An IO (e.g. StringIO, FileIO) object containing a PDBx/mmCIF file.
            Mutually exclusive with file_path and pdb_file.
    Returns:
        io.StringIO: A StringIO object containing a PDB file. Formatted as a PDB file.
    """
    alterations_info = {}
    if file_path is not None:
        file_path = str(file_path)
    fixed_pdb = fix_pdb(
        file_path=file_path,
        pdb_file=pdb_file,
        pdbx_file=pdbx_file,
        alterations_info=alterations_info,
        **kwargs,
    )
    fixed_pdb = clean_structure(fixed_pdb, alterations_info)
    print(f"Applied alterations: {alterations_info}")
    return fixed_pdb


def fix_pdb(
    *,
    file_path: str = None,
    pdb_file: io.TextIOBase = None,
    pdbx_file: io.TextIOBase = None,
    alterations_info: Optional[dict] = None,
    keep_water: bool = False,
    add_hydrogen: bool = False,
    pH: float = 7.0,
) -> io.StringIO:
    """
    Apply pdbfixer to the contents of a PDB file; return a PDB StringIO object result.
    1) Replaces nonstandard residues.
    2) Removes heterogens (non protein residues) including water.
    3) Adds missing residues and missing atoms within existing residues.
    4) Adds hydrogens assuming pH=7.0.
    5) KeepIds is currently true, so the fixer must keep the existing chain and
       residue identifiers. This will fail for some files in wider PDB that have
       invalid IDs.
    Args:
        file_path (str): Path to PDB file to fix. Mutually exclusive with pdb_file and pdbx_file.
        pdb_file (io.StringIO): PDB file to fix. Mutually exclusive with file_path and pdbx_file.
        pdbx_file (io.StringIO): PDBx/mmCIF file to fix. Mutually exclusive with file_path and pdb_file.
        alterations_info (dict): A dictionary to store information about the alterations made by
            pdbfixer. This is useful for debugging. If None, no information is stored.
        keep_water (bool): If True, water molecules are kept in the output PDB file. Defaults to False.
        add_hydrogen (bool): If True, hydrogens are added to the output PDB file. Defaults to False.
        pH (float): The pH to use when adding hydrogens. Defaults to 7.0.
    Returns:
        io.StringIO: A StringIO object containing the fixed PDB file (in PDB format).
    """
    fixer = PDBFixer(filename=file_path, pdbfile=pdb_file, pdbxfile=pdbx_file)
    alterations_info = {} if alterations_info is None else alterations_info
    # Replace nonstandard residues.
    fixer.findNonstandardResidues()
    alterations_info["nonstandard_residues"] = fixer.nonstandardResidues
    fixer.replaceNonstandardResidues()
    # Remove heterogens (non protein residues) including water. Use custom function to track alterations.
    _remove_heterogens(fixer, alterations_info, keep_water=keep_water)
    # Add missing residues and missing atoms within existing residues.
    fixer.findMissingResidues()
    alterations_info["missing_residues"] = fixer.missingResidues
    # Add all missing heavy atoms, as specified by the missingAtoms, missingTerminals, and missingResidues fields
    fixer.findMissingAtoms()
    alterations_info["missing_heavy_atoms"] = fixer.missingAtoms
    # NOTE: addMissingAtoms also adds missing residues.
    fixer.addMissingAtoms(seed=0)
    if add_hydrogen:
        fixer.addMissingHydrogens(pH=pH)
    return _openmm_pdb_to_strIO(fixer.topology, fixer.positions)


def clean_structure(pdb_file: io.TextIOBase, alterations_info: dict) -> io.StringIO:
    """Applies additional fixes to an OpenMM structure, to handle edge cases.
    Args:
        pdb_file (io.TextIOBase): An IO object (e.g. StringIO, FileIO) containing a PDB file. Formatted as a PDB file.
        alterations_info (dict): A dictionary to store information about the alterations made by
            pdbfixer. This is useful for debugging. If None, no information is stored.
    Returns:
        io.StringIO: A StringIO object containing the fixed PDB file (in PDB format).
    """
    openmm_pdb_structure = openmm.app.internal.pdbstructure.PdbStructure(pdb_file)
    _replace_met_se(openmm_pdb_structure, alterations_info)
    _remove_chains_of_length_one(openmm_pdb_structure, alterations_info)
    openmm_pdb_file = openmm.app.PDBFile(openmm_pdb_structure)
    return _openmm_pdb_to_strIO(openmm_pdb_file.topology, openmm_pdb_file.positions)


def _openmm_pdb_to_strIO(
    topology: openmm.app.Topology, positions: openmm.unit.quantity.Quantity
) -> io.StringIO:
    """Returns a pdb string provided OpenMM topology and positions.
    Args:
        topology (openmm.app.Topology): OpenMM topology object.
        positions (openmm.unit.quantity.Quantity): OpenMM positions object.
    Returns:
        io.StringIO: A StringIO object containing a PDB file. Formatted as a PDB file.
    """
    handle = io.StringIO()
    openmm.app.PDBFile.writeFile(
        topology, positions, handle, keepIds=True
    )  # Keep residue and atom ids
    handle.seek(0)
    return handle


def _remove_heterogens(
    fixer: PDBFixer, alterations_info: Optional[dict] = None, keep_water: bool = False
) -> None:
    """Removes the residues that Pdbfixer considers to be heterogens.
    Args:
        fixer (PDBFixer): A PDBFixer object to modify.
        alterations_info (dict): A dictionary to store information about the alterations made by
            pdbfixer. This is useful for debugging. If None, no information is stored.
        keep_water (bool): If True, water molecules (HOH) are kept in the output PDB file. Defaults to False.
    Returns:
        None
    """
    if alterations_info:
        initial_resnames = set()
        for chain in fixer.topology.chains():
            for residue in chain.residues():
                initial_resnames.add(residue.name)
    fixer.removeHeterogens(keepWater=keep_water)
    if alterations_info:
        final_resnames = set()
        for chain in fixer.topology.chains():
            for residue in chain.residues():
                final_resnames.add(residue.name)
        alterations_info["removed_heterogens"] = initial_resnames.difference(final_resnames)


def _replace_met_se(
    pdb_structure: openmm.app.internal.pdbstructure.PdbStructure, alterations_info: dict
) -> None:
    """Replace the Se in any MET residues that were not marked as modified."""
    modified_met_residues = []
    for res in pdb_structure.iter_residues():
        name = res.get_name_with_spaces().strip()
        if name == "MET":
            s_atom = res.get_atom("SD")
            if s_atom.element_symbol == "Se":
                s_atom.element_symbol = "S"
                s_atom.element = element.get_by_symbol("S")
                modified_met_residues.append(s_atom.residue_number)
    alterations_info["Se_in_MET"] = modified_met_residues


def _remove_chains_of_length_one(
    pdb_structure: openmm.app.internal.pdbstructure.PdbStructure, alterations_info: dict
) -> None:
    """Removes chains that correspond to a single amino acid.
    A single amino acid in a chain is both N and C terminus. There is no force
    template for this case.
    Args:
      pdb_structure: An OpenMM pdb_structure to modify and fix.
      alterations_info: A dict that will store details of changes made.
    """
    removed_chains = {}
    for model in pdb_structure.iter_models():
        valid_chains = [c for c in model.iter_chains() if len(c) > 1]
        invalid_chain_ids = [c.chain_id for c in model.iter_chains() if len(c) <= 1]
        model.chains = valid_chains
        for chain_id in invalid_chain_ids:
            model.chains_by_id.pop(chain_id)
        removed_chains[model.number] = invalid_chain_ids
    alterations_info["removed_chains"] = removed_chains