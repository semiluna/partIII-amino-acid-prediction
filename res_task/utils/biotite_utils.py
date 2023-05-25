import gzip
import io
import os
import pathlib
from typing import Dict, Union

import biotite.structure as bs
import biotite.structure.io.mmtf as mmtf
import biotite.structure.io.pdb as pdb
import biotite.structure.io.pdbx as pdbx
import numpy as np
from Bio.Data.IUPACData import protein_letters_3to1

_3to1 = _3to1 = {key.upper(): val for key, val in protein_letters_3to1.items()}


def structure_to_buffer(struct: bs.AtomArray, type="cif") -> io.StringIO:
    buffer = io.StringIO()
    if type == "cif":
        file = pdbx.PDBxFile()
        pdbx.set_structure(file, struct, data_block="structure")
        file.write(buffer)
    elif type == "pdb":
        file = pdb.PDBFile()
        pdb.set_structure(file, struct)
        file.write(buffer)
    else:
        raise ValueError(f"Unknown type: {type}")
    buffer.seek(0)
    return buffer


def structure_from_buffer(
    buffer: Union[io.StringIO, io.BytesIO], file_type="cif", **load_kwargs
) -> bs.AtomArray:
    buffer.seek(0)
    if file_type in ("cif", "mmcif", "pdbx"):
        file = pdbx.PDBxFile()
        file.read(buffer)
        if "assembly_id" in load_kwargs:
            return pdbx.get_assembly(file, **load_kwargs)
        return pdbx.get_structure(file, **load_kwargs)
    elif file_type == "pdb":
        file = pdb.PDBFile()
        file.read(buffer)
        return pdb.get_structure(file, **load_kwargs)
    elif file_type == "mmtf":
        file = mmtf.MMTFFile()
        file.read(buffer)
        return mmtf.get_structure(file, **load_kwargs)
    else:
        raise ValueError(f"Unknown type: {file_type}")


def load_structure(
    path_or_buffer: Union[io.StringIO, pathlib.Path, str], file_type: str = None, **load_kwargs
) -> bs.AtomArray:
    if isinstance(path_or_buffer, io.StringIO):
        assert file_type is not None, "Type must be specified when loading from buffer"
        return structure_from_buffer(path_or_buffer, file_type=file_type, **load_kwargs)

    path = pathlib.Path(path_or_buffer)
    assert path.exists(), f"File does not exist: {path}"

    if path.suffix in (".gz", ".gzip"):
        open_func = gzip.open
        file_type = os.path.splitext(path.stem)[-1].split(".")[-1]
    else:
        open_func = open
        file_type = path.suffix.split(".")[-1]

    buffer = io.BytesIO() if file_type == "mmtf" else io.StringIO()
    mode = "rb" if file_type == "mmtf" else "rt"
    with open_func(path, mode) as file:
        buffer.write(file.read())
    return structure_from_buffer(buffer, file_type=file_type, **load_kwargs)


def get_sequence(structure: bs.AtomArray) -> Dict[str, str]:
    structure = structure[bs.filter_amino_acids(structure)]
    sequence = {}
    for chain_id in bs.get_chains(structure):
        sequence_array = bs.residues.get_residues(structure[structure.chain_id == chain_id])[1]
        sequence[chain_id] = "".join([_3to1[res] for res in sequence_array])
    return sequence


# MMTF assembly:
def _get_label_asym_id_per_atom(mmtf_structure, mmtf_file):
    label_asym_id_per_chain = mmtf_file["chainIdList"]
    label_asym_id_per_residue = np.repeat(label_asym_id_per_chain, mmtf_file["groupsPerChain"])
    label_asym_id_per_atom = bs.spread_residue_wise(mmtf_structure, label_asym_id_per_residue)
    return label_asym_id_per_atom


def get_assembly(mmtf_file, assembly=0, model=1, **kwargs):
    mmtf_structure = mmtf.get_structure(mmtf_file, model=model, **kwargs)

    # Get the transformation for the first biological assembly
    transforms = mmtf_file["bioAssemblyList"][assembly]
    # Get the label_asym_chain_id for each atom, as this is used to identify the chains
    # in the biological assembly
    label_asym_chain_id_per_atom = _get_label_asym_id_per_atom(mmtf_structure, mmtf_file)

    assembly = None
    for transform in transforms["transformList"]:
        # Subset to the relevant atoms in the biological assembly
        _relevant_chains = np.array(mmtf_file["chainIdList"])[transform["chainIndexList"]]
        _relevant_atoms = np.isin(label_asym_chain_id_per_atom, _relevant_chains)

        sub_structure = mmtf_structure[_relevant_atoms]

        # Apply the transformation to the relevant chains
        _transform_4d = np.asarray(transform["matrix"]).reshape(4, 4)
        rotation, translation = _transform_4d[:3, :3], _transform_4d[:3, 3]
        sub_structure.coord = bs.transform.matrix_rotate(sub_structure.coord, rotation)
        sub_structure.coord += translation

        if assembly is None:
            assembly = sub_structure
        else:
            assembly += sub_structure

    return assembly


# MMTF sequence
def get_polymer_seqres(mmtf_file):
    chains = mmtf_file["chainIdList"]
    return {
        ",".join(chains[x["chainIndexList"]]): x["sequence"]
        for x in mmtf_file["entityList"]
        if x["type"] == "polymer"
    }