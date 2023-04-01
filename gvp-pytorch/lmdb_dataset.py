import gzip
import io
import logging
import msgpack
import os
from pathlib import Path
import pickle as pkl

import lmdb
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, IterableDataset

try:
    import torch_geometric.data as ptg
except:
    print('torch geometric not found, GNN examples will not work until it is.')
    ptg = None

import atom3d.util.rosetta as ar
import atom3d.util.file as fi
import atom3d.util.formats as fo

from utils.serialisation import serialise, deserialise
logger = logging.getLogger(__name__)


class LMDBDataset(Dataset):
    """
    Creates a dataset from an lmdb file. Adapted from `TAPE <https://github.com/songlab-cal/tape/blob/master/tape/datasets.py>`_.
    :param data_file: path to LMDB file containing dataset
    :type data_file: Union[str, Path]
    :param transform: Transformation function to apply to each item.
    :type transform: Function, optional
    """

    def __init__(self, data_file, transform=None):
        """constructor
        """
        if type(data_file) is list:
            if len(data_file) != 1:
                raise RuntimeError("Need exactly one filepath for lmdb")
            data_file = data_file[0]

        self.data_file = Path(data_file).absolute()
        if not self.data_file.exists():
            raise FileNotFoundError(self.data_file)

        self._connect()

        with self._env.begin(write=False) as txn:
            self._num_examples = int(txn.get(b'num_examples'))
            self._serialization_format = \
                txn.get(b'serialization_format').decode()
            self._id_to_idx = deserialise(
                txn.get(b'id_to_idx'), self._serialization_format)

        # NOTE: We remove the `_env` variable in `init` on purpose as it messes with
        #   multiprocessing.
        #   c.f. https://github.com/pytorch/vision/issues/689#issuecomment-787215916
        self._disconnect()
        self._transform = transform

    def __len__(self) -> int:
        return self._num_examples

    def get(self, id: str):
        idx = self.id_to_idx(id)
        return self[idx]

    def id_to_idx(self, id: str):
        if id not in self._id_to_idx:
            raise IndexError(id)
        idx = self._id_to_idx[id]
        return idx

    def ids_to_indices(self, ids):
        return [self.id_to_idx(id) for id in ids]

    def ids(self):
        return list(self._id_to_idx.keys())

    def __getitem__(self, index: int):
        if not 0 <= index < self._num_examples:
            raise IndexError(index)
        
        if self._env is None:
            self._connect()
            
        with self._env.begin(write=False) as txn:

            compressed = txn.get(str(index).encode())
            buf = io.BytesIO(compressed)
            with gzip.GzipFile(fileobj=buf, mode="rb") as f:
                serialized = f.read()
            try:
                item = deserialise(serialized, self._serialization_format)
            except:
                return None
        # Recover special data types (currently only pandas dataframes).
        if 'types' in item.keys():
            for x in item.keys():
                if item['types'][x] == str(pd.DataFrame):
                    item[x] = pd.DataFrame(**item[x])
        else:
            logging.warning('Data types in item %i not defined. Will use basic types only.'%index)

        if 'file_path' not in item:
            item['file_path'] = str(self.data_file)
        if 'id' not in item:
            item['id'] = str(index)
        if self._transform:
            item = self._transform(item)
        return item
    
    def _connect(self) -> None:
        self._env = lmdb.open(
            str(self.data_file),
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def _disconnect(self) -> None:
        if self._env:
            self._env = None
