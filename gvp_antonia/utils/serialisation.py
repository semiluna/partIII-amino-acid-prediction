from __future__ import annotations

import pickle as pkl
import dill
import msgpack
import json 

from typing import Any, Literal, Sequence

def serialise(x, serialization_format):
    """
    Serializes dataset `x` in format given by `serialization_format` (pkl, json, msgpack).
    """
    if serialization_format == 'pkl':
        # Pickle
        # Memory efficient but brittle across languages/python versions.
        return pkl.dumps(x)
    elif serialization_format == 'json':
        # JSON
        # Takes more memory, but widely supported.
        serialized = json.dumps(
            x, default=lambda df: json.loads(
                df.to_json(orient='split', double_precision=6))).encode()
    elif serialization_format == 'msgpack':
        # msgpack
        # A bit more memory efficient than json, a bit less supported.
        serialized = msgpack.packb(
            x, default=lambda df: df.to_dict(orient='split'))
    else:
        raise RuntimeError('Invalid serialization format')
    return serialized


def deserialise(x, serialization_format):
    """
    Deserializes dataset `x` assuming format given by `serialization_format` (pkl, json, msgpack).
    """
    if serialization_format == 'pkl':
        return pkl.loads(x)
    elif serialization_format == 'json':
        serialized = json.loads(x)
    elif serialization_format == 'msgpack':
        serialized = msgpack.unpackb(x)
    else:
        raise RuntimeError('Invalid serialization format')
    return serialized