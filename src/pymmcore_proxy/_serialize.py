"""JSON-safe serialization for pymmcore-proxy wire format.

Handles: numpy arrays, tuples, bytes, MDAEvent, enums, and basic types.
Tagged dicts with "__type__" key distinguish special types from plain dicts.
"""

from __future__ import annotations

import base64
import enum
from typing import Any

import numpy as np


def encode(obj: Any) -> Any:
    """Encode a Python object into a JSON-safe representation."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)

    if isinstance(obj, np.ndarray):
        return {
            "__type__": "ndarray",
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
            "data": base64.b64encode(obj.tobytes()).decode("ascii"),
        }

    if isinstance(obj, tuple):
        return {"__type__": "tuple", "items": [encode(x) for x in obj]}

    if isinstance(obj, list):
        return [encode(x) for x in obj]

    if isinstance(obj, dict):
        return {k: encode(v) for k, v in obj.items()}

    if isinstance(obj, bytes):
        return {"__type__": "bytes", "data": base64.b64encode(obj).decode("ascii")}

    if isinstance(obj, enum.Enum):
        return {"__type__": "enum", "class": type(obj).__qualname__, "value": obj.value}

    # useq MDAEvent and other pydantic models
    if hasattr(obj, "model_dump"):
        return {
            "__type__": "model",
            "class": type(obj).__qualname__,
            "module": type(obj).__module__,
            "data": obj.model_dump(mode="json"),
        }

    # Fallback: try to convert to string
    return str(obj)


def decode(obj: Any) -> Any:
    """Decode a JSON representation back into Python objects."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    if isinstance(obj, list):
        return [decode(x) for x in obj]

    if isinstance(obj, dict):
        t = obj.get("__type__")

        if t == "ndarray":
            raw = base64.b64decode(obj["data"])
            return np.frombuffer(raw, dtype=np.dtype(obj["dtype"])).reshape(obj["shape"]).copy()

        if t == "tuple":
            return tuple(decode(x) for x in obj["items"])

        if t == "bytes":
            return base64.b64decode(obj["data"])

        if t == "model":
            return _reconstruct_model(obj)

        if t == "enum":
            # Return raw value — caller can cast if needed
            return obj["value"]

        # Plain dict
        return {k: decode(v) for k, v in obj.items()}

    return obj


def _reconstruct_model(obj: dict) -> Any:
    """Reconstruct a pydantic model from serialized form."""
    module = obj.get("module", "")
    cls_name = obj.get("class", "")
    data = obj.get("data", {})

    # Try to import and reconstruct (check MDASequence before MDAEvent)
    if cls_name == "MDASequence":
        try:
            from useq import MDASequence
            return MDASequence(**data)
        except Exception:
            pass
    if "useq" in module or cls_name == "MDAEvent":
        try:
            from useq import MDAEvent
            return MDAEvent(**data)
        except Exception:
            pass

    # Fallback: return the raw dict
    return data
