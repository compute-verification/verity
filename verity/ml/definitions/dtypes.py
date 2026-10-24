"""Torch-style dtype strings, element sizes, and exact BF16 <-> float32 conversion in numpy.

The canonical dtype string is the torch dtype name without the ``torch.`` prefix
(``"bfloat16"``, ``"float8_e4m3fn"``, ``"int32"``).  :func:`normalize_dtype` accepts the
prefixed spelling and returns the canonical one.

Two non-tensor dtypes exist for boundary leaves only: ``"json"`` (canonical JSON bytes) and
``"utf8"`` (a UTF-8 string).  Their leaves have shape ``[]`` and a variable byte length.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

# dtype -> (itemsize, numpy view dtype or None).  BF16 and FP8 have no numpy dtype and are viewed
# as unsigned integers of the same width.
_TENSOR_DTYPES: dict[str, tuple[int, str | None]] = {
    "bool": (1, "bool"),
    "uint8": (1, "uint8"),
    "int8": (1, "int8"),
    "int16": (2, "int16"),
    "uint16": (2, "uint16"),
    "int32": (4, "int32"),
    "uint32": (4, "uint32"),
    "int64": (8, "int64"),
    "uint64": (8, "uint64"),
    "float16": (2, "float16"),
    "bfloat16": (2, "uint16"),
    "float32": (4, "float32"),
    "float64": (8, "float64"),
    "float8_e4m3fn": (1, "uint8"),
    "float8_e4m3fnuz": (1, "uint8"),
    "float8_e5m2": (1, "uint8"),
    "float8_e5m2fnuz": (1, "uint8"),
}

VARIABLE_DTYPES = frozenset({"json", "utf8"})

TENSOR_DTYPES = frozenset(_TENSOR_DTYPES)
ALL_DTYPES = TENSOR_DTYPES | VARIABLE_DTYPES


class DtypeError(ValueError):
    pass


def normalize_dtype(dtype: str) -> str:
    """``"torch.bfloat16"`` -> ``"bfloat16"``; rejects anything not in the table."""
    if not isinstance(dtype, str):
        raise DtypeError(f"dtype must be a string, got {type(dtype).__name__}")
    name = dtype.removeprefix("torch.")
    if name not in ALL_DTYPES:
        raise DtypeError(f"unknown dtype {dtype!r}")
    return name


def itemsize(dtype: str) -> int | None:
    """Bytes per element, or ``None`` for the variable-length dtypes."""
    name = normalize_dtype(dtype)
    if name in VARIABLE_DTYPES:
        return None
    return _TENSOR_DTYPES[name][0]


def numel(shape: Sequence[int]) -> int:
    return math.prod(int(d) for d in shape)


def expected_nbytes(dtype: str, shape: Sequence[int]) -> int | None:
    size = itemsize(dtype)
    if size is None:
        return None
    return size * numel(shape)


def check_nbytes(dtype: str, shape: Sequence[int], nbytes: int) -> None:
    """Raise :class:`DtypeError` when ``nbytes`` is not what dtype x shape requires."""
    expected = expected_nbytes(dtype, shape)
    if expected is not None and expected != nbytes:
        raise DtypeError(
            f"{normalize_dtype(dtype)} x {list(shape)} needs {expected} bytes, got {nbytes}"
        )


def view_dtype(dtype: str) -> np.dtype:
    """The numpy dtype that views the raw bytes (BF16/FP8 as unsigned ints)."""
    name = normalize_dtype(dtype)
    if name in VARIABLE_DTYPES:
        raise DtypeError(f"{name} has no array view")
    return np.dtype(_TENSOR_DTYPES[name][1]).newbyteorder("<")


def bytes_to_array(dtype: str, shape: Sequence[int], data: bytes) -> np.ndarray:
    """Raw little-endian bytes -> numpy array in the view dtype (a copy, C order)."""
    check_nbytes(dtype, shape, len(data))
    return np.frombuffer(data, dtype=view_dtype(dtype)).reshape(tuple(int(d) for d in shape)).copy()


def array_to_bytes(array: np.ndarray) -> bytes:
    """Little-endian C-order bytes of an array (the exact device bytes when the array is a view)."""
    arr = np.ascontiguousarray(array)
    if arr.dtype.byteorder == ">":
        arr = arr.astype(arr.dtype.newbyteorder("<"))
    return arr.tobytes(order="C")


# --- BF16 ---------------------------------------------------------------------------------------


def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    """uint16 BF16 bit patterns -> float32, exactly (BF16 is the top half of float32)."""
    bits = np.asarray(bits, dtype=np.uint16)
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)


def f32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    """float32 -> uint16 BF16 bit patterns with round-to-nearest-even on the dropped 16 bits.

    NaN keeps its sign and is quieted (the payload's top bit is set) so it never rounds to an
    infinity; overflow rounds to the signed infinity exactly as IEEE RNE would.
    """
    f32 = np.ascontiguousarray(np.asarray(values, dtype=np.float32))
    bits = f32.view(np.uint32)
    nan_mask = np.isnan(f32)
    lsb = (bits >> np.uint32(16)) & np.uint32(1)
    rounded = (bits + np.uint32(0x7FFF) + lsb) >> np.uint32(16)
    nan_bits = ((bits >> np.uint32(16)) | np.uint32(0x0040)) & np.uint32(0xFFFF)
    out = np.where(nan_mask, nan_bits, rounded).astype(np.uint16)
    return out


def bf16_bytes_to_f32(data: bytes, shape: Sequence[int]) -> np.ndarray:
    return bf16_bits_to_f32(bytes_to_array("bfloat16", shape, data))


def f32_to_bf16_bytes(values: np.ndarray) -> bytes:
    return array_to_bytes(f32_to_bf16_bits(values))


def to_float32(dtype: str, shape: Sequence[int], data: bytes) -> np.ndarray:
    """Decode a float leaf (bf16/f16/f32/f64) to float32 exactly where exact (f64 rounds)."""
    name = normalize_dtype(dtype)
    if name == "bfloat16":
        return bf16_bytes_to_f32(data, shape)
    if name in ("float16", "float32", "float64"):
        return bytes_to_array(name, shape, data).astype(np.float32)
    raise DtypeError(f"{name} is not a float dtype this helper decodes")
