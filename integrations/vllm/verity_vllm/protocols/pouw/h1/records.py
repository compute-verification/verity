"""H1's z/y records (`pouw_h1`'s z table, compute accounting's `Accounting.Served.ValueChecked`): what
``verity_int8::scaled_mm_zy`` writes beside the served y over a call's T slots, one `ZY_BYTES` record a 32 × 32 unit in the
compiled Program's unit order, the unit's exact s32 z over the whole K then its y = dq(z, s_a, s_w) in bf16, little-endian
and row-major, an empty slot's bytes zero (`engine.int8_exact.zy_records`). Unit u's record is the z table's leaf u. The
credit's value check (`reason`) holds the records to the call's committed scales and its served y."""
from __future__ import annotations

from typing import Any

import numpy as np

#: a unit's rows and columns (`pouw_h1.TILE`, `engine.int8_exact.UNIT`)
UNIT = (32, 32)
#: a unit's record: its s32 z, then its bf16 y
ZY_BYTES = 6 * UNIT[0] * UNIT[1]


def of(z: Any, units: int) -> np.ndarray | None:
    """A call of `units` units' records on the host, [units, `ZY_BYTES`] uint8, when `z` is them (``scaled_mm_zy``'s zy),
    else None. A CUDA tensor is copied into pinned host memory, which the array holds."""
    if z is None or str(getattr(z, "dtype", "")).removeprefix("torch.") != "uint8" or tuple(z.shape) != (units, ZY_BYTES):
        return None
    if getattr(z, "is_cuda", False):
        import torch
        host = torch.empty(tuple(z.shape), dtype=torch.uint8, pin_memory=True)
        host.copy_(z.detach())
        return host.numpy()
    return np.ascontiguousarray(z.detach().cpu().numpy() if hasattr(z, "detach") else np.asarray(z))


def split(records: np.ndarray, rows: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(z [rows, n] int32, y [rows, n] bf16 bits as int16) of a call's records, unit (b, c) record b·(n / 32) + c."""
    ur, uc = UNIT
    half = 4 * ur * uc

    def grid(v: np.ndarray) -> np.ndarray:
        return v.reshape(rows // ur, n // uc, ur, uc).transpose(0, 2, 1, 3).reshape(rows, n)
    return (grid(np.ascontiguousarray(records[:, :half]).view("<i4")),
            grid(np.ascontiguousarray(records[:, half:]).view("<i2")))


def _dq(z: np.ndarray, sa: np.ndarray, s_w: Any) -> np.ndarray:
    """dq(z, s_a, s_w)'s bf16 bits as int16 (`engine.int8_exact.dequantize`, row by row)."""
    import torch

    from verity_vllm.engine import int8_exact as E
    return E.dequantize(torch.from_numpy(z), torch.from_numpy(sa),
                        torch.from_numpy(np.ascontiguousarray(s_w, np.float32).reshape(-1))).view(torch.int16).numpy()


def _holds(records: np.ndarray, rows: int, n: int, y: Any, s_a: Any, s_w: Any) -> bool:
    """Whether `reason` finds nothing, read in full only on the row blocks that hold a served row. Every later slot has
    scale 0, so `reason` holds there iff its z is 0 and its y is dq(0, 0, s_w), one row for them all: those blocks are
    compared in the records' own layout. False also when the shapes are not a call's, s_w is not finite or a dq is a NaN
    (torch's CPU cast doesn't give every NaN the same bits), for `reason` to read the records whole."""
    ur, uc = UNIT
    m = int(np.asarray(s_a).size)
    if not (isinstance(records, np.ndarray) and records.dtype == np.uint8 and rows > 0 and rows % ur == 0 and n > 0
            and n % uc == 0 and records.shape == ((rows // ur) * (n // uc), ZY_BYTES) and m <= rows
            and np.asarray(y).size == m * n and np.asarray(s_w).size == n
            and np.isfinite(np.asarray(s_w, np.float32)).all()):
        return False
    cols, half, served = n // uc, 4 * ur * uc, -(-m // ur) * ur
    head, tail = records[:served // ur * cols], records[served // ur * cols:]
    if served:
        z, yr = split(head, served, n)
        sa = np.zeros(served, np.float32)
        sa[:m] = np.asarray(s_a, np.float32).reshape(-1)
        dq = _dq(z, sa, s_w)
        if ((dq & 0x7FFF) > 0x7F80).any() or (dq != yr).any() or (yr[:m] != np.asarray(y, np.int16).reshape(m, n)).any() \
                or z[m:].any():
            return False
    if not len(tail):
        return True
    y0 = _dq(np.zeros((1, n), np.int32), np.zeros(1, np.float32), s_w).reshape(cols, 1, uc)
    return not tail[:, :half].any() and bool((tail[:, half:].view("<i2").reshape(-1, cols, ur, uc) == y0).all())


def reason(records: np.ndarray, rows: int, n: int, y: np.ndarray, s_a: np.ndarray, s_w: np.ndarray) -> str | None:
    """Why a call's records over `rows` slots are not its served product's (None if they are): every unit's y is dq of its
    z at the call's committed scales, s_a [m] on the m served slots, 0 on the empty ones, and s_w [n] (`DequantOf`,
    `engine.int8_exact.dequantize`); the served y [m, n] bf16 bits are the records' on the served rows; an empty slot's z
    is 0. Records that hold are accepted by `_holds`; the rest are read whole, here, for the first unit that doesn't."""
    if _holds(records, rows, n, y, s_a, s_w):
        return None
    return _reason(records, rows, n, y, s_a, s_w)


def _reason(records: np.ndarray, rows: int, n: int, y: np.ndarray, s_a: np.ndarray, s_w: np.ndarray) -> str | None:
    """`reason` over every slot: the reference `_holds` agrees with."""
    m = int(np.asarray(s_a).size)
    z, yr = split(records, rows, n)
    sa = np.zeros(rows, np.float32)
    sa[:m] = np.asarray(s_a, np.float32).reshape(-1)
    dq = _dq(z, sa, s_w)

    def unit(bad: np.ndarray, row0: int = 0) -> int:
        i, j = (int(v) for v in np.argwhere(bad)[0])
        return ((row0 + i) // UNIT[0]) * (n // UNIT[1]) + j // UNIT[1]
    if (bad := dq != yr).any():
        return f"unit {unit(bad)}'s y is not dq of its z at the call's scales"
    if (bad := yr[:m] != np.asarray(y, np.int16).reshape(m, n)).any():
        return f"the served y is not unit {unit(bad)}'s records' y"
    if (bad := z[m:] != 0).any():
        return f"unit {unit(bad, m)}'s empty slot has a z that isn't 0"
    return None


def refusal(held: tuple[np.ndarray, np.ndarray, np.ndarray] | None, call: Any, y: np.ndarray) -> str | None:
    """Why a call (its ``units``, ``rows`` and ``n``) the credit gate's trace credits earns nothing (None if it earns):
    `held`, its (records, s_a, s_w) as `pouw_h1.H1Run.record` held them, is None when its z wasn't its records, else
    `reason` with the served y's bf16 bits `y`."""
    if held is None:
        return f"its z is not its {call.units} z/y records of {ZY_BYTES} bytes"
    return reason(held[0], call.rows, call.n, y, held[1], held[2])


def holds_on(zy: Any, rows: int, n: int, y: Any, s_a: Any, s_w: Any) -> bool:
    """Whether `reason` finds nothing, read on the records' own device from the call's tensors (``scaled_mm_zy``'s zy, the
    served y [m, n] bf16, s_a [m] and s_w [n]) with one sync for the answer. dq there is `engine.int8_exact.dequantize`: an
    s32 to f32 conversion, two f32 products and a bf16 rounding, each to nearest, so its bits are the host's but for a NaN's.
    False, for `refusal` to read the host's copy, when they aren't a call's tensors on one device, s_w isn't finite, a dq is
    a NaN, or anything doesn't hold."""
    import torch

    from verity_vllm.engine import int8_exact as E
    ur, uc = UNIT
    if not all(torch.is_tensor(t) for t in (zy, y, s_a, s_w)):
        return False
    m, dev = s_a.numel(), zy.device
    if not (zy.dtype == torch.uint8 and y.dtype == torch.bfloat16 and rows > 0 and rows % ur == 0 and n > 0
            and n % uc == 0 and tuple(zy.shape) == ((rows // ur) * (n // uc), ZY_BYTES) and m <= rows
            and y.numel() == m * n and s_w.numel() == n and all(t.device == dev for t in (y, s_a, s_w))):
        return False

    def grid(v: Any) -> Any:
        return v.reshape(rows // ur, n // uc, ur, uc).permute(0, 2, 1, 3).reshape(rows, n)
    zy = zy.detach().contiguous()
    z, yr = grid(zy.view(torch.int32)[:, :ur * uc]), grid(zy.view(torch.int16)[:, 2 * ur * uc:])
    sw = s_w.detach().reshape(-1).to(torch.float32)
    sa = torch.zeros(rows, dtype=torch.float32, device=dev)
    sa[:m] = s_a.detach().reshape(-1).to(torch.float32)
    dq = E.dequantize(z, sa, sw).view(torch.int16)
    bad = (~torch.isfinite(sw).all() | ((dq & 0x7FFF) > 0x7F80).any() | (dq != yr).any() | (z[m:] != 0).any()
           | (yr[:m] != y.detach().contiguous().view(torch.int16).reshape(m, n)).any())
    return not bool(bad)


__all__ = ["UNIT", "ZY_BYTES", "holds_on", "of", "reason", "refusal", "split"]
