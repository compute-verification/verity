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
    else None."""
    if z is None or str(getattr(z, "dtype", "")).removeprefix("torch.") != "uint8" or tuple(z.shape) != (units, ZY_BYTES):
        return None
    return np.ascontiguousarray(z.detach().cpu().numpy() if hasattr(z, "detach") else np.asarray(z))


def split(records: np.ndarray, rows: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(z [rows, n] int32, y [rows, n] bf16 bits as int16) of a call's records, unit (b, c) record b·(n / 32) + c."""
    ur, uc = UNIT
    half = 4 * ur * uc

    def grid(v: np.ndarray) -> np.ndarray:
        return v.reshape(rows // ur, n // uc, ur, uc).transpose(0, 2, 1, 3).reshape(rows, n)
    return (grid(np.ascontiguousarray(records[:, :half]).view("<i4")),
            grid(np.ascontiguousarray(records[:, half:]).view("<i2")))


def reason(records: np.ndarray, rows: int, n: int, y: np.ndarray, s_a: np.ndarray, s_w: np.ndarray) -> str | None:
    """Why a call's records over `rows` slots are not its served product's (None if they are): every unit's y is dq of its
    z at the call's committed scales, s_a [m] on the m served slots, 0 on the empty ones, and s_w [n] (`DequantOf`,
    `engine.int8_exact.dequantize`); the served y [m, n] bf16 bits are the records' on the served rows; an empty slot's z
    is 0."""
    import torch

    from verity_vllm.engine import int8_exact as E
    m = int(np.asarray(s_a).size)
    z, yr = split(records, rows, n)
    sa = np.zeros(rows, np.float32)
    sa[:m] = np.asarray(s_a, np.float32).reshape(-1)
    dq = E.dequantize(torch.from_numpy(z), torch.from_numpy(sa),
                      torch.from_numpy(np.ascontiguousarray(s_w, np.float32).reshape(-1))).view(torch.int16).numpy()

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


__all__ = ["UNIT", "ZY_BYTES", "of", "reason", "refusal", "split"]
