"""H1's z/y records (`pouw_h1`'s z table, compute accounting's `Accounting.Served.ValueChecked`): what
``verity_int8::scaled_mm_zy`` writes beside the served y over a call's T slots, one `ZY_BYTES` record a 32 × 32 unit in the
compiled Program's unit order, the unit's exact s32 z over the whole K then its y = dq(z, s_a, s_w) in bf16, little-endian
and row-major, an empty slot's bytes zero (`engine.int8_exact.zy_records`). Unit u's record is the z table's leaf u. The
credit's value check (`reason`) holds the records to the call's committed scales and its served y, on the device the
records are on. The records are read where the arm wrote them (`of`); `host` copies them only for what reads them on the
host."""
from __future__ import annotations

from typing import Any

import numpy as np

#: a unit's rows and columns (`pouw_h1.TILE`, `engine.int8_exact.UNIT`)
UNIT = (32, 32)
#: a unit's record: its s32 z, then its bf16 y
ZY_BYTES = 6 * UNIT[0] * UNIT[1]


def of(z: Any, units: int) -> Any:
    """A call of `units` units' records, [units, `ZY_BYTES`] uint8, as its arm handed them over (``scaled_mm_zy``'s zy: a
    tensor on its device, or an array), when `z` is them, else None."""
    if z is None or str(getattr(z, "dtype", "")).removeprefix("torch.") != "uint8" or tuple(z.shape) != (units, ZY_BYTES):
        return None
    return z.detach() if hasattr(z, "detach") else np.ascontiguousarray(z)


def host(records: Any) -> np.ndarray | None:
    """`records` (`of`'s) on the host, [units, `ZY_BYTES`] uint8."""
    if records is None:
        return None
    return np.ascontiguousarray(records.detach().cpu().numpy() if hasattr(records, "detach") else np.asarray(records))


def split(records: np.ndarray, rows: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(z [rows, n] int32, y [rows, n] bf16 bits as int16) of a call's records, unit (b, c) record b·(n / 32) + c."""
    ur, uc = UNIT
    half = 4 * ur * uc

    def grid(v: np.ndarray) -> np.ndarray:
        return v.reshape(rows // ur, n // uc, ur, uc).transpose(0, 2, 1, 3).reshape(rows, n)
    return (grid(np.ascontiguousarray(records[:, :half]).view("<i4")),
            grid(np.ascontiguousarray(records[:, half:]).view("<i2")))


def reason(records: Any, rows: int, n: int, y: Any, s_a: Any, s_w: Any) -> str | None:
    """Why a call's records over `rows` slots are not its served product's (None if they are): every unit's y is dq of its
    z at the call's committed scales, s_a [m] on the m served slots, 0 on the empty ones, and s_w [n] (`DequantOf`,
    `engine.int8_exact.dequantize`); the served y [m, n] (bf16, or its bits as int16) is the records' on the served rows;
    an empty slot's z is 0. Computed on the records' device (a tensor's, else the host), the same bits on either."""
    import torch

    from verity_vllm.engine import int8_exact as E
    ur, uc = UNIT
    r = torch.as_tensor(records)
    dev, half = r.device, 4 * ur * uc

    def grid(v: Any) -> Any:
        return v.reshape(rows // ur, n // uc, ur, uc).transpose(1, 2).reshape(rows, n)
    z, yr = grid(r[:, :half].contiguous().view(torch.int32)), grid(r[:, half:].contiguous().view(torch.int16))
    sv = torch.as_tensor(s_a).to(dev, torch.float32).reshape(-1)
    m = sv.numel()
    sa = torch.zeros(rows, dtype=torch.float32, device=dev)
    sa[:m] = sv
    dq = E.dequantize(z, sa, torch.as_tensor(s_w).to(dev, torch.float32).reshape(-1)).view(torch.int16)
    yt = torch.as_tensor(y).detach()
    ys = (yt.contiguous().view(torch.int16) if yt.dtype == torch.bfloat16 else yt.to(torch.int16)).to(dev).reshape(m, n)
    bad = (dq != yr, yr[:m] != ys, z[m:] != 0)

    def unit(b: Any, row0: int = 0) -> int:
        i, j = torch.nonzero(b)[0].tolist()
        return ((row0 + i) // ur) * (n // uc) + j // uc
    dq_bad, y_bad, z_bad = torch.stack([b.any() for b in bad]).tolist()
    if dq_bad:
        return f"unit {unit(bad[0])}'s y is not dq of its z at the call's scales"
    if y_bad:
        return f"the served y is not unit {unit(bad[1])}'s records' y"
    if z_bad:
        return f"unit {unit(bad[2], m)}'s empty slot has a z that isn't 0"
    return None


def refusal(held: tuple[Any, Any, Any] | None, call: Any, y: Any) -> str | None:
    """Why a call (its ``units``, ``rows`` and ``n``) the credit gate's trace credits earns nothing (None if it earns):
    `held`, its (records, s_a, s_w) as `pouw_h1.H1Run.record` held them, is None when its z wasn't its records, else
    `reason` with the served y."""
    if held is None:
        return f"its z is not its {call.units} z/y records of {ZY_BYTES} bytes"
    return reason(held[0], call.rows, call.n, y, held[1], held[2])


__all__ = ["UNIT", "ZY_BYTES", "host", "of", "reason", "refusal", "split"]
