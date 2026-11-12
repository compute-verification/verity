"""Cuts `fa2_train_bwd_sm120.npz` from the capture `verity.definitions.evidence` names `cc12_fa2_backward_capture`: the words
of torch's flash attention backward on cc 12.0 under deterministic algorithms, its `out/bwd.npz`.

    python fa2_train_bwd_sm120.py CAPTURE/out/bwd.npz

It keeps, of one head each:

    rows/NAME/...    query rows (`rows`) with their q, dO, O, LSE and dQ, and the keys and values of the key blocks they see
                     (`k`, `v`: keys 0 .. the last such block's end); `ns` is the kernel's dQ buffers at that capture's B and H
    block/NAME/...   the last key block (`BN` keys) with its dK and dV, and the rows at and below it with their q, dO, O and LSE

    rows/xl/wide     B 2, H 24 (4 buffers), q and k N(0, 9): rows 0, 5 and 127 (one key block, of which they see 1, 6 and all
                     128 keys), 130 (two) and 621 (five, so one buffer holds two parts: its word 2 is another under the sum
                     in the key blocks' order)
    block/l16/unit   B 2, H 16, q and k N(0, 1): keys and rows 896 to 1,023, two query blocks

The archive's members are written in order with one fixed date, so the same capture gives the same bytes.
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path

import numpy as np

from verity.definitions.registry.fa2_train_bwd import train_sdpa_backward

#: name -> (capture case, which of its heads, (B, H), rows)
ROWS = {"xl/wide": ("bwd/xl/det/wide", 0, (2, 24), (0, 5, 127, 130, 621))}
#: name -> (capture case, which of its heads, (B, H))
BLOCK = {"l16/unit": ("bwd/l16/det/unit", 1, (2, 16))}


def _head(d, case: str, bh: tuple[int, int]) -> dict:
    """The bindings of the head Definition at the case's shape: its key block `BN` and dQ buffers `NS`."""
    return train_sdpa_backward(*bh, *d[f"{case}/q"].shape[1:]).bindings


def cut(d) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name, (case, head, bh, rows) in ROWS.items():
        r, b = list(rows), _head(d, case, bh)
        keys = (max(r) // b["BN"] + 1) * b["BN"]
        out[f"rows/{name}/rows"] = np.array(r, np.int32)
        out[f"rows/{name}/ns"] = np.array(b["NS"], np.int32)
        for f in ("q", "do", "o", "lse", "dq"):
            out[f"rows/{name}/{f}"] = d[f"{case}/{f}"][head][r]
        for f in ("k", "v"):
            out[f"rows/{name}/{f}"] = d[f"{case}/{f}"][head][:keys]
    for name, (case, head, bh) in BLOCK.items():
        T, BN = d[f"{case}/q"].shape[1], _head(d, case, bh)["BN"]
        for f in ("q", "do", "o", "lse", "k", "v", "dk", "dv"):
            out[f"block/{name}/{f}"] = d[f"{case}/{f}"][head][T - BN:]
    return out


def write(arrays: dict[str, np.ndarray], path: Path) -> None:
    with zipfile.ZipFile(path, "w") as z:
        for name, a in arrays.items():
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.asarray(a), version=(1, 0))
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info, buf.getvalue(), compresslevel=9)


if __name__ == "__main__":
    write(cut(np.load(sys.argv[1])), Path(__file__).with_suffix(".npz"))
