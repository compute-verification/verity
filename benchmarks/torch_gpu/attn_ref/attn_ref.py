"""A shim over attn_bf16's reference path: `qk` and `pv` are `attn.qk` and `attn.pv`
(benchmarks/torch_gpu/attn_bf16/attn.py, which says what each word is) with ``path="ref"``.
$ATTN_REF_CUBIN, when set, names the cubin (build.sh here ships attn_bf16's sm_90a cubin as attn_ref.sm_90a.cubin), else
attn_bf16.sm_90a.cubin beside attn.py.  New code imports attn_bf16's `attn`; this directory goes once e2e_h200 and
commit_slots do.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "attn_bf16"))
import attn

NEG_INF = attn.NEG_INF
PV_K = attn.PAD_K
WARPS = attn.WARPS
_CUBIN = os.environ.get("ATTN_REF_CUBIN")
if _CUBIN:
    os.environ["ATTN_BF16_CUBIN"] = _CUBIN


def cubin_path() -> Path:
    return Path(_CUBIN or os.path.join(attn.HERE, "attn_bf16.sm_90a.cubin"))


def qk(q, k, lens=None, **kw):
    return attn.qk(q, k, lens, path="ref", **kw)


def pv(p, v, lens=None, **kw):
    return attn.pv(p, v, lens, path="ref", **kw)
