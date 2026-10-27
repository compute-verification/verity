"""Comparing an evaluated member with its committed words: which output a committed member names (a rank partial, an
interior slice, a padded MoE slot) and the lifted bit comparison."""
from __future__ import annotations

import re
from typing import Any

import numpy as np

from verity_vllm.query.manifest import format as RM
from verity_vllm.check.replay.opening import CommittedStore


def _tp_member_alias(got: dict[str, Any], member: str) -> Any | None:
    """[TP-08d, MOE-TP-01] The evaluated value a rank Program's TP member (`part_rank<r>` / `shard_rank<r>`) compares against when the
    relation did not name it: the evaluator's `0` / `out`, else the relation's ONE output whatever its name (MoeSum's `sum` = the rank's
    MoE-block partial before the all-reduce).  None for any other member name, or for a multi-output relation (never a guess between two)."""
    if member in got or not member.startswith((RM.TP_PARTIAL_MEMBER.split("{")[0], RM.TP_SHARD_MEMBER.split("{")[0])):
        return None
    if "0" in got:
        return got["0"]
    if "out" in got:
        return got["out"]
    if len(got) == 1:
        return next(iter(got.values()))
    return None


def _interior_slice(got: dict[str, np.ndarray], lo: int, hi: int) -> np.ndarray | None:
    """The words [lo, hi) of an INTERIOR producer's output refs = the concatenation of its members in return order (a
    Fp8GroupQuant row returns (xq[K], sx[K/G]); a member-less Gemm returns its one output).  None when the slice leaves the output."""
    parts = [np.asarray(v).reshape(-1) for v in got.values()]
    total = sum(int(p.size) for p in parts)
    if lo < 0 or hi > total:
        return None
    if len(parts) == 1:
        return parts[0][lo:hi]
    off, out = 0, []
    for p in parts:
        a, b = max(lo, off), min(hi, off + p.size)
        if a < b:
            out.append(p[a - off:b - off])
        off += p.size
    if len(out) == 1:
        return out[0]
    return np.concatenate([o.astype(np.uint32) for o in out])       # a slice across members of different widths: widened words (values kept)


_SLOT_MEMBER_RE = re.compile(r"^slot(?P<e>\d+)/(up|act|down|g)$")


def padded_slot_enable(member: str, got: dict[str, np.ndarray], store: "CommittedStore", rid: str, step: int, op_path: str, row: int,
                       topk: int) -> tuple[bool | None, str]:
    """The enable a_e of a padded slot member of one token row, from the COMMITTED router_ids of that row (the block's
    enable is the step's run bit -> the router members are Active in every executed step; a_e = [e in ids]).  None + why when the committed
    ids are not available (the slot's verdict is then not evaluable by name -- never the twin's ids)."""
    m = _SLOT_MEMBER_RE.match(member)
    if m is None:
        return True, ""                                                          # router_ids / router_w: Active under the block's enable
    e = int(m.group("e"))
    w, why = store.words(rid, step, op_path, "router_ids", row * topk, (row + 1) * topk)
    if w is None:
        return None, f"slot enable of {member}: the row's committed router_ids are not available -- {why}"
    ids = np.asarray(w).reshape(-1)
    bot = RM.lifted_bot(RM.LIFTED_F32)
    if any(int(v) & bot for v in ids):
        return None, f"slot enable of {member}: a committed router id of the row is bot ({[hex(int(v)) for v in ids]}) in an executed step"
    return bool(any(int(v) == e for v in ids)), ""


def lifted_compare(have: np.ndarray, payload: np.ndarray, active: bool, dtype: str) -> tuple[bool, int, int | None, str]:
    """Committed LIFTED words (`have`, the plane's machine words) vs the twin's unlifted payload under the slot enable:
    Active -> every word must be Active(payload) = payload exactly (tag clear); disabled -> every word must be enc(bot) = 1 << w exactly
    (never INV, never a bot with a nonzero payload).  Returns (equal, differing_elements, first_diff, verdict)."""
    wbits, nbytes = RM.LIFTED_DTYPES[dtype]
    bot = 1 << wbits
    dt = np.uint32 if nbytes == 4 else np.uint64
    have = np.asarray(have).reshape(-1).astype(dt, copy=False)
    if active:
        want = np.asarray(payload).reshape(-1).astype(dt, copy=False)
        verdict = "active: word == Active(twin value)"
    else:
        want = np.full(have.shape if have.size else np.asarray(payload).reshape(-1).shape, bot, dtype=dt)
        verdict = "disabled: word == enc(bot) exactly"
    if have.shape != want.shape:
        return False, -1, None, verdict + f" -- width {have.shape} vs {want.shape}"
    diff = have != want
    nd = int(np.count_nonzero(diff))
    if nd == 0:
        return True, 0, None, verdict
    first = int(np.flatnonzero(diff)[0])
    hv = int(have[first])
    kind = ("bot word under an Active enable" if (active and hv == bot) else
            "value word under a disabled enable" if (not active and not (hv & bot)) else
            "bot with a nonzero payload / INV" if (not active and (hv & bot) and hv != bot) else
            "INV word (bot bit + nonzero payload) under an Active enable" if (active and (hv & bot)) else "payload differs")
    return False, nd, first, verdict + f" -- FAIL: {kind}"
