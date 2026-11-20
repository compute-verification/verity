"""The name of a request Program's Build dir (`trace_dir`), which the verifier's own Builds are written under
(`naming.write_build`). `steps` re-exports it."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from verity_vllm.program import prefix_plan as PP


def _label(policy: dict) -> str:
    return "greedy" if policy["kind"] == "greedy" else f"t{policy['temperature']:g}-p{policy['top_p']:g}" + (f"-r{policy['rid']}" if "rid" in policy else "")


def trace_dir(u: Any) -> str:
    """The derive dir name of a request Program's trace: `build_request_<prefix_plan.shape_name>_<policy>`, in class G then `_l` and
    the first 12 hex digits of the SHA-256 of its launch context's compact JSON (with its chunks, `{"batch": …, "chunks": …}`, and
    its entries, `"entries": …`; batch invariant, its launches' max_seqlen_q, `[n0, n1, …]`), and with a length rule `_s` and the
    first 12 of its config's (`RequestUnit.stop`). `u`: a
    `steps.RequestUnit` or its summary."""
    LP, T, C, policy, launch, chunks, stop, rows, seqlen_q = (
        (u["LP"], u["T"], u.get("C") or 0, u["policy"], u.get("launch"), u.get("chunks"), u.get("stop"), u.get("entries"), u.get("seqlen_q"))
        if isinstance(u, Mapping) else (u.LP, u.T, u.C, u.policy, u.batch(), list(u.chunks), u.stop, list(u.entries), list(u.seqlen_q)))
    launch = launch or seqlen_q
    ctx = {"batch": launch, "chunks": chunks, **({"entries": rows} if rows else {})} if chunks else launch
    tail = f"_l{hashlib.sha256(json.dumps(ctx, separators=(',', ':')).encode()).hexdigest()[:12]}" if launch else ""
    tail += f"_s{hashlib.sha256(json.dumps(stop, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:12]}" if stop else ""
    return f"build_request_{PP.shape_name(PP.shape(LP, T, C))}_{_label(policy)}{tail}"


