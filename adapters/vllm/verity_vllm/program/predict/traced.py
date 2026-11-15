"""The traced side of the compiler (`steps`): the Build traces of a compilation's request Programs, which `steps.compare` holds
the compiled digests to. `steps` re-exports these names."""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from typing import Any

from verity_vllm.program import prefix_plan as PP


def _label(policy: dict) -> str:
    return "greedy" if policy["kind"] == "greedy" else f"t{policy['temperature']:g}-p{policy['top_p']:g}" + (f"-r{policy['rid']}" if "rid" in policy else "")


NO_EDGE_TRACE = "no trace: the Build's request wrapper returns its samples alone, never a window's edge rows"
NO_PARTIAL_TRACE = ("no trace: the Build's chunked request wrapper computes each chunk as its sequence's whole rows in its launch, "
                    "never the first rows of a longer one (a recomputed hit's)")


def trace_dir(u: Any) -> str:
    """The derive dir name of a request Program's trace: `build_request_<prefix_plan.shape_name>_<policy>`, in class G then `_l` and
    the first 12 hex digits of the SHA-256 of its launch context's compact JSON (with its chunks, `{"batch": …, "chunks": …}`, and
    its entries, `"entries": …`), and with a length rule `_s` and the first 12 of its config's (`RequestUnit.stop`). `u`: a
    `steps.RequestUnit` or its summary."""
    LP, T, C, policy, launch, chunks, stop, rows = (
        (u["LP"], u["T"], u.get("C") or 0, u["policy"], u.get("launch"), u.get("chunks"), u.get("stop"), u.get("entries"))
        if isinstance(u, Mapping) else (u.LP, u.T, u.C, u.policy, u.batch(), list(u.chunks), u.stop, list(u.entries)))
    ctx = {"batch": launch, "chunks": chunks, **({"entries": rows} if rows else {})} if chunks else launch
    tail = f"_l{hashlib.sha256(json.dumps(ctx, separators=(',', ':')).encode()).hexdigest()[:12]}" if launch else ""
    tail += f"_s{hashlib.sha256(json.dumps(stop, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:12]}" if stop else ""
    return f"build_request_{PP.shape_name(PP.shape(LP, T, C))}_{_label(policy)}{tail}"


def trace_commands(summary: dict, registered: Any, root: str, *, max_model_len: int = 2048) -> list[list[str]]:
    """One `verity-vllm build --wrapper request` per distinct request Program of a compilation (`Compiled.summary()`) under
    `registered` (`steps.Registered`), on the pinned vLLM (CPU, on meta), each writing `<root>/<trace_dir>` at a `max_model_len` of
    at least its LP + T, under φ's GPU's name where φ states one (`--device-name`: the tuned kernel configs a GPU-free host reads);
    at tensor-parallel world W, φ's rank's (`--tp W --rank r --dead-free`); in class G batched (`--batched`, with
    `--dsa-table-reads` under `Registered.dsa_table_reads`) at the unit's launch context (`--launch-context '{"batch": ...}'`), with
    its prefill chunks there when it has them (`'{"batch": ..., "chunks": ...}'`: the Build's chunked request wrapper,
    `vllm_meta.ChunkedGreedyRequest`); its length rule `--stop`. A Program with a window's edge rows has none (`NO_EDGE_TRACE`),
    nor one with a chunk of fewer rows than its launch ran (`NO_PARTIAL_TRACE`), so `steps.compare` counts it `no_trace`."""
    from verity_vllm.program.frontend.target_profile import TargetProfile
    phi, seen, cmds = registered.phi, set(), []
    target = {k: v for k, v in phi.target.items() if k in TargetProfile.__dataclass_fields__}
    # the target's GPU name, which vLLM's get_device_name answers on any host (its tuned fused-MoE configs' key)
    device = ["--device-name", str(phi.target["device_name"])] if phi.target.get("device_name") else []
    world = int(phi.tp.get("world", 1))
    batched = [] if phi.target_profile().batch_invariant else ["--batched", *(["--dsa-table-reads"] if registered.dsa_table_reads else [])]
    tp = ["--tp", str(world), "--rank", str(int(phi.tp.get("rank", 0))), "--dead-free"] if world > 1 else []
    for u in summary["requests"]:
        if u is None or u.get("edge") or u.get("entries") or (d := trace_dir(u)) in seen:
            continue
        seen.add(d)
        pol, ctx = u["policy"], {"batch": u.get("launch"), **({"chunks": list(u["chunks"])} if u.get("chunks") else {})}
        sampling = "greedy" if pol["kind"] == "greedy" else json.dumps({k: pol[k] for k in ("temperature", "top_p", "top_k", "rid") if k in pol}, sort_keys=True)
        cmds.append(["verity-vllm", "build", "--out", os.path.join(root, d), "--wrapper", "request", "--LP", str(u["LP"]), "--T", str(u["T"]),
                     *(["--cached", str(u["C"])] if u.get("C") else []),
                     *(["--launch-context", json.dumps(ctx, separators=(",", ":"))] if u.get("launch") else []),
                     *(["--stop", json.dumps(u["stop"], sort_keys=True, separators=(",", ":"))] if u.get("stop") else []),
                     "--model", phi.model["repo"], "--revision", phi.model["revision"],
                     "--max-model-len", str(max(max_model_len, int(u["LP"]) + int(u["T"]))),
                     "--target", json.dumps(target, sort_keys=True), *device, "--profile", phi.profile, "--sampling", sampling, *tp, *batched])
    return cmds
