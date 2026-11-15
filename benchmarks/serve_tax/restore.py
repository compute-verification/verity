"""Restore a replay block's boundary state from a served ``restore`` arm (`serve_tax`) into a fresh engine, replay the
replay block's steps to its far edge, and say how close the replay came.

    restore.py --arm-dir OUT/restore [--from-edge 0] [--to-edge 1] [--kv-dir DIR] [--checkpoint DIR] [--eager] --out DIR

The served arm took the boundary state at each edge (`servetax_plugin.edges_of`, module `boundary`): the KV pool's bytes
(``--kv-dir``/edge-K.kv) and root, and the scheduler's, block pool's and model runner's state (edges/edge-K.state.pkl).
This builds the served engine again in this process (``VLLM_ENABLE_V1_MULTIPROCESSING=0``) from the step trace's header
(the scheduler's settings, and the KV cache's format and the speculative decoding where it holds them:
`replay.HEADER_ARGS`), with the served server's stats setting, loads the KV bytes into the same block ids, puts the state back
(`boundary.restore`), and checks the restore: the KV root again, and every leaf `boundary.dump_core` renders against the
served engine core's at the edge. It then replays: each request that arrived after the edge is added under its served
engine id, prompt and sampling right before the scheduled step it arrived before, and the engine core drains and takes the
state again once it has scheduled as many steps as the served engine had at the far edge.

restore.json compares, from edge to edge: the schedule step by step (the step trace's records), every sampled token and
its logprob's bits step by step, and at the far edge the KV root, each block's fingerprint, the block pool and every
dumped leaf; and times the engine's build, the KV load, the state's restore and the replay, with the GPU's busy seconds
of the replay beside serving's over the same steps.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools" / "config"))
sys.path.append(str(ROOT))
from verity_config.settings import SHARED  # noqa: E402 (from the tree's tools/config, put on sys.path above)
import boundary as B  # noqa: E402 (beside this file)
import replay  # noqa: E402
import serve_tax as S  # noqa: E402

#: the step trace's step fields compared (not ``t``, the served clock)
STEP_FIELDS = ("rows", "new", "resumed", "cache_hit", "preempted", "emitted", "finished")


def served(arm_dir: Path, kv_dir: Path, a: int, b: int) -> dict[str, Any]:
    """The served arm's edges ``a`` and ``b`` (records and states), the step trace between them, the arrivals the replay
    adds (each with its plan prompt: those before a step between the edges, and those that arrived while edge ``b``
    drained, which its state holds) and the served steps' outputs."""
    srv = arm_dir / "server"
    recs = [json.loads(x) for f in sorted(srv.glob("edges.*.jsonl")) for x in f.read_text().splitlines() if x.strip()]
    edges = {r["edge"]: r for r in recs if "next_step" in r}
    files = {r["edge"]: r for r in recs if "state_write_s" in r}
    if a not in edges or b not in edges:
        raise SystemExit(f"{srv}: edges {sorted(edges)}, not {a} and {b}")
    st = {k: pickle.loads((kv_dir / f"edge-{k}.state.pkl").read_bytes()) for k in (a, b)}
    dumps = {k: json.loads((srv / "edges" / f"edge-{k}.dump.json").read_text()) for k in (a, b)}
    lo, hi = edges[a]["next_step"], edges[b]["next_step"]
    (trace,) = S.step_traces(arm_dir)
    tr = S.trace_records(trace)
    held, held_far = ({r.request_id for r in B.unpickle_requests(st[k]["requests"], {"block_hasher": None})} for k in (a, b))
    named = json.loads((arm_dir / "meta.json").read_text()).get("plan")
    plan_file = next((p for p in (Path(named) if named else None, arm_dir / "plan.jsonl") if p and p.is_file()), None)
    if plan_file is None:
        raise SystemExit(f"no plan at {named} or {arm_dir / 'plan.jsonl'}")
    plan = {r["i"]: r for r in S.trace_records(plan_file)}
    client = {r["id"]: r for f in arm_dir.glob("*/rep-*/client.jsonl") for r in map(json.loads, f.read_text().splitlines())}
    client |= {r["id"]: r for f in arm_dir.glob("warm/client.jsonl") for r in map(json.loads, f.read_text().splitlines())}
    arrive = []
    for r in tr:
        if r["kind"] != "arrive" or r["request"] in held:
            continue
        if lo <= r["before_step"] < hi or (r["before_step"] == hi and r["request"] in held_far):
            http = r["request"].split("-", 1)[1].rsplit("-", 2)[0]
            arrive.append({"engine_id": r["request"], "before_step": r["before_step"], "sampling": r["sampling"],
                           "max_tokens": r["sampling"].get("max_tokens"), "prompt_ids": plan[client[http]["i"]]["prompt_ids"]})
    steps = {r["step"]: r for r in tr if r["kind"] == "step" and lo <= r["step"] < hi}
    outs: dict[int, dict] = {}
    for f in sorted((srv / "edges").glob("outputs.*.jsonl*")):
        outs |= {o["step"]: o["out"] for o in S.trace_records(f) if lo <= o["step"] < hi}
    return {"edges": edges, "files": files, "state": st, "dumps": dumps, "lo": lo, "hi": hi, "header": tr[0],
            "arrive": arrive, "steps": steps, "outputs": outs}


def engine(checkpoint: Path, header: dict, eager: bool, log_stats: bool) -> Any:
    from vllm import EngineArgs
    from vllm.v1.engine.llm_engine import LLMEngine

    cfg = header["config"]
    args = EngineArgs(model=str(checkpoint), dtype="bfloat16", seed=0, trust_remote_code=True, enforce_eager=eager,
                      max_model_len=cfg["max_model_len"], max_num_seqs=cfg["max_num_seqs"],
                      max_num_batched_tokens=cfg["max_num_batched_tokens"], enable_chunked_prefill=cfg["enable_chunked_prefill"],
                      enable_prefix_caching=cfg["enable_prefix_caching"], async_scheduling=cfg["async_scheduling"],
                      block_size=cfg["block_size"], num_gpu_blocks_override=cfg["num_gpu_blocks"],
                      disable_log_stats=not log_stats,
                      **{arg: cfg[key] for key, arg in replay.HEADER_ARGS.items() if key in cfg})
    return LLMEngine.from_engine_args(args)


def params(r: dict) -> Any:
    """The served request's sampling parameters as the server's completions endpoint built them for the streaming client,
    but for ``detokenize`` (`BY_CONSTRUCTION`)."""
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    s = r["sampling"]
    keep = {k: s[k] for k in ("temperature", "top_p", "top_k", "min_p", "seed", "min_tokens", "presence_penalty",
                              "frequency_penalty", "repetition_penalty", "stop_token_ids", "ignore_eos") if s.get(k) is not None}
    return SamplingParams(n=1, max_tokens=r["max_tokens"], logprobs=0, detokenize=False, output_kind=RequestOutputKind.DELTA,
                          skip_clone=True, **keep)


def warm(eng: Any, core: Any, requests: list[dict]) -> int:
    """Serve ``requests`` (the replay block's own prompts and sampling, under other ids) to the end in the engine before
    the restore, so the kernels serving compiled during its warm-up are compiled here too; the steps it took. Under async
    scheduling the engine core has scheduled a batch past the last finish, so it steps on until its batch queue is
    empty. The restore then overwrites what they left, which the restored engine's dump checks."""
    for i, r in enumerate(requests):
        eng.add_request(f"restore-warm-{i}", {"prompt_token_ids": r["prompt_ids"]}, params(r))
    steps = 0
    while eng.has_unfinished_requests() or getattr(core, "batch_queue", None):
        eng.step()
        steps += 1
    return steps


def by_construction(path: str, served: dict[str, Any], replayed: dict[str, Any]) -> bool:
    """A dump path where the replay differs from serving by construction, read by no engine core: the served engine
    core's process wrapper's own attributes (`EngineCoreProc`: the replay's engine core runs in this process), and each
    request's ``detokenize``, which the served frontend set and this replay, which detokenizes nothing, leaves off."""
    return (path.startswith("core.") and path not in replayed) or path.endswith(".sampling_params.detokenize")


def dump_diff(served: dict[str, Any], replayed: dict[str, Any]) -> dict[str, Any]:
    """`boundary.diff` of the leaves but those `by_construction`, which are counted by pattern apart."""
    skip = {p for p in set(served) | set(replayed)
            if served.get(p) != replayed.get(p) and by_construction(p, served, replayed)}
    out = B.diff({p: v for p, v in served.items() if p not in skip}, {p: v for p, v in replayed.items() if p not in skip})
    out["by_construction"] = dict(Counter(map(B.pattern, skip)))
    out["clean"] = out["differing"] == 0 and out["only_served_n"] == 0 and out["only_restored_n"] == 0
    return out


def step_diff(want: dict[int, dict], got: dict[int, dict]) -> dict[str, Any]:
    steps = sorted(set(want) | set(got))
    bad = [s for s in steps if {f: want.get(s, {}).get(f) for f in STEP_FIELDS} != {f: got.get(s, {}).get(f) for f in STEP_FIELDS}]
    first = None
    if bad:
        s = bad[0]
        first = {"step": s, "fields": [f for f in STEP_FIELDS if want.get(s, {}).get(f) != got.get(s, {}).get(f)],
                 "served": {f: want.get(s, {}).get(f) for f in STEP_FIELDS}, "replayed": {f: got.get(s, {}).get(f) for f in STEP_FIELDS}}
    return {"steps_served": len(want), "steps_replayed": len(got), "steps_differing": len(bad),
            "rows_served": sum(len(r.get("rows", ())) for r in want.values()), "first": first}


def fingerprint_diff(a: Any, b: Any, limit: int = 50) -> dict[str, Any]:
    import numpy as np

    if a is None or b is None or a.shape != b.shape:
        return {"compared": False}
    rows = np.nonzero((a != b).any(axis=1))[0]
    return {"compared": True, "blocks": int(a.shape[0]), "blocks_differing": int(rows.size), "first": rows[:limit].tolist(),
            "layers_differing_first_block": np.nonzero(a[rows[0]] != b[rows[0]])[0].tolist() if rows.size else []}


def run(a: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import torch

    sys.path.insert(0, str(ROOT / "adapters/vllm"))
    sys.path.insert(0, str(HERE))
    from servetax_plugin import step_outputs_hook
    from verity_vllm.engine.hooks import Hooks
    from verity_vllm.engine.step_trace import StepTrace

    meta = json.loads((a.arm_dir / "meta.json").read_text())
    kv_dir = Path(a.kv_dir or (meta.get("edges") or {}).get("kv_dir") or a.arm_dir / "server" / "edges")
    w = served(a.arm_dir, kv_dir, a.from_edge, a.to_edge)
    st0 = w["state"][a.from_edge]
    kv_file = kv_dir / f"edge-{a.from_edge}.kv"
    checkpoint = a.checkpoint or Path(meta["checkpoint"])
    log_stats = "--disable-log-stats" not in meta["server_argv"]

    t = time.monotonic()
    eng = engine(checkpoint, w["header"], a.eager, log_stats)
    out: dict[str, Any] = {"engine_ready_s": time.monotonic() - t}
    core = eng.engine_core.engine_core
    sched, _kvm, pool, runner = B.parts(core)
    trace = StepTrace().install(eng)
    kv = B.KvPool(runner.kv_caches, len(pool.blocks))
    if kv.layout() != st0["kv_layout"]:
        raise SystemExit("the fresh engine's KV pool is laid out unlike the served one's")
    if a.warm:
        t = time.monotonic()
        out["warm_steps"] = warm(eng, core, w["arrive"][:a.warm])
        out["warm_s"] = time.monotonic() - t

    torch.cuda.synchronize()
    t = time.monotonic()
    out["kv_load_bytes"] = kv.load(str(kv_file))
    out["kv_load_s"] = time.monotonic() - t
    t = time.monotonic()
    missing = B.restore(core, st0)
    trace.scheduled = st0["next_step"]
    torch.cuda.synchronize()
    out["state_restore_s"] = time.monotonic() - t
    out["restore_s"] = out["kv_load_s"] + out["state_restore_s"]
    out["runner_paths_missing"] = missing
    t = time.monotonic()
    out["restored"] = {"dump": dump_diff(w["dumps"][a.from_edge], B.dump_core(core)),
                       "pool": B.pool_diff(st0["pool"], B.pool_state(pool))}
    kv0 = kv.take()
    out["restored"]["kv_root_equal"] = kv0["root"] == st0["kv"]["root"]
    out["restored"]["check_s"] = time.monotonic() - t

    busy = B.Busy(lambda: torch.cuda.Event(enable_timing=True), first=st0["next_step"])
    runner._verity_busy = busy
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    B.busy_runner(GPUModelRunner, lambda r: getattr(r, "_verity_busy", None))
    outs: dict[int, dict] = {}
    hooks = Hooks(tag="serve-tax-restore")
    hooks.wrap(sched, "update_from_output", step_outputs_hook(trace, lambda rec: outs.__setitem__(rec["step"], rec["out"]),
                                                              lambda: True))
    pending = sorted(w["arrive"], key=lambda r: r["before_step"])

    def add_due() -> None:
        while pending and pending[0]["before_step"] == trace.scheduled:
            r = pending.pop(0)
            eng.add_request(r["engine_id"], {"prompt_token_ids": r["prompt_ids"]}, params(r))
        if pending and pending[0]["before_step"] < trace.scheduled:
            raise SystemExit(f"the replay passed step {pending[0]['before_step']} before {pending[0]['engine_id']} arrived")

    def schedule(orig):
        def wrapped(*args, **kw):
            add_due()
            return orig(*args, **kw)
        return wrapped

    hooks.wrap(sched, "schedule", schedule)
    taken: dict[str, Any] = {}

    def take(e: B.Edges, info: dict) -> None:
        torch.cuda.synchronize()
        taken["drained_t"] = info["drained_t"]
        for r in pending:
            eng.add_request(r["engine_id"], {"prompt_token_ids": r["prompt_ids"]}, params(r))
        taken["added_at_edge"] = len(pending)
        pending.clear()
        taken["state"] = B.capture(core)
        taken["dump"] = B.dump_core(core)
        taken["kv"] = kv.take()
        busy.reap(wait=True)

    edges = B.Edges(core, B.at_steps([w["hi"]], lambda: trace.scheduled), take, time.monotonic)
    stepper = type(core).step_with_batch_queue
    core.step_fn = lambda: edges.step(stepper)
    start = time.monotonic()
    while not taken:
        if not (sched.has_requests() or core.batch_queue):
            add_due()
            if not sched.has_requests():
                raise SystemExit(f"the replay is idle at step {trace.scheduled} short of {w['hi']}")
        eng.step()
    out["replay_wall_s"] = taken["drained_t"] - start
    busy_s, busy_steps = busy.total(w["lo"], w["hi"])
    srv_edge = w["edges"][a.to_edge]
    out |= {"replay_busy_s": busy_s, "replay_busy_steps": busy_steps, "served_busy_s": srv_edge["since_last"]["busy_s"],
            "served_busy_steps": srv_edge["since_last"]["busy_steps"], "served_wall_s": srv_edge["since_last"]["wall_s"]}
    out["replay_over_served_busy"] = busy_s / out["served_busy_s"] if out["served_busy_s"] else None
    served_steps_s = srv_edge["since_last"].get("busy_steps_s")
    out["busy_by_step"] = B.busy_compare(served_steps_s, busy.steps(w["lo"], w["hi"])) if served_steps_s else None
    steps = {r["step"]: r for r in trace.records if r["kind"] == "step" and w["lo"] <= r["step"] < w["hi"]}
    fp_served = np.load(a.arm_dir / "server" / "edges" / f"edge-{a.to_edge}.fp.npy")
    out |= {"schedule": step_diff(w["steps"], steps), "outputs": B.outputs_diff(w["outputs"], outs),
            "far_edge": {"kv_root_equal": taken["kv"]["root"] == w["state"][a.to_edge]["kv"]["root"],
                         "kv_fingerprints": fingerprint_diff(fp_served, taken["kv"]["fingerprint"]),
                         "pool": B.pool_diff(w["state"][a.to_edge]["pool"], taken["state"]["pool"]),
                         "dump": dump_diff(w["dumps"][a.to_edge], taken["dump"]),
                         "added_at_edge": taken["added_at_edge"]}}
    far, near = out["far_edge"], out["restored"]
    out["bit_exact"] = (near["kv_root_equal"] and not near["pool"] and near["dump"]["clean"]
                        and out["schedule"]["steps_differing"] == 0 and out["outputs"]["first"] is None
                        and far["kv_root_equal"] and not far["pool"] and far["dump"]["clean"])
    hooks.uninstall()
    (a.out / f"replay-edge-{a.to_edge}.dump.json").write_text(json.dumps(taken["dump"]))
    with open(a.out / "replay_outputs.jsonl", "w") as f:
        for s in sorted(outs):
            f.write(json.dumps({"step": s, "out": outs[s]}, separators=(",", ":")) + "\n")
    with open(a.out / "replay_steps.jsonl", "w") as f:
        for s in sorted(steps):
            f.write(json.dumps(steps[s], separators=(",", ":")) + "\n")
    return out | {"replay_block": {"from_edge": a.from_edge, "to_edge": a.to_edge, "from_step": w["lo"], "to_step": w["hi"],
                                   "arrivals": len(w["arrive"]), "requests_at_edge": len(st0["scheduler"]["requests"])}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--arm-dir", type=Path, required=True)
    ap.add_argument("--from-edge", type=int, default=0)
    ap.add_argument("--to-edge", type=int, default=1)
    ap.add_argument("--kv-dir", type=Path, help="where the served arm wrote the KV pool's bytes (default: its edges/)")
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--warm", type=int, default=0, metavar="N",
                    help="serve the replay block's first N arrivals, under other ids, before the restore (`warm`)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if sys.flags.hash_randomization:
        raise SystemExit("run with PYTHONHASHSEED=0, as the served engine did: the scheduler iterates sets of request ids")
    os.environ.update(VLLM_ENABLE_V1_MULTIPROCESSING="0", VLLM_DISABLE_REQUEST_ID_RANDOMIZATION="1")
    a.out.mkdir(parents=True, exist_ok=True)
    got = run(a)
    rec = {"schema": "verity-bench/serve-tax/restore/v0", "arm_dir": str(a.arm_dir), "run_id": SHARED.RESEARCH_RUN_ID,
           "mode": "eager" if a.eager else "compiled", **got}
    (a.out / "restore.json").write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps({k: rec.get(k) for k in ("bit_exact", "restore_s", "kv_load_s", "state_restore_s", "replay_wall_s",
                                               "replay_busy_s", "served_busy_s")} |
                     {"restored_dump_clean": rec["restored"]["dump"]["clean"],
                      "steps_differing": rec["schedule"]["steps_differing"], "outputs_first": rec["outputs"]["first"],
                      "far_kv_root_equal": rec["far_edge"]["kv_root_equal"], "far_dump_clean": rec["far_edge"]["dump"]["clean"],
                      "far_dump_patterns": rec["far_edge"]["dump"]["patterns"]}, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
