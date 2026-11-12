"""GPU run: a stand-in for GLM-5.3 served with its multi-token-prediction draft on (vLLM's ``speculative_config`` method
``mtp``), then replayed from the scheduler's step trace alone; the replay's steps and every token are held to the served ones.

The stand-in is GLM-5.3's registered config (`CONFIG`) cut to its first `LAYERS` layers and `EXPERTS` routed experts, with
its MTP layer, served tensor-parallel 1 on one card from vLLM's dummy weights (``load_format="dummy"``), without a tokenizer.
``--weights dummy`` keeps vLLM's values, under which a token depends on little but the token before it; ``--weights scaled``
redraws every weight at `STD` after the load (`rescale`), so a token depends on its whole context and one differing bit shows.

Each arm of `ARMS` is one speculative config (``off`` has none). Each phase is a process of its own:

- ``--phase serve``: an engine with the arm's config under the step trace (`verity_vllm.engine.step_trace.StepTrace`). The
  requests of `plan` (greedy and sampled, each under its own seed, some sharing a cached prefix, more than fit one batch)
  arrive between engine iterations. It writes the trace, each request's tokens and the draft tokens each step scheduled.
- ``--phase replay``: an engine built from the trace's header (the scheduler's settings and its ``speculative`` entry), each
  request put back before the step the trace says it arrived before; the scheduler decides the rest. It writes the same.

Without ``--phase`` it runs the serve and the replay of each of ``--runs`` (`RUNS`: weights and arm) in their order, compares
each replay with its serve (steps, prefix-cache hits, draft counts, tokens) and each draft arm's tokens with ``off``'s under
the same weights, rewrites OUT/specdec_replay.json after each and exits 0 only if every arm took a prefix-cache hit, every
draft arm scheduled draft tokens and every replay equals its serve. A phase that fails is recorded and the rest go on.

In the pod venv, on one idle card, from the tree's root:

    PYTHONHASHSEED=0 $PY adapters/vllm/tests/properties/specdec_replay_gpu.py --out DIR [--runs dummy:off,dummy:mtp1,...]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

ADAPTER = Path(__file__).resolve().parents[2]
CONFIG = ADAPTER / "verity_vllm/engine/profiles/hf_configs/GLM53.config.json"
#: the stand-in's decoder layers (GLM-5.3's three dense layers, then MoE layers with a shared and an own index) and experts
LAYERS, EXPERTS = 7, 16
#: ``--weights scaled``: every weight's standard deviation, and the scale beside an FP8 weight sampled uniform on [-1, 1]
STD = 0.02
FP8_SCALE = STD * 3 ** 0.5
#: one speculative config an arm; ``synthetic`` accepts by a rate instead of the model's probabilities, so a stand-in whose
#: draft rarely agrees with its target still takes accepted draft tokens through the cache and the scheduler
ARMS: dict[str, dict[str, Any] | None] = {
    "off": None,
    "mtp1": {"method": "mtp", "num_speculative_tokens": 1},
    "mtp3": {"method": "mtp", "num_speculative_tokens": 3},
    "mtp5": {"method": "mtp", "num_speculative_tokens": 5},
    "synthetic3": {"method": "mtp", "num_speculative_tokens": 3, "rejection_sample_method": "synthetic",
                   "synthetic_acceptance_rates": [0.8, 0.6, 0.4]},
}
#: the default ``--runs``, in the order they run: the redrawn weights with the draft off, which a draft arm's tokens are
#: compared with, then with GLM-5.3's five draft tokens, one and three; vLLM's dummy weights, whose answer stands if redrawing
#: the weights fails; the arms that accept by a rate last
RUNS = "scaled:off,scaled:mtp5,scaled:mtp1,scaled:mtp3,dummy:off,dummy:mtp1,scaled:synthetic3,dummy:synthetic3"
#: the served engine's settings; the replay's are the trace header's
SERVE = {"max_model_len": 4096, "max_num_seqs": 8, "max_num_batched_tokens": 512, "enable_chunked_prefill": True,
         "enable_prefix_caching": True, "async_scheduling": None, "block_size": None, "num_gpu_blocks": 8192}
REQUESTS = 14
#: what of a step record the replay must repeat
STEP_KEYS = ("T", "order", "rows", "new", "resumed", "cache_hit", "preempted", "emitted", "finished")


def stand_in(where: Path) -> None:
    cfg = json.loads(CONFIG.read_text())
    cut = {**cfg, "num_hidden_layers": LAYERS, "n_routed_experts": EXPERTS}
    for k in ("indexer_types", "mlp_layer_types"):
        cut[k] = cfg[k][:LAYERS]
    (where / "config.json").write_text(json.dumps(cut, indent=1) + "\n")


def plan(default_seed: int) -> list[dict[str, Any]]:
    """The requests, the same in every arm: its id, the engine iteration it arrives before, its prompt and its sampling."""
    rng = random.Random(0)

    def ids(n: int) -> list[int]:
        return [rng.randrange(1000, 150000) for _ in range(n)]

    shared = ids(160)
    out = []
    for i in range(REQUESTS):
        n = rng.choice((24, 70, 150, 330, 610))
        prompt = shared + ids(max(n - len(shared), 8)) if i % 3 == 2 else ids(n)
        sampling: dict[str, Any] = {"temperature": 0.0, "seed": default_seed}
        if i % 4 == 1:
            sampling = {"temperature": 0.8, "top_k": 50, "seed": 1000 + i}
        if i % 4 == 3:
            # GLM-5.3's own generation config
            sampling = {"temperature": 1.0, "top_p": 0.95, "seed": 1000 + i}
        out.append({"id": f"specdec-{i:02d}", "iteration": 2 * i, "prompt_ids": prompt, "sampling": sampling,
                    "max_tokens": rng.choice((24, 40, 56))})
    return out


def engine(model: Path, cfg: dict[str, Any], eager: bool) -> Any:
    from vllm import EngineArgs
    from vllm.v1.engine.llm_engine import LLMEngine

    args = EngineArgs(model=str(model), load_format="dummy", skip_tokenizer_init=True, generation_config="vllm",
                      dtype="bfloat16", seed=0, trust_remote_code=True, enforce_eager=eager,
                      max_model_len=cfg["max_model_len"], max_num_seqs=cfg["max_num_seqs"],
                      max_num_batched_tokens=cfg["max_num_batched_tokens"], enable_chunked_prefill=cfg["enable_chunked_prefill"],
                      enable_prefix_caching=cfg["enable_prefix_caching"], async_scheduling=cfg["async_scheduling"],
                      block_size=cfg["block_size"], num_gpu_blocks_override=cfg["num_gpu_blocks"], disable_log_stats=True,
                      speculative_config=cfg.get("speculative"),
                      **{k: cfg[k] for k in ("long_prefill_token_threshold", "kv_cache_dtype") if k in cfg})
    return LLMEngine.from_engine_args(args)


def rescale(eng: Any) -> dict[str, int]:
    """Redraw the target's and the draft's weights from one seeded generator, in the modules' own order: a norm's gain 1, a
    quantization scale `FP8_SCALE`, an FP8 weight uniform on [-1, 1], every other float weight normal at `STD`."""
    import torch

    from verity_vllm.engine import handle

    runner = handle.model_runner(eng)
    gen = torch.Generator(device=runner.device)
    gen.manual_seed(1234)
    seen: set[int] = set()
    count = {"norm": 0, "scale": 0, "fp8": 0, "float": 0, "draft_only": 0}
    models = [runner.model, getattr(getattr(runner, "speculator", None), "model", None)]
    with torch.no_grad():
        for model in models:
            for name, p in (model.named_parameters() if model is not None else ()):
                if id(p) in seen or not torch.is_floating_point(p):
                    continue
                seen.add(id(p))
                count["draft_only"] += model is not runner.model
                if "scale" in name:
                    p.fill_(FP8_SCALE)
                    count["scale"] += 1
                elif "norm" in name and p.ndim == 1:
                    p.fill_(1.0)
                    count["norm"] += 1
                elif torch.finfo(p.dtype).bits < 16:
                    fresh = torch.empty(p.shape, dtype=torch.float16, device=p.device).uniform_(-1.0, 1.0, generator=gen)
                    p.copy_(fresh.to(p.dtype))
                    count["fp8"] += 1
                else:
                    p.normal_(0.0, STD, generator=gen)
                    count["float"] += 1
    return count


def params(sampling: dict[str, Any], max_tokens: int) -> Any:
    from vllm import SamplingParams

    return SamplingParams(n=1, temperature=sampling.get("temperature", 0.0), top_p=sampling.get("top_p", 1.0),
                          top_k=sampling.get("top_k", 0), seed=sampling.get("seed"), max_tokens=max_tokens, ignore_eos=True,
                          detokenize=False)


def arrivals(recs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Where a trace's records say each request arrived, for `drive`'s replay."""
    return [{"id": r["request"], "before_step": r["before_step"], "after_empty": r.get("after_empty", 0)}
            for r in recs if r["kind"] == "arrive"]


def drive(eng: Any, trace: Any, phase: str, due: list[dict[str, Any]], add: Callable[[str], None]) -> dict[str, Any]:
    """Step ``eng``, whose scheduler ``trace`` is installed on, until every request of ``due`` has arrived (``add``) and
    finished and every step the scheduler scheduled has its record. On the serve a request arrives before the engine
    iteration its entry names, or at once where the engine is idle. On the replay it goes back where the trace says
    (`arrivals`), before the first schedule at its step: one that came after a schedule that scheduled nothing waits for
    such a schedule, or for the engine to be idle, so the step in flight has finished its requests and cached their blocks
    as it had when the request was served. What came out: the trace's steps, each request's tokens, the draft tokens each
    step scheduled for each request, and how many schedules scheduled nothing.

    A step gets its record when its output is processed. Under a draft the scheduler schedules a request's next row before
    the output that ends it is back, so the engine can hold a step when no request is unfinished; vLLM's own loop steps
    while its batch queue holds one, and so does this, until the trace has a record for every step scheduled. An engine
    that holds no output for a scheduled step is refused at the second engine step in a row that processes none, instead
    of being stepped forever."""
    from verity_vllm.engine.hooks import Hooks

    due = list(due)
    drafted: dict[int, dict[str, int]] = {}
    tokens: dict[str, list[int]] = {}
    empty = 0

    def put_back(idle: bool) -> None:
        while due and due[0]["before_step"] == trace.scheduled and (idle or not due[0]["after_empty"] or trace.empty):
            add(due.pop(0)["id"])
        if due and due[0]["before_step"] < trace.scheduled:
            raise SystemExit(f"specdec-replay: the replay passed step {due[0]['before_step']} before {due[0]['id']} arrived")

    def schedule(orig: Any) -> Any:
        def wrapped(*args: Any, **kw: Any) -> Any:
            nonlocal empty
            if phase == "replay":
                put_back(idle=False)
            before = trace.scheduled
            so = orig(*args, **kw)
            if trace.scheduled > before:
                drafted[before] = {rid: len(ids) for rid, ids in (getattr(so, "scheduled_spec_decode_tokens", None) or {}).items()}
            else:
                empty += 1
            return so
        return wrapped

    hooks = Hooks(tag="specdec-replay")
    hooks.wrap(trace.sched, "schedule", schedule)
    iteration = unprocessed = 0
    while due or eng.has_unfinished_requests() or trace.executed < trace.scheduled:
        if phase == "serve":
            idle = not eng.has_unfinished_requests()
            while due and (due[0]["iteration"] <= iteration or idle):
                add(due.pop(0)["id"])
                idle = False
        elif due and not eng.has_unfinished_requests():
            put_back(idle=True)
            if not eng.has_unfinished_requests():
                raise SystemExit(f"specdec-replay: the replay is idle at step {trace.scheduled} with {due[0]['id']} due before "
                                 f"step {due[0]['before_step']}")
        owed = trace.executed if not due and not eng.has_unfinished_requests() else None
        for o in eng.step():
            if o.finished:
                tokens[o.request_id] = [int(t) for t in o.outputs[0].token_ids]
        iteration += 1
        unprocessed = unprocessed + 1 if owed == trace.executed else 0
        if unprocessed > 1:
            raise SystemExit(f"specdec-replay: step {owed} was scheduled and the engine holds no output for it")
    hooks.uninstall()
    return {"steps": [r for r in trace.records if r["kind"] == "step"], "drafted": {str(s): d for s, d in sorted(drafted.items())},
            "tokens": tokens, "empty_schedules": empty,
            "arrived_after_an_empty_schedule": {r["request"]: r["after_empty"] for r in trace.records
                                                if r["kind"] == "arrive" and "after_empty" in r}}


def run(a: argparse.Namespace) -> dict[str, Any]:
    """One engine: the serve of ``a.arm``, or the replay of the serve in ``a.dir``."""
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("specdec-replay: PYTHONHASHSEED=0 (a set's order decides a batch's row order)")
    os.environ["VLLM_DISABLE_REQUEST_ID_RANDOMIZATION"] = "1"
    sys.path.insert(0, str(ADAPTER))
    from verity_vllm.engine import env, served_seed
    from verity_vllm.engine.step_trace import StepTrace, load

    env.apply_engine(None)
    out = Path(a.dir)
    reqs = {r["id"]: r for r in plan(served_seed.DEFAULT_SEED)}
    if a.phase == "serve":
        cfg = {**SERVE, "speculative": ARMS[a.arm]}
        due = [{"id": r["id"], "iteration": r["iteration"]} for r in reqs.values()]
        (out / "trace.jsonl").unlink(missing_ok=True)
    else:
        recs = load(out / "trace.jsonl")
        cfg = recs[0]["config"]
        due = arrivals(recs)
    model = Path(tempfile.mkdtemp(prefix="specdec-stand-in-"))
    stand_in(model)
    eng = engine(model, cfg, eager=not a.compiled)
    redrawn = rescale(eng) if a.weights == "scaled" else None
    trace = StepTrace(out / "trace.jsonl" if a.phase == "serve" else None).install(eng)

    def add(rid: str) -> None:
        r = reqs[rid]
        got = eng.add_request(rid, {"prompt_token_ids": r["prompt_ids"]}, params(r["sampling"], r["max_tokens"]))
        if got not in (None, rid):
            raise SystemExit(f"specdec-replay: the engine named {rid} {got}")

    doc = {"phase": a.phase, "arm": a.arm, "weights": a.weights, "compiled": a.compiled, "config": trace.records[0]["config"],
           "redrawn": redrawn, **drive(eng, trace, a.phase, due, add), "sampling": {rid: r["sampling"] for rid, r in reqs.items()}}
    trace.uninstall()
    (out / f"{a.phase}.json").write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    return doc


def acceptance(doc: dict[str, Any]) -> dict[str, Any]:
    """Over the rows that verified draft tokens: the draft tokens scheduled, those accepted (a row's emitted tokens less the
    one the target samples), how many rows accepted each count, and the rows that break the record's reading of a draft: a row
    whose tokens are not one more than its drafts, or that emitted fewer than one token or more than all of them.

    ``first_position``: where the trace puts a row's first position against the true one, over every row after a request's
    first. The true one is the request's first position plus what each earlier row kept: all of a row without drafts, the
    emitted tokens of one with. A row scheduled while the row before was still running is ``ahead_by_rejected``: the
    scheduler had counted every draft of that row as kept.

    And the prefix-cache tokens each request started from."""
    rows = scheduled = accepted = 0
    by_count: dict[str, int] = {}
    broken = {"tokens_not_one_more_than_drafts": 0, "emitted_outside_one_to_all": 0}
    first = {"true": 0, "ahead_by_rejected": 0, "other": 0}
    true: dict[str, int] = {}
    rejected: dict[str, int] = {}
    for s in doc["steps"]:
        drafts = doc["drafted"].get(str(s["step"]), {})
        for rid, start, n in s["rows"]:
            d, e = drafts.get(rid, 0), s["emitted"].get(rid)
            if start is None:
                true.pop(rid, None)
            elif rid not in true or rid in s["new"] or rid in s["resumed"]:
                true[rid], rejected[rid] = start, 0
            else:
                ahead = start - true[rid]
                first["true" if not ahead else "ahead_by_rejected" if ahead == rejected[rid] else "other"] += 1
            if d and e is not None:
                rows += 1
                scheduled += d
                accepted += e - 1
                by_count[str(e - 1)] = by_count.get(str(e - 1), 0) + 1
                broken["tokens_not_one_more_than_drafts"] += n != 1 + d
                broken["emitted_outside_one_to_all"] += not 1 <= e <= 1 + d
            if rid in true:
                kept = n if not d else e
                if kept is None:
                    # a row of a request that finished at the step before, which async scheduling still ran
                    del true[rid]
                else:
                    true[rid], rejected[rid] = true[rid] + kept, n - kept
    return {"rows_with_drafts": rows, "draft_tokens": scheduled, "accepted": accepted, "rows_by_accepted": by_count, **broken,
            "first_position": first, "cache_hits": {rid: n for s in doc["steps"] for rid, n in s["cache_hit"].items()},
            "distinct_tokens": len({t for ids in doc["tokens"].values() for t in ids})}


def token_diff(want: dict[str, list[int]], got: dict[str, list[int]], sampling: dict[str, dict]) -> dict[str, Any]:
    out: dict[str, Any] = {"greedy": {"requests": 0, "equal": 0}, "sampled": {"requests": 0, "equal": 0}, "differing": []}
    for rid, x in sorted(want.items()):
        kind = out["sampled" if sampling[rid].get("temperature") else "greedy"]
        y = got.get(rid, [])
        kind["requests"] += 1
        if x == y:
            kind["equal"] += 1
            continue
        first = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
        out["differing"].append({"request": rid, "tokens": len(x), "tokens_other": len(y), "first_position": first})
    return out


def compare(serve: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    s, r = serve["steps"], replay["steps"]
    first = next((i for i, (x, y) in enumerate(zip(s, r)) if any(x[k] != y[k] for k in STEP_KEYS)), None)
    if first is None and len(s) != len(r):
        first = min(len(s), len(r))
    tok = token_diff(serve["tokens"], replay["tokens"], serve["sampling"])
    equal = first is None and serve["drafted"] == replay["drafted"] and serve["tokens"] == replay["tokens"]
    return {"equal": equal, "steps": len(s), "steps_replayed": len(r), "first_step_differing": first,
            "drafted_equal": serve["drafted"] == replay["drafted"], "requests": len(serve["tokens"]),
            "tokens": sum(map(len, serve["tokens"].values())), "tokens_differing": tok["differing"],
            "empty_schedules": serve["empty_schedules"], "arrived_after_an_empty_schedule": serve["arrived_after_an_empty_schedule"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", default=RUNS, help="WEIGHTS:ARM,... (dummy or scaled; an arm of ARMS)")
    ap.add_argument("--weights", choices=("dummy", "scaled"), default="dummy")
    ap.add_argument("--compiled", action="store_true", help="vLLM's defaults (torch.compile, CUDA graphs) instead of eager")
    ap.add_argument("--phase", choices=("serve", "replay"))
    ap.add_argument("--arm", choices=tuple(ARMS))
    ap.add_argument("--dir")
    a = ap.parse_args()
    if a.phase:
        doc = run(a)
        print(f"specdec-replay: {a.weights} {a.arm} {a.phase}: {len(doc['steps'])} steps, {len(doc['tokens'])} requests, "
              f"{sum(map(len, doc['tokens'].values()))} tokens, drafts {acceptance(doc)}", flush=True)
        return 0
    out = Path(a.out)
    env = {**os.environ, "PYTHONHASHSEED": "0"}
    report: dict[str, Any] = {"layers": LAYERS, "experts": EXPERTS, "compiled": a.compiled, "runs": {}}
    ok = True
    runs = [tuple(r.split(":")) for r in a.runs.split(",")]
    unknown = [r for r in runs if len(r) != 2 or r[0] not in ("dummy", "scaled") or r[1] not in ARMS]
    if unknown:
        raise SystemExit(f"specdec-replay: --runs names {unknown}: WEIGHTS:ARM, dummy or scaled, an arm of {list(ARMS)}")
    off: dict[str, dict[str, Any]] = {}
    for weights, arm in runs:
        d = out / f"{weights}-{arm}"
        d.mkdir(parents=True, exist_ok=True)
        entry: dict[str, Any] = {"speculative": ARMS[arm]}
        report["runs"][f"{weights}-{arm}"] = entry
        for phase in ("serve", "replay"):
            argv = [sys.executable, __file__, "--out", str(out), "--phase", phase, "--arm", arm, "--dir", str(d),
                    "--weights", weights, *(["--compiled"] if a.compiled else [])]
            with (d / f"{phase}.log").open("w") as log:
                rc = subprocess.run(argv, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            tail = (d / f"{phase}.log").read_text(errors="replace").splitlines()[-1:]
            print(f"== {weights} {arm} {phase}: rc {rc}: {tail[0] if tail else ''}", flush=True)
            entry[phase] = {"rc": rc}
            if rc:
                entry[phase]["log_tail"] = (d / f"{phase}.log").read_text(errors="replace").splitlines()[-25:]
                ok = False
                break
        if entry["serve"]["rc"] == 0:
            serve = json.loads((d / "serve.json").read_text())
            entry.update(config=serve["config"], redrawn=serve["redrawn"], drafts=acceptance(serve))
            ok = ok and bool(entry["drafts"]["cache_hits"]) and (ARMS[arm] is None or entry["drafts"]["draft_tokens"] > 0)
            if arm == "off":
                off[weights] = serve
            elif weights in off:
                entry["tokens_against_off"] = token_diff(off[weights]["tokens"], serve["tokens"], serve["sampling"])
        if entry.get("replay", {}).get("rc") == 0:
            entry["replay_equal"] = compare(serve, json.loads((d / "replay.json").read_text()))
            ok = ok and entry["replay_equal"]["equal"]
        report["ok"] = ok
        (out / "specdec_replay.json").write_text(json.dumps(report, indent=1) + "\n")
    for name, e in report["runs"].items():
        print(f"{name}: drafts {e.get('drafts')} replay {e.get('replay_equal')} against off {e.get('tokens_against_off')}", flush=True)
    print(f"specdec-replay: {'ok' if ok else 'FAILED'}: {out / 'specdec_replay.json'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
