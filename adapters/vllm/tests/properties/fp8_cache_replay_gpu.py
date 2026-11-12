"""GPU run: a stand-in for GLM-5.3 served with an FP8 KV cache (vLLM's ``kv_cache_dtype``) and the prefix cache on, under the
scheduler's step trace, then replayed from the trace's header alone; the replay's steps, every token and every cached row's
bytes are held to the served ones.

The stand-in is GLM-5.3's registered config (`CONFIG`) cut to its first `LAYERS` layers and `EXPERTS` routed experts, served
tensor-parallel 1 on one card from vLLM's dummy weights (``load_format="dummy"``), without a tokenizer. ``--weights dummy``
keeps vLLM's values; ``--weights scaled`` redraws every weight at `STD` after the load (`rescale`), so a token depends on its
whole context and one differing bit of a cached row shows in the tokens.

Each arm of `ARMS` is one cache format, as a launch line names it (``auto`` is the bf16 cache). Each phase is a process of
its own:

- ``--phase serve``: an engine with the arm's cache format under the step trace (`verity_vllm.engine.step_trace.StepTrace`).
  The requests of `plan` (greedy and sampled, each under its own seed, some sharing a cached prefix, one longer than the
  indexer's top-k, more than fit one batch) arrive between engine iterations. After each step's forward, `Rows` reads at
  every layer with a KV cache the rows the step wrote: the layer's cache at the step's slots. The phase writes the trace, each
  request's tokens, a digest of each request's rows at each layer, and what the engine chose (`chosen`): the cache format its
  config holds once built, and each layer's attention backend, cache dtype and row bytes.
- ``--phase replay``: an engine built from the trace's header, each request put back before the step the trace says it
  arrived before; the scheduler decides the rest. It writes the same. ``--cache named`` gives the engine the cache format
  the served engine's config held instead, for a tree whose header does not carry the format.

What of ours meets the engine is recorded with the serve (``ours``): whether the trace's header holds the cache format, what
the history tables' guard (`verity_vllm.pipeline.boundary_tables.mla_layers`) says of the model, and, where the guard passes,
the tables' own step hook beside `Rows`: every row it read must be `Rows`' bytes.

Without ``--phase`` it runs the serve and the replay of each of ``--runs`` (`RUNS`: weights and arm), compares each replay
with its serve (steps, prefix-cache hits, tokens, rows), writes OUT/fp8_cache_replay.json and exits 0 only if every serve took
a prefix-cache hit, every arm but ``auto`` held byte rows, and every replay from the header equals its serve. Where the header
holds no cache format and the arm's is not ``auto``, the replay from the header runs another engine than the served one; the
run then replays once more with ``--cache named`` and reports both, and still fails. A phase that fails is recorded and the
rest go on. The rows' bytes stay in a scratch directory, removed at the end; OUT holds digests.

In the pod venv, on one idle card, from the tree's root:

    PYTHONHASHSEED=0 $PY adapters/vllm/tests/properties/fp8_cache_replay_gpu.py --out DIR [--runs dummy:auto,dummy:fp8,...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

ADAPTER = Path(__file__).resolve().parents[2]
CONFIG = ADAPTER / "verity_vllm/engine/profiles/hf_configs/GLM53.config.json"
#: the stand-in's decoder layers (GLM-5.3's three dense layers, then MoE layers with a shared and an own index) and experts
LAYERS, EXPERTS = 7, 16
#: ``--weights scaled``: every weight's standard deviation, and the scale beside an FP8 weight drawn uniform on [-1, 1]
STD = 0.02
FP8_SCALE = STD * 3 ** 0.5
#: an arm -> the ``kv_cache_dtype`` its served engine is given (None: vLLM's default, ``auto``)
ARMS: dict[str, str | None] = {"auto": None, "fp8": "fp8", "fp8_ds_mla": "fp8_ds_mla", "fp8_e4m3": "fp8_e4m3"}
#: the default ``--runs``: vLLM's dummy weights first, whose answer stands if redrawing the weights fails; ``fp8`` is what
#: vLLM's launch line for GLM-5.3 names, ``auto`` the bf16 control
RUNS = "dummy:auto,dummy:fp8,scaled:fp8,scaled:auto"
#: the served engine's settings; the replay's are the trace header's
SERVE = {"max_model_len": 4096, "max_num_seqs": 8, "max_num_batched_tokens": 512, "enable_chunked_prefill": True,
         "enable_prefix_caching": True, "async_scheduling": None, "block_size": None, "num_gpu_blocks": 8192}
REQUESTS = 12
#: a prompt longer than GLM-5.3's ``index_topk`` (2048), so that its last steps attend a strict subset of its cached rows
LONG = 2200
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

    shared = ids(200)
    out = []
    for i in range(REQUESTS):
        n = LONG if i == 4 else rng.choice((24, 70, 150, 330, 610))
        prompt = shared + ids(max(n - len(shared), 8)) if i % 3 == 2 else ids(n)
        sampling: dict[str, Any] = {"temperature": 0.0, "seed": default_seed}
        if i % 2:
            # GLM-5.3's own generation config
            sampling = {"temperature": 1.0, "top_p": 0.95, "seed": 1000 + i}
        out.append({"id": f"fp8-{i:02d}", "iteration": 3 * i, "prompt_ids": prompt, "sampling": sampling,
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
                      **({"kv_cache_dtype": cfg["kv_cache_dtype"]} if cfg.get("kv_cache_dtype") else {}))
    return LLMEngine.from_engine_args(args)


def rescale(runner: Any) -> dict[str, int]:
    """Redraw the model's weights from one seeded generator, in the modules' own order: a norm's gain 1, a quantization
    scale `FP8_SCALE`, an FP8 weight uniform on [-1, 1], every other float weight normal at `STD`."""
    import torch

    gen = torch.Generator(device=runner.device)
    gen.manual_seed(1234)
    count = {"norm": 0, "scale": 0, "fp8": 0, "float": 0}
    with torch.no_grad():
        for name, p in runner.model.named_parameters():
            if not torch.is_floating_point(p):
                continue
            if "scale" in name:
                p.fill_(FP8_SCALE)
                count["scale"] += 1
            elif "norm" in name and p.ndim == 1:
                p.fill_(1.0)
                count["norm"] += 1
            elif torch.finfo(p.dtype).bits < 16:
                draw = torch.empty(p.shape, dtype=torch.float16, device=p.device).uniform_(-1.0, 1.0, generator=gen)
                p.copy_(draw.to(p.dtype))
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


def kv_layers(model: Any) -> dict[str, Any]:
    """Layer name -> the module the engine bound that layer's KV cache to (the outermost under each name)."""
    out: dict[str, Any] = {}
    for _path, m in model.named_modules():
        name, cache = getattr(m, "layer_name", None), getattr(m, "kv_cache", None)
        if isinstance(name, str) and hasattr(cache, "stride") and cache.numel() and name not in out:
            out[name] = m
    return out


def chosen(eng: Any, runner: Any) -> dict[str, Any]:
    """What the built engine holds: its config's cache format and block size, and each layer with a KV cache."""
    cc = eng.vllm_config.cache_config
    layers = []
    for name, m in kv_layers(runner.model).items():
        c, backend = m.kv_cache, getattr(m, "attn_backend", None)
        layers.append({"layer": name, "module": type(m).__name__, "mla": getattr(m, "kv_lora_rank", None) is not None,
                       "kv_cache_dtype": getattr(m, "kv_cache_dtype", None),
                       "backend": getattr(backend, "__name__", None) if backend is not None else None,
                       "impl": type(m.impl).__name__ if getattr(m, "impl", None) is not None else None,
                       "dtype": str(c.dtype).removeprefix("torch."), "shape": list(c.shape),
                       "row_bytes": int(c.shape[-1]) * c.element_size()})
    return {"cache_dtype": str(cc.cache_dtype), "block_size": int(cc.block_size), "layers": layers}


class Rows:
    """Each request's cached rows as the engine wrote them: after a step's forward, every layer's cache at the step's
    slots (the layer's entry of the step's ``slot_mapping``), by request, at the positions the step computed. A layer whose
    cache is not ``[blocks, block_size, row]`` or has no slot mapping is named in ``skipped`` and not read."""

    def __init__(self, runner: Any) -> None:
        import torch

        self.torch, self.layers, self.skipped = torch, {}, {}
        for name, m in kv_layers(runner.model).items():
            if m.kv_cache.dim() == 3:
                self.layers[name] = m.kv_cache
            else:
                self.skipped[name] = f"a cache of shape {list(m.kv_cache.shape)}"
        self.got: dict[str, dict[str, list[tuple[int, Any]]]] = {}
        self.steps = 0

    def after(self, runner: Any) -> None:
        import numpy as np

        torch, state = self.torch, runner.execute_model_state
        ib, slots = state.input_batch, state.slot_mappings_by_layer
        n = len(ib.req_ids)
        qsl = np.asarray(ib.query_start_loc_np[:n + 1], dtype=np.int64)
        nct = np.asarray(ib.num_computed_tokens_np[:n], dtype=np.int64)
        for name, cache in self.layers.items():
            if name not in slots:
                self.skipped[name] = "no slot mapping at a step"
                continue
            s = slots[name][:int(qsl[n])].to(torch.int64)
            if bool((s < 0).any()):
                raise ValueError(f"step {self.steps}: a scheduled token has no slot at {name}")
            rows = cache[s // cache.shape[1], s % cache.shape[1]].contiguous()
            host = rows.view(torch.uint8).reshape(rows.shape[0], -1).cpu().numpy()
            for i, rid in enumerate(ib.req_ids[:n]):
                self.got.setdefault(str(rid), {}).setdefault(name, []).append((int(nct[i]), host[int(qsl[i]):int(qsl[i + 1])].copy()))
        self.steps += 1

    def digest(self) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
        """Each request's rows at each layer: its first computed position (the prefix-cache hit's length), how many it
        computed and their SHA-512; and the bytes, by ``request|layer``. A step that starts below what a request holds (a
        recompute after a preemption) writes over its rows from there."""
        import numpy as np

        out: dict[str, dict[str, Any]] = {}
        raw: dict[str, Any] = {}
        for rid, by in sorted(self.got.items()):
            out[rid] = {}
            for name, chunks in by.items():
                first, end = min(s for s, _ in chunks), max(s + len(c) for s, c in chunks)
                buf = np.zeros((end - first, chunks[0][1].shape[1]), np.uint8)
                for s, c in chunks:
                    buf[s - first:s - first + len(c)] = c
                out[rid][name] = {"first": first, "positions": end - first, "row_bytes": int(buf.shape[1]),
                                  "sha512": hashlib.sha512(buf.tobytes()).hexdigest()}
                raw[f"{rid}|{name}"] = buf
        return out, raw


def ours(runner: Any, header: dict[str, Any]) -> tuple[dict[str, Any], Any]:
    """What of ours says of this engine before it serves, and the history tables' step hook where their guard passes."""
    from verity_vllm.pipeline import boundary_tables as BT

    out: dict[str, Any] = {"header_kv_cache_dtype": header.get("kv_cache_dtype")}
    hook = None
    try:
        hook = BT.StepHook(runner.model)
        out["tables_guard"] = {"verdict": "ok", "layers": list(hook.layers)}
    except ValueError as e:
        out["tables_guard"] = {"verdict": "refuses", "message": str(e)}
    return out, hook


def tables_rows(hook: Any, raw: dict[str, Any]) -> dict[str, Any]:
    """The tables' hook against `Rows`: over the rows the hook read before it failed, if it did (it refuses a request that
    starts past what it holds, as one with a prefix-cache hit does), whether each is `Rows`' bytes."""
    lat = hook.latents
    out: dict[str, Any] = {"failed": lat.failed, "requests": 0, "rows": 0, "differing": []}
    for rid in lat.requests():
        k, mine = lat.length(rid), lat.take(rid)
        for x, (name, _cache) in hook.layers.items():
            buf = raw.get(f"{rid}|{name}")
            if buf is None or not k:
                continue
            out["rows"] += k
            if mine[x][:k].tobytes() != buf[:k].tobytes():
                out["differing"].append({"request": rid, "layer": name})
        out["requests"] += 1
    return out


def run(a: argparse.Namespace) -> dict[str, Any]:
    """One engine: the serve of ``a.arm``, or the replay of the serve in ``a.dir``."""
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("fp8-cache-replay: PYTHONHASHSEED=0 (a set's order decides a batch's row order)")
    os.environ["VLLM_DISABLE_REQUEST_ID_RANDOMIZATION"] = "1"
    sys.path.insert(0, str(ADAPTER))
    import numpy as np
    import torch

    from verity_vllm.engine import env, handle, served_seed
    from verity_vllm.engine.hooks import Hooks
    from verity_vllm.engine.step_trace import StepTrace, load

    env.apply_engine(None)
    out = Path(a.dir)
    reqs = {r["id"]: r for r in plan(served_seed.DEFAULT_SEED)}
    if a.phase == "serve":
        cfg = {**SERVE, **({"kv_cache_dtype": ARMS[a.arm]} if ARMS[a.arm] else {})}
        due = [{"id": r["id"], "iteration": r["iteration"]} for r in reqs.values()]
        (out / "trace.jsonl").unlink(missing_ok=True)
    else:
        recs = load(out / "trace.jsonl")
        cfg = dict(recs[0]["config"])
        if a.cache == "named":
            cfg["kv_cache_dtype"] = json.loads((out / "serve.json").read_text())["chosen"]["cache_dtype"]
        due = [{"id": r["request"], "before_step": r["before_step"]} for r in recs if r["kind"] == "arrive"]
    model = Path(tempfile.mkdtemp(prefix="fp8-cache-stand-in-"))
    stand_in(model)
    eng = engine(model, cfg, eager=not a.compiled)
    runner = handle.model_runner(eng)
    redrawn = rescale(runner) if a.weights == "scaled" else None
    trace = StepTrace(out / "trace.jsonl" if a.phase == "serve" else None).install(eng)
    said, hook = ours(runner, trace.records[0]["config"])
    rows = Rows(runner)
    tokens: dict[str, list[int]] = {}
    empty: dict[int, int] = {}

    def add(rid: str) -> None:
        r = reqs[rid]
        got = eng.add_request(rid, {"prompt_token_ids": r["prompt_ids"]}, params(r["sampling"], r["max_tokens"]))
        if got not in (None, rid):
            raise SystemExit(f"fp8-cache-replay: the engine named {rid} {got}")

    def schedule(orig: Any) -> Any:
        def wrapped(*args: Any, **kw: Any) -> Any:
            while a.phase == "replay" and due and due[0]["before_step"] == trace.scheduled:
                add(due.pop(0)["id"])
            before = trace.scheduled
            so = orig(*args, **kw)
            if trace.scheduled == before:
                empty[before] = empty.get(before, 0) + 1
            return so
        return wrapped

    def execute_model(orig: Any) -> Any:
        def wrapped(scheduler_output: Any, *args: Any, **kw: Any) -> Any:
            res = orig(scheduler_output, *args, **kw)
            if kw.get("dummy_run") or torch.cuda.is_current_stream_capturing() \
                    or not int(getattr(scheduler_output, "total_num_scheduled_tokens", 0) or 0):
                return res
            rows.after(runner)
            if hook is not None:
                hook.after(runner, scheduler_output)
            return res
        return wrapped

    hooks = Hooks(tag="fp8-cache-replay")
    hooks.wrap(trace.sched, "schedule", schedule)
    hooks.wrap(runner, "execute_model", execute_model)
    iteration = 0
    while due or eng.has_unfinished_requests():
        if a.phase == "serve":
            idle = not eng.has_unfinished_requests()
            while due and (due[0]["iteration"] <= iteration or idle):
                add(due.pop(0)["id"])
                idle = False
        elif due and (due[0]["before_step"] < trace.scheduled or not eng.has_unfinished_requests()):
            if due[0]["before_step"] != trace.scheduled:
                raise SystemExit(f"fp8-cache-replay: at step {trace.scheduled} with {due[0]['id']} due before step "
                                 f"{due[0]['before_step']}")
            add(due.pop(0)["id"])
        for o in eng.step():
            if o.finished:
                tokens[o.request_id] = [int(t) for t in o.outputs[0].token_ids]
        iteration += 1
    hooks.uninstall()
    trace.uninstall()
    digests, raw = rows.digest()
    if a.rows:
        np.savez(a.rows, **raw)
    steps = [r for r in trace.records if r["kind"] == "step"]
    doc = {"phase": a.phase, "arm": a.arm, "weights": a.weights, "compiled": a.compiled, "cache": a.cache,
           "config": trace.records[0]["config"], "engine_given": {k: cfg.get(k) for k in ("kv_cache_dtype", "block_size")},
           "chosen": chosen(eng, runner), "ours": {**said, **({"tables_rows": tables_rows(hook, raw)} if hook is not None else {})},
           "device": {"name": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0))},
           "vllm": __import__("vllm").__version__, "redrawn": redrawn, "steps": steps, "tokens": tokens, "rows": digests,
           "rows_skipped": rows.skipped, "sampling": {rid: r["sampling"] for rid, r in reqs.items()},
           "empty_schedules": sum(empty.values())}
    (out / f"{a.phase}{'-named' if a.cache == 'named' else ''}.json").write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    return doc


def summary(doc: dict[str, Any]) -> dict[str, Any]:
    """What a serve or a replay ran on: the engine's cache format, the distinct layers' kinds, and its hits and tokens."""
    kinds: dict[str, int] = {}
    for x in doc["chosen"]["layers"]:
        k = f"{x['module']}/{x['backend'] or x['impl']} {x['kv_cache_dtype']} {x['dtype']} {x['row_bytes']} B"
        kinds[k] = kinds.get(k, 0) + 1
    return {"cache_dtype": doc["chosen"]["cache_dtype"], "block_size": doc["chosen"]["block_size"], "layers": kinds,
            "cache_hits": {rid: n for s in doc["steps"] for rid, n in s["cache_hit"].items()}, "steps": len(doc["steps"]),
            "tokens": sum(map(len, doc["tokens"].values())), "distinct_tokens": len({t for ids in doc["tokens"].values() for t in ids}),
            "rows": sum(x["positions"] for by in doc["rows"].values() for x in by.values()), "rows_skipped": doc["rows_skipped"]}


def rows_diff(serve: dict[str, Any], replay: dict[str, Any], files: tuple[Path, Path], limit: int = 8) -> dict[str, Any]:
    """The (request, layer) row sets whose digests differ, and for the first few where: the first differing position and the
    byte's offset in its row."""
    import numpy as np

    keys = sorted({(r, n) for d in (serve, replay) for r, by in d["rows"].items() for n in by})
    bad = [(r, n) for r, n in keys if serve["rows"].get(r, {}).get(n) != replay["rows"].get(r, {}).get(n)]
    out: dict[str, Any] = {"sets": len(keys), "differing": len(bad), "first": []}
    if bad:
        a, b = (np.load(f) for f in files)
        for r, n in bad[:limit]:
            x, y = (z[f"{r}|{n}"] if f"{r}|{n}" in z.files else None for z in (a, b))
            where: dict[str, Any] = {"request": r, "layer": n, "served": serve["rows"].get(r, {}).get(n, {}).get("positions"),
                                     "replayed": replay["rows"].get(r, {}).get(n, {}).get("positions")}
            if x is not None and y is not None and x.shape == y.shape:
                rows, cols = np.nonzero(x != y)
                where.update(rows_differing=int(len(set(rows.tolist()))), first_position=int(rows[0]), first_byte=int(cols[0]))
            elif x is not None and y is not None:
                where.update(served_row_bytes=int(x.shape[1]), replayed_row_bytes=int(y.shape[1]))
            out["first"].append(where)
    return out


def compare(serve: dict[str, Any], replay: dict[str, Any], files: tuple[Path, Path]) -> dict[str, Any]:
    s, r = serve["steps"], replay["steps"]
    first = next((i for i, (x, y) in enumerate(zip(s, r)) if any(x[k] != y[k] for k in STEP_KEYS)), None)
    if first is None and len(s) != len(r):
        first = min(len(s), len(r))
    toks = [{"request": rid, "tokens": len(x), "first_position": next(
        (i for i, (p, q) in enumerate(zip(x, replay["tokens"].get(rid, []))) if p != q), min(len(x), len(replay["tokens"].get(rid, []))))}
        for rid, x in sorted(serve["tokens"].items()) if x != replay["tokens"].get(rid)]
    rows = rows_diff(serve, replay, files)
    same_engine = serve["chosen"] == replay["chosen"]
    return {"equal": first is None and not toks and not rows["differing"] and same_engine, "same_engine": same_engine,
            "replay_cache_dtype": replay["chosen"]["cache_dtype"], "steps": len(s), "steps_replayed": len(r),
            "first_step_differing": first, "requests": len(serve["tokens"]), "tokens": sum(map(len, serve["tokens"].values())),
            "tokens_differing": toks, "rows": rows, "empty_schedules": [serve["empty_schedules"], replay["empty_schedules"]]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", default=RUNS, help="WEIGHTS:ARM,... (dummy or scaled; an arm of ARMS)")
    ap.add_argument("--weights", choices=("dummy", "scaled"), default="dummy")
    ap.add_argument("--compiled", action="store_true", help="vLLM's defaults (torch.compile, CUDA graphs) instead of eager")
    ap.add_argument("--phase", choices=("serve", "replay"))
    ap.add_argument("--cache", choices=("header", "named"), default="header")
    ap.add_argument("--arm", choices=tuple(ARMS))
    ap.add_argument("--dir")
    ap.add_argument("--rows", help="where a phase writes its rows' bytes (.npz)")
    a = ap.parse_args()
    if a.phase:
        doc = run(a)
        print(f"fp8-cache-replay: {a.weights} {a.arm} {a.phase} ({a.cache}): {summary(doc)}", flush=True)
        return 0
    out = Path(a.out)
    scratch = Path(tempfile.mkdtemp(prefix="fp8-cache-rows-"))
    env = {**os.environ, "PYTHONHASHSEED": "0"}
    report: dict[str, Any] = {"layers": LAYERS, "experts": EXPERTS, "compiled": a.compiled, "runs": {}}
    ok = True
    runs = [tuple(r.split(":")) for r in a.runs.split(",")]
    unknown = [r for r in runs if len(r) != 2 or r[0] not in ("dummy", "scaled") or r[1] not in ARMS]
    if unknown:
        raise SystemExit(f"fp8-cache-replay: --runs names {unknown}: WEIGHTS:ARM, dummy or scaled, an arm of {list(ARMS)}")

    def phase(d: Path, weights: str, arm: str, name: str, cache: str) -> tuple[int, Path]:
        tag = name if cache == "header" else f"{name}-{cache}"
        rows = scratch / f"{d.name}-{tag}.npz"
        argv = [sys.executable, __file__, "--out", str(out), "--phase", name, "--cache", cache, "--arm", arm, "--dir", str(d),
                "--weights", weights, "--rows", str(rows), *(["--compiled"] if a.compiled else [])]
        with (d / f"{tag}.log").open("w") as log:
            rc = subprocess.run(argv, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        tail = (d / f"{tag}.log").read_text(errors="replace").splitlines()[-1:]
        print(f"== {weights} {arm} {tag}: rc {rc}: {tail[0][:600] if tail else ''}", flush=True)
        return rc, rows

    try:
        for weights, arm in runs:
            d = out / f"{weights}-{arm}"
            d.mkdir(parents=True, exist_ok=True)
            entry: dict[str, Any] = {"kv_cache_dtype": ARMS[arm]}
            report["runs"][d.name] = entry
            rc, served_rows = phase(d, weights, arm, "serve", "header")
            entry["serve"] = {"rc": rc}
            if rc:
                entry["serve"]["log_tail"] = (d / "serve.log").read_text(errors="replace").splitlines()[-25:]
                ok = False
            else:
                serve = json.loads((d / "serve.json").read_text())
                entry.update(config=serve["config"], redrawn=serve["redrawn"], ours=serve["ours"], served=summary(serve),
                             device=serve["device"], vllm=serve["vllm"])
                mla = [x for x in serve["chosen"]["layers"] if x["mla"]]
                entry["byte_rows"] = bool(mla) and all(x["dtype"] == "uint8" for x in mla)
                ok = ok and bool(entry["served"]["cache_hits"]) and (arm == "auto" or entry["byte_rows"])
                named = ARMS[arm] is not None and serve["config"].get("kv_cache_dtype") is None
                for cache in ("header", "named") if named else ("header",):
                    key = "replay" if cache == "header" else "replay_named"
                    rc, replayed_rows = phase(d, weights, arm, "replay", cache)
                    entry[key] = {"rc": rc}
                    if rc:
                        entry[key]["log_tail"] = (d / f"{'replay' if cache == 'header' else 'replay-named'}.log").read_text(
                            errors="replace").splitlines()[-25:]
                    else:
                        replay = json.loads((d / f"replay{'-named' if cache == 'named' else ''}.json").read_text())
                        entry[key].update(compare(serve, replay, (served_rows, replayed_rows)))
                    replayed_rows.unlink(missing_ok=True)
                ok = ok and bool(entry["replay"].get("equal"))
            served_rows.unlink(missing_ok=True)
            report["ok"] = ok
            (out / "fp8_cache_replay.json").write_text(json.dumps(report, indent=1) + "\n")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    for name, e in report["runs"].items():
        print(f"{name}: served {e.get('served')} ours {e.get('ours')} replay {e.get('replay')} named {e.get('replay_named')}",
              flush=True)
    print(f"fp8-cache-replay: {'ok' if ok else 'FAILED'}: {out / 'fp8_cache_replay.json'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
