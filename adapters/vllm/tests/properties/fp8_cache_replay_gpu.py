"""GPU run: a stand-in for GLM-5.3 served with an FP8 KV cache (vLLM's ``kv_cache_dtype``) and the prefix cache on, under the
scheduler's step trace, then replayed from the trace's header alone; the replay's steps, every token and every cached row's
bytes are held to the served ones.

The stand-in is GLM-5.3's registered config (`CONFIG`) cut to its first `LAYERS` layers, `EXPERTS` routed experts and `HEADS`
attention heads (one rank's of tensor-parallel 8, which decides how an FP8 cache is read), served tensor-parallel 1 on one
card from vLLM's dummy weights (``load_format="dummy"``), without a tokenizer. ``--weights dummy``
keeps vLLM's values; ``--weights scaled`` redraws every weight at `STD` after the load (`rescale`), so a token depends on its
whole context and one differing bit of a cached row shows in the tokens.

Each arm of `ARMS` is one cache format, as a launch line names it (``auto`` is the bf16 cache). Each phase is a process of
its own:

- ``--phase serve``: an engine with the arm's cache format under the step trace (`verity_vllm.engine.step_trace.StepTrace`).
  The requests of `plan` (greedy and sampled, each under its own seed, some sharing a cached prefix, more than fit one
  batch, and where the process's top-k is canonical one longer than the indexer's top-k: `LONG`) arrive between engine
  iterations. After each step's forward, `Rows` reads at
  every layer with a KV cache (each attention layer's latent rows, each indexer's key rows) the rows the step wrote: the
  layer's cache at the step's slots. The phase writes the trace, each
  request's tokens, a digest of each request's rows at each layer, what the engine chose (`chosen`: the cache format its
  config holds once built, each layer's attention backend, cache dtype and row bytes, and each backend's read path) and how
  its first attention layer attended (`Routes`).
- ``--phase replay``: an engine built from the trace's header, each request put back before the step the trace says it
  arrived before; the scheduler decides the rest. It writes the same. ``--cache named`` gives the engine the cache format
  the served engine's config held instead, for a tree whose header does not carry the format. It refuses a process whose
  top-k is not the serve's.

What of ours meets the engine is recorded with the serve (``ours``): whether the trace's header holds the cache format, what
the history tables' guard (`verity_vllm.pipeline.boundary_tables.mla_layers`) says of the model, and, where the guard passes,
the tables' own step hook beside `Rows`: every block of rows it reads must be `Rows`' bytes.

Without ``--phase`` it runs the serve and the replay of each of ``--runs`` (`RUNS`: weights and arm, and ``:fast`` for a run
on vLLM's own top-k whatever the environment asks for, whose plan then holds no long prompt), compares each replay
with its serve (steps, prefix-cache hits, tokens, rows), writes OUT/fp8_cache_replay.json and exits 0 only if every serve took
a prefix-cache hit, every arm but ``auto`` held byte rows, and every replay from the header equals its serve. Where the header
holds no cache format and the arm's is not ``auto``, the replay from the header runs another engine than the served one; the
run then replays once more with ``--cache named`` and reports both, and still fails. A phase that fails is recorded and the
rest go on. The rows' bytes stay in a scratch directory, removed at the end; OUT holds digests, and under each run's ``ptx/``
the PTX Triton compiled for vLLM's fused norm, RoPE and cache write (`keep_ptx`) when that run's serve was the first to need it.

In the pod venv, on one idle card, from the tree's root, with the DSA top-k plugin's site on the path for the long prompt:

    $PY benchmarks/dsa_topk/dsa_topk.py site SITE
    export PYTHONHASHSEED=0 PYTHONPATH=SITE VERITY_DSA_TOPK=canonical
    $PY adapters/vllm/tests/properties/fp8_cache_replay_gpu.py --out DIR [--runs dummy:auto,dummy:fp8,...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
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
#: the stand-in's attention heads: a rank's when GLM-5.3's 64 are split over tensor-parallel 8. vLLM's FlashMLA sparse
#: backend reads an FP8 cache with its FP8 kernel at every token below 32 heads a rank; at 32 or more a prefill token reads
#: a bf16 copy of the rows instead, a path GLM-5.3 on eight cards never takes
HEADS = 8
#: ``--weights scaled``: every weight's standard deviation, and the scale beside an FP8 weight uniform on [-1, 1]
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
#: a prompt longer than GLM-5.3's ``index_topk`` (2048), so that its last steps attend a strict subset of its cached rows.
#: vLLM's kernels write that subset in an order that varies run to run, and the attention sums in that order, so a row
#: past 2048 keys replays only where the top-k is canonical (`canonical_topk`): the plan holds this prompt only there
LONG = 2200
#: vLLM's indexer top-k ops (``torch.ops._C``), which the DSA top-k plugin wraps
TOPK_OPS = ("top_k_per_row_prefill", "top_k_per_row_decode", "persistent_topk", "cooperative_topk")
#: vLLM's Triton kernel that norms a token's latent, rotates its RoPE part and writes its cache row, quantising it for an
#: FP8 cache
WRITE_KERNEL = "_fused_norm_rope_kernel"
#: what of a step record the replay must repeat
STEP_KEYS = ("T", "order", "rows", "new", "resumed", "cache_hit", "preempted", "emitted", "finished")


def stand_in(where: Path) -> None:
    cfg = json.loads(CONFIG.read_text())
    cut = {**cfg, "num_hidden_layers": LAYERS, "n_routed_experts": EXPERTS, "num_attention_heads": HEADS,
           "num_key_value_heads": HEADS}
    for k in ("indexer_types", "mlp_layer_types"):
        cut[k] = cfg[k][:LAYERS]
    (where / "config.json").write_text(json.dumps(cut, indent=1) + "\n")


def canonical_topk() -> Any:
    """The DSA top-k plugin's device pass where this process's indexer top-k is the canonical one, else None: the plugin's
    wrappers on ``torch.ops._C``, which vLLM loads with the engine where the plugin's site is on the path and
    ``VERITY_DSA_TOPK=canonical``. Its ``record()`` counts the calls and the rows it rewrote."""
    import torch

    return next((c for n in TOPK_OPS
                 if (c := getattr(getattr(torch.ops._C, n, None), "__dict__", {}).get("dsa_topk_canon")) is not None), None)


def plan(default_seed: int, long: bool = True) -> list[dict[str, Any]]:
    """The requests, the same in every arm: its id, the engine iteration it arrives before, its prompt and its sampling.
    ``long``: whether one prompt is `LONG`."""
    rng = random.Random(0)

    def ids(n: int) -> list[int]:
        return [rng.randrange(1000, 150000) for _ in range(n)]

    shared = ids(200)
    out = []
    for i in range(REQUESTS):
        n = LONG if i == 4 and long else rng.choice((24, 70, 150, 330, 610))
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
    """Layer name -> the module the engine bound that layer's KV cache to (the outermost under each name). The name is
    the one vLLM's forward context and a step's slot mappings hold the layer under: an attention layer's ``layer_name``,
    and the ``prefix`` of an indexer's key cache, which has no ``layer_name``."""
    out: dict[str, Any] = {}
    for _path, m in model.named_modules():
        cache: Any = getattr(m, "kv_cache", None)
        name = next((n for n in (getattr(m, "layer_name", None), getattr(m, "prefix", None)) if isinstance(n, str)), None)
        if name is not None and hasattr(cache, "stride") and cache.numel() and name not in out:
            out[name] = m
    return out


def chosen(eng: Any, runner: Any) -> dict[str, Any]:
    """What the built engine holds: its config's cache format and block size, each layer with a KV cache, and each
    attention backend's metadata builder: whether it holds an FP8 cache and reads it in one mixed batch."""
    cc = eng.vllm_config.cache_config
    layers = []
    for name, m in kv_layers(runner.model).items():
        c, backend = m.kv_cache, getattr(m, "attn_backend", None)
        layers.append({"layer": name, "module": type(m).__name__, "mla": getattr(m, "kv_lora_rank", None) is not None,
                       "kv_cache_dtype": getattr(m, "kv_cache_dtype", None), "fp8_query": getattr(m, "_fp8_query", None),
                       "backend": getattr(backend, "__name__", None) if backend is not None else None,
                       "impl": type(m.impl).__name__ if getattr(m, "impl", None) is not None else None,
                       "dtype": str(c.dtype).removeprefix("torch."), "shape": list(c.shape),
                       "row_bytes": int(c.shape[-1]) * c.element_size()})
    builders = []
    for group in (g for groups in getattr(runner, "attn_groups", None) or [] for g in groups):
        b = (getattr(group, "metadata_builders", None) or [None])[0]
        builders.append({"backend": getattr(group.backend, "__name__", None), "layers": len(group.layer_names),
                         "heads": getattr(b, "num_heads", None), "fp8_cache": getattr(b, "use_fp8_kv_cache", None),
                         "fp8_mixed_batch": getattr(b, "fp8_use_mixed_batch", None)})
    return {"cache_dtype": str(cc.cache_dtype), "block_size": int(cc.block_size), "layers": layers, "builders": builders}


class Routes:
    """How the first attention layer with vLLM's route decision (``_use_sparse_mha``) attended, by call of that decision
    (a layer asks once a step, and again where the step has prefill tokens): ``top-k`` reads every key from the cache at the
    indexer's selection; ``dense`` reads a prefill step's own keys as the step computed them, never from the cache;
    ``dense, cached context`` reads, beside them, the rows of an earlier chunk or of a prefix-cache hit from the cache."""

    def __init__(self, hooks: Any, model: Any) -> None:
        self.count: dict[str, int] = {}
        layer = next((m for m in kv_layers(model).values() if callable(getattr(m, "_use_sparse_mha", None))), None)
        if layer is not None:
            hooks.wrap(layer, "_use_sparse_mha", self.counting)

    def counting(self, orig: Any) -> Any:
        def wrapped(attn_metadata: Any, *args: Any, **kw: Any) -> Any:
            dense = orig(attn_metadata, *args, **kw)
            context = getattr(getattr(attn_metadata, "prefill", None), "chunked_context", None) is not None
            route = "top-k" if not dense else "dense, cached context" if context else "dense"
            self.count[route] = self.count.get(route, 0) + 1
            return dense
        return wrapped


def keep_ptx(cache: Path, dest: Path, seen: set[str]) -> list[dict[str, Any]]:
    """Copies to `dest` each PTX of `WRITE_KERNEL` in Triton's cache directory `cache` that no earlier call saw (one file a
    specialization; the cache format is among its constants), and counts its f32 divisions by mnemonic: the FP8 write
    divides a latent by its tile's scale, and the mnemonic says how that quotient rounds."""
    out = []
    for f in sorted(cache.rglob(f"{WRITE_KERNEL}*.ptx")):
        if str(f) in seen:
            continue
        seen.add(str(f))
        dest.mkdir(parents=True, exist_ok=True)
        name = f"{f.parent.name[:12]}-{f.name}"
        shutil.copy(f, dest / name)
        div: dict[str, int] = {}
        for m in re.findall(r"\bdiv\.[a-z.]*f32\b", f.read_text(errors="replace")):
            div[m] = div.get(m, 0) + 1
        out.append({"file": name, "bytes": f.stat().st_size, "div": div})
    return out


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
    """What of ours says of this engine before it serves, and the history tables' step hook where their guard passes. The
    hook's rows go to `Seen`, which keeps every block as read: the tables' own store refuses a request that starts past what
    it holds, as one with a prefix-cache hit does."""
    from verity_vllm.pipeline import boundary_tables as BT

    class Seen(BT.Latents):
        def add(self, request: str, start: int, rows: Any) -> None:
            self.seen.append((request, start, {x: r.copy() for x, r in rows.items()}))

    out: dict[str, Any] = {"header_kv_cache_dtype": header.get("kv_cache_dtype")}
    hook = None
    try:
        seen = Seen(BT.mla_layers(runner.model))
        seen.seen = []
        hook = BT.StepHook(runner.model, latents=seen)
        out["tables_guard"] = {"verdict": "ok", "layers": list(hook.layers)}
    except ValueError as e:
        out["tables_guard"] = {"verdict": "refuses", "message": str(e)}
    return out, hook


def tables_rows(hook: Any, rows: Rows) -> dict[str, Any]:
    """The tables' hook against `Rows`: each block of rows the hook read at a step, in order, against the block `Rows` read
    for that request at that layer at that step."""
    out: dict[str, Any] = {"failed": hook.latents.failed, "blocks": 0, "rows": 0, "differing": []}
    at: dict[str, int] = {}
    for rid, start, by in hook.latents.seen:
        i = at[rid] = at.get(rid, -1) + 1
        for x, (name, _cache) in hook.layers.items():
            s, mine = rows.got[rid][name][i]
            out["blocks"] += 1
            out["rows"] += len(mine)
            if s != start or by[x].tobytes() != mine.tobytes():
                out["differing"].append({"request": rid, "layer": name, "start": start})
    return out


def run(a: argparse.Namespace) -> dict[str, Any]:
    """One engine: the serve of ``a.arm``, or the replay of the serve in ``a.dir``."""
    if os.environ.get("PYTHONHASHSEED") != "0":
        raise SystemExit("fp8-cache-replay: PYTHONHASHSEED=0 (a set's order decides a batch's row order)")
    os.environ["VLLM_DISABLE_REQUEST_ID_RANDOMIZATION"] = "1"
    sys.path.insert(0, str(ADAPTER))
    import numpy as np
    import torch

    from tests.properties import specdec_replay_gpu as loop
    from verity_vllm.engine import env, handle, served_seed
    from verity_vllm.engine.hooks import Hooks
    from verity_vllm.engine.step_trace import StepTrace, load

    env.apply_engine(None)
    out = Path(a.dir)
    served = None if a.phase == "serve" else json.loads((out / "serve.json").read_text())
    if served is None:
        cfg = {**SERVE, **({"kv_cache_dtype": ARMS[a.arm]} if ARMS[a.arm] else {})}
        (out / "trace.jsonl").unlink(missing_ok=True)
    else:
        recs = load(out / "trace.jsonl")
        cfg = dict(recs[0]["config"])
        if a.cache == "named":
            cfg["kv_cache_dtype"] = served["chosen"]["cache_dtype"]
    model = Path(tempfile.mkdtemp(prefix="fp8-cache-stand-in-"))
    stand_in(model)
    eng = engine(model, cfg, eager=not a.compiled)
    runner = handle.model_runner(eng)
    canon = canonical_topk()
    canonical = canon is not None
    if served is not None and canonical != served["canonical_topk"]:
        raise SystemExit(f"fp8-cache-replay: the serve's top-k was {'' if served['canonical_topk'] else 'not '}canonical "
                         f"and this process's is {'' if canonical else 'not '}: replay under the serve's environment")
    reqs = {r["id"]: r for r in plan(served_seed.DEFAULT_SEED, long=canonical)}
    due = [{"id": r["id"], "iteration": r["iteration"]} for r in reqs.values()] if served is None else loop.arrivals(recs)
    redrawn = rescale(runner) if a.weights == "scaled" else None
    trace = StepTrace(out / "trace.jsonl" if a.phase == "serve" else None).install(eng)
    said, hook = ours(runner, trace.records[0]["config"])
    rows = Rows(runner)

    def add(rid: str) -> None:
        r = reqs[rid]
        got = eng.add_request(rid, {"prompt_token_ids": r["prompt_ids"]}, params(r["sampling"], r["max_tokens"]))
        if got not in (None, rid):
            raise SystemExit(f"fp8-cache-replay: the engine named {rid} {got}")

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
    routes = Routes(hooks, runner.model)
    hooks.wrap(runner, "execute_model", execute_model)
    # the speculative-decoding stand-in's loop: the serve's arrivals by engine iteration, the replay's where the trace says
    drove = loop.drive(eng, trace, a.phase, due, add)
    hooks.uninstall()
    trace.uninstall()
    digests, raw = rows.digest()
    if a.rows:
        np.savez(a.rows, **raw)
    steps, tokens = drove["steps"], drove["tokens"]
    doc = {"phase": a.phase, "arm": a.arm, "weights": a.weights, "compiled": a.compiled, "cache": a.cache,
           "config": trace.records[0]["config"], "engine_given": {k: cfg.get(k) for k in ("kv_cache_dtype", "block_size")},
           "chosen": chosen(eng, runner), "ours": {**said, **({"tables_rows": tables_rows(hook, rows)} if hook is not None else {})},
           "device": {"name": torch.cuda.get_device_name(0), "capability": list(torch.cuda.get_device_capability(0))},
           "vllm": __import__("vllm").__version__, "redrawn": redrawn, "steps": steps, "tokens": tokens, "rows": digests,
           "rows_skipped": rows.skipped, "sampling": {rid: r["sampling"] for rid, r in reqs.items()},
           "empty_schedules": drove["empty_schedules"], "routes": routes.count, "canonical_topk": canonical,
           "topk": canon.record() if canonical else None}
    (out / f"{a.phase}{'-named' if a.cache == 'named' else ''}.json").write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    return doc


def summary(doc: dict[str, Any]) -> dict[str, Any]:
    """What a serve or a replay ran on: the engine's cache format, the distinct layers' kinds, and its hits and tokens."""
    kinds: dict[str, int] = {}
    for x in doc["chosen"]["layers"]:
        k = f"{x['module']}/{x['backend'] or x['impl']} {x['kv_cache_dtype']} {x['dtype']} {x['row_bytes']} B"
        kinds[k] = kinds.get(k, 0) + 1
    return {"cache_dtype": doc["chosen"]["cache_dtype"], "block_size": doc["chosen"]["block_size"], "layers": kinds,
            "builders": doc["chosen"]["builders"], "routes": doc["routes"], "topk": doc["topk"],
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
            "tokens_differing": toks, "rows": rows, "empty_schedules": [serve["empty_schedules"], replay["empty_schedules"]],
            "routes_replayed": replay["routes"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", default=RUNS, help="WEIGHTS:ARM[:fast],... (dummy or scaled; an arm of ARMS; fast: vLLM's own "
                    "top-k, whatever the environment asks for)")
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
    env = {**os.environ, "PYTHONHASHSEED": "0", "TRITON_CACHE_DIR": str(scratch / "triton")}
    ptx_seen: set[str] = set()
    report: dict[str, Any] = {"layers": LAYERS, "experts": EXPERTS, "compiled": a.compiled, "runs": {}}
    ok = True
    runs = [tuple(r.split(":")) for r in a.runs.split(",")]
    unknown = [r for r in runs if len(r) < 2 or r[0] not in ("dummy", "scaled") or r[1] not in ARMS or r[2:] not in ((), ("fast",))]
    if unknown:
        raise SystemExit(f"fp8-cache-replay: --runs names {unknown}: WEIGHTS:ARM[:fast], dummy or scaled, an arm of {list(ARMS)}")

    def phase(d: Path, weights: str, arm: str, name: str, cache: str, env: dict[str, str]) -> tuple[int, Path]:
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
        for weights, arm, *fast in runs:
            d = out / "-".join((weights, arm, *fast))
            d.mkdir(parents=True, exist_ok=True)
            entry: dict[str, Any] = {"kv_cache_dtype": ARMS[arm]}
            report["runs"][d.name] = entry
            run_env = {k: v for k, v in env.items() if not (fast and k == "VERITY_DSA_TOPK")}
            rc, served_rows = phase(d, weights, arm, "serve", "header", run_env)
            entry["serve"] = {"rc": rc}
            entry["write_ptx"] = keep_ptx(scratch / "triton", d / "ptx", ptx_seen)
            if rc:
                entry["serve"]["log_tail"] = (d / "serve.log").read_text(errors="replace").splitlines()[-25:]
                ok = False
            else:
                serve = json.loads((d / "serve.json").read_text())
                entry.update(config=serve["config"], redrawn=serve["redrawn"], ours=serve["ours"], served=summary(serve),
                             device=serve["device"], vllm=serve["vllm"], canonical_topk=serve["canonical_topk"])
                mla = [x for x in serve["chosen"]["layers"] if x["mla"]]
                entry["byte_rows"] = bool(mla) and all(x["dtype"] == "uint8" for x in mla)
                ok = ok and bool(entry["served"]["cache_hits"]) and (arm == "auto" or entry["byte_rows"])
                named = ARMS[arm] is not None and serve["config"].get("kv_cache_dtype") is None
                for cache in ("header", "named") if named else ("header",):
                    key = "replay" if cache == "header" else "replay_named"
                    rc, replayed_rows = phase(d, weights, arm, "replay", cache, run_env)
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
