"""A model served mid-traffic with drained edges (`tp_edge`), restored at its first edge into a fresh engine, replayed to
its second and compared. GLM-5.3 at TP8 is the case it is for; any `serve_tax` model runs.

    tp_restore.py preflight --model-id glm53 [--max-model-len N] [--tp N] [--kv-cache-dtype FORMAT]
    tp_restore.py serve --model-id glm53 --c4 C4 --out OUT --scratch DIR [--rate 10] [--span-s 240]
                        [--edges-at 120,170,220] [--plan-file PLAN.jsonl] [--unseeded-share 0.5] [--sampling own]
                        [--max-model-len 40960] [--tp N] [--kv-cache-dtype FORMAT] [--long-share 0.05]
                        [--long-tokens 8192,32768]
                        [--saver MODULE:FUNCTION ...] [--dump 1] [--num-gpu-blocks N]
    tp_restore.py restore --served OUT --scratch DIR --out OUT2 --loader MODULE:FUNCTION [--no-fill] [--rank-map-free]

``serve`` runs `serve_tax`'s ``record`` arm (vLLM's OpenAI server with the scheduler's step trace), with `tp_edge` as a
second general plugin. It is pinned: PYTHONHASHSEED=0, request ids not randomized, Inductor's reduction block untimed,
Model Runner V2. It drives the plan under Poisson arrivals, every request asking for its sampled token's logprob.
With ``--sampling own`` each request sends the sampling it carries (``--plan-file``, the replay ladder's production
plan: ShareGPT, multi-turn and long prompts, each greedy or at temperature 0.7), the rest of it the server's
generation-config defaults; ``--unseeded-share`` of the sampled requests go without a seed (`request_class`), so vLLM
draws theirs, and the rest carry the public seed rule's. A named ``--sampling`` goes on every request instead, where
``production`` sends no sampling fields: the generation config's defaults apply (GLM-5.3: temperature 1.0, top-p 0.95)
and nothing is seeded, so with no greedy or seeded row in a batch vLLM samples with FlashInfer's sampler and the CUDA
generator. Each edge writes under ``--scratch``, because its blocks are tens of GB and stay out of the run's record; the
first ``--dump`` edges write their blocks, the rest only their digests and state. OUT
gets each ``edge.json``, the step outputs from the first edge to the last, the client's records and the step trace.
Once the last edge is written the server stops, unless ``--run-out``: nothing after it is compared, and the requests in
flight then are recorded with the tokens they streamed.

``restore`` builds the served engine again in this process, from the served server's own argv and block count, and:
- calls ``--loader`` (the scheduler's state and the runner's rows) with the engine core and the first edge's directory
  (without one, it stops once the edge's blocks and RNG are checked);
- loads every rank's blocks and RNG (`tp_state.load_rank`, `set_rng_ranked`), filling every other block with NaN unless
  ``--no-fill``;
- checks every rank's digest of the edge's blocks against the served one. If they differ, the restore isn't exact at
  the edge and the run stops there;
- compares the rank map with the served one, and refuses a different map unless ``--rank-map-free``;
- steps the engine core to the second edge, drained as served, adding each request that arrived after the first edge
  right before the step it arrived before in the served step trace;
- writes ``restore.json``: bit-exact or not, with the first step whose scheduled rows differ, the first step and
  request whose sampled tokens or logprob bits differ (`boundary.outputs_diff`, and per sampling class, plan kind
  and keys class: at most or past GLM-5.3's indexer top-k, `TOPK_KEYS`), the first request position whose client token
  differs, and the first layer and block whose bytes differ at the second edge, with the differing blocks per keys
  class of the requests holding them, each edge's KV root (`kv_root`), the requests held at the first edge (per keys
  class too) and the snapshot's bytes.
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools" / "config"))
sys.path.append(str(ROOT))
import serve_tax as S  # noqa: E402 (beside this file)
import tp_edge as E  # noqa: E402
import tp_state  # noqa: E402

SCHEMA = "verity-bench/tp-restore/v0"
#: the pins both engines run under: the prefix cache's hash seed, unrandomized request ids, Inductor's reduction block
#: not timed per process, and Model Runner V2 (vLLM's default here, stated so a feature can't silently switch it)
PINS = {"PYTHONHASHSEED": "0", "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION": "1", "TORCHINDUCTOR_DYNAMIC_SCALE_RBLOCK": "0",
        "VLLM_USE_V2_MODEL_RUNNER": "1"}
#: sampling a request asks for: ``production`` asks for nothing, so the generation config's defaults apply, unseeded
SAMPLING = {"production": {}}
#: vLLM's directory override for FlashInfer's autotune results (`resolve_flashinfer_autotune_file`)
AUTOTUNE_ENV = "VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR"
#: the client's request ids, ``tpr-<plan index>``; the engine's are ``cmpl-<client id>-0`` (with request ids not randomized)
ID = "tpr"
#: the logprobs every request asks for: its sampled token's (the step trace's sampling record leaves this field out)
LOGPROBS = 0
#: ``--sampling`` for each request's own (a production plan's)
OWN = "own"


def client_id(engine_id: str) -> str:
    m = re.search(rf"{ID}-\d+", engine_id)
    if m is None:
        raise ValueError(f"{engine_id!r}: no client id ({ID}-<n>) in it")
    return m.group(0)


def request_class(r: Mapping[str, Any], rid: str, unseeded_share: float, sampling: str = OWN) -> str:
    """A request's sampling class: under a named ``sampling``, or for a request without its own, that name; else
    ``greedy`` unless the plan samples it, and a sampled one ``unseeded`` when the first byte of its id's SHA-512 falls
    in ``unseeded_share``, else ``seeded``."""
    if sampling != OWN or "sampling" not in r:
        return sampling
    if not r.get("seeded"):
        return "greedy"
    return "unseeded" if hashlib.sha512(rid.encode()).digest()[0] < 256 * unseeded_share else "seeded"


def own_body(real: Callable[..., dict], unseeded_share: float, sampling: str = OWN) -> Callable[..., dict]:
    """The request body, asking for each sampled token's logprob: for a plan's request without sampling of its own,
    ``real`` (serve_tax's, decoding to its cap); for a production plan's, which stops at its end of text or its cap,
    with `OWN` its own sampling and the public seed rule's seed (`corpus.request_seed`) unless it is unseeded, and under
    a named ``sampling`` that one with no seed."""
    C = S.corpus()

    def body(served: str, r: dict, rid: str, name: str) -> dict:
        if "sampling" not in r:
            return real(served, r, rid, name) | {"logprobs": LOGPROBS}
        params = r["sampling"] if sampling == OWN else C.SAMPLING[sampling]
        b = {"model": served, "prompt": r["prompt_ids"], "max_tokens": r["max_tokens"], **params, "stream": True,
             "stream_options": {"include_usage": True}, "return_token_ids": True, "user": rid, "logprobs": LOGPROBS}
        if request_class(r, rid, unseeded_share, sampling) == "seeded":
            b["seed"] = C.request_seed(rid)
        return b
    return body


def sampling_params(sampling: Mapping[str, Any], logprobs: int | None = None) -> Any:
    """The request's SamplingParams from every field the step trace recorded of the served one (the server's
    generation-config defaults applied) and the ``logprobs`` the client asked for, detokenizing nothing, its outputs as a
    streaming server's."""
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    return SamplingParams(**{k: v for k, v in sampling.items() if v is not None and k != "n"}, n=1, logprobs=logprobs,
                          detokenize=False, output_kind=RequestOutputKind.DELTA)


def edge_site(site: Path, arm: S.Arm) -> Path:
    """`serve_tax.plugin_site`'s directory for ``arm`` with `tp_edge.register` declared beside the arm's plugins."""
    S.plugin_site(site, arm)
    d = site / "verity_tp_edge-0.0.dist-info"
    d.mkdir(parents=True, exist_ok=True)
    (d / "METADATA").write_text("Metadata-Version: 2.1\nName: verity-tp-edge\nVersion: 0.0\n")
    (d / "entry_points.txt").write_text("[vllm.general_plugins]\nverity_tp_edge = tp_edge:register\n")
    return site


def drive(port: int, served: str, plan: list[dict], rate: float, span_s: float, d: Path, drain_s: float,
          sampling: str, requests: int | None = None, unseeded_share: float = 0.0) -> dict:
    """`serve_tax.drive_load` under ``sampling`` (one of the corpus's, or ``production``), or with `OWN` each request's
    own (`own_body`); each request's class goes to ``classes.json``."""
    C = S.corpus()
    for name, body in SAMPLING.items():
        C.SAMPLING.setdefault(name, body)
    d.mkdir(parents=True, exist_ok=True)
    arr = C.traffic.arrivals(plan, rate)
    k = requests or max(1, sum(1 for a in arr if a <= span_s))
    reqs, arr = plan[:k], arr[:k]
    if sampling == OWN and any("sampling" not in r for r in reqs):
        raise SystemExit("the plan's requests carry no sampling of their own: name one with --sampling")
    ids = [f"{ID}-{r['i']:05d}" for r in reqs]
    classes = {rid: request_class(r, rid, unseeded_share, sampling) for r, rid in zip(reqs, ids)}
    (d / "classes.json").write_text(json.dumps(classes))
    real = C.request_body
    C.request_body = own_body(real, unseeded_share, sampling)
    t = time.monotonic()
    try:
        got = asyncio.run(C.drive(port, served, reqs, arr, ids, d / "client.jsonl", drain_s, sampling))
    finally:
        C.request_body = real
    counts: dict[str, int] = {}
    for c in classes.values():
        counts[c] = counts.get(c, 0) + 1
    meta = {"rate": rate, "span_s": span_s, "requests": k, "sampling": sampling, "logprobs": LOGPROBS,
            "unseeded_share": unseeded_share, "classes": counts, "wall_s": round(time.monotonic() - t, 3), **got}
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def lengthen(plan: list[dict], share: float, lo: int, hi: int, seed: int) -> list[dict]:
    """The plan with a ``share`` of its requests (drawn under ``seed``) given long prompts, of ``lo`` to ``hi`` tokens
    each: the prompts of the plan's requests from the next one on, joined in order. A long prompt's prefill spans steps
    of the scheduler's token budget, so an edge can find one part done."""
    rng = random.Random(seed)
    out = []
    for n, r in enumerate(plan):
        if rng.random() >= share:
            out.append(r)
            continue
        want, ids, k = rng.randint(lo, hi), [], n
        while len(ids) < want:
            k = (k + 1) % len(plan)
            ids += plan[k]["prompt_ids"]
        out.append(r | {"prompt_ids": ids[:want], "prompt_len": want, "kind": "long"})
    return out


def served_argv(a: argparse.Namespace, checkpoint: Path, port: int) -> list[str]:
    """`serve_tax`'s ``record`` arm's server argv for ``a``'s model, with ``a``'s context, tensor-parallel size, pool and
    KV cache format."""
    return S.server_argv(S.ARMS["record"], S.MODELS[a.model_id], checkpoint, port, getattr(a, "num_gpu_blocks", None)) \
        + (["--max-model-len", str(a.max_model_len)] if a.max_model_len else []) \
        + (["--tensor-parallel-size", str(a.tp)] if a.tp else []) \
        + (["--kv-cache-dtype", a.kv_cache_dtype] if getattr(a, "kv_cache_dtype", None) else [])


def preflight(a: argparse.Namespace) -> int:
    """Before any engine starts: the served argv parses as `restore` parses it, and every name `restore` reads
    imports."""
    sys.path.insert(0, str(ROOT / "adapters/vllm"))
    from verity_vllm.engine.hooks import Hooks  # noqa: F401
    from verity_vllm.engine.step_trace import StepTrace  # noqa: F401
    from vllm.config import VllmConfig
    from vllm.v1.engine.core_client import InprocClient
    from vllm.v1.engine.llm_engine import LLMEngine
    from vllm.v1.executor import Executor

    checkpoint = Path(a.checkpoint) if a.checkpoint else S.MODELS[a.model_id].checkpoint()
    args = engine_args(served_argv(a, checkpoint, 0)[1:], {"num_gpu_blocks": 1})
    got = {k: getattr(args, k) for k in ("model", "tensor_parallel_size", "max_model_len", "seed", "num_gpu_blocks_override",
                                         "enable_prefix_caching", "async_scheduling", "kv_cache_dtype", "block_size")}
    missing = [f"{c.__name__}.{n}" for c, n in ((LLMEngine, "__init__"), (InprocClient, "get_output"),
                                                (Executor, "get_class"), (VllmConfig, "compute_hash"),
                                                (type(args), "create_engine_config")) if not hasattr(c, n)]
    print(f"[tp-restore] preflight: {got}" + (f"; missing {missing}" if missing else ""), flush=True)
    return 1 if missing or got["tensor_parallel_size"] != (a.tp or (8 if a.model_id == "glm53" else 1)) else 0


def serve(a: argparse.Namespace) -> int:
    model = S.MODELS[a.model_id]
    checkpoint = Path(a.checkpoint) if a.checkpoint else model.checkpoint()
    out, scratch = Path(a.out), Path(a.scratch)
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    need = S.plan_need({"load": a.rate}, a.span_s, a.requests)
    if a.plan_file:
        base = Path(a.plan_file)
        plan = [json.loads(x) for x in base.read_text().splitlines() if x.strip()]
        if len(plan) < need:
            raise SystemExit(f"{base} holds {len(plan)} requests, fewer than {need} for {a.rate} req/s over {a.span_s:g} s")
    else:
        base = Path(a.plan) if a.plan else out / "plan.base.jsonl"
        plan = S.plan_of(base, Path(a.c4), checkpoint, need, a.docs, a.seed)
        if a.long_share:
            lo, hi = a.long_tokens
            plan = lengthen(plan, a.long_share, lo, hi, a.seed)
    plan_file = out / "plan.jsonl"
    plan_file.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in plan))
    arm = S.ARMS["record"]
    site = edge_site(scratch / ".site", arm)
    port = S.corpus().free_port()
    argv = served_argv(a, checkpoint, port)
    srv = out / "server"
    srv.mkdir(exist_ok=True)
    env = S.server_env(arm, srv, site, None, 600.0, 600.0) | PINS | {AUTOTUNE_ENV: str(scratch / "autotune")}
    env[E.ENV] = json.dumps({"out": str(scratch / "edges"), "at_s": a.edges_at, "dump": a.dump, "savers": a.saver,
                             "outputs": True, "whole_pool": True})
    meta: dict[str, Any] = {"schema": SCHEMA, "model_id": a.model_id, "model": f"{model.repo}@{model.revision}",
                            "checkpoint": str(checkpoint), "plan": str(plan_file), "plan_base": str(base),
                            "plan_file": a.plan_file, "unseeded_share": a.unseeded_share,
                            "long_share": None if a.plan_file else a.long_share, "long_tokens": a.long_tokens,
                            "server_argv": argv[1:], "pins": PINS, "edges_at": a.edges_at, "dump": a.dump, "savers": a.saver,
                            "scratch": str(scratch), "autotune_dir": str(scratch / "autotune"), "versions": S.versions(),
                            "host": os.uname().nodename}
    meta["open_files"] = S.open_files()
    rc = 0
    t = time.monotonic()
    with open(srv / "server.log", "wb") as log:
        server = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    finished = threading.Event()

    def stop_after_last_edge() -> None:
        while not finished.wait(1.0):
            if len(list((scratch / "edges").glob("edge-*/edge.json"))) >= len(a.edges_at):
                meta["stopped_after_last_edge_s"] = round(time.monotonic() - t, 1)
                S.stop(server)
                return

    watcher = threading.Thread(target=stop_after_last_edge, name="tp-restore-last-edge", daemon=True)
    try:
        S.wait_ready(server, port, a.startup_s)
        meta["startup_s"] = round(time.monotonic() - t, 1)
        if not a.run_out:
            watcher.start()
        meta["load"] = drive(port, model.served, plan, a.rate, a.span_s, out / "load", a.drain_s, a.sampling, a.requests,
                             a.unseeded_share)
    except Exception as e:  # noqa: BLE001 -- the record says why the run stopped
        meta["error"] = f"{type(e).__name__}: {e}"
        rc = 1
    finally:
        finished.set()
        if watcher.is_alive():
            watcher.join()
        meta["server_stop"] = S.stop(server)
        meta["edges"] = copy_edges(scratch / "edges", out / "edges")
        (out / "meta.json").write_text(json.dumps(meta, indent=1))
    if len(meta["edges"]) < 2:
        print(f"[tp-restore] serve: {len(meta['edges'])} edge(s) taken, 2 needed (edges at {a.edges_at} s)"
              + (f"; {meta['error']}" if meta.get("error") else ""), flush=True)
        rc = rc or 1
    return rc


def copy_edges(src: Path, dst: Path) -> list[dict]:
    """Each edge's ``edge.json``, and the step outputs, from the scratch directory into the run's record, and a summary
    of each edge with its KV root and its files' bytes."""
    got = []
    dst.mkdir(parents=True, exist_ok=True)
    if (src / "outputs.jsonl").exists():
        shutil.copy2(src / "outputs.jsonl", dst / "outputs.jsonl")
    for e in sorted(src.glob("edge-*/edge.json")):
        (dst / e.parent.name).mkdir(parents=True, exist_ok=True)
        shutil.copy2(e, dst / e.parent.name / "edge.json")
        rec = json.loads(e.read_text())
        got.append({k: rec.get(k) for k in ("edge", "current_step", "trace_step", "running", "in_flight", "waiting", "pool",
                                            "check", "seconds")}
                   | {"dir": str(e.parent), "blocks": len(rec["ids"]), "kv_root": kv_root(rec), "files": file_bytes(e.parent)})
    return got


def kv_root(rec: Mapping[str, Any]) -> str:
    """An edge's KV root: SHA-512 over its block ids and rank 0's per-layer digests (`tp_state.digest`), in order."""
    return hashlib.sha512(json.dumps({"ids": rec["ids"], "layers": rec["ranks"][0]["layer_digests"]},
                                     separators=(",", ":")).encode()).hexdigest()


def file_bytes(d: Path) -> dict[str, int]:
    """Each file's bytes under an edge's directory, by its path there."""
    return {str(p.relative_to(d)): p.stat().st_size for p in sorted(d.rglob("*")) if p.is_file()}


# ---------------------------------------------------------------------------------------------------- the restore

#: the step-trace header's scheduler and cache settings an engine's arguments set: vLLM resolves the batch limits by
#: usage context and device, so the restore sets each to the served engine's
PINNED_CONFIG = ("max_num_batched_tokens", "max_num_seqs", "max_model_len", "enable_chunked_prefill",
                 "long_prefill_token_threshold", "max_num_partial_prefills", "async_scheduling", "block_size",
                 "enable_prefix_caching")


def engine_args(server_argv: Sequence[str], config: Mapping[str, Any]) -> Any:
    """The served engine's arguments, parsed from its server's argv as vLLM's OpenAI server parses them (its stats
    setting too: the scheduler's state a loader restores keeps the served one's), with the served step trace header's
    block count (``num_gpu_blocks_override``) and `PINNED_CONFIG`."""
    from vllm.engine.arg_utils import EngineArgs
    from vllm.entrypoints.launchers.cli_args import make_arg_parser
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    argv = list(server_argv)
    if argv[:2] == ["-m", "vllm.entrypoints.openai.api_server"]:
        argv = argv[2:]
    args = EngineArgs.from_cli_args(make_arg_parser(FlexibleArgumentParser()).parse_args(argv))
    args.num_gpu_blocks_override = int(config["num_gpu_blocks"])
    for k in PINNED_CONFIG:
        if config.get(k) is not None and hasattr(args, k):
            setattr(args, k, config[k])
    return args


def config_difference(served: Mapping[str, Any], restored: Mapping[str, Any]) -> dict[str, list]:
    """The scheduler and cache settings (step-trace header ``config``) whose values differ: name -> [served, restored]."""
    return {k: [served.get(k), restored.get(k)] for k in sorted(set(served) | set(restored)) if served.get(k) != restored.get(k)}


def served_trace(served: Path) -> list[dict]:
    """The served engine core's step trace, from ``serve``'s OUT (its ``server/``)."""
    (trace,) = S.step_traces(served)
    return S.trace_records(trace)


def arrivals(recs: list[dict], plan: Mapping[int, dict], client: Mapping[str, dict], lo: int, hi: int) -> list[dict]:
    """The requests that arrived before served steps ``lo`` … ``hi`` - 1 (step-trace indices), each with its engine id,
    the step it arrived before, its prompt and its sampling, in arrival order."""
    out = []
    for r in recs:
        if r["kind"] == "arrive" and lo <= r["before_step"] < hi:
            c = client[client_id(r["request"])]
            out.append({"engine_id": r["request"], "before_step": r["before_step"], "prompt_ids": plan[c["i"]]["prompt_ids"],
                        "max_tokens": r["sampling"].get("max_tokens"), "sampling": r["sampling"]})
    return out


def interval_tokens(recs: list[dict], client: Mapping[str, dict], lo: int, hi: int) -> dict[str, list[int]]:
    """Each request's tokens emitted in served steps ``lo`` … ``hi`` - 1: its client tokens from the position its
    earlier steps' emitted counts reach."""
    before: dict[str, int] = {}
    during: dict[str, int] = {}
    for r in recs:
        if r["kind"] != "step":
            continue
        tgt = before if r["step"] < lo else during if r["step"] < hi else None
        if tgt is None:
            continue
        for rid, n in (r.get("emitted") or {}).items():
            tgt[rid] = tgt.get(rid, 0) + int(n)
    out = {}
    for rid, n in during.items():
        toks = client[client_id(rid)]["output_ids"]
        k = before.get(rid, 0)
        out[rid] = toks[k:k + n]
    return out


def first_step_difference(served: list[dict], replayed: list[dict]) -> dict | None:
    """The first step whose scheduled rows differ between two step traces' step records (by step index), or None."""
    a = {s["step"]: s["rows"] for s in served}
    b = {s["step"]: s["rows"] for s in replayed}
    for step in sorted(set(a) | set(b)):
        if a.get(step) != b.get(step):
            return {"step": step, "served": a.get(step), "replayed": b.get(step)}
    return None


def first_token_difference(want: Mapping[str, list[int]], got: Mapping[str, list[int]]) -> dict | None:
    """The request whose tokens first differ, by position (then request id), or None."""
    firsts = []
    for rid in sorted(set(want) | set(got)):
        a, b = want.get(rid, []), got.get(rid, [])
        if a != b:
            i = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
            firsts.append((i, rid, len(a), len(b)))
    if not firsts:
        return None
    i, rid, na, nb = min(firsts)
    return {"request": rid, "position": i, "tokens_served": na, "tokens_replayed": nb, "requests_differing": len(firsts)}


def edge_difference(served: Mapping[str, Any], replayed: Mapping[str, Any], served_dir: Path, replayed_dir: Path) -> dict | None:
    """Where rank 0's blocks first differ at an edge, served against replayed: the block sets, else the first layer and
    block (from the two digest files), or None if every layer's digest agrees."""
    if served["ids"] != replayed["ids"]:
        a, b = set(served["ids"]), set(replayed["ids"])
        return {"blocks": "differ", "served_only": sorted(a - b)[:16], "replayed_only": sorted(b - a)[:16]}
    if served["ranks"][0]["layer_digests"] == replayed["ranks"][0]["layer_digests"]:
        return None
    return tp_state.first_difference((served_dir / "digests.rank0.bin").read_bytes(),
                                     (replayed_dir / "digests.rank0.bin").read_bytes(), served["layers"], served["ids"])


def read_outputs(path: Path, lo: int, hi: int) -> dict[int, dict]:
    """`tp_edge.record_outputs`'s steps ``lo`` … ``hi`` - 1: step -> request -> [tokens, logprob bits], from ``path`` or
    the runner's gzipped copy of it."""
    gz = path.with_name(path.name + ".gz")
    if not path.exists() and gz.exists():
        path = gz
    if not path.exists():
        return {}
    text = gzip.decompress(path.read_bytes()).decode() if path.suffix == ".gz" else path.read_text()
    recs = (json.loads(x) for x in text.splitlines() if x)
    return {r["step"]: r["out"] for r in recs if lo <= r["step"] < hi}


def by_request(outs: Mapping[int, Mapping[str, list]]) -> dict[str, tuple[list[int], list[int]]]:
    """Step outputs joined per request, in step order: its tokens and their logprob bits."""
    got: dict[str, tuple[list[int], list[int]]] = {}
    for s in sorted(outs):
        for rid, (toks, bits) in outs[s].items():
            t, b = got.setdefault(rid, ([], []))
            t.extend(toks)
            b.extend(bits or ())
    return got


def class_outputs(served: Mapping[int, Mapping[str, list]], replayed: Mapping[int, Mapping[str, list]],
                  classes: Mapping[str, str]) -> dict[str, dict[str, int]]:
    """Per sampling class (`request_class`, by client id), the requests with tokens in either run's steps, their served
    tokens, and the requests whose tokens or logprob bits differ."""
    a, b = by_request(served), by_request(replayed)
    out: dict[str, dict[str, int]] = {}
    for rid in sorted(set(a) | set(b)):
        k = out.setdefault(classes.get(client_id(rid), "unknown"), {"requests": 0, "tokens": 0, "differing": 0})
        k["requests"] += 1
        k["tokens"] += len(a[rid][0]) if rid in a else 0
        k["differing"] += a.get(rid) != b.get(rid)
    return out


TOPK_KEYS = 2048
"""GLM-5.3's DSA indexer's top-k: a query over at most this many keys attends to every one of them, so the order and
ties of the indexer's selection don't enter (past it, a replay from the start differs too)."""


def keys_class(keys: int) -> str:
    return f"keys<={TOPK_KEYS}" if keys <= TOPK_KEYS else f"keys>{TOPK_KEYS}"


def max_keys(served: Mapping[int, Mapping[str, list]], first: Mapping[str, Mapping[str, Any]],
             far: Mapping[str, Mapping[str, Any]], plan: Mapping[int, Mapping[str, Any]]) -> dict[str, int]:
    """By client id, the most keys each request attends over by the far edge: its prompt and every token it produced but
    the last, or the tokens in its KV if more (a prefill part done). ``first`` and ``far`` are the two edges' requests
    (`tp_edge.held`): the far one's counts are a held request's own, the first one's its tokens before the step
    outputs start."""
    def keys(h: Mapping[str, Any], more: int = 0) -> int:
        out = h["output"] + more
        return max(h["computed"], h["prompt"] + out - 1) if out else h["computed"]

    at1, at2 = ({client_id(r): h for r, h in e.items()} for e in (first, far))
    n = {client_id(r): len(t) for r, (t, _) in by_request(served).items()}
    got = {}
    for cid in sorted(set(at1) | set(at2) | set(n)):
        if cid in at2:
            got[cid] = keys(at2[cid])
            continue
        prompt = len(plan.get(int(cid.split("-")[1]), {}).get("prompt_ids", ()))
        got[cid] = keys(at1.get(cid) or {"prompt": prompt, "output": 0, "computed": 0}, n.get(cid, 0))
    return got


def held_classes(held: Mapping[str, Mapping[str, Any]], classes: Mapping[str, str]) -> dict[str, dict[str, int]]:
    """An edge's requests (`tp_edge.held`) per class (by client id), by state."""
    out: dict[str, dict[str, int]] = {}
    for rid, h in held.items():
        k = out.setdefault(classes.get(client_id(rid), "unknown"), {"prefilling": 0, "decoding": 0, "waiting": 0})
        k[h["state"]] += 1
    return out


def block_classes(blocks: Iterable[int], held: Mapping[str, Mapping[str, Any]], classes: Mapping[str, str]) -> dict[str, int]:
    """``blocks`` by the classes of the edge's requests holding them (`tp_edge.held`; a block requests of two classes
    share counts in both), the rest as ``free``: cached blocks no request holds."""
    owner: dict[int, set[str]] = {}
    for rid, h in held.items():
        for b in h["blocks"]:
            owner.setdefault(b, set()).add(classes.get(client_id(rid), "unknown"))
    out: dict[str, int] = {}
    for b in blocks:
        for k in owner.get(b, {"free"}):
            out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


def differing_requests(served: Mapping[int, Mapping[str, list]], replayed: Mapping[int, Mapping[str, list]],
                       plan: Mapping[int, Mapping[str, Any]], keys: Mapping[str, int],
                       first: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The requests whose step outputs differ, by client id, each with its plan kind, prompt length, most keys
    (`max_keys`) and state at the first edge (None if it arrived after it)."""
    a, b = by_request(served), by_request(replayed)
    state = {client_id(r): h["state"] for r, h in first.items()}
    got = []
    for rid in sorted(set(a) | set(b)):
        if a.get(rid) != b.get(rid):
            cid = client_id(rid)
            r = plan.get(int(cid.split("-")[1]), {})
            got.append({"id": cid, "kind": r.get("kind", "plain"), "prompt_len": len(r.get("prompt_ids", ())),
                        "keys": keys.get(cid), "held": state.get(cid)})
    return got


def snapshot_bytes(files: Mapping[str, int]) -> dict[str, Any]:
    """An edge's snapshot by part (`file_bytes`): the KV dump, rank 0's that every rank loads when the ranks agree; the
    scheduler's state; each rank's runner rows and generators (`boundary.save_rank`); and what one rank reads."""
    kv = sum(n for f, n in files.items() if f.startswith("kv.rank"))
    ranks = {int(m.group(1)): n for f, n in files.items() if (m := re.fullmatch(r"rank-(\d+)\.pkl", f))}
    return {"kv": kv, "state": files.get("state.pkl", 0), "ranks": ranks,
            "per_rank": kv + max(ranks.values(), default=0)}


def generators(rng: Mapping[str, Any]) -> dict[str, Any]:
    """An `tp_state.rng_state` without V2's per-request seeds, which the runner's rows hold (the loader's to restore)."""
    return {k: v for k, v in rng.items() if k != "seeds"}


def restored_exact(served: Sequence[Mapping[str, Any]], restored: Sequence[Mapping[str, Any]], seeds: bool) -> dict[str, bool]:
    """Whether every rank's blocks and generators, and with ``seeds`` its requests' seeds, equal the served edge's."""
    out = {"kv": [g["layer_digests"] for g in restored] == [r["layer_digests"] for r in served],
           "generators": [generators(g["rng"]) for g in restored] == [generators(r["rng"]) for r in served]}
    if seeds:
        out["seeds"] = [g["rng"].get("seeds") for g in restored] == [r["rng"].get("seeds") for r in served]
    return out


def autotune_file(root: Path, config_hash: str) -> Path:
    """Where an engine of this config hash (``VllmConfig.compute_hash(include_version=False)``) keeps FlashInfer's
    autotune results under ``root``, as `resolve_flashinfer_autotune_file` puts them with the directory override."""
    return root / hashlib.sha256(config_hash.encode()).hexdigest() / "autotune_configs.json"


def carry_autotune(served_root: Path, root: Path, config_hash: str) -> dict[str, Any]:
    """The served engine's FlashInfer autotune results put where an engine of ``config_hash`` reads them at startup, so
    that it runs the served kernels instead of timing its own (its config hash may differ, the block count pinned)."""
    files = sorted(served_root.glob("*/autotune_configs.json"))
    dst = autotune_file(root, config_hash)
    if len(files) > 1:
        raise SystemExit(f"{served_root}: {len(files)} autotune files, where one served engine writes one")
    if not files:
        return {"served": None, "restored": dst.parent.name}
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(files[0], dst)
    return {"served": files[0].parent.name, "restored": dst.parent.name, "same_config": files[0].parent.name == dst.parent.name}


def resolve(name: str) -> Callable[..., Any]:
    return E.resolve(name)


def restore(a: argparse.Namespace) -> int:
    served, scratch, out = Path(a.served), Path(a.scratch), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((served / "meta.json").read_text())
    if len(meta.get("edges", ())) < 2:
        raise SystemExit(f"{served}: {len(meta.get('edges', ()))} edge(s), a restore needs 2")
    e1, e2 = meta["edges"][0], meta["edges"][1]
    edge1, edge2 = Path(e1["dir"]), Path(e2["dir"])
    rec1, rec2 = (json.loads((d / "edge.json").read_text()) for d in (edge1, edge2))
    sys.path.insert(0, str(ROOT / "adapters/vllm"))
    os.environ.update(PINS | {"VLLM_ENABLE_V1_MULTIPROCESSING": "0", AUTOTUNE_ENV: str(scratch / "autotune"),
                              "PYTHONPATH": os.pathsep.join(dict.fromkeys([str(HERE), *filter(None, sys.path)]))})
    recs = served_trace(served)
    header = recs[0]
    client = {c["id"]: c for c in (json.loads(x) for x in (served / "load" / "client.jsonl").read_text().splitlines() if x)}
    plan_file = Path(meta["plan"])
    plan = {r["i"]: r for r in (json.loads(x) for x in plan_file.read_text().splitlines() if x.strip())}
    s1, s2 = rec1["trace_step"], rec2["trace_step"]
    res: dict[str, Any] = {"schema": SCHEMA, "served": str(served), "edge1": rec1["edge"], "edge2": rec2["edge"],
                           "trace_steps": [s1, s2], "fill": not a.no_fill, "loader": a.loader,
                           "held": {k: rec1[k] for k in ("running", "in_flight", "waiting")},
                           "edge_seconds": [e["seconds"] for e in meta["edges"]],
                           "snapshot_bytes": snapshot_bytes(file_bytes(edge1)) | {
                               "kv_live": len(rec1["ids"]) * sum(x["block_bytes"] for x in rec1["layers"])}}

    from verity_vllm.engine.hooks import Hooks
    from verity_vllm.engine.step_trace import StepTrace
    from vllm.usage.usage_lib import UsageContext
    from vllm.v1.engine.llm_engine import LLMEngine
    from vllm.v1.executor import Executor

    verdict = out / "restore.json"
    t0 = time.monotonic()
    usage = UsageContext.OPENAI_API_SERVER
    args = engine_args(meta["server_argv"], header["config"])
    vc = args.create_engine_config(usage)
    res["autotune"] = carry_autotune(Path(meta["autotune_dir"]), scratch / "autotune", vc.compute_hash(include_version=False))
    res["log_stats"] = not args.disable_log_stats
    eng = LLMEngine(vllm_config=vc, executor_class=Executor.get_class(vc), log_stats=res["log_stats"],
                    usage_context=usage, multiprocess_mode=False)
    core = eng.engine_core.engine_core
    rpc = core.model_executor.collective_rpc
    res["build_s"] = round(time.monotonic() - t0, 1)
    probe = StepTrace().install(core.scheduler)
    probe.uninstall()
    res["config_difference"] = config_difference(header["config"], probe.records[0]["config"])
    if res["config_difference"]:
        res["verdict"] = "the restored engine's scheduler or cache settings differ from the served one's"
        verdict.write_text(json.dumps(res, indent=1))
        return 1

    t0 = time.monotonic()
    res["loader_out"] = resolve(a.loader)(core, edge1) if a.loader else None
    res["kv_load"] = rpc(tp_state.load_rank, args=(str(edge1), None, not a.no_fill))
    res["rng_set"] = rpc(tp_state.set_rng_ranked, args=({r["rank"]: r["rng"] for r in rec1["ranks"]},))
    check_dir = scratch / "restore-check"
    got = rpc(tp_state.digest, args=(rec1["ids"], str(check_dir)))
    res["restore_s"] = round(time.monotonic() - t0, 1)
    mine = sorted(got, key=lambda r: r["rank"]["rank"])
    res["restored_exact"] = restored_exact(rec1["ranks"], mine, seeds=bool(a.loader))
    res["rank_map"] = E.check_rank_map(rec1["rank_map"], E.rank_map(got))
    if not res["restored_exact"]["kv"]:
        res["verdict"] = "the restore is not exact at the edge"
        res["first_difference"] = edge_difference(rec1, {"ids": rec1["ids"], "ranks": [{"layer_digests": mine[0]["layer_digests"]}]},
                                                  edge1, check_dir)
        verdict.write_text(json.dumps(res, indent=1))
        return 1
    if res["rank_map"] and not a.rank_map_free:
        res["verdict"] = "the rank map differs from the served one (--rank-map-free to run anyway)"
        verdict.write_text(json.dumps(res, indent=1))
        return 1
    if not a.loader:
        ok = all(res["restored_exact"].values())
        res["verdict"] = (("the edge's blocks and generators restored exactly" if ok else
                           "the edge's blocks restored exactly, its generators not") +
                          "; no --loader, so the scheduler's and runner's state wasn't and nothing is replayed")
        verdict.write_text(json.dumps(res, indent=1))
        return 0 if ok else 1

    trace = StepTrace().install(core.scheduler)
    trace.scheduled = s1
    core.scheduler._verity_trace = trace
    edges = E.install(core, E.Edges(scratch / "replay", steps=[s2], dump=0, counter=lambda c: trace.scheduled))
    pending = [r for r in arrivals(recs, plan, client, s1, s2) if r["engine_id"] not in core.scheduler.requests]
    tokens: dict[str, list[int]] = {}

    def add_due() -> None:
        while pending and pending[0]["before_step"] <= trace.scheduled:
            r = pending.pop(0)
            eng.add_request(r["engine_id"], {"prompt_token_ids": r["prompt_ids"]},
                            sampling_params(r["sampling"], meta["load"].get("logprobs")))

    def schedule(orig):
        def wrapped(*args, **kw):
            add_due()
            return orig(*args, **kw)
        return wrapped

    import boundary as B
    from servetax_plugin import step_outputs_hook

    outs: dict[int, dict] = {}
    hooks = Hooks(tag="tp-restore")
    hooks.wrap(core.scheduler, "schedule", schedule)
    hooks.wrap(core.scheduler, "update_from_output",
               step_outputs_hook(trace, lambda rec: outs.__setitem__(rec["step"], rec["out"]), lambda: True))
    t0 = time.monotonic()
    while not edges.taken:
        if not core.scheduler.has_requests() and not core.batch_queue:
            add_due()
            if not core.scheduler.has_requests():
                res["verdict"] = f"the replay went idle at trace step {trace.scheduled} before the second edge ({s2})"
                verdict.write_text(json.dumps(res, indent=1))
                return 1
        outputs, executed = core.step_fn()
        core.post_step(model_executed=executed)
        for eco in (outputs or {}).values():
            for o in eco.outputs:
                if o.new_token_ids:
                    tokens.setdefault(o.request_id, []).extend(int(t) for t in o.new_token_ids)
    res["replay_s"] = round(time.monotonic() - t0, 1)
    hooks.uninstall()
    trace.uninstall()

    rep2 = json.loads((scratch / "replay" / f"edge-{edges.taken[0]['edge']:06d}" / "edge.json").read_text())
    served_steps = [r for r in recs if r["kind"] == "step" and s1 <= r["step"] < s2]
    replayed_steps = [r for r in trace.records if r["kind"] == "step" and r["step"] < s2]
    res["steps"] = {"served": len(served_steps), "replayed": len(replayed_steps),
                    "first_difference": first_step_difference(served_steps, replayed_steps)}
    want = interval_tokens(recs, client, s1, s2)
    res["tokens"] = {"requests": len(want), "served": sum(map(len, want.values())), "replayed": sum(map(len, tokens.values())),
                     "first_difference": first_token_difference(want, tokens)}
    served_outs = read_outputs(served / "edges" / "outputs.jsonl", s1, s2)
    replayed_outs = {s: o for s, o in outs.items() if s < s2}
    classes_file = served / "load" / "classes.json"
    classes = json.loads(classes_file.read_text()) if classes_file.exists() else {}
    kinds = {f"{ID}-{i:05d}": r.get("kind", "plain") for i, r in plan.items()}
    keys = max_keys(served_outs, rec1["requests"], rec2["requests"], plan)
    by_keys = {cid: keys_class(k) for cid, k in keys.items()}
    res["held"]["keys"] = held_classes(rec1["requests"], by_keys)
    res["outputs"] = B.outputs_diff(served_outs, replayed_outs) | {
        "recorded": bool(served_outs), "classes": class_outputs(served_outs, replayed_outs, classes),
        "kinds": class_outputs(served_outs, replayed_outs, kinds),
        "keys": class_outputs(served_outs, replayed_outs, by_keys),
        "differing": differing_requests(served_outs, replayed_outs, plan, keys, rec1["requests"])}
    rep2_dir = scratch / "replay" / f"edge-{edges.taken[0]['edge']:06d}"
    first2 = edge_difference(rec2, rep2, edge2, rep2_dir)
    res["edge2"] = {"blocks": len(rec2["ids"]), "first_difference": first2,
                    "kv_root": {"served": kv_root(rec2), "replayed": kv_root(rep2)}, "replay_check": rep2["check"],
                    "replay_seconds": rep2["seconds"], "held": held_classes(rec2["requests"], by_keys)}
    if first2 and "layer" in first2:
        res["edge2"]["differing_by_keys"] = block_classes(
            tp_state.differing_blocks((edge2 / "digests.rank0.bin").read_bytes(), (rep2_dir / "digests.rank0.bin").read_bytes(),
                                      rec2["layers"], rec2["ids"]), rec2["requests"], by_keys)
    res["bit_exact"] = (res["steps"]["first_difference"] is None and res["tokens"]["first_difference"] is None
                        and res["outputs"]["recorded"] and res["outputs"]["first"] is None
                        and res["edge2"]["first_difference"] is None)
    res["verdict"] = "bit-exact" if res["bit_exact"] else "differs"
    verdict.write_text(json.dumps(res, indent=1))
    print(f"[tp-restore] {res['verdict']}: steps {res['steps']['first_difference']}, outputs {res['outputs']['first']} "
          f"(recorded {res['outputs']['recorded']}), tokens {res['tokens']['first_difference']}, edge 2 "
          f"{res['edge2']['first_difference']}; requests differing by keys {res['outputs']['keys']}, far-edge blocks "
          f"{res['edge2'].get('differing_by_keys')}", flush=True)
    return 0 if res["bit_exact"] else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("preflight")
    f.add_argument("--model-id", default="glm53", choices=sorted(S.MODELS))
    f.add_argument("--checkpoint")
    f.add_argument("--max-model-len", type=int)
    f.add_argument("--tp", type=int)
    f.add_argument("--kv-cache-dtype")
    s = sub.add_parser("serve")
    s.add_argument("--model-id", default="glm53", choices=sorted(S.MODELS))
    s.add_argument("--checkpoint")
    s.add_argument("--c4", required=True)
    s.add_argument("--plan")
    s.add_argument("--plan-file", help="a ready plan, used as it is (its requests may carry their own sampling)")
    s.add_argument("--unseeded-share", type=float, default=0.5,
                   help="the share of a plan's sampled requests sent without a seed (request_class)")
    s.add_argument("--docs", type=int, default=4000)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--long-share", type=float, default=0.0, help="the share of requests given long prompts (lengthen)")
    s.add_argument("--long-tokens", type=lambda x: [int(t) for t in x.split(",")], default=[8192, 32768])
    s.add_argument("--rate", type=float, default=10.0)
    s.add_argument("--span-s", type=float, default=240.0)
    s.add_argument("--requests", type=int)
    s.add_argument("--edges-at", type=lambda x: [float(t) for t in x.split(",")], default=[120.0, 170.0, 220.0])
    s.add_argument("--sampling", default=OWN, help=f"{OWN} (each request's), or one of the corpus's, or production")
    s.add_argument("--max-model-len", type=int)
    s.add_argument("--tp", type=int, help="the tensor-parallel size, when not the model's own")
    s.add_argument("--kv-cache-dtype", help="the KV cache's format (vLLM's --kv-cache-dtype), when not vLLM's default; the "
                   "restore's engine takes it from the served argv")
    s.add_argument("--saver", action="append", default=[])
    s.add_argument("--dump", type=int, default=1, help="how many of the first edges write their blocks")
    s.add_argument("--num-gpu-blocks", type=int, help="the KV pool's blocks (vLLM's --num-gpu-blocks-override), each dump's size")
    s.add_argument("--startup-s", type=float, default=3600.0)
    s.add_argument("--drain-s", type=float, default=600.0)
    s.add_argument("--run-out", action="store_true",
                   help="serve the whole load and its drain; by default the server stops once the last edge is written")
    s.add_argument("--out", required=True)
    s.add_argument("--scratch", required=True)
    r = sub.add_parser("restore")
    r.add_argument("--served", required=True)
    r.add_argument("--scratch", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--loader")
    r.add_argument("--no-fill", action="store_true")
    r.add_argument("--rank-map-free", action="store_true")
    a = p.parse_args(argv)
    return {"preflight": preflight, "serve": serve, "restore": restore}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
