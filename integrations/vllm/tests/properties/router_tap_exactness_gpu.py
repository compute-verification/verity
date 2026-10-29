"""GPU run of the router-tap exactness property (`verity_vllm.properties.router_tap_exactness`): check the tap build on this GPU against
the installed `_moe_C.topk_softmax` and the circuit's ordered router, the router-softmax source on vLLM's routing path, and live on vLLM
engines serving tiny random-weight models of the two served router shapes (OLMoE's, Qwen3-MoE's); write the sealed record `<out>/router_tap_exactness.json`.  Exit status 1 unless ok; `--verify RECORD` re-derives a record's
digest and verdict without a GPU.

It lives under tests/ because it seeds random generators, sets VLLM_BATCH_INVARIANT for its own process and launches vLLM's kernels
directly (lints P3, P7).  In the pod venv, with the tree on PYTHONPATH as for gate (b):

    $PY tests/properties/router_tap_exactness_gpu.py --so /workspace/cp/router_tap/build/verity_router_tap.so --out DIR [--quick] [--no-live]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import multiprocessing as mp
import os
import platform
import random
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np

from verity_vllm.properties import router_tap_exactness as X
from verity_vllm.query import router_softmax as RS

LAYERS_TOY = {3: (64, 8, 8, False), 1: (128, 8, 8, True)}
NAN, PINF, NINF, PZ, NZ = 0x7FC0, 0x7F80, 0xFF80, 0x0000, 0x8000
PROMPTS = ("The capital of France is", "def fibonacci(n):\n    ", "Mixture-of-experts models route each token to")


def _bf16(v: float) -> int:
    import struct
    return struct.unpack("<I", struct.pack("<f", v))[0] >> 16


def edge_rows(E: int, rng: random.Random) -> dict[str, list[int]]:
    """Named bf16 logit rows at the router's edges (vu-export's `router_eq.cases`, plus a NaN and a
    subnormal at the head of a thread's columns, where the max's NaN and subnormal handling decide it)."""
    g = lambda: [_bf16(rng.gauss(0.0, 1.0)) for _ in range(E)]   # noqa: E731
    out: dict[str, list[int]] = {"all +0": [PZ] * E, "all -0": [NZ] * E, "mixed +-0": [PZ if i % 2 else NZ for i in range(E)],
                                 "all -inf": [NINF] * E, "all NaN": [NAN] * E, "all +inf": [PINF] * E}
    out["one +inf"] = [PZ] * E
    out["one +inf"][E // 3] = PINF
    out["two +inf"] = g()
    out["two +inf"][3] = out["two +inf"][E - 2] = PINF
    out["one NaN"] = g()
    out["one NaN"][5] = NAN
    out["NaN first"] = g()
    out["NaN first"][0] = NAN
    out["NaN heads thread 1"] = g()
    out["NaN heads thread 1"][8] = NAN
    out["negative NaN"] = g()
    out["negative NaN"][E - 1] = 0xFFC1
    out["some -inf"] = g()
    for i in range(0, E, 3):
        out["some -inf"][i] = NINF
    out["-inf and one finite"] = [NINF] * E
    out["-inf and one finite"][E - 1] = _bf16(1.5)
    out["all equal"] = [_bf16(2.0)] * E
    base = [rng.gauss(0.0, 1.0) for _ in range(E // 3 + 1)]
    out["ties of 3"] = [_bf16(base[i // 3]) for i in range(E)]
    out["huge (expf overflow)"] = [_bf16(rng.gauss(0.0, 1.0) * 1e4) for _ in range(E)]
    out["tiny spread (p ties after rounding)"] = [_bf16(10.0 + rng.gauss(0.0, 1.0) * 1e-3) for _ in range(E)]
    out["underflow to 0 but one"] = [_bf16(-200.0)] * (E - 1) + [_bf16(100.0)]
    out["subnormal logits"] = [rng.randrange(1, 0x7F) for _ in range(E)]
    out["neg subnormal"] = [0x8000 | rng.randrange(1, 0x7F) for _ in range(E)]
    out["subnormal and -0"] = [NZ if i % 3 else rng.randrange(1, 0x7F) for i in range(E)]
    out["max finite"] = [0x7F7F] * (E // 2) + [0xFF7F] * (E - E // 2)
    out["one zero rest -inf"] = [NINF] * E
    out["one zero rest -inf"][E // 2] = PZ
    return out


def random_rows(E: int, rng: random.Random, n: int) -> dict[str, list[int]]:
    """`n` seeded rows: gaussian at three scales, few distinct values (heavy ties), small integers (exact ties), and rows sprinkled with
    specials (NaN, +-inf, +-0, the largest finite, subnormals)."""
    out = {}
    for r in range(n):
        kind = r % 4
        if kind == 0:
            s = rng.choice([0.5, 2.0, 8.0])
            row = [_bf16(rng.gauss(0.0, 1.0) * s) for _ in range(E)]
        elif kind == 1:
            vals = [rng.gauss(0.0, 3.0) for _ in range(4)]
            row = [_bf16(rng.choice(vals)) for _ in range(E)]
        elif kind == 2:
            row = [_bf16(float(rng.randint(-3, 3))) for _ in range(E)]
        else:
            row = [_bf16(rng.gauss(0.0, 4.0)) for _ in range(E)]
            for i in rng.sample(range(E), rng.randint(1, 5)):
                row[i] = rng.choice([NAN, PINF, NINF, PZ, NZ, 0x7F7F, 0xFF7F, 0x0001, 0x8001])
        out[f"random {r} kind {kind}"] = row
    return out


def rows_of(E: int, seed: int, n_random: int) -> dict[str, list[int]]:
    rng = random.Random(seed * 1_000_003 + E)
    return {**edge_rows(E, rng), **random_rows(E, rng, n_random)}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _ir_job(job: tuple) -> tuple[list[int], list[int], list[int]]:
    E, TOPK, VPT, renorm, row = job
    return X.ir_row(E, TOPK, VPT, renorm, row)


def _environment() -> dict[str, Any]:
    import torch

    env = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0),
           "capability": ".".join(map(str, torch.cuda.get_device_capability(0))), "VLLM_BATCH_INVARIANT": os.environ.get("VLLM_BATCH_INVARIANT")}
    try:
        env["vllm"] = __import__("vllm").__version__
    except Exception as e:  # noqa: BLE001
        env["vllm"] = f"unavailable: {e}"[:160]
    return env


def _u32(t) -> np.ndarray:
    import torch

    return t.contiguous().view(torch.int32).cpu().numpy().view(np.uint32)


def _bf16_tensor(rows: list[list[int]]):
    import torch

    return torch.from_numpy(np.asarray(rows, dtype=np.uint16).view(np.int16)).view(torch.bfloat16).cuda()


def _launch(op, x, TOPK: int, renorm: bool, ind=None, tap=None, pad=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    M = x.shape[0]
    w = torch.empty(M, TOPK, dtype=torch.float32, device="cuda")
    ids = torch.empty(M, TOPK, dtype=ind or torch.int32, device="cuda")
    tei = torch.empty(M, TOPK, dtype=torch.int32, device="cuda")
    if tap is None:
        op(w, ids, tei, x, renorm, None, pad)
    else:
        op(w, ids, tei, x, renorm, None, pad, tap)
    torch.cuda.synchronize()
    return _u32(w), ids.cpu().numpy().astype(np.int64) & 0xFFFFFFFF, tei.cpu().numpy().astype(np.int64)


def _same(a: tuple, b: tuple) -> bool:
    return all(np.array_equal(x, y) for x, y in zip(a, b))


def check_config(tap, pool, E: int, TOPK: int, VPT: int, renorm: bool, n_random: int, seed: int) -> dict[str, Any]:
    import torch

    named = rows_of(E, seed, n_random)
    names, rows = list(named), list(named.values())
    n_edge = len(edge_rows(E, random.Random(0)))
    W = RS.words(E, TOPK, renorm)
    rec: dict[str, Any] = {"name": f"E{E}_k{TOPK}_{'norm' if renorm else 'plain'}", "E": E, "TOPK": TOPK, "VPT": VPT, "renormalize": renorm,
                           "rows": len(rows), "edge_rows": n_edge, "words_per_row": W}
    x = _bf16_tensor(rows)
    installed = torch.ops._moe_C.topk_softmax
    ref = _launch(installed, x, TOPK, renorm)
    untapped = _launch(tap.topk_softmax, x, TOPK, renorm)
    taps, outs = [], []
    for fill in X.FILLS:
        buf = torch.full((len(rows), W), fill, dtype=torch.int32, device="cuda").view(torch.float32)
        outs.append(_launch(tap.topk_softmax_tap, x, TOPK, renorm, tap=buf))
        taps.append(_u32(buf).reshape(len(rows), W))
    ref64 = _launch(installed, x, TOPK, renorm, ind=torch.int64)
    buf64 = torch.empty(len(rows), W, dtype=torch.float32, device="cuda")
    tapped64 = _launch(tap.topk_softmax_tap, x, TOPK, renorm, ind=torch.int64, tap=buf64)
    pad = torch.arange(len(rows), device="cuda") % 3 == 1                  # vLLM's is_padding mask: a padding row's ids, never its softmax
    ref_pad = _launch(installed, x, TOPK, renorm, pad=pad)
    buf_pad = torch.empty(len(rows), W, dtype=torch.float32, device="cuda")
    tapped_pad = _launch(tap.topk_softmax_tap, x, TOPK, renorm, tap=buf_pad, pad=pad)
    rec["padding_eq_installed"] = _same(tapped_pad, ref_pad) and bool(np.array_equal(_u32(buf_pad).reshape(len(rows), W), taps[0]))
    rec["out_eq_installed"] = all(_same(o, ref) for o in outs) and _same(tapped64, ref64) and rec["padding_eq_installed"]
    rec["out_eq_untapped"] = _same(untapped, ref)
    rec["unwritten"] = int(np.count_nonzero(taps[0] != taps[1]))
    t0 = time.time()
    ir = pool.map(_ir_job, [(E, TOPK, VPT, renorm, r) for r in rows], chunksize=8)
    rec["ir_seconds"] = round(time.time() - t0, 1)
    rec["ir_rows"] = len(ir)
    ir_words = np.asarray([w for w, _o, _i in ir], dtype=np.uint32)
    ir_w = np.asarray([o for _w, o, _i in ir], dtype=np.uint32)
    ir_ids = np.asarray([i for _w, _o, i in ir], dtype=np.int64)
    bad_rows = np.flatnonzero((taps[0] != ir_words).any(axis=1))
    rec["tap_ne_ir"] = int(len(bad_rows))
    by_value: dict[str, int] = {}
    for r, c in zip(*np.nonzero(taps[0] != ir_words)):
        k = X.classify(E, TOPK, renorm, int(c))
        by_value[k] = by_value.get(k, 0) + 1
    rec["tap_ne_ir_words_by_value"] = by_value
    rec["tap_ne_ir_first"] = [{"row": names[r], "logits": rows[r],
                               "words": [{"i": int(c), "value": X.classify(E, TOPK, renorm, int(c)), "tap": int(taps[0][r, c]), "ir": int(ir_words[r, c])}
                                         for c in np.flatnonzero(taps[0][r] != ir_words[r])[:6]]} for r in bad_rows[:4]]
    rec["out_ne_ir"] = int(np.count_nonzero((ref[0] != ir_w).any(axis=1) | (ref[1] != ir_ids).any(axis=1)))
    rec["edge_taps"] = {names[i]: [int(v) for v in taps[0][i]] for i in range(n_edge)}
    rec["ok"] = X.config_ok(rec)
    return rec


class _Recorder:
    """The committer surface the source uses: the current step's job, the error list, `_acquire` (kept here, cloned)."""

    def __init__(self) -> None:
        self._cur: Any = None
        self.errors: list[str] = []
        self.acquired: list[tuple[str, Any]] = []

    def _acquire(self, name: str, t: Any, job: Any) -> None:
        self.acquired.append((name, t.detach().clone()))


def _toy():
    import torch.nn as nn
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk

    class FusedMoE(nn.Module):
        """Stands for vLLM's MoE layer as the source attributes it (`moe_source.moe_layers_of`): vLLM's `fused_topk` over its logits."""

        def __init__(self, topk: int, renorm: bool) -> None:
            super().__init__()
            self.topk, self.renorm = topk, renorm

        def forward(self, h, logits, renorm=None):
            return fused_topk(h, logits, self.topk, self.renorm if renorm is None else renorm)

    root = nn.Module()
    root.model = nn.Module()
    root.model.layers = nn.ModuleList([nn.Module() for _ in range(4)])
    for k, (_E, TOPK, _VPT, renorm) in LAYERS_TOY.items():
        root.model.layers[k].mlp = nn.Module()
        root.model.layers[k].mlp.experts = FusedMoE(TOPK, renorm)
    return root


def check_source(so: str, seed: int) -> dict[str, Any]:
    import torch

    from verity_vllm.acquire.sources.router_softmax_source import RouterSoftmaxSource
    from verity_vllm.engine import hooks

    toy = _toy()
    layers = {k: {"module": f"model.layers.{k}.mlp.experts", "experts": E, "topk": K, "renormalize": r, "width": RS.words(E, K, r)}
              for k, (E, K, _V, r) in LAYERS_TOY.items()}
    g = torch.Generator(device="cuda").manual_seed(seed)
    T = 37
    inputs = {k: ((torch.randn(T, 16, device="cuda", generator=g)).to(torch.bfloat16),
                  (torch.randn(T, E, device="cuda", generator=g) * 3).to(torch.bfloat16)) for k, (E, _K, _V, _r) in LAYERS_TOY.items()}

    def forward(renorm_override=None):
        out = {}
        for k in sorted(LAYERS_TOY):
            h, lg = inputs[k]
            w, ids, _tei = toy.model.layers[k].mlp.experts(h, lg, renorm_override)
            torch.cuda.synchronize()
            out[k] = (_u32(w), ids.cpu().numpy())
        return out

    ref = forward()
    site = torch.ops._moe_C.topk_softmax
    com = _Recorder()
    src = RouterSoftmaxSource(com, toy, layers, so)
    src.install()
    rec: dict[str, Any] = {"T": T, "layers": {str(k): v for k, v in layers.items()}}
    try:
        outside = forward()
        rec["outside_step_acquired"] = len(com.acquired)
        com._cur = types.SimpleNamespace(step=0)
        got = forward()
        n_in = len(com.acquired)
        toy.model.layers[3].mlp.experts(*inputs[3], True)                      # layer 3's manifest router does not renormalize
        torch.cuda.synchronize()
        rec["refused_other_geometry"] = len(com.acquired) == n_in and len(com.errors) == 1 and "layer 3" in com.errors[0]
        com._cur = None
    finally:
        src.uninstall()
    rec["restored"] = torch.ops._moe_C.topk_softmax is site and not any(getattr(p, "owner", None) is torch.ops._moe_C for p in hooks.live())
    rec["outputs_equal"] = all(_same(ref[k], got[k]) and _same(ref[k], outside[k]) for k in LAYERS_TOY)
    names = [n for n, _t in com.acquired]
    rec["names"] = names
    rec["names_ok"] = names == [f"moe/L{k}/{RS.MEMBER}" for k in sorted(LAYERS_TOY)]
    tap = getattr(torch.ops, "verity_router_tap")
    eq = []
    for (n, t), k in zip(com.acquired, sorted(LAYERS_TOY)):
        E, K, _V, r = LAYERS_TOY[k]
        buf = torch.empty(T, RS.words(E, K, r), dtype=torch.float32, device="cuda")
        _launch(tap.topk_softmax_tap, inputs[k][1], K, r, tap=buf)
        eq.append(np.array_equal(_u32(t), _u32(buf)))
    rec["words_equal"] = bool(eq) and all(eq)
    rec["errors"], rec["error_msgs"] = len(com.errors), com.errors[:4]
    rec["ok"] = X.source_ok(rec)
    return rec


#: tiny random-weight models of the two served router shapes, served by vLLM's own MoE layers (E = 64 top-8 as OLMoE, E = 128 top-8
#: renormalized as Qwen3-30B-A3B), with the B0 checkpoint's tokenizer: the router path is vLLM's, no full checkpoint is downloaded
TINY = {"olmoe": {"E": 64, "K": 8, "renorm": False}, "qwen3moe": {"E": 128, "K": 8, "renorm": True}}


def make_tiny(kind: str, tok_dir: str, out_dir: str, seed: int) -> dict[str, Any]:
    import shutil

    import torch
    import transformers

    base = json.loads(Path(tok_dir, "config.json").read_text())
    ids = {k: base.get(k) for k in ("bos_token_id", "eos_token_id", "pad_token_id")}
    t = TINY[kind]
    common = dict(vocab_size=int(base["vocab_size"]), hidden_size=128, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
                  num_experts=t["E"], num_experts_per_tok=t["K"], norm_topk_prob=t["renorm"], max_position_embeddings=2048,
                  tie_word_embeddings=False, **ids)
    if kind == "olmoe":
        cfg = transformers.OlmoeConfig(intermediate_size=64, **common)
    else:
        cfg = transformers.Qwen3MoeConfig(intermediate_size=128, moe_intermediate_size=64, head_dim=32, decoder_sparse_step=1, mlp_only_layers=[],
                                          **common)
    torch.manual_seed(seed)
    model = transformers.AutoModelForCausalLM.from_config(cfg, torch_dtype=torch.bfloat16)
    model.save_pretrained(out_dir, safe_serialization=True)
    for f in Path(tok_dir).iterdir():
        if f.is_file() and f.name != "config.json" and (f.suffix in (".json", ".txt", ".model") and not f.name.endswith(".index.json")):
            shutil.copy(f, Path(out_dir, f.name))
    return {"kind": kind, "architecture": cfg.architectures[0] if cfg.architectures else type(model).__name__, "experts": t["E"], "topk": t["K"],
            "renormalize": t["renorm"], "vocab": common["vocab_size"], "seed": seed}


def check_live(so: str, pool, kind: str, tok_dir: str, work: str, max_tokens: int = 8) -> dict[str, Any]:
    import torch
    from vllm import LLM, SamplingParams

    from verity_vllm.acquire.sources.moe_source import moe_layers_of
    from verity_vllm.acquire.sources.router_softmax_source import RouterSoftmaxSource
    from verity_vllm.engine import handle, hooks

    d = Path(work, f"tiny-{kind}")
    rec: dict[str, Any] = make_tiny(kind, tok_dir, str(d), 20260926)
    E, K, renorm = rec["experts"], rec["topk"], rec["renormalize"]
    llm = LLM(model=str(d), tokenizer=str(d), dtype="bfloat16", enforce_eager=True, max_model_len=512, gpu_memory_utilization=0.3,
              enable_prefix_caching=False, max_num_seqs=4, seed=0)
    model = handle.model_runner(llm).get_model()
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)

    def generate():
        return [list(o.outputs[0].token_ids) for o in llm.generate(list(PROMPTS), sp, use_tqdm=False)]

    bare = generate()
    mods = moe_layers_of(model)
    layers = {k: {"module": n, "experts": E, "topk": K, "renormalize": renorm, "width": RS.words(E, K, renorm)} for n, k in mods.items()}
    com = _Recorder()
    com._cur = types.SimpleNamespace(step=0)
    src = RouterSoftmaxSource(com, model, layers, so)
    logits: list[Any] = []
    forwards = [0]
    src.install()
    rec_hooks = hooks.Hooks()
    inner = torch.ops._moe_C.topk_softmax

    def recording(w, ids, tei, gating, renormalize=False, bias=None, is_padding=None):
        logits.append(gating.detach().clone())
        return inner(w, ids, tei, gating, renormalize, bias, is_padding)

    rec_hooks.patch(torch.ops._moe_C, "topk_softmax", recording)
    h = model.register_forward_pre_hook(lambda _m, _a: forwards.__setitem__(0, forwards[0] + 1))
    try:
        served = generate()
    finally:
        h.remove()
        rec_hooks.uninstall()
        src.uninstall()
    rec.update(moe_layers=len(layers), forwards=forwards[0], acquired=len(com.acquired), launches=len(logits), prompts=len(PROMPTS),
               max_tokens=max_tokens, tokens_bare=bare, tokens_served=served, tokens_equal_bare=bare == served)
    rec["every_layer_every_step"] = len(com.acquired) == len(logits) == len(layers) * forwards[0] and forwards[0] > 0 and len(layers) > 0
    jobs, taps = [], []
    for (_name, t), lg in zip(com.acquired, logits):
        rows = lg.reshape(-1, E).contiguous().view(torch.int16).cpu().numpy().view(np.uint16).astype(np.int64)
        tw = _u32(t).reshape(len(rows), -1)
        for i in range(len(rows)):
            jobs.append((E, K, 8, renorm, [int(v) for v in rows[i]]))
            taps.append(tw[i])
    t0 = time.time()
    ir = pool.map(_ir_job, jobs, chunksize=16)
    rec["ir_seconds"] = round(time.time() - t0, 1)
    ne = [i for i, (words, _w, _ids) in enumerate(ir) if not np.array_equal(np.asarray(words, dtype=np.uint32), taps[i])]
    rec["rows_checked"], rec["rows_ne_ir"] = len(jobs), len(ne)
    rec["errors"], rec["error_msgs"] = len(com.errors), com.errors[:4]
    del llm
    __import__("gc").collect()
    torch.cuda.empty_cache()
    return rec


def _guarded(fn, base: dict[str, Any], *args) -> dict[str, Any]:
    try:
        return fn(*args)
    except Exception as e:  # noqa: BLE001
        return {**base, "error": f"{type(e).__name__}: {e}"[:400], "ok": False}


def run(so: str, out: str, quick: bool = False, live: bool = True) -> dict[str, Any]:
    """Check the tap build at `so` on this GPU and write the sealed record to `<out>/router_tap_exactness.json`."""
    pool = mp.get_context("fork").Pool(max(2, min(32, (os.cpu_count() or 4) - 2)))   # before CUDA: the circuit evaluation needs none
    from verity_vllm.engine import engine_profile as prof

    env_flags = prof.apply_env()                               # the engine's pins (batch invariance, in-process core, no FlashInfer sampler)
    os.environ["VLLM_BATCH_INVARIANT"] = "1"
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    import torch
    import vllm
    import vllm._custom_ops  # loads the installed _moe_C ops

    from verity_vllm.acquire.router_tap_src.build_ext import NAME, TAP_VERSION

    torch.ops.load_library(so)
    tap = getattr(torch.ops, NAME)
    installed = sorted(glob.glob(os.path.join(os.path.dirname(vllm.__file__), "_moe_C*.so")))
    doc: dict[str, Any] = {
        "schema": X.SCHEMA, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "argv": sys.argv, "environment": _environment(), "quick": quick,
        "env_flags": env_flags,
        "tap": {"so": so, "sha256": _sha256(so), "tap_version": int(tap.tap_version()), "want_version": TAP_VERSION,
                "build_info": json.loads(Path(so).with_name("build_info.json").read_text()) if Path(so).with_name("build_info.json").exists() else None},
        "installed": {"op": "_moe_C.topk_softmax", "so": {os.path.basename(p): _sha256(p) for p in installed}}, "configs": []}
    n_random = X.QUICK_RANDOM if quick else X.RANDOM
    for i, (E, K, V, r) in enumerate(X.CONFIGS):
        rec = _guarded(check_config, {"name": f"E{E}_k{K}_{'norm' if r else 'plain'}", "E": E, "TOPK": K, "VPT": V, "renormalize": r},
                       tap, pool, E, K, V, r, n_random, 2026 + i)
        doc["configs"].append(rec)
        print(f"[router-tap] {rec['name']:14s} " + (rec.get("error") or
              f"rows {rec['rows']} ==installed {rec['out_eq_installed']} ==untapped {rec['out_eq_untapped']} unwritten {rec['unwritten']} "
              f"!=ir {rec['tap_ne_ir']} {rec['tap_ne_ir_words_by_value']} out!=ir {rec['out_ne_ir']} ({rec['ir_seconds']} s)")
              + f" -> {'OK' if rec['ok'] else 'FAIL'}", flush=True)
        for m in rec.get("tap_ne_ir_first") or []:
            print(f"[router-tap]    {m['row']}: {m['words']}", flush=True)
    doc["source"] = _guarded(check_source, {}, so, 7)
    print(f"[router-tap] source {json.dumps({k: v for k, v in doc['source'].items() if k != 'layers'})[:900]}", flush=True)
    if live and not quick:
        tok_dir = prof.checkpoint_for("B0", "manifests/checkpoints.json")["local_path"]
        doc["live"] = {"tokenizer": tok_dir, "models": [_guarded(check_live, {"kind": k}, so, pool, k, tok_dir, out) for k in TINY]}
        for m in doc["live"]["models"]:
            m["ok"] = X.live_model_ok(m)
        doc["live"]["ok"] = X.live_ok(doc["live"])
        for m in doc["live"]["models"]:
            print(f"[router-tap] live {json.dumps({k: v for k, v in m.items() if not k.startswith('tokens_')})[:900]}", flush=True)
    pool.close()
    doc = X.seal(doc)
    path = Path(out) / X.RECORD
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"ROUTER-TAP-EXACTNESS {'OK' if doc['ok'] else 'FAIL'} {doc['digest']} {path}", flush=True)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="router-tap exactness: the tapped topk_softmax computes vLLM's outputs and the IR's committed softmax")
    ap.add_argument("--so", help="the tap build: verity_router_tap.so (ops/pod_router_tap.sh)")
    ap.add_argument("--out", help=f"directory for {X.RECORD}")
    ap.add_argument("--quick", action="store_true", help=f"{X.QUICK_RANDOM} random rows per configuration, no live check")
    ap.add_argument("--no-live", action="store_true", help="skip the live check on the tiny MoE models")
    ap.add_argument("--verify", metavar="RECORD", help="check a record's digest and verdict (no GPU)")
    a = ap.parse_args(argv)
    if a.verify:
        doc = json.loads(Path(a.verify).read_text())
        problems = X.verify_record(doc)
        print(f"{a.verify}: {'intact, ok=' + str(doc.get('ok')) if not problems else '; '.join(problems)} (digest {doc.get('digest')})")
        return 1 if problems or not doc.get("ok") else 0
    if not (a.so and a.out):
        ap.error("--so and --out are required (or --verify RECORD)")
    out = str(Path(a.out).resolve())
    os.chdir(Path(__file__).resolve().parents[2])
    return 0 if run(a.so, out, a.quick, live=not a.no_live)["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
