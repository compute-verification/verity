"""GLM-5.3's MLA RMSNorms on an H200: `_fused_norm_rope_kernel`'s `_rms_norm` (program 2's q_a_layernorm over q_c [2048], program
1's kv_a_layernorm over kv_c [512], eps 1e-5) against `dsa_norm.DsaRMSNorm_v1{N, EPS=1e-05}`, the Definition read from the kernel's
H200 PTX (its module docstring), and against the models it was chosen over (`ALTERNATIVES`).

One GPU process writes `<out>/glm53_fused_rmsnorm_capture.json`, the property record `glm53_fused_rmsnorm_capture`
(`properties.record`), and the kernel's PTX under `<out>/ptx/`.  The device gate runs first and nothing else runs when it fails
(exit 3).  vLLM runs as in `glm53_dsa_capture_gpu` (its `environment`: the Triton cache in the scratch directory).

  gate     `fa2_target_capture_gpu.gate`: driver, compute capability 9.0, 132 SMs.
  served   `fused_norm_rope` as `DeepseekV32Attention.forward` calls it (`glm53_dsa_capture_gpu.norm_rope`) on that capture's operands
           (`operands`: SEQ + EXTRA tokens, each row of q_c and kv_c at a scale 2^U(-3, 3), weights 1 + 0.1 N(0, 1)) with
           `special_rows` planted at tokens SPECIAL: twice on fresh caches, the words equal, and every written MLA cache row's first
           512 words the kv_c output.
  probe    the same launch with the first rows of q_c and kv_c and both weights `probe`'s: at eps 1e-5 the models agree on almost
           every word of served rows (a bf16 output moves only when an f32 ulp of r crosses its rounding boundary), so `probe`
           picks rows where an alternative's r (or product order) differs and puts, in columns of their own, weights in [1, 2) at
           which the bf16 words differ, PER_ALT words for each alternative.
  words    every output row of both launches against the twin (`dsa_norm_rows.dsa_rmsnorm_rows`); the reference evaluator on the
           SPECIAL rows, the probe rows and SAMPLE; and each alternative's words against the kernel's, counted (`refuted`).
  ptx      the Triton cache holds one specialisation of `_fused_norm_rope_kernel`, and its code (the text before the first
           `.file`) is the PTX the Definition was read from (PTX_CODE_SHA256; PTX_COUNTS are its loads, reductions and epilogue).
  bind     DsaRMSNorm_v1 and its Boolean version bind at N 2048 and 512, EPS 1e-5.

GO = the gate, both launches deterministic with the cache rows, every word of every row equal to the twin, the reference equal to
the twin on every sampled row, every alternative refuted on at least PER_ALT words of each norm, the PTX and the bindings, no error.

    cd integrations/vllm; PY=/workspace/venv312/bin/python     # PYTHONPATH as pod_bootstrap.sh sets it
    $PY tests/properties/glm53_fused_rmsnorm_capture_gpu.py --out DIR --scratch /tmp --want-cc 9.0 --want-sms 132 --min-driver 575
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import multiprocessing
import os
import re
import shutil
import sys
import tempfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable

import numpy as np

from tests.properties import fa2_target_capture_gpu as C
from tests.properties import fa_tap_exactness_gpu as G
from tests.properties import glm53_dsa_capture_gpu as DS
from verity_vllm.program.kernels import derived_rows as DR
from verity_vllm.program.kernels import dsa_norm_rows as NR
from verity_vllm.properties.record import seal as seal_record

SCHEMA = "verity-vllm/glm53-fused-rmsnorm-capture/v1"
RECORD = "glm53_fused_rmsnorm_capture.json"
EPS = DS.RMS_EPS
#: (output, its width, its weight, its index in `norm_rope`'s result)
NORMS = (("q_c", DS.Q_LORA, "q_rms_w", 0), ("kv_c", DS.KV_LORA, "kv_rms_w", 1))
SPECIAL = tuple(range(8, 19))
SAMPLE = (0, 1, 7, 19, 1000, 2047, 2048, 3071, 3072, 3135)
CHUNK = 512
PER_ALT, BATCH, POOL = 4, 256, 4096
#: the probe's weights: every bf16 word in [1, 2) (a power of two moves no product across a rounding boundary)
PROBE_W = np.arange(0x3F80, 0x4000, dtype=np.uint16)
TOTALS = ("rows", "words", "mismatch", "reference", "twin_mismatch")
CASES = ("served", "probe")
KERNEL = "_fused_norm_rope_kernel"
#: r20261013-001802-7e95's `_fused_norm_rope_kernel.a3d95dc37831.ptx` (Triton 3.7.1, vLLM d9105ea80), before its first `.file`
PTX_CODE_SHA256 = "2218fc75966f0924bcfac4e4d21f743cd33424d60dcc5c16e6f512a0137d5090"
PTX_COUNTS = {"div.full.f32": 4, "rsqrt.approx.ftz.f32": 3, "ld.global.v4.b32": 4, "ld.global.v2.b32": 2, "cvt.rn.bf16x2.f32": 10,
              "shfl.sync.bfly.b32": 35}
_PTX_OPS = re.compile(r"\b(" + "|".join(re.escape(k) for k in PTX_COUNTS) + r")(?=[\s;])")


# ------------------------------------------------------------------------------------------------------------------ the models
def bf16_row(rng: np.random.Generator, n: int, scale: float) -> np.ndarray:
    return ((rng.standard_normal(n).astype(np.float32) * np.float32(scale)).view(np.uint32) >> 16).astype(np.uint16)


def weight_row(rng: np.random.Generator, n: int) -> np.ndarray:
    """A norm weight as the checkpoints' and `operands`' are: 1 + 0.1 N(0, 1) in bf16."""
    return ((1 + 0.1 * rng.standard_normal(n)).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def special_rows(n: int, seed: int) -> np.ndarray:
    """One row per SPECIAL token: x 2^40; squares past f32 (2^70: the sum is inf, r = +0); f32-subnormal squares (2^-70); bf16
    subnormals; all zeros (r = rsqrt(eps)); one outlier over tiny values; +-0 / +-1 alternating; an inf; a NaN; the largest and
    the smallest normal bf16 magnitudes."""
    rng = np.random.default_rng(seed)
    out = [bf16_row(rng, n, s) for s in (2.0 ** 40, 2.0 ** 70, 2.0 ** -70, 2.0 ** -130)]
    out.append(np.zeros(n, dtype=np.uint16))
    tiny = bf16_row(rng, n, 2.0 ** -9)
    tiny[7] = 0x4580
    out.append(tiny)
    out.append(np.tile(np.array([0x0000, 0x8000, 0x3F80, 0xBF80], dtype=np.uint16), n // 4))
    inf, nan = bf16_row(rng, n, 1.0), bf16_row(rng, n, 1.0)
    inf[3], nan[5] = 0x7F80, 0x7FC1
    out += [inf, nan, np.full(n, 0x7F7F, dtype=np.uint16), np.full(n, 0x0080, dtype=np.uint16) | (rng.integers(0, 2, n) << 15).astype(np.uint16)]
    assert len(out) == len(SPECIAL)
    return np.stack(out)


def _chain(v: np.ndarray) -> np.ndarray:
    s = DR.fma_rn(v[..., 0], v[..., 0], DR.mul_rn(v[..., 1], v[..., 1]))
    for k in range(2, v.shape[-1]):
        s = DR.fma_rn(v[..., k], v[..., k], s)
    return s


def model_r(x: np.ndarray, eps: float, layout: str = "strided", chains: str = "one", cta: str = "02_13",
            epilogue: str = "rsqrt") -> np.ndarray:
    """Each row's f32 scale r under one choice of each step, `DsaRMSNorm_v1`'s at the defaults: thread t's registers `layout`
    ("strided" = `dsa_norm.layout`, "contiguous" = elements E t .. E t + E - 1), squared and summed as one FMA chain or one per
    8-element vector added in order (`chains` "one" / "per_vector"), each warp's butterfly, the CTA's two adds ("02_13" =
    (w0 + w2) + (w1 + w3), "01_23" = (w0 + w1) + (w2 + w3)), then `epilogue` "rsqrt" (RsqrtApprox(eps + DivFull(sum, N))) or
    "sqrt" (`RMSNormTriton_v1`'s DivFull(1, MufuSqrtFtz(fma(DivFullRcp(N), sum, eps))))."""
    from verity_vllm.program.registry import dsa_norm as DN
    from verity_vllm.program.registry.prims import bits_of

    R, N = x.shape
    E = N // DN.THREADS
    regs = DN.layout(N) if layout == "strided" else [list(range(E * t, E * t + E)) for t in range(DN.THREADS)]
    g = DR.bf16_to_f32_words(x)[:, np.asarray(regs)]
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        if chains == "one":
            lanes = _chain(g)
        else:
            parts = [_chain(g[..., i:i + 8]) for i in range(0, E, 8)]
            lanes = parts[0]
            for p in parts[1:]:
                lanes = DR.add_rn(lanes, p)
        v = lanes.reshape(R, DN.THREADS // DN.WARP, DN.WARP)
        while v.shape[-1] > 1:
            h = v.shape[-1] // 2
            v = DR.add_rn(v[..., :h], v[..., h:])
        w4 = v[..., 0]
        a, b, c, d = (0, 2, 1, 3) if cta == "02_13" else (0, 1, 2, 3)
        total = DR.add_rn(DR.add_rn(w4[:, a], w4[:, b]), DR.add_rn(w4[:, c], w4[:, d]))
        n = np.full(R, bits_of(np.float32(N)), dtype=np.uint32)
        e = np.full(R, bits_of(np.float32(eps)), dtype=np.uint32)
        if epilogue == "rsqrt":
            return DR.rsqrt_approx(DR.add_rn(e, DR.mul_rn(DR.div_full_scale_a(total, n), DR.div_full_rcp(n))))
        rms = DR.mufu_sqrt_ftz(DR.fma_rn(DR.div_full_rcp(n), total, e))
        return DR.mul_rn(DR.div_full_scale_a(np.full(R, 0x3F800000, dtype=np.uint32), rms), DR.div_full_rcp(rms))


def _r_ours(x: np.ndarray, eps: float) -> np.ndarray:
    return NR.dsa_rmsnorm_rows(x, np.zeros(x.shape[1], dtype=np.uint16), eps)["r"]


def _r_triton(x: np.ndarray, eps: float) -> np.ndarray:
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return DR.rmsnorm_triton_rows(x, np.zeros(x.shape[1], dtype=np.uint16), eps)["inv"]


def _r_f64(x: np.ndarray, eps: float) -> np.ndarray:
    """`glm53_dsa_capture_gpu._rms_info`'s model: the mean of squares in f64, then 1 / sqrt in f32."""
    xf = DR.as_f32(DR.bf16_to_f32_words(x)).astype(np.float64)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        return DR.as_u32(np.float32(1.0) / np.sqrt((np.mean(xf * xf, axis=1) + eps).astype(np.float32)))


Model = tuple[Callable[[np.ndarray, float], np.ndarray], str]


def alternatives(N: int) -> dict[str, Model]:
    """The models a capture equal to `DsaRMSNorm_v1` refutes, each (its r, its product order: "rx_w" = (r x) w, as the Definition,
    "x_rw" = x (r w)): vLLM's batch-invariant Triton RMSNorm, the f64 model the DSA capture reports, the Triton RMSNorm epilogue on
    this tree, the other pairing of the warps, the other product order; and where a thread holds more than one 8-element vector,
    contiguous elements a thread and one chain per vector."""
    out: dict[str, Model] = {"RMSNormTriton_v1": (_r_triton, "rx_w"), "f64_sum": (_r_f64, "rx_w"),
                             "sqrt_epilogue": (lambda x, e: model_r(x, e, epilogue="sqrt"), "rx_w"),
                             "warp_pairs": (lambda x, e: model_r(x, e, cta="01_23"), "rx_w"), "x_times_rw": (_r_ours, "x_rw")}
    if N // 128 > 8:
        out["contiguous"] = (lambda x, e: model_r(x, e, layout="contiguous"), "rx_w")
        out["per_vector_chains"] = (lambda x, e: model_r(x, e, chains="per_vector"), "rx_w")
    return out


def words(x: np.ndarray, w: np.ndarray, r: np.ndarray, product: str) -> np.ndarray:
    """bf16 outputs of rows x (R, N) at scales r (R,) and weights w (N,) or broadcastable, in ``product``'s order."""
    xf = DR.bf16_to_f32_words(x)
    wf = DR.bf16_to_f32_words(w)
    r = np.asarray(r, dtype=np.uint32)[..., None]
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        y = DR.mul_rn(DR.mul_rn(r, xf), wf) if product == "rx_w" else DR.mul_rn(xf, DR.mul_rn(r, wf))
    return DR.f2fp_bf16(y)


def probe(N: int, eps: float, seed: int) -> dict[str, Any]:
    """Rows and one weight row on which each alternative's words differ from `DsaRMSNorm_v1`'s in PER_ALT columns of its own.  Rows
    at served scales, drawn BATCH at a time (at most POOL), in order: a row whose r differs (any row, for a product order), then
    every free column with a weight in PROBE_W at which the two bf16 words differ, the first such weight taken.  The r of the two
    models are a few f32 ulps apart and x and w carry 8 significant bits, so about one such row in eight has a column at all.  A weight
    changes only its column's words and r only its row's, so the targets hold together; the other columns keep a served weight."""
    rng = np.random.default_rng(seed)
    w = weight_row(rng, N)
    pool: list[np.ndarray] = []
    ours: list[np.ndarray] = []
    free = np.ones(N, dtype=bool)
    rows: list[int] = []
    targets: list[dict[str, Any]] = []
    for name, (fn, product) in alternatives(N).items():
        found, b = 0, 0
        while found < PER_ALT and b * BATCH < POOL:
            if b == len(pool):
                pool.append(np.stack([bf16_row(rng, N, 2.0 ** rng.uniform(-3, 3)) for _ in range(BATCH)]))
                ours.append(_r_ours(pool[b], eps))
            r = fn(pool[b], eps)
            for i in range(BATCH):
                if found == PER_ALT:
                    break
                if product == "rx_w" and r[i] == ours[b][i]:
                    continue
                cols = np.flatnonzero(free)
                x = np.broadcast_to(pool[b][i, cols][:, None], (cols.size, PROBE_W.size))
                ya = words(x, PROBE_W, np.full(cols.size, ours[b][i]), "rx_w")
                yb = words(x, PROBE_W, np.full(cols.size, r[i]), product)
                for c in np.flatnonzero((ya != yb).any(axis=1))[:PER_ALT - found]:
                    k, j, g = int(np.flatnonzero(ya[c] != yb[c])[0]), int(cols[c]), b * BATCH + i
                    free[j], w[j] = False, PROBE_W[k]
                    if g not in rows:
                        rows.append(g)
                    targets.append({"alternative": name, "row": rows.index(g), "col": j, "w": int(PROBE_W[k]), "ours": int(ya[c, k]),
                                    "theirs": int(yb[c, k])})
                    found += 1
            b += 1
        if found < PER_ALT:
            raise RuntimeError(f"probe: {name} differs from DsaRMSNorm_v1 on {found} words of {POOL} rows at N={N}")
    return {"x": np.stack([pool[g // BATCH][g % BATCH] for g in rows]), "w": w, "targets": targets}


# ---------------------------------------------------------------------------------------------------------------------------- gpu
def _tensor(a: np.ndarray, like):
    """bf16 words as a bf16 tensor on ``like``'s device."""
    import torch

    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.uint16).view(np.int16)).view(torch.bfloat16).to(like.device)


def jobs_for(work: Path, case: str, saved: dict[str, dict[str, np.ndarray]], probe_rows: dict[str, int]) -> list[dict[str, Any]]:
    """One launch's comparison jobs: each norm's words (``saved[norm]``: x, w, out) saved under ``work``, its rows in chunks of
    CHUNK, each with the rows of SAMPLE, SPECIAL and the probe it holds."""
    jobs = []
    for norm, N, _, _ in NORMS:
        paths = DS._save(work / case / norm, **saved[norm])
        sample = sorted(set(SAMPLE) | set(SPECIAL) | set(range(probe_rows.get(norm, 0))))
        jobs += [{"case": case, "norm": norm, "N": N, "span": [r0, r1], "sample": [s for s in sample if r0 <= s < r1], "paths": paths}
                 for r0, r1 in DS._chunks(len(saved[norm]["x"]), CHUNK)]
    return jobs


def launch(work: Path, o: dict[str, Any], case: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """`fused_norm_rope` twice on fresh caches: the launch record (its checks) and the comparison jobs of its two norms."""
    import torch

    T = o["pos"].numel()
    c1, c2 = DS.caches(DS.NB, T), DS.caches(DS.NB, T)
    out1 = DS.norm_rope(o, c1, o["cs"], o["cs"])
    out2 = DS.norm_rope(o, c2, o["cs"], o["cs"])
    torch.cuda.synchronize()
    slots = o["slots"].cpu().numpy()
    written = np.flatnonzero(slots >= 0)
    mla = DS._u16(c1["mla"]).reshape(DS.NB * DS.BLOCK, DS.KV_LORA + DS.DR)
    kv_out = DS._u16(out1[1])
    rec = {"case": case, "tokens": int(T), "checks": {
        "deterministic": bool(all(torch.equal(a.view(torch.int16), b.view(torch.int16)) for a, b in zip(out1, out2))
                              and torch.equal(c1["mla"].view(torch.int16), c2["mla"].view(torch.int16))),
        "mla_kv_c": bool(written.size and np.array_equal(mla[slots[written], :DS.KV_LORA], kv_out[written]))}}
    saved = {norm: {"x": DS._u16(o[norm]), "w": DS._u16(o[wkey]), "out": DS._u16(out1[k])} for norm, _, wkey, k in NORMS}
    return rec, jobs_for(work, case, saved, o.get("probe_rows", {}))


def served_operands(seed: int) -> dict[str, Any]:
    """`glm53_dsa_capture_gpu.operands` with `special_rows` at the SPECIAL tokens of q_c and kv_c."""
    o = DS.operands(seed)
    for norm, N, _, _ in NORMS:
        o[norm][list(SPECIAL)] = _tensor(special_rows(N, seed + N), o["qkv"])
    return o


def probe_operands(o: dict[str, Any], probes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The served operands with each norm's probe rows over its first rows and its probe weights (qkv copied, so the layout and the
    strides are the served launch's)."""
    p = dict(o)
    p["qkv"] = o["qkv"].clone()
    p["q_c"] = p["qkv"][:, :DS.Q_LORA]
    p["kv_c"] = p["qkv"][:, DS.Q_LORA:DS.Q_LORA + DS.KV_LORA]
    p["k_pe"] = p["qkv"][:, DS.Q_LORA + DS.KV_LORA:]
    p["probe_rows"] = {}
    for norm, _, wkey, _ in NORMS:
        x = probes[norm]["x"]
        p[norm][:len(x)] = _tensor(x, p["qkv"])
        p[wkey] = _tensor(probes[norm]["w"], p["qkv"])
        p["probe_rows"][norm] = len(x)
    return p


def gpu(work: Path, doc: dict[str, Any]) -> list[dict[str, Any]]:
    """The served and the probe launches (doc["launches"], doc["probes"]); returns the comparison jobs."""
    import torch
    import triton
    from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config

    doc["triton"] = triton.__version__
    probes = {norm: probe(N, EPS, 20261013 + N) for norm, N, _, _ in NORMS}
    doc["probes"] = {norm: {"rows": len(p["x"]), "targets": p["targets"]} for norm, p in probes.items()}
    jobs: list[dict[str, Any]] = []
    with set_current_vllm_config(VllmConfig(compilation_config=CompilationConfig(custom_ops=["all"]))):
        o = served_operands(20261005)
        for case, ops in (("served", o), ("probe", probe_operands(o, probes))):
            rec, js = launch(work, ops, case)
            doc["launches"].append(rec)
            jobs += js
            print(f"[glm53-rmsnorm] launch {case:6s} tokens={rec['tokens']} {json.dumps(rec['checks'])}", flush=True)
        del o, ops
    torch.cuda.empty_cache()
    return jobs


def ptx_record(text: str) -> dict[str, Any]:
    code = text[:text.index("\t.file")] if "\t.file" in text else text
    return {"code_sha256": hashlib.sha256(code.encode()).hexdigest(), "counts": dict(Counter(_PTX_OPS.findall(code))),
            "reqntid": re.findall(r"\.reqntid\s+([\d, ]+)", code)}


def ptx(triton_cache: Path, outdir: Path) -> dict[str, Any]:
    """ok when the cache holds one `_fused_norm_rope_kernel` specialisation and its code is the pinned PTX's; copied to <out>/ptx/."""
    dest = outdir / "ptx"
    dest.mkdir(parents=True, exist_ok=True)
    entries = []
    for f in sorted(glob.glob(str(triton_cache / "**" / f"{KERNEL}*.ptx"), recursive=True)):
        text = Path(f).read_text(errors="replace")
        shutil.copy(f, dest / f"{KERNEL}.{hashlib.sha256(text.encode()).hexdigest()[:12]}.ptx")
        entries.append(ptx_record(text))
    ok = len(entries) == 1 and entries[0]["code_sha256"] == PTX_CODE_SHA256 and entries[0]["counts"] == PTX_COUNTS
    return {"specialisations": entries, "want": {"code_sha256": PTX_CODE_SHA256, "counts": PTX_COUNTS}, "ok": ok}


def bind_check() -> dict[str, Any]:
    from verity.ml.circuits.defs import bind
    from verity_vllm.program import boolean as SW
    from verity_vllm.program.registry import dsa_norm as DN

    out: dict[str, Any] = {}
    for _, N, _, _ in NORMS:
        try:
            fn = bind(DN.DsaRMSNorm, N=N, EPS=EPS)
            out[fn.id] = bool(fn.signature()) and SW.boolean_version(fn).id == f"DsaRMSNorm_v2{{N={N},EPS={EPS}}}"
        except Exception as e:  # noqa: BLE001
            out[f"DsaRMSNorm_v1{{N={N}}}"] = DS._error(e)
    return {"bindings": out, "ok": len(out) == len(NORMS) and all(v is True for v in out.values())}


# ------------------------------------------------------------------------------------------------------------------------ compare
def compare(job: dict[str, Any]) -> dict[str, Any]:
    """One chunk of one norm's rows: against the twin, the reference on its sampled rows, each alternative counted."""
    try:
        return _compare(job)
    except Exception as e:  # noqa: BLE001
        return {k: job[k] for k in ("case", "norm", "span")} | {"error": DS._error(e)}


def _compare(job: dict[str, Any]) -> dict[str, Any]:
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation import evaluate
    from verity_vllm.program.registry import dsa_norm as DN

    (r0, r1), N = job["span"], job["N"]
    a = DS._load(job["paths"])
    x, w, got = np.ascontiguousarray(a["x"][r0:r1]), np.ascontiguousarray(a["w"]), np.ascontiguousarray(a["out"][r0:r1])
    twin = NR.dsa_rmsnorm_rows(x, w, EPS)["out"]
    bad = np.argwhere(twin != got)
    rec: dict[str, Any] = {"rows": r1 - r0, "words": int(got.size), "mismatch": int(len(bad)), "reference": 0, "twin_mismatch": 0,
                           "first_mismatch": None}
    if len(bad):
        i, j = (int(v) for v in bad[0])
        rec["first_mismatch"] = {"row": r0 + i, "col": j, "want": int(twin[i, j]), "kernel": int(got[i, j])}
    fn = bind(DN.DsaRMSNorm, N=N, EPS=EPS)
    for s in job["sample"]:
        rec["reference"] += 1
        rec["twin_mismatch"] += [int(v) for v in evaluate(fn, x[s - r0].tolist(), w.tolist())] != twin[s - r0].tolist()
    rec["refuted"] = {name: int((words(x, w, r(x, EPS), product) != got).sum()) for name, (r, product) in alternatives(N).items()}
    return {k: job[k] for k in ("case", "norm", "span")} | rec


# ---------------------------------------------------------------------------------------------------------------------------- run
def summary(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    """Totals per (case, norm), and each norm's refuted words per alternative over both launches."""
    out: dict[str, Any] = {"cases": {}, "refuted": {}}
    for case in CASES:
        for norm, *_ in NORMS:
            rs = [r for r in comparisons if r["case"] == case and r["norm"] == norm]
            out["cases"][f"{case}/{norm}"] = {k: sum(r[k] for r in rs) for k in TOTALS} | {
                "first_mismatch": next((r["first_mismatch"] for r in rs if r["first_mismatch"]), None)}
    for norm, N, _, _ in NORMS:
        rs = [r for r in comparisons if r["norm"] == norm]
        out["refuted"][norm] = {name: sum(r["refuted"][name] for r in rs) for name in alternatives(N)}
    return out


def _case_ok(t: dict[str, Any]) -> bool:
    return t["rows"] > 0 and t["mismatch"] == 0 and t["reference"] > 0 and t["twin_mismatch"] == 0


def _verdict(doc: dict[str, Any]) -> bool:
    s = doc["summary"]
    return bool(doc["gate"]["ok"] and not doc["errors"] and doc["ptx"].get("ok") is True and doc["bind"].get("ok") is True
                and {x["case"] for x in doc["launches"]} == set(CASES)
                and all(x["checks"] and all(v is True for v in x["checks"].values()) for x in doc["launches"])
                and len(s["cases"]) == len(CASES) * len(NORMS) and all(_case_ok(t) for t in s["cases"].values())
                and all(n >= PER_ALT for alt in s["refuted"].values() for n in alt.values()))


def run(out: str, scratch: str | None, min_driver: int, want_cc: str, want_sms: int, workers: int) -> dict[str, Any]:
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.TemporaryDirectory(dir=scratch)
    work = Path(tmp.name)
    env = DS.environment(work)
    doc: dict[str, Any] = {"schema": SCHEMA, "argv": sys.argv, "env": env, "environment": G._environment(),
                           "gate": C.gate(min_driver, want_cc, want_sms), "errors": {}, "launches": []}
    print(f"[glm53-rmsnorm] gate {json.dumps(doc['gate'])}", flush=True)
    if not doc["gate"]["ok"]:
        (outdir / RECORD).write_text(json.dumps(seal_record(doc, False), indent=1) + "\n")
        print("GLM53-FUSED-RMSNORM GATE-FAIL", flush=True)
        raise SystemExit(3)
    jobs: list[dict[str, Any]] = []
    try:
        jobs = gpu(work, doc)
    except Exception as e:  # noqa: BLE001
        doc["errors"]["gpu"] = DS._error(e)
        print(f"[glm53-rmsnorm] gpu ERROR {doc['errors']['gpu']}", flush=True)
    for name, fn in (("ptx", lambda: ptx(work / "triton_cache", outdir)), ("bind", bind_check)):
        try:
            doc[name] = fn()
        except Exception as e:  # noqa: BLE001
            doc["errors"][name] = DS._error(e)
            doc[name] = {"ok": False}
    print(f"[glm53-rmsnorm] ptx {json.dumps(doc['ptx'])[:1500]}", flush=True)
    print(f"[glm53-rmsnorm] bind {json.dumps(doc['bind'])}; comparing {len(jobs)} jobs on {workers} workers", flush=True)
    doc["comparisons"] = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init) as ex:
        for res in ex.map(compare, jobs):
            if "error" in res:
                doc["errors"][f"compare/{res['case']}/{res['norm']}/{res['span'][0]}"] = res["error"]
                continue
            doc["comparisons"].append(res)
    doc["summary"] = summary(doc["comparisons"])
    for key, t in doc["summary"]["cases"].items():
        print(f"[glm53-rmsnorm] words {key:13s} rows={t['rows']} words={t['words']} !={t['mismatch']} ref={t['reference']} "
              f"twin!={t['twin_mismatch']} first={json.dumps(t['first_mismatch'])}", flush=True)
    doc = seal_record(doc, _verdict(doc))
    (outdir / RECORD).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    tmp.cleanup()
    totals = {k: sum(t[k] for t in doc["summary"]["cases"].values()) for k in TOTALS}
    ls = doc["launches"]
    print(f"GLM53-FUSED-RMSNORM {'GO' if doc['ok'] else 'NO-GO'} {doc['digest']} words={json.dumps(totals)} "
          f"refuted={json.dumps(doc['summary']['refuted'])} launches_ok={sum(all(x['checks'].values()) for x in ls)}/{len(CASES)} "
          f"ptx={doc['ptx'].get('ok')} bind={doc['bind'].get('ok')} errors={sorted(doc['errors'])}", flush=True)
    return doc


def _worker_init() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GLM-5.3's fused MLA RMSNorms (_fused_norm_rope_kernel) on this Hopper GPU against "
                                             "DsaRMSNorm_v1")
    ap.add_argument("--out", required=True, help=f"directory for {RECORD} and ptx/")
    ap.add_argument("--scratch", default=None, help="parent of the scratch directory (the words and the Triton cache)")
    ap.add_argument("--min-driver", type=int, default=575)
    ap.add_argument("--want-cc", default="9.0")
    ap.add_argument("--want-sms", type=int, default=132)
    ap.add_argument("--workers", type=int, default=C.cpu_workers())
    a = ap.parse_args(argv)
    doc = run(a.out, a.scratch, a.min_driver, a.want_cc, a.want_sms, a.workers)
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
