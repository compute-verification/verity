"""GLM-5.3's MLA cache write under `--kv-cache-dtype fp8` (the layout `fp8_ds_mla`, 656 bytes a token) on an H200: vLLM's own
`fused_norm_rope` at the pin, program 1 of `_fused_norm_rope_kernel`, against `dsa_ds_mla.DsaLatentRowDsMla_v1{KVL=512, R=64,
TILE=128, EPS=1e-05}` (the Definition read from the kernel's PTX, its module docstring), and its kv_c_out and k_pe_out against
`DsaRMSNorm_v1{N=512}` and `RoPEHeadGptJ_v2{D=64}` as registered.

Two phases, so that the card is held only while the kernel runs:

  gpu      (`--phase gpu`, under the card's lease) the gate (`fa2_target_capture_gpu.gate`; nothing else runs when it fails, exit
           3), then `fused_norm_rope` as `DeepseekV32Attention.forward` calls it (attention.py:338-366: kv_c_out and k_pe_out always
           passed, `mla_kv_cache_dtype="fp8_ds_mla"`, a uint8 cache [blocks, 64, 656] every byte 0xA5) on `glm53_dsa_capture_gpu`'s
           operands (SEQ + EXTRA tokens, rows at scales 2^U(-3, 3), the shared rotary cache at far positions too), in two cases:
             served  the served weights, with `glm53_fused_rmsnorm_capture_gpu.special_rows` at its SPECIAL tokens and `tile_rows`
                     after them;
             corner  the same tokens with `tile_rows` over the first CORNER_ROWS and the weight row `corner_weight`: 1.0 but in
                     tile 3, where columns PLANT hold 2^127 (an infinite y), NaN, -0, the least bf16 subnormal and -1.
           Each case runs with the indexer (`HAS_INDEXER`, the indexer layers' specialisation) and without (the shared layers'),
           twice each on fresh caches.  The kernel's inputs and outputs go to <out>/words/<case>/ (kv = [kv_c | k_pe] and the
           rotary row of each token, the weights, the slots; per mode kv_c_out, k_pe_out and the 656-byte row at each slotted
           token) and the Triton cache's PTX of the kernel to <out>/ptx/, with each specialisation's code digest and counts.
  compare  (`--phase compare`, on CPU after the lease) every token of both cases: the reference evaluator of the Definition, the
           cache row's 656 bytes (e4m3 values, four f32 scales, 64 bf16 RoPE words) and kv_c_out and k_pe_out against it in both
           modes, and kv_c_out and k_pe_out against DsaRMSNorm_v1 and RoPEHeadGptJ_v2; the corners each case reached, counted from
           the kernel's bytes (`corners`).

GO = the gate, no error, every launch deterministic, every byte of the cache outside the slotted rows still 0xA5, both modes'
words equal, the PTX of both FP8 specialisations the pinned code (PTX_CODE_SHA256), every row of both cases equal to the
Definition in all 656 bytes and in kv_c_out and k_pe_out, and to the two registered Definitions, and every corner of CORNERS
reached.  The record, <out>/glm53_ds_mla_write_capture.json, carries `ok` and the digest `properties.record.digest_of` takes.

    cd adapters/vllm; PY=/workspace/venv312/bin/python      # PYTHONPATH as pod_bootstrap.sh sets it
    $PY tests/properties/glm53_ds_mla_write_capture_gpu.py --phase gpu --out DIR --scratch /tmp
    $PY tests/properties/glm53_ds_mla_write_capture_gpu.py --phase compare --out DIR
"""
from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

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
from typing import Any

import numpy as np

from tests.properties import fa2_target_capture_gpu as C
from tests.properties import fa_tap_exactness_gpu as G
from tests.properties import glm53_dsa_capture_gpu as DS
from tests.properties import glm53_fused_rmsnorm_capture_gpu as FR
from verity_vllm.properties.record import digest_of

SCHEMA = "verity-vllm/glm53-ds-mla-write-capture/v1"
RECORD = "glm53_ds_mla_write_capture.json"
GPU_RECORD = "gpu.json"
KVL, R, TILE, EPS = DS.KV_LORA, DS.DR, 128, DS.RMS_EPS
NT = KVL // TILE
ROW = KVL + 4 * NT + 2 * R
SENTINEL = 0xA5
FLT_MIN = 0x00800000
CASES = ("served", "corner")
MODES = ("indexer", "shared")
CORNER_ROWS = 256
TILE_ROWS = 64
#: tile 3's planted weight columns in the corner case: (column, bf16 weight)
PLANT = ((500, 0x7F00), (501, 0x7FC0), (502, 0x8000), (503, 0x0001), (504, 0xBF80))
CHUNK = 128
KERNEL = "_fused_norm_rope_kernel"
#: the code (the text before the first `.file`) of the two FP8 specialisations the pin's serve compiled on an H200 under
#: `--kv-cache-dtype fp8`: the indexer layers' and the shared layers'
PTX_CODE_SHA256 = {"indexer": "f732e7c0d387b3edffb6a36a34ef1682c6fef273ab3d6d3292cdc748accf48cf",
                   "shared": "e2f8335c18c6db5dd329b4dd9d1cbbe71b75ba2ef55d13e37637f50ca4b60ef0"}
PTX_OPS = ("div.full.f32", "cvt.rn.satfinite.e4m3x2.f32", "abs.f32", "max.f32", "rsqrt.approx.ftz.f32", "cvt.rn.bf16x2.f32",
           "cvt.rn.bf16.f32", "shfl.sync.bfly.b32")
_PTX_OPS = re.compile(r"\b(" + "|".join(re.escape(k) for k in PTX_OPS) + r")(?=[\s;])")
#: what the two cases must reach between them, each counted over tiles or bytes of the kernel's rows (`corners`)
CORNERS = ("floor_empty", "floor_subnormal", "scale_inf", "nan_in_finite_tile", "nan_byte", "neg_zero_byte", "max_448")


# ---------------------------------------------------------------------------------------------------------------------- inputs
def tile_rows(rng: np.random.Generator, n: int) -> np.ndarray:
    """n rows of kv_c (bf16 words): tiles 0-2 each one of five forms in turn (zeros of random sign; a scale swept from 2^-121 to
    2^-113, whose amax * f32(1/448) crosses FLT_MIN; a served scale; one value of magnitude [1, 2) over the rest at 2^-15, which
    quantise to e4m3 subnormals and zeros; 16 exact -0), tile 3 values of scale about 1, which give the row its sum of squares.  No
    NaN or inf: in kv_c one makes r NaN or +0 for the whole row (`special_rows` has those); here they come from the weights."""
    out = np.empty((n, KVL), dtype=np.uint16)
    for i in range(n):
        for t in range(NT - 1):
            form, cols = (i + 3 * t) % 5, slice(TILE * t, TILE * t + TILE)
            if form == 0:
                v = (rng.integers(0, 2, TILE) << 15).astype(np.uint16)
            elif form == 1:
                v = FR.bf16_row(rng, TILE, 2.0 ** (-121 + 0.25 * ((i + t) % 32)))
            elif form == 2:
                v = FR.bf16_row(rng, TILE, 2.0 ** rng.uniform(-3, 3))
            elif form == 3:
                v = FR.bf16_row(rng, TILE, 2.0 ** -15)
                v[rng.integers(TILE)] = 0x3F80 | int(rng.integers(0, 128)) | int(rng.integers(0, 2)) << 15
            else:
                v = FR.bf16_row(rng, TILE, 1.0)
                v[rng.choice(TILE, 16, replace=False)] = 0x8000
            out[i, cols] = v
        out[i, TILE * (NT - 1):] = FR.bf16_row(rng, TILE, 2.0 ** rng.uniform(-1, 1))
        out[i, PLANT[0][0]] = (0x4080 | (rng.integers(0, 2) << 15)) if i % 3 == 0 else 0   # +-4: an infinite y under 2^127
    return out


def corner_weight() -> np.ndarray:
    w = np.full(KVL, 0x3F80, dtype=np.uint16)
    for col, word in PLANT:
        w[col] = word
    return w


# ------------------------------------------------------------------------------------------------------------------------- gpu
def operands(seed: int) -> dict[str, dict[str, Any]]:
    """The two cases' operands: `glm53_dsa_capture_gpu.operands` with the served planting, and a copy (qkv cloned, so the strides
    are the served launch's) with the corner planting."""
    o = DS.operands(seed)
    rng = np.random.default_rng(seed)
    o["kv_c"][list(FR.SPECIAL)] = FR._tensor(FR.special_rows(KVL, seed + KVL), o["qkv"])
    first = FR.SPECIAL[-1] + 1
    o["kv_c"][first:first + TILE_ROWS] = FR._tensor(tile_rows(rng, TILE_ROWS), o["qkv"])
    c = dict(o)
    c["qkv"] = o["qkv"].clone()
    c["q_c"] = c["qkv"][:, :DS.Q_LORA]
    c["kv_c"] = c["qkv"][:, DS.Q_LORA:DS.Q_LORA + KVL]
    c["k_pe"] = c["qkv"][:, DS.Q_LORA + KVL:]
    c["kv_c"][:CORNER_ROWS] = FR._tensor(tile_rows(rng, CORNER_ROWS), c["qkv"])
    c["kv_rms_w"] = FR._tensor(corner_weight(), c["qkv"])
    return {"served": o, "corner": c}


def launch(o: dict[str, Any], has_indexer: bool) -> dict[str, Any]:
    """One `fused_norm_rope` on fresh caches as attention.py:338-366 calls it under `fp8_ds_mla`: kv_c_out, k_pe_out and the MLA
    cache's bytes."""
    import torch
    from vllm.models.deepseek_v32.common.kernels import fused_norm_rope

    T = o["pos"].numel()
    mla = torch.full((DS.NB, DS.BLOCK, ROW), SENTINEL, dtype=torch.uint8, device="cuda")
    idx = torch.full((DS.NB, DS.BLOCK, DS.STRIDE), SENTINEL, dtype=torch.uint8, device="cuda") if has_indexer else None
    topk = torch.full((T, DS.TOPK), 7, dtype=torch.int32, device="cuda")
    kv_c_out, k_pe_out = torch.empty_like(o["kv_c"]), torch.empty_like(o["k_pe"])
    fused_norm_rope(o["pos"], o["q_c"], o["q_rms_w"], EPS, o["kv_c"], o["kv_rms_w"], EPS, o["k_pe"], o["cs"], o["index_k"], o["k_w"],
                    o["k_b"], DS.K_EPS, o["cs"], topk, slot_mapping=o["slots"], indexer_k_cache=idx, mla_kv_cache=mla,
                    mla_kv_cache_dtype="fp8_ds_mla", mla_k_scale=o["k_scale"], has_indexer=has_indexer, index_rope_interleave=True,
                    kv_c_out=kv_c_out, k_pe_out=k_pe_out, index_k_out=None)
    torch.cuda.synchronize()
    return {"kv_out": DS._u16(kv_c_out), "kpe_out": DS._u16(k_pe_out), "cache": DS._u8(mla).reshape(DS.NB * DS.BLOCK, ROW)}


def case_words(outdir: Path, case: str, o: dict[str, Any]) -> dict[str, Any]:
    """Both modes, twice each: the case's record (its checks) with its words saved under <out>/words/<case>/."""
    slots = o["slots"].cpu().numpy()
    written = np.flatnonzero(slots >= 0)
    rest = np.ones(DS.NB * DS.BLOCK, dtype=bool)
    rest[slots[written]] = False
    arrays: dict[str, np.ndarray] = {"kv": np.concatenate([DS._u16(o["kv_c"]), DS._u16(o["k_pe"])], axis=1),
                                     "w": DS._u16(o["kv_rms_w"]), "cs": DS._u16(o["cs"][o["pos"]]), "slots": slots}
    checks: dict[str, bool] = {}
    for mode in MODES:
        a, b = launch(o, mode == "indexer"), launch(o, mode == "indexer")
        checks[f"{mode}/deterministic"] = all(np.array_equal(a[k], b[k]) for k in a)
        checks[f"{mode}/untouched_sentinel"] = bool((a["cache"][rest] == SENTINEL).all())
        rows = np.zeros((len(slots), ROW), dtype=np.uint8)
        rows[written] = a["cache"][slots[written]]
        arrays |= {f"{mode}_kv_out": a["kv_out"], f"{mode}_kpe_out": a["kpe_out"], f"{mode}_rows": rows}
    checks["modes_equal"] = all(np.array_equal(arrays[f"indexer_{k}"], arrays[f"shared_{k}"]) for k in ("kv_out", "kpe_out", "rows"))
    paths = DS._save(outdir / "words" / case, **arrays)
    return {"case": case, "tokens": len(slots), "slotted": int(written.size), "checks": checks,
            "paths": {k: os.path.relpath(p, outdir) for k, p in paths.items()}}


def ptx_record(text: str) -> dict[str, Any]:
    code = text[:text.index("\t.file")] if "\t.file" in text else text
    return {"code_sha256": hashlib.sha256(code.encode()).hexdigest(), "counts": dict(Counter(_PTX_OPS.findall(code))),
            "reqntid": re.findall(r"\.reqntid\s+([\d, ]+)", code)}


def ptx(triton_cache: Path, outdir: Path) -> dict[str, Any]:
    """Every `_fused_norm_rope_kernel` specialisation in the Triton cache, copied to <out>/ptx/ under its cache directory's name;
    ok when the FP8 ones are the two PTX_CODE_SHA256 names."""
    dest = outdir / "ptx"
    dest.mkdir(parents=True, exist_ok=True)
    entries = []
    for f in sorted(glob.glob(str(triton_cache / "**" / f"{KERNEL}*.ptx"), recursive=True)):
        name = f"{Path(f).parent.name}-{Path(f).name}"
        shutil.copy(f, dest / name)
        entries.append({"file": name} | ptx_record(Path(f).read_text(errors="replace")))
    fp8 = {e["code_sha256"] for e in entries if e["counts"].get("cvt.rn.satfinite.e4m3x2.f32")}
    return {"specialisations": entries, "want": PTX_CODE_SHA256, "ok": fp8 == set(PTX_CODE_SHA256.values())}


def gpu_phase(outdir: Path, scratch: str | None, min_driver: int, want_cc: str, want_sms: int) -> int:
    tmp = tempfile.TemporaryDirectory(dir=scratch)
    work = Path(tmp.name)
    env, cleared = DS.environment(work)
    doc: dict[str, Any] = {"schema": SCHEMA, "argv": sys.argv, "env": env, "env_cleared": cleared, "served": DS.served(),
                           "environment": G._environment(), "gate": C.gate(min_driver, want_cc, want_sms), "errors": {},
                           "cases": []}
    print(f"[ds-mla-write] gate {json.dumps(doc['gate'])} served {json.dumps(doc['served'])}", flush=True)
    if doc["gate"]["ok"]:
        import triton
        import vllm
        from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config

        doc["versions"] = {"vllm": vllm.__version__, "triton": triton.__version__}
        with set_current_vllm_config(VllmConfig(compilation_config=CompilationConfig(custom_ops=["all"]))):
            for case, o in operands(20261112).items():
                rec = case_words(outdir, case, o)
                doc["cases"].append(rec)
                print(f"[ds-mla-write] {case} tokens={rec['tokens']} slotted={rec['slotted']} {json.dumps(rec['checks'])}",
                      flush=True)
        try:
            doc["ptx"] = ptx(work / "triton_cache", outdir)
        except OSError as e:
            doc["errors"]["ptx"] = DS._error(e)
            doc["ptx"] = {"ok": False}
        print(f"[ds-mla-write] ptx ok={doc['ptx'].get('ok')} "
              f"{json.dumps([(e['file'], e['code_sha256'][:16]) for e in doc['ptx'].get('specialisations', [])])}", flush=True)
    (outdir / GPU_RECORD).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    tmp.cleanup()
    if not doc["gate"]["ok"]:
        print("GLM53-DS-MLA-WRITE GATE-FAIL", flush=True)
        return 3
    return 0 if not doc["errors"] else 1


# --------------------------------------------------------------------------------------------------------------------- compare
def want_row(ref: list[int]) -> np.ndarray:
    """The 656 bytes the Definition's outputs give: nope, the scales little-endian, latent[KVL:] little-endian."""
    nope = np.asarray(ref[KVL + R:KVL + R + KVL], dtype=np.uint8)
    scale = np.asarray(ref[KVL + R + KVL:], dtype=np.uint32).view(np.uint8)
    rope = np.asarray(ref[KVL:KVL + R], dtype=np.uint16).view(np.uint8)
    return np.concatenate([nope, scale, rope])


def corners(rows: np.ndarray) -> dict[str, int]:
    """Counted from cache rows (n, 656): tiles whose scale is FLT_MIN with every byte a zero or a NaN (`floor_empty`) or with a
    finite nonzero one (`floor_subnormal`: amax * f32(1/448) at most FLT_MIN); tiles with scale +inf; tiles with a finite scale
    above FLT_MIN and a NaN byte (the max dropped it); NaN bytes; -0 bytes; tiles holding +-448."""
    q = rows[:, :KVL].reshape(-1, NT, TILE)
    s = np.ascontiguousarray(rows[:, KVL:KVL + 4 * NT]).view(np.uint32).reshape(-1, NT)
    mag = q & 0x7F
    empty = ((mag == 0) | (mag == 0x7F)).all(axis=2)
    nan = (mag == 0x7F).any(axis=2)
    finite = (s & 0x7F800000) != 0x7F800000
    return {"floor_empty": int(((s == FLT_MIN) & empty).sum()), "floor_subnormal": int(((s == FLT_MIN) & ~empty).sum()),
            "scale_inf": int((s == 0x7F800000).sum()), "nan_in_finite_tile": int((finite & (s != FLT_MIN) & nan).sum()),
            "nan_byte": int((mag == 0x7F).sum()), "neg_zero_byte": int((q == 0x80).sum()), "max_448": int((mag == 0x7E).any(axis=2).sum())}


def _compare(job: dict[str, Any]) -> dict[str, Any]:
    from verity.circuits.defs import bind
    from verity.circuits.evaluation import evaluate
    from verity.definitions.registry import dsa_ds_mla as M, dsa_norm as DN
    from verity.definitions.registry.boolean_rope import RoPEHeadGptJV2

    row_fn = bind(M.DsaLatentRowDsMla, KVL=KVL, R=R, TILE=TILE, EPS=EPS)
    norm_fn, rope_fn = bind(DN.DsaRMSNorm, N=KVL, EPS=EPS), bind(RoPEHeadGptJV2, D=R)
    a = DS._load({k: str(Path(job["out"]) / p) for k, p in job["paths"].items()})
    r0, r1 = job["span"]
    w = a["w"].tolist()
    keys = ("rows", "row_bytes", "kv_out", "kpe_out", "norm", "rope")
    rec: dict[str, Any] = {"tokens": r1 - r0, "slotted": 0, "differ": {f"{m}/{k}": 0 for m in MODES for k in keys},
                           "differ_bytes": {m: 0 for m in MODES}, "first": {}}
    for t in range(r0, r1):
        kv, cs = a["kv"][t].tolist(), a["cs"][t].tolist()
        ref = [int(v) for v in evaluate(row_fn, kv, w, cs)]
        norm = [int(v) for v in evaluate(norm_fn, kv[:KVL], w)]
        rope = [int(v) for v in evaluate(rope_fn, kv[KVL:], cs)]
        slotted = int(a["slots"][t]) >= 0
        rec["slotted"] += slotted
        want = want_row(ref)
        for m in MODES:
            kv_out, kpe_out = a[f"{m}_kv_out"][t].tolist(), a[f"{m}_kpe_out"][t].tolist()
            bad = {"kv_out": kv_out != ref[:KVL], "kpe_out": kpe_out != ref[KVL:KVL + R], "norm": kv_out != norm,
                   "rope": kpe_out != rope, "row_bytes": False}
            if slotted:
                nb = int((a[f"{m}_rows"][t] != want).sum())
                rec["differ_bytes"][m] += nb
                bad["row_bytes"] = nb > 0
            bad["rows"] = any(bad.values())
            for k, v in bad.items():
                rec["differ"][f"{m}/{k}"] += int(v)
                if v and f"{m}/{k}" not in rec["first"]:
                    rec["first"][f"{m}/{k}"] = t
    slotted_rows = np.asarray(a["indexer_rows"][r0:r1])[np.asarray(a["slots"][r0:r1]) >= 0]
    rec["corners"] = corners(slotted_rows)
    return {k: job[k] for k in ("case", "span")} | rec


def summary(doc: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for case in CASES:
        rs = [r for r in doc["comparisons"] if r["case"] == case]
        s: dict[str, Any] = {"tokens": sum(r["tokens"] for r in rs), "slotted": sum(r["slotted"] for r in rs)}
        for k in ("differ", "differ_bytes", "corners"):
            s[k] = dict(sum((Counter(r[k]) for r in rs), Counter()))
            s[k] |= {name: 0 for name in (rs[0][k] if rs else {}) if name not in s[k]}
        s["first"] = {k: min(r["first"][k] for r in rs if k in r["first"]) for k in {k for r in rs for k in r["first"]}}
        out[case] = s
    return out


def _verdict(doc: dict[str, Any], gpu: dict[str, Any]) -> bool:
    s = doc["summary"]
    reached = Counter()
    for case in CASES:
        reached.update(s.get(case, {}).get("corners", {}))
    return bool(gpu["gate"]["ok"] and not gpu["errors"] and not doc["errors"] and gpu["ptx"].get("ok") is True
                and {c["case"] for c in gpu["cases"]} == set(CASES)
                and all(c["checks"] and all(v is True for v in c["checks"].values()) for c in gpu["cases"])
                and all(s[case]["tokens"] == next(c["tokens"] for c in gpu["cases"] if c["case"] == case) for case in CASES)
                and all(v == 0 for case in CASES for v in s[case]["differ"].values())
                and all(reached[k] > 0 for k in CORNERS))


def compare_phase(outdir: Path, workers: int) -> int:
    gpu = json.loads((outdir / GPU_RECORD).read_text())
    doc: dict[str, Any] = {"schema": SCHEMA, "gpu": gpu, "errors": {}, "comparisons": []}
    jobs = [{"case": c["case"], "span": [r0, r1], "paths": c["paths"], "out": str(outdir)}
            for c in gpu["cases"] for r0, r1 in DS._chunks(c["tokens"], CHUNK)]
    print(f"[ds-mla-write] comparing {len(jobs)} jobs on {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init) as ex:
        for res in ex.map(_compare, jobs):
            doc["comparisons"].append(res)
    doc["summary"] = summary(doc)
    for case, s in doc["summary"].items():
        rows = {m: s["differ"].get(f"{m}/rows", 0) for m in MODES}
        print(f"[ds-mla-write] {case}: tokens={s['tokens']} slotted={s['slotted']} rows_differ={json.dumps(rows)} "
              f"differ={json.dumps(s['differ'])} bytes_differ={json.dumps(s['differ_bytes'])} first={json.dumps(s['first'])} "
              f"corners={json.dumps(s['corners'])}", flush=True)
    doc["ok"] = _verdict(doc, gpu)
    doc["digest"] = digest_of(doc)
    (outdir / RECORD).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    tokens = sum(s["tokens"] for s in doc["summary"].values())
    differ = sum(s["differ"].get(f"{m}/rows", 0) for s in doc["summary"].values() for m in MODES)
    print(f"GLM53-DS-MLA-WRITE {'GO' if doc['ok'] else 'NO-GO'} {doc['digest']} tokens={tokens} (each in {len(MODES)} modes) "
          f"rows_differ={differ} ptx={gpu['ptx'].get('ok')} errors={sorted(doc['errors']) + sorted(gpu['errors'])}", flush=True)
    return 0 if doc["ok"] else 1


def _worker_init() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GLM-5.3's fp8_ds_mla cache write (_fused_norm_rope_kernel program 1) on this Hopper "
                                             "GPU against DsaLatentRowDsMla_v1")
    ap.add_argument("--phase", choices=("gpu", "compare"), required=True)
    ap.add_argument("--out", required=True, help=f"directory for {GPU_RECORD}, words/, ptx/ and {RECORD}")
    ap.add_argument("--scratch", default=None, help="parent of the scratch directory (the Triton cache)")
    ap.add_argument("--min-driver", type=int, default=575)
    ap.add_argument("--want-cc", default="9.0")
    ap.add_argument("--want-sms", type=int, default=132)
    ap.add_argument("--workers", type=int, default=C.cpu_workers())
    a = ap.parse_args(argv)
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    if a.phase == "gpu":
        return gpu_phase(outdir, a.scratch, a.min_driver, a.want_cc, a.want_sms)
    return compare_phase(outdir, a.workers)


if __name__ == "__main__":
    sys.exit(main())
