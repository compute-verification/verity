"""GLM-5.3's MTP draft layer's input norms on an H200: vLLM d9105ea80's `fused_eh_norm` (`_fused_eh_norm_kernel`: one program a token,
the embedding row +0 at position 0, `enorm` of it and `hnorm` of the previous hidden state side by side) against
`mtp_draft.MtpEhNorm_v1{H=6144, BLOCK=8192, WARPS=4, EPS=1e-05}`, each half `dsa_norm.TritonBlockRMSNorm_v1` at those statics, and
against the models it was chosen over (`alternatives`): the reduction orders a tile of 8192 over 6144 values could have, and the
position tests the select could make.

One GPU process writes `<out>/glm53_mtp_eh_norm_capture.json`, the property record `glm53_mtp_eh_norm_capture`
(`properties.record`), the kernel's PTX and SASS under `<out>/ptx/`, and each launch's inputs and output under `<out>/words/`.  The
device gate runs first and nothing else runs when it fails (exit 3).  vLLM runs as in `glm53_dsa_capture_gpu`, at its defaults (its
`environment`: `served_defaults`, a served engine's toolchain, the Triton cache in the scratch directory), its switches read back by
that capture's `served`.

  gate     `fa2_target_capture_gpu.gate`: driver, compute capability 9.0, 132 SMs.
  served   `fused_eh_norm(positions, inputs_embeds, previous_hidden, enorm_w, hnorm_w, eps)` as `mtp.py` calls it, on T tokens: e rows
           at 2^U(-6, 0), p rows at 2^U(-3, 3), weights 1 + 0.1 N(0, 1), positions 1..T but 0 at ZERO_POS and 2^32 (low word 0) at
           HIGH_POS, `glm53_fused_rmsnorm_capture_gpu.special_rows` at tokens SPECIAL of e and p; twice, the words equal.
  probe    the same launch with the first rows of e and p and both weights the probe's (`probe`): rows where an alternative's r
           differs and, in columns of their own, weights in [1, 2) at which the bf16 words then differ, PER_ALT words for each.
  words    every output word of both launches against the twin (`dsa_norm_rows.block_rmsnorm_rows` of e, zeroed where pos is 0, and
           of p); the reference evaluator of `MtpEhNorm_v1` on the SPECIAL, ZERO_POS, HIGH_POS, probe and SAMPLE rows; each
           alternative's words against the kernel's, counted (`refuted`).
  ptx      the Triton cache holds one specialisation of `_fused_eh_norm_kernel`; its code (the text before the first `.file`, its
           `.loc` lines and their labels removed: they name the source's lines) is the PTX `MtpEhNorm_v1` was read from
           (PTX_CODE_SHA256, PTX_COUNTS); its SASS takes rsqrt of `FFMA sum, rcp(6144), eps` (SASS_COUNTS).
  bind     MtpEhNorm_v1, TritonBlockRMSNorm_v1 and the units of MtpEhNorm_v1's cut (`mtp_draft.eh_cut`) bind at the served statics.

GO = the gate, the switches, both launches deterministic, every word of every row equal to the twin, the reference equal to the
twin on every sampled row, every alternative refuted on at least PER_ALT words, the PTX, the SASS and the bindings. A phase that
raises ends the script with its traceback and writes no record.

    cd adapters/vllm; PY=/workspace/venv312/bin/python     # PYTHONPATH as pod_bootstrap.sh sets it
    $PY tests/properties/glm53_mtp_eh_norm_capture_gpu.py --out DIR --scratch /tmp --want-cc 9.0 --want-sms 132 --min-driver 575
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
import subprocess
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
from tests.properties import glm53_fused_rmsnorm_capture_gpu as FR
from verity.definitions.numerics import derived_rows as DR
from verity.definitions.numerics import dsa_norm_rows as NR
from verity_vllm.properties.record import seal as seal_record

SCHEMA = "verity-vllm/glm53-mtp-eh-norm-capture/v1"
RECORD = "glm53_mtp_eh_norm_capture.json"
#: GLM-5.3's hidden size, `next_power_of_2` of it, Triton's default warps, and rms_norm_eps
H, BLOCK, WARPS, EPS = 6144, 8192, 4, 1e-5
T = 1024
SPECIAL = tuple(range(8, 19))
ZERO_POS = (0, 3, 300, T - 1)
HIGH_POS = 5
SAMPLE = (1, 2, 7, 19, 511, 512, T - 2)
CHUNK = 128
PER_ALT, BATCH, POOL = 4, 256, 4096
HALVES = ("e", "p")
CASES = ("served", "probe")
TOTALS = ("rows", "words", "mismatch", "rows_differ", "reference", "twin_mismatch")
KERNEL = "_fused_eh_norm_kernel"
#: Triton 3.7.1's sm_90 PTX of `_fused_eh_norm_kernel` at H 6144, BLOCK 8192, 4 warps, every pointer and stride divisible by 16: its
#: code before the first `.file` without `.loc` lines and `$L__tmp` labels (`ptx_record`)
PTX_CODE_SHA256 = "88978f9f23bf0bf743216e79cbc564e294a845fda819eeccf30c8ca355408b7d"
PTX_COUNTS = {"ld.global.v4.b32": 32, "ld.global.b64": 1, "fma.rn.f32": 126, "mul.f32": 258, "add.f32": 16, "div.full.f32": 2,
              "rsqrt.approx.ftz.f32": 2, "shfl.sync.bfly.b32": 14, "cvt.rn.bf16x2.f32": 64, "st.global.v4.b32": 16, "selp.f32": 64,
              "setp.eq.b64": 1}
_PTX_OPS = re.compile(r"\b(" + "|".join(re.escape(k) for k in PTX_COUNTS) + r")(?=[\s;])")
#: the SASS of that PTX: rsqrt of `FFMA sum, 0x392AAAAB (= DivFullRcp(6144)), eps` in each norm, no MUFU.RCP
SASS_COUNTS = {"ffma_rcp_eps": 2, "MUFU.RSQ": 2, "MUFU.RCP": 0}
_SASS_FFMA = re.compile(r"FFMA R\d+, R\d+, 0\.00016276042151730507612, R\d+ ;")


# ------------------------------------------------------------------------------------------------------------------ the models
def zeroed(e: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """e with every word of a position-0 row +0 (`tl.where(pos == 0, 0.0, e)` on the whole int64)."""
    out = np.array(e, dtype=np.uint16)
    out[np.asarray(pos, dtype=np.int64) == 0] = 0
    return out


def _chain(v: np.ndarray) -> np.ndarray:
    s = DR.fma_rn(v[..., 0], v[..., 0], DR.mul_rn(v[..., 1], v[..., 1]))
    for k in range(2, v.shape[-1]):
        s = DR.fma_rn(v[..., k], v[..., k], s)
    return s


def _tree(v: np.ndarray) -> np.ndarray:
    while v.shape[-1] > 1:
        h = v.shape[-1] // 2
        v = DR.add_rn(v[..., :h], v[..., h:])
    return v[..., 0]


def model_r(x: np.ndarray, eps: float, regs: list[list[int]], warps: int = WARPS, chains: str = "one") -> np.ndarray:
    """Each row's f32 scale r with thread t's registers ``regs[t]`` (an element past N +0), squared and summed as one FMA chain or one
    per 8-element vector added in order (``chains`` "one" / "per_vector"), each warp's butterfly, the warp sums' halving tree, and
    rsqrt of the contracted `fma(scale_a(sum), rcp(N), eps)`."""
    from verity.definitions.registry.prims import bits_of

    R, N = x.shape
    g = np.pad(DR.bf16_to_f32_words(x), ((0, 0), (0, max(max(r) for r in regs) + 1 - N)))[:, np.asarray(regs)]
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        if chains == "one":
            lanes = _chain(g)
        else:
            parts = [_chain(g[..., i:i + 8]) for i in range(0, g.shape[-1], 8)]
            lanes = parts[0]
            for p in parts[1:]:
                lanes = DR.add_rn(lanes, p)
        total = _tree(_tree(lanes.reshape(R, warps, 32)))
        n = np.full(R, bits_of(np.float32(N)), dtype=np.uint32)
        e = np.full(R, bits_of(np.float32(eps)), dtype=np.uint32)
        return DR.rsqrt_approx(DR.fma_rn(DR.div_full_scale_a(total, n), DR.div_full_rcp(n), e))


def _r_ours(x: np.ndarray, eps: float) -> np.ndarray:
    return NR.block_rmsnorm_rows(x, np.zeros(x.shape[1], dtype=np.uint16), eps, BLOCK, WARPS)["r"]


def _r_unfused(x: np.ndarray, eps: float) -> np.ndarray:
    return NR.block_rmsnorm_rows(x, np.zeros(x.shape[1], dtype=np.uint16), eps, BLOCK, WARPS, fused_eps=False)["r"]


def _r_warps8(x: np.ndarray, eps: float) -> np.ndarray:
    return NR.block_rmsnorm_rows(x, np.zeros(x.shape[1], dtype=np.uint16), eps, BLOCK, 8)["r"]


def _r_triton(x: np.ndarray, eps: float) -> np.ndarray:
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        return DR.rmsnorm_triton_rows(x, np.zeros(x.shape[1], dtype=np.uint16), eps)["inv"]


CONTIGUOUS = [list(range(64 * t, 64 * t + 64)) for t in range(32 * WARPS)]
Model = tuple[Callable[[np.ndarray, float], np.ndarray], str]


def alternatives() -> dict[str, Model]:
    """The models a capture equal to `TritonBlockRMSNorm_v1{6144, 8192, 4}` refutes, each (its r, its product order: "rx_w" =
    (r x) w, as the Definition, "x_rw" = x (r w)): `rsqrt(eps + div.full(sum, N))` as the PTX writes it, before ptxas contracts it;
    the block on 8 warps; 64 contiguous elements a thread; one chain per 8-element vector; vLLM's batch-invariant Triton RMSNorm;
    and the other product order."""
    return {"unfused_eps": (_r_unfused, "rx_w"), "warps8": (_r_warps8, "rx_w"),
            "contiguous": (lambda x, e: model_r(x, e, CONTIGUOUS), "rx_w"),
            "per_vector_chains": (lambda x, e: model_r(x, e, NR_LAYOUT, chains="per_vector"), "rx_w"),
            "RMSNormTriton_v1": (_r_triton, "rx_w"), "x_times_rw": (_r_ours, "x_rw")}


def _layout() -> list[list[int]]:
    from verity.definitions.registry import dsa_norm as DN

    return DN.layout(BLOCK, 32 * WARPS)


NR_LAYOUT = _layout()


def probe(eps: float, seed: int) -> dict[str, Any]:
    """`glm53_fused_rmsnorm_capture_gpu.probe` over this block's alternatives: rows at served scales, BATCH at a time (at most POOL),
    a row whose r differs (any row, for a product order), then every free column with a weight in PROBE_W at which the two bf16
    words differ, the first such weight taken, PER_ALT columns each."""
    rng = np.random.default_rng(seed)
    w = FR.weight_row(rng, H)
    pool: list[np.ndarray] = []
    ours: list[np.ndarray] = []
    free = np.ones(H, dtype=bool)
    rows: list[int] = []
    targets: list[dict[str, Any]] = []
    for name, (fn, product) in alternatives().items():
        found, b = 0, 0
        while found < PER_ALT and b * BATCH < POOL:
            if b == len(pool):
                pool.append(np.stack([FR.bf16_row(rng, H, 2.0 ** rng.uniform(-3, 3)) for _ in range(BATCH)]))
                ours.append(_r_ours(pool[b], eps))
            r = fn(pool[b], eps)
            for i in range(BATCH):
                if found == PER_ALT:
                    break
                if product == "rx_w" and r[i] == ours[b][i]:
                    continue
                cols = np.flatnonzero(free)
                x = np.broadcast_to(pool[b][i, cols][:, None], (cols.size, FR.PROBE_W.size))
                ya = FR.words(x, FR.PROBE_W, np.full(cols.size, ours[b][i]), "rx_w")
                yb = FR.words(x, FR.PROBE_W, np.full(cols.size, r[i]), product)
                for c in np.flatnonzero((ya != yb).any(axis=1))[:PER_ALT - found]:
                    k, j, g = int(np.flatnonzero(ya[c] != yb[c])[0]), int(cols[c]), b * BATCH + i
                    free[j], w[j] = False, FR.PROBE_W[k]
                    if g not in rows:
                        rows.append(g)
                    targets.append({"alternative": name, "row": rows.index(g), "col": j, "w": int(FR.PROBE_W[k]),
                                    "ours": int(ya[c, k]), "theirs": int(yb[c, k])})
                    found += 1
            b += 1
        if found < PER_ALT:
            raise RuntimeError(f"probe: {name} differs from TritonBlockRMSNorm_v1 on {found} words of {POOL} rows")
    return {"x": np.stack([pool[g // BATCH][g % BATCH] for g in rows]), "w": w, "targets": targets}


# ---------------------------------------------------------------------------------------------------------------------------- gpu
def positions() -> np.ndarray:
    pos = np.arange(1, T + 1, dtype=np.int64)
    pos[list(ZERO_POS)] = 0
    pos[HIGH_POS] = 1 << 32
    return pos


def served_operands(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    e = np.stack([FR.bf16_row(rng, H, 2.0 ** rng.uniform(-6, 0)) for _ in range(T)])
    p = np.stack([FR.bf16_row(rng, H, 2.0 ** rng.uniform(-3, 3)) for _ in range(T)])
    e[list(SPECIAL)] = FR.special_rows(H, seed + 1)
    p[list(SPECIAL)] = FR.special_rows(H, seed + 2)
    return {"pos": positions(), "e": e, "p": p, "ew": FR.weight_row(rng, H), "hw": FR.weight_row(rng, H)}


def probe_operands(o: dict[str, np.ndarray], probes: dict[str, dict[str, Any]]) -> dict[str, np.ndarray]:
    """The served operands with each half's probe rows over rows from 20 on (past SPECIAL, at nonzero positions) and its weights."""
    q = {k: np.array(v) for k, v in o.items()}
    for half, wkey in (("e", "ew"), ("p", "hw")):
        x = probes[half]["x"]
        q[half][PROBE0:PROBE0 + len(x)] = x
        q[wkey] = probes[half]["w"]
    return q


PROBE0 = 20


def launch(outdir: Path, o: dict[str, np.ndarray], case: str) -> tuple[dict[str, Any], dict[str, str]]:
    """`fused_eh_norm` twice: the launch record and the saved words (inputs and output)."""
    import torch
    from vllm.models.deepseek_v32.common.kernels import fused_eh_norm

    def bf16(a):
        return torch.from_numpy(np.ascontiguousarray(a, dtype=np.uint16).view(np.int16)).view(torch.bfloat16).cuda()

    pos = torch.from_numpy(o["pos"]).cuda()
    e, p, ew, hw = bf16(o["e"]), bf16(o["p"]), bf16(o["ew"]), bf16(o["hw"])
    outs = [fused_eh_norm(pos, e, p, ew, hw, EPS) for _ in range(2)]
    torch.cuda.synchronize()
    rec = {"case": case, "tokens": T, "shape": list(outs[0].shape), "dtype": str(outs[0].dtype), "strides": {
        "e": list(e.stride()), "p": list(p.stride()), "out": list(outs[0].stride())}, "checks": {
        "deterministic": bool(torch.equal(outs[0].view(torch.int16), outs[1].view(torch.int16))),
        "shape": list(outs[0].shape) == [T, 2 * H] and outs[0].dtype == torch.bfloat16}}
    paths = DS._save(outdir / "words" / case, pos=o["pos"], e=o["e"], p=o["p"], ew=o["ew"], hw=o["hw"], out=DS._u16(outs[0]))
    return rec, paths


def gpu(outdir: Path, doc: dict[str, Any]) -> list[dict[str, Any]]:
    """The served and the probe launches; returns the comparison jobs."""
    import torch
    import triton

    doc["triton"] = triton.__version__
    probes = {half: probe(EPS, 20261112 + i) for i, half in enumerate(HALVES)}
    doc["probes"] = {half: {"rows": len(p["x"]), "targets": p["targets"]} for half, p in probes.items()}
    o = served_operands(20261113)
    jobs: list[dict[str, Any]] = []
    for case, ops in (("served", o), ("probe", probe_operands(o, probes))):
        rec, paths = launch(outdir, ops, case)
        doc["launches"].append(rec)
        extra = set(range(PROBE0, PROBE0 + max(len(p["x"]) for p in probes.values()))) if case == "probe" else set()
        sample = sorted(set(SAMPLE) | set(SPECIAL) | set(ZERO_POS) | {HIGH_POS} | extra)
        jobs += [{"case": case, "span": [r0, r1], "sample": [s for s in sample if r0 <= s < r1], "paths": paths}
                 for r0, r1 in DS._chunks(T, CHUNK)]
        print(f"[glm53-mtp-eh-norm] launch {case:6s} {json.dumps(rec)}", flush=True)
    torch.cuda.empty_cache()
    return jobs


def ptx_record(text: str) -> dict[str, Any]:
    code = text[:text.index("\t.file")] if "\t.file" in text else text
    kept = "\n".join(line for line in code.splitlines()
                     if not re.match(r"\s*\.loc\b", line) and not re.match(r"\$L__tmp\d+:\s*$", line))
    return {"code_sha256": hashlib.sha256(kept.encode()).hexdigest(), "counts": dict(Counter(_PTX_OPS.findall(code))),
            "reqntid": re.findall(r"\.reqntid\s+([\d, ]+)", code)}


def sass_record(cubin: Path) -> dict[str, Any]:
    """The cubin's SASS by Triton's own cuobjdump (else nvdisasm), and SASS_COUNTS' counts in it."""
    import triton

    nv = Path(triton.__file__).parent / "backends" / "nvidia" / "bin"
    for tool, args in (("cuobjdump", ["-sass"]), ("nvdisasm", ["-c"])):
        if (nv / tool).exists():
            r = subprocess.run([str(nv / tool), *args, str(cubin)], capture_output=True, text=True)
            if r.returncode == 0 and r.stdout.strip():
                text = r.stdout
                return {"tool": tool, "text": text, "counts": {"ffma_rcp_eps": len(_SASS_FFMA.findall(text)),
                                                               "MUFU.RSQ": text.count("MUFU.RSQ"), "MUFU.RCP": text.count("MUFU.RCP")}}
    return {"tool": None, "text": "", "counts": {}}


def ptx(triton_cache: Path, outdir: Path) -> dict[str, Any]:
    """ok when the cache holds one `_fused_eh_norm_kernel` specialisation, its code is the pinned PTX's and its SASS has the
    contracted epilogue; the PTX and SASS copied to <out>/ptx/."""
    dest = outdir / "ptx"
    dest.mkdir(parents=True, exist_ok=True)
    entries = []
    for f in sorted(glob.glob(str(triton_cache / "**" / f"{KERNEL}*.ptx"), recursive=True)):
        text = Path(f).read_text(errors="replace")
        tag = hashlib.sha256(text.encode()).hexdigest()[:12]
        shutil.copy(f, dest / f"{KERNEL}.{tag}.ptx")
        rec = ptx_record(text)
        cubin = Path(f).with_suffix(".cubin")
        s = sass_record(cubin) if cubin.exists() else {"tool": None, "text": "", "counts": {}}
        (dest / f"{KERNEL}.{tag}.sass").write_text(s.pop("text"))
        entries.append(rec | {"sass": s})
    ok = (len(entries) == 1 and entries[0]["code_sha256"] == PTX_CODE_SHA256 and entries[0]["counts"] == PTX_COUNTS
          and entries[0]["reqntid"] == ["128"] and entries[0]["sass"]["counts"] == SASS_COUNTS)
    return {"specialisations": entries, "want": {"code_sha256": PTX_CODE_SHA256, "counts": PTX_COUNTS, "sass": SASS_COUNTS}, "ok": ok}


def bind_check() -> dict[str, Any]:
    from verity.circuits.defs import bind
    from verity.definitions.registry import dsa_norm as DN
    from verity.definitions.registry import mtp_draft as MD

    out: dict[str, Any] = {}

    def whole() -> Any:
        return bind(MD.MtpEhNorm, H=H, BLOCK=BLOCK, WARPS=WARPS, EPS=EPS)

    makes = [whole, lambda: bind(DN.TritonBlockRMSNorm, N=H, BLOCK=BLOCK, WARPS=WARPS, EPS=EPS),
             *((lambda i=i: MD.eh_cut(whole())[i][0]) for i in range(3))]
    for make in makes:
        fn = make()
        out[fn.id] = bool(fn.signature())
    return {"bindings": out, "ok": len(out) == len(makes) and all(v is True for v in out.values())}


# ------------------------------------------------------------------------------------------------------------------------ compare
def compare(job: dict[str, Any]) -> dict[str, Any]:
    """Rows [r0, r1) of one launch: each half against the twin, `MtpEhNorm_v1`'s reference on the sampled rows, each alternative's
    words counted where they differ from the kernel's, and the position alternatives on the e half."""
    from verity.circuits.defs import bind
    from verity.circuits.evaluation import evaluate
    from verity.definitions.registry import mtp_draft as MD

    r0, r1 = job["span"]
    a = DS._load(job["paths"])
    pos = np.asarray(a["pos"][r0:r1])
    got = np.ascontiguousarray(a["out"][r0:r1])
    xs = {"e": zeroed(a["e"][r0:r1], pos), "p": np.ascontiguousarray(a["p"][r0:r1])}
    ws = {"e": np.ascontiguousarray(a["ew"]), "p": np.ascontiguousarray(a["hw"])}
    halves = {"e": got[:, :H], "p": got[:, H:]}
    out: dict[str, Any] = {"case": job["case"], "span": [r0, r1], "halves": {}}
    twins = {}
    for half in HALVES:
        twin = NR.block_rmsnorm_rows(xs[half], ws[half], EPS, BLOCK, WARPS)["out"]
        twins[half] = twin
        bad = twin != halves[half]
        rec: dict[str, Any] = {"rows": r1 - r0, "words": int(twin.size), "mismatch": int(bad.sum()),
                               "rows_differ": int(bad.any(axis=1).sum()), "reference": 0, "twin_mismatch": 0, "first_mismatch": None}
        if bad.any():
            i, j = (int(v[0]) for v in np.nonzero(bad))
            rec["first_mismatch"] = {"row": r0 + i, "col": j, "want": int(twin[i, j]), "kernel": int(halves[half][i, j])}
        rec["refuted"] = {name: int((FR.words(xs[half], ws[half], r(xs[half], EPS), product) != halves[half]).sum())
                          for name, (r, product) in alternatives().items()}
        out["halves"][half] = rec
    raw = np.ascontiguousarray(a["e"][r0:r1])
    position_alts = {"position_ignored": raw, "low_word_only": zeroed(raw, pos & 0xFFFFFFFF)}
    out["halves"]["e"]["refuted"] |= {name: int((NR.block_rmsnorm_rows(x, ws["e"], EPS, BLOCK, WARPS)["out"] != halves["e"]).sum())
                                      for name, x in position_alts.items()}
    fn = bind(MD.MtpEhNorm, H=H, BLOCK=BLOCK, WARPS=WARPS, EPS=EPS)
    want = np.concatenate([twins["e"], twins["p"]], axis=1)
    for s in job["sample"]:
        i = s - r0
        y = [int(v) for v in evaluate(fn, int(pos[i]) & ((1 << 64) - 1), a["e"][s].tolist(), a["p"][s].tolist(), ws["e"].tolist(),
                                      ws["p"].tolist())]
        for half, cols in (("e", slice(0, H)), ("p", slice(H, 2 * H))):
            out["halves"][half]["reference"] += 1
            out["halves"][half]["twin_mismatch"] += y[cols] != want[i, cols].tolist()
    return out


# ---------------------------------------------------------------------------------------------------------------------------- run
def summary(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {"cases": {}, "refuted": {}}
    for case in CASES:
        for half in HALVES:
            rs = [r["halves"][half] for r in comparisons if r["case"] == case]
            out["cases"][f"{case}/{half}"] = {k: sum(r[k] for r in rs) for k in TOTALS} | {
                "first_mismatch": next((r["first_mismatch"] for r in rs if r["first_mismatch"]), None)}
    for half in HALVES:
        rs = [r["halves"][half]["refuted"] for r in comparisons]
        out["refuted"][half] = {name: sum(r[name] for r in rs) for name in rs[0]} if rs else {}
    return out


def _case_ok(t: dict[str, Any]) -> bool:
    return t["rows"] > 0 and t["mismatch"] == 0 and t["reference"] > 0 and t["twin_mismatch"] == 0


def _verdict(doc: dict[str, Any]) -> bool:
    s = doc["summary"]
    return bool(doc["gate"]["ok"] and doc["served"].get("ok") is True and doc["ptx"].get("ok") is True
                and doc["bind"].get("ok") is True
                and {x["case"] for x in doc["launches"]} == set(CASES)
                and all(x["checks"] and all(v is True for v in x["checks"].values()) for x in doc["launches"])
                and len(s["cases"]) == len(CASES) * len(HALVES) and all(_case_ok(t) for t in s["cases"].values())
                and all(s["refuted"].get(h) and all(n >= PER_ALT for n in s["refuted"][h].values()) for h in HALVES))


def run(out: str, scratch: str | None, min_driver: int, want_cc: str, want_sms: int, workers: int) -> dict[str, Any]:
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.TemporaryDirectory(dir=scratch)
    work = Path(tmp.name)
    env, cleared = DS.environment(work)
    doc: dict[str, Any] = {"schema": SCHEMA, "argv": sys.argv, "env": env, "env_cleared": cleared, "served": DS.served(),
                           "environment": G._environment(), "gate": C.gate(min_driver, want_cc, want_sms),
                           "launches": []}
    print(f"[glm53-mtp-eh-norm] gate {json.dumps(doc['gate'])} served {json.dumps(doc['served'])} cleared {sorted(cleared)}", flush=True)
    if not doc["gate"]["ok"]:
        (outdir / RECORD).write_text(json.dumps(seal_record(doc, False), indent=1) + "\n")
        print("GLM53-MTP-EH-NORM GATE-FAIL", flush=True)
        raise SystemExit(3)
    jobs = gpu(outdir, doc)
    doc["ptx"] = ptx(work / "triton_cache", outdir)
    doc["bind"] = bind_check()
    print(f"[glm53-mtp-eh-norm] ptx {json.dumps(doc['ptx'])[:1500]}", flush=True)
    print(f"[glm53-mtp-eh-norm] bind {json.dumps(doc['bind'])}; comparing {len(jobs)} jobs on {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"), initializer=_worker_init) as ex:
        doc["comparisons"] = list(ex.map(compare, jobs))
    doc["summary"] = summary(doc["comparisons"])
    for key, t in doc["summary"]["cases"].items():
        print(f"[glm53-mtp-eh-norm] words {key:9s} rows={t['rows']} words={t['words']} !={t['mismatch']} rows!={t['rows_differ']} "
              f"ref={t['reference']} twin!={t['twin_mismatch']} first={json.dumps(t['first_mismatch'])}", flush=True)
    doc = seal_record(doc, _verdict(doc))
    (outdir / RECORD).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    tmp.cleanup()
    totals = {k: sum(t[k] for t in doc["summary"]["cases"].values()) for k in TOTALS}
    print(f"GLM53-MTP-EH-NORM {'GO' if doc['ok'] else 'NO-GO'} {doc['digest']} served={doc['served'].get('ok')} "
          f"words={json.dumps(totals)} refuted={json.dumps(doc['summary']['refuted'])} "
          f"ptx={doc['ptx'].get('ok')} bind={doc['bind'].get('ok')}", flush=True)
    return doc


def _worker_init() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="GLM-5.3's MTP fused enorm/hnorm (_fused_eh_norm_kernel) on this Hopper GPU against "
                                             "MtpEhNorm_v1")
    ap.add_argument("--out", required=True, help=f"directory for {RECORD}, ptx/ and words/")
    ap.add_argument("--scratch", default=None, help="parent of the scratch directory (the Triton cache)")
    ap.add_argument("--min-driver", type=int, default=575)
    ap.add_argument("--want-cc", default="9.0")
    ap.add_argument("--want-sms", type=int, default=132)
    ap.add_argument("--workers", type=int, default=C.cpu_workers())
    a = ap.parse_args(argv)
    doc = run(a.out, a.scratch, a.min_driver, a.want_cc, a.want_sms, a.workers)
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
