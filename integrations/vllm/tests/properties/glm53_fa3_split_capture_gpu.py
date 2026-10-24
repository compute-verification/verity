"""FA3's split sparse MLA on Hopper against `sparse_mla_split`: the H200's go/no-go for `SparseMlaHeadSplit_v1`, vLLM d9105ea80's
default `FLASH_ATTN_MLA_SPARSE` decode (`forward_mqa` passes no `num_splits`, so FA3 splits each token's key range by its
heuristic and `FlashAttnFwdCombine` merges the splits), and for `MufuLg2Ftz_v1` on sm_90, at GLM-5.3's TP8 statics (8 heads,
rope 64, latent 512, top-k 2048, the scale 1/16).

One GPU process writes `<out>/glm53_fa3_split_capture.json`, the property record `glm53_fa3_split_capture` (`properties.record`),
and prints one line per launch, then `GLM53-FA3-SPLIT GO` or `NO-GO`.  `ok` (GO) = every check below.  The device gate runs first
and nothing else runs when it fails (exit 3).  A phase that raises records its error and fails the verdict without stopping the
others.

  gate     `fa2_target_capture_gpu.gate`: driver, cc 9.0, 132 SMs (an H200).
  switches vLLM at its defaults, as it serves (`environment`: `engine.env.served_defaults` clears every switch the process
           inherited), and `served` reads batch invariance back from vLLM, off.
  mufu     lg2.approx.ftz.f32 on every word of the exponents LG2_EXPONENTS (FA3 takes the log of a split's sum, in [1 - 2^-24,
           2^12): exponents 126 to 138), on the sm_120 measurement's 2^20-word sweep, and on the powers of two below 1: each word
           against `MufuLg2Ftz_v1` (twin `mufu_lg2_words`, the reference on a sample) but the words the model leaves open
           (`mufu_lg2_undetermined`), whose device words are recorded (``open``); whether sm_90's words are sm_120's (the table's
           digests) is recorded.  ex2.approx.ftz.f32 and rcp.approx.ftz.f32, the combine's scale and inverse sum, on core's table
           domains and on the sweep against core's tables.
  attn     FA3 through vLLM's `flash_attn_varlen_func` as `forward_mqa` calls it (flashattn_mla_sparse.py:201-250: q = q_pe, q_v =
           q_nope (a transposed view), k and v the flat KV rows' k_pe and kv_c, max_seqlen_k the top-k width, seqused_k each token's
           valid count, the block table its selected rows, causal, fa_version 3) for each of LAUNCHES: the operand set's tokens
           (SETS: the DSA capture's batch, the attention probe's tokens, tokens alone, a ladder of lengths, wide batches, the
           batch-size limit at 992 and 993 tokens) with no `num_splits` (vLLM's call) or an explicit one (1, the serving pin; 28,
           the heuristic's own bound; 5).  Each launch twice (deterministic); its split counts `launch_num_splits` at the device's
           SM count; every (token, head) of the compared tokens (SAMPLE) against `SparseMlaHeadSplit_v1{T, NS}` (twin
           `head_split_rows`; the reference itself on REFERENCE); a token that differs is tried at every distinct split of its
           blocks (``matching_ns``, a diagnosis, not a check).  A launch of SAME gives its counterpart's words.
  bind     the Programs bind at the served statics.

GO needs the switches, every case's launch checks, its words compared with 0 mismatches and 0 unresolved, the reference sampled with 0
disagreements with the twin, and the bindings.

    cd integrations/vllm; PY=/workspace/venv312/bin/python     # PYTHONPATH as pod_bootstrap.sh sets it
    $PY tests/properties/glm53_fa3_split_capture_gpu.py --out DIR [--scratch DIR] [--quick]
"""
from __future__ import annotations

__models__ = ("zai-org/GLM-5.3",)

import argparse
import hashlib
import json
import multiprocessing
import os
import sys
import tempfile
import traceback
import zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from tests.properties import fa2_target_capture_gpu as C
from tests.properties import fa_tap_exactness_gpu as G
from verity_vllm.engine.env import served_defaults
from verity_vllm.properties.record import seal as seal_record

SCHEMA = "verity-vllm/glm53-fa3-split-capture/v2"
RECORD = "glm53_fa3_split_capture.json"
TAG = "glm53-fa3-split"
#: GLM-5.3 (tests/check/fixtures/glm_moe_dsa/config.json) at TP8: 8 local MLA heads, rope 64, latent 512, nope 192, top 2048
NH, DV, DR, NOPE, TOPK = 8, 512, 64, 192, 2048
KW = DV + DR
SCALE = (NOPE + DR) ** -0.5
#: the KV rows every token selects from, and the scales of q (alternating by token) and of the rows
POOL, Q_SCALES, KV_SCALE = 16384, (1.0, 3.0), 0.5
SEED = 20261013
#: the operand sets: each token's valid count; its q and selected rows are drawn from SEED and the set's name
SOLO = (1, 64, 65, 192, 320, 448, 640, 1000, 1500, 1793, 1856, 2047, 2048)
SETS: dict[str, tuple[int, ...]] = {
    "dsa14": (1, 2, 5, 64, 65, 66, 128, 129, 130, 1001, 2047, 2048, 2048, 2048),
    "probe5": (129, 130, 1001, 2048, 2048),
    "ladder32": tuple(64 * k for k in range(1, 33)),
    "wide8": (2048,) * 8,
    "wide24": (2048,) * 24,
    "edge992": (1,) * 991 + (2048,),
    "edge993": (1,) * 992 + (2048,),
    **{f"solo{t}": (t,) for t in SOLO},
}
#: the launches: name -> (operand set, num_splits argument; 0 passes none, as `forward_mqa` does)
LAUNCHES: dict[str, tuple[str, int]] = {**{s: (s, 0) for s in SETS}, "pin1": ("dsa14", 1), "bound28": ("dsa14", 28),
                                        "bound5": ("probe5", 5)}
#: a launch whose words must be its counterpart's: the explicit bound the heuristic itself takes
SAME = {"bound28": "dsa14"}
#: the tokens compared where not every one is (the batch-size limit's sets: the long token and a few short ones)
SAMPLE = {"edge992": (0, 1, 990, 991), "edge993": (0, 1, 991, 992)}
#: (launch, token) whose head 0 the reference itself evaluates
REFERENCE = (("dsa14", 8), ("dsa14", 9), ("probe5", 0), ("solo1856", 0), ("pin1", 13), ("edge992", 991))
QUICK_LAUNCHES = ("dsa14", "probe5", "pin1", "bound28", "solo2048")
LG2_EXPONENTS, QUICK_LG2_EXPONENTS = tuple(range(126, 141)), (126, 127, 128)
#: mantissas per lg2 comparison job
LG2_JOB = 1 << 21
TOTALS = ("words", "mismatch", "unresolved", "reference", "twin_mismatch")
#: the token diagnoses kept per case
DIFFERING = 32


def _error(e: BaseException) -> str:
    return "".join(traceback.format_exception_only(type(e), e)).strip()[-600:] + " | " + traceback.format_exc(limit=-3)[-900:]


def _bf16(rng: np.random.Generator, shape, scale) -> np.ndarray:
    x = rng.standard_normal(shape).astype(np.float32) * np.asarray(scale, dtype=np.float32)
    return (x.view(np.uint32) >> 16).astype(np.uint16)


def _save(d: Path, **arrays: np.ndarray) -> dict[str, str]:
    d.mkdir(parents=True, exist_ok=True)
    out = {}
    for k, a in arrays.items():
        np.save(d / f"{k}.npy", np.ascontiguousarray(a))
        out[k] = str(d / f"{k}.npy")
    return out


def _load(paths: dict[str, str]) -> dict[str, np.ndarray]:
    return {k: np.load(p, mmap_mode="r") for k, p in paths.items()}


def _chunks(n: int, step: int) -> list[tuple[int, int]]:
    return [(i, min(n, i + step)) for i in range(0, n, step)]


def _sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype="<u4").tobytes()).hexdigest()


def _mla() -> dict[str, Any]:
    from verity.ml.definitions.registry import sparse_mla as M

    return {k: v for k, v in M.SPARSE_MLA_GLM53.items() if k != "NH"}


# ------------------------------------------------------------------------------------------------------------------- operands
def operands(names) -> dict[str, Any]:
    """The KV pool (POOL rows of [kv_c | k_pe] bf16 words) and, per operand set of the launches ``names``, each token's q_nope
    (S, NH, DV), q_rope (S, NH, DR), its valid count and its selected rows (S, TOPK), distinct and padded with -1."""
    pool = _bf16(np.random.default_rng(SEED), (POOL, KW), KV_SCALE)
    sets = {}
    for s in sorted({LAUNCHES[n][0] for n in names}):
        valid = np.array(SETS[s], dtype=np.int32)
        rng = np.random.default_rng([SEED, zlib.crc32(s.encode())])
        S = valid.size
        qs = np.array([Q_SCALES[i % len(Q_SCALES)] for i in range(S)], dtype=np.float32)[:, None, None]
        rows = np.full((S, TOPK), -1, dtype=np.int32)
        for i, t in enumerate(valid):
            rows[i, :t] = rng.choice(POOL, int(t), replace=False)
        sets[s] = {"valid": valid, "qn": _bf16(rng, (S, NH, DV), qs), "qr": _bf16(rng, (S, NH, DR), qs), "rows": rows}
    return {"pool": pool, "sets": sets}


# --------------------------------------------------------------------------------------------------------------------- the jobs
def lg2_jobs(work: Path, case: str, y: np.ndarray, *, e: int | None = None, x: np.ndarray | None = None) -> list[dict[str, Any]]:
    """The comparison jobs of one lg2 launch: the band of exponent ``e`` (x = e << 23 | m) or the words ``x``."""
    paths = _save(work / f"lg2_{case}", y=y, **({} if x is None else {"x": x}))
    return [{"phase": "mufu", "case": "lg2", "kind": "lg2", "paths": paths, "e": e, "range": list(r)} for r in _chunks(y.size, LG2_JOB)]


def attn_jobs(work: Path, name: str, s: dict[str, np.ndarray], out: np.ndarray, num_sm: int, pool: dict[str, str],
              checks: dict[str, bool]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """One launch's record (its split counts at ``num_sm`` SMs) and its comparison jobs, one per compared token."""
    from verity.ml.definitions.registry import sparse_mla_split as W

    set_name, arg = LAUNCHES[name]
    valid = [int(t) for t in s["valid"]]
    ns = W.launch_num_splits(valid, num_sm, TOPK, arg)
    paths = _save(work / f"attn_{name}", qn=s["qn"], qr=s["qr"], rows=s["rows"], out=out)
    rec = {"phase": "attn", "case": name, "operands": set_name, "num_splits_arg": arg, "num_sm": num_sm, "valid": valid,
           "num_splits": ns, "static_bound": W.static_num_splits(num_sm, TOPK, len(valid)),
           "splits": {str(k): v for k, v in sorted(Counter(ns).items())}, "checks": checks}
    tokens = SAMPLE.get(set_name, range(len(valid)))
    jobs = [{"phase": "attn", "case": name, "kind": "attn", "paths": paths, "pool": pool, "token": int(i), "T": valid[i],
             "NS": ns[i], "reference": (name, int(i)) in REFERENCE} for i in tokens]
    return rec, jobs


# ------------------------------------------------------------------------------------------------------------------ comparisons
def matching_ns(qn: np.ndarray, qr: np.ndarray, kv: np.ndarray, T: int, got: np.ndarray) -> list[int]:
    """The split counts at which the twin gives every head's words: the least NS of each distinct `split_blocks` partition."""
    from verity.ml.definitions.numerics import sparse_mla_split_twins as TW
    from verity.ml.definitions.registry import sparse_mla_split as W

    seen, out = set(), []
    for ns in range(1, W.LANES + 1):
        key = tuple(W.split_blocks(T, ns))
        if key in seen:
            continue
        seen.add(key)
        if np.array_equal(TW.head_split_rows(qn, qr, kv, T=T, NS=ns, scale=SCALE)[0], got):
            out.append(ns)
    return out


def compare_attn(job: dict[str, Any], rec: dict[str, Any]) -> None:
    """One token's NH heads against the twin of `SparseMlaHeadSplit_v1{T, NS}`; head 0 against the reference on REFERENCE."""
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation import evaluate
    from verity.ml.definitions.numerics import sparse_mla_split_twins as TW
    from verity.ml.definitions.registry import sparse_mla_split as W

    a = _load(job["paths"])
    i, T, NS = job["token"], job["T"], job["NS"]
    kv = np.asarray(_load(job["pool"])["pool"])[np.asarray(a["rows"][i, :T])]
    qn, qr, got = (np.asarray(a[k][i]) for k in ("qn", "qr", "out"))
    want, nf = TW.head_split_rows(qn, qr, kv, T=T, NS=NS, scale=SCALE)
    bad = (want != got) & ~nf[:, None]
    rec["words"] += int((~nf).sum()) * DV
    rec["unresolved"] += int(nf.sum()) * DV
    rec["mismatch"] += int(bad.sum())
    if bad.any():
        h, w = (int(v) for v in np.argwhere(bad)[0])
        d = {"launch": job["case"], "token": i, "T": T, "NS": NS, "mismatch": int(bad.sum()), "head": h, "word": w,
             "want": int(want[h, w]), "kernel": int(got[h, w]), "matching_ns": matching_ns(qn, qr, kv, T, got)}
        rec["first_mismatch"] = rec["first_mismatch"] or d
        rec["differing"].append(d)
    if job["reference"]:
        fn = bind(W.SparseMlaHeadSplit, T=T, NS=NS, **_mla())
        y = np.asarray(evaluate(fn, qn[0].tolist(), qr[0].tolist(), kv.reshape(-1).tolist()), dtype=np.uint16)
        rec["reference"] += 1
        rec["twin_mismatch"] += int(not np.array_equal(y, want[0]))


def compare_lg2(job: dict[str, Any], rec: dict[str, Any]) -> None:
    """Words [lo, hi) of an lg2 launch against `MufuLg2Ftz_v1`'s twin, the open words skipped; the reference on the first two."""
    from verity.ml.circuits.evaluation import evaluate
    from verity.ml.definitions.numerics import sparse_mla_split_twins as TW
    from verity.ml.definitions.registry import sparse_mla_split as W

    a = _load(job["paths"])
    lo, hi = job["range"]
    if job["e"] is None:
        x = np.asarray(a["x"][lo:hi], dtype=np.uint32)
    else:
        x = (np.uint32(job["e"]) << np.uint32(23)) | np.arange(lo, hi, dtype=np.uint32)
    y = np.asarray(a["y"][lo:hi], dtype=np.uint32)
    want = TW.mufu_lg2_words(x)
    open_ = TW.undetermined_words(x, want)
    bad = np.flatnonzero((want != y) & ~open_)
    rec["words"] += int((~open_).sum())
    rec["mismatch"] += int(bad.size)
    if bad.size and rec["first_mismatch"] is None:
        rec["first_mismatch"] = {"x": int(x[bad[0]]), "want": int(want[bad[0]]), "kernel": int(y[bad[0]])}
    for j in range(min(2, x.size)):
        rec["reference"] += 1
        rec["twin_mismatch"] += evaluate(W.MufuLg2Ftz, int(x[j]))[0] != int(want[j])


def compare_mufu(job: dict[str, Any], rec: dict[str, Any]) -> None:
    """ex2.approx.ftz / rcp.approx.ftz on the sweep against core's tables (`derived_rows.mufu_ex2_ftz`, `mufu_rcp_ftz`)."""
    from verity.ml.definitions.numerics import derived_rows as DR

    a = _load(job["paths"])
    x, y = np.asarray(a["x"], dtype=np.uint32), np.asarray(a["y"], dtype=np.uint32)
    want = {"ex2": DR.mufu_ex2_ftz, "rcp": DR.mufu_rcp_ftz}[job["case"]](x)
    bad = np.flatnonzero(want != y)
    rec["words"] += int(y.size)
    rec["mismatch"] += int(bad.size)
    if bad.size and rec["first_mismatch"] is None:
        rec["first_mismatch"] = {"x": int(x[bad[0]]), "want": int(want[bad[0]]), "kernel": int(y[bad[0]])}


COMPARE = {"attn": compare_attn, "lg2": compare_lg2, "mufu": compare_mufu}


def compare(job: dict[str, Any]) -> dict[str, Any]:
    rec = {**{k: 0 for k in TOTALS}, "first_mismatch": None, "differing": []}
    try:
        COMPARE[job["kind"]](job, rec)
    except Exception as e:  # noqa: BLE001
        return {"phase": job["phase"], "case": job["case"], "error": _error(e)}
    return {"phase": job["phase"], "case": job["case"], **rec}


# ------------------------------------------------------------------------------------------------------------------------ the GPU
def _mufu(op: str, x: np.ndarray) -> np.ndarray:
    import torch

    from verity_vllm.program import mla_capture as MC

    return MC._mufu(op, torch.from_numpy(x.view(np.float32).copy()).to("cuda")).cpu().numpy().view(np.uint32)


def sweep_words() -> np.ndarray:
    """The sm_120 measurement's sweep (`mla_capture.mufu_probe`)."""
    from verity_vllm.program import mla_capture as MC

    return MC._splitmix(np.arange(1 << 20, dtype=np.uint64) + (np.uint64(0xABC) << np.uint64(40))).astype(np.uint32)


def phase_mufu(work: Path, quick: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """lg2 on the exponent bands, the sweep and the powers of two below 1; ex2 and rcp on core's table domains and the sweep."""
    from verity.ml.circuits.ops import mufu as MU
    from verity.ml.definitions.numerics import sparse_mla_split_twins as TW
    from verity.ml.definitions.registry import sparse_mla_split as W

    jobs: list[dict[str, Any]] = []
    m = np.arange(1 << 23, dtype=np.uint32)
    exps = QUICK_LG2_EXPONENTS if quick else LG2_EXPONENTS
    band = {}
    for e in exps:
        y = _mufu("lg2_ftz", (np.uint32(e) << np.uint32(23)) | m)
        band[e] = y
        jobs += lg2_jobs(work, f"E{e}", y, e=e)
    sweep = sweep_words()
    ys = _mufu("lg2_ftz", sweep)
    jobs += lg2_jobs(work, "sweep", ys, x=sweep)
    powers = np.arange(1, 127, dtype=np.uint32) << np.uint32(23)
    yp = _mufu("lg2_ftz", powers)
    exact = (np.float32(np.arange(1, 127)) - np.float32(127)).view(np.uint32)
    prov = W.lg2_table()["provenance"]["measured"]
    sm120 = {"mantissas": prov["mantissas"].rsplit(" ", 1)[1], "sweep": prov["sweep"].rsplit(" ", 1)[1]}
    lg2 = {"phase": "mufu", "case": "lg2", "exponents": list(exps), "sweep_in": _sha(sweep), "sweep_out": _sha(ys),
           "open": {f"{int(x):#010x}": f"{int(v):#010x}" for x, v in zip(powers, yp)},
           "open_model": {f"{int(x):#010x}": f"{int(v):#010x}" for x, v in zip(powers, TW.mufu_lg2_words(powers))},
           "open_exact": bool(np.array_equal(yp, exact)),
           "same_as_sm120": {"sweep": _sha(ys) == sm120["sweep"], "mantissas": _sha(band[127]) == sm120["mantissas"] if 127 in band else None},
           "checks": {"sweep_input": _sha(sweep) in prov["sweep"], "open_are_the_powers": bool(TW.undetermined_words(powers).all())}}
    recs = [lg2]
    j = np.arange(1 << 23, dtype=np.uint32)
    domains = {"ex2": (j.astype(np.float64) / (1 << 23)).astype(np.float32).view(np.uint32), "rcp": np.uint32(0x3F800000) | j}
    for op in ("ex2", "rcp"):
        y = _mufu(f"{op}_ftz", domains[op])
        bad = np.flatnonzero(y != MU.table(op))
        recs.append({"phase": "mufu", "case": op, "table_words": int(y.size), "table_mismatch": int(bad.size),
                     "checks": {"table": not bad.size}})
        jobs.append({"phase": "mufu", "case": op, "kind": "mufu", "paths": _save(work / f"{op}_sweep", x=sweep, y=_mufu(f"{op}_ftz", sweep))})
    return recs, jobs


def phase_attn(work: Path, names, num_sm: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Every launch of ``names`` twice through vLLM's `flash_attn_varlen_func`, as `forward_mqa` calls it."""
    import torch
    from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

    def dev(a: np.ndarray):
        return torch.from_numpy(np.ascontiguousarray(a).view(np.int16)).to("cuda").view(torch.bfloat16)

    o = operands(names)
    rows = dev(o["pool"])
    k_cache, v_cache = rows[:, DV:].unsqueeze(1).unsqueeze(1), rows[:, :DV].unsqueeze(1).unsqueeze(1)
    pool = _save(work / "pool", pool=o["pool"])
    recs, jobs, outs = {}, [], {}
    for name in names:
        set_name, arg = LAUNCHES[name]
        s = o["sets"][set_name]
        S = s["valid"].size
        q_nope = dev(s["qn"].transpose(1, 0, 2)).transpose(0, 1)
        q_rope = dev(s["qr"])
        valid = torch.from_numpy(s["valid"]).to("cuda")
        table = torch.from_numpy(s["rows"]).to("cuda")
        cu = torch.arange(0, S + 1, dtype=torch.int32, device="cuda")
        kw = {"num_splits": arg} if arg else {}

        def fa3():
            return flash_attn_varlen_func(q=q_rope, k=k_cache, v=v_cache, q_v=q_nope, max_seqlen_q=1, cu_seqlens_q=cu,
                                          max_seqlen_k=TOPK, seqused_k=valid, block_table=table, softmax_scale=SCALE, causal=True,
                                          fa_version=3, **kw)

        out1, out2 = fa3(), fa3()
        torch.cuda.synchronize()
        checks = {"deterministic": bool(torch.equal(out1.view(torch.int16), out2.view(torch.int16))),
                  "shape": list(out1.shape) == [S, NH, DV] and out1.dtype == torch.bfloat16}
        outs[name] = out1.contiguous().view(torch.int16).cpu().numpy().view(np.uint16)
        recs[name], js = attn_jobs(work, name, s, outs[name], num_sm, pool, checks)
        jobs += js
    for a, b in SAME.items():
        if a in recs and b in recs:
            recs[a]["checks"][f"same_as_{b}"] = bool(np.array_equal(outs[a], outs[b]))
    return list(recs.values()), jobs


def bind_check() -> dict[str, Any]:
    """The Programs bind at GLM-5.3's statics."""
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import sparse_mla as M
    from verity.ml.definitions.registry import sparse_mla_split as W

    mla = _mla()
    targets = {"MufuLg2Ftz_v1": lambda: W.MufuLg2Ftz,
               f"SparseMlaSplit_v1{{T={TOPK},NS=16,NH={NH},...}}": lambda: bind(W.SparseMlaSplit, T=TOPK, NS=16, **M.SPARSE_MLA_GLM53)}
    for T, NS in ((130, 2), (1001, 8), (TOPK, 16), (TOPK, 28)):
        targets[f"SparseMlaHeadSplit_v1{{T={T},NS={NS},...}}"] = lambda T=T, NS=NS: bind(W.SparseMlaHeadSplit, T=T, NS=NS, **mla)
    out = {}
    for name, make in targets.items():
        try:
            out[name] = bool(make().signature())
        except Exception as e:  # noqa: BLE001
            out[name] = _error(e)
    return {"bindings": out, "ok": all(v is True for v in out.values())}


# ------------------------------------------------------------------------------------------------------------------------- run
def environment(work: Path) -> tuple[dict[str, str], dict[str, str]]:
    """vLLM's defaults (`served_defaults`) and the Triton cache in scratch, set before vLLM is imported: (the env set, the switches
    cleared)."""
    cleared = served_defaults()
    env = {"VLLM_ENABLE_V1_MULTIPROCESSING": "0", "TRITON_CACHE_DIR": str(work / "triton_cache")}
    os.environ.update(env)
    return env, cleared


def served() -> dict[str, Any]:
    """The one switch on the decode's path, as vLLM read it: batch invariance, off (`forward_mqa` passes fa_version 3 and reads no
    switch; `fa_utils` reads batch invariance for FA4 only)."""
    import vllm.envs as envs

    rec: dict[str, Any] = {"VLLM_BATCH_INVARIANT": bool(envs.VLLM_BATCH_INVARIANT)}
    rec["ok"] = not rec["VLLM_BATCH_INVARIANT"]
    return rec


def gpu(work: Path, quick: bool, doc: dict[str, Any]) -> list[dict[str, Any]]:
    """The GPU phases; each phase's error goes to doc["errors"].  Fills doc["launches"]; returns the comparison jobs."""
    jobs: list[dict[str, Any]] = []
    names = QUICK_LAUNCHES if quick else tuple(LAUNCHES)
    for phase, fn in (("mufu", lambda: phase_mufu(work, quick)), ("attn", lambda: phase_attn(work, names, doc["gate"]["sms"]))):
        try:
            recs, js = fn()
        except Exception as e:  # noqa: BLE001
            doc["errors"][phase] = _error(e)
            print(f"[{TAG}] {phase} ERROR {doc['errors'][phase][:600]}", flush=True)
            continue
        doc["launches"] += recs
        jobs += js
        for r in recs:
            info = {k: r[k] for k in ("num_splits_arg", "splits", "static_bound", "same_as_sm120", "open_exact") if k in r}
            print(f"[{TAG}] {phase} {r['case']:10s} {json.dumps(r['checks'])} {json.dumps(info)[:600]}", flush=True)
    return jobs


def _launch_ok(x: dict[str, Any]) -> bool:
    return bool(x["checks"]) and all(v is True for v in x["checks"].values())


def _words_ok(t: dict[str, Any]) -> bool:
    return t["words"] > 0 and t["mismatch"] == 0 and t["unresolved"] == 0 and t["twin_mismatch"] == 0


def cases(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Each launch's verdict (mufu/lg2, mufu/ex2, mufu/rcp, attn/<launch>): its checks, its words and its errors."""
    out: dict[str, dict[str, Any]] = {}
    for x in doc["launches"]:
        key = f"{x['phase']}/{x['case']}"
        comps = [r for r in doc.get("comparisons", []) if (r["phase"], r["case"]) == (x["phase"], x["case"])]
        t = {k: sum(r[k] for r in comps) for k in TOTALS}
        errors = [e for e in doc["errors"] if e == f"compare/{key}"]
        out[key] = {"launch_ok": _launch_ok(x), "totals": t, "errors": errors, "ok": _launch_ok(x) and _words_ok(t) and not errors}
    return out


def _verdict(doc: dict[str, Any]) -> bool:
    c = cases(doc)
    want = {"mufu/lg2", "mufu/ex2", "mufu/rcp"} | {f"attn/{n}" for n in (QUICK_LAUNCHES if doc.get("quick") else LAUNCHES)}
    return bool(doc["gate"]["ok"] and doc["served"].get("ok") is True and not doc["errors"] and doc["bind"].get("ok") is True
                and want <= set(c)
                and all(v["ok"] for v in c.values()) and _words_ok(doc["totals"]) and doc["totals"]["reference"] > 0)


def run(out: str, scratch: str | None, min_driver: int, want_cc: str, want_sms: int, workers: int, quick: bool) -> dict[str, Any]:
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    tmp = tempfile.TemporaryDirectory(dir=scratch)
    work = Path(tmp.name)
    env, cleared = environment(work)
    doc: dict[str, Any] = {"schema": SCHEMA, "argv": sys.argv, "env": env, "env_cleared": cleared, "served": served(), "quick": quick,
                           "environment": G._environment(), "gate": C.gate(min_driver, want_cc, want_sms), "errors": {},
                           "launches": []}
    print(f"[{TAG}] gate {json.dumps(doc['gate'])} served {json.dumps(doc['served'])} cleared {sorted(cleared)}", flush=True)
    if not doc["gate"]["ok"]:
        (outdir / RECORD).write_text(json.dumps(seal_record(doc, False), indent=1) + "\n")
        print("GLM53-FA3-SPLIT GATE-FAIL", flush=True)
        raise SystemExit(3)
    jobs = gpu(work, quick, doc)
    try:
        doc["bind"] = bind_check()
    except Exception as e:  # noqa: BLE001
        doc["errors"]["bind"] = _error(e)
        doc["bind"] = {"ok": False}
    print(f"[{TAG}] bind ok={doc['bind']['ok']}; comparing {len(jobs)} jobs on {workers} workers", flush=True)
    results: dict[tuple, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        for res in ex.map(compare, jobs):
            if "error" in res:
                doc["errors"].setdefault(f"compare/{res['phase']}/{res['case']}", res["error"])
                continue
            agg = results.setdefault((res["phase"], res["case"]), {"phase": res["phase"], "case": res["case"],
                                                                  **{k: 0 for k in TOTALS}, "first_mismatch": None, "differing": []})
            for k in TOTALS:
                agg[k] += res[k]
            agg["first_mismatch"] = agg["first_mismatch"] or res["first_mismatch"]
            agg["differing"] = (agg["differing"] + res["differing"])[:DIFFERING]
    doc["comparisons"] = list(results.values())
    for r in doc["comparisons"]:
        print(f"[{TAG}] words {r['phase']:4s} {r['case']:10s} words={r['words']} !={r['mismatch']} unresolved={r['unresolved']} "
              f"ref={r['reference']} twin!={r['twin_mismatch']} first={json.dumps(r['first_mismatch'])}", flush=True)
    doc["totals"] = {k: sum(r[k] for r in doc["comparisons"]) for k in TOTALS}
    doc["cases"] = cases(doc)
    doc = seal_record(doc, _verdict(doc))
    (outdir / RECORD).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    tmp.cleanup()
    print(f"GLM53-FA3-SPLIT {'GO' if doc['ok'] else 'NO-GO'} {doc['digest']} served={doc['served'].get('ok')} "
          f"totals={json.dumps(doc['totals'])} "
          f"no-go cases={[k for k, c in doc['cases'].items() if not c['ok']]} errors={sorted(doc['errors'])}", flush=True)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="FA3's split sparse MLA and MUFU.LG2 on this Hopper GPU against SparseMlaHeadSplit_v1")
    ap.add_argument("--out", required=True, help=f"directory for {RECORD}")
    ap.add_argument("--scratch", default=None, help="parent of the scratch directory (operands, outputs, the Triton cache)")
    ap.add_argument("--min-driver", type=int, default=575)
    ap.add_argument("--want-cc", default="9.0")
    ap.add_argument("--want-sms", type=int, default=132)
    ap.add_argument("--workers", type=int, default=C.cpu_workers())
    ap.add_argument("--quick", action="store_true", help=f"launches {QUICK_LAUNCHES}, lg2 on exponents {QUICK_LG2_EXPONENTS}")
    a = ap.parse_args(argv)
    doc = run(a.out, a.scratch, a.min_driver, a.want_cc, a.want_sms, a.workers, a.quick)
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
