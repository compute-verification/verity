"""FA2 with a logit soft cap on a target: the stock vllm-flash-attn `Is_softcap` kernel vLLM launches for Gemma-2, against
`AttentionSoftcap_v2` (`registry.fa2_softcap`), the softcapped `Attention_v5` the target binds.

One GPU process writes `<out>/fa2_softcap_capture.json`, the property record `fa2_softcap_capture` (`properties.record`).  `ok` = the
gate, MUFU ex2 / rcp, MUFU.TANH, every launch's tile and determinism, and every row equal to `AttentionSoftcap_v2` (which the target
binds) on every head.  The device gate runs first and nothing else runs when it fails (exit 3):

  gate     `fa2_target_capture_gpu.gate`.
  binary   `fa2_target_capture_gpu.binary`: the SASS and PTX `_vllm_fa2_C` and `_C` carry.
  mufu     `fa2_target_capture_gpu.mufu`: ex2 / rcp tables and reduction rule on all 2^32 inputs, JIT compute_80 PTX and native.
  tanh     `program/kernels/cuda/tanh_probe.cu` built as compute_80 PTX (JIT-compiled by the driver, as FA2 is): `tanh.approx.f32` on
           every 32-bit word against `MufuTanh_v1`'s three rules and its measured table, and its odd symmetry.  A mismatch keeps the
           device's table over the shard (`tanh_sm*.bin`), the table another MufuTanh version would pin.
  launch   `fa2_target_capture_gpu.launch_geometry` of every case (torch.profiler).
  cases    at head dims 128 and 256, `fa_tap_exactness.cases(2)` with a softcap (the sliding window (4095, 0) is wider than every key
           length, so the kernel runs causal), a `wide` case (every operand word scaled by 2^U(-8, 8)), and a mixed and a decode launch at
           each pinned Gemma-2 shape.  Every output row, run twice, against `AttentionSoftcap_v2{T,NH,KVH,D,BN=fa2_kblock_n(D),CAP,
           DOT=HopperBF16WgmmaDot16_v1,INV=Fa2InvSum_v1,MASKED_FROM}` by its vectorised twin (`softcap_rows.softcap_head_states`), by the
           reference evaluator on every head the twin flags non-finite and on a sample of finite ones (where the twin must agree), by the
           replay row kernel (`rows.attention_softcap`, on the row's q / K / V words; it declines rows with a non-finite head), and
           the target's own binding of every row (`targets.attention_spec(..., cap=)`).

    cd integrations/vllm; PY=/workspace/venv312/bin/python     # PYTHONPATH as pod_bootstrap.sh sets it
    $PY tests/properties/fa2_softcap_capture_gpu.py --out DIR --want-cc 12.0 --want-sms 188 --min-driver 575
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from tests.properties import fa2_target_capture_gpu as C
from tests.properties import fa_tap_exactness_gpu as G
from verity_vllm.program.kernels import rows as RK
from verity_vllm.program.kernels import softcap_rows as SR
from verity_vllm.properties.fa_tap_exactness import PAGE, cases as tap_cases
from verity_vllm.properties.record import seal as seal_record

SCHEMA = "verity-vllm/fa2-softcap-capture/v1"
RECORD = "fa2_softcap_capture.json"
HDIMS = (128, 256)
CAP = 50.0
TANH_SRC = Path(__file__).resolve().parents[2] / "verity_vllm/program/kernels/cuda/tanh_probe.cu"
#: (heads, KV heads, head dim) of the pinned Gemma-2 checkpoints whose selector scale is D^-0.5 (`query_pre_attn_scalar == head_dim`)
MODELS = {"gemma-2-2b": (8, 4, 256), "gemma-2-9b": (16, 8, 256)}
#: finite heads per launch the reference evaluator also runs (the twin's own check)
FINITE_SAMPLE = 2


# ------------------------------------------------------------------------------------------------------------------------ tanh
def tanh_table() -> tuple[np.ndarray, int, dict[str, Any]]:
    """`MufuTanh_v1`'s table words over its shard, decoded from the in-tree blocks: (words, lo, the shard's record)."""
    from verity.ml.circuits.ops import mufu

    (s,) = mufu.mufu_tanh_shards()
    words = np.concatenate([s.block(k) for k in range(len(s.header["blocks"]))]).astype("<u4")
    sha = hashlib.sha256(words.tobytes()).hexdigest()
    return words, s.lo, {"lo": f"0x{s.lo:08X}", "hi": f"0x{s.hi:08X}", "words": int(words.size), "sha256_words": sha,
                         "sha256_words_recorded": s.sha_words, "ok": words.size == s.hi - s.lo and sha == s.sha_words}


def tanh_probe(work: Path, outdir: Path, ref: Path, lo: int, hi: int, sm: str) -> dict[str, Any]:
    from verity.ml.circuits.ops import mufu

    R = mufu.MUFU_TANH_RULES
    exe = work / "tanh_probe_jit_compute_80"
    b = C._run([C._cuda_tool("nvcc"), "-O2", "-gencode", "arch=compute_80,code=compute_80", "-o", str(exe), str(TANH_SRC)])
    if b.returncode:
        return {"error": f"nvcc: {b.stderr[-400:]}", "ok": False}
    meas = work / f"tanh_{sm}.bin"
    r = C._run([str(exe), str(ref), hex(lo), hex(hi), hex(R["identity_below"]), hex(R["saturate_from"]), str(meas)])
    try:
        got = json.loads(r.stdout)
    except ValueError:
        return {"rc": r.returncode, "error": (r.stdout + r.stderr)[-400:], "ok": False}
    classes = ("nan", "saturate", "identity", "table", "odd_symmetry")
    rec = {"rc": r.returncode, **got, "rules": {k: f"0x{v:08X}" for k, v in R.items()}}
    rec["ok"] = (r.returncode == 0 and got["nan"]["checked"] + got["saturate"]["checked"] + got["identity"]["checked"]
                 + got["table"]["checked"] == 1 << 32 and all(got[c]["mismatches"] == 0 for c in classes))
    if rec["ok"]:
        meas.unlink()
    else:
        kept = outdir / meas.name
        os.replace(meas, kept)
        rec["table_kept"] = str(kept)
        rec["table_kept_sha256"] = C._sha256_file(str(kept))
    return rec


mufu_tanh, softcap_head_states = SR.mufu_tanh, SR.softcap_head_states


# ----------------------------------------------------------------------------------------------------------------------- cases
def case_list() -> list[dict[str, Any]]:
    out = [{**c, "hdims": list(HDIMS)} for c in tap_cases(2) if c["softcap"] > 0]
    out.append({"name": "wide_softcap", "H": 8, "Hk": 4, "seqlens_q": [1, 64, 65, 200, 1], "seqlens_k": [1, 64, 65, 200, 700],
                "softcap": CAP, "window": [-1, -1], "wide": True, "hdims": list(HDIMS)})
    for m, (H, Hk, D) in MODELS.items():
        out.append({"name": f"{m}_mixed", "H": H, "Hk": Hk, "seqlens_q": [1, 17, 64, 65, 129, 1], "seqlens_k": [1, 17, 64, 65, 129, 600],
                    "softcap": CAP, "window": [4095, 0], "hdims": [D]})
        out.append({"name": f"{m}_decode", "H": H, "Hk": Hk, "seqlens_q": [1] * 5, "seqlens_k": [1, 64, 65, 300, 1031],
                    "softcap": CAP, "window": [4095, 0], "hdims": [D]})
    return out


def compare(job: dict[str, Any]) -> dict[str, Any]:
    """Every output row of one launch against `AttentionSoftcap_v2` (a worker process: numpy words in, a record out)."""
    from verity.ml.circuits.evaluation.reference import evaluate
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.frontend.target_profile import TargetProfile
    from verity_vllm.program.kernels import derived_rows as DR
    from verity_vllm.program.registry import b1, hopper
    from verity_vllm.program.registry import fa2_softcap as F2S
    from verity_vllm.program.registry import prims as P
    from verity_vllm.program.frontend import targets as TG

    table = np.memmap(job["tanh_table"], dtype="<u4", mode="r")
    tanh = lambda x: mufu_tanh(x, table, job["tanh_lo"])  # noqa: E731
    c, D, q, k, v, o, bt, cap = job["case"], job["D"], job["q"], job["k"], job["v"], job["out"], job["block_table"], job["case"]["softcap"]
    tp = TargetProfile.default().with_overrides(compute_capability=tuple(job["cc"]), num_sms=int(job["sms"]))
    H, Hk = c["H"], c["Hk"]
    BN = b1.fa2_kblock_n(D)
    kvh = np.arange(H) // (H // Hk)
    sq_max = max(c["seqlens_q"])
    dot16, inv = DR.TC_DOT16_BY_PRIM["HopperBF16WgmmaDot16_v1"], DR.INV_SUM_BY_PRIM["Fa2InvSum_v1"]
    rec: dict[str, Any] = {"rows": 0, "heads": 0, "nonfinite_heads": 0, "reference_heads": 0, "finite_reference_heads": 0,
                           "mismatch": 0, "mismatch_finite": 0, "twin_mismatch": 0, "binding_mismatch": 0, "first_mismatch": None,
                           "row_kernel_rows": 0, "row_kernel_declined": 0, "row_kernel_mismatch": 0}
    sampled = 0
    tok = 0
    for seq, (sq, sk) in enumerate(zip(c["seqlens_q"], c["seqlens_k"])):
        keys = np.arange(sk)
        kseq, vseq = k[bt[seq, keys // PAGE], keys % PAGE], v[bt[seq, keys // PAGE], keys % PAGE]
        for r in range(sq):
            T = r + 1 + sk - sq
            kk = np.ascontiguousarray(np.transpose(kseq[:T], (1, 0, 2))[kvh])
            vv = np.ascontiguousarray(np.transpose(vseq[:T], (1, 0, 2))[kvh])
            got = o[tok + r]
            gfrom = C.guarded_from(sq_max, sq, sk, r, BN)
            st = softcap_head_states(q[tok + r], kk, vv, BN, cap=cap, masked_from=gfrom, dot16=dot16, inv=inv, tanh=tanh)
            want = np.asarray(st["out"], dtype=np.uint16)
            nonfinite = np.asarray(st["nonfinite"], dtype=bool)
            head = bind(F2S.AttentionHeadSoftcapV2, T=T, D=D, BN=BN, CAP=cap, DOT=hopper.HopperBF16WgmmaDot16, INV=P.Fa2InvSum,
                        MASKED_FROM=gfrom)
            spec = TG.attention_spec(tp, T, H, Hk, D, sq_max, row=r, own_rows=sq, cap=cap)
            try:
                rk = RK.attention_softcap({"T": T, "NH": H, "KVH": Hk, "D": D, "BN": BN, "CAP": cap, "MASKED_FROM": gfrom,
                                           "DOT": hopper.HopperBF16WgmmaDot16.id, "INV": P.Fa2InvSum.id},
                                          [q[tok + r].reshape(-1), kseq[:T].reshape(-1), vseq[:T].reshape(-1)])["out"]
                rec["row_kernel_rows"] += 1
                rec["row_kernel_mismatch"] += int(not np.array_equal(rk, got.reshape(-1)))
            except SR.Declined:
                rec["row_kernel_declined"] += 1
            rec["binding_mismatch"] += spec.id != bind(F2S.AttentionSoftcapV2, T=T, NH=H, KVH=Hk, D=D, BN=BN, CAP=cap,
                                                       DOT=hopper.HopperBF16WgmmaDot16, INV=P.Fa2InvSum, MASKED_FROM=gfrom).id
            ref_heads = list(np.flatnonzero(nonfinite))
            if sampled < FINITE_SAMPLE and T > BN and not nonfinite.all():
                ref_heads.append(int(np.flatnonzero(~nonfinite)[-1]))
                sampled += 1
                rec["finite_reference_heads"] += 1
            for h in ref_heads:
                y = [int(x) for x in evaluate(head, q[tok + r, h].tolist(), kk[h].reshape(-1).tolist(), vv[h].reshape(-1).tolist())]
                rec["twin_mismatch"] += not nonfinite[h] and y != [int(x) for x in want[h]]     # the twin is exact on finite heads only
                want[h] = y
                rec["reference_heads"] += 1
            bad = np.flatnonzero((want != got).any(axis=1))
            rec["rows"] += 1
            rec["heads"] += H
            rec["nonfinite_heads"] += int(nonfinite.sum())
            rec["mismatch"] += int(bad.size)
            rec["mismatch_finite"] += int(np.count_nonzero(~nonfinite[bad]))
            if bad.size and rec["first_mismatch"] is None:
                h = int(bad[0])
                rec["first_mismatch"] = {"token": tok + r, "head": h, "T": T, "guarded_from": gfrom, "nonfinite": bool(nonfinite[h]),
                                         "want": [int(x) for x in want[h][:4]], "kernel": [int(x) for x in got[h][:4]]}
        tok += sq
    return {"case": c["name"], "D": D, **rec}


def gpu_case(op, c: dict[str, Any], D: int, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    t = G._inputs(c, D, seed)
    if c.get("wide"):
        C._widen(t, seed + 1)
    args = G._vllm_args(2, c, t)
    out1, _ = G._call(op, args)
    out2, _ = G._call(op, args)
    launch = {"case": c["name"], "D": D, "H": c["H"], "Hk": c["Hk"], "seqlens_q": c["seqlens_q"], "seqlens_k": c["seqlens_k"],
              "softcap": c["softcap"], "window": c["window"], "deterministic": G._same(out1, out2)}
    job = {"case": {k: c[k] for k in ("name", "H", "Hk", "seqlens_q", "seqlens_k", "softcap")}, "D": D, "q": G._words(t["q"]),
           "k": G._words(t["k"]), "v": G._words(t["v"]), "out": G._words(out1), "block_table": t["block_table"].cpu().numpy()}
    return launch, job


# ------------------------------------------------------------------------------------------------------------------------- run
def _verdict(doc: dict[str, Any]) -> bool:
    launches, rows = doc["launches"], doc["cases"]
    return (doc["gate"]["ok"] and doc["mufu"].get("ok") is True and doc["tanh"]["table"].get("ok") is True
            and doc["tanh"]["probe"].get("ok") is True
            and bool(launches) and all(x["ok"] and x["deterministic"] for x in launches)
            and bool(rows) and all(x.get("mismatch") == 0 and x.get("twin_mismatch") == 0 and x.get("binding_mismatch") == 0
                                   and x.get("row_kernel_mismatch") == 0 for x in rows)
            and sum(x["nonfinite_heads"] for x in rows) > 0 and sum(x["finite_reference_heads"] for x in rows) > 0)


def run(out: str, min_driver: int, want_cc: str, want_sms: int, workers: int, only: list[str] | None) -> dict[str, Any]:
    import torch

    so = C.fa2_library()
    if so:
        torch.ops.load_library(so)
    outdir = Path(out)
    outdir.mkdir(parents=True, exist_ok=True)
    doc: dict[str, Any] = {"schema": SCHEMA, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "argv": sys.argv,
                           "environment": G._environment(), "gate": C.gate(min_driver, want_cc, want_sms)}
    print(f"[softcap-capture] gate {json.dumps(doc['gate'])}", flush=True)
    if not doc["gate"]["ok"]:
        (outdir / RECORD).write_text(json.dumps(seal_record(doc, False), indent=1) + "\n")
        print("FA2-SOFTCAP-CAPTURE GATE-FAIL", flush=True)
        raise SystemExit(3)
    p = torch.cuda.get_device_properties(0)
    sm = f"sm_{p.major}{p.minor}"
    doc["binary"] = C.binary()
    with tempfile.TemporaryDirectory(dir=outdir) as work:
        doc["mufu"] = C.mufu(Path(work), sm)
        for b in doc["mufu"]["builds"].values():
            if b.get("tables_kept"):
                kept = outdir / Path(b["tables_kept"]).name
                os.replace(b["tables_kept"], kept)
                b["tables_kept"] = str(kept)
        print(f"[softcap-capture] mufu ok={doc['mufu']['ok']}", flush=True)
        words, lo, trec = tanh_table()
        ref = Path(work) / "tanh_sm89_table.bin"
        words.tofile(ref)
        del words
        doc["tanh"] = {"table": trec, "probe": tanh_probe(Path(work), outdir, ref, lo, int(trec["hi"], 16), sm)}
        print(f"[softcap-capture] tanh {json.dumps(doc['tanh'])[:900]}", flush=True)
        from torch.profiler import ProfilerActivity, profile, record_function

        op = torch.ops._vllm_fa2_C.varlen_fwd
        doc["launches"], jobs = [], []
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for i, c in enumerate(case_list()):
                if only and c["name"] not in only:
                    continue
                for D in c["hdims"]:
                    with record_function(f"{C.LABEL}{len(jobs)}"):
                        launch, job = gpu_case(op, c, D, 1000 * D + i)
                        job.update(cc=[p.major, p.minor], sms=int(p.multi_processor_count), tanh_table=str(ref), tanh_lo=lo)
                        torch.cuda.synchronize()
                    doc["launches"].append(launch)
                    jobs.append(job)
        names = C.kernels_by_launch(prof)
        for i, launch in enumerate(doc["launches"]):
            launch.update(C.launch_geometry(names.get(i, []), launch["D"]))
            print(f"[softcap-capture] launch hd{launch['D']} {launch['case']:30s} traits={launch['traits']} ok={launch['ok']} "
                  f"det={launch['deterministic']}", flush=True)
        doc["cases"] = []
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as ex:
            for rec in ex.map(compare, jobs):
                doc["cases"].append(rec)
                print(f"[softcap-capture] rows hd{rec['D']} {rec['case']:30s} rows={rec['rows']} heads={rec['heads']} "
                      f"nonfinite={rec['nonfinite_heads']} ref={rec['reference_heads']} !={rec['mismatch']}(finite {rec['mismatch_finite']}) "
                      f"twin!={rec['twin_mismatch']} bind!={rec['binding_mismatch']} row={rec['row_kernel_rows']} "
                      f"row!={rec['row_kernel_mismatch']} declined={rec['row_kernel_declined']}", flush=True)
    doc["totals"] = {k: sum(x[k] for x in doc["cases"]) for k in ("rows", "heads", "nonfinite_heads", "reference_heads",
                                                                  "finite_reference_heads", "mismatch", "mismatch_finite",
                                                                  "twin_mismatch", "binding_mismatch", "row_kernel_rows",
                                                                  "row_kernel_declined", "row_kernel_mismatch")}
    doc = seal_record(doc, _verdict(doc))
    (outdir / RECORD).write_text(json.dumps(doc, indent=1) + "\n")
    print(f"FA2-SOFTCAP-CAPTURE {'OK' if doc['ok'] else 'FAIL'} {doc['digest']} totals={json.dumps(doc['totals'])}", flush=True)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="FA2's softcapped kernel on this GPU against AttentionSoftcap_v2")
    ap.add_argument("--out", required=True, help=f"directory for {RECORD}")
    ap.add_argument("--min-driver", type=int, default=575)
    ap.add_argument("--want-cc", default="12.0")
    ap.add_argument("--want-sms", type=int, default=188)
    ap.add_argument("--workers", type=int, default=C.cpu_workers())
    ap.add_argument("--only", default=None, help="comma list of case names")
    a = ap.parse_args(argv)
    doc = run(a.out, a.min_driver, a.want_cc, a.want_sms, a.workers, a.only.split(",") if a.only else None)
    return 0 if doc["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
