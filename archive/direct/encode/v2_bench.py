"""Measure every B-Ligero prover bucket at the checker-v2 witness shape (b-v2 lane; ``V2_SHAPE.md``, ``v2_shape.json``).

    python3 -m encode.v2_bench --reps 7 --out v2_bench.json [--quick] [--buckets commit,irs,quadratic,quadratic_fc,lintest,inverse,openings]

Shape (B = 4096, K = 1536, 2^-128, BabyBear / BabyBear^6, HVZK rows, transposed layout, table side as helper rows):
  rows 104,217 x [16384, 4096]; 3,502 F_t product rows; 9,785 F_c helper-triple rows (6 coordinate rows + the query's
  C witness rows each, grouped by table); linear test M = 2 fixed rows + 9,780 exceptional (table-side) rows; 43.6M helpers
  to invert; t = 192 openings.  Variant (b) (public table coefficients): 94,439 rows, 1,632 exceptional rows, 37.4M helpers.
Synthetic witness: uniform BabyBear values (the kernels are shape-driven and data-independent; values < p are 'in range' for
every kernel here -- the constraint satisfaction is not what is timed).  Every kernel is checked bit-exactly against its
numpy / Python reference on a small shape first (``checks``); the b-commit / b-lintest kernels are reused unchanged.
CUDA events, warm, median of ``--reps``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "verity" / "ml" / "kernels"))

import cupy as cp

from encode import babybear as bb
from encode import buckets as bk
from encode import buckets_bench as bbench
from encode import buckets_v2 as bv2
from encode import commit_bench as cbench
from encode import lintest_structured as lt
from encode.commit_stream import StreamingCommitter
from encode.encode_ntt_simt import RSEncoderSIMT

HERE = Path(__file__).resolve().parent
K, D, L = bb.K_CODE, 6, 3840
PEAK_BW = 1.008e12


def load_shape(variant: str = "helper_rows") -> dict:
    return json.loads((HERE / "v2_shape.json").read_text())[variant]


def rand_u32(shape, seed: int, lo: int = 0):
    return cp.random.RandomState(seed).randint(lo, bb.P, size=shape, dtype=cp.int32).view(cp.uint32)


# -- checks ------------------------------------------------------------------------------------------------------------
def check_quadratic_fc(enc: RSEncoderSIMT, cols: int = 4, rows: int = 24) -> bool:
    rng = np.random.default_rng(21)
    H = np.stack([bb.fill_rows(rows, seed=30 + c) for c in range(D)])              # (6, rows, k)
    W = np.stack([bb.fill_rows(rows, seed=40 + q) for q in range(cols)])           # (C, rows, k)
    beta = [int(x) for x in rng.integers(0, bb.P, D)]
    z = rng.integers(0, bb.P, D).astype(np.uint64)
    r = rng.integers(0, bb.P, (rows, D)).astype(np.uint64)
    bc = bv2.beta_coords(beta, cols)
    want = bv2.quadratic_fc_ref(r, H, W, bc, z)
    q = bv2.QuadraticFc(cols, chunk_rows=8)
    Hd, Wd = cp.asarray(H.astype(np.uint32)), cp.asarray(W.astype(np.uint32))
    H1 = cp.empty_like(Hd)
    W1 = cp.empty_like(Wd)
    for c in range(D):
        enc(Hd[c], H1[c], write_systematic=False, ncosets=1)
    for c in range(cols):
        enc(Wd[c], W1[c], write_systematic=False, ncosets=1)
    one1 = enc(cp.asarray(bv2.one_row()), cp.empty((1, K), dtype=cp.uint32), write_systematic=False, ncosets=1)[0]
    acc = q.acc(rows)
    q(Hd, H1, Wd, W1, cp.asarray(bk.mont(bc)), cp.asarray(bk.mont(z)), one1, cp.asarray(np.concatenate([bk.mont(r), bk.mont2(r)], axis=1)), acc)
    got = cp.asnumpy(q.reduce(acc)).astype(np.uint64)
    return bool(np.array_equal(got, want))


# -- the new bucket -------------------------------------------------------------------------------------------------------
def bench_quadratic_fc(shape: dict, reps: int, enc: RSEncoderSIMT, slice_rows: int = 256, from_store: bool = False) -> list[dict]:
    """Helper triples grouped by table (``cols`` witness rows each).  ``from_store``: coset-1 rows read from a resident
    store instead of re-encoded (the fused commitment keeps cosets 1-3 for the openings anyway)."""
    blocks = shape["blocks"]
    res = []
    stream = cp.cuda.Stream(non_blocking=True)
    total_ms, total_rows, total_enc = 0.0, 0, 0
    one1 = enc(cp.asarray(bv2.one_row()), cp.empty((1, K), dtype=cp.uint32), write_systematic=False, ncosets=1)[0]
    per_table = []
    for name, tb in sorted(shape["tables"].items(), key=lambda kv: -kv[1]["queries_per_unit"]):
        cols, n = tb["columns"], tb["queries_per_unit"] * blocks
        q = bv2.QuadraticFc(cols, chunk_rows=32)
        Hs = rand_u32((D, n, K), 50)
        Ws = rand_u32((cols, n, K), 51)
        H1s = rand_u32((D, n, K), 52) if from_store else None
        W1s = rand_u32((cols, n, K), 53) if from_store else None
        r12 = rand_u32((n, 2 * D), 54)
        beta = rand_u32((cols, D), 55)
        z = rand_u32((D,), 56)
        acc = q.acc(slice_rows)
        out = cp.empty((D, 2 * K), dtype=cp.uint32)
        H1 = cp.empty((D, slice_rows, K), dtype=cp.uint32)
        W1 = cp.empty((cols, slice_rows, K), dtype=cp.uint32)

        def loop(q=q, acc=acc, out=out, n=n, cols=cols, Hs=Hs, Ws=Ws, H1s=H1s, W1s=W1s, H1=H1, W1=W1, r12=r12, beta=beta, z=z):
            acc.fill(0)
            for r0 in range(0, n, slice_rows):
                s = min(slice_rows, n - r0)
                if from_store:
                    h1, w1 = H1s[:, r0:r0 + s], W1s[:, r0:r0 + s]
                else:
                    for c in range(D):
                        enc(Hs[c, r0:r0 + s], H1[c, :s], write_systematic=False, ncosets=1)
                    for c in range(cols):
                        enc(Ws[c, r0:r0 + s], W1[c, :s], write_systematic=False, ncosets=1)
                    h1, w1 = H1[:, :s], W1[:, :s]
                q(Hs[:, r0:r0 + s], h1, Ws[:, r0:r0 + s], w1, beta, z, one1, r12[r0:r0 + s], acc)
            q.reduce(acc, out)
        with stream:
            g = bbench.graph_of(loop, stream)
            med, ts = bbench.timed(lambda g=g: g.launch(stream=stream), reps)
        per_table.append({"table": name, "cols": cols, "rows": n, "ms_median": med, "ms_all": ts, "us_per_row": med * 1e3 / n,
                          "encoder_calls_per_row": 0 if from_store else D + cols})
        total_ms += med
        total_rows += n
        total_enc += 0 if from_store else (D + cols) * n
        print(f"v2 quadratic_fc {name} (C={cols}, {n} rows): {med:.2f} ms ({med * 1e3 / n:.3f} us/row)", flush=True)
        del Hs, Ws, H1s, W1s, acc, H1, W1
        bbench.free_all()
    mm = D * 2 * K * total_rows * (6 * shape["quadratic"]["fc_triple_witness_reads_per_unit"] / shape["per_unit"]["lookups"] + 36 + 36 + 6)
    res.append({"name": "quadratic_test_fc" + ("_from_store" if from_store else "_reencode"), "bucket": "quadratic_test_fc", "shape": "v2",
                "rows": total_rows, "slice_rows": slice_rows, "ms_median": total_ms, "ms_all": [total_ms], "us_per_row": total_ms * 1e3 / total_rows,
                "transforms_per_batch": total_enc, "coset1_source": "store" if from_store else "re-encoded", "per_table": per_table,
                "mont_muls_est": mm, "note": "sum of the per-table timings (each its own CUDA graph); y = z_T - sum beta^c w_c and z = 1 virtual",
                "device_memory_in_use_bytes": bbench.mem()})
    print(f"v2 quadratic_test_fc total: {total_ms:.2f} ms over {total_rows} F_c triple rows ({'store' if from_store else 're-encode'})", flush=True)
    return res


def bench_commit(R: int, reps: int, quick: bool) -> list[dict]:
    res = []
    rows = cbench.random_rows(R, seed=1)
    for keep in (True, False):
        sc = StreamingCommitter(hash="blake3", slice_rows=512, overlap=False, keep_codeword=keep)
        ok = cbench.check(sc)
        med, ts = cbench.timed(lambda sc=sc, rows=rows: sc.commit(rows), reps)
        r = {"name": f"blake3_s512_ov0_keep{int(keep)}_fused", "hash": "blake3", "slice_rows": 512, "overlap": False, "keep": keep, "mode": "fused",
             "bucket": "commitment", "shape": "v2", "rows": R, "ms_median": med, "ms_all": ts, "us_per_row": med * 1e3 / R, "bit_exact": ok,
             "bytes_in": R * K * 4, "codeword_bytes": R * bb.N_CODE * 4, "read_floor_ms": R * K * 4 / PEAK_BW * 1e3,
             "codeword_gb_per_s": R * bb.N_CODE * 4 / (med / 1e3) / 1e9, "store_bytes": R * 3 * K * 4 if keep else 0,
             "device_memory_in_use_bytes": bbench.mem()}
        res.append(r)
        print(f"v2 commitment keep={keep}: {med:.2f} ms ({r['us_per_row']:.3f} us/row), bit-exact {ok}", flush=True)
        del sc
        bbench.free_all()
    del rows
    bbench.free_all()
    return res


def bench_lintest(R: int, M: int, X: int, reps: int, enc: RSEncoderSIMT, lazy: bool = True) -> dict:
    W = rand_u32((R, K), 7)
    a = types.SimpleNamespace(chunk_rows=1024, reps=reps)
    stream = cp.cuda.Stream(non_blocking=True)
    r = lt._bench_structured(enc, W, R, M, X, a, stream, bbench, lazy=lazy)
    r["shape"] = "v2"
    del W
    bbench.free_all()
    return r


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--out", default="v2_bench.json")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--buckets", default="commit,irs,quadratic,quadratic_fc,lintest,inverse,openings")
    a = ap.parse_args(argv)
    sh = load_shape("helper_rows")
    shb = load_shape("public_coefficients")
    R, Q = sh["rows"]["total"], sh["quadratic"]["ft_triple_rows"]
    NINV, NINV_B = sh["helpers_to_invert"], shb["helpers_to_invert"]
    X, XB, M = sh["lintest"]["exceptional_rows"], shb["lintest"]["exceptional_rows"], sh["lintest"]["M"]
    RB = shb["rows"]["total"]
    if a.quick:
        R, RB, Q, NINV, NINV_B, X, XB = 12_000, 11_000, 400, 4_000_000, 3_500_000, 600, 100
        sh = {**sh, "blocks": 4}
        a.reps = min(a.reps, 3)
    res = {"device": bbench.device_info(), "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "args": vars(a),
           "shape": {"rows": R, "rows_variant_b": RB, "quad_ft_rows": Q, "helpers": NINV, "helpers_variant_b": NINV_B, "M": M, "X": X, "X_variant_b": XB,
                     "fc_triple_rows": sh["quadratic"]["fc_triple_rows"] if not a.quick else None, "t_open": bbench.T_OPEN},
           "results": [], "bench_kind": "bucket", "checker_variant": "v2",
           "scope": "component: every B-Ligero prover bucket at the checker-v2 witness shape (V2_SHAPE.md), measured standalone on a "
                    "synthetic witness of that shape (uniform BabyBear values); r^T A scalars / pattern generation, witness and hint "
                    "generation and the transcript are not included",
           "bit_exact_note": "b-commit / b-lintest kernels re-checked against buckets_ref / the general path (checks); the new F_c-triple "
                             "quadratic kernel against quadratic_fc_ref (numpy)"}
    print(json.dumps(res["device"]), flush=True)
    enc = RSEncoderSIMT()
    res["checks"] = bbench.check_all(enc)
    res["checks"]["quadratic_fc_C4"] = check_quadratic_fc(enc, cols=4)
    res["checks"]["quadratic_fc_C1"] = check_quadratic_fc(enc, cols=1, rows=9)
    res["checks"].update({f"lintest_{k}": v for k, v in lt.checks(enc).items()})
    print("checks:", json.dumps(res["checks"]), flush=True)
    want = a.buckets.split(",")
    if "commit" in want:
        res["results"] += bench_commit(R, a.reps, a.quick)
    if "irs" in want:
        res["results"] += [{**r, "shape": "v2"} for r in bbench.bench_irs("v2", R, a.reps, chunk_rows_list=(1024,))]
    if "quadratic" in want:
        res["results"] += [{**r, "shape": "v2"} for r in bbench.bench_quadratic("v2", Q, a.reps, enc, slice_list=(512,))]
    if "quadratic_fc" in want:
        res["results"] += bench_quadratic_fc(sh, a.reps, enc, from_store=False)
        res["results"] += bench_quadratic_fc(sh, a.reps, enc, from_store=True)
    if "lintest" in want:
        res["results"].append(bench_lintest(R, M, X, a.reps, enc))
        res["results"].append({**bench_lintest(RB, M, XB, a.reps, enc), "variant": "public_coefficients"})
    if "inverse" in want:
        res["results"] += [{**r, "shape": "v2"} for r in bbench.bench_inverse("v2", NINV, a.reps, threads_list=(131072, 262144))]
        res["results"] += [{**r, "shape": "v2", "variant": "public_coefficients"} for r in bbench.bench_inverse("v2", NINV_B, a.reps, threads_list=(262144,))]
    if "openings" in want:
        res["results"] += [{**r, "shape": "v2"} for r in bbench.bench_openings("v2", R, a.reps, enc)]
    res["all_bit_exact"] = all(v is True for k, v in res["checks"].items() if not (k.endswith("toy_exceptional_rows") or "int8_tc" in k))
    res["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    res["run_id"] = os.environ.get("RESEARCH_RUN_ID")
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    rd = os.environ.get("RESEARCH_RUN_DIR")
    if rd:
        Path(rd, "v2_bench.json").write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main()
