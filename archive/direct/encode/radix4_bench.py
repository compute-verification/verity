"""Radix-4 shared-memory stages in the SIMT encoder (b-v2, task 4): bit-exactness against ``babybear.encode_rows`` and timing of
the encoder alone (cosets-only store) and of the fused commitment (``StreamingCommitter``) with the radix-2 vs radix-4 encoder,
at the v2 (104,217) and v1 (310,602) row counts.

    python3 encode/radix4_bench.py --reps 7 --out "$RESEARCH_RUN_DIR/result.json"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "verity" / "ml" / "kernels"))

import cupy as cp  # noqa: E402

from encode import babybear as bb  # noqa: E402
from encode import buckets_bench as bbench  # noqa: E402
from encode import commit_bench as cbench  # noqa: E402
from encode.commit_stream import StreamingCommitter  # noqa: E402
from encode.encode_ntt_simt import RSEncoderSIMT  # noqa: E402

K = bb.K_CODE
PEAK_BW = 1.008e12


def exactness() -> dict:
    out = {}
    rows = bb.fill_rows(40, seed=5)
    ref = bb.encode_rows(rows, bb.N_CODE, layout="coset")
    for threads in (256, 512, 1024):
        for radix4 in (False, True):
            enc = RSEncoderSIMT(threads=threads, radix4=radix4)
            got = cp.asnumpy(enc(cp.asarray(rows))).astype(np.uint64)
            out[f"t{threads}_radix{4 if radix4 else 2}"] = bool(np.array_equal(got, ref))
            store = cp.empty((40, 3 * K), dtype=cp.uint32)
            enc(cp.asarray(rows), store, write_systematic=False)
            out[f"t{threads}_radix{4 if radix4 else 2}_store"] = bool(np.array_equal(cp.asnumpy(store).astype(np.uint64), ref[:, K:]))
            g = RSEncoderSIMT(threads=threads, radix4=radix4, gather=True)
            cols = np.sort(np.random.default_rng(1).choice(bb.N_CODE, 192, replace=False))
            out[f"t{threads}_radix{4 if radix4 else 2}_gather"] = bool(np.array_equal(cp.asnumpy(g.gather(cp.asarray(rows), cols)).astype(np.uint64), ref[:, cols].T))
    return out


def bench_encoder(R: int, reps: int) -> list[dict]:
    res = []
    rows = bbench.rand_u32((R, K), 3)
    store = cp.empty((R, 3 * K), dtype=cp.uint32)
    for threads, minb in ((512, 1), (1024, 2)):
        for radix4 in (False, True):
            sms = cp.cuda.runtime.getDeviceProperties(0)["multiProcessorCount"]
            enc = RSEncoderSIMT(threads=threads, radix4=radix4, min_blocks_per_sm=minb, blocks=sms * (16 if threads == 512 else 1))
            med, ts = bbench.timed(lambda: enc(rows, store, write_systematic=False), reps)
            res.append({"name": f"encoder_t{threads}_mb{minb}_radix{4 if radix4 else 2}", "bucket": "encoder", "rows": R, "threads": threads,
                        "min_blocks_per_sm": minb, "radix4": radix4, "ms_median": med, "ms_all": ts, "us_per_row": med * 1e3 / R,
                        "butterflies_per_s": RSEncoderSIMT.butterflies_per_row(K) * R / (med / 1e3)})
            print(f"R={R} {res[-1]['name']}: {med:.2f} ms ({res[-1]['us_per_row']:.3f} us/row)", flush=True)
            del enc
    del rows, store
    bbench.free_all()
    return res


def bench_commit(R: int, reps: int) -> list[dict]:
    res = []
    rows = cbench.random_rows(R, seed=1)
    sms = cp.cuda.runtime.getDeviceProperties(0)["multiProcessorCount"]
    for radix4 in (False, True):
        for keep in (True, False):
            enc = RSEncoderSIMT(threads=1024, min_blocks_per_sm=2, blocks=sms, radix4=radix4)
            sc = StreamingCommitter(hash="blake3", slice_rows=512, overlap=False, keep_codeword=keep, encoder=enc)
            ok = cbench.check(sc)
            med, ts = cbench.timed(lambda sc=sc, rows=rows: sc.commit(rows), reps)
            res.append({"name": f"blake3_s512_ov0_keep{int(keep)}_fused_radix{4 if radix4 else 2}", "bucket": "commitment", "keep": keep, "radix4": radix4,
                        "rows": R, "ms_median": med, "ms_all": ts, "us_per_row": med * 1e3 / R, "bit_exact": ok,
                        "read_floor_ms": R * K * 4 / PEAK_BW * 1e3, "device_memory_in_use_bytes": bbench.mem()})
            print(f"R={R} {res[-1]['name']}: {med:.2f} ms ({res[-1]['us_per_row']:.3f} us/row), bit-exact {ok}", flush=True)
            del sc, enc
            bbench.free_all()
    del rows
    bbench.free_all()
    return res


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--out", default="radix4_bench.json")
    ap.add_argument("--rows", default="104217,310602")
    a = ap.parse_args(argv)
    result = {"exact": exactness(), "device": cbench.device_info() if hasattr(cbench, "device_info") else {}}
    print(result["exact"], flush=True)
    assert all(result["exact"].values()), result["exact"]
    result["results"] = []
    for R in (int(x) for x in a.rows.split(",")):
        result["results"] += bench_encoder(R, a.reps)
        result["results"] += bench_commit(R, a.reps)
    Path(a.out).write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
