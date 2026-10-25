"""stage.py FORM K N INST OUT: INST token rows of ``ScaledMmFp8Block_v2{K,N,128}`` (one x row, N weight columns per
instance) staged for C-Flock as `verity_flock.multi_unit`'s statements, from synthetic values drawn as ``served_units``'
``fp8-coord`` draws them (x = Fp8GroupQuant of a seeded N(0, 1) bf16 row; each weight column N(0, 0.02) quantized per 128, a
weight block's scale its first column's).

    FORM  c       one statement: N * K/128 k-step units and a scale stage of N slots per VU (OUT/c)
          b       two statements joined by a row: the writer's k-step units return temp and s (OUT/b-writer), the reader's
                  units are the scale stage (OUT/b-reader), its input row the writer's output row

Every instance's outputs are checked against the served kernel's numpy twin (`derived_rows.scaled_mm_fp8_block_row`) and, when
the Definition is at most 2^20 ANDs a row (``--reference``, or K * N <= 4096), against the whole Definition's IR reference
evaluated on the same leaves. Writes OUT/<form>/stage.json: each slot type's ANDs and nonzeros, k_log, VUs per block, seconds."""
from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

from verity_flock import multi_unit as MU
from verity.ml.definitions.numerics import derived_rows as DR

G = 128
SALT_KEY = b"multi-unit-benchmark"


def bf16(v: np.ndarray) -> np.ndarray:
    u = v.astype(np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def words(*parts: np.ndarray) -> np.ndarray:
    raw = b"".join(np.ascontiguousarray(p).astype(p.dtype.newbyteorder("<")).tobytes() for p in parts)
    return np.frombuffer(raw, dtype="<u2").copy()


def draw(K: int, N: int, n: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    xs, ws, twin = [], [], []
    for _ in range(n):
        xq, sx = DR.fp8_group_quant_row(bf16(rng.standard_normal(K)), G=G)
        cols = [DR.fp8_group_quant_row(bf16(rng.standard_normal(K) * 0.02), G=G) for _ in range(N)]
        wq = np.stack([q for q, _ in cols])
        sw = np.stack([cols[nb * G][1] for nb in range(-(-N // G))])
        out, nonfin = DR.scaled_mm_fp8_block_row(xq, sx, wq, sw, G=G)
        assert not nonfin.any(), "a synthetic coordinate is non-finite"
        xs.append(words(xq, sx))
        ws.append(words(wq, sw))
        twin.append(out)
    return {"x": np.stack(xs), "w": np.stack(ws)}, np.stack(twin)


def definition(K: int, N: int):
    from verity.ml.circuits.defs import bind
    from verity.ml.definitions.registry import boolean_fp8 as BF

    return bind(BF.ScaledMmFp8BlockV2, K=K, N=N, G=G)


def reference(K: int, N: int, rows: dict) -> np.ndarray:
    """The whole Definition's IR reference on each instance's leaves (the rows' bits, low first)."""
    from verity.ml.circuits.evaluation import evaluate

    d = definition(K, N)
    out = []
    for x, w in zip(rows["x"], rows["w"]):
        bx = np.unpackbits(np.asarray(x, dtype="<u2").view(np.uint8), bitorder="little").tolist()
        bw = np.unpackbits(np.asarray(w, dtype="<u2").view(np.uint8), bitorder="little").tolist()
        got = list(evaluate(d, bx[:8 * K], bx[8 * K:], bw[:8 * K * N], bw[8 * K * N:]))
        out.append(np.packbits(np.asarray(got, dtype=np.uint8), bitorder="little").view("<u2"))
    return np.stack(out)


def main(argv: list[str]) -> int:
    form, K, N, n, out = argv[0], int(argv[1]), int(argv[2]), int(argv[3]), Path(argv[4])
    t0 = time.monotonic()
    rows, twin = draw(K, N, n)
    t_draw = time.monotonic() - t0
    ref = None
    if "--reference" in argv or K * N <= 4096:
        ref = reference(K, N, rows)
        if not np.array_equal(ref, twin):
            raise AssertionError("the Definition's IR reference is not the served twin")
    t_ref = time.monotonic() - t0 - t_draw
    res = []
    if form == "c":
        t = time.monotonic()
        split = MU.fp8_block(K, N, G, definition(K, N))
        t_build = time.monotonic() - t
        res.append((out / "c", MU.stage(split, rows, out / "c", SALT_KEY, twin), t_build))
    elif form == "b":
        t = time.monotonic()
        writer, reader = MU.fp8_block_linked(K, N, G)
        t_build = time.monotonic() - t
        st_w = MU.stage(writer, rows, out / "b-writer", SALT_KEY)
        st_w |= {"units": N * (K // G), "units_padded": 1 << (N * (K // G) - 1).bit_length()}
        c = np.load(out / "b-writer" / "outs.npy")
        st_r = MU.stage(reader, {"c": c}, out / "b-reader", SALT_KEY, twin)
        res += [(out / "b-writer", st_w, t_build), (out / "b-reader", st_r, 0.0)]
    else:
        raise SystemExit(f"FORM {form}: c or b")
    for d, st, tb in res:
        st |= {"form": form, "K": K, "N": N, "G": G, "twin": "agrees", "reference": "agrees" if ref is not None else None,
               "seconds": {"draw": round(t_draw, 1), "reference": round(t_ref, 1), "build": round(tb, 1),
                           "total": round(time.monotonic() - t0, 1)},
               "maxrss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)}
        (d / "stage.json").write_text(json.dumps(st, indent=1, sort_keys=True))
        print(json.dumps({k: st[k] for k in ("name", "n", "k_log", "vus_per_block", "per_vu", "counts", "seconds", "maxrss_gb")}))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
