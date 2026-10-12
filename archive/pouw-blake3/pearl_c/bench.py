#!/usr/bin/env python3
"""Pearl-C (`pearl-c-h100-v1`) on fixed matmul shapes on one H100 SXM, PoUW alone (no served model): the prover's cost
against the plain FP8 GEMM with the adopted α, hashing reported apart, and γ per shape.  Stdlib only, from the prebuilt
pearl_c.cubin through run.py's driver.

1. The gate check at check.json's shape: run.py's `check`, the $0.30 run's bit-exact gates.
2. Per shape of bench.json (twin.py's): A's real rows and B generated on the device (`gen_f32`; filler rows stay zero),
   every kernel once, then each sampled 64 x 64 tile compared with the CPU twin: C̃, U, the leaf, Pearl's tickets, α on
   both sides, the A row's leaf in A's tree and the adopted baseline's y.  A shape is timed only when the check and its tiles
   pass (the baseline when its own buffers do); PEARLC_UNGATED=1 times regardless and says so.
3. The timing: run.py's `measure`, then per shape
   - `prover_x`: the unit without hashing (A's forming, A′·F_B, P_A, the product with its peel) over the adopted baseline
     (A's statistics, e4m3(αx), the FP8 GEMM; B's quantization and B̃, B̃·F_A, P_B are per job);
   - `with_hashing_x`: the unit with its hashing over the same, and `hashing_ms` by itself;
   - `per_call_b_x`: with B's forming per call as well (POUS on), over the baseline quantizing B per call;
   - γ from (n, k) by the cheap-binding lane's accounting (`gamma`).
   The kernels round m up to 128; `m_real` rows are the matmul's, and only they are credited.

One JSON object per line on stdout, a table on stderr.  Exit 3: not an H100 SXM; 4: a check or a sampled tile failed.

  python3 bench.py [bench.json [check.json]]      (env: REPS=5, ONLY="label ...", PEARLC_UNGATED, PEARLC_LIBCUDA)
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from fractions import Fraction

import run as R

HERE = os.path.dirname(os.path.abspath(__file__))
T = 64                        # the leaf's tile
SUBTILE = 16                  # Pearl's ticket subtile

# γ's constants (research store docs/pouw/cheap-binding.md §1; pearl_c.py's W_ref, H100 W1 units per MAC)
GAMMA0 = RHO_CAP = Fraction(1, 400)
FORMING_CREDITED, FORMING_A_ONLY = 64, 4      # the noise atom and the cast; ρ
RANK, PEEL_UNITS = 32, 2 * 64                 # A′·F_B's fused columns; the peel, 2 units per BF16 MAC over depth 64
FADD_PER_MAC = Fraction(32, 128)              # one 32-unit FP32 add per 128 products
WINDOW = 128                                  # products per promotion add


def gamma(n: int, k: int) -> dict[str, Fraction]:
    """γ of one unit of width n and depth k, per element of A, under rev1 (`DeviceRev1.lean`): the credit is the chain
    n(1 + 1/4) less the first promotion add of each word (it adds into +0, so 32n/k), A′·F_B's 32 columns, the peel
    128n/k and the credited forming; W_ref adds ρ.  `cap` is the worst case under the debit cap ρ = 1/400, `floor` a
    zero debit, and `*_chain_only` the reading where TT_OUT credits no forming.  It reads neither m nor B's per-job work."""
    credit = n + n * FADD_PER_MAC * (1 - Fraction(WINDOW, k)) + RANK + Fraction(PEEL_UNITS * n, k) + FORMING_CREDITED
    wref = credit + FORMING_A_ONLY
    return {"cap": 1 - (1 - GAMMA0) * (1 - RHO_CAP) * credit / wref,
            "cap_chain_only": 1 - (1 - GAMMA0) * (credit - FORMING_CREDITED - RHO_CAP * credit) / wref,
            "floor": 1 - (1 - GAMMA0) * credit / wref,
            "floor_chain_only": 1 - (1 - GAMMA0) * (credit - FORMING_CREDITED) / wref}


def tile_bytes(read, n: int, k: int, r0: int, c0: int) -> dict[str, bytes]:
    """The sampled tile's bytes as the kernels lay them out: read(buffer, offset, bytes)."""
    rows = range(r0, r0 + T)
    return {"leaf": read("leaves", 32 * ((r0 // T) * (n // T) + c0 // T), 32),
            "c": b"".join(read("c", 4 * (i * n + c0), 4 * T) for i in rows),
            "u": b"".join(read("u", 4 * (i * n + c0), 4 * T) for i in rows),
            "tickets": b"".join(read("tickets", 32 * ((r0 // SUBTILE + s) * (n // SUBTILE) + c0 // SUBTILE), 32 * T // SUBTILE)
                                for s in range(T // SUBTILE)),
            "y_adopt": b"".join(read("y_adopt", 2 * (i * n + c0), 2 * T) for i in rows),
            "alpha_a": read("alpha_a", 4 * r0, 4 * T),
            "alpha_b": read("alpha_b", 4 * c0, 4 * T),
            "row_leaf": read("tree_a", 32 * r0, 32)}


def compare_tile(got: dict[str, bytes], want: dict) -> list[str]:
    """The names of the buffers whose bytes differ from the twin's."""
    return sorted(name for name, raw in got.items()
                  if (raw.hex() if name == "leaf" else hashlib.sha256(raw).hexdigest()) != want[name])


BASELINE = {"y_adopt"}


def shape_run(dev: R.Dev, nv: R.Nvml, spec: dict, reps: int, check: dict, ungated: bool) -> dict:
    m, m_real, n, k = spec["m"], spec["m_real"], spec["n"], spec["k"]
    keys = {name: dev.put(bytes.fromhex(h)) for name, h in spec["keys"].items()}
    xa, xb = dev.alloc(m * 4 * k), dev.alloc(n * 4 * k)
    for x, rows, seed in ((xa, m_real, spec["gen"]["a"]), (xb, n, spec["gen"]["b"])):
        dev.launch("gen_f32", R.cdiv(rows * k, 256), 256, p=(x,), i=(rows, k, seed))
    pipe = R.Pipeline(dev, m, n, k, "v1", keys, xa, xb, unit=None)      # E_A by bench.json's line key: the twin samples tiles
    pipe.run_all()
    read = lambda name, off, size: dev.read(pipe.b[name] + off, size)  # noqa: E731
    tiles = [{"tile": s["tile"], "failed": compare_tile(tile_bytes(read, n, k, *s["tile"]), s)} for s in spec["samples"]]
    unit_ok = all(not set(t["failed"]) - BASELINE for t in tiles)
    base_ok = all(not set(t["failed"]) & BASELINE for t in tiles)
    gates = {"pearl_c": check["pearl_c_ok"] and unit_ok or ungated, "plain": check["plain_ok"] and base_ok or ungated,
             "ungated": ungated}
    out = R.measure(dev, nv, pipe, reps, gates)
    row = {"kind": "bench", "label": spec["label"], "m_real": m_real, "shape": [m, n, k], "tiles": tiles,
           "tiles_ok": unit_ok and base_ok, "gamma": {kk: str(v) for kk, v in gamma(n, k).items()},
           "gamma_pct": {kk: round(float(v) * 100, 3) for kk, v in gamma(n, k).items()}, "timing": out}
    if "ratio_vs_adopted" in out:
        adopted = out["adopted_unit_ms"]
        row["summary"] = {"adopted_ms": adopted, "prover_ms": out["unit_e2e_no_hash_ms"], "with_hashing_ms": out["unit_e2e_ms"],
                          "hashing_ms": round(out["split_ms"]["commit"] + out["split_ms"]["hashing"], 4), "job_ms": out["job_ms"],
                          "prover_x": out["ratio_vs_adopted"]["no_hash_e2e"],
                          "with_hashing_x": out["ratio_vs_adopted"]["unit_e2e"],
                          "hashing_x": round((out["split_ms"]["commit"] + out["split_ms"]["hashing"]) / adopted, 4),
                          "per_call_b_x": out["ratio_vs_adopted"]["with_job"],
                          "chain_only_x": out["ratio_vs_adopted"]["chain_only"],
                          "adopted_tflops_real": round(2 * m_real * n * k / adopted / 1e9, 1)}
    pipe.free()
    for p in (xa, xb, *keys.values()):
        dev.free(p)
    return row


def line(row: dict) -> str:
    g = row["gamma_pct"]
    head = f"{row['label']:24s} {row['m_real']}x{row['shape'][1]}x{row['shape'][2]}"
    tiles = "tiles ok" if row["tiles_ok"] else "tiles FAILED " + "; ".join(f"{t['tile']}: {t['failed']}" for t in row["tiles"] if t["failed"])
    if "summary" not in row:
        return f"{head}: not timed; {tiles}; γ {g['cap']:.3f}% / {g['cap_chain_only']:.3f}%"
    s = row["summary"]
    return (f"{head}: prover {s['prover_x']:.3f}x, with hashing {s['with_hashing_x']:.3f}x (hashing {s['hashing_ms']:.3f} ms = "
            f"{s['hashing_x']:.3f}x), B per call {s['per_call_b_x']:.3f}x; adopted FP8 {s['adopted_ms']:.3f} ms "
            f"({s['adopted_tflops_real']} TFLOPS on real rows); γ {g['cap']:.3f}% / {g['cap_chain_only']:.3f}% chain-only; {tiles}")


def main(argv: list[str]) -> int:
    bench_path = argv[1] if len(argv) > 1 else os.path.join(HERE, "bench.json")
    check_path = argv[2] if len(argv) > 2 else os.path.join(HERE, "check", "check.json")
    bench, man = json.load(open(bench_path)), json.load(open(check_path))
    image = open(os.path.join(HERE, "pearl_c.cubin"), "rb").read()
    nv = R.Nvml()
    dev = R.Dev(image)
    R.emit({"kind": "env", "device": dev.name, "sms": dev.sms, "cuda_driver_api": dev.driver, "nvml_driver": nv.driver,
            "cubin_sha256": hashlib.sha256(image).hexdigest(),
            "bench_json_sha256": hashlib.sha256(open(bench_path, "rb").read()).hexdigest(),
            "check_json_sha256": hashlib.sha256(open(check_path, "rb").read()).hexdigest(),
            "mem_free_total": dev.free_mem(), "clk": nv.read(), "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    if dev.name != R.H100 and not os.environ.get("PEARLC_ANY_GPU"):
        print(f"REFUSED: GPU is {dev.name!r}, not an H100 SXM ({R.H100})", file=sys.stderr)
        return 3
    t0 = time.time()
    res = R.check(dev, man, os.path.dirname(check_path))
    R.emit({"kind": "check", "shape": man["shape"], "seconds": round(time.time() - t0, 2), "formings": res})
    print(f"check at {man['shape']}: v1 {res['v1']['buffers']} buffers, failed {res['v1']['failed'] or 'none'}", file=sys.stderr)
    ungated, reps = bool(os.environ.get("PEARLC_UNGATED")), int(os.environ.get("REPS", "5"))
    only = set(os.environ.get("ONLY", "").split())
    ok = res["v1"]["pearl_c_ok"] and res["v1"]["plain_ok"]
    for spec in bench["shapes"]:
        if only and spec["label"] not in only:
            continue
        t0 = time.time()
        row = shape_run(dev, nv, spec, reps, res["v1"], ungated)
        row["seconds"] = round(time.time() - t0, 2)
        R.emit(row)
        print(line(row), file=sys.stderr)
        ok = ok and row["tiles_ok"]
    return 0 if ok else 4


if __name__ == "__main__":
    sys.exit(main(sys.argv))
