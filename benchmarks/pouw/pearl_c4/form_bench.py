"""form_nv's block-step variants on one card: the gate, then the timing.

  python form_bench.py CUBIN OUT_JSON [--expect-uuid GPU-...] [--shapes 8192x8192,64x8192] [--reps 7] [--item-ms 40]

Variants (pearl_c4.cu, `form_body`'s V): form_nv_v0 (0, the integer-max amax and a 128-entry table: the forming shipped
until lut256), form_nv_v1 (the amax by FMNMX, the floor in FP32 and one packed e4m3x2 encode), form_nv (2, the shipped
one: v1 with lut256's reciprocal table, the path the block scale is priced on).  Per shape (rows x k, A's forming as the
call launches it, `Pipeline.steps`' geometry and E lines from seed_a):
  1. the gate: on four inputs (Gaussian rows, rows over 10^±3 with outliers, a zero row and near-zero rows, spikes of
     10^18) every variant's codes, scales, α, 1/α, a_E, P and E lines byte-equal to form_nv_v0's; nothing is timed unless
     every variant passes.  The variants agree where D is finite (FMNMX drops a NaN the integer max keeps): ρ² must
     not overflow, and a row whose a_E is NaN is outside D-SB anyway;
  2. the timing: `--reps` rounds, each variant once per round in turn, an item of launches sized to at least `--item-ms`,
     the SM clock read after each item; per variant the median ms per launch and its FP4 units per formed element
     (ms · f_SM · SMs · 2048 / (rows · k), 2048 dense FP4 MACs per SM per clock), and occupancy (CTAs per SM).
"""

from __future__ import annotations

import argparse
import ctypes as C
import json
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dev as D

REF = "form_nv_v0"
VARIANTS = (REF, "form_nv_v1", "form_nv")
OUTS = ("a_codes", "a_scales", "a_alpha", "a_inv", "a_ae", "pa", "lines_ea")
FP4_PER_SM_CLK = 2048


def bf16_rows(x: np.ndarray) -> bytes:
    return (x.astype(np.float32).view(np.uint32) >> 16).astype("<u2").tobytes()


def inputs(rows: int, k: int, seed: int) -> list[tuple[str, bytes]]:
    rng = np.random.default_rng(seed)
    wide = rng.standard_normal((rows, k)) * 10.0 ** rng.uniform(-3, 3, (rows, 1))
    wide[rng.random((rows, k)) < 0.01] *= 50
    small = rng.standard_normal((rows, k)) * 1e-30
    small[0] = 0
    spikes = rng.standard_normal((rows, k))
    spikes[:, ::97] = 1.0e18
    return [("gaussian", bf16_rows(rng.standard_normal((rows, k)))), ("wide", bf16_rows(wide)),
            ("small", bf16_rows(small)), ("spikes", bf16_rows(spikes))]


def sm_clock(uuid: str) -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits", "-i", uuid],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return int(out)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def occupancy(d: D.Dev, name: str, smem: int) -> int:
    f = d.cu.cuOccupancyMaxActiveBlocksPerMultiprocessor
    f.argtypes, f.restype = [C.POINTER(C.c_int), C.c_void_p, C.c_int, C.c_size_t], C.c_int
    n = C.c_int()
    d.ck("cuOccupancyMaxActiveBlocksPerMultiprocessor", C.byref(n), d.fn(name), 256, smem)
    return n.value


def shape_run(d: D.Dev, rows: int, k: int, args) -> dict:
    pl = D.Pipeline(d, "nvfp4", rows, 128, k)
    b = pl.buf
    d.write(b["seed_a"], bytes(range(32)))
    d.write(b["seed_b"], bytes(range(32, 64)))
    d.launch("lines", D.cdiv(k, 512), 128, p=[b["seed_b"], b["lines_fa"]], i=[D.ROLE_FA, k])
    _, geo, kw = next(s for s in pl.steps("hash") if s[0] == pl.form)
    smem = {v: pl.form_smem if v == pl.form else 0 for v in VARIANTS}
    bufs = {v: {o: b[o] if v == pl.form else d.alloc(pl.sizes[o]) for o in OUTS} for v in VARIANTS}

    def launch(v: str) -> None:
        o = bufs[v]
        p = [b["xa"], o["lines_ea"], b["lines_fa"], o["a_codes"], o["a_scales"], o["a_alpha"], o["a_inv"], o["a_ae"], o["pa"],
             b["seed_a"]]
        d.launch(v, geo[0], geo[1], smem=smem[v], p=p, i=kw["i"])

    gate = {}
    for label, data in inputs(rows, k, args.seed):
        d.write(b["xa"], data)
        for v in VARIANTS:
            launch(v)
        d.sync()
        ref = {o: d.read(bufs[REF][o], pl.sizes[o]) for o in OUTS}
        gate[label] = {v: {o: d.read(bufs[v][o], pl.sizes[o]) == ref[o] for o in OUTS} for v in VARIANTS[1:]}
    ok = all(all(r.values()) for g in gate.values() for r in g.values())
    res = {"rows": rows, "k": k, "grid": list(geo[0]), "gate": gate, "gate_ok": ok,
           "occupancy_ctas_per_sm": {v: occupancy(d, v, smem[v]) for v in VARIANTS}, "dyn_smem": smem}
    if ok:
        per = {v: max(1, round(args.item_ms / max(d.timed(lambda v=v: launch(v), 3)[-1], 1e-3))) for v in VARIANTS}
        reps = {v: [] for v in VARIANTS}
        for _ in range(args.reps):
            for v in VARIANTS:
                ms = d.timed(lambda v=v: [launch(v) for _ in range(per[v])], 1)[0] / per[v]
                reps[v].append({"ms": ms, "sm_mhz": sm_clock(d.uuid)})
        res["timing"] = {}
        for v in VARIANTS:
            ms = statistics.median(r["ms"] for r in reps[v])
            clocks = [r["sm_mhz"] for r in reps[v] if r["sm_mhz"]]
            mhz = statistics.median(clocks) if clocks else args.clock_mhz
            units = ms * 1e-3 * mhz * 1e6 * d.sms * FP4_PER_SM_CLK / (rows * k)
            res["timing"][v] = {"launches_per_item": per[v], "reps": reps[v], "ms_median": ms, "sm_mhz_median": mhz,
                                "fp4_units_per_element": units}
        base = res["timing"][REF]["fp4_units_per_element"]
        for v in VARIANTS[1:]:
            res["timing"][v][f"fp4_units_per_element_vs_{REF}"] = res["timing"][v]["fp4_units_per_element"] - base
    for v in VARIANTS:
        if v != pl.form:
            for o in OUTS:
                d.free(bufs[v][o])
    pl.free()
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cubin")
    ap.add_argument("out")
    ap.add_argument("--expect-uuid", default=None)
    ap.add_argument("--shapes", default="8192x8192,64x8192")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--item-ms", type=float, default=40.0)
    ap.add_argument("--clock-mhz", type=float, default=2100.0, help="used only when no clock could be read")
    ap.add_argument("--seed", type=int, default=20260930)
    args = ap.parse_args(argv)
    d = D.Dev(Path(args.cubin).read_bytes(), args.expect_uuid)
    out = {"device": d.identity, "shapes": []}
    for s in args.shapes.split(","):
        rows, k = (int(v) for v in s.split("x"))
        out["shapes"].append(shape_run(d, rows, k, args))
        Path(args.out).write_text(json.dumps(out, indent=1))
    ok = all(s["gate_ok"] for s in out["shapes"])
    out["ok"] = ok
    Path(args.out).write_text(json.dumps(out, indent=1))
    for s in out["shapes"]:
        t = s.get("timing", {})
        print(s["rows"], s["k"], "gate", s["gate_ok"], s["occupancy_ctas_per_sm"],
              {v: round(x["fp4_units_per_element"], 2) for v, x in t.items()})
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
