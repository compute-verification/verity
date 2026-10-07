#!/usr/bin/env python3
"""Pearl-C4's noise line under TurboSHAKE128 and SHA-512 on one card: the gate, then the interleaved timing of the call.

  python line_bench.py OUT_JSON --arm NAME=CUBIN ... [--expect-uuid GPU-...] [--shapes 8192x8192x8192,64x8192x8192]
                       [--rounds 31] [--item-ms 40]

Arms are pearl_c4.cu built at LINE_ARM 0 (``ts128``: TurboSHAKE128(seed ‖ role ‖ index u32 LE, 0x24), 16 bytes), 1
(``sha512-1``: the first 16 bytes of H_"pearl-c4/v2/line"(seed ‖ role ‖ index u32 LE), one compression per line) and 2
(``sha512-4``: H_"pearl-c4/v2/line"(seed ‖ role ‖ index/4 u32 LE), line index's 16-byte quarter index mod 4, one
compression per four lines); H_ℓ(x) = SHA-512(ℓ ‖ \\0 zero-padded to 128 bytes ‖ x).  Per shape, every arm gets the same
operands, salt and domain ids (dev.py's self-timing labels), and:
  1. the gate: F_A's and F_B's lines (the lines kernel), E_A's lines as the hash call's forming writes them, each byte-equal
     to the host's rule for its arm over the seeds the device derived; and the hash call's codes, scales and P_A equal to
     the nohash call's (which forms from the lines kernel's E_A).  Nothing is timed unless every arm passes;
  2. the timing: `--rounds` rounds, each round every arm once in a rotated order, an item being a CUDA graph of the arm's
     launches sized to at least `--item-ms`, the SM clock read after each item.  Items: ``call`` (dev.py's hash call:
     A's commitment and seed, the forming with E_A's lines from seed_a, A′F_B, the chain with the peel and digests, the
     leaves), ``form`` (that forming launch alone), ``form_nohash`` (the forming reading primed lines) and ``gemm``.
Per arm and item: the median ms per launch, its 10th and 90th percentiles, and the median clock; per SHA-512 arm, its
median call and form against ts128's, in µs and as a share of ts128's call.
"""

from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import importlib.util
import json
import statistics
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dev as D  # noqa: E402

_ts = importlib.util.spec_from_file_location("turboshake", HERE.parents[2] / "verity/ml/crypto/turboshake.py")
TS = importlib.util.module_from_spec(_ts)
_ts.loader.exec_module(TS)

LINE_BLOCK = (b"pearl-c4/v2/line" + b"\0").ljust(128, b"\0")
ITEMS = ("call", "form", "form_nohash", "gemm")
GRAPH = {"cuStreamCreate": [C.POINTER(C.c_void_p), C.c_uint], "cuStreamBeginCapture_v2": [C.c_void_p, C.c_int],
         "cuStreamEndCapture": [C.c_void_p, C.POINTER(C.c_void_p)],
         "cuGraphInstantiateWithFlags": [C.POINTER(C.c_void_p), C.c_void_p, C.c_uint64],
         "cuGraphLaunch": [C.c_void_p, C.c_void_p], "cuGraphExecDestroy": [C.c_void_p], "cuGraphDestroy": [C.c_void_p],
         "cuStreamSynchronize": [C.c_void_p]}


def host_line(arm: str, seed: bytes, role: int, index: int) -> bytes:
    if arm == "ts128":
        return TS.turboshake128(seed + bytes([role]) + index.to_bytes(4, "little"), 0x24, 16)
    if arm == "sha512-1":
        return hashlib.sha512(LINE_BLOCK + seed + bytes([role]) + index.to_bytes(4, "little")).digest()[:16]
    q = index % 4
    return hashlib.sha512(LINE_BLOCK + seed + bytes([role]) + (index // 4).to_bytes(4, "little")).digest()[16 * q:16 * q + 16]


def sm_clock(uuid: str) -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits", "-i", uuid],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return int(out)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


class Graphs:
    """CUDA graphs of a launch sequence repeated, on one stream, timed by events on that stream."""

    def __init__(self, d: D.Dev):
        self.d = d
        for f, a in GRAPH.items():
            getattr(d.cu, f).argtypes, getattr(d.cu, f).restype = a, C.c_int
        self.stream = C.c_void_p()
        d.ck("cuStreamCreate", C.byref(self.stream), 1)

    def build(self, steps, reps: int):
        d, s = self.d, self.stream
        d.ck("cuStreamBeginCapture_v2", s, 2)
        for _ in range(reps):
            for name, geo, kw in steps:
                d.launch(name, *geo, stream=s.value, **kw)
        g = C.c_void_p()
        d.ck("cuStreamEndCapture", s, C.byref(g))
        ex = C.c_void_p()
        d.ck("cuGraphInstantiateWithFlags", C.byref(ex), g, 0)
        return g, ex

    def time(self, ex) -> float:
        d, s = self.d, self.stream
        a, b, ms = C.c_void_p(), C.c_void_p(), C.c_float()
        d.ck("cuEventCreate", C.byref(a), 0)
        d.ck("cuEventCreate", C.byref(b), 0)
        d.ck("cuEventRecord", a, s)
        d.ck("cuGraphLaunch", ex, s)
        d.ck("cuEventRecord", b, s)
        d.ck("cuEventSynchronize", b)
        d.ck("cuEventElapsedTime", C.byref(ms), a, b)
        d.cu.cuEventDestroy_v2(a)
        d.cu.cuEventDestroy_v2(b)
        return ms.value

    def free(self, g, ex) -> None:
        self.d.ck("cuGraphExecDestroy", ex)
        self.d.ck("cuGraphDestroy", g)


def operands(m: int, n: int, k: int) -> dict[str, bytes]:
    import numpy as np
    rng = np.random.default_rng(20260930)
    out = {}
    for name, rows in (("xa", m), ("xb", n)):
        x = rng.standard_normal((rows, k)).astype(np.float32).view(np.uint32) >> 16
        out[name] = x.astype("<u2").tobytes()
    for name in ("salt", "dom_a", "dom_b"):
        out[name] = hashlib.sha256(f"verity/pouw/pearl-c4-dev/{name}".encode()).digest()
    return out


def gate(arm: str, d: D.Dev, pl: D.Pipeline) -> dict:
    b, m, k = pl.buf, pl.m, pl.k
    pl.launch(mode="hash")
    d.sync()
    hashed = {o: d.read(b[o], pl.sizes[o]) for o in ("a_codes", "a_scales", "pa", "lines_ea")}
    seed_a, seed_b = d.read(b["seed_a"], 32), d.read(b["seed_b"], 32)
    pl.prime()
    pl.launch(mode="nohash")
    d.sync()
    out = {}
    for name, seed, role, count in (("lines_fa", seed_b, D.ROLE_FA, k), ("lines_fb", seed_b, D.ROLE_FB, k),
                                    ("lines_ea", seed_a, D.ROLE_EA, m)):
        got = hashed["lines_ea"] if name == "lines_ea" else d.read(b[name], 16 * count)
        want = b"".join(host_line(arm, seed, role, i) for i in range(count))
        out[name] = got == want
        if name == "lines_ea":
            out["lines_ea_primed"] = d.read(b["lines_ea"], 16 * count) == want
    for o in ("a_codes", "a_scales", "pa"):
        out[f"{o}_hash_eq_nohash"] = d.read(b[o], pl.sizes[o]) == hashed[o]
    out["ok"] = all(out.values())
    return out


def shape_run(arms: dict[str, tuple[D.Dev, Graphs]], m: int, n: int, k: int, args) -> dict:
    ops = operands(m, n, k)
    pls, res = {}, {"shape": [m, n, k], "gate": {}}
    for arm, (d, _) in arms.items():
        pl = D.Pipeline(d, "nvfp4", m, n, k)
        for name, data in ops.items():
            d.write(pl.buf[name], data)
        pl.weight_side()
        pl.prime()
        d.sync()
        pls[arm] = pl
        res["gate"][arm] = gate(arm, d, pl)
    res["gate_ok"] = all(g["ok"] for g in res["gate"].values())
    any_pl = next(iter(pls.values()))
    res["geometry"] = {"bm": any_pl.bm, "bn": any_pl.bn, "split_a": any_pl.split_a}
    if not res["gate_ok"]:
        return res
    graphs = {}
    for arm, (d, gr) in arms.items():
        pl = pls[arm]
        hash_steps, nohash_steps = pl.steps("hash"), pl.steps("nohash")
        seqs = {"call": hash_steps, "form": [s for s in hash_steps if s[0] == pl.form],
                "form_nohash": [s for s in nohash_steps if s[0] == pl.form],
                "gemm": [s for s in hash_steps if s[0].startswith("gemm_")]}
        for item, steps in seqs.items():
            g1, e1 = gr.build(steps, 1)
            gr.time(e1)
            once = min(gr.time(e1) for _ in range(3))
            gr.free(g1, e1)
            reps = max(1, round(args.item_ms / max(once, 1e-3)))
            graphs[arm, item] = (reps, *gr.build(steps, reps))
            gr.time(graphs[arm, item][2])
    names = list(arms)
    samples = {(a, i): [] for a in names for i in ITEMS}
    for r in range(args.rounds):
        order = names[r % len(names):] + names[:r % len(names)]
        for item in ITEMS:
            for arm in order:
                reps, _, ex = graphs[arm, item]
                ms = arms[arm][1].time(ex) / reps
                samples[arm, item].append({"ms": ms, "sm_mhz": sm_clock(arms[arm][0].uuid)})
    res["timing"] = {}
    for arm in names:
        res["timing"][arm] = {}
        for item in ITEMS:
            xs = sorted(s["ms"] for s in samples[arm, item])
            clocks = [s["sm_mhz"] for s in samples[arm, item] if s["sm_mhz"]]
            res["timing"][arm][item] = {"launches_per_item": graphs[arm, item][0], "ms_median": statistics.median(xs),
                                        "ms_p10": xs[len(xs) // 10], "ms_p90": xs[len(xs) * 9 // 10],
                                        "sm_mhz_median": statistics.median(clocks) if clocks else None,
                                        "reps": samples[arm, item]}
    base = res["timing"][names[0]]
    call0 = base["call"]["ms_median"]
    res["vs_" + names[0]] = {arm: {item: {"us": 1e3 * (res["timing"][arm][item]["ms_median"] - base[item]["ms_median"]),
                                          "pct_of_call": 100 * (res["timing"][arm][item]["ms_median"]
                                                                - base[item]["ms_median"]) / call0}
                                   for item in ITEMS} for arm in names[1:]}
    for arm, (d, gr) in arms.items():
        for item in ITEMS:
            _, g, ex = graphs[arm, item]
            gr.free(g, ex)
        pls[arm].free()
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--arm", action="append", required=True, help="NAME=CUBIN, ts128 first (the baseline)")
    ap.add_argument("--expect-uuid", default=None)
    ap.add_argument("--shapes", default="8192x8192x8192,64x8192x8192")
    ap.add_argument("--rounds", type=int, default=31)
    ap.add_argument("--item-ms", type=float, default=40.0)
    args = ap.parse_args(argv)
    arms = {}
    for spec in args.arm:
        name, path = spec.split("=", 1)
        d = D.Dev(Path(path).read_bytes(), args.expect_uuid)
        arms[name] = (d, Graphs(d))
    any_dev = next(iter(arms.values()))[0]
    out = {"device": any_dev.identity, "arms": {s.split("=", 1)[0]: hashlib.sha256(Path(s.split("=", 1)[1]).read_bytes())
                                                .hexdigest() for s in args.arm}, "shapes": []}
    for s in args.shapes.split(","):
        m, n, k = (int(v) for v in s.split("x"))
        out["shapes"].append(shape_run(arms, m, n, k, args))
        Path(args.out).write_text(json.dumps(out, indent=1))
    out["ok"] = all(s["gate_ok"] for s in out["shapes"])
    Path(args.out).write_text(json.dumps(out, indent=1))
    for s in out["shapes"]:
        print(s["shape"], "gate", s["gate_ok"], json.dumps(s.get("vs_" + next(iter(arms)), {})))
        for arm, t in s.get("timing", {}).items():
            print(" ", arm, {i: round(1e3 * t[i]["ms_median"], 2) for i in ITEMS}, "us")
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
