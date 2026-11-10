"""Synthesize the blocks the always-on logging design adds beside FT (gov.v) with SI-ALG's flow (yosys + ABC on
generic.lib, NAND2 = 1 GE, registers at a DFF a bit), and check both netlists of each block bit for bit against an
exact Python reference.

    python3 gov_synth.py --alg DIR_OF_SILICON_ALG --out OUT [--fold DIR_OF_SILICON_FOLD] [--vectors N]

DIR_OF_SILICON_ALG is SI-ALG's benchmarks/pouw/silicon_alg (branch cursor/compute-research-silicon-alg-8e03), whose
synth.py runs unchanged. With --fold (SI-FOLD's benchmarks/pouw/theory/silicon_fold), it also maps FT's Keccak round
from SI-FOLD's rtl.py and the plane-limited unit (gov_ft.v) built from four of them, checked against SI-FOLD's rho.py.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
M48 = (1 << 48) - 1
NC = 14


def cnt_gen(rng: random.Random, n: int) -> dict:
    def ctr() -> int:
        return sum(((M48 - rng.getrandbits(6)) if rng.random() < 0.2 else rng.getrandbits(48)) << (48 * k)
                   for k in range(NC))
    return {"v": [rng.getrandbits(1) for _ in range(n)], "cls": [rng.getrandbits(4) for _ in range(n)],
            "am": [rng.choice([0, (1 << 32) - 1, rng.getrandbits(32)]) for _ in range(n)],
            "we": [rng.getrandbits(1) for _ in range(n)], "ab": [rng.getrandbits(1) for _ in range(n)],
            "c": [ctr() for _ in range(n)]}


def cnt_ref(v: dict, i: int) -> dict:
    c = v["c"][i]
    out = 0
    for k in range(NC):
        x = (c >> (48 * k)) & M48
        if k < 12 and v["v"][i] and v["cls"][i] == k:
            x += bin(v["am"][i]).count("1")
        if k == 12:
            x += v["we"][i]
        if k == 13:
            x += v["ab"][i]
        out |= (x & M48) << (48 * k)
    return {"y": out}


def gf64(a: int, b: int) -> int:
    p = 0
    for i in range(64):
        if (b >> i) & 1:
            p ^= a << i
    for j in range(126, 63, -1):
        if (p >> j) & 1:
            p ^= ((1 << 64) | 0x1B) << (j - 64)
    return p & ((1 << 64) - 1)


def gf_gen(rng: random.Random, n: int) -> dict:
    edge = [0, 1, (1 << 64) - 1, 1 << 63]
    return {"a": [rng.choice(edge) if rng.random() < 0.05 else rng.getrandbits(64) for _ in range(n)],
            "b": [rng.choice(edge) if rng.random() < 0.05 else rng.getrandbits(64) for _ in range(n)]}


M64 = (1 << 64) - 1


def kc_gen(rng: random.Random, n: int) -> dict:
    return {"st": [rng.getrandbits(1600) for _ in range(n)], "m": [rng.getrandbits(1024) for _ in range(n)],
            "rc": [rng.getrandbits(256) for _ in range(n)]}


def kc_ref(rho, plane: bool):
    """FT's round (16 lanes absorbed, one round) or the plane-limited unit (4 x 4 lanes of plane 0, four rounds)."""
    def ref(v: dict, i: int) -> dict:
        lanes = [(v["st"][i] >> (64 * j)) & M64 for j in range(25)]
        steps = [(k, 4) for k in range(4)] if plane else [(0, 16)]
        for k, nl in steps:
            blk = v["m"][i] >> (256 * k) if plane else v["m"][i]
            for j in range(nl):
                lanes[j] ^= (blk >> (64 * j)) & M64
            lanes = rho.keccak_round(lanes, (v["rc"][i] >> (64 * k)) & M64 if plane else v["rc"][i] & M64, 64)
        return {"y": sum(x << (64 * j) for j, x in enumerate(lanes))}
    return ref


DESIGNS = {
    "cnt_step": dict(gen=cnt_gen, ref=cnt_ref, regs=NC * 48,
                     role="one SMSP: 12 issue-class lane-op counters + RF-write and absorb counters, 48-bit"),
    "gf64_mul": dict(gen=gf_gen, ref=lambda v, i: {"y": gf64(v["a"][i], v["b"][i])}, regs=0,
                     role="GF(2^64) multiply, the step of an order-independent polynomial hash"),
}


def fold_designs(fold: Path, work: Path) -> tuple[dict, list[Path]]:
    sys.path.insert(0, str(fold))
    import rho
    import rtl

    v = work / "keccak1600_rnd.v"
    v.write_text(rtl.verilog("keccak1600_rnd"))
    return {
        "keccak1600_rnd": dict(gen=kc_gen, ref=kc_ref(rho, False), regs=1600,
                               role="FT's round: 32 words (16 lanes) a round, a round a clock (SI-FOLD's unit)"),
        "ft_plane4": dict(gen=kc_gen, ref=kc_ref(rho, True), regs=1600,
                          role="plane-limited keyed absorb: 4 rounds a clock, 8 words into plane 0 a round"),
    }, [v, HERE / "gov_ft.v"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alg", required=True, help="SI-ALG's benchmarks/pouw/silicon_alg directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fold", help="SI-FOLD's benchmarks/pouw/theory/silicon_fold: adds FT's round and ft_plane4")
    ap.add_argument("--vectors", type=int, default=4096)
    a = ap.parse_args()
    sys.path.insert(0, a.alg)
    import synth  # SI-ALG's flow, unchanged

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    synth.RTL = [HERE / "gov.v"]
    designs = dict(DESIGNS)
    if a.fold:
        extra, files = fold_designs(Path(a.fold), out)
        designs.update(extra)
        synth.RTL += files
    rows = {}
    for top, spec in designs.items():
        with tempfile.TemporaryDirectory() as td:
            r = synth.run_yosys(top, Path(td))
        nets = r.pop("nets")
        rng = random.Random(f"{top}:0")
        vec = spec["gen"](rng, a.vectors)
        want = [spec["ref"](vec, i) for i in range(a.vectors)]
        chk = {}
        for kind, net in nets.items():
            got = synth.simulate(net, top, vec, a.vectors)
            bad = [i for i in range(a.vectors) for k in want[i] if got[k][i] != want[i][k]]
            chk[kind] = {"vectors": a.vectors, "mismatches": len(bad), "first": bad[:1]}
        r["check"] = chk
        r["ge_reg"] = spec["regs"] * synth.DFF_GE
        r["ge"] = r["ge_comb"] + r["ge_reg"]
        r["role"] = spec["role"]
        rows[top] = r
        ok = all(c["mismatches"] == 0 for c in chk.values())
        print(f"{top:10s} comb {r['ge_comb']:9.1f} reg {r['ge_reg']:8.1f} total {r['ge']:9.1f} GE  "
              f"ANDs {r['ands']:6d} depth {r['depth']:3d}  check {'ok' if ok else 'FAIL'}", flush=True)
    ver = subprocess.run([synth.yosys(), "-V"], capture_output=True, text=True).stdout.strip()
    (out / "gov_synth.json").write_text(json.dumps({"yosys": ver, "dff_ge": synth.DFF_GE, "rows": rows}, indent=1,
                                                   default=str))
    bad = [t for t, r in rows.items() if any(c["mismatches"] for c in r["check"].values())]
    if bad:
        print("CHECK FAILED:", bad)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
