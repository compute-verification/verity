"""Synthesize SI-NOISE's blocks with yosys + ABC and price them in gate equivalents (GE) and in AND gates.

Each block in `BLOCKS` is a top module of `rtl/*.v` (with parameters). Two mappings a block:

- **area:** `abc -g AND,NAND,OR,NOR,XOR,XNOR,ANDNOT,ORNOT,MUX` on the flattened netlist, cells weighted by `GE` (NAND2 = 1;
  a flip-flop 4.5), the generic library this lane uses until SI-ALG's baseline lands;
- **proof:** `abc -g AND,XOR` (XOR free in C-Flock's Boolean circuits), whose AND count is an upper bound on the block's
  multiplicative complexity, so the C-Flock gates a block costs per evaluation.

`per` divides a block's figures to the unit it is quoted in (a noise bit, a MAC, an accumulator step). Output: one JSON
with the raw cell counts and the derived GE / ANDs per unit. Usage: `python3 synth.py OUT.json [--only=a,b] [--jobs=N]`,
with a native `yosys` on PATH (yowasp-yosys's ABC pass doesn't run).
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
RTL = sorted((HERE / "rtl").glob("*.v"))
YOSYS = shutil.which("yosys") or "yosys"

GE = {"$_NAND_": 1.0, "$_NOR_": 1.0, "$_AND_": 1.33, "$_OR_": 1.33, "$_ANDNOT_": 1.33, "$_ORNOT_": 1.33,
      "$_XOR_": 2.33, "$_XNOR_": 2.33, "$_MUX_": 2.33, "$_NOT_": 0.67, "$_BUF_": 0.0}
DFF_GE = 4.5
AREA_GATES = "AND,NAND,OR,NOR,XOR,XNOR,ANDNOT,ORNOT,MUX"

# name: (top, {param: value}, unit, units a cycle, what)
BLOCKS = {
    "mac_s8_dp4": ("mac_s8_dp4", {}, "MAC", 4, "s8 x s8 -> s32 4-wide dot-product step, registered accumulator"),
    "mac_fp8_dp4": ("mac_fp8_dp4", {}, "MAC", 4, "FP8 E4M3 4-wide dot product into an FP32 accumulator, one rounding"),
    "trivium_x64": ("trivium_x64", {}, "bit", 64, "Trivium, 64 updates a cycle (288-bit state)"),
    "chacha_round": ("chacha_round", {}, "round-bit", 512, "one pipelined ChaCha round (4 QRs, 512 state FFs)"),
    "chacha_ff": ("chacha_ff", {}, "bit", 512, "ChaCha's 16 feed-forward adds"),
    "sha512_round": ("sha512_round", {}, "round-bit", 512, "one pipelined SHA-512 round + schedule step (1,536 FFs)"),
    "keccak_round": ("keccak_round", {}, "round-bit", 1088, "one pipelined Keccak-f[1600] round, per rate bit"),
    "walk_b1_w16": ("noise_walk", {"B": 1, "W": 16}, "acc-step", 1, "hidden walk, 1-bit noise, 16-bit register"),
    "walk_b8_w16": ("noise_walk", {"B": 8, "W": 16}, "acc-step", 1, "hidden walk, 8-bit noise, 16-bit register"),
    "walk_b32_w32": ("noise_walk", {"B": 32, "W": 32}, "acc-step", 1, "hidden walk, 32-bit noise, 32-bit register"),
    "sk_xorrot": ("sk_xorrot", {}, "acc-step", 1, "s <= rotl(s,1) ^ (acc + z)"),
    "sk_xorrot_x": ("sk_xorrot_x", {}, "acc-step", 1, "s <= rotl(s,1) ^ acc ^ z (no adder)"),
    "sk_crc32": ("sk_crc32", {}, "acc-step", 1, "CRC-32 of acc + z"),
    "sk_gf32": ("sk_gf32", {}, "acc-step", 1, "GF(2^32) Horner, s <= s*r ^ (acc + z)"),
    "sk_nh": ("sk_nh", {}, "pair-step", 1, "NH, s <= s + (acc0+z0)(acc1+z1) mod 2^32"),
    "sk_lin": ("sk_lin", {}, "acc-step", 1, "linear, s <= s + r(acc + z) mod 2^32"),
}


def run_yosys(top: str, params: dict, gates: str) -> dict:
    reads = "; ".join(f"read_verilog -sv {p}" for p in RTL)
    chp = "".join(f"chparam -set {k} {v} {top}; " for k, v in params.items())
    script = (f"{reads}; {chp}hierarchy -top {top}; synth -flatten -top {top}; abc -g {gates}; opt_clean; stat")
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "y.log"
        subprocess.run([YOSYS, "-q", "-l", str(log), "-p", script], check=True, capture_output=True)
        text = log.read_text()
    tail = text[text.rfind("Number of cells:"):]
    cells = {m.group(2): int(m.group(1)) for m in re.finditer(r"^\s+(\d+)\s+(\$_\w+_)\s*$", tail, re.M)}
    if not cells:
        cells = {m.group(1): int(m.group(2)) for m in re.finditer(r"^\s+(\$_\w+_)\s+(\d+)\s*$", tail, re.M)}
    if not cells:
        raise SystemExit(f"{top}: no cells in {YOSYS}'s stat. yowasp-yosys's ABC doesn't run (WASI has no subprocess); "
                         "use a native yosys (apt install yosys)")
    return cells


def price(cells: dict) -> tuple[float, int, float]:
    ge = dff = 0.0
    for c, n in cells.items():
        if "DFF" in c or "DLATCH" in c or "SDFF" in c:
            dff += n
            ge += DFF_GE * n
        else:
            if c not in GE:
                raise SystemExit(f"unpriced cell {c}")
            ge += GE[c] * n
    return ge, int(dff), ge - DFF_GE * dff


def one(name: str) -> dict:
    top, params, unit, per, what = BLOCKS[name]
    area = run_yosys(top, params, AREA_GATES)
    proof = run_yosys(top, params, "AND,XOR")
    ge, dff, comb = price(area)
    ands = proof.get("$_AND_", 0)
    return {"top": top, "params": params, "what": what, "unit": unit, "per_cycle": per, "cells": area,
            "ge": round(ge, 1), "dff": dff, "ge_comb": round(comb, 1), "ands": ands,
            "ge_per_unit": round(ge / per, 3), "ands_per_unit": round(ands / per, 3)}


def main() -> int:
    from concurrent.futures import ProcessPoolExecutor

    out = Path(sys.argv[1])
    only, jobs = None, 4
    for a in sys.argv[2:]:
        if a.startswith("--only="):
            only = set(a.split("=", 1)[1].split(","))
        if a.startswith("--jobs="):
            jobs = int(a.split("=", 1)[1])
    names = [n for n in BLOCKS if not only or n in only]
    res = {"library": {"GE": GE, "DFF": DFF_GE, "area_gates": AREA_GATES, "proof_gates": "AND,XOR"},
           "yosys": subprocess.run([YOSYS, "-V"], capture_output=True, text=True).stdout.strip(), "blocks": {}}
    with ProcessPoolExecutor(jobs) as ex:
        for name, b in zip(names, ex.map(one, names)):
            res["blocks"][name] = b
            print(f"{name:14s} {b['ge']:10.1f} GE ({b['dff']:5d} FF)  {b['ands']:7d} AND   {b['ge_per_unit']:9.2f} "
                  f"GE/{b['unit']}  {b['ands_per_unit']:8.2f} AND/{b['unit']}   {b['what']}", flush=True)
    out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
