"""Synthesize every module in `rtl.py` with yosys + ABC to a generic gate library, and check each mapped netlist
against `rho.py` with yosys's `eval` on random vectors.

For each module it reports cells by type, gate equivalents (NAND2 = 1, Nangate45's ratios, `GE`), the longest path in
gates (`ltp`), and the AND count of a second mapping to {AND, XOR, NOT}: an upper bound on the multiplicative complexity
a Boolean proof pays (C-Flock's ANDs). Needs `yowasp-yosys` on PATH (`pip install yowasp-yosys`).

    python3 synth.py OUT.json [--vectors N] [--scratch DIR]
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import rho
import rtl

GATES = "AND,NAND,OR,NOR,XOR,XNOR,ANDNOT,ORNOT,MUX,AOI3,OAI3,AOI4,OAI4"
# Gate equivalents from Nangate45's cell areas over NAND2_X1's 0.798 um^2.
GE = {
    "$_NOT_": 0.67, "$_BUF_": 0.67, "$_NAND_": 1.0, "$_NOR_": 1.0, "$_AND_": 1.33, "$_OR_": 1.33,
    "$_ANDNOT_": 1.33, "$_ORNOT_": 1.33, "$_XOR_": 2.0, "$_XNOR_": 2.0, "$_MUX_": 2.33, "$_NMUX_": 2.33,
    "$_AOI3_": 1.33, "$_OAI3_": 1.33, "$_AOI4_": 1.67, "$_OAI4_": 1.67,
}
GE_DFF = 5.67  # DFF_X1 4.522 um^2


def _bits(rng: random.Random, n: int) -> int:
    return rng.getrandbits(n)


def _words(v: int, n: int, w: int) -> list[int]:
    return [(v >> (w * i)) & ((1 << w) - 1) for i in range(n)]


def _pack(ws: list[int], w: int) -> int:
    return sum(x << (w * i) for i, x in enumerate(ws))


def _sx8(v: int) -> int:
    return v - 256 if v & 0x80 else v


def _e4m3(v: int) -> float:
    s = -1.0 if v & 0x80 else 1.0
    e, m = (v >> 3) & 0xF, v & 7
    return s * (m / 8 * 2.0**-6 if e == 0 else (1 + m / 8) * 2.0 ** (e - 7))


def _fp8_inputs(rng: random.Random) -> dict[str, int]:
    while True:
        a, b = rng.getrandbits(8), rng.getrandbits(8)
        if (a & 0x7F) != 0x7F and (b & 0x7F) != 0x7F:
            break
    r = rng.random()
    if r < 0.1:
        acc = 0
    else:
        e = rng.randint(110, 145)
        acc = (rng.getrandbits(1) << 31) | (e << 23) | rng.getrandbits(23)
    return {"a": a, "b": b, "acc": acc}


def _fp8_ref(i: dict[str, int]) -> int | None:
    p = np.float32(_e4m3(i["a"]) * _e4m3(i["b"]))
    c = np.frombuffer(struct.pack("<I", i["acc"]), dtype=np.float32)[0]
    s = np.float32(c + p)
    out = struct.unpack("<I", s.tobytes())[0]
    if (out >> 23) & 0xFF == 0:  # flushed by the RTL
        return None if out & 0x7FFFFF else out
    return out


def _keccak_ref(b: int):
    w = b // 25
    r = rtl.KECCAK_RATE[b]

    def f(i: dict[str, int]) -> int:
        s = i["st"] ^ i["m"]
        return _pack(rho.keccak_round(_words(s, 25, w), i["rc"], w), w)

    return {"st": b, "m": r, "rc": w}, f


def _speck(i):
    x, v = rho.speck64_round(i["st"] & rho.M32, i["st"] >> 32, i["k"])
    return x | (v << 32)


def _simon(i):
    x, v = rho.simon64_round(i["st"] & rho.M32, i["st"] >> 32, i["k"])
    return x | (v << 32)


def _xoodoo(i):
    s = _words(i["st"] ^ i["m"], 12, 32)
    a = rho.xoodoo_round([s[0:4], s[4:8], s[8:12]], i["rc"])
    return _pack(a[0] + a[1] + a[2], 32)


def _dround(i):
    s = _words(i["st"] ^ i["m"], 16, 32)
    return _pack(rho.chacha_diagonal_round(rho.chacha_column_round(s)), 32)


REF = {
    "mac4_s8": ({"a": 32, "b": 32, "acc": 32}, lambda i: (i["acc"] + sum(_sx8((i["a"] >> 8 * k) & 0xFF) * _sx8((i["b"] >> 8 * k) & 0xFF) for k in range(4))) & rho.M32),
    "fp8_mac": ({"a": 8, "b": 8, "acc": 32}, _fp8_ref),
    "sha512_rnd": ({"st": 512, "w": 64, "k": 64}, lambda i: _pack(rho.sha512_round(_words(i["st"], 8, 64), i["w"], i["k"]), 64)),
    "sha512_krom": ({"t": 7}, lambda i: rho.SHA512_K[i["t"]] if i["t"] < 80 else 0),
    "keccak_rcrom": ({"t": 5}, lambda i: rho.KECCAK_RC64[i["t"]] if i["t"] < 24 else 0),
    "keccak1600_rnd": _keccak_ref(1600),
    "keccak800_rnd": _keccak_ref(800),
    "keccak400_rnd": _keccak_ref(400),
    "keccak200_rnd": _keccak_ref(200),
    "chacha_qr": ({"st": 128}, lambda i: _pack(list(rho.chacha_qr(*_words(i["st"], 4, 32))), 32)),
    "chacha_dround": ({"st": 512, "m": 512}, _dround),
    "aes_rnd": ({"st": 128, "k": 128}, lambda i: int.from_bytes(rho.aes_round(i["st"].to_bytes(16, "little"), i["k"].to_bytes(16, "little")), "little")),
    "ascon_rnd": ({"st": 320, "m": 320, "c": 8}, lambda i: _pack(rho.ascon_round(_words(i["st"] ^ i["m"], 5, 64), i["c"]), 64)),
    "xoodoo_rnd": ({"st": 384, "m": 384, "rc": 32}, _xoodoo),
    "speck64_rnd": ({"st": 64, "k": 32}, _speck),
    "simon64_rnd": ({"st": 64, "k": 32}, _simon),
    "nv_arx": ({"t": 32, "w": 32}, lambda i: rho.nv_arx(i["t"], i["w"])),
    "affine": ({"t": 32, "w": 32}, lambda i: rho.affine(i["t"], i["w"])),
}


def _yosys(script: str, cwd: Path) -> None:
    exe = shutil.which("yowasp-yosys") or shutil.which("yosys")
    if exe is None:
        sys.exit("synth.py: yosys not found; run `pip install yowasp-yosys`")
    r = subprocess.run([exe, "-q", "-p", script], cwd=cwd, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"synth.py: yosys failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")


def _cells(stat: str) -> dict[str, int]:
    out = {}
    for m in re.finditer(r"^\s+(\d+)\s+(\$_\w+_)\s*$", stat, re.M):
        out[m.group(2)] = out.get(m.group(2), 0) + int(m.group(1))
    return out


def synth_one(name: str, scratch: Path, vectors: int, seed: int) -> dict:
    d = scratch / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "m.v").write_text(rtl.verilog(name))
    ports, ref = REF[name]
    rng = random.Random(f"{seed}:{name}")
    cases = []
    while len(cases) < vectors:
        i = _fp8_inputs(rng) if name == "fp8_mac" else {p: _bits(rng, w) for p, w in ports.items()}
        want = ref(i)
        if want is not None:
            cases.append((i, want))
    evals = "; ".join(
        "tee -q -a eval.txt eval " + " ".join(f"-set {p} {ports[p]}'h{v:x}" for p, v in i.items()) + " -show y"
        for i, _ in cases
    )
    (d / "eval.txt").write_text("")
    flatten = "synth -flatten -top " + name + (" -booth" if name == "mac4_s8" else "")
    _yosys(
        f"read_verilog -sv m.v; {flatten}; abc -g {GATES}; opt_clean; tee -q -o stat.txt stat; "
        f"tee -q -o ltp.txt ltp -noff; {evals}",
        d,
    )
    _yosys(f"read_verilog -sv m.v; {flatten}; abc -g AND,XOR; opt_clean; tee -q -o stat_ax.txt stat", d)
    cells = _cells((d / "stat.txt").read_text())
    unknown = sorted(set(cells) - set(GE))
    if unknown:
        sys.exit(f"synth.py: {name}: cells without a GE weight: {unknown}")
    ge = sum(GE[c] * n for c, n in cells.items())
    depth = int(re.search(r"length=(\d+)", (d / "ltp.txt").read_text()).group(1))
    ax = _cells((d / "stat_ax.txt").read_text())
    got = [
        int(m.group(2), 2) if m.group(2) else int(m.group(3)) & rho.M32
        for m in re.finditer(r"Eval result: \\y = (?:(\d+)'([01]+)|(-?\d+))\.", (d / "eval.txt").read_text())
    ]
    ok = len(got) == len(cases) and all(g == w for g, (_, w) in zip(got, cases))
    return {
        "cells": cells,
        "ge": round(ge, 2),
        "depth": depth,
        "ands": ax.get("$_AND_", 0),
        "xors": ax.get("$_XOR_", 0),
        "vectors": len(cases),
        "vectors_ok": sum(g == w for g, (_, w) in zip(got, cases)),
        "checked": ok,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--vectors", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scratch")
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    scratch = Path(a.scratch) if a.scratch else Path(tempfile.mkdtemp(prefix="si-fold-synth-"))
    res = {}
    for name in a.only or rtl.MODULES:
        res[name] = synth_one(name, scratch, a.vectors, a.seed)
        r = res[name]
        print(f"{name:16s} GE {r['ge']:9.1f}  depth {r['depth']:3d}  ANDs {r['ands']:6d}  eval {r['vectors_ok']}/{r['vectors']}", flush=True)
    meta = {"ge_weights": GE, "ge_dff": GE_DFF, "gates": GATES, "selftest": rho.selftest()}
    Path(a.out).write_text(json.dumps({"modules": res, "meta": meta}, indent=1, sort_keys=True))
    if not a.scratch:
        shutil.rmtree(scratch)
    bad = [n for n, r in res.items() if not r["checked"]] + [n for n, ok in meta["selftest"] if not ok]
    if bad:
        sys.exit(f"synth.py: unchecked: {bad}")


if __name__ == "__main__":
    main()
