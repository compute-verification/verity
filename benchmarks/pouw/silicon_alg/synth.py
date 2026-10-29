"""Synthesize SI-ALG's MACs (rtl/*.v) with yosys + ABC, check every netlist against refs.py, and tabulate.

For each design (yosys's array/macc multiplier mapping; yosys 0.33 has no Booth pass):
- GE: ABC maps to generic.lib (gate equivalents, NAND2 = 1); the accumulator register is added at a DFF's 5.667 GE a bit.
- ANDs: ABC maps to AND/XOR (NOT free), the cost model of C-Flock's Boolean circuits; an upper bound on the
  multiplicative complexity, since ABC doesn't minimize ANDs specifically.
- depth: the longest path in the AND/XOR netlist (`ltp -noff`), a latency proxy.
Both netlists of every design are simulated bit for bit against its exact reference on the vectors refs.py draws.

    python3 synth.py --out OUT [--vectors N] [--only NAME,...] [--jobs J]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import refs

HERE = Path(__file__).resolve().parent
RTL = [HERE / "rtl" / "baseline.v", HERE / "rtl" / "formats.v"]
LIB = HERE / "generic.lib"
DFF_GE = 4.522 / 0.798

CELL = {
    "$_NOT_": lambda p, m: ~p["A"] & m, "INV": lambda p, m: ~p["A"] & m,
    "$_BUF_": lambda p, m: p["A"], "BUF": lambda p, m: p["A"],
    "$_AND_": lambda p, m: p["A"] & p["B"], "AND2": lambda p, m: p["A"] & p["B"],
    "$_OR_": lambda p, m: p["A"] | p["B"], "OR2": lambda p, m: p["A"] | p["B"],
    "$_XOR_": lambda p, m: p["A"] ^ p["B"], "XOR2": lambda p, m: p["A"] ^ p["B"],
    "$_NAND_": lambda p, m: ~(p["A"] & p["B"]) & m, "NAND2": lambda p, m: ~(p["A"] & p["B"]) & m,
    "$_NOR_": lambda p, m: ~(p["A"] | p["B"]) & m, "NOR2": lambda p, m: ~(p["A"] | p["B"]) & m,
    "$_XNOR_": lambda p, m: ~(p["A"] ^ p["B"]) & m, "XNOR2": lambda p, m: ~(p["A"] ^ p["B"]) & m,
    "$_ANDNOT_": lambda p, m: p["A"] & ~p["B"] & m, "$_ORNOT_": lambda p, m: (p["A"] | ~p["B"]) & m,
    "$_MUX_": lambda p, m: (p["A"] & ~p["S"] | p["B"] & p["S"]) & m,
    "MUX2": lambda p, m: (p["A"] & ~p["S"] | p["B"] & p["S"]) & m,
    "NAND3": lambda p, m: ~(p["A"] & p["B"] & p["C"]) & m, "NOR3": lambda p, m: ~(p["A"] | p["B"] | p["C"]) & m,
    "AOI21": lambda p, m: ~((p["A"] & p["B"]) | p["C"]) & m, "OAI21": lambda p, m: ~((p["A"] | p["B"]) & p["C"]) & m,
    "AOI22": lambda p, m: ~((p["A"] & p["B"]) | (p["C"] & p["D"])) & m,
    "OAI22": lambda p, m: ~((p["A"] | p["B"]) & (p["C"] | p["D"])) & m,
}


def yosys() -> str:
    for exe in ("yosys", "yowasp-yosys"):
        if shutil.which(exe):
            return exe
    sys.exit("no yosys: pip install yowasp-yosys")


def run_yosys(top: str, work: Path) -> dict:
    reads = "; ".join(f"read_verilog {p}" for p in RTL)
    front = f"{reads}; hierarchy -top {top}; synth -flatten -top {top}"
    lib_json, xag_json, xag_net, lib_net, ltp = (work / f for f in
                                                 ("lib.json", "xag.json", "xag_net.json", "lib_net.json", "ltp.txt"))
    script = (f"{front}; design -save rtl; "
              f"abc -liberty {LIB}; opt_clean; tee -q -o {lib_json} stat -json -liberty {LIB}; "
              f"write_json {lib_net}; design -load rtl; "
              f"abc -g AND,XOR; opt_clean; tee -q -o {xag_json} stat -json; write_json {xag_net}; "
              f"tee -q -o {ltp} ltp -noff")
    log = work / "yosys.log"
    with open(log, "w") as fh:
        r = subprocess.run([yosys(), "-q", "-p", script], stdout=fh, stderr=subprocess.STDOUT, cwd=work,
                           env={**os.environ, "TMPDIR": str(work)})
    if r.returncode:
        raise RuntimeError(f"yosys failed on {top}: {log.read_text()[-3000:]}")
    lib = json.loads(lib_json.read_text())
    xag = json.loads(xag_json.read_text())
    mod = lambda s: s["modules"][next(iter(s["modules"]))] if "modules" in s else s["design"]
    lib_m, xag_m = mod(lib), mod(xag)
    types = xag_m.get("num_cells_by_type", {})
    depth = int(ltp.read_text().split("Longest topological path in")[1].split("(length=")[1].split(")")[0])
    return {
        "ge_comb": float(lib_m.get("area", lib_m.get("chip_area", 0.0))),
        "lib_cells": lib_m.get("num_cells_by_type", {}),
        "ands": int(types.get("$_AND_", 0)),
        "xors": int(types.get("$_XOR_", 0)),
        "nots": int(types.get("$_NOT_", 0)),
        "depth": depth,
        "nets": {"lib": json.loads(lib_net.read_text()), "xag": json.loads(xag_net.read_text())},
    }


def simulate(net: dict, top: str, vec: dict, n: int) -> dict:
    """Evaluate a yosys JSON netlist on n vectors at once (a Python int a net, bit i = vector i)."""
    mod = net["modules"][top]
    mask = (1 << n) - 1
    val: dict = {"0": 0, "1": mask, "x": 0}
    for name, port in mod["ports"].items():
        if port["direction"] != "input":
            continue
        col = vec[name]
        for bit, net_id in enumerate(port["bits"]):
            if isinstance(net_id, int):
                val[net_id] = sum(((x >> bit) & 1) << i for i, x in enumerate(col))
    cells = list(mod["cells"].values())
    users: dict = {}
    need = []
    for k, c in enumerate(cells):
        ins = [b[0] for p, b in c["connections"].items() if p != "Y"]
        pend = [b for b in ins if b not in val]
        need.append(len(pend))
        for b in pend:
            users.setdefault(b, []).append(k)
    ready = [k for k in range(len(cells)) if need[k] == 0]
    while ready:
        k = ready.pop()
        c = cells[k]
        y = c["connections"]["Y"][0]
        val[y] = CELL[c["type"]]({p: val[b[0]] for p, b in c["connections"].items() if p != "Y"}, mask)
        for u in users.get(y, ()):
            need[u] -= 1
            if need[u] == 0:
                ready.append(u)
    out = {}
    for name, port in mod["ports"].items():
        if port["direction"] != "output":
            continue
        word = [0] * n
        for bit, net_id in enumerate(port["bits"]):
            v = val[net_id] if isinstance(net_id, int) else (mask if net_id == "1" else 0)
            for i in range(n):
                word[i] |= ((v >> i) & 1) << bit
        out[name] = word
    return out


def check(top: str, spec: dict, nets: dict, n: int, seed: int) -> dict:
    rng = random.Random(f"{top}:{seed}")
    vec = spec["gen"](rng, n)
    n = len(next(iter(vec.values())))
    want = [spec["ref"](vec, i) for i in range(n)]
    res = {}
    for kind, net in nets.items():
        got = simulate(net, top, vec, n)
        bad = [i for i in range(n) for k in want[i] if got[k][i] != want[i][k]]
        res[kind] = {"vectors": n, "mismatches": len(bad),
                     "first": None if not bad else {"i": bad[0], **{k: vec[k][bad[0]] for k in vec},
                                                    "want": want[bad[0]],
                                                    "got": {k: got[k][bad[0]] for k in got}}}
    return res


def one(top: str, n: int, seed: int) -> dict:
    spec = refs.designs()[top]
    with tempfile.TemporaryDirectory() as td:
        r = run_yosys(top, Path(td))
    nets = r.pop("nets")
    r["check"] = check(top, spec, nets, n, seed)
    r.update(top=top, lanes=spec["lanes"], accw=spec["accw"], role=spec["role"])
    r["ge_reg"] = spec["accw"] * DFF_GE
    r["ge_per_mac"] = (r["ge_comb"] + r["ge_reg"]) / spec["lanes"]
    r["ge_comb_per_mac"] = r["ge_comb"] / spec["lanes"]
    r["ands_per_mac"] = r["ands"] / spec["lanes"]
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--vectors", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    names = [x for x in a.only.split(",") if x] or list(refs.designs())
    rows = []
    with cf.ProcessPoolExecutor(a.jobs) as ex:
        futs = {ex.submit(one, t, a.vectors, a.seed): t for t in names}
        for f in cf.as_completed(futs):
            r = f.result()
            ok = all(c["mismatches"] == 0 for c in r["check"].values())
            print(f"{r['top']:22s} GE {r['ge_comb']:9.1f} +reg {r['ge_reg']:6.1f} "
                  f"/MAC {r['ge_per_mac']:8.1f}  AND/MAC {r['ands_per_mac']:7.1f} depth {r['depth']:4d} "
                  f"check {'ok' if ok else 'FAIL'} ({r['check']['xag']['vectors']} vectors)", flush=True)
            rows.append(r)
    rows.sort(key=lambda r: names.index(r["top"]))
    best = {r["top"]: r for r in rows}
    base = {"s8": best.get("mac_s8x4_s32"), "fp8": best.get("mac_fp8_fp32")}
    lines = ["| design | role | lanes | acc bits | GE comb | GE reg | GE a MAC | GE comb a MAC | vs s8 | vs FP8 | "
             "ABC ANDs a MAC | depth | check |", "|" + "---|" * 13]
    for t in names:
        r = best[t]
        mins = {k: min(x[k] for x in rows if x["top"] == t) for k in ("ands_per_mac",)}
        ok = all(c["mismatches"] == 0 for x in rows if x["top"] == t for c in x["check"].values())
        nv = sum(c["vectors"] for x in rows if x["top"] == t for c in x["check"].values())
        vs = lambda k: f"{r['ge_per_mac'] / base[k]['ge_per_mac']:.2f}x" if base[k] else "-"
        lines.append(f"| {t} | {r['role']} | {r['lanes']} | {r['accw']} | "
                     f"{r['ge_comb']:.0f} | {r['ge_reg']:.0f} | {r['ge_per_mac']:.1f} | {r['ge_comb_per_mac']:.1f} | "
                     f"{vs('s8')} | {vs('fp8')} | {mins['ands_per_mac']:.1f} | {r['depth']} | "
                     f"{'ok' if ok else 'FAIL'} ({nv}) |")
    table = "\n".join(lines) + "\n"
    (out / "table.md").write_text(table)
    src = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in RTL + [LIB, HERE / "refs.py",
                                                                            Path(__file__).resolve()]}
    ver = subprocess.run([yosys(), "-V"], capture_output=True, text=True).stdout.strip()
    (out / "synth.json").write_text(json.dumps({"yosys": ver, "dff_ge": DFF_GE, "sources": src,
                                                "rows": rows}, indent=1, default=str))
    print(table)
    bad = [r["top"] for r in rows if any(c["mismatches"] for c in r["check"].values())]
    if bad:
        print("CHECK FAILED:", sorted(set(bad)))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
