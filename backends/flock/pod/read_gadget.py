"""The hidden-read kill experiment (hiding model architectures): today's ``GemmCoordinate<K>`` class statement, and the same
statement with R hidden-index Merkle reads of depth D per VU, as ``verity/flock-circuit`` statements for C-Flock's CPU prover
(``76-read-gadget.sh``).

    python read_gadget.py check                                       each slot type against `verity.ml.boolean.merkle`'s references
    python read_gadget.py stage --k K --reads none|A:D:L:R --n N --sets DIR --out DIR
                                                                      circuit.txt, tables/, inst-N.bin, pub-N.bin; prints the
                                                                      statement's shape (JSON)
    python read_gadget.py measure [--shape A:D:J ...] [--jobs N] --out DIR
                                                                      subtree reads built whole (default: runs of 1, 16 and 256
                                                                      cells in stores of 2^30 to 2^33 cells), their ANDs and rows
                                                                      against the sum of their parts, and the rows C-Flock lays
                                                                      each slot type out in: measure.jsonl, measure-summary.json

A read is `verity.ml.boolean.merkle`'s circuit cut into slots of L levels each (a slot type ``merkle<A>x<L>``, like
``sha512x3``): a slot takes the current digest (16 ports of 32 bits), L (A - 1) sibling digests and a 32-bit port whose low
L log2(A) bits are the L digits, and gives the digest L levels up (16 ports of 32 bits); the D / L slots of a read are chained
by wires, the last one's output the root. A read's position, path and leaf are its slots' inputs, which here are wired from the
units' 32-bit cut outputs (round robin), so the read computes a root nobody checks: the experiment prices the circuit (its rows,
block, prover memory and time), which an honest read with a public root region changes by no slot and one 512-bit region. The
slots are packed after every range today's statement places (`verity_flock.circuit._ranges`).
"""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing
import random
import sys
from array import array
from pathlib import Path

from verity.ir import boolean as BL
from verity.ml.boolean import merkle as M
from verity.ml.boolean import forms as F
from verity.ml.boolean.forms import Circuit

from verity_flock import circuit as FC
from verity_flock import ir_lower as IL
from verity_flock import sha512_circuit as SC

SEMANTICS = "sm80.mma.m16n8k16.bf16"
SEED = 20261003


def level_circuit(A: int, L: int) -> tuple[Circuit, list[str], list[str]]:
    """L levels of an A-ary read: ``cur0..15``, ``s<l>_<j>_<i>`` (level l's sibling j, 32-bit part i), ``d``; ``o0..15``."""
    B = A.bit_length() - 1
    if A not in M.ARITIES or B * L > 32:
        raise ValueError(f"arity {A} with {L} levels a slot")
    C = Circuit()
    cur_n = [f"cur{i}" for i in range(16)]
    sib_n = [[[f"s{l}_{j}_{i}" for i in range(16)] for j in range(A - 1)] for l in range(L)]
    cur = [C.input(n, 32) for n in cur_n]
    sibs = [[[C.input(n, 32) for n in ns] for ns in lv] for lv in sib_n]
    d = C.input("d", 32)
    flat = lambda ports: [b for p in ports for b in p]  # noqa: E731
    x = flat(cur)
    for l in range(L):
        x = M._node(C, M._slots(C, d[B * l:B * l + B], x, [flat(s) for s in sibs[l]]))
    outs = [f"o{i}" for i in range(16)]
    for i, nm in enumerate(outs):
        C.output(nm, x[32 * i:32 * i + 32])
    return C, cur_n + [n for lv in sib_n for ns in lv for n in ns] + ["d"], outs


def _digest_out(C: Circuit, x: list) -> list[str]:
    outs = [f"o{i}" for i in range(16)]
    for i, nm in enumerate(outs):
        C.output(nm, x[32 * i:32 * i + 32])
    return outs


def leaf_circuit() -> tuple[Circuit, list[str], list[str]]:
    """A cell's leaf (the subtree read's first stage): ``bc0..31`` (its b || c, 32-bit parts in word order); ``o0..15``."""
    C = Circuit()
    ins = [f"bc{i}" for i in range(32)]
    bc = [b for nm in ins for b in C.input(nm, 32)]
    return C, ins, _digest_out(C, M._leaf(C, bc))


def node_circuit(A: int) -> tuple[Circuit, list[str], list[str]]:
    """A node inside a run: ``c<j>_<i>`` (child j's 32-bit part i); ``o0..15``."""
    C = Circuit()
    ins = [[f"c{j}_{i}" for i in range(16)] for j in range(A)]
    kids = [[b for nm in ns for b in C.input(nm, 32)] for ns in ins]
    return C, [nm for ns in ins for nm in ns], _digest_out(C, M._node(C, kids))


def check() -> dict:
    """Each slot type against `verity.ml.boolean.merkle`'s references on random digests, digits and commit strings."""
    rng, out = random.Random(SEED), {}

    def run(C, ins, outs, bits, want, name):
        got, _ = C.evaluate({nm: bits[32 * k:32 * k + 32] for k, nm in enumerate(ins)}, 1)
        if [b for nm in outs for b in got[nm]] != M._digest_bits(want):
            raise SystemExit(f"{name}: the slot's output is not the reference's")

    for A, L in ((2, 1), (2, 3), (4, 1)):
        C, ins, outs = level_circuit(A, L)
        for _ in range(3):
            leaf, path = rng.randbytes(64), [[rng.randbytes(64) for _ in range(A - 1)] for _ in range(L)]
            pos = rng.randrange(A ** L)
            bits = M._digest_bits(leaf) + [b for lv in path for s in lv for b in M._digest_bits(s)] + [(pos >> j) & 1 for j in range(32)]
            run(C, ins, outs, bits, M.root_ref(A, leaf, pos, path), f"merkle{A}x{L}")
        out[f"merkle{A}x{L}"] = "ok"
    C, ins, outs = leaf_circuit()
    for _ in range(3):
        bc = rng.randbytes(128)
        run(C, ins, outs, M._digest_bits(bc[:64]) + M._digest_bits(bc[64:]), M.leaf_ref(bc), "hm96leaf")
    out["hm96leaf"] = "ok"
    for A in M.ARITIES:
        C, ins, outs = node_circuit(A)
        for _ in range(3):
            kids = [rng.randbytes(64) for _ in range(A)]
            run(C, ins, outs, [b for k in kids for b in M._digest_bits(k)], M.node_ref(kids), f"node{A}")
        out[f"node{A}"] = "ok"
    return out


# --- measure: subtree reads built whole ----------------------------------------------------------------------------------------

#: runs of 1, 16 and 256 cells (J per arity) in stores of 2^30 to 2^33 cells (D per arity: the smallest tree that holds them)
RUNS = {2: (0, 4, 8), 4: (0, 2, 4)}
STORE_LOG2 = (30, 31, 32, 33)


def _depth(A: int, log2_cells: int) -> int:
    b = A.bit_length() - 1
    return -(-log2_cells // b)


def shapes() -> list[tuple[int, int, int]]:
    return sorted({(A, _depth(A, s), J) for A in M.ARITIES for s in STORE_LOG2 for J in RUNS[A]})


class _Lean(Circuit):
    """`forms.Circuit` keeping each committed bit's kind and its operands' bits in flat arrays and no forms, enough for liveness
    and the AND and copy counts of a read of 10^8 rows. Operands come before their AND, so one backward sweep finds the live
    bits."""

    def __init__(self):
        super().__init__()
        self.code, self.off, self.dep = bytearray(), array("Q", [0]), array("I")

    def _new(self, kind, data):
        if kind == "in":
            self.code.append(0)
        elif kind == "and":
            a, b = data
            self.code.append(2 if F.ONE.key() in (a.key(), b.key()) else 1)
            self.dep.extend(a.s)
            self.dep.extend(b.s)
        else:
            raise ValueError(f"a Merkle read commits no {kind}")
        self.off.append(len(self.dep))
        return F.L(frozenset((len(self.code) - 1,)), 0)

    def tally(self) -> dict:
        live = bytearray(len(self.code))
        for bits in self.outputs.values():
            for w in bits:
                for v in w.s:
                    live[v] = 1
        code, off, dep = self.code, self.off, self.dep
        for v in range(len(code) - 1, -1, -1):
            if live[v] and code[v]:
                for u in dep[off[v]:off[v + 1]]:
                    live[u] = 1
        n, dead = [0, 0, 0], 0
        for v, c in enumerate(code):
            if live[v]:
                n[c] += 1
            elif c:
                dead += 1
        return {"inputs": n[0], "ands": n[1], "copies": n[2], "rows": n[1] + n[2], "dead": dead}


def whole(A: int, D: int, J: int | None) -> dict:
    """The read (``J`` None: `MerkleRead`) built whole and tallied."""
    C = _Lean()
    ps, _ = M._types(A, D, J)
    els = [C.input(f"e{i}", w) for i, w in enumerate(w for t in ps for w in BL.elements(t))]
    C.output("ok", M.build(C, A, D, *els) if J is None else M.build_subtree(C, A, D, J, *els))
    return {"arity": A, "D": D, "J": J, **C.tally()}


def _whole(shape):
    return whole(*shape)


def parts(rows: dict) -> dict:
    """Each part's (ANDs, rows) from small reads built whole: a level (D 2 less D 1), the comparison (a 1-level read less its level),
    a cell's leaf (a 1-cell subtree read less a 1-level read) and a run node (a 2-level read of A cells less its leaves, level and
    comparison)."""
    def d(x, y, *rest):
        return [x[k] - y[k] - sum(r[k] for r in rest) for k in (0, 1)]

    out = {}
    for A in M.ARITIES:
        g = {key: (r["ands"], r["rows"]) for key, r in rows.items() if key[0] == A}
        level = d(g[(A, 2, None)], g[(A, 1, None)])
        compare = d(g[(A, 1, None)], level)
        leaf = d(g[(A, 1, 0)], g[(A, 1, None)])
        node = d(g[(A, 2, 1)], level, compare, *[leaf] * A)
        out[A] = {"level": level, "compare": compare, "leaf": leaf, "run_node": node}
    return out


def predicted(P: dict, A: int, D: int, J: int) -> list[int]:
    p, R = P[A], A ** J
    return [R * p["leaf"][k] + (R - 1) // (A - 1) * p["run_node"][k] + (D - J) * p["level"][k] + p["compare"][k] for k in (0, 1)]


def slot_rows() -> dict:
    """The useful rows and slot of each slot type C-Flock would lay a read out in (`ir_lower.layout`), and of its row hashing
    (`sha512x3`, three compressions, and `hm96`, a row's b || c from its last chaining value and salt)."""
    nets = {"hm96leaf": leaf_circuit(), "node2": node_circuit(2), "node4": node_circuit(4), "merkle2x1": level_circuit(2, 1),
            "merkle2x3": level_circuit(2, 3), "merkle4x1": level_circuit(4, 1),
            "sha512x3": (SC.compression3(), ["cv0", "msg0", "cv1", "msg1", "cv2", "msg2"], ["out0", "out1", "out2"]),
            "hm96": (SC.hm96_row(), ["cv", "pad", "y"], ["bc"])}
    out = {}
    for name, (C, ins, outs) in nets.items():
        lay = IL.layout(C, ins, outs)
        out[name] = {"useful_rows": lay.useful, "slot_log": FC._slot_log(lay.useful)}
    return out


def measure(want: list[tuple[int, int, int]], jobs: int, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    small = [(A, D, None) for A in M.ARITIES for D in (1, 2)] + [(A, 1, 0) for A in M.ARITIES] + [(A, 2, 1) for A in M.ARITIES]
    todo = sorted(set(small) | set(want), key=lambda s: -((s[0] ** (s[2] or 0)) * 4 + s[1]))
    got = {}
    with (out / "measure.jsonl").open("w") as f, multiprocessing.get_context("fork").Pool(jobs, maxtasksperchild=1) as pool:
        for r in pool.imap_unordered(_whole, todo):
            got[(r["arity"], r["D"], r["J"])] = r
            f.write(json.dumps({"kind": "read", **r}, sort_keys=True) + "\n")
            f.flush()
        slots = slot_rows()
        for name, s in slots.items():
            f.write(json.dumps({"kind": "slot", "name": name, **s}, sort_keys=True) + "\n")
    P = parts(got)
    reads = []
    for A, D, J in want:
        r = got[(A, D, J)]
        reads.append({**r, "cells": A ** J, "predicted": predicted(P, A, D, J), "exact": [r["ands"], r["rows"]] == predicted(P, A, D, J),
                      "separate": [A ** J * got[(A, D, 0)][k] for k in ("ands", "rows")] if (A, D, 0) in got else None})
    summary = {"parts": P, "reads": reads, "slots": slots, "dead": sum(r["dead"] for r in got.values()),
               "all_exact": all(r["exact"] for r in reads)}
    (out / "measure-summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    return {"reads": len(reads), "all_exact": summary["all_exact"], "dead": summary["dead"]}


def _sections(text: str) -> dict[str, int]:
    """Each CIRCUIT section's name and line count (a flock-ir-unit/v2 expanded circuit: one line per row, then the header)."""
    return {ln.split()[1]: int(ln.split()[2]) for ln in text.split("\n") if ln.startswith("CIRCUIT ")}


def with_reads(text: str, A: int, D: int, L: int, R: int) -> str:
    """``text`` with R reads of depth D per VU in slots of L levels, packed after its ranges."""
    if D % L:
        raise ValueError(f"depth {D} is not whole slots of {L} levels")
    head, meta_line, rest = text.split("\n", 2)
    meta = json.loads(meta_line[len("META "):])
    C, ins, outs = level_circuit(A, L)
    lay = IL.layout(C, ins, outs)
    name = f"merkle{A}x{L}"
    net = FC._expanded(lay, name)
    G, per_read = meta["vus_per_block"], D // L
    items = [(r["type"], r["slot_log"], r["count"], {k: v for k, v in r.items() if k not in ("type", "slot_log", "pos0", "count")})
             for r in meta["ranges"]]
    if FC._ranges(items)[0] != meta["ranges"]:
        raise AssertionError("the statement's ranges are not _ranges of their items")
    items.append(("unit", FC._slot_log(lay.useful), G * R * per_read, {"net": name, "packed": True}))
    meta["ranges"], meta["k_log"] = FC._ranges(items)
    if meta["k_log"] > FC.K_MAX:
        raise ValueError(f"a block of 2^{meta['k_log']} bits (at most 2^{FC.K_MAX})")
    src = itertools.cycle(range(meta["units_per_vu"]))
    wires = []
    for r in range(R):
        for s in range(per_read):
            i = r * per_read + s
            for p in range(len(ins)):
                wires.append([name, i, p, name, i - 1, p] if p < 16 and s else [name, i, p, "unit", next(src), 0])
    meta["wires"] = meta["wires"] + wires
    meta["per_vu"] = {**meta["per_vu"], name: R * per_read}
    return (f"{head}\nMETA {json.dumps(meta, sort_keys=True, separators=(',', ':'))}\n{rest}"
            f"CIRCUIT {name} {net.count(chr(10))}\n{net}")


def shape(text: str) -> dict:
    meta = json.loads(text.split("\n", 2)[1][len("META "):])
    lines = _sections(text)
    slot = {r["net"]: r["slot_log"] for r in meta["ranges"] if r["type"] == "unit"}
    useful = {nm: meta["per_vu"][nm] * (lines[nm] - 1) for nm in meta["per_vu"]}
    slotted = {nm: meta["per_vu"][nm] << slot[nm] for nm in meta["per_vu"]}
    return {"k_log": meta["k_log"], "block_bits": 1 << meta["k_log"], "vus_per_block": meta["vus_per_block"], "per_vu": meta["per_vu"],
            "slot_log": slot, "useful_rows_per_vu": useful, "useful_rows": sum(useful.values()), "slotted_rows": sum(slotted.values()),
            "mask_rows": 1 << meta["mask_log"], "circuit_bytes": len(text.encode()), "circuit_sha512": FC.digest(text),
            "ranges": meta["ranges"]}


def stage(K: int, reads: str, n: int, sets: Path, out: Path) -> dict:
    from verity_numerical.bench import generate as GEN
    from verity_numerical.bench import templates as TM
    from verity_numerical.bench.input_sets import InputSet

    sub = TM.get("gemm-coordinate").bind(K=K, semantics=SEMANTICS)
    path = sets / "sets" / sub.set_name
    if not (path / "manifest.json").exists():
        GEN.generate(sub, n, SEED, sets)
    s = InputSet.open(path)
    if s.n < n:
        raise SystemExit(f"{path}: {s.n} instances, {n} wanted")
    low = FC.lowering_for_set(s)
    text = FC.compose(low, s.subcircuit, statics=FC.set_statics(s))
    if reads != "none":
        A, D, L, R = (int(x) for x in reads.split(":"))
        text = with_reads(text, A, D, L, R)
    out.mkdir(parents=True, exist_ok=True)
    (out / "circuit.txt").write_text(text)
    IL.write_tables(out / "tables")
    FC.write(text, s, 0, n, out / f"inst-{n}.bin", out / f"pub-{n}.bin")
    return {"subcircuit": s.subcircuit.id, "unit": low.name, "reads": reads, "n": n, "pin": FC.digest(text), **shape(text)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("check")
    st = sp.add_parser("stage")
    st.add_argument("--k", type=int, required=True)
    st.add_argument("--reads", default="none", help="none, or A:D:L:R (arity, depth, levels a slot, reads per VU)")
    st.add_argument("--n", type=int, default=2)
    st.add_argument("--sets", required=True, help="where the synthetic input sets are generated (and reused)")
    st.add_argument("--out", required=True)
    me = sp.add_parser("measure")
    me.add_argument("--shape", action="append", default=[], help="A:D:J (arity, depth, run height); default: shapes()")
    me.add_argument("--jobs", type=int, default=8)
    me.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "check":
        res = check()
    elif a.cmd == "measure":
        want = [tuple(int(x) for x in s.split(":")) for s in a.shape] or shapes()
        res = measure(want, a.jobs, Path(a.out))
    else:
        res = stage(a.k, a.reads, a.n, Path(a.sets), Path(a.out))
    json.dump(res, sys.stdout, sort_keys=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
