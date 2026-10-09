"""A served unit as a hidden netlist: one Q_word unit class of a vLLM Definition, lowered by C-Flock's pieces
(`verity_flock.partition_units.lower_class`) and written as the universal unit's SSA netlist (`universal.Netlist`).

    python benchmarks/private_circuit/served_netlist.py --definition 'GemmCoordinate_v1{K=8192}' --out DIR [--vectors 16]

The class is the Definition's largest unit class under `Q_word` v1 {16, 32} (by word gates) that lowers. Its circuit is ANDs over
GF(2) linear forms; each distinct form an AND or an output reads becomes a chain of 2-input gates over its wires (XOR, the last one
XNOR when its constant is 1; a one-wire form with constant 1 is NAND(w, w); a constant is XOR / XNOR of wire 0 with itself), then
each AND is one AND gate. Wires 0 .. N_IN - 1 are the unit's input bits, port by port, each word LSB first; outputs likewise. A
unit with a table read (`lookups`) is refused: the netlist has no read.

The netlist is checked against the reference: the unit's inputs and outputs on ``--vectors`` synthetic draws lifted through its
flat Definition (`partition_units.lanes`), the netlist run bit-sliced. ``DIR/netlist.json.gz`` is
``{"format": "verity/universal-netlist/v0", "n_in", "gates": [[op, a, b], ...], "outputs": [...], ...}`` (ops as in
`universal`: 0 XOR, 1 AND, 2 XNOR, 3 NAND), and ``DIR/netlist.summary.json`` its class and counts.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

from verity.ml.circuits.codec import program_digest
from verity.ml.private_circuits import universal as U
from verity.ml.randomness import derive
from verity_flock import partition_units as PU

XOR, AND, XNOR, NAND = U.XOR, U.AND, U.XNOR, U.NAND


def to_netlist(C) -> tuple[U.Netlist, dict]:
    live = C.live()
    ins = [w for name in sorted(C.inputs, key=lambda k: int(k[2:])) for w in C.inputs[name]]
    wire: dict[int, int] = {}
    for i, w in enumerate(ins):
        (b,) = w.s
        wire[b] = i
    n_in = len(ins)
    gates: list[tuple[int, int, int]] = []
    forms: dict = {}

    def new(op: int, a: int, b: int) -> int:
        gates.append((op, a, b))
        return n_in + len(gates) - 1

    def form(f) -> int:
        k = f.key()
        if k in forms:
            return forms[k]
        ws = sorted(wire[b] for b in f.s)
        if not ws:
            v = new(XNOR if f.c else XOR, 0, 0)
        elif len(ws) == 1:
            v = new(NAND, ws[0], ws[0]) if f.c else ws[0]
        else:
            acc = ws[0]
            for j, w in enumerate(ws[1:]):
                acc = new(XNOR if f.c and j == len(ws) - 2 else XOR, acc, w)
            v = acc
        forms[k] = v
        return v

    for v in sorted(live):
        kind = C.kind[v]
        if kind == "in":
            continue
        if kind != "and":
            raise ValueError(f"the unit's circuit has a {kind} bit: the netlist has none")
        a, b = C.data[v]
        wire[v] = new(AND, form(a), form(b))
    outs = [form(w) for name in sorted(C.outputs, key=lambda k: int(k[3:])) for w in C.outputs[name]]
    ops = [0, 0, 0, 0]
    for g in gates:
        ops[g[0]] += 1
    return U.Netlist(n_in, tuple(gates), tuple(outs)), {"xor": ops[XOR], "and": ops[AND], "xnor": ops[XNOR], "nand": ops[NAND]}


def run_sliced(net: U.Netlist, x: list[int], T: int) -> list[int]:
    """`universal.interpret` over T lanes at once: ``x[i]`` holds input bit i of every lane."""
    full = (1 << T) - 1
    w = list(x)
    for op, a, b in net.gates:
        va, vb = w[a], w[b]
        v = (va & vb) if op & 1 else (va ^ vb)
        w.append(v ^ full if op >> 1 else v)
    return [w[o] for o in net.outputs]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--definition", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--vectors", type=int, default=16)
    ap.add_argument("--max-gates", type=int, default=1 << 20)
    a = ap.parse_args(argv)
    from circuit_check import targets as TG

    TG.load_registries()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    units = PU.partition_units(PU.one_call(TG.definition(a.definition)), "q-word")
    classes = units.classes()
    order = sorted(classes.items(), key=lambda kv: -len(kv[1].representative.gates))
    summary = {"definition": a.definition, "query": units.query, "classes": len(classes),
               "by_word_gates": [{"shape": s[:16], "unit": cl.representative.unit, "units": cl.units, "word_gates": len(cl.representative.gates),
                                  "in_bits": sum(cl.representative.in_widths), "out_bits": sum(cl.representative.out_widths)}
                                 for s, cl in order[:10]]}
    for s, cl in order:
        uc = cl.representative
        act = units.activation(cl.definition)
        low = PU.lower_class(act, uc, a.max_gates)
        if low.circuit is None or low.counts["lookups"]:
            summary.setdefault("skipped", []).append({"shape": s[:16], "why": low.reason or "a table read", "missing": list(low.missing)})
            continue
        break
    else:
        summary["netlist"] = None
        (out / "netlist.summary.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary))
        return 1
    lower_s = time.monotonic() - t0
    net, ops = to_netlist(low.circuit)
    key = derive(PU.SOURCE, PU.DOMAIN, {"program": program_digest(units.program)[:32], "seed": 0})
    ins, want = PU.lanes(act, [uc], key, a.vectors)[(uc.definition, uc.unit)]
    T = a.vectors
    x = [sum(((ins[p][t] >> j) & 1) << t for t in range(T)) for p, wd in enumerate(uc.in_widths) for j in range(wd)]
    got = run_sliced(net, x, T)
    exp = [sum(((want[p][t] >> j) & 1) << t for t in range(T)) for p, wd in enumerate(uc.out_widths) for j in range(wd)]
    mismatched_bits = sum(bin(g ^ e).count("1") for g, e in zip(got, exp))
    G, N_IN, N_OUT = len(net.gates), net.n_in, len(net.outputs)
    summary["netlist"] = {
        "shape": uc.shape, "unit": uc.unit, "flat_definition": uc.definition, "word_gates": len(uc.gates),
        "in_widths": list(uc.in_widths), "out_widths": list(uc.out_widths),
        "gf2": low.counts, "gates": ops, "G": G, "N_IN": N_IN, "N_OUT": N_OUT,
        "check": {"vectors": T, "mismatched_bits": mismatched_bits, "output_bits": N_OUT * T},
        "class": {"G": G, "N_IN": N_IN, "N_OUT": N_OUT, "K": U.index_bits(G, N_IN),
                  "ands_closed_form": 2 * G * N_IN + G * G + N_OUT * (N_IN + G) if G > 200000 else U.and_count(G, N_IN, N_OUT),
                  "ands_exact": G <= 200000},
        "seconds": {"lower": round(lower_s, 1), "total": round(time.monotonic() - t0, 1)}}
    with gzip.open(out / "netlist.json.gz", "wt") as f:
        json.dump({"format": "verity/universal-netlist/v0", "definition": a.definition, "shape": uc.shape, "n_in": N_IN,
                   "in_widths": list(uc.in_widths), "out_widths": list(uc.out_widths), "ops": {"0": "XOR", "1": "AND", "2": "XNOR", "3": "NAND"},
                   "gates": net.gates, "outputs": net.outputs}, f, separators=(",", ":"))
    (out / "netlist.summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))
    return 0 if mismatched_bits == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
