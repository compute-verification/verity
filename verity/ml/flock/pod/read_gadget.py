"""The hidden-read kill experiment (hiding model architectures): today's ``GemmCoordinate<K>`` class statement, and the same
statement with R hidden-index Merkle reads of depth D per VU, as ``verity/flock-circuit`` statements for Flock's CPU prover
(``76-read-gadget.sh``).

    python read_gadget.py check                                       a level slot against `verity.ml.commitments.gates.merkle.root_ref`
    python read_gadget.py stage --k K --reads none|A:D:L:R --n N --sets DIR --out DIR
                                                                      circuit.txt, tables/, inst-N.bin, pub-N.bin; prints the
                                                                      statement's shape (JSON)

A read is `verity.ml.commitments.gates.merkle`'s circuit cut into slots of L levels each (a slot type ``merkle<A>x<L>``, like
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
import random
import sys
from pathlib import Path

from verity.ml.commitments.gates import merkle as M
from verity.ml.circuits.boolean.forms import Circuit

from verity_flock import circuit as FC
from verity_flock import ir_lower as IL

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


def check() -> dict:
    """Each slot type against ``root_ref`` on random digests and digits."""
    rng, out = random.Random(SEED), {}
    for A, L in ((2, 1), (2, 3), (4, 1)):
        C, ins, outs = level_circuit(A, L)
        B = A.bit_length() - 1
        for _ in range(3):
            leaf, path = rng.randbytes(64), [[rng.randbytes(64) for _ in range(A - 1)] for _ in range(L)]
            pos = rng.randrange(A ** L)
            want = M._digest_bits(M.root_ref(A, leaf, pos, path))
            bits = M._digest_bits(leaf) + [b for lv in path for s in lv for b in M._digest_bits(s)] + [(pos >> j) & 1 for j in range(32)]
            vals = {nm: bits[32 * k:32 * k + 32] for k, nm in enumerate(ins)}
            got, _ = C.evaluate(vals, 1)
            if [b for nm in outs for b in got[nm]] != want:
                raise SystemExit(f"merkle{A}x{L}: the slot's output is not root_ref's")
        out[f"merkle{A}x{L}"] = "ok"
    return out


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
    a = ap.parse_args(argv)
    res = check() if a.cmd == "check" else stage(a.k, a.reads, a.n, Path(a.sets), Path(a.out))
    json.dump(res, sys.stdout, sort_keys=True)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
