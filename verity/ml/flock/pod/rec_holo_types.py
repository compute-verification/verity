"""The unit types of rec-holo's kill experiment: each traced once from Verity's Boolean builders, written two ways.

- the gate list (`verity.ml.circuits.boolean.trace`): And/Xor/Not gates over wires 0, 1 (constants), the inputs, then one wire per
  gate; route P's gate table and wiring permutation are built from it.
- the C-Flock rows of the same builder (`verity.ml.circuits.boolean.forms`): committed bits are the inputs then the live ANDs, in the
  trace's order; each AND row is two GF(2) linear forms over committed bits (C-Flock's `A·z ∘ B·z = z`, XOR free).

Both are checked to compute the same outputs on random inputs before a file is written. One file a type, `<name>.rhn`, all
little-endian u32: magic, n_in, n_gates, n_out, n_and; then n_gates × (kind, op0, op1) with kind 0 and, 1 xor, 2 not (op1 = 0);
n_out output wires; n_and rows, each two forms; n_out output forms. A form is (len << 1 | constant) then len committed ids.

    python3 rec_holo_types.py --out DIR [NAME ...]
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from array import array
from pathlib import Path

from verity.ml.circuits.boolean import trace as T
from verity.ml.circuits.ops.boolean import elementwise as E, gemm as G, mufu as MU, scalar as SC
from verity_flock import sha512_circuit as SH

MAGIC = 0x524E4C31


def _sha512(C, cv, msg):
    words = SH.compress_gadget(C, [cv[64 * i:64 * i + 64] for i in range(8)], [msg[64 * i:64 * i + 64] for i in range(16)])
    return [b for w in words for b in w]


def _bits(n):
    from verity.ml.circuits.types import Array
    from verity.ml.circuits import boolean as BL
    return Array(n, BL.BIT)


def _defn(d):
    params, ret = d.signature({})
    return d.build, [t for _, t in params], ret


TYPES = {
    "sha512": lambda: (_sha512, [_bits(512), _bits(1024)], _bits(512)),
    "f32_fma": lambda: _defn(E.F32Fma),
    "f32_div": lambda: _defn(E.F32Div),
    "f32_add": lambda: _defn(E.F32Add),
    "f32_mul": lambda: _defn(E.F32Mul),
    "bf16_mul_f32": lambda: _defn(E.Bf16MulF32),
    "mufu_ex2": lambda: _defn(MU.MufuEx2Ftz),
    "mufu_rcp": lambda: _defn(MU.MufuRcpFtz),
    "mufu_tanh": lambda: _defn(MU.MufuTanh),
    "mufu_sqrt": lambda: _defn(MU.MufuSqrtFtz),
    "tc_dot16": lambda: _defn(G.AmpereBF16TcDot16),
    "f32_to_e4m3": lambda: _defn(SC.F32ToE4m3Sat),
}


def both(build, params, ret):
    """The gate list and the C-Flock rows of one builder run (the same live ANDs, in the same order)."""
    held = []

    def capture(C, *ins):
        held.append((C, ins))
        return build(C, *ins)

    net = T.trace(capture, params, ret)
    C, ins = held[0]
    outs = C.outputs["out"]
    in_bits = [next(iter(f.s)) for e in ins for f in e]
    ands = sorted(v for v in C.live() if C.kind[v] == "and")
    cid = {v: i for i, v in enumerate(in_bits)}
    cid.update({v: len(in_bits) + i for i, v in enumerate(ands)})

    def form(f):
        return (f.c, sorted(cid[v] for v in f.s))

    rows = [(form(C.data[v][0]), form(C.data[v][1])) for v in ands]
    return net, rows, [form(f) for f in outs]


def eval_net(net, x, T_):
    mask = (1 << T_) - 1
    w = [0, mask] + list(x)
    for kind, ops in net.gates:
        if kind == "and":
            w.append(w[ops[0]] & w[ops[1]])
        elif kind == "xor":
            w.append(w[ops[0]] ^ w[ops[1]])
        else:
            w.append(w[ops[0]] ^ mask)
    return [w[o] for o in net.outputs]


def eval_rows(n_in, rows, fouts, x, T_):
    mask = (1 << T_) - 1
    z = list(x)

    def val(f):
        c, ids = f
        acc = mask if c else 0
        for i in ids:
            acc ^= z[i]
        return acc

    for a, b in rows:
        z.append(val(a) & val(b))
    return [val(f) for f in fouts]


def write(path: Path, net, rows, fouts):
    u = array("I", [MAGIC, net.n_inputs, len(net.gates), len(net.outputs), len(rows)])
    kinds = {"and": 0, "xor": 1, "not": 2}
    for kind, ops in net.gates:
        u.extend((kinds[kind], ops[0], ops[1] if len(ops) > 1 else 0))
    u.extend(net.outputs)

    def put(f):
        c, ids = f
        u.append(len(ids) << 1 | c)
        u.extend(ids)

    for a, b in rows:
        put(a)
        put(b)
    for f in fouts:
        put(f)
    if sys.byteorder != "little":
        u.byteswap()
    path.write_bytes(u.tobytes())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("names", nargs="*")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    for name in a.names or list(TYPES):
        t0 = time.time()
        net, rows, fouts = both(*TYPES[name]())
        rng = random.Random(name)
        x = [rng.getrandbits(64) for _ in range(net.n_inputs)]
        if eval_net(net, x, 64) != eval_rows(net.n_inputs, rows, fouts, x, 64):
            print(f"{name}: the gate list and the C-Flock rows disagree", file=sys.stderr)
            return 1
        write(a.out / f"{name}.rhn", net, rows, fouts)
        nnz = sum(len(r[0][1]) + len(r[1][1]) + r[0][0] + r[1][0] for r in rows)
        print(f"{name}: inputs {net.n_inputs} outputs {len(net.outputs)} gates {net.counts} rows {len(rows)} "
              f"nnz/row {nnz / max(1, len(rows)):.1f} ({time.time() - t0:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
