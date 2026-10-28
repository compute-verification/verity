"""ANDs of the aggregate block's parts in the repo's Boolean lowering: f_mix on bits (checked against agg_f.mix on all
2^16 inputs), the sum of g words mod 2^16, the int8 noise add and the s8 segment step (`segment.PouwI8Add`,
`PouwSegDot_v2`), and `dither.PouwDitherBlock_v1` for comparison."""
from __future__ import annotations

import json
import os
import sys

import numpy as np

from verity.ml.circuits import boolean as BL
from verity.ml.circuits.boolean.forms import add, compress_columns, NOT, ZERO
from verity.ml.circuits.boolean.trace import definition
from verity.ml.circuits.defs import bind
from verity.ml.circuits.evaluation import bits as BS
from verity.ml.circuits.types import Array
from verity.core.protocols.pouw.circuit import dither as D
from verity.core.protocols.pouw.circuit import segment as SG

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import agg_f as F  # noqa: E402

BIT = BL.BIT
U16 = Array(16, BIT)


def _xsr(x, s):
    return [x[i] ^ x[i + s] if i + s < 16 else x[i] for i in range(16)]


def _mulc(C, x, c):
    cols = [[] for _ in range(16)]
    for b in range(16):
        if c >> b & 1:
            for i in range(16 - b):
                cols[i + b].append(x[i])
    rows = compress_columns(C, cols)
    return list(rows[:16]) + [ZERO] * (16 - len(rows[:16]))


def _rounds_bits(p):
    """xorshift-multiply rounds (s, c)... then a last xorshift, from a parameter tuple of agg_f's form."""
    def bits(C, z):
        x = list(z)
        for s, c in zip(p[0:-1:2], p[1:-1:2]):
            x = _xsr(x, s)
            x = _mulc(C, x, c)
        x = _xsr(x, p[-1])
        return C.commit(x)
    return bits


def _mix_def(name, p, ref):
    return definition(name, 1, _rounds_bits(p), (("z", U16),), U16, word=lambda z: int(ref(np.array([z]))[0]),
                      conformance="protocol-defined", doc=f"{name} on bits (research prototype, not a registered Definition)")


def _mulc_w(C, x, c, w):
    cols = [[] for _ in range(w)]
    for b in range(w):
        if c >> b & 1:
            for i in range(min(len(x), w - b)):
                cols[i + b].append(x[i])
    rows = compress_columns(C, cols)
    return list(rows[:w]) + [ZERO] * (w - len(rows[:w]))


def _m32a_bits(C, z):
    x = _mulc_w(C, list(z), 0x85EBCA6B, 32)
    x = [x[i] ^ x[i + 13] if i + 13 < 32 else x[i] for i in range(32)]
    x = _mulc_w(C, x, 0xC2B2AE35, 32)
    return C.commit([x[i] ^ x[i + 16] for i in range(16)])


AggM32a = definition("AggM32a", 1, _m32a_bits, (("z", U16),), U16, word=lambda z: int(F.m32a(np.array([z]))[0]),
                     conformance="protocol-defined", doc="m32a on bits (research prototype, not a registered Definition)")
AggMix = _mix_def("AggMix", F.MIX, F.mix)
AggMix3 = _mix_def("AggMix3", F.MIX3, F.mix3)
AggMix4 = _mix_def("AggMix4", F.MIX4, F.mix4)


def _sum_bits(C, *ws):
    cols = [[] for _ in range(16)]
    for w in ws:
        for i in range(16):
            cols[i].append(w[i])
    rows = compress_columns(C, cols)
    return C.commit(list(rows[:16]))


def sum_def(g: int):
    return definition(f"AggSum{g}", 1, _sum_bits, tuple((f"w{k}", U16) for k in range(g)), U16,
                      word=lambda *ws: sum(ws) & 0xFFFF, conformance="protocol-defined", doc="sum mod 2^16")


def ands(t) -> int:
    return BS.compile(t).counts["and"]


def main(out: str) -> None:
    res = {}
    for key, t, ref in (("mix", AggMix, F.mix), ("mix3", AggMix3, F.mix3), ("mix4", AggMix4, F.mix4),
                        ("m32a", AggM32a, F.m32a)):
        res[f"{key}_ands"] = ands(t)
        got = np.asarray(BS.evaluate_words(t, np.arange(1 << 16, dtype=np.uint64))).astype(np.uint32).reshape(-1)
        res[f"{key}_matches_reference_on_all_2^16"] = bool(np.array_equal(got, ref(F.Z)))
    res["sum_ands"] = {g: ands(sum_def(g)) for g in (2, 8, 64)}
    res["i8add_ands"] = ands(SG.PouwI8Add)
    res["segdot_ands_per_32"] = ands(SG.PouwSegDot)
    res["segdot0_ands_per_32"] = ands(SG.PouwSegDot0)
    res["dither_block_ands"] = {L: ands(bind(D.PouwDitherBlock, L=L, FIRST=False)) for L in (64, 256, 1024)}
    print(json.dumps(res, indent=1))
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/agg/ands.json")
