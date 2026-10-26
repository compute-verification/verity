"""fold_sat.py's instruction models and CEGIS on small menus: the targets are segment.py's folds, the models match their
reference semantics, a lower bound comes out UNSAT and the honest program comes out SAT (verified for all inputs)."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import z3

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "theory"))
import fold_sat as F  # noqa: E402
from verity.core.protocols.pouw.circuit import segment as S  # noqa: E402


def _rest_of_chain0(y, z):
    """Chain 0's remaining entries of a set (t = 2, 4, 6, e = 0) with zero words."""
    for t in range(2, S.MI * S.NI, 2):
        u = y ^ S.KK[t]
        y = (u * S.KA[t] + z) & F.M32
        z = (S.rotl(u, S.ROT[t]) + y) & F.M32
    return [y, z]


def test_targets_are_segment_folds():
    rng = random.Random(7)
    for _ in range(50):
        y, z, p, w0, w1 = (rng.getrandbits(32) for _ in range(5))
        st = [0] * (2 * S.NCH)
        st[0], st[S.NCH] = y, z
        words = [0] * 32
        words[0] = p
        out = S._duo_absorb(st, words)
        assert _rest_of_chain0(*F.TARGETS["duo"].eval([y, z, p])) == [out[0], out[S.NCH]]
        out = S._fx1_absorb(st, words)
        assert _rest_of_chain0(*F.TARGETS["fx1"].eval([y, z, p])) == [out[0], out[S.NCH]]
        e0, e1 = S.li_entries(0)[:2]
        words = [0] * 32
        words[e0], words[e1] = w0, w1
        lst = [0] * S.NCH
        lst[0] = y
        # chain 0's first half step alone: the second half step with zero words undone by recomputing it
        half = F.TARGETS["li"].eval([y, w0, w1])[0]
        assert S._li_absorb(lst, words)[0] == ((half ^ S.LK[0][1]) * S.LA[0][1]) & F.M32
        assert F.TARGETS["li_split2"].eval([y, w0 - 5, 5, w1 - 9, 9]) == [half]


def _eval_op(op, a, b, c, d=0, **kw):
    slot = F.Slot(0, 4, (op,))
    avail = [z3.BitVecVal(v, 32) for v in (a, b, c, d)]
    s = z3.Solver()
    s.add([slot.sel[j] == j for j in range(4)])
    for name, val in kw.items():
        s.add(getattr(slot, name) == val)
    lo, hi = slot.outputs(avail)
    assert s.check() == z3.sat
    m = s.model()
    return m.eval(lo, model_completion=True).as_long(), m.eval(hi, model_completion=True).as_long()


def test_instruction_models():
    rng = random.Random(3)
    for _ in range(10):
        a, b, c, d = (rng.getrandbits(32) for _ in range(4))
        assert _eval_op("LOP3", a, b, c, lut=0x96)[0] == a ^ b ^ c
        assert _eval_op("IMAD", a, b, c)[0] == (a * b + c) & F.M32
        assert _eval_op("IMAD.HI", a, b, c, signed=False)[0] == ((a * b >> 32) + c) & F.M32
        wide = a * b + (d << 32 | c)
        assert _eval_op("IMAD.WIDE", a, b, c, d, signed=False) == (wide & F.M32, (wide >> 32) & F.M32)
        assert _eval_op("LEA.HI", a, a, c, n=13)[0] == (S.rotl(a, 13) + c) & F.M32
        assert _eval_op("SHF", a, b, c, n=7, mode=0)[0] == ((b << 32 | a) >> 7) & F.M32
        assert _eval_op("SHF", a, b, 40, mode=0, fn=3)[0] == b   # clamped: min(40, 32)
        assert _eval_op("SHF", a, b, 40, mode=0, fn=1)[0] == ((b << 32 | a) >> 8) & F.M32
        assert _eval_op("PRMT", a, b, 0x3210)[0] == a
        assert _eval_op("PRMT", a, b, 1, fn=1)[0] == ((b << 32 | a) >> 8) & F.M32   # f4e, c[1:0] = 1
        assert _eval_op("SEL", a, b, c, d, fn=0, signed=False)[0] == (c if a < b else d)


def test_li_one_instruction_unsat_small_menu():
    r = F.synth(F.TARGETS["li"], 1, ops=("LOP3", "IADD3", "IMAD"), timeout_s=60)
    assert r["verdict"] == "unsat"


def test_li_honest_found():
    r = F.synth(F.TARGETS["li"], 2, ops=("LOP3", "IMAD"), timeout_s=60)
    assert r["verdict"] == "sat", r
