"""DEVELOPMENT BINDING, NOT FOR REVIEW: scratch C-Flock pieces for the PoUW word primitives Keccak-f[1600] is written in
(`verity.ml.protocols.pouw.circuit.words`: PouwXor32_v1, PouwChi32_v1, PouwRotl64Lo_v1, PouwRotl64Hi_v1), which C-Flock has no piece for
on this branch. `stage.py` puts them in `ir_lower.PIECES` for the hiding tile only, and every result staged with them is
labelled 'hash scratch'. Xor is free, χ is 32 ANDs and 32 committed bits, and a rotation by a constant is wiring; a rotation by a variable count
is refused (Keccak's are constants). χ's output is committed (`sha512_circuit.commit`, one row a bit), or the XOR forms
would compound round over round. `check()` holds each to its primitive on random words."""
import random

from verity.ml.commitments.gates.sha512 import commit
from verity.ml.protocols.pouw.circuit import words as W


def _const(word) -> int:
    if not all(w.is_const() for w in word):
        raise NotImplementedError("a PouwRotl64 by a variable count (Keccak-f rotates by constants)")
    return sum(w.c << i for i, w in enumerate(word)) & 63


def _rotl(lo, hi, c):
    x, r = list(lo) + list(hi), _const(c)
    return [x[(i - r) % 64] for i in range(64)]


PIECES = {"PouwXor32_v1": lambda C, a, b: [x ^ y for x, y in zip(a, b)],
          "PouwChi32_v1": lambda C, a, b, c: commit(C, [x ^ C.AND(y.inv(), z) for x, y, z in zip(a, b, c)]),
          "PouwRotl64Lo_v1": lambda C, lo, hi, c: _rotl(lo, hi, c)[:32],
          "PouwRotl64Hi_v1": lambda C, lo, hi, c: _rotl(lo, hi, c)[32:]}


def check(n: int = 512, seed: int = 20261003) -> dict:
    from verity.ml.circuits.boolean.forms import const_bits
    from verity_flock import gf2
    from verity_flock import ir_lower as IL

    rng = random.Random(seed)

    def run(piece, widths, vecs):
        C = gf2.Circuit()
        ins = [C.input(f"i{k}", w) for k, w in enumerate(widths)]
        C.output("o", piece(C, *ins))
        outs, _ = C.evaluate({f"i{k}": IL.slice_words([v[k] for v in vecs], w) for k, w in enumerate(widths)}, len(vecs))
        return [int(x) for x in IL.unslice_words(outs["o"], len(vecs))]

    vecs = [tuple(rng.getrandbits(32) for _ in range(3)) for _ in range(n)]
    bad = {"PouwXor32_v1": sum(g != W.PouwXor32.evaluate(a, b)
                               for (a, b, _), g in zip(vecs, run(PIECES["PouwXor32_v1"], (32, 32), vecs))),
           "PouwChi32_v1": sum(g != W.PouwChi32.evaluate(*v) for v, g in zip(vecs, run(PIECES["PouwChi32_v1"], (32,) * 3, vecs)))}
    for name, prim in (("PouwRotl64Lo_v1", W.PouwRotl64Lo), ("PouwRotl64Hi_v1", W.PouwRotl64Hi)):
        bad[name] = 0
        for c in range(64):
            got = run(lambda C, lo, hi, c=c: PIECES[name](C, lo, hi, const_bits(c, 32)), (32, 32), vecs[:64])
            bad[name] += sum(g != prim.evaluate(lo, hi, c) for (lo, hi, _), g in zip(vecs[:64], got))
    return bad


if __name__ == "__main__":
    print(check())
