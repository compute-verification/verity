"""PoUW's tile checks register the Boolean versions of the words they call (`verity.circuits.ops.boolean.words`), and the four
Keccak words' builders compose, in `hashes`, to core's TurboSHAKE128."""
from __future__ import annotations

import random
import subprocess
import sys

from verity.circuits.boolean.forms import ONE, Circuit, const_bits
from verity.circuits.ops import words as W
from verity.circuits.ops.boolean import words as B
from verity.commitments.crypto.turboshake import turboshake128
from verity.protocols.accounting.compute.pouw.prover.circuit import hashes as H

M32 = (1 << 32) - 1
KECCAK = ((B.PouwXor32, W.PouwXor32), (B.PouwChi32, W.PouwChi32), (B.PouwRotl64Lo, W.PouwRotl64Lo),
          (B.PouwRotl64Hi, W.PouwRotl64Hi))


def test_building_a_tile_check_registers_them():
    """Flock's lowering finds a primitive's Boolean Definition only in core's registry, so importing a tile check
    (`pc8`, through `tile_units`) must register all fifteen (the hiding tile's digest calls the four Keccak words, the cap the five
    window words), in a fresh interpreter where nothing else imported `boolean`."""
    ids = [w.id.replace("_v1", "_v2") for w in (W.PouwF32SqrtRn, W.PouwETilde, W.PouwBf16Neg, W.PouwSqAcc, W.PouwSqCeil,
                                                W.PouwSqGe, W.PouwBlackwellE4m3Window, W.PouwHopperE4m3Window,
                                                W.PouwWindowWord, W.PouwBlackwellE4m3SaltDead, W.PouwHopperE4m3SaltDead)
                                               + tuple(w for _d, w in KECCAK)]
    code = ("import sys, verity.protocols.accounting.compute.pouw.prover.circuit.pc8\nfrom verity.circuits.defs import REGISTRY\n"
            f"missing = [i for i in {ids!r} if i not in REGISTRY]\nsys.exit(','.join(missing) if missing else 0)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


def test_the_keccak_words_compose_to_turboshake128(monkeypatch):
    """The four builders in place of their word primitives wherever `hashes` calls them (θ's rotations by 1, ρ's offsets,
    χ, ι's constants) compute core's TurboSHAKE128 on a message of input lanes, and every AND they build is χ's: 32 per
    half-lane per round, and as many commits of its outputs, so χ's operands are θ's outputs, 11 committed bits at most;
    without the commits χ's XOR terms compound round over round, and the 12 rounds' AND operands grow to thousands of bits
    each."""
    build = {w.id: d.build for d, w in KECCAK}

    class OnForms(H.Emit):
        """`hashes.Emit` on core's forms: a half-lane is 32 forms, a word primitive its Boolean Definition's builder, and the
        permutation `hashes`' own body on this emitter."""

        def __init__(self, C):
            super().__init__(None)
            self.B, self.C = self, C

        def c(self, v):
            return const_bits(v & M32, 32)

        def call(self, fn, *args):
            if fn is H.KeccakP1600R12:
                return H._keccak_body(self, None, args[0], 12)
            if all(isinstance(a, int) for a in args):
                return fn.evaluate(*args)
            return build[fn.id](self.C, *(self.g(a) for a in args))

        def array(self, halves):
            return [self.g(h) for h in halves]

    monkeypatch.setattr(H, "Emit", lambda e: e)
    C, n, T = Circuit(cse=True), 20, 8
    lanes = [(C.input(f"lo{i}", 32), C.input(f"hi{i}", 32)) for i in range(n)]
    digest = H.turboshake128(OnForms(C), lanes, 32, 0x1F)
    ands = [d for k, d in zip(C.kind, C.data) if k == "and"]
    assert sum(1 for d in ands if d[1] is not ONE) == sum(1 for d in ands if d[1] is ONE) == 12 * 25 * 2 * 32
    assert max(len(x.s) for d in ands if d[1] is not ONE for x in d) == 11 and max(len(d[0].s) for d in ands if d[1] is ONE) == 12
    R = random.Random(20261009)
    msgs = [R.randbytes(8 * n) for _ in range(T)]
    vals = {f"{part}{i}": [sum((int.from_bytes(m[8 * i:8 * i + 8], "little") >> (32 * h + j) & 1) << t
                                for t, m in enumerate(msgs)) for j in range(32)]
            for i in range(n) for h, part in enumerate(("lo", "hi"))}
    _val, ev = C.values(vals, T)
    for t, m in enumerate(msgs):
        got = b"".join(sum((ev(b) >> t & 1) << j for j, b in enumerate(half)).to_bytes(4, "little") for half in digest)
        assert got == turboshake128(m, 0x1F, 32)
