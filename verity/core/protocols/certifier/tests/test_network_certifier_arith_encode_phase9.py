"""Red-team (REDTEAM-ADVICE, phase 9 target 2) against ENC-CODE's in-circuit encoder (`arith_encode`), the counting
variant that charges a window's committed segment length. These cases do not fault the encoder against its own
reference (the author's suite does that); they measure the degrees of freedom the piece leaves to a prover under the
DECIDED composition, where the charge is the coded length and must be <= the declared public c.

Findings, each confirmed executably below:

1. The model context `ctx` is a FREE WITNESS. `KTContext_v1` checks only `ctx < C`; nothing here ties it to the
   replay's position (the module says it is "the replay's (plain: min(passes, 2))"). A prover who knows a decision bit
   can route it to whichever context makes it cheapest, under-charging the window. The tie of `ctx` to the replay
   position is the (unbuilt) compile unit's job.
2. The step's frequencies `f0`/`f1` are a FREE WITNESS in `ArithEncodeStep_v1`: checked only `>= 1` and sum `<= 2^16`.
   A near-certain model codes a known bit in ~0 segment bits. The tie of `f0`/`f1` to `KTContext_v1`'s output is the
   compile unit's wiring; in isolation the step charges whatever frequencies it is handed.
3. The window boundary is CLOSED for interval state: the coder restarts at `fresh()` each window and the finish
   terminates the window's segment exactly, so no un-terminated interval state carries across a boundary to amortize
   charge. What this piece does NOT pin is the decision->window assignment (A5's charge rule), which is the replay's
   step position, set by the compile unit.
"""
from __future__ import annotations

import random

from verity.core.protocols.certifier import arith_encode as AE

K = AE.K_MAX


class _WNC:
    """A plain 32-bit WNC encoder (the author suite's helper), to turn a symbol list into the segment the circuit
    would check, so its length is the charge."""

    def __init__(self):
        self.low, self.high, self.pending, self.used, self.bits = 0, AE.TOP, 0, False, []

    def emit(self, b):
        self.bits += [b] + [1 - b] * self.pending
        self.pending = 0

    def encode(self, cum, freq, total):
        if freq == total:
            return
        self.used, r = True, self.high - self.low + 1
        self.high, self.low = self.low + r * (cum + freq) // total - 1, self.low + r * cum // total
        while True:
            if self.high < AE.HALF:
                self.emit(0)
            elif self.low >= AE.HALF:
                self.emit(1)
                self.low, self.high = self.low - AE.HALF, self.high - AE.HALF
            elif self.low >= AE.Q1 and self.high < 3 * AE.Q1:
                self.pending += 1
                self.low, self.high = self.low - AE.Q1, self.high - AE.Q1
            else:
                break
            self.low, self.high = 2 * self.low, 2 * self.high + 1

    def finish(self):
        if self.used:
            self.pending += 1
            self.emit(0 if self.low < AE.Q1 else 1)
        return self.bits


def _cft(bit, f0, f1):
    """ArithEncodeStep_v1's (cum, freq, total) for a bit under (f0, f1), as `_symbol(BIN=1)` lays it out (`_segment`)."""
    return (f0 if bit else 0, f1 if bit else f0, f0 + f1)


def _segment(bit_syms):
    """The committed segment a list of (bit, f0, f1) binary symbols produces, the finish included."""
    e = _WNC()
    for bit, f0, f1 in bit_syms:
        e.encode(*_cft(bit, f0, f1))
    return e.finish()


def _charge(bit_syms):
    return len(_segment(bit_syms))


def test_context_is_a_free_witness_so_the_charge_is_the_provers_to_lower():
    """Finding 1. KTContext_v1 accepts any `ctx < C`. A prover who knows each bit routes it to its cheapest context;
    the resulting segment is strictly shorter than the one the replay's own context would charge, and the model unit
    accepts every adversarial (ctx, bit) step."""
    C, NW = 3, 24
    # a skewed start: context 0 favours bit 0, context 2 favours bit 1, context 1 is the replay's "honest" context.
    priors = [(100, 1), (1, 1), (1, 100)]
    bits = [0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1]

    def run(pick):
        counts, bit_syms, kt_cases = [tuple(p) for p in priors], [], []
        for b in bits:
            ctx = pick(b)
            kt_cases.append((tuple(counts), 1, ctx, b))
            ok, f0, f1, counts = AE.reference_kt(C, NW, counts, 1, ctx, b)
            assert ok
            bit_syms.append((b, f0, f1))
        return _charge(bit_syms), kt_cases

    honest_bits, _ = run(lambda b: 1)                          # the replay's single context, bit 1 is expensive there
    cheap_bits, cheap_cases = run(lambda b: 2 if b else 0)     # route each bit to the context that makes it likely

    # the model unit accepts EVERY adversarial context choice (the piece never ties ctx to the replay position).
    assert AE.evaluate_kt(C, NW, cheap_cases) == [AE.reference_kt(C, NW, *c) for c in cheap_cases]
    assert all(ok for ok, *_ in AE.evaluate_kt(C, NW, cheap_cases))

    # and the free choice charges strictly fewer segment bits for the same decisions.
    assert cheap_bits < honest_bits
    print(f"\n[enc-code] free-context under-charge: honest ctx charges {honest_bits} bits, "
          f"the prover's per-bit context {cheap_bits} bits ({honest_bits - cheap_bits} saved over {len(bits)} decisions)")


def test_step_frequencies_are_a_free_witness():
    """Finding 2. ArithEncodeStep_v1 checks only f0,f1 >= 1 and f0+f1 <= 2^16; it charges whatever frequencies it is
    handed. A near-certain model codes a known bit in far fewer segment bits than a uniform one, and the step accepts
    both. The tie of f0/f1 to KTContext_v1's output is the compile unit's wiring."""
    bits = [0] * 20
    uniform = [(b, 1, 1) for b in bits]                      # ~1 segment bit a decision
    certain = [(b, AE.MAX_TOTAL - 1, 1) for b in bits]       # p(0) ~ 1 - 2^-16: ~0 bits a decision

    uc, cc = _charge(uniform), _charge(certain)
    assert cc < uc

    # the step accepts the skewed frequencies on every symbol (matches the reference), so nothing here refuses them.
    segment = _segment(certain)
    st, pos, cases = AE.fresh(), 0, []
    for s in certain:
        look = (segment[pos:pos + K] + [0] * K)[:K]
        cases.append((st, 1, s, look))
        nxt, n = AE.reference_step(K, 1, st, 1, s, look)
        st, pos = nxt, pos + n
    got = AE.evaluate_steps(K, 1, cases)
    assert all(g[0].ok for g in got)
    print(f"\n[enc-code] free-frequency under-charge: a uniform model charges {uc} bits, a near-certain model {cc} bits "
          f"for the same {len(bits)} decisions")


def test_window_boundary_is_closed_for_interval_state():
    """Finding 3. The coder restarts at fresh() each window and the finish terminates the segment exactly, so a window
    cannot leave interval state un-terminated to amortize the charge into a neighbour. A segment that does not end where
    the finish does is refused (both a bit short and a bit long)."""
    rng = random.Random(0)
    for _ in range(40):
        bit_syms = []
        for _ in range(rng.randrange(1, 12)):
            t = rng.choice([2, 3, 17, 1000, AE.MAX_TOTAL])
            f0 = rng.randrange(1, t)
            bit_syms.append((rng.randrange(2), f0, t - f0))
        seg = _segment(bit_syms)
        assert AE.check_window(K, 1, bit_syms, seg)
        # one bit short (stop before the finish) or one bit long (carry past it) is refused: the window owns exactly
        # its own terminated segment, nothing leaks across the boundary.
        assert not AE.check_window(K, 1, bit_syms, seg[:-1])
        assert not AE.check_window(K, 1, bit_syms, seg + [rng.randrange(2)])
