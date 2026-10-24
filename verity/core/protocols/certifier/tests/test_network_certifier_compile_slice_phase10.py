"""Red-team (REDTEAM-ADVICE, phase 10 target 3) against ADV-SPEC's compile slice (`compile_slice.CompileSlice_v1`, #3661),
on corpus-smallkv's first seven steps: does the composition close phase 9's bindings (3)-(5), the encoder's context and
frequencies, the cache hit fed into ReplayStep_v1 and the chain's ``used``/``taken``, and what does its ``rows`` bit bind?

- the context and the frequencies are wired: the cheapest context assignment and a near-certain codeword, both accepted
  by the coder piece alone, are refused by ``code``;
- ``used`` and ``taken`` are the step's own: a state row whose decision or queue head moves by one is refused;
- no witness field carries a wired value, and the ReplayStep_v1 inputs held at zero are exactly the cache counts;
- ``rows`` binds the run's step rows to the committed ``srv`` row (#3773, which closed this): once the served rows are
  committed, an otherwise-valid run whose rows are another advice's is refused at ``rows``, so a cheaper substituted
  advice can't pass; ``pub`` is the declared charge, no longer a digest of the prover's own rows;
- the code's position counter is sized to cover the worst case (``max(SEG, P·A·EK + 2)``), so a long code can't wrap it.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity.core.protocols.certifier import arith_encode as AE
from verity.core.protocols.certifier import compile_slice as CS
from verity.core.protocols.certifier import replay_chain as RC
from verity.core.protocols.certifier import replay_step as RS

FIXTURE = Path(__file__).parent / "fixtures" / "compile_slice_smallkv.json"
CODE, ROWS = 1 << CS.FLAGS.index("code"), 1 << CS.FLAGS.index("rows")
LOAD, STEPS = 1 << CS.FLAGS.index("load"), 1 << CS.FLAGS.index("steps")


@pytest.fixture(scope="module")
def slice_():
    fx = json.loads(FIXTURE.read_text())
    st = CS.statics(fx["R"], fx["W"], fx["A"], fx["engine"], fx["IW"], Q=fx["Q"], D=fx["D"], P=fx["P"], K=fx["K"],
                    SEG=fx["SEG"])
    sl = CS.Slice(tuple(map(tuple, fx["arrivals"])), tuple(fx["decisions"]), fx["steps"])
    rows = [tuple(map(tuple, r)) for r in fx["rows"]]
    return SimpleNamespace(st=st, S=SimpleNamespace(**st), sl=sl, rows=rows, honest=CS.case(SimpleNamespace(**st), sl))


def _span(S, row: str, name: str) -> tuple[int, int]:
    off = 0
    for n, k, w in CS.fields(S)[row]:
        if n == name:
            return off, k * w
        off += k * w
    raise KeyError(name)


def _with(S, c: CS.Case, row: str, name: str, bits, at: int = 0) -> CS.Case:
    """`c` with `bits` written into `row`'s field `name` at bit `at` of the field (the rest of the field kept)."""
    off, w = _span(S, row, name)
    assert at + len(bits) <= w
    r = list(c.rows[row])
    r[off + at:off + at + len(bits)] = list(bits)
    return replace(c, rows={**c.rows, row: r})


def _coded(S, c: CS.Case, seg) -> CS.Case:
    """`c` with the window's codeword `seg` and the charge its length."""
    c = _with(S, c, "wit", "seg", list(seg) + [0] * (S.SEG - len(seg)))
    return _with(S, c, "pub", "c", CS._bits(len(seg), CS.CW))


def _symbols(bits, ctxs):
    """The (bit, f0, f1) symbols the plain KT model gives `bits` under the context assignment `ctxs`."""
    n, out = {c: (0, 0) for c in range(CS.EC)}, []
    for b, c in zip(bits, ctxs):
        out.append((b, *AE.scaled(2 * n[c][0] + 1, 2 * n[c][1] + 1)))
        n[c] = (n[c][0] + (not b), n[c][1] + bool(b))
    return out


def test_the_context_and_frequencies_are_wired_so_a_cheaper_codeword_is_refused(slice_):
    """Bindings (3). Phase 9: KTContext_v1 takes any ``ctx < C`` and ArithEncodeStep_v1 any f0, f1 with f0 + f1 <= 2^16.
    The cheapest context assignment of the honest decisions codes them in 4 bits (honest: 8) and near-certain
    frequencies in 2; the coder piece accepts both. In the slice both fail ``code``: ctx is the in-circuit pass counter
    and f0/f1 are KTContext_v1's outputs from the in-circuit counts."""
    x = slice_
    dec = x.sl.decisions
    cheapest = min((len(CS.codeword(_symbols(dec, a))), a) for a in itertools.product(range(CS.EC), repeat=len(dec)))[1]
    by_context = _symbols(dec, cheapest)
    sure = [(b, 1, (1 << 16) - 1) if b else (b, (1 << 16) - 1, 1) for b in dec]
    forged = {name: CS.codeword(s) for name, s in (("context", by_context), ("frequencies", sure))}
    assert {k: len(v) for k, v in forged.items()} == {"context": 4, "frequencies": 2} and x.honest.length == 8
    assert all(AE.check_window(CS.EK, 1, s, forged[k]) for k, s in (("context", by_context), ("frequencies", sure)))

    got = CS.evaluate(x.st, [x.honest] + [_coded(x.S, x.honest, seg) for seg in forged.values()])
    assert got[0] == CS.ACCEPT
    assert all(not v & CODE and v != CS.ACCEPT for v in got[1:]), [hex(v) for v in got]


def test_used_and_taken_are_the_steps_own_so_a_moved_head_is_refused(slice_):
    """Bindings (5). Phase 9: on a non-last step ReplayAdvance accepts a band of ``used`` and ``taken``. In the slice they
    are ReplayStep_v1's outputs, so every state row whose decision head (``dh``) or queue head (``qh``) moves by one,
    the rest of the witness kept, is refused: S_1 by ``load``, every later row by ``steps``."""
    x = slice_
    Sc = CS.chain(x.S)
    Z = RC.widths(Sc)
    sc = CS.scan(x.S, x.sl)
    width = RC.width(Sc)
    lim = {"dh": 1 << Z.DW, "qh": 1 << dict((n, w) for n, _k, w in RC.row_fields(Sc))["qh"]}
    moved, where = [], []
    for u, row in enumerate(sc.states):
        for head, d in itertools.product(("dh", "qh"), (-1, 1)):
            v = getattr(row, head) + d
            if 0 <= v < lim[head]:
                moved.append(_with(x.S, x.honest, "wit", "state", RC.encode(replace(row, **{head: v}), Sc), at=u * width))
                where.append(u)
    assert len(moved) >= 2 * len(sc.states)
    got = CS.evaluate(x.st, moved)
    assert all(not v & (LOAD if u == 0 else STEPS) for u, v in zip(where, got)), list(zip(where, map(hex, got)))


def test_no_witness_field_carries_a_wired_value_and_only_the_cache_counts_are_held_at_zero(slice_):
    """Bindings (3)-(5) structurally, and (4)'s stand-in. The witness is the state and window rows, each step's arrivals
    and bits, and the codeword; no row has a context, frequency, ``used``, ``taken`` or cache-count field (and #3773's
    committed ``srv`` row dropped the in-circuit digest salt the witness once carried). The
    ReplayStep_v1 inputs the slice feeds neither from a state row, a front nor the bits, and so holds at zero, are
    exactly ``hit``, ``evict`` and ``ret``: the phase 9 hit and return levers have no input to ride on (at the cost of
    every slice that hits a prefix or stops a request)."""
    S = slice_.S
    Sc, St = CS.chain(S), CS.step_statics(S)
    assert [n for n, _k, _w in CS.fields(S)["wit"]] == ["state", "window", "arr", "bits", "seg"]
    names = {n for fs in CS.fields(S).values() for n, _k, _w in fs}
    assert not names & {"ctx", "f0", "f1", "used", "taken", "hit", "evict", "ret", "pc", "pos"}
    fed = {n for n, _k, _w in RC.row_fields(Sc)} | {n for n, _k, _w in RC.front_fields(Sc)} | {"bits"}
    assert {n for n, _k, _w in RS._in_fields(St)} - fed == {"hit", "evict", "ret"}


def test_the_committed_srv_binds_the_rows_so_a_substituted_advice_is_refused(slice_):
    """Phase 10 (4), closed by #3773. The finding was that ``rows`` bound the rows to the prover's own ``pub.digest``, a
    self-consistency check, so a cheaper substituted advice passed under its own digest. Now the served rows are a
    committed ``srv`` row (held by b‖c to served_rows' registered leaf), and ``pub`` is only the declared charge: an
    otherwise-valid run (load, steps, code and charge all pass) whose committed ``srv`` is another advice's rows — here
    the six-passes run's, which codes in 5 bits to honest's 8 — is refused at exactly ``rows`` (0x0F), so the cheaper
    advice can't be substituted once the served rows are committed. The arrival negative still passes the circuit; its
    refusal is ``cert``'s native b‖c alone."""
    x = slice_
    S, sl = x.S, x.sl
    cheap = CS.case(S, CS.Slice(sl.arrivals, (0,) * len(sl.decisions), sl.steps))
    swapped = CS.case(S, sl, served=cheap.steps)
    cert = list(sl.arrivals)
    cert[1] = (cert[1][0], cert[1][1] + 1, cert[1][2])
    arrival = CS.case(S, sl, cert=cert)
    assert cheap.length < x.honest.length and tuple(cheap.steps) != tuple(x.honest.steps)
    assert swapped.flags == {"load": True, "steps": True, "code": True, "charge": True, "rows": False}
    got = CS.evaluate(x.st, [x.honest, swapped, arrival])
    assert got == [CS.ACCEPT, CS.ACCEPT & ~ROWS, CS.ACCEPT]
    assert [c.verdict for c in (x.honest, swapped, arrival)] == got


def test_the_position_counter_covers_the_worst_case_not_SEG_alone(slice_):
    """Hardening (phase 10 (5), now sized to cover the worst case). The code's position is a PW-bit counter,
    PW = bitlen(max(SEG, P·A·EK + 2)): every ask consumes at most EK bits and the finish 2, so a code of all P·A asks
    can't wrap it. At the pinned statics the worst case P·A·EK + 2 dominates SEG, so PW is 8 (not SEG's 5), and the
    longest code of any decision vector up to D under any context assignment, 12 bits, is below SEG and 2^PW = 256.
    Unlike the earlier SEG-only width, PW now reads P and A: scaling the steps grows it, so a longer run can't wrap."""
    S = slice_.S
    pw = CS._pos_width(S)
    worst = S.P * S.A * CS.EK + 2
    longest = max(len(CS.codeword(_symbols(bits, a))) for n in range(S.D + 1)
                  for bits in itertools.product((0, 1), repeat=n) for a in itertools.product(range(CS.EC), repeat=n))
    assert (pw, longest) == (8, 12) and worst >= S.SEG and longest <= S.SEG < 1 << pw
    big = SimpleNamespace(**{**slice_.st, "P": S.P * 8})
    assert CS._pos_width(big) > pw
