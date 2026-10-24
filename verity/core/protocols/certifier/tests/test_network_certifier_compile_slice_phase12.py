"""REDTEAM phase 12, target 4 (REDTEAM-ADVICE, note:network-accounting/20261024T0558Z-finding-redteam-v3) against #3661's
compile slice (`compile_slice.CompileSlice_v1`) with the phase 9 bindings wired: the context computed in-circuit (the
plain counter), the frequencies KTContext_v1's outputs, and the heads moved by ReplayStep_v1's own ``taken`` and ``used``.

Every interface REDTEAM's phase 9 found free in a piece alone is closed end to end by the composition. ADV-SPEC rebuilt
them as `compile_slice.phase9`; here each is run through the *compiled circuit* (`evaluate`, not only the reference), and
the charge is shown pinned to the honest coded length from both sides, so the context/frequency findings (a cheaper
charge) and the schedule findings (a shorter run) buy nothing on the slice.

The one residual is a named stand-in, not an exploit: the cache counts (ReplayStep_v1's ``hit``/``evict``/``ret`` held at
0, CacheStep_v1 unwired) refuse a claimed hit by the 0 and by ``rows``, but an honest hit can't be taken either; the slice
class hits no prefix, so this is a declared delegation, carried here as the ``hit`` case's refusal.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity.core.protocols.certifier import compile_slice as CS

FIXTURE = Path(__file__).parent / "fixtures" / "compile_slice_smallkv.json"
SALT = bytes(range(32))


@pytest.fixture(scope="module")
def x():
    fx = json.loads(FIXTURE.read_text())
    st = CS.statics(fx["R"], fx["W"], fx["A"], fx["engine"], fx["IW"], Q=fx["Q"], D=fx["D"], P=fx["P"], K=fx["K"],
                    SEG=fx["SEG"])
    S = SimpleNamespace(**st)
    sl = CS.Slice(tuple(map(tuple, fx["arrivals"])), tuple(fx["decisions"]), fx["steps"])
    rows = [tuple(map(tuple, r)) for r in fx["rows"]]
    return SimpleNamespace(st=st, S=S, sl=sl, rows=rows, digest=CS.rows_digest(S, rows, SALT))


def test_every_phase9_interface_is_refused_by_the_compiled_circuit(x):
    """The five phase 9 interfaces, each with the clear recompile's digest, are refused, and the compiled circuit's
    verdict is the reference's: no witness field carries a context, a frequency, a cache count or a consumed count, so a
    piece's freedom is gone once the pieces are composed. context/frequency lose on `code` and `charge` (a codeword the
    in-circuit contexts don't produce, a charge below the true length); hit/reread/untaken on `steps` and `rows` (the
    heads move by ReplayStep's own taken and used, and the rows are the clear recompile's)."""
    cases = {k: CS.case(x.S, x.sl, SALT, digest=x.digest, **kw) for k, kw in CS.phase9(x.S, x.sl).items()}
    assert tuple(cases) == CS.PHASE9
    got = CS.evaluate(x.st, list(cases.values()))
    assert got == [c.verdict for c in cases.values()]
    assert all(v != CS.ACCEPT for v in got)
    refused = {k: {f for f, ok in c.flags.items() if not ok} for k, c in cases.items()}
    assert refused == {"context": {"code", "charge"}, "frequency": {"code", "charge"},
                       "hit": {"steps", "rows"}, "reread": {"steps", "rows"}, "untaken": {"steps", "rows"}}


def test_the_charge_cannot_drop_below_the_honest_coded_length(x):
    """The context/frequency findings were a cheaper charge; the slice pins the charge to the honest coded length (8
    bits), and neither side moves it:
    - the honest codeword with a charge one below its length is refused by `charge` alone — the length is the circuit's,
      not the prover's declaration;
    - the phase 9 `context` codeword (4 bits) declared at the honest charge and digest is refused by `code` alone — the
      codeword has to decode under the in-circuit contexts, which the prover can't move, so a cheaper one never passes
      and so never lowers the charge.
    The compiled circuit agrees with the reference on both."""
    honest = CS.case(x.S, x.sl, SALT)
    assert honest.verdict == CS.ACCEPT and honest.length == 8

    low = CS.case(x.S, x.sl, SALT, digest=x.digest, charge=honest.length - 1)
    assert {f for f, ok in low.flags.items() if not ok} == {"charge"}

    cheap = CS.phase9(x.S, x.sl)["context"]["seg"]
    forced = CS.case(x.S, x.sl, SALT, digest=x.digest, charge=honest.length, seg=cheap)
    assert {f for f, ok in forced.flags.items() if not ok} == {"code"}

    assert CS.evaluate(x.st, [low, forced]) == [low.verdict, forced.verdict]


def test_combining_interfaces_only_stacks_refusals(x):
    """A cheaper codeword and a head left in place together (phase 9 `context` and `untaken`) refuse by the union of
    their checks, not by canceling: the combined verdict leaves only `load` set, so composing the leaks opens no gap the
    circuit misses."""
    p9 = CS.phase9(x.S, x.sl)
    combo = CS.case(x.S, x.sl, SALT, digest=x.digest, **{**p9["context"], **p9["untaken"]})
    refused = {f for f, ok in combo.flags.items() if not ok}
    assert refused == {"code", "charge", "steps", "rows"}
    assert CS.evaluate(x.st, [combo]) == [combo.verdict] and combo.verdict == (1 << CS.FLAGS.index("load"))
