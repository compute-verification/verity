"""REDTEAM phase 17, Target 3 (#3884, PROVE-PATH): ServedStaticsTie_v1, one committed served-rows row a block of G steps,
read through the Lean verifier's registered reads. The PR's own test covers the per-field edits and an idle step at G = 1;
these add the angles the lead named that it doesn't: G not dividing P refused for every non-divisor (not only G = 2), the
split between the circuit builder (which trusts G) and the registration (which enforces the tiling), a whole multi-step
block at G = P (idle with a forged row, and each step held to its own Program), and rows committed under fresh salts
failing the registered read at the verifier's own host prediction (`served_rows.leaf_ok`).

The smallkv fixture's P is 7 (prime), so the only blocks are G = 1 (seven) and G = P (one); a genuine between-block
boundary at an intermediate G can't be built from it, so the multi-step cases use G = P."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from verity.core.protocols.certifier import compile_slice as CS, served_rows as SR, served_statics as SS

FIXTURE = Path(__file__).parent / "fixtures" / "compile_slice_smallkv.json"
COUNT, POS = 1 << SS.FLAGS.index("count"), 1 << SS.FLAGS.index("pos")
RUN_ROOT = bytes(range(64))


@pytest.fixture(scope="module")
def x():
    fx = json.loads(FIXTURE.read_text())
    S = SimpleNamespace(**CS.statics(fx["R"], fx["W"], fx["A"], fx["engine"], fx["IW"], Q=fx["Q"], D=fx["D"], P=fx["P"],
                                     K=fx["K"], SEG=fx["SEG"], CN=fx["CN"], CTW=fx["CTW"], CU=fx["CU"]))
    rows = [tuple(map(tuple, r)) for r in fx["rows"]]
    return SimpleNamespace(S=S, rows=rows, programs=SS.program_of(rows))


def test_every_g_not_dividing_p_is_refused_at_the_registration(x):
    """Not only G = 2: every G in 2 .. P that doesn't divide P (all of them, P = 7 being prime), and G = 0 and G = P + 1,
    are refused by `_blocks`, `block_bits` and `register` alike, so no off-tiling registration can be built."""
    S = x.S
    bad = [g for g in range(0, S.P + 2) if g < 1 or g > S.P or S.P % g]
    assert bad == [0, 2, 3, 4, 5, 6, 8]
    for g in bad:
        with pytest.raises(ValueError, match="G ."):
            SR._blocks(S, g)
        with pytest.raises(ValueError, match="G dividing P|G steps"):
            SR.block_bits(S, x.rows, g, 0)
        with pytest.raises(ValueError, match="G dividing P|G steps"):
            SR.register(S, [x.rows], RUN_ROOT, G=g)


def test_the_circuit_builder_trusts_g_while_the_registration_enforces_the_tiling(x):
    """`served_statics` builds a unit for any G >= 1, G = 3 (which does not divide P = 7) among them, so the circuit is
    not where the tiling is checked. It is fail-closed anyway: the matching registration can't exist (`register(G=3)`
    raises), so a unit built off the tiling has no registered leaf for the Lean reads to open."""
    st = SS.statics(x.S, G=3)
    assert st["G"] == 3
    assert SS.size(**st)["ands"] > 0 and dict(SS.rows(SimpleNamespace(**st)))["srb"] > 0
    with pytest.raises(ValueError, match="G dividing P"):
        SR.register(x.S, [x.rows], RUN_ROOT, G=3)


def test_an_idle_multi_step_block_with_a_forged_row_is_refused_at_g_equals_p(x):
    """At G = P a fully-idle block (every step None) is accepted, its served rows zero against idle Programs; forging one
    real row into one otherwise-idle step drops `count`, so the whole block is refused. The circuit agrees with the
    reference."""
    S, G = x.S, x.S.P
    st = SS.statics(S, G=G)
    T = SimpleNamespace(**st)
    real = next(r for r in x.rows if r is not None)
    idle_window = [None] * S.P
    forged_window = [real] + [None] * (S.P - 1)
    idle = SS.case(T, SR.block_bits(S, idle_window, G, 0), [None] * G)
    forged = SS.case(T, SR.block_bits(S, forged_window, G, 0), [None] * G)
    assert idle.verdict == SS.ACCEPT
    assert forged.verdict == SS.ACCEPT & ~COUNT and not forged.flags["count"]
    assert SS.evaluate(st, [idle, forged]) == [idle.verdict, forged.verdict]


def test_each_step_of_a_g_equals_p_block_is_held_to_its_own_program(x):
    """At G = P the single block still binds each step to its own Program: editing step k's start in the served rows
    drops `pos`, editing its n drops `count`, and the block is accepted only when every step matches. This exercises the
    per-step alignment inside a multi-step block (the PR's edits are all at G = 1)."""
    S, G = x.S, x.S.P
    st = SS.statics(S, G=G)
    T = SimpleNamespace(**st)
    honest = SS.case(T, SR.block_bits(S, x.rows, G, 0), x.programs)
    assert honest.verdict == SS.ACCEPT
    tried = 0
    for k, step in enumerate(x.rows):
        if step is None:
            continue
        tried += 1
        for at, want in ((3, SS.ACCEPT & ~POS), (4, SS.ACCEPT & ~COUNT)):
            edited = [list(map(list, s)) if s is not None else None for s in x.rows]
            edited[k][0][at] = edited[k][0][at] + 1
            window = [tuple(map(tuple, s)) if s is not None else None for s in edited]
            case = SS.case(T, SR.block_bits(S, window, G, 0), x.programs)
            assert case.verdict == want, (k, at)
            assert SS.evaluate(st, [case]) == [case.verdict]
    assert tried >= 2


def test_rows_committed_under_fresh_salts_fail_the_registered_read(x):
    """The 'unregistered' attack at the verifier's own host prediction: a prover commits the honest rows under fresh
    salts, so their b‖c differs from the one in the registered root. `leaf_ok` opens the registered root, so the fresh
    b‖c fails at every leaf while the registered one passes — the Lean registered reads would refuse the statement."""
    S = x.S
    served = SR.register(S, [x.rows], RUN_ROOT, G=1)      # the registration of record
    fresh = SR.register(S, [x.rows], RUN_ROOT, G=1)       # the same rows, fresh salts
    rec = served.record()
    assert served.registration.rows == fresh.registration.rows          # same rows ...
    assert served.registration.root != fresh.registration.root          # ... different salted root
    for i in range(rec["count"]):
        path = served.registration.open(i).path
        assert served.bc(i) != fresh.bc(i)
        assert SR.leaf_ok(rec, RUN_ROOT, i, served.bc(i), path)
        assert not SR.leaf_ok(rec, RUN_ROOT, i, fresh.bc(i), path)
