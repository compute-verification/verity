"""The Pearl-C period credit rule (CMP-82, `pearl_c_period`): a period earns only when every clause holds.

`verity/core/protocols/pouw/lean/check/tests/pouw_period_cases.py` draws the cases (an honest period of one to three
Pearl-C salt epochs and their passes, then one mutation aimed at one clause); that package's `tests/test_difftest.py`
holds the Lean `PouwCheck.pearlCPeriodCredit` to this module on them. The tests here hold this module to the rule: an
honest period's passes are credited, two passes of one epoch whose call indices continue one another among them, and
each mutation is refused by the clause it aims at."""
from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

import pytest

from verity.core.protocols.pouw import core_check as CC
from verity.core.protocols.pouw import pearl_c_period as PP

ROOT = Path(__file__).resolve().parents[5]
PKG = ROOT / "verity/core/protocols/pouw/lean/check"


def _load(name: str):
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, PKG / f"tests/{name}.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


_load("pouw_rounds_cases")
_load("pouw_pass_cases")
cases = _load("pouw_period_cases")

#: a part of each clause's reason, past the code
REASON = {PP.FORM: "", PP.EXACT: "the period's sessions are not exactly its passes' epochs, so it earns nothing: ",
          PP.SALT_ONLY: PP.NOT_SALT_ONLY, PP.SESSION: "", PP.PASS: " is refused, so the period earns nothing: ",
          PP.INDEX: PP.OUT_OF_TURN, PP.REPEATED_CALL: ""}


def _credit(d: dict) -> tuple[list[dict] | None, tuple[str, str] | None]:
    return PP.pearl_c_period_credit(*cases.args(d))


def _manifests(d: dict) -> list[dict]:
    return [json.loads(bytes.fromhex(p["manifest"])) for p in d["passes"]]


def test_the_refusals_are_numbered_by_the_rules_clauses():
    """The codes, 0 the inputs' form and 1 to 6 the rule's clauses in CMP-82's order, and every clause has a mutation
    the cases aim at it."""
    assert PP.REFUSALS == ("form", "exact", "salt-only", "session", "pass", "index", "repeated-call")
    assert set(cases.CLAUSE.values()) == set(PP.REFUSALS)


@pytest.mark.parametrize("seed", range(6))
def test_an_honest_period_credits_its_passes_in_order(seed):
    """An honest Pearl-C period, its passes of one to three salt epochs over one or two weights' roots numbered on from
    the lane's earlier credited epochs, is credited exactly its passes' manifests, in the order presented."""
    d = cases.instance(random.Random(f"honest/{seed}"))
    got, refused = _credit(d)
    assert refused is None and got == _manifests(d), refused


@pytest.mark.parametrize("seed", range(4))
def test_two_passes_of_one_epoch_with_continuing_indices_are_credited(seed):
    """Two passes of one salt epoch, the second's call indices continuing the first's, read one session and are both
    credited; once they share a call index, the period is refused (``repeated-call``)."""
    d = cases.instance(random.Random(f"two passes/{seed}"), epochs=1, passes=2)
    got, refused = _credit(d)
    assert refused is None and len(got) == 2, refused
    first, second = got
    assert first["session"] == second["session"] and first["pass_index"] == second["pass_index"]
    a, b = ([u["index"] for u in m["matmuls"]] for m in (first, second))
    assert max(a) + 1 == min(b) and sorted(a + b) == list(range(min(a), max(b) + 1))
    d = cases.instance(random.Random(f"two passes/{seed}"), "repeated-call", epochs=1, passes=2)
    _, refused = _credit(d)
    assert refused is not None and refused[0] == PP.REPEATED_CALL and PP.REPEATED in refused[1], refused


@pytest.mark.parametrize("name", cases.NAMES)
def test_every_mutation_is_refused_by_its_clause(name):
    """Each mutation is refused with its clause's code (`pouw_period_cases.CLAUSE`): the inputs' form; a period the
    ledger doesn't close complete with exactly the passes' epochs (an epoch no pass presents, an abandoned complete
    session among them); a session not reserved for the salt coin alone (an H1 session's); a pass whose session doesn't
    count toward the period or is past its pin floor; a refused pass; an epoch out of its weights' root's turn; and two
    passes of one epoch sharing a call index (a pass presented twice among them)."""
    want = cases.CLAUSE[name]
    for seed in range(3):
        got, refused = _credit(cases.instance(random.Random(f"{name}/{seed}"), name))
        assert got is None and refused is not None and refused[0] == want and REASON[want] in refused[1], \
            (name, seed, refused)


def test_an_abandoned_salt_session_refuses_its_period():
    """Best-of-n: a complete salt-only session the period's close lists, which no pass presents, refuses the period as
    `credit.period`'s ``missing`` window."""
    for seed in range(3):
        _, refused = _credit(cases.instance(random.Random(f"abandoned salt/{seed}"), "abandoned"))
        assert refused[0] == PP.EXACT and ": missing: window " in refused[1], refused


def test_h1_refuses_a_pearl_c_session():
    """H1 and Pearl-C refuse each other's sessions: a Pearl-C epoch's session, which the period credits, is not a live
    window's to H1's salt rule (`window.live_epoch_reason`), and the period refuses an H1 session (``salt-only``)."""
    d = cases.instance(random.Random("h1 reads a pearl-c session"))
    assert _credit(d)[1] is None
    p, m = d["passes"][0], _manifests(d)[0]
    refused = CC.salt(d["key"], d["ledger"], d["pins"], d["main"], {"root": m["session"]["root"]}, p["session"],
                      "ab" * 64, "cd" * 64, m["salt_opening"], cases.RC.lms_verify)
    assert refused == ("coin", "the coin commitment's coins are not a live epoch's: one salt, one draw, then the rounds'")
    _, refused = _credit(cases.instance(random.Random("pearl-c reads an h1 session"), "h1-session"))
    assert refused[0] == PP.SALT_ONLY, refused
