"""`credit.window` on one honest window over a pinned circuit with a SUPPLIED verifier: the verifier `run` here is not
`proofs.verify` but a stand-in that answers every session the Lean names with the outcome this test chooses, accepted
(its findings built from the ``--registered`` file credit hands it) or one sampled proof block's session rejected. So these tests
show the Lean's judgement of the verifier outcomes it is handed, on the served path (`verifier.RECORD_VERDICT` is
``"lean"``): that the Lean `credit-verified` (`credit._lean_verified`), the Lean `credit-live-verified`
(`credit._lean_live_verified`) and the Lean `window` of record (`credit._of_record`) are each called, and what each
answers. They do NOT show that a tampered proof is caught: no proof is read, and a supplied verifier accepts whatever
it is told to. The real-proof case, a kept proof of a pinned H1 unit made with its window verified by `proofs.verify`
inside `credit.window`, is CMP-451; it waits on that kept proof.

The circuit is the registry's window at K = 64 over J10-R's ``PouwH1Tile_v5``, named ``h1/``
(`pouw_credit_cases.registry_window`), `test_period_real`'s, which `credit.TILE_PINS` pins: over a stand-in window
`credit.layout_of` refuses at the layout before any proof is asked, so no cheaper circuit reaches the Lean verifiers.
The Lean calls are counted by wrapping `verifier.lean`, which `verifier.of_record` also calls; `credit.py` is unchanged.

Runtime: about two minutes on a node, what no stand-in window can cover. The window is built once a module (about ten
seconds, five GB), and each case carries its 60 MB descriptor to the Lean three times in the honest case (twice to
`credit-verified`, once to `window`) and twice in the rejected one, each a parse of the whole descriptor, beside
`credit.layout_of`'s own read of it."""
from __future__ import annotations

import copy
import functools
import hashlib
import random
import shutil
from collections import Counter
from pathlib import Path

import pouw_credit_cases as credit_cases
import pouw_rounds_cases as rc
import pytest

from verity.protocols.accounting.compute.pouw.verifier import credit as Cr
from verity.protocols.accounting.compute.pouw.verifier import verifier as V
from verity.protocols.proof.spot.verifier.one_stage import proofs as PF

if shutil.which("lake") is None:
    pytest.skip("lake is not on the PATH", allow_module_level=True)

#: the supplied verifier's reason for the session it rejects
REJECTED = "rejected by the supplied verifier"
#: `credit.window`'s arguments before `verify`: `window_args`' with the statements in place of the outcomes, and the
#: committed circuit files after the live evidence
ARGS = tuple(x for a in credit_cases.WINDOW_ARGS
             for x in {"proofs": ("statements",), "live": ("live", "circuit_files")}.get(a, (a,)))


@functools.cache
def _window() -> dict:
    """An honest window (`honest_window`) whose circuit is the registry's window at K = 64 over v5, named ``h1/``, its
    ledger signed as `window_instance` signs an honest one's."""
    rng = random.Random("supplied/0")
    desc = credit_cases.desc_bytes(credit_cases.registry_window(64, "v5", "h1/")).hex()
    d, s = credit_cases.honest_window(rng, lay=copy.deepcopy(credit_cases.REGISTRY_LAYOUT), desc=desc, prefix="h1/")
    signed = {**credit_cases.signs(d["weights"]), Cr.LENGTH_FIELD: dict(d["length"])}
    d["ledger"] = rc.ledger_lines(rc.ledger_events(rng, d["window"], s, period=signed), rc._key(0))
    return d


def _unit(session: Path) -> int:
    return int(session.name.removeprefix("unit"))


def _evidence(d: dict) -> tuple[list[dict], dict, dict, list[int]]:
    """`d`'s statements (one a call with a sampled proof block, one session a sampled proof block), its live evidence in the statements'
    form (one a round), the committed circuit files (digests of no file: the supplied verifier reads none) and its
    sample's units. The paths name no file."""
    sampled = credit_cases._served_units(d)
    starts = [0]
    for size, _work in Cr.call_work(d["layout"]):
        starts.append(starts[-1] + size)
    statements = []
    for c in range(len(starts) - 1):
        units = [u for u in sampled if starts[c] <= u < starts[c + 1]]
        if units:
            statements.append({"call": c, "circuit": "supplied/circuit", "public": f"supplied/call{c}/public",
                               "sessions": [f"supplied/call{c}/unit{u}" for u in units]})
    live = {"descriptor": d["live"]["descriptor"],
            "statements": [{"round": s, "circuit": "supplied/live/circuit", "public": f"supplied/live/unit{s}/public",
                            "sessions": [f"supplied/live/unit{s}"]} for s in range(len(d["live"]["proofs"]))]}
    ks = sorted({k for _lin, k, _n in d["layout"]["linears"]})
    files = {**{f"k{k}": hashlib.sha512(f"supplied k{k}".encode()).hexdigest() for k in ks},
             Cr.LIVE: hashlib.sha512(b"supplied live").hexdigest()}
    return statements, live, files, sampled


def _supplied(asked: list[tuple[Path, ...]], reject: int | None = None) -> Cr.Run:
    """A supplied verifier: each session of the statement it is asked about is accepted, its finding the ports of the
    ``--registered`` file credit hands it at their values and roots and its unit the one its directory names; the
    session of sampled proof block `reject` is rejected instead, proving no unit. Each statement's sessions are
    appended to `asked`."""
    def run(proof: PF.Proof, *, program: bytes, registered: dict, **_inputs):
        asked.append(proof.sessions)
        ports = {p: {"value": x["value"], "root": x["root"]} for p, x in registered["ports"].items()}
        digest = hashlib.sha512(program).hexdigest()
        return [PF.Finding(str(s), not rejected, REJECTED if rejected else "accepted", digest, ports,
                           () if rejected else (_unit(s),))
                for s in proof.sessions
                for rejected in [reject is not None and _unit(s) == reject and s.parent.name != "live"]], None
    return run


@pytest.fixture
def lean_calls(monkeypatch) -> list[tuple[str, dict]]:
    """Each call of the Lean verifier (`verifier.lean`, which `verifier.of_record` calls too) while a test runs: its
    function and its answer."""
    calls: list[tuple[str, dict]] = []
    lean = V.lean

    def counted(fn: str, value):
        got = lean(fn, value)
        calls.append((fn, got))
        return got
    monkeypatch.setattr(V, "lean", counted)
    return calls


def _credit(d: dict, run: Cr.Run) -> tuple[list[int] | None, tuple[str, str] | None]:
    statements, live, files, _sampled = _evidence(d)
    args = {**d, "statements": statements, "live": live, "circuit_files": files}
    return Cr.window(*(args[x] for x in ARGS), rc.lms_verify, run)


@pytest.mark.slow
def test_supplied_verifier_accepting_honest_window_is_credited_by_the_lean_of_record(lean_calls):
    """With a SUPPLIED verifier that accepts every session, `credit.window` credits the honest pinned window, and the
    credit is the Lean `window`'s of record; the Lean `credit-verified` named the sample's runs and judged their
    answers, and so did the Lean `credit-live-verified` for the live rounds. It shows the Lean's judgement of accepted
    outcomes it is handed, not that any proof is sound."""
    d = _window()
    statements, live, _files, sampled = _evidence(d)
    asked: list[tuple[Path, ...]] = []
    credited, why = _credit(d, _supplied(asked))
    fns = Counter(fn for fn, _got in lean_calls)
    answers = {fn: [got for f, got in lean_calls if f == fn] for fn in fns}
    assert why is None and credited, (why, fns)
    assert fns == {"credit-verified": 2, "credit-live-verified": 2, "window": 1}, fns
    named, judged = answers["credit-verified"]
    assert [a["statement"] for a in named["run"]] == list(range(len(statements))), named
    assert sorted(o["unit"] for o in judged["outcomes"]) == sampled, judged
    live_named, live_judged = answers["credit-live-verified"]
    assert len(live_named["run"]) == len(live["statements"]), live_named
    assert sorted(o["unit"] for o in live_judged["outcomes"]) == list(range(len(live["statements"]))), live_judged
    assert answers["window"] == [{"credited": credited}], answers["window"]
    assert len(asked) == len(statements) + len(live["statements"]), asked


@pytest.mark.slow
def test_supplied_verifier_rejecting_one_sampled_session_is_refused_by_the_lean(lean_calls):
    """With a SUPPLIED verifier that rejects the session of one sampled proof block, `credit.window` refuses the window with the
    Lean `credit-verified`'s refusal (``failed``, naming that unit's session), and neither the live rounds nor the Lean
    `window` of record are asked. It shows the Lean's judgement of a rejection it is handed, not that a tampered proof
    is caught: the supplied verifier rejects by fiat."""
    d = _window()
    _statements, _live, _files, sampled = _evidence(d)
    unit = sampled[0]
    asked: list[tuple[Path, ...]] = []
    credited, why = _credit(d, _supplied(asked, reject=unit))
    fns = Counter(fn for fn, _got in lean_calls)
    assert fns == {"credit-verified": 2}, fns
    judged = lean_calls[-1][1]
    assert credited is None and why == tuple(judged["refused"]), (credited, why, judged)
    assert why[0] == "failed" and f"unit{unit} " in why[1] and REJECTED in why[1], why
    assert any(f"unit{unit}" in str(s) for sessions in asked for s in sessions), asked
