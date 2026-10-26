"""The network certifier over a period (`NetworkCheck.Period`, `verity/certifier/v0`, `network-period serve`) against its
Python mirror, through the difftest harness's `ProtocolPort`.

The mirror and the generator are `period_case.py`'s: `reference` (its `answer`) checks a period in the core interface's
answer format, `build` makes a period with one change (`EXPECT`, each aimed at one path of the check, or malformed
parameters, commitments, evidence and cases) in the clock form or, for a change named `rows-…`, the rows form, and
`case` a small period of either form aimed at a verdict or at a decoding failure (`KINDS`). This module re-exports
what the composition's difftest reads of it (`reference`, `build`, `EXPECT`): it draws onsite's certifier cases from
`build` and answers them with `reference` (`verity/core/protocols/composition/tests/reference.py`)."""

from __future__ import annotations

import dataclasses
import json
import shutil
from collections import Counter
from fractions import Fraction
from pathlib import Path

import pytest
from hypothesis import strategies as st

import period_case as PC
import verity_difftest as D
from period_case import (ACCEPT, EXPECT, FORMS, INGRESS, KINDS, MUTATIONS, REJECT, UNDECIDED, Facts, Params, at_most,
                         build, clog2, public, reference, verdict_of)
from verity.core.protocols.certifier import warden_rows as WR
from verity.core.protocols.certifier.certificate import Certificate
from verity.core.protocols.certifier.schedule import SyncSet
from verity.core.protocols.certifier.warden import WardenParams

__all__ = ["EXPECT", "MUTATIONS", "build", "reference"]

LEAN = Path(__file__).resolve().parents[1] / "lean"
pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

TAGS: dict[str, str] = {}
SEEN: Counter = Counter()


@st.composite
def cases(draw, only: tuple[str, ...] = tuple(MUTATIONS)) -> dict:
    m = draw(st.sampled_from([h for h in ("honest", "rows-honest") if h in only] * 2 + list(only)))
    case = build(m, draw(st.integers(0, 1 << 16)))
    TAGS[D.canonical(case)] = m
    return case


def counted(case) -> dict:
    out = reference(case)
    SEEN[(TAGS.get(D.canonical(case), "?"), out["verdict"])] += 1
    return out


PORT = D.ProtocolPort("network-period", counted, cases(), package=LEAN)
server = D.server_fixture(PORT)


@pytest.mark.slow
def test_the_period_check_agrees_with_its_mirror(server):
    """Lean and the mirror agree on every case, and each change of either form reaches its verdict at least once."""
    SEEN.clear()
    examples = 1200
    stats = D.run(PORT, examples=examples, server=server)
    print("\n".join(f"{tag:22} {verdict:10} {k}" for (tag, verdict), k in sorted(SEEN.items())))
    missed = [tag for tag, verdict in EXPECT.items() if not SEEN[(tag, verdict)]]
    assert not missed, (missed, SEEN)
    assert min(stats.counts.get(v, 0) for v in (ACCEPT, REJECT, UNDECIDED)) > examples // 50, stats


def test_each_verdict_is_the_mirrors(server):
    """A short run over an honest period, an undecided link-window and a forged certificate, in each form: all three
    verdicts."""
    port = dataclasses.replace(PORT, inputs=cases(("honest", "answer-undecided", "cert-forged", "rows-honest",
                                                   "rows-answer-undecided", "rows-cert-forged")))
    stats = D.run(port, examples=40, server=server)
    assert all(stats.counts.get(v) for v in (ACCEPT, REJECT, UNDECIDED)), stats


def test_a_mirror_that_skips_the_ingress_certificate_is_caught(server, monkeypatch):
    """A mirror that reads only the egress certificate accepts a period whose ingress certificate names another link,
    and Lean rejects it."""
    real = PC._certifies
    monkeypatch.setattr(PC, "_certifies", lambda p, d, w, c: d is INGRESS or real(p, d, w, c))
    with pytest.raises(D.Mismatch) as x:
        D.run(dataclasses.replace(PORT, inputs=cases(("cert-link",))), examples=20, server=server)
    assert (x.value.lean[0], x.value.python[0]) == (REJECT, ACCEPT)


def test_a_mirror_that_skips_the_rows_check_is_caught(server, monkeypatch):
    """A mirror that doesn't read the row registrations accepts a rows-form period whose rows a key other than the
    deployment key signed, and Lean rejects it."""
    monkeypatch.setattr(PC, "_rows_check", lambda p, w, d, doc: None)
    for n in range(2):
        with pytest.raises(D.Mismatch) as x:
            D.replay(PORT, build("rows-reg-key", n, small=True), server)
        assert (x.value.lean[0], x.value.python[0]) == (REJECT, ACCEPT)


#: The rows form's refusals the period must make, each with what Lean's reason and the mirror's say.
ROWS_REFUSALS = {
    "rows-cert-missing": ("certificates: the requests', responses' and fails' records'", "certificates"),
    "rows-reg-key": ("not under the deployment key in the session", "doesn't verify under"),
    "rows-answer-window": ("another window's answer", "another window's answer"),
    "rows-schedule": ("another schedule's digest than the bound one", "the registration's parameters is"),
}


def test_the_rows_form_refuses_what_it_must(server):
    """A link-window missing one of its records' certificates, a registration not under the deployment key, the scan's
    answer for another window's rows, and rows under a schedule whose digest isn't the bound one: Lean and the mirror
    each refuse every one at that check."""
    for m, (lean, mirror) in ROWS_REFUSALS.items():
        for n in range(4):
            case = build(m, n, small=n % 2 == 0)
            assert mirror in reference(case)["reason"], (m, n, reference(case))
            verdict, out = D.replay(PORT, case, server)
            assert verdict == REJECT and lean in json.loads(out)["reason"], (m, n, out)


def test_the_mirror_reads_a_registration_as_the_network_certifier_writes_it():
    """The rows form's honest registrations are `warden_rows.registration`'s documents, which its own
    `check_registration` reads back under the period's key, session and schedule."""
    for n in range(4):
        case = build("rows-honest", n, small=n % 2 == 0)
        p = PC.params(case["params"])
        for w, x in enumerate(json.loads(bytes.fromhex(case["evidence"]))["windows"], start=p.first):
            assert [doc["format"] for doc in x["registrations"]] == [WR.REGISTRATION] * 2
            for doc, d in zip(x["registrations"], (PC.INGRESS, PC.EGRESS)):
                head, _ = WR.check_registration(doc, p.key, p.session, p.schedule, p.link, d, w)
                c = {k: bytes.fromhex(v) for k, v in doc["certificate"].items()}
                cert = Certificate(c["key"], p.link, d, w, c["digest"], c["signature"])
                assert WR.registration(head, cert, p.schedule) == doc


def test_a_small_period_reaches_its_verdict(server):
    """Each change's small period (`build(..., small=True)`, which the composition's difftest draws) gets the change's
    verdict from the mirror, and Lean agrees."""
    small = [build(m, n, small=True) for m in MUTATIONS if m != "case" for n in range(2)]
    assert [reference(c)["verdict"] for c in small] == [EXPECT[m] for m in MUTATIONS if m != "case" for _ in range(2)]
    D.run(dataclasses.replace(PORT, inputs=st.sampled_from(small)), examples=2 * len(small), server=server)


@pytest.mark.parametrize("form", FORMS)
def test_each_kind_of_case_is_answered_as_lean_answers(server, form):
    """`case(kind, seed, form)` reaches its kind's verdict for each kind, the ones that fail to decode refused where
    they fail (a digest only once the commitment decodes); an accepted period's eps is its audits' and scans' deltas
    summed,
    and its profile names no sync; in the clock form its evidence holds each clock certificate alone, and in the rows
    form each link-window's three records' certificates and two row registrations, with one scan a link-window and no
    schedules term; `cases(seed, form)` is one of each with its answer; and Lean gives every one of them the mirror's
    answer."""
    drawn = {(k, s): PC.case(k, s, form) for k in KINDS for s in range(6)}
    answers = {ks: PC.answer(c) for ks, c in drawn.items()}
    assert {ks: a["verdict"] for ks, a in answers.items()} == {(k, s): verdict_of(k) for k, s in drawn}
    assert all(a["reason"].startswith(k.removeprefix("broken-") + ":")
               and (a["commitment"] is None) == (k != "broken-evidence")
               for (k, _), a in answers.items() if k.startswith("broken-"))
    accepted = [a["public"] for (k, _), a in answers.items() if k == ACCEPT]
    assert all(isinstance(x["K"], int) and x["eps"] == str(sum(
        (Fraction(t["delta"]) for t in x["audits"] + x.get("scans", [])), Fraction(0))) for x in accepted)
    assert all(sorted(a["profile"]) == ["link", "windows"] for (k, _), a in answers.items() if k == ACCEPT)
    shown = [w for (k, _), c in drawn.items() if k == ACCEPT
             for w in json.loads(bytes.fromhex(c["evidence"]))["windows"]]
    if form == "clock":
        assert all(sorted(w["clock"]) == PC.CERTIFICATE_KEYS for w in shown)
        assert all("scans" not in x for x in accepted)
    else:
        assert all(sorted(w) == PC.ROWS_WINDOW_KEYS and len(w["certificates"]) == 3 and len(w["registrations"]) == 2
                   for w in shown)
        assert all(x["schedules"] == 0 and len(x["scans"]) == x["windows"] for x in accepted)
    assert PC.cases(3, form) == {k: (drawn[k, 3], answers[k, 3]) for k in KINDS}
    D.run(dataclasses.replace(PORT, inputs=st.sampled_from(list(drawn.values()))), examples=2 * len(drawn),
          server=server)


def test_the_bound_counts_what_it_says():
    """K's terms on a period worked by hand: 10 units drawn subset:3 at δ = 1/10 with at most 4 wrong, 7 cells on one
    connection and none on another, 2 syncs over 3 link-windows, jitter below 2 over 3 steps; and eps over audits at
    δ = 1/10, 1/15 and 1/6, and over none (Lean's `#guard`s)."""
    p = Params(((0, 3),), (7, 0), 0, 2, "a", None, b"", "", SyncSet((0,), (Fraction(1), Fraction(2))), 3)
    out = public(p, [Facts(10, 3, Fraction(1, 10), 4)])
    assert (out["schedules"], out["cells"], out["audits"][0]["faulty_sets"], out["audits"][0]["faulty_bits"],
            out["audits"][0]["jitter"], out["K"]) == (3, 3, 9, 64, 3, 3 + 3 + 9 + 64 + 3)
    assert [clog2(x) for x in range(6)] == [0, 0, 1, 2, 2, 3]
    assert [at_most(10, 0), at_most(10, 1), at_most(10, 2), at_most(4, 9), at_most(0, 3)] == [1, 11, 56, 16, 1]
    assert out["eps"] == "1/10"
    three = dataclasses.replace(p, audits=((0, 0),) * 3)
    assert public(three, [Facts(1, 1, Fraction(1, d), 0) for d in (10, 15, 6)])["eps"] == "1/3"
    assert public(dataclasses.replace(p, audits=()), [])["eps"] == "0"


def test_the_rows_forms_bound_counts_what_it_says():
    """The rows form's K worked by hand (Lean's `rowsTerms` `#guard`s): no schedules term under its one schedule, 7
    cells on one connection, the audit above, and over 3 link-windows one scan of 20 units drawn at δ = 1/15 with at
    most 3 wrong (1351 sets), 1 free input and jitter below 2 over 2 steps; and eps over the audit's δ = 1/10 and the
    scans' 1/15 and 1/6."""
    p = Params(((0, 3),), (7,), 0, 2, "a", None, b"", "", None, 3, WardenParams(), (1, 2))
    out = public(p, [Facts(10, 3, Fraction(1, 10), 4)], [Facts(20, 5, Fraction(1, 15), 3)])
    (scan,) = out["scans"]
    assert at_most(20, 3) == 1351 and clog2(1351) == 11
    assert (out["schedules"], out["cells"], scan["free"], scan["faulty_sets"], scan["faulty_bits"], scan["jitter"],
            out["K"]) == (0, 3, 1, 11, 48, 2, 0 + 3 + (0 + 9 + 64 + 3) + (1 + 11 + 48 + 2))
    two = dataclasses.replace(p, audits=((0, 0),), windows=2, scan=(0, 0))
    assert public(two, [Facts(1, 1, Fraction(1, 10), 0)],
                  [Facts(1, 1, Fraction(1, d), 0) for d in (15, 6)])["eps"] == "1/3"
