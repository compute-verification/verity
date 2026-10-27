"""The network certifier over a period (`NetworkCheck.Period`, `verity/certifier/v0`, `network-period serve`) against its
Python mirror, through the difftest harness's `ProtocolPort`.

The mirror and the generator are `period_case.py`'s: `reference` (its `answer`) checks a period in the core interface's
answer format, running one-stage v2 on each link-window's proof as the check does, `build` makes a period with one
change (`EXPECT`, each aimed at one path of the check, or malformed parameters, commitments, evidence and cases) in the
clock form or, for a change named `rows-…`, the rows form, and `case` a small period of either form aimed at a verdict
or at a decoding failure (`KINDS`). No case carries a proof v2 accepts, so the best verdict here is undecided
(`UndecidedPort`). This module re-exports what the composition's difftest reads of it (`reference`, `build`, `EXPECT`):
it draws onsite's certifier cases from `build` and answers them with `reference`
(`verity/core/protocols/composition/tests/reference.py`)."""

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
from period_case import (ACCEPT, EXPECT, FORMS, INGRESS, MUTATIONS, REJECT, UNDECIDED, Facts, Params, at_most, build,
                         clog2, public, reference, verdict_of)
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


class UndecidedPort(D.ProtocolPort):
    """The period's port, whose best verdict here is undecided: one-stage v2 accepts a link-window only on a session's
    proof, which no case carries, so a run's non-vacuity bar is an undecided period."""
    accepted = UNDECIDED


PORT = UndecidedPort("network-period", counted, cases(), package=LEAN)
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
    assert min(stats.counts.get(v, 0) for v in (REJECT, UNDECIDED)) > examples // 50, stats
    assert not stats.counts.get(ACCEPT), stats


def test_each_verdict_is_the_mirrors(server):
    """A short run over an honest period and a forged certificate in each form: the clock form's honest period is
    undecided (v2 on a link-window with no session's proof), the rows form's rejected at its scan's answer."""
    port = dataclasses.replace(PORT, inputs=cases(("honest", "audit-undecided", "cert-forged", "rows-honest",
                                                   "rows-cert-forged")))
    stats = D.run(port, examples=40, server=server)
    assert all(stats.counts.get(v) for v in (REJECT, UNDECIDED)), stats


def test_a_mirror_that_skips_the_ingress_certificate_is_caught(server, monkeypatch):
    """A mirror that reads only the egress certificate leaves a period whose ingress certificate names another link to
    one-stage v2 (undecided), and Lean rejects it."""
    real = PC._certifies
    monkeypatch.setattr(PC, "_certifies", lambda p, d, w, c: d is INGRESS or real(p, d, w, c))
    with pytest.raises(D.Mismatch) as x:
        D.run(dataclasses.replace(PORT, inputs=cases(("cert-link",))), examples=20, server=server)
    assert (x.value.lean[0], x.value.python[0]) == (REJECT, UNDECIDED)


def test_a_mirror_that_skips_one_stage_v2_is_caught(server, monkeypatch):
    """A mirror that checks each link-window's fields but doesn't run one-stage v2 on its proof accepts an honest
    period, and Lean, which runs v2 and finds no session's proof, leaves it undecided."""
    monkeypatch.setattr(PC, "_proved", lambda *a: None)
    for n in range(2):
        with pytest.raises(D.Mismatch) as x:
            D.replay(PORT, build("honest", n, small=True), server)
        assert (x.value.lean[0], x.value.python[0]) == (UNDECIDED, ACCEPT)


#: The refusals the period must make by name, each with what Lean's reason and the mirror's say: a link-window shown
#: with an answer of its own (the made-up `{"verdict": "accept", …}` the check once read), one whose proof v2 refuses
#: (a public file or a session record in no blob, units no session proved, a circuit file in no blob), one whose
#: sessions opened a root other than the registered one, and in the rows form a link-window missing one of its
#: records' certificates, a registration not under the deployment key, the scan's answer for another window's rows,
#: rows under a schedule whose digest isn't the bound one, and the scan's answer of an honest period, which the check
#: takes from no one.
REFUSALS = {
    "answer-made-up": ("answer: the check runs one-stage v2 on the link-window's proof itself and takes no answer",
                       "answer: the check runs one-stage v2 on the link-window's proof itself"),
    "v2-refused": ("one-stage v2: ", "one-stage v2: "),
    "v2-session": ("one-stage v2: the audit's checks failed: roots", "one-stage v2: "),
    "rows-cert-missing": ("certificates: the requests', responses' and fails' records'", "certificates"),
    "rows-reg-key": ("not under the deployment key in the session", "doesn't verify under"),
    "rows-answer-window": ("another window's answer", "another window's answer"),
    "rows-schedule": ("another schedule's digest than the bound one", "the registration's parameters is"),
    "rows-honest": ("the scan's answer: the check takes no answer for a link-window, and one-stage v2 can't run",
                    "the scan's answer: the check takes no answer for a link-window"),
    "state-read": (f"its read of {PC.SCAN_STATE}: the clock form has no scan",) * 2,
    "rows-state-root": (f"its read of {PC.SCAN_STATE}: another root than link-window ",) * 2,
    "rows-state-domain": (f"its read of {PC.SCAN_STATE}: its domain is no link-window's scan state's",) * 2,
}


def test_the_period_refuses_what_it_must_by_name(server):
    """Each of `REFUSALS`: Lean and the mirror each refuse it at that check, by its name."""
    for m, (lean, mirror) in REFUSALS.items():
        for n in range(4):
            case = build(m, n, small=n % 2 == 0)
            assert mirror in reference(case)["reason"], (m, n, reference(case))
            verdict, out = D.replay(PORT, case, server)
            assert verdict == REJECT and lean in json.loads(out)["reason"], (m, n, out)


def test_an_audits_read_of_the_scan_state_under_a_link_windows_root_and_domain_passes():
    """The rows form's honest periods' audits read the scan's state rows under the root and domain a link-window's scan
    registration names them under, and reach the scan's answer (`REFUSALS`' rows-honest holds Lean to it); the clock
    form's honest audits read none."""
    reads = 0
    for n in range(8):
        for form in FORMS:
            case = build("honest" if form == "clock" else "rows-honest", n, small=n % 2 == 0)
            c = PC.commitment(bytes.fromhex(case["commitment"]))
            states = [PC.state_of(reg) for reg in c.windows]
            got = [(e["root"], e["domain"]) for reg in c.audits for e in reg.record.get("registered", [])
                   if e["value"] == PC.SCAN_STATE]
            assert all(x in states for x in got) and (form == "rows" or not got), (form, n)
            reads += len(got)
            if form == "rows":
                assert "the scan's answer" in reference(case)["reason"], n
    assert reads


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
    """`case(kind, seed, form)` reaches its kind's verdict for each kind the form reaches (`AIMS`: no case is
    accepted, and the rows form, whose scan's answer the check refuses, has no undecided one), the ones that fail to
    decode refused where they fail (a digest only once the commitment decodes); an honest period's link-windows show in
    the clock form two certificates, the clock certificate, one-stage v2's evidence and the two delivered salts, its
    case carrying v2's blobs, and in the rows form three records' certificates, two row registrations and the scan's
    answer; `cases(seed, form)` is one of each with its answer; and Lean gives every one of them the mirror's answer."""
    kinds = list(PC.AIMS[form])
    assert kinds == [k for k in PC.KINDS if form == "clock" or k != UNDECIDED]
    drawn = {(k, s): PC.case(k, s, form) for k in kinds for s in range(6)}
    answers = {ks: PC.answer(c) for ks, c in drawn.items()}
    assert {ks: a["verdict"] for ks, a in answers.items()} == {(k, s): verdict_of(k) for k, s in drawn}
    assert all(a["reason"].startswith(k.removeprefix("broken-") + ":")
               and (a["commitment"] is None) == (k != "broken-evidence")
               for (k, _), a in answers.items() if k.startswith("broken-"))
    honest = [build("honest" if form == "clock" else "rows-honest", s, small=True) for s in range(6)]
    shown = [w for c in honest for w in json.loads(bytes.fromhex(c["evidence"]))["windows"]]
    if form == "clock":
        assert all(sorted(w) == PC.CLOCK_WINDOW_KEYS and len(w["certificates"]) == 2
                   and sorted(w["salts"]) == ["o0", "pub"] and sorted(w["clock"]) == PC.CERTIFICATE_KEYS for w in shown)
        assert all(c["blobs"] == [PC.ZK.CIRCUIT.hex(), PC.ZK.PUBLIC.hex()] for c in honest)
    else:
        assert all(sorted(w) == PC.ROWS_WINDOW_KEYS and len(w["certificates"]) == 3 and len(w["registrations"]) == 2
                   for w in shown)
    assert PC.cases(3, form) == {k: (drawn[k, 3], answers[k, 3]) for k in kinds}
    for c in drawn.values():
        D.replay(PORT, c, server)


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
