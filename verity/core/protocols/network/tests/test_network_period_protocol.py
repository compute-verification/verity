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
import types
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
from verity.ml.circuits import codec
from verity.ml.circuits.defs import bind
from verity.ml.circuits.program import Program
from verity.ml.windows import interface as IF

__all__ = ["EXPECT", "MUTATIONS", "build", "reference"]

LEAN = Path(__file__).resolve().parents[1] / "lean"
pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

TAGS: dict[str, str] = {}
SEEN: Counter = Counter()


#: The changes drawn more often than the rest: each form's accepted periods, its honest one and the rows form's tied
#: audit sampled in part, and its undecided answers, the only undecided ones.
WEIGHTS = {"honest": 4, "rows-honest": 4, "rows-tied-partial": 2, "rows-tied-sampled": 2, "answer-undecided": 2,
           "audit-undecided": 2, "rows-answer-undecided": 1, "rows-audit-undecided": 1}


@st.composite
def cases(draw, only: tuple[str, ...] = tuple(MUTATIONS)) -> dict:
    m = draw(st.sampled_from([h for h, k in WEIGHTS.items() if h in only for _ in range(k)] + list(only)))
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


def refused_by_lean_alone(case: dict, server, why: str) -> None:
    """A mirror with a check skipped accepts `case`, and Lean refuses it, its reason naming `why`."""
    with pytest.raises(D.Mismatch) as x:
        D.replay(PORT, case, server)
    assert (x.value.lean[0], x.value.python[0]) == (REJECT, ACCEPT), x.value
    assert why in json.loads(x.value.lean[1])["reason"], x.value


def test_a_mirror_that_skips_the_rows_check_is_caught(server, monkeypatch):
    """A mirror that doesn't read the row registrations accepts a rows-form period whose rows a key other than the
    deployment key signed, and Lean refuses it at the rows."""
    monkeypatch.setattr(PC, "_rows_check", lambda p, w, d, doc: None)
    for n in range(2):
        refused_by_lean_alone(build("rows-reg-key", n, small=True), server, "not under the deployment key")


#: The rows form's refusals the period must make, each with what Lean's reason and the mirror's say.
ROWS_REFUSALS = {
    "rows-cert-missing": ("certificates: the requests', responses' and fails' records'", "certificates"),
    "rows-past-slots": ("certifies none of a window in which it refused a connection", "refused a connection"),
    "rows-reg-key": ("not under the deployment key in the session", "doesn't verify under"),
    "rows-answer-window": ("another window's answer", "another window's answer"),
    "rows-schedule": ("another schedule's digest than the bound one", "the registration's parameters is"),
    "rows-delta": ("is not the period's", "is not the period's"),
    "rows-units": ("not the scan's 2T", "not the scan's 2T"),
    "rows-prog-untied": ("is no compute audit's output",) * 2,
    "rows-prog-unwired": ("is no compute audit's slot entries",) * 2,
    "rows-state-untied": (PC.UNSTATED,) * 2,
    "rows-state-misindexed": (PC.MISREAD,) * 2,
    "rows-state-unpinned": (PC.UNPINNED,) * 2,
    "rows-state-misanchored": (PC.MISANCHORED,) * 2,
}


def test_the_rows_form_refuses_what_it_must(server):
    """A link-window missing one of its records' certificates, one the warden left uncertified for refusing a
    connection (traffic past its Program slots), a registration not under the deployment key, the scan's answer for
    another window's rows, rows under a schedule whose digest isn't the bound one, scans at two δ, a scan answered over
    other than its 2T units, a scan whose `prog` is no compute audit's output, one whose `prog` is an audit's output but
    no `SlotEntry_v1` or `SelectEntry_v1` template's entries, a scan whose state is not its wired audit's
    `SCAN_STATE`, one whose units read their state off the chain, one whose row 0 isn't opened to the empty state and
    one whose tied audit reads the state off its slots' anchors: Lean and the mirror each refuse every one at that
    check."""
    for m, (lean, mirror) in ROWS_REFUSALS.items():
        for n in range(4):
            case = build(m, n, small=n % 2 == 0)
            assert mirror in reference(case)["reason"], (m, n, reference(case))
            verdict, out = D.replay(PORT, case, server)
            assert verdict == REJECT and lean in json.loads(out)["reason"], (m, n, out)


def test_the_rows_form_accepts_a_wired_state_tied_period(server):
    """A rows-form period whose every link-window's scan's `prog` is a compute audit's slot entries and whose state is
    that audit's `SCAN_STATE` passes every check: Lean accepts it with the mirror's public, and leaves one with an
    undecided scan or audit answer undecided, as the mirror does."""
    for m, want in (("rows-honest", ACCEPT), ("rows-answer-undecided", UNDECIDED), ("rows-audit-undecided", UNDECIDED)):
        for n in range(3):
            case = build(m, n, small=n % 2 == 0)
            out = reference(case)
            assert out["verdict"] == want, (m, n, out)
            verdict, got = D.replay(PORT, case, server)
            assert verdict == want and json.loads(got).get("public") == out.get("public"), (m, n, got)


def test_a_scan_whose_state_is_not_its_wired_audits_is_refused(server, monkeypatch):
    """A slot's Program reads the requests from its audit's registered `SCAN_STATE`, and the scan's units read and
    register their states at the scan's state's root, each past the first its transition's output from the one before,
    so the rows form refuses, in Lean and in the mirror and by name, each way a link-window's state tie can fail
    (`PC.UNSTATES`): a `SCAN_STATE` that skips or alters one of the scan's transitions (the scan's state entry, or the
    audit's, at another root or domain), no state entry on either side or the scan's under another name, the state of
    an audit the scan's `prog` isn't wired to, a root that is no string, a second state entry at another root on
    either side, the scan's held in its window's `activations.values` alone (where an audit's may be, a scan's never
    is), and the audit's held there alone under another name; and a mirror that skips the tie accepts what Lean
    refuses, on the audit's side (the scan's own state entry broken also leaves its row 0's opening under no root it
    states, which the mirror then refuses)."""
    for u in PC.UNSTATES:
        monkeypatch.setattr(PC, "UNSTATES", (u,))
        for n in range(3):
            case = build("rows-state-untied", n, small=n % 2 == 0)
            assert PC.UNSTATED in reference(case)["reason"], (u, n, reference(case))
            verdict, out = D.replay(PORT, case, server)
            assert verdict == REJECT and PC.UNSTATED in json.loads(out)["reason"], (u, n, out)
    monkeypatch.setattr(PC, "UNSTATES", ("audit", "audit-root", "audit-extra", "audit-misnamed"))
    monkeypatch.setattr(PC, "_state_tied", lambda p, audits, reg: True)
    for n in range(2):
        refused_by_lean_alone(build("rows-state-untied", n, small=True), server, PC.UNSTATED)


def test_a_tied_audit_whose_state_is_in_its_windows_activations_alone_is_tied(server, monkeypatch):
    """A window circuit's registration commits the scan's state rows its slots read as a per-window value, its window's
    `activations.values` entry of `SCAN_STATE`, with no `roots` or `registered` entry of it (`PC.STATE_PLACES`): a
    period whose every tied audit holds its state there alone passes with each audit's state held, and Lean accepts it
    with the mirror's public; one whose aimed audit's entry there is at another root, beside a second state entry at
    another root, or under another name (its `value` another or none, or under another value's key), is refused, in
    Lean and in the mirror and by name."""
    monkeypatch.setattr(PC, "STATE_PLACES", ("activations",))
    for m in ("rows-honest", "rows-state-anchored"):
        for n in range(3):
            case = build(m, n, small=n % 2 == 0)
            c = PC.commitment(bytes.fromhex(case["commitment"]))
            tied = {i for reg in c.windows for i in PC._tied_audits(c.audits, reg)}
            assert tied and all(not PC._states(c.audits[i]) and PC._audit_states(c.audits[i]) for i in tied), (m, n)
            assert all(PC._state_held(c.audits, c.windows, i) for i in tied), (m, n)
            out = reference(case)
            assert out["verdict"] == ACCEPT, (m, n, out)
            verdict, got = D.replay(PORT, case, server)
            assert verdict == ACCEPT and json.loads(got)["public"] == out["public"], (m, n, got)
    for u in ("audit-root", "audit-extra", "audit-misnamed"):
        monkeypatch.setattr(PC, "UNSTATES", (u,))
        for n in range(2):
            case = build("rows-state-untied", n, small=n % 2 == 0)
            assert PC.UNSTATED in reference(case)["reason"], (u, n, reference(case))
            verdict, out = D.replay(PORT, case, server)
            assert verdict == REJECT and PC.UNSTATED in json.loads(out)["reason"], (u, n, out)


def test_a_tied_audit_that_runs_another_circuit_is_refused(server, monkeypatch):
    """The rows form holds each phase's tied audits to the class circuit, the parameters' `circuit`: a period
    whose every tied audit is registered under it passes, and Lean accepts it with the mirror's public, whose `held`
    names `circuit` and `free`; each way the hold can fail (`PC.UNHOLDS`: the aimed phase's tied audit registered
    under another Program, or the parameters' `circuit` another digest) Lean and the mirror refuse by name; and a mirror
    that skips the hold accepts what Lean refuses."""
    for n in range(3):
        case = build("rows-honest", n, small=n % 2 == 0)
        c = PC.commitment(bytes.fromhex(case["commitment"]))
        tied = [i for reg in c.windows for i in PC._tied_audits(c.audits, reg)]
        assert tied and all(c.audits[i]["program"] == case["params"]["circuit"] for i in tied), (n, c.audits)
        out = reference(case)
        assert out["verdict"] == ACCEPT and out["public"]["held"] == ["circuit", "free"], (n, out)
        verdict, got = D.replay(PORT, case, server)
        assert verdict == ACCEPT and json.loads(got)["public"] == out["public"], (n, got)
    unhold = PC.UNHOLDS
    for u in unhold:
        monkeypatch.setattr(PC, "UNHOLDS", (u,))
        for n in range(3):
            case = build("rows-circuit-swapped", n, small=n % 2 == 0)
            assert PC.UNHELD in reference(case)["reason"], (u, n, reference(case))
            verdict, out = D.replay(PORT, case, server)
            assert verdict == REJECT and PC.UNHELD in json.loads(out)["reason"], (u, n, out)
    monkeypatch.setattr(PC, "UNHOLDS", unhold)
    monkeypatch.setattr(PC, "_held", lambda p, audits, reg: True)
    for n in range(2):
        refused_by_lean_alone(build("rows-circuit-swapped", n, small=True), server, PC.UNHELD)


def test_a_scan_whose_units_read_their_state_off_the_chain_is_refused(server, monkeypatch):
    """Ingress unit t reads `sin` at state leaf t and registers its `o1` at leaf t + 1, as the scan's registration
    states (`PC.state_reads`), so the rows form refuses, in Lean and in the mirror and by name, each way those reads can
    leave the chain while the state stays tied by root and domain (`PC.MISREADS`): a `sin` at leaf t + 1, an `o1` at
    leaf t, a run over other units, the two runs swapped, the reads or a run missing or renamed, and a field no
    natural; and a mirror that skips the reads accepts what Lean refuses."""
    misreads = PC.MISREADS
    for u in misreads:
        monkeypatch.setattr(PC, "MISREADS", (u,))
        for n in range(3):
            case = build("rows-state-misindexed", n, small=n % 2 == 0)
            assert PC.MISREAD in reference(case)["reason"], (u, n, reference(case))
            verdict, out = D.replay(PORT, case, server)
            assert verdict == REJECT and PC.MISREAD in json.loads(out)["reason"], (u, n, out)
    monkeypatch.setattr(PC, "MISREADS", misreads)
    monkeypatch.setattr(PC, "_state_read", lambda p, reg: True)
    for n in range(2):
        refused_by_lean_alone(build("rows-state-misindexed", n, small=True), server, PC.MISREAD)


def test_a_scan_whose_row_0_is_not_the_empty_state_is_refused(server, monkeypatch):
    """Row 0 of the scan's state, which no unit computes and the Programs anchored at bucket 0 read, must open to the
    empty state, so the rows form refuses, in Lean and in the mirror and by name, each way the opening can fail
    (`PC.UNPINS`): a state tree whose row 0 the prover chose, opened as committed, no opening, another salt or path,
    another word count, leaf 1's opening, and a field of another type; and a mirror that skips the opening accepts
    what Lean refuses."""
    unpins = PC.UNPINS
    for u in unpins:
        monkeypatch.setattr(PC, "UNPINS", (u,))
        for n in range(3):
            case = build("rows-state-unpinned", n, small=n % 2 == 0)
            assert PC.UNPINNED in reference(case)["reason"], (u, n, reference(case))
            verdict, out = D.replay(PORT, case, server)
            assert verdict == REJECT and PC.UNPINNED in json.loads(out)["reason"], (u, n, out)
    monkeypatch.setattr(PC, "UNPINS", unpins)
    monkeypatch.setattr(PC, "_state_empty", lambda p, reg: True)
    for n in range(2):
        refused_by_lean_alone(build("rows-state-unpinned", n, small=True), server, PC.UNPINNED)


def test_a_tied_audit_that_reads_the_state_off_its_slots_anchors_is_refused(server, monkeypatch):
    """Lane l's slot at boundary b, unit l · nb + b of a tied audit's slot entries, computes its entry on the state
    before its anchor's bucket, row b · every of the scan's state, so a period whose tied audits' circuit records read
    the state there (`PC.anchor_audit`, a digit of radix 1 between or not) passes, and Lean accepts it with the mirror's
    public; the rows form refuses, in Lean and in the mirror and by name, each way the read can leave the anchors while
    every tie, wiring, state and hold stays (`PC.MISANCHORS`): the row after the anchor's, lane 1's slots at it, the
    lanes' and boundaries' digits swapped, a read of the slot entries' output, a state parameter of T + 1 rows, and a
    table over the state; and a mirror that skips the anchors accepts what Lean refuses."""
    for n in range(3):
        case = build("rows-state-anchored", n, small=n % 2 == 0)
        c = PC.commitment(bytes.fromhex(case["commitment"]))
        tied = {i for reg in c.windows for i in PC._tied_audits(c.audits, reg)}
        assert tied and all("circuit" in case["params"]["audits"][i] for i in tied), (n, tied)
        out = reference(case)
        assert out["verdict"] == ACCEPT, (n, out)
        verdict, got = D.replay(PORT, case, server)
        assert verdict == ACCEPT and json.loads(got)["public"] == out["public"], (n, got)
    misanchors = PC.MISANCHORS
    for u in misanchors:
        monkeypatch.setattr(PC, "MISANCHORS", (u,))
        for n in range(2):
            case = build("rows-state-misanchored", n, small=n % 2 == 0)
            assert PC.MISANCHORED in reference(case)["reason"], (u, n, reference(case))
            verdict, out = D.replay(PORT, case, server)
            assert verdict == REJECT and PC.MISANCHORED in json.loads(out)["reason"], (u, n, out)
    monkeypatch.setattr(PC, "MISANCHORS", misanchors)
    monkeypatch.setattr(PC, "_anchored", lambda s, a: True)
    for n in range(2):
        refused_by_lean_alone(build("rows-state-misanchored", n, small=True), server, PC.MISANCHORED)


def test_a_tied_audit_is_charged_its_state_reads_held_only_at_the_scans_state(server, monkeypatch):
    """A phase's term charges a tied audit its free inputs with its reads of the scan's state held (`state_free`) only
    where its `SCAN_STATE` is the scan's state in every phase tied to it (`PC._state_held`), the root the scan's
    registration's own: where each tied audit reads it at its slots' anchors (`rows-state-anchored`) each is charged its
    `state_free`, below its `free`, and where the aimed phase ties a second audit whose `SCAN_STATE` is another
    link-window's or none's (`rows-state-unheld`) that one is charged its `free` in every phase tied to it, each
    phase's scan sampled in part (`PC.sampled_in_part`). Lean accepts each with the mirror's public, and a mirror that
    holds every tied audit's reads, or none, gives another public than Lean's. With every unit of each scan proved
    (`PC.proved_whole`) a term counts the scan's own free inputs alone, no tied audit's, held or not."""
    for m in ("rows-state-anchored", "rows-state-unheld"):
        for n in range(3):
            case = PC.sampled_in_part(build(m, n, small=n % 2 == 0))
            p, c = PC.params(case["params"]), PC.commitment(bytes.fromhex(case["commitment"]))
            tied = [PC._tied_audits(c.audits, reg) for reg in c.windows]
            held = {i: PC._state_held(c.audits, c.windows, i) for ts in tied for i in ts}
            assert all(p.audits[i].state_free < p.audits[i].free for i in held), (m, n, p.audits)
            assert all(held.values()) == (m == "rows-state-anchored"), (m, n, held)
            want = [p.scan.free + sum(p.audits[i].state_free if held[i] else p.audits[i].free for i in ts)
                    for ts in tied]
            out = reference(case)
            assert out["verdict"] == ACCEPT and [s["free"] for s in out["public"]["scans"]] == want, (m, n, out)
            verdict, got = D.replay(PORT, case, server)
            assert verdict == ACCEPT and json.loads(got)["public"] == out["public"], (m, n, got)
            whole = PC.proved_whole(case)
            out = reference(whole)
            assert out["verdict"] == ACCEPT and [s["free"] for s in out["public"]["scans"]] == [
                p.scan.free] * p.windows, (m, n, out)
            verdict, got = D.replay(PORT, whole, server)
            assert verdict == ACCEPT and json.loads(got)["public"] == out["public"], (m, n, got)
    for m, always in (("rows-state-unheld", True), ("rows-state-anchored", False)):
        monkeypatch.setattr(PC, "_state_held", lambda audits, windows, i, always=always: always)
        for n in range(2):
            with pytest.raises(D.Mismatch) as x:
                D.replay(PORT, PC.sampled_in_part(build(m, n, small=True)), server)
            assert x.value.lean[0] == x.value.python[0] == ACCEPT, x.value


def test_a_scan_whose_prog_is_no_audits_output_is_refused(server, monkeypatch):
    """The scan reads each lane's Programs' rows (`scan/prog<j>`) from a compute audit's output, so the rows form
    refuses, in Lean and in the mirror and by name, each way a link-window's tie can fail at every lane
    (`PC.UNTIES`): its `scan/prog<j>` entries at another root or domain than the audit's ports', the audit's entries at
    ports that are no unit's output nor a chunk's entries port (`o0`, an input, `o01`, a non-ASCII digit, no unit, a
    chunk index with a leading zero or none), no `scan/prog<j>` entry or each under another name (`scan/prog`
    among them), the chunks' entries in the scan's own registration and in no audit's, and no root a string; and a
    mirror that skips the tie refuses only at the wiring (`PC.UNWIRED`) what Lean refuses at the tie."""
    untie = PC.UNTIES
    for u in untie:
        monkeypatch.setattr(PC, "UNTIES", (u,))
        for n in range(3):
            case = build("rows-prog-untied", n, small=n % 2 == 0)
            assert "is no compute audit's output" in reference(case)["reason"], (u, n, reference(case))
            verdict, out = D.replay(PORT, case, server)
            assert verdict == REJECT and "is no compute audit's output" in json.loads(out)["reason"], (u, n, out)
    monkeypatch.setattr(PC, "UNTIES", untie)
    monkeypatch.setattr(PC, "_prog_tied", lambda audits, reg: True)
    for n in range(2):
        case = build("rows-prog-untied", n, small=True)
        assert PC.UNWIRED in reference(case)["reason"], (n, reference(case))
        verdict, out = D.replay(PORT, case, server)
        assert verdict == REJECT and "is no compute audit's output" in json.loads(out)["reason"], (n, out)


def test_a_stray_audit_beside_the_tied_one_is_accepted_and_alone_refused(server):
    """A composite's tie of the network certifier's compute audits to its policy's record has two rows-form fixtures
    (`PC.audits_of`, `PC.with_audits`): a small `rows-honest` period, whose one execution phase's scan ties its one
    audit, and the same period with another period's audit beside its own, a stray no phase's scan ties. Lean and the
    mirror accept both alike, with the commitment registering exactly those audits' records, so the stray is the
    composite's to refuse; and with the stray in place of the tied audit, the scan's prog is no compute audit's output
    and both refuse it."""
    for n in range(3):
        tied = build("rows-honest", n, small=True)
        (own,) = PC.audits_of(tied)
        stray = PC.audits_of(build("rows-honest", 100 + n, small=True))[0]
        for case, audits in ((tied, [own]), (PC.with_audits(tied, [own, stray]), [own, stray])):
            out = reference(case)
            assert out["verdict"] == ACCEPT, (n, out)
            verdict, got = D.replay(PORT, case, server)
            assert verdict == ACCEPT and json.loads(got)["public"] == out["public"], (n, got)
            assert [PC.R.digest(a) for a in json.loads(bytes.fromhex(case["commitment"]))["audits"]] == [
                PC.R.digest(a[1]) for a in audits]
        alone = PC.with_audits(tied, [stray])
        assert "is no compute audit's output" in reference(alone)["reason"], (n, reference(alone))
        verdict, got = D.replay(PORT, alone, server)
        assert verdict == REJECT and "is no compute audit's output" in json.loads(got)["reason"], (n, got)


def test_a_scan_whose_prog_is_no_slot_entries_is_refused(server, monkeypatch):
    """The scan reads each slot's entry, read bit included, from the `SlotEntry_v1` unit of that slot's Program, lane
    j's from chunk j, so the rows form refuses, in Lean and in the mirror and by name, each way a tied link-window's
    wiring can fail at one lane (`PC.UNWIRES`): the audit's chunk entry at an output port (another unit's `o<j>`, or
    the chunk's own `o1`), at the entries port of a `SlotEntry_v1` template the audit's Program doesn't have, at
    another chunk's entries port, of other than T / every leaves, the scan's lane entry named another lane's, that
    template one member off a slot each, or that template under the slot entry's name and descriptor id with another
    body; and a mirror that skips the wiring, which the state tie reads too, accepts what Lean refuses at the
    wiring."""
    unwire = PC.UNWIRES
    for u in unwire:
        monkeypatch.setattr(PC, "UNWIRES", (u,))
        for n in range(3):
            case = build("rows-prog-unwired", n, small=n % 2 == 0)
            assert PC.UNWIRED in reference(case)["reason"], (u, n, reference(case))
            verdict, out = D.replay(PORT, case, server)
            assert verdict == REJECT and PC.UNWIRED in json.loads(out)["reason"], (u, n, out)
    monkeypatch.setattr(PC, "UNWIRES", unwire)
    monkeypatch.setattr(PC, "_wired", lambda p, audits, reg: True)
    for n in range(2):
        refused_by_lean_alone(build("rows-prog-unwired", n, small=True), server, PC.UNWIRED)


def test_a_template_named_slot_entry_with_another_body_is_refused(server, monkeypatch):
    """`wired` pins the slot entry's and the select entry's Definitions by closure digest (`PC.SLOT_ENTRY_PINS` and
    `PC.SELECT_ENTRY_PINS`, Lean's `entryPins`): the cases' templates (`PC.slot_entry`, `io_shell.SlotEntry_v1` at its
    smallest circuit-check row, and `PC.select_entry`, `SelectEntry_v1` at its smallest binding) have each list's
    first, and a template under either's name and
    descriptor id (`PC.SLOT_KEY`, `PC.SELECT_KEY`) with another body (`PC.RENAMED_SLOT_ENTRY`,
    `PC.RENAMED_SELECT_ENTRY`) has none, so Lean and the mirror refuse its link-window at the wiring, Lean by the pin;
    and a mirror that pins that digest too accepts it, the pin being all that refuses it."""
    others = {}
    for select, key, entry, pins in ((False, PC.SLOT_KEY, PC.SLOT_ENTRY, "SLOT_ENTRY_PINS"),
                                     (True, PC.SELECT_KEY, PC.SELECT_ENTRY, "SELECT_ENTRY_PINS")):
        good = json.loads(PC.audit_descriptor(((1, 2),), (2,), False, select, (1,) if select else ()))
        bad = json.loads(PC.audit_descriptor(((1, 2),), (2,), True, select, (1,) if select else ()))
        assert good["definitions"][key]["id"] == bad["definitions"][key]["id"] == entry
        assert PC.slot_entry_digest(good, key) == getattr(PC, pins)[0]
        assert (other := PC.slot_entry_digest(bad, key)) not in PC.SLOT_ENTRY_PINS + PC.SELECT_ENTRY_PINS
        others[pins] = other
    monkeypatch.setattr(PC, "UNWIRES", ("body",))
    selects = set()
    for n in range(4):
        case = build("rows-prog-unwired", n, small=n % 2 == 0)
        selects.add(any(PC.SELECT_KEY in a["program"]["definitions"] for a in case["params"]["audits"]))
        assert PC.UNWIRED in reference(case)["reason"], (n, reference(case))
        verdict, out = D.replay(PORT, case, server)
        assert verdict == REJECT and "whose template is a pinned SlotEntry_v1 or SelectEntry_v1" in json.loads(
            out)["reason"], (n, out)
        with monkeypatch.context() as mp:
            for pins, other in others.items():
                mp.setattr(PC, pins, (*getattr(PC, pins), other))
            assert reference(case)["verdict"] == ACCEPT, (n, reference(case))
    assert selects == {False, True}, selects


def test_a_tied_audits_free_inputs_are_in_its_link_windows_term(server):
    """The egress renders the outputs of the audit the scan's `prog` is tied to, so the bits the prover chose of that
    audit's inputs are the link-window's: over honest rows periods, which Lean and the mirror accept with one public,
    each scan sampled in part has as its `free` the input gates of the Programs of the audits whose registration holds
    its `prog`'s root (`input_gates`; the scan's own is 0), some of them nonzero, and no `entries`; a scan whose every
    unit is proved has the scan's own `free`, 0, and its slot entries' bits as `entries`; and each one's `tied` names
    those audits by index."""
    charged, whole = 0, 0
    for n in range(8):
        case = build("rows-honest", n, small=n % 2 == 0)
        c, p = json.loads(bytes.fromhex(case["commitment"])), PC.params(case["params"])
        out = reference(case)
        assert out["verdict"] == ACCEPT, (n, out)
        verdict, got = D.replay(PORT, case, server)
        assert verdict == ACCEPT and json.loads(got)["public"] == out["public"], (n, got)
        for reg, t, proved in zip(c["windows"], out["public"]["scans"], PC.wholes(case), strict=True):
            roots = {e["root"] for e in reg.get("roots", []) + reg.get("registered", [])
                     if PC.prog_port(e.get("port")) or PC.prog_port(e.get("value"))}
            tied = [i for i, r in enumerate(c["audits"]) if any(o.get("root") in roots for o in r["roots"])]
            want = sum(PC.input_gates(case["params"]["audits"][i]["program"]) for i in tied)
            assert (t["free"], t["entries"]) == ((0, p.scan.entries) if proved else (want, 0)), (n, t, want, proved)
            assert t["tied"] == tied and tied, (n, t, tied)
            whole += proved
        charged += any(t["free"] for t in out["public"]["scans"])
    assert charged and whole, (charged, whole)


def charged(case: dict, public: dict) -> list[int]:
    """The audits of an accepted rows-form `case` its `public` charges: drawn in part and tied to a link-window's scan
    that is sampled in part (`PC.wholes`)."""
    c = json.loads(bytes.fromhex(case["commitment"]))
    audits = c["audits"]
    part = [reg for reg, proved in zip(c["windows"], PC.wholes(case), strict=True) if not proved]
    return [i for i, f in enumerate(public["audits"]) if f["drawn"] < f["units"] and PC._tied_in(audits, part, i)]


def test_an_untied_audit_sampled_in_part_is_charged_nothing(server):
    """K charges an audit a scan's `prog` is tied to its free inputs and, sampled in part, its incorrect proof blocks as
    the entries they can set, and an untied audit nothing: over honest rows periods, which the mirror accepts, every
    tied audit samples all its proof blocks and some untied audit fewer, none is charged, and Lean accepts each such
    period with the mirror's public."""
    partial = []
    for n in range(24):
        case = build("rows-honest", n)
        out = reference(case)
        assert out["verdict"] == ACCEPT, (n, out)
        assert not charged(case, out["public"]), (n, out)
        if any(f["drawn"] < f["units"] for f in out["public"]["audits"]):
            partial.append(case)
    assert partial
    for case in partial[:3]:
        verdict, got = D.replay(PORT, case, server)
        assert verdict == ACCEPT and json.loads(got)["public"] == reference(case)["public"], got


#: The tied audits sampled in part: by template, its slot entries' and scan slots' strata whole (`rows-tied-partial`),
#: or its slot entries' or its scan slots' stratum in part too, and one or two of its proof blocks, the demo's sample.
SAMPLED_IN_PART = ("rows-tied-partial", "rows-tied-entries-sampled", "rows-tied-slots-sampled", "rows-tied-sampled")


def test_a_tied_audit_sampled_in_part_is_charged_its_link_windows_entries(server):
    """The scan reads a tied audit only through its `prog`, E_w entry bits a link-window, and its state is the scan's
    own, so an incorrect proof block of any of the audit's templates, an unsampled `SelectEntry_v1` or `ScanSlots_v2`
    one among them, changes only those bits: the rows form accepts a tied audit at any sample, Lean with the mirror's
    public, which, where that audit's link-window's scan is sampled in part too, reads the audit's draw at
    δ_i = 2^-64 (`charged`), where its at most `b` incorrect proof blocks are at least one, so its phases' terms charge
    every entry they can set, `faulty_entries` = E_w = lanes · T / every · the entry's bits, every other phase's none,
    and `eps` holds δ_i beside the scans' δ. Each way of sampling in part is charged, the demo's sample at one proof
    block and at two, and the cases' audits select (`SelectEntry_v1` beside `ScanSlots_v2`) or don't (`SlotEntry_v1`).
    Where its link-window's scan has every unit proved, that term charges E_w once as `entries`, the audit is not
    charged and `eps` is the scans' δ alone, in each way of sampling too."""
    seen = Counter()
    for m in SAMPLED_IN_PART:
        for n in range(16):
            case = build(m, n, small=n % 2 == 0)
            p, out = PC.params(case["params"]), reference(case)
            assert out["verdict"] == ACCEPT, (m, n, out)
            verdict, got = D.replay(PORT, case, server)
            assert verdict == ACCEPT and json.loads(got)["public"] == out["public"], (m, n, got)
            pub, proved = out["public"], PC.wholes(case)
            s, prog = p.schedule, PC._scan_widths(case["params"]["scan"])[1]
            assert p.scan.entries == s.lanes * (s.window // s.every) * prog, (m, n, p.scan)
            c = json.loads(bytes.fromhex(case["commitment"]))
            (i,) = [j for j, f in enumerate(pub["audits"])
                    if f["drawn"] < f["units"] and PC._tied_in(c["audits"], c["windows"], j)]
            assert [t["entries"] for t in pub["scans"]] == [p.scan.entries if w else 0 for w in proved], (m, n, pub)
            if not charged(case, pub):
                assert pub["charged"] == [] and not any(t["faulty_entries"] for t in pub["scans"]), (m, n, pub)
                assert pub["eps"] == str(sum((Fraction(t["delta"]) for t in pub["scans"]), Fraction(0))), (m, n, pub)
                seen[m, "whole"] += 1
                continue
            assert charged(case, pub) == [i], (m, n, pub)
            (read,) = pub["charged"]
            b = read["wrong_units"]
            assert read["delta"] == str(PC.TIED_DELTA) and b >= 1, (m, n, read)
            assert {k: read[k] for k in ("drawn", "units")} == {k: pub["audits"][i][k] for k in ("drawn", "units")}
            assert [t["faulty_entries"] for t in pub["scans"]] == [
                p.scan.entries if i in PC._tied_audits(c["audits"], reg) and not w else 0
                for reg, w in zip(c["windows"], proved, strict=True)], (m, n, pub)
            assert pub["eps"] == str(sum((Fraction(t["delta"]) for t in pub["scans"]), PC.TIED_DELTA))
            seen[m, "charged"] += 1
            if m == "rows-tied-sampled":
                seen[f"subset:{pub['audits'][i]['drawn']}"] += 1
            seen["select" if PC.SELECT_KEY in case["params"]["audits"][i]["program"]["definitions"] else "slot"] += 1
    want = [(m, k) for m in SAMPLED_IN_PART for k in ("charged", "whole")] + ["subset:1", "subset:2", "select", "slot"]
    assert all(seen[k] for k in want), seen


def test_a_tied_audits_draw_is_read_at_the_periods_own_delta(server):
    """One draw read at two δs, each fixed before the coins: a tied audit drawn `subset:1` of its n proof blocks, whose
    answer sampled proofs gave at δ = 1/2 with its bound there, n // 2, beside scans at δ = 2^-64. Where its
    link-window's scan is sampled in part, the rows form reads
    the same draw at δ_i = 2^-64 (`charged`): n − 1 wrong units, every undrawn one, since a set of them escapes with
    1/n ≥ δ_i, so its phases' terms charge E_w; it keeps the answer's δ and bound in `audits`, sampled proofs'
    own reading; and its eps is the scans' δ and δ_i, each its own term, (m + 1) · 2^-64 exactly. Lean gives every
    such period the mirror's public."""
    one = Facts(16, 1, Fraction(1, 2), 8, ((16, 1),))
    assert (one.at(PC.TIED_DELTA).wrong, one.at(Fraction(1, 2)).wrong, one.at(PC.TIED_DELTA).delta) == (
        15, 8, PC.TIED_DELTA)
    assert Facts(16, 16, Fraction(1, 2), 0, ((16, 16),)).at(PC.TIED_DELTA).wrong == 0
    assert Facts(16, 1, Fraction(1, 2), 8).at(PC.TIED_DELTA).wrong == 16
    assert Facts(9, 5, Fraction(1, 10), 0, ((4, 4), (5, 1))).at(Fraction(1, 2)).wrong == 2
    read = 0
    for n in range(24):
        case = build("rows-tied-sampled", n, small=n % 2 == 0)
        if not (tied := charged(case, reference(case)["public"])):
            continue
        (i,) = tied
        e = json.loads(bytes.fromhex(case["evidence"]))
        units = e["audits"][i]["profile"]["levels"][0]["size"]
        if e["audits"][i]["profile"]["levels"][0]["draw"] != {"subset": 1}:
            continue
        e["audits"][i]["profile"]["delta"], e["audits"][i]["public"]["bound"] = "1/2", units // 2
        for w in e["windows"]:
            w["answer"]["profile"]["delta"] = str(PC.TIED_DELTA)
        case = case | {"evidence": D.canonical(e).encode().hex()}
        out = reference(case)
        assert out["verdict"] == ACCEPT, (n, out)
        verdict, got = D.replay(PORT, case, server)
        assert verdict == ACCEPT and json.loads(got)["public"] == out["public"], (n, got)
        pub, c = out["public"], json.loads(bytes.fromhex(case["commitment"]))
        assert pub["audits"][i] == {"delta": "1/2", "drawn": 1, "units": units, "wrong_units": units // 2}, (n, pub)
        assert pub["charged"] == [{"delta": str(PC.TIED_DELTA), "drawn": 1, "units": units, "wrong_units": units - 1}]
        p = PC.params(case["params"])
        assert [t["faulty_entries"] for t in pub["scans"]] == [
            p.scan.entries if i in PC._tied_audits(c["audits"], reg) and not w else 0
            for reg, w in zip(c["windows"], PC.wholes(case), strict=True)], (n, pub)
        assert pub["eps"] == str(Fraction(len(pub["scans"]) + 1, 2**64)), (n, pub)
        read += 1
    assert read >= 2, read


def test_a_scan_proved_whole_is_charged_its_slot_entries_once(server):
    """Where every unit of a link-window's scan is proved, the scan's input gates the prover chooses are its slot
    entries, E_w bits, so its term charges those once (`entries`) in place of its tied audits' free inputs and
    `faulty_entries` together, whatever its tied audits' draws, and an audit tied only to such scans is not charged, so
    no δ_i is in `eps`. Over honest rows periods and each way of sampling a tied audit in part, with every scan proved
    whole (`PC.proved_whole`: a subset draw of all its units, or both directions' strata whole), Lean accepts with the
    mirror's public: each term's `free` is the scan's own, 0, its `entries` E_w and its `faulty_entries` 0, `charged`
    is empty, `eps` is the scans' δ alone, and K is the schedules, the cells and each link-window's E_w and jitter.
    With the first link-window's scan alone proved whole, an audit sampled in part stays charged exactly where a scan
    it is tied to is sampled in part. The term reads none of the tied audits' own count, so takes no minimum with it
    (`test_the_rows_forms_bound_counts_what_it_says` has that count the smaller)."""
    seen = Counter()
    for m in ("rows-honest", *SAMPLED_IN_PART):
        for n in range(6):
            base = build(m, n, small=n % 2 == 0)
            p, E = PC.params(base["params"]), PC.params(base["params"]).scan.entries
            for stratified in (False, True):
                case = PC.proved_whole(base, stratified=stratified)
                assert all(PC.wholes(case)), (m, n)
                out = reference(case)
                assert out["verdict"] == ACCEPT, (m, n, out)
                verdict, got = D.replay(PORT, case, server)
                assert verdict == ACCEPT and json.loads(got)["public"] == out["public"], (m, n, got)
                pub = out["public"]
                assert [(t["free"], t["entries"], t["faulty_entries"], t["faulty_sets"], t["faulty_bits"])
                        for t in pub["scans"]] == [(0, E, 0, 0, 0)] * p.windows, (m, n, pub)
                assert pub["charged"] == [] and not charged(case, pub), (m, n, pub)
                assert pub["eps"] == str(sum((Fraction(t["delta"]) for t in pub["scans"]), Fraction(0))), (m, n, pub)
                assert pub["K"] == pub["schedules"] + pub["cells"] + sum(E + t["jitter"] for t in pub["scans"])
            mixed = PC.proved_whole(base, windows=(0,))
            out = reference(mixed)
            verdict, got = D.replay(PORT, mixed, server)
            assert verdict == out["verdict"] == ACCEPT and json.loads(got)["public"] == out["public"], (m, n, got)
            pub, proved = out["public"], PC.wholes(mixed)
            assert proved[0] and [t["entries"] for t in pub["scans"]] == [E if w else 0 for w in proved], (m, n, pub)
            tied = charged(mixed, pub)
            assert [f["delta"] for f in pub["charged"]] == [str(PC.TIED_DELTA)] * len(tied), (m, n, pub)
            seen["mixed", bool(tied), all(proved)] += 1
    assert seen["mixed", True, False] and seen["mixed", False, False] and seen["mixed", False, True], seen


def test_an_output_port_is_a_units_past_o0():
    """`output_port` is Lean's `outputPort` on its `#guard`s: a unit's port `o<j>`, j ≥ 1 in ASCII digits without a
    leading zero."""
    assert all(map(PC.output_port, ["block-3/o1", "a/b/o10", "/o2"]))
    assert not any(map(PC.output_port, ["o1", "u/o0", "u/o01", "u/o", "u/O1", "u/o1x", "u/o\u0663", "u/o\u00b9",
                                        "u/prog", "", 1, None]))


def test_an_entries_port_is_a_slot_entry_templates_chunks_o0():
    """`entries_port_name` is Lean's `entriesPortName` on its `#guard`s: a `SlotEntry_v1` or `SelectEntry_v1`
    template's `<id>/c<j>/o0`, the chunk's index in ASCII digits without a leading zero."""
    assert PC.entries_port("SlotEntry_v1{IDB=8,KO=2,NP=4,STOP=1}") == "SlotEntry_v1{IDB=8,KO=2,NP=4,STOP=1}/c0/o0"
    assert PC.entries_port("SelectEntry_v1{}", 3) == "SelectEntry_v1{}/c3/o0"
    assert all(map(PC.entries_port_name, [PC.entries_port("SlotEntry_v1{IDB=8,KO=2,NP=4,STOP=1}"),
                                          "SlotEntry_v1{}/c0/o0", "SlotEntry_v1{KO=2}/c1/o0",
                                          PC.entries_port(PC.SELECT_KEY, 3), "SelectEntry_v1{}/c0/o0",
                                          "SelectEntry_v1{KO=2}/c10/o0"]))
    assert not any(map(PC.entries_port_name, ["SlotEntry_v2{KO=2}/c0/o0", "SlotEntry_v1{KO=2}/c01/o0",
                                              "SlotEntry_v1{KO=2}/c0/o1", "SlotEntry_v1/c0/o0", "", 1, None,
                                              "SlotEntry_v1{KO=2}/c/o0", "SlotEntry_v1{KO=2}c0/o0",
                                              "SlotEntry_v1{KO=2}/cx/o0", "SlotEntry_v1{KO=2}/c\u0663/o0",
                                              "SelectEntry_v2{KO=2}/c0/o0", "SelectEntry_v1{KO=2}/c1/o1",
                                              "SelectEntry_v1/c0/o0", "ScanSlots_v2{KO=2}/c0/o0"]))


def test_a_prog_port_is_a_lanes_programs_rows():
    """`prog_port` is Lean's `progPort` on its `#guard`s: `scan/prog<j>`, the lane's index in ASCII digits without a
    leading zero."""
    assert PC.prog_name(3) == "scan/prog3"
    assert all(map(PC.prog_port, [PC.prog_name(0), "scan/prog3", "scan/prog12"]))
    assert not any(map(PC.prog_port, ["scan/prog", "scan/prog01", "scan/prog/", "Scan/prog0", "scan/prog0x", "prog0",
                                      "scan/state", "", "scan/prog\u0663", 1, None]))


def test_each_lanes_prog_is_its_own_chunk():
    """`_wired` is Lean's `wired` on its two-lane `#guard` (T = 4, `every` 2): lane j's `scan/prog<j>` is chunk j's
    entries port, of 2 leaves, of one pinned template of 4 members; not with the lanes swapped, a chunk of other
    leaves, lane 1's tree missing, or another template's chunk 1; and the scan's lane-1 tree alone ties the audit."""
    s = types.SimpleNamespace(schedule=types.SimpleNamespace(window=4, every=2, lanes=2),
                              audits=[PC.Audit(0, 0, "", 4, 0, (("SlotEntry_v1{KO=1}", 4, PC.SLOT_ENTRY_PINS[0]),))])

    def reg(*ports) -> dict:
        return {"roots": [{"port": q, "leaves": n, "root": root, "domain": "d"} for q, root, n in ports]}

    def chunks(l0: int, l1: int, k1: str) -> dict:
        return reg(("SlotEntry_v1{KO=1}/c0/o0", "r0", l0), (k1 + "/c1/o0", "r1", l1))

    def scan(r0: str, r1: str) -> dict:
        return reg(("scan/prog0", r0, 2), ("scan/prog1", r1, 2))

    assert PC._wired(s, [chunks(2, 2, "SlotEntry_v1{KO=1}")], scan("r0", "r1"))
    assert not PC._wired(s, [chunks(2, 2, "SlotEntry_v1{KO=1}")], scan("r1", "r0"))
    assert not PC._wired(s, [chunks(3, 1, "SlotEntry_v1{KO=1}")], scan("r0", "r1"))
    assert not PC._wired(s, [chunks(2, 2, "SlotEntry_v1{KO=1}")], reg(("scan/prog0", "r0", 2)))
    assert not PC._wired(s, [chunks(2, 2, "SlotEntry_v1{KO=2}")], scan("r0", "r1"))
    assert PC._tied_audits([chunks(2, 2, "SlotEntry_v1{KO=1}")], reg(("scan/prog1", "r1", 2))) == [0]


def test_the_scan_is_read_off_its_program(server):
    """The rows form's scan terms are its registered Program's, in the generator form: T, no free input (its `prog` a
    compute audit's output), an ingress and an egress instance's committed bits (their parameters, the more of bucket
    0's and a later bucket's for ingress, and their returns: a u16 with the state and view, or with the row), and its
    slot entries' bits (lanes · T / every entries of its `prog` ports' width); and
    Lean and the mirror refuse, as parameters, a scan that isn't a descriptor, another Program's, one in the checker
    form (its egress row an input), one whose templates have other parameters or other than 1, T − 1 and T instances,
    or rows other than the schedule's, one whose lane reads another row of its input than its bucket's or another
    lane's input, one with an input gate no template's, and each negative of a check of what
    `NetTiming.BoundaryGates.Fits` asks."""
    for n in range(3):
        case = build("rows-honest", n, small=n % 2 == 0)
        p, desc = PC.params(case["params"]), case["params"]["scan"]
        state, prog, view, words = PC._scan_widths(desc)
        s, T, row = p.schedule, p.schedule.window, WR.row_bytes(p.schedule)
        ins = 8 * row + (state if T > 1 else 0) + s.lanes * prog
        assert p.scan == PC.Scan(T, 0, ins + 16 + state + view, 16 * words + 16 + 8 * row,
                                 s.lanes * (T // s.every) * prog), (n, p.scan)
        for bad in PC.bad_scans(case["params"]):
            broken = case | {"params": bad}
            assert reference(broken)["reason"].startswith("params: scan:"), (n, reference(broken))
            verdict, out = D.replay(PORT, broken, server)
            assert verdict == REJECT and "scan" in json.loads(out)["reason"], (n, out)


def test_each_fits_check_refuses_its_negative(server):
    """Each of the scan's checks of what `NetTiming.BoundaryGates.Fits` asks refuses, in Lean and in the mirror, a scan that
    passes every check before it (`PC.scan_negatives`): one past the resolution's walk, one whose root doesn't return
    an egress leaf, one whose egress template passes a parameter's leaves through, one whose bucket 0 row is a constant
    gate's, one with an egress argument a later ingress output, one whose egress state is an ingress row's input, and
    one whose ingress template names a later root node."""
    for n in range(3):
        case = build("rows-honest", n, small=n % 2 == 0)
        for name, (bad, fragment) in PC.scan_negatives(case["params"]).items():
            broken = case | {"params": bad}
            assert fragment in reference(broken)["reason"], (name, n, reference(broken))
            verdict, out = D.replay(PORT, broken, server)
            assert verdict == REJECT and fragment in json.loads(out)["reason"], (name, n, out)


def test_a_scan_in_the_checker_form_is_refused_by_name(server):
    """A scan in the checker form, whose egress template takes the row its units compare as an input, is refused at the
    parameters by Lean and the mirror, by name, though its terms would read: no egress gate is an input
    (`NetTiming.BoundaryGates.Fits`), so `K` counts only what a Program that computes its egress rows leaves free."""
    for n in range(2):
        case = build("rows-honest", n, small=True)
        s = PC.params(case["params"]).schedule
        d = json.loads(PC.scan_descriptor(s.window, WR.row_bytes(s), PC._scan_widths(case["params"]["scan"]),
                                          checker=True))
        broken = case | {"params": case["params"] | {"scan": d, "program": PC.program_sha512(d)}}
        assert "(the checker form)" in reference(broken)["reason"], (n, reference(broken))
        verdict, out = D.replay(PORT, broken, server)
        assert verdict == REJECT and "(the checker form)" in json.loads(out)["reason"], (n, out)


def test_a_wide_unit_is_charged_its_committed_bits(server):
    """A compute audit's incorrect proof blocks' bits (`faulty_bits`) are its bound times the most bits one of its
    Program's units commits, its parameters and its return, read off the Program, not a proof block's 16-bit public
    output: over honest clock-form periods whose audits' stand-in units take a 16-bit parameter and return 8 to 320
    bits, each audit is charged `b · bits`, some audit of wide units more than `16 · b`, and Lean's public is the
    mirror's."""
    wide = 0
    for n in range(12):
        case = build("honest", n)
        p, out = PC.params(case["params"]), reference(case)
        assert out["verdict"] == ACCEPT, (n, out)
        for audit, t in zip(p.audits, out["public"]["audits"]):
            assert (t["units"], t["faulty_bits"]) == (audit.units, t["wrong_units"] * audit.bits), (n, t, audit)
            wide += t["faulty_bits"] > 16 * t["wrong_units"]
        verdict, got = D.replay(PORT, case, server)
        assert verdict == ACCEPT and json.loads(got)["public"] == out["public"], (n, got)
    assert wide


#: The refusals that hold a compute audit to its Program, in either form, each with what Lean's reason and the
#: mirror's say.
AUDIT_REFUSALS = {m: why for f in ("", "rows-") for m, why in (
    (f"{f}audit-program", ("registration: not its Program",) * 2),
    (f"{f}audit-units", ("not its Program's",) * 2))}


def test_an_audit_is_held_to_its_program(server):
    """An audit whose registration names another Program than its descriptor's, or whose answer is over another number
    of units than its Program's template instances, is refused at that check by Lean and the mirror; and an audit
    whose Program isn't a descriptor, is of another schema or has no template is refused as parameters."""
    for m, (lean, mirror) in AUDIT_REFUSALS.items():
        for n in range(4):
            case = build(m, n, small=n % 2 == 0)
            assert mirror in reference(case)["reason"], (m, n, reference(case))
            verdict, out = D.replay(PORT, case, server)
            assert verdict == REJECT and lean in json.loads(out)["reason"], (m, n, out)
    for form in ("honest", "rows-honest"):
        case = build(form, 0, small=True)
        for bad in PC.bad_audits(case["params"]):
            broken = case | {"params": bad}
            assert reference(broken)["reason"].startswith("params:"), (form, reference(broken))
            verdict, out = D.replay(PORT, broken, server)
            assert verdict == REJECT and json.loads(out)["reason"].startswith("params:"), (form, bad, out)


RECORD_REFUSALS = {
    "audit-record-circuit": ("binds another circuit record",) * 2,
    "audit-record-pin": ("is not its window's registered root",) * 2,
    "audit-record-state": (f"params: an audit: circuit: a pin of {PC.SCAN_STATE}",) * 2,
}


def test_an_audit_in_the_record_form_is_counted_on_its_record(server):
    """An audit carrying its window's circuit record, wire order and pins (`record_audit`) is accepted with the mirror's
    public, its free inputs `interface.input_count`'s on the record, not its Program's input gates, a wire from a
    Boolean-profile writer's one-bit leaves to a reader's 16-bit leaves included (`audit-record-bits`); and one whose
    registration's window binds another record, or records no pin's root, or that pins the scan's state, is refused at
    that check by Lean and the mirror."""
    for m, bits in (("audit-record", False), ("audit-record-bits", True)):
        prog, doc, _ = (json.loads(x) for x in PC.record_audit(bits))
        free = IF.input_count(doc)["free"]
        assert 0 < free < PC.input_gates(prog)
        for n in range(4):
            case = build(m, n, small=n % 2 == 0)
            out = reference(case)
            assert out["verdict"] == ACCEPT, (m, n, out)
            assert [PC._audit(a, "").free for a in case["params"]["audits"] if "circuit" in a] == [free]
            verdict, got = D.replay(PORT, case, server)
            assert verdict == ACCEPT and json.loads(got).get("public") == out.get("public"), (m, n, got)
    for m, (lean, mirror) in RECORD_REFUSALS.items():
        for n in range(4):
            case = build(m, n, small=n % 2 == 0)
            assert mirror in reference(case)["reason"], (m, n, reference(case))
            verdict, out = D.replay(PORT, case, server)
            assert verdict == REJECT and lean in json.loads(out)["reason"], (m, n, out)


def test_the_record_forms_circuit_check_refuses_what_it_must(server):
    """Lean alone refuses as parameters a record-form audit whose record `NetworkCheck.Circuit.check` refuses, checks
    the mirror leaves to it (`period_case._record_count`), so Lean's answer is read without the mirror's: an order
    ranking a wire's readers below its writers, one of another format, a record over another Program, one that lists
    its members, and one that declares ingress."""
    case = build("audit-record", 0, small=True)
    i, a = next((i, a) for i, a in enumerate(case["params"]["audits"]) if "circuit" in a)
    order, doc = a["order"], a["circuit"]
    seg = order["segments"][0]
    for edit, why in [
        ({"order": order | {"segments": [seg | {"r": [-x for x in seg["r"]]}]}}, "order: wire"),
        ({"order": order | {"format": "verity/wire-order/v2"}}, "order: not a verity/wire-order/v1 certificate"),
        ({"circuit": doc | {"program": "ab" * 64}}, "not the audit's Program's SHA-512"),
        ({"circuit": doc | {"members": []}}, "lists its members"),
        ({"circuit": doc | {"ingress": [{}]}}, "declares ingress"),
    ]:
        audits = [*case["params"]["audits"][:i], a | edit, *case["params"]["audits"][i + 1:]]
        verdict, out = PORT.lean(server, case | {"params": case["params"] | {"audits": audits}})
        reason = json.loads(out)["reason"]
        assert verdict == REJECT and reason.startswith("params: an audit: circuit: ") and why in reason, (why, out)


def test_the_rows_forms_caps_are_the_schedules(server):
    """The rows form's `cells` are the schedule's caps (`Schedule.caps`, `WardenParams.caps`): `egress_cells` for each
    of the `lanes · m · window` connections the network certifier can carry, so an accepted period's schedules and
    cells terms are each that many `⌈log2 (egress_cells + 1)⌉`, never 0 for caps that miss its connections; no caps, one
    connection short or over, and a cap below `egress_cells` are refused as parameters by Lean and the mirror alike;
    and w63k4's schedule charges 384 bits for each over one link-window (Lean's `#guard`)."""
    for n in range(4):
        case = build("rows-honest", n, small=n % 2 == 0)
        p, out = PC.params(case["params"]), reference(case)
        s = p.schedule
        assert list(p.cells) == s.caps(p.windows) == [s.egress_cells] * (s.lanes * p.windows * s.window)
        term = s.lanes * p.windows * s.window * clog2(s.egress_cells + 1)
        assert out["verdict"] == ACCEPT and term > 0, (n, out)
        assert (out["public"]["schedules"], out["public"]["cells"]) == (term, term), (n, out["public"])
        verdict, got = D.replay(PORT, case, server)
        assert verdict == ACCEPT and json.loads(got)["public"] == out["public"], (n, got)
        for bad in PC.bad_caps(case["params"]):
            broken = case | {"params": bad}
            assert reference(broken)["reason"].startswith("params: cells:"), (n, reference(broken))
            verdict, got = D.replay(PORT, broken, server)
            assert verdict == REJECT and json.loads(got)["reason"].startswith("params: cells:"), (n, bad, got)
    w63k4 = WardenParams(cell_bytes=512, slots=4, bucket_ms=100, delay=27, lanes=4, queue=4, window=32,
                         ingress_cells=47, ingress_buckets=12, egress_cells=5, every=16)
    assert len(w63k4.caps(1)) == 128 and sum(clog2(c + 1) for c in w63k4.caps(1)) == 384


def test_a_period_whose_frames_overrun_a_u32_is_refused(server):
    """The rows form refuses a period whose `lanes · windows · window` connections, the k-th at ordinal k + 1, aren't
    distinct u32s, or whose cell's B bytes aren't under 2^32 (`rowsWf`): its frames could not name every connection or
    every payload's length, so its record would not read back (`NetTiming.RowsFormat.Wf`). As parameters it is refused
    before its caps, which a period that long could never list, by Lean and the mirror alike, and the honest period's
    schedule passes `rowsWf` up to the last whole period under 2^32. The check refuses it too (`rowsWfCheck`), which the
    mirror's check shows at an empty period whose B is 2^32: no parameters `params` reads reach it."""
    fragment = "rows format: a frame's ordinal and payload length are u32s"
    for n in range(2):
        case = build("rows-honest", n, small=True)
        p = PC.params(case["params"])
        per = p.schedule.lanes * p.schedule.window
        over = -(-PC.U32 // per)
        assert PC.rows_wf(p.schedule, p.windows) and PC.rows_wf(p.schedule, (PC.U32 - 1) // per), (n, per)
        assert not PC.rows_wf(p.schedule, over), (n, per)
        broken = case | {"params": case["params"] | {"windows": over}}
        assert reference(broken)["reason"].startswith(f"params: {fragment}"), (n, reference(broken))
        verdict, out = D.replay(PORT, broken, server)
        assert verdict == REJECT and json.loads(out)["reason"].startswith(f"params: {fragment}"), (n, out)
        empty = dataclasses.replace(p, audits=(), windows=0)
        for b, refused in ((PC.U32 - 1, False), (PC.U32, True)):
            q = dataclasses.replace(empty, schedule=dataclasses.replace(p.schedule, cell_bytes=b))
            got = PC.check(q, PC.Commitment([], [], b""), ([], []))
            assert (got[0] == REJECT and got[1].startswith(fragment)) == refused, (n, b, got)


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
    they fail (a digest only once the commitment decodes); an accepted period's eps is its units' deltas summed (the
    clock form's audits', the rows form's scans' and δ_i for each tied audit drawn in part, which its `charged` reads
    there), and its profile names no sync; in
    the clock form its evidence holds
    each clock certificate alone, and in the rows form each link-window's three records' certificates and two row
    registrations, with one scan a link-window, all at one δ, and a schedules term equal to the cells';
    `cases(seed, form)` is one of each with its answer; and Lean gives every one of them the mirror's answer."""
    kinds = PC.kinds(form)
    assert kinds == KINDS, kinds
    drawn = {(k, s): PC.case(k, s, form) for k in kinds for s in range(6)}
    answers = {ks: PC.answer(c) for ks, c in drawn.items()}
    assert {ks: a["verdict"] for ks, a in answers.items()} == {(k, s): verdict_of(k) for k, s in drawn}
    assert all(a["reason"].startswith(k.removeprefix("broken-") + ":")
               and (a["commitment"] is None) == (k != "broken-evidence")
               for (k, _), a in answers.items() if k.startswith("broken-"))
    accepted = [a["public"] for (k, _), a in answers.items() if k == ACCEPT]
    for ks, a in answers.items():
        if ks[0] == ACCEPT:
            x = a["public"]
            reads = [f["delta"] for f in x.get("charged", [])]
            assert reads == [str(PC.TIED_DELTA)] * len(charged(drawn[ks], x) if "scans" in x else []), (ks, x)
            assert isinstance(x["K"], int) and x["eps"] == str(sum(
                (Fraction(t["delta"]) for t in x.get("scans", x["audits"])), PC.TIED_DELTA * len(reads))), (ks, x)
    assert all(sorted(a["profile"]) == ["link", "windows"] for (k, _), a in answers.items() if k == ACCEPT)
    shown = [w for (k, _), c in drawn.items() if k == ACCEPT
             for w in json.loads(bytes.fromhex(c["evidence"]))["windows"]]
    if form == "clock":
        assert all(sorted(w["clock"]) == PC.CERTIFICATE_KEYS for w in shown)
        assert all("scans" not in x for x in accepted)
    else:
        assert all(sorted(w) == PC.ROWS_WINDOW_KEYS and len(w["certificates"]) == 3 and len(w["registrations"]) == 2
                   for w in shown)
        assert all(x["schedules"] == x["cells"] and len(x["scans"]) == x["windows"]
                   and len({t["delta"] for t in x["scans"]}) == 1 for x in accepted)
    assert PC.cases(3, form) == {k: (drawn[k, 3], answers[k, 3]) for k in kinds}
    D.run(dataclasses.replace(PORT, inputs=st.sampled_from(list(drawn.values()))), examples=2 * len(drawn),
          server=server)


def test_the_bound_counts_what_it_says():
    """K's terms on a period worked by hand: 10 units, each committing 16 bits, drawn subset:3 at δ = 1/10 with at most
    4 incorrect, 7 cells on one connection and none on another, 2 syncs over 3 link-windows, jitter below 2 over 3
    steps; the same audit with units committing 40 bits, its incorrect proof blocks' bits 4 · 40; and eps over audits at
    δ = 1/10, 1/15 and 1/6, and over none (Lean's `#guard`s)."""
    p = Params((PC.Audit(0, 3, "", 10, 16),), (7, 0), 0, 2, "a", None, b"", "",
               SyncSet((0,), (Fraction(1), Fraction(2))), 3)
    out = public(p, [Facts(10, 3, Fraction(1, 10), 4)])
    assert (out["schedules"], out["cells"], out["audits"][0]["faulty_sets"], out["audits"][0]["faulty_bits"],
            out["audits"][0]["jitter"], out["K"]) == (3, 3, 9, 64, 3, 3 + 3 + 9 + 64 + 3)
    wide = public(dataclasses.replace(p, audits=(PC.Audit(0, 3, "", 10, 40),)), [Facts(10, 3, Fraction(1, 10), 4)])
    assert (wide["audits"][0]["faulty_bits"], wide["K"]) == (160, 3 + 3 + 9 + 160 + 3)
    assert [clog2(x) for x in range(6)] == [0, 0, 1, 2, 2, 3]
    assert [at_most(10, 0), at_most(10, 1), at_most(10, 2), at_most(4, 9), at_most(0, 3)] == [1, 11, 56, 16, 1]
    assert out["eps"] == "1/10"
    three = dataclasses.replace(p, audits=(PC.Audit(0, 0, "", 1, 16),) * 3)
    assert public(three, [Facts(1, 1, Fraction(1, d), 0) for d in (10, 15, 6)])["eps"] == "1/3"
    assert public(dataclasses.replace(p, audits=()), [])["eps"] == "0"


def test_an_audits_free_inputs_are_its_programs_input_gates():
    """`auditOf` counts an audit's free inputs off its descriptor (`inputGates`, `input_gates`), never from the
    parameters: each template's instances times its parameters' bits, from the templates' signatures, over the stand-in
    unit, the slot entries' and the select entries' templates and mixes of them; a parameter bound at registration
    counts as one bound per run does; and parameters stating a `free` are refused."""
    for units, entries, select in [(((3, 2),), (), False), (((1, 1), (2, 40)), (4,), False), (((2, 5),), (3,), True)]:
        desc = json.loads(PC.audit_descriptor(units, entries, select=select))
        entry = PC.select_entry() if select else PC.slot_entry()
        templates = [(bind(PC.AUDIT_UNIT, OUT=out), n) for n, out in units] + [(entry, n) for n in entries]
        want = sum(n * sum(sum(t.leaf_widths()) for _, t in d.signature()[0]) for d, n in templates)
        assert PC.input_gates(desc) == want > 0, (units, entries, select)
    run, reg = (json.loads(D.canonical(codec.encode_program(Program(bind(PC.AUDIT_UNIT, OUT=2), registered=r))))
                for r in ((), ("x",)))
    assert run != reg and PC.input_gates(run) == PC.input_gates(reg) == 16
    case = next(c for n in range(8) if (c := build("rows-honest", n))["params"]["audits"])
    p = case["params"]
    stated = p | {"audits": [a | {"free": PC.input_gates(a["program"])} for a in p["audits"]]}
    assert reference(case)["verdict"] == ACCEPT and reference(case | {"params": stated})["verdict"] == REJECT


def test_the_rows_forms_bound_counts_what_it_says():
    """The rows form's K worked by hand (Lean's `rowsTerms` `#guard`s): 7 cells on one connection and 2 on another, 5
    bits for the schedules and 5 again for the fail cells, an audit whose own terms are in none, and over 1 link-window
    one scan of 20 units (T = 10) drawn at δ = 1/15 with at most 3 incorrect (1351 sets), 1,120 free inputs, instances
    committing 300 and 512 bits, and jitter below 2 over its 10 steps; with every unit proved, no term for incorrect
    proof blocks, and the scan's 96 entry bits once in place of its tied audits' 500 free inputs and the 96 their
    incorrect proof blocks can set; at w63's class with the egress tie (T = 32), sampled at all but one unit with at
    most 1 incorrect, K the tied slot entries' free inputs, 8 · 1,136 = 9,088 bits, beside its one of 65 sets and 512
    bits, and with every unit proved its 8 slot entries of 112 bits, 896; at w63k4's with its tied audit drawn in part
    with at most 2 incorrect, K those 9,088 and its E_w = 896 bits beside the scan's own where the scan is sampled in
    part, and E_w once where every unit of it is proved; no minimum, a scan proved whole whose tied audits count 100
    free inputs charged 896; and eps over the scans' δ alone, or with a charged audit's beside them. `Facts.whole` on
    Lean's `#guard`: a subset draw of every unit or strata each whole that hold them all, not a draw in part, a
    stratum drawn past its size beside one in part, strata short of the units, or no draw."""
    p = Params((PC.Audit(0, 3, "", 10, 16),), (7, 2), 0, 2, "a", None, b"", "", None, 1, WardenParams(),
               PC.Scan(10, 1120, 300, 512))
    out = public(p, [Facts(10, 3, Fraction(1, 10), 4)], [(Facts(20, 5, Fraction(1, 15), 3), 0, 0, [])])
    (scan,) = out["scans"]
    assert at_most(20, 3) == 1351 and clog2(1351) == 11
    assert (out["schedules"], out["cells"], scan["free"], scan["faulty_sets"], scan["faulty_bits"], scan["jitter"],
            scan["faulty_entries"], scan["entries"], scan["tied"], out["K"]) == (
                5, 5, 1120, 11, 3 * 512, 10, 0, 0, [], 5 + 5 + (1120 + 11 + 3 * 512 + 10))
    assert out["audits"] == [{"delta": "1/10", "drawn": 3, "units": 10, "wrong_units": 4}]
    d = Fraction(1, 100)
    all64, most64 = Facts(64, 64, d, 0, ((64, 64),)), Facts(64, 63, d, 1, ((64, 63),))
    assert all64.whole and Facts(64, 64, d, 0, ((32, 32), (32, 32))).whole
    assert not any(f.whole for f in (most64, Facts(64, 64, d, 0, ((32, 33), (32, 31))),
                                     Facts(64, 32, d, 0, ((32, 32),)), Facts(64, 64, d, 0)))
    drawn = dataclasses.replace(p, audits=(), cells=(), jitter=1, scan=PC.Scan(4, 0, 300, 512, 96))
    assert public(drawn, [], [(Facts(8, 8, d, 0, ((8, 8),)), 500, 96, [])])["K"] == 96
    w63 = dataclasses.replace(p, audits=(PC.Audit(8 * 1136, 0, "", 8, 1136),), cells=(), jitter=1,
                              scan=PC.Scan(32, 0, 300, 512, 8 * 112))
    tied = public(w63, [Facts(8, 8, d, 0)], [(most64, 8 * 1136, 0, [0])])
    assert (tied["scans"][0]["free"], tied["scans"][0]["entries"], tied["scans"][0]["tied"], tied["K"]) == (
        9088, 0, [0], 9088 + 7 + 512)
    tied = public(w63, [Facts(8, 8, d, 0)], [(all64, 8 * 1136, 0, [0])])
    assert (tied["scans"][0]["free"], tied["scans"][0]["entries"], tied["scans"][0]["tied"], tied["K"]) == (
        0, 896, [0], 896)
    part = public(w63, [Facts(16, 9, d, 2)], [(most64, 9088, 8 * 112, [0])])
    assert (part["scans"][0]["faulty_entries"], part["scans"][0]["tied"], part["K"]) == (
        896, [0], 9088 + 7 + 512 + 896)
    part = public(w63, [Facts(16, 9, d, 2)], [(all64, 9088, 8 * 112, [0])])
    assert (part["scans"][0]["faulty_entries"], part["scans"][0]["entries"], part["K"]) == (0, 896, 896)
    assert public(w63, [], [(all64, 100, 0, [])])["K"] == 896
    assert public(w63, [], [(Facts(64, 63, d, 0, ((64, 63),)), 100, 0, [])])["K"] == 100
    two = dataclasses.replace(p, cells=(), windows=2, scan=PC.Scan(0, 0, 0, 0))
    assert public(two, [Facts(1, 1, Fraction(1, 10), 0)],
                  [(Facts(1, 1, Fraction(1, 15), 0), 0, 0, [])] * 2)["eps"] == "2/15"
    one = dataclasses.replace(two, windows=1)
    assert public(one, [Facts(2, 1, Fraction(1, 10), 0)], [(Facts(1, 1, Fraction(1, 15), 0), 0, 0, [])],
                  [Facts(2, 1, Fraction(1, 10), 0)])["eps"] == "1/6"
