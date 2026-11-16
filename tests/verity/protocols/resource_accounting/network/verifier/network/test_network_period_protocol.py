"""The network attester over a period (`NetworkCheck.Period`, `verity/certifier/v0`, `network-period serve`) against its
Python mirror, through the difftest harness's `ProtocolPort`.

The mirror and the generator are `period_case.py`'s: `reference` (its `answer`) checks a period in the core interface's
answer format, `build` makes a period with one change (`EXPECT`, each aimed at one path of the check, or malformed
parameters, commitments, evidence and cases), and `case` a small period aimed at a verdict or at a decoding failure
(`KINDS`). This module re-exports
what the composition's difftest reads of it (`reference`, `build`, `EXPECT`): it samples onsite's certifier cases from
`build` and answers them with `reference` (`tests/verity/protocols/compose/verifier/lean/reference.py`)."""

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
from period_case import (ACCEPT, EXPECT, INGRESS, KINDS, MUTATIONS, REJECT, UNDECIDED, Facts, Params, at_most, build,
                         clog2, public, reference, verdict_of)
from verity.protocols.resource_accounting.network.verifier import scan_ok
from verity.protocols.resource_accounting.network.verifier import network_attester_rows as WR
from verity.protocols.resource_accounting.network.verifier.certificate import Certificate
from verity.protocols.resource_accounting.network.verifier.network_attester import NetworkCertifierParams
from verity.protocols.proofs.spot.verifier.one_stage import docket as R
from verity.protocols.proofs.spot.verifier.profile import Stratified, Stratum

__all__ = ["EXPECT", "MUTATIONS", "build", "reference"]

LEAN = Path(__file__).resolve().parents[7] / "verity" / "protocols" / "resource_accounting" / "network" / "verifier" / "lean"
pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

TAGS: dict[str, str] = {}
SEEN: Counter = Counter()


#: The changes sampled more often than the rest: the accepted periods, the honest one and the tied audit sampled in
#: part, and the undecided answers, the only undecided ones.
WEIGHTS = {"rows-honest": 4, "rows-tied-partial": 2, "rows-tied-sampled": 2, "rows-answer-undecided": 2,
           "rows-audit-undecided": 2}


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
    """Lean and the mirror agree on every case, and each change reaches its verdict at least once: a change the run's
    sample leaves out is replayed once, at seed 0, so adding a change doesn't fail another for its samples."""
    SEEN.clear()
    examples = 1200
    stats = D.run(PORT, examples=examples, server=server)
    for tag in (tag for tag in EXPECT if not any(SEEN[(tag, v)] for v in (ACCEPT, REJECT, UNDECIDED))):
        case = build(tag, 0)
        TAGS[D.canonical(case)] = tag
        D.replay(PORT, case, server)
    print("\n".join(f"{tag:22} {verdict:10} {k}" for (tag, verdict), k in sorted(SEEN.items())))
    missed = [tag for tag, verdict in EXPECT.items() if not SEEN[(tag, verdict)]]
    assert not missed, (missed, SEEN)
    assert min(stats.counts.get(v, 0) for v in (ACCEPT, REJECT, UNDECIDED)) > examples // 50, stats


def test_each_verdict_is_the_mirrors(server):
    """A short run over an honest period, an undecided execution phase and a forged certificate: all three verdicts."""
    port = dataclasses.replace(PORT, inputs=cases(("rows-honest", "rows-answer-undecided", "rows-cert-forged")))
    stats = D.run(port, examples=40, server=server)
    assert all(stats.counts.get(v) for v in (ACCEPT, REJECT, UNDECIDED)), stats


def refused_by_lean_alone(case: dict, server, why: str) -> None:
    """A mirror with a check skipped accepts `case`, and Lean refuses it, its reason naming `why`."""
    with pytest.raises(D.Mismatch) as x:
        D.replay(PORT, case, server)
    assert (x.value.lean[0], x.value.python[0]) == (REJECT, ACCEPT), x.value
    assert why in json.loads(x.value.lean[1])["reason"], x.value


def test_a_mirror_that_skips_the_ingress_certificate_is_caught(server, monkeypatch):
    """A mirror that reads only the egress records' certificates accepts a period whose requests' certificate is forged,
    and Lean refuses it at the certificates."""
    real = PC._certifies_on
    monkeypatch.setattr(PC, "_certifies_on", lambda p, link, d, w, c: d is INGRESS or real(p, link, d, w, c))
    skipped = [c for n in range(12) if reference(c := build("rows-cert-forged", n, small=True))["verdict"] == ACCEPT]
    assert skipped
    for case in skipped:
        refused_by_lean_alone(case, server, "certificates")


def test_a_mirror_that_skips_the_rows_check_is_caught(server, monkeypatch):
    """A mirror that doesn't read the row registrations accepts a rows-form period whose rows a key other than the
    listed one signed, and Lean refuses it at the rows."""
    monkeypatch.setattr(PC, "_rows_check", lambda p, w, d, doc: None)
    for n in range(2):
        refused_by_lean_alone(build("rows-reg-key", n, small=True), server, "not under a listed key")


#: The rows form's refusals the period must make, each with what Lean's reason and the mirror's say.
ROWS_REFUSALS = {
    "rows-cert-missing": ("certificates: the requests', responses' and fails' records'", "certificates"),
    "rows-past-slots": ("certifies none of a window in which it refused a connection", "refused a connection"),
    "rows-reg-key": ("not under a listed key in its session", "doesn't verify under"),
    "rows-answer-window": ("another window's answer", "another window's answer"),
    "rows-schedule": ("another schedule's digest than the bound one", "the registration's parameters is"),
    "rows-cap-schedule": ("another schedule's digest than the bound one", "the registration's parameters is"),
    "rows-delta": ("is not the period's", "is not the period's"),
    "rows-units": ("not the scan's 2T", "not the scan's 2T"),
    "rows-prog-untied": ("is no compute audit's output",) * 2,
    "rows-prog-unwired": ("is no compute audit's slot entries",) * 2,
    "rows-state-untied": (PC.UNSTATED,) * 2,
    "rows-answer-tied-unheld": (PC.TIE_UNHELD,) * 2,
    "rows-state-misindexed": (PC.MISREAD,) * 2,
    "rows-start-unpinned": (PC.UNPINNED,) * 2,
    "rows-start-carried": (PC.CARRIED,) * 2,
    "rows-circuit-swapped": (PC.UNHELD,) * 2,
    "rows-ok-unheld": (PC.UNOK,) * 2,
    "rows-one-tick": (f"{PC.UNCOMPUTED}: T = 1: a scan has at least two ticks",) * 2,
}


def test_the_rows_form_refuses_what_it_must(server):
    """An execution phase missing one of its records' certificates, one the network attester left uncertified for
    refusing a connection (traffic past its Program slots), a registration not under a listed key, the scan's answer
    for another window's rows, rows under a schedule whose digest isn't the bound one (one field off, or another
    response cap or none), scans at two δ, a scan answered over
    other than its 2T + 1 units, a scan whose `prog` is no compute audit's output, one whose `prog` is an audit's output
    but no `SlotEntry_v1` or `SelectEntry_v1` template's entries, a scan whose state is not its wired audit's
    `SCAN_STATE`, a scan whose answer names a tied Input that is no lane's `prog<j>`, one whose start is not opened to
    the empty state, a period of more than one execution phase, a tied audit registered under another Program than the
    class circuit, a scan whose ok bits are not
    at the schedule's constants, and one under a schedule of one tick, which has no constants: Lean and the mirror each
    refuse every one at that check."""
    for m, (lean, mirror) in ROWS_REFUSALS.items():
        for n in range(4):
            case = build(m, n, small=n % 2 == 0)
            assert mirror in reference(case)["reason"], (m, n, reference(case))
            verdict, out = D.replay(PORT, case, server)
            assert verdict == REJECT and lean in json.loads(out)["reason"], (m, n, out)


def certificates(case: dict) -> list[Certificate]:
    """A case's certificates, its row registrations' among them."""
    windows = json.loads(bytes.fromhex(case["evidence"]))["windows"]
    regs = [{"key_id": (c := doc["certificate"])["key"], "link": doc["link"], "direction": doc["direction"],
             "window": doc["window"], "digest": c["digest"], "signature": c["signature"]}
            for x in windows for doc in x.get("registrations", [])]
    return [PC._certificate(c, "a certificate") for c in [
        *(c for x in windows for c in x["certificates"]), *regs]]


def signers(case: dict) -> set[str]:
    """The identifiers of the keys a case's certificates and row registrations name as their signers."""
    return {c.key_id.hex() for c in certificates(case)}


def by_window(case: dict) -> list[tuple[int, list[Certificate]]]:
    """A case's execution phases by the period's numbers, each with its certificates, its row registrations' among
    them."""
    windows = json.loads(bytes.fromhex(case["evidence"]))["windows"]
    return [(case["params"]["first"] + i, certificates({"evidence": D.canonical({"windows": [x]}).encode().hex()}))
            for i, x in enumerate(windows)]


def named(reason: str) -> set[str]:
    """The names a refusal's reason gives its signers: not listed, or a listed key outside its stretch."""
    return {name for name in (PC.UNLISTED, PC.OUTSIDE) if name in reason}


def listed_ids(case: dict) -> list[str]:
    """The key identifiers of the entries a case's parameters list, in their order."""
    return [e["key"][:32] for e in case["params"]["keys"]]


def weaken(monkeypatch, pick) -> None:
    """A mirror that holds a certificate or a row registration for the period's execution phase `w` to the entry
    `pick(p, w, key_id)` gives for the key identifier it names, where the mirror picks by the execution phase alone
    (`PC._entry_at`)."""
    def certifies_on(p, link, d, w, c):
        e = pick(p, w, c.key_id)
        return (e is not None and c.direction is d and c.link == link and c.window == w - e.start
                and c.verify(e.key, e.session))

    def rows_check(p, w, d, doc):
        e = pick(p, w, bytes.fromhex(doc["certificate"]["key"]))
        try:
            if e is None:
                raise ValueError("no entry")
            WR.check_registration(doc, e.key, e.session, p.schedule, p.link, d, w - e.start)
        except ValueError as x:
            raise PC.Finding(REJECT, f"window {w}: the {d.value} rows: {x}") from None

    monkeypatch.setattr(PC, "_certifies_on", certifies_on)
    monkeypatch.setattr(PC, "_rows_check", rows_check)


def test_an_execution_phase_picks_its_entry_and_is_signed_at_its_sessions_number(server, monkeypatch):
    """The parameters list the certifier's entries, each a key, its session and the start of its stretch, and the
    period's execution phase alone picks the entry its certificates and row registrations are held to: that entry's key,
    its session, and that session's number for the execution phase, the period's less the entry's start. A period
    whose every signature is the second of two entries' (`second-key`), and one whose one execution phase is the
    second's stretch's first, the one after the first entry's last, signed by the second at its number 0
    (`two-stretches`), are accepted by Lean with the mirror's public; and a mirror that reads only the first listed
    entry, or holds a certificate to the period's number, refuses what Lean accepts."""
    cases = {(m, n): build(m, n, small=n % 2 == 0)
             for m in ("rows-second-key", "rows-two-stretches") for n in range(4)}
    for (m, n), case in cases.items():
        ids = listed_ids(case)
        assert len(ids) == 2 and signers(case) == {ids[1]}, (m, n)
        p = PC.params(case["params"])
        assert p.entries[0].session != p.entries[1].session and p.entries[1].start >= 1, (m, n)
        for w, certs in by_window(case):
            e = PC._entry_at(p, w)
            (other,) = [x for x in p.entries if x is not e]
            for c in certs:
                assert c.key_id == e.key.key_id and c.window == w - e.start and c.verify(e.key, e.session), (m, n, w, c)
                assert not c.verify(e.key, other.session), (m, n, w, c)
        if "two-stretches" in m:
            assert p.windows == 1 and p.first == p.entries[1].start > p.entries[0].start, (m, n)
        out = reference(case)
        assert out["verdict"] == ACCEPT, (m, n, out)
        verdict, got = D.replay(PORT, case, server)
        assert verdict == ACCEPT and json.loads(got)["public"] == out["public"], (m, n, got)
    real = PC._entry_at
    for skipped in (lambda p, w: p.entries[0] if p.entries[0].start <= w else None,
                    lambda p, w: (e := real(p, w)) and dataclasses.replace(e, start=0)):
        monkeypatch.setattr(PC, "_entry_at", skipped)
        for m in ("rows-second-key", "rows-two-stretches"):
            with pytest.raises(D.Mismatch) as x:
                D.replay(PORT, cases[m, 0], server)
            assert (x.value.lean[0], x.value.python[0]) == (ACCEPT, REJECT), x.value


#: The changes in which a listed entry signs outside its stretch: two listed entries in turn over a period the second's
#: stretch holds, and one certificate or one row registration under the first.
OUTSIDE_STRETCH = ("rows-two-signers", "rows-cert-outside-stretch", "rows-reg-outside-stretch")


def test_a_mirror_that_looks_the_entry_up_by_the_certificates_id_is_caught(server, monkeypatch):
    """The execution phase picks the entry, and nothing a certificate names does. A period of two listed entries whose
    execution phases the second's stretch holds, in which the first entry signs too, in its own session at its own
    number for the execution phase (every other signature in `two-signers`, one certificate or one row registration in
    `…-outside-stretch`), is refused by Lean and by the mirror and by name (`PC.OUTSIDE`), though every signature
    verifies under a listed key in that key's session, and every certificate at that session's number; and a mirror that
    looks the entry up by the key identifier the certificate names accepts what Lean refuses by that name."""
    cases = {(m, n): build(m, n, small=n % 2 == 0) for m in OUTSIDE_STRETCH for n in range(4)}
    for (m, n), case in cases.items():
        p = PC.params(case["params"])
        first, second = p.entries
        by_id = {e.key.key_id: e for e in p.entries}
        assert signers(case) == set(listed_ids(case)), (m, n)
        outside = 0
        for w, certs in by_window(case):
            assert PC._entry_at(p, w) is second, (m, n, w)
            for c in certs:
                e = by_id[c.key_id]
                assert c.verify(e.key, e.session), (m, n, w, c)
                assert "reg" in m or c.window == w - e.start, (m, n, w, c)
                outside += e is first
        assert outside >= 1, (m, n)
        out = reference(case)
        assert out["verdict"] == REJECT and PC.OUTSIDE in out["reason"], (m, n, out)
        verdict, got = D.replay(PORT, case, server)
        assert verdict == REJECT and PC.OUTSIDE in json.loads(got)["reason"], (m, n, got)
    weaken(monkeypatch, lambda p, w, key_id: next((e for e in p.entries if e.key.key_id == key_id), None))
    for m in OUTSIDE_STRETCH[:2]:
        for n in range(3):
            refused_by_lean_alone(cases[m, n], server, PC.OUTSIDE)


#: The changes that put a certificate or a row registration under the picked entry's key in the other entry's session.
OTHER_SESSION = ("rows-cert-other-session", "rows-reg-other-session")
#: The changes that put a certificate or a row registration under the picked entry at the period's number for the
#: execution phase, not its session's.
PERIOD_NUMBER = ("rows-cert-period-number", "rows-reg-period-number")


def test_a_signature_in_another_entrys_session_or_at_the_periods_number_is_refused(server, monkeypatch):
    """An entry's session and numbers are its own: a period of two listed entries, the second's stretch holding its
    execution phases, in which one certificate (one of the three records') or one row registration is the second entry's key's in the first entry's session, or names
    the period's number for its execution phase where the session's belongs, is refused by Lean and by the mirror, as no
    unlisted signer's and no listed key's outside its stretch; and a mirror that takes the picked key's certificate in
    any listed session, or at either number, accepts what Lean refuses."""
    for m in (*OTHER_SESSION, *PERIOD_NUMBER):
        for n in range(4):
            case = build(m, n, small=n % 2 == 0)
            p = PC.params(case["params"])
            first, second = p.entries
            assert first.session != second.session and second.start >= 1, (m, n)
            good = bad = 0
            for w, certs in by_window(case):
                assert PC._entry_at(p, w) is second, (m, n, w)
                for c in certs:
                    assert c.key_id == second.key.key_id, (m, n, w, c)
                    ok = c.window == w - second.start and c.verify(second.key, second.session)
                    good += ok
                    bad += not ok and (c.window == w - second.start and c.verify(second.key, first.session)
                                       if m in OTHER_SESSION else c.window == w and c.verify(second.key, second.session))
            assert bad == 1 and good == len(certificates(case)) - 1, (m, n, good, bad)
            out = reference(case)
            assert out["verdict"] == REJECT and not {PC.UNLISTED, PC.OUTSIDE} & named(out["reason"]), (m, n, out)
            verdict, got = D.replay(PORT, case, server)
            assert verdict == REJECT and not {PC.UNLISTED, PC.OUTSIDE} & named(json.loads(got)["reason"]), (m, n, got)
    real = PC._entry_at

    def in_any_session(p, link, d, w, c):
        e = real(p, w)
        return (e is not None and c.direction is d and c.link == link and c.window == w - e.start
                and any(c.verify(e.key, x.session) for x in p.entries))

    def at_either_number(p, link, d, w, c):
        e = real(p, w)
        return (e is not None and c.direction is d and c.link == link and c.window in (w, w - e.start)
                and c.verify(e.key, e.session))

    for weaker, ms in ((in_any_session, OTHER_SESSION[:1]), (at_either_number, PERIOD_NUMBER[:1])):
        monkeypatch.setattr(PC, "_certifies_on", weaker)
        for m in ms:
            for n in range(3):
                refused_by_lean_alone(build(m, n, small=True), server, "window ")


#: The changes that put a certificate or a row registration under a key the parameters don't list.
UNLISTED = ("rows-cert-unlisted", "rows-reg-unlisted")
#: The changes to a signature's key or session: a key that isn't listed or another key under the picked entry's
#: identifier, and another session.
KEY_AND_SESSION = ("rows-cert-key", "rows-cert-session", "rows-reg-key", "rows-reg-session")


def test_a_signer_that_is_not_listed_is_refused_by_name(server, monkeypatch):
    """A certificate (one of the three records') or a row registration whose signer is not one of the listed keys is refused, by Lean and by the mirror and by name
    (`PC.UNLISTED`), though its signature verifies under its own key; the changes to a signature's key and session
    still refuse, none of a listed signer's naming it; and a mirror that takes the unlisted key as the picked entry's
    accepts what Lean refuses by that name."""
    for m in UNLISTED:
        for n in range(4):
            case = build(m, n, small=n % 2 == 0)
            assert not signers(case) <= set(listed_ids(case)), (m, n)
            out = reference(case)
            assert out["verdict"] == REJECT and PC.UNLISTED in out["reason"], (m, n, out)
            verdict, got = D.replay(PORT, case, server)
            assert verdict == REJECT and PC.UNLISTED in json.loads(got)["reason"], (m, n, got)
    for m in KEY_AND_SESSION:
        for n in range(4):
            case = build(m, n, small=n % 2 == 0)
            listed = signers(case) <= set(listed_ids(case))
            out = reference(case)
            assert out["verdict"] == REJECT and (PC.UNLISTED in out["reason"]) == (not listed), (m, n, out)
            verdict, got = D.replay(PORT, case, server)
            assert verdict == REJECT and (PC.UNLISTED in json.loads(got)["reason"]) == (not listed), (m, n, got)
        assert "key" in m or listed, m
    # a small period's keys are `PC.KEYS[0]`'s, and its unlisted signer that parameter set's key of seed 1
    other = PC.LmsKey(PC._key(0).params, PC.lms.public_key(PC._key(0, 1)))
    real = PC._entry_at
    weaken(monkeypatch, lambda p, w, key_id: (e := real(p, w)) and (
        dataclasses.replace(e, key=other) if key_id == other.key_id else e))
    for m in UNLISTED:
        refused_by_lean_alone(build(m, 0, small=True), server, PC.UNLISTED)


#: The entries the parameters must not list (`PC.bad_entries`), each with what Lean's reason and the mirror's say.
ENTRIES_REFUSALS = {
    "empty": PC.NO_KEY, "out-of-order": PC.OUT_OF_ORDER, "overlap": PC.OVERLAP, "twice": PC.TWICE,
    "one-id": PC.TWO_KEYS, "before": PC.BEFORE, "no-entry": PC.NO_ENTRY,
    "key": PC.KEY_NOT_KEYS, "key-too": PC.KEY_NOT_KEYS, "session": PC.SESSION_IN_ENTRY,
    "sessions": PC.SESSIONS_IN_ENTRY}


def test_entries_the_parameters_must_not_list_are_refused_by_name(server):
    """Parameters that list no entry; entries out of order, one starting before the one listed before it; two entries
    with one start, whose stretches would overlap; one key identifier and one session twice, whose execution phase n
    would be two of the period's; another key under a listed key's identifier; a period that starts before the first entry's stretch; an entry that
    is the key alone; or that still name `"key"`, `"session"` or `"sessions"`: each is refused at the parameters by Lean
    and by the mirror and by name. So is a list or an entry that doesn't read."""
    for m in ("rows-honest", "rows-second-key"):
        case = build(m, 0, small=True)
        bad = PC.bad_entries(case["params"])
        assert set(ENTRIES_REFUSALS) < set(bad), (m, sorted(bad))
        for name, params in bad.items():
            why = ENTRIES_REFUSALS.get(name, "")
            out = reference(case | {"params": params})
            assert out["verdict"] == REJECT and why in out["reason"], (m, name, out)
            verdict, got = D.replay(PORT, case | {"params": params}, server)
            assert verdict == REJECT and why in json.loads(got)["reason"], (m, name, got)


def shown_starts(case: dict) -> list[PC.CertifiedStart]:
    """The signed starts a case's evidence shows, in its order."""
    return [PC._start(x, "a start") for x in json.loads(bytes.fromhex(case["evidence"])).get("starts", [])]


def with_starts(case: dict, starts: list | None) -> dict:
    """`case` with its evidence's `"starts"` replaced, or left out (`None`)."""
    e = {k: v for k, v in json.loads(bytes.fromhex(case["evidence"])).items() if k != "starts"}
    return case | {"evidence": D.canonical(e if starts is None else e | {"starts": starts}).encode().hex()}


def test_a_key_listed_in_two_sessions_is_held_to_its_signed_starts(server, monkeypatch):
    """The period reads each listed entry's session's signed start, so a key may be listed in two sessions. A
    period of one execution phase in the second of two stretches of one key, each signature in that stretch's session
    at its number, the second entry starting as many execution phases after the first as its session's signed base is
    after the first's on the key's life (`two-sessions`), is accepted by Lean with the mirror's public. The same list
    with the second session's signed base another, every signature still verifying, is refused by Lean and by the
    mirror and by name (`PC.NOT_SIGNED`); and a mirror that reads the starts' certificates and not their bases accepts
    what Lean refuses by that name. A period is one execution phase, so one spanning the restart, an execution phase in
    each stretch, its starts held, is refused by both by name (`restart`, `PC.CARRIED`). A key's first listed entry is
    held to no start: moving every base of a key by one amount changes no verdict."""
    for n in range(6):
        for m, want, why in (("rows-two-sessions", ACCEPT, None), ("rows-start-not-signed", REJECT, PC.NOT_SIGNED),
                             ("rows-restart", REJECT, PC.CARRIED)):
            case = build(m, n, small=n % 2 == 0)
            p = PC.params(case["params"])
            first, second = p.entries
            starts = shown_starts(case)
            assert first.key == second.key and first.session != second.session, (m, n)
            spans = {first, second} if m == "rows-restart" else {second}
            assert {PC._entry_at(p, w) for w, _ in by_window(case)} == spans, (m, n)
            for w, certs in by_window(case):
                e = PC._entry_at(p, w)
                assert all(c.window == w - e.start and c.verify(e.key, e.session) for c in certs), (m, n, w)
            assert [x.refusal(e.key, e.session, p.link, p.schedule.digest) for e, x in zip(p.entries, starts)] == \
                [None, None], (m, n)
            L = p.schedule.window * p.schedule.bucket_ms
            held = (second.start - first.start) * L == starts[1].base_ms - starts[0].base_ms
            assert held == (why != PC.NOT_SIGNED), (m, n)
            out = reference(case)
            verdict, got = D.replay(PORT, case, server)
            assert out["verdict"] == verdict == want, (m, n, out, got)
            if want == ACCEPT:
                assert json.loads(got)["public"] == out["public"], (m, n, got)
            else:
                assert why in out["reason"] and why in json.loads(got)["reason"], (m, n, out, got)
    monkeypatch.setattr(PC, "_held_to", lambda L, a, xa, e, x: True)
    for n in range(3):
        refused_by_lean_alone(build("rows-start-not-signed", n, small=True), server, PC.NOT_SIGNED)


def test_a_keys_later_session_is_the_one_after_its_entry_listed_before_it(server, monkeypatch):
    """A key listed in two sessions has the second's ordinal the first's next (`two-sessions`, accepted above): its device
    state gives each session it starts the next ordinal, so a session of the key between the two, which the list
    leaves out, leaves a gap. The same list with the second session's ordinal another, its start still certified and
    its base where the first puts it (`start-skipped`), is refused by Lean and by the mirror, by name (`PC.SKIPPED`);
    and a mirror that reads the ordinals' signatures and not their order accepts what Lean refuses by that name."""
    for n in range(6):
        case = build("rows-start-skipped", n, small=n % 2 == 0)
        p = PC.params(case["params"])
        first, second = p.entries
        starts = shown_starts(case)
        assert first.key == second.key and first.session != second.session, n
        assert [x.refusal(e.key, e.session, p.link, p.schedule.digest) for e, x in zip(p.entries, starts)] == \
            [None, None], n
        assert PC._held_to(p.schedule.window * p.schedule.bucket_ms, first, starts[0], second, starts[1]), n
        assert starts[1].ordinal != starts[0].ordinal + 1, n
        out = reference(case)
        verdict, got = D.replay(PORT, case, server)
        assert out["verdict"] == verdict == REJECT, (n, out, got)
        assert PC.SKIPPED in out["reason"] and PC.SKIPPED in json.loads(got)["reason"], (n, out, got)
    monkeypatch.setattr(PC, "_follows", lambda xb, x: True)
    for n in range(3):
        refused_by_lean_alone(build("rows-start-skipped", n, small=True), server, PC.SKIPPED)


def test_a_start_its_session_did_not_certify_is_refused_by_name(server, monkeypatch):
    """A period shows one signed start for each listed entry, each its session's certified start as the
    network attester's own verifier reads it. A shown start that is no egress certificate of window 0 under the link's
    start name, that names a key other than its entry's, that names another schedule's digest than the period's, whose
    base, ordinal and schedule do not open the digest it signs, or whose signature does not verify under its entry's key
    in its entry's session is refused by Lean and by the mirror, by name (`PC.UNCERTIFIED`) and by the
    same code. So is evidence that shows a start too few or too many, or none (`PC.STARTS_COUNT`): parameters that
    list a key again in a session whose start the evidence does not show read, and the period is refused there. A
    mirror that reads no start accepts what Lean refuses by `PC.UNCERTIFIED`."""
    codes = Counter()
    for n in range(48):
        case = build("rows-start-uncertified", n, small=n % 2 == 0)
        p = PC.params(case["params"])
        whys = [x.refusal(e.key, e.session, p.link, p.schedule.digest) for e, x in zip(p.entries, shown_starts(case))]
        (why,) = [w for w in whys if w is not None]
        codes[why] += 1
        out = reference(case)
        verdict, got = D.replay(PORT, case, server)
        named_by = f"{PC.UNCERTIFIED} ({why})"
        assert out["verdict"] == verdict == REJECT, (n, out, got)
        assert named_by in out["reason"] and named_by in json.loads(got)["reason"], (n, why, out, got)
    assert set(codes) == set(PC.UNCERTIFIED_BY), codes
    for m, why in (("rows-start-count", PC.STARTS_COUNT),):
        for n in range(6):
            case = build(m, n, small=n % 2 == 0)
            out = reference(case)
            verdict, got = D.replay(PORT, case, server)
            assert out["verdict"] == verdict == REJECT, (m, n, out, got)
            assert why in out["reason"] and why in json.loads(got)["reason"], (m, n, out, got)
    case = build("rows-honest", 0, small=True)
    e = case["params"]["keys"][-1]
    again = e | {"session": e["session"][:-1] + ("1" if e["session"][-1] == "0" else "0"), "start": e["start"] + 1}
    restarted = case | {"params": case["params"] | {"keys": [*case["params"]["keys"], again]}}
    assert len(PC.params(restarted["params"]).entries) == len(shown_starts(restarted)) + 1
    for bad in (restarted, with_starts(case, None)):
        out = reference(bad)
        verdict, got = D.replay(PORT, bad, server)
        assert out["verdict"] == verdict == REJECT, (out, got)
        assert PC.STARTS_COUNT in out["reason"] and PC.STARTS_COUNT in json.loads(got)["reason"], (out, got)
    monkeypatch.setattr(PC, "_start_check", lambda p, starts, i, e, x: None)
    for n in range(3):
        refused_by_lean_alone(build("rows-start-uncertified", n, small=True), server, PC.UNCERTIFIED)


def test_the_rows_form_accepts_a_wired_state_tied_period(server):
    """A rows-form period whose every execution phase's scan's `prog` is a compute audit's slot entries and whose state
    is that audit's `SCAN_STATE` passes every check: Lean accepts it with the mirror's public, and leaves one with an
    undecided scan or audit answer undecided, as the mirror does."""
    for m, want in (("rows-honest", ACCEPT), ("rows-answer-undecided", UNDECIDED), ("rows-audit-undecided", UNDECIDED)):
        for n in range(3):
            case = build(m, n, small=n % 2 == 0)
            out = reference(case)
            assert out["verdict"] == want, (m, n, out)
            verdict, got = D.replay(PORT, case, server)
            assert verdict == want and json.loads(got).get("public") == out.get("public"), (m, n, got)


def test_a_capped_period_passes_and_publishes_its_cap(server):
    """A rows-form period under a schedule with a response cap passes the period check, and its public names the cap
    (`max_tokens_cap`), Lean's as the mirror's; an uncapped period's public names none, so its bytes are the ones it had
    before the cap was a parameter."""
    for n in range(4):
        case = build("rows-capped", n, small=n % 2 == 0)
        out = reference(case)
        cap = case["params"]["schedule"]["max_tokens_cap"]
        assert out["verdict"] == ACCEPT and out["public"]["max_tokens_cap"] == cap, (n, out)
        verdict, got = D.replay(PORT, case, server)
        assert verdict == ACCEPT and json.loads(got)["public"] == out["public"], (n, got)
        verdict, got = D.replay(PORT, build("rows-honest", n, small=n % 2 == 0), server)
        assert verdict == ACCEPT and "max_tokens_cap" not in json.loads(got)["public"], (n, got)


def test_a_bad_response_cap_is_refused(server):
    """A schedule whose response cap is no positive u32 below `egress_cells`, or whose cell doesn't hold an id request's
    `max_tokens`, is refused at the parameters, by Lean as by the mirror."""
    case = build("rows-honest", 0, small=True)
    for params in PC._bad_caps(case["params"]["schedule"], case["params"]):
        bad = case | {"params": params}
        assert reference(bad)["verdict"] == REJECT, params["schedule"]
        verdict, out = D.replay(PORT, bad, server)
        assert verdict == REJECT and "schedule" in json.loads(out)["reason"], (params["schedule"], out)


def test_a_schedule_with_connection_rows_is_refused_by_name(server):
    """A schedule with lengths rows, of either form, with content rows or without, and with request words, is one
    the network attester builds (`NetworkCertifierParams`), and the period check reads it and refuses it by name at the
    parameters, Lean as the
    mirror: the check reads no connection rows' registration, so it accepts no period under one. The check refuses
    it too (`connectionRowsCheck`), which the mirror's check shows at an empty period under each schedule, with no
    entries and so no starts to show, accepted under the honest one: no parameters `params` reads reach it."""
    case = build("rows-honest", 0, small=True)
    empty = dataclasses.replace(PC.params(case["params"]), audits=(), windows=0, entries=())
    assert PC.check(empty, PC.Commitment([], [], b""), ([], [], []))[0] == ACCEPT
    for params in PC.connection_rows_params(case["params"]):
        schedule = PC._schedule(params["schedule"])
        assert PC.connection_rows(schedule) and PC.schedule_json(schedule) == params["schedule"]
        bad = case | {"params": params}
        out = reference(bad)
        assert out["verdict"] == REJECT and PC.CONNECTION_ROWS in out["reason"], (params["schedule"], out)
        verdict, got = D.replay(PORT, bad, server)
        assert verdict == REJECT and PC.CONNECTION_ROWS in json.loads(got)["reason"], (params["schedule"], got)
        held = dataclasses.replace(empty, schedule=schedule)
        assert PC.check(held, PC.Commitment([], [], b""), ([], [], [])) == (REJECT, PC.CONNECTION_ROWS), schedule


def test_bad_lengths_rows_and_content_rows_are_refused_as_the_schedules(server):
    """A schedule whose lengths rows, content rows or request words `NetworkCertifierParams` refuses (`PC.bad_lengths`) is
    refused at the parameters as no schedule, by Lean as by the mirror, and not by the name a schedule with
    connection rows is."""
    case = build("rows-honest", 0, small=True)
    for params in PC.bad_lengths(case["params"]):
        bad = case | {"params": params}
        for reason in (reference(bad)["reason"], json.loads(D.replay(PORT, bad, server)[1])["reason"]):
            assert "schedule" in reason and PC.CONNECTION_ROWS not in reason, (params["schedule"], reason)
        assert reference(bad)["verdict"] == REJECT == D.replay(PORT, bad, server)[0], params["schedule"]


def test_a_scan_whose_state_is_not_its_wired_audits_is_refused(server, monkeypatch):
    """A slot's Program reads the requests from its audit's registered `SCAN_STATE`, and the scan's units read and
    register their states at the scan's state's root, each past the first its transition's output from the one before,
    so the rows form refuses, in Lean and in the mirror and by name, each way an execution phase's state tie can fail
    (`PC.UNSTATES`): a `SCAN_STATE` that skips or alters one of the scan's transitions (the scan's state entry, or the
    audit's, at another root or domain), no state entry on either side or the scan's under another name, the state of
    an audit the scan's `prog` isn't wired to, a root that is no string, a second state entry at another root on
    either side, and the audit's held in its window's `activations.values` alone; and a mirror that skips the tie
    accepts what Lean refuses, on the audit's side (the scan's own state entry broken also leaves its row 0's opening
    under no root it states, which the mirror then refuses)."""
    unstate = PC.UNSTATES
    for u in unstate:
        monkeypatch.setattr(PC, "UNSTATES", (u,))
        for n in range(3):
            case = build("rows-state-untied", n, small=n % 2 == 0)
            assert PC.UNSTATED in reference(case)["reason"], (u, n, reference(case))
            verdict, out = D.replay(PORT, case, server)
            assert verdict == REJECT and PC.UNSTATED in json.loads(out)["reason"], (u, n, out)
    monkeypatch.setattr(PC, "UNSTATES", ("audit", "audit-root", "audit-extra", "audit-activations"))
    monkeypatch.setattr(PC, "_state_tied", lambda p, audits, reg: True)
    for n in range(2):
        refused_by_lean_alone(build("rows-state-untied", n, small=True), server, PC.UNSTATED)


def tied_cases(m: str, kind: str, monkeypatch) -> list[tuple[int, dict, dict]]:
    """The cases with the change `m` at the one list `kind` of its lists (`PC.TIED_HELD`, `PC.TIED_UNHELD`), over seeds
    that reach scans of one lane and of two: each seed, its case and its aimed execution phase's scan's schedule."""
    monkeypatch.setattr(PC, "TIED_HELD" if m == "rows-answer-tied-held" else "TIED_UNHELD", (kind,))
    out = []
    for n in range(8):
        case = build(m, n, small=n % 2 == 0)
        sched = case["params"]["schedule"]
        windows = json.loads(bytes.fromhex(case["evidence"]))["windows"]
        want = PC.tied_list(kind, sched["lanes"])
        assert any(PC.TIED_KEY in x["answer"]["public"] and x["answer"]["public"][PC.TIED_KEY] == want
                   for x in windows), (kind, n)
        out.append((n, case, sched))
    assert {sched["lanes"] for _, _, sched in out} == {1, 2}, (kind, [sched["lanes"] for _, _, sched in out])
    return out


def test_a_scan_whose_answer_names_a_tie_the_check_does_not_wire_is_refused(server, monkeypatch):
    """One-stage's registration check lets several proof blocks read a run Input only where its verifier names it tied,
    and its answer's public outputs then name it (`PC.TIED_KEY`): the acceptance rests on a later check that ties the
    Input's rows to the registrations that computed them. The period check holds one such tie a lane, its `prog<j>`'s
    (`wired`), so the rows form refuses, in Lean and in the mirror and by name, a list that names anything else
    (`PC.TIED_UNHELD`), at scans of one lane and of two: beside the lanes' names the name past the last lane's
    (`prog<lanes>`), another Input's, before or after them, or a lane's registration entry in place of its parameter;
    another Input's alone; a lane's name without its lane (`prog`, which at one lane is no `prog0`); and what is no
    list of names (a string, a nested list, a number, null). Each refusal names the entry that is no lane's and the
    lanes' names, Lean's as the mirror's and then the fix; and a mirror that skips the check accepts what Lean
    refuses."""
    kinds = PC.TIED_UNHELD
    for u in kinds:
        for n, case, sched in tied_cases("rows-answer-tied-unheld", u, monkeypatch):
            L, held = sched["lanes"], PC.prog_params(sched["lanes"])
            bad = {"next": f'names "prog{L}"', "state": 'names "state"', "entry": 'names "scan/prog0"',
                   "first": 'names "state"', "alone": 'names "state"', "bare": 'names "prog"',
                   "string": 'is "prog0", no list of names',
                   "nested": "names " + json.dumps(held, separators=(",", ":")),
                   "number": "names 0", "null": "is null, no list of names"}[u]
            mine = reference(case)["reason"]
            assert PC.TIE_UNHELD in mine and f"{PC.TIED_KEY} {bad}" in mine, (u, n, mine)
            assert mine.endswith(f"the scan's lanes' {', '.join(held)}"), (u, n, mine)
            verdict, out = D.replay(PORT, case, server)
            assert verdict == REJECT and json.loads(out)["reason"].startswith(mine + ". "), (u, n, out, mine)
    monkeypatch.setattr(PC, "TIED_UNHELD", kinds)
    monkeypatch.setattr(PC, "tied_input_wired", lambda s, a: True)
    for n in range(2):
        refused_by_lean_alone(build("rows-answer-tied-unheld", n, small=True), server, PC.TIE_UNHELD)


def test_a_scan_whose_answer_names_only_its_lanes_tied_is_accepted(server, monkeypatch):
    """The rows form accepts a scan's answer that names only lanes' `prog<j>` tied, whatever the list's shape
    (`PC.TIED_HELD`), at scans of one lane and of two: every lane's in the lanes' order, a proper subset of them, the
    lanes' in another order, one twice, and an empty list, which rests on nothing. Every name there is one the check
    wires (`wired` holds each lane's), so Lean accepts each with the mirror's public; at one lane `["prog0"]` is the
    whole list."""
    for u in PC.TIED_HELD:
        for n, case, sched in tied_cases("rows-answer-tied-held", u, monkeypatch):
            out = reference(case)
            assert out["verdict"] == ACCEPT, (u, n, out)
            verdict, got = D.replay(PORT, case, server)
            assert verdict == ACCEPT and json.loads(got)["public"] == out["public"], (u, n, got)
    assert PC.tied_list("all", 1) == ["prog0"] and PC.tied_list("subset", 2) == ["prog0"]
    assert PC.tied_list("order", 2) == ["prog1", "prog0"] and PC.tied_list("twice", 2) == ["prog0", "prog1", "prog0"]


def test_an_honest_scans_answer_names_its_lanes_tied_or_nothing(server):
    """Over honest rows periods, each execution phase's scan's answer names its lanes' `prog<j>` tied, every lane's in
    the lanes' order and nothing else, exactly where several of its units read one slot entry (the schedule's `every`
    ≥ 2), as one-stage's registration check publishes them, and has no such key where each unit reads its own
    (`every` = 1): Lean accepts each with the mirror's public, some of either kind and of one lane and of two."""
    seen = Counter()
    for n in range(28):
        case = build("rows-honest", n, small=n % 2 == 0)
        sched = case["params"]["schedule"]
        shared = sched["every"] > 1
        windows = json.loads(bytes.fromhex(case["evidence"]))["windows"]
        want = list(PC.prog_params(sched["lanes"])) if shared else None
        assert all(x["answer"]["public"].get(PC.TIED_KEY) == want for x in windows), n
        out = reference(case)
        assert out["verdict"] == ACCEPT, (n, out)
        verdict, got = D.replay(PORT, case, server)
        assert verdict == ACCEPT and json.loads(got)["public"] == out["public"], (n, got)
        seen[shared, sched["lanes"]] += 1
    assert all(seen[shared, lanes] for shared in (True, False) for lanes in (1, 2)), seen


def test_a_tied_name_is_a_lanes_or_refused():
    """The mirror's read of an answer's tied Inputs (`PC.tied_input_wired`, `PC.tied_unheld`), at one lane and at
    two: no public outputs, none under the key, and each list of lanes' names pass (`PC.TIED_HELD`); each list with
    another name, and each value that is no list of names, is refused (`PC.TIED_UNHELD`), and the refusal names its
    first entry that is no lane's. At one lane `["prog0"]` passes and `["prog"]` is refused."""
    for lanes in (1, 2):
        s = types.SimpleNamespace(lanes=lanes)
        assert all(PC.tied_input_wired(s, {"public": pub}) for pub in (None, {}, {"bound": 0}, [], "x"))
        for kind in PC.TIED_HELD:
            assert PC.tied_input_wired(s, {"public": {PC.TIED_KEY: PC.tied_list(kind, lanes)}}), (lanes, kind)
        for kind in PC.TIED_UNHELD:
            a = {"public": {PC.TIED_KEY: PC.tied_list(kind, lanes)}}
            assert not PC.tied_input_wired(s, a), (lanes, kind)
            assert PC.tied_unheld(s, a).endswith(f"the scan's lanes' {', '.join(PC.prog_params(lanes))}"), (lanes, kind)
    one = types.SimpleNamespace(lanes=1)
    assert PC.tied_input_wired(one, {"public": {PC.TIED_KEY: ["prog0"]}})
    assert not PC.tied_input_wired(one, {"public": {PC.TIED_KEY: ["prog"]}})
    assert PC.tied_unheld(one, {"public": {PC.TIED_KEY: ["prog"]}}) == 'names "prog", none of the scan\'s lanes\' prog0'
    assert PC.tied_unheld(one, {"public": {PC.TIED_KEY: ["prog0", "prog1"]}}) == (
        'names "prog1", none of the scan\'s lanes\' prog0')
    two = types.SimpleNamespace(lanes=2)
    assert PC.tied_unheld(two, {"public": {PC.TIED_KEY: "prog0"}}) == (
        'is "prog0", no list of names of the scan\'s lanes\' prog0, prog1')


def test_a_tied_audit_whose_state_is_in_its_windows_activations_alone_is_refused(server, monkeypatch):
    """A window circuit's registration commits the scan's state rows its slots read as a per-window value, its window's
    `activations.values` entry of `SCAN_STATE`, with no `roots` or `registered` entry of it. The check reads an audit's
    state entry in those two places alone (`stateEntries`), so a period that is honest but for that place, its tied
    audit's entry there under its own name and at the scan's state's root and domain, is refused at the state tie, in
    Lean and in the mirror and by name."""
    monkeypatch.setattr(PC, "UNSTATES", ("audit-activations",))
    for n in range(3):
        case = build("rows-state-untied", n, small=n % 2 == 0)
        c = PC.commitment(bytes.fromhex(case["commitment"]))
        held = [(i, a["window"]["activations"]["values"]) for i, a in enumerate(c.audits)
                if "activations" in a["window"]]
        assert len(held) == 1 and list(held[0][1]) == [PC.SCAN_STATE], (n, held)
        i, e = held[0][0], held[0][1][PC.SCAN_STATE]
        assert e["value"] == PC.SCAN_STATE and not PC._states(c.audits[i]), (n, e)
        assert any(i in PC._tied_audits(c.audits, reg) and PC._states(reg) and all(
            (s["root"], s["domain"]) == (e["root"], e["domain"]) for s in PC._states(reg)) for reg in c.windows), n
        out = reference(case)
        assert out["verdict"] == REJECT and PC.UNSTATED in out["reason"], (n, out)
        verdict, got = D.replay(PORT, case, server)
        assert verdict == REJECT and PC.UNSTATED in json.loads(got)["reason"], (n, got)


def test_a_tied_audit_that_runs_another_circuit_is_refused(server, monkeypatch):
    """The rows form holds each execution phase's tied audits to the class circuit, the parameters' `circuit`: a period
    whose every tied audit is registered under it passes, and Lean accepts it with the mirror's public; each way the
    hold can fail (`PC.UNHOLDS`: the aimed phase's tied audit registered under another Program, or the parameters'
    `circuit` another digest) Lean and the mirror refuse by name; and a mirror that skips the hold accepts what Lean
    refuses."""
    for n in range(3):
        case = build("rows-honest", n, small=n % 2 == 0)
        c = PC.commitment(bytes.fromhex(case["commitment"]))
        tied = [i for reg in c.windows for i in PC._tied_audits(c.audits, reg)]
        assert tied and all(c.audits[i]["program"] == case["params"]["circuit"] for i in tied), (n, c.audits)
        out = reference(case)
        assert out["verdict"] == ACCEPT, (n, out)
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
    """The start unit registers its `o0` at state leaf 0, and bucket t, unit t + 1, reads `sin` at leaf t and
    registers its `o1` at leaf t + 1, as the scan's registration states (`PC.state_reads`), so the rows form refuses,
    in Lean and in the mirror and by name, each way those reads can leave the chain while the state stays tied by
    root and domain (`PC.MISREADS`): a `sin` at leaf t + 1, an `o1` at leaf t or t + 2, the start unit's `o0` at leaf
    1, a run over other units, the buckets' two runs swapped, the reads or a run missing or renamed, a field no
    natural, and the reads of a scan with no start unit; and a mirror that skips the reads accepts what Lean
    refuses."""
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


def test_a_scan_whose_start_is_not_the_empty_state_is_refused(server, monkeypatch):
    """The start unit's input, which no unit computes, is a registered row the run's first execution phase must open to
    the empty state, so the rows form refuses, in Lean and in the mirror and by name, each way the opening can fail
    (`PC.UNPINS`): a start whose row the prover chose, opened as committed, no opening, another salt, another word
    count, no start entry, one under another name, one named by port alone, a second at another root, and a field of
    another type; and a mirror that skips the opening accepts what Lean refuses."""
    unpins = PC.UNPINS
    for u in unpins:
        monkeypatch.setattr(PC, "UNPINS", (u,))
        for n in range(3):
            case = build("rows-start-unpinned", n, small=n % 2 == 0)
            assert PC.UNPINNED in reference(case)["reason"], (u, n, reference(case))
            verdict, out = D.replay(PORT, case, server)
            assert verdict == REJECT and PC.UNPINNED in json.loads(out)["reason"], (u, n, out)
    monkeypatch.setattr(PC, "UNPINS", unpins)
    monkeypatch.setattr(PC, "_start_empty", lambda reg: True)
    for n in range(2):
        refused_by_lean_alone(build("rows-start-unpinned", n, small=True), server, PC.UNPINNED)


def test_a_period_of_more_than_one_phase_is_refused(server, monkeypatch):
    """An execution phase after its period's first starts from the state the one before it ended in, which the check
    neither holds nor charges, so the rows form refuses, in Lean and in the mirror and by name, a period of the run's
    first execution phase and the next and one of a later one and the next (`PC.CARRIES`), each stating a start opened
    to the empty state; and a mirror that takes any execution phase's start accepts what Lean refuses."""
    carries = PC.CARRIES
    for u in carries:
        monkeypatch.setattr(PC, "CARRIES", (u,))
        for n in range(3):
            case = build("rows-start-carried", n, small=n % 2 == 0)
            assert case["params"]["windows"] > 1 and (case["params"]["first"] > 0) == (u == "later"), (u, n)
            assert PC.CARRIED in reference(case)["reason"], (u, n, reference(case))
            verdict, out = D.replay(PORT, case, server)
            assert verdict == REJECT and PC.CARRIED in json.loads(out)["reason"], (u, n, out)
    monkeypatch.setattr(PC, "CARRIES", carries)
    monkeypatch.setattr(PC, "_start_refusal", lambda w, i, reg: None)
    for n in range(2):
        refused_by_lean_alone(build("rows-start-carried", n, small=True), server, PC.CARRIED)


def test_a_period_that_does_not_begin_the_run_is_charged_its_start(server, monkeypatch):
    """A period of one execution phase that is not the run's first starts from the state the one before it ended in,
    which the check holds to nothing: Lean accepts it at any start it states (`PC.CHARGES`) with the mirror's public,
    whose `start` is the scan's state's full width, the start unit's `sin`, counted once in `K`; the same period at the
    run's first execution phase, whose start is the empty state, is charged none."""
    for u in PC.CHARGES:
        monkeypatch.setattr(PC, "CHARGES", (u,))
        for n in range(2):
            case = build("rows-start-charged", n, small=n % 2 == 0)
            out = reference(case)
            p = PC.params(case["params"])
            assert p.first > 0 and p.windows == 1 and out["verdict"] == ACCEPT, (u, n, out)
            pub = out["public"]
            assert pub["start"] == p.scan.start > 0 and pub["K"] == pub["start"] + pub["schedules"] + pub["cells"] + sum(
                t["free"] + t["faulty_sets"] + t["faulty_bits"] + t["jitter"] + t["faulty_entries"]
                for t in pub["scans"]), (u, n, pub)
            verdict, got = D.replay(PORT, case, server)
            assert verdict == ACCEPT and json.loads(got)["public"] == pub, (u, n, got)
    for n in range(2):
        out = reference(build("rows-honest", n, small=n % 2 == 0))
        assert out["verdict"] == ACCEPT and out["public"]["start"] == 0, (n, out)


def test_a_scan_whose_ok_bits_are_not_the_schedules_constants_is_refused(server, monkeypatch):
    """A scan proof block's public word is its ok bit, and a member's ok rows are committed under public salts, so the
    rows form computes each member's root from the schedule's T and refuses, in Lean and in the mirror and by name,
    each way an execution phase's scan's registration can name another (`PC.UNOKS`): a member's `<member>/o0` at another
    root (a tree with a row that is not the ok row, or under other salts), domain or leaf count, with no entry or
    under another port, named a second time at another root, beside a port ending `/o0` that is no member's, every
    member at another schedule's constants, a root or domain that is no string, the entries by value under
    `registered`, and the members of a group of two execution phases' scan (`PC.PAIR`), which the rows form, one
    execution phase a registration, does not read; the start unit's `start/o0`, a state and no ok bit, is at any
    root; and a mirror that skips the comparison accepts what Lean refuses."""
    unok = PC.UNOKS
    for u in unok:
        monkeypatch.setattr(PC, "UNOKS", (u,))
        for n in range(3):
            case = build("rows-ok-unheld", n, small=n % 2 == 0)
            assert PC.UNOK in reference(case)["reason"], (u, n, reference(case))
            verdict, out = D.replay(PORT, case, server)
            assert verdict == REJECT and PC.UNOK in json.loads(out)["reason"], (u, n, out)
    monkeypatch.setattr(PC, "UNOKS", unok)
    monkeypatch.setattr(scan_ok, "held", lambda consts, reg: True)
    for n in range(2):
        refused_by_lean_alone(build("rows-ok-unheld", n, small=True), server, PC.UNOK)


def ok_entries(case: dict) -> list[dict[str, tuple]]:
    """Each execution phase's scan's registration's ok entries, by port: (domain, root, leaves)."""
    return [{e["port"]: (e.get("domain"), e.get("root"), e.get("leaves")) for e in reg.get("roots", [])
             if scan_ok.ok_port(e.get("port"))}
            for reg in json.loads(bytes.fromhex(case["commitment"]))["windows"]]


def test_two_members_at_one_root_are_each_held_to_it(server, monkeypatch):
    """At T = 2 the scan's `ingress-0` and `last` are one proof block of the ingress template each, so their ok bits
    are at one frame domain and one root under two ports (`rows-ok-equal`): Lean accepts the period with the mirror's
    public, each port held to its own constant. At that schedule each way of naming another (`PC.UNOKS`) is refused
    by both, by name, a registration that names one of the two ports and not the other among them: a root another
    member is at stands for no member but its own."""
    twins = ("ingress-0/o0", "last/o0")
    for n in range(4):
        case = build("rows-ok-equal", n, small=n % 2 == 0)
        assert case["params"]["schedule"]["window"] == PC.EQUAL_T
        for at in ok_entries(case):
            at.pop(scan_ok.START, None)
            assert sorted(at) == ["egress-0/o0", *twins]
            assert at[twins[0]] == at[twins[1]] and len(set(at.values())) == 2, at
        out = reference(case)
        assert out["verdict"] == ACCEPT, (n, out)
        verdict, got = D.replay(PORT, case, server)
        assert verdict == ACCEPT and json.loads(got)["public"] == out["public"], (n, got)
    real, unok, one_twin = PC._small_schedule, PC.UNOKS, 0
    monkeypatch.setattr(PC, "_small_schedule", lambda r, small, capped=False, T=None: real(r, small, capped, PC.EQUAL_T))
    for u in unok:
        monkeypatch.setattr(PC, "UNOKS", (u,))
        for n in range(8 if u == "missing" else 2):
            case = build("rows-ok-unheld", n, small=n % 2 == 0)
            assert case["params"]["schedule"]["window"] == PC.EQUAL_T
            one_twin += any(len(set(twins) & set(at)) == 1 and len(set(at) - {scan_ok.START}) == 2
                            for at in ok_entries(case))
            assert PC.UNOK in reference(case)["reason"], (u, n, reference(case))
            verdict, out = D.replay(PORT, case, server)
            assert verdict == REJECT and PC.UNOK in json.loads(out)["reason"], (u, n, out)
    assert one_twin, "no case named one of the two ports at one root and not the other"


def test_a_scan_whose_prog_is_no_audits_output_is_refused(server, monkeypatch):
    """The scan reads each lane's Programs' rows (`scan/prog<j>`) from a compute audit's output, so the rows form
    refuses, in Lean and in the mirror and by name, each way an execution phase's tie can fail at every lane
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


def test_a_scan_whose_prog_is_no_slot_entries_is_refused(server, monkeypatch):
    """The scan reads each slot's entry, read bit included, from the `SlotEntry_v1` unit of that slot's Program, lane
    j's from chunk j, so the rows form refuses, in Lean and in the mirror and by name, each way a tied execution phase's
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
    `PC.RENAMED_SELECT_ENTRY`) has none, so Lean and the mirror refuse its execution phase at the wiring, Lean by the
    pin; and a mirror that pins that digest too accepts it, the pin being all that refuses it."""
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


def test_a_tied_audits_free_inputs_are_in_its_phases_term(server):
    """The egress renders the outputs of the audit the scan's `prog` is tied to, so the bits the prover chose of that
    audit's inputs are the execution phase's: over honest rows periods, which Lean and the mirror accept with one
    public, each scan's `free` is the parameters' `free` of the audits whose registration holds its `prog`'s root (the
    scan's own is 0), some of them nonzero."""
    charged = 0
    for n in range(8):
        case = build("rows-honest", n, small=n % 2 == 0)
        c = json.loads(bytes.fromhex(case["commitment"]))
        out = reference(case)
        assert out["verdict"] == ACCEPT, (n, out)
        verdict, got = D.replay(PORT, case, server)
        assert verdict == ACCEPT and json.loads(got)["public"] == out["public"], (n, got)
        for reg, t in zip(c["windows"], out["public"]["scans"]):
            roots = {e["root"] for e in reg.get("roots", []) + reg.get("registered", [])
                     if PC.prog_port(e.get("port")) or PC.prog_port(e.get("value"))}
            want = sum(a["free"] for a, r in zip(case["params"]["audits"], c["audits"])
                       if any(o.get("root") in roots for o in r["roots"]))
            assert t["free"] == want, (n, t, want)
        charged += any(t["free"] for t in out["public"]["scans"])
    assert charged


def as_facts(entry: dict) -> Facts:
    """An entry of a public's `audits` or `charged` as the facts it writes (`_facts_json`), less their design."""
    return Facts(entry["units"], entry["drawn"], Fraction(entry["delta"]), entry["wrong_units"])


def sample_size(entry: dict) -> tuple[int, int]:
    """An entry of a public's `audits` or `charged`: its units and how many of them its sample takes."""
    f = as_facts(entry)
    return f.units, f.sampled


def charged(case: dict, public: dict) -> list[int]:
    """The audits of an accepted rows-form `case` its `public` charges: tied to an execution phase's scan and sampled in
    part."""
    c = json.loads(bytes.fromhex(case["commitment"]))
    audits, windows = c["audits"], c["windows"]
    return [i for i, f in enumerate(public["audits"]) if f["drawn"] < f["units"] and PC._tied_in(audits, windows, i)]


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


def test_a_tied_audit_sampled_in_part_is_charged_its_phases_entries(server):
    """The scan reads a tied audit only through its `prog`, E_w entry bits an execution phase, and its state is the
    scan's own, so an incorrect proof block of any of the audit's templates, an unsampled `SelectEntry_v1` or
    `ScanSlots_v2` one among them, changes only those bits: the rows form accepts a tied audit at any sample, Lean with
    the mirror's public, which reads that audit's sample at δ_i = 2^-64 (`charged`), where its at most `b` incorrect
    proof blocks are at least one, so its execution phases' terms charge every entry they can set, `faulty_entries` =
    E_w = lanes · T / every · the entry's bits, every other execution phase's none, and `eps` holds δ_i beside the
    scans' δ. `b` is the count the check reads there, never below the count it reads at the answer's δ nor the
    sample's own bound, so each way of sampling in part is charged, an answer that states a `bound` of 0 among them,
    the demo's sample at one proof block and at two, and the cases' audits select (`SelectEntry_v1` beside
    `ScanSlots_v2`) or don't (`SlotEntry_v1`)."""
    seen = Counter()
    for m in SAMPLED_IN_PART:
        for n in range(8):
            case = build(m, n, small=n % 2 == 0)
            p, out = PC.params(case["params"]), reference(case)
            assert out["verdict"] == ACCEPT, (m, n, out)
            verdict, got = D.replay(PORT, case, server)
            assert verdict == ACCEPT and json.loads(got)["public"] == out["public"], (m, n, got)
            pub, (i,) = out["public"], charged(case, out["public"])
            s, prog = p.schedule, PC._scan_widths(case["params"]["scan"])[1]
            assert p.scan.entries == s.lanes * (s.window // s.every) * prog, (m, n, p.scan)
            c, e = (json.loads(bytes.fromhex(case[k])) for k in ("commitment", "evidence"))
            (read,), own = (as_facts(f) for f in pub["charged"]), as_facts(pub["audits"][i])
            assert read.delta == PC.TIED_DELTA and read.wrong >= max(1, own.wrong), (m, n, read)
            assert sample_size(pub["charged"][0]) == sample_size(pub["audits"][i]), (m, n, read)
            assert [t["faulty_entries"] for t in pub["scans"]] == [
                p.scan.entries if i in PC._tied_audits(c["audits"], reg) else 0 for reg in c["windows"]], (m, n, pub)
            assert pub["eps"] == str(sum((Fraction(t["delta"]) for t in pub["scans"]), PC.TIED_DELTA))
            seen[m, "charged"] += 1
            seen["stated 0"] += e["audits"][i]["public"]["bound"] == 0
            if m == "rows-tied-sampled":
                seen[f"subset:{pub['audits'][i]['drawn']}"] += 1
            seen["select" if PC.SELECT_KEY in case["params"]["audits"][i]["program"]["definitions"] else "slot"] += 1
    want = [(m, "charged") for m in SAMPLED_IN_PART] + ["stated 0", "subset:1", "subset:2", "select", "slot"]
    assert all(seen[k] for k in want), seen


def test_a_tied_audits_sample_is_read_at_the_periods_own_delta(server):
    """One sample read at two δs, each fixed before the coins: a tied audit sampled `subset:1` of its n proof blocks,
    whose answer is at δ = 1/2 with its bound there, n // 2, beside scans at δ = 2^-64. The rows form reads the same
    sample at δ_i = 2^-64 (`charged`): n − 1 incorrect proof blocks, every unsampled one, since the sample misses
    any one of them with probability 1/n ≥ δ_i, so the terms of the scans it is tied to charge E_w; it keeps the
    answer's δ and count in `audits`, the answer's own reading; and its eps is the scans' δ and δ_i, each its own
    term, (m + 1) · 2^-64 exactly. The reading keeps a count above the sample's bound, and a stratified sample's is
    its strata's own, summed. Lean gives every such period the mirror's public."""
    one = Facts(16, 1, Fraction(1, 2), 8, {"subset": 1})
    assert (one.at(PC.TIED_DELTA).wrong, one.at(Fraction(1, 2)).wrong, one.at(PC.TIED_DELTA).delta) == (
        15, 8, PC.TIED_DELTA)
    assert Facts(16, 16, Fraction(1, 2), 0, {"subset": 16}).at(PC.TIED_DELTA).wrong == 0
    assert Facts(16, 1, Fraction(1, 2), 8).at(PC.TIED_DELTA).wrong == 16
    assert Facts(16, 16, Fraction(1, 2), 3, {"subset": 16}).at(PC.TIED_DELTA).wrong == 3
    assert Facts(9, 5, Fraction(1, 10), 0, strata_of(("a", 4, 4), ("b", 5, 1))).at(Fraction(1, 2)).wrong == 3
    read = 0
    for n in range(12):
        case = build("rows-tied-sampled", n, small=n % 2 == 0)
        (i,) = charged(case, reference(case)["public"])
        e = json.loads(bytes.fromhex(case["evidence"]))
        units = e["audits"][i]["profile"]["levels"][0]["size"]
        if PC._sample_of(json.loads(bytes.fromhex(case["commitment"]))["audits"][i])[1] != {"subset": 1}:
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
        assert as_facts(pub["audits"][i]) == Facts(units, 1, Fraction(1, 2), units // 2), (n, pub)
        assert [as_facts(f) for f in pub["charged"]] == [Facts(units, 1, PC.TIED_DELTA, units - 1)], (n, pub)
        p = PC.params(case["params"])
        assert [t["faulty_entries"] for t in pub["scans"]] == [
            p.scan.entries if i in PC._tied_audits(c["audits"], reg) else 0 for reg in c["windows"]], (n, pub)
        assert pub["eps"] == str(Fraction(len(pub["scans"]) + 1, 2**64)), (n, pub)
        read += 1
    assert read >= 2, read


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
                              audits=[PC.Audit(0, 0, "", 4, (("SlotEntry_v1{KO=1}", 4, PC.SLOT_ENTRY_PINS[0]),))])

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
    compute audit's output), an ingress and an egress instance's committed bits (their returns: a u16 with the state
    and view, or with the row), its slot entries' bits (lanes · T / every entries of its `prog` ports' width) and its
    state's width, the start unit's `sin`, which a period that does not begin the run is charged; and
    Lean and the mirror refuse, as parameters, a scan that isn't a descriptor, another Program's, one in the checker
    form (its egress row an input), one whose templates have other parameters or other than 1, T and T instances,
    one with no start unit's template, or rows other than the schedule's, one whose lane reads another row of its input than its bucket's or another
    lane's input, one with an input gate no template's, and each negative of a check of the rows the scan's ties
    equate, of what `NetTiming.BoundaryGates.Fits` asks or of the input nodes its arguments read."""
    for n in range(3):
        case = build("rows-honest", n, small=n % 2 == 0)
        p, desc = PC.params(case["params"]), case["params"]["scan"]
        state, prog, view, _words = PC._scan_widths(desc)
        s, T, row = p.schedule, p.schedule.window, WR.row_bytes(p.schedule)
        assert p.scan == PC.Scan(T, 0, 16 + state + view, 16 + 8 * row, s.lanes * (T // s.every) * prog, state), \
            (n, p.scan)
        for bad in PC.bad_scans(case["params"]):
            broken = case | {"params": bad}
            assert reference(broken)["reason"].startswith("params: scan:"), (n, reference(broken))
            verdict, out = D.replay(PORT, broken, server)
            assert verdict == REJECT and "scan" in json.loads(out)["reason"], (n, out)


def test_each_scan_check_refuses_its_negative(server):
    """Each of the scan's checks of the rows its ties equate, of what `NetTiming.BoundaryGates.Fits` asks and of the
    input nodes its arguments read refuses, in Lean and in the mirror, a scan that passes every check before it
    (`PC.scan_negatives`): one whose ingress view is not its egress `state`'s leaves, one whose `sin` is not the
    returned state's, one whose start unit takes or returns a state of other leaves, one past the resolution's walk,
    one whose root doesn't return an egress leaf, one whose egress template passes a parameter's leaves through, one
    whose ingress row is a constant gate's, one with an egress argument an ingress output, one whose egress state is
    the ingress rows' input, one whose `sin` reads its row's input node, one whose start unit reads the buckets'
    `sin`'s input node, one whose egress state is lane 0's slot entries' input node, one with an input node no argument reads, and one whose ingress template names a later
    root node."""
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


#: The refusals that hold a compute audit to its Program, each with what Lean's reason and the mirror's say.
AUDIT_REFUSALS = {"rows-audit-program": ("registration: not its Program",) * 2,
                  "rows-audit-units": ("not its Program's",) * 2}


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
    case = build("rows-honest", 0, small=True)
    for bad in PC.bad_audits(case["params"]):
        broken = case | {"params": bad}
        assert reference(broken)["reason"].startswith("params:"), reference(broken)
        verdict, out = D.replay(PORT, broken, server)
        assert verdict == REJECT and json.loads(out)["reason"].startswith("params:"), (bad, out)


#: The changes whose answer's profile samples more than its level holds: a compute audit's, the tied audit's of the
#: aimed scan, and a scan's.
OVERSAMPLED = ("rows-audit-oversampled", "rows-tied-oversampled", "rows-answer-oversampled")
#: The change whose answer's profile samples more of a stratum than the stratum holds, in a level whose total holds: the
#: aimed scan's tied audit's.
STRATUM_OVERSAMPLED = "rows-tied-stratum-oversampled"
#: The change whose answer's profile has strata whose sizes do not sum to its level's, no stratum sampling more than it
#: holds: the aimed scan's tied audit's.
STRATA_SIZES = "rows-tied-strata-sizes"


def sampled(level: dict) -> int:
    """How many of its units a profile's level samples: its `subset`'s k, or its strata's summed."""
    design = level["draw"]
    return design["subset"] if "subset" in design else sum(s["k"] for s in design["strata"])


def unheld(level: dict) -> str | None:
    """Which of `factsOf`'s three refusals a profile's level meets, the first in its order: the level samples more than
    it holds, a stratum more than the stratum holds, or the strata's sizes do not sum to the level's; `None`: none."""
    strata = level["draw"].get("strata")
    if sampled(level) > level["size"]:
        return "level"
    if any(s["k"] > s["size"] for s in strata or ()):
        return "stratum"
    return "sizes" if strata is not None and sum(s["size"] for s in strata) != level["size"] else None


def strata_of(*ss: tuple[str, int, int]) -> dict:
    """A stratified sampling design as a profile's level writes it, each stratum its name, its size and its count."""
    return {"law": "stratified", "strata": [{"name": name, "size": size, "k": k} for name, size, k in ss]}


def level_of(x: dict) -> tuple[int, dict]:
    """The one level of the answer `x`'s profile: its size and its sampling design."""
    (level,) = x["profile"]["levels"]
    return level["size"], level["draw"]


def stated(size: int, design: dict) -> dict:
    """A registration's `population` and sampling design that state `design` of `size` units, a stratum's units the
    range after the last stratum's."""
    if "subset" in design:
        return {"population": size, "law": {"k": design["subset"], "law": "subset"}}
    lo, strata = 0, []
    for s in design["strata"]:
        strata.append({"k": s["k"], "template": s["name"], "units": [[lo, lo + s["size"]]]})
        lo += s["size"]
    return {"population": size, "law": {"k": sum(s["k"] for s in strata), "law": "stratified", "strata": strata}}


def registered(case: dict, c: dict, e: dict, reg: dict, x: dict) -> dict:
    """`case` with the commitment `c` and the evidence `e` it decodes to, after the registration `reg` of `c` is made
    to state the sample the profile of its answer `x` of `e` names: its level's size as the `population` and the
    level's sampling design as its own, a stratum's units the range after the last stratum's. The answer is for that
    registration."""
    size, design = level_of(x)
    reg |= stated(size, design)
    x["commitment"] = R.digest(reg)
    return case | {"commitment": D.canonical(c).encode().hex(), "evidence": D.canonical(e).encode().hex()}


def test_a_profile_that_is_no_sample_of_its_level_is_refused(server):
    """A profile whose level samples more than it holds is no sample of the level. Neither is a stratified one with a
    stratum that samples more than the stratum holds though the level's total holds, or whose strata's sizes do not sum
    to the level's though no stratum samples more than it holds. The charge of a tied audit sampled in part would miss
    all three (the last where the strata hold more than the level), the last two as an audit sampled whole with a
    stratum in part. Lean and the mirror each refuse the one answer that carries it, where they read its sample: by the
    level a compute audit's, tied to a scan or not, and a scan's, sampled `subset:k` or stratified; by
    the stratum and by the strata's sizes, above the level's and below, the aimed scan's tied audit's. They refuse it
    before they read the registration's sample or the public's bound (with the bound gone too the reason is still the
    sample's), and for that alone: with the level sampled whole instead, in its registration as in its profile and
    in one stratum where the strata's sizes were not the level's, Lean accepts the period with the mirror's public."""
    seen = Counter()
    for m, why in [*((m, PC.OVERSAMPLED) for m in OVERSAMPLED), (STRATUM_OVERSAMPLED, PC.STRATUM_OVERSAMPLED),
                   (STRATA_SIZES, PC.STRATA_SIZES)]:
        # seed 5 of `rows-audit-oversampled` aims an audit no scan ties
        for n in (*range(4), *((5,) if m == "rows-audit-oversampled" else ())):
            case = build(m, n, small=n % 2 == 0)
            c, e = (json.loads(bytes.fromhex(case[k])) for k in ("commitment", "evidence"))
            answers = [("audit", x) for x in e["audits"]] + [("scan", w["answer"]) for w in e["windows"]]
            regs = c["audits"] + c["windows"]
            (i, kind, x), = [(i, kind, x) for i, (kind, x) in enumerate(answers)
                             if unheld(x["profile"]["levels"][0])]
            level, bound = x["profile"]["levels"][0], x["public"]
            design, how = level["draw"], unheld(level)
            tied = kind == "audit" and PC._tied_in(c["audits"], c["windows"], i)
            assert tied or m not in ("rows-tied-oversampled", STRATUM_OVERSAMPLED, STRATA_SIZES), (m, n)
            assert how == {PC.OVERSAMPLED: "level", PC.STRATUM_OVERSAMPLED: "stratum", PC.STRATA_SIZES: "sizes"}[why]
            seen.update([kind, "subset" if "subset" in design else "stratified", how,
                         *(["tied" if tied else "untied"] if kind == "audit" else []),
                         *(["above" if sum(s["size"] for s in design["strata"]) > level["size"] else "below"]
                           if how == "sizes" else [])])
            x["public"] = {}
            for broken in (case, case | {"evidence": D.canonical(e).encode().hex()}):
                assert why in reference(broken)["reason"], (m, n, reference(broken))
                verdict, out = D.replay(PORT, broken, server)
                assert verdict == REJECT and why in json.loads(out)["reason"], (m, n, out)
            x["public"] = bound
            level["draw"] = {"subset": level["size"]} if "subset" in design else design | {"strata": (
                [design["strata"][0] | {"k": level["size"], "size": level["size"]}] if how == "sizes"
                else [s | {"k": s["size"]} for s in design["strata"]])}
            whole = registered(case, c, e, regs[i], x)
            assert reference(whole)["verdict"] == ACCEPT, (m, n, reference(whole))
            verdict, out = D.replay(PORT, whole, server)
            assert verdict == ACCEPT and json.loads(out)["public"] == reference(whole)["public"], (m, n, out)
    assert all(seen[k] for k in ("audit", "scan", "subset", "stratified", "tied", "untied", "level", "stratum", "sizes",
                                 "above", "below")), seen


#: The changes whose one answer's registration states no sample, or another than the answer's profile names, each
#: with the refusal's name.
UNREGISTERED = {"rows-audit-unsampled": PC.NO_SAMPLE, "rows-answer-unsampled": PC.NO_SAMPLE,
                "rows-audit-unregistered": PC.NOT_REGISTERED, "rows-answer-unregistered": PC.NOT_REGISTERED}


def bounds(reg: dict, x: dict) -> tuple[int, dict, int, int]:
    """Of an accepted answer `x` to the registration `reg`: its level's size and sampling design, the bound its public
    states and the bound of its registration's sample at its δ (`PC._sample_bound`)."""
    need = PC._sample_bound(*PC._sample_of(reg), Fraction(x["profile"]["delta"]))
    return *level_of(x), x["public"]["bound"], need


def test_an_answer_whose_sample_is_not_its_registrations_is_refused(server):
    """A sample's strata, each stratum's units and how many of them are sampled are the registration's, fixed before
    any coin; only which units the sample takes comes with the answer. So a compute audit's answer, and a scan's, is
    refused by name, by Lean and by the mirror, where its registration states no `subset:k` or stratified
    sample (no sampling design or population, a Bernoulli design, or a count, strata, a template or a range
    one-stage's profile reads none of) and where its profile is another sample of its level than the registered one
    (another `k`, a stratum renamed, sampled otherwise or holding another unit, the strata as one subset or the subset
    as one stratum). It is refused for that alone: with the registration made to state the profile's sample, Lean
    accepts the period with the mirror's public."""
    seen = Counter()
    for m, why in UNREGISTERED.items():
        for n in range(6):
            case = build(m, n, small=n % 2 == 0)
            assert why in reference(case)["reason"], (m, n, reference(case))
            verdict, out = D.replay(PORT, case, server)
            assert verdict == REJECT and why in json.loads(out)["reason"], (m, n, out)
            c, e = (json.loads(bytes.fromhex(case[k])) for k in ("commitment", "evidence"))
            answers = [("audit", reg, x) for reg, x in zip(c["audits"], e["audits"])] + [
                ("scan", reg, w["answer"]) for reg, w in zip(c["windows"], e["windows"])]
            (kind, reg, x), = [(kind, reg, x) for kind, reg, x in answers if PC._sample_of(reg) != level_of(x)]
            _, design = level_of(x)
            seen.update([kind, why, "subset" if "subset" in design else "stratified"])
            if all(s["size"] for s in design.get("strata", [])):  # a stratum of no unit has no range to register
                fixed = registered(case, c, e, reg, x)
                assert reference(fixed)["verdict"] == ACCEPT, (m, n, reference(fixed))
                verdict, out = D.replay(PORT, fixed, server)
                assert verdict == ACCEPT and json.loads(out)["public"] == reference(fixed)["public"], (m, n, out)
                seen[m, "accepted"] += 1
    assert all(seen[k] for k in ("audit", "scan", "subset", "stratified", *UNREGISTERED.values(),
                                 *((m, "accepted") for m in UNREGISTERED))), seen


def test_an_answers_count_is_never_below_its_samples_bound(server):
    """The count of incorrect proof blocks the period reads of an answer is the larger of the `bound` the answer
    states and the bound of its registration's sample at its δ, which the check computes itself, so an answer that
    states too small a bound gains nothing and none is refused for it: over accepted periods, every compute
    audit's and every scan's count in the public is that maximum, an answer below its sample's bound is read at
    the sample's and one at or above it as it states, over `subset:k` samples and stratified ones, and Lean's public
    is the mirror's."""
    seen = Counter()
    for m in ("rows-honest", *SAMPLED_IN_PART):
        for n in range(4):
            case = build(m, n, small=n % 2 == 0)
            out = reference(case)
            assert out["verdict"] == ACCEPT, (m, n, out)
            verdict, got = D.replay(PORT, case, server)
            assert verdict == ACCEPT and json.loads(got)["public"] == out["public"], (m, n, got)
            c, e = (json.loads(bytes.fromhex(case[k])) for k in ("commitment", "evidence"))
            read = [("audit", reg, x, t) for reg, x, t in zip(c["audits"], e["audits"], out["public"]["audits"])] + [
                ("scan", reg, w["answer"], t)
                for reg, w, t in zip(c["windows"], e["windows"], out["public"].get("scans", []))]
            for kind, reg, x, t in read:
                size, design, bound, need = bounds(reg, x)
                got = as_facts(t)
                assert (got.units, got.wrong) == (size, max(bound, need)), (m, n, kind, t, bound, need)
                how = "raised" if bound < need else "as stated"
                seen.update([(kind, how), ("subset" if "subset" in design else "stratified", how)])
    assert all(seen[k, how] for k in ("audit", "scan", "subset", "stratified") for how in ("raised", "as stated")), seen


def test_a_samples_bound_is_its_stratas_own_summed():
    """`_sample_bound` and `_facts` on Lean's `#guard`s of `sampleBound` and `factsOf`. 3 of 10 at δ = 1/10 leave at
    most 4 incorrect, so a `bound` of 3 is read as 4 and one of 5 as 5. Strata of 9 of 10 and 1 of 10 at δ = 1/100 leave
    at most 1 and 9, each at 1/200, where their totals read as one sample, 10 of 20, would leave 5. Two strata of 4 of
    8 at δ = 1/10 leave at most 3 each at 1/20, where one-stage's bound of the strata together is 3. A level sampled
    whole leaves none."""
    def count(size, design, delta, states):
        prof = {"delta": delta, "levels": [{"draw": design, "size": size}]}
        return PC._facts("x", stated(size, design), prof, {"bound": states}).wrong

    tenth, bound = Fraction(1, 10), PC._sample_bound
    assert [count(10, {"subset": 3}, tenth, b) for b in (3, 4, 5)] == [4, 4, 5]
    uneven, halves = strata_of(("a", 10, 9), ("b", 10, 1)), strata_of(("a", 8, 4), ("b", 8, 4))
    assert [count(20, uneven, Fraction(1, 100), b) for b in (9, 12)] == [10, 12]
    assert [bound(10, {"subset": k}, Fraction(1, 200)) for k in (9, 1)] == [1, 9]
    assert bound(20, {"subset": 10}, Fraction(1, 100)) == 5
    assert [count(16, halves, tenth, b) for b in (0, 7)] == [6, 7]
    assert bound(8, {"subset": 4}, Fraction(1, 20)) == 3
    assert Stratified((Stratum("a", 8, 4), Stratum("b", 8, 4))).count_bound(tenth) == 3
    assert [count(10, {"subset": 10}, tenth, b) for b in (0, 2)] == [0, 2]


def test_the_rows_forms_cells_are_the_schedules(server):
    """The rows form's `cells` are the schedule's own (`Schedule.rowsCells`, `NetworkCertifierParams.rows_cells`): `egress_cells`
    for each of the `lanes · m · window` connections the network attester can carry, so an accepted period's schedules
    and cells terms are each that many `⌈log2 (egress_cells + 1)⌉`, never 0 for cells that miss its connections; no
    cells, one connection short or over, and a cell count below `egress_cells` are refused as parameters by Lean and the
    mirror alike; and a schedule of 4 lanes over a link's 32-bucket execution phase, 5 egress cells a connection,
    charges 384 bits for each over one such phase (Lean's `#guard`)."""
    for n in range(4):
        case = build("rows-honest", n, small=n % 2 == 0)
        p, out = PC.params(case["params"]), reference(case)
        s = p.schedule
        assert list(p.cells) == s.rows_cells(p.windows) == [s.egress_cells] * (s.lanes * p.windows * s.window)
        term = s.lanes * p.windows * s.window * clog2(s.egress_cells + 1)
        assert out["verdict"] == ACCEPT and term > 0, (n, out)
        assert (out["public"]["schedules"], out["public"]["cells"]) == (term, term), (n, out["public"])
        verdict, got = D.replay(PORT, case, server)
        assert verdict == ACCEPT and json.loads(got)["public"] == out["public"], (n, got)
        for bad in PC.bad_cells(case["params"]):
            broken = case | {"params": bad}
            assert reference(broken)["reason"].startswith("params: cells:"), (n, reference(broken))
            verdict, got = D.replay(PORT, broken, server)
            assert verdict == REJECT and json.loads(got)["reason"].startswith("params: cells:"), (n, bad, got)
    four = NetworkCertifierParams(cell_bytes=512, slots=4, bucket_ms=100, delay=27, lanes=4, queue=4, window=32,
                        ingress_cells=47, ingress_buckets=12, egress_cells=5, every=16)
    assert len(four.rows_cells(1)) == 128 and sum(clog2(c + 1) for c in four.rows_cells(1)) == 384


def test_a_period_whose_frames_overrun_a_u32_is_refused(server):
    """The rows form refuses a period whose `lanes · windows · window` connections, the k-th at ordinal k + 1, aren't
    distinct u32s, or whose cell's B bytes aren't under 2^32 (`rowsWf`): its frames could not name every connection or
    every payload's length, so its record would not read back (`NetTiming.RowsFormat.Wf`). As parameters it is refused
    before its cells, which a period that long could never list, by Lean and the mirror alike, and the honest period's
    schedule passes `rowsWf` up to the last whole period under 2^32. The check refuses it too (`rowsWfCheck`), which the
    mirror's check shows at an empty period whose B is 2^32, with no entries and so no starts to show: no parameters
    `params` reads reach it."""
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
        empty = dataclasses.replace(p, audits=(), windows=0, entries=())
        for b, refused in ((PC.U32 - 1, False), (PC.U32, True)):
            q = dataclasses.replace(empty, schedule=dataclasses.replace(p.schedule, cell_bytes=b))
            got = PC.check(q, PC.Commitment([], [], b""), ([], [], []))
            assert (got[0] == REJECT and got[1].startswith(fragment)) == refused, (n, b, got)


def test_the_check_refuses_entries_out_of_order_and_two_keys_with_one_identifier():
    """The check holds the entries' order and their keys' identifiers itself, before the starts, of any parameters:
    an entry that starts before the one listed before it, or at its start, is refused (`entriesOrderRefusal`), and so
    are two keys with one identifier (`entriesKeyRefusal`); one key listed in two sessions is neither. The decoder
    refuses both lists first (`out-of-order`, `overlap`, `one-id`), so no parameters `params` reads reach these: the
    cases are the Lean `#guard`'s beside `startChecks`, with its keys and its words."""
    p = PC.params(build("rows-honest", 0, small=True)["params"])
    lp = p.entries[0].key.params

    def found(*entries: tuple[int, int, int, int]) -> tuple:
        es = tuple(PC.Entry(PC.LmsKey(lp, bytes([b]) * 16 + bytes([c]) * lp.N), bytes([s]) * 32, a)
                   for b, c, s, a in entries)
        return PC.check(dataclasses.replace(p, audits=(), windows=0, entries=es), PC.Commitment([], [], b""),
                        ([], [], []))

    order = "list the entries by their stretches' starts, no two at one"
    assert found((1, 1, 7, 3), (2, 2, 8, 2)) == (
        REJECT, f"{PC.ENTRIES_OUT_OF_ORDER}: entry 1 starts at 2, not after entry 0's 3: {order}")
    assert found((1, 1, 7, 2), (2, 2, 8, 2)) == (
        REJECT, f"{PC.ENTRIES_OUT_OF_ORDER}: entry 1 starts at 2, not after entry 0's 2: {order}")
    assert found((1, 1, 7, 0), (2, 2, 8, 1), (1, 3, 9, 2)) == (
        REJECT, f"{PC.ENTRIES_ONE_ID} (a key's first 16 bytes), entries 0 and 2: an identifier names one key")
    assert found((1, 1, 7, 0), (1, 1, 9, 2))[1].startswith("starts: 0 shown for 2")


def test_the_mirror_reads_a_registration_as_the_network_attester_writes_it():
    """The rows form's honest registrations are `network_attester_rows.registration`'s documents, which its own
    `check_registration` reads back under the entry its execution phase picks: that entry's key, its session, the
    session's number for the execution phase, and the period's schedule."""
    for m, n in ((m, n) for m in ("rows-honest", "rows-two-stretches") for n in range(4)):
        case = build(m, n, small=n % 2 == 0)
        p = PC.params(case["params"])
        for w, x in enumerate(json.loads(bytes.fromhex(case["evidence"]))["windows"], start=p.first):
            assert [doc["format"] for doc in x["registrations"]] == [WR.REGISTRATION] * 2
            e = PC._entry_at(p, w)
            for doc, d in zip(x["registrations"], (PC.INGRESS, PC.EGRESS)):
                c = {k: bytes.fromhex(v) for k, v in doc["certificate"].items()}
                head, _ = WR.check_registration(doc, e.key, e.session, p.schedule, p.link, d, w - e.start)
                cert = Certificate(c["key"], p.link, d, w - e.start, c["digest"], c["signature"])
                assert WR.registration(head, cert, p.schedule) == doc


def test_a_small_period_reaches_its_verdict(server):
    """Each change's small period (`build(..., small=True)`, which the composition's difftest samples) gets the change's
    verdict from the mirror, and Lean agrees."""
    small = [build(m, n, small=True) for m in MUTATIONS if m != "case" for n in range(2)]
    assert [reference(c)["verdict"] for c in small] == [EXPECT[m] for m in MUTATIONS if m != "case" for _ in range(2)]
    D.run(dataclasses.replace(PORT, inputs=st.sampled_from(small)), examples=2 * len(small), server=server)


def test_each_kind_of_case_is_answered_as_lean_answers(server):
    """`case(kind, seed)` reaches its kind's verdict for each kind, the ones that fail to decode refused where they fail
    (a digest only once the commitment decodes); an accepted period's eps is its scans' deltas summed with its tied
    audits' sampled in part, and its profile names no sync; its evidence holds each execution phase's three records'
    certificates and two row registrations, with one scan an execution phase, all at one δ, and a schedules term equal to
    the cells'; `cases(seed)` is one of each with its answer; and Lean gives every one of them the mirror's answer."""
    sampled = {(k, s): PC.case(k, s) for k in KINDS for s in range(6)}
    answers = {ks: PC.answer(c) for ks, c in sampled.items()}
    assert {ks: a["verdict"] for ks, a in answers.items()} == {(k, s): verdict_of(k) for k, s in sampled}
    assert all(a["reason"].startswith(k.removeprefix("broken-") + ":")
               and (a["commitment"] is None) == (k != "broken-evidence")
               for (k, _), a in answers.items() if k.startswith("broken-"))
    accepted = [a["public"] for (k, _), a in answers.items() if k == ACCEPT]
    assert all(isinstance(x["K"], int) and x["eps"] == str(sum(
        (Fraction(t["delta"]) for t in x["scans"]), Fraction(0)) + sum(
        (Fraction(f["delta"]) for f in x.get("charged", ())), Fraction(0)))
        and [sample_size(f) for f in x.get("charged", ())] == [
            sample_size(x["audits"][i]) for i in charged(sampled[ks], x)]
        and all(f["delta"] == str(PC.TIED_DELTA) for f in x.get("charged", ()))
        for ks, a in answers.items() if ks[0] == ACCEPT for x in [a["public"]])
    assert all(sorted(a["profile"]) == ["link", "windows"] for (k, _), a in answers.items() if k == ACCEPT)
    shown = [w for (k, _), c in sampled.items() if k == ACCEPT
             for w in json.loads(bytes.fromhex(c["evidence"]))["windows"]]
    assert all(sorted(w) == PC.ROWS_WINDOW_KEYS and len(w["certificates"]) == 3 and len(w["registrations"]) == 2
               for w in shown)
    assert all(x["schedules"] == x["cells"] and len(x["scans"]) == x["windows"]
               and len({t["delta"] for t in x["scans"]}) == 1 for x in accepted)
    assert PC.cases(3) == {k: (sampled[k, 3], answers[k, 3]) for k in KINDS}
    D.run(dataclasses.replace(PORT, inputs=st.sampled_from(list(sampled.values()))), examples=2 * len(sampled),
          server=server)


def test_the_clock_form_is_refused(server):
    """Parameters in the clock form, a Σ_sync (`sync_set`) where the rows form has its schedule and scan, are refused as
    parameters before anything else of them is read, by Lean and by the mirror."""
    case = build("rows-honest", 0, small=True)
    params = {k: v for k, v in case["params"].items() if k not in ("schedule", "scan")}
    clock = case | {"params": params | {"sync_set": {"deltas": [0], "rhos": [[1, 1]]}}}
    out = reference(clock)
    assert out["verdict"] == REJECT and out["reason"] == f"params: {PC.CLOCK_FORM}", out
    verdict, got = D.replay(PORT, clock, server)
    assert verdict == REJECT and json.loads(got)["reason"] == f"params: {PC.CLOCK_FORM}", got


def test_the_rows_forms_bound_counts_what_it_says():
    """The rows form's K worked by hand (Lean's `rowsTerms` `#guard`s): 7 cells on one connection and 2 on another, 5
    bits for the schedules and 5 again for the fail cells, an audit whose own terms are in none, and over 1 execution phase
    one scan of 20 units (T = 10) sampled at δ = 1/15 with at most 3 incorrect (1351 sets), 1,120 free inputs, instances
    committing 300 and 512 bits, and jitter below 2 over its 10 steps; with every unit sampled, no term for incorrect
    proof blocks; at w63's class with the egress tie (T = 32, every unit sampled), K the tied slot entries' free inputs,
    8 · 1,136 = 9,088 bits; at w63k4's (8 slot entries of 112 bits) with its tied audit sampled in part with at most 2
    incorrect, K those and its E_w = 896 bits; and eps over the scans' δ alone, or with a charged audit's beside
    them."""
    p = Params((PC.Audit(0, 3, "", 10),), (7, 2), 0, 2, "a", (), "", 1, NetworkCertifierParams(),
               PC.Scan(10, 1120, 300, 512))
    out = public(p, [Facts(10, 3, Fraction(1, 10), 4)], [(Facts(20, 5, Fraction(1, 15), 3), 0, 0)])
    (scan,) = out["scans"]
    assert at_most(20, 3) == 1351 and clog2(1351) == 11
    assert [clog2(x) for x in range(6)] == [0, 0, 1, 2, 2, 3]
    assert [at_most(10, 0), at_most(10, 1), at_most(10, 2), at_most(4, 9), at_most(0, 3)] == [1, 11, 56, 16, 1]
    assert (out["schedules"], out["cells"], scan["free"], scan["faulty_sets"], scan["faulty_bits"], scan["jitter"],
            scan["faulty_entries"], out["K"]) == (5, 5, 1120, 11, 3 * 512, 10, 0, 5 + 5 + (1120 + 11 + 3 * 512 + 10))
    assert out["audits"] == [{"delta": "1/10", "drawn": 3, "units": 10, "wrong_units": 4}]
    sampled = dataclasses.replace(p, audits=(), cells=(), jitter=1, scan=PC.Scan(4, 0, 300, 512))
    assert public(sampled, [], [(Facts(8, 8, Fraction(1, 100), 0), 0, 0)])["K"] == 0
    w63 = dataclasses.replace(p, audits=(PC.Audit(8 * 1136, 0, "", 8),), cells=(), jitter=1,
                              scan=PC.Scan(32, 0, 300, 512))
    tied = public(w63, [Facts(8, 8, Fraction(1, 100), 0)], [(Facts(64, 64, Fraction(1, 100), 0), 8 * 1136, 0)])
    assert (tied["scans"][0]["free"], tied["K"]) == (9088, 9088)
    k4 = dataclasses.replace(w63, scan=PC.Scan(32, 0, 300, 512, 8 * 112))
    part = public(k4, [Facts(16, 9, Fraction(1, 100), 2)], [(Facts(64, 64, Fraction(1, 100), 0), 9088, 8 * 112)])
    assert (part["scans"][0]["faulty_entries"], part["K"]) == (896, 9088 + 896)
    two = dataclasses.replace(p, cells=(), windows=2, scan=PC.Scan(0, 0, 0, 0))
    assert public(two, [Facts(1, 1, Fraction(1, 10), 0)],
                  [(Facts(1, 1, Fraction(1, 15), 0), 0, 0)] * 2)["eps"] == "2/15"
    one = dataclasses.replace(two, windows=1)
    assert public(one, [Facts(2, 1, Fraction(1, 10), 0)], [(Facts(1, 1, Fraction(1, 15), 0), 0, 0)],
                  [Facts(2, 1, Fraction(1, 10), 0)])["eps"] == "1/6"
