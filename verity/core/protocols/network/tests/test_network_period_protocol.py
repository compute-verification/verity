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

__all__ = ["EXPECT", "MUTATIONS", "build", "reference"]

LEAN = Path(__file__).resolve().parents[1] / "lean"
pytestmark = pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")

TAGS: dict[str, str] = {}
SEEN: Counter = Counter()


#: The changes drawn more often than the rest: each form's accepted periods, its honest one and the rows form's tied
#: audit drawn in part, and its undecided answers, the only undecided ones.
WEIGHTS = {"honest": 4, "rows-honest": 4, "rows-tied-partial": 2, "answer-undecided": 2, "audit-undecided": 2,
           "rows-answer-undecided": 1, "rows-audit-undecided": 1}


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
    "rows-tied-undrawn": (PC.UNDRAWN,) * 2,
    "rows-tied-entries-undrawn": (PC.UNDRAWN,) * 2,
    "rows-tied-slots-undrawn": (PC.UNDRAWN,) * 2,
    "rows-state-untied": (PC.UNSTATED,) * 2,
}


def test_the_rows_form_refuses_what_it_must(server):
    """A link-window missing one of its records' certificates, one the warden left uncertified for refusing a
    connection (traffic past its Program slots), a registration not under the deployment key, the scan's answer for
    another window's rows, rows under a schedule whose digest isn't the bound one, scans at two δ, a scan answered over
    other than its 2T units, a scan whose `prog` is no compute audit's output, one whose `prog` is an audit's output but
    no `SlotEntry_v1` or `SelectEntry_v1` template's entries, an audit a scan's `prog` is tied to with a unit undrawn
    and no strata, or with its slot entries' or its scan slots' stratum drawn in part, and a scan whose state is not its
    wired audit's `SCAN_STATE`: Lean and the mirror each refuse every one at that check."""
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
    an audit the scan's `prog` isn't wired to, a root that is no string, and a second state entry at another root on
    either side; and a mirror that skips the tie accepts what Lean refuses."""
    unstate = PC.UNSTATES
    for u in unstate:
        monkeypatch.setattr(PC, "UNSTATES", (u,))
        for n in range(3):
            case = build("rows-state-untied", n, small=n % 2 == 0)
            assert PC.UNSTATED in reference(case)["reason"], (u, n, reference(case))
            verdict, out = D.replay(PORT, case, server)
            assert verdict == REJECT and PC.UNSTATED in json.loads(out)["reason"], (u, n, out)
    monkeypatch.setattr(PC, "UNSTATES", unstate)
    monkeypatch.setattr(PC, "_state_tied", lambda p, audits, reg: True)
    for n in range(2):
        refused_by_lean_alone(build("rows-state-untied", n, small=True), server, PC.UNSTATED)


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
    each scan's `free` is the parameters' `free` of the audits whose registration holds its `prog`'s root (the scan's
    own is 0), some of them nonzero."""
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


def charged(case: dict, public: dict) -> list[int]:
    """The audits of an accepted rows-form `case` its `public` charges: tied to a link-window's scan and drawn in part."""
    c = json.loads(bytes.fromhex(case["commitment"]))
    audits, windows = c["audits"], c["windows"]
    return [i for i, f in enumerate(public["audits"]) if f["drawn"] < f["units"] and PC._tied_in(audits, windows, i)]


def test_only_a_tied_audit_must_draw_its_entries_whole(server, monkeypatch):
    """K charges an audit a scan's `prog` is tied to its free inputs, and its wrong units only as the entries they can
    set, so the rows form refuses one drawn in part unless its slot entries' and scan slots' templates are strata drawn
    whole (`rows-tied-undrawn`, `rows-tied-entries-undrawn` and `rows-tied-slots-undrawn` above), and only such an
    audit: over honest rows periods, which the mirror accepts, every tied audit draws all its units and some untied
    audit fewer, and Lean accepts each such period with the mirror's public; a mirror that skips the draw's refusal
    accepts what Lean refuses at the draw, and so does one that holds only the slot entries' template whole a period
    whose scan slots' stratum is drawn in part."""
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
    with monkeypatch.context() as mp:
        mp.setattr(PC, "_draws_whole", lambda audit, prof, f: True)
        for m in ("rows-tied-undrawn", "rows-tied-entries-undrawn", "rows-tied-slots-undrawn"):
            for n in range(2):
                refused_by_lean_alone(build(m, n, small=True), server, PC.UNDRAWN)
    monkeypatch.setattr(PC.Audit, "whole", property(lambda a: tuple(k for k, _n, _d in a.entries)))
    for n in range(2):
        refused_by_lean_alone(build("rows-tied-slots-undrawn", n, small=True), server, PC.UNDRAWN)


def test_a_tied_audit_drawn_in_part_is_charged_its_link_windows_entries(server):
    """A tied audit whose slot entries' and scan slots' templates are strata drawn whole may draw the rest in part at
    its δ (`rows-tied-partial`): Lean accepts the period with the mirror's public, in which that audit's link-window's
    term charges every entry its at most `b` wrong units can set, `faulty_entries` = E_w = lanes · T / every · the
    entry's bits, when `b ≥ 1` and none when `b = 0`, every other link-window's none, and `eps` holds that audit's δ
    beside the scans'; the cases' audits select (`SelectEntry_v1` beside `ScanSlots_v2`) or don't (`SlotEntry_v1`)."""
    seen = Counter()
    for n in range(8):
        case = build("rows-tied-partial", n, small=n % 2 == 0)
        p, out = PC.params(case["params"]), reference(case)
        assert out["verdict"] == ACCEPT, (n, out)
        verdict, got = D.replay(PORT, case, server)
        assert verdict == ACCEPT and json.loads(got)["public"] == out["public"], (n, got)
        pub, (i,) = out["public"], charged(case, out["public"])
        s, prog = p.schedule, PC._scan_widths(case["params"]["scan"])[1]
        assert p.scan.entries == s.lanes * (s.window // s.every) * prog, (n, p.scan)
        c = json.loads(bytes.fromhex(case["commitment"]))
        audits = c["audits"]
        b = pub["audits"][i]["wrong_units"]
        assert [t["faulty_entries"] for t in pub["scans"]] == [
            p.scan.entries if b and i in PC._tied_audits(audits, reg) else 0
            for reg in c["windows"]], (n, pub)
        assert pub["eps"] == str(sum((Fraction(t["delta"]) for t in pub["scans"]), Fraction(pub["audits"][i]["delta"])))
        seen["charged" if b else "uncharged"] += 1
        seen["select" if p.audits[i].scan_slots else "slot"] += 1
    assert all(seen[k] for k in ("charged", "uncharged", "select", "slot")), seen


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
    compute audit's output), an ingress and an egress instance's committed bits (their returns: a u16 with the state
    and view, or with the row), and its slot entries' bits (lanes · T / every entries of its `prog` ports' width); and
    Lean and the mirror refuse, as parameters, a scan that isn't a descriptor, another Program's, one in the checker
    form (its egress row an input), one whose templates have other parameters or other than 1, T − 1 and T instances,
    or rows other than the schedule's, one whose lane reads another row of its input than its bucket's or another
    lane's input, one with an input gate no template's, and each negative of a check of what `NetTiming.Layout.Fits`
    asks."""
    for n in range(3):
        case = build("rows-honest", n, small=n % 2 == 0)
        p, desc = PC.params(case["params"]), case["params"]["scan"]
        state, prog, view, _words = PC._scan_widths(desc)
        s, T, row = p.schedule, p.schedule.window, WR.row_bytes(p.schedule)
        assert p.scan == PC.Scan(T, 0, 16 + state + view, 16 + 8 * row, s.lanes * (T // s.every) * prog), (n, p.scan)
        for bad in PC.bad_scans(case["params"]):
            broken = case | {"params": bad}
            assert reference(broken)["reason"].startswith("params: scan:"), (n, reference(broken))
            verdict, out = D.replay(PORT, broken, server)
            assert verdict == REJECT and "scan" in json.loads(out)["reason"], (n, out)


def test_each_fits_check_refuses_its_negative(server):
    """Each of the scan's checks of what `NetTiming.Layout.Fits` asks refuses, in Lean and in the mirror, a scan that
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
    (`NetTiming.Layout.Fits`), so `K` counts only what a Program that computes its egress rows leaves free."""
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
    """A compute audit's faulty bits are its bound times the most bits one of its Program's units commits, read off the
    Program, not a proof unit's 16-bit public output: over honest clock-form periods whose audits' units commit 8 to
    320 bits, each audit is charged `b · bits`, some audit of wide units more than `16 · b`, and Lean's public is the
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
    clock form's audits', the rows form's scans' and its tied audits' drawn in part), and its profile names no sync; in
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
    assert all(isinstance(x["K"], int) and x["eps"] == str(sum(
        (Fraction(t["delta"]) for t in x.get("scans", x["audits"])), Fraction(0)) + sum(
        (Fraction(x["audits"][i]["delta"]) for i in (charged(drawn[ks], x) if "scans" in x else ())), Fraction(0)))
        for ks, a in answers.items() if ks[0] == ACCEPT for x in [a["public"]])
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
    4 wrong, 7 cells on one connection and none on another, 2 syncs over 3 link-windows, jitter below 2 over 3 steps;
    the same audit with units committing 40 bits, its faulty bits 4 · 40; and eps over audits at δ = 1/10, 1/15 and
    1/6, and over none (Lean's `#guard`s)."""
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


def test_the_rows_forms_bound_counts_what_it_says():
    """The rows form's K worked by hand (Lean's `rowsTerms` `#guard`s): 7 cells on one connection and 2 on another, 5
    bits for the schedules and 5 again for the fail cells, an audit whose own terms are in none, and over 1 link-window
    one scan of 20 units (T = 10) drawn at δ = 1/15 with at most 3 wrong (1351 sets), 1,120 free inputs, instances
    committing 300 and 512 bits, and jitter below 2 over its 10 steps; with every unit drawn, no faulty term; at w63's
    class with the egress tie (T = 32, every unit drawn), K the tied slot entries' free inputs, 8 · 1,136 = 9,088 bits;
    at w63k4's (8 slot entries of 112 bits) with its tied audit drawn in part with at most 2 wrong, K those and its
    E_w = 896 bits; and eps over the scans' δ alone, or with a charged audit's beside them."""
    p = Params((PC.Audit(0, 3, "", 10, 16),), (7, 2), 0, 2, "a", None, b"", "", None, 1, WardenParams(),
               PC.Scan(10, 1120, 300, 512))
    out = public(p, [Facts(10, 3, Fraction(1, 10), 4)], [(Facts(20, 5, Fraction(1, 15), 3), 0, 0)])
    (scan,) = out["scans"]
    assert at_most(20, 3) == 1351 and clog2(1351) == 11
    assert (out["schedules"], out["cells"], scan["free"], scan["faulty_sets"], scan["faulty_bits"], scan["jitter"],
            scan["faulty_entries"], out["K"]) == (5, 5, 1120, 11, 3 * 512, 10, 0, 5 + 5 + (1120 + 11 + 3 * 512 + 10))
    assert out["audits"] == [{"delta": "1/10", "drawn": 3, "units": 10, "wrong_units": 4}]
    drawn = dataclasses.replace(p, audits=(), cells=(), jitter=1, scan=PC.Scan(4, 0, 300, 512))
    assert public(drawn, [], [(Facts(8, 8, Fraction(1, 100), 0), 0, 0)])["K"] == 0
    w63 = dataclasses.replace(p, audits=(PC.Audit(8 * 1136, 0, "", 8, 1136),), cells=(), jitter=1,
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
