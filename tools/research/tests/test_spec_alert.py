"""`spec_alert`: the DM the spec's reader gets when a push lands a change to what a guarantee says, and nothing for a move or a reprint."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import research.merge as merge
import research.spec_alert as spec
from research.slack import SlackError

H1, H2, H3 = "a" * 64, "b" * 64, "c" * 64
REG = {"humans": [{"name": "Daniel Reuter", "slack_user": "U0DANIEL"}], "channels": {"agent-alerts": {"id": "C0ALERTS"}}}


def _g(sig: str, h: str) -> dict:
    return {"signature": [sig], "assumptions": [], "type_hash": h}


def _doc(guarantees: dict, defs: dict | None = None) -> dict:
    return {"guarantees": guarantees, "reads": {"P.Defs": {"digest": "x", "definitions": defs or {}}}}


class Api:
    def __init__(self, fail: str | None = None, error: Exception | None = None):
        self.calls: list[dict] = []
        self.fail = fail
        self.error = error

    def call(self, method: str, **params):
        self.calls.append({"method": method, **params})
        if self.fail and params.get("channel") == self.fail:
            raise self.error or SlackError(f"{method}: channel_not_found")
        return {"ok": True}


def test_a_restated_guarantee_shows_its_statement_before_and_after():
    lines = spec.changes(_doc({"P.t": _g("P.t : 1 = 1", H1)}), _doc({"P.t": _g("P.t : 2 = 2", H2)}))
    assert len(lines) == 1 and lines[0].startswith("• changed guarantee `P.t`, before:")
    assert "P.t : 1 = 1" in lines[0] and lines[0].index("P.t : 1 = 1") < lines[0].index("after:") < lines[0].index("P.t : 2 = 2")


def test_a_restated_definition_a_guarantee_is_stated_as_shows_it_changed():
    """`theorem G : Guarantees.G` keeps its signature and type hash when `Guarantees.G`'s body is restated; its
    `unfolded_hash` moves, and the DM shows it changed. A record that gains the hash, or a push with no Lean input changed,
    shows nothing."""
    sig = "P.SecurityProofs.G : P.Guarantees.G"
    was, now = (_doc({"P.SecurityProofs.G": _g(sig, H1) | {"unfolded_hash": u}}) for u in (H2, H3))
    (line,) = spec.changes(was, now)
    assert line.startswith("• changed guarantee `P.SecurityProofs.G`: its definition, unfolded once, says something else")
    assert sig in line
    assert spec.changes(_doc({"P.SecurityProofs.G": _g(sig, H1)}), now) == []
    assert [k for _, k, _ in spec._across({"a": was}, {"a": now}, sources=False)] == []
    (rehashed,) = spec.changes(was, _doc({"P.SecurityProofs.G": _g(sig, H3) | {"unfolded_hash": H2}}))
    assert rehashed.startswith("• changed guarantee `P.SecurityProofs.G`, before:")


def test_new_and_removed_guarantees_are_listed_and_a_move_is_not():
    old = _doc({"P.t": _g("P.t : A", H1), "P.gone": _g("P.gone : B", H2)})
    new = _doc({"Q.t": _g("Q.t : A", H1), "P.new": _g("P.new : C", H3)})
    lines = spec.changes(old, new)
    assert lines[0] == "• removed guarantee `P.gone`"                       # P.t -> Q.t has the same type hash: a move
    assert len(lines) == 2 and lines[1].startswith("• new guarantee `P.new`") and "P.new : C" in lines[1]


def test_a_reprint_and_a_rehash_send_nothing():
    assert spec.changes(_doc({"P.t": _g("P.t : A", H1)}), _doc({"P.t": _g("P.t :  A", H1)})) == []
    assert spec.changes(_doc({"P.t": _g("P.t : A", "1234")}), _doc({"P.t": _g("P.t : A", H1)})) == []
    assert spec.changes(_doc({"P.t": _g("P.t : A", "1234")}), _doc({"P.t": _g("P.t : B", H1)})) != []


def test_a_guarantee_whose_premises_change_is_shown_and_one_that_first_records_them_is_tallied():
    """A premise gained or dropped is shown, whichever kind it is; one that only moved between `assumptions` and `open`
    is no change to what the statement takes, and `reads` holds what it only reads."""
    def rests(*premises: str, open_: tuple = (), reads: tuple = ()) -> dict:
        return _doc({"P.t": _g("P.t : A", H1) | {"premises": {"assumptions": list(premises), "open": list(open_),
                                                             "reads": list(reads)}}})
    old, new = rests("P.Hyp", "P.Old"), rests("P.Hyp", open_=("P.TapOpens",))
    assert spec.changes(rests("P.Hyp", "P.Old"), rests("P.Hyp", open_=("P.Old",), reads=("P.Seen",))) == []
    assert spec.changes(old, new) == ["• guarantee `P.t` rests now also on `P.TapOpens`; no longer on `P.Old`"]
    assert spec.changes(rests("P.Hyp"), old) == ["• guarantee `P.t` rests now also on `P.Old`"]
    assert spec.changes(new, new) == [] and spec.changes(new, _doc({"P.t": _g("P.t : A", H1)})) == []
    entries = spec._across({"a": _doc({"P.t": _g("P.t : A", H1)})}, {"a": new})
    assert [(k, t) for _, k, t in entries] == [("premised", "")]
    assert spec._tally(["premises", "premised", "premised", "new"]) == \
        "guarantees: 1 new, 1 with new premises, 2 premises recorded"


def test_a_changed_definition_names_its_module():
    old = _doc({"P.t": _g("P.t : D", H1)}, {"P.D": H1, "P.E": H2})
    new = _doc({"P.t": _g("P.t : D", H1)}, {"P.D": H3, "P.E": H2})
    assert spec.changes(old, new) == ["• definition `P.D` (`P.Defs`) changed"]
    moved = {"reads": {"P.Other": {"definitions": {"P.D": H1, "P.E": H2}}}, "guarantees": old["guarantees"]}
    assert spec.changes(old, moved) == []                                   # same names and hashes in another module


def test_a_lock_that_drops_its_readers_sends_nothing(tmp_path):
    """A lock that drops the guarantees each `reads` group listed, its definitions and hashes as they were."""
    repo = _repo(tmp_path)
    listed = _doc({"P.t": _g("P.t : A", H1)}, {"P.D": H1})
    listed["reads"]["P.Defs"]["guarantees"] = ["P.t"]
    old = _commit(repo, "pkg/lean-audit.json", listed, "base")
    _source(repo, "pkg/P.lean")
    new = _commit(repo, "pkg/lean-audit.json", _doc({"P.t": _g("P.t : A", H1)}, {"P.D": H1}), "m")
    assert spec.message(repo, old, new) is None


def test_an_assumption_no_reads_record_holds_is_shown_when_its_digest_changes_or_is_first_recorded():
    """A premise outside `meaning` has only its own record's digest, so a strengthened premise (such as
    `RecDeploy.PlainLeavesAdmitted`) would reach no DM. A changed digest is shown (tallied as rehashed when no Lean input
    changed), and so is one first recorded; one a `reads` record holds shows as its definition, an unchanged one shows
    nothing, and a new one shows as a named assumption listed."""
    def doc(assumed: dict, defs: dict | None = None) -> dict:
        return _doc({"P.t": _g("P.t : A", H1)}, defs) | {"assumptions": assumed}
    old = doc({"P.Leaves": {"owner": "@proofs", "digest": H1}, "P.Bare": {"owner": "@proofs"}, "P.Hyp": {"digest": H1}},
              {"P.Hyp": H1})
    new = doc({"P.Leaves": {"owner": "@proofs", "digest": H2}, "P.Bare": {"owner": "@proofs", "digest": H3},
               "P.Hyp": {"digest": H2}, "P.New": {"digest": H3}}, {"P.Hyp": H2})
    assert spec.changes(old, new) == [
        "• named assumption `P.New` listed",
        "• definition `P.Hyp` (`P.Defs`) changed",
        "• assumption `P.Bare` recorded: its statement and the definitions it reads are now digested in its record",
        "• assumption `P.Leaves` changed: its statement or a definition it reads"]
    assert spec.changes(new, new) == []
    entries = spec._across({"lean-audit.json": old}, {"lean-audit.json": new}, sources=False)
    assert [k for _, k, _ in entries] == ["assumed", "rehashed", "recorded", "rehashed"]
    assert spec._tally([k for _, k, _ in spec._across({"a": old}, {"a": new})]) == \
        "definitions: 1 changed; premises: 1 changed, 1 recorded, 1 named"


def test_the_named_assumptions_and_the_open_premises_are_shown_as_they_change():
    """The lock's real assumptions are the few under `assumptions`, so one named, one no longer named and an open premise
    that becomes one are shown; an open premise listed, discharged or re-ticketed is shown, and the named assumptions an
    update first moves to `open` are tallied."""
    def doc(assumed: dict, opened: dict) -> dict:
        return _doc({"P.t": _g("P.t : A", H1)}) | {"assumptions": assumed, "open": opened}
    ticket = {"owner": "@proofs", "issue": "PROOFS-1", "due": "2026-11-30"}
    old = doc({"P.Coll": {"owner": "@lean"}, "P.Gone": {"owner": "@lean"}, "P.Moved": {"owner": "@lean"}},
              {"P.Real": ticket, "P.Done": ticket, "P.Late": ticket})
    new = doc({"P.Coll": {"owner": "@lean"}, "P.Real": {"owner": "@lean"}, "P.Hash": {"owner": "@lean"}},
              {"P.Moved": ticket, "P.Late": ticket | {"due": "2026-12-31"}, "P.Tap": ticket})
    assert spec.changes(old, new) == [
        "• named assumption `P.Hash` listed",
        "• open premise `P.Late`: PROOFS-1, due 2026-12-31 (was PROOFS-1, due 2026-11-30)",
        "• named assumption `P.Real` listed (was open: PROOFS-1, due 2026-11-30)",
        "• open premise `P.Tap` listed (PROOFS-1, due 2026-11-30)",
        "• open premise `P.Done` discharged: no longer listed (PROOFS-1, due 2026-11-30)",
        "• named assumption `P.Gone` no longer listed"]
    assert spec._tally([k for _, k, _ in spec._across({"a": old}, {"a": new})]) == \
        "premises: 2 named, 1 no longer named, 1 open listed, 1 now open, 1 discharged, 1 re-ticketed"
    assert spec.changes(new, new) == []
    read = _doc({"P.t": _g("P.t : A", H1)}, {"P.Gone": H2}) | {"assumptions": {}, "open": new["open"]}
    assert spec.changes(new | {"assumptions": {"P.Gone": {"owner": "@lean"}}}, read) == [
        "• named assumption `P.Gone` no longer listed: statements only read it, and its `reads` record holds it"]


def test_a_domain_hypothesis_is_shown_with_what_it_restricts_as_it_is_listed_moved_or_restated():
    """A domain hypothesis owes no issue, so each one listed, each premise that moves in or out of `domain`, and a new
    `why` are shown with what it restricts."""
    def doc(**sections) -> dict:
        return _doc({"P.t": _g("P.t : A", H1)}) | sections
    ticket = {"owner": "@proofs", "issue": "PROOFS-1", "due": "2026-11-30"}
    scope = {"owner": "@proofs", "why": "the circuits the bound is over"}
    old = doc(open={"P.Scope": ticket, "P.Kept": ticket}, domain={"P.Event": scope, "P.Back": scope, "P.Gone": scope})
    new = doc(open={"P.Kept": ticket, "P.Back": ticket},
              domain={"P.Scope": scope, "P.Event": scope | {"why": "the runs that accepted"},
                      "P.Below": scope | {"why": "the layer below's guarantee", "cites": "P.u"}})
    assert spec.changes(old, new) == [
        "• open premise `P.Back` listed (PROOFS-1, due 2026-11-30) (was a domain hypothesis: the circuits the bound is over)",
        "• domain hypothesis `P.Below` listed: the layer below's guarantee, citing `P.u`",
        "• domain hypothesis `P.Event`: the runs that accepted (was the circuits the bound is over)",
        "• domain hypothesis `P.Scope` listed: the circuits the bound is over (was open: PROOFS-1, due 2026-11-30)",
        "• domain hypothesis `P.Gone` no longer listed (the circuits the bound is over)"]
    assert spec._tally([k for _, k, _ in spec._across({"a": old}, {"a": new})]) == \
        "premises: 2 domain listed, 1 domain dropped, 1 domain restated, 1 open listed"


def test_an_entry_added_to_the_frozen_shape_list_is_shown_with_its_reason_and_one_removed_is_not():
    """Adding to `shape.frozen` exempts a guarantee from the shape rule: a claims change, so the DM shows it."""
    base = _doc({"P.t": _g("P.t : A", H1)})
    frozen = base | {"shape": {"frozen": {"P.t": "RESTATE: split the iff."}}}
    assert spec.changes(base, frozen) == ["• frozen `P.t` (doesn't have the shape yet): RESTATE: split the iff."]
    assert spec.changes(frozen, base) == [] and spec.changes(frozen, frozen) == []
    assert spec._tally(["frozen", "frozen", "new"]) == "guarantees: 1 new; shape: 2 frozen"


def test_a_conjecture_added_or_marked_falsified_is_shown_with_its_reason():
    """`shape.conjectures`: a property resting on an open one is conditional, and a falsified one takes every property
    resting on it off the list, so a new entry or a new status is shown; an unchanged one is not."""
    base = _doc({"P.t": _g("P.t : A", H1)})
    opened = base | {"shape": {"conjectures": {"P.C": {"status": "open", "test": "t.py", "why": "unsettled"}}}}
    dead = base | {"shape": {"conjectures": {"P.C": {"status": "falsified", "why": "known false"}}}}
    assert spec.changes(base, opened) == ["• conjecture `P.C` open: unsettled"]
    assert spec.changes(opened, dead) == ["• conjecture `P.C` falsified (was open): known false"]
    assert spec.changes(dead, dead) == [] and spec.changes(dead, base) == []
    assert spec._tally(["frozen", "conjecture"]) == "shape: 1 frozen, 1 conjectures marked"


def test_compiled_code_newly_allowed_on_a_verifiers_path_is_shown():
    """`verifier.allow` lists the `@[extern]` and `@[implemented_by]` constants the verifier may reach, which no
    proof sees: an entry added is shown; one dropped, or the list unchanged, is not."""
    base = _doc({"P.t": _g("P.t : A", H1)}) | {"verifier": {"entries": ["P.verify"], "allow": ["Nat.add"]}}
    more = base | {"verifier": {"entries": ["P.verify"], "allow": ["ByteArray.push", "Nat.add"]}}
    assert spec.changes(base, more) == ["• compiled code allowed on the verifier's path, which no proof sees: `ByteArray.push`"]
    assert spec.changes(more, base) == [] and spec.changes(more, more) == []
    assert spec._tally(["compiled", "compiled", "new"]) == "guarantees: 1 new; verifier: 2 allowed"


TRUST_KINDS = {"grown", "deployed", "undeployed", "trusted", "untrusted", "shrunk", "trust"}


def _trust(doc: dict, deployed: list[str], gap: dict, base: dict | None = None, extra: dict | None = None) -> dict:
    return doc | {"deployed": deployed, "trust": {"base": base or {}, "extra": extra or {}}, "trust_gap": gap}


def test_a_trust_gap_grown_by_hand_comes_first_and_one_that_shrank_is_shown_after_the_statements():
    """An update only shrinks `trust_gap`, so a premise a deployed guarantee newly rests on outside its trust was added by
    hand, the one way past that ratchet: it leads the DM, ahead of a restated guarantee."""
    old = _trust(_doc({"A.G": _g("G", "h1"), "A.H": _g("H", "h2")}), ["A.G", "A.H"], {"A.G": ["P1"], "A.H": ["Q1", "Q2"]})
    new = _trust(_doc({"A.G": _g("G", "h1"), "A.H": _g("H'", "h3")}), ["A.G", "A.H"], {"A.G": ["P1", "P2"], "A.H": ["Q1"]})
    got = spec.changes(old, new)
    assert got[0] == "• trust gap grown, by hand: deployed guarantee `A.G` now also rests outside its trust on `P2`"
    assert got[1].startswith("• changed guarantee `A.H`, before:")
    assert got[2] == "• trust gap shrunk: `A.H` no longer rests outside its trust on `Q2`"
    assert spec._tally([k for k, _ in spec._entries(old, new)]) == "trust: 1 gap grown by hand, 1 gap shrunk; guarantees: 1 changed"


def test_a_guarantee_deployed_or_no_longer_and_a_premise_trusted_or_no_longer_are_shown():
    old = _trust(_doc({"A.G": _g("G", "h1"), "A.H": _g("H", "h2")}), ["A.G"], {}, {"C.CR": "collision resistance",
                                                                                   "C.Old": "an old bound"},
                 {"A.G": {"C.Src": "the firewall's randomness"}})
    new = _trust(_doc({"A.G": _g("G", "h1"), "A.H": _g("H", "h2")}), ["A.H"], {}, {"C.CR": "collision resistance, strict",
                                                                                   "C.New": "a new bound"},
                 {"A.H": {"C.Src": "the firewall's randomness"}})
    assert spec.changes(old, new) == [
        "• deployed guarantee `A.H` listed: it states the deployed system",
        "• `A.G` no longer listed as deployed",
        "• trust base `C.CR`: collision resistance, strict (was: collision resistance)",
        "• trust base `C.New`: a new bound, listed",
        "• deployed guarantee `A.H` also trusts `C.Src`: the firewall's randomness",
        "• trust base `C.Old` no longer listed (it stated: an old bound)",
        "• deployed guarantee `A.G` no longer trusts `C.Src`"]
    assert spec._tally([k for k, _ in spec._entries(old, new)]) == \
        "trust: 1 deployed, 1 no longer deployed, 3 trusted, 2 no longer trusted"


def test_a_rename_carries_the_trust_with_it_and_changes_nothing_of_it():
    """A guarantee renamed with its type hash and a premise renamed with its digest keep their places in `deployed`, the
    trust and `trust_gap`: no trust entry, though the renamed assumption itself still shows as listed and no longer listed."""
    def doc(g: str, p: str) -> dict:
        d = _trust(_doc({g: _g("G", "h1")}), [g], {g: [p, "A.Assumptions.Kept"]}, {p: "a trusted bound"}, {g: {p: "a ruling"}})
        return d | {"assumptions": {p: {"digest": "d1"}, "A.Assumptions.Kept": {"digest": "d2"}}}

    old, new = doc("A.SpotSound", "A.Assumptions.SampledHash"), doc("A.SpotSound", "A.Assumptions.SpotHash")
    kinds = [k for k, _ in spec._entries(old, new)]
    assert not TRUST_KINDS & set(kinds), spec.changes(old, new)
    assert sorted(kinds) == ["assumed", "unassumed"]


def test_a_gap_premise_renamed_with_its_text_is_no_growth_and_one_the_records_already_held_is():
    """A rename that changes a premise's text changes its hash too (`DrawIsCoin` to `SampleIsCoin`): a name no old record
    held, in place of one no new record holds, is that rename. A premise the records already held, newly in the gap, was
    put there by hand."""
    def doc(renamed: str, h: str, gap: list[str]) -> dict:
        d = _trust(_doc({"A.G": _g("G", "h1")}, {renamed: h, "A.Other": "h2"}), ["A.G"], {"A.G": gap})
        return d | {"open": {renamed: {"issue": "X-1"}, "A.Other": {"issue": "X-2"}}}

    old = doc("A.DrawIsCoin", "h3", ["A.DrawIsCoin"])
    new = doc("A.SampleIsCoin", "h4", ["A.SampleIsCoin", "A.Other"])
    assert [t for k, t in spec._entries(old, new) if k in TRUST_KINDS] == [
        "• trust gap grown, by hand: deployed guarantee `A.G` now also rests outside its trust on `A.Other`"]


def test_a_record_that_holds_a_trust_for_the_first_time_is_tallied_and_sends_nothing(tmp_path):
    old = _doc({"A.G": _g("G", "h1")})
    new = _trust(_doc({"A.G": _g("G", "h1")}), ["A.G"], {"A.G": ["P1"]}, {"C.CR": "collision resistance"})
    assert spec._entries(old, new) == [] and [k for _, k, _ in spec._across({"x": old}, {"x": new})] == ["trust"]
    repo = _repo(tmp_path)
    a = _commit(repo, "security/lean-audit.json", old, "a")
    b = _commit(repo, "security/lean-audit.json", new, "b")
    assert spec.message(repo, a, b) is None


def test_a_record_from_before_the_rename_compares_by_what_it_says():
    new = _doc({"P.t": _g("P.t : D", H1)}, {"P.D": H1})
    old = {"pins": new["guarantees"], "reads": {"P.Defs": {"digest": "x", "definitions": {"P.D": H1}, "pins": ["P.t"]}}}
    assert spec.changes(old, new) == []
    assert spec.changes(old, _doc({"P.t": _g("P.t : E", H2)}, {"P.D": H3})) == [
        "• changed guarantee `P.t`, before:\n```\nP.t : D\n```\nafter:\n```\nP.t : E\n```",
        "• definition `P.D` (`P.Defs`) changed"]


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    for args in (["init", "-q", "-b", "main"], ["config", "user.email", "t@t"], ["config", "user.name", "t"],
                 ["remote", "add", "origin", "git@github.com:owner/verity.git"]):
        subprocess.run(["git", "-C", str(repo), *args], check=True)
    return repo


def _commit(repo: Path, rel: str, doc: dict, msg: str) -> str:
    (repo / rel).parent.mkdir(parents=True, exist_ok=True)
    (repo / rel).write_text(json.dumps(doc, indent=1))
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-q", "-m", msg], check=True)
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


def _source(repo: Path, rel: str, text: str = "def x := 1\n") -> None:
    """A Lean input the next `_commit` takes along, so its definitions' changed hashes say something changed."""
    (repo / rel).parent.mkdir(parents=True, exist_ok=True)
    (repo / rel).write_text(text + str(len(list(repo.rglob("*.lean")))))


def test_a_definition_rehashed_or_regrouped_with_no_lean_input_changed_is_tallied_and_not_listed(tmp_path):
    """The audit regrouping its reads records can change a hundred definitions' hashes with no Lean source behind them, and
    the DM would list every one; a Lean input changed in the push, or one outside the audit's own directory, lists
    them as before."""
    repo = _repo(tmp_path)
    old = _commit(repo, "pkg/lean-audit.json", _doc({"P.t": _g("P.t : A", H1)}, {"P.D": H1, "P.E": H2}), "base")
    (repo / "tools/verity/lean").mkdir(parents=True)
    (repo / "tools/verity/lean/Facts.lean").write_text("-- the audit's own extractor\n")
    regrouped = _commit(repo, "pkg/lean-audit.json", _doc({"P.t": _g("P.t : A", H1)}, {"P.D": H3, "P.E": H2}), "m")
    assert spec.message(repo, old, regrouped) is None
    with_guarantee = _doc({"P.t": _g("P.t : A", H1), "P.n": _g("P.n : N", H2)}, {"P.D": H3, "P.E": H2})
    tallied = _commit(repo, "pkg/lean-audit.json", with_guarantee, "m")
    text = spec.message(repo, old, tallied)
    assert "• `pkg`: guarantees: 1 new; definitions: 1 rehashed" in text and "`P.D`" not in text
    prev, h = regrouped, H3
    for source in ("pkg/P.lean", "pkg/lean-toolchain", "other/lake-manifest.json", "other/lakefile.toml"):
        _source(repo, source)
        h = H1 if h == H3 else H3
        cur = _commit(repo, "pkg/lean-audit.json", _doc({"P.t": _g("P.t : A", H1)}, {"P.D": h, "P.E": H2}), "m")
        assert spec.message(repo, prev, cur).endswith("*pkg*\n• definition `P.D` (`P.Defs`) changed"), source
        prev = cur


def test_a_newly_pinned_dependency_bundle_is_a_lean_input_and_a_rekeyed_or_dropped_one_is_not(tmp_path):
    """Every definition that reads Mathlib or ArkLib is built against the bundle `lean-deps.json` pins: pinning another one with
    no manifest change would otherwise show only as `rehashed`. Re-keying a bundle or dropping a pin is no Lean input."""
    repo = _repo(tmp_path)
    pins = lambda **deps: {"deps": {k: {"package": "pkg", "sha256": s} for k, s in deps.items()}}
    _commit(repo, spec.DEPS_PINS, pins(k1=H1, k2=H2), "pins")
    old = _commit(repo, "pkg/lean-audit.json", _doc({"P.t": _g("P.t : A", H1)}, {"P.D": H1}), "base")
    _commit(repo, spec.DEPS_PINS, pins(k3=H1), "re-key one bundle, drop the other")
    rekeyed = _commit(repo, "pkg/lean-audit.json", _doc({"P.t": _g("P.t : A", H1)}, {"P.D": H3}), "m")
    assert spec.message(repo, old, rekeyed) is None
    _commit(repo, spec.DEPS_PINS, pins(k3=H1, k4=H3), "a new bundle")
    pinned = _commit(repo, "pkg/lean-audit.json", _doc({"P.t": _g("P.t : A", H1)}, {"P.D": H2}), "m")
    assert spec.message(repo, rekeyed, pinned).endswith("*pkg*\n• definition `P.D` (`P.Defs`) changed")


def test_the_dm_names_the_package_the_pull_requests_and_the_compare_link(tmp_path):
    repo = _repo(tmp_path)
    old = _commit(repo, "pkg/lean/lean-audit.json", _doc({"P.t": _g("P.t : A", H1)}), "base")
    new = _commit(repo, "pkg/lean/lean-audit.json", _doc({"P.t": _g("P.t : B", H2)}), "Merge #12 #34 (abc)")
    text = spec.message(repo, old, new, spec.web(repo, "origin"))
    assert text.startswith(f"A spec change landed on main (<https://github.com/owner/verity/compare/{old[:12]}...{new[:12]}|")
    assert "<https://github.com/owner/verity/pull/12|#12>, <https://github.com/owner/verity/pull/34|#34>" in text
    assert "*pkg/lean*\n• changed guarantee `P.t`" in text


def test_a_record_that_moved_with_its_package_is_compared_at_its_old_path(tmp_path):
    repo = _repo(tmp_path)
    doc = _doc({"P.t": _g("P.t : A", H1), "P.u": _g("P.u : B", H2)})
    old = _commit(repo, "protocols/p/lean/lean-audit.json", doc, "base")
    subprocess.run(["git", "-C", str(repo), "mv", "protocols/p", "p"], check=True)
    moved = _commit(repo, "p/lean/lean-audit.json", doc, "Layout move (#9)")
    assert spec.message(repo, old, moved) is None
    edited = _commit(repo, "p/lean/lean-audit.json", _doc({"P.t": _g("P.t : C", H3), "P.u": _g("P.u : B", H2)}), "m")
    text = spec.message(repo, old, edited)
    assert "*p/lean*\n• changed guarantee `P.t`" in text and "new guarantee" not in text and "P.u" not in text


def test_a_fixture_of_the_audits_tests_is_no_record(tmp_path):
    """A layout move renamed a test fixture named `lean-audit.json` under the audit's directory, whose `reads` are lists:
    read as a record it failed the DM, and Daniel was told the records couldn't be read."""
    repo = _repo(tmp_path)
    toy = {"guarantees": {"Core.G": {"owner": "@lean"}}, "reads": {"Core.G": ["Core.helper"]}}
    old = _commit(repo, "tools/verity/lean/tests/fixtures/unread/Old/lean-audit.json", toy, "base")
    subprocess.run(["git", "-C", str(repo), "mv", "tools/verity/lean/tests/fixtures/unread/Old",
                    "tools/verity/lean/tests/fixtures/unread/new"], check=True)
    moved = _commit(repo, "tools/verity/lean/tests/fixtures/unread/new/lean-audit.json", toy, "Layout move (#9)")
    assert spec._records_changed(repo, old, moved) == [] and spec.message(repo, old, moved) is None
    edited = _commit(repo, "tools/verity/lean/tests/fixtures/unread/new/lean-audit.json", {**toy, "roots": ["A"]}, "m")
    assert spec.message(repo, moved, edited) is None


def test_records_that_moved_between_lock_files_are_matched_by_name_and_tallied_as_moves(tmp_path):
    """The Lean move merged six packages' records into one: matched file by file, the DM listed 990 guarantees removed from
    the six and 990 new in the one, where all but a handful kept their names and type hashes."""
    repo = _repo(tmp_path)
    _commit(repo, "a/lean-audit.json", _doc({"P.t": _g("P.t : A", H1), "P.u": _g("P.u : B", H2)}, {"P.D": H1}), "a")
    old = _commit(repo, "b/lean-audit.json", _doc({"Q.v": _g("Q.v : C", H3)}, {"Q.E": H2}), "b")
    subprocess.run(["git", "-C", str(repo), "rm", "-q", "a/lean-audit.json", "b/lean-audit.json"], check=True)
    both = _doc({"P.t": _g("P.t : A", H1), "P.u": _g("P.u : B", H2), "Q.v": _g("Q.v : C", H3)})
    both["reads"] = {"P.Defs": {"definitions": {"P.D": H1, "Q.E": H2}}}
    moved = _commit(repo, "c/lean-audit.json", both, "Layout move (#9)")
    assert spec.message(repo, old, moved) is None
    both["guarantees"]["Q.v"] = _g("Q.v : D", H1)
    both["reads"]["P.Defs"]["definitions"]["Q.E"] = H3
    _source(repo, "c/Q.lean")
    edited = _commit(repo, "c/lean-audit.json", both, "m")
    text = spec.message(repo, old, edited)
    assert "• `c`: guarantees: 1 changed, 2 moved here; definitions: 1 changed" in text
    assert "removed" not in text and "new guarantee" not in text and "`P.D`" not in text
    assert "*c*\n• changed guarantee `Q.v`" in text and "• definition `Q.E` (`P.Defs`) changed" in text


def test_a_long_dm_ends_with_the_compare_link_and_closes_its_code_block(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    monkeypatch.setattr(spec, "LIMIT", 400)
    old = _commit(repo, "lean-audit.json", _doc({}), "base")
    new = _commit(repo, "lean-audit.json", _doc({f"P.t{i}": _g(f"P.t{i} : " + "x" * 60, f"{i:064d}") for i in range(20)}), "m")
    text = spec.message(repo, old, new, None)
    assert len(text) <= 400 and text.count("```") % 2 == 0 and text.endswith(f"… the rest is in {old[:9]}..{new[:9]}.")


def test_the_dm_opens_with_a_tally_and_a_cut_loses_new_guarantees_first(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    monkeypatch.setattr(spec, "LIMIT", 600)
    many = {f"P.n{i}": _g(f"P.n{i} : " + "x" * 60, f"{i:064d}") for i in range(20)}
    old = _commit(repo, "pkg/lean-audit.json", _doc({"P.t": _g("P.t : A", H1)}, {"P.D": H1}), "base")
    _source(repo, "pkg/P.lean")
    new = _commit(repo, "pkg/lean-audit.json", _doc({"P.t": _g("P.t : B", H2), **many}, {"P.D": H3}), "m")
    text = spec.message(repo, old, new, None)
    assert text.split("\n")[1] == "• `pkg`: guarantees: 1 changed, 20 new; definitions: 1 changed"
    assert text.index("changed guarantee `P.t`") < text.index("definition `P.D` (`P.Defs`) changed") < text.index("new guarantee")
    assert "P.n19" not in text and text.endswith(f"… the rest is in {old[:9]}..{new[:9]}.")


def test_alert_dms_daniel_and_sends_nothing_without_a_spec_change(tmp_path):
    repo = _repo(tmp_path)
    old = _commit(repo, "lean-audit.json", _doc({"P.t": _g("P.t : A", H1)}), "base")
    same = _commit(repo, "lean-audit.json", _doc({"Q.t": _g("Q.t : A", H1)}), "move")
    api = Api()
    assert spec.alert(repo, old, same, api=api, reg=REG, log=lambda *_: None) and api.calls == []
    new = _commit(repo, "lean-audit.json", _doc({"Q.t": _g("Q.t : B", H2)}), "Merge #7 (abc)")
    assert spec.alert(repo, same, new, api=api, reg=REG, log=lambda *_: None)
    assert [(c["method"], c["channel"]) for c in api.calls] == [("chat.postMessage", "U0DANIEL")]
    assert "changed guarantee `Q.t`" in api.calls[0]["text"]


def test_a_failed_dm_is_printed_and_posted_to_agent_alerts(tmp_path, capsys):
    repo = _repo(tmp_path)
    old = _commit(repo, "lean-audit.json", _doc({"P.t": _g("P.t : A", H1)}), "base")
    new = _commit(repo, "lean-audit.json", _doc({"P.t": _g("P.t : B", H2)}), "m")
    api = Api(fail="U0DANIEL")
    assert not spec.alert(repo, old, new, api=api, reg=REG, log=lambda *_: None)
    assert [c["channel"] for c in api.calls] == ["U0DANIEL", "C0ALERTS"]
    assert "the DM to Daniel failed: chat.postMessage: channel_not_found" in api.calls[1]["text"]
    assert "the DM to Daniel failed" in capsys.readouterr().err


def test_any_failure_after_the_push_is_reported_and_never_raised(tmp_path, monkeypatch, capsys):
    repo = _repo(tmp_path)
    old = _commit(repo, "lean-audit.json", _doc({"P.t": _g("P.t : A", H1)}), "base")
    new = _commit(repo, "lean-audit.json", _doc({"P.t": _g("P.t : B", H2)}), "m")
    api = Api(fail="U0DANIEL", error=RuntimeError("socket closed"))
    assert not spec.alert(repo, old, new, api=api, reg=REG, log=lambda *_: None)
    assert [c["channel"] for c in api.calls] == ["U0DANIEL", "C0ALERTS"]
    assert "the DM to Daniel failed: RuntimeError: socket closed" in api.calls[1]["text"]
    api = Api(fail="U0DANIEL")
    assert not spec.alert(repo, old, new, api=api, reg={"humans": REG["humans"]}, log=lambda *_: None)
    assert "posting that to #agent-alerts failed too: KeyError" in capsys.readouterr().err
    monkeypatch.setattr(spec, "message", lambda *a: {}["pins"])
    api = Api()
    assert spec.alert(repo, old, new, api=api, reg=REG, log=lambda *_: None)
    assert "reading its `lean-audit.json` records failed: KeyError: 'pins'" in api.calls[0]["text"]


def test_a_push_to_main_alerts_on_what_it_moved_main_across(tmp_path, monkeypatch):
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(remote)], check=True)
    repo = _repo(tmp_path)
    subprocess.run(["git", "-C", str(repo), "remote", "set-url", "origin", str(remote)], check=True)
    old = _commit(repo, "lean-audit.json", _doc({"P.t": _g("P.t : A", H1)}), "base")
    subprocess.run(["git", "-C", str(repo), "push", "-q", "origin", "main"], check=True)
    mid = _commit(repo, "lean-audit.json", _doc({"P.t": _g("P.t : B", H2)}), "Merge #8 (abc)")
    subprocess.run(["git", "-C", str(repo), "push", "-q", "origin", "main"], check=True)
    subprocess.run(["git", "-C", str(repo), "update-ref", "refs/remotes/origin/main", old], check=True)  # a stale fetch
    new = _commit(repo, "lean-audit.json", _doc({"P.t": _g("P.t : C", H3)}), "Merge #9 (abc)")
    seen = []
    monkeypatch.setattr(spec, "alert", lambda repo, was, now, **kw: seen.append((was, now)) or True)
    monkeypatch.setattr(merge, "_sweep_after", lambda *a, **k: None)
    monkeypatch.setattr(merge, "report_landings", lambda *a, **kw: True)     # it asks GitHub (`gh pr list`) and posts to Slack
    assert merge._push_and_sweep(repo, "main", "origin", True) == 0
    assert seen == [(mid, new)]


def test_the_queue_names_the_pull_requests_it_landed(tmp_path):
    repo = _repo(tmp_path)
    old = _commit(repo, "lean-audit.json", _doc({"P.t": _g("P.t : A", H1)}), "base")
    new = _commit(repo, "lean-audit.json", _doc({"P.t": _g("P.t : B", H2)}), "queue: #5 at abc")
    assert " from #5, #6.\n" in spec.message(repo, old, new, None, prs=[5, 6])
