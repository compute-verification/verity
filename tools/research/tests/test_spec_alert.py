"""`spec_alert`: the DM Daniel gets when a push lands a change to what a guarantee says, and nothing for a move or a reprint."""
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


def _doc(guarantees: dict, defs: dict | None = None, readers: tuple[str, ...] = ("P.t",)) -> dict:
    return {"guarantees": guarantees, "reads": {"P.Defs": {"digest": "x", "definitions": defs or {}, "guarantees": list(readers)}}}


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


def test_a_changed_definition_names_the_guarantees_that_read_it():
    readers = ("P.t", "P.u", "P.v", "P.w", "P.x", "P.y")
    old = _doc({"P.t": _g("P.t : D", H1)}, {"P.D": H1, "P.E": H2}, readers)
    new = _doc({"P.t": _g("P.t : D", H1)}, {"P.D": H3, "P.E": H2}, readers)
    assert spec.changes(old, new) == ["• definition `P.D` changed; read by `P.t`, `P.u`, `P.v`, `P.w` and 2 more"]
    moved = {"reads": {"P.Other": {"definitions": {"P.D": H1, "P.E": H2}, "guarantees": ["P.t"]}}, "guarantees": old["guarantees"]}
    assert spec.changes(old, moved) == []                                   # same names and hashes in another module


def test_an_assumption_no_reads_record_holds_is_shown_when_its_digest_changes_or_is_first_recorded():
    """LEAN-60: a premise outside `meaning` has only its own record's digest, so #4641's strengthened
    `RecDeploy.PlainLeavesAdmitted` reached no DM. A changed digest is shown (tallied as rehashed when no Lean input
    changed), and so is one first recorded; one a `reads` record holds shows as its definition, and a new entry or an
    unchanged one shows nothing."""
    def doc(assumed: dict, defs: dict | None = None) -> dict:
        return _doc({"P.t": _g("P.t : A", H1)}, defs) | {"assumptions": assumed}
    old = doc({"P.Leaves": {"owner": "@proofs", "digest": H1}, "P.Bare": {"owner": "@proofs"}, "P.Hyp": {"digest": H1}},
              {"P.Hyp": H1})
    new = doc({"P.Leaves": {"owner": "@proofs", "digest": H2}, "P.Bare": {"owner": "@proofs", "digest": H3},
               "P.Hyp": {"digest": H2}, "P.New": {"digest": H3}}, {"P.Hyp": H2})
    assert spec.changes(old, new) == [
        "• definition `P.Hyp` changed; read by `P.t`",
        "• assumption `P.Bare` recorded: its statement and the definitions it reads are now digested in its record",
        "• assumption `P.Leaves` changed: its statement or a definition it reads"]
    assert spec.changes(new, new) == []
    entries = spec._across({"lean-audit.json": old}, {"lean-audit.json": new}, sources=False)
    assert [k for _, k, _ in entries] == ["rehashed", "recorded", "rehashed"]
    assert spec._tally([k for _, k, _ in spec._across({"a": old}, {"a": new})]) == \
        "definitions: 1 changed; assumptions: 1 changed, 1 recorded"


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
    """`verifier.allow` (LEAN-33) lists the `@[extern]` and `@[implemented_by]` constants the verifier may reach, which no
    proof sees: an entry added is shown; one dropped, or the list unchanged, is not."""
    base = _doc({"P.t": _g("P.t : A", H1)}) | {"verifier": {"entries": ["P.verify"], "allow": ["Nat.add"]}}
    more = base | {"verifier": {"entries": ["P.verify"], "allow": ["ByteArray.push", "Nat.add"]}}
    assert spec.changes(base, more) == ["• compiled code allowed on the verifier's path, which no proof sees: `ByteArray.push`"]
    assert spec.changes(more, base) == [] and spec.changes(more, more) == []
    assert spec._tally(["compiled", "compiled", "new"]) == "guarantees: 1 new; verifier: 2 allowed"


def test_a_record_from_before_the_rename_compares_by_what_it_says():
    new = _doc({"P.t": _g("P.t : D", H1)}, {"P.D": H1}, ("P.t",))
    old = {"pins": new["guarantees"], "reads": {"P.Defs": {"digest": "x", "definitions": {"P.D": H1}, "pins": ["P.t"]}}}
    assert spec.changes(old, new) == []
    assert spec.changes(old, _doc({"P.t": _g("P.t : E", H2)}, {"P.D": H3})) == [
        "• changed guarantee `P.t`, before:\n```\nP.t : D\n```\nafter:\n```\nP.t : E\n```",
        "• definition `P.D` changed; read by `P.t`"]


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
    """The audit regrouping its reads records (#1544) changed 116 definitions' hashes with no Lean source behind them, and
    the DM would have listed every one; a Lean input changed in the push, or one outside the audit's own directory, lists
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
    assert "• `pkg`: guarantees: 1 new; definitions: 1 rehashed" in text and "P.D" not in text
    prev, h = regrouped, H3
    for source in ("pkg/P.lean", "pkg/lean-toolchain", "other/lake-manifest.json", "other/lakefile.toml"):
        _source(repo, source)
        h = H1 if h == H3 else H3
        cur = _commit(repo, "pkg/lean-audit.json", _doc({"P.t": _g("P.t : A", H1)}, {"P.D": h, "P.E": H2}), "m")
        assert spec.message(repo, prev, cur).endswith("*pkg*\n• definition `P.D` changed; read by `P.t`"), source
        prev = cur


def test_a_newly_pinned_dependency_bundle_is_a_lean_input_and_a_rekeyed_or_dropped_one_is_not(tmp_path):
    """Every definition that reads Mathlib or ArkLib is built against the bundle `lean-deps.json` pins (top, 06:06Z): pinning
    another one with no manifest change would otherwise show only as `rehashed`. #1544 re-keys two bundles and drops a pin."""
    repo = _repo(tmp_path)
    pins = lambda **deps: {"deps": {k: {"package": "pkg", "sha256": s} for k, s in deps.items()}}
    _commit(repo, spec.DEPS_PINS, pins(k1=H1, k2=H2), "pins")
    old = _commit(repo, "pkg/lean-audit.json", _doc({"P.t": _g("P.t : A", H1)}, {"P.D": H1}), "base")
    _commit(repo, spec.DEPS_PINS, pins(k3=H1), "re-key one bundle, drop the other")
    rekeyed = _commit(repo, "pkg/lean-audit.json", _doc({"P.t": _g("P.t : A", H1)}, {"P.D": H3}), "m")
    assert spec.message(repo, old, rekeyed) is None
    _commit(repo, spec.DEPS_PINS, pins(k3=H1, k4=H3), "a new bundle")
    pinned = _commit(repo, "pkg/lean-audit.json", _doc({"P.t": _g("P.t : A", H1)}, {"P.D": H2}), "m")
    assert spec.message(repo, rekeyed, pinned).endswith("*pkg*\n• definition `P.D` changed; read by `P.t`")


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


def test_records_that_moved_between_lock_files_are_matched_by_name_and_tallied_as_moves(tmp_path):
    """The Lean move merged six packages' records into one: matched file by file, the DM listed 990 guarantees removed from
    the six and 990 new in the one, where all but a handful kept their names and type hashes."""
    repo = _repo(tmp_path)
    _commit(repo, "a/lean-audit.json", _doc({"P.t": _g("P.t : A", H1), "P.u": _g("P.u : B", H2)}, {"P.D": H1}), "a")
    old = _commit(repo, "b/lean-audit.json", _doc({"Q.v": _g("Q.v : C", H3)}, {"Q.E": H2}, ("Q.v",)), "b")
    subprocess.run(["git", "-C", str(repo), "rm", "-q", "a/lean-audit.json", "b/lean-audit.json"], check=True)
    both = _doc({"P.t": _g("P.t : A", H1), "P.u": _g("P.u : B", H2), "Q.v": _g("Q.v : C", H3)})
    both["reads"] = {"P.Defs": {"definitions": {"P.D": H1, "Q.E": H2}, "guarantees": ["P.t", "Q.v"]}}
    moved = _commit(repo, "c/lean-audit.json", both, "Layout move (#9)")
    assert spec.message(repo, old, moved) is None
    both["guarantees"]["Q.v"] = _g("Q.v : D", H1)
    both["reads"]["P.Defs"]["definitions"]["Q.E"] = H3
    _source(repo, "c/Q.lean")
    edited = _commit(repo, "c/lean-audit.json", both, "m")
    text = spec.message(repo, old, edited)
    assert "• `c`: guarantees: 1 changed, 2 moved here; definitions: 1 changed" in text
    assert "removed" not in text and "new guarantee" not in text and "P.D" not in text
    assert "*c*\n• changed guarantee `Q.v`" in text and "• definition `Q.E` changed; read by `P.t`, `Q.v`" in text


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
    assert text.index("changed guarantee `P.t`") < text.index("definition `P.D` changed") < text.index("new guarantee")
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
    monkeypatch.setattr(merge, "_sweep_after", lambda *a: None)
    monkeypatch.setattr(merge, "report_landings", lambda *a, **kw: True)     # it asks GitHub (`gh pr list`) and posts to Slack
    assert merge._push_and_sweep(repo, "main", "origin", True) == 0
    assert seen == [(mid, new)]


def test_the_queue_names_the_pull_requests_it_landed(tmp_path):
    repo = _repo(tmp_path)
    old = _commit(repo, "lean-audit.json", _doc({"P.t": _g("P.t : A", H1)}), "base")
    new = _commit(repo, "lean-audit.json", _doc({"P.t": _g("P.t : B", H2)}), "queue: #5 at abc")
    assert " from #5, #6.\n" in spec.message(repo, old, new, None, prs=[5, 6])
