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


def _doc(pins: dict, defs: dict | None = None, readers: tuple[str, ...] = ("P.t",)) -> dict:
    return {"pins": pins, "reads": {"P.Defs": {"digest": "x", "definitions": defs or {}, "pins": list(readers)}}}


class Api:
    def __init__(self, fail: str | None = None):
        self.calls: list[dict] = []
        self.fail = fail

    def call(self, method: str, **params):
        self.calls.append({"method": method, **params})
        if self.fail and params.get("channel") == self.fail:
            raise SlackError(f"{method}: channel_not_found")
        return {"ok": True}


def test_a_restated_guarantee_shows_its_statement_before_and_after():
    lines = spec.changes(_doc({"P.t": _g("P.t : 1 = 1", H1)}), _doc({"P.t": _g("P.t : 2 = 2", H2)}))
    assert len(lines) == 1 and lines[0].startswith("• changed guarantee `P.t`, before:")
    assert "P.t : 1 = 1" in lines[0] and lines[0].index("P.t : 1 = 1") < lines[0].index("after:") < lines[0].index("P.t : 2 = 2")


def test_new_and_removed_guarantees_are_listed_and_a_move_is_not():
    old = _doc({"P.t": _g("P.t : A", H1), "P.gone": _g("P.gone : B", H2)})
    new = _doc({"Q.t": _g("Q.t : A", H1), "P.new": _g("P.new : C", H3)})
    lines = spec.changes(old, new)
    assert lines[0].startswith("• new guarantee `P.new`") and "P.new : C" in lines[0]
    assert lines[1:] == ["• removed guarantee `P.gone`"]                    # P.t -> Q.t has the same type hash: a move


def test_a_reprint_and_a_rehash_send_nothing():
    assert spec.changes(_doc({"P.t": _g("P.t : A", H1)}), _doc({"P.t": _g("P.t :  A", H1)})) == []
    assert spec.changes(_doc({"P.t": _g("P.t : A", "1234")}), _doc({"P.t": _g("P.t : A", H1)})) == []
    assert spec.changes(_doc({"P.t": _g("P.t : A", "1234")}), _doc({"P.t": _g("P.t : B", H1)})) != []


def test_a_changed_definition_names_the_guarantees_that_read_it():
    readers = ("P.t", "P.u", "P.v", "P.w", "P.x", "P.y")
    old = _doc({"P.t": _g("P.t : D", H1)}, {"P.D": H1, "P.E": H2}, readers)
    new = _doc({"P.t": _g("P.t : D", H1)}, {"P.D": H3, "P.E": H2}, readers)
    assert spec.changes(old, new) == ["• definition `P.D` changed; read by `P.t`, `P.u`, `P.v`, `P.w` and 2 more"]
    moved = {"reads": {"P.Other": {"definitions": {"P.D": H1, "P.E": H2}, "pins": ["P.t"]}}, "pins": old["pins"]}
    assert spec.changes(old, moved) == []                                   # same names and hashes in another module


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


def test_the_dm_names_the_package_the_pull_requests_and_the_compare_link(tmp_path):
    repo = _repo(tmp_path)
    old = _commit(repo, "pkg/lean/lean-audit.json", _doc({"P.t": _g("P.t : A", H1)}), "base")
    new = _commit(repo, "pkg/lean/lean-audit.json", _doc({"P.t": _g("P.t : B", H2)}), "Merge #12 #34 (abc)")
    text = spec.message(repo, old, new, spec.web(repo, "origin"))
    assert text.startswith(f"A spec change landed on main (<https://github.com/owner/verity/compare/{old[:12]}...{new[:12]}|")
    assert "<https://github.com/owner/verity/pull/12|#12>, <https://github.com/owner/verity/pull/34|#34>" in text
    assert "*pkg/lean*\n• changed guarantee `P.t`" in text


def test_a_long_dm_ends_with_the_compare_link_and_closes_its_code_block(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    monkeypatch.setattr(spec, "LIMIT", 400)
    old = _commit(repo, "lean-audit.json", _doc({}), "base")
    new = _commit(repo, "lean-audit.json", _doc({f"P.t{i}": _g(f"P.t{i} : " + "x" * 60, f"{i:064d}") for i in range(20)}), "m")
    text = spec.message(repo, old, new, None)
    assert len(text) < 500 and text.count("```") % 2 == 0 and text.endswith(f"… the rest is in {old[:9]}..{new[:9]}.")


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


def test_a_push_to_main_alerts_on_what_it_moved_main_across(tmp_path, monkeypatch):
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(remote)], check=True)
    repo = _repo(tmp_path)
    subprocess.run(["git", "-C", str(repo), "remote", "set-url", "origin", str(remote)], check=True)
    old = _commit(repo, "lean-audit.json", _doc({"P.t": _g("P.t : A", H1)}), "base")
    subprocess.run(["git", "-C", str(repo), "push", "-q", "origin", "main"], check=True)
    new = _commit(repo, "lean-audit.json", _doc({"P.t": _g("P.t : B", H2)}), "Merge #9 (abc)")
    seen = []
    monkeypatch.setattr(spec, "alert", lambda repo, was, now, **kw: seen.append((was, now)) or True)
    monkeypatch.setattr(merge, "_sweep_after", lambda *a: None)
    assert merge._push_and_sweep(repo, "main", "origin", True) == 0
    assert seen == [(old, new)]
