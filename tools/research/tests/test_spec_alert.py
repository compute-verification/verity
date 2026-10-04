"""`research merge --push` DMs Daniel each guarantee change it lands (Daniel, 3 Oct): a `lean-audit.json` whose `pins` or `reads`
differ between main before and after the push. A change an `audit.py --update` report printed as the same statements (only moved)
sends nothing, and a DM that fails is printed and posted to #agent-alerts. GitHub and Slack are fakes; nothing leaves tmp."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import research.merge as M
from research import spec_alert as S

AUDIT = "pkg/lean/lean-audit.json"


def _git(repo: Path, *args: str) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    return subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=repo, check=True, capture_output=True, text=True,
                          env=env).stdout.strip()


def _commit(repo: Path, files: dict[str, str], msg: str) -> str:
    for p, text in files.items():
        (repo / p).parent.mkdir(parents=True, exist_ok=True)
        (repo / p).write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


def _record(pins: dict, reads: dict | None = None) -> str:
    return json.dumps({"pins": pins, "reads": reads or {}, "allowed_axioms": []}, indent=1)


def _repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "r"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    return repo, _commit(repo, {AUDIT: _record({"Pkg.G": {"type_hash": "a"}}), "x.txt": "x\n"}, "init")


def _gh(prs_by_commit: dict[str, list[int]]):
    calls = []

    def gh(*args: str) -> str:
        calls.append(args)
        sha = args[1].split("/commits/")[1].split("/")[0]
        return json.dumps([{"number": n, "html_url": f"https://github.com/o/r/pull/{n}", "head": {"sha": sha}}
                           for n in prs_by_commit.get(sha, [])])
    gh.calls = calls
    return gh


def test_a_changed_pin_sends_daniel_the_names_and_the_pull_request(tmp_path):
    repo, old = _repo(tmp_path)
    c = _commit(repo, {AUDIT: _record({"Pkg.G": {"type_hash": "b"}, "Pkg.H": {"type_hash": "c"}})}, "change G, add H")
    sent = []
    assert S.alert(repo, old, c, gh=_gh({c: [41]}), reviews=lambda pr, head: [], send=sent.append,
                   fallback=lambda t: None, log=lambda s: None) == 0
    assert len(sent) == 1
    text = sent[0]
    assert AUDIT in text and "#41" in text and "https://github.com/o/r/pull/41" in text
    assert "added `Pkg.H`" in text and "changed `Pkg.G`" in text
    assert old[:12] in text and c[:12] in text


def test_no_guarantee_change_sends_nothing_and_opens_no_store(tmp_path):
    repo, old = _repo(tmp_path)
    c = _commit(repo, {"x.txt": "y\n", AUDIT: _record({"Pkg.G": {"type_hash": "a"}}).replace('"allowed_axioms": []',
                                                                                               '"allowed_axioms": ["propext"]')},
                "an audit change outside pins and reads")
    sent = []
    assert S.alert(repo, old, c, gh=_gh({}), send=sent.append, fallback=sent.append) == 0
    assert sent == []


def test_a_report_that_printed_only_a_move_sends_nothing(tmp_path):
    repo, old = _repo(tmp_path)
    new_pins = {"Pkg.Moved.G": {"type_hash": "a"}}
    c = _commit(repo, {AUDIT: _record(new_pins)}, "move G")
    report = {"package": "pkg/lean", "statement_changes": False, "review": ["moved: Pkg.G -> Pkg.Moved.G (same statement)"],
              "before": {"pins": {"Pkg.G": {"type_hash": "a"}}, "reads": {}}, "pins": new_pins, "reads": {}}
    sent = []
    S.alert(repo, old, c, gh=_gh({c: [7]}), reviews=lambda pr, head: [report] if pr == 7 else [], send=sent.append,
            fallback=sent.append, log=lambda s: None)
    assert sent == []


def test_a_report_with_statement_changes_is_quoted(tmp_path):
    repo, old = _repo(tmp_path)
    new_pins = {"Pkg.G": {"type_hash": "z"}}
    c = _commit(repo, {AUDIT: _record(new_pins)}, "restate G")
    report = {"package": "pkg/lean", "statement_changes": True, "review": ["Pkg.G: before `P`, now `Q`"],
              "before": {"pins": {"Pkg.G": {"type_hash": "a"}}, "reads": {}}, "pins": new_pins, "reads": {}}
    sent = []
    S.alert(repo, old, c, gh=_gh({c: [8]}), reviews=lambda pr, head: [report], send=sent.append, fallback=sent.append,
            log=lambda s: None)
    assert len(sent) == 1 and "> Pkg.G: before `P`, now `Q`" in sent[0]


def test_a_failed_dm_is_printed_and_posted_to_agent_alerts(tmp_path, capsys):
    repo, old = _repo(tmp_path)
    c = _commit(repo, {AUDIT: _record({"Pkg.G": {"type_hash": "b"}})}, "change G")
    alerts = []

    def refuse(text):
        raise RuntimeError("chat.postMessage: channel_not_found")
    assert S.alert(repo, old, c, gh=_gh({c: [9]}), reviews=lambda pr, head: [], send=refuse, fallback=alerts.append,
                   log=lambda s: None) == 1
    err = capsys.readouterr().err
    assert "GUARANTEE CHANGE NOT SENT TO DANIEL" in err and "channel_not_found" in err and AUDIT in err
    assert len(alerts) == 1 and "couldn't DM Daniel" in alerts[0] and AUDIT in alerts[0]


def test_push_alerts_on_the_change_between_the_remote_before_and_after(tmp_path, monkeypatch):
    repo, old = _repo(tmp_path)
    origin = tmp_path / "origin.git"
    _git(tmp_path, "init", "-q", "--bare", "-b", "main", str(origin))
    _git(repo, "remote", "add", "origin", str(origin))
    _git(repo, "push", "-q", "origin", "main")
    c = _commit(repo, {AUDIT: _record({"Pkg.G": {"type_hash": "b"}})}, "change G")
    seen = []
    monkeypatch.setattr(S, "alert", lambda repo, was, now, **kw: seen.append((was, now)) or 0)
    monkeypatch.setattr(M, "_sweep_after", lambda *a: None)
    assert M._push_and_sweep(repo, "main", "origin", True) == 0
    assert seen == [(old, c)]
    assert _git(repo, "ls-remote", "origin", "refs/heads/main").split()[0] == c
