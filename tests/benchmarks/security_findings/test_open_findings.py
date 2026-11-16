"""The count of the security review's open findings (`open_findings.py`): only the check the script holds for a finding's id
closes it; a finding with no check is open whatever the list holds, and the list holds no reading; the mark is `measured`
only when every finding is decided by a check of the tree; tonight's list closes only through its checks and its exceptions
are pinned; the guard check takes only commands that run the guard from `/opt/agent-guard/`; the address check counts every
tracked file the list doesn't except; a recorded reading decides only with its check's command, a reader and a time within
seven days; the agent-env, store, root and updates checks decide as their findings say, the machine checks reading this
machine itself and the others from the reading; a check that can't read its thing leaves the finding unread, which `main`
counts not closed and names apart; and the list and its data name no address, key or auditor's machine.

No test reads the clock or the network: the clock, this machine's reading and the store are stand-ins, the checks run on toy
checkouts under `tmp_path`, and the script is imported with the repository's root on `sys.path` only through `monkeypatch`."""
from __future__ import annotations

import importlib
import json
import re
import subprocess
import sys
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FINDING = {"id": "x", "statement": "s", "would_close": "a check that would"}
CHECKED = {"agent-holds-personal-and-cloud-keys": ("agent-env-names", "recorded"),
           "guard-runs-from-the-branch": ("guard-outside-checkout", "declared"),
           "agents-have-root": ("sudo-refused", "machine"),
           "security-updates-and-kernel": ("updates-and-kernel", "machine"),
           "store-key-writes": ("store-write-refused", "recorded"),
           "machine-addresses-in-the-tree": ("no-public-address", "tree")}
PUBLIC = ".".join(["81", "2", "3", "4"])


@pytest.fixture
def O(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(ROOT))
    return importlib.import_module("benchmarks.security_findings.open_findings")


def stand_in(O: ModuleType, closed: bool | None) -> dict:
    return {fid: O.Check(c.name, c.reads, lambda root, f, closed=closed: (closed, "stand-in")) for fid, c in O.CHECKS.items()}


def test_the_suite_leaves_sys_path_alone() -> None:
    assert str(ROOT) not in sys.path


def test_a_finding_without_a_check_is_open_whatever_its_entry_holds(O: ModuleType) -> None:
    why = "no check; what would close it: a check that would"
    assert O.status(FINDING, checks={}) == O.Status("x", True, "none", why)
    reading = {"command": "true", "reader": "infra", "showed": "ok", "closed": True}
    assert O.status({**FINDING, "reading": reading}, checks={}).open
    named = O.status({**FINDING, "check": "no-public-address", "reads": "tree"}, checks=stand_in(O, True))
    assert (named.open, named.basis) == (True, "refused")


def test_the_list_takes_no_reading_and_says_what_would_close_an_unchecked_finding(O: ModuleType, tmp_path: Path) -> None:
    p = tmp_path / "findings.toml"
    for body, why in [('would_close = "w"\n[finding.reading]\ncommand = "true"\nclosed = true\n', "holds reading"),
                      ("", "would close it"), ('would_close = """two\nlines"""\n', "would close it"),
                      ('would_close = "w"\ncheck = "c"\nreads = "tree"\nhow = "h"\n', "would close it"),
                      ('check = "c"\nreads = "tree"\n', "would close it")]:
        p.write_text(f'[[finding]]\nid = "a"\nstatement = "s"\n{body}', encoding="utf-8")
        with pytest.raises(ValueError, match=why):
            O.load(p)
    p.write_text('[[finding]]\nid = "a"\nstatement = "s"\nwould_close = "w"\n' * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="share an id"):
        O.load(p)


def test_each_check_is_the_scripts_for_its_finding_and_the_list_cant_move_it(O: ModuleType) -> None:
    assert {fid: (c.name, c.reads) for fid, c in O.CHECKS.items()} == CHECKED
    f = {"id": "machine-addresses-in-the-tree", "statement": "s", "check": "no-public-address", "reads": "tree", "how": "h"}
    assert O.status(f, checks=stand_in(O, True)) == O.Status(f["id"], False, "tree", "stand-in")
    assert O.status(f, checks=stand_in(O, False)) == O.Status(f["id"], True, "tree", "stand-in")
    for other in ({**f, "reads": "declared"}, {**f, "check": "guard-outside-checkout"}, {**f, "id": "agents-have-root"}):
        assert O.status(other, checks=stand_in(O, True)).open, other


def test_a_check_that_cannot_run_counts_open(O: ModuleType) -> None:
    def broken(root: Path, f: dict) -> tuple[bool, str]:
        raise OSError("no checkout")
    got = O.status({**FINDING, "check": "c", "reads": "tree"}, checks={"x": O.Check("c", "tree", broken)})
    assert (got.open, got.basis) == (True, "refused")


def test_the_mark_is_measured_only_when_every_finding_rests_on_a_check_of_the_tree(O: ModuleType) -> None:
    def s(basis: str, open_: bool = False):
        return O.Status("x", open_, basis, "")
    assert O.mark([s("tree"), s("tree", True)]) == "measured"
    for other in ("declared", "machine", "recorded", "unread", "none", "refused"):
        assert O.mark([s("tree"), s(other)]) == "partial", other
    assert O.mark([]) == "partial"


def test_tonights_list_closes_only_through_its_checks(O: ModuleType) -> None:
    """Every finding of the list has its check, and only the checks decide: 6 open when they find their findings open, 0 when
    they find them closed, and 6 not closed, each unread, when none can read its thing where it runs."""
    findings = O.load()
    assert [f["id"] for f in findings] == ["agent-holds-personal-and-cloud-keys", "guard-runs-from-the-branch",
                                           "agents-have-root", "security-updates-and-kernel", "store-key-writes",
                                           "machine-addresses-in-the-tree"]
    open_ = [O.status(f, checks=stand_in(O, False)) for f in findings]
    assert sum(s.open for s in open_) == 6 and O.mark(open_) == "partial"
    closed = [O.status(f, checks=stand_in(O, True)) for f in findings]
    assert [s.id for s in closed if s.open] == [] and O.mark(closed) == "partial"
    unread = [O.status(f, checks=stand_in(O, None)) for f in findings]
    assert [(s.open, s.basis) for s in unread] == [(True, "unread")] * 6


def test_the_address_exceptions_are_these_two(O: ModuleType) -> None:
    """Each exception drops a file's addresses from finding 6's count, so adding one changes this test in the same change."""
    six = next(f for f in O.load() if f["id"] == "machine-addresses-in-the-tree")
    allowed = O.excepted(six)
    assert (len(allowed), sum(allowed.values())) == (2, 2)
    assert {e["kind"] for e in six["except"]} == set(O.EXCEPT_KINDS)


def _checkout(tmp_path: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    _commit(tmp_path)
    return tmp_path


def _commit(root: Path) -> None:
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-qm", "c", "--allow-empty"],
                   cwd=root, check=True)


def test_the_address_check_counts_every_tracked_file_the_list_doesnt_except(O: ModuleType, tmp_path: Path) -> None:
    root = _checkout(tmp_path, {"a.sh": "ssh research@192.0.2.7\n", "b/c.toml": 'v = "12.9.1.4"\n'})
    none = {"id": "machine-addresses-in-the-tree"}
    assert O.no_public_address(root, none) == (True, "0 tracked files hold a public IPv4 address the list doesn't except")
    for rel in ("tests/d.py", "e/tests/f.txt", "g.md"):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(f"host {PUBLIC} and {PUBLIC}\n", encoding="utf-8")
    _commit(root)
    assert O.no_public_address(root, none)[0] is False and O.no_public_address(root, none)[1].startswith("3 ")

    def entry(path: str, count: int, kind: str = "version") -> dict:
        return {"path": path, "count": count, "kind": kind, "reason": "r"}
    some = {**none, "except": [entry("tests/d.py", 2), entry("e/tests/f.txt", 1)]}
    assert O.no_public_address(root, some)[1].startswith("2 ")
    every = {**none, "except": [entry("tests/d.py", 2), entry("e/tests/f.txt", 2), entry("g.md", 2)]}
    assert O.no_public_address(root, every)[0] is True
    for bad in (entry("g.md", 2, "ours"), entry("g.md", 2, "example"), entry("g.md", 0),
                {"path": "g.md", "count": 2, "kind": "version"}):
        with pytest.raises(ValueError):
            O.no_public_address(root, {**none, "except": [bad]})


MAIN_LIKE = ("sh -c 'h=tools/agent-guard/hook.sh; [ -f \"$h\" ] && exec sh \"$h\" \"$1\"; d=$(ls -td "
             "\"${AGENT_GUARD_CACHE:-${HOME:-/tmp}/.cache/agent-guard}\"/[0-9a-f]*/ 2>/dev/null | head -1); "
             "[ -f \"$d$1\" ] || d=/opt/agent-guard/; exec python3 \"$d$1\"' guard shell_hook.py")


@pytest.mark.parametrize(("command", "outside"), [
    ("/opt/agent-guard/shell_hook.py", True),
    ("/opt/agent-guard/hook guard shell_hook.py --timeout=10", True),
    (MAIN_LIKE, False),
    ("python3 /opt/agent-guard/shell_hook.py", False),
    ('sh -c \'exec python3 "$PWD"/tools/agent-guard/shell_hook.py\'', False),
    ('python3 "$PWD"/tools/agent-guard/shell_hook.py', False),
    ('/opt/agent-guard/hook "$PWD"/tools/agent-guard/shell_hook.py', False),
    ('/opt/agent-guard/hook "${CURSOR_PROJECT_DIR}"/tools/agent-guard/shell_hook.py', False),
    ("/opt/agent-guard/hook $(git rev-parse --show-toplevel)", False),
    ("git show HEAD:tools/agent-guard/shell_hook.py | python3 -", False),
    ("python3 -m research.addresses", False),
    (".venv/bin/python tools/agent-guard/shell_hook.py", False),
    ("./tools/agent-guard/hook.sh shell_hook.py", False),
    ("/opt/agent-guard/../../agent/repos/verity/tools/agent-guard/hook.sh", False),
    ("/opt/agent-guard/hook /agent/repos/verity/tools/agent-guard/shell_hook.py", False),
    ("/opt/agent-guard/hook ..", False),
    ("/opt/agent-guard/hook .", False),
    ("/opt/agent-guard/hook --dir=.", False),
    ("/opt/agent-guard/hook tools", False),
    ("/opt/agent-guard/hook --root=tools", False),
    ("/opt/agent-guard/hook; sh tools/agent-guard/hook.sh", False),
    ("/opt/agent-guard/hook && sh x", False),
    ("/opt/agent-guard/hook < tools/x", False),
    ("/opt/agent-guard/hook 'a b'", False),
    ('/opt/agent-guard/hook "unterminated', False),
    ("/opt/agent-guard/sub/hook", False),
    ("/opt/tools/agent-guard/shell_hook.py", False),
    ("", False),
])
def test_only_a_command_that_runs_the_guard_from_opt_alone_is_outside(O: ModuleType, command: str, outside: bool) -> None:
    assert O.outside_checkout(command, {"tools", ".cursor", ".venv", "tests"}) is outside


def test_the_one_wrapper_is_outside_only_word_for_word_with_a_hook_file(O: ModuleType) -> None:
    """The wrapper (`GUARD_WRAPPER`) counts as outside only as `/bin/sh -c WRAPPER guard NAME` with its exact text and a hook
    file NAME: a changed character, a shell found on PATH, a wrapper that keeps the caller's PATH, another argument or a name of
    the checkout reopens the finding until the check vouches for it."""
    names = {"tools", ".cursor", ".venv", "tests", "x.py"}
    w = O.GUARD_WRAPPER
    assert O.outside_checkout(f"/bin/sh -c '{w}' guard shell_hook.py", names) is True
    assert O.outside_checkout(f"/bin/sh -c '{w}' guard task_hook.py", names) is True
    for command in (f"/bin/sh -c '{w.replace('/opt/agent-guard', '/opt/agent-guard/../x')}' guard shell_hook.py",
                    f"/bin/sh -c '{w} ' guard shell_hook.py", f"/bin/sh -c '{w}' guard shell_hook.py extra",
                    f"/bin/sh -c '{w}' other shell_hook.py", f"/bin/sh -c '{w}' guard tools",
                    f"/bin/sh -c '{w}' guard x.py",
                    f"/bin/sh -c '{w}' guard ../tools/agent-guard/shell_hook.py", f"bash -c '{w}' guard shell_hook.py",
                    f"/bin/sh -c '{w}'", f"sh -c '{w}' guard shell_hook.py",
                    f"/bin/sh -c '{w.replace('PATH=/usr/local/bin:/usr/bin:/bin; export PATH; ', '')}' guard shell_hook.py"):
        assert O.outside_checkout(command, names) is False, command


def _hooks(*commands: str) -> str:
    return json.dumps({"version": 1, "hooks": {"stop": [{"command": c} for c in commands]}})


def test_the_guard_check_reads_every_hook_the_commit_declares(O: ModuleType, tmp_path: Path) -> None:
    root = _checkout(tmp_path, {".cursor/hooks.json": _hooks("/opt/agent-guard/stop_hook.py"),
                                "tools/agent-guard/hook.sh": "#!/bin/sh\n"})
    tail = "hook commands of .cursor/hooks.json at the commit don't run the guard from /opt/agent-guard/ alone"
    assert O.guard_outside_checkout(root, {}) == (True, f"0 of 1 {tail}")
    (root / ".cursor/hooks.json").write_text(_hooks("/opt/agent-guard/stop_hook.py", MAIN_LIKE), encoding="utf-8")
    assert O.guard_outside_checkout(root, {}) == (True, f"0 of 1 {tail}"), "the working tree's edit is not the commit's"
    _commit(root)
    assert O.guard_outside_checkout(root, {}) == (False, f"1 of 2 {tail}")
    (root / ".cursor/hooks.json").write_text(_hooks(), encoding="utf-8")
    _commit(root)
    assert O.guard_outside_checkout(root, {})[0] is False


def test_a_commit_that_tracks_no_hooks_file_declares_no_cure(O: ModuleType, tmp_path: Path) -> None:
    """A VM's boot writes an untracked hooks file when the commit tracks none, and any branch can still bring its own: the
    tree then declares nothing about the guard, so the finding stays open."""
    root = _checkout(tmp_path, {"tools/x": "x\n"})
    (root / ".cursor").mkdir()
    (root / ".cursor/hooks.json").write_text(_hooks("/opt/agent-guard/stop_hook.py"), encoding="utf-8")
    closed, why = O.guard_outside_checkout(root, {})
    assert closed is False and why.startswith("the commit tracks no .cursor/hooks.json")
    got = O.status({"id": "guard-runs-from-the-branch", "statement": "s", "check": "guard-outside-checkout",
                    "reads": "declared", "how": "h"}, root)
    assert (got.open, got.basis) == (True, "declared")


def test_main_writes_the_count_of_open_findings(O: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                 capsys: pytest.CaptureFixture) -> None:
    """Every finding of the list counts, checked or not, and the value written is the count printed."""
    root = tmp_path / "tree"
    (root / "list").mkdir(parents=True)
    path = root / "list/findings.toml"
    path.write_text("".join(f'[[finding]]\nid = "{i}"\nstatement = "s"\nwould_close = "w"\n' for i in ("a", "b", "c")) +
                    '[[finding]]\nid = "shut"\nstatement = "s"\ncheck = "c"\nreads = "tree"\nhow = "h"\n'
                    '[[finding]]\nid = "ajar"\nstatement = "s"\ncheck = "d"\nreads = "declared"\nhow = "h"\n', encoding="utf-8")
    checks = {"shut": O.Check("c", "tree", lambda r, f: (True, "shut")), "ajar": O.Check("d", "declared", lambda r, f: (False, "ajar"))}
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(tmp_path / "run"))
    O.main(["--out", str(tmp_path / "out")], path=path, root=root, checks=checks)
    assert capsys.readouterr().out.splitlines()[0] == "security findings open: 4 of 5 (mark partial)"
    result = json.loads((tmp_path / "run/result.json").read_text(encoding="utf-8"))
    (m,) = result["measurements"]
    assert (m["name"], m["value"], m["unit"], m["target"], m["mark"]) == (O.METRIC, 4, "findings", 0, "partial")
    assert result["validation"]["status"] == "passed"


def test_main_on_tonights_list_writes_six(O: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(tmp_path))
    O.main([], checks=stand_in(O, False))
    (m,) = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))["measurements"]
    assert (m["value"], m["mark"]) == (6, "partial")


def test_main_names_an_unread_finding_apart_and_counts_it_not_closed(O: ModuleType, tmp_path: Path,
                                                                     monkeypatch: pytest.MonkeyPatch,
                                                                     capsys: pytest.CaptureFixture) -> None:
    path = tmp_path / "findings.toml"
    path.write_text('[[finding]]\nid = "far"\nstatement = "s"\ncheck = "c"\nreads = "recorded"\nhow = "h"\n'
                    '[[finding]]\nid = "near"\nstatement = "s"\ncheck = "d"\nreads = "tree"\nhow = "h"\n', encoding="utf-8")
    checks = {"far": O.Check("c", "recorded", lambda r, f: (None, "not here")), "near": O.Check("d", "tree", lambda r, f: (True, "ok"))}
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(tmp_path / "run"))
    O.main(["--out", str(tmp_path / "out")], path=path, root=tmp_path, checks=checks)
    assert capsys.readouterr().out.splitlines() == [
        "security findings open: 1 of 2 (mark partial), 1 of them unread: no check could read them where this ran",
        "  far: unread: not here", "  near: closed (tree): ok"]
    result = json.loads((tmp_path / "run/result.json").read_text(encoding="utf-8"))
    assert (result["measurements"][0]["value"], result["validation"]["status"]) == (1, "passed")


# --- the checks of things outside the tree: readings, machines, the store -------------------------------------------------

NOW = 1_800_000_000.0
DAY = 86400.0


def _at(t: float) -> str:
    return datetime.fromtimestamp(t, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _tree(tmp_path: Path, readings: str = "", nodes: str = "") -> Path:
    """A toy checkout: readings.toml, the deploy map's nodes and the real kernel table where the checks look for them."""
    here = tmp_path / "benchmarks/security_findings"
    here.mkdir(parents=True)
    (here / "readings.toml").write_text(readings, encoding="utf-8")
    (here / "kernels.toml").write_text((ROOT / "benchmarks/security_findings/kernels.toml").read_text(encoding="utf-8"), encoding="utf-8")
    deploy = tmp_path / "tools/research/src/research/pods/nebius/deploy.toml"
    deploy.parent.mkdir(parents=True)
    deploy.write_text(nodes or ('[nodes.vy-nebius-cpu-10]\n[nodes.vy-nebius-cpu-2]\n[nodes.vy-nebius-cpu-3]\nstaged = "not yet"\n'
                                '[nodes.vy-nebius-1]\n[nodes.vy-control-verity]\nadmin = true\n'), encoding="utf-8")
    return tmp_path


def _reading(name: str, command: str, at: float, body: str, reader: str = "infra") -> str:
    return f"[{name}]\ncommand = {json.dumps(command)}\nreader = {json.dumps(reader)}\nat = {_at(at)}\n{body}\n"


@pytest.fixture
def now(O: ModuleType, monkeypatch: pytest.MonkeyPatch) -> float:
    monkeypatch.setattr(O, "clock", lambda: NOW)
    return NOW


def test_a_reading_decides_only_with_the_checks_command_a_reader_and_a_time_within_seven_days(O: ModuleType, tmp_path: Path,
                                                                                               now: float) -> None:
    assert O.MAX_AGE_DAYS == 7
    root = _tree(tmp_path)
    entry, why = O.reading(root, "x", "cmd")
    assert entry is None and why == "no reading is recorded: run `cmd` and record what it prints as [x] in readings.toml"
    readings = root / "benchmarks/security_findings/readings.toml"
    readings.write_text(_reading("x", "cmd", now - 6.5 * DAY, "seen = 1"), encoding="utf-8")
    assert O.reading(root, "x", "cmd") == ({"command": "cmd", "reader": "infra", "at": O.datetime.fromtimestamp(now - 6.5 * DAY, O.UTC),
                                            "seen": 1}, "read by infra 6.5 days ago")
    readings.write_text(_reading("x", "cmd", now - 7.5 * DAY, "seen = 1"), encoding="utf-8")
    assert O.reading(root, "x", "cmd") == (None, "the reading [x] is 8 days old, over 7: run `cmd` again")
    for text, why in [(_reading("x", "other", now, ""), "not `cmd`"), (_reading("x", "cmd", now + DAY, ""), "after now"),
                      (_reading("x", "cmd", now, "", reader=" "), "who read it"),
                      ('[x]\ncommand = "cmd"\nreader = "infra"\nat = 2026-11-15T18:00:00\n', "with its zone")]:
        readings.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match=why):
            O.reading(root, "x", "cmd")


def test_the_agent_env_check_closes_only_on_a_fresh_reading_that_names_none_of_the_keys(O: ModuleType, tmp_path: Path,
                                                                                        now: float) -> None:
    root = _tree(tmp_path)
    readings = root / "benchmarks/security_findings/readings.toml"
    closed, why = O.agent_env_names(root, {})
    assert closed is None and why.startswith("an agent VM's environment can't be read where this runs, only recorded; no reading")
    finding = {"id": "agent-holds-personal-and-cloud-keys", "statement": "s", "check": "agent-env-names", "reads": "recorded", "how": "h"}
    assert (O.status(finding, root).open, O.status(finding, root).basis) == (True, "unread")
    readings.write_text(_reading("agent-env", O.AGENT_ENV_COMMAND, now - DAY, 'names = ["RUNPOD_API_KEY", "LINEAR_API_KEY"]'))
    assert O.agent_env_names(root, {}) == (False, "an agent VM read by infra 1.0 days ago held 2 of the 5 names: LINEAR_API_KEY, "
                                                  "RUNPOD_API_KEY")
    readings.write_text(_reading("agent-env", O.AGENT_ENV_COMMAND, now - DAY, "names = []"))
    assert O.agent_env_names(root, {}) == (True, "an agent VM read by infra 1.0 days ago held 0 of the 5 names")
    assert O.status(finding, root) == O.Status(finding["id"], False, "recorded", "an agent VM read by infra 1.0 days ago held 0 of the 5 names")
    readings.write_text(_reading("agent-env", O.AGENT_ENV_COMMAND, now - DAY, 'names = ["PATH"]'))
    with pytest.raises(ValueError, match="names are those of AGENT_KEYS"):
        O.agent_env_names(root, {})
    assert O.AGENT_ENV_COMMAND.startswith("compgen -e | grep -xE '") and all(k in O.AGENT_ENV_COMMAND for k in O.AGENT_KEYS)


@pytest.mark.parametrize(("status", "code", "closed"), [(403, "AccessDenied", True), (404, "NoSuchKey", False), (200, "OK", False),
                                                       (403, "SignatureDoesNotMatch", None), (401, "Unauthorized", None)])
def test_the_store_check_closes_only_on_access_denied(O: ModuleType, tmp_path: Path, now: float, status: int, code: str,
                                                      closed: bool | None) -> None:
    root = _tree(tmp_path, _reading("store-write", O.STORE_COMMAND, now - DAY, f"status = {status}\ncode = {json.dumps(code)}"))
    assert O.store_write_refused(root, {})[0] is closed


def test_the_store_check_without_a_reading_says_where_the_key_is(O: ModuleType, tmp_path: Path, now: float) -> None:
    closed, why = O.store_write_refused(_tree(tmp_path), {})
    assert closed is None and why.startswith("the agents' store key is on agent VMs, not where this runs; no reading")


class _Store:
    """A stand-in for the store's client: answers each request with `status`, and records it."""
    bucket = "b"

    def __init__(self, status: int, body: bytes) -> None:
        self.status, self.body, self.asked = status, body, []

    def _request(self, method: str, key: str, headers: dict | None = None, **kw):
        import io
        import urllib.error
        self.asked.append((method, key, headers, kw))
        raise urllib.error.HTTPError("u", self.status, "r", {}, io.BytesIO(self.body))  # type: ignore[arg-type]


def test_the_store_probe_copies_from_a_missing_scratch_key_and_returns_only_status_and_code(O: ModuleType) -> None:
    store = _Store(403, b"<Error><Code>AccessDenied</Code><Message>no</Message></Error>")
    assert O.probe_store({}, store) == (403, "AccessDenied")
    ((method, key, headers, kw),) = store.asked
    source = headers["x-amz-copy-source"]
    assert method == "PUT" and kw == {} and key.startswith(f"{O.SCRATCH}/") and source.startswith(f"/b/{O.SCRATCH}/missing-")
    assert source != f"/b/{key}" and list(headers) == ["x-amz-copy-source"]
    assert O.probe_store({}, _Store(404, b"<Error><Code>NoSuchKey</Code></Error>")) == (404, "NoSuchKey")
    assert O.probe_store({}, _Store(400, b"")) == (400, "HTTP 400")
    with pytest.raises(SystemExit, match="AWS_SECRET_ACCESS_KEY is not set: run it on an agent VM"):
        O.probe_store({"R2_ENDPOINT": "https://example.invalid", "R2_BUCKET": "b", "AWS_ACCESS_KEY_ID": "dummy"})


#: what MACHINE_COMMAND prints for the agents' login on a CPU machine with root, 6.11 and the strict allow-list
STRICT = ["research", "1001", "0", "0", "6.11.0-1016-nvidia", "24.04", 'APT::Periodic::Unattended-Upgrade "1";',
          'Unattended-Upgrade::Allowed-Origins:: "${distro_id}:${distro_codename}";',
          'Unattended-Upgrade::Allowed-Origins:: "${distro_id}:${distro_codename}-security";',
          'Unattended-Upgrade::Allowed-Origins:: "${distro_id}ESMApps:${distro_codename}-apps-security";',
          'Unattended-Upgrade::Package-Whitelist:: "ca-certificates";', 'Unattended-Upgrade::Package-Whitelist-Strict "true";']
FIXED = ["research", "1001", "refused", "0", "6.8.0-90-nvidia", "24.04", *STRICT[6:10]]


@pytest.mark.parametrize(("lines", "closed", "why"), [
    (FIXED, True, "refused"),
    (STRICT, False, "sudo gives research root"),
    (["root", "0", "refused", "0", "6.8.0-147-generic", "22.04"], False, "root is root"),
    (["research", "1001", "refused", "1", *FIXED[4:]], None, "read with no new privileges, where sudo refuses every login"),
    (["job", "1002", "refused", "0", *FIXED[4:]], None, "read as job, not the agents' research"),
    (["research", "1001", "1002", "0", *FIXED[4:]], None, "sudo printed '1002'"),
])
def test_root_is_refused_only_by_a_sudo_refusal_of_the_agents_login_with_privileges_to_gain(O: ModuleType, lines: list, closed,
                                                                                           why: str) -> None:
    assert O.root_refused(O.Machine.of(lines)) == (closed, why)


def test_a_machine_reading_is_six_lines_and_apts(O: ModuleType) -> None:
    assert O.Machine.of(FIXED).apt == tuple(STRICT[6:10])
    with pytest.raises(ValueError, match="six and apt's"):
        O.Machine.of(FIXED[:5])


@pytest.mark.parametrize(("lines", "why"), [
    (FIXED, "maintained"),
    (STRICT, "kernel 6.11 on 24.04 is not maintained (kernels.toml); a package allow-list"),
    ([*FIXED, STRICT[10]], "a package allow-list"),
    ([*FIXED, STRICT[10], 'Unattended-Upgrade::Package-Whitelist-Strict "false";'], "a package allow-list"),
    ([*FIXED, 'Unattended-Upgrade::Package-Whitelist "";', 'Unattended-Upgrade::Package-Whitelist-Strict "true";'], "maintained"),
    ([*FIXED[:4], "7.0.0-12-generic", "24.04", *FIXED[6:]], "maintained"),
    ([*FIXED[:4], "6.8.0-90-generic", "22.04", *FIXED[6:]], "kernel 6.8 on 22.04 is not maintained (kernels.toml)"),
    ([*FIXED[:6], 'APT::Periodic::Unattended-Upgrade "0";', *FIXED[7:]], "no unattended upgrades"),
    ([*FIXED[:6], *FIXED[6:8], FIXED[9]], "Ubuntu's security pocket is no origin"),
    (FIXED[:6], "no unattended upgrades; Ubuntu's security pocket is no origin"),
])
def test_updates_hold_on_a_maintained_kernel_the_security_pocket_and_no_allow_list(O: ModuleType, monkeypatch: pytest.MonkeyPatch,
                                                                                    lines: list, why: str) -> None:
    """The kernel table's dates are its own: 6.8 on 22.04 ends in May 2027, so on a day after it the series is not maintained."""
    monkeypatch.setattr(O, "clock", lambda: O.datetime(2027, 6, 1, tzinfo=O.UTC).timestamp())
    judge = O.updates_judge(O.kernels(ROOT))
    assert judge(O.Machine.of(lines)) == (why == "maintained", why)


def test_the_kernel_table_names_its_source_and_each_series_by_release_with_its_end(O: ModuleType) -> None:
    table = O.kernels(ROOT)
    assert tomllib.loads(O.KERNELS.read_text(encoding="utf-8"))["source"] == "https://ubuntu.com/kernel/lifecycle"
    assert all(isinstance(until, O.date) and not isinstance(until, O.datetime) for until in table.values())
    assert ("6.8", "24.04") in table and not any(series == "6.11" for series, _ in table)


def test_the_machines_are_the_deploy_maps_cpu_machines_but_the_staged(O: ModuleType, tmp_path: Path) -> None:
    assert O.cpu_machines(_tree(tmp_path)) == ["vy-nebius-cpu-2", "vy-nebius-cpu-10"]
    assert "vy-nebius-cpu-12" in O.cpu_machines(ROOT)


def test_here_is_the_machine_the_run_recorded_else_the_host(O: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(O.socket, "gethostname", lambda: "a-host")
    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
    assert O.here() == "a-host"
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(tmp_path))
    assert O.here() == "a-host"
    (tmp_path / "job.json").write_text(json.dumps({"run": {"remote": {"machine": "vy-nebius-cpu-12"}}}), encoding="utf-8")
    assert O.here() == "vy-nebius-cpu-12"


def _machines(O: ModuleType, at: float, output: dict[str, list[str]]) -> str:
    body = "".join(f"{name} = {json.dumps(lines)}\n" for name, lines in output.items())
    return _reading("machines", O.MACHINE_COMMAND, at, f"[machines.output]\n{body}")


def test_the_machine_checks_read_this_machine_itself_and_the_others_from_the_reading(O: ModuleType, tmp_path: Path, now: float,
                                                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    """Root: one machine giving root opens the finding; one unread leaves it unread; this machine's own decisive read counts
    over the reading, and an undecided one falls back to it; past seven days only this machine's own read counts."""
    reads: list[str] = []

    def live(lines: list[str]):
        monkeypatch.setattr(O, "read_here", lambda: reads.append("live") or lines)

    monkeypatch.setattr(O, "here", lambda: "vy-nebius-cpu-2")
    pod = ["root", "0", "refused", "0", "6.8.0-147-generic", "22.04"]
    root = _tree(tmp_path, _machines(O, now - DAY, {"vy-nebius-cpu-2": STRICT, "vy-nebius-cpu-10": FIXED, "vy-control-verity": pod}))
    live(FIXED)
    closed, why = O.sudo_refused(root, {})
    assert closed is False and reads == ["live"]
    assert why == ("1 of 3 machines give the agents root, 0 unread; read vy-nebius-cpu-2 itself, the rest read by infra 1.0 days "
                   "ago (root is root: vy-control-verity)")
    assert O.updates_and_kernel(root, {})[0] is True
    live(["research", "1001", "refused", "1", *FIXED[4:]])
    assert O.sudo_refused(root, {})[1].startswith("2 of 3 machines give the agents root"), "an undecided read falls back"
    readings = root / "benchmarks/security_findings/readings.toml"
    readings.write_text(_machines(O, now - DAY, {"vy-nebius-cpu-2": FIXED, "vy-nebius-cpu-10": FIXED,
                                                 "vy-control-verity": ["agent", "1003", "refused", "0", *FIXED[4:6]]}))
    live(FIXED)
    closed, why = O.sudo_refused(root, {})
    assert closed is None and why.endswith("(unread: read as agent, not the agents' research: vy-control-verity)")
    readings.write_text(_machines(O, now - DAY, {"vy-nebius-cpu-2": FIXED, "vy-nebius-cpu-10": FIXED,
                                                 "vy-control-verity": ["research", "1001", "refused", "0", *FIXED[4:6]]}))
    assert O.sudo_refused(root, {}) == (True, "0 of 3 machines give the agents root, 0 unread; read vy-nebius-cpu-2 itself, "
                                              "the rest read by infra 1.0 days ago")
    readings.write_text(_machines(O, now - 8 * DAY, {"vy-nebius-cpu-2": FIXED, "vy-nebius-cpu-10": FIXED}))
    closed, why = O.updates_and_kernel(root, {})
    assert closed is None and why.startswith("0 of 2 machines fall short, 1 unread; read vy-nebius-cpu-2 itself, the rest unrecorded")
    assert "the reading [machines] is 8 days old, over 7" in why and why.endswith(": vy-nebius-cpu-10)")
    monkeypatch.setattr(O, "here", lambda: "an-agent-vm")
    readings.write_text("")
    closed, why = O.updates_and_kernel(root, {})
    assert closed is None and "read none of them itself (it runs on none)" in why


def test_tonights_readings_decide_each_finding_with_the_checks_commands(O: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    """The recorded readings parse under their checks: a day after each was taken, every check reading them decides."""
    readings = tomllib.loads(O.READINGS.read_text(encoding="utf-8"))
    assert sorted(readings) == ["agent-env", "machines", "store-write"]
    monkeypatch.setattr(O, "here", lambda: "an-agent-vm")
    for name, check in [("agent-env", O.agent_env_names), ("store-write", O.store_write_refused), ("machines", O.updates_and_kernel)]:
        day_after = readings[name]["at"].timestamp() + DAY
        monkeypatch.setattr(O, "clock", lambda: day_after)
        assert check(ROOT, {})[0] is not None, name


#: a dotted quad of any kind, and a run of key material: 24 or more base64 or hex characters holding a digit and a letter
_QUAD = re.compile(r"(?<![\d.])\d{1,3}(?:\.\d{1,3}){3}(?![\d.])")
_KEYLIKE = re.compile(r"[A-Za-z0-9+/=]{24,}")
_KEY_PREFIXES = ("-----BEGIN", "AKIA", "ASIA", "ghp_", "gho_", "ghs_", "github_pat_", "xox", "eyJ", "sk-", "vsk_", "rpa_")


@pytest.mark.parametrize("name", ["findings.toml", "readings.toml", "kernels.toml"])
def test_the_list_and_its_data_name_no_address_key_or_auditors_machine(O: ModuleType, name: str) -> None:
    """The readings name the control pod, which is the team's machine and not an auditor's, and no other machine the deploy
    map reaches only through its admin account."""
    from research import addresses

    text = (O.HERE / name).read_text(encoding="utf-8")
    assert addresses.findings(text) == [] and not _QUAD.search(text)
    keylike = [m for m in _KEYLIKE.findall(text) if re.search(r"\d", m) and re.search(r"[A-Za-z]", m)]
    assert keylike == [] and not [p for p in _KEY_PREFIXES if p in text]
    deploy = tomllib.loads((ROOT / "tools/research/src/research/pods/nebius/deploy.toml").read_text(encoding="utf-8"))
    admin = [n for n, node in deploy["nodes"].items() if node.get("admin") and n != O.CONTROL_POD]
    assert admin and [n for n in admin if n in text] == []


def test_the_list_checks_refuse_what_they_must() -> None:
    assert _QUAD.search("at " + ".".join(["10", "0", "0", "2"]))
    assert [m for m in _KEYLIKE.findall("key " + "Ab3" * 9) if re.search(r"\d", m)]
