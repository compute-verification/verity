"""The count of the security review's open findings (`open_findings.py`): only the check the script holds for a finding's id
closes it; a finding with no check is open whatever the list holds, and the list holds no reading; the mark is `measured`
only when every finding is decided by a check of the tree; tonight's list closes nothing by itself and its exceptions are
pinned; the guard check takes only commands that run the guard from `/opt/agent-guard/`; the address check counts every
tracked file the list doesn't except; `main` writes the open count; and the list names no address, key or auditor's machine.

No test reads the clock or the network: the checks run on stand-ins or on toy git checkouts under `tmp_path`, and the
script is imported with the repository's root on `sys.path` only through `monkeypatch`."""
from __future__ import annotations

import importlib
import json
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from types import ModuleType

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FINDING = {"id": "x", "statement": "s", "would_close": "a check that would"}
CHECKED = {"guard-runs-from-the-branch": ("guard-outside-checkout", "declared"),
           "machine-addresses-in-the-tree": ("no-public-address", "tree")}
PUBLIC = ".".join(["81", "2", "3", "4"])


@pytest.fixture
def O(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(ROOT))
    return importlib.import_module("benchmarks.security_findings.open_findings")


def stand_in(O: ModuleType, closed: bool) -> dict:
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
    for other in ("declared", "none", "refused"):
        assert O.mark([s("tree"), s(other)]) == "partial", other
    assert O.mark([]) == "partial"


def test_tonights_list_closes_nothing_by_itself(O: ModuleType) -> None:
    """Whatever the checks find, nothing in the list closes a finding: 6 open when both checks find their finding open, and
    only the two checked findings close when both find them closed."""
    findings = O.load()
    assert [f["id"] for f in findings] == ["agent-holds-personal-and-cloud-keys", "guard-runs-from-the-branch",
                                           "agents-have-root", "security-updates-and-kernel", "store-key-writes",
                                           "machine-addresses-in-the-tree"]
    open_ = [O.status(f, checks=stand_in(O, False)) for f in findings]
    assert sum(s.open for s in open_) == 6 and O.mark(open_) == "partial"
    closed = [O.status(f, checks=stand_in(O, True)) for f in findings]
    assert sorted(s.id for s in closed if not s.open) == sorted(CHECKED)
    assert O.mark(closed) == "partial"


def test_the_address_exceptions_are_these_nine(O: ModuleType) -> None:
    """Each exception drops a file's addresses from finding 6's count, so adding one changes this test in the same change."""
    six = next(f for f in O.load() if f["id"] == "machine-addresses-in-the-tree")
    allowed = O.excepted(six)
    assert (len(allowed), sum(allowed.values())) == (9, 15)
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

    def entry(path: str, count: int, kind: str = "example") -> dict:
        return {"path": path, "count": count, "kind": kind, "reason": "r"}
    some = {**none, "except": [entry("tests/d.py", 2), entry("e/tests/f.txt", 1)]}
    assert O.no_public_address(root, some)[1].startswith("2 ")
    every = {**none, "except": [entry("tests/d.py", 2), entry("e/tests/f.txt", 2), entry("g.md", 2)]}
    assert O.no_public_address(root, every)[0] is True
    for bad in (entry("g.md", 2, "ours"), entry("g.md", 0), {"path": "g.md", "count": 2, "kind": "example"}):
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


#: a dotted quad of any kind, and a run of key material: 24 or more base64 or hex characters holding a digit and a letter
_QUAD = re.compile(r"(?<![\d.])\d{1,3}(?:\.\d{1,3}){3}(?![\d.])")
_KEYLIKE = re.compile(r"[A-Za-z0-9+/=]{24,}")
_KEY_PREFIXES = ("-----BEGIN", "AKIA", "ASIA", "ghp_", "gho_", "ghs_", "github_pat_", "xox", "eyJ", "sk-", "vsk_", "rpa_")


def test_the_list_names_no_address_key_or_auditors_machine(O: ModuleType) -> None:
    from research import addresses

    text = O.LIST.read_text(encoding="utf-8")
    assert addresses.findings(text) == [] and not _QUAD.search(text)
    keylike = [m for m in _KEYLIKE.findall(text) if re.search(r"\d", m) and re.search(r"[A-Za-z]", m)]
    assert keylike == [] and not [p for p in _KEY_PREFIXES if p in text]
    deploy = tomllib.loads((ROOT / "tools/research/src/research/pods/nebius/deploy.toml").read_text(encoding="utf-8"))
    admin = [n for n, node in deploy["nodes"].items() if node.get("admin")]
    assert admin and [n for n in admin if n in text] == []


def test_the_list_checks_refuse_what_they_must() -> None:
    assert _QUAD.search("at " + ".".join(["10", "0", "0", "2"]))
    assert [m for m in _KEYLIKE.findall("key " + "Ab3" * 9) if re.search(r"\d", m)]
