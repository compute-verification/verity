"""The count of the security review's open findings (`open_findings.py`): a check decides its finding either way; a reading
counts only whole and less than seven days old at the run's time, which is injected; a finding with neither is open; the
mark is `measured` only when every finding is decided by a check of the tree; tonight's list closes nothing by itself;
the two checks on toy checkouts; and the list names no address, key or auditor's machine.

No test reads the clock or the network: the run's time is a fixed instant, and the checks run on stand-ins or on toy git
checkouts under `tmp_path`."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.security_findings import open_findings as O  # noqa: E402

NOW = datetime(2026, 1, 10, 12, 0, tzinfo=UTC)
FINDING = {"id": "x", "statement": "s", "how": "h"}


def reading(**kw: object) -> dict:
    return {"command": "sudo -n id -u", "reader": "infra", "at": NOW - timedelta(days=1), "showed": "refused on each",
            "closed": True, **kw}


def stand_in(closed: bool) -> dict[str, O.Check]:
    return {name: O.Check(c.reads, lambda root, closed=closed: (closed, "stand-in")) for name, c in O.CHECKS.items()}


def test_a_finding_with_neither_check_nor_reading_is_open() -> None:
    assert O.status(FINDING, NOW, checks={}) == O.Status("x", True, "none", "no check and no reading")


def test_a_whole_fresh_reading_closes_its_finding_and_one_that_says_open_does_not() -> None:
    assert O.status({**FINDING, "reading": reading()}, NOW) == O.Status("x", False, "reading", "read by infra: refused on each")
    assert O.status({**FINDING, "reading": reading(closed=False)}, NOW).open


def test_an_expired_reading_counts_open() -> None:
    assert not O.status({**FINDING, "reading": reading(at=NOW - O.FRESH + timedelta(seconds=1))}, NOW).open
    expired = O.status({**FINDING, "reading": reading(at=NOW - O.FRESH)}, NOW)
    assert (expired.open, expired.basis) == (True, "none") and "expired" in expired.why
    assert O.status({**FINDING, "reading": reading(at=NOW - timedelta(days=30))}, NOW).open


@pytest.mark.parametrize("field", O.READING)
def test_a_reading_without_a_field_is_refused(field: str) -> None:
    for r in ({k: v for k, v in reading().items() if k != field}, reading(**{field: ""})):
        got = O.status({**FINDING, "reading": r}, NOW)
        assert (got.open, got.basis, got.why) == (True, "refused", f"its reading is refused: it lacks {field}")


def test_a_reading_without_its_command_or_reader_is_refused() -> None:
    for field in ("command", "reader"):
        got = O.status({**FINDING, "reading": reading(**{field: None})}, NOW)
        assert (got.open, got.basis) == (True, "refused") and field in got.why
    assert O.status({**FINDING, "reading": reading(reader="  ")}, NOW).basis == "refused"


def test_a_reading_whose_time_is_not_utc_or_later_than_the_run_is_refused() -> None:
    for at in (NOW.replace(tzinfo=None) - timedelta(days=1), "2026-01-09T12:00:00Z", NOW + timedelta(seconds=1)):
        got = O.status({**FINDING, "reading": reading(at=at)}, NOW)
        assert (got.open, got.basis) == (True, "refused"), at
    assert O.status({**FINDING, "reading": reading(closed="yes")}, NOW).basis == "refused"


def test_a_check_decides_its_finding_either_way() -> None:
    f = {**FINDING, "check": "no-public-address", "reads": "tree"}
    assert O.status(f, NOW, checks=stand_in(True)) == O.Status("x", False, "tree", "stand-in")
    assert O.status(f, NOW, checks=stand_in(False)) == O.Status("x", True, "tree", "stand-in")
    assert O.status({**f, "reading": reading()}, NOW, checks=stand_in(False)).basis == "refused"
    assert O.status({**f, "reading": reading()}, NOW, checks=stand_in(True)).open
    assert O.status({**f, "reads": "declared"}, NOW, checks=stand_in(True)).basis == "refused"
    assert O.status({**f, "check": "nothing"}, NOW, checks=stand_in(True)).basis == "refused"


def test_a_check_that_cannot_run_counts_open() -> None:
    def broken(root: Path) -> tuple[bool, str]:
        raise OSError("no checkout")
    f = {**FINDING, "check": "c", "reads": "tree"}
    got = O.status(f, NOW, checks={"c": O.Check("tree", broken)})
    assert (got.open, got.basis) == (True, "refused")


def test_the_mark_is_measured_only_when_every_finding_rests_on_a_check_of_the_tree() -> None:
    def s(basis: str, open_: bool = False) -> O.Status:
        return O.Status("x", open_, basis, "")
    assert O.mark([s("tree"), s("tree", True)]) == "measured"
    for other in ("declared", "reading", "none", "refused"):
        assert O.mark([s("tree"), s(other)]) == "partial", other
    assert O.mark([]) == "partial"


def test_tonights_list_closes_nothing_by_itself() -> None:
    """Whatever the checks find, no reading or entry of the list closes a finding: 6 open when both checks find their
    finding open, and only the two checked findings close when both find them closed."""
    findings = O.load()
    assert len(findings) == 6
    open_ = [O.status(f, NOW, checks=stand_in(False)) for f in findings]
    assert sum(s.open for s in open_) == 6 and O.mark(open_) == "partial"
    closed = [O.status(f, NOW, checks=stand_in(True)) for f in findings]
    assert sorted(s.id for s in closed if not s.open) == ["guard-runs-from-the-branch", "machine-addresses-in-the-tree"]
    assert O.mark(closed) == "partial"


def test_the_list_refuses_an_entry_without_its_words_or_a_repeated_id(tmp_path: Path) -> None:
    p = tmp_path / "findings.toml"
    p.write_text('[[finding]]\nid = "a"\nstatement = "s"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="how it is checked"):
        O.load(p)
    p.write_text('[[finding]]\nid = "a"\nstatement = "s"\nhow = "h"\n' * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="share an id"):
        O.load(p)


def _checkout(tmp_path: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text, encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    return tmp_path


def test_the_address_check_finds_a_public_address_and_no_documentation_one(tmp_path: Path) -> None:
    public = ".".join(["81", "2", "3", "4"])
    root = _checkout(tmp_path, {"a.sh": "ssh research@192.0.2.7\n", "b/c.toml": 'v = "12.9.1.4"\n'})
    assert O.no_public_address(root) == (True, "0 tracked files hold a public IPv4 address")
    (root / "b" / "d.txt").write_text(f"host {public}\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    assert O.no_public_address(root) == (False, "1 tracked files hold a public IPv4 address")


MAIN_LIKE = ("sh -c 'h=tools/agent-guard/hook.sh; [ -f \"$h\" ] && exec sh \"$h\" \"$1\"; d=$(ls -td "
             "\"${AGENT_GUARD_CACHE:-${HOME:-/tmp}/.cache/agent-guard}\"/[0-9a-f]*/ 2>/dev/null | head -1); "
             "[ -f \"$d$1\" ] || d=/opt/agent-guard/; exec python3 \"$d$1\"' guard shell_hook.py")


@pytest.mark.parametrize(("command", "inside"), [
    (MAIN_LIKE, True),
    ("python3 /opt/agent-guard/shell_hook.py", False),
    ("sh -c 'd=${AGENT_GUARD_HOME:-/opt/agent-guard}; exec python3 \"$d/task_hook.py\"'", False),
    ("python3 \"$CURSOR_PROJECT_DIR/tools/agent-guard/shell_hook.py\"", True),
    ("python3 \"$(git rev-parse --show-toplevel)/tools/agent-guard/shell_hook.py\"", True),
    ("./tools/agent-guard/hook.sh shell_hook.py", True),
    ("cd tools && python3 agent-guard/shell_hook.py", True),
    ("python3 /opt/tools/agent-guard/shell_hook.py", False),
])
def test_a_hook_command_naming_a_file_of_the_checkout_is_found(command: str, inside: bool) -> None:
    assert O.runs_checkout_file(command, {"tools", ".cursor", "tests"}) is inside


def test_the_guard_check_reads_every_hook_command(tmp_path: Path) -> None:
    def hooks(*commands: str) -> str:
        return json.dumps({"version": 1, "hooks": {"stop": [{"command": c} for c in commands]}})
    root = _checkout(tmp_path, {".cursor/hooks.json": hooks("python3 /opt/agent-guard/stop_hook.py"),
                                "tools/agent-guard/hook.sh": "#!/bin/sh\n"})
    assert O.guard_outside_checkout(root) == (True, "0 of 1 hook commands of .cursor/hooks.json run a file of the checkout")
    (root / ".cursor/hooks.json").write_text(hooks("python3 /opt/agent-guard/stop_hook.py", MAIN_LIKE), encoding="utf-8")
    assert O.guard_outside_checkout(root) == (False, "1 of 2 hook commands of .cursor/hooks.json run a file of the checkout")
    (root / ".cursor/hooks.json").write_text(hooks(), encoding="utf-8")
    assert O.guard_outside_checkout(root)[0] is False


#: a dotted quad of any kind, and a run of key material: 24 or more base64 or hex characters holding a digit and a letter
_QUAD = re.compile(r"(?<![\d.])\d{1,3}(?:\.\d{1,3}){3}(?![\d.])")
_KEYLIKE = re.compile(r"[A-Za-z0-9+/=]{24,}")
_KEY_PREFIXES = ("-----BEGIN", "AKIA", "ASIA", "ghp_", "gho_", "ghs_", "github_pat_", "xox", "eyJ", "sk-", "vsk_", "rpa_")


def test_the_list_names_no_address_key_or_auditors_machine() -> None:
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
