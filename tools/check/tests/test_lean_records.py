"""`tools/check/lean_changed.py --records`: a lane's records audited without the kernel replay on a tree kept per check slot,
with its own copy of the audit's dependencies, taken through `slot.py` where the machine has check slots."""
from __future__ import annotations

import fcntl
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location("check_lean_changed_records", ROOT / "tools" / "check" / "lean_changed.py")
LC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(LC)

#: `audit.py` as far as `--records` sees it: its arguments, working directory and threads logged, `--update` writing a record
FAKE_AUDIT = """\
import json, os, sys
with open(os.environ["AUDIT_LOG"], "a") as f:
    f.write(json.dumps({"argv": sys.argv[1:], "cwd": os.getcwd(), "threads": os.environ.get("LEAN_NUM_THREADS")}) + "\\n")
if "--update" in sys.argv:
    with open(os.path.join(sys.argv[-1], "lean-audit.json"), "w") as f:
        f.write('{"pins": {"A.t": {}}}')
sys.exit(int(os.environ.get("AUDIT_RC", "0")))
"""


class Exec(Exception):
    pass


@pytest.fixture
def repo(tmp_path, monkeypatch) -> Path:
    """A checkout with package `a` (a git dependency, and `p` by path), `p` and `c`; `audit.py` as `FAKE_AUDIT`; no check slot
    and no slots file, whatever machine runs the tests."""
    root = tmp_path / "repo"
    files = {
        "a/lakefile.toml": "", "a/lean-toolchain": "v1\n", "a/lean-audit.json": "{}", "a/Aa.lean": "def a := 1\n",
        "a/lake-manifest.json": json.dumps({"packages": [{"name": "dep", "type": "git", "rev": "abc"},
                                                         {"name": "p", "type": "path", "dir": "../p"}]}),
        "p/lakefile.toml": "", "p/lean-toolchain": "v1\n", "p/lean-audit.json": "{}", "p/Pp.lean": "def p := 1\n",
        "c/lakefile.toml": "", "c/lean-toolchain": "v1\n", "c/lean-audit.json": "{}", "c/C.lean": "def c := 1\n",
        "c/lake-manifest.json": json.dumps({"packages": [{"name": "dep", "type": "git", "rev": "def"}]}),
    }
    for f, text in files.items():
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        (root / f).write_text(text)
    for m in (LC, LC.LA):
        monkeypatch.setattr(m, "ROOT", root)
    monkeypatch.setattr(LC.LA.COMMON, "tracked_files", lambda r=None: sorted(files))
    monkeypatch.setattr(LC.LA.COMMON, "cores", lambda: 2)
    (tmp_path / "audit.py").write_text(FAKE_AUDIT)
    monkeypatch.setattr(LC, "AUDIT_PY", tmp_path / "audit.py")
    monkeypatch.setenv("AUDIT_LOG", str(tmp_path / "audit.log"))
    monkeypatch.setenv("CHECK_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("CHECK_SLOT", raising=False)
    monkeypatch.setattr(LC.SLOT, "SLOTS", tmp_path / "no-slots")
    return root


def _calls(tmp: Path) -> list[dict]:
    log = tmp / "audit.log"
    calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    log.unlink(missing_ok=True)
    return calls


def test_the_audit_runs_without_the_replay_on_the_slots_tree_with_a_copy_of_the_audits_dependencies(repo, tmp_path,
                                                                                                    monkeypatch, capsys):
    monkeypatch.setenv("CHECK_SLOT", "c")
    tree, deps = LC.records_tree(), tmp_path / "deps"
    assert tree == tmp_path / "cache" / "lean-records" / "c"
    k = LC.LA.deps_key(repo / "a")
    (deps / k / "dep").mkdir(parents=True)
    (deps / k / "dep" / "Dep.olean").write_text("olean")
    assert LC.audit_records("a", deps, tree) == 0
    assert _calls(tmp_path) == [{"argv": ["--build", "--no-replay", "--no-runs", str(tree / "a")], "cwd": str(tree),
                                 "threads": "2"}]
    assert (tree / "a" / "Aa.lean").is_file() and (tree / "p" / "Pp.lean").is_file() and not (tree / "c").exists()
    copied, kept = tree / "a" / ".lake" / "packages" / "dep" / "Dep.olean", deps / k / "dep" / "Dep.olean"
    assert copied.read_text() == "olean" and copied.stat().st_ino != kept.stat().st_ino, "a copy, never a link to the audit's"
    assert json.loads((tree / LC.LINKS).read_text()) == {"a": k}
    out = capsys.readouterr().out
    assert f"dependencies copied ({k[:12]})" in out and "lean-records: a: audit.py exit 0" in out

    assert LC.audit_records("a", deps, tree, update=True) == 0
    assert _calls(tmp_path)[0]["argv"] == ["--build", "--no-replay", "--no-runs", "--update", str(tree / "a")]
    assert (repo / "a" / "lean-audit.json").read_text() == '{"pins": {"A.t": {}}}', "the record comes back to the checkout"
    out = capsys.readouterr().out
    assert f"lean-records: wrote {repo / 'a' / 'lean-audit.json'}" in out
    assert "dependencies copied" not in out, "the tree keeps its copy while the manifest and toolchain stay"

    (repo / "a" / "lean-audit.json").write_text("{}")
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(tmp_path / "run"))
    moved = tmp_path / "moved.json"
    assert LC.audit_records("a", deps, tree, update=True, moved=moved, out=tmp_path / "reports") == 0
    assert _calls(tmp_path)[0]["argv"][3:] == ["--update", "--moved", str(moved), "--out", str(tmp_path / "reports"),
                                               str(tree / "a")]
    assert (repo / "a" / "lean-audit.json").read_text() == "{}", "a recorded run's source tree is shared by its commit's runs"

    monkeypatch.setenv("AUDIT_RC", "1")
    assert LC.audit_records("a", deps, tree) == 1, "the audit's verdict on the records is the exit status"


def test_without_the_audits_dependencies_the_tree_fetches_its_own_once(repo, tmp_path, capsys):
    tree, deps = tmp_path / "tree", tmp_path / "deps"
    k = LC.LA.deps_key(repo / "a")
    assert LC.audit_records("a", deps, tree) == 0
    assert json.loads((tree / LC.LINKS).read_text()) == {"a": k}
    assert "lean-records: a: its dependencies are fetched cold, once for this tree" in capsys.readouterr().out
    (tree / "a" / ".lake" / "packages" / "dep").mkdir(parents=True)          # what the audit's `setup.sh` fetched there
    (deps / k / "dep").mkdir(parents=True)
    assert LC.audit_records("a", deps, tree) == 0
    out = capsys.readouterr().out
    assert "fetched cold" not in out and "copied" not in out, "the tree keeps what it fetched"
    assert LC.audit_records("p", deps, tree) == 0 and json.loads((tree / LC.LINKS).read_text()) == {"a": k}, \
        "a package that fetches nothing needs none"


def test_a_tree_keeps_one_packages_dependencies(repo, tmp_path):
    tree, deps = tmp_path / "tree", tmp_path / "deps"
    ka, kc = LC.LA.deps_key(repo / "a"), LC.LA.deps_key(repo / "c")
    assert ka != kc
    for k in (ka, kc):
        (deps / k / "dep").mkdir(parents=True)
    assert LC.audit_records("a", deps, tree) == 0 and (tree / "a" / ".lake" / "packages" / "dep").is_dir()
    assert LC.audit_records("c", deps, tree) == 0
    assert json.loads((tree / LC.LINKS).read_text()) == {"c": kc}
    assert not (tree / "a" / ".lake" / "packages").exists(), "each copy costs inodes the node is short of"
    assert (tree / "c" / ".lake" / "packages" / "dep").is_dir() and (deps / ka / "dep").is_dir()


def test_it_names_one_package_and_one_run_holds_a_tree(repo, tmp_path, capsys):
    tree, deps = tmp_path / "tree", tmp_path / "deps"
    assert LC.audit_records("q", deps, tree) == 2
    assert "lean-records: 'q' matches 0 packages: a, c, p" in capsys.readouterr().out
    assert LC.audit_records("p/", deps, tree) == 0 and _calls(tmp_path)[0]["argv"][-1] == str(tree / "p")
    with open(tree / ".lock", "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        assert LC.audit_records("a", deps, tree) == 2
    assert f"lean-records: another run holds {tree}" in capsys.readouterr().out and _calls(tmp_path) == []


def test_it_takes_a_check_slot_first_where_the_machine_has_check_slots(repo, tmp_path, monkeypatch):
    def execv(exe, argv):
        raise Exec(exe, argv)

    seen = []
    monkeypatch.setattr(LC.os, "execv", execv)
    monkeypatch.setattr(LC, "audit_records", lambda name, deps, tree, *a, **kw: seen.append((name, tree, a)) or 0)
    assert LC.main(["--records", "a", "--update"]) == 0, "no slots file here: it runs as it is"
    assert seen.pop() == ("a", tmp_path / "cache" / "lean-records" / "local", (True, None, None))

    (tmp_path / "slots").write_text("c 8-31\n")
    monkeypatch.setattr(LC.SLOT, "SLOTS", tmp_path / "slots")
    monkeypatch.setenv("RESEARCH_LANE", "lane-x")
    with pytest.raises(Exec) as e:
        LC.main(["--records", "a", "--update"])
    lean_changed = str((ROOT / "tools" / "check" / "lean_changed.py").resolve())
    assert e.value.args == (sys.executable, [sys.executable, str(ROOT / "tools" / "check" / "slot.py"), "--by", "lane-x", "--",
                                             sys.executable, lean_changed, "--records", "a", "--update"])
    assert not seen

    monkeypatch.setenv("CHECK_SLOT", "c")
    assert LC.main(["--records", "a"]) == 0, "in the slot that slot.py took"
    assert seen.pop()[1] == tmp_path / "cache" / "lean-records" / "c"

    with pytest.raises(SystemExit):
        LC.main(["--update"])
