"""`stage.py`: a move staged on a base as one certified commit per map, on toy repositories holding this checkout's move
scripts, as the window's base holds its own."""
from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
spec = importlib.util.spec_from_file_location("stage_under_test", HERE.parent / "stage.py")
ST = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ST)
spec = importlib.util.spec_from_file_location("stage_move_check", ROOT / ST.MOVE_CHECK)
MC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MC)

ENV = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
       "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1"}
SCRIPTS = [f"tools/move/{p.name}" for p in sorted((ROOT / "tools/move").glob("*.py"))] + [ST.MOVE_CHECK, ST.PINS]
WORDS = 'scope = "prose"\n\n[words]\n"zorble" = "quux"\n\n[rewrite]\nexclude = ["tools/**"]\n'
PATHS = ('[roots]\nold = {}\nnew = {}\n\n[paths]\n"alpha/" = "beta/"\n\n[modules]\n"alpha" = "beta"\n\n'
         '[rewrite]\nexclude = ["tools/move/**", "tools/verity/**"]\n')
TREE = {
    "notes.md": "A zorble is `zorble`.\n",
    "a.py": '"""Each zorble."""\nZORBLE = "zorble"\n',
    "alpha/__init__.py": "",
    "alpha/x.py": "def f():\n    return 1\n",
    "user.py": "from alpha.x import f  # a zorble\n",
    "tools/research/src/research/store/moves.json": '{"moves": []}\n',
}


@pytest.fixture(autouse=True)
def _git_identity(monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=repo, check=True, capture_output=True,
                          text=True).stdout.strip()


def _repo(tmp_path: Path, files: dict[str, str], scripts: dict[str, str] | None = None) -> tuple[Path, str]:
    """A repository whose one commit holds this checkout's move scripts (``scripts`` replacing some), and ``files``."""
    repo = tmp_path / "repo"
    files = {**{p: (ROOT / p).read_text() for p in SCRIPTS}, **(scripts or {}), **files}
    for p, text in files.items():
        (repo / p).parent.mkdir(parents=True, exist_ok=True)
        (repo / p).write_text(text)
    git(repo, "init", "-q", "-b", "main")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "base")
    return repo, git(repo, "rev-parse", "HEAD")


def _stage(monkeypatch, repo: Path, *args: str) -> int:
    monkeypatch.chdir(repo)
    return ST.main([*args, "--no-fetch"])


def _show(repo: Path, rev: str, path: str) -> str:
    return git(repo, "show", f"{rev}:{path}")


def test_a_word_map_and_a_path_map_stage_as_a_chain_of_certified_commits(tmp_path, monkeypatch, capsys):
    """The word map's commit is on BASE and the path map's on it; each names its command and map, and move_check.py
    certifies each on its parent as the lander runs it; the checkout itself is left as it was."""
    repo, base = _repo(tmp_path, {**TREE, "tools/move/zorble_map.toml": WORDS, "tools/move/alpha_map.toml": PATHS})
    cert = tmp_path / "cert.json"
    assert _stage(monkeypatch, repo, "tools/move/zorble_map.toml", "tools/move/alpha_map.toml", "--base", base,
                  "--no-lock", "--certificate", str(cert)) == 0
    out = capsys.readouterr().out
    words, paths = json.loads(cert.read_text())["moves"]
    assert (words["parent"], paths["parent"]) == (base, words["commit"])
    assert git(repo, "rev-parse", f"{paths['commit']}^") == words["commit"]

    w = words["commit"]
    assert _show(repo, w, "notes.md") == "A quux is `zorble`."
    assert _show(repo, w, "a.py") == '"""Each quux."""\nZORBLE = "zorble"'
    message = git(repo, "log", "-1", "--format=%B", w)
    assert message.startswith("Move by tools/move/zorble_map.toml\n\n`python3 tools/move/rename.py --map "
                              f"tools/move/zorble_map.toml` at the root of {base}")
    assert message.endswith("\n\nMove: tools/move/zorble_map.toml")

    p = paths["commit"]
    assert _show(repo, p, "beta/x.py") == "def f():\n    return 1"
    assert _show(repo, p, "user.py") == "from beta.x import f  # a quux"
    assert git(repo, "log", "-1", "--format=%B", p).endswith("\n\nLayout-move: tools/move/alpha_map.toml")

    for m in (words, paths):
        r = MC.certify(m["parent"], m["commit"], cwd=str(repo))
        assert r["certified"] and m["report"]["certified"], r["refusals"]
        assert m["check"] == f"verity check move {m['parent']} {m['commit']}" and m["check"] in out
    assert words["report"]["words"] == "tools/move/zorble_map.toml"
    assert git(repo, "rev-parse", "HEAD") == base and git(repo, "status", "--porcelain") == ""
    assert len(git(repo, "worktree", "list").splitlines()) == 1


def test_a_map_the_base_lacks_is_refused_before_anything_runs(tmp_path, monkeypatch, capsys):
    repo, base = _repo(tmp_path, TREE)
    (repo / "tools/move/zorble_map.toml").write_text(WORDS)
    assert _stage(monkeypatch, repo, "tools/move/zorble_map.toml", "--base", base, "--branch", "staged") == 1
    assert "tools/move/zorble_map.toml isn't at" in capsys.readouterr().err
    assert len(git(repo, "worktree", "list").splitlines()) == 1
    assert git(repo, "branch", "--list", "staged") == ""


def test_a_map_outside_tools_move_or_of_both_kinds_or_neither_is_refused(tmp_path, monkeypatch, capsys):
    repo, base = _repo(tmp_path, {**TREE, "maps/zorble_map.toml": WORDS})
    assert _stage(monkeypatch, repo, "maps/zorble_map.toml", "--base", base) == 1
    assert "a map is a file directly in tools/move/" in capsys.readouterr().err
    with pytest.raises(ST.Refused, match=r"both \[words\] and \[paths\]"):
        ST.kind_of("m.toml", WORDS + PATHS.replace("[rewrite]", "[rewrite2]"))
    with pytest.raises(ST.Refused, match=r"neither \[words\] nor \[paths\]"):
        ST.kind_of("m.toml", '[rewrite]\nexclude = []\n')


def test_the_scripts_are_the_bases_so_a_base_whose_move_check_refuses_refuses_the_move(tmp_path, monkeypatch, capsys):
    """The base's `move_check.py` judges its move, whatever this checkout's says: here it refuses everything."""
    refuses = ('import json, sys\nprint(json.dumps({"certified": False, "refusals": [{"path": "a.py", "reason": "not a move"}]}))'
               '\nprint("move-check: refused", file=sys.stderr)\nsys.exit(1)\n')
    repo, base = _repo(tmp_path, {**TREE, "tools/move/zorble_map.toml": WORDS}, {ST.MOVE_CHECK: refuses})
    assert _stage(monkeypatch, repo, "tools/move/zorble_map.toml", "--base", base) == 1
    err = capsys.readouterr().err
    assert "isn't a pure move: move-check: refused\n  a.py: not a move\neach with its hunk: verity check move" in err


def test_a_map_a_second_run_changes_again_is_refused(tmp_path, monkeypatch, capsys):
    """`restack.py` carries branches across a move by rerunning it, so a map whose rerun renames again is refused."""
    again = WORDS.replace('"zorble" = "quux"', '"zorble" = "zorble zorble"')
    repo, base = _repo(tmp_path, {**TREE, "tools/move/zorble_map.toml": again})
    assert _stage(monkeypatch, repo, "tools/move/zorble_map.toml", "--base", base) == 1
    assert "tools/move/zorble_map.toml isn't idempotent" in capsys.readouterr().err


def test_a_map_that_renames_nothing_is_refused(tmp_path, monkeypatch, capsys):
    repo, base = _repo(tmp_path, {**TREE, "tools/move/none_map.toml": WORDS.replace("zorble", "frobnitz")})
    assert _stage(monkeypatch, repo, "tools/move/none_map.toml", "--base", base) == 1
    assert "changes nothing on" in capsys.readouterr().err


def test_the_branch_is_pushed_from_the_fetched_base_and_never_over_one_that_exists(tmp_path, monkeypatch, capsys):
    repo, base = _repo(tmp_path, {**TREE, "tools/move/zorble_map.toml": WORDS})
    remote = tmp_path / "remote.git"
    git(tmp_path, "init", "-q", "--bare", str(remote))
    git(repo, "remote", "add", "origin", str(remote))
    git(repo, "push", "-q", "origin", "main")
    monkeypatch.chdir(repo)
    assert ST.main(["tools/move/zorble_map.toml", "--branch", "staged", "--push"]) == 0
    head = git(remote, "rev-parse", "refs/heads/staged")
    assert git(repo, "rev-parse", "staged") == head and git(repo, "rev-parse", f"{head}^") == base
    git(repo, "branch", "-D", "staged")
    assert ST.main(["tools/move/zorble_map.toml", "--branch", "staged", "--push"]) == 1
    assert "origin has a branch staged" in capsys.readouterr().err
    assert git(remote, "rev-parse", "refs/heads/staged") == head
