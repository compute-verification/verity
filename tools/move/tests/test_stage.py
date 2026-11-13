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
    ".gitignore": "__pycache__/\n",  # as the tree's: the scripts import pins.py from the root they run at
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
    assert git(repo, "log", "-1", "--format=%B", p).endswith("\n\nLayout-move: tools/move/alpha_map.toml --no-lock")

    for m in (words, paths):
        r = MC.certify(m["parent"], m["commit"], cwd=str(repo))
        assert r["certified"] and m["report"]["certified"], r["refusals"]
        assert m["check"] == f"verity check move {m['parent']} {m['commit']}" and m["check"] in out
    assert words["report"]["words"] == "tools/move/zorble_map.toml"
    assert git(repo, "rev-parse", "HEAD") == base and git(repo, "status", "--porcelain") == ""
    assert len(git(repo, "worktree", "list").splitlines()) == 1


def test_an_open_branch_crosses_every_landed_move_with_one_restack_command(tmp_path, monkeypatch, capsys):
    """The window: two moves staged on main and landed as `research merge` lands a ref (`--no-ff`), then a commit after
    them; a branch from before them crosses both with `restack.py BRANCH`, which finds each by its move line and reruns its
    script on the branch's own changes, so what the branch added is renamed and moved too."""
    repo, base = _repo(tmp_path, {**TREE, "tools/move/zorble_map.toml": WORDS, "tools/move/alpha_map.toml": PATHS})
    git(repo, "checkout", "-q", "-b", "pr")
    (repo / "alpha/y.py").write_text("from alpha.x import f  # a zorble helper\n")
    (repo / "notes.md").write_text(TREE["notes.md"] + "Another zorble.\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "the branch's own change")
    git(repo, "checkout", "-q", "main")
    assert _stage(monkeypatch, repo, "tools/move/zorble_map.toml", "tools/move/alpha_map.toml", "--base", base,
                  "--no-lock", "--branch", "staged") == 0
    assert "restack.py BRANCH" in capsys.readouterr().out
    git(repo, "merge", "-q", "--no-ff", "-m", "land the moves", "staged")
    (repo / "later.md").write_text("after the moves\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "after the moves")

    assert ST.RS.main(["pr", "--main", "main", "--no-pins", "--json"]) == 0
    [r] = json.loads(capsys.readouterr().out)
    assert r["status"] == "clean" and r["crossed"] == [git(repo, "rev-parse", "staged~1"), git(repo, "rev-parse", "staged")]
    result = r["result"]
    assert git(repo, "merge-base", "--is-ancestor", "main", result) == ""
    assert _show(repo, result, "beta/y.py") == "from beta.x import f  # a quux helper"
    assert _show(repo, result, "notes.md") == "A quux is `zorble`.\nAnother quux."
    assert _show(repo, result, "later.md") == "after the moves"
    assert "alpha/y.py" not in git(repo, "ls-tree", "-r", "--name-only", result)
    assert git(repo, "rev-parse", "pr") == r["head"]


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
    with pytest.raises(ST.Refused, match=r"neither \[words\], \[\[words_in\]\] nor \[paths\]"):
        ST.kind_of("m.toml", '[rewrite]\nexclude = []\n')


def test_a_word_map_whose_words_are_all_scoped_stages_under_an_empty_words_table(tmp_path, monkeypatch):
    """A name two modules declare is renamed only where its own module reads it, by a `[[words_in]]` entry under an
    empty `[words]`: staged and certified, the other module's left as it was. A map without the `[words]` table is
    refused, naming it, since `rename.py` and `move_check.py` read it."""
    scoped = ('[words]\n\n[[words_in]]\nfiles = ["Z.lean"]\nwords = { "zorble_le" = "quux_le" }\n\n'
              '[rewrite]\nexclude = ["tools/**"]\n')
    lean = {"Z.lean": "namespace Z\ntheorem zorble_le : True := trivial\nexample : True := zorble_le\nend Z\n",
            "W.lean": "namespace W\ntheorem zorble_le : True := trivial\nend W\n"}
    repo, base = _repo(tmp_path, {**TREE, **lean, "tools/move/zorble_map.toml": scoped})
    cert = tmp_path / "cert.json"
    assert _stage(monkeypatch, repo, "tools/move/zorble_map.toml", "--base", base, "--certificate", str(cert)) == 0
    [m] = json.loads(cert.read_text())["moves"]
    assert m["report"]["certified"] and MC.certify(m["parent"], m["commit"], cwd=str(repo))["certified"]
    assert _show(repo, m["commit"], "Z.lean") == lean["Z.lean"].replace("zorble_le", "quux_le").strip()
    assert _show(repo, m["commit"], "W.lean") == lean["W.lean"].strip()
    with pytest.raises(ST.Refused, match=r"\[\[words_in\]\] but no \[words\]: give it an empty \[words\]"):
        ST.kind_of("m.toml", scoped.removeprefix("[words]\n\n"))


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


def test_a_path_map_that_acknowledges_a_split_reference_stages(tmp_path, monkeypatch, capsys):
    """A map that splits a directory acknowledges each reference to it in [split], and `layout.py` refuses an
    acknowledgement no reference needs. On the move's own commit every reference is rewritten, so none needs one: the
    second run takes the map as `restack.py` runs it on a branch (`--report-splits`), and still must change nothing.
    An entry of a list of paths needs none: the mover writes it as where its files went, so a row for it is refused."""
    split = PATHS.replace('"alpha/" = "beta/"\n', '"alpha/" = "beta/"\n"alpha/sub/" = "gamma/"\n').replace(
        '"alpha" = "beta"\n', '"alpha" = "beta"\n"alpha.sub" = "gamma"\n') + '\n[split]\n"gates.toml: alpha" = "the gate"\n'
    files = {**TREE, "alpha/sub/__init__.py": "", "alpha/sub/z.py": "Z = 1\n", "gates.toml": 'requires = "alpha/"\n',
             "tools/move/alpha_map.toml": split}
    repo, base = _repo(tmp_path, files)
    assert _stage(monkeypatch, repo, "tools/move/alpha_map.toml", "--base", base, "--no-lock", "--branch", "staged") == 0, \
        capsys.readouterr().err
    assert _show(repo, "staged", "gates.toml") == 'requires = "beta/"'
    assert _show(repo, "staged", "gamma/z.py") == "Z = 1"
    repo, base = _repo(tmp_path / "list", {**files, "gates.toml": 'requires = ["alpha/"]\n'})
    assert _stage(monkeypatch, repo, "tools/move/alpha_map.toml", "--base", base, "--no-lock", "--branch", "staged") == 1
    assert "[split] acknowledges what no reference needs any more: gates.toml: alpha" in capsys.readouterr().err


def test_a_path_map_that_would_delete_an_init_with_content_is_refused_naming_it_and_the_fix(tmp_path, monkeypatch, capsys):
    """The map takes the data sheet out of `gamma/`, whose `__init__.py` holds a docstring: the mover refuses, and the
    refusal names the file and the fix; no branch is made."""
    data = PATHS.replace('"alpha/" = "beta/"\n', '"gamma/g.toml" = "data/g.toml"\n').replace('"alpha" = "beta"\n', "")
    repo, base = _repo(tmp_path, {**TREE, "gamma/__init__.py": '"""Gamma."""\n', "gamma/g.toml": "g = 1\n",
                                  "tools/move/data_map.toml": data})
    assert _stage(monkeypatch, repo, "tools/move/data_map.toml", "--base", base, "--no-lock", "--branch", "staged") == 1
    err = capsys.readouterr().err
    assert "failed on" in err and "(exit 1)" in err and "comments and whitespace: gamma/__init__.py\n" in err
    assert err.rstrip().endswith("or delete it in a hand commit before the move")
    assert git(repo, "branch", "--list", "staged") == ""


def test_a_map_that_renames_nothing_is_refused(tmp_path, monkeypatch, capsys):
    repo, base = _repo(tmp_path, {**TREE, "tools/move/none_map.toml": WORDS.replace("zorble", "frobnitz")})
    assert _stage(monkeypatch, repo, "tools/move/none_map.toml", "--base", base) == 1
    assert "changes nothing on" in capsys.readouterr().err


def test_a_git_that_cant_commit_is_refused_before_anything_runs(tmp_path, monkeypatch, capsys):
    """A node's git has no identity: refused at once, not after the codemod's run."""
    repo, base = _repo(tmp_path, {**TREE, "tools/move/zorble_map.toml": WORDS})
    for k in ("GIT_AUTHOR_NAME", "GIT_AUTHOR_EMAIL", "GIT_COMMITTER_NAME", "GIT_COMMITTER_EMAIL"):
        monkeypatch.delenv(k)
    for k, v in {"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "user.useConfigOnly", "GIT_CONFIG_VALUE_0": "true"}.items():
        monkeypatch.setenv(k, v)
    assert _stage(monkeypatch, repo, "tools/move/zorble_map.toml", "--base", base, "--branch", "staged") == 1
    assert "git has no identity here to commit the moves as" in capsys.readouterr().err
    assert len(git(repo, "worktree", "list").splitlines()) == 1
    assert git(repo, "branch", "--list", "staged") == ""


def test_the_staged_branch_bundles_for_the_checkout_that_pushes_it(tmp_path, monkeypatch, capsys):
    """A staging run on a node hands its branch back as a bundle over BASE, which a checkout at BASE fetches whole."""
    repo, base = _repo(tmp_path, {**TREE, "tools/move/zorble_map.toml": WORDS})
    lead = tmp_path / "lead"
    git(tmp_path, "clone", "-q", str(repo), str(lead))
    bundle = tmp_path / "move.bundle"
    assert _stage(monkeypatch, repo, "tools/move/zorble_map.toml", "--base", base, "--branch", "staged",
                  "--bundle", str(bundle)) == 0
    assert f"bundled to {bundle}" in capsys.readouterr().out
    git(lead, "fetch", "-q", str(bundle), "staged:staged")
    assert git(lead, "rev-parse", "staged") == git(repo, "rev-parse", "staged")
    assert git(lead, "rev-parse", "staged^") == base


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
