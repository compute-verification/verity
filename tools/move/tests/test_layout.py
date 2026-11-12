"""`layout.py` itself on a toy tree whose map moves core's `verity.randomness`, `census/` and a recorded run's directory: what
it must leave alone."""
from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
LAYOUT = HERE.parent / "layout.py"
spec = importlib.util.spec_from_file_location("layout_under_test", LAYOUT)
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)

ENV = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t",
       "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}
MAP = """[roots]
old = { verity = "pkg/src/verity" }
new = { verity = "verity" }

[paths]
"pkg/src/verity/randomness/" = "verity/primitives/randomness/"
"census/" = "catalog/census/"
"runs/" = "data/runs/"

[modules]
"census" = "catalog.census"
"runs" = "data.runs"
"verity.randomness" = "verity.primitives.randomness"
"""
PROVER = '''//! Coins come from `verity.randomness.derive`; the hardware from census/hardware.json.
/* a block comment naming verity.randomness, and a quote " that opens nothing */
const TAG: &str = "sha256 (verity.randomness)";
const RAW: &str = r#"tag "verity.randomness" in a raw string"#;
fn f<'a>(x: &'a str) -> serde_json::Value {
    let _quote = '"';
    let _path = "census/hardware.json";
    json!({"census": 1, "coins": "verity.randomness"}) // verity.randomness after the literals
}
'''
TAGS = 'def coinDerivation : String := "sha256 (verity.randomness)"\n'


@pytest.fixture(autouse=True)
def _git_identity(monkeypatch):
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "commit.gpgsign=false", *args], cwd=repo, check=True, capture_output=True,
                          text=True).stdout


@pytest.fixture
def moved(tmp_path) -> Path:
    r = tmp_path / "repo"
    files = {"pkg/src/verity/__init__.py": "", "pkg/src/verity/randomness/__init__.py": "def derive():\n    return 0\n",
             "census/hardware.json": "{}\n", "prover.rs": PROVER, "Tags.lean": TAGS, ".gitignore": "out/\n",
             "runs/r1/out/session.txt": "a recorded session\n",
             "tools/research/src/research/store/moves.json": '{"moves": []}\n'}
    for p, t in files.items():
        (r / p).parent.mkdir(parents=True, exist_ok=True)
        (r / p).write_text(t)
    (tmp_path / "map.toml").write_text(MAP)
    git(r, "init", "-q", "-b", "main")
    git(r, "add", "-A")
    git(r, "add", "-f", "runs/r1/out/session.txt")
    git(r, "commit", "-q", "-m", "old")
    subprocess.run(["python3", str(LAYOUT), "--root", str(r), "--map", str(tmp_path / "map.toml"), "--no-lock"], check=True,
                   capture_output=True)
    return r


def test_a_second_run_on_the_moved_tree_changes_nothing(moved):
    """`restack.py` carries a branch across a move by rerunning it, and `stage.py` refuses a move whose rerun changes
    anything: with nothing of the map left to move, no second (empty) entry goes to `moves.json`."""
    git(moved, "commit", "-q", "-m", "moved")
    subprocess.run(["python3", str(LAYOUT), "--root", str(moved), "--map", str(moved.parent / "map.toml"), "--no-lock"],
                   check=True, capture_output=True)
    assert git(moved, "status", "--porcelain") == ""
    assert len(json.loads((moved / "tools/research/src/research/store/moves.json").read_text())["moves"]) == 1


def test_a_string_literal_in_rust_or_lean_keeps_its_text_and_only_a_path_in_one_moves(moved):
    rs = (moved / "prover.rs").read_text()
    assert 'const TAG: &str = "sha256 (verity.randomness)";' in rs, "a domain tag is protocol text"
    assert '"census": 1' in rs and '"coins": "verity.randomness"' in rs, "a JSON key and value keep their text"
    assert 'r#"tag "verity.randomness" in a raw string"#' in rs
    assert '"catalog/census/hardware.json"' in rs, "a path in a literal still names the file where it moved"
    assert "`verity.primitives.randomness.derive`" in rs and "catalog/census/hardware.json." in rs, "comments are rewritten"
    assert "naming verity.primitives.randomness, and a quote" in rs
    assert "// verity.primitives.randomness after the literals" in rs, "a char literal's quote and a lifetime open no string"
    assert (moved / "Tags.lean").read_text() == TAGS


def test_a_tracked_file_a_gitignore_matches_stays_tracked_where_it_moves(moved):
    tracked = set(git(moved, "ls-files").split())
    assert "data/runs/r1/out/session.txt" in tracked and "runs/r1/out/session.txt" not in tracked


LIMIT_MAP = '[roots]\nold = {}\nnew = {}\n\n[paths]\n"census/" = "catalog/census/"\n\n[modules]\n"census" = "catalog.census"\n'


def _limit_tree(root: Path, files: dict[str, str | bytes], rows: str = "") -> tuple[Path, list[str]]:
    """A tree whose map moves `census/`, and the arguments that move it."""
    r = root / "repo"
    for p, t in {"census/hardware.json": "{}\n", "tools/research/src/research/store/moves.json": '{"moves": []}\n', **files}.items():
        (r / p).parent.mkdir(parents=True, exist_ok=True)
        (r / p).write_bytes(t if isinstance(t, bytes) else t.encode())
    (root / "map.toml").write_text(LIMIT_MAP + rows)
    git(r, "init", "-q", "-b", "main")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "old")
    return r, ["--root", str(r), "--map", str(root / "map.toml"), "--no-lock"]


def test_a_lean_lock_over_the_limit_is_rewritten_as_one_under_it(tmp_path, monkeypatch):
    """A lock that grew past the limit moved with every old path in it, and the move said nothing."""
    lock = json.dumps({"shape": {"frozen": {"T": "reads census/hardware.json, " + "as it was frozen, " * 30}}}, indent=1) + "\n"
    under, argv = _limit_tree(tmp_path / "under", {"Security/lean-audit.json": lock})
    assert L.main(argv) == 0
    over, argv = _limit_tree(tmp_path / "over", {"Security/lean-audit.json": lock})
    monkeypatch.setattr(L, "TEXT_LIMIT", len(lock) - 1)
    assert L.main(argv) == 0
    assert "reads catalog/census/hardware.json, " in (over / "Security/lean-audit.json").read_text()
    assert (over / "Security/lean-audit.json").read_bytes() == (under / "Security/lean-audit.json").read_bytes()


def test_a_text_file_over_the_limit_stops_the_move_which_names_it_until_the_map_excludes_it(tmp_path, monkeypatch):
    """No file is left with its old paths for its size alone and passed over: text is what has no NUL in its first 8 KiB."""
    notes = "see census/hardware.json\n" * 20
    files = {"docs/notes.md": notes, "blob.bin": b"\0" + notes.encode()}
    monkeypatch.setattr(L, "TEXT_LIMIT", len(notes) - 1)
    _, argv = _limit_tree(tmp_path / "named", files)
    with pytest.raises(SystemExit, match=rf"\n  docs/notes\.md \({len(notes)} bytes\)\n") as stop:
        L.main(argv)
    assert "blob.bin" not in str(stop.value) and "exclude in the map" in str(stop.value)
    _, argv = _limit_tree(tmp_path / "dry", files)
    with pytest.raises(SystemExit, match=r"docs/notes\.md"):
        L.main([*argv, "--dry-run"])
    excluded, argv = _limit_tree(tmp_path / "excluded", files, '\n[rewrite]\nexclude = ["docs/notes.md"]\n')
    assert L.main(argv) == 0 and (excluded / "docs/notes.md").read_text() == notes
    monkeypatch.setattr(L, "TEXT_LIMIT", len(notes))
    at, argv = _limit_tree(tmp_path / "at", files)
    assert L.main(argv) == 0 and (at / "docs/notes.md").read_text() == notes.replace("census/", "catalog/census/")
    assert (at / "blob.bin").read_bytes() == files["blob.bin"]


def test_the_maps_paths_as_a_trie_match_what_the_list_of_them_longest_first_matches():
    """`_paths` finds the map's old paths with one regex over every file. Written as a list, `re` tries each path at every
    character: nine of a run's ten minutes on a draft of the one map. `alternation` writes the same set as a
    trie, and a branch crosses a move only if both find the same spans: the longest path that matches, or a shorter one
    where what follows refuses the longer (`a/b/c.py` before `a/b/c`, and `a/b` in `a/b/c-d`)."""
    words = ["a/b", "a/b/c", "a/b/c.py", "a/bc", "a.b", "a", "x-y", "x-y/z", "tools/x+y", "tools/x", "census", "(odd)/[p]"]
    assert L.alternation([]) == "" and re.fullmatch(L.alternation(["a.b"]), "a.b") and not re.fullmatch(L.alternation(["a.b"]), "axb")
    after = r"(?![\w\-]|\.\w)(/\w*)*"
    listed = re.compile("(" + "|".join(re.escape(w) for w in sorted(words, key=len, reverse=True)) + ")" + after)
    trie = re.compile("(" + L.alternation(words) + ")" + after)
    follows = ["", "/", ".", ".py", ".p y", "x", "-d", "/c", "/c.py", "/c-d", "c", ".b", "+y", "+", "/z", "/[p]"]
    cases = [f"{w}{f}" for w in words for f in follows] + [w[:i] for w in words for i in range(1, len(w))]
    for text in [*cases, " ".join(cases), "".join(cases)]:
        assert [(m.span(), m.groups()) for m in trie.finditer(text)] == [(m.span(), m.groups()) for m in listed.finditer(text)], text


SPLIT_MAP = """[roots]
old = {}
new = {}

[paths]
"backends/flock/" = "verity/ml/flock/"
"backends/flock/verifier/lean/" = "verity/core/service/flock/"

[modules]
"backends.flock" = "verity.ml.flock"
"backends.flock.verifier.lean" = "verity.core.service.flock"
"""


def _split_tree(tmp_path: Path, split: str, *args: str) -> subprocess.CompletedProcess:
    r = tmp_path / "repo"
    files = {"backends/flock/verifier/ci.py": "", "backends/flock/verifier/lean/lakefile.toml": "",
             "gates.toml": 'merge_requires = ["backends/flock/", "backends/flock/verifier/lean/"]\n',
             "check.py": 'ROOT = None\nVERIFIER = ROOT / "backends" / "flock" / "verifier"\n',
             "tools/research/src/research/store/moves.json": '{"moves": []}\n'}
    for p, t in files.items():
        (r / p).parent.mkdir(parents=True, exist_ok=True)
        (r / p).write_text(t)
    (tmp_path / "map.toml").write_text(SPLIT_MAP + split)
    git(r, "init", "-q", "-b", "main")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "old")
    return subprocess.run(["python3", str(LAYOUT), "--root", str(r), "--map", str(tmp_path / "map.toml"), "--no-lock",
                           "--dry-run", *args], capture_output=True, text=True)


def test_a_reference_to_a_directory_the_map_splits_fails_unless_split_acknowledges_it(tmp_path):
    """The rewrite of a split directory follows its larger part, so a gate naming `backends/flock/` would stop covering the
    Lean verifier that left for core/ (the Flock gates that failed open): each such reference is an error until [split]
    names it with its reason, and an acknowledgement nothing needs is an error too. A reference to the part that left is
    exact, and needs none. On a branch's tree (restack.py's `--report-splits`) they're noted instead: the owner's to check."""
    p = _split_tree(tmp_path / "bare", "")
    assert p.returncode != 0
    assert "gates.toml: backends/flock (" in p.stderr and "check.py: backends/flock/verifier (" in p.stderr, p.stderr
    assert "backends/flock/verifier/lean" not in p.stderr.replace("backends/flock/verifier (", "")
    ok = '\n[split]\n"gates.toml: backends/flock" = "the gate"\n"check.py: backends/flock/verifier" = "the scripts"\n'
    assert _split_tree(tmp_path / "acknowledged", ok).returncode == 0
    stale = _split_tree(tmp_path / "stale", ok + '"other.py: backends/flock" = "gone"\n')
    assert stale.returncode != 0 and "no reference needs any more: other.py: backends/flock" in stale.stderr
    branch = _split_tree(tmp_path / "branch", '\n[split]\n"other.py: backends/flock" = "gone"\n', "--report-splits")
    assert branch.returncode == 0 and "gates.toml: backends/flock (" in branch.stdout, branch.stderr


def test_a_relative_path_through_a_moved_directory_is_normalized_though_neither_end_moved(tmp_path):
    """Lake writes an inherited path requirement as its requirer's path joined to the requirement's own, so a manifest that
    stays can name a directory that stays through one that moves (`lms/lean/../../../../core/service/flock`); as written it
    would stop resolving. One that never leaves a moved directory is a path the move doesn't touch, and keeps its text."""
    r = tmp_path / "repo"
    manifest = '{"packages": [{"type": "path", "dir": "../old/lms/lean/../../../keep/flock"}, {"type": "path", "dir": "../keep/x/../flock"}]}\n'
    files = {"old/lms/lean/lakefile.toml": "", "keep/flock/lakefile.toml": "", "keep/x/a.txt": "", "pkg/lake-manifest.json": manifest,
             "tools/research/src/research/store/moves.json": '{"moves": []}\n'}
    for p, t in files.items():
        (r / p).parent.mkdir(parents=True, exist_ok=True)
        (r / p).write_text(t)
    (tmp_path / "map.toml").write_text('[roots]\nold = {}\nnew = {}\n\n[paths]\n"old/" = "new/"\n\n[modules]\n"old" = "new"\n')
    git(r, "init", "-q", "-b", "main")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "old")
    subprocess.run(["python3", str(LAYOUT), "--root", str(r), "--map", str(tmp_path / "map.toml"), "--no-lock"], check=True,
                   capture_output=True)
    assert (r / "pkg/lake-manifest.json").read_text() == manifest.replace("../old/lms/lean/../../../keep/flock", "../keep/flock")


LAKEFILE = '''name = "requirer"
path = "inner"

[[require]]
name = "inner"
path = "inner"

[[require]]
name = "deep"
path = "sub/deep"

[[lean_lib]]
name = "inner"
'''
LAKE_MANIFEST = '''{"packages": [{"type": "path", "name": "inner", "dir": "inner"},
              {"type": "path", "name": "deep", "dir": "sub/deep"},
              {"type": "git", "name": "inner", "dir": "inner"}]}
'''


def _lake_tree(tmp_path: Path, *maps: str, manifest: str = LAKE_MANIFEST) -> tuple[Path, list[subprocess.CompletedProcess]]:
    """A package `outer/` that requires two packages inside its own directory by paths with no `./`, as Lake takes them,
    after each of `maps` has run on it in turn."""
    r = tmp_path / "repo"
    files = {"outer/lakefile.toml": LAKEFILE, "outer/lake-manifest.json": manifest, "outer/inner/lakefile.toml": "",
             "outer/sub/deep/lakefile.lean": "", "tools/research/src/research/store/moves.json": '{"moves": []}\n'}
    for p, t in files.items():
        (r / p).parent.mkdir(parents=True, exist_ok=True)
        (r / p).write_text(t)
    git(r, "init", "-q", "-b", "main")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "old")
    runs = []
    for i, m in enumerate(maps):
        (tmp_path / f"map{i}.toml").write_text("[roots]\nold = {}\nnew = {}\n\n" + m)
        runs.append(subprocess.run(["python3", str(LAYOUT), "--root", str(r), "--map", str(tmp_path / f"map{i}.toml"),
                                    "--no-lock"], capture_output=True, text=True))
        git(r, "add", "-A")
        git(r, "commit", "-q", "--allow-empty", "-m", f"map {i}")
    return r, runs


def _lake_dirs(r: Path, package: str) -> tuple[list[str], list[str]]:
    return (L.lake_dirs("lakefile.toml", (r / package / "lakefile.toml").read_text()),
            L.lake_dirs("lake-manifest.json", (r / package / "lake-manifest.json").read_text()))


def test_a_lake_package_inside_the_one_that_requires_it_is_followed_when_it_leaves(tmp_path):
    """Lake reads a requirement's `path` and a manifest's `dir` from the package's own directory, with or without `./`, and a
    package inside the requiring one is written as its bare name (`path = "composition"`). No other reader takes one word
    for a path: left as written, the commit is certified a pure move, since only paths changed, of a tree whose package
    can no longer find its requirement. Only those two are read so: a `name`, a `path` outside a requirement and a fetched
    package's `dir` stay. A later map that moves the requirer rewrites the path again, and each map run again changes
    nothing."""
    leave = ('[paths]\n"outer/inner/" = "apart/inner/"\n"outer/sub/deep/" = "apart/deep/"\n\n'
             '[modules]\n"outer.inner" = "apart.inner"\n"outer.sub.deep" = "apart.deep"\n')
    away = '[paths]\n"outer/" = "far/away/outer/"\n\n[modules]\n"outer" = "far.away.outer"\n'
    r, (first,) = _lake_tree(tmp_path / "one", leave)
    assert first.returncode == 0, first.stderr
    assert _lake_dirs(r, "outer") == (["../apart/inner", "../apart/deep"], ["../apart/inner", "../apart/deep"])
    assert (r / "outer/lakefile.toml").read_text() == LAKEFILE.replace('name = "inner"\npath = "inner"',
                                                                       'name = "inner"\npath = "../apart/inner"') \
        .replace('"sub/deep"', '"../apart/deep"')
    assert '{"type": "git", "name": "inner", "dir": "inner"}' in (r / "outer/lake-manifest.json").read_text()
    r, runs = _lake_tree(tmp_path / "two", leave, away, leave, away)
    assert [p.returncode for p in runs] == [0, 0, 0, 0], [p.stderr for p in runs]
    assert _lake_dirs(r, "far/away/outer") == (["../../../apart/inner", "../../../apart/deep"],) * 2
    assert git(r, "diff", "--stat", "HEAD~2", "HEAD") == ""


def test_a_lake_package_that_moves_with_the_one_that_requires_it_keeps_its_path(tmp_path):
    r, (run,) = _lake_tree(tmp_path, '[paths]\n"outer/" = "far/outer/"\n\n[modules]\n"outer" = "far.outer"\n')
    assert run.returncode == 0, run.stderr
    assert _lake_dirs(r, "far/outer") == (["inner", "sub/deep"], ["inner", "sub/deep"])
    assert (r / "far/outer/lakefile.toml").read_text() == LAKEFILE


def test_a_move_that_leaves_a_lake_path_naming_no_package_stops(tmp_path):
    """A map can still part a package from the path that names it: by excluding the requirer's Lake files from the rewrite.
    The certificate would pass, since nothing but paths changed, and the Lean build would fail in a candidate; so the mover
    stops and names the file and the path."""
    r, (run,) = _lake_tree(tmp_path, '[paths]\n"outer/inner/" = "apart/inner/"\n\n[modules]\n"outer.inner" = "apart.inner"\n\n'
                                     '[rewrite]\nexclude = ["outer/*"]\n')
    assert run.returncode != 0
    assert "outer/lakefile.toml: inner (it named outer/inner)" in run.stderr, run.stderr
    assert "outer/lake-manifest.json: inner (it named outer/inner)" in run.stderr
    assert "sub/deep" not in run.stderr


def test_a_lake_file_that_does_not_parse_stops_the_move(tmp_path):
    """The mover can't say that a Lake file it can't read still names its packages, so it stops and names the file rather
    than take it to hold no path. A Lake file of a shape Lake wouldn't write holds none."""
    _, (run,) = _lake_tree(tmp_path, '[paths]\n"outer/inner/" = "apart/inner/"\n\n[modules]\n"outer.inner" = "apart.inner"\n',
                           manifest='{"packages": [')
    assert run.returncode != 0
    assert "outer/lake-manifest.json: not a Lake file the mover can read" in run.stderr, run.stderr
    assert L.lake_dirs("lake-manifest.json", "") == [] == L.lake_dirs("lake-manifest.json", '{"packages": {"dir": "x"}}')
    assert L.lake_dirs("lakefile.toml", 'require = "inner"\n') == [] == L.lake_dirs("README.md", "not TOML [")
    assert L.lake_dirs("lake-manifest.json", '[{"type": "path", "dir": "x"}]') == []


def test_a_subpackage_renamed_under_the_same_parent_is_renamed_where_it_is_imported_from_that_parent(tmp_path):
    """`from pkg import old as O`, where `pkg.old` becomes `pkg.new`: the statement's module stays `pkg`, only the imported
    name changes, and an unchanged module is no reason to leave the statement as it was (the vLLM move left 30 such
    `from verity_vllm import protocol_options as PO` lines, and one in a script's string, naming a package that no longer
    exists)."""
    r = tmp_path / "repo"
    files = {"pkg/__init__.py": "", "pkg/old/__init__.py": "X = 1\n", "pkg/keep.py": "Y = 2\n",
             "use.py": "from pkg import old as O\nfrom pkg import keep, old\nfrom pkg import keep as K\n",
             "run.py": 'SCRIPT = """\nfrom pkg import old as O\nfrom pkg import keep\n"""\n',
             "tools/research/src/research/store/moves.json": '{"moves": []}\n'}
    for p, t in files.items():
        (r / p).parent.mkdir(parents=True, exist_ok=True)
        (r / p).write_text(t)
    (tmp_path / "map.toml").write_text('[roots]\nold = { pkg = "pkg" }\nnew = { pkg = "pkg" }\n\n[paths]\n"pkg/old/" = "pkg/new/"'
                                       '\n\n[modules]\n"pkg.old" = "pkg.new"\n')
    git(r, "init", "-q", "-b", "main")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "old")
    subprocess.run(["python3", str(LAYOUT), "--root", str(r), "--map", str(tmp_path / "map.toml"), "--no-lock"], check=True,
                   capture_output=True)
    assert (r / "use.py").read_text() == "from pkg import new as O\nfrom pkg import keep, new as old\nfrom pkg import keep as K\n"
    assert (r / "run.py").read_text() == 'SCRIPT = """\nfrom pkg import new as O\nfrom pkg import keep\n"""\n', "and in a string"


def test_a_module_whose_row_keeps_it_stays_importable_from_its_package_when_the_package_moves(tmp_path):
    """A package's `__init__.py` and part of its modules move, and a row with old == new keeps another where it is (the fold
    half of `engine/profiles/` moving to `observe/fold/profiles/` with the model facts staying): `from pkg.prof import facts`
    still names `pkg.prof`, and only the names that moved follow the package."""
    r = tmp_path / "repo"
    files = {"pkg/__init__.py": "", "pkg/obs/__init__.py": "", "pkg/prof/__init__.py": "from . import facts, fold\n\n\ndef f():\n    return 0\n",
             "pkg/prof/fold.py": "", "pkg/prof/facts.py": "",
             "use.py": "from pkg.prof import facts, fold\nfrom pkg.prof import facts as F\nfrom pkg.prof import f\nimport pkg.prof.facts\n",
             "run.py": 'SCRIPT = "from pkg.prof import facts, fold"\n',
             "tools/research/src/research/store/moves.json": '{"moves": []}\n'}
    for p, t in files.items():
        (r / p).parent.mkdir(parents=True, exist_ok=True)
        (r / p).write_text(t)
    (tmp_path / "map.toml").write_text(
        '[roots]\nold = { pkg = "pkg" }\nnew = { pkg = "pkg" }\n\n[paths]\n"pkg/prof/__init__.py" = "pkg/obs/prof/__init__.py"\n'
        '"pkg/prof/fold.py" = "pkg/obs/prof/fold.py"\n"pkg/prof/facts.py" = "pkg/prof/facts.py"\n\n[modules]\n'
        '"pkg.prof" = "pkg.obs.prof"\n"pkg.prof.facts" = "pkg.prof.facts"\n"pkg.prof.fold" = "pkg.obs.prof.fold"\n')
    git(r, "init", "-q", "-b", "main")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "old")
    subprocess.run(["python3", str(LAYOUT), "--root", str(r), "--map", str(tmp_path / "map.toml"), "--no-lock"], check=True,
                   capture_output=True)
    assert (r / "use.py").read_text() == ("from pkg.prof import facts\nfrom pkg.obs.prof import fold\nfrom pkg.prof import facts as F\n"
                                          "from pkg.obs.prof import f\nimport pkg.prof.facts\n")
    assert (r / "pkg/obs/prof/__init__.py").read_text().startswith("from pkg.prof import facts\nfrom pkg.obs.prof import fold\n")
    assert (r / "pkg/prof/facts.py").exists() and (r / "pkg/prof/__init__.py").exists()
    assert (r / "run.py").read_text() == 'SCRIPT = "from pkg.prof import facts; from pkg.obs.prof import fold"\n', "and in a string"


def test_a_distributions_tests_path_follows_the_move_and_moves_json_records_only_rows_the_tree_has(tmp_path):
    """In a moved file below a distribution, `tests/old/t.py` is that distribution's (the vLLM tp2 fixtures' README names
    `tests/collectives/...` from `integrations/vllm/`), and the root's `tests/` stays. A map row for a file an open branch
    adds is no row of this tree's move: `moves.json` would name a path the tree lacks."""
    r = tmp_path / "repo"
    files = {"dist/pyproject.toml": "[project]\nname = 'dist'\n", "dist/tests/__init__.py": "",
             "dist/tests/old/__init__.py": "", "dist/tests/old/t.py": "",
             "dist/tests/old/fixtures/README.md": "Inputs of `tests/old/t.py`, beside `tests/test_root.py`.\n",
             "tests/test_root.py": "", "tools/research/src/research/store/moves.json": '{"moves": []}\n'}
    for p, t in files.items():
        (r / p).parent.mkdir(parents=True, exist_ok=True)
        (r / p).write_text(t)
    (tmp_path / "map.toml").write_text(
        '[roots]\nold = {}\nnew = {}\n\n[paths]\n"dist/tests/old/" = "dist/tests/new/"\n"dist/tests/later.py" = '
        '"dist/tests/new/later.py"\n\n[modules]\n"dist.tests.later" = "dist.tests.new.later"\n"dist.tests.old" = "dist.tests.new"\n'
        '\n[rewrite]\n'
        'include = ["**/fixtures/README.md"]\n')
    git(r, "init", "-q", "-b", "main")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "old")
    subprocess.run(["python3", str(LAYOUT), "--root", str(r), "--map", str(tmp_path / "map.toml"), "--no-lock"], check=True,
                   capture_output=True)
    assert (r / "dist/tests/new/fixtures/README.md").read_text() == \
        "Inputs of `tests/new/t.py`, beside `tests/test_root.py`.\n"
    moves = json.loads((r / "tools/research/src/research/store/moves.json").read_text())["moves"]
    assert moves == [{"paths": {"dist/tests/old/": "dist/tests/new/"}}]


def test_uv_lock_names_a_moved_distribution_by_its_new_directory(tmp_path):
    """A map that moves a distribution and renames no member (the machine model to `verity/hardware/`): `uv lock` reads the
    locked distribution's metadata at its old directory and fails, and `uv sync` with it, so every suite of the moved tree
    fails before it starts. The mover rewrites the directory, under `--no-lock` too; one that stays keeps its line, and
    so does a second run."""
    r = tmp_path / "repo"
    lock = ('[manifest]\nmembers = ["model", "other"]\n\n[[package]]\nname = "model"\nsource = { editable = "bench/model" }\n\n'
            '[[package]]\nname = "probe"\nsource = { virtual = "bench/model/probe" }\n\n'
            '[[package]]\nname = "other"\nsource = { editable = "bench/other" }\n\n'
            '[package.metadata]\nrequires-dist = [{ name = "model", editable = "bench/model" }]\n')
    files = {"bench/model/pyproject.toml": "[project]\nname = 'model'\n", "bench/model/probe/pyproject.toml": "[project]\nname = 'probe'\n",
             "bench/other/pyproject.toml": "[project]\nname = 'other'\n", "uv.lock": lock,
             "tools/research/src/research/store/moves.json": '{"moves": []}\n'}
    for p, t in files.items():
        (r / p).parent.mkdir(parents=True, exist_ok=True)
        (r / p).write_text(t)
    (tmp_path / "map.toml").write_text('[roots]\nold = {}\nnew = {}\n\n[paths]\n"bench/model/" = "hardware/model/"\n\n[modules]\n'
                                       '"bench.model" = "hardware.model"\n\n[rewrite]\nexclude = ["uv.lock"]\n')
    git(r, "init", "-q", "-b", "main")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "old")
    run = ["python3", str(LAYOUT), "--root", str(r), "--map", str(tmp_path / "map.toml"), "--no-lock"]
    subprocess.run(run, check=True, capture_output=True)
    want = lock.replace('"bench/model/probe"', '"hardware/model/probe"').replace('"bench/model"', '"hardware/model"')
    assert (r / "uv.lock").read_text() == want and '"bench/other"' in want
    git(r, "commit", "-q", "-m", "moved")
    subprocess.run(run, check=True, capture_output=True)
    assert git(r, "status", "--porcelain") == ""


def test_a_failed_uv_lock_fails_the_move(tmp_path, monkeypatch):
    """A move whose `uv lock` failed left a lock `uv sync` fails on, and exited 0, so `stage.py` certified a commit no suite
    could start on. It exits 1 with uv's reason, which `stage.py` and `restack.py` refuse on."""
    r = tmp_path / "repo"
    files = {"old/a.txt": "", "tools/verity/check/pod_setup.sh": "UV_VERSION=${UV_VERSION:-9.9.9}\n",
             "tools/research/src/research/store/moves.json": '{"moves": []}\n'}
    for p, t in files.items():
        (r / p).parent.mkdir(parents=True, exist_ok=True)
        (r / p).write_text(t)
    uv = tmp_path / "bin" / "uv"
    uv.parent.mkdir()
    uv.write_text('#!/bin/sh\n[ "$1" = --version ] && echo "uv 9.9.9" && exit 0\necho "error: Distribution not found at: old" >&2\nexit 2\n')
    uv.chmod(0o755)
    (tmp_path / "map.toml").write_text('[roots]\nold = {}\nnew = {}\n\n[paths]\n"old/" = "new/"\n\n[modules]\n"old" = "new"\n')
    git(r, "init", "-q", "-b", "main")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "old")
    monkeypatch.setenv("PATH", str(uv.parent), prepend=os.pathsep)
    p = subprocess.run(["python3", str(LAYOUT), "--root", str(r), "--map", str(tmp_path / "map.toml")], capture_output=True,
                       text=True)
    assert p.returncode == 1, p.stderr
    assert "Distribution not found at: old" in p.stdout and "`uv sync` fails on it" in p.stderr


def test_a_shell_scripts_paths_from_its_own_directory_are_recounted_from_where_it_moved(tmp_path):
    """The streamed wipe's `demo.sh` goes from three directories down to two (`examples/wipe/`): its climb to the root through
    `$0`, `${BASH_SOURCE[0]}` or a variable it sets to its own path is recounted, as is a path past the root to a file that
    moved; a script that stays keeps its climb, and a `dirname` of anything else (`$NVCC`) is no anchor."""
    r = tmp_path / "repo"
    demo = ('#!/usr/bin/env bash\nSELF=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")\n'
            'ROOT=$(cd "$(dirname "$SELF")/../../.." && pwd)\ncd "$(dirname "${BASH_SOURCE[0]}")/../../.."\n'
            'bash "$(dirname "$0")/../stay/run.sh"\nCUDA=$(cd "$(dirname "$NVCC")/.." && pwd)\n')
    stay = 'cd "$(dirname "$0")/../../.."\nbash "$(dirname "$0")/../wipe/demo.sh"\n'
    files = {"bench/pous/wipe/demo.sh": demo, "bench/pous/stay/run.sh": stay,
             "tools/research/src/research/store/moves.json": '{"moves": []}\n'}
    for p, t in files.items():
        (r / p).parent.mkdir(parents=True, exist_ok=True)
        (r / p).write_text(t)
    (tmp_path / "map.toml").write_text('[roots]\nold = { pkg = "pkg" }\nnew = { pkg = "pkg" }\n\n[paths]\n'
                                       '"bench/pous/wipe/demo.sh" = "examples/wipe/demo.sh"\n\n[modules]\n')
    git(r, "init", "-q", "-b", "main")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "old")
    subprocess.run(["python3", str(LAYOUT), "--root", str(r), "--map", str(tmp_path / "map.toml"), "--no-lock"], check=True,
                   capture_output=True)
    assert (r / "examples/wipe/demo.sh").read_text() == (
        demo.replace('"$SELF")/../../..', '"$SELF")/../..').replace('"${BASH_SOURCE[0]}")/../../..', '"${BASH_SOURCE[0]}")/../..')
        .replace('"$0")/../stay/run.sh', '"$0")/../../bench/pous/stay/run.sh'))
    assert (r / "bench/pous/stay/run.sh").read_text() == stay.replace('"$0")/../wipe/demo.sh', '"$0")/../../../examples/wipe/demo.sh')


def test_a_path_after_a_variable_a_shell_script_sets_to_its_own_directory_is_recounted(tmp_path):
    """`here=$(cd "$(dirname "$0")" && pwd)` is the script's own directory, so a path after `$here` is one from there: the
    climb to the root loses the level the script lost, and the script beside it that stayed is named from the new place.
    A file in the script's directory goes with it. A path after a variable the script sets again, or after any other
    (`$out`), is left."""
    demo = ('#!/usr/bin/env bash\nhere=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)\nroot=$(cd "$here/../../.." && pwd)\n'
            'bash "${here}/../stay/run.sh" "$here/notes.txt" "$out/../stay/run.sh"\n')
    again = '#!/bin/sh\nhere=$(cd "$(dirname "$0")" && pwd)\nhere=$1\nbash "$here/../stay/run.sh"\n'
    r, p = _run(tmp_path, {"bench/pous/wipe/demo.sh": demo, "bench/pous/wipe/again.sh": again, "bench/pous/wipe/notes.txt": "n\n",
                           "bench/pous/stay/run.sh": "#!/bin/sh\n"},
                '[roots]\nold = { pkg = "pkg" }\nnew = { pkg = "pkg" }\n\n[paths]\n"bench/pous/wipe/" = "examples/wipe/"\n\n[modules]\n'
                '"bench.pous.wipe" = "examples.wipe"\n')
    assert p.returncode == 0, p.stderr
    assert (r / "examples/wipe/demo.sh").read_text() == demo.replace('"$here/../../.."', '"$here/../.."').replace(
        '"${here}/../stay/run.sh"', '"${here}/../../bench/pous/stay/run.sh"')
    assert (r / "examples/wipe/again.sh").read_text() == again


WORDS_PY = ('"""The packages of `Analysis` and `Analysis/Proofs`; Analysis in prose is a word."""\n'
            'ROOT = None\nSPEC = ROOT / "Analysis"\nCAPS = {"Analysis": 3, "Analysis/Proofs": 4, \'Analysis\': 5}\n'
            'HEAVY = [p for p in CAPS if p.split("/")[0] == "Analysis"]\nNAME = "Analysis is a title"\n')
WORDS_MAP = ('[roots]\nold = {}\nnew = {}\n\n[paths]\n"Analysis/" = "analysis/"\n\n[modules]\n"Analysis" = "analysis"\n')


def _words_tree(tmp_path: Path, rewrite: str, more: dict[str, str] | None = None) -> Path:
    r = tmp_path / "repo"
    files = {"Analysis/Core.lean": "-- stated in `Analysis`\ndef dir : String := \"Analysis\"\n", "caps.py": WORDS_PY,
             "kinds.toml": 'inputs = ["Analysis", "tools"]  # `Analysis`, and Analysis alone\n',
             "README.md": "The spec is `Analysis`; Analysis's lock is `Analysis/lean-audit.json`.\n",
             "tag.rs": 'const DIR: &str = "Analysis";\n',
             "tools/research/src/research/store/moves.json": '{"moves": []}\n', **(more or {})}
    for p, t in files.items():
        (r / p).parent.mkdir(parents=True, exist_ok=True)
        (r / p).write_text(t)
    (tmp_path / "map.toml").write_text(WORDS_MAP + rewrite)
    git(r, "init", "-q", "-b", "main")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "old")
    subprocess.run(["python3", str(LAYOUT), "--root", str(r), "--map", str(tmp_path / "map.toml"), "--no-lock"], check=True,
                   capture_output=True)
    return r


def test_a_moved_directorys_name_alone_is_a_name_in_python_unless_the_map_lists_it_under_words(tmp_path):
    """`Security/` to `security/`: the directory is a package's path wherever it stands alone in quotes (a ratchet's key, a
    tuple of packages, `ROOT / "Security"`), which the default leaves in Python, where a lone quoted word is as often a key
    or a run's own directory. Without [rewrite] words only the paths with a slash and the data file's entry follow the move,
    which leaves `{"Analysis": 3, "analysis/Proofs": 4}`; with it the name alone between quotes or backticks follows too, in
    every file but Rust and Lean, and the bare word in prose or in a longer string stays."""
    r = _words_tree(tmp_path / "default", "")
    assert (r / "caps.py").read_text() == WORDS_PY.replace("Analysis/Proofs", "analysis/Proofs")
    assert (r / "kinds.toml").read_text() == 'inputs = ["analysis", "tools"]  # `Analysis`, and Analysis alone\n'
    r = _words_tree(tmp_path / "words", '\n[rewrite]\nwords = ["Analysis"]\n')
    assert (r / "caps.py").read_text() == (
        '"""The packages of `analysis` and `analysis/Proofs`; Analysis in prose is a word."""\n'
        'ROOT = None\nSPEC = ROOT / "analysis"\nCAPS = {"analysis": 3, "analysis/Proofs": 4, \'analysis\': 5}\n'
        'HEAVY = [p for p in CAPS if p.split("/")[0] == "analysis"]\nNAME = "Analysis is a title"\n')
    assert (r / "kinds.toml").read_text() == 'inputs = ["analysis", "tools"]  # `analysis`, and Analysis alone\n'
    assert (r / "README.md").read_text() == "The spec is `analysis`; Analysis's lock is `analysis/lean-audit.json`.\n"
    assert (r / "tag.rs").read_text() == 'const DIR: &str = "Analysis";\n', "a Rust literal is protocol text"
    assert (r / "analysis/Core.lean").read_text() == "-- stated in `Analysis`\ndef dir : String := \"Analysis\"\n"


def test_a_key_made_of_a_word_and_a_moved_directory_follows_where_the_map_lists_its_file(tmp_path):
    """The caps' room is keyed `"CAP package"`: `"INLINE Security"` beside `"INLINE Security/Properties"`. The path with a
    slash follows the move, and the name alone as a string's last word only in a file [rewrite] word_ends lists: a
    reason that ends "nothing of Security" elsewhere is prose."""
    text = 'ROOM = {"INLINE Analysis": 3, "INLINE Analysis/Props": 2}\nWHY = "nothing of Analysis"\n'
    r = _words_tree(tmp_path, '\n[rewrite]\nwords = ["Analysis"]\nword_ends = {"room.py" = ["Analysis"]}\n',
                    {"room.py": text, "why.py": text})
    assert (r / "room.py").read_text() == text.replace("Analysis", "analysis")
    assert (r / "why.py").read_text() == text.replace("Analysis/Props", "analysis/Props")


def test_a_path_that_is_a_shell_variables_default_follows_the_move(tmp_path):
    """`${DEFS:-flock/pod/defs.json}` names a file as any path does. The `-` before it belongs to the expansion (elsewhere a
    `-` before a path makes the path part of a longer word, which is left alone), and the `}` after it closes the
    expansion, so the file's own row is followed, not its directory's: `flock/pod/` is split here, and a reference to it
    would be refused."""
    r = tmp_path / "repo"
    run = ('export RESOLVER=${RESOLVER:-flock/pod/catalog.py:definition}\nDEFS=$REPO/${DEFS:-flock/pod/defs.json}\n'
           'T=${TESTS:-flock/tests/test_a.py flock/tests/test_b.py}\nU=${UNSET-flock/pod/catalog.py}\n'
           'W=pre-flock/pod/catalog.py\n')
    files = {"flock/pod/run.sh": run, "flock/pod/catalog.py": "", "flock/pod/defs.json": "{}\n", "flock/tests/test_a.py": "",
             "flock/tests/test_b.py": "", "tools/research/src/research/store/moves.json": '{"moves": []}\n'}
    for p, t in files.items():
        (r / p).parent.mkdir(parents=True, exist_ok=True)
        (r / p).write_text(t)
    (tmp_path / "map.toml").write_text(
        '[roots]\nold = {}\nnew = {}\n\n[paths]\n"flock/pod/catalog.py" = "prover/catalog.py"\n'
        '"flock/pod/defs.json" = "prover/data/defs.json"\n"flock/tests/" = "prover/tests/"\n\n[modules]\n'
        '"flock.pod.catalog" = "prover.catalog"\n"flock.tests" = "prover.tests"\n')
    git(r, "init", "-q", "-b", "main")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "old")
    p = subprocess.run(["python3", str(LAYOUT), "--root", str(r), "--map", str(tmp_path / "map.toml"), "--no-lock"],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert (r / "flock/pod/run.sh").read_text() == (
        'export RESOLVER=${RESOLVER:-prover/catalog.py:definition}\nDEFS=$REPO/${DEFS:-prover/data/defs.json}\n'
        'T=${TESTS:-prover/tests/test_a.py prover/tests/test_b.py}\nU=${UNSET-prover/catalog.py}\n'
        'W=pre-flock/pod/catalog.py\n')


def test_the_brace_that_closes_a_shell_default_is_no_part_of_its_path(tmp_path):
    """`${T:-flock/tests/test_a.py}`: the file has no row of its own, so the path is read from its directory's row on,
    and that reading takes braces in (`tests/{a,b}.py` is a path too). The `}` no `{` of the path opens closes the
    expansion. Read as part of the path it makes `test_a.py}` a pattern under `flock/tests/`, which this map splits
    (`data.json` goes elsewhere), and a reference to a split directory stops the move."""
    r = tmp_path / "repo"
    files = {"run.sh": 'T=${T:-flock/tests/test_a.py}\nU="${U:-flock/tests/test_a.py}"\n', "flock/tests/test_a.py": "",
             "flock/tests/data.json": "{}\n", "tools/research/src/research/store/moves.json": '{"moves": []}\n'}
    for p, t in files.items():
        (r / p).parent.mkdir(parents=True, exist_ok=True)
        (r / p).write_text(t)
    (tmp_path / "map.toml").write_text(
        '[roots]\nold = {}\nnew = {}\n\n[paths]\n"flock/tests/" = "prover/tests/"\n'
        '"flock/tests/data.json" = "data/data.json"\n\n[modules]\n"flock.tests" = "prover.tests"\n')
    git(r, "init", "-q", "-b", "main")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "old")
    p = subprocess.run(["python3", str(LAYOUT), "--root", str(r), "--map", str(tmp_path / "map.toml"), "--no-lock"],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr
    assert (r / "run.sh").read_text() == 'T=${T:-prover/tests/test_a.py}\nU="${U:-prover/tests/test_a.py}"\n'


def test_a_brace_list_whose_paths_go_two_ways_is_split(tmp_path):
    """`flock/{tool.sh,verifier/tool.sh}` names two files. A list is carried by the row of the directory before it, which
    is right while every path of it goes where that directory goes (`flock/{a,b}.sh`, and a pattern such as
    `flock/k{1,2}.json`, which names no file). Here `flock/verifier/` has a row of its own: carried by the directory's
    row the list would name `new/prover/verifier/tool.sh`, which exists nowhere, and the move would still exit 0. A
    second map that moves one path of a list the first one split finds it there."""
    r = tmp_path / "repo"
    toml = 'flock = ["flock/{tool.sh,verifier/tool.sh}", "flock/{a,b}.sh", "flock/k{1,2}.json", "flock/{a.sh,verifier}"]\n'
    files = {"boundary.toml": toml, "flock/tool.sh": "", "flock/verifier/tool.sh": "", "flock/verifier/x.sh": "",
             "flock/a.sh": "", "flock/b.sh": "", "tools/research/src/research/store/moves.json": '{"moves": []}\n'}
    for p, t in files.items():
        (r / p).parent.mkdir(parents=True, exist_ok=True)
        (r / p).write_text(t)
    rows = '"flock/tool.sh" = "new/prover/tool.sh"\n"flock/a.sh" = "new/prover/a.sh"\n"flock/b.sh" = "new/prover/b.sh"\n'
    head = '[roots]\nold = {}\nnew = {}\n\n[paths]\n'
    (tmp_path / "one.toml").write_text(head + rows + '"flock/verifier/" = "new/agreement/"\n\n'
                                       '[tokens]\n"flock/" = "new/prover/"\n\n[modules]\n"flock.verifier" = "new.agreement"\n')
    (tmp_path / "first.toml").write_text(head + rows + '\n[tokens]\n"flock/" = "new/prover/"\n\n[modules]\n')
    (tmp_path / "second.toml").write_text(head + '"flock/verifier/" = "new/agreement/"\n\n[modules]\n'
                                          '"flock.verifier" = "new.agreement"\n')
    git(r, "init", "-q", "-b", "main")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "old")
    want = ('flock = ["{new/prover/tool.sh,new/agreement/tool.sh}", "new/prover/{a,b}.sh", "new/prover/k{1,2}.json", '
            '"{new/prover/a.sh,new/agreement}"]\n')

    def move(name: str) -> dict:
        p = subprocess.run(["python3", str(LAYOUT), "--root", str(r), "--map", str(tmp_path / name), "--no-lock",
                            "--report-splits", "--report", str(tmp_path / "report.json")], capture_output=True, text=True)
        assert p.returncode == 0, p.stdout + p.stderr
        return json.loads((tmp_path / "report.json").read_text())["notes"]

    assert move("one.toml")["brace lists split"] == [
        "boundary.toml: flock/{tool.sh,verifier/tool.sh} -> {new/prover/tool.sh,new/agreement/tool.sh}",
        "boundary.toml: flock/{a.sh,verifier} -> {new/prover/a.sh,new/agreement}"]
    assert (r / "boundary.toml").read_text() == want
    assert (r / "new/agreement/tool.sh").is_file() and (r / "new/prover/tool.sh").is_file()
    git(r, "reset", "-q", "--hard")
    git(r, "clean", "-fdq")
    move("first.toml")
    assert '"{new/prover/tool.sh,flock/verifier/tool.sh}"' in (r / "boundary.toml").read_text()
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "first")
    move("second.toml")
    assert (r / "boundary.toml").read_text() == want


def _run(tmp_path: Path, files: dict[str, str], layout: str, *args: str) -> tuple[Path, subprocess.CompletedProcess]:
    """The mover's run by the map ``layout`` on a new repository of ``files``."""
    r = tmp_path / "repo"
    for p, t in {"tools/research/src/research/store/moves.json": '{"moves": []}\n', **files}.items():
        (r / p).parent.mkdir(parents=True, exist_ok=True)
        (r / p).write_text(t)
    (tmp_path / "map.toml").write_text(layout)
    git(r, "init", "-q", "-b", "main")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "old")
    return r, subprocess.run(["python3", str(LAYOUT), "--root", str(r), "--map", str(tmp_path / "map.toml"), "--no-lock", *args],
                             capture_output=True, text=True)


LEVELS_MAP = """[roots]
old = { verity = "verity" }
new = { verity = "verity" }

[paths]
"benchmarks/live/" = "verity/composite/device/live/"
"verity/core/wire/" = "verity/configuration/wire/"
"verity/core/scripts/" = "verity/configuration/scripts/"

[modules]
"benchmarks.live" = "verity.composite.device.live"
"verity.core.wire" = "verity.configuration.wire"
"verity.core.scripts" = "verity.configuration.scripts"
"""


def test_a_module_moved_into_a_package_gets_each_level_and_an_emptied_package_loses_its_marks(tmp_path):
    """`benchmarks/live/` was no package, and below `verity/` its modules import by name only if each directory up to the
    root is one; its tests, run by path, get no level, and neither do scripts that a package held as no module of its
    own. `verity/core/` then holds only the `__init__.py` files that marked it and its packages, each of nothing but
    comments and whitespace: they go, and `verity/__init__.py` and a package that never held more (`verity/empty/`) stay."""
    r, p = _run(tmp_path, {
        "verity/__init__.py": "", "verity/core/__init__.py": "# The trusted core.\n\n", "verity/core/wire/__init__.py": "",
        "verity/core/wire/frames.py": "X = 1\n", "verity/core/scripts/vectors.py": "Y = 1\n", "verity/core/sub/__init__.py": "",
        "verity/empty/__init__.py": "", "benchmarks/live/auditor.py": "Z = 1\n", "benchmarks/live/tests/test_auditor.py": "W = 1\n",
    }, LEVELS_MAP)
    assert p.returncode == 0, p.stderr
    assert sorted(git(r, "ls-files", "verity").split()) == [
        "verity/__init__.py", "verity/composite/__init__.py", "verity/composite/device/__init__.py",
        "verity/composite/device/live/__init__.py", "verity/composite/device/live/auditor.py",
        "verity/composite/device/live/tests/test_auditor.py", "verity/configuration/__init__.py",
        "verity/configuration/scripts/vectors.py", "verity/configuration/wire/__init__.py", "verity/configuration/wire/frames.py",
        "verity/empty/__init__.py"]
    assert not (r / "verity/core").exists()
    assert "`__init__.py` of nothing but comments and whitespace, left in a package the move empties, deleted: 2" in p.stdout


@pytest.mark.parametrize("text, comments", [
    ("", True), ("\n\n", True), ("# the data sheets\n", True), ("  # indented\r\n\t\n\f\n", True),
    ("# -*- coding: utf-8 -*-\n# one a GPU\n", True), ("#!/usr/bin/env python3\n", True),
    ('"""The data sheets."""\n', False), ("'''x'''\n# then a comment\n", False), ("pass\n", False), ("X = 1\n", False),
    ("from .site import Card\n", False), ("def load(:\n", False), ('"""never closed\n', False), ("# x\0\n", False), (None, False),
])
def test_only_comments_is_the_text_of_comments_and_whitespace_alone(text, comments):
    """What the mover deletes with an emptied package, and what `move_check.py` reads at BASE by this same function: a
    docstring is content, and so are code, text that doesn't parse, and text that isn't UTF-8 (None)."""
    assert L.only_comments(text) is comments


HELD_MAP = """[roots]
old = { verity = "verity" }
new = { verity = "verity" }

[paths]
"verity/hardware/cards/a100.toml" = "hardware/cards/a100.toml"
"verity/hardware/nodes/n1.toml" = "hardware/nodes/n1.toml"
"""
HELD_TREE = {"verity/__init__.py": "", "verity/hardware/__init__.py": "", "verity/hardware/site.py": "S = 1\n",
             "verity/hardware/cards/a100.toml": "peak = 1\n", "verity/hardware/nodes/n1.toml": "gpus = 8\n"}
CODE = "from verity.hardware.site import S\n\n\ndef load(name):\n    return {name: S}\n"


def _certify(r: Path, base: str) -> dict:
    """`move_check.py`'s verdict on the mover's run, committed on ``base``."""
    git(r, "commit", "-q", "-m", "move")
    spec = importlib.util.spec_from_file_location("layout_move_check", HERE.parents[2] / "tools/verity/check/move_check.py")
    mc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mc)
    return mc.certify(base.strip(), git(r, "rev-parse", "HEAD").strip(), cwd=str(r))


@pytest.mark.parametrize("held", ["", "# one data sheet a GPU\n\n   \n"])
def test_an_emptied_packages_init_of_comments_alone_is_deleted_and_certified(tmp_path, held):
    """The map moves only the data sheets out of `cards/` and `nodes/`; their `__init__.py` files, empty or of comments and
    whitespace alone, go with the packages, and `move_check.py` certifies the commit."""
    r, p = _run(tmp_path, {**HELD_TREE, "verity/hardware/cards/__init__.py": held, "verity/hardware/nodes/__init__.py": held},
                HELD_MAP)
    assert p.returncode == 0, p.stderr
    assert not (r / "verity/hardware/cards").exists() and not (r / "verity/hardware/nodes").exists()
    cert = _certify(r, git(r, "rev-parse", "HEAD"))
    assert cert["certified"], cert["refusals"]
    assert cert["rules"]["emptied"] == ["verity/hardware/cards/__init__.py", "verity/hardware/nodes/__init__.py"]


@pytest.mark.parametrize("held", ['"""The data sheets, one a GPU."""\n', CODE])
@pytest.mark.parametrize("dry", [False, True])
def test_an_emptied_packages_init_that_holds_more_stops_the_move_naming_it_and_the_fix(tmp_path, held, dry):
    """A docstring or code in an `__init__.py` the move would delete with its package stops the whole move, dry run or not:
    exit 1, each such file named with the fix, and nothing written; `nodes/`, of nothing but a comment, is not named."""
    r, p = _run(tmp_path, {**HELD_TREE, "verity/hardware/cards/__init__.py": held,
                           "verity/hardware/nodes/__init__.py": "# the nodes\n"}, HELD_MAP, *(["--dry-run"] if dry else []))
    assert p.returncode == 1, p.stdout
    first, fix = p.stderr.rstrip().splitlines()[-2:]
    assert first == ("the move would delete 1 `__init__.py` file left in packages it empties that hold more than comments "
                     "and whitespace: verity/hardware/cards/__init__.py")
    assert fix.startswith("give each a row in the map saying where it goes")
    assert p.stderr.rstrip().endswith("or delete it in a hand commit before the move")
    assert git(r, "status", "--porcelain") == "" and (r / "verity/hardware/cards/__init__.py").read_text() == held


def test_an_init_that_holds_more_moves_by_its_row_and_the_move_is_certified(tmp_path):
    """The fix: a row in the map sends the code where it goes (the package becomes a module of the same name), and the
    range certifies."""
    r, p = _run(tmp_path, {**HELD_TREE, "verity/hardware/cards/__init__.py": CODE, "verity/hardware/nodes/__init__.py": ""},
                HELD_MAP.replace('[paths]\n', '[paths]\n"verity/hardware/cards/__init__.py" = "verity/hardware/cards.py"\n'))
    assert p.returncode == 0, p.stderr
    assert (r / "verity/hardware/cards.py").read_text() == CODE
    cert = _certify(r, git(r, "rev-parse", "HEAD"))
    assert cert["certified"], cert["refusals"]
    assert {"old": "verity/hardware/cards/__init__.py", "new": "verity/hardware/cards.py"} in cert["renames"]


COVER_MAP = """[roots]
old = { lib = "lib" }
new = { verity = "verity" }

[paths]
"lib/commit/" = "verity/commit/"
"lib/core/wire/" = "verity/config/wire/"
"lib/core/proto/" = "verity/protocols/proto/"
"lib/core/claims.py" = "verity/config/claims.py"

[modules]
"lib.commit" = "verity.commit"
"lib.core.claims" = "verity.config.claims"
"lib.core.proto" = "verity.protocols.proto"
"lib.core.wire" = "verity.config.wire"
"""


def test_a_list_entry_naming_a_directory_the_map_empties_becomes_the_cover_of_where_its_files_went(tmp_path):
    """`lib/core/` goes three ways and its own `__init__.py` is deleted, so no one path stands for it. In a list of paths
    it becomes the fewest directories and files of the moved tree that hold what the list's old entries held, taken over
    all of them: beside `lib/commit/` that is `verity/` whole, alone it is the two parts, and so it is beside an entry
    below it, since no row gives it a place. An entry naming the deleted `__init__.py` becomes the package levels above
    what the move wrote anew in its list (`marks`), and goes where the list names nothing the move wrote (`bare`). What
    is no such list stays as it was: a tuple that an assignment unpacks or that is a row of a table, a word of a
    command that takes one directory, and a list in a test's file, which may name a stand-in tree."""
    r, p = _run(tmp_path, {
        "lib/core/__init__.py": "", "lib/core/wire/__init__.py": "", "lib/core/wire/frames.py": "X = 1\n",
        "lib/core/proto/check.py": "Y = 1\n", "lib/core/claims.py": "Z = 1\n", "lib/commit/c.py": "C = 1\n",
        "docs/a.md": "a\n", "other/x.py": "W = 1\n",
        "gates.toml": ('low_risk = ["docs/", "lib/core/", "lib/commit/"]\ncore = ["lib/core/"]\n'
                       'levels = ["lib/core/", "lib/core/wire/"]\nmarks = [\n    "lib/core/__init__.py",\n    "lib/core/claims.py",\n]\n'
                       'bare = ["lib/core/__init__.py", "docs/a.md"]\n'),
        "tools/lists.py": 'ROOTS = ("lib/core", "other")\nA, B = "lib/core", "docs"\nROWS = [("lib/core", "why")]\n',
        "tools/tests/test_lists.py": 'ROOTS = ("lib/core", "other")\n',
        "job.sh": "#!/bin/sh\ngit rev-parse HEAD:lib/core HEAD:other\ncd lib/core\n",
    }, COVER_MAP)
    assert p.returncode == 0, p.stderr
    assert (r / "gates.toml").read_text() == (
        'low_risk = ["docs/", "verity/", "verity/commit/"]\ncore = ["verity/config/", "verity/protocols/proto/"]\n'
        'levels = ["verity/config/", "verity/protocols/proto/", "verity/config/wire/"]\nmarks = [\n'
        '    "verity/__init__.py", "verity/config/__init__.py",\n    "verity/config/claims.py",\n]\nbare = ["docs/a.md"]\n')
    assert (r / "tools/lists.py").read_text() == (
        'ROOTS = ("verity/config", "verity/protocols/proto", "other")\nA, B = "lib/core", "docs"\nROWS = [("lib/core", "why")]\n')
    assert (r / "tools/tests/test_lists.py").read_text() == 'ROOTS = ("lib/core", "other")\n'
    assert (r / "job.sh").read_text() == "#!/bin/sh\ngit rev-parse HEAD:verity/config HEAD:verity/protocols/proto HEAD:other\ncd lib/core\n"
    assert "gates.toml: lib/core/ -> verity\n" in p.stdout and "gates.toml: lib/core/__init__.py -> nothing" in p.stdout


BELOW_MAP = """[roots]
old = { lib = "lib" }
new = { verity = "verity" }

[paths]
"lib/flock/src/" = "verity/flock/prover/src/"
"lib/flock/tool.py" = "verity/flock/prover/tool.py"
"lib/flock/flock-link.patch" = "verity/flock/prover/flock-link.patch"
"lib/flock/pod/a.sh" = "verity/flock/pod/a.sh"
"lib/flock/verifier/" = "verity/flock/agreement/"

[tokens]
"lib/flock" = "verity/flock/prover"

[modules]
"lib.flock.src" = "verity.flock.prover.src"
"lib.flock.tool" = "verity.flock.prover.tool"
"lib.flock.verifier" = "verity.flock.agreement"
"""
BELOW = {
    "lib/flock/src/main.rs": "fn main() {}\n", "lib/flock/tool.py": "T = 1\n", "lib/flock/flock-link.patch": "patch\n",
    "lib/flock/pod/a.sh": "#!/bin/sh\n", "lib/flock/verifier/ci.py": "C = 1\n",
    "tests/test_gate.py": ('FILES = {"lib/flock/src.rs": "fn main() {}", "lib/flock/tool/x.py": "", "lib/flock/pod/zz.sh": "",\n'
                           '         "lib/flock/verifier/x.py": ""}\n\n\ndef test(repo):\n    (repo / "lib" / "flock" / "new.rs").write_text("")\n'),
    "job.sh": '#!/bin/sh\nfor p in link glue; do git apply "$SRC/lib/flock/flock-$p.patch"; done\n',
}


def test_a_path_below_a_split_directory_that_names_no_tracked_file_follows_the_tokens_row(tmp_path):
    """`lib/flock/` goes three ways and its `[tokens]` row says a reference to it means the prover. A path below it that
    the tree does not have, a test's own file (`src.rs`, a chain's `new.rs`) or the start of a name a variable ends
    (`flock-$p.patch`), follows that row, and `[split]` must acknowledge it as it must a reference to the directory. One
    whose first name the tree has there is left: `tool/` (`tool.py` has its own row) and `pod/` (a directory with no
    row). One below a directory with a row follows that row, as before."""
    ack = '\n[split]\n"tests/test_gate.py: lib/flock" = "the toy files are the prover\'s"\n"job.sh: lib/flock" = "the patches"\n'
    r, p = _run(tmp_path / "acknowledged", BELOW, BELOW_MAP + ack)
    assert p.returncode == 0, p.stderr
    assert (r / "tests/test_gate.py").read_text() == (
        'FILES = {"verity/flock/prover/src.rs": "fn main() {}", "lib/flock/tool/x.py": "", "lib/flock/pod/zz.sh": "",\n'
        '         "verity/flock/agreement/x.py": ""}\n\n\ndef test(repo):\n    (repo / "verity" / "flock" / "prover" / "new.rs").write_text("")\n')
    assert (r / "job.sh").read_text() == '#!/bin/sh\nfor p in link glue; do git apply "$SRC/verity/flock/prover/flock-$p.patch"; done\n'
    _, bare = _run(tmp_path / "bare", BELOW, BELOW_MAP)
    assert bare.returncode != 0 and "job.sh: lib/flock (" in bare.stderr and "tests/test_gate.py: lib/flock (" in bare.stderr


def test_a_workspaces_members_and_a_directory_beside_its_own_parts_follow_the_tokens_row(tmp_path):
    """`lib/flock/` is a distribution and goes three ways. In a list of paths it becomes where its files went, here all of
    `verity/flock/`. The workspace's `members` names the directory its `pyproject.toml` is in, which the `[tokens]` row
    says: written as the cover it would name no distribution, and the workspace would lose it. And in a list that names
    it beside a part of it (`levels`), it is the part the row says, not all of them."""
    r, p = _run(tmp_path, {
        **BELOW, "tests/test_gate.py": "", "job.sh": "#!/bin/sh\n", "lib/flock/pyproject.toml": '[project]\nname = "flock"\n',
        "other/pyproject.toml": '[project]\nname = "other"\n',
        "pyproject.toml": ('[tool.uv.workspace]\nmembers = ["lib/flock", "other"]\n\n[tool.uv.sources]\nother = { workspace = true }\n\n'
                           '[tool.gate]\nkeyed = ["lib/flock", "other"]\nlevels = ["lib/flock", "lib/flock/verifier"]\n'),
    }, BELOW_MAP.replace("[tokens]", '"lib/flock/pyproject.toml" = "verity/flock/prover/pyproject.toml"\n\n[tokens]')
        + '\n[split]\n"pyproject.toml: lib/flock" = "the distribution"\n')
    assert p.returncode == 0, p.stderr
    assert (r / "pyproject.toml").read_text() == (
        '[tool.uv.workspace]\nmembers = ["verity/flock/prover", "other"]\n\n[tool.uv.sources]\nother = { workspace = true }\n\n'
        '[tool.gate]\nkeyed = ["verity/flock", "other"]\nlevels = ["verity/flock/prover", "verity/flock/agreement"]\n')


NAMES = {**BELOW, "tests/test_gate.py": "", "job.sh": "#!/bin/sh\n", "other/x.py": "X = 1\n",
         "suites.py": 'AFTER = ("flock-suite", "lib/flock", "other")\nARGV = ["tool", "--project", "lib/flock"]\nPATHS = ("lib/flock", "other")\n',
         "gates.toml": 'after = ["lib/flock", "flock-suite"]\npaths = ["lib/flock", "a.sh", "lib/gone/x.py"]\n'}


def test_a_list_with_an_entry_of_one_name_that_no_path_ends_in_holds_names_and_is_no_cover(tmp_path):
    """`lib/flock/` goes three ways. A list that names it beside one name no path of the tree ends in, a suite named by its
    distribution (`flock-suite`) or a command's words (`tool`, `--project`), holds names, and the directory's is the name
    of one thing: it follows the `[tokens]` row, and `[split]` must acknowledge it as it must any reference to the
    directory. A list of paths is one whose every entry is a path or a name some path ends in (`a.sh`, a pattern), also
    a path the tree no longer has (`lib/gone/x.py`): there the directory becomes where its files went (`PATHS`,
    `paths`). Without the rows the move fails and names each file with the directory."""
    ack = '\n[split]\n"suites.py: lib/flock" = "the suite and the project are the prover\'s"\n"gates.toml: lib/flock" = "the suite"\n'
    r, p = _run(tmp_path / "acknowledged", NAMES, BELOW_MAP + ack)
    assert p.returncode == 0, p.stderr
    assert (r / "suites.py").read_text() == ('AFTER = ("flock-suite", "verity/flock/prover", "other")\n'
                                             'ARGV = ["tool", "--project", "verity/flock/prover"]\nPATHS = ("verity/flock", "other")\n')
    assert (r / "gates.toml").read_text() == ('after = ["verity/flock/prover", "flock-suite"]\n'
                                              'paths = ["verity/flock", "a.sh", "lib/gone/x.py"]\n')
    _, bare = _run(tmp_path / "bare", NAMES, BELOW_MAP)
    assert bare.returncode != 0 and "suites.py: lib/flock (" in bare.stderr and "gates.toml: lib/flock (" in bare.stderr


PART = {**BELOW, "tests/test_gate.py": "", "job.sh": "#!/bin/sh\n",
        "suites.py": 'AFTER = ("lib/flock", "other-suite")\n', "run.sh": '#!/bin/sh\nexe "lib/flock/$x"\n',
        "doc.md": "The prover is `lib/flock/`.\n"}


def test_a_split_row_that_names_the_part_its_file_means_sends_the_files_references_there(tmp_path):
    """`lib/flock/` goes three ways. A `[split]` row acknowledges one file's reference to it, which then follows the
    `[tokens]` row (`doc.md`); a row with `to` names the part that file means instead, and its references go there: an
    entry of a list, which is then no cover, and the start of a path a variable ends. A row that names a directory no
    file of `lib/flock/` goes to is refused, by its key."""
    rows = ('\n[split]\n"doc.md: lib/flock" = "the prover"\n"run.sh: lib/flock" = { to = "verity/flock/agreement", why = "its packages" }\n'
            '"suites.py: lib/flock" = { to = "TO", why = "the suite is the verifier\'s" }\n')
    r, p = _run(tmp_path / "part", PART, BELOW_MAP + rows.replace("TO", "verity/flock/agreement"))
    assert p.returncode == 0, p.stderr
    assert (r / "suites.py").read_text() == 'AFTER = ("verity/flock/agreement", "other-suite")\n'
    assert (r / "run.sh").read_text() == '#!/bin/sh\nexe "verity/flock/agreement/$x"\n'
    assert (r / "doc.md").read_text() == "The prover is `verity/flock/prover/`.\n"
    _, bad = _run(tmp_path / "elsewhere", PART, BELOW_MAP + rows.replace("TO", "verity/flock/agreement/lean"))
    assert bad.returncode != 0 and '[split] "suites.py: lib/flock" names verity/flock/agreement/lean' in bad.stderr


def test_a_path_after_a_prefix_the_map_lists_for_the_root_is_rewritten(tmp_path):
    """A pod's unit names a file by where the pod installs the tree. The map lists that prefix as standing for the root
    (`[rewrite] root_prefixes`), and the path after it is rewritten; after a prefix the map does not list it stays. A
    prefix that does not end in `/` is refused."""
    unit = ("ExecStart=/usr/bin/python3 -I /opt/pod/current/lib/flock/tool.py run %i\n"
            "ExecStart=/usr/bin/python3 -I /opt/other/current/lib/flock/tool.py run %i\n")
    files = {**BELOW, "tests/test_gate.py": "", "job.sh": "#!/bin/sh\n", "pod/unit@.service": unit}
    listed = BELOW_MAP + '\n[rewrite]\nroot_prefixes = ["/opt/pod/current/"]\n'
    r, p = _run(tmp_path / "listed", files, listed)
    assert p.returncode == 0, p.stderr
    assert (r / "pod/unit@.service").read_text() == unit.replace("/opt/pod/current/lib/flock/", "/opt/pod/current/verity/flock/prover/")
    r, p = _run(tmp_path / "bare", files, BELOW_MAP)
    assert p.returncode == 0 and (r / "pod/unit@.service").read_text() == unit
    _, bad = _run(tmp_path / "bad", files, listed.replace("current/", "current"))
    assert bad.returncode != 0 and "[rewrite] root_prefixes" in bad.stderr


def test_a_directorys_name_alone_between_quotes_is_rewritten_in_the_files_the_map_lists_it_for(tmp_path):
    """`examples/` goes to `workloads/`, and `"examples"` is the directory in the command line's code and a key
    elsewhere. `[rewrite] words_in` lists the name for the files where it is the directory: there it is rewritten, alone
    between quotes; in a Python file its glob does not match it stays, as every lone quoted word does."""
    layout = ('[roots]\nold = { lib = "lib" }\nnew = { verity = "verity" }\n\n[paths]\n"examples/" = "workloads/"\n\n'
              '[modules]\n"examples" = "workloads"\n\n[rewrite]\nwords_in = { "cli/*.py" = ["examples/"] }\n')
    r, p = _run(tmp_path, {"examples/demo/config.toml": "", "cli/main.py": 'CONFIG = ROOT / "examples" / NAME / "config.toml"\n',
                           "bench/keys.py": 'KEYS = {"examples": 1}\n'}, layout)
    assert p.returncode == 0, p.stderr
    assert (r / "cli/main.py").read_text() == 'CONFIG = ROOT / "workloads" / NAME / "config.toml"\n'
    assert (r / "bench/keys.py").read_text() == 'KEYS = {"examples": 1}\n'


def test_an_import_split_in_two_keeps_its_lines_comment_and_the_lines_it_went_on_over(tmp_path):
    """`lib.spot` goes two ways, so one `from lib.spot import` becomes two. Each keeps the comment that ended the line. An
    import that went on over more lines, in brackets or after a backslash, keeps those lines as they were, under the
    statement that has their names, which goes last; its comment stays where it was, after the last line. Where the names
    on those lines go two ways no statement has them, and the import is written again, one statement a line."""
    use = ("from lib.spot import coins as CC, service as S  # noqa: E402\n"
           "from lib.spot import (coins as C2, second as SS,\n                      service as S2)  # the rest\n"
           "from lib.spot import coins as C3, service as S3, \\\n    second as S4\n"
           "from lib.spot import (second as S5,\n                      service as S6, coins as C6)\n")
    r, p = _run(tmp_path, {"lib/spot/__init__.py": "", "lib/spot/coins.py": "C = 1\n", "lib/spot/service.py": "S = 1\n",
                           "lib/spot/second.py": "T = 1\n", "use.py": use},
                '[roots]\nold = { lib = "lib" }\nnew = { verity = "verity" }\n\n[paths]\n"lib/spot/coins.py" = "verity/device/coins.py"\n'
                '"lib/spot/" = "verity/verifier/"\n\n[modules]\n"lib.spot" = "verity.verifier"\n"lib.spot.coins" = "verity.device.coins"\n')
    assert p.returncode == 0, p.stderr
    assert (r / "use.py").read_text() == (
        "from verity.device import coins as CC  # noqa: E402\nfrom verity.verifier import service as S  # noqa: E402\n"
        "from verity.device import coins as C2\nfrom verity.verifier import (second as SS,\n                      service as S2)  # the rest\n"
        "from verity.device import coins as C3\nfrom verity.verifier import service as S3, \\\n    second as S4\n"
        "from verity.verifier import second as S5, service as S6\nfrom verity.device import coins as C6\n")


def test_rust_literals_spans_strings_and_skips_comments_chars_and_lifetimes():
    text = '''// "no" \n/* "no" /* "nested" */ */ let a = "x\\"y"; let b = br##"r"#"##; let c = '"'; fn f<'a>() { "z" }'''
    got = [text[s:e] for s, e in L.rust_literals(text)]
    assert got == ['"x\\"y"', 'br##"r"#"##', '"z"']


ROWS_MAP = """[roots]
old = {}
new = {}

[paths]
"old/lib/__init__.py" = "new/lib/__init__.py"
"old/lib/a.py" = "new/lib/a.py"
"old/whole/" = "new/whole/"

[retired]
paths = ["old/lib/"]

[drop]
paths = ["old/lib/gone.py"]

[modules]
"old.lib" = "new.lib"
"old.lib.a" = "new.lib.a"
"old.whole" = "new.whole"

[rewrite]
exclude = ["tools/move/**"]
"""
#: what an open branch adds: three files under the path the map retires, one in a directory that stays, and their readers
ADDED = {"old/lib/added.py": "import old.lib.a\n", "old/lib/sub/deep.py": "", "old/lib/sub/deeper/x.py": "",
         "stays/extra.py": "", "doc.md": "See old/lib/added.py, old/lib/sub/ and `old.lib.sub.deep`.\n"}
ROWS = "tools/move/layout_map.rows.toml"
MAP_AT = "tools/move/layout_map.toml"


def rows_repo(tmp_path: Path, rows: str | None, name: str = "repo", map_text: str = ROWS_MAP) -> Path:
    """A checkout whose map retires `old/lib/`, with a branch's files in it (`ADDED`) and its rows file when `rows` is given."""
    r = tmp_path / name
    files = {"old/lib/__init__.py": "", "old/lib/a.py": "X = 1\n", "old/lib/gone.py": "", "old/whole/w.py": "",
             "stays/s.py": "", "use.py": "import old.lib.a\n", MAP_AT: map_text,
             "tools/research/src/research/store/moves.json": '{"moves": []}\n', **ADDED}
    if rows is not None:
        files[ROWS] = rows
    for p, t in files.items():
        (r / p).parent.mkdir(parents=True, exist_ok=True)
        (r / p).write_text(t)
    git(r, "init", "-q", "-b", "main")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "a branch, merged with the move's base")
    return r


def layout(r: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["python3", str(LAYOUT), "--root", str(r), "--map", MAP_AT, "--no-lock", *args], cwd=r,
                          capture_output=True, text=True)


def test_a_tracked_file_under_a_retired_path_with_no_row_stops_the_move_by_name_with_an_exit_of_its_own(tmp_path):
    """The map is written from main's tree, so a file a branch adds under a directory the map empties has no row and would
    stay behind. The move stops before it changes anything, with `NO_ROW` (not the 1 of any other failure), the fix and
    the rows file's name in the first line of stderr, and each file on a line of its own. A file in a directory the map
    does not retire is not named: nothing says it should have moved."""
    r = rows_repo(tmp_path, None)
    p = layout(r)
    first, *named = p.stderr.splitlines()
    assert p.returncode == L.NO_ROW == 3, p.stdout + p.stderr
    assert first.startswith("no row for 3 tracked files under a path the map retires: ") and ROWS in first and "[paths]" in first
    assert named == ["  old/lib/added.py", "  old/lib/sub/deep.py", "  old/lib/sub/deeper/x.py"]
    assert git(r, "status", "--porcelain") == ""


def test_a_row_of_the_rows_file_moves_its_file_and_rewrites_its_references_as_a_row_of_the_map_does(tmp_path):
    """The rows file is read after the map: a file row and a directory row move the branch's files, and every reference to
    them is rewritten, a module name too, though the rows file declares none. A row the map already has changes nothing.
    The tree is, file for file, the one the same rows give as rows of the map; the rows file is deleted once read, so a
    second run finds none and changes nothing; and `moves.json` records the map's rows alone."""
    rows = '[paths]\n"old/lib/added.py" = "new/lib/added.py"\n"old/lib/sub/" = "new/lib/sub/"\n"old/whole/" = "new/whole/"\n'
    r = rows_repo(tmp_path, rows)
    assert layout(r, "--dry-run").returncode == 0 and git(r, "status", "--porcelain") == "", "a dry run reads it and keeps it"
    p = layout(r)
    assert p.returncode == 0, p.stdout + p.stderr
    assert f"rows read from {ROWS}: 2" in p.stdout
    assert (r / "new/lib/added.py").read_text() == "import new.lib.a\n"
    assert (r / "new/lib/sub/deep.py").is_file() and (r / "new/lib/sub/deeper/x.py").is_file() and not (r / "old").exists()
    assert (r / "doc.md").read_text() == "See new/lib/added.py, new/lib/sub/ and `new.lib.sub.deep`.\n"
    assert (r / "stays/extra.py").is_file() and not (r / ROWS).exists()
    moves = json.loads((r / "tools/research/src/research/store/moves.json").read_text())["moves"]
    assert moves == [{"paths": {"old/lib/__init__.py": "new/lib/__init__.py", "old/lib/a.py": "new/lib/a.py",
                                "old/whole/": "new/whole/"}}]
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "moved")
    assert layout(r).returncode == 0 and git(r, "status", "--porcelain") == ""

    in_map = ROWS_MAP.replace('"old/whole/" = "new/whole/"\n', '"old/whole/" = "new/whole/"\n"old/lib/added.py" = '
                              '"new/lib/added.py"\n"old/lib/sub/" = "new/lib/sub/"\n')
    in_map = in_map.replace('"old.whole" = "new.whole"\n', '"old.lib.added" = "new.lib.added"\n"old.lib.sub" = "new.lib.sub"\n'
                            '"old.whole" = "new.whole"\n')
    m = rows_repo(tmp_path, None, "as-map-rows", in_map)
    assert layout(m).returncode == 0
    git(m, "add", "-A")
    trees = [{line.split("\t")[1]: line for line in git(repo, "ls-files", "-s").splitlines()
              if not line.endswith(("/layout_map.toml", "/moves.json"))} for repo in (m, r)]
    assert trees[0] == trees[1] and "new/lib/sub/deeper/x.py" in trees[0]


@pytest.mark.parametrize("rows, why", [
    ('[paths]\n"old/lib/a.py" = "elsewhere/a.py"\n', "'old/lib/a.py' = 'elsewhere/a.py': the map sends it to 'new/lib/a.py'"),
    ('[paths]\n"old/whole/w.py" = "elsewhere/w.py"\n', "'old/whole/w.py' = 'elsewhere/w.py': the map sends it to 'new/whole/w.py'"),
    ('[paths]\n"old/whole/" = "elsewhere/"\n', "'old/whole/' = 'elsewhere/': the map sends it to 'new/whole/'"),
    ('[paths]\n"old/" = "elsewhere/"\n', "'old/': the map names 4 paths at or under it (old/lib/__init__.py, ...)"),
    ('[paths]\n"old/lib/gone.py" = "new/lib/gone.py"\n', "'old/lib/gone.py': the map deletes it"),
    ('[paths]\n"old/lib/missing.py" = "new/lib/missing.py"\n', "'old/lib/missing.py': no tracked file is at it"),
    ('[paths]\n"old/lib/nothing/" = "new/lib/nothing/"\n', "'old/lib/nothing/': no tracked file is under it"),
    ('[paths]\n"old/lib/added.py" = "new/lib/"\n', "'old/lib/added.py' = 'new/lib/': a file to a file"),
    ('[paths]\n"old/lib/added.py" = "new/lib/added.py"\n\n[retired]\npaths = []\n', "holds [paths] alone, not ['retired']"),
    ("[paths\n", "Expected"),
])
def test_a_rows_file_that_changes_what_the_map_says_or_moves_nothing_is_refused_and_nothing_moves(tmp_path, rows, why):
    """A branch's rows add to the map and never change it: a row for a path the map sends elsewhere (by its own row or a
    directory's) or deletes, a directory row over paths the map names (it would take main's files with it), a row no
    tracked file is at or under (a path mistyped would leave its file behind unsaid), a file sent to a directory, and any
    table but [paths] each stop the move with the row and the reason, exit 1, before anything changes."""
    r = rows_repo(tmp_path, rows)
    p = layout(r)
    assert p.returncode == 1 and p.stderr.startswith(f"{ROWS}: ") and why in p.stderr, p.stdout + p.stderr
    assert git(r, "status", "--porcelain") == ""


def test_a_rows_file_renaming_a_module_the_map_renames_another_way_is_refused(tmp_path):
    """`old/lib/a.py` and a directory `old/lib/a/` share the module name `old.lib.a`. The map sends the file to `new.lib.a`;
    a row sending the directory elsewhere would rename that module a second way, in every file that imports it."""
    r = rows_repo(tmp_path, '[paths]\n"old/lib/a/" = "other/a/"\n"old/lib/added.py" = "new/lib/added.py"\n'
                            '"old/lib/sub/" = "new/lib/sub/"\n')
    (r / "old/lib/a").mkdir()
    (r / "old/lib/a/z.py").write_text("")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "a directory named as a module is")
    p = layout(r)
    assert p.returncode == 1 and "module old.lib.a: the map renames it new.lib.a, and these rows other.a" in p.stderr, p.stderr
    assert git(r, "status", "--porcelain") == ""


def test_a_rows_file_outside_the_checkout_is_refused(tmp_path):
    """The rows file is its branch's own, a tracked file beside the map: one beside a map kept outside the checkout is
    nobody's commit, and deleting it once read would lose it."""
    r = rows_repo(tmp_path, None)
    (tmp_path / "map.toml").write_text(ROWS_MAP)
    (tmp_path / "map.rows.toml").write_text('[paths]\n"old/lib/added.py" = "new/lib/added.py"\n')
    p = subprocess.run(["python3", str(LAYOUT), "--root", str(r), "--map", str(tmp_path / "map.toml"), "--no-lock"],
                       capture_output=True, text=True)
    assert p.returncode == 1 and "a rows file is its branch's own" in p.stderr, p.stdout + p.stderr
    assert (tmp_path / "map.rows.toml").is_file() and git(r, "status", "--porcelain") == ""
