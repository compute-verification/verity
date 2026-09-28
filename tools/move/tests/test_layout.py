"""`layout.py` itself on a toy tree whose map moves core's `verity.randomness`, `census/` and a recorded run's directory: what
it must leave alone."""
from __future__ import annotations

import importlib.util
import os
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
    Lean verifier that left for core/ (the C-Flock gates that failed open): each such reference is an error until [split]
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


def test_rust_literals_spans_strings_and_skips_comments_chars_and_lifetimes():
    text = '''// "no" \n/* "no" /* "nested" */ */ let a = "x\\"y"; let b = br##"r"#"##; let c = '"'; fn f<'a>() { "z" }'''
    got = [text[s:e] for s, e in L.rust_literals(text)]
    assert got == ['"x\\"y"', 'br##"r"#"##', '"z"']
