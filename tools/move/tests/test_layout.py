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


def test_the_maps_paths_as_a_trie_match_what_the_list_of_them_longest_first_matches():
    """`_paths` finds the map's old paths with one regex over every file. Written as a list, `re` tries each path at every
    character: nine of a run's ten minutes on a draft of the one map (LEAN-258). `alternation` writes the same set as a
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
    own. `verity/core/` then holds only the `__init__.py` files that marked it and its packages: they go, and
    `verity/__init__.py` and a package that never held more (`verity/empty/`) stay."""
    r, p = _run(tmp_path, {
        "verity/__init__.py": "", "verity/core/__init__.py": '"""The trusted core."""\n', "verity/core/wire/__init__.py": "",
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
    assert "`__init__.py` left alone in a package the move empties, deleted: 2" in p.stdout


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
    all of them: beside `lib/commit/` that is `verity/` whole, alone it is the two parts. An entry naming the deleted
    `__init__.py` goes. What is no such list stays as it was: entries one above the other (levels), a tuple that an
    assignment unpacks or that is a row of a table, and a word of a command that takes one directory."""
    r, p = _run(tmp_path, {
        "lib/core/__init__.py": "", "lib/core/wire/__init__.py": "", "lib/core/wire/frames.py": "X = 1\n",
        "lib/core/proto/check.py": "Y = 1\n", "lib/core/claims.py": "Z = 1\n", "lib/commit/c.py": "C = 1\n",
        "docs/a.md": "a\n", "other/x.py": "W = 1\n",
        "gates.toml": ('low_risk = ["docs/", "lib/core/", "lib/commit/"]\ncore = ["lib/core/"]\n'
                       'levels = ["lib/core/", "lib/core/wire/"]\nmarks = [\n    "lib/core/__init__.py",\n    "lib/core/claims.py",\n]\n'),
        "tools/lists.py": 'ROOTS = ("lib/core", "other")\nA, B = "lib/core", "docs"\nROWS = [("lib/core", "why")]\n',
        "job.sh": "#!/bin/sh\ngit rev-parse HEAD:lib/core HEAD:other\ncd lib/core\n",
    }, COVER_MAP)
    assert p.returncode == 0, p.stderr
    assert (r / "gates.toml").read_text() == (
        'low_risk = ["docs/", "verity/", "verity/commit/"]\ncore = ["verity/config/", "verity/protocols/proto/"]\n'
        'levels = ["lib/core/", "verity/config/wire/"]\nmarks = [\n    "verity/config/claims.py",\n]\n')
    assert (r / "tools/lists.py").read_text() == (
        'ROOTS = ("verity/config", "verity/protocols/proto", "other")\nA, B = "lib/core", "docs"\nROWS = [("lib/core", "why")]\n')
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


def test_rust_literals_spans_strings_and_skips_comments_chars_and_lifetimes():
    text = '''// "no" \n/* "no" /* "nested" */ */ let a = "x\\"y"; let b = br##"r"#"##; let c = '"'; fn f<'a>() { "z" }'''
    got = [text[s:e] for s, e in L.rust_literals(text)]
    assert got == ['"x\\"y"', 'br##"r"#"##', '"z"']
