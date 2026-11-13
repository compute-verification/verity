"""`tools/check/move_check.py`: which commit ranges it certifies as pure moves, on tiny git repositories."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
GIT = shutil.which("git")
spec = importlib.util.spec_from_file_location("check_move_check", ROOT / "tools" / "verity" / "check" / "move_check.py")
MC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(MC)

ENV = {**os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1"}      # not this machine's git config
PYPROJECT = 'name = "pous"\n[tool.verity.tests]\ninputs = ["protocols/pous/data", "census/hardware.json"]\n'
LAKEFILE = '[[require]]\nname = "verity"\npath = "../../../packages/verity/lean"\n'


def _git(repo: Path, *args: str) -> str:
    return subprocess.run([GIT, "-c", "user.email=t@t", "-c", "user.name=t", *args], cwd=repo, check=True,
                          capture_output=True, text=True, env=ENV).stdout.strip()


def _write(repo: Path, files: dict[str, str]) -> None:
    for p, text in files.items():
        (repo / p).parent.mkdir(parents=True, exist_ok=True)
        (repo / p).write_text(text)


def _repo(tmp_path: Path, files: dict[str, str]) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True)
    _git(repo, "init", "-q")
    _write(repo, files)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    return repo, _git(repo, "rev-parse", "HEAD")


def _move(repo: Path, old: str, new: str) -> None:
    (repo / new).parent.mkdir(parents=True, exist_ok=True)
    _git(repo, "mv", old, new)


def _check(repo: Path, base: str, extra: dict[str, str] | None = None) -> dict:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "move")
    return MC.certify(base, _git(repo, "rev-parse", "HEAD"), extra, cwd=str(repo))


BASE_FILES = {
    "protocols/pous/src/verity_pous/__init__.py": "",
    "protocols/pous/src/verity_pous/codec.py": "def encode(x):\n    return x + 1\n",
    "protocols/pous/lean/Pous.lean": "theorem t : 1 = 1 := rfl\n",
    "protocols/pous/lean/lean-audit.json": '{"guarantees": {"Pous.t": "abc"}}\n',
    "protocols/pouw/src/verity_pouw/x.py": "X = 1\n",
    "tests/test_pous.py": "from verity_pous.codec import encode\nimport verity_pous.codec as c\n\n\ndef test():\n    assert encode(1) == 2\n",
}


def test_pure_rename_is_certified(tmp_path):
    repo, base = _repo(tmp_path, BASE_FILES)
    _move(repo, "protocols/pous", "security_proofs/pous")
    r = _check(repo, base)
    assert r["certified"], r["refusals"]
    assert len(r["renames"]) == 4 and not r["edits"]
    # the map is the moved directory, not `protocols`, since `protocols/pouw` stayed
    assert r["map"] == {"protocols/pous": "security_proofs/pous"}


def test_rename_with_a_one_byte_edit_is_refused(tmp_path):
    repo, base = _repo(tmp_path, BASE_FILES)
    _move(repo, "protocols/pous", "security_proofs/pous")
    _write(repo, {"security_proofs/pous/src/verity_pous/codec.py": "def encode(x):\n    return x + 2\n"})
    r = _check(repo, base)
    assert not r["certified"]
    [f] = r["refusals"]
    assert f["path"] == "security_proofs/pous/src/verity_pous/codec.py" and f["old"] == "protocols/pous/src/verity_pous/codec.py"
    assert "-    return x + 1" in f["hunk"] and "+    return x + 2" in f["hunk"]


def test_pyproject_path_edit_is_certified(tmp_path):
    repo, base = _repo(tmp_path, {**BASE_FILES, "protocols/pous/pyproject.toml": PYPROJECT,
                                  "pyproject.toml": '[tool.uv.workspace]\nmembers = ["protocols/pous", "protocols/pouw"]\n'})
    _move(repo, "protocols/pous", "security_proofs/pous")
    _write(repo, {"security_proofs/pous/pyproject.toml": PYPROJECT.replace("protocols/pous/data", "security_proofs/pous/data"),
                  "pyproject.toml": '[tool.uv.workspace]\nmembers = ["security_proofs/pous", "protocols/pouw"]\n'})
    r = _check(repo, base)
    assert r["certified"], r["refusals"]
    assert {e["path"] for e in r["edits"]} == {"pyproject.toml", "security_proofs/pous/pyproject.toml"}


def test_path_edit_to_a_directory_that_did_not_move_is_refused(tmp_path):
    repo, base = _repo(tmp_path, {**BASE_FILES, "pyproject.toml": 'members = ["protocols/pous", "protocols/pouw"]\n'})
    _move(repo, "protocols/pous", "security_proofs/pous")
    _write(repo, {"pyproject.toml": 'members = ["security_proofs/pous", "security_proofs/pouw"]\n'})
    [f] = _check(repo, base)["refusals"]
    assert f["path"] == "pyproject.toml" and "security_proofs/pouw" in f["hunk"]


def test_path_after_a_checkout_variable_is_certified(tmp_path):
    """`tools/move/layout.py` rewrites a moved path after `$REPO/` or `${ROOT}/` (a shell script's checkout), so the check
    reads it from the root there too; a moved path inside a longer one is still not the move's."""
    run = 'SRC=$REPO/protocols/pous/src:${ROOT}/protocols/pous\nOUT=$W/x/protocols/pous\n'
    repo, base = _repo(tmp_path, {**BASE_FILES, "run.sh": run})
    _move(repo, "protocols/pous", "security_proofs/pous")
    _write(repo, {"run.sh": run.replace("$REPO/protocols", "$REPO/security_proofs").replace("}/protocols", "}/security_proofs")})
    r = _check(repo, base)
    assert r["certified"], r["refusals"]
    _write(repo, {"run.sh": run.replace("protocols/pous", "security_proofs/pous")})
    [f] = _check(repo, base)["refusals"]
    assert f["path"] == "run.sh" and "+OUT=$W/x/security_proofs/pous" in f["hunk"]


def test_non_path_edit_in_pyproject_is_refused(tmp_path):
    repo, base = _repo(tmp_path, {**BASE_FILES, "protocols/pous/pyproject.toml": PYPROJECT})
    _move(repo, "protocols/pous", "security_proofs/pous")
    _write(repo, {"security_proofs/pous/pyproject.toml":
                  PYPROJECT.replace("protocols/pous/data", "security_proofs/pous/data").replace('"pous"', '"pous2"')})
    r = _check(repo, base)
    [f] = r["refusals"]
    assert f["path"] == "security_proofs/pous/pyproject.toml" and f["reason"] == "edit beyond moved paths and module names"
    assert f["hunk"].splitlines()[1:] == ['-name = "pous"', '+name = "pous2"']


def test_import_line_rewrite_is_certified(tmp_path):
    repo, base = _repo(tmp_path, BASE_FILES)
    _move(repo, "protocols/pous/src/verity_pous", "security_proofs/pous/src/pous")
    _write(repo, {"tests/test_pous.py": BASE_FILES["tests/test_pous.py"].replace("verity_pous", "pous")})
    r = _check(repo, base)
    assert r["certified"], r["refusals"]
    assert r["edits"] == [{"path": "tests/test_pous.py", "kind": "python"}]


def test_import_of_a_module_that_did_not_move_is_refused(tmp_path):
    repo, base = _repo(tmp_path, BASE_FILES)
    _move(repo, "protocols/pous/src/verity_pous", "security_proofs/pous/src/pous")
    _write(repo, {"tests/test_pous.py": BASE_FILES["tests/test_pous.py"].replace("verity_pous.codec", "pous.other")})
    [f] = _check(repo, base)["refusals"]
    assert f["path"] == "tests/test_pous.py" and f["reason"] == "edit beyond moved paths and module names"


def test_a_new_file_is_refused(tmp_path):
    repo, base = _repo(tmp_path, BASE_FILES)
    _move(repo, "protocols/pous", "security_proofs/pous")
    _write(repo, {"security_proofs/pous/NEW.md": "hello\n"})
    [f] = _check(repo, base)["refusals"]
    assert (f["path"], f["status"], f["reason"]) == ("security_proofs/pous/NEW.md", "A", "added")


def test_an_empty_init_for_a_package_level_the_move_makes_is_certified(tmp_path):
    """`tools/move/layout.py` writes an empty `__init__.py` in each new directory from a module moved into a package up to
    its package's root, so the module imports by its new name: a module that was a package's at BASE, and one that was
    outside every package (`verity_pouw/x.py` has no `__init__.py` at or above it). Only that is let in: one with a line in
    it, one above a script a package held as no module of its own (`verity_pous/scripts/gen.py`), or one in a directory
    run by path (`tests/`), is a new file like any other."""
    roots = {"tools/move/layout_map.toml": '[roots]\nnew = { security = "security", loose = "loose" }\n',
             "protocols/pous/src/verity_pous/scripts/gen.py": "G = 1\n", "protocols/pouw/src/verity_pouw/tests/test_x.py": "T = 1\n"}
    repo, base = _repo(tmp_path, {**BASE_FILES, **roots})
    _move(repo, "protocols/pous/src/verity_pous/codec.py", "security/pous/codec.py")
    _move(repo, "protocols/pouw/src/verity_pouw/x.py", "loose/pouw/x.py")
    _move(repo, "protocols/pous/src/verity_pous/scripts/gen.py", "loose/scripts/gen.py")
    _move(repo, "protocols/pouw/src/verity_pouw/tests/test_x.py", "loose/pouw/tests/test_x.py")
    _write(repo, {"security/__init__.py": "", "security/pous/__init__.py": "", "loose/pouw/__init__.py": "", "loose/__init__.py": "",
                  "loose/scripts/__init__.py": "", "loose/pouw/tests/__init__.py": "",
                  "tests/test_pous.py": BASE_FILES["tests/test_pous.py"].replace("verity_pous.codec", "security.pous.codec")})
    r = _check(repo, base, {"verity_pous.codec": "security.pous.codec"})
    assert [(f["path"], f["reason"]) for f in r["refusals"]] == [("loose/pouw/tests/__init__.py", "added"),
                                                                  ("loose/scripts/__init__.py", "added")]
    assert {e["path"] for e in r["edits"] if e["kind"] == "package level"} == {
        "security/__init__.py", "security/pous/__init__.py", "loose/__init__.py", "loose/pouw/__init__.py"}
    assert r["rules"] == {"level": ["loose/__init__.py", "loose/pouw/__init__.py"]}
    _write(repo, {"security/pous/__init__.py": "X = 1\n"})
    assert ("security/pous/__init__.py", "added") in [(f["path"], f["reason"]) for f in _check(repo, base)["refusals"]]


def test_the_init_of_a_package_the_move_emptied_may_be_deleted(tmp_path):
    """The mover deletes the `__init__.py` left alone where a package was: its files left by the range's renames and HEAD
    has nothing at or below its directory. One whose directory still holds a file, and one of a package no file left by a
    rename, is a deleted file like any other."""
    files = {**BASE_FILES, "protocols/pous/src/verity_pous/sub/__init__.py": "", "protocols/pous/src/verity_pous/sub/kept.py": "K = 1\n",
             "protocols/pouw/src/verity_pouw/idle/__init__.py": '"""Nothing here."""\n'}
    repo, base = _repo(tmp_path, files)
    _move(repo, "protocols/pous/src/verity_pous/codec.py", "security/pous/codec.py")
    _write(repo, {"tests/test_pous.py": BASE_FILES["tests/test_pous.py"].replace("verity_pous.codec", "security.pous.codec")})
    _git(repo, "rm", "-q", "protocols/pous/src/verity_pous/sub/__init__.py", "protocols/pouw/src/verity_pouw/idle/__init__.py")
    r = _check(repo, base, {"verity_pous.codec": "security.pous.codec"})
    assert sorted(f["path"] for f in r["refusals"]) == ["protocols/pous/src/verity_pous/sub/__init__.py",
                                                        "protocols/pouw/src/verity_pouw/idle/__init__.py"]
    _move(repo, "protocols/pous/src/verity_pous/sub/kept.py", "security/pous/kept.py")
    _git(repo, "rm", "-q", "protocols/pous/src/verity_pous/__init__.py")
    r = _check(repo, base, {"verity_pous.codec": "security.pous.codec"})
    assert [f["path"] for f in r["refusals"]] == ["protocols/pouw/src/verity_pouw/idle/__init__.py"]
    assert r["rules"] == {"emptied": ["protocols/pous/src/verity_pous/__init__.py", "protocols/pous/src/verity_pous/sub/__init__.py"]}


COVERED = {
    "tools/move/layout_map.toml": '[roots]\nnew = { verity = "verity" }\n',
    "lib/core/__init__.py": "", "lib/core/wire/__init__.py": "", "lib/core/wire/frames.py": "X = 1\n",
    "lib/core/proto/check.py": "Y = 1\n", "lib/core/claims.py": "Z = 1\n", "lib/commit/c.py": "C = 1\n", "docs/a.md": "a\n",
    "other/x.py": "W = 1\n",
    "gates.toml": 'low_risk = ["docs/", "lib/core/", "lib/commit/"]\ncore = [\n    "lib/core/",  # the verifiers\n]\n',
    "job.sh": "#!/bin/sh\ngit rev-parse HEAD:lib/core HEAD:other\ncd lib/core\n",
}
CORE_COVER = '    "verity/config/",\n    "verity/protocols/proto/",  # the verifiers\n'


def _covered(tmp_path: Path, core: str = CORE_COVER, cd: str = "lib/core") -> dict:
    """The range that sends `lib/core/` three ways and deletes its own `__init__.py`, with `gates.toml`'s `core` list
    written as `core` and `job.sh` changing directory to `cd`."""
    repo, base = _repo(tmp_path, COVERED)
    for old, new in (("lib/core/wire", "verity/config/wire"), ("lib/core/proto", "verity/protocols/proto"),
                     ("lib/core/claims.py", "verity/config/claims.py"), ("lib/commit", "verity/commit")):
        _move(repo, old, new)
    _git(repo, "rm", "-q", "lib/core/__init__.py")
    job = f"#!/bin/sh\ngit rev-parse HEAD:verity/config HEAD:verity/protocols/proto HEAD:other\ncd {cd}\n"
    _write(repo, {"verity/__init__.py": "", "verity/config/__init__.py": "", "job.sh": job,
                  "gates.toml": f'low_risk = ["docs/", "verity/", "verity/commit/"]\ncore = [\n{core}]\n'})
    return _check(repo, base)


def test_a_list_of_paths_rewritten_as_an_exact_cover_of_where_its_files_went_is_certified(tmp_path):
    """No one path stands for `lib/core/` after the range, so the mover writes a list entry that names it as the fewest
    paths that hold its files (`tools/move/layout.py`'s `_covers`). The certificate takes any list whose paths hold, of the
    files HEAD has that the range did not add, exactly those the old list's paths held: beside `lib/commit/` that is
    `verity/` whole, alone the two parts, in either order. git reads the deleted `lib/core/__init__.py` as renamed to the
    new `verity/__init__.py`, one empty file as good as another: where no file of its directory went with it, the first is
    deleted and the second added, so the two parts still hold all `lib/core/` held."""
    r = _covered(tmp_path / "mover")
    assert r["certified"], r["refusals"]
    assert {"old": "lib/core/__init__.py", "new": "verity/__init__.py"} in r["renames"]
    assert r["rules"] == {"cover": ["gates.toml", "job.sh"]}
    r = _covered(tmp_path / "order", '    "verity/protocols/proto/",\n    "verity/config/",  # the verifiers\n')
    assert r["certified"], r["refusals"]


def test_a_list_of_paths_that_holds_more_or_less_than_its_old_entries_held_is_refused(tmp_path):
    """A cover is exact. `verity/` holds `lib/commit/`'s file too, which `core` never named; `verity/config/` alone lost the
    protocol; a path that holds only what the range added (`verity/__init__.py`) covers nothing; a list whose comment
    changed is another edit; and a word of a command that takes one directory (`cd`) is no list."""
    for case, core in (("wider", '    "verity/",  # the verifiers\n'), ("narrower", '    "verity/config/",  # the verifiers\n'),
                       ("added", CORE_COVER + '    "verity/__init__.py",\n'),
                       ("comment", CORE_COVER.replace("the verifiers", "the trusted code"))):
        r = _covered(tmp_path / case, core)
        assert [f["path"] for f in r["refusals"]] == ["gates.toml"] and r["rules"]["cover"] == ["job.sh"], case
    r = _covered(tmp_path / "cd", cd="verity/config")
    assert [f["path"] for f in r["refusals"]] == ["job.sh"] and r["rules"]["cover"] == ["gates.toml"]


def test_a_covered_lists_comment_and_its_entry_that_is_no_path_change_as_a_line_may(tmp_path):
    """A list the range covers also holds a comment that names a moved file and an entry the tree does not have, below a
    directory that moved whole. Both follow the move as they would on any line, and the list is a cover. Refused: a
    comment whose other word changed, and the made-up entry sent anywhere else."""
    core = '    # read beside {c}, {why}\n    "verity/config/",\n    "verity/protocols/proto/",\n    "{made}",\n'
    for case, c, why, made in (("follows", "verity/commit/c.py", "the verifiers", "verity/commit/made.json"),
                               ("word", "verity/commit/c.py", "the checkers", "verity/commit/made.json"),
                               ("elsewhere", "verity/commit/c.py", "the verifiers", "verity/config/made.json")):
        gates = 'core = [\n    # read beside lib/commit/c.py, the verifiers\n    "lib/core/",\n    "lib/commit/made.json",\n]\n'
        repo, base = _repo(tmp_path / case, {**COVERED, "gates.toml": gates, "job.sh": "#!/bin/sh\n"})
        for old, new in (("lib/core/wire", "verity/config/wire"), ("lib/core/proto", "verity/protocols/proto"),
                         ("lib/core/claims.py", "verity/config/claims.py"), ("lib/commit", "verity/commit")):
            _move(repo, old, new)
        _git(repo, "rm", "-q", "lib/core/__init__.py")
        _write(repo, {"verity/__init__.py": "", "verity/config/__init__.py": "",
                      "gates.toml": f"core = [\n{core.format(c=c, why=why, made=made)}]\n"})
        r = _check(repo, base)
        if case == "follows":
            assert r["certified"] and r["rules"] == {"cover": ["gates.toml"]}, r["refusals"]
        else:
            assert [f["path"] for f in r["refusals"]] == ["gates.toml"], case


def test_a_cover_may_keep_the_place_the_range_declares_for_an_entry_and_add_its_other_parts(tmp_path):
    """`moves.json` sends `lib/h/` to `old/lib/h/` and its `face/` elsewhere, so a line that names `lib/h/` may follow
    to `old/lib/h/`, which no longer holds `face/`. A list that names it there and `face/`'s new place beside it holds
    what it held: a cover. With a second entry that holds a file the list never held, it is refused."""
    moves = {"about": "x", "moves": []}
    for case, face in (("parts", "bench/face/"), ("wider", "bench/")):
        repo, base = _repo(tmp_path / case, {MC.MOVES: json.dumps(moves, indent=2) + "\n", "lib/h/a.py": "A = 1\n",
                                             "lib/h/face/b.py": "B = 1\n", "bench/other.py": "O = 1\n",
                                             "gates.toml": 'owns = ["lib/h/"]\n'})
        _move(repo, "lib/h/face", "bench/face")
        _move(repo, "lib/h/a.py", "old/lib/h/a.py")
        paths = {"lib/h/": "old/lib/h/", "lib/h/face/": "bench/face/"}
        _write(repo, {MC.MOVES: json.dumps({**moves, "moves": [{"paths": paths}]}, indent=2) + "\n",
                      "gates.toml": f'owns = ["old/lib/h/", "{face}"]\n'})
        r = _check(repo, base)
        if case == "parts":
            assert r["certified"] and r["rules"] == {"cover": ["gates.toml"]}, r["refusals"]
        else:
            assert [f["path"] for f in r["refusals"]] == ["gates.toml"]


def _levels(tmp_path: Path, low_risk: str, marks: str) -> dict:
    """`_covered`'s range, which also moves `other/notes.md` into `docs/` and makes `verity/commit/` a package, on a
    `gates.toml` whose `low_risk` names `docs/` and `lib/core/` and whose `marks` names `lib/core/`'s own `__init__.py`
    and a module of it; the two lists as given."""
    gates = 'low_risk = ["docs/", "lib/core/"]\nmarks = ["lib/core/__init__.py", "lib/core/claims.py"]\n'
    repo, base = _repo(tmp_path, {**COVERED, "other/notes.md": "n\n", "gates.toml": gates, "job.sh": "#!/bin/sh\n"})
    for old, new in (("lib/core/wire", "verity/config/wire"), ("lib/core/proto", "verity/protocols/proto"),
                     ("lib/core/claims.py", "verity/config/claims.py"), ("lib/commit", "verity/commit"),
                     ("other/notes.md", "docs/notes.md")):
        _move(repo, old, new)
    _git(repo, "rm", "-q", "lib/core/__init__.py")
    _write(repo, {"verity/__init__.py": "", "verity/config/__init__.py": "", "verity/commit/__init__.py": "",
                  "gates.toml": f"low_risk = [{low_risk}]\nmarks = [{marks}]\n"})
    return _check(repo, base)


LOW_RISK = '"docs/", "verity/config/", "verity/protocols/proto/"'
MARKS = '"verity/__init__.py", "verity/config/__init__.py", "verity/config/claims.py"'


def test_a_cover_judges_what_the_range_rewrote_and_takes_the_package_levels_above_it(tmp_path):
    """An entry the range sent whole is where it went, whatever else the range moved there: `docs/` gains a file the list
    never held and is still the list's entry, beside the two parts of `lib/core/`. And a list that named a package's own
    `__init__.py`, which the range deleted with the package, may name the levels above what the range wrote anew in it:
    the mover writes them there, since an import of the moved module runs each."""
    r = _levels(tmp_path, LOW_RISK, MARKS)
    assert r["certified"], r["refusals"]
    assert r["rules"]["cover"] == ["gates.toml"]


def test_a_level_above_nothing_the_list_names_or_in_a_list_that_named_none_is_refused(tmp_path):
    """`verity/commit/` is a package the range made, but `marks` names nothing below it; `low_risk` named no level, so it
    gains none; and a part that holds a file the old list never held (`docs/notes.md`, written beside `docs/` as its own
    entry) is no part."""
    for case, low_risk, marks in (("above nothing", LOW_RISK, MARKS + ', "verity/commit/__init__.py"'),
                                  ("named none", LOW_RISK + ', "verity/__init__.py"', MARKS),
                                  ("never held", LOW_RISK.replace('"docs/"', '"docs/a.md", "docs/notes.md"'), MARKS)):
        r = _levels(tmp_path / case, low_risk, marks)
        assert [f["path"] for f in r["refusals"]] == ["gates.toml"], case


def test_an_empty_init_above_the_packages_root_is_refused(tmp_path):
    """The mover stops at the root of the moved module's package, which the path map at BASE names. An `__init__.py`
    above it makes a package of a directory that was none (here `security/`, over the root `security/pous`), which no
    move needs; and with no map naming a root, no level is the mover's."""
    for case, maps, let_in in (("root", {"tools/move/layout_one/map.toml": '[roots]\nnew = { pous = "security/pous" }\n'},
                                {"security/pous/__init__.py"}), ("no-map", {}, set())):
        repo, base = _repo(tmp_path / case, {**BASE_FILES, **maps})
        _move(repo, "protocols/pous/src/verity_pous/codec.py", "security/pous/codec.py")
        _write(repo, {"security/__init__.py": "", "security/pous/__init__.py": "",
                      "tests/test_pous.py": BASE_FILES["tests/test_pous.py"].replace("verity_pous.codec", "pous.codec")})
        r = _check(repo, base, {"verity_pous.codec": "pous.codec"})
        assert {e["path"] for e in r["edits"] if e["kind"] == "package level"} == let_in, case
        assert {f["path"] for f in r["refusals"]} == {"security/__init__.py", "security/pous/__init__.py"} - let_in, case


def test_a_map_that_does_not_parse_at_base_stops_the_check(tmp_path):
    """The package roots come from every `.toml` under `tools/move/` at BASE, and a table of another shape names none.
    One that doesn't parse might have named a root, so the check stops on it by name rather than read it as naming none."""
    repo, base = _repo(tmp_path / "shape", {**BASE_FILES, "tools/move/other.toml": 'roots = "none"\n[words]\na = "b"\n'})
    _move(repo, "protocols/pous/src/verity_pous/codec.py", "security/pous/codec.py")
    assert "certified" in _check(repo, base, {"verity_pous.codec": "security.pous.codec"})
    repo, base = _repo(tmp_path / "broken", {**BASE_FILES, "tools/move/layout_one/map.toml": "[roots\n"})
    _move(repo, "protocols/pous/src/verity_pous/codec.py", "security/pous/codec.py")
    with pytest.raises(SystemExit, match=r"tools/move/layout_one/map\.toml doesn't parse at BASE"):
        _check(repo, base, {"verity_pous.codec": "security.pous.codec"})


def test_names_with_spaces_and_odd_characters(tmp_path):
    files = {"old dir/a file.txt": "one\n", "old dir/tab\there.txt": "two\n", 'old dir/quo"te.txt': "three\n"}
    repo, base = _repo(tmp_path, files)
    _move(repo, "old dir", "new dir")
    r = _check(repo, base)
    assert r["certified"], r["refusals"]
    assert sorted(x["new"] for x in r["renames"]) == ["new dir/a file.txt", 'new dir/quo"te.txt', "new dir/tab\there.txt"]
    assert r["map"] == {"old dir": "new dir"}


def test_lean_package_move_with_relative_require(tmp_path):
    files = {**BASE_FILES, "packages/verity/lean/Verity.lean": "def v := 1\n", "protocols/pous/lean/lakefile.toml": LAKEFILE}
    repo, base = _repo(tmp_path, files)
    _move(repo, "packages/verity/lean", "verity/lean")
    _move(repo, "protocols/pous/lean", "security_proofs/pous")
    _write(repo, {"security_proofs/pous/lakefile.toml": LAKEFILE.replace("../../../packages/verity/lean", "../../verity/lean")})
    r = _check(repo, base)
    assert r["certified"], r["refusals"]
    assert r["edits"] == [{"path": "security_proofs/pous/lakefile.toml", "old": "protocols/pous/lean/lakefile.toml", "kind": "path"}]


def test_byte_identical_lakefile_whose_relative_require_broke_is_refused(tmp_path):
    files = {**BASE_FILES, "packages/verity/lean/Verity.lean": "def v := 1\n", "protocols/pous/lean/lakefile.toml": LAKEFILE}
    repo, base = _repo(tmp_path, files)
    _move(repo, "protocols/pous/lean", "security_proofs/pous")
    [f] = _check(repo, base)["refusals"]
    assert f["reason"] == "relative path now resolves elsewhere"


def test_lean_audit_pin_change_is_refused(tmp_path):
    repo, base = _repo(tmp_path, BASE_FILES)
    _move(repo, "protocols/pous", "security_proofs/pous")
    _write(repo, {"security_proofs/pous/lean/lean-audit.json": '{"guarantees": {"Pous.t": "abd"}}\n'})
    [f] = _check(repo, base)["refusals"]
    assert f["path"] == "security_proofs/pous/lean/lean-audit.json"


def test_explicit_map_and_cli(tmp_path, capsys, monkeypatch):
    repo, base = _repo(tmp_path, {**BASE_FILES, "AGENTS.md": "See `protocols/pous/lean/Pous.lean`.\n"})
    _write(repo, {"AGENTS.md": "See `elsewhere/Pous.lean`.\n"})
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "doc")
    monkeypatch.chdir(repo)
    assert MC.main([base, "HEAD", "--json"]) == 1
    assert not json.loads(capsys.readouterr().out)["certified"]
    assert MC.main([base, "HEAD", "--map", "protocols/pous/lean=elsewhere"]) == 0
    assert capsys.readouterr().out.startswith("move-check: certified")


SPEC = "packages/verity/lean"
PROOFS = "security_proofs/core"
LAKEFILE_SPEC = ('name = "verity"\ndefaultTargets = ["Verity"]\n\n[[require]]\nname = "mathlib"\n'
                 'git = "https://github.com/leanprover-community/mathlib4"\nrev = "abc"\n\n# Verity core\'s Lean.\n'
                 '[[lean_lib]]\nname = "Verity"\n')
LAKEFILE_PROOFS = ('name = "coreproofs"\ndefaultTargets = ["CoreProofs"]\n\n[[require]]\nname = "mathlib"\n'
                   'git = "https://github.com/leanprover-community/mathlib4"\nrev = "abc"\n\n[[require]]\nname = "verity"\n'
                   'path = "../../packages/verity/lean"\n\n# The proofs of core\'s guarantees.\n[[lean_lib]]\nname = "CoreProofs"\n'
                   'globs = ["CoreProofs.+"]\n')
MATHLIB = {"url": "https://github.com/leanprover-community/mathlib4", "type": "git", "name": "mathlib", "rev": "abc",
           "inherited": False, "configFile": "lakefile.lean"}
VERITY_PATH = {"type": "path", "scope": "", "name": "verity", "manifestFile": "lake-manifest.json", "inherited": False,
               "dir": "../../packages/verity/lean", "configFile": "lakefile.toml"}
PINS = {"Verity.SecurityProofs.G": {"signature": ["Verity.SecurityProofs.G : Verity.G"], "assumptions": [], "type_hash": "aa"}}
READS = {"Verity.Guarantees": {"digest": "dd", "definitions": {"Verity.G": "ee"}}}
FOO = "import Specs.Core.Guarantees\n\nnamespace Proofs.Core\n\ntheorem G : Verity.G := rfl\n\nend Proofs.Core\n"
BAR = ("import Verity.SecurityProofs.Foo\nimport Mathlib.Tactic\n\nnamespace Proofs.Core\n\n"
       "theorem H : 1 = 1 := rfl\n\nend Verity.SecurityProofs\n")


def _manifest(name: str, *packages: dict) -> str:
    return json.dumps({"version": "1.2.0", "packagesDir": ".lake/packages", "packages": list(packages), "name": name}, indent=1)


def _lock(**fields) -> str:
    return json.dumps(fields, indent=2) + "\n"


SPEC_FILES = {
    f"{SPEC}/lakefile.toml": LAKEFILE_SPEC, f"{SPEC}/lean-toolchain": "leanprover/lean4:v4.99.0\n",
    f"{SPEC}/lake-manifest.json": _manifest("verity", MATHLIB),
    f"{SPEC}/lean-audit.json": _lock(roots=["Verity"], layers={"Verity.Protocol": []}, pins=PINS, reads=READS),
    f"{SPEC}/Verity.lean": "import Definitions.Core\nimport Specs.Core.Guarantees\nimport Proofs.Core\n",
    f"{SPEC}/Verity/Protocol.lean": "def Verity.p := 1\n",
    f"{SPEC}/Verity/Guarantees.lean": "import Definitions.Core\n\ndef Verity.G : Prop := Verity.p = 1\n",
    f"{SPEC}/Verity/SecurityProofs.lean": "import Verity.SecurityProofs.Foo\nimport Verity.SecurityProofs.Bar\n",
    f"{SPEC}/Verity/SecurityProofs/Foo.lean": FOO,
    f"{SPEC}/Verity/SecurityProofs/Bar.lean": BAR,
}


def _split(tmp_path: Path, bar: str | None = None, pins: dict | None = None, files: dict | None = None,
           base: dict | None = None) -> tuple[Path, str, dict]:
    """Core's proofs split out to `security_proofs/core` under the module root `CoreProofs`; the spec and its umbrella stay."""
    repo, base = _repo(tmp_path, {**SPEC_FILES, **(base or {})})
    for m in ("Foo", "Bar"):
        _move(repo, f"{SPEC}/Verity/SecurityProofs/{m}.lean", f"{PROOFS}/CoreProofs/{m}.lean")
    split = {
        f"{PROOFS}/CoreProofs/Bar.lean": bar if bar is not None else BAR.replace("import Verity.SecurityProofs.Foo", "import CoreProofs.Foo"),
        f"{SPEC}/Verity/SecurityProofs.lean": "import CoreProofs.Foo\nimport CoreProofs.Bar\n",
        f"{SPEC}/lean-audit.json": _lock(roots=["Verity.Protocol", "Verity.Guarantees", "Verity.SecurityProofs"],
                                         layers={"Verity.Protocol": []}, pins=pins or PINS, reads=READS,
                                         proved_in=["../../../security_proofs/core"]),
        f"{PROOFS}/lakefile.toml": LAKEFILE_PROOFS, f"{PROOFS}/lean-toolchain": "leanprover/lean4:v4.99.0\n",
        f"{PROOFS}/lake-manifest.json": _manifest("coreproofs", VERITY_PATH, MATHLIB),
        f"{PROOFS}/lean-audit.json": _lock(roots=["CoreProofs"], layers={"Verity.Protocol": []}),
    }
    _write(repo, {**split, **(files or {})})
    return repo, base, _check(repo, base)


def test_lean_split_is_certified(tmp_path):
    _, _, r = _split(tmp_path)
    assert r["certified"], r["refusals"]
    assert r["lean_modules"] == {"Verity.SecurityProofs.Bar": "CoreProofs.Bar", "Verity.SecurityProofs.Foo": "CoreProofs.Foo"}
    assert r["renames"] == [{"old": f"{SPEC}/Verity/SecurityProofs/Foo.lean", "new": f"{PROOFS}/CoreProofs/Foo.lean"}]
    kinds = {e["path"]: e["kind"] for e in r["edits"]}
    assert kinds == {f"{PROOFS}/CoreProofs/Bar.lean": "lean", f"{SPEC}/Verity/SecurityProofs.lean": "lean",
                     f"{SPEC}/lean-audit.json": "lock", **{f"{PROOFS}/{n}": "copy" for n in sorted(MC.LAKE_COPIES)
                                                             if n != "lakefile.lean"}}


def test_module_names_follow_src_dir(tmp_path):
    """A nested package with `srcDir = ".."` names its modules from the directory above it, as Lake does."""
    repo, base = _repo(tmp_path, {"verity/Security/lakefile.toml": 'name = "security"\n[[lean_lib]]\nname = "Spec"\n',
                                  "verity/Security/Spec/Pouw.lean": "",
                                  "verity/Security/Proofs/lakefile.toml": 'name = "proofs"\nsrcDir = ".."\n',
                                  "verity/Security/Proofs/Pouw/End.lean": "",
                                  "p/lakefile.toml": '[[lean_lib]]\nname = "P"\nsrcDir = "src"\n', "p/src/P/A.lean": "",
                                  "p/Check.lean": ""})
    tree = MC.Tree(base, str(repo))
    assert tree.module("verity/Security/Spec/Pouw.lean") == "Spec.Pouw"
    assert tree.module("verity/Security/Proofs/Pouw/End.lean") == "Proofs.Pouw.End"
    assert tree.module("p/src/P/A.lean") == "P.A" and tree.module("p/Check.lean") == "Check"


def test_lean_split_that_changes_a_proof_line_is_refused(tmp_path):
    bar = BAR.replace("import Verity.SecurityProofs.Foo", "import CoreProofs.Foo").replace(":= rfl", ":= by rfl")
    [f] = _split(tmp_path, bar=bar)[2]["refusals"]
    assert f["path"] == f"{PROOFS}/CoreProofs/Bar.lean" and "+theorem H : 1 = 1 := by rfl" in f["hunk"]


def test_lean_split_that_changes_a_guarantee_record_is_refused(tmp_path):
    pins = {"Verity.SecurityProofs.G": {**PINS["Verity.SecurityProofs.G"], "type_hash": "ab"}}
    [f] = _split(tmp_path, pins=pins)[2]["refusals"]
    assert f["path"] == f"{SPEC}/lean-audit.json" and f["reason"].startswith("`pins` changed")
    assert '"type_hash": "aa"' in f["hunk"] and '"type_hash": "ab"' in f["hunk"]


def test_lean_import_of_a_module_that_did_not_move_is_refused(tmp_path):
    bar = BAR.replace("import Verity.SecurityProofs.Foo", "import CoreProofs.Other")
    [f] = _split(tmp_path, bar=bar)[2]["refusals"]
    assert f["path"] == f"{PROOFS}/CoreProofs/Bar.lean" and "+import CoreProofs.Other" in f["hunk"]


def test_proofs_package_lakefile_with_a_new_git_dependency_is_refused(tmp_path):
    lakefile = LAKEFILE_PROOFS + '\n[[require]]\nname = "extra"\ngit = "https://example.com/extra"\n'
    [f] = _split(tmp_path, files={f"{PROOFS}/lakefile.toml": lakefile})[2]["refusals"]
    assert f["path"] == f"{PROOFS}/lakefile.toml" and '+git = "https://example.com/extra"' in f["hunk"]


def test_proofs_package_lock_that_lists_guarantees_is_refused(tmp_path):
    [f] = _split(tmp_path, files={f"{PROOFS}/lean-audit.json": _lock(roots=["CoreProofs"], pins=PINS)})[2]["refusals"]
    assert f["path"] == f"{PROOFS}/lean-audit.json" and "lists no guarantees" in f["reason"]


def test_spec_lock_proved_in_that_names_no_package_is_refused(tmp_path):
    repo, base = _repo(tmp_path, SPEC_FILES)
    _write(repo, {f"{SPEC}/lean-audit.json": _lock(roots=["Verity"], layers={"Verity.Protocol": []}, pins=PINS, reads=READS,
                                                   proved_in=["../../security_proofs/core"])})
    [f] = _check(repo, base)["refusals"]
    assert "`proved_in` entry" in f["reason"]


#: the numerical bench's lowerings, which name a directory in a pathlib chain
LOWERINGS = "benchmarks/numerical/python/verity_numerical/bench/lowerings.py"
REGISTRY = 'REGISTRY = {\n    "flock_pure": "backends.flock.tool:FLOCK_PURE",\n    "commit_cost": "benchmarks.commitments.tool:COMMIT_COST",\n}\n'
REGISTRY_FILES = {"backends/flock/tool.py": "FLOCK_PURE = 1\n", "backends/flock/prove.py": "",
                  "benchmarks/commitments/tool.py": "COMMIT_COST = 1\n",
                  "tools/research/src/research/store/tools_registry.py": REGISTRY,
                  LOWERINGS:
                      'DIRECTORIES = {\n    "C-Flock": REPO / "backends" / "flock" / "templates",\n}\n',
                  "tools/research/src/research/store/parity.py":
                      'argv = [sys.executable, "-m", f"backends.flock.{mod}", "--root", str(root)]\n'}


def test_tools_registry_dotted_rewrite_is_certified(tmp_path):
    repo, base = _repo(tmp_path, REGISTRY_FILES)
    _move(repo, "backends/flock", "catalog/flock")
    _write(repo, {"tools/research/src/research/store/tools_registry.py": REGISTRY.replace("backends.flock", "catalog.flock"),
                  LOWERINGS:
                      'DIRECTORIES = {\n    "C-Flock": REPO / "catalog" / "flock" / "templates",\n}\n',
                  "tools/research/src/research/store/parity.py":
                      'argv = [sys.executable, "-m", f"catalog.flock.{mod}", "--root", str(root)]\n'})
    r = _check(repo, base)
    assert r["certified"], r["refusals"]
    assert {e["path"] for e in r["edits"]} == {"tools/research/src/research/store/tools_registry.py",
                                               LOWERINGS,
                                               "tools/research/src/research/store/parity.py"}


def test_tools_registry_non_path_change_is_refused(tmp_path):
    repo, base = _repo(tmp_path, REGISTRY_FILES)
    _move(repo, "backends/flock", "catalog/flock")
    _write(repo, {"tools/research/src/research/store/tools_registry.py":
                  REGISTRY.replace("backends.flock", "catalog.flock").replace("FLOCK_PURE", "FLOCK_FAST")})
    [f] = _check(repo, base)["refusals"]
    assert f["path"] == "tools/research/src/research/store/tools_registry.py" and "FLOCK_FAST" in f["hunk"]


def test_dotted_name_no_rename_explains_is_refused(tmp_path):
    repo, base = _repo(tmp_path, {**REGISTRY_FILES, "tools/research/src/research/x.py": 'NAME = "backends.flock.tool"\n'})
    _move(repo, "backends/flock", "catalog/flock")
    _write(repo, {"tools/research/src/research/x.py": 'NAME = "catalog.other.tool"\n'})
    [f] = _check(repo, base)["refusals"]
    assert f["path"] == "tools/research/src/research/x.py" and "catalog.other.tool" in f["hunk"]


WORKSPACE = '[tool.uv.workspace]\nmembers = ["protocols/pous", "protocols/pouw"]\n\n[tool.uv.sources]\npous = { workspace = true }\n'


def _new_member(tmp_path: Path, members: str, sources: str = "") -> dict:
    repo, base = _repo(tmp_path, {**BASE_FILES, "pyproject.toml": WORKSPACE})
    _move(repo, "protocols/pous", "security_proofs/pous")
    _write(repo, {"pyproject.toml": WORKSPACE.replace('"protocols/pous"', '"security_proofs/pous"')
                  .replace('"protocols/pouw"]', members + "]") + sources})
    return _check(repo, base)


def test_workspace_member_for_a_new_package_is_certified(tmp_path):
    r = _new_member(tmp_path, '"protocols/pouw", "security_proofs/pous/src"')
    assert r["certified"], r["refusals"]


def test_workspace_member_for_an_old_or_missing_directory_is_refused(tmp_path):
    for i, extra in enumerate(('"protocols/pouw", "tests"', '"protocols/pouw", "nowhere"')):
        [f] = _new_member(tmp_path / str(i), extra)["refusals"]
        assert f["path"] == "pyproject.toml"


def test_uv_source_for_a_package_not_created_by_the_range_is_refused(tmp_path):
    [f] = _new_member(tmp_path, '"protocols/pouw"', "other = { workspace = true }\n")["refusals"]
    assert f["path"] == "pyproject.toml" and "+other = { workspace = true }" in f["hunk"]


LEAN_PACKAGES = 'PACKAGES = [\n    ROOT / "packages" / "verity" / "lean",\n]\n'


def test_lean_packages_entry_for_a_new_lake_package_is_certified(tmp_path):
    entry = '    ROOT / "security_proofs" / "core",\n'
    _, _, r = _split(tmp_path, base={"tests/test_lean_packages.py": LEAN_PACKAGES},
                     files={"tests/test_lean_packages.py": LEAN_PACKAGES.replace("]\n", entry + "]\n")})
    assert r["certified"], r["refusals"]
    assert {"path": "tests/test_lean_packages.py", "kind": "python"} in r["edits"]


def test_lean_packages_entry_for_an_existing_package_is_refused(tmp_path):
    entry = '    ROOT / "packages" / "verity" / "lean",\n'
    _, _, r = _split(tmp_path, base={"tests/test_lean_packages.py": LEAN_PACKAGES},
                     files={"tests/test_lean_packages.py": LEAN_PACKAGES.replace("]\n", entry + "]\n")})
    [f] = r["refusals"]
    assert f["path"] == "tests/test_lean_packages.py"


RUN = {"why": "random instances", "args": ["{difftest/instances.jsonl}", "10000"],
       "data": {"difftest/instances.jsonl": "c048"}, "generate": ["python3", "scripts/difftest_vectors.py", "{repo}", "--out", "{out}"]}


def _runs_split(tmp_path: Path, proofs_runs: dict | None, spec_runs: dict | None = None, was: dict = RUN) -> dict:
    """The split, with core's lock holding a difftest (``was``) whose runner module and generator move to the proofs package."""
    base_files = {**SPEC_FILES, f"{SPEC}/Verity/SecurityProofs/Difftest.lean": "def main : IO Unit := pure ()\n",
                  f"{SPEC}/scripts/difftest_vectors.py": "print(1)\n",
                  f"{SPEC}/lean-audit.json": _lock(roots=["Verity"], layers={"Verity.Protocol": []}, pins=PINS, reads=READS,
                                                   runs={"Verity.SecurityProofs.Difftest": was})}
    repo, base = _repo(tmp_path, base_files)
    for m in ("Foo", "Bar", "Difftest"):
        _move(repo, f"{SPEC}/Verity/SecurityProofs/{m}.lean", f"{PROOFS}/CoreProofs/{m}.lean")
    _move(repo, f"{SPEC}/scripts", f"{PROOFS}/scripts")
    spec_lock = dict(roots=["Verity.Protocol", "Verity.Guarantees", "Verity.SecurityProofs"], layers={"Verity.Protocol": []},
                     pins=PINS, reads=READS, proved_in=["../../../security_proofs/core"])
    _write(repo, {
        f"{PROOFS}/CoreProofs/Bar.lean": BAR.replace("import Verity.SecurityProofs.Foo", "import CoreProofs.Foo"),
        f"{SPEC}/Verity/SecurityProofs.lean": "import CoreProofs.Foo\nimport CoreProofs.Bar\n",
        f"{SPEC}/lean-audit.json": _lock(**spec_lock, **({"runs": spec_runs} if spec_runs is not None else {})),
        f"{PROOFS}/lakefile.toml": LAKEFILE_PROOFS, f"{PROOFS}/lean-toolchain": "leanprover/lean4:v4.99.0\n",
        f"{PROOFS}/lake-manifest.json": _manifest("coreproofs", VERITY_PATH, MATHLIB),
        f"{PROOFS}/lean-audit.json": _lock(roots=["CoreProofs"], layers={"Verity.Protocol": []},
                                           **({"runs": proofs_runs} if proofs_runs is not None else {})),
    })
    return _check(repo, base)


def test_runs_entry_moved_to_the_proofs_lock_is_certified(tmp_path):
    r = _runs_split(tmp_path, {"CoreProofs.Difftest": RUN})
    assert r["certified"], r["refusals"]


def test_runs_entry_that_vanishes_is_refused(tmp_path):
    [f] = _runs_split(tmp_path, None)["refusals"]
    assert f["path"] == f"{SPEC}/lean-audit.json" and "is in no lock at HEAD" in f["reason"]


def test_runs_entry_with_a_changed_data_hash_is_refused(tmp_path):
    [f] = _runs_split(tmp_path, {"CoreProofs.Difftest": {**RUN, "data": {"difftest/instances.jsonl": "c049"}}})["refusals"]
    assert f["path"] == f"{PROOFS}/lean-audit.json" and "changed beyond moved paths" in f["reason"] and "c049" in f["hunk"]


def test_runs_entry_duplicated_into_two_locks_is_refused(tmp_path):
    r = _runs_split(tmp_path, {"CoreProofs.Difftest": RUN}, spec_runs={"CoreProofs.Difftest": RUN})
    assert any("locks at HEAD" in f["reason"] for f in r["refusals"])


WORD_MAP = ('[words]\n"Models/Core" = "Definitions/Core"\n"Models.Core" = "Definitions.Core"\n"Trusted" = "Checked"\n'
            '"S/Models/" = "S/Definitions/"\n\'"Models" / "Core"\' = \'"Definitions" / "Core"\'\n'
            "'Models\\.Core' = 'Definitions\\.Core'\n'\"Models\" / \"Hash\"' = '\"Definitions\" / \"Hash\"'\n"
            '\n[rewrite]\nexclude = []\n')
WORD_BASE = {
    "tools/move/demo_map.toml": WORD_MAP,
    "S/lakefile.toml": 'name = "s"\n\n[[lean_lib]]\nname = "Models"\n',
    "S/Models/Core/A.lean": "import Models.Core.B\n\n/-! Trusted: `Models/Core/B.lean` states `Models.Core.B`. -/\n",
    "S/Models/Core/B.lean": "def x := 1\n",
    "S/lean-audit.json": _lock(roots=["Models"], layers={"Models.Core": {"imports": ["Models.Core"]}},
                               reads={"Models.Core.B": {"digest": "dd", "definitions": {"x": "ee"}}}, dependencies={"m": "1"}),
    "notes.md": "The core's statements are `S/Models/Core/` (`Models.Core`). Trusted.\n",
    "more.md": "`S/Models/` states them.\n",
    "t.py": 'CORE = ROOT / "S" / "Models" / "Core"\nHASH = ROOT / "S" / "Models" / "Hash"\nB = r"Models\\.Core\\.B"\n',
    "p/TRUSTED.sha256": "",
    "tools/research/src/research/store/moves.json": json.dumps({"about": "x", "moves": [{"paths": {"S/Specs/": "S/Models/"}}]},
                                                               indent=2) + "\n",
}


def _sha(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode()).hexdigest()


def _worded(tmp_path: Path, base_pins: str | None = None, base_extra: dict | None = None, **over: str) -> dict:
    """`Models/Core/` moved to `Definitions/Core/` with the map's words applied, as `tools/move/rename.py` does it."""
    base_files = {**WORD_BASE, **(base_extra or {}),
                  "p/TRUSTED.sha256": base_pins or f"{_sha(WORD_BASE['S/Models/Core/A.lean'])}  ../S/Models/Core/A.lean\n"}
    repo, base = _repo(tmp_path, base_files)
    for m in ("A", "B"):
        _move(repo, f"S/Models/Core/{m}.lean", f"S/Definitions/Core/{m}.lean")
    a = "import Definitions.Core.B\n\n/-! Trusted: `Definitions/Core/B.lean` states `Definitions.Core.B`. -/\n"
    head = {
        "S/Definitions/Core/A.lean": a,
        "S/lean-audit.json": _lock(roots=["Models"], layers={"Definitions.Core": {"imports": ["Definitions.Core"]}},
                                   reads={"Definitions.Core.B": {"digest": "dd", "definitions": {"x": "ee"}}},
                                   dependencies={"m": "2"}),
        "notes.md": "The core's statements are `S/Definitions/Core/` (`Definitions.Core`). Trusted.\n",
        "more.md": "`S/Definitions/` states them.\n",
        "t.py": 'CORE = ROOT / "S" / "Definitions" / "Core"\nHASH = ROOT / "S" / "Models" / "Hash"\nB = r"Definitions\\.Core\\.B"\n',
        "p/TRUSTED.sha256": f"{_sha(a)}  ../S/Definitions/Core/A.lean\n",
        "tools/research/src/research/store/moves.json": json.dumps(
            {"about": "x", "moves": [{"paths": {"S/Specs/": "S/Models/"}}, {"paths": {"S/Models/": "S/Definitions/"}}]},
            indent=2) + "\n",
    }
    _write(repo, {**head, **over})
    return _check(repo, base)


def test_a_word_map_move_is_certified(tmp_path):
    """What `tools/move/rename.py` writes: the map's words in any text file, a pin's digest following its file, the record of
    the move appended to `moves.json`, a lock's `reads` keyed by the moved module and its `dependencies` record."""
    r = _worded(tmp_path)
    assert r["certified"], r["refusals"]
    kinds = {e["path"]: e["kind"] for e in r["edits"]}
    assert kinds["notes.md"] == kinds["more.md"] == kinds["S/Definitions/Core/A.lean"] == kinds["t.py"] == "words"
    assert kinds["p/TRUSTED.sha256"] == "text"


def test_a_list_entry_the_words_repeat_may_go(tmp_path):
    """Two directories the range merged, both listed: the words list the one twice, and the repeat may go; any other entry
    stays."""
    old = '{\n  "owns": [\n    "S/Definitions/Core/",\n    "S/Models/Core/",\n    "S/Proofs/"\n  ]\n}\n'
    once = '{\n  "owns": [\n    "S/Definitions/Core/",\n    "S/Proofs/"\n  ]\n}\n'
    (tmp_path / "ok").mkdir()
    r = _worded(tmp_path / "ok", base_extra={"owners.json": old}, **{"owners.json": once})
    assert r["certified"], r["refusals"]
    (tmp_path / "dropped").mkdir()
    dropped = '{\n  "owns": [\n    "S/Definitions/Core/",\n    "S/Definitions/Core/"\n  ]\n}\n'
    assert not _worded(tmp_path / "dropped", base_extra={"owners.json": old}, **{"owners.json": dropped})["certified"]


def _caps(areas: dict, indent: int = 1) -> str:
    """A file whose writer keeps its areas sorted, as `tools/verity/lean/unread.py` writes its caps."""
    return json.dumps({"schema": "caps/v1", "areas": areas}, indent=indent) + "\n"


CAPS = {"Definitions/Hash": {"modules": 1}, "Models/Core": {"modules": 7}, "Proofs/Core": {"modules": 2}}
RESORTED = {"Definitions/Core": {"modules": 7}, "Definitions/Hash": {"modules": 1}, "Proofs/Core": {"modules": 2}}


def test_rename_py_sorts_a_renamed_key_into_its_place_and_that_is_a_move(tmp_path):
    """`rename.py` renames the key `Models/Core` and puts it in its sorted place, and its move is certified, the caps as the
    words' edit."""
    repo, base = _repo(tmp_path, {
        "tools/move/caps_map.toml": '[words]\n"Models/Core" = "Definitions/Core"\n\n[rewrite]\nexclude = ["tools/move/**"]\n',
        "S/Models/Core/A.lean": "def a := 1\n",
        "caps.json": _caps(CAPS),
        "tools/research/src/research/store/moves.json": json.dumps({"about": "x", "moves": []}, indent=2) + "\n"})
    assert MC._rename().main(["--map", str(repo / "tools/move/caps_map.toml"), "--root", str(repo)]) == 0
    _git(repo, "commit", "-q", "-m", "move")
    assert (repo / "caps.json").read_text() == _caps(RESORTED)
    r = MC.certify(base, _git(repo, "rev-parse", "HEAD"), cwd=str(repo))
    assert r["certified"], r["refusals"]
    assert {"path": "caps.json", "kind": "words"} in r["edits"]


def test_a_resorted_object_with_anything_else_changed_is_refused(tmp_path):
    """The reorder is a move only as `rename.py` writes it: a changed value, or the same value laid out anew, is refused
    with its first hunk."""
    for case, after in (("value", _caps({**RESORTED, "Definitions/Core": {"modules": 8}})), ("layout", _caps(RESORTED, 2))):
        (tmp_path / case).mkdir()
        [f] = _worded(tmp_path / case, base_extra={"caps.json": _caps(CAPS)}, **{"caps.json": after})["refusals"]
        assert f["path"] == "caps.json" and f["hunk"], case


def test_a_reorder_of_an_object_unsorted_before_or_after_is_refused(tmp_path):
    """`rename.py` sorts again only an object that was sorted, and sorts it: one that wasn't sorted at BASE keeps the order
    the words leave it in, and one that was comes back sorted."""
    for case, before, after in (("before", dict(reversed(CAPS.items())), RESORTED),
                                ("after", CAPS, dict(reversed(RESORTED.items())))):
        (tmp_path / case).mkdir()
        [f] = _worded(tmp_path / case, base_extra={"caps.json": _caps(before)}, **{"caps.json": _caps(after)})["refusals"]
        assert f["path"] == "caps.json" and f["hunk"], case


def test_a_chain_or_regex_spelling_is_kept_only_with_the_word_it_spells(tmp_path):
    """`"Models" / "Hash"` spells `Models/Hash`, which no kept word renames (nothing under `Models/Hash` moved), so the map
    doesn't explain renaming it."""
    base = {"t.py": 'CORE = S / "Models" / "Core"\nHASH = S / "Models" / "Hash"\nB = r"Models\\.Core\\.B"\n'}
    kept = 'CORE = S / "Definitions" / "Core"\nHASH = S / "Models" / "Hash"\nB = r"Definitions\\.Core\\.B"\n'
    (tmp_path / "kept").mkdir()
    r = _worded(tmp_path / "kept", base_extra=base, **{"t.py": kept})
    assert r["certified"], r["refusals"]
    (tmp_path / "hash").mkdir()
    [f] = _worded(tmp_path / "hash", base_extra=base, **{"t.py": kept.replace('"Models" / "Hash"', '"Definitions" / "Hash"')})["refusals"]
    assert f["path"] == "t.py"


def test_a_word_that_renames_nothing_the_range_moved_is_refused(tmp_path):
    """`Trusted` is in the map, but no path or module the range moved has it, so the map doesn't explain renaming it."""
    [f] = _worded(tmp_path, **{"notes.md": "The core's statements are `S/Definitions/Core/` (`Definitions.Core`). Checked.\n"})["refusals"]
    assert f["path"] == "notes.md" and "+The core's statements are `S/Definitions/Core/` (`Definitions.Core`). Checked." in f["hunk"]


DECL_BASE = {
    "tools/move/sample_map.toml": '[words]\n"DrawIsCoin" = "SampleIsCoin"\n\n[[words_in]]\nfiles = ["P/S/Law.lean", '
                                  '"P/S/Use.lean"]\nwords = { "escape" = "acceptProb" }\n\n[rewrite]\n'
                                  'exclude = ["tools/move/**", "**/CHANGELOG.md"]\n',
    "P/lakefile.toml": 'name = "p"\n',
    "P/S/Law.lean": "/-- The coin is a draw. -/\ndef DrawIsCoin : Prop := True\n\n/-- An escape. -/\ndef escape : Nat := 1\n",
    "P/S/Use.lean": "import S.Law\n\ntheorem t : DrawIsCoin := trivial\ntheorem u : escape = 1 := rfl\n",
    "P/lean-audit.json": _lock(assumptions={"DrawIsCoin": {"owner": "@lean"}},
                               reads={"S.Law": {"digest": "dd", "definitions": {"DrawIsCoin": "ee", "escape": "ff"}}}),
    "notes.md": "`DrawIsCoin` holds; an escape is rare.\n",
    "CHANGELOG.md": "Added `DrawIsCoin`.\n",
}
DECL_HEAD = {
    "P/S/Law.lean": "/-- The coin is a draw. -/\ndef SampleIsCoin : Prop := True\n\n/-- An acceptProb. -/\ndef acceptProb : Nat := 1\n",
    "P/S/Use.lean": "import S.Law\n\ntheorem t : SampleIsCoin := trivial\ntheorem u : acceptProb = 1 := rfl\n",
    "P/lean-audit.json": _lock(assumptions={"SampleIsCoin": {"owner": "@lean"}},
                               reads={"S.Law": {"digest": "dd", "definitions": {"DrawIsCoin": "ee", "escape": "ff"}}}),
    "notes.md": "`SampleIsCoin` holds; an escape is rare.\n",
}


def _declared(tmp_path: Path, base_extra: dict | None = None, **over: str) -> dict:
    """`DrawIsCoin` and the scoped `escape` renamed as `tools/move/rename.py` renames them: no path moves."""
    repo, base = _repo(tmp_path, {**DECL_BASE, **(base_extra or {})})
    _write(repo, {**DECL_HEAD, **over})
    return _check(repo, base)


def test_a_lean_declaration_rename_is_certified(tmp_path):
    """A word that renames a declaration of an edited `.lean` file to a name the base never spells is kept, as is a
    `[[words_in]]` entry's in its files; a lock's records keep the old names for `check.py --update --moved`."""
    r = _declared(tmp_path)
    assert r["certified"], r["refusals"]
    assert {e["path"]: e["kind"] for e in r["edits"]} == dict.fromkeys(["P/S/Law.lean", "P/S/Use.lean", "P/lean-audit.json",
                                                                        "notes.md"], "words")


def test_a_declaration_rename_onto_a_name_the_base_spells_is_refused(tmp_path):
    """`SampleIsCoin` is already a name at the base, so renaming `DrawIsCoin` to it would merge two names."""
    r = _declared(tmp_path, base_extra={"other.md": "`SampleIsCoin` is another premise.\n"})
    assert {f["path"] for f in r["refusals"]} == {"P/S/Law.lean", "P/S/Use.lean", "P/lean-audit.json", "notes.md"}


def test_a_new_name_a_lean_comment_already_uses_is_no_merge(tmp_path):
    """A `.lean` file's prose may already say `acceptProb`: a comment names nothing, so the rename stays certified. The same
    word in its code is a name the rename would merge with."""
    for kind, line in [("comment", "-- the acceptProb is P_a\n"), ("code", "theorem v : acceptProb = acceptProb := rfl\n")]:
        (tmp_path / kind).mkdir()
        old, new = DECL_BASE["P/S/Use.lean"] + line, DECL_HEAD["P/S/Use.lean"] + line
        r = _declared(tmp_path / kind, base_extra={"P/S/Use.lean": old}, **{"P/S/Use.lean": new})
        assert r["certified"] == (kind == "comment"), r["refusals"]


def test_a_map_of_scoped_words_alone_is_read(tmp_path):
    """A map whose `[words]` is empty renames only in the files of its `[[words_in]]` entries, and is read all the same."""
    scoped = ('[words]\n\n[[words_in]]\nfiles = ["P/S/Law.lean", "P/S/Use.lean"]\nwords = { "escape" = "acceptProb" }\n\n'
              '[rewrite]\nexclude = ["tools/move/**", "**/CHANGELOG.md"]\n')
    keep = {p: DECL_BASE[p] for p in ("notes.md", "P/lean-audit.json")}
    r = _declared(tmp_path, base_extra={"tools/move/sample_map.toml": scoped}, **keep, **{
        "P/S/Law.lean": DECL_BASE["P/S/Law.lean"].replace("escape", "acceptProb"),
        "P/S/Use.lean": DECL_BASE["P/S/Use.lean"].replace("escape", "acceptProb")})
    assert r["certified"], r["refusals"]
    assert {e["path"] for e in r["edits"]} == {"P/S/Law.lean", "P/S/Use.lean"}


def test_a_scoped_word_renames_only_in_its_files(tmp_path):
    """`escape` is renamed only in the files its `[[words_in]]` entry names; a prose `escape` elsewhere stays."""
    [f] = _declared(tmp_path, **{"notes.md": "`SampleIsCoin` holds; an acceptProb is rare.\n"})["refusals"]
    assert f["path"] == "notes.md"


def test_a_stale_word_of_another_map_is_not_kept(tmp_path):
    """An earlier move's map also renames `DrawIsCoin`, to a name the base no longer spells; the range didn't make that
    rename, so its word is not applied."""
    stale = '[words]\n"DrawIsCoin" = "CoinIsDrawn"\n\n[rewrite]\nexclude = ["tools/move/**"]\n'
    r = _declared(tmp_path, base_extra={"tools/move/a_stale_map.toml": stale})
    assert r["certified"], r["refusals"]


def test_a_word_no_edited_lean_file_declares_is_refused(tmp_path):
    """Renaming a declaration's mentions while its `.lean` files stay as they were renames no declaration."""
    repo, base = _repo(tmp_path, DECL_BASE)
    _write(repo, {"notes.md": DECL_HEAD["notes.md"]})
    [f] = _check(repo, base)["refusals"]
    assert f["path"] == "notes.md"


def test_lean_declared_reads_declarations_constructors_and_fields():
    """Each name as written and in full: below its namespaces, a field or a constructor below its structure or type; a
    section or a `mutual` block adds no part, and `end` closes it."""
    text = ("/-- `def notMe` in a docstring. -/\n@[simp] private theorem a_b : True := trivial\nnamespace Law.X\n"
            "structure S where\n  draw : Nat\n  h : draw = 1 := by rfl\ninductive T\n  | leaf\n  | node (l : T) : T\n"
            "section\ndef f : T → Nat\n  | .leaf => 0\n  | x => 1\nend\n-- def notMeEither\nend Law.X\n"
            "noncomputable def escape := 1\n")
    assert MC.lean_declared(text) == ["a_b", "Law.X", "S", "Law.X.S", "draw", "Law.X.S.draw", "h", "Law.X.S.h", "T",
                                      "Law.X.T", "leaf", "Law.X.T.leaf", "node", "Law.X.T.node", "f", "Law.X.f", "escape"]


AREAS = {
    "tools/move/areas_map.toml": '[words]\n"S/Area" = "S/Area/Check"\n"S.Area" = "S.Area.Check"\n',
    "P/lakefile.toml": 'name = "p"\n',
    "P/S/Area.lean": "import S.Area.Check\n",
    "P/S/Area/Check.lean": "import S.Area.Grid\n\ntheorem t : True := trivial\n",
    "P/S/Area/Grid.lean": "def g := 1\n",
    MC.MOVES: json.dumps({"about": "x", "moves": []}, indent=2) + "\n",
}
AREAS_MOVED = {"P/S/Area.lean": "P/S/Area/Check.lean", "P/S/Area/Check.lean": "P/S/Area/Check/Check.lean",
               "P/S/Area/Grid.lean": "P/S/Area/Check/Grid.lean"}


def _areas(tmp_path: Path, **over: str) -> dict:
    """`S.Area` becomes `S.Area.Check` and its `Check` `S.Area.Check.Check`: a file lands where another left, and an
    aggregator whose every line the words rename goes where the map sends it, as `tools/move/rename.py` writes them."""
    repo, base = _repo(tmp_path, AREAS)
    _move(repo, "P/S/Area/Grid.lean", "P/S/Area/Check/Grid.lean")
    _move(repo, "P/S/Area/Check.lean", "P/S/Area/Check/Check.lean")
    _move(repo, "P/S/Area.lean", "P/S/Area/Check.lean")
    _write(repo, {"P/S/Area/Check.lean": "import S.Area.Check.Check\n",
                  "P/S/Area/Check/Check.lean": "import S.Area.Check.Grid\n\ntheorem t : True := trivial\n",
                  MC.MOVES: json.dumps({"about": "x", "moves": [{"paths": AREAS_MOVED}]}, indent=2) + "\n", **over})
    return _check(repo, base)


def test_a_file_landing_where_another_left_is_a_move(tmp_path):
    r = _areas(tmp_path)
    assert r["certified"], r["refusals"]
    assert r["lean_modules"] == {"S.Area": "S.Area.Check", "S.Area.Check": "S.Area.Check.Check",
                                 "S.Area.Grid": "S.Area.Check.Grid"}


def test_a_file_landing_where_another_left_is_recorded_where_it_went(tmp_path):
    swapped = {**AREAS_MOVED, "P/S/Area/Check.lean": "P/S/Area/Grid.lean"}
    [f] = _areas(tmp_path, **{MC.MOVES: json.dumps({"about": "x", "moves": [{"paths": swapped}]}, indent=2) + "\n"})["refusals"]
    assert f["path"] == MC.MOVES


def test_a_file_landing_where_another_left_is_judged_like_any_edit(tmp_path):
    [f] = _areas(tmp_path, **{"P/S/Area/Check.lean": "import S.Area.Check.Check\ntheorem u : True := trivial\n"})["refusals"]
    assert f["path"] == "P/S/Area/Check.lean" and f.get("old") == "P/S/Area.lean"


def test_a_pin_whose_digest_is_not_its_file_is_refused(tmp_path):
    [f] = _worded(tmp_path, **{"p/TRUSTED.sha256": f"{'0' * 64}  ../S/Definitions/Core/A.lean\n"})["refusals"]
    assert f["path"] == "p/TRUSTED.sha256"


MANIFEST_PIN = "verity/protocols/flock/prover/tests/fixtures/randomness_lean.json"
MANIFEST = '{"packages": [{"type": "path", "dir": "../pkg/old/lean"}]}\n'


def _manifest_pin(tmp_path: Path, pinned) -> dict:
    """`pkg/old/` moved to `pkg/new/`, so `Sec/lake-manifest.json`'s `dir` follows it, and the flock Lean vectors' pin
    (a JSON pin of `pins.py`) records the manifest and a Lean file as ``pinned(manifest, lean)`` gives them at HEAD."""
    lean = "theorem a : 1 = 1 := rfl\n"

    def pin(manifest: str, lean_digest: str) -> str:
        return json.dumps({"inputs": {"Sec/A.lean": lean_digest, "Sec/lake-manifest.json": _sha(manifest)}, "kat": [1]},
                          indent=2) + "\n"
    repo, base = _repo(tmp_path, {"Sec/lake-manifest.json": MANIFEST, "Sec/A.lean": lean, "pkg/old/lean/B.lean": "def b := 1\n",
                                  MANIFEST_PIN: pin(MANIFEST, _sha(lean))})
    _move(repo, "pkg/old/lean/B.lean", "pkg/new/lean/B.lean")
    moved = MANIFEST.replace("pkg/old", "pkg/new")
    _write(repo, {"Sec/lake-manifest.json": moved, MANIFEST_PIN: pin(*pinned(moved, _sha(lean)))})
    return _check(repo, base)


def test_a_json_pins_manifest_digest_follows_the_manifest(tmp_path):
    """A move rewrites a Lake manifest's `dir`, and a JSON pin that records the manifest takes its new digest: no pinned
    output reads where a path package sits."""
    r = _manifest_pin(tmp_path, lambda manifest, lean: (manifest, lean))
    assert r["certified"], r["refusals"]


def test_a_json_pins_other_digests_and_wrong_ones_are_refused(tmp_path):
    for case, pinned in {"wrong": lambda manifest, lean: (manifest + " ", lean),
                         "lean": lambda manifest, lean: (manifest, "0" * 64)}.items():
        (tmp_path / case).mkdir()
        [f] = _manifest_pin(tmp_path / case, pinned)["refusals"]
        assert f["path"] == MANIFEST_PIN, case


def test_pins_sorted_anew_after_their_paths_moved_are_a_move(tmp_path):
    """The command that writes a pin file sorts its paths, so a regeneration after the move may order them anew."""
    a, b = "import Definitions.Core.B\n\n/-! Trusted: `Definitions/Core/B.lean` states `Definitions.Core.B`. -/\n", "def x := 1\n"
    base_pins = (f"{_sha(WORD_BASE['S/Models/Core/A.lean'])}  ../S/Models/Core/A.lean\n"
                 f"{_sha(b)}  ../S/Models/Core/B.lean\n")
    swapped = f"{_sha(b)}  ../S/Definitions/Core/B.lean\n{_sha(a)}  ../S/Definitions/Core/A.lean\n"
    (tmp_path / "ok").mkdir()
    r = _worded(tmp_path / "ok", base_pins, **{"p/TRUSTED.sha256": swapped})
    assert r["certified"], r["refusals"]
    (tmp_path / "dropped").mkdir()
    dropped = f"{_sha(a)}  ../S/Definitions/Core/A.lean\n"
    assert not _worded(tmp_path / "dropped", base_pins, **{"p/TRUSTED.sha256": dropped})["certified"]


def _doc_paths(tmp_path: Path, lean: str, md: str) -> dict:
    """`P/x` moved to `P/y`; a Lean doc comment names it from the root, a Markdown file relative to itself."""
    repo, base = _repo(tmp_path, {"P/x/A.lean": "def a := 1\n", "S/B.lean": "/-! `P/x/A.lean` states `a`. -/\ndef b := 1\n",
                                  "docs/README.md": "`../P/x/A.lean` states it ([A](../P/x/A.lean)).\n"})
    _move(repo, "P/x", "P/y")
    _write(repo, {"S/B.lean": lean, "docs/README.md": md})
    return _check(repo, base)


def test_moved_paths_in_a_lean_comment_and_markdown_relative_paths_are_a_move(tmp_path):
    r = _doc_paths(tmp_path, "/-! `P/y/A.lean` states `a`. -/\ndef b := 1\n", "`../P/y/A.lean` states it ([A](../P/y/A.lean)).\n")
    assert r["certified"], r["refusals"]


def test_a_lean_comment_edit_that_is_not_path_only_is_refused(tmp_path):
    """Only a moved path may change in a Lean comment: not its prose, not a path the move didn't make, nor the code."""
    md = "`../P/y/A.lean` states it ([A](../P/y/A.lean)).\n"
    for case, lean in {"prose": "/-! `P/y/A.lean` states `b`. -/\ndef b := 1\n",
                       "unmade": "/-! `P/z/A.lean` states `a`. -/\ndef b := 1\n",
                       "code": "/-! `P/y/A.lean` states `a`. -/\ndef b := 2\n"}.items():
        (tmp_path / case).mkdir()
        [f] = _doc_paths(tmp_path / case, lean, md)["refusals"]
        assert f["path"] == "S/B.lean", case


def test_a_markdown_relative_path_the_move_did_not_make_is_refused(tmp_path):
    """A Markdown relative path may only follow the move: one naming a path the move didn't make, or another file, is
    refused, in a link's target and in backticks."""
    lean = "/-! `P/y/A.lean` states `a`. -/\ndef b := 1\n"
    for case, md in {"link": "`../P/y/A.lean` states it ([A](../P/z/A.lean)).\n",
                     "backticks": "`../P/z/A.lean` states it ([A](../P/y/A.lean)).\n",
                     "outside": "`../P/y/A.lean` states it ([A](../../P/y/A.lean)).\n"}.items():
        (tmp_path / case).mkdir()
        [f] = _doc_paths(tmp_path / case, lean, md)["refusals"]
        assert f["path"] == "docs/README.md", case


def test_a_moves_record_of_a_path_that_did_not_move_whole_is_refused(tmp_path):
    moves = json.dumps({"about": "x", "moves": [{"paths": {"S/Specs/": "S/Models/"}}, {"paths": {"S/": "T/"}}]}, indent=2) + "\n"
    [f] = _worded(tmp_path, **{"tools/research/src/research/store/moves.json": moves})["refusals"]
    assert f["path"] == "tools/research/src/research/store/moves.json" and "T/Models/Core/A.lean" in f["reason"]


SPLIT_DIR = {
    "p/circuit/__init__.py": "",
    "p/circuit/a.py": "from p.circuit.live import T\n\nA = T + 1\n",
    "p/circuit/live.py": "T = 1\n",
    "p/circuit/a.sidecar.py": "from p.circuit.a import A\n",
    "p/other.py": "X = 1\n",
    "suite/pyproject.toml": 'inputs = ["p/circuit", "p/other.py"]\n',
    MC.MOVES: json.dumps({"about": "x", "moves": []}, indent=2) + "\n",
}


def _split_dir(tmp_path: Path, paths: dict[str, str] | None) -> dict:
    """`p/circuit/` moves to `q/circuit/` but for `live.py`, which goes to `p/`: no rename pairs the directory, and the
    one-line sidecar whose import is rewritten is below git's similarity. ``paths`` is the entry appended to `moves.json`."""
    repo, base = _repo(tmp_path, SPLIT_DIR)
    _move(repo, "p/circuit/live.py", "p/live.py")
    for f in ("__init__.py", "a.py", "a.sidecar.py"):
        _move(repo, f"p/circuit/{f}", f"q/circuit/{f}")
    _write(repo, {"q/circuit/a.py": "from p.live import T\n\nA = T + 1\n", "q/circuit/a.sidecar.py": "from q.circuit.a import A\n",
                  "suite/pyproject.toml": 'inputs = ["q/circuit", "p/other.py"]\n'})
    if paths is not None:
        _write(repo, {MC.MOVES: json.dumps({"about": "x", "moves": [{"paths": paths}]}, indent=2) + "\n"})
    return _check(repo, base)


def test_the_appended_moves_entry_pairs_what_no_rename_does(tmp_path):
    r = _split_dir(tmp_path, {"p/circuit/live.py": "p/live.py", "p/circuit/": "q/circuit/"})
    assert r["certified"], r["refusals"]
    assert r["map"]["p/circuit"] == "q/circuit"
    assert {e["path"] for e in r["edits"]} >= {"suite/pyproject.toml", "q/circuit/a.sidecar.py", MC.MOVES}


def test_without_the_entry_a_split_directory_is_refused(tmp_path):
    paths = {f["path"] for f in _split_dir(tmp_path, None)["refusals"]}
    assert {"suite/pyproject.toml", "q/circuit/a.sidecar.py"} <= paths


def test_an_entry_the_range_does_not_bear_out_adds_nothing(tmp_path):
    """An entry is a map only where the range moved each file it names to its place: `r/circuit/` is nowhere at HEAD."""
    r = _split_dir(tmp_path, {"p/circuit/live.py": "p/live.py", "p/circuit/": "r/circuit/"})
    assert "p/circuit" not in r["map"]
    assert {MC.MOVES, "suite/pyproject.toml"} <= {f["path"] for f in r["refusals"]}


def test_a_moves_file_that_is_not_json_declares_nothing_and_is_refused(tmp_path):
    repo, base = _repo(tmp_path, SPLIT_DIR)
    _write(repo, {MC.MOVES: '{"about": "x", "moves": [\n'})
    r = _check(repo, base)
    assert [(f["path"], f["reason"]) for f in r["refusals"]] == [(MC.MOVES, "not JSON")]


def test_a_lock_field_the_writer_wraps_anew_is_still_a_move(tmp_path):
    """`check.py --update` wraps a list a renamed module lengthened past its line: the same value, laid out anew."""
    lock = _lock(roots=["Models"], layers={"Definitions.Core": {"imports": ["Definitions.Core"]}},
                 reads={"Definitions.Core.B": {"digest": "dd", "definitions": {"x": "ee"}}}, dependencies={"m": "2"})
    wrapped = lock.replace('[\n        "Definitions.Core"\n      ]', '["Definitions.Core"]')
    assert wrapped != lock
    r = _worded(tmp_path, **{"S/lean-audit.json": wrapped})
    assert r["certified"], r["refusals"]


def test_a_reads_record_that_changes_with_its_key_is_refused(tmp_path):
    lock = _lock(roots=["Models"], layers={"Definitions.Core": {"imports": ["Definitions.Core"]}},
                 reads={"Definitions.Core.B": {"digest": "de", "definitions": {"x": "ee"}}}, dependencies={"m": "2"})
    [f] = _worded(tmp_path, **{"S/lean-audit.json": lock})["refusals"]
    assert f["reason"].startswith("`reads` changed")


PROPS = 'name = "props"\nsrcDir = ".."\n\n[[lean_lib]]\nname = "Props"\nroots = ["Props.A", "Props.Proofs.A"]\n'
PROPS_PATH = {"type": "path", "scope": "", "name": "props", "manifestFile": "lake-manifest.json", "inherited": False,
              "dir": "../Sec/Props", "configFile": "lakefile.toml"}
PP_PATH = {**PROPS_PATH, "name": "pp", "dir": "../Sec/Proofs/Props"}


def _in_place(tmp_path: Path, **over: str) -> dict:
    """`Props.Proofs` split out of package `props` into a new package `pp` inside `Sec/Proofs/`: `props` drops its roots,
    `q`, which imports the proof, requires `pp`, and a suite's inputs name `pp`'s Lake files."""
    repo, base = _repo(tmp_path, {
        "Sec/Props/lakefile.toml": PROPS, "Sec/Props/lean-toolchain": "leanprover/lean4:v4.99.0\n",
        "Sec/Props/lake-manifest.json": _manifest("props"), "Sec/Props/lean-audit.json": _lock(roots=["Props.A", "Props.Proofs.A"]),
        "Sec/Props/A.lean": "def a := 1\n", "Sec/Props/Proofs/A.lean": "import Props.A\n\ntheorem t : a = 1 := rfl\n",
        "q/lakefile.toml": 'name = "q"\n\n[[require]]\nname = "props"\npath = "../Sec/Props"\n',
        "q/lake-manifest.json": _manifest("q", PROPS_PATH), "q/Q.lean": "import Props.Proofs.A\n",
        "x/pyproject.toml": 'inputs = [\n    "Sec/Props/lakefile.toml",\n]\n'})
    _move(repo, "Sec/Props/Proofs/A.lean", "Sec/Proofs/Props/A.lean")
    _write(repo, {
        "Sec/Proofs/Props/lakefile.toml": ('name = "pp"\nsrcDir = "../.."\n\n[[require]]\nname = "props"\npath = "../../Props"\n'
                                           '\n[[lean_lib]]\nname = "PP"\nroots = ["Proofs.Props.A"]\n'),
        "Sec/Proofs/Props/lean-toolchain": "leanprover/lean4:v4.99.0\n",
        "Sec/Proofs/Props/lake-manifest.json": _manifest("pp", {**PROPS_PATH, "dir": "../../Props"}),
        "Sec/Proofs/Props/lean-audit.json": _lock(roots=["Proofs.Props.A"], guarantees={}),
        "Sec/Props/lakefile.toml": PROPS.replace(', "Props.Proofs.A"', ""),
        "Sec/Props/lean-audit.json": _lock(roots=["Props.A"], proved_in=["../Proofs/Props"]),
        "q/lakefile.toml": ('name = "q"\n\n[[require]]\nname = "props"\npath = "../Sec/Props"\n\n# the proofs\n[[require]]\n'
                            'name = "pp"\npath = "../Sec/Proofs/Props"\n'),
        "q/lake-manifest.json": _manifest("q", PROPS_PATH, PP_PATH), "q/Q.lean": "import Proofs.Props.A\n",
        "x/pyproject.toml": 'inputs = [\n    "Sec/Props/lakefile.toml",\n    "Sec/Proofs/Props/lakefile.toml", "Sec/Proofs/Props/lean-toolchain",\n]\n',
        **over})
    return _check(repo, base)


def test_a_split_into_a_package_inside_another_tree_is_certified(tmp_path):
    r = _in_place(tmp_path)
    assert r["certified"], r["refusals"]
    assert r["lean_modules"] == {"Props.Proofs.A": "Proofs.Props.A"}


def test_an_in_place_lakefile_that_drops_a_root_that_did_not_move_is_refused(tmp_path):
    [f] = _in_place(tmp_path, **{"Sec/Props/lakefile.toml": PROPS.replace('"Props.A", ', "").replace(', "Props.Proofs.A"', "")})["refusals"]
    assert f["path"] == "Sec/Props/lakefile.toml"


def test_an_in_place_require_of_an_existing_package_is_refused(tmp_path):
    lakefile = 'name = "q"\n\n[[require]]\nname = "props"\npath = "../Sec/Props"\n\n[[require]]\nname = "q2"\npath = "../Sec"\n'
    r = _in_place(tmp_path, **{"q/lakefile.toml": lakefile})
    assert [f["path"] for f in r["refusals"]] == ["q/lakefile.toml"]


def test_an_inputs_line_naming_an_existing_package_is_refused(tmp_path):
    r = _in_place(tmp_path, **{"x/pyproject.toml": 'inputs = [\n    "Sec/Props/lakefile.toml",\n    "Sec/Props/lean-toolchain",\n]\n'})
    assert [f["path"] for f in r["refusals"]] == ["x/pyproject.toml"]


def test_a_fold_into_an_existing_package_moves_its_roots(tmp_path):
    """`Props.Proofs` folded into `Sec/Proofs`'s existing package (srcDir `..`): `props` drops the roots and `proofs` gains them."""
    proofs = 'name = "proofs"\nsrcDir = ".."\n\n[[lean_lib]]\nname = "Proofs"\nroots = ["Proofs.B"]\n'
    repo, base = _repo(tmp_path, {
        "Sec/Props/lakefile.toml": PROPS, "Sec/Props/A.lean": "def a := 1\n",
        "Sec/Props/Proofs/A.lean": "import Props.A\n\ntheorem t : a = 1 := rfl\n",
        "Sec/Proofs/lakefile.toml": proofs, "Sec/Proofs/B.lean": "import Props.Proofs.A\n"})
    _move(repo, "Sec/Props/Proofs/A.lean", "Sec/Proofs/Props/A.lean")
    _write(repo, {"Sec/Props/lakefile.toml": PROPS.replace(', "Props.Proofs.A"', ""),
                  "Sec/Proofs/lakefile.toml": proofs.replace('"Proofs.B"]', '"Proofs.B", "Proofs.Props.A"]'),
                  "Sec/Proofs/B.lean": "import Proofs.Props.A\n"})
    r = _check(repo, base)
    assert r["certified"], r["refusals"]
    _write(repo, {"Sec/Proofs/lakefile.toml": proofs.replace('"Proofs.B"]', '"Proofs.B", "Proofs.Props.A", "Proofs.C"]')})
    [f] = _check(repo, base)["refusals"]
    assert f["path"] == "Sec/Proofs/lakefile.toml"


#: a package below `pkg/` that moves deeper, the shape of the vLLM integration's first move
APP = {"pkg/app/__init__.py": "", "pkg/app/store/__init__.py": "", "pkg/app/store/remote.py": "class Session:\n    pass\n",
       "pkg/app/store/host.py": "x = 1\n", "pkg/app/linear/__init__.py": "", "pkg/app/linear/api.py": "y = 2\n",
       "pkg/tests/store/__init__.py": "", "pkg/tests/store/test_remote.py": "ROOT = Path(__file__).resolve().parents[2]\n",
       ".gitignore": "pkg/tests/store/fixtures/periods\n"}
APP_MOVES = [("pkg/app/store", "pkg/app/protocols/bandwidth"), ("pkg/app/linear", "pkg/app/protocols/linear"),
             ("pkg/tests/store", "pkg/tests/protocols/bandwidth")]


def _app(tmp_path: Path, files: dict[str, str], edits: dict[str, str]) -> dict:
    repo, base = _repo(tmp_path, {**APP, **files})
    for old, new in APP_MOVES:
        _move(repo, old, new)
    _write(repo, edits)
    return _check(repo, base)


def test_prose_paths_and_module_names_in_any_text_file_are_certified(tmp_path):
    r = _app(tmp_path, {
        "README.md": "The store (`pkg/app/store/`, `app/store/remote.py`) is `app.store.remote.Session`.\n"
                     "Its test is `tests/store/test_remote.py`.\n",
        "run.sh": 'git show "$SHA:pkg/app/store/host.py" > host.py  # app.linear.api\n',
        "keep.json": '{"app.linear.api": "pkg/app/linear/api.py"}\n',
        "pkg/app/doc.py": '"""Through `app.store.remote.Session` (pkg/app/store/remote.py)."""\n'}, {
        "README.md": "The store (`pkg/app/protocols/bandwidth/`, `app/protocols/bandwidth/remote.py`) is "
                     "`app.protocols.bandwidth.remote.Session`.\nIts test is `tests/store/test_remote.py`.\n",
        "run.sh": 'git show "$SHA:pkg/app/protocols/bandwidth/host.py" > host.py  # app.protocols.linear.api\n',
        "keep.json": '{"app.protocols.linear.api": "pkg/app/protocols/linear/api.py"}\n',
        "pkg/app/doc.py": '"""Through `app.protocols.bandwidth.remote.Session` (pkg/app/protocols/bandwidth/remote.py)."""\n',
        "pkg/tests/protocols/bandwidth/test_remote.py": "ROOT = Path(__file__).resolve().parents[3]\n",
        ".gitignore": "pkg/tests/protocols/bandwidth/fixtures/periods\n"})
    assert r["certified"], r["refusals"]
    assert {e["path"]: e["kind"] for e in r["edits"]} == {
        "README.md": "text", "run.sh": "text", "keep.json": "text", "pkg/app/doc.py": "python", ".gitignore": "text",
        "pkg/tests/protocols/bandwidth/test_remote.py": "python"}


PY_OLD = ("from app import linear, store as ST\nfrom app.store import remote as R\n\n"
          'HOST = Path(__file__).resolve().parents[0] / "store" / "host.py"\nUP = Path(R.__file__).parents[1]\n')
PY_NEW = ("from app.protocols import linear\nfrom app.protocols import bandwidth as ST\nfrom app.protocols.bandwidth import remote as R\n\n"
          'HOST = Path(__file__).resolve().parents[0] / "protocols" / "bandwidth" / "host.py"\nUP = Path(R.__file__).parents[2]\n')


def test_moved_modules_imported_by_name_and_anchors_with_a_path_are_certified(tmp_path):
    r = _app(tmp_path, {"pkg/app/use.py": PY_OLD}, {"pkg/app/use.py": PY_NEW})
    assert r["certified"], r["refusals"]


def test_imports_in_a_shell_heredoc_and_in_a_string_literal_are_certified(tmp_path):
    r = _app(tmp_path, {"run.sh": "python - <<'EOF'\nfrom app import linear, store as ST\nEOF\n",
                        "pkg/app/gen.py": 'SRC = (\n    "from app.store import remote as R\\n"\n)\n'},
             {"run.sh": "python - <<'EOF'\nfrom app.protocols import linear\nfrom app.protocols import bandwidth as ST\nEOF\n",
              "pkg/app/gen.py": 'SRC = (\n    "from app.protocols.bandwidth import remote as R\\n"\n)\n'})
    assert r["certified"], r["refusals"]


def test_an_import_in_a_string_literal_that_binds_another_name_is_refused(tmp_path):
    [f] = _app(tmp_path, {"pkg/app/gen.py": 'SRC = (\n    "from app.store import remote as R\\n"\n)\n'},
               {"pkg/app/gen.py": 'SRC = (\n    "from app.protocols.bandwidth import remote as S\\n"\n)\n'})["refusals"]
    assert f["path"] == "pkg/app/gen.py" and "as S" in f["hunk"]


def test_an_import_that_binds_another_name_is_refused(tmp_path):
    [f] = _app(tmp_path, {"pkg/app/use.py": PY_OLD},
               {"pkg/app/use.py": PY_NEW.replace("bandwidth as ST", "bandwidth as BW")})["refusals"]
    assert f["path"] == "pkg/app/use.py" and "BW" in f["hunk"]


PAREN_OLD = "from app import store as ST\nfrom app.store.remote import (Session,\n                              Other)\n"


def test_a_parenthesized_import_continued_on_an_unchanged_line_is_certified(tmp_path):
    r = _app(tmp_path, {"pkg/app/use.py": PAREN_OLD}, {"pkg/app/use.py": PAREN_OLD.replace(
        "from app import store as ST", "from app.protocols import bandwidth as ST").replace(
        "app.store.remote", "app.protocols.bandwidth.remote")})
    assert r["certified"], r["refusals"]


def test_a_parenthesized_import_of_a_module_that_did_not_move_there_is_refused(tmp_path):
    for old in (PAREN_OLD, PAREN_OLD.replace("import (Session,", "import (")):
        new = old.replace("from app import store as ST", "from app.protocols import bandwidth as ST").replace(
            "app.store.remote", "app.protocols.linear.api")
        [f] = _app(tmp_path / str(len(old)), {"pkg/app/use.py": old}, {"pkg/app/use.py": new})["refusals"]
        assert f["path"] == "pkg/app/use.py" and "linear.api" in f["hunk"]


def test_a_module_anchor_that_names_another_directory_is_refused(tmp_path):
    [f] = _app(tmp_path, {"pkg/app/use.py": PY_OLD},
               {"pkg/app/use.py": PY_NEW.replace("parents[2]", "parents[3]")})["refusals"]
    assert f["path"] == "pkg/app/use.py" and "parents[3]" in f["hunk"]


def test_a_word_that_names_no_moved_path_is_refused_in_prose(tmp_path):
    [f] = _app(tmp_path, {"README.md": "The store is `app.store`.\n"},
               {"README.md": "The store is `app.protocols.storage`.\n"})["refusals"]
    assert f["path"] == "README.md" and f["reason"] == "edit beyond moved paths and module names"


WORD_OLD = {"examples/toy/run.sh": "echo toy\n",
            "README.md": "The end-to-end examples (`examples/toy/run.sh`) are the milestone's.\n",
            "registry.json": '{"examples": ["who runs `examples/`?"], "owns": ["examples/toy/"]}\n',
            "pkg/app/ladder.py": 'def examples(root):\n    return sorted((root / "examples").iterdir())\n'}
WORD_NEW = {"README.md": "The end-to-end examples (`demos/toy/run.sh`) are the milestone's.\n",
            "registry.json": '{"examples": ["who runs `demos/`?"], "owns": ["demos/toy/"]}\n',
            "pkg/app/ladder.py": 'def examples(root):\n    return sorted((root / "demos").iterdir())\n'}


def _word(tmp_path: Path, edits: dict[str, str]) -> dict:
    repo, base = _repo(tmp_path, WORD_OLD)
    _move(repo, "examples/toy/run.sh", "demos/toy/run.sh")
    _write(repo, edits)
    return _check(repo, base)


def test_a_moved_top_level_name_left_standing_as_a_word_is_certified(tmp_path):
    r = _word(tmp_path, WORD_NEW)
    assert r["certified"], r["refusals"]


def test_a_moved_top_level_name_rewritten_to_anything_else_is_refused(tmp_path):
    [f] = _word(tmp_path, {**WORD_NEW, "README.md": WORD_NEW["README.md"].replace("end-to-end examples", "end-to-end samples")})[
        "refusals"]
    assert f["path"] == "README.md" and "samples" in f["hunk"]


def test_anchor_that_names_another_directory_is_refused(tmp_path):
    [f] = _app(tmp_path, {}, {"pkg/tests/protocols/bandwidth/test_remote.py": "ROOT = Path(__file__).resolve().parents[4]\n"})[
        "refusals"]
    assert f["path"] == "pkg/tests/protocols/bandwidth/test_remote.py" and "parents[4]" in f["hunk"]


LEFT_BODY = "".join(f"STEP_{i} = {i}  # unchanged by the move\n" for i in range(12))
LEFT_OLD = ("HERE = Path(__file__).resolve().parent\nsys.path.insert(0, str(Path(__file__).resolve().parents[1]))\n"
            + LEFT_BODY)
LEFT_NEW = ('HERE = (Path(__file__).resolve().parent.parent.parent / "bench" / "kv")\n'
            'sys.path.insert(0, str((Path(__file__).resolve().parents[2] / "bench")))\n' + LEFT_BODY)
LEFT_BY_HAND = ('HERE = Path(__file__).resolve().parent.parent.parent / "bench" / "kv"\n'
                'sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench"))\n' + LEFT_BODY)


def _left(tmp_path: Path, new: str) -> dict:
    """`bench/kv/run.py` moves to `pkg/kv/`, and the rest of `bench/kv/` stays."""
    repo, base = _repo(tmp_path, {"bench/kv/run.py": LEFT_OLD, "bench/kv/stay.py": "x = 1\n", "pkg/__init__.py": ""})
    _move(repo, "bench/kv/run.py", "pkg/kv/run.py")
    _write(repo, {"pkg/kv/run.py": new})
    return _check(repo, base)


def test_an_anchor_joined_back_to_the_directory_its_file_left_is_certified(tmp_path):
    """As `tools/move/layout.py` writes it: the anchor gains the path back, and parentheses around the two."""
    r = _left(tmp_path, LEFT_NEW)
    assert r["certified"], r["refusals"]


def test_an_anchor_joined_back_by_hand_with_no_parentheses_of_its_own_is_certified(tmp_path):
    """Where the anchor is a call's whole argument, the call's parentheses are around it and are not ones it gained."""
    r = _left(tmp_path, LEFT_BY_HAND)
    assert r["certified"], r["refusals"]


def test_an_anchor_joined_to_another_directory_than_its_file_left_is_refused(tmp_path):
    for i, (was, now) in enumerate((('"bench" / "kv"', '"bench" / "other"'), ("parents[2]", "parents[1]"),
                                    ('(Path(__file__).resolve().parent.parent.parent / "bench" / "kv")',
                                     '(Path(__file__).resolve().parent.parent.parent / "bench" / "kv").parent'))):
        [f] = _left(tmp_path / str(i), LEFT_NEW.replace(was, now))["refusals"]
        assert f["path"] == "pkg/kv/run.py" and now.split(" / ")[-1] in f["hunk"]


SH_BODY = "".join(f'echo "step {i} of the session, unchanged by the move"\n' for i in range(12))
SH_OLD = ('#!/usr/bin/env bash\nSELF=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")\n'
          'ROOT=$(cd "$(dirname "$SELF")/../../.." && pwd)\ncd "$(dirname "${BASH_SOURCE[0]}")/../../.."\n'
          'bash "$(dirname "$0")/../../../pkg/app/store/host.py"\n' + SH_BODY)


def _shell(tmp_path: Path, new: str) -> dict:
    """A script three directories down moves to two (the streamed wipe's `demo.sh` to `examples/wipe/`)."""
    repo, base = _repo(tmp_path, {**APP, "bench/pous/wipe/demo.sh": SH_OLD})
    for old, new_dir in APP_MOVES:
        _move(repo, old, new_dir)
    _move(repo, "bench/pous/wipe/demo.sh", "examples/wipe/demo.sh")
    _write(repo, {"examples/wipe/demo.sh": new})
    return _check(repo, base)


SH_NEW = ('#!/usr/bin/env bash\nSELF=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")\n'
          'ROOT=$(cd "$(dirname "$SELF")/../.." && pwd)\ncd "$(dirname "${BASH_SOURCE[0]}")/../.."\n'
          'bash "$(dirname "$0")/../../pkg/app/protocols/bandwidth/host.py"\n' + SH_BODY)


def test_a_shell_scripts_paths_from_its_own_directory_recounted_are_certified(tmp_path):
    r = _shell(tmp_path, SH_NEW)
    assert r["certified"], r["refusals"]


def test_a_shell_path_from_its_own_directory_that_names_another_one_is_refused(tmp_path):
    [f] = _shell(tmp_path, SH_NEW.replace('"$SELF")/../..', '"$SELF")/..'))["refusals"]
    assert f["path"] == "examples/wipe/demo.sh" and '"$SELF")/.." &&' in f["hunk"]


def test_a_shell_path_from_a_variable_that_is_not_the_scripts_own_path_is_refused(tmp_path):
    other = SH_OLD.replace('SELF=$(cd "$(dirname "$0")" && pwd)/$(basename "$0")', 'SELF=$NVCC')
    repo, base = _repo(tmp_path, {**APP, "bench/pous/wipe/demo.sh": other})
    _move(repo, "bench/pous/wipe/demo.sh", "examples/wipe/demo.sh")
    _write(repo, {"examples/wipe/demo.sh": other.replace('"$SELF")/../../..', '"$SELF")/../..')})
    [f] = _check(repo, base)["refusals"]
    assert f["path"] == "examples/wipe/demo.sh" and "SELF" in f["hunk"]


HERE_OLD = ('#!/usr/bin/env bash\nhere=$(cd "$(dirname "$0")" && pwd)\nroot=$(cd "$here/../../.." && pwd)\n'
            'bash "$here/../../../pkg/app/store/host.py" "$out/../../x"\n' + SH_BODY)
HERE_NEW = HERE_OLD.replace('"$here/../../.."', '"$here/../.."').replace(
    '"$here/../../../pkg/app/store/host.py"', '"$here/../../pkg/app/protocols/bandwidth/host.py"')


def _here(tmp_path: Path, old: str, new: str) -> dict:
    repo, base = _repo(tmp_path, {**APP, "bench/pous/wipe/demo.sh": old})
    for was, new_dir in APP_MOVES:
        _move(repo, was, new_dir)
    _move(repo, "bench/pous/wipe/demo.sh", "examples/wipe/demo.sh")
    _write(repo, {"examples/wipe/demo.sh": new})
    return _check(repo, base)


def test_a_path_after_a_variable_a_shell_script_sets_to_its_own_directory_is_recounted(tmp_path):
    """`here=$(cd "$(dirname "$0")" && pwd)` is the script's directory, so `$here/../..` is a path from it as
    `$(dirname "$0")/../..` is: recounted from the script's new place to where what it named went. Refused as before: a
    climb that names another directory, a path after a variable the script sets again, and one after any other variable."""
    r = _here(tmp_path / "taken", HERE_OLD, HERE_NEW)
    assert r["certified"], r["refusals"]
    assert r["rules"] == {"anchor": ["examples/wipe/demo.sh"]}
    again = 'here=$(cd "$(dirname "$0")" && pwd)\nhere=$1\n'
    for case, old, new in (
            ("elsewhere", HERE_OLD, HERE_NEW.replace('"$here/../.."', '"$here/.."')),
            ("set again", HERE_OLD.replace('here=$(cd "$(dirname "$0")" && pwd)\n', again),
             HERE_NEW.replace('here=$(cd "$(dirname "$0")" && pwd)\n', again)),
            ("another variable", HERE_OLD, HERE_NEW.replace('"$out/../../x"', '"$out/../x"'))):
        r = _here(tmp_path / case, old, new)
        assert [f["path"] for f in r["refusals"]] == ["examples/wipe/demo.sh"] and "anchor" not in r["rules"], case


SPOT = {"pkg/app/__init__.py": "", "pkg/app/spot/__init__.py": "", "pkg/app/spot/coins.py": "C = 1\n",
        "pkg/app/spot/service.py": "S = 1\n", "pkg/app/spot/second.py": "T = 1\n", "pkg/app/spot/one/draw.py": "D = 1\n"}
SPOT_USE = ("from app.spot import coins as CC, service as S  # noqa: E402\n" + SH_BODY
            + "from app.spot import (coins as C2, second as SS,\n                      service as S2)\n" + SH_BODY
            + "from app.spot import coins as C3, service as S3\nfrom app.spot.one import draw as D, \\\n    draw as D2\n")
SPOT_SPLIT = ("from app.device import coins as CC  # noqa: E402\nfrom app.verifier import service as S  # noqa: E402\n" + SH_BODY
              + "from app.device import coins as C2\nfrom app.verifier import (second as SS,\n                      service as S2)\n"
              + SH_BODY + "from app.device import coins as C3\nfrom app.verifier import service as S3\n"
              "from app.verifier.one import draw as D, \\\n    draw as D2\n")


def _spot(tmp_path: Path, use: str) -> dict:
    """`app.spot` goes two ways, `coins` to `app.device` and the rest to `app.verifier`, and `use.py` imports from both."""
    repo, base = _repo(tmp_path, {**SPOT, "use.py": SPOT_USE})
    _move(repo, "pkg/app/spot/coins.py", "pkg/app/device/coins.py")
    for name in ("__init__.py", "service.py", "second.py", "one/draw.py"):
        _move(repo, f"pkg/app/spot/{name}", f"pkg/app/verifier/{name}")
    _write(repo, {"use.py": use})
    return _check(repo, base)


def test_an_import_split_in_two_keeps_its_lines_comment_and_the_lines_it_went_on_over(tmp_path):
    """One `from` import whose names now come from two modules becomes two (`tools/move/layout.py`): each keeps the
    comment that ended the line, and where the import went on over more lines, in brackets or after a backslash, those
    lines stay as they were under the statement that has their names. Refused: a line that lost the comment, and a
    continued line whose module is not where its names went."""
    r = _spot(tmp_path / "split", SPOT_SPLIT)
    assert r["certified"] and r["rules"] == {"continued": ["use.py"]}, r["refusals"]
    for case, use in (("comment", SPOT_SPLIT.replace("coins as CC  # noqa: E402", "coins as CC")),
                      ("bracket", SPOT_SPLIT.replace("from app.verifier import (second", "from app.other import (second")),
                      ("backslash", SPOT_SPLIT.replace("from app.verifier.one import", "from app.other.one import"))):
        assert [f["path"] for f in _spot(tmp_path / case, use)["refusals"]] == ["use.py"], case


def test_relative_import_rewritten_absolute_is_certified(tmp_path):
    body = "".join(f"\n\ndef f{i}():\n    return put({i})\n" for i in range(8))
    files = {"pkg/app/options/__init__.py": "", "pkg/app/options/pearl.py": "from .pearl_host import put\n" + body,
             "pkg/app/options/pearl_host.py": "def put():\n    pass\n"}
    repo, base = _repo(tmp_path, {**APP, **files})
    _move(repo, "pkg/app/options/pearl.py", "pkg/app/protocols/pearl/__init__.py")
    _move(repo, "pkg/app/options/pearl_host.py", "pkg/app/protocols/pearl/host.py")
    _move(repo, "pkg/app/options/__init__.py", "pkg/app/protocols/__init__.py")
    _write(repo, {"pkg/app/protocols/pearl/__init__.py": "from app.protocols.pearl.host import put\n" + body})
    r = _check(repo, base)
    assert r["certified"], r["refusals"]
    _write(repo, {"pkg/app/protocols/pearl/__init__.py": "from app.protocols.pearl.other import put\n" + body})
    assert _check(repo, base)["refusals"]


def test_identical_files_git_pairs_across_directories_follow_their_directory():
    renames = [("a/__init__.py", "y/b/__init__.py"), ("b/__init__.py", "x/a/__init__.py"),
               ("a/m.py", "x/a/m.py"), ("b/n.py", "y/b/n.py")]
    base = ["a/__init__.py", "a/m.py", "b/__init__.py", "b/n.py"]
    added = {"x/a/__init__.py", "x/a/m.py", "y/b/__init__.py", "y/b/n.py"}
    assert MC.derive_map(renames, base, set(base), added) == {"a": "x/a", "b": "y/b"}


MOVES_DOC = {"about": "the moves", "moves": [{"paths": {"old/": "older/"}}]}


def _moves(tmp_path: Path, paths: dict[str, str]) -> dict:
    files = {MC.MOVES: json.dumps(MOVES_DOC, indent=2) + "\n"}
    return _app(tmp_path, files, {MC.MOVES: json.dumps({**MOVES_DOC, "moves": [*MOVES_DOC["moves"], {"paths": paths}]},
                                                       indent=2) + "\n"})


def test_moves_entry_the_range_agrees_with_is_certified(tmp_path):
    r = _moves(tmp_path, {"pkg/app/store/": "pkg/app/protocols/bandwidth/", "pkg/app/linear/api.py": "pkg/app/protocols/linear/api.py",
                          "pkg/app/store/added_on_a_branch.py": "pkg/app/protocols/bandwidth/added_on_a_branch.py"})
    assert r["certified"], r["refusals"]
    assert {"path": MC.MOVES, "kind": "moves"} in r["edits"]


def test_moves_entry_the_range_contradicts_is_refused(tmp_path):
    [f] = _moves(tmp_path, {"pkg/app/store/": "pkg/app/protocols/storage/"})["refusals"]
    assert f["path"] == MC.MOVES and "pkg/app/protocols/storage/" in f["reason"]


JOIN = ('import sys\nfrom pathlib import Path\n\nHERE = Path(__file__).resolve().parent\n'
        'sys.path.insert(0, str(HERE.parent / "rho"))\n')
CIRCUITS = "# Steps as Boolean circuits (rho/circuits.py): what the census counts\nx = 1\n"
RHO = {"bench/rho/rho/__init__.py": "", "bench/rho/rho/classes.py": "Y = 1\n",
       "bench/rho/rho/circuits.py": 'X = 1\nSOURCE = f"circuit.{X}: its gates over its FLOPs (rho/circuits.py units)"\n',
       "bench/rho/data/circuits.toml": CIRCUITS, "bench/nci/statement.py": JOIN, "bench/nci/notes.toml": CIRCUITS}


def _rho(tmp_path: Path, edits: dict[str, str], files: dict[str, str] | None = None) -> dict:
    """`bench/rho` renamed `bench/overhead` and its package `rho` with it, the directory's pair given as a range's
    `moves.json` entry gives it."""
    repo, base = _repo(tmp_path, {**RHO, **(files or {})})
    _move(repo, "bench/rho/rho", "bench/overhead/overhead")
    _move(repo, "bench/rho/data", "bench/overhead/data")
    _write(repo, edits)
    return _check(repo, base, {"bench/rho": "bench/overhead"})


def test_join_on_the_files_own_place_naming_a_moved_directory_is_certified(tmp_path):
    r = _rho(tmp_path, {"bench/nci/statement.py": JOIN.replace('"rho"', '"overhead"')})
    assert r["certified"], r["refusals"]
    assert {"path": "bench/nci/statement.py", "kind": "python"} in r["edits"]


def test_join_with_any_other_change_or_on_a_shadowed_name_is_refused(tmp_path):
    [f] = _rho(tmp_path / "a", {"bench/nci/statement.py": JOIN.replace('(0, str(HERE.parent / "rho"',
                                                                         '(1, str(HERE.parent / "overhead"')})["refusals"]
    assert f["path"] == "bench/nci/statement.py" and '+sys.path.insert(1, str(HERE.parent / "overhead"))' in f["hunk"]
    shadowed = JOIN + '\n\ndef f(HERE):\n    return HERE.parent / "rho"\n'
    [f] = _rho(tmp_path / "b", {"bench/nci/statement.py": shadowed.replace('"rho"', '"overhead"')},
               {"bench/nci/statement.py": shadowed})["refusals"]
    assert f["path"] == "bench/nci/statement.py" and '+sys.path.insert(0, str(HERE.parent / "overhead"))' in f["hunk"]


def test_mention_of_a_moved_file_inside_its_moved_package_is_certified(tmp_path):
    circuits = RHO["bench/rho/rho/circuits.py"].replace("(rho/", "(overhead/")
    r = _rho(tmp_path, {"bench/overhead/data/circuits.toml": CIRCUITS.replace("(rho/", "(overhead/"),
                        "bench/overhead/overhead/circuits.py": circuits})
    assert r["certified"], r["refusals"]
    assert {e["path"] for e in r["edits"]} == {"bench/overhead/data/circuits.toml", "bench/overhead/overhead/circuits.py"}


def test_mention_with_any_other_change_elsewhere_or_outside_the_package_is_refused(tmp_path):
    for i, (path, text) in enumerate([
            ("bench/overhead/data/circuits.toml", CIRCUITS.replace("(rho/", "(overhead/").replace("counts", "measures")),
            ("bench/overhead/data/circuits.toml", CIRCUITS.replace("(rho/circuits.py", "(overhead/classes.py")),
            ("bench/nci/notes.toml", CIRCUITS.replace("(rho/", "(overhead/"))]):
        [f] = _rho(tmp_path / str(i), {path: text})["refusals"]
        assert f["path"] == path and "+" + text.splitlines()[0] in f["hunk"]


VECTORS = ('"""The vectors.\n\n    vectors.py --out ../../../proto/certifier/tests/v.json\n"""\n'
           "# reads ../../../proto/certifier/ for the schema\nX = 1\n")
WHY = "instances on which L.f gives what proto/certifier/cert.py gives, byte for byte"


def _why(why: str) -> str:
    return _lock(guarantees={}, runs={"L.Difftest": {"why": why, "args": ["10"]}})


def _certifier(tmp_path: Path, edits: dict[str, str]) -> dict:
    """`proto/certifier` renamed `proto/network_certifier`, which a script three levels down and a lock's run name."""
    repo, base = _repo(tmp_path, {"proto/certifier/cert.py": "X = 1\n", "proto/certifier/tests/v.json": "{}\n",
                                  "bench/e2e/enc/vectors.py": VECTORS, "L/lean-audit.json": _why(WHY)})
    _move(repo, "proto/certifier", "proto/network_certifier")
    _write(repo, edits)
    return _check(repo, base)


def test_relative_path_in_a_comment_naming_a_moved_place_is_certified(tmp_path):
    r = _certifier(tmp_path, {"bench/e2e/enc/vectors.py": VECTORS.replace("/certifier/", "/network_certifier/")})
    assert r["certified"], r["refusals"]
    assert {"path": "bench/e2e/enc/vectors.py", "kind": "python"} in r["edits"]


def test_relative_path_with_any_other_change_or_resolving_elsewhere_is_refused(tmp_path):
    moved = VECTORS.replace("/certifier/", "/network_certifier/")
    for i, (text, line) in enumerate([
            (moved.replace("the schema", "its schema"), "+# reads ../../../proto/network_certifier/ for its schema"),
            (moved.replace("../../../", "../../"), "+    vectors.py --out ../../proto/network_certifier/tests/v.json")]):
        [f] = _certifier(tmp_path / str(i), {"bench/e2e/enc/vectors.py": text})["refusals"]
        assert f["path"] == "bench/e2e/enc/vectors.py" and line in f["hunk"]


def test_runs_why_naming_a_moved_path_from_the_root_is_certified(tmp_path):
    r = _certifier(tmp_path, {"L/lean-audit.json": _why(WHY.replace("/certifier/", "/network_certifier/"))})
    assert r["certified"], r["refusals"]


def test_runs_why_with_any_other_change_is_refused(tmp_path):
    why = WHY.replace("/certifier/", "/network_certifier/").replace("byte for byte", "bit for bit")
    [f] = _certifier(tmp_path, {"L/lean-audit.json": _why(why)})["refusals"]
    assert f["path"] == "L/lean-audit.json" and "changed beyond moved paths" in f["reason"] and "bit for bit" in f["hunk"]


FROZEN = "frozen (no-function): Sec: it is about an arbitrary run (Sec/Defs/A.lean:3)"


def _renamed(tmp_path: Path, reason: str) -> dict:
    """The directory `Sec/` renamed `sec/`, whose lock gives a reason that names a file of it and a conclusion `Sec`."""
    repo, base = _repo(tmp_path, {"Sec/Defs/A.lean": "def a := 1\n",
                                  "Sec/lean-audit.json": _lock(guarantees={}, shape={"frozen": {"T.Sec": FROZEN}})})
    _move(repo, "Sec", "sec")
    _write(repo, {"sec/lean-audit.json": _lock(guarantees={}, shape={"frozen": {"T.Sec": reason}})})
    return _check(repo, base)


def test_a_lock_field_that_leaves_a_moved_directorys_name_as_a_word_is_certified(tmp_path):
    """A top-level directory's name is also a word, in a lock's reasons as in a text file's lines: the move rewrites
    the path and leaves the conclusion's name."""
    moved = FROZEN.replace("Sec/Defs", "sec/Defs")
    r = _renamed(tmp_path / "left", moved)
    assert r["certified"], r["refusals"]
    assert _renamed(tmp_path / "all", moved.replace("Sec:", "sec:"))["certified"]
    [f] = _renamed(tmp_path / "more", moved.replace("arbitrary", "any"))["refusals"]
    assert f["path"] == "sec/lean-audit.json" and "`shape` changed beyond moved paths" in f["reason"]


TERMS = 'scope = "prose"\n\n[words]\n"draw" = "sample"\n\n[[words_in]]\nfiles = ["c.py"]\nwords = { "draw_one" = "pick_one" }\n'
TERMS_BASE = {
    "tools/move/terms_map.toml": TERMS + '\n[rewrite]\nexclude = ["tools/move/**"]\n',
    "a.py": '"""Each draw is fresh."""\nDRAW = b"draw"\n\n\ndef draw_one(n):\n    # a draw of n coins\n'
            '    return f"draw {n}" + str(draw_one(n - 1))\n',
    "b.py": "from a import draw_one\n\nX = draw_one(2)  # one draw\n",
    "c.py": "from a import draw_one\n",
    "notes.md": "A draw picks `draw_one()`; see [draw](draw.md).\n",
    "P/L.lean": "/-- A draw. -/\ndef drawOne : Nat := 1\n\ndef drawTwo : Nat := 2 -- the draw\n",
    "P/lean-audit.json": _lock(guarantees={}, reads={"L": {"digest": "dd", "definitions": {"drawOne": "ee"}}}),
    "x.json": '{"draw": "a draw"}\n',
}
MOVE_MSG = "move\n\nMove: tools/move/terms_map.toml"


def _terms(tmp_path: Path, base_extra: dict | None = None, message: str = MOVE_MSG, words: str | None = None,
           **head: str) -> dict:
    """The prose map's move: ``head`` edits BASE, committed with ``message``."""
    repo, base = _repo(tmp_path, {**TERMS_BASE, **(base_extra or {})})
    _write(repo, head)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return MC.certify(base, _git(repo, "rev-parse", "HEAD"), cwd=str(repo), words=words)


def _refused(r: dict) -> dict[str, str]:
    return {f["path"]: f["reason"] for f in r["refusals"]}


def test_a_prose_move_by_rename_is_certified(tmp_path):
    """What `tools/move/rename.py` writes with a `scope = "prose"` map: words in comments, docstrings and Markdown's text,
    with code, strings, backticks and data files as they were."""
    repo, base = _repo(tmp_path, TERMS_BASE)
    assert MC._rename().main(["--map", str(repo / "tools/move/terms_map.toml"), "--root", str(repo)]) == 0
    _git(repo, "commit", "-q", "-m", MOVE_MSG)
    assert (repo / "a.py").read_text() == TERMS_BASE["a.py"].replace("Each draw", "Each sample").replace("a draw", "a sample")
    assert (repo / "notes.md").read_text() == "A sample picks `draw_one()`; see [sample](draw.md).\n"
    r = MC.certify(base, _git(repo, "rev-parse", "HEAD"), cwd=str(repo))
    assert r["certified"], r["refusals"]
    assert r["words"] == "tools/move/terms_map.toml"
    assert {e["path"]: e["kind"] for e in r["edits"]} == dict.fromkeys(["a.py", "b.py", "notes.md", "P/L.lean"], "words")
    assert MC.summary(r).endswith("(0 renames, 4 edits by tools/move/terms_map.toml's words)")


def test_a_word_renamed_in_a_string_literal_or_a_data_file_is_refused(tmp_path):
    a = TERMS_BASE["a.py"].replace('b"draw"', 'b"sample"')
    r = _terms(tmp_path, **{"a.py": a, "x.json": '{"draw": "a sample"}\n'})
    assert set(_refused(r)) == {"a.py", "x.json"}
    assert all("string literal or a data file" in why for why in _refused(r).values())


def test_a_name_renamed_to_a_fresh_name_the_same_everywhere_is_certified(tmp_path):
    """`draw_one` becomes `sample_one` in each file that renames it, and no file spells `sample_one` at BASE; a Lean name no
    lock records may follow its word too."""
    r = _terms(tmp_path, **{"a.py": TERMS_BASE["a.py"].replace("draw_one", "sample_one"),
                            "b.py": TERMS_BASE["b.py"].replace("draw_one", "sample_one"),
                            "P/L.lean": TERMS_BASE["P/L.lean"].replace("drawTwo", "sampleTwo")})
    assert r["certified"], r["refusals"]


def test_a_name_renamed_onto_one_the_file_spells_is_refused(tmp_path):
    base = TERMS_BASE["b.py"] + "sample_one = 3\n"
    r = _terms(tmp_path, base_extra={"b.py": base}, **{"b.py": base.replace("draw_one", "sample_one")})
    assert _refused(r) == {"b.py": "renames draw_one to sample_one, which b.py spells at BASE: the two names would merge"}


def test_a_name_renamed_otherwise_in_another_file_is_refused(tmp_path):
    """`c.py`'s scoped word renames `draw_one` to `pick_one`, while `b.py` renames it to `sample_one`."""
    r = _terms(tmp_path, **{"b.py": TERMS_BASE["b.py"].replace("draw_one", "sample_one"),
                            "c.py": TERMS_BASE["c.py"].replace("draw_one", "pick_one")})
    assert set(_refused(r)) == {"b.py", "c.py"}
    assert all("to different names in different files" in why for why in _refused(r).values())


def test_a_lean_name_a_lock_records_is_refused(tmp_path):
    r = _terms(tmp_path, **{"P/L.lean": TERMS_BASE["P/L.lean"].replace("drawOne", "sampleOne")})
    [(path, why)] = _refused(r).items()
    assert path == "P/L.lean" and why.startswith("renames drawOne, which a lock's records name")


def test_an_edit_beyond_the_maps_words_is_refused(tmp_path):
    r = _terms(tmp_path, **{"notes.md": "A sample chooses `draw_one()`; see [sample](draw.md).\n"})
    [f] = r["refusals"]
    assert f["path"] == "notes.md" and f["reason"] == "edit beyond the words of tools/move/terms_map.toml"
    assert "+A sample chooses" in f["hunk"]


def test_a_file_a_json_pin_records_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(MC, "json_pin_keys", lambda: {"fix/vec.json": ("inputs", "")})
    r = _terms(tmp_path, base_extra={"fix/vec.json": '{"inputs": {"notes.md": "00"}}\n'},
               **{"notes.md": "A sample picks `draw_one()`; see [sample](draw.md).\n"})
    assert _refused(r)["notes.md"].startswith("the JSON pin fix/vec.json records this file's digest")


def test_a_prose_move_is_judged_by_the_map_its_move_line_or_words_names(tmp_path):
    """Without a `Move:` line the prose map is not read, so a prose edit is no pure move; `--words` names the map instead."""
    notes = {"notes.md": "A sample picks `draw_one()`; see [sample](draw.md).\n"}
    (tmp_path / "bare").mkdir()
    assert not _terms(tmp_path / "bare", message="move", **notes)["certified"]
    (tmp_path / "named").mkdir()
    r = _terms(tmp_path / "named", message="move", words="tools/move/terms_map.toml", **notes)
    assert r["certified"], r["refusals"]


LAYOUT_MAP = "tools/move/layout_map.toml"
LAYOUT_MSG = f"Move by {LAYOUT_MAP}\n\nLayout-move: {LAYOUT_MAP}\n"
#: `verity/ml/flock/` splits: `tool.py` to `verity/flock/prover/`, `pod/` whole to `verity/flock/pod/`
FLOCK = {
    LAYOUT_MAP: '[tokens]\n"verity/ml/flock/" = "verity/flock/prover/"\n"verity/ml/" = "verity/elsewhere/"\n',
    "verity/ml/flock/tool.py": "FLOCK_PURE = 1\n",
    "verity/ml/flock/pod/10-link.sh": "echo link\n",
    "verity/ml/other.py": "X = 1\n",
    "SKILL.md": ("files beside the code (`verity/ml/flock/`). Its patches are `verity/ml/flock/*.patch`,\n"
                 "and a test names `verity/ml/flock/src.rs`.\nThe pod scripts are in `verity/ml/flock/pod`.\n"),
    "tests/_release.py": 'FLOCK = ROOT / "verity" / "ml" / "flock"\n',
    "Security/Proofs/lakefile.toml": '[[require]]\nname = "vstar"\npath = "../../verity/ml/flock"\n',
    "Security/Sha.lean": "/-- (`verity/ml/flock`) step for step -/\ntheorem t : 1 = 1 := rfl\n",
}
FLOCK_MOVED = {
    "SKILL.md": FLOCK["SKILL.md"].replace("verity/ml/flock/pod", "verity/flock/pod").replace("verity/ml/flock", "verity/flock/prover"),
    "tests/_release.py": 'FLOCK = ROOT / "verity" / "flock" / "prover"\n',
    "Security/Proofs/lakefile.toml": FLOCK["Security/Proofs/lakefile.toml"].replace("verity/ml/flock", "verity/flock/prover"),
    "Security/Sha.lean": FLOCK["Security/Sha.lean"].replace("verity/ml/flock", "verity/flock/prover"),
}


def _flock(tmp_path: Path, message: str = LAYOUT_MSG, **over: str) -> dict:
    repo, base = _repo(tmp_path, FLOCK)
    _move(repo, "verity/ml/flock/tool.py", "verity/flock/prover/tool.py")
    _move(repo, "verity/ml/flock/pod", "verity/flock/pod")
    _write(repo, {**FLOCK_MOVED, **over})
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return MC.certify(base, _git(repo, "rev-parse", "HEAD"), cwd=str(repo))


def test_a_reference_to_a_split_directory_follows_the_maps_tokens_row(tmp_path):
    """The map the move's commit names says where a reference to a directory it splits goes (`[tokens]`), which no rename
    does: the directory itself, a pattern below it, a path below it that names nothing, a pathlib chain, a lakefile's
    relative path, a Lean comment. The report lists the paths the row took."""
    r = _flock(tmp_path)
    assert r["certified"], r["refusals"]
    assert "verity/ml/flock" not in r["map"]
    assert r["rules"] == {"tokens": ["SKILL.md", "Security/Proofs/lakefile.toml", "Security/Sha.lean", "tests/_release.py"]}


def test_a_path_below_a_split_directory_that_the_tree_has_follows_only_its_own_move(tmp_path):
    """`verity/ml/flock/pod` moved whole to `verity/flock/pod`: the directory's row doesn't send it below `verity/flock/prover/`.
    Neither does a row send a reference anywhere but where it says, and a row the range doesn't bear out (no file of
    `verity/ml/` is below `verity/elsewhere/`) is no row."""
    for case, line in (("below", "The pod scripts are in `verity/flock/prover/pod`."),
                       ("elsewhere", "The pod scripts are in `verity/flock/pod` (`verity/flock/pod/`)."),
                       ("not borne out", "The pod scripts are in `verity/flock/pod`, not in `verity/elsewhere/`.")):
        base = FLOCK["SKILL.md"] if case == "below" else FLOCK["SKILL.md"].replace("pod`.", "pod` (`verity/ml/`).") \
            if case == "not borne out" else FLOCK["SKILL.md"].replace("pod`.", "pod` (`verity/ml/flock/`).")
        (tmp_path / case).mkdir()
        repo, at = _repo(tmp_path / case, {**FLOCK, "SKILL.md": base})
        _move(repo, "verity/ml/flock/tool.py", "verity/flock/prover/tool.py")
        _move(repo, "verity/ml/flock/pod", "verity/flock/pod")
        _write(repo, {**FLOCK_MOVED, "SKILL.md": "\n".join([*FLOCK_MOVED["SKILL.md"].splitlines()[:2], line]) + "\n"})
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", LAYOUT_MSG)
        r = MC.certify(at, _git(repo, "rev-parse", "HEAD"), cwd=str(repo))
        assert [f["path"] for f in r["refusals"]] == ["SKILL.md"], case
        assert "SKILL.md" not in r["rules"]["tokens"], case


def test_an_anchor_that_names_a_split_directory_is_recounted_to_where_its_row_points(tmp_path):
    """A test of `verity/ml/flock/` names that directory by counting up from its own file. The directory goes two ways and
    its `[tokens]` row says a reference to it means the prover, so from the file's new place the anchor counts up to
    there. Refused: a count that names another directory, and the same recount where no row names the directory."""
    body = "".join(f"V{i} = {i}\n" for i in range(30)) + "sys.path.insert(0, str(Path(__file__).resolve().parents[{n}]))\n"
    for case, n, rows in (("row", 2, FLOCK[LAYOUT_MAP]), ("other", 3, FLOCK[LAYOUT_MAP]),
                          ("no row", 2, '[tokens]\n"verity/ml/" = "verity/elsewhere/"\n')):
        repo, base = _repo(tmp_path / case, {**FLOCK, LAYOUT_MAP: rows, "verity/ml/flock/tests/test_x.py": body.format(n=1)})
        for old, new in (("verity/ml/flock/tool.py", "verity/flock/prover/tool.py"), ("verity/ml/flock/pod", "verity/flock/pod"),
                         ("verity/ml/flock/tests/test_x.py", "verity/flock/prover/unit/tests/test_x.py")):
            _move(repo, old, new)
        _write(repo, {**(FLOCK_MOVED if case != "no row" else {}), "verity/flock/prover/unit/tests/test_x.py": body.format(n=n)})
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", LAYOUT_MSG)
        r = MC.certify(base, _git(repo, "rev-parse", "HEAD"), cwd=str(repo))
        if case == "row":
            assert r["certified"] and "verity/flock/prover/unit/tests/test_x.py" in r["rules"]["tokens"], r["refusals"]
        else:
            assert "verity/flock/prover/unit/tests/test_x.py" in [f["path"] for f in r["refusals"]], case


def test_a_cover_may_hold_an_entry_its_tokens_row_sends(tmp_path):
    """A list names the split `verity/ml/flock` beside its own `pod/`, where the mover writes it as its `[tokens]` row
    says, and `lib/a/`, which goes two ways with no row and is covered. The certificate takes the row's entry in the
    cover and lists the file under both rules. An entry that holds a file the old list never held (`verity`, with
    `verity/ml/other.py`) is still refused."""
    for case, flock in (("row", "verity/flock/prover"), ("wider", "verity")):
        repo, base = _repo(tmp_path / case, {**FLOCK, "verity/ml/flock/agree.py": "A = 1\n", "lib/a/x.py": "X = 1\n", "lib/a/y.py": "Y = 1\n",
                                             "gates.toml": 'keyed = ["verity/ml/flock", "verity/ml/flock/pod", "lib/a"]\n'})
        for old, new in (("verity/ml/flock/tool.py", "verity/flock/prover/tool.py"), ("verity/ml/flock/pod", "verity/flock/pod"),
                         ("verity/ml/flock/agree.py", "verity/flock/agreement/agree.py"),
                         ("lib/a/x.py", "p/x.py"), ("lib/a/y.py", "q/y.py")):
            _move(repo, old, new)
        _write(repo, {**FLOCK_MOVED, "gates.toml": f'keyed = ["{flock}", "verity/flock/pod", "p", "q"]\n'})
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", LAYOUT_MSG)
        r = MC.certify(base, _git(repo, "rev-parse", "HEAD"), cwd=str(repo))
        if case == "row":
            assert r["certified"], r["refusals"]
            assert r["rules"]["cover"] == ["gates.toml"] and "gates.toml" in r["rules"]["tokens"]
        else:
            assert [f["path"] for f in r["refusals"]] == ["gates.toml"]


def test_a_cover_whose_one_part_is_where_the_row_points_is_a_cover(tmp_path):
    """`verity/ml/flock` goes three ways and its `[tokens]` row names one of them. A list that names it alone is written
    as all three, and the row's part among them is a part like the others: a cover, under no row. With one of the three
    left out the list is neither what the row says nor a cover, and is refused."""
    for case, parts in (("all", '"verity/flock/agreement", "verity/flock/pod", "verity/flock/prover"'),
                        ("less", '"verity/flock/pod", "verity/flock/prover"')):
        repo, base = _repo(tmp_path / case, {**FLOCK, "verity/ml/flock/agree.py": "A = 1\n", "gates.toml": 'keyed = ["verity/ml/flock"]\n'})
        for old, new in (("verity/ml/flock/tool.py", "verity/flock/prover/tool.py"), ("verity/ml/flock/pod", "verity/flock/pod"),
                         ("verity/ml/flock/agree.py", "verity/flock/agreement/agree.py")):
            _move(repo, old, new)
        _write(repo, {**FLOCK_MOVED, "gates.toml": f"keyed = [{parts}]\n"})
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", LAYOUT_MSG)
        r = MC.certify(base, _git(repo, "rev-parse", "HEAD"), cwd=str(repo))
        if case == "all":
            assert r["certified"], r["refusals"]
            assert r["rules"]["cover"] == ["gates.toml"] and "gates.toml" not in r["rules"]["tokens"]
        else:
            assert [f["path"] for f in r["refusals"]] == ["gates.toml"]


def test_the_tokens_rows_are_read_only_from_the_map_a_commit_of_the_range_names(tmp_path):
    """Without the `Layout-move:` line no map says where the split directory's references go, as before."""
    r = _flock(tmp_path, message="move")
    assert {f["path"] for f in r["refusals"]} == set(FLOCK_MOVED) and r["rules"] == {}


ROOTED = ('export PYTHONPATH=<tree>:<tree>/protocols/pous/src\n'
          'T=${TESTS:-protocols/pous/tests/test_x.py}\n'
          'cp -r "$T/wt-$c/protocols/pous/live" .\n'
          'REPLAY=${REPLAY:-$ROOT/protocols/pous/replay.py}\n'
          "echo '{r}/protocols/pous/python' ~/verity/protocols/pous /workspace/protocols/pous\n")


def test_a_moved_path_after_what_the_mover_reads_as_the_root_is_certified(tmp_path):
    """`tools/move/layout.py` rewrites a moved path after each prefix it reads as the root (its `ROOT_PREFIX`: `<tree>/`,
    `$T/wt-$c/`, `{r}/`, `~/verity/`, `/workspace/`) and as a shell variable's default, and the check reads a line nothing
    else explains by the same pattern. A moved path inside a longer one, or after a word's `-`, is still not the move's."""
    for case, miss in (("taken", ""), ("inside", "OUT=$W/x/protocols/pous\n"), ("word", "OUT=wt-protocols/pous\n")):
        run = ROOTED + "echo done\n" + miss
        repo, base = _repo(tmp_path / case, {**BASE_FILES, "run.sh": run})
        _move(repo, "protocols/pous", "security_proofs/pous")
        _write(repo, {"run.sh": run.replace("protocols/pous", "security_proofs/pous")})
        r = _check(repo, base)
        if not miss:
            assert r["certified"], r["refusals"]
            assert r["rules"] == {"prefixes": ["run.sh"]}
        else:
            [f] = r["refusals"]
            assert f["path"] == "run.sh" and f["hunk"].splitlines()[1:] == ["-" + miss.strip(), "+" + miss.strip().replace(
                "protocols/pous", "security_proofs/pous")] and r["rules"] == {}, case


def test_a_moved_path_after_a_prefix_the_map_lists_for_the_root_is_certified(tmp_path):
    """A pod's unit names a file by where the pod installs the tree, which no pattern knows for the root. The move's map
    lists the prefix (`[rewrite] root_prefixes`), the mover rewrites the path after it, and the check reads it there.
    Refused as before: the same line when the map does not list the prefix, or lists it and no commit names the map, and
    a path after another directory."""
    unit = "ExecStart=/usr/bin/python3 -I /opt/pod/current/verity/ml/flock/tool.py run %i\n"
    listed = FLOCK[LAYOUT_MAP] + '\n[rewrite]\nroot_prefixes = ["/opt/pod/current/"]\n'
    for case, layout, message, line in (("taken", listed, LAYOUT_MSG, unit), ("unlisted", FLOCK[LAYOUT_MAP], LAYOUT_MSG, unit),
                                        ("unnamed", listed, "move", unit),
                                        ("elsewhere", listed, LAYOUT_MSG, unit.replace("/opt/pod/current/", "/opt/other/current/"))):
        repo, base = _repo(tmp_path / case, {**FLOCK, LAYOUT_MAP: layout, "pod/unit@.service": line})
        _move(repo, "verity/ml/flock/tool.py", "verity/flock/prover/tool.py")
        _move(repo, "verity/ml/flock/pod", "verity/flock/pod")
        _write(repo, {**FLOCK_MOVED, "pod/unit@.service": line.replace("verity/ml/flock/tool.py", "verity/flock/prover/tool.py")})
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", message)
        r = MC.certify(base, _git(repo, "rev-parse", "HEAD"), cwd=str(repo))
        if case == "taken":
            assert r["certified"], r["refusals"]
            assert r["rules"]["root_prefixes"] == ["pod/unit@.service"] and "prefixes" not in r["rules"]
        else:
            assert "pod/unit@.service" in {f["path"] for f in r["refusals"]} and "root_prefixes" not in r["rules"], case


def test_a_reference_to_a_split_directory_follows_the_part_the_maps_split_row_names_for_its_file(tmp_path):
    """`verity/ml/flock/` splits and its `[tokens]` row says the prover, but `run.sh` means the pod scripts by it. The
    map's `[split]` row for that file names the part (`to`), the mover writes it, and the check reads it there, for the
    directory and for a path below it that a variable ends. Refused: the same line where the row is another file's,
    where it names a directory no file of the split one went to, where no commit names the map, and where the line
    follows the `[tokens]` row the file's own row replaces."""
    line, part = 'exe "verity/ml/flock/$x" verity/ml/flock\n', "verity/flock/pod"
    row = '\n[split]\n"FILE: verity/ml/flock" = { to = "TO", why = "the pod scripts" }\n'
    for case, file, to, message, new in (("taken", "run.sh", part, LAYOUT_MSG, part), ("another", "other.sh", part, LAYOUT_MSG, part),
                                         ("nowhere", "run.sh", "verity/flock/gone", LAYOUT_MSG, "verity/flock/gone"),
                                         ("unnamed", "run.sh", part, "move", part),
                                         ("tokens", "run.sh", part, LAYOUT_MSG, "verity/flock/prover")):
        layout = FLOCK[LAYOUT_MAP] + row.replace("FILE", file).replace("TO", to)
        repo, base = _repo(tmp_path / case, {**FLOCK, LAYOUT_MAP: layout, "run.sh": line})
        _move(repo, "verity/ml/flock/tool.py", "verity/flock/prover/tool.py")
        _move(repo, "verity/ml/flock/pod", "verity/flock/pod")
        _write(repo, {**FLOCK_MOVED, "run.sh": line.replace("verity/ml/flock", new)})
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", message)
        r = MC.certify(base, _git(repo, "rev-parse", "HEAD"), cwd=str(repo))
        if case == "taken":
            assert r["certified"], r["refusals"]
            assert r["rules"]["split"] == ["run.sh"] and "run.sh" not in r["rules"]["tokens"]
        else:
            assert "run.sh" in {f["path"] for f in r["refusals"]} and "split" not in r["rules"], case


def test_a_rule_lists_only_the_paths_whose_lines_it_rewrote(tmp_path):
    """A line with a moved path after a root prefix, rewritten, and a reference to the split directory left as written
    is the prefix rule's alone."""
    line = 'cp "$T/wt-$c/verity/ml/flock/pod/10-link.sh" verity/ml/flock/\n'
    repo, base = _repo(tmp_path, {**FLOCK, "run.sh": line})
    _move(repo, "verity/ml/flock/tool.py", "verity/flock/prover/tool.py")
    _move(repo, "verity/ml/flock/pod", "verity/flock/pod")
    _write(repo, {**FLOCK_MOVED, "run.sh": line.replace("verity/ml/flock/pod", "verity/flock/pod")})
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", LAYOUT_MSG)
    r = MC.certify(base, _git(repo, "rev-parse", "HEAD"), cwd=str(repo))
    assert r["certified"], r["refusals"]
    assert r["rules"]["prefixes"] == ["run.sh"] and "run.sh" not in r["rules"]["tokens"]


#: the package `verity/ml/circuits/` moves to `verity/circuits/prover/`, but for `ops/` (to `verity/definitions/ops/`) and
#: `pod/` (to `verity/circuits/pod/`); `benchmarks/pouw/` stays, and `tools/config/` goes to `verity/configuration/config/`.
#: The script's variable is one `tests/test_pythonpath_entries.py` does not hold to this tree, which has no such paths
REST = "".join(f"line {i} of the file, which the move leaves as it is\n" for i in range(8))
REBASE = {
    LAYOUT_MAP: '[tokens]\n"verity/ml/circuits/" = "verity/circuits/"\n',
    "verity/ml/circuits/pyproject.toml": '[project]\nname = "circuits"\n',
    "verity/ml/circuits/README.md": "| [`ops`](ops/README.md) | the primitive circuits |\n" + REST,
    "verity/ml/circuits/population.py": '"""A tile\'s own check is `ops/gemm.circuit-check.py`, the unit\'s `tests/fixtures/unit.json`.\n'
                                        + REST + '"""\n',
    "verity/ml/circuits/ops/README.md": "# ops\n",
    "verity/ml/circuits/ops/gemm.circuit-check.py": "X = 1\n",
    "verity/ml/circuits/pod/90-unit.sh": "echo unit\n",
    "verity/ml/circuits/tests/fixtures/unit.json": "{}\n",
    "verity/ml/circuits/tests/lean/Unit.lean": "/-- Built by `pod/90-unit.sh` on `tests/fixtures/unit.json`.\n" + REST + "-/\ntheorem t : 1 = 1 := rfl\n",
    "benchmarks/pouw/pyproject.toml": '[project]\nname = "pouw"\n',
    "benchmarks/pouw/served/legs.sh": 'env PYTHONPATH="$POUW/../..:$POUW/../../tools/config:$POUW" python -m served\n' + REST.replace("line", "# line"),
    "tools/config/settings.py": "Y = 1\n",
    "tools/other.py": "Z = 1\n",
}
REBASED = {
    "verity/circuits/prover/README.md": REBASE["verity/ml/circuits/README.md"].replace("(ops/", "(../../definitions/ops/"),
    "verity/circuits/prover/population.py": REBASE["verity/ml/circuits/population.py"].replace("`ops/", "`../../definitions/ops/"),
    "verity/circuits/prover/tests/lean/Unit.lean": REBASE["verity/ml/circuits/tests/lean/Unit.lean"].replace("`pod/", "`../pod/"),
    "benchmarks/pouw/served/legs.sh": REBASE["benchmarks/pouw/served/legs.sh"].replace("tools/config", "verity/configuration/config"),
}


def _rebase(tmp_path: Path, extra: dict[str, str], **over: str) -> dict:
    repo, base = _repo(tmp_path, {**REBASE, **extra})
    for name in ("pyproject.toml", "README.md", "population.py", "tests"):
        _move(repo, f"verity/ml/circuits/{name}", f"verity/circuits/prover/{name}")
    _move(repo, "verity/ml/circuits/ops", "verity/definitions/ops")
    _move(repo, "verity/ml/circuits/pod", "verity/circuits/pod")
    _move(repo, "tools/config", "verity/configuration/config")
    _write(repo, {**REBASED, **over})
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", LAYOUT_MSG)
    return MC.certify(base, _git(repo, "rev-parse", "HEAD"), cwd=str(repo))


def test_a_relative_path_is_based_again_to_where_its_file_went(tmp_path):
    """A path a line names from its file's directory (a Markdown link, a docstring's), or from its package's root (the
    directory of the nearest `pyproject.toml` above it; after a variable too), may be written again from where that
    directory went to where the path went: `ops/README.md` from `verity/circuits/prover/` is `../../definitions/ops/README.md`,
    and `pod/90-unit.sh` from the package now at `verity/circuits/prover/` is `../pod/90-unit.sh`."""
    r = _rebase(tmp_path, {})
    assert r["certified"], r["refusals"]
    assert r["rules"] == {"relative": ["benchmarks/pouw/served/legs.sh", *list(REBASED)[:3]]}


def test_a_directory_written_with_its_slash_is_based_again_in_markdown(tmp_path):
    """Markdown's dotted paths are read from their file's directory before any rule, which drops a directory's slash:
    `ops/tests/` in backticks, based again to `../../definitions/ops/tests/`, reads there as the directory with none. Based
    to its parent it is refused."""
    old, new = "verity/ml/circuits/README.md", "verity/circuits/prover/README.md"
    for case, to in (("based", "`../../definitions/ops/tests/`"), ("parent", "`../../definitions/ops/`")):
        r = _rebase(tmp_path / case, {old: REBASE[old] + "Its tests are `ops/tests/`.\n", "verity/ml/circuits/ops/tests/test_x.py": ""},
                    **{new: REBASED[new] + f"Its tests are {to}.\n"})
        assert (r["certified"], [f["path"] for f in r["refusals"]]) == ((True, []) if case == "based" else (False, [new])), case


def test_a_path_from_the_packages_root_may_be_based_where_the_directorys_row_points(tmp_path):
    """The package's `pyproject.toml` went to `verity/circuits/prover/`, and its directory's `[tokens]` row says
    `verity/circuits/`: the mover writes a path from the package's root from there (`pod/90-unit.sh` is still that, and
    `tests/fixtures/unit.json` is `prover/tests/fixtures/unit.json`). The certificate takes the line with every such path
    based so, under `relative` and `tokens`, and refuses one that bases one path at each."""
    lean = "verity/circuits/prover/tests/lean/Unit.lean"
    row = REBASE["verity/ml/circuits/tests/lean/Unit.lean"].replace("`tests/", "`prover/tests/")
    r = _rebase(tmp_path / "row", {}, **{lean: row})
    assert r["certified"], r["refusals"]
    assert lean in r["rules"]["relative"] and r["rules"]["tokens"] == [lean]
    r = _rebase(tmp_path / "mixed", {}, **{lean: row.replace("`pod/", "`../pod/")})
    assert [f["path"] for f in r["refusals"]] == [lean] and "tokens" not in r["rules"]


def test_a_relative_path_based_anywhere_else_is_refused(tmp_path):
    """The based path is the one the move implies: one that names another place is refused; so is one written from a
    directory that is neither where the package's `pyproject.toml` went nor where its directory's `[tokens]` row points
    (`circuits/prover/tests/fixtures/unit.json`, from `verity/`); and a plain path the root also has is a path from the
    root."""
    readme, lean = "verity/circuits/prover/README.md", "verity/circuits/prover/tests/lean/Unit.lean"
    for case, extra, path, text in (
            ("elsewhere", {}, readme, REBASED[readme].replace("definitions/ops/", "definitions/")),
            ("a third directory", {}, lean, REBASED[lean].replace("`tests/", "`circuits/prover/tests/")),
            ("the root's", {"ops/README.md": "# the root's\n"}, readme, REBASED[readme])):
        (tmp_path / case).mkdir()
        r = _rebase(tmp_path / case, extra, **{path: text})
        assert [f["path"] for f in r["refusals"]] == [path], case
        assert path not in r["rules"]["relative"], case


def test_a_file_the_maps_drop_lists_may_be_deleted(tmp_path):
    """The map the move's commit names lists the files the move deletes (`[drop] paths`); any other deleted file, and the
    same one in a range whose commit names no map, is still refused."""
    drop = "benchmarks/pouw/contents/filler/receipts.toml"
    files = {**BASE_FILES, LAYOUT_MAP: f'[drop]\npaths = ["{drop}"]\n', drop: 'title = "receipts"\n',
             "benchmarks/pouw/contents/paths/clear.toml": 'title = "clear"\n'}
    for case, gone, message, refused in (("listed", [drop], LAYOUT_MSG, []),
                                         ("unlisted", [drop, "benchmarks/pouw/contents/paths/clear.toml"], LAYOUT_MSG,
                                          ["benchmarks/pouw/contents/paths/clear.toml"]),
                                         ("no map", [drop], "move", [drop])):
        repo, base = _repo(tmp_path / case, files)
        _git(repo, "rm", "-q", *gone)
        _git(repo, "commit", "-q", "-m", message)
        r = MC.certify(base, _git(repo, "rev-parse", "HEAD"), cwd=str(repo))
        assert [(f["path"], f["reason"]) for f in r["refusals"]] == [(p, "deleted") for p in refused], case
        assert r["rules"] == ({"drop": [drop]} if message == LAYOUT_MSG else {}), case
        assert [e for e in r["edits"] if e["kind"] == "dropped"] == [{"path": p, "kind": "dropped"} for p in r["rules"].get("drop", [])]


def test_a_runs_why_may_follow_a_moved_modules_dotted_name(tmp_path):
    """A run's `why` is prose that names the Python it holds the Lean to by module: the name follows the module's move
    (`scripts/difftest_vectors.py` goes to the proofs package with the run), and no other."""
    why = "instances on which lean.scripts.difftest_vectors (packages/verity/lean/scripts) gives what the Lean gives"
    moved = why.replace("lean.scripts", "core.scripts").replace("packages/verity/lean/scripts", "security_proofs/core/scripts")
    for case, now in (("moved", moved), ("elsewhere", moved.replace("core.scripts.difftest", "core.scripts.other"))):
        r = _runs_split(tmp_path / case, {"CoreProofs.Difftest": {**RUN, "why": now}}, was={**RUN, "why": why})
        if case == "moved":
            assert r["certified"], r["refusals"]
            assert r["rules"] == {"why": [f"{PROOFS}/lean-audit.json"]}
        else:
            [f] = r["refusals"]
            assert f["path"] == f"{PROOFS}/lean-audit.json" and "changed beyond moved paths" in f["reason"] and r["rules"] == {}


def test_a_runs_generator_named_from_the_root_follows_its_move(tmp_path):
    """A run names its generator from the root after `{repo}/`, which the mover reads as the root: the path follows the
    script's move, as on a line; one sent anywhere else is refused."""
    old = ["python3", "{repo}/packages/verity/lean/scripts/difftest_vectors.py", "{repo}", "--out", "{out}"]
    for case, to in (("moved", "security_proofs/core/scripts"), ("elsewhere", "security_proofs/scripts")):
        now = {**RUN, "generate": [old[0], old[1].replace("packages/verity/lean/scripts", to), *old[2:]]}
        r = _runs_split(tmp_path / case, {"CoreProofs.Difftest": now}, was={**RUN, "generate": old})
        if case == "moved":
            assert r["certified"], r["refusals"]
            assert r["rules"] == {"prefixes": [f"{PROOFS}/lean-audit.json"]}
        else:
            [f] = r["refusals"]
            assert f["path"] == f"{PROOFS}/lean-audit.json" and "changed beyond moved paths" in f["reason"] and r["rules"] == {}

