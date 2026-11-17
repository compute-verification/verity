"""drive.toml's `lowrisk_excludes` keeps the code the auditor's devices and the verifiers run off the low-risk lane, where the
verified queue's loop lands a change on its ready mark alone, with no reader.

Two ways the list goes wrong without failing anything: a `device/` or `verifier/` directory it doesn't name as a whole (a
new file there would land unread, even while the files there now are caught by a name glob or a subdirectory's entry), and
an entry a move left behind, which matches nothing and so excludes nothing. Both are read through the queue's own parse
(`DriveFlags.parse`) and matcher (`research.queue.lowrisk`), against `git ls-files`.

The land verdicts gate reads `lowrisk_excludes` but the entries `verdict_plain` lists: those may free for it no file the
whole check must see before it lands (`NEVER_PLAIN`), nor a test of one at `tests/X` (`suite_layout.stands_for`).

An entry `D/` holds `D`'s tests at `tests/D/` too, and `tests/` only the repository suite's own (`suite_layout.held`), which
the queue tells apart by path alone: the files of `tests/` itself and its `suite_layout.ROOT_OWN` directories. So each
directory of `tests/` is one of those or mirrors a directory of the tree.
"""
from __future__ import annotations

import functools
import re
import subprocess
import tomllib
from collections.abc import Iterable
from pathlib import Path

from research import suite_layout
from research.queue import DRIVE_FLAGS, DriveFlags, lowrisk

ROOT = Path(__file__).resolve().parents[1]
#: what `verdict_plain` may never make plain, beside every file under a `TRUSTED_DIRS` directory and every file of a trusted
#: place of tests/import_order.toml or of a place one reaches (the soundness path): Lean, its locks and Lake files, the
#: research tool and the move tool it certifies moves against, the check, the Lean check, and the code an auditor's device runs
#: outside a `device/` directory
NEVER_PLAIN = re.compile(r"(.*/)?(.*\.lean|lean-audit\.json|unread-caps\.json|lakefile\..*|lean-toolchain|lake-manifest\.json"
                         r"|TRUSTED\.sha256)|tools/research/.*|tools/move/.*|tools/verity/check/.*|tools/check/.*"
                         r"|tools/verity/lean/.*"
                         r"|benchmarks/one_stage/.*|benchmarks/network_traces/.*|benchmarks/poub_session/auditor_line\.py")
#: the directories whose every file, present or added later, is the auditor's devices' or a verifier's code
TRUSTED_DIRS = ("device", "verifier", "attester")
#: a file name no name glob of the list matches, standing for a file added later to a directory
NEW_FILE = "a-file-added-later"


@functools.cache
def tracked() -> tuple[str, ...]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True).stdout.decode()
    return tuple(sorted(p for p in out.split("\0") if p))


def excludes() -> tuple[str, ...]:
    return DriveFlags.parse((ROOT / DRIVE_FLAGS).read_text(), DRIVE_FLAGS).lowrisk_excludes


def trusted_dirs(files: Iterable[str]) -> list[str]:
    """Every directory of ``files`` named one of `TRUSTED_DIRS`, from the root, ending in `/`."""
    dirs = (f.split("/")[:-1] for f in files)
    return sorted({"/".join(parts[:i + 1]) + "/" for parts in dirs for i, part in enumerate(parts) if part in TRUSTED_DIRS})


def open_dirs(files: Iterable[str], entries: tuple[str, ...]) -> list[str]:
    """The `trusted_dirs` of ``files`` where a file added later would be low-risk under ``entries``."""
    return [d for d in trusted_dirs(files) if lowrisk([d + NEW_FILE], entries)]


def dead_entries(files: Iterable[str], entries: tuple[str, ...]) -> list[str]:
    """The ``entries`` under which no file of ``files`` is kept off the low-risk lane."""
    files = list(files)
    return [e for e in entries if lowrisk(files, [e])]


def test_every_device_and_verifier_directory_is_excluded():
    entries = excludes()
    assert trusted_dirs(tracked()), "no `device/` or `verifier/` directory is tracked: the test reads the wrong tree"
    assert not (bad := open_dirs(tracked(), entries)), (
        f"{DRIVE_FLAGS}'s `lowrisk_excludes` doesn't name these directories as a whole, so a file added to one would land on "
        f"its ready mark alone, with no reader: " + ", ".join(bad) + ". Fix: add each, as written here, to `lowrisk_excludes`.")


def test_no_excludes_entry_is_dead():
    assert not (bad := dead_entries(tracked(), excludes())), (
        f"these `lowrisk_excludes` entries of {DRIVE_FLAGS} match no tracked path, so they exclude nothing: "
        + ", ".join(bad) + ". Fix: point each at the path it was meant to cover (where a move put it), or delete it.")


def soundness_paths() -> dict[str, bool]:
    """{each path of a place of tests/import_order.toml: whether its place is trusted (marked `trusted`, or named under a
    `TRUSTED_DIRS` directory) or one a trusted place `reaches`: the soundness path}."""
    places = tomllib.loads((ROOT / "tests/import_order.toml").read_text())["place"]
    trusted = [p for p in places if p.get("trusted") or set(TRUSTED_DIRS) & set(p["paths"][0].split("/"))]
    names = {p["paths"][0] for p in trusted} | {r for p in trusted for r in p.get("reaches", ())}
    return {path: p["paths"][0] in names for p in places for path in p["paths"]}


def never_plain(files: Iterable[str], paths: dict[str, bool]) -> list[str]:
    """The ``files`` `verdict_plain` may never make plain: under a `TRUSTED_DIRS` directory, matching `NEVER_PLAIN`, or of
    the place whose path is the longest above it, when ``paths`` (`soundness_paths`) marks it; or the tests of one such
    (`tests/X` for `X`: `suite_layout.stands_for`)."""
    def sound(f: str) -> bool:
        above = [p for p in paths if f == p or f.startswith(p + "/")]
        return bool(above) and paths[max(above, key=len)]

    def kept(f: str) -> bool:
        return bool(set(TRUSTED_DIRS) & set(f.split("/")[:-1]) or NEVER_PLAIN.fullmatch(f) or sound(f))
    return [f for f in files if kept(f) or ((src := suite_layout.stands_for(f)) is not None and kept(src))]


def unmirrored(files: Iterable[str]) -> list[str]:
    """The directories directly under `tests/` among ``files`` that are neither the repository suite's own
    (`suite_layout.ROOT_OWN`) nor a directory of the tree they mirror, and the `ROOT_OWN` entries that name one."""
    files = list(files)
    tops = {f.split("/", 1)[0] for f in files if "/" in f}
    under = {r.split("/", 1)[0] for f in files if (r := suite_layout.mirror_of(f)) is not None and "/" in r}
    return sorted({d for d in under if (d in suite_layout.ROOT_OWN) == (d in tops)} | (set(suite_layout.ROOT_OWN) & tops))


def test_each_directory_of_tests_is_the_repository_suites_own_or_mirrors_one_of_the_tree():
    assert not (bad := unmirrored(tracked())), (
        "the queue reads which files of `tests/` are the repository suite's own by path alone (`suite_layout.root_owns`), and "
        f"these directories of `tests/` are ambiguous: {', '.join(f'tests/{d}/' for d in bad)}. Fix: list a directory of the "
        "repository suite's own in `ROOT_OWN` (tools/research/src/research/suite_layout.py), and take out of it any that names "
        "a directory of the tree; a directory of tests mirroring `X` goes at `tests/X`.")


def test_every_verdict_plain_entry_frees_no_file_the_whole_check_must_see_and_matches_a_tracked_path():
    """An entry of `verdict_plain` names a tracked path, and together they make plain no file under a `verifier/` or
    `device/` directory, of the soundness path, of Lean, a lock or a Lake file, of the research tool, the move tool, the
    check or the Lean check, or of the code an auditor's device runs: such a file keeps the whole check before it lands,
    whatever its land verdicts."""
    flags = DriveFlags.parse((ROOT / DRIVE_FLAGS).read_text(), DRIVE_FLAGS)
    assert flags.verdict_plain and not (bad := dead_entries(tracked(), flags.verdict_plain)), (
        f"these `verdict_plain` entries of {DRIVE_FLAGS} match no tracked path: {', '.join(bad)}. Fix: delete each, here and "
        f"from `lowrisk_excludes`, or point both at the path it was meant to cover")
    gate = [x for x in flags.lowrisk_excludes if x not in flags.verdict_plain]
    freed = [f for f in tracked() if not lowrisk([f], flags.lowrisk_excludes) and lowrisk([f], gate)]
    paths = soundness_paths()
    assert freed and any(paths.values()) and not (bad := never_plain(freed, paths)), (
        f"{DRIVE_FLAGS}'s `verdict_plain` lets these land on land verdicts and the quick tier, before the whole check: "
        + ", ".join(bad[:20]) + ". Fix: take out of `verdict_plain` each entry that frees them.")


def test_each_check_catches_its_failure():
    files = ["a/device/run.py", "a/device/sub/x.py", "b/verifier/registration.py", "b/verifier/lean/X.lean", "c/devices/y.py"]
    assert trusted_dirs(files) == ["a/device/", "b/verifier/"]
    assert open_dirs(files, ("a/device/", "registration.py", "*.lean")) == ["b/verifier/"]
    assert open_dirs(files, ("a/", "b/verifier/")) == []
    assert dead_entries(files, ("a/device/", "gone/", "a/device/run.py", "*.lean", "*.rs")) == ["gone/", "*.rs"]
    paths = {"v/circuits": True, "v/circuits/ops": False, "v/channel.py": True, "v/defs": False}
    assert never_plain(["a/device/x.py", "b/verifier/__init__.py", "x/Y.lean", "p/lean-audit.json", "p/lakefile.toml",
                        "tools/research/x.py", "tools/verity/check/merge_map.py", "tools/verity/lean/check.py",
                        "v/circuits/program.py", "v/channel.py", "tools/move/restack.py", "benchmarks/one_stage/a5.py",
                        "benchmarks/network_traces/x.py", "benchmarks/poub_session/auditor_line.py",
                        "v/circuits/ops/add.py", "v/defs/__init__.py", "benchmarks/poub_session/x.py", "pyproject.toml"],
                       paths) == [
        "a/device/x.py", "b/verifier/__init__.py", "x/Y.lean", "p/lean-audit.json", "p/lakefile.toml", "tools/research/x.py",
        "tools/verity/check/merge_map.py", "tools/verity/lean/check.py", "v/circuits/program.py", "v/channel.py",
        "tools/move/restack.py", "benchmarks/one_stage/a5.py", "benchmarks/network_traces/x.py",
        "benchmarks/poub_session/auditor_line.py"]
    assert never_plain(["tests/b/verifier/t.py", "tests/tools/research/test_x.py", "tests/v/channel.py",
                        "tests/v/defs/test_d.py", "tests/test_repo.py", "tests/lint/test_y.py", "tests/v/circuits/ops/test_a.py"],
                       paths) == ["tests/b/verifier/t.py", "tests/tools/research/test_x.py", "tests/v/channel.py"]
    assert unmirrored(["v/a.py", "tests/test_r.py", "tests/lint/test_l.py", "tests/v/test_a.py"]) == []
    assert unmirrored(["v/a.py", "tests/fixtures/x.json", "tests/v/test_a.py"]) == ["fixtures"]
    assert unmirrored(["v/a.py", "lint/b.py", "tests/lint/test_l.py"]) == ["lint"]
    assert unmirrored(["lint/b.py"]) == ["lint"]
