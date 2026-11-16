"""A worker's fetched packages (`runners/lean_module.packages`): made from the check's warm copy (`lean_audit.WarmDeps`) and
topped up from each copy a Lean check gives back, which holds what that check's build compiled alone, never losing a module
a task may be reading."""
from __future__ import annotations

import json
from pathlib import Path

from verity_graph.runners import lean_module as lm

la = lm.la
LIB = Path("dep") / ".lake" / "build" / "lib" / "lean"


def package(root: Path, name: str) -> Path:
    """A package of the tree that fetches `dep`, sharing its manifest and toolchain with every other one here."""
    pkg = root / name
    pkg.mkdir(parents=True)
    (pkg / "lean-toolchain").write_text("leanprover/lean4:v4.0.0\n")
    (pkg / "lake-manifest.json").write_text(json.dumps({"packagesDir": ".lake/packages", "packages": [
        {"type": "git", "name": "dep", "rev": "0" * 40, "url": "https://example.invalid/dep"}]}))
    return pkg


def build(pkg: Path, *mods: str) -> None:
    """A Lean check's build of ``pkg`` compiling ``mods`` of `dep` in its `.lake/packages`."""
    (lib := pkg / ".lake" / "packages" / LIB).mkdir(parents=True, exist_ok=True)
    for m in mods:
        (lib / f"{m}.olean").write_text(m)


def built(home: Path) -> set[str]:
    return {o.stem for o in (home / LIB).glob("*.olean")}


def quiet(s: str) -> None:
    pass


def test_the_fetched_packages_take_each_module_a_given_back_copy_adds_and_keep_the_ones_it_drops(tmp_path, monkeypatch):
    monkeypatch.setenv("CHECK_CACHE", str(tmp_path / "cache"))
    store, work = tmp_path / "cache" / "lean-deps", tmp_path / "work"
    a, b = package(tmp_path, "a"), package(tmp_path, "b")
    first = la.WarmDeps(store)
    first.take(a, log=quiet)
    build(a, "A")
    first.give_back({a})
    home = lm.packages(a, work)
    assert built(home) == {"A"} and lm.packages(b, work) == home, "one copy for each manifest and toolchain"
    inode = (home / LIB / "A.olean").stat().st_ino

    ours, theirs = la.WarmDeps(store), la.WarmDeps(store)
    ours.take(a, log=quiet)
    theirs.take(b, log=quiet)            # `a`'s check holds the warm copy, so `b`'s builds cold
    assert lm.packages(a, work) == home and built(home) == {"A"}, "a held copy leaves the fetched packages as they are"
    build(a, "B")
    build(b, "C")
    ours.give_back({a})
    assert built(lm.packages(a, work)) == {"A", "B"}
    theirs.give_back({b})                # a cold build's tree replaces the fuller one
    assert built(lm.packages(b, work)) == {"A", "B", "C"}, "what the warm copy dropped stays: a task may be reading it"
    assert (home / LIB / "A.olean").stat().st_ino == inode, "nothing the fetched packages hold is replaced"

    monkeypatch.setattr(lm, "_add_missing", None)
    assert lm.packages(a, work) == home, "an unchanged warm copy isn't walked again"


def test_with_no_warm_copy_the_fetched_packages_are_made_as_check_py_makes_them_the_workspace_loaded_in_a_sandbox(
        tmp_path, monkeypatch):
    """The worker ran `setup.sh PKG` itself, with the caller's whole environment and no sandbox, so a candidate's
    lakefiles and Mathlib's `cache get` ran outside every sandbox with the network. It now runs `check.py`'s `setup`:
    the first half as itself, then the fetch in the fetch sandbox with `fetch_env`, nothing else."""
    monkeypatch.setenv("CHECK_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("VERITY_CALLER_ONLY", "kept from the fetch")
    a, c, ran = package(tmp_path, "a"), la.lean_check(), []

    def run(argv, cwd, log=None, timeout=0, on_line=None, env=None):
        ran.append((argv, env))
        if "--fetch" in argv:
            (a / ".lake" / "packages" / "dep").mkdir(parents=True)
        return 0, ""

    monkeypatch.setattr(c, "run", run)
    monkeypatch.setattr(c, "sandboxed", lambda: True)
    monkeypatch.setattr(lm, "call", lambda argv, *rest, **kw: ran.append((argv, None)) or 0)
    (work := tmp_path / "work").mkdir()
    home = lm.packages(a, work)
    assert (home / "dep").is_dir()
    setup = [(argv[argv.index(str(c.TOOL / "setup.sh")):], argv, env) for argv, env in ran
             if str(c.TOOL / "setup.sh") in argv]
    assert [s for s, _, _ in setup] == [[str(c.TOOL / "setup.sh"), "--outside", str(a)],
                                        [str(c.TOOL / "setup.sh"), "--fetch", str(a)]], ran
    (_, outside, _), (_, fetch, env) = setup
    assert outside[0] == "bash" and fetch[:len(c.UNSHARE_NET)] == c.UNSHARE_NET and "--net" in fetch, fetch
    assert env == c.fetch_env() and "VERITY_CALLER_ONLY" not in env


def test_fetched_packages_whose_record_holds_the_key_alone_take_what_they_lack_from_the_warm_copy(tmp_path, monkeypatch):
    """As a worker's are where this runner made them before it topped them up: from a warm copy that lacked modules."""
    monkeypatch.setenv("CHECK_CACHE", str(tmp_path / "cache"))
    a = package(tmp_path, "a")
    check = la.WarmDeps(tmp_path / "cache" / "lean-deps")
    check.take(a, log=quiet)
    build(a, "A", "B")
    check.give_back({a})
    home = lm.fetched_dir(k := la.deps_key(a))
    (home / LIB).mkdir(parents=True)
    (home / LIB / "A.olean").write_text("A")
    (home / lm.READY).write_text(k + "\n")
    assert lm.packages(a, tmp_path / "work") == home and built(home) == {"A", "B"}
    assert json.loads((home / lm.READY).read_text())["warm"] is not None
