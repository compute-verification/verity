"""The sibling check (`siblings.py`) that `verity lean try --siblings` runs once FILE elaborates: which modules its stub
imports, and, with a real toolchain, that a module redeclaring a name another built module of its package declares fails
the try naming both, as the package's root fails a records run an hour in. Its stub loads no build older than a source it
was built from, in the package or in one it requires by path, and none without a source: with a real toolchain, a try
passes over the products a tree keeps from before names moved between modules (INF-667), and still fails a clash that
is in the sources. And the package's Lean options every try takes from it: with a real toolchain, an auto-bound name
fails a try where the package sets `autoImplicit = false`."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lean_siblings", HERE.parent / "siblings.py")
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)


def _cli():
    spec = importlib.util.spec_from_file_location("verity_cli_for_siblings", HERE.parents[1] / "cli" / "verity_cli" / "__init__.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def write(pkg: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        (pkg / rel).parent.mkdir(parents=True, exist_ok=True)
        (pkg / rel).write_text(text)


def built(pkg: Path, *mods: str, at: float = 2_000_000_000.0, trace: bool = True) -> None:
    """Give each of ``mods`` an `.olean` written at ``at``, newer than every source written before it, and, as Lake leaves
    a module it built, its trace (``trace``)."""
    for m in mods:
        f = (pkg / S.LIB).joinpath(*m.split(".")).with_suffix(".olean")
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"olean")
        os.utime(f, (at, at))
        if trace:
            f.with_suffix(".trace").write_text("{}")


def written(at: float, *files: Path) -> None:
    """Set when each of ``files`` was last written, as a sync that rewrote it then leaves it."""
    for f in files:
        os.utime(f, (at, at))


#: a stand-in workspace: `proofs` requires `commit` by path, which requires `ident`, as `proofs`' manifest lists them
T0 = 2_000_000_000.0
LIB = '\n[[lean_lib]]\nname = "{}"\n'
MANIFEST = json.dumps({"packages": [{"name": "commit", "type": "path", "dir": "../commit", "inherited": False},
                                    {"name": "mathlib", "type": "git", "inherited": True},
                                    {"name": "ident", "type": "path", "dir": "../commit/../ident", "inherited": True}]})


def test_the_stub_imports_the_built_modules_that_import_file_neither_directly_nor_through_others(tmp_path):
    """`P.B` imports FILE, `P.Top` imports `P.B`: neither is a sibling. `P.C` isn't built and `P.E`'s import `P.C` changed
    after `P.E` was built, so both are unchecked. A nested package's modules and `.lake` aren't the package's."""
    pkg = tmp_path / "p"
    write(pkg, {"lakefile.toml": 'name = "p"\n\n[[lean_lib]]\nname = "P"\n', "P/D.lean": "module\n",
                "P/A.lean": "module\n", "P/B.lean": "module\npublic import P.D\n", "P/Top.lean": "module\nimport P.B\n",
                "P/C.lean": "module\n", "P/E.lean": "module\nimport P.C\nimport Mathlib\n",
                "P/Nested/lakefile.toml": 'name = "n"\n', "P/Nested/X.lean": "module\n", "P/.lake/Y.lean": ""})
    built(pkg, "P.A", "P.B", "P.Top", "P.E")
    os.utime(pkg / "P" / "C.lean", (2_000_000_100.0,) * 2)
    p = S.plan(pkg, "P/D.lean")
    assert p == S.Plan("P.D", True, ["P.A"], ["P.C", "P.E"])
    work = tmp_path / "work"
    (work / "P").mkdir(parents=True)
    (work / "P" / "D.olean").write_bytes(b"")
    (work / "P" / "D.olean.server").write_bytes(b"")
    argv = S.stub(p, work)
    assert (work / f"{S.STUB}.lean").read_text() == "module\n\npublic import P.D\npublic import P.A\n"
    assert '"P.D": [["' + str(work / "P" / "D.olean") + '", "' + str(work / "P" / "D.olean.server") + '"]]' in (
        work / f"{S.STUB}.json").read_text()
    assert argv == ["--setup", str(work / f"{S.STUB}.json"), str(work / f"{S.STUB}.lean")]


def test_the_stub_leaves_out_the_other_executables_roots(tmp_path):
    """`P.Main` and `P.Tool` are executables' roots under the library's directory: each declares `main`, so a stub
    importing both would refuse it whatever FILE is. Neither is a sibling of a library module, and an executable's root
    as FILE has the library's modules for siblings and not the other root."""
    pkg = tmp_path / "p"
    write(pkg, {"lakefile.toml": 'name = "p"\n\n[[lean_lib]]\nname = "P"\n\n[[lean_exe]]\nname = "p-main"\nroot = "P.Main"\n'
                                 '\n[[lean_exe]]\nname = "P.Tool"\n',
                "P/A.lean": "module\n", "P/D.lean": "module\n", "P/Main.lean": "module\nimport P.A\n",
                "P/Tool.lean": "module\n"})
    built(pkg, "P.A", "P.Main", "P.Tool")
    assert S.executables(pkg) == {"P.Main", "P.Tool"}
    assert S.plan(pkg, "P/D.lean") == S.Plan("P.D", True, ["P.A"], [])
    assert S.plan(pkg, "P/Main.lean") == S.Plan("P.Main", True, ["P.A"], ["P.D"])
    assert S.plan(pkg, "P/Tool.lean") == S.Plan("P.Tool", True, ["P.A"], ["P.D"])


def test_modules_sit_under_src_dir_and_a_plain_sibling_makes_the_stub_plain(tmp_path):
    """Security/Proofs' layout: `srcDir = ".."`, so its library `Proofs` is the package's own directory, beside the outer
    package's `Defs`. A sibling that isn't a `module` can't be imported by one, so the stub is a plain file."""
    outer = tmp_path / "Sec"
    pkg = outer / "Proofs"
    write(outer, {"lakefile.toml": 'name = "sec"\n\n[[lean_lib]]\nname = "Defs"\n', "Defs/X.lean": "module\n",
                  "Proofs/lakefile.toml": 'name = "proofs"\nsrcDir = ".."\n\n[[lean_lib]]\nname = "Proofs"\n',
                  "Proofs/Core.lean": "module\n", "Proofs/Core/Old.lean": "theorem t : True := trivial\n"})
    built(pkg, "Proofs.Core.Old")
    p = S.plan(pkg, "Core.lean")
    assert p == S.Plan("Proofs.Core", False, ["Proofs.Core.Old"], [])
    work = (pkg / S.COPIES / "w").resolve()
    assert S.lean_args(pkg, p, work) == ["-R", ".lake/axiom-probe/w", "-o", str(work / "Proofs" / "Core.olean")]
    assert S.source(p, work) == work / "Proofs" / "Core.lean", "the probed copy, which -R names Proofs.Core"
    assert S.plan(outer, "Defs/X.lean") == S.Plan("Defs.X", True, [], [])
    assert "not a module of its package" in S.plan(outer, "Proofs/Core.lean")


def moved(ws: Path) -> tuple[Path, Path, Path]:
    """INF-667's tree, (`ident`, `commit`, `proofs`): everything was built at `T0 + 100`, and at `T0 + 200` a sync brought a
    move of `Identity.Domains`' names into the new `Identity.Domains.Pouw`. It rewrote `Identity/Domains.lean` and what
    imports the new module from then on, `PouwCommit/Bindings.lean` and `Proofs/New.lean`, and wrote
    `Identity/Domains/Pouw.lean` and FILE. Since then Lake built only what a target imported, `Identity.Domains.Pouw`
    and `Proofs.New`: the products of `Identity.Domains`, of `PouwCommit.Bindings`, of `Proofs.Old`, which imports that,
    and of `Proofs.Far`, which imports `Proofs.Old`, are from before the move, and the last two's sources never
    changed."""
    ident, commit, proofs = ws / "ident", ws / "commit", ws / "proofs"
    write(ident, {"lakefile.toml": 'name = "ident"\n' + LIB.format("Identity"), "Identity/Domains.lean": "module\n",
                  "Identity/Domains/Pouw.lean": "module\n"})
    write(commit, {"lakefile.toml": 'name = "commit"\n' + LIB.format("PouwCommit"),
                   "PouwCommit/Bindings.lean": "module\npublic import Identity.Domains.Pouw\n"})
    write(proofs, {"lakefile.toml": 'name = "proofs"\n' + LIB.format("Proofs"), "lake-manifest.json": MANIFEST,
                   "Proofs/File.lean": "module\npublic import Identity.Domains.Pouw\n",
                   "Proofs/Old.lean": "module\npublic import PouwCommit.Bindings\n",
                   "Proofs/Far.lean": "module\nimport Proofs.Old\n",
                   "Proofs/New.lean": "module\npublic import Identity.Domains.Pouw\n",
                   "Proofs/Alone.lean": "module\nimport Mathlib\n"})
    written(T0, *(f for q in (ident, commit, proofs) for f in q.rglob("*.lean")))
    built(ident, "Identity.Domains", at=T0 + 100)
    built(commit, "PouwCommit.Bindings", at=T0 + 100)
    built(proofs, "Proofs.Old", "Proofs.Far", "Proofs.Alone", at=T0 + 100)
    written(T0 + 200, ident / "Identity" / "Domains.lean", ident / "Identity" / "Domains" / "Pouw.lean",
            commit / "PouwCommit" / "Bindings.lean", proofs / "Proofs" / "New.lean", proofs / "Proofs" / "File.lean")
    built(ident, "Identity.Domains.Pouw", at=T0 + 300)
    built(proofs, "Proofs.New", at=T0 + 300)
    return ident, commit, proofs


def test_a_sibling_built_before_a_source_it_imports_through_another_package_changed_is_unchecked(tmp_path):
    """INF-667 (`moved`). `Proofs.Old`'s build is newer than every source of its own package, and it loads
    `PouwCommit.Bindings` and `Identity.Domains` as they were built before the move, the second declaring what
    `Identity.Domains.Pouw` declares now: a stub that imports it beside FILE fails on a clash that is in no source. Its
    import `PouwCommit.Bindings` has a source newer than its build, in a package `proofs` requires through another, so
    it is unchecked, and so is `Proofs.Far`, which imports it. Each is a sibling again once Lake has rebuilt it."""
    ident, commit, proofs = moved(tmp_path)
    assert S.workspace(proofs) == [proofs, commit, ident] and S.workspace(ident) == [ident], "its manifest's path packages"
    assert S.plan(proofs, "Proofs/File.lean") == S.Plan("Proofs.File", True, ["Proofs.Alone", "Proofs.New"],
                                                        ["Proofs.Far", "Proofs.Old"])
    built(commit, "PouwCommit.Bindings", at=T0 + 400)
    built(proofs, "Proofs.Old", at=T0 + 400)
    assert S.plan(proofs, "Proofs/File.lean") == S.Plan("Proofs.File", True, ["Proofs.Alone", "Proofs.New", "Proofs.Old"],
                                                        ["Proofs.Far"]), "Far's own build is still from before the move"
    built(proofs, "Proofs.Far", at=T0 + 400)
    assert S.plan(proofs, "Proofs/File.lean").unchecked == []
    written(T0 + 500, ident / "Identity" / "Domains" / "Pouw.lean")
    assert S.plan(proofs, "Proofs/File.lean") == S.Plan("Proofs.File", True, ["Proofs.Alone"],
                                                        ["Proofs.Far", "Proofs.New", "Proofs.Old"]), \
        "a source two packages down, newer than every build that loads its product"


def test_no_stub_imports_a_product_with_no_source_or_a_module_whose_build_loads_one(tmp_path):
    """A sync unlinks a source its branch doesn't have and leaves the product, for the branch that has it. `Proofs.Dead`
    and `Identity.Gone` are such products: neither is a module of a package, so neither is a sibling, and nothing that
    loads one is. `Proofs.Uses` still imports `Identity.Gone` and its build is newer than its source, and `Proofs.Via`
    imports `Proofs.Uses`: both unchecked. So is `Proofs.Shadowed`, whose name `ident` holds a product of with no source,
    which Lean could load in its place. A module without its trace is one Lake rebuilds (`lean_changed.clashed` drops
    the trace of a module whose build it can't trust): `Proofs.Forgot`, and `Proofs.OnForgot`, which imports it. With its
    source in place a module is one like any other."""
    ident, proofs = tmp_path / "ident", tmp_path / "proofs"
    write(ident, {"lakefile.toml": 'name = "ident"\n' + LIB.format("Identity"), "Identity/Kept.lean": "module\n"})
    write(proofs, {"lakefile.toml": 'name = "proofs"\n' + LIB.format("Proofs"),
                   "lake-manifest.json": json.dumps({"packages": [{"name": "ident", "type": "path", "dir": "../ident"}]}),
                   "Proofs/File.lean": "module\n", "Proofs/Fine.lean": "module\nimport Identity.Kept\n",
                   "Proofs/Uses.lean": "module\nimport Identity.Gone\n", "Proofs/Via.lean": "module\nimport Proofs.Uses\n",
                   "Proofs/Forgot.lean": "module\n", "Proofs/OnForgot.lean": "module\nimport Proofs.Forgot\n",
                   "Proofs/Shadowed.lean": "module\n"})
    written(T0, *(f for q in (ident, proofs) for f in q.rglob("*.lean")))
    built(ident, "Identity.Kept", "Identity.Gone", "Proofs.Shadowed", at=T0 + 100)
    built(proofs, "Proofs.Fine", "Proofs.Uses", "Proofs.Via", "Proofs.OnForgot", "Proofs.Shadowed", "Proofs.Dead", at=T0 + 100)
    built(proofs, "Proofs.Forgot", at=T0 + 100, trace=False)
    p = S.plan(proofs, "Proofs/File.lean")
    assert p == S.Plan("Proofs.File", True, ["Proofs.Fine"],
                       ["Proofs.Forgot", "Proofs.OnForgot", "Proofs.Shadowed", "Proofs.Uses", "Proofs.Via"])
    (work := tmp_path / "work").mkdir()
    S.stub(p, work)
    assert (work / f"{S.STUB}.lean").read_text() == "module\n\npublic import Proofs.File\npublic import Proofs.Fine\n"
    write(ident, {"Identity/Gone.lean": "module\n"})
    written(T0, ident / "Identity" / "Gone.lean")
    (ident / S.LIB / "Proofs" / "Shadowed.olean").unlink()
    (proofs / S.LIB / "Proofs" / "Forgot.trace").write_text("{}")
    assert S.plan(proofs, "Proofs/File.lean") == S.Plan(
        "Proofs.File", True, ["Proofs.Fine", "Proofs.Forgot", "Proofs.OnForgot", "Proofs.Shadowed", "Proofs.Uses", "Proofs.Via"], [])


def test_an_import_chain_longer_than_python_recurses_is_followed(tmp_path):
    """A function that calls itself once per import stops at Python's recursion limit, some hundreds of imports deep; the
    newest source is found at the far end of a chain several times that."""
    pkg, n = tmp_path / "p", 1500
    write(pkg, {"lakefile.toml": 'name = "p"\n' + LIB.format("P"), "P/File.lean": "module\n", "P/M0.lean": "module\n",
                **{f"P/M{i}.lean": f"module\nimport P.M{i - 1}\n" for i in range(1, n)}})
    written(T0, *pkg.rglob("*.lean"))
    built(pkg, *(f"P.M{i}" for i in range(n)), at=T0 + 100)
    assert S.plan(pkg, "P/File.lean").unchecked == []
    written(T0 + 200, pkg / "P" / "M0.lean")
    built(pkg, "P.M0", at=T0 + 300)
    p = S.plan(pkg, "P/File.lean")
    assert p.siblings == ["P.M0"] and len(p.unchecked) == n - 1


def test_options_are_the_packages_lean_options_each_weak(tmp_path):
    """A dotted option name is a TOML table, flattened back; a package with a `lakefile.lean` gives none."""
    write(tmp_path / "p", {"lakefile.toml": 'name = "p"\n\n[leanOptions]\nautoImplicit = false\nbackward.proofsInPublic = '
                                            'true\nmaxHeartbeats = 400000\nlinter.style = "x"\n'})
    assert S.options(tmp_path / "p") == ["-Dweak.autoImplicit=false", "-Dweak.backward.proofsInPublic=true",
                                         "-Dweak.maxHeartbeats=400000", "-Dweak.linter.style=x"]
    write(tmp_path / "q", {"lakefile.toml": 'name = "q"\n', "r/lakefile.lean": ""})
    assert S.options(tmp_path / "q") == [] and S.options(tmp_path / "q" / "r") == []


def test_the_line_names_both_modules_of_a_clash_and_counts_what_it_left_unchecked():
    p = S.Plan("P.D", True, ["P.C"], [f"P.U{i}" for i in range(S.SHOWN + 2)])
    rc, line = S.verdict(p, 1, "error: import P.C failed, environment already contains 'Q.bar' from P.D\n", 3.2, 900)
    assert rc == 1 and "P.D and P.C both declare 'Q.bar'" in line and "rename one" in line
    assert "1 imported in 3 s, peak 900 MB" in line and line.endswith("and 2 more")
    assert f"; {S.SHOWN + 2} unchecked, not built since their sources or imports changed" in line, "the count, not only names"
    rc, line = S.verdict(p, 1, "error: import P.X failed, environment already contains 'Q.z' from P.C\n", 1, 1)
    assert rc == 1 and "P.X and P.C both declare 'Q.z', neither of them P.D: a clash already in the sources" in line
    assert "rename one" in line and "lake build" not in line and f"; {S.SHOWN + 2} unchecked" in line
    assert S.verdict(p, 0, "", 1, 1)[0] == 0
    assert S.verdict(p, 1, "error: object file ... does not exist\n", 1, 1)[0] == 1


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_try_with_siblings_fails_a_module_that_redeclares_a_name_another_built_module_declares(tmp_path, capsys):
    """Each of `P.C` and `P.D` elaborates; only importing both refuses `Q.bar`, which `verity lean try` alone passes."""
    pkg = tmp_path / "p"
    shutil.copy(HERE / "fixtures" / "control" / "lean-toolchain", tmp_path)
    write(pkg, {"lakefile.toml": 'name = "p"\ndefaultTargets = ["P"]\n\n[[lean_lib]]\nname = "P"\n',
                "lean-toolchain": (tmp_path / "lean-toolchain").read_text(),
                "P.lean": "module\n\npublic import P.B\npublic import P.C\n",
                "P/A.lean": "module\n\n@[expose] public section\n\ndef P.foo : Nat := 1\n",
                "P/B.lean": "module\n\npublic import P.A\n@[expose] public section\n\ntheorem P.foo_eq : P.foo = 1 := rfl\n",
                "P/C.lean": "module\n\n@[expose] public section\n\nnamespace Q\n\ndef bar : Nat := 2\n\nend Q\n"})
    subprocess.run(["lake", "build"], cwd=pkg, check=True, capture_output=True)
    dup = "module\n\npublic import P.A\n@[expose] public section\n\nnamespace Q\n\ndef bar : Nat := P.foo\n\nend Q\n"
    write(pkg, {"P/D.lean": dup})
    cli = _cli()
    assert cli.main(["lean", "try", str(pkg / "P" / "D.lean")]) == 0, "D elaborates on its own"
    capsys.readouterr()
    assert cli.main(["lean", "try", str(pkg / "P" / "D.lean"), "--siblings"]) == 1
    out = capsys.readouterr().out
    assert "import P.C failed, environment already contains 'Q.bar' from P.D" in out, out
    assert "siblings: P.D and P.C both declare 'Q.bar'" in out and "rename one" in out, out
    assert not list((pkg / S.COPIES).glob("siblings-*")), "the scratch build is removed"
    write(pkg, {"P/D.lean": dup.replace("def bar", "def baz")})
    assert cli.main(["lean", "try", str(pkg / "P" / "D.lean"), "--siblings"]) == 0
    out = capsys.readouterr().out
    assert "siblings: no built module of its package that doesn't import P.D declares a name it does (4 imported" in out, out
    assert out.splitlines()[-1].startswith("verity lean try: P/D.lean: axiom probe:"), "the probe's line stays last"


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_try_with_siblings_passes_over_products_built_before_names_moved_and_fails_a_clash_in_the_sources(tmp_path, capsys):
    """INF-667, built with Lake. `proofs` requires `commit`, which requires `ident`, and all three are built. Then
    `Identity.tag` moves from `Identity.Domains` into the new `Identity.Domains.Pouw`, which `PouwCommit.Bindings` imports
    from then on, and of all that Lake builds only what FILE imports, as a try does on a slot tree synced across the move.
    The products of `Identity.Domains` (which still declares `Identity.tag`), of `PouwCommit.Bindings` and of `Proofs.Old`,
    which imports it, are from before the move, and Lean refuses `Proofs.Old` beside `Identity.Domains.Pouw`: a clash that
    is in no source, on which the try failed while it took another package's products as built. Now `Proofs.Old` and the
    root that imports it are unchecked, and the try passes. A clash that is in the sources still fails it: `Proofs.Dup`,
    built from a source that declares `Identity.tag` again. With that source unlinked and its product left, as `sync`
    leaves one, nothing loads the product. Once Lake has rebuilt `Proofs.Old`, the stub imports it again.

    The plan's tests above say which modules the stub names; only this one loads them, so only it shows that what the
    plan leaves out is what Lean refuses and that no product from before the move comes in through what it keeps."""
    past = 1_000_000_000.0
    toolchain = (HERE / "fixtures" / "control" / "lean-toolchain").read_text()
    ident, commit, proofs = tmp_path / "ident", tmp_path / "commit", tmp_path / "proofs"
    tag = "module\n\n@[expose] public section\n\ndef Identity.tag : Nat := 1\n"
    bindings = "module\n\npublic import {}\n@[expose] public section\n\ndef PouwCommit.bound : Nat := Identity.tag\n"
    write(ident, {"lakefile.toml": 'name = "ident"\n' + LIB.format("Identity"), "lean-toolchain": toolchain,
                  "Identity/Domains.lean": tag})
    write(commit, {"lakefile.toml": 'name = "commit"\n\n[[require]]\nname = "ident"\npath = "../ident"\n' + LIB.format("PouwCommit"),
                   "lean-toolchain": toolchain, "PouwCommit/Bindings.lean": bindings.format("Identity.Domains")})
    write(proofs, {"lakefile.toml": 'name = "proofs"\ndefaultTargets = ["Proofs"]\n\n[[require]]\nname = "commit"\n'
                                    'path = "../commit"\n' + LIB.format("Proofs"),
                   "lean-toolchain": toolchain, "Proofs.lean": "module\n\npublic import Proofs.Old\npublic import Proofs.Kept\n",
                   "Proofs/Old.lean": "module\n\npublic import PouwCommit.Bindings\n@[expose] public section\n\n"
                                      "theorem Proofs.old : PouwCommit.bound = 1 := rfl\n",
                   "Proofs/Kept.lean": "module\n\n@[expose] public section\n\ndef Proofs.kept : Nat := 3\n"})

    def lake(*argv: str) -> subprocess.CompletedProcess:
        return subprocess.run(["lake", *argv], cwd=proofs, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def sources() -> list[Path]:
        return [f for q in (ident, commit, proofs) for f in q.rglob("*.lean") if ".lake" not in f.parts]

    assert (first := lake("build")).returncode == 0, first.stdout
    assert S.workspace(proofs) == [proofs, commit, ident], "Lake's own manifest lists the package `commit` requires"
    # every source so far older than every product so far, and both in the past, whenever this runs
    written(past, *sources())
    written(past + 100, *(f for q in (ident, commit, proofs) for f in (q / ".lake" / "build").rglob("*") if f.is_file()))
    before = set(sources())
    write(ident, {"Identity/Domains.lean": "module\n", "Identity/Domains/Pouw.lean": tag})
    write(commit, {"PouwCommit/Bindings.lean": bindings.format("Identity.Domains.Pouw")})
    write(proofs, {"Proofs/File.lean": "module\n\npublic import Identity.Domains.Pouw\n@[expose] public section\n\n"
                                       "theorem Proofs.file : Identity.tag = 1 := rfl\n"})
    written(past + 200, ident / "Identity" / "Domains.lean", commit / "PouwCommit" / "Bindings.lean", *(set(sources()) - before))
    assert (deps := lake("build", "+Proofs.File:deps")).returncode == 0, deps.stdout
    assert (ident / S.LIB / "Identity" / "Domains.olean").stat().st_mtime == past + 100, "no target imports it, so it stays"
    (tmp_path / "Both.lean").write_text("import Proofs.Old\nimport Identity.Domains.Pouw\n")
    both = lake("env", "lean", str(tmp_path / "Both.lean"))
    assert both.returncode and "environment already contains 'Identity.tag' from Identity.Domains" in both.stdout, both.stdout

    cli, file = _cli(), str(proofs / "Proofs" / "File.lean")
    assert cli.main(["lean", "try", file, "--siblings"]) == 0, capsys.readouterr().out
    out = capsys.readouterr().out
    assert "siblings: no built module of its package that doesn't import Proofs.File declares a name it does (1 imported" in out, out
    assert ("; 2 unchecked, not built since their sources or imports changed, in their package or one it requires: Proofs, "
            "Proofs.Old") in out, out

    write(proofs, {"Proofs/Dup.lean": "module\n\n@[expose] public section\n\ndef Identity.tag : Nat := 2\n"})
    written(past + 200, proofs / "Proofs" / "Dup.lean")
    assert (dup := lake("build", "+Proofs.Dup")).returncode == 0, dup.stdout
    assert cli.main(["lean", "try", file, "--siblings"]) == 1
    out = capsys.readouterr().out
    assert ("siblings: Proofs.Dup and Identity.Domains.Pouw both declare 'Identity.tag', neither of them Proofs.File: a clash "
            "already in the sources") in out, out
    (proofs / "Proofs" / "Dup.lean").unlink()
    assert (proofs / S.LIB / "Proofs" / "Dup.olean").is_file()
    assert cli.main(["lean", "try", file, "--siblings"]) == 0, capsys.readouterr().out
    assert "(1 imported" in capsys.readouterr().out, "the product with no source is not a sibling, and nothing loads it"

    assert (again := lake("build")).returncode == 0, again.stdout
    assert "Proofs.Old" in S.plan(proofs, "Proofs/File.lean").siblings, "rebuilt after the move, so checked again"
    assert cli.main(["lean", "try", file, "--siblings"]) == 0, capsys.readouterr().out


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_try_refuses_an_auto_bound_name_where_its_package_sets_auto_implicit_false(tmp_path, capsys):
    """`lake env lean` alone takes `α` for an implicit argument, as the package's build doesn't (`options`)."""
    shutil.copy(HERE / "fixtures" / "control" / "lean-toolchain", tmp_path)
    for name, opts in (("loose", ""), ("strict", "[leanOptions]\nautoImplicit = false\n\n")):
        write(tmp_path / name, {"lakefile.toml": f'name = "{name}"\n\n{opts}[[lean_lib]]\nname = "P"\n',
                                "lean-toolchain": (tmp_path / "lean-toolchain").read_text(),
                                "P/A.lean": "def P.same (x : α) : α := x\n"})
    cli = _cli()
    assert cli.main(["lean", "try", str(tmp_path / "loose" / "P" / "A.lean")]) == 0
    capsys.readouterr()
    assert cli.main(["lean", "try", str(tmp_path / "strict" / "P" / "A.lean")]) == 1
    out = capsys.readouterr().out
    assert "lake env lean -Dweak.autoImplicit=false .lake/axiom-probe/P/A.lean" in out, out
    assert "unknown identifier" in out.lower() and "α" in out, out
