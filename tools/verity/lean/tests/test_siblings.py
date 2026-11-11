"""The sibling check (`siblings.py`) that `verity lean try --siblings` runs once FILE elaborates: which modules its stub
imports, and, with a real toolchain, that a module redeclaring a name another built module of its package declares fails
the try naming both, as the package's root fails a records run an hour in. And the package's Lean options every try
takes from it: with a real toolchain, an auto-bound name fails a try where the package sets `autoImplicit = false`."""
from __future__ import annotations

import importlib.util
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


def built(pkg: Path, *mods: str, at: float = 2_000_000_000.0) -> None:
    """Give each of ``mods`` an `.olean` newer than every source written before it."""
    for m in mods:
        f = (pkg / S.LIB).joinpath(*m.split(".")).with_suffix(".olean")
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"olean")
        os.utime(f, (at, at))


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
    """security/Proofs' layout: `srcDir = ".."`, so its library `Proofs` is the package's own directory, beside the outer
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
    rc, line = S.verdict(p, 1, "error: import P.X failed, environment already contains 'Q.z' from P.C\n", 1, 1)
    assert rc == 1 and "neither of them P.D" in line and "lake build +P.X +P.C" in line
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
