"""The sibling check (`siblings.py`) that `verity lean try --siblings` runs once FILE elaborates: which modules its stub
imports, and, with a real toolchain, that a module redeclaring a name another built module of its package declares fails
the try naming both, as the package's root fails a records run an hour in. Its stub loads no build that is not of its
source as it is, by the hash Lake's trace states or, of a build Lake took rather than made, by Lake's own word, in the
package or in one it requires by path, and none without a source: with a real toolchain, a try passes over the products
a tree keeps from before names moved between modules, and still fails a clash that is in the sources; and a build Lake
took is a sibling until its source's text changes, and again once the source is written as it was. A first refusal that names a sibling has Lake rebuild it, once, before
the stub's word is the verdict: with a real toolchain, a theorem Lean makes in two modules, one of them left built
against an older build of an import, is refused in the stub and in a root alike until Lake rebuilds that module, and a
name two sources declare is refused after the rebuild too. And the package's Lean options every try takes from it:
with a real toolchain, an auto-bound name fails a try where the package sets `autoImplicit = false`."""
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


def built(pkg: Path, *mods: str, trace: bool = True, of: str | None = None) -> None:
    """Give each of ``mods`` an `.olean` and, as Lake leaves a module it built, a trace (``trace``) that states the hash of
    the source it was built from: the module's source as it is in ``pkg`` now, or ``of``, a text it had before."""
    src = S.source_dir(pkg)[0]
    for m in mods:
        f = (pkg / S.LIB).joinpath(*m.split(".")).with_suffix(".olean")
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(b"olean")
        source = src.joinpath(*m.split(".")).with_suffix(".lean")
        text = of if of is not None else source.read_text() if source.is_file() else ""
        if trace:
            f.with_suffix(".trace").write_text(json.dumps({"inputs": [["options", "00000000000006bb"],
                                                                      [f"/tree/{source}", S.lake_hash(text.encode())]]}))


#: a stand-in workspace: `proofs` requires `commit` by path, which requires `ident`, as `proofs`' manifest lists them
LIB = '\n[[lean_lib]]\nname = "{}"\n'
MANIFEST = json.dumps({"packages": [{"name": "commit", "type": "path", "dir": "../commit", "inherited": False},
                                    {"name": "mathlib", "type": "git", "inherited": True},
                                    {"name": "ident", "type": "path", "dir": "../commit/../ident", "inherited": True}]})


def test_the_hash_is_the_one_lakes_trace_states_for_a_source():
    """Three sources Lean 4.34's Lake built, each with the hash its trace states: one shorter than a word of the hash, one
    with a character of two bytes, and one with CRLF line ends, which Lake hashes as LF. `built_from` reads that hash
    from a trace by the source's file name, and nothing from a trace that states none."""
    assert S.lake_hash(b"module\n") == "f1512c6392544dc8"
    assert S.lake_hash("module\n\n@[expose] public section\n\ndef P.b : Nat := 2 -- \u00e9\n".encode()) == "97b565cade92b543"
    crlf = b"module\r\n\r\npublic import P.B\r\n@[expose] public section\r\n\r\ndef P.a : Nat := 1\r\n"
    assert S.lake_hash(crlf) == S.lake_hash(crlf.replace(b"\r\n", b"\n")) == "42d00b9fd865c9a6"


def test_a_trace_states_its_source_by_file_name_or_states_none(tmp_path):
    trace = tmp_path / "A.trace"
    trace.write_text(json.dumps({"depHash": "950f7100ae4d8693", "inputs": [
        ["deps", [["imports", [["P.B:importArts", "722ab1e256688428"]]]]], ["Lean 4.34.0", "a1c6c7aa3e9abfe7"],
        ["/tree/p/P/A.lean", "42d00b9fd865c9a6"], ["options", "00000000000006bb"]]}))
    assert S.built_from(trace, "A.lean") == ("42d00b9fd865c9a6", False)
    assert S.built_from(trace, "B.lean") == (None, False) == S.built_from(tmp_path / "none.trace", "A.lean")
    for text in ("{}", "[]", '{"inputs": [["/tree/p/P/A.lean", "42d0', '{"inputs": {"/tree/p/P/A.lean": "42d00b9fd865c9a6"}}',
                 '{"inputs": [["/a/A.lean", "1"], ["/b/A.lean", "2"]]}',
                 '{"synthetic": true, "schemaVersion": "2025-09-10", "inputs": [["options", "06bb"]], "depHash": "b3c1e3c25b4816c6"}',
                 '{"synthetic": true, "inputs": []}', '{"synthetic": true, "schemaVersion": "2025-09-10", "inputs": []}',
                 '{"depHash": "12345678901234567890"}', '{"schemaVersion": "2025-09-10", "depHash": 7, "outputs": {}}',
                 '{"schemaVersion": "2025-09-10", "depHash": "b3c1e3c25b4816c6", "inputs": {}}', "12345678901234567890"):
        trace.write_text(text)
        assert S.built_from(trace, "A.lean") == (None, False), text
    trace.write_text('{"synthetic": true, "schemaVersion": "2025-09-10", "outputs": {}, "log": [], "inputs": [], "depHash": "b3c1e3c25b4816c6"}')
    assert S.built_from(trace, "A.lean") == (None, True), "as Lake writes it for a module it took from its cache"
    trace.write_text('{"schemaVersion":"2025-09-10","depHash":"255a6a20482e9767","outputs":{"m":true,"o":["3775597e598b9774.olean"]}}')
    assert S.built_from(trace, "A.lean") == (None, True), "as leantar writes it for a module unpacked from its .ltar"


def test_the_stub_imports_the_built_modules_that_import_file_neither_directly_nor_through_others(tmp_path):
    """`P.B` imports FILE, `P.Top` imports `P.B`: neither is a sibling. `P.C`'s source changed after it was built, `P.E`
    imports `P.C`, and `P.F` isn't built: all three are unchecked. A nested package's modules and `.lake` aren't the
    package's."""
    pkg = tmp_path / "p"
    write(pkg, {"lakefile.toml": 'name = "p"\n\n[[lean_lib]]\nname = "P"\n', "P/D.lean": "module\n",
                "P/A.lean": "module\n", "P/B.lean": "module\npublic import P.D\n", "P/Top.lean": "module\nimport P.B\n",
                "P/C.lean": "module\n", "P/E.lean": "module\nimport P.C\nimport Mathlib\n", "P/F.lean": "module\n",
                "P/Nested/lakefile.toml": 'name = "n"\n', "P/Nested/X.lean": "module\n", "P/.lake/Y.lean": ""})
    built(pkg, "P.A", "P.B", "P.Top", "P.E")
    built(pkg, "P.C", of="module\n\ndef c := 1\n")
    p = S.plan(pkg, "P/D.lean")
    assert p == S.Plan("P.D", True, ["P.A"], ["P.C", "P.E", "P.F"])
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


#: a script run with `lean --run`: a module of the library that no module imports, and that declares `main` in public
SCRIPT = "module\n\n@[expose] public section\n\ndef main (_ : List String) : IO UInt32 := pure 0\n"


def test_the_stub_leaves_out_the_modules_under_an_exempt_prefix_of_the_lock(tmp_path):
    """The lock's `exempt` lists the module prefixes nothing imports: `P.Run`, which holds two scripts, and the script
    `P.Probe` by its own name. No root loads one beside the library, so none is a sibling of a library module, nor
    counted as unchecked where it isn't built (`P.Run.Later`). A prefix is matched as the audit matches it, the name or
    below it: `P.Runner` and `P.Probes` are under neither. FILE's own module stays in: a script as FILE has the
    library's modules for siblings, and not the other scripts."""
    pkg = tmp_path / "p"
    write(pkg, {"lakefile.toml": 'name = "p"\n' + LIB.format("P"),
                "lean-audit.json": json.dumps({"exempt": {"P.Run": "scripts the audit runs", "P.Probe": "run by hand"}}),
                "P/A.lean": "module\n", "P/D.lean": "module\n", "P/Runner.lean": "module\n", "P/Probes.lean": "module\n",
                "P/Run/Bulk.lean": SCRIPT, "P/Run/Later.lean": SCRIPT, "P/Probe.lean": SCRIPT})
    built(pkg, "P.A", "P.Runner", "P.Run.Bulk", "P.Probe")
    assert S.exempt(pkg) == ["P.Run", "P.Probe"]
    p = S.plan(pkg, "P/D.lean")
    assert p == S.Plan("P.D", True, ["P.A", "P.Runner"], ["P.Probes"])
    (work := tmp_path / "work").mkdir()
    S.stub(p, work)
    assert (work / f"{S.STUB}.lean").read_text() == "module\n\npublic import P.D\npublic import P.A\npublic import P.Runner\n"
    assert S.plan(pkg, "P/Run/Bulk.lean") == S.Plan("P.Run.Bulk", True, ["P.A", "P.Runner"], ["P.D", "P.Probes"])
    assert S.plan(pkg, "P/Probe.lean") == S.Plan("P.Probe", True, ["P.A", "P.Runner"], ["P.D", "P.Probes"])


def test_a_plan_never_fails_on_the_lock(tmp_path, capsys):
    """The audit judges the lock; the sibling check only reads the names under its `exempt`. With no lock (or a
    directory in its place), one whose `exempt` holds no names, or one that can't be read (no JSON, bytes that aren't
    text), a package plans as one whose lock exempts nothing: both scripts are siblings. A lock that can't be read is
    said on stderr, once a plan, and nothing else is. Nor does the plan ask for the reason the audit asks of an entry."""
    pkg = tmp_path / "p"
    write(pkg, {"lakefile.toml": 'name = "p"\n' + LIB.format("P"), "P/A.lean": "module\n", "P/D.lean": "module\n",
                "P/Run/Bulk.lean": SCRIPT, "P/Probe.lean": SCRIPT})
    built(pkg, "P.A", "P.Run.Bulk", "P.Probe")
    lock, without = pkg / "lean-audit.json", S.Plan("P.D", True, ["P.A", "P.Probe", "P.Run.Bulk"], [])
    assert S.exempt(pkg) == [] and S.plan(pkg, "P/D.lean") == without, "no lock"
    lock.mkdir()
    assert S.plan(pkg, "P/D.lean") == without, "a directory in its place"
    lock.rmdir()
    for text in ("[]", "null", '"P.Run"', '{"roots": ["P"]}', '{"exempt": null}', '{"exempt": "P.Run"}', '{"exempt": 7}',
                 '{"exempt": [["P.Run"], 7, null]}', '{"exempt": {}}'):
        lock.write_text(text)
        assert S.exempt(pkg) == [] and S.plan(pkg, "P/D.lean") == without, text
    assert capsys.readouterr().err == "", "none of these is a lock that can't be read"
    for data in (b"", b"{", b'{"exempt": {"P.Run": "why", "P.Probe": "why"}', b'{"exempt": {"P.Run\xff": "why"}}'):
        lock.write_bytes(data)
        assert S.plan(pkg, "P/D.lean") == without, data
        err = capsys.readouterr().err
        assert err.startswith(f"siblings: {lock} can't be read (") and err.count("\n") == 1, err
        assert err.endswith("): no module is left out for its `exempt`\n"), err
    for text in ('{"exempt": {"P.Run": "", "P.Probe": null}}', '{"exempt": ["P.Run", "P.Probe", 7]}'):
        lock.write_text(text)
        assert S.plan(pkg, "P/D.lean") == S.Plan("P.D", True, ["P.A"], []), text
    assert capsys.readouterr().err == ""


def scripts(tmp_path: Path, exempt: dict[str, str]) -> Path:
    """FILE, `P/D.lean`, in a package Lake built: the library's root `P` and its `P.A`, and two scripts that each
    declare `main`, `P.Run.Bulk` and `P.Probe`, which the lock lists under ``exempt`` or doesn't."""
    pkg = tmp_path / "p"
    shutil.copy(HERE / "fixtures" / "control" / "lean-toolchain", tmp_path)
    write(pkg, {"lakefile.toml": 'name = "p"\ndefaultTargets = ["P"]\n' + LIB.format("P"),
                "lean-toolchain": (tmp_path / "lean-toolchain").read_text(),
                "lean-audit.json": json.dumps({"roots": ["P"], "exempt": exempt}),
                "P.lean": "module\n\npublic import P.A\n",
                "P/A.lean": "module\n\n@[expose] public section\n\ndef P.foo : Nat := 1\n",
                "P/Run/Bulk.lean": SCRIPT, "P/Probe.lean": SCRIPT})
    subprocess.run(["lake", "build", "P", "+P.Run.Bulk", "+P.Probe"], cwd=pkg, check=True, capture_output=True)
    write(pkg, {"P/D.lean": "module\n\npublic import P.A\n@[expose] public section\n\ndef P.bar : Nat := P.foo\n"})
    assert S.plan(pkg, "P/D.lean").unchecked == [], "Lake built both scripts"
    return pkg / "P" / "D.lean"


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_try_with_siblings_passes_beside_two_scripts_that_declare_main_under_exempt_prefixes(tmp_path, capsys):
    """`P.Run.Bulk` and `P.Probe` are built and each declares `main`. The lock lists `P.Run` and `P.Probe` under
    `exempt`, so the stub imports neither, only the library's two modules, and the try of a library module passes."""
    file = scripts(tmp_path, {"P.Run": "scripts the audit runs", "P.Probe": "run by hand"})
    assert S.plan(file.parents[1], "P/D.lean").siblings == ["P", "P.A"]
    assert _cli().main(["lean", "try", str(file), "--siblings"]) == 0, capsys.readouterr().out
    out = capsys.readouterr().out
    assert "siblings: no built module of its package that doesn't import P.D declares a name it does (2 imported" in out, out
    assert "unchecked" not in out and "'main'" not in out, out


@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_try_with_siblings_refuses_two_scripts_that_declare_main_under_no_exempt_prefix(tmp_path, capsys):
    """The control: the same package, its lock with no `exempt` entry. The stub imports both scripts beside the library,
    Lean refuses the second for the `main` the first declares, and the line names both. Lake, which the try has build
    the two, holds both current and rebuilds neither, and the stub's second refusal is the verdict."""
    file = scripts(tmp_path, {})
    assert S.plan(file.parents[1], "P/D.lean").siblings == ["P", "P.A", "P.Probe", "P.Run.Bulk"]
    assert _cli().main(["lean", "try", str(file), "--siblings"]) == 1
    out = capsys.readouterr().out
    assert "import P.Run.Bulk failed, environment already contains 'main' from P.Probe" in out, out
    assert "siblings: P.Run.Bulk and P.Probe both declare 'main' as they are built here: Lake rebuilt nothing in " in out, out
    assert ("siblings: P.Run.Bulk and P.Probe both declare 'main', neither of them P.D, and Lake holds both current, which "
            "this try had it build: a clash in the sources") in out, out


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


def moved(ws: Path) -> tuple[Path, Path, Path]:
    """A tree of three packages (`ident`, `commit`, `proofs`): everything was built, and then a sync brought a move of
    `Identity.Domains`' names into the new `Identity.Domains.Pouw`. It rewrote `Identity/Domains.lean` and what imports
    the new module from then on, `PouwCommit/Bindings.lean` and `Proofs/New.lean`, and wrote `Identity/Domains/Pouw.lean`
    and FILE. Since then Lake built only what a target imported, `Identity.Domains.Pouw` and `Proofs.New`: the products
    of `Identity.Domains`, of `PouwCommit.Bindings`, of `Proofs.Old`, which imports that, and of `Proofs.Far`, which
    imports `Proofs.Old`, are from before the move, and the last two's sources never changed."""
    ident, commit, proofs = ws / "ident", ws / "commit", ws / "proofs"
    write(ident, {"lakefile.toml": 'name = "ident"\n' + LIB.format("Identity"),
                  "Identity/Domains.lean": "module\n\ndef Identity.tag := 1\n"})
    write(commit, {"lakefile.toml": 'name = "commit"\n' + LIB.format("PouwCommit"),
                   "PouwCommit/Bindings.lean": "module\npublic import Identity.Domains\n"})
    write(proofs, {"lakefile.toml": 'name = "proofs"\n' + LIB.format("Proofs"), "lake-manifest.json": MANIFEST,
                   "Proofs/Old.lean": "module\npublic import PouwCommit.Bindings\n",
                   "Proofs/Far.lean": "module\nimport Proofs.Old\n",
                   "Proofs/New.lean": "module\npublic import Identity.Domains\n",
                   "Proofs/Alone.lean": "module\nimport Mathlib\n"})
    built(ident, "Identity.Domains")
    built(commit, "PouwCommit.Bindings")
    built(proofs, "Proofs.Old", "Proofs.Far", "Proofs.New", "Proofs.Alone")
    write(ident, {"Identity/Domains.lean": "module\n", "Identity/Domains/Pouw.lean": "module\n\ndef Identity.tag := 1\n"})
    write(commit, {"PouwCommit/Bindings.lean": "module\npublic import Identity.Domains.Pouw\n"})
    write(proofs, {"Proofs/New.lean": "module\npublic import Identity.Domains.Pouw\n",
                   "Proofs/File.lean": "module\npublic import Identity.Domains.Pouw\n"})
    built(ident, "Identity.Domains.Pouw")
    built(proofs, "Proofs.New")
    return ident, commit, proofs


def test_a_sibling_that_loads_a_build_from_before_a_source_in_another_package_changed_is_unchecked(tmp_path):
    """On `moved`'s tree, `Proofs.Old`'s source never changed, and its build loads `PouwCommit.Bindings` and through it
    `Identity.Domains` as they were built before the move, the second declaring what `Identity.Domains.Pouw` declares
    now: a stub that imports it beside FILE fails on a clash that is in no source. `PouwCommit.Bindings`' build is not
    of its source as it is, in a package `proofs` requires through another, so `Proofs.Old` is unchecked, and so is
    `Proofs.Far`, which imports it. Both are siblings again once Lake has rebuilt `PouwCommit.Bindings`: what their own
    builds name is what their sources name."""
    ident, commit, proofs = moved(tmp_path)
    assert S.workspace(proofs) == [proofs, commit, ident] and S.workspace(ident) == [ident], "its manifest's path packages"
    for text in ('{"packages": [{"type": "path", "dir": "../comm', "[]", '{"packages": {}}'):
        (ident / "lake-manifest.json").write_text(text)
        assert "lake-manifest.json can't be read for the packages it requires" in S.plan(ident, "Identity/Domains.lean"), \
            f"no check on a manifest that would hide a required package: {text}"
    (ident / "lake-manifest.json").unlink()
    assert S.plan(proofs, "Proofs/File.lean") == S.Plan("Proofs.File", True, ["Proofs.Alone", "Proofs.New"],
                                                        ["Proofs.Far", "Proofs.Old"])
    built(commit, "PouwCommit.Bindings")
    assert S.plan(proofs, "Proofs/File.lean") == S.Plan("Proofs.File", True, ["Proofs.Alone", "Proofs.Far", "Proofs.New",
                                                                              "Proofs.Old"], [])


def test_a_source_written_again_as_it_was_built_from_leaves_its_build_checked(tmp_path):
    """A slot tree serves one branch after another. A sync to a branch that changes `Identity/Domains/Pouw.lean`, two
    packages down, leaves every build that loads its product unchecked; the sync back writes the source again as it was
    built from, later than every product in the tree, and Lake rebuilds nothing. Each is a sibling again, which no
    comparison of modification times gives: by those every tree that serves two branches ends with nothing checked."""
    ident, commit, proofs = moved(tmp_path)
    built(commit, "PouwCommit.Bindings")
    was = (pouw := ident / "Identity" / "Domains" / "Pouw.lean").read_text()
    pouw.write_text(was + "\ndef Identity.other := 2\n")
    assert S.plan(proofs, "Proofs/File.lean") == S.Plan("Proofs.File", True, ["Proofs.Alone"],
                                                        ["Proofs.Far", "Proofs.New", "Proofs.Old"])
    pouw.write_text(was)
    assert S.plan(proofs, "Proofs/File.lean").unchecked == []


#: a trace as Lake writes it for a module whose outputs it found in its artifact cache, and as `leantar` writes it for
#: one unpacked from its `.ltar`
TOOK = {"cache": '{"synthetic": true, "schemaVersion": "2025-09-10", "outputs": {}, "log": [], "inputs": [], '
                 '"depHash": "b3c1e3c25b4816c6"}',
        "ltar": '{"schemaVersion":"2025-09-10","depHash":"255a6a20482e9767","outputs":{}}'}


def lake_says(said: str, asked: list | None = None):
    """A stand-in for `plan`'s ``lake`` that prints ``said``, and lists in ``asked`` what it is run with."""
    def run(args: list[str]) -> str:
        if asked is not None:
            asked.append(args)
        return said
    return run


@pytest.mark.parametrize("took", list(TOOK.values()), ids=list(TOOK))
def test_a_module_lake_took_rather_than_built_counts_on_lakes_word_alone(tmp_path, took):
    """Lake writes a trace with no input for a module it finds in its artifact cache, and `leantar` one for a module
    unpacked from its `.ltar`, as a node's records runs leave most of a slot tree, the definitions everything imports
    among them: there is no source hash to compare. Such a module is a sibling only where Lake, asked once about the
    modules that could be siblings, says that it holds it as its inputs are now. The times say nothing either way: a
    source written again as it was is newer than its build, and counts; one whose text changed while its trace was
    written later is older, and doesn't. Where Lake would compile the module, says nothing of it, or isn't asked, it is
    unchecked, and so is what imports it."""
    pkg, then = tmp_path / "p", 2_000_000_000.0
    write(pkg, {"lakefile.toml": 'name = "p"\n' + LIB.format("P"), "P/File.lean": "module\n", "P/A.lean": "module\n",
                "P/B.lean": "module\nimport P.A\n", "P/Up.lean": "module\nimport P.File\n"})
    built(pkg, "P.A", "P.B", "P.Up")
    (trace := pkg / S.LIB / "P" / "A.trace").write_text(took)
    asked, none = [], S.Plan("P.File", True, [], ["P.A", "P.B"])
    for source, written in ((then + 100, then), (then, then + 100)):
        os.utime(pkg / "P" / "A.lean", (source, source))
        os.utime(trace, (written, written))
        assert S.plan(pkg, "P/File.lean", lake_says("✔ [1/2] Replayed P.A\n✔ [2/2] Replayed P.B\n", asked)) == \
            S.Plan("P.File", True, ["P.A", "P.B"], [])
        for said in ("✖ [1/2] Building P.A\nerror: target is out-of-date and needs to be rebuilt\n✔ [2/2] Ran P.B\n",
                     "✔ [1/2] Ran P.A\n✔ [2/2] Replayed P.B\n", "error: no such target\n", ""):
            assert S.plan(pkg, "P/File.lean", lake_says(said)) == none, said
        assert S.plan(pkg, "P/File.lean") == none, "Lake isn't asked"
    assert asked == [["build", "--no-build", "-v", "--no-ansi", "+P.A", "+P.B"]] * 2, \
        "once a plan, about the modules that import FILE neither directly nor through others"
    built(pkg, "P.A")
    assert S.plan(pkg, "P/File.lean").unchecked == [], "built from the source as it is, which its trace states"


def test_lake_is_asked_before_a_product_is_read_since_it_puts_in_place_what_it_holds_packed(tmp_path):
    """`lake build --no-build` unpacks the archive it holds a module's outputs in, and says so: `P.A` has no product
    until Lake is asked, and is a sibling after. An executable's root, a module under an `exempt` prefix and FILE's
    own importers are no siblings, so Lake isn't asked about them; with no module to ask about, it isn't run."""
    pkg = tmp_path / "p"
    write(pkg, {"lakefile.toml": 'name = "p"\n' + LIB.format("P") + '\n[[lean_exe]]\nname = "tool"\nroot = "P.Tool"\n',
                "lean-audit.json": json.dumps({"exempt": ["P.Scripts"]}), "P/File.lean": "module\n", "P/A.lean": "module\n",
                "P/Tool.lean": "module\n", "P/Scripts/Run.lean": "module\n", "P/Up.lean": "module\nimport P.File\n"})
    asked: list = []

    def lake(args: list[str]) -> str:
        asked.append(args)
        built(pkg, "P.A")
        (pkg / S.LIB / "P" / "A.trace").write_text(TOOK["ltar"])
        return "ℹ [1/1] Unpacked P.A (12ms)\n"

    assert S.plan(pkg, "P/File.lean") == S.Plan("P.File", True, [], ["P.A"])
    assert S.plan(pkg, "P/File.lean", lake) == S.Plan("P.File", True, ["P.A"], [])
    assert asked == [[*S.WORD, "+P.A"]]
    assert S.plan(pkg, "P/A.lean", lake) == S.Plan("P.A", True, [], ["P.File", "P.Up"]) and asked[1] == [*S.WORD, "+P.File", "+P.Up"]
    (pkg / "P" / "A.lean").unlink()
    (pkg / "P" / "Up.lean").unlink()
    assert S.plan(pkg, "P/File.lean", lake) == S.Plan("P.File", True, [], []) and len(asked) == 2, "nothing to ask about"


def test_lakes_word_is_read_on_the_line_of_the_job_that_makes_a_modules_olean():
    """Lines as `lake build --no-build -v --no-ansi` prints them. A job that ended without an error and without
    compiling has a verb in the past: Lake replayed the build the tree holds, reused its cache's or unpacked its
    archive, with or without a log under the line, a time after it, or the mark of an optional job. A job Lake would
    compile for fails with the verb of what it tried, and so does any job that fails; one whose import failed only
    ran. A module's other jobs carry captions that are no module's name, and a line of a build's log is no job's."""
    said = ("✔ [1/20] Replayed P.A:setup\n"
            "✔ [2/20] Replayed P.A\n"
            "ℹ [3/20] Replayed P.B (1.2s)\n"
            "info: P/B.lean:3:0: said while it compiled\n"
            "⚠ [4/20] Replayed P.«C-d» (340ms)\n"
            "warning: P/C-d.lean:1:0: declaration uses `sorry`\n"
            "ℹ [5/20] Unpacked P.E (12s)\n"
            "✔ [6/20] Reused P.F\n"
            "✔ [7/20] (Optional) Replayed P.G\n"
            "✔ [8/20] Ran P.H\n"
            "✖ [9/20] Building P.I\n"
            "trace: .> LEAN_PATH=/tree/.lake/build/lib/lean lean /tree/P/I.lean\n"
            "error: target is out-of-date and needs to be rebuilt\n"
            "✖ [10/20] Replaying P.J\n"
            "✖ [11/20] Unpacking P.K\n"
            "✖ [12/20] Running P.L\n"
            "✔ [13/20] Fetched P.M\n"
            "✔ [14/20] Built P.N (2.0s)\n"
            "  ✔ [15/20] Replayed P.O\n"
            "✔ [16/20] Replayed P.Q and more\n"
            "✔ [17/20] Replayed P.R:c.o (with exports)\n"
            "Some required targets logged failures:\n- P.I\n- P.J\n")
    assert S.held(said) == {"P.A:setup", "P.A", "P.B", "P.C-d", "P.E", "P.F", "P.G"}
    assert S.held("") == set() == S.held("error: unknown target `+P.A`\n")


def test_no_stub_imports_a_product_with_no_source_or_a_module_whose_build_loads_one(tmp_path):
    """A sync unlinks a source its branch doesn't have and leaves the product, for the branch that has it. `Proofs.Dead`
    and `Identity.Gone` are such products: neither is a module of a package, so neither is a sibling, and nothing that
    loads one is. `Proofs.Uses` still imports `Identity.Gone` and its build is of its own source, and `Proofs.Via`
    imports `Proofs.Uses`: both unchecked. So is `Proofs.Shadowed`, whose name `ident` holds a product of with no source,
    which Lean could load in its place. A module without its trace is one Lake rebuilds (`lean_changed.clashed` drops
    the trace of a module whose build it can't trust), and a trace that states no source says nothing of which it was
    built from: `Proofs.Forgot`, and `Proofs.OnForgot`, which imports it. With its source in place a module is one like
    any other."""
    ident, proofs = tmp_path / "ident", tmp_path / "proofs"
    write(ident, {"lakefile.toml": 'name = "ident"\n' + LIB.format("Identity"), "Identity/Kept.lean": "module\n"})
    write(proofs, {"lakefile.toml": 'name = "proofs"\n' + LIB.format("Proofs"),
                   "lake-manifest.json": json.dumps({"packages": [{"name": "ident", "type": "path", "dir": "../ident"}]}),
                   "Proofs/File.lean": "module\n", "Proofs/Fine.lean": "module\nimport Identity.Kept\n",
                   "Proofs/Uses.lean": "module\nimport Identity.Gone\n", "Proofs/Via.lean": "module\nimport Proofs.Uses\n",
                   "Proofs/Forgot.lean": "module\n", "Proofs/OnForgot.lean": "module\nimport Proofs.Forgot\n",
                   "Proofs/Shadowed.lean": "module\n"})
    built(ident, "Identity.Kept", "Identity.Gone", "Proofs.Shadowed", of="module\n")
    built(proofs, "Proofs.Fine", "Proofs.Uses", "Proofs.Via", "Proofs.OnForgot", "Proofs.Shadowed", "Proofs.Dead")
    built(proofs, "Proofs.Forgot", trace=False)
    unchecked = ["Proofs.Forgot", "Proofs.OnForgot", "Proofs.Shadowed", "Proofs.Uses", "Proofs.Via"]
    p = S.plan(proofs, "Proofs/File.lean")
    assert p == S.Plan("Proofs.File", True, ["Proofs.Fine"], unchecked)
    (work := tmp_path / "work").mkdir()
    S.stub(p, work)
    assert (work / f"{S.STUB}.lean").read_text() == "module\n\npublic import Proofs.File\npublic import Proofs.Fine\n"
    (proofs / S.LIB / "Proofs" / "Forgot.trace").write_text('{"outputs": {"o": ["892a0563bc7b7627.olean"]}}')
    assert S.plan(proofs, "Proofs/File.lean") == p, "a trace that states no source"
    write(ident, {"Identity/Gone.lean": "module\n"})
    (ident / S.LIB / "Proofs" / "Shadowed.olean").unlink()
    built(proofs, "Proofs.Forgot")
    assert S.plan(proofs, "Proofs/File.lean") == S.Plan("Proofs.File", True, ["Proofs.Fine", *unchecked], [])


def test_an_import_chain_longer_than_python_recurses_is_followed(tmp_path):
    """A function that calls itself once per import stops at Python's recursion limit, some hundreds of imports deep; the
    changed source is found at the far end of a chain several times that."""
    pkg, n = tmp_path / "p", 1500
    write(pkg, {"lakefile.toml": 'name = "p"\n' + LIB.format("P"), "P/File.lean": "module\n", "P/M0.lean": "module\n",
                **{f"P/M{i}.lean": f"module\nimport P.M{i - 1}\n" for i in range(1, n)}})
    built(pkg, *(f"P.M{i}" for i in range(n)))
    assert S.plan(pkg, "P/File.lean").unchecked == []
    write(pkg, {"P/M0.lean": "module\n\ndef m := 0\n"})
    p = S.plan(pkg, "P/File.lean")
    assert p.siblings == [] and len(p.unchecked) == n
    built(pkg, "P.M0")
    assert S.plan(pkg, "P/File.lean").unchecked == []


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
    assert f"; {S.SHOWN + 2} unchecked, with no build of their source as it is or importing a module with none" in line, \
        "the count, not only names"
    rc, line = S.verdict(p, 1, "error: import P.X failed, environment already contains 'Q.z' from P.C\n", 1, 1)
    assert rc == 1 and "P.X and P.C both declare 'Q.z', neither of them P.D: a clash already in the sources" in line
    assert "rename one" in line and "`lake build +P.X +P.C` and try again" in line and f"; {S.SHOWN + 2} unchecked" in line
    assert S.verdict(p, 0, "", 1, 1)[0] == 0
    assert S.verdict(p, 1, "error: object file ... does not exist\n", 1, 1)[0] == 1


REFUSAL = "VeritySiblings.lean:1:0: error: import {} failed, environment already contains '{}' from {}\n"
#: Lake's output for a rebuild that compiled `P.B` alone, and `check`'s line for it
REBUILT = (0, "✔ [3/4] Built P.B (1.1s)\nBuild completed successfully.\n", 12.4)
ONCE = ("siblings: P.B and P.A both declare 'P.step.eq_1' as they are built here: Lake rebuilt P.B in 12 s, and the stub is "
        "elaborated once more")
COST = "(5 imported in 20 s, peak 800 MB)"


def checked(tmp_path, outputs: list[str], lake: tuple[int, str, float] | None = None):
    """`check` of `P.D` beside five siblings, its stub answered by ``outputs`` in turn (a refusal, or `""` for a pass) and
    its rebuild by ``lake`` (None: `check` is given no `rebuild`): (the status, the lines, how many times the stub ran,
    the outputs `rebuild` was given)."""
    p, ran, given = S.Plan("P.D", True, ["P.A", "P.B", "P.C", "P.E", "P.F"], []), [], []

    def lean(args: list[str]) -> tuple[int, str, float, int]:
        ran.append(args)
        return (1 if outputs[len(ran) - 1] else 0), outputs[len(ran) - 1], 20.0, 800

    def rebuild(output: str) -> tuple[int, str, float]:
        given.append(output)
        return lake

    rc, output, lines = S.check(p, tmp_path, lean, rebuild if lake else None)
    assert output == outputs[len(ran) - 1], "the last stub's output"
    return rc, lines.splitlines(), len(ran), given


def test_a_first_refusal_has_lake_rebuild_the_siblings_it_names_and_the_stub_elaborated_once_more(tmp_path):
    """`P.B` holds a theorem Lean made for it with the type an older build of an import gave it, and `P.A` with today's.
    Lake rebuilds `P.B` alone, and the stub passes: the rebuild's line, then the line of a check that passes."""
    refusal = REFUSAL.format("P.B", "P.step.eq_1", "P.A")
    rc, lines, ran, given = checked(tmp_path, [refusal, ""], REBUILT)
    assert (rc, ran, given) == (0, 2, [refusal])
    assert lines == [ONCE, f"siblings: no built module of its package that doesn't import P.D declares a name it does {COST}"]


def test_a_second_refusal_is_the_verdict_and_its_line_says_whether_this_try_rebuilt_the_modules_it_names(tmp_path):
    """One rebuild and one more stub, never a second rebuild. The same two again are builds Lake holds current: a clash
    in the sources. Another two, or one of them, this try did not rebuild, so its line says they may be stale builds
    too, which a second try rebuilds."""
    first = REFUSAL.format("P.B", "P.step.eq_1", "P.A")
    rc, lines, ran, given = checked(tmp_path, [first, first], REBUILT)
    assert (rc, ran, given) == (1, 2, [first])
    assert lines == [ONCE, (
        "siblings: P.B and P.A both declare 'P.step.eq_1', neither of them P.D, and Lake holds both current, which this try "
        "had it build: a clash in the sources, which a root that imports both refuses too (rename one); P.D's own names are "
        f"unchecked {COST}")]

    second = REFUSAL.format("P.E", "P.run.eq_2", "P.C")
    rc, lines, ran, given = checked(tmp_path, [first, second, ""], REBUILT)
    assert (rc, ran, given) == (1, 2, [first]), "no second rebuild, and no third stub"
    assert lines == [ONCE, (
        "siblings: P.E and P.C both declare 'P.run.eq_2', neither of them P.D, and this try did not rebuild P.E or P.C (it "
        "rebuilds once): a clash in the sources, which a root that imports both refuses too (rename one), or a build made "
        "against an older build of a module it imports, which a second try rebuilds; P.D's own names are unchecked "
        f"{COST}")]

    shared = REFUSAL.format("P.F", "P.step.eq_1", "P.A")
    rc, lines, ran, given = checked(tmp_path, [first, shared], REBUILT)
    assert (rc, ran, given) == (1, 2, [first])
    assert "P.F and P.A both declare 'P.step.eq_1', neither of them P.D, and this try did not rebuild P.F (it rebuilds " \
        "once): a clash in the sources" in lines[1], lines


def test_a_refusal_of_the_files_own_build_has_the_sibling_rebuilt_before_it_is_the_verdict(tmp_path):
    """FILE's build is the try's own, so only the sibling is rebuilt; the same refusal then says to rename one, and one
    that names another sibling says that one was not rebuilt."""
    refusal = REFUSAL.format("P.C", "Q.bar", "P.D")
    assert S.named(S.Plan("P.D", True, ["P.C"], []), refusal) == ["P.C"]
    rc, lines, ran, given = checked(tmp_path, [refusal, refusal], (0, "✔ [1/1] Built P.C (1s)\n", 2.0))
    assert (rc, ran, given) == (1, 2, [refusal])
    assert lines == [
        "siblings: P.D and P.C both declare 'Q.bar' as they are built here: Lake rebuilt P.C in 2 s, and the stub is "
        "elaborated once more",
        "siblings: P.D and P.C both declare 'Q.bar', which Lean refuses when it imports both, as the package's root does: "
        f"rename one, and Lake holds P.C current, which this try had it build {COST}"]

    rc, lines, ran, given = checked(tmp_path, [refusal, REFUSAL.format("P.D", "Q.baz", "P.F")], (0, "", 0.3))
    assert (rc, ran, given) == (1, 2, [refusal])
    assert lines == [
        "siblings: P.D and P.C both declare 'Q.bar' as they are built here: Lake rebuilt nothing in 0 s, and the stub is "
        "elaborated once more",
        "siblings: P.D and P.F both declare 'Q.baz', which Lean refuses when it imports both, as the package's root does: "
        "rename one; or P.F, which this try did not rebuild (it rebuilds once), was built against an older build of a module "
        f"it imports, which a second try rebuilds {COST}"]


def test_a_rebuild_that_fails_is_the_verdict_with_lakes_last_line(tmp_path):
    refusal = REFUSAL.format("P.B", "P.step.eq_1", "P.A")
    failed = (1, "✖ [3/4] Building P.B\nerror: P/B.lean:3:0: unknown identifier 'x'\nerror: build failed\n\n", 4.0)
    rc, lines, ran, given = checked(tmp_path, [refusal], failed)
    assert (rc, ran, given) == (1, 1, [refusal]), "the stub is not elaborated again"
    assert lines == ["siblings: P.B and P.A both declare 'P.step.eq_1' as they are built here, and Lake's rebuild of P.B and "
                     "P.A failed (exit 1), its last line: error: build failed; P.D's own names are unchecked"]


def test_nothing_is_rebuilt_where_the_first_stub_names_no_sibling_in_a_refusal_or_no_rebuild_is_given(tmp_path):
    """A check that passes, and a stub that fails on something else, give with `rebuild` what they give without it, and
    never call it. Without `rebuild` a refusal is the verdict as it was before `check` took one: one stub, one line,
    the text unchanged."""
    passed = checked(tmp_path, [""], REBUILT)
    assert passed == (0, [f"siblings: no built module of its package that doesn't import P.D declares a name it does {COST}"], 1, [])
    assert passed == checked(tmp_path, [""])
    missing = "error: object file 'P/A.olean' of module P.A does not exist\n"
    failed = checked(tmp_path, [missing], REBUILT)
    assert failed == (1, [f"siblings: the stub that imports P.D beside them failed (its output above) {COST}"], 1, [])
    assert failed == checked(tmp_path, [missing])

    rc, lines, ran, given = checked(tmp_path, [REFUSAL.format("P.B", "P.step.eq_1", "P.A")])
    assert (rc, ran, given) == (1, 1, [])
    assert lines == [
        "siblings: P.B and P.A both declare 'P.step.eq_1', neither of them P.D: a clash already in the sources, which a root "
        "that imports both refuses too (rename one), since each build the stub loaded is of its source as it is; or one of "
        "them was built against an older build of a module it imports (`lake build +P.B +P.A` and try again); P.D's own "
        f"names are unchecked {COST}"]
    rc, lines, ran, given = checked(tmp_path, [REFUSAL.format("P.C", "Q.bar", "P.D")])
    assert (rc, ran, given) == (1, 1, [])
    assert lines == ["siblings: P.D and P.C both declare 'Q.bar', which Lean refuses when it imports both, as the package's "
                     f"root does: rename one {COST}"]


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
    assert "siblings: P.D and P.C both declare 'Q.bar' as they are built here: Lake rebuilt nothing in " in out, out
    assert "rename one, and Lake holds P.C current, which this try had it build (4 imported" in out, out
    assert not list((pkg / S.COPIES).glob("siblings-*")), "the scratch build is removed"
    write(pkg, {"P/D.lean": dup.replace("def bar", "def baz")})
    assert cli.main(["lean", "try", str(pkg / "P" / "D.lean"), "--siblings"]) == 0
    out = capsys.readouterr().out
    assert "siblings: no built module of its package that doesn't import P.D declares a name it does (4 imported" in out, out
    assert out.splitlines()[-1].startswith("verity lean try: P/D.lean: axiom probe:"), "the probe's line stays last"


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_try_with_siblings_passes_over_products_built_before_names_moved_and_fails_a_clash_in_the_sources(tmp_path, capsys):
    """Built with Lake. `proofs` requires `commit`, which requires `ident`, and all three are built. Then
    `Identity.tag` moves from `Identity.Domains` into the new `Identity.Domains.Pouw`, which `PouwCommit.Bindings` imports
    from then on, and of all that Lake builds only what FILE imports, as a try does on a slot tree synced across the move.
    The products of `Identity.Domains` (which still declares `Identity.tag`), of `PouwCommit.Bindings` and of `Proofs.Old`,
    which imports it, are from before the move, and Lean refuses `Proofs.Old` beside `Identity.Domains.Pouw`: a clash that
    is in no source, on which the try failed while it took another package's products as built. Now `Proofs.Old` and the
    root that imports it are unchecked, and the try passes. A clash that is in the sources still fails it: `Proofs.Dup`,
    built from a source that declares `Identity.tag` again. With that source unlinked and its product left, as `sync`
    leaves one, nothing loads the product. A source changed and written back as it was built from, with no build
    between, leaves its module a sibling. Once Lake has rebuilt `PouwCommit.Bindings`, the stub imports `Proofs.Old`
    again, with the build it had.

    The plan's tests above say which modules the stub names, from traces they write themselves; only this one reads the
    traces Lake writes and loads the builds, so only it shows that the hash is Lake's, that what the plan leaves out is
    what Lean refuses, and that no product from before the move comes in through what it keeps."""
    toolchain = (HERE / "fixtures" / "control" / "lean-toolchain").read_text()
    ident, commit, proofs = tmp_path / "ident", tmp_path / "commit", tmp_path / "proofs"
    tag = "module\n\n@[expose] public section\n\ndef Identity.tag : Nat := 1\n"
    bindings = "module\n\npublic import {}\n@[expose] public section\n\ndef PouwCommit.bound : Nat := Identity.tag\n"
    kept = "module\n\n@[expose] public section\n\ndef Proofs.kept : Nat := 3\n"
    write(ident, {"lakefile.toml": 'name = "ident"\n' + LIB.format("Identity"), "lean-toolchain": toolchain,
                  "Identity/Domains.lean": tag})
    write(commit, {"lakefile.toml": 'name = "commit"\n\n[[require]]\nname = "ident"\npath = "../ident"\n' + LIB.format("PouwCommit"),
                   "lean-toolchain": toolchain, "PouwCommit/Bindings.lean": bindings.format("Identity.Domains")})
    write(proofs, {"lakefile.toml": 'name = "proofs"\ndefaultTargets = ["Proofs"]\n\n[[require]]\nname = "commit"\n'
                                    'path = "../commit"\n' + LIB.format("Proofs"),
                   "lean-toolchain": toolchain, "Proofs.lean": "module\n\npublic import Proofs.Old\npublic import Proofs.Kept\n",
                   "Proofs/Old.lean": "module\n\npublic import PouwCommit.Bindings\n@[expose] public section\n\n"
                                      "theorem Proofs.old : PouwCommit.bound = 1 := rfl\n",
                   "Proofs/Kept.lean": kept})

    def lake(*argv: str) -> subprocess.CompletedProcess:
        return subprocess.run(["lake", *argv], cwd=proofs, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def stated(pkg: Path, mod: str) -> tuple[str | None, str]:
        """(the hash ``mod``'s trace states for its source, the hash of that source as it is)."""
        rel = Path(*mod.split("."))
        return (S.built_from((pkg / S.LIB / rel).with_suffix(".trace"), f"{rel.name}.lean")[0],
                S.lake_hash((pkg / rel).with_suffix(".lean").read_bytes()))

    assert (first := lake("build")).returncode == 0, first.stdout
    assert S.workspace(proofs) == [proofs, commit, ident], "Lake's own manifest lists the package `commit` requires"
    for pkg, mod in ((ident, "Identity.Domains"), (commit, "PouwCommit.Bindings"), (proofs, "Proofs.Old"), (proofs, "Proofs")):
        got, now = stated(pkg, mod)
        assert got == now, f"{mod}: Lake's trace states {got} for its source, and `lake_hash` gives {now}"
    write(ident, {"Identity/Domains.lean": "module\n", "Identity/Domains/Pouw.lean": tag})
    write(commit, {"PouwCommit/Bindings.lean": bindings.format("Identity.Domains.Pouw")})
    file = "module\n\npublic import Identity.Domains.Pouw\n@[expose] public section\n\ntheorem Proofs.file : Identity.tag = 1 := rfl\n"
    write(proofs, {"Proofs/File.lean": file})
    assert (deps := lake("build", "+Proofs.File:deps")).returncode == 0, deps.stdout
    got, now = stated(ident, "Identity.Domains")
    assert (ident / S.LIB / "Identity" / "Domains.olean").is_file() and got is not None and got != now, \
        "no target imports it, so its product stays, from the source it had"
    (tmp_path / "Both.lean").write_text("import Proofs.Old\nimport Identity.Domains.Pouw\n")
    both = lake("env", "lean", str(tmp_path / "Both.lean"))
    assert both.returncode and "environment already contains 'Identity.tag' from Identity.Domains" in both.stdout, both.stdout

    cli, file = _cli(), str(proofs / "Proofs" / "File.lean")
    assert cli.main(["lean", "try", file, "--siblings"]) == 0, capsys.readouterr().out
    out = capsys.readouterr().out
    assert "siblings: no built module of its package that doesn't import Proofs.File declares a name it does (1 imported" in out, out
    assert ("; 2 unchecked, with no build of their source as it is or importing a module with none, in their package or one "
            "it requires: Proofs, Proofs.Old") in out, out

    write(proofs, {"Proofs/Dup.lean": "module\n\n@[expose] public section\n\ndef Identity.tag : Nat := 2\n"})
    assert (dup := lake("build", "+Proofs.Dup")).returncode == 0, dup.stdout
    assert cli.main(["lean", "try", file, "--siblings"]) == 1
    out = capsys.readouterr().out
    assert "both declare 'Identity.tag' as they are built here: Lake rebuilt nothing in " in out, out
    assert ("siblings: Proofs.Dup and Identity.Domains.Pouw both declare 'Identity.tag', neither of them Proofs.File, and "
            "Lake holds both current, which this try had it build: a clash in the sources") in out, out
    (proofs / "Proofs" / "Dup.lean").unlink()
    assert (proofs / S.LIB / "Proofs" / "Dup.olean").is_file()
    assert cli.main(["lean", "try", file, "--siblings"]) == 0, capsys.readouterr().out
    assert "(1 imported" in capsys.readouterr().out, "the product with no source is not a sibling, and nothing loads it"

    write(proofs, {"Proofs/Kept.lean": kept + "\ndef Proofs.more : Nat := 4\n"})
    assert S.plan(proofs, "Proofs/File.lean").siblings == [], "its build is of the source it had"
    write(proofs, {"Proofs/Kept.lean": kept})
    assert S.plan(proofs, "Proofs/File.lean").siblings == ["Proofs.Kept"], "written back as it was built from"

    assert (again := lake("build", "+PouwCommit.Bindings")).returncode == 0, again.stdout
    assert S.plan(proofs, "Proofs/File.lean") == S.Plan("Proofs.File", True, ["Proofs", "Proofs.Kept", "Proofs.Old"], [])
    assert cli.main(["lean", "try", file, "--siblings"]) == 0, capsys.readouterr().out
    assert "(3 imported" in capsys.readouterr().out


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
@pytest.mark.parametrize("how", ["cache", "ltar"])
def test_a_build_lake_took_is_a_sibling_while_lake_holds_it_whatever_the_times_say(tmp_path, monkeypatch, capsys, how):
    """Built with Lake, then taken by Lake into a tree with no build: from its artifact cache (`cache`), or unpacked
    from the archives of a published build (`ltar`), as a node's tree takes main's. Every trace there states one hash
    and no source. Without Lake asked none of the modules is a sibling; with it each is, and the try passes on them.
    Then `P.A`'s text changes, and its trace is made the newer of the two: `P.A` is unchecked, and so are `P.B` and the
    root, which import it and of which Lake says nothing. Written again as it was, later than every build, all are
    siblings again.

    Only this test takes builds with Lake and has Lake say which it holds, so only it shows that `built_from` knows
    the traces and `held` the lines Lake writes for them, and that a stub loads the builds a plan counts on that word."""
    pkg, cache = tmp_path / "tree" / "p", tmp_path / "cache"
    files = {"lakefile.toml": 'name = "p"\ndefaultTargets = ["P"]\n\n[[lean_lib]]\nname = "P"\n',
             "lean-toolchain": (HERE / "fixtures" / "control" / "lean-toolchain").read_text(),
             "P.lean": "module\n\npublic import P.A\npublic import P.B\npublic import P.D\n",
             "P/A.lean": "module\n\n@[expose] public section\n\ndef P.a : Nat := 1\n",
             "P/B.lean": "module\n\npublic import P.A\n@[expose] public section\n\ntheorem P.b : P.a = 1 := rfl\n",
             "P/D.lean": "module\n\n@[expose] public section\n\ndef P.d : Nat := 4\n\n#eval IO.println \"said while it compiled\"\n",
             "P/C.lean": "module\n\n@[expose] public section\n\ndef P.c : Nat := 3\n"}
    monkeypatch.delenv("LAKE_ARTIFACT_CACHE", raising=False)
    monkeypatch.setenv("LAKE_CACHE_DIR", str(cache))

    def lake(at: Path, *argv: str, **env: str) -> str:
        r = subprocess.run(["lake", *argv], cwd=at, env={**os.environ, **env}, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
        assert r.returncode in (0, 3) if "--no-build" in argv else r.returncode == 0, r.stdout
        return r.stdout

    write(pkg, files)
    if how == "cache":
        lake(pkg, "build", LAKE_ARTIFACT_CACHE="true")
        shutil.rmtree(pkg / ".lake" / "build")
    else:
        publisher, out = write(tmp_path / "publisher" / "p", files), tmp_path / "published"
        writable = {"LAKE_ARTIFACT_CACHE": "true", "LAKE_CACHE_DIR": str(out / "lake-cache")}
        lake(publisher, "build", "-o", str(out / "outputs.jsonl"), **writable)
        lake(publisher, "cache", "stage", str(out / "outputs.jsonl"), str(out / "stage"), **writable)
        lake(publisher, "cache", "unstage", str(out / "stage"))
        shutil.copy(publisher / "lake-manifest.json", pkg)
    took = lake(pkg, "build", "-v", "--no-ansi")
    assert "Built" not in took and ("Reused P.A" if how == "cache" else "Unpacked P.A") in took, took
    mods = ["P", "P.A", "P.B", "P.D"]
    for m in mods:
        trace = (pkg / S.LIB).joinpath(*m.split(".")).with_suffix(".trace")
        assert S.built_from(trace, f"{m.split('.')[-1]}.lean") == (None, True), f"{m}: {trace.read_text()}"

    def ask(args: list[str]) -> str:
        return lake(pkg, *args)

    assert S.plan(pkg, "P/C.lean") == S.Plan("P.C", True, [], mods), "no build Lake took counts without Lake's word"
    assert S.plan(pkg, "P/C.lean", ask) == S.Plan("P.C", True, mods, [])
    cli = _cli()
    assert cli.main(["lean", "try", str(pkg / "P" / "C.lean"), "--siblings"]) == 0, capsys.readouterr().out
    out = capsys.readouterr().out
    assert "P/C.lean: Lake holds 4 of the 4 modules that could be siblings as their inputs are now (lake build" in out, out
    assert "siblings: no built module of its package that doesn't import P.C declares a name it does (4 imported" in out, out

    source, trace = pkg / "P" / "A.lean", pkg / S.LIB / "P" / "A.trace"
    write(pkg, {"P/A.lean": files["P/A.lean"] + "\ndef P.more : Nat := 2\n"})
    os.utime(source, (2_000_000_000, 2_000_000_000))
    os.utime(trace, (2_000_000_100, 2_000_000_100))
    assert S.plan(pkg, "P/C.lean", ask) == S.Plan("P.C", True, ["P.D"], ["P", "P.A", "P.B"]), \
        "a build taken for another text, though its trace is the newer"
    write(pkg, {"P/A.lean": files["P/A.lean"]})
    os.utime(source, (2_100_000_000, 2_100_000_000))
    assert S.plan(pkg, "P/C.lean", ask) == S.Plan("P.C", True, mods, []), "written again as it was, later than every build"


DEFS = ("module\n\n@[expose] public section\n\nnamespace P\n\nstructure Trace where\n  log : List Nat\n  out : Nat\n\n"
        "def step : Nat → Trace → Trace\n  | 0, t => { t with out := 0 }\n  | n + 1, t => step n { t with log := n :: t.log }\n")
USER = ("module\n\npublic import P.D\n\n@[expose] public section\n\nnamespace P\n\n"
        "theorem NAME : ∀ t, (step 0 t).out = 0 := by\n  intro t\n  simp [step]\n")
FILE = "module\n\npublic import P.D\n\n@[expose] public section\n\ntheorem P.c : ∀ n : Nat, n = n := fun _ => rfl\n"
DUP = "module\n\n@[expose] public section\n\ntheorem P.a : True := trivial\n"


def left_behind(tmp_path: Path):
    """A package built with Lake in which `P.A` and `P.B` each unfold `P.step` of `P.D` (`simp [step]`), so the build of
    each holds the equation lemma Lean makes for it, the root imports both, and FILE (`P/C.lean`) imports `P.D` alone;
    then `P.D` renames the structure in that lemma's type and Lake builds `P.B` and FILE's imports, as a try does. `P.A`,
    whose source names neither, stays built against the older `P.D`, with the lemma at its older type: (the package,
    `lake` run there)."""
    pkg = tmp_path / "p"
    write(pkg, {"lakefile.toml": 'name = "p"\ndefaultTargets = ["P"]\n\n[[lean_lib]]\nname = "P"\n',
                "lean-toolchain": (HERE / "fixtures" / "control" / "lean-toolchain").read_text(),
                "P.lean": "module\n\npublic import P.A\npublic import P.B\n", "P/D.lean": DEFS,
                "P/A.lean": USER.replace("NAME", "a"), "P/B.lean": USER.replace("NAME", "b"), "P/C.lean": FILE})

    def lake(*argv: str) -> subprocess.CompletedProcess:
        return subprocess.run(["lake", *argv], cwd=pkg, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    assert (first := lake("build")).returncode == 0, first.stdout
    assert S.plan(pkg, "P/C.lean") == S.Plan("P.C", True, ["P", "P.A", "P.B", "P.D"], [])
    write(pkg, {"P/D.lean": DEFS.replace("Trace", "Transcript")})
    assert (some := lake("build", "+P.B", "+P.C:deps")).returncode == 0 and "Built P.B" in some.stdout, some.stdout
    assert "Built P.A" not in some.stdout, "no target imports it, so it stays as it was built"
    assert S.plan(pkg, "P/C.lean") == S.Plan("P.C", True, ["P", "P.A", "P.B", "P.D"], []), \
        "its trace states its source as it is, so it is a sibling still"
    return pkg, lake


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_a_theorem_lean_makes_in_two_modules_is_refused_only_between_builds_lake_holds_current(tmp_path):
    """Built with Lake (`left_behind`). Lean refuses `P.A`, left built against an older build of its import, beside
    `P.B`, in the stub and in the root elaborated on these builds alike, though no source declares a name twice.
    Without `rebuild` that refusal is the verdict, and its line says how to rebuild the two. With it Lake rebuilds `P.A`
    alone, the check passes and the root elaborates. A name two sources declare is refused after the rebuild too, in
    which Lake builds nothing, and the line names both.

    Only this test loads builds Lake made against two versions of an import, so only it shows that the refusal is
    Lean's own on such builds, a root's as much as the stub's, and that Lake's rebuild of what it names ends it."""
    pkg, lake = left_behind(tmp_path)

    def check(rebuild: bool) -> tuple[int, str, list[str], list[str]]:
        """The check of `P/C.lean` as a try makes it: (its status, the stub's last output, its lines, what Lake said)."""
        p, said = S.plan(pkg, "P/C.lean"), []

        def again(output: str) -> tuple[int, str, float]:
            rc, text, seconds, _ = S.run(["lake", "build", *(f"+{m}" for m in S.named(p, output))], pkg)
            said.append(text)
            return rc, text, seconds

        with S.scratch(pkg) as work:
            args = S.lean_args(pkg, p, work)
            S.source(p, work).write_text(FILE)
            assert (made := lake("env", "lean", *args, str(S.source(p, work)))).returncode == 0, made.stdout
            rc, output, lines = S.check(p, work, lambda a: S.run(["lake", "env", "lean", *a], pkg), again if rebuild else None)
        return rc, output, lines.splitlines(), said

    setup = pkg / ".lake" / "build" / "ir" / "P.setup.json"
    root = lake("env", "lean", "--setup", str(setup), str(pkg / "P.lean"))
    assert root.returncode and "import P.B failed, environment already contains 'P.step.eq_" in root.stdout, \
        f"the root itself, on these builds and with Lake's own setup of it: {root.stdout}"
    rc, output, lines, _ = check(rebuild=False)
    assert rc == 1 and "import P.B failed, environment already contains 'P.step.eq_" in output, output
    assert len(lines) == 1 and "siblings: P.B and P.A both declare 'P.step.eq_" in lines[0], lines
    assert "or one of them was built against an older build of a module it imports (`lake build +P.B +P.A` and try again)" \
        in lines[0], lines

    rc, output, lines, said = check(rebuild=True)
    assert rc == 0, (output, lines)
    assert len(said) == 1 and "Built P.A" in said[0] and "Built P.B" not in said[0], f"Lake rebuilds the one it holds stale: {said}"
    assert len(lines) == 2 and "siblings: P.B and P.A both declare 'P.step.eq_" in lines[0], lines
    assert "as they are built here: Lake rebuilt P.A in " in lines[0], lines
    assert "siblings: no built module of its package that doesn't import P.C declares a name it does (4 imported" in lines[1], lines
    assert lake("env", "lean", "--setup", str(setup), str(pkg / "P.lean")).returncode == 0, "and the root elaborates on them"

    write(pkg, {"P/E.lean": DUP})
    assert (dup := lake("build", "+P.E")).returncode == 0, dup.stdout
    rc, output, lines, said = check(rebuild=True)
    assert rc == 1 and len(said) == 1 and "Built" not in said[0], f"one rebuild, in which Lake holds both current: {said}"
    assert len(lines) == 2 and "siblings: P.E and P.A both declare 'P.a' as they are built here: Lake rebuilt nothing in " in lines[0], lines
    assert ("siblings: P.E and P.A both declare 'P.a', neither of them P.C, and Lake holds both current, which this try had "
            "it build: a clash in the sources, which a root that imports both refuses too (rename one)") in lines[1], lines


def test_the_local_try_rebuilds_what_the_stubs_refusal_names_once_and_elaborates_the_stub_once_more(
        tmp_path, monkeypatch, capsys):
    """Against a stand-in `lake` that refuses the stub over `P.B` and `P.A` until it has built them: the local try runs
    `lake build +P.B +P.A` in the package, once, then the stub once more, and prints the rebuild's line before the
    verdict's, as `lean_changed.py --try` does on a node for such a refusal (`test_lean_changed.py`). Before any of it
    the try asks Lake, once, which builds it holds: `P.A`, which Lake took, is a sibling on the line that says so, and
    the try's line counts it."""
    pkg, bin_, log = tmp_path / "p", tmp_path / "bin", tmp_path / "ran"
    write(pkg, {"lakefile.toml": 'name = "p"\n' + LIB.format("P"), "P/A.lean": "module\n", "P/B.lean": "module\n",
                "P/C.lean": "module\n"})
    built(pkg, "P.A", "P.B")
    (pkg / S.LIB / "P" / "A.trace").write_text(TOOK["cache"])
    write(bin_, {"lake": f"""#!/bin/sh
echo "$@" >> "{log}"
case "$1 $2 $3" in
    "env lean --setup")
        [ -e "{tmp_path}/rebuilt" ] && exit 0
        echo "VeritySiblings.lean:1:0: error: import P.B failed, environment already contains 'P.step.eq_1' from P.A"
        exit 1;;
    "build --no-build -v")
        echo "✔ [1/2] Reused P.A"
        echo "✔ [2/2] Replayed P.B"
        exit 0;;
    build\\ *)
        : > "{tmp_path}/rebuilt"
        echo "[3/4] Built P.A (1.0s)";;
esac
exit 0
"""})
    (bin_ / "lake").chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_), prepend=os.pathsep)
    assert _cli().main(["lean", "try", str(pkg / "P" / "C.lean"), "--siblings"]) == 0
    ran = log.read_text().splitlines()
    assert ran[0] == "build --no-build -v --no-ansi +P.A +P.B"
    assert [" ".join(line.split()[:3]) for line in ran[1:]] == [
        "env lean -R", "env lean --setup", "build +P.B +P.A", "env lean --setup"]
    out = capsys.readouterr().out
    assert ("verity lean try: P/C.lean: Lake holds 2 of the 2 modules that could be siblings as their inputs are now "
            "(lake build --no-build -v --no-ansi, ") in out, out
    first, last = [line for line in out.splitlines() if "siblings:" in line]
    assert first.startswith("verity lean try: P/C.lean: siblings: P.B and P.A both declare 'P.step.eq_1' as they are built "
                            "here: Lake rebuilt P.A in ") and first.endswith(" s, and the stub is elaborated once more")
    assert last.startswith("verity lean try: P/C.lean: siblings: no built module of its package that doesn't import P.C "
                           "declares a name it does (2 imported in ")


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lake") is None, reason="no Lean toolchain")
def test_the_local_try_passes_a_tree_where_a_sibling_was_left_built_against_an_older_build_of_its_import(tmp_path, capsys):
    """`left_behind` through `verity lean try --siblings`: it has Lake rebuild `P.A` and passes, and a second try, with
    nothing left to rebuild, says what a try says on builds made together. A name two sources declare still fails it,
    after a rebuild in which Lake builds nothing.

    The test above this one's package shows what the check does given a `rebuild`; only this one shows that the local
    try gives it one that Lake takes."""
    pkg, lake = left_behind(tmp_path)
    cli, file = _cli(), str(pkg / "P" / "C.lean")
    assert cli.main(["lean", "try", file, "--siblings"]) == 0, capsys.readouterr().out
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if "siblings:" in line]
    assert len(lines) == 2 and lines[0].startswith(
        "verity lean try: P/C.lean: siblings: P.B and P.A both declare 'P.step.eq_"), out
    assert "as they are built here: Lake rebuilt P.A in " in lines[0], out
    assert lines[1].startswith("verity lean try: P/C.lean: siblings: no built module of its package that doesn't import P.C "
                               "declares a name it does (4 imported"), out
    assert out.splitlines()[-1].startswith("verity lean try: P/C.lean: axiom probe:"), "the probe's line stays last"
    assert cli.main(["lean", "try", file, "--siblings"]) == 0, capsys.readouterr().out
    assert len([line for line in capsys.readouterr().out.splitlines() if "siblings:" in line]) == 1, "one line, as before"

    write(pkg, {"P/E.lean": DUP})
    assert (dup := lake("build", "+P.E")).returncode == 0, dup.stdout
    assert cli.main(["lean", "try", file, "--siblings"]) == 1
    out = capsys.readouterr().out
    assert "siblings: P.E and P.A both declare 'P.a' as they are built here: Lake rebuilt nothing in " in out, out
    assert ("siblings: P.E and P.A both declare 'P.a', neither of them P.C, and Lake holds both current, which this try had "
            "it build: a clash in the sources") in out, out


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
