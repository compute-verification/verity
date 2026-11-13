"""The sibling check (`siblings.py`) that `verity lean try --siblings` runs once FILE elaborates: which modules its stub
imports, and, with a real toolchain, that a module redeclaring a name another built module of its package declares fails
the try naming both, as the package's root fails a records run an hour in. Its stub loads no build that is not of its
source as it is, by the hash Lake's trace states, in the package or in one it requires by path, and none without a
source: with a real toolchain, a try passes over the products a tree keeps from before names moved between modules
(INF-667), and still fails a clash that is in the sources. And the package's Lean options every try takes from it: with
a real toolchain, an auto-bound name fails a try where the package sets `autoImplicit = false`."""
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
    """INF-667's tree, (`ident`, `commit`, `proofs`): everything was built, and then a sync brought a move of
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
    """INF-667 (`moved`). `Proofs.Old`'s source never changed, and its build loads `PouwCommit.Bindings` and through it
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


@pytest.mark.parametrize("took", ['{"synthetic": true, "schemaVersion": "2025-09-10", "outputs": {}, "log": [], "inputs": [], '
                                  '"depHash": "b3c1e3c25b4816c6"}',
                                  '{"schemaVersion":"2025-09-10","depHash":"255a6a20482e9767","outputs":{}}'], ids=["cache", "ltar"])
def test_a_module_lake_took_rather_than_built_counts_until_its_source_is_written_again(tmp_path, took):
    """Lake writes a trace with no input for a module it finds in its artifact cache, and `leantar` one for a module
    unpacked from its `.ltar`, as a node's records runs leave most of a slot tree, the definitions everything imports
    among them: there is no source hash to compare. Lake took it for the source as it was then, so it is a sibling while
    its trace is at least as new as its source, and unchecked, with what imports it, once a sync has written the source
    again. The `.olean` itself is a link into the cache, touched whenever the cache is read: its time says nothing."""
    pkg, then = tmp_path / "p", 2_000_000_000.0
    write(pkg, {"lakefile.toml": 'name = "p"\n' + LIB.format("P"), "P/File.lean": "module\n", "P/A.lean": "module\n",
                "P/B.lean": "module\nimport P.A\n"})
    built(pkg, "P.A", "P.B")
    (trace := pkg / S.LIB / "P" / "A.trace").write_text(took)
    os.utime(pkg / "P" / "A.lean", (then, then))
    os.utime(trace, (then, then))
    os.utime(pkg / S.LIB / "P" / "A.olean", (then - 500, then - 500))
    assert S.plan(pkg, "P/File.lean") == S.Plan("P.File", True, ["P.A", "P.B"], [])
    os.utime(pkg / "P" / "A.lean", (then + 100, then + 100))
    os.utime(pkg / S.LIB / "P" / "A.olean", (then + 500, then + 500))
    assert S.plan(pkg, "P/File.lean") == S.Plan("P.File", True, [], ["P.A", "P.B"])
    built(pkg, "P.A")
    assert S.plan(pkg, "P/File.lean").unchecked == [], "built from the source as it is"


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
    assert ("siblings: Proofs.Dup and Identity.Domains.Pouw both declare 'Identity.tag', neither of them Proofs.File: a clash "
            "already in the sources") in out, out
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
