"""`tools/verity/lean/modules.py`: the port it makes of a plain file (which reads back as a module and which a rerun leaves
alone), the transition options, each package's lakefile option, and which files of a package are its modules."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lean_modules", HERE.parent / "modules.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def test_a_header_lean_can_t_read_is_left_to_a_hand():
    """`headers.py` reads the header; one cut off (a comment left open, an import without a name) is no module and isn't
    ported."""
    assert M.is_module("-- c\nmodule\n\npublic import A\n") and not M.is_module("import A\n")
    for cut in ("/- never closed\nimport A\n", "import A\nimport\n"):
        assert not M.is_module(cut)
        with pytest.raises(ValueError, match="can't be read"):
            M.port(cut)


@pytest.mark.parametrize("src, want", [
    ("def x := 1\n", "module\n\n@[expose] public section\n\ndef x := 1\n"),
    ("", "module\n\n@[expose] public section\n\n"),
    ("import A\nimport all B\nmeta import C\n\ndef x := 1\n",
     "module\n\npublic import A\nimport all B\npublic meta import C\n\n@[expose] public section\n\ndef x := 1\n"),
    ("/- Copyright. -/\nimport A\n/-! Doc. -/\n\nnamespace N\n",
     "/- Copyright. -/\nmodule\n\npublic import A\n/-! Doc. -/\n\n@[expose] public section\n\nnamespace N\n"),
    ("/-! Doc. -/\n\ndef x := 1\n", "module\n\n/-! Doc. -/\n\n@[expose] public section\n\ndef x := 1\n"),
    ("import A", "module\n\npublic import A\n\n@[expose] public section"),
])
def test_a_port_reads_back_as_a_module_and_a_rerun_changes_nothing(src, want):
    assert M.port(src) == want
    assert M.port(want) == want
    h = M.H.header(want)
    assert h.module and all(i.public or i.all for i in h.imports)


def test_a_file_that_runs_code_while_it_compiles_also_meta_imports_what_it_imports():
    src = "import A\nimport all B\nmeta import C\n\n#guard a = 1\n"
    want = "module\n\npublic import A\nimport all B\npublic meta import C\nmeta import A\nmeta import B\n\n" \
           f"{M.SECTION}\n\n#guard a = 1\n"
    assert M.port(src) == want and M.port(want) == want
    for run in ("#eval a", "  #eval! a", "run_cmd pure ()", "#guard_msgs in #eval a"):
        assert "meta import A" in M.port(f"import A\n\n{run}\n"), run
    for quiet in ("#guard_msgs in\nexample : a = 1 := rfl", "#guard_expr a = 1", "def evalA := 1"):
        assert "meta" not in M.port(f"import A\n\n{quiet}\n"), quiet


def test_a_file_that_declares_a_private_definition_reads_it_from_public_bodies():
    """Every definition of a port is exposed, so a public one that reads a private one needs `backward.privateInPublic`."""
    for decl in ("private def h := 1", "@[inline] private abbrev h := 1", "private partial def h : Nat → Nat := id",
                 "  private structure H where"):
        want = (f"module\n\npublic import A\n\n{M.SECTION}\nset_option backward.privateInPublic true\n"
                f"set_option backward.privateInPublic.warn false\n\n{decl}\ndef x := h\n")
        assert M.port(f"import A\n\n{decl}\ndef x := h\n") == want and M.port(want) == want, decl
    for quiet in ("private theorem t : True := trivial", "def private_h := 1", "-- private def h := 1"):
        assert "backward" not in M.port(f"import A\n\n{quiet}\n"), quiet


def test_with_option_adds_a_transition_option_after_the_section_once():
    ported = M.port("import A\n\ndef x := 1\n")
    once = M.with_option(ported, "backward.privateInPublic")
    assert once == ported.replace(M.SECTION, M.SECTION + "\nset_option backward.privateInPublic true"
                                                         "\nset_option backward.privateInPublic.warn false")
    assert M.with_option(once, "backward.privateInPublic") == once
    with pytest.raises(ValueError, match="not ported"):
        M.with_option("import A\n", "backward.privateInPublic")


def test_a_package_keeps_a_public_statement_s_proofs_in_it():
    """`backward.proofsInPublic` goes after the last key of the lakefile's `[leanOptions]`, above a comment that opens the
    next table, once; a lakefile without the table gets one; one whose leanOptions can't take the line is left to a hand."""
    text = 'name = "p"\n\n[leanOptions]\nautoImplicit = false\n\n# its dependency\n[[require]]\nname = "q"\npath = "../q"\n'
    want = text.replace("autoImplicit = false\n", f"autoImplicit = false\n{M.LAKE_OPTION}\n")
    assert M.with_lake_option(text) == want and M.with_lake_option(want) == want
    assert M.has_lake_option(want) and not M.has_lake_option(text)
    assert M.with_lake_option('name = "p"\n[[lean_lib]]\nname = "P"') == \
        f'name = "p"\n[[lean_lib]]\nname = "P"\n\n[leanOptions]\n{M.LAKE_OPTION}\n'
    for odd in ('[leanOptions]\nbackward.proofsInPublic = false\n', 'leanOptions = [{name = "autoImplicit", value = false}]\n',
                'leanOptions = {autoImplicit = false}\n'):
        with pytest.raises(ValueError, match="by hand"):
            M.with_lake_option(odd)


def test_a_package_s_modules_are_its_libraries_and_executable_roots(tmp_path, monkeypatch, capsys):
    """A script beside them (run with `lean --run`), a nested package's files and the controls' fixtures are not; `--check`
    names each file left to port with the command that ports it."""
    monkeypatch.setattr(M, "ROOT", tmp_path)
    files = {"p/lakefile.toml": 'name = "p"\nsrcDir = "src"\n[[lean_lib]]\nname = "P"\n[[lean_exe]]\nname = "run"\nroot = "Main"\n',
             "p/src/P.lean": "import P.A\n", "p/src/P/A.lean": "def a := 1\n", "p/src/Main.lean": "def main : IO Unit := pure ()\n",
             "p/src/Other.lean": "def o := 1\n", "p/scripts/Gen.lean": "def g := 1\n",
             "p/inner/lakefile.toml": 'name = "inner"\n[[lean_lib]]\nname = "I"\n', "p/inner/I.lean": "def i := 1\n",
             M.FIXTURES + "control/lakefile.toml": 'name = "control"\n[[lean_lib]]\nname = "Control"\n',
             M.FIXTURES + "control/Control.lean": "import Control.Defs\n"}
    for rel, text in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)
    monkeypatch.setattr(M, "tracked", lambda: sorted(files))
    assert M.packages(sorted(files)) == ["p", "p/inner"]
    assert M.modules([], sorted(files)) == ["p/inner/I.lean", "p/src/Main.lean", "p/src/P.lean", "p/src/P/A.lean"]
    assert M.main(["--check", "p/src"]) == 1
    assert capsys.readouterr().out.splitlines() == [f"{f}: not a module: run `{M.FIX} {f}`"
                                                    for f in ("p/src/Main.lean", "p/src/P.lean", "p/src/P/A.lean")] + \
        [f"p/lakefile.toml: its [leanOptions] lack `{M.LAKE_OPTION}`: run `{M.FIX} p`"]
    assert M.main(["p"]) == 0 and "4 of 4 module(s) ported, 2 lakefile(s)" in capsys.readouterr().out
    assert M.main(["p"]) == 0 and "0 of 4 module(s) ported, 0 lakefile(s)" in capsys.readouterr().out
    assert M.main(["--check"]) == 0
    assert (tmp_path / "p/lakefile.toml").read_text() == files["p/lakefile.toml"] + f"\n[leanOptions]\n{M.LAKE_OPTION}\n"
    assert (tmp_path / "p/scripts/Gen.lean").read_text() == "def g := 1\n"
    assert (tmp_path / M.FIXTURES / "control/Control.lean").read_text() == "import Control.Defs\n"
    assert (tmp_path / M.FIXTURES / "control/lakefile.toml").read_text() == files[M.FIXTURES + "control/lakefile.toml"]
