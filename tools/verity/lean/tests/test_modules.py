"""`tools/verity/lean/modules.py`: the header reader, the port it makes of a plain file (which reads back as a module and
which a rerun leaves alone), the transition options, and which files of a package are its modules."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lean_modules", HERE.parent / "modules.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def imports(src: str) -> list[tuple[tuple[str, ...], str]]:
    return [(i.words, i.module) for i in M.header(src).imports]


def test_a_header_is_read_as_lean_reads_it():
    src = ("\ufeff-- a comment\n/- a /- nested -/ comment -/\nmodule\n\npublic import A.B\nmeta import A.Tac  -- why\n"
           "public meta import A.Elab\nimport all A.Open\n/- between -/ import «A.Odd name»\nimport C\n\n"
           "/-! The module docstring ends the header. -/\n\nimport D\n")
    h = M.header(src)
    assert h.module and src[h.first:].startswith("module\n")
    assert imports(src) == [(("public",), "A.B"), (("meta",), "A.Tac"), (("public", "meta"), "A.Elab"), (("all",), "A.Open"),
                            ((), "«A.Odd name»"), ((), "C")]
    assert src[:h.end].endswith("import C") and src[h.body:].startswith("/-! The module docstring")
    assert M.header("prelude\nimport Init.Core\ndef x := 1\n").imports[0].module == "Init.Core"
    plain = M.header("/-- A docstring is a command. -/\ndef x := 1\n")
    assert not plain.module and plain.imports == () and plain.end == plain.first == plain.body == 0


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
    h = M.header(want)
    assert h.module and all("public" in i.words or "all" in i.words for i in h.imports)


def test_with_option_adds_a_transition_option_after_the_section_once():
    ported = M.port("import A\n\ndef x := 1\n")
    once = M.with_option(ported, "backward.privateInPublic")
    assert once == ported.replace(M.SECTION, M.SECTION + "\nset_option backward.privateInPublic true")
    assert M.with_option(once, "backward.privateInPublic") == once
    with pytest.raises(ValueError, match="not ported"):
        M.with_option("import A\n", "backward.proofsInPublic")


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
                                                    for f in ("p/src/Main.lean", "p/src/P.lean", "p/src/P/A.lean")]
    assert M.main(["p"]) == 0 and "4 of 4 module(s) ported" in capsys.readouterr().out
    assert M.main(["p"]) == 0 and "0 of 4 module(s) ported" in capsys.readouterr().out
    assert M.main(["--check"]) == 0
    assert (tmp_path / "p/scripts/Gen.lean").read_text() == "def g := 1\n"
    assert (tmp_path / M.FIXTURES / "control/Control.lean").read_text() == "import Control.Defs\n"
