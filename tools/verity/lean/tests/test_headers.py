"""`tools/verity/lean/headers.py`: each header form Lean 4.34 accepts is read as Lean reads it, and the header ends where
Lean's does. Lean's verdict on each form here was taken by compiling it against a `module` package."""
from __future__ import annotations

import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("lean_headers_under_test", Path(__file__).resolve().parent.parent / "headers.py")
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)
I = H.Import


def test_a_plain_import():
    assert H.header("import A\nimport B.C\n") == H.Header(False, False, (I("A"), I("B.C")), len("import A\nimport B.C"))


def test_module():
    assert H.header("module\n\nimport A\n") == H.Header(True, False, (I("A"),), len("module\n\nimport A"))
    assert H.header("module -- no imports\n") == H.Header(True, False, (), len("module"))


def test_prelude_after_module():
    assert H.header("module\nprelude\nimport Init.Prelude\n")[:3] == (True, True, (I("Init.Prelude"),))
    assert H.header("prelude\nimport Init.Core\n")[:3] == (False, True, (I("Init.Core"),))


def test_public_import():
    assert H.header("module\npublic import A\n").imports == (I("A", public=True),)


def test_meta_import():
    assert H.header("module\nmeta import A\n").imports == (I("A", meta=True),)


def test_public_meta_import():
    assert H.header("module\npublic meta import A\n").imports == (I("A", public=True, meta=True),)


def test_import_all():
    assert H.header("module\nimport all A\n").imports == (I("A", all=True),)
    assert H.header("module\nmeta import all A\n").imports == (I("A", meta=True, all=True),)


def test_public_import_all_is_read_and_left_to_lean_to_refuse():
    assert H.header("module\npublic import all A\n").imports == (I("A", public=True, all=True),)
    assert H.header("module\npublic meta import all A\n").imports == (I("A", True, True, True),)


def test_the_words_of_an_import_go_in_lean_s_order():
    assert H.header("module\nimport A\nmeta public import B\n") == H.Header(True, False, (I("A"),), len("module\nimport A"))
    assert H.imports("module\npublic public import A\n") == []
    assert H.imports("module\nprivate import A\n") == []


def test_prelude_before_module_ends_the_header():
    assert H.header("prelude\nmodule\nimport Init.Prelude\n") == H.Header(False, True, (), len("prelude"))
    assert H.header("module\nmodule\nimport A\n") == H.Header(True, False, (), len("module"))


def test_comments_and_blank_lines_before_and_among_imports():
    src = "/- a /- nested -/ comment -/\n-- a line\n\nmodule -- here\n\n-- among\npublic   meta\n  import -- x\n  all\n A\n" \
          "/- between -/ import «B».«C» import D -- two on a line\n\n"
    assert H.header(src) == H.Header(True, False, (I("A", True, True, True), I("B.C"), I("D")), src.index(" -- two"))


def test_a_doc_comment_ends_the_header():
    for doc in ("/-! A module doc. -/", "/-- A declaration's doc. -/"):
        assert H.header(f"{doc}\nmodule\nimport A\n") == H.Header(False, False, (), 0)
        assert H.header(f"module\nimport A\n{doc}\nimport B\n") == H.Header(True, False, (I("A"),), len("module\nimport A"))


def test_the_first_other_token_ends_the_header():
    for rest in ("public section\n", "@[expose] public section\n", "public def x := 1\n", "meta def x := 1\n",
                 "def x := 1\nimport Z\n", "import_x\n"):
        h = H.header(f"module\npublic import A\n\n{rest}")
        assert h == H.Header(True, False, (I("A", public=True),), len("module\npublic import A")), rest
    assert H.header("") == H.Header(False, False, (), 0)


def test_a_tab_or_a_byte_order_mark_is_no_space():
    assert H.imports("\ufeffimport A\n") == []
    assert H.imports("import A\n\timport B\n") == ["A"]


def test_a_header_cut_off_has_none():
    for src in ("/- never closed\nimport A\n", "import A\nimport", "import all", "module\nimport all\n",
                "import import\n", "module\npublic import -- no name\n", "import /-- doc -/ A\n",
                "module\nimport ModSys.\n", "import .A\n", "import A..B\n"):
        assert H.header(src) is None and H.imports(src) is None, src


def test_names_lose_their_quotes():
    assert H.imports("import «A».B.«c d»\n") == ["A.B.c d"]
    assert H.parts("«A».B.«c.d»") == ["A", "B", "c.d"] and H.name("«A».B") == "A.B"


def test_each_import_starts_at_its_first_word():
    """Where the codemod (`modules.py`) writes `public ` before an import."""
    src = "/- c -/\nmodule\n\npublic import A\nmeta import B -- why\n/- x -/ import all C import D\n\ndef x := 1\n"
    assert [src[a:].split("\n")[0] for a in H.starts(src)] == [
        "public import A", "meta import B -- why", "import all C import D", "import D"]
    assert H.starts("def x := 1\n") == () and H.starts("import\n") is None
