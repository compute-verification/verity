"""`code_digest` and `ref_vocab_digest`: a comment, a docstring or a sorted import moves neither; a changed constant moves both."""
from __future__ import annotations

import ast
import inspect

import pytest

from verity_vllm.definitions import code_digest as C
from verity_vllm.definitions.registry import prims
from verity_vllm.definitions.registry import ref_prims as R

PKG = "verity_vllm.definitions.registry"


def comment(src: str) -> str:
    lines = src.splitlines(keepends=True)
    at = next(i for i, ln in enumerate(lines) if ln.lstrip().startswith(("def ", "return ")))
    indent = lines[at][: len(lines[at]) - len(lines[at].lstrip())]
    return "".join(lines[:at]) + f"{indent}# a comment\n" + "".join(lines[at:]) + "\n# a closing comment\n"


def docstring(src: str) -> str:
    """Every docstring rewritten."""
    lines = src.splitlines(keepends=True)
    docs = [n.body[0] for n in ast.walk(ast.parse(src))
            if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.body
            and isinstance(n.body[0], ast.Expr) and isinstance(n.body[0].value, ast.Constant) and isinstance(n.body[0].value.value, str)]
    assert docs, "no docstring to rewrite"
    for d in sorted(docs, key=lambda d: -d.lineno):
        lines[d.lineno - 1:d.end_lineno] = [" " * d.col_offset + '"""Another docstring."""\n']
    return "".join(lines)


def sorted_imports(src: str, package: str = PKG) -> str:
    """Each top-level `from M import a, b` of attributes split into one statement a name, in reverse order: what an import
    sort may do where the order means nothing."""
    lines = src.splitlines(keepends=True)
    done = 0
    for s in sorted(ast.parse(src).body, key=lambda s: -s.lineno):
        if not (isinstance(s, ast.ImportFrom) and len(s.names) > 1 and s.level == 0):
            continue
        if any(C._may_be_submodule(s.module, a.name) for a in s.names):
            continue
        lines[s.lineno - 1:s.end_lineno] = [ast.unparse(ast.ImportFrom(s.module, [a], 0)) + "\n" for a in reversed(s.names)]
        done += 1
    assert done, "no import to sort"
    return "".join(lines)


def constant(src: str) -> str:
    """The first integer constant one more."""
    lines = src.splitlines(keepends=True)
    c = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Constant) and type(n.value) is int)
    assert c.lineno == c.end_lineno
    ln = lines[c.lineno - 1]
    lines[c.lineno - 1] = ln[:c.col_offset] + str(c.value + 1) + ln[c.end_col_offset:]
    return "".join(lines)


@pytest.mark.parametrize("module", [R, prims], ids=["ref_prims", "prims"])
@pytest.mark.parametrize("edit", [comment, docstring, sorted_imports], ids=["comment", "docstring", "sorted-imports"])
def test_a_modules_code_digest_holds_under_edits_that_compute_nothing(module, edit):
    src = inspect.getsource(module)
    edited = edit(src)
    assert edited != src
    assert C.source_digest(edited, PKG) == C.source_digest(src, PKG) == C.code_digest(module)


@pytest.mark.parametrize("module", [R, prims], ids=["ref_prims", "prims"])
def test_a_modules_code_digest_moves_with_a_constant(module):
    src = inspect.getsource(module)
    assert C.source_digest(constant(src), PKG) != C.code_digest(module)


SYNTHETIC = '''"""A module."""
from __future__ import annotations

import math

from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
from verity_vllm.definitions.registry import b1, prims

SCALE = 3


def f(x):
    """Doc."""
    return math.floor(x * SCALE)  # rounds down
'''


@pytest.mark.parametrize("edited, holds", [
    (SYNTHETIC.replace("from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind",
                       "from verity.circuits.defs import (bind,\n    PrimitiveDefinition)\nfrom verity.circuits.defs import CompositeDefinition"),
     True),
    (SYNTHETIC.replace("import math\n\nfrom", "import math\nfrom"), True),
    (SYNTHETIC.replace("import math\n\nfrom verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind",
                       "from verity.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind\nimport math"), False),
    (SYNTHETIC.replace("import b1, prims", "import prims, b1"), False),
    (SYNTHETIC.replace("import b1, prims", "import b1, prims as P"), False),
], ids=["names-split-and-reordered", "layout", "modules-reordered", "submodules-reordered", "binding-renamed"])
def test_an_import_run_is_its_first_imports_in_order_and_its_bindings(edited, holds):
    """Reordering the modules a run imports moves the digest: a first import runs the module's body (a registration)."""
    assert edited != SYNTHETIC
    assert (C.source_digest(edited, PKG) == C.source_digest(SYNTHETIC, PKG)) is holds


def test_a_source_that_does_not_parse_is_refused():
    with pytest.raises(ValueError, match="does not parse"):
        C.source_digest("lambda x: x,)")


def _ref_vocab_digest_with(monkeypatch, obj, src: str) -> str:
    real = inspect.getsource
    monkeypatch.setattr(inspect, "getsource", lambda o: src if o is obj else real(o))
    try:
        return R.ref_vocab_digest()
    finally:
        monkeypatch.setattr(inspect, "getsource", real)


@pytest.mark.parametrize("target", ["module", "evaluator"])
def test_ref_vocab_digest_holds_under_comments_docstrings_and_sorted_imports_and_moves_with_a_constant(monkeypatch, target):
    obj = R if target == "module" else prims.F32Add.evaluate
    src = inspect.getsource(obj)
    before = R.ref_vocab_digest()
    edits = [comment, docstring] + ([sorted_imports] if target == "module" else [])
    for edit in edits:
        assert _ref_vocab_digest_with(monkeypatch, obj, edit(src)) == before, edit.__name__
    assert _ref_vocab_digest_with(monkeypatch, obj, constant(src)) != before
