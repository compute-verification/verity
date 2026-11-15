"""Each module of `verity_vllm.engine` that states `__all__` states it once, as a literal list of names, lists no name twice
and binds every one of them at its top level: a def, a class, an assignment or an import, also one under a top-level ``if``,
``try`` or ``with``.  The modules are read as source and none is loaded, so those that need CUDA, triton or vLLM to load
(`int8_fused`, `int8_vllm`) are read here as the rest are.  `tests/test_vllm_public_api.py` holds the modules it lists as
public to more than this; the engine's other modules it holds to nothing."""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

import verity_vllm.engine

ENGINE = Path(verity_vllm.engine.__file__).parent
MODULES = sorted(p.relative_to(ENGINE).as_posix() for p in ENGINE.rglob("*.py"))
NOT_LOADED_ON_A_CPU = ("int8_fused.py", "int8_vllm.py")


def _stated(tree: ast.Module) -> list[ast.expr | None]:
    return [n.value for n in tree.body if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign))
            for t in (n.targets if isinstance(n, ast.Assign) else [n.target]) if isinstance(t, ast.Name) and t.id == "__all__"]


def _bound(nodes: list[ast.stmt]) -> set[str]:
    out: set[str] = set()
    for n in nodes:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, (ast.Assign, ast.AnnAssign)):
            for t in n.targets if isinstance(n, ast.Assign) else [n.target]:
                out |= {x.id for x in ast.walk(t) if isinstance(x, ast.Name)}
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            out |= {(a.asname or a.name).split(".")[0] for a in n.names}
        elif isinstance(n, ast.If):
            out |= _bound(n.body) | _bound(n.orelse)
        elif isinstance(n, ast.Try):
            out |= _bound(n.body) | _bound(n.orelse) | _bound(n.finalbody) | {x for h in n.handlers for x in _bound(h.body)}
        elif isinstance(n, ast.With):
            out |= _bound(n.body)
    return out


def faults(text: str) -> list[str]:
    """What is wrong with the `__all__` a module's source states; nothing where it states none."""
    tree = ast.parse(text)
    faces = _stated(tree)
    if not faces:
        return []
    if len(faces) > 1:
        return [f"states __all__ {len(faces)} times"]
    try:
        names = ast.literal_eval(faces[0]) if faces[0] is not None else None
    except ValueError:
        names = None
    if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
        return ["__all__ is not a literal list of names"]
    have = _bound(tree.body)
    return ([f"lists {n!r} twice" for n in sorted({n for n in names if names.count(n) > 1})]
            + [f"lists {n!r}, which its top level does not bind" for n in names if n not in have])


@pytest.mark.parametrize("module", MODULES)
def test_a_module_binds_each_name_its_face_lists(module: str) -> None:
    assert faults((ENGINE / module).read_text(encoding="utf-8")) == []


@pytest.mark.parametrize("module", NOT_LOADED_ON_A_CPU)
def test_the_modules_no_cpu_loads_state_a_face_and_are_read(module: str) -> None:
    assert module in MODULES
    assert len(_stated(ast.parse((ENGINE / module).read_text(encoding="utf-8")))) == 1


@pytest.mark.parametrize("text, found", [
    ("x = 1\n", []),
    ("__all__ = ['f', 'K', 'm', 'n', 'T']\nimport m\nfrom p import q as n\nK = 1\ndef f(): ...\nclass T: ...\n", []),
    (("__all__ = ['a', 'b', 'c', 'd', 'e']\nif K:\n    a = 1\nelse:\n    b = 2\ntry:\n    import c\nexcept ImportError:\n    c = None\n"
      "else:\n    d = 1\nwith open(p) as h:\n    e = h.read()\n"), []),
    ("__all__ = ['f', 'g']\ndef f(): ...\n", ["lists 'g', which its top level does not bind"]),
    ("__all__ = ['g']\ndef f():\n    g = 1\n", ["lists 'g', which its top level does not bind"]),
    ("__all__ = ['g']\nclass C:\n    def g(self): ...\n", ["lists 'g', which its top level does not bind"]),
    ("__all__ = ['f', 'f']\ndef f(): ...\n", ["lists 'f' twice"]),
    ("__all__ = ['f']\n__all__ = ['f']\ndef f(): ...\n", ["states __all__ 2 times"]),
    ("__all__ = ['f']\n__all__ += ['g']\ndef f(): ...\ndef g(): ...\n", ["states __all__ 2 times"]),
    ("__all__ = [n for n in ('f',)]\ndef f(): ...\n", ["__all__ is not a literal list of names"]),
    ("__all__ = ('f',)\ndef f(): ...\n", ["__all__ is not a literal list of names"]),
    ("__all__ = ['f', 1]\ndef f(): ...\n", ["__all__ is not a literal list of names"]),
    ("__all__: list[str]\ndef f(): ...\n", ["__all__ is not a literal list of names"]),
])
def test_the_reader_names_each_fault(text: str, found: list[str]) -> None:
    assert faults(text) == found
