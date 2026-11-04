"""No Python file of the vLLM integration imports `research`, the ops tool that runs and records jobs, which lives outside
this repository: the integration runs it only as a command (`research data ...`, `python -m research`).  Every `*.py` under
integrations/vllm is read with `ast` (library, tests, tools and examples alike), importing nothing.  Exact: no allowlist."""
from __future__ import annotations

import ast
from pathlib import Path

from . import _ratchet as R

#: generated output, environments and caches, never the integration's code
SKIP = frozenset({"out", ".venv", "__pycache__"})
LOADERS = frozenset({"importlib.import_module", "import_module", "__import__"})
FIX = ("Take what the integration needs from the standard library or as an argument, run the store as a command "
       "(`verity_vllm.pipeline.store_data.data`), and keep ops-only code beside its callers, outside integrations/vllm.")


def research_imports(tree: ast.AST) -> list[tuple[int, str]]:
    """(line, module) of each import of the package `research`: `import`, `from ... import`, and `importlib.import_module` or
    `__import__` with a literal name."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(node.lineno, a.name) for a in node.names]
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            found.append((node.lineno, node.module))
        elif isinstance(node, ast.Call) and (R.dotted(node.func) or "") in LOADERS and node.args \
                and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
            found.append((node.lineno, node.args[0].value))
    return sorted((line, m) for line, m in found if m.split(".")[0] == "research")


def files() -> list[Path]:
    return [p for p in sorted(R.ROOT.rglob("*.py")) if not SKIP & set(p.relative_to(R.ROOT).parts[:-1])]


def test_no_file_imports_research() -> None:
    found = [f"  {p.relative_to(R.ROOT).as_posix()}:{line} {m}" for p in files()
             for line, m in research_imports(ast.parse(p.read_text(encoding="utf-8"), str(p)))]
    assert not found, f"{len(found)} import(s) of `research` in integrations/vllm. {FIX}\n" + "\n".join(found)


def test_every_form_of_import_is_found_and_the_integrations_own_research_module_is_not() -> None:
    src = ("import research\nimport research.store as S\nfrom research.store.tool import Tool\nimport importlib\n"
           "importlib.import_module('research.clone')\n__import__('research')\nfrom verity_vllm.pipeline import research\n"
           "import researcher\nfrom . import research\nsubprocess.run([py, '-c', 'import research'])\n")
    assert research_imports(ast.parse(src)) == [(1, "research"), (2, "research.store"), (3, "research.store.tool"),
                                                (5, "research.clone"), (6, "research")]
    assert R.ROOT / "verity_vllm" / "pipeline" / "source_identity.py" in files()
    assert not any("out" in p.relative_to(R.ROOT).parts[:-1] for p in files())
