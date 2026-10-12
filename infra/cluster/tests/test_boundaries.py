"""The cluster tooling stays stdlib-only: it imports neither Verity nor the research tool, so it runs on a bare node and
can't break when either changes."""
import ast
import sys
from pathlib import Path

SRC = Path(__file__).parents[1] / "src" / "cluster"


def test_imports_are_stdlib_or_own():
    allowed = set(sys.stdlib_module_names) | {"cluster"}
    for path in SRC.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else \
                [node.module] if isinstance(node, ast.ImportFrom) and node.level == 0 else []
            for name in names:
                assert name.split(".")[0] in allowed, f"{path.name} imports {name}"
