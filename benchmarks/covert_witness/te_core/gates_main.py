"""te_core's gate table held to `main`'s, training_slowdown.gates: the same code under the package's new name.

    python -m te_core.gates_main PATH/TO/training_slowdown    # one JSON line; exit 1 where they differ

`main`'s benchmarks/training_slowdown/training_slowdown is te_core's measuring code under its new name. A panel's row
is priced by te_core.gates and a reader of `main` reads training_slowdown.gates, so the two are compared as code: each
of `FILES` parsed on both sides, the package's name made one and docstrings dropped (they name each side's own
modules), so the files agree where every statement does: every Definition's pinned counts, every price, every dtype
and op class, and the functions that read them.
"""
from __future__ import annotations

import ast
import hashlib
import json
import sys
from pathlib import Path

from . import VERSION

HERE = Path(__file__).resolve().parent
#: the gate table and the op names it classes rows by
FILES = ("gates.py", "opnames.py")
OURS, THEIRS = "te_core", "training_slowdown"


def statements(source: str) -> dict[str, str]:
    """A module's top-level statements by the names they bind (a statement that binds none by its position), each as
    its ast's dump, docstrings dropped."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and body
                and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)):
            node.body = body[1:] or [ast.Pass()]
    out = {}
    for i, node in enumerate(tree.body):
        names = ([node.name] if hasattr(node, "name") else
                 sorted(n.id for t in getattr(node, "targets", [getattr(node, "target", None)]) if t is not None
                        for n in ast.walk(t) if isinstance(n, ast.Name)))
        out[", ".join(names) or f"statement {i}"] = out.get(", ".join(names), "") + ast.dump(node)
    return out


def compared(main: Path) -> dict:
    """Each of `FILES` here against `main`'s (the directory of training_slowdown's modules): both files' sha256, and
    the top-level names whose statements differ or that one side lacks."""
    files = {}
    for name in FILES:
        ours, theirs = (HERE / name).read_bytes(), (Path(main) / name).read_bytes()
        a, b = statements(ours.decode()), statements(theirs.decode().replace(THEIRS, OURS))
        files[name] = {OURS: hashlib.sha256(ours).hexdigest(), THEIRS: hashlib.sha256(theirs).hexdigest(),
                       "differ": sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k))}
    init = (Path(main) / "__init__.py").read_text()
    theirs = next((ast.literal_eval(n.value) for n in ast.parse(init).body if isinstance(n, ast.Assign)
                   and [getattr(t, "id", None) for t in n.targets] == ["VERSION"]), None)
    return {"same": not any(f["differ"] for f in files.values()), "versions": {OURS: VERSION, THEIRS: theirs},
            "files": files}


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        raise SystemExit(__doc__.split("\n\n")[1])
    out = compared(Path(args[0]))
    print(json.dumps(out))
    return 0 if out["same"] else 1


if __name__ == "__main__":
    sys.exit(main())
