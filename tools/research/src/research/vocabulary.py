"""The tree's retired terms and what to write instead, as `tests/test_retired_terms.py` states them: `TERMS`, retired
everywhere but the Lean, and `DOC_TERMS`, retired in the maintained docs' prose (GLOSSARY.md's older names). `research msg`
warns when a post uses one (`retired`).

The table has one home, the test, which every lane's rename edits; this reads its assignments from the checkout rather
than keeping a second copy. A checkout without it, or a table this can't read (`table()["error"]`), flags nothing: a post is
never held on it."""
from __future__ import annotations

import ast
import re
from functools import cache
from pathlib import Path

TABLE = "tests/test_retired_terms.py"
#: The table's names this reads, each an assignment of literals, f-strings, `re.compile` and the names before it.
NAMES = ("TERMS", "_SEP", "_TERM", "DOC_TERMS", "_SPACE", "_DOC_TERM", "_CODE")


def table_path() -> Path | None:
    here = Path(__file__).resolve()
    return next((d / TABLE for d in (Path.cwd(), *Path.cwd().parents, *here.parents) if (d / TABLE).is_file()), None)


@cache
def table(path: Path | None = None) -> dict:
    """The table's NAMES as the test file assigns them, `skipped` naming any it couldn't evaluate; {"error": why} when it
    can't be read, which flags nothing."""
    path = path or table_path()
    try:
        tree = ast.parse(path.read_text()) if path else None
    except (OSError, SyntaxError, ValueError) as e:
        return {"error": f"{path}: {e}"}
    ns: dict = {}
    for node in tree.body if tree else []:
        target = node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else getattr(node, "target", None)
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(target, ast.Name) and target.id in NAMES and node.value:
            try:
                ns[target.id] = _value(node.value, ns)
            except (ValueError, KeyError, TypeError, re.error) as e:   # a form this reader doesn't know: skip that name
                ns.setdefault("skipped", []).append(f"{target.id}: {e}")
    return ns


_FLAGS = {"I": re.I, "M": re.M, "S": re.S, "X": re.X, "IGNORECASE": re.I, "MULTILINE": re.M, "DOTALL": re.S}


def _value(e: ast.expr, ns: dict) -> object:
    """An assignment's value: literals, f-strings over the names before it, `re.compile` and `re`'s flags joined by `|`."""
    if isinstance(e, ast.Name):
        return ns[e.id]
    if isinstance(e, ast.JoinedStr):
        return "".join(str(_value(v.value, ns)) if isinstance(v, ast.FormattedValue) else str(v.value) for v in e.values)
    if isinstance(e, ast.Attribute) and isinstance(e.value, ast.Name) and e.value.id == "re" and e.attr in _FLAGS:
        return _FLAGS[e.attr]
    if isinstance(e, ast.BinOp) and isinstance(e.op, ast.BitOr):
        return _value(e.left, ns) | _value(e.right, ns)
    if (isinstance(e, ast.Call) and isinstance(e.func, ast.Attribute) and e.func.attr == "compile"
            and isinstance(e.func.value, ast.Name) and e.func.value.id == "re" and not e.keywords):
        return re.compile(*(_value(a, ns) for a in e.args))
    return ast.literal_eval(e)


def retired(text: str, t: dict | None = None) -> list[tuple[str, str]]:
    """(the term as written, what to write instead) for each retired term in `text`'s prose: code spans and fences, and a term
    written as a quoted mention ("leg"), are skipped, as the docs' scan skips them."""
    t = table() if t is None else t
    code, patterns = t.get("_CODE"), [t[k] for k in ("_TERM", "_DOC_TERM") if isinstance(t.get(k), re.Pattern)]
    prose = code.sub(lambda m: re.sub(r"[^\n]", " ", m[0]), text) if isinstance(code, re.Pattern) else text
    out: list[tuple[str, str]] = []
    for m in (m for p in patterns for m in p.finditer(prose)):
        if prose[m.start() - 1:m.start()] in ('"', "\u201c") and prose[m.end():m.end() + 1] in ('"', "\u201d"):
            continue
        said = " ".join(m[0].replace("-", " ").split())
        if all(said.lower() != s.lower() for s, _ in out):
            out.append((said, instead(said, t)))
    return out


def instead(term: str, t: dict | None = None) -> str:
    """What to write for a retired `term` as written: the entry of TERMS or DOC_TERMS that names it."""
    t, said = table() if t is None else t, term.lower()
    for names, write in [*t.get("TERMS", {}).items(), *t.get("DOC_TERMS", {}).items()]:
        for name in (n.strip().lower() for n in names.split(",")):
            if name and (name in said or said in name or name.startswith(said.rstrip("s"))):
                return write
    return "see GLOSSARY.md's older names"
