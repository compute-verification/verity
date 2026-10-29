"""The tree's retired terms and what to write instead, as `tests/test_retired_terms.py` states them: `TERMS`, retired
everywhere but the Lean, and `DOC_TERMS`, retired in the maintained docs' prose (GLOSSARY.md's older names). `research msg`
warns when a post uses one (`retired`).

The table has one home, the test, which every lane's rename edits; this reads its assignments from the checkout rather
than keeping a second copy. A checkout without it, or a table this can't read, flags nothing: a post is never held on it."""
from __future__ import annotations

import ast
import re
from functools import cache
from pathlib import Path

TABLE = "tests/test_retired_terms.py"
#: The table's names this reads, each an assignment of literals, `re` calls and the names before it.
NAMES = ("TERMS", "_SEP", "_TERM", "DOC_TERMS", "_SPACE", "_DOC_TERM", "_CODE")


def table_path() -> Path | None:
    here = Path(__file__).resolve()
    return next((d / TABLE for d in (Path.cwd(), *Path.cwd().parents, *here.parents) if (d / TABLE).is_file()), None)


@cache
def table(path: Path | None = None) -> dict:
    """The table's NAMES as the test file assigns them, or {} when it can't be read."""
    path = path or table_path()
    try:
        tree = ast.parse(path.read_text()) if path else None
    except (OSError, SyntaxError, ValueError):
        return {}
    ns: dict = {"re": re}
    for node in tree.body if tree else []:
        target = node.targets[0] if isinstance(node, ast.Assign) and len(node.targets) == 1 else getattr(node, "target", None)
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and isinstance(target, ast.Name) and target.id in NAMES and node.value:
            try:
                exec(compile(ast.fix_missing_locations(ast.Module([ast.Assign([target], node.value)], [])), str(path), "exec"), ns)
            except Exception:                                     # a form this reader doesn't know: skip that name
                continue
    return {k: ns[k] for k in NAMES if k in ns}


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
