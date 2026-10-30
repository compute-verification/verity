"""Direct environment reads in every tracked Python file outside ``integrations/vllm/`` and ``archive/``, held to
``allowlists/repo_env_reads.json``: S1 of the quality rollout, so no area adds one
while its owner builds the settings registry that declares each variable once.

The integration has its own settings rule (P7, ``integrations/vllm/tests/lint/test_p07_declared_inputs.py``) and ``archive/``
is kept but not developed, so neither is held here (``LEFT_OUT``).  A read is ``os.environ[...]``, ``os.environ.get(...)``,
``os.environ.setdefault(...)``, ``os.getenv(...)``, a membership test (``"X" in os.environ``), and ``os.environ.pop(...)``
whose value is used, through any alias (``from os import environ``, ``import os as o``, ``env = os.environ``), ``os.environb``
and ``os.getenvb`` alike.  A write, a delete, and passing ``os.environ`` whole (to a subprocess, ``dict(...)``, ``.copy()``)
are not.  Reads are counted per file, the key the ratchet gate counts by too: an entry ``{"file", "count"}`` allows ``count``
reads in its file.  A file read more often than its entry allows, or read with no entry, fails.  An entry that allows more
than is found doesn't: the list can run ahead of a tree, holding the reads of PRs queued below it on ``next``, and ``--prune``
deletes and lowers what this tree no longer finds.

An area's settings registry is where reads belong, so it is exempt: a module named ``settings.py``, or one the list marks
``{"file": PATH, "registry": true}``.  A file that moves takes its entries with it, in the same change (``--move``): this lint
reads no history, so a renamed file's reads count under its new path.  The ratchet gate (``integrations/vllm/tools/
ratchet_gate.py``) refuses a change whose Python gains a read, or that grows this list or leaves a renamed file's entries
behind, and judges a change to this module by running it as the change's base has it on the change's tree.  So it imports only
the standard library, ``test_no_new_reads`` takes no fixture, and the gate reads ``LEFT_OUT``, ``FIX``, ``reads``, ``parse``
and ``is_registry`` from the base's copy of this module.

    python3 tests/lint/test_repo_env_reads.py --move OLD NEW   # a file (or directory) moved: its entries follow it
    python3 tests/lint/test_repo_env_reads.py --prune          # delete and lower what this tree no longer finds
"""
from __future__ import annotations

import ast
import bisect
import json
import re
import subprocess
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path, PurePosixPath

NAME = "repo_env_reads"
ROOT = Path(__file__).resolve().parents[2]
LIST = Path(__file__).resolve().parent / "allowlists" / f"{NAME}.json"
#: the integration's own settings rule holds it, and the archive isn't developed
LEFT_OUT = ("integrations/vllm/", "archive/")
#: a settings registry's module name: reads belong there
REGISTRY = "settings.py"
TEMPLATE = "integrations/vllm/verity_vllm/settings.py"
FIX = (f"Read the variable through your area's settings registry, which declares each one once (under verity/, verity/settings.py, "
       f"the distribution's one; the template: {TEMPLATE}, with integrations/vllm/tests/test_settings.py and "
       f"integrations/vllm/tests/lint/test_p07_declared_inputs.py), or create one: a module named `{REGISTRY}`, or one "
       f"tests/lint/allowlists/{NAME}.json marks "
       f'{{"file": PATH, "registry": true}}. If the file moved, move its entries in the same change: '
       f"`python3 tests/lint/test_{NAME}.py --move OLD NEW`.")
ENVIRON = frozenset({"os.environ", "os.environb"})
GETENV = frozenset({"os.getenv", "os.getenvb"})
#: the attributes of ``os`` that read the environment; a line naming none of them, nor an alias of one, is not walked
ATTRS = frozenset(name.removeprefix("os.") for name in ENVIRON | GETENV)
NAMES = re.compile(r"\b(?:environb?|getenvb?)\b")
#: on a line that names the environment, a name it may bind to it: an import's ``as X``, or ``X = <...>.environ``
ALIAS = re.compile(r"\bas\s+(\w+)|^\s*(\w+)\s*(?::[^=]*)?=\s*(?:[\w.]*\.)?environb?\s*(?:#.*)?$")
SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)


def _named(node: ast.AST | None, form: str) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
        name = node.value if isinstance(node.value, str) else node.value.decode(errors="replace")
        return f"{name} ({form})"
    return f"<computed> ({form})"


def reads(source: str) -> list[tuple[int, str]]:
    """(line, what) of each direct environment read in ``source``: ``what`` is the variable, when it is a literal, and the
    form (``HF_HOME (os.environ.get)``, ``<computed> (os.environ[...])``).  A file that doesn't parse counts one, at its
    error, since its reads can't be seen."""
    if "environ" not in source and "getenv" not in source:
        return []
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError) as e:
        return [(getattr(e, "lineno", None) or 0, f"<unparseable: {getattr(e, 'msg', e)}>")]
    lines = source.splitlines()
    aliases = {w for line in lines if NAMES.search(line) for m in ALIAS.finditer(line) for w in m.groups() if w}
    hot_line = re.compile(NAMES.pattern + "".join(f"|\\b{re.escape(w)}\\b" for w in sorted(aliases)))
    hot = [n for n, line in enumerate(lines, 1) if hot_line.search(line)]

    def warm(node: ast.stmt) -> bool:
        """Whether ``node``'s lines hold one that names the environment or an alias of it: only those are walked."""
        i = bisect.bisect_left(hot, min([node.lineno, *(d.lineno for d in getattr(node, "decorator_list", ()))]))
        return i < len(hot) and hot[i] <= (node.end_lineno or node.lineno)

    imported: dict[str, str] = {"os": "os"}
    parent: dict[int, ast.AST] = {}
    scope: dict[int, ast.AST] = {}
    assigns: list[ast.Assign | ast.AnnAssign] = []
    candidates: list[ast.Name | ast.Attribute] = []
    stack: list[tuple[ast.AST, ast.AST]] = [(tree, tree)]
    while stack:
        node, inside = stack.pop()
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "os" or a.name.startswith("os."):
                    imported[a.asname or "os"] = a.name if a.asname else "os"
            continue
        if isinstance(node, ast.ImportFrom):
            if node.module == "os" and not node.level:
                imported.update({a.asname or a.name: f"os.{a.name}" for a in node.names})
            continue
        if isinstance(node, ast.stmt) and not warm(node):
            continue
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            assigns.append(node)
        elif isinstance(node, ast.Name) or isinstance(node, ast.Attribute) and node.attr in ATTRS:
            candidates.append(node)
        for child in ast.iter_child_nodes(node):
            parent[id(child)], scope[id(child)] = node, inside
            stack.append((child, child if isinstance(child, SCOPES) else inside))
    bound: dict[int, set[str]] = {}

    def origin(node: ast.AST) -> str | None:
        if isinstance(node, ast.Name):
            if node.id in imported:
                return imported[node.id]
            at = scope.get(id(node))
            while at is not None:
                if node.id in bound.get(id(at), ()):
                    return "os.environ"
                at = scope.get(id(at))
            return None
        if isinstance(node, ast.Attribute):
            base = origin(node.value)
            return f"{base}.{node.attr}" if base else None
        return None

    for node in assigns:
        if node.value is not None and origin(node.value) in ENVIRON:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            bound.setdefault(id(scope[id(node)]), set()).update(t.id for t in targets if isinstance(t, ast.Name))
    out = []
    for node in candidates:
        o = origin(node)
        if o is None:
            continue
        up = parent.get(id(node))
        if o in GETENV:
            call = up if isinstance(up, ast.Call) and up.func is node else None
            out.append((node.lineno, _named(call.args[0] if call and call.args else None, o)))
        elif o not in ENVIRON:
            continue
        elif isinstance(up, ast.Subscript) and up.value is node and isinstance(up.ctx, ast.Load):
            out.append((node.lineno, _named(up.slice, f"{o}[...]")))
        elif isinstance(up, ast.Attribute) and up.value is node and up.attr in ("get", "setdefault", "pop"):
            call = parent.get(id(up))
            call = call if isinstance(call, ast.Call) and call.func is up else None
            if up.attr != "pop" or call is not None and not isinstance(parent.get(id(call)), ast.Expr):
                out.append((node.lineno, _named(call.args[0] if call and call.args else None, f"{o}.{up.attr}")))
        elif isinstance(up, ast.Compare):
            i = next((i for i, c in enumerate(up.comparators) if c is node), None)
            if i is not None and isinstance(up.ops[i], (ast.In, ast.NotIn)):
                out.append((node.lineno, _named(up.comparators[i - 1] if i else up.left, f"in {o}")))
    return sorted(out)


def parse(doc: object) -> tuple[Counter, frozenset[str]]:
    """(each file's allowed reads, the files marked settings registries) of a list's document."""
    assert isinstance(doc, dict) and isinstance(doc.get("entries"), list), f"{LIST.name}: {{'entries': [...]}} expected"
    counts: Counter = Counter()
    marks: set[str] = set()
    for e in doc["entries"]:
        if isinstance(e, dict) and set(e) == {"file", "registry"}:
            assert e["registry"] is True and isinstance(e["file"], str), f"{LIST.name}: a mark is {{file, registry: true}}: {e}"
            assert e["file"] not in marks, f"{LIST.name}: duplicate mark {e}"
            marks.add(e["file"])
            continue
        assert isinstance(e, dict) and set(e) == {"file", "count"} and isinstance(e["file"], str), \
            f"{LIST.name}: an entry is {{file, count}} or {{file, registry: true}}: {e}"
        assert type(e["count"]) is int and e["count"] >= 1, f"{LIST.name}: count must be a positive int: {e}"
        assert e["file"] not in counts, f"{LIST.name}: duplicate entry {e}"
        counts[e["file"]] = e["count"]
    return counts, frozenset(marks)


def is_registry(path: str, marks: frozenset[str]) -> bool:
    return PurePosixPath(path).name == REGISTRY or path in marks


def files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z", "--", "*.py"], cwd=ROOT, capture_output=True, check=True).stdout
    return sorted(p for p in out.decode().split("\0") if p and not p.startswith(LEFT_OUT))


@lru_cache(maxsize=None)
def scan() -> tuple[tuple[str, int, str], ...]:
    """Each read's file, line and what it reads."""
    return tuple((f, line, what) for f in files() if (ROOT / f).is_file()
                 for line, what in reads((ROOT / f).read_text(encoding="utf-8", errors="replace")))


def allowed() -> tuple[Counter, frozenset[str]]:
    return parse(json.loads(LIST.read_text(encoding="utf-8")))


def found(marks: frozenset[str]) -> Counter:
    return Counter(f for f, *_ in scan() if not is_registry(f, marks))


def test_no_new_reads() -> None:
    have, marks = allowed()
    sites: dict[str, list[tuple[int, str]]] = {}
    for f, line, what in scan():
        sites.setdefault(f, []).append((line, what))
    new = [f"  {f}: {n} found, {have[f]} allowed\n" + "\n".join(f"    {f}:{line} {what}" for line, what in sites[f])
           for f, n in sorted(found(marks).items()) if n > have[f]]
    assert not new, (f"direct environment reads: {len(new)} file(s) read the environment more often than "
                     f"tests/lint/allowlists/{NAME}.json allows. {FIX}\n" + "\n".join(new))


def test_what_is_a_read_and_what_is_not() -> None:
    src = (
        "import os\n"
        "import os as o\n"
        "from os import environ, getenv as ge\n"
        "A = os.environ['A']\n"
        "B = os.environ.get('B', '')\n"
        "C = o.getenv('C')\n"
        "D = environ.setdefault('D', '1')\n"
        "E = ge('E')\n"
        "if 'F' in os.environ and k not in environ:\n"
        "    G = os.environ.pop('G')\n"
        "def f():\n"
        "    env = os.environ\n"
        "    return env.get('H'), os.environb[b'I']\n"
        "def g(env):\n"
        "    return env.get('NOT_ENV')\n"
        "os.environ['W'] = '1'\n"
        "del os.environ['W']\n"
        "os.environ.pop('W', None)\n"
        "os.environ.update(X='1')\n"
        "subprocess.run(cmd, env=os.environ)\n"
        "copy = dict(os.environ); copy2 = os.environ.copy(); copy2.get('NOT_ENV')\n"
        "'''os.environ.get(\"DOC\")'''\n"
    )
    assert reads(src) == [
        (4, "A (os.environ[...])"), (5, "B (os.environ.get)"), (6, "C (os.getenv)"), (7, "D (os.environ.setdefault)"),
        (8, "E (os.getenv)"), (9, "<computed> (in os.environ)"), (9, "F (in os.environ)"), (10, "G (os.environ.pop)"),
        (13, "H (os.environ.get)"), (13, "I (os.environb[...])")]
    assert reads("import os\nos.environ.get(\n") == [(2, "<unparseable: '(' was never closed>")]
    assert reads("X = (\n") == [] and reads("X = 1\n") == []


def test_a_settings_registry_is_exempt_and_the_list_is_well_formed() -> None:
    marks = frozenset({"tools/x/config.py"})
    assert is_registry("tools/x/settings.py", frozenset()) and is_registry("tools/x/config.py", marks)
    assert not is_registry("tools/x/config.py", frozenset()) and not is_registry("tools/x/my_settings.py", marks)
    doc = {"entries": [{"file": "a.py", "count": 2}, {"file": "tools/x/config.py", "registry": True}]}
    assert parse(doc) == (Counter({"a.py": 2}), marks)
    for bad in ({"file": "a.py", "count": 0}, {"file": "a.py", "count": True}, {"file": "a.py", "registry": False},
                {"file": "a.py", "count": 1, "rule": "x"}):
        try:
            parse({"entries": [bad]})
        except AssertionError as e:
            assert str(e).startswith(f"{LIST.name}: "), e
        else:
            raise AssertionError(f"{bad} parsed")
    assert moved(Counter({"a/x.py": 2, "b.py": 1}), frozenset({"a/settings_x.py"}), "a", "c") == (
        Counter({"c/x.py": 2, "b.py": 1}), frozenset({"c/settings_x.py"}))


def moved(counts: Counter, marks: frozenset[str], old: str, new: str) -> tuple[Counter, frozenset[str]]:
    """``counts`` and ``marks`` with the file ``old``'s, or the directory ``old``'s files', under ``new``: a count already
    there adds."""
    old, new = old.rstrip("/"), new.rstrip("/")

    def at(f: str) -> str:
        return new + f[len(old):] if f == old or f.startswith(f"{old}/") else f

    out: Counter = Counter()
    for f, n in counts.items():
        out[at(f)] += n
    return out, frozenset(map(at, marks))


def dump(counts: Counter, marks: frozenset[str]) -> None:
    rows = sorted([(f, 0, {"file": f, "count": n}) for f, n in counts.items() if n > 0]
                  + [(f, 1, {"file": f, "registry": True}) for f in marks])
    lines = [f"    {json.dumps(e, ensure_ascii=False)}" for *_, e in rows]
    LIST.write_text('{\n  "rule": "direct environment reads (tests/lint/test_repo_env_reads.py)",\n  "entries": [\n'
                    + ",\n".join(lines) + "\n  ]\n}\n", encoding="utf-8")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--move"] and len(sys.argv) == 4:
        have, marks = allowed()
        after = moved(have, marks, *sys.argv[2:])
        dump(*after)
        print(f"{sum(1 for f in have if f not in after[0]) + len(marks - after[1])} entries moved from {sys.argv[2]} to "
              f"{sys.argv[3]}")
    elif sys.argv[1:] == ["--prune"]:
        have, marks = allowed()
        tracked = set(files())
        dump(have & found(marks), frozenset(m for m in marks if m in tracked))
    else:
        sys.exit(__doc__.rsplit("\n\n", 1)[-1])
