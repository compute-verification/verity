"""cut.py ROOT TABLE.json [--report FILE] [--dry]: move top-level definitions between Python files by name, on any base.

The table (JSON) names:

* ``modules``: each new module, with the definitions it takes from each file by name (``take``), the short name other
  files import it under (``alias``) and its docstring (``doc``); a module that stays beside a file it takes from may
  read what stays of it (``reads_source``) and end in text of its own (``tail``, with the ``imports`` it reads);
* ``whole``: the files that move whole, old path to new;
* ``edits``: exact replacements, each of which must match its file once (what no rule here writes: a command's parser,
  a list a test holds the tree to), a new module's text as the cut wrote it among them;
* ``files``: new files given whole;
* ``bare``: the directory whose files import each other by a bare name;
* ``own``: for ``allow.py``, the sites of each counted lint the cut may add of its own, with why.

A definition is its statement, its decorators and the comment lines right above it. Nothing in a definition's body
changes but the name it reads another moved definition by, and what an ``edits`` row names. Every tracked Python file that reads a moved definition,
by ``alias.NAME`` or ``from module import NAME``, through whatever re-exports, reads it from its new module after;
the file a definition left imports back the ones it still reads, and loses the imports only they used.

The script edits ROOT's working tree and commits nothing. It refuses, changing nothing and naming the fix, where a
name of the table is not a top-level definition of its file, where two definitions of one name meet in a module, where
a moved definition reads one that stays, where a module under ``verity/`` would import a file of ``bare``, and where
an edit does not match once. It imports and runs none of the tree's code. ``allow.py``, run after it, moves the
counted lints' entries with the findings the definitions took along."""
from __future__ import annotations

import ast
import builtins
import io
import json
import re
import subprocess
import symtable
import sys
import tokenize
from pathlib import Path

ROOT = Path(sys.argv[1]).resolve()
TABLE = json.loads(Path(sys.argv[2]).read_text())
REPORT = Path(sys.argv[sys.argv.index("--report") + 1]) if "--report" in sys.argv else None
DRY = "--dry" in sys.argv
DEFS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
BARE = TABLE["bare"]
WIDTH = 120
problems: list[str] = []
notes: list[str] = []


def dotted(rel: str) -> str:
    return rel[:-3].replace("/", ".")


def globals_of(table: symtable.SymbolTable) -> set[str]:
    out = {s.get_name() for s in table.get_symbols() if s.is_global() and s.is_referenced()}
    for child in table.get_children():
        out |= globals_of(child)
    return out


def loads(node: ast.AST | None) -> set[str]:
    return set() if node is None else {x.id for x in ast.walk(node) if isinstance(x, ast.Name) and isinstance(x.ctx, ast.Load)}


def _parts(node: ast.AST) -> list[str] | None:
    """[a, b, c] of the dotted name `a.b.c`."""
    out: list[str] = []
    while isinstance(node, ast.Attribute):
        out.append(node.attr)
        node = node.value
    return [node.id, *reversed(out)] if isinstance(node, ast.Name) else None


def _renames(stmt: ast.Assign) -> list[tuple[str, list[str]]]:
    """(NAME, [a, b, c]) for `NAME = a.b.c`, and for each pair of `N1, N2 = a.b, c.d`."""
    target, value = stmt.targets[0], stmt.value
    if isinstance(target, ast.Name):
        return [(target.id, parts)] if (parts := _parts(value)) else []
    if isinstance(target, ast.Tuple) and isinstance(value, ast.Tuple) and len(target.elts) == len(value.elts):
        return [(t.id, parts) for t, v in zip(target.elts, value.elts) if isinstance(t, ast.Name) and (parts := _parts(v))]
    return []


class File:
    """One file's text, parsed: its top-level definitions, its import statements and what each binds where."""

    def __init__(self, rel: str, text: str) -> None:
        self.rel, self.text = rel, text
        self.lines = text.split("\n")
        self.tree = ast.parse(text, filename=rel)
        self.top: dict[str, ast.stmt] = {}
        for n in self.tree.body:
            if isinstance(n, DEFS):
                self.top.setdefault(n.name, n)
            elif isinstance(n, (ast.Assign, ast.AnnAssign)):
                for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                    for x in ast.walk(t):
                        if isinstance(x, ast.Name):
                            self.top.setdefault(x.id, n)
        # every import statement with the function it is in (None at module level), innermost
        self.imports: list[tuple[ast.stmt, ast.AST | None]] = []
        self.functions: list[ast.AST] = []
        self.calls: list[tuple[ast.Assign, ast.AST | None]] = []  # NAME = f(...), where f may load a file by its path
        #: NAME = a.b.c, alone or in a tuple: another name for a module, perhaps; (statement, NAME, [a, b, c], scope)
        self.renames: list[tuple[ast.Assign, str, list[str], ast.AST | None]] = []

        def walk(node: ast.AST, inside: ast.AST | None) -> None:
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.Import, ast.ImportFrom)):
                    self.imports.append((child, inside))
                elif isinstance(child, ast.Assign) and len(child.targets) == 1 and isinstance(child.targets[0], ast.Name) \
                        and isinstance(child.value, ast.Call) and (isinstance(child.value.func, ast.Name)
                                                                   or getattr(child.value.func, "attr", "") == "import_module"):
                    self.calls.append((child, inside))
                elif isinstance(child, ast.Assign) and len(child.targets) == 1 and (pairs := _renames(child)):
                    self.renames += [(child, name, parts, inside) for name, parts in pairs]
                elif isinstance(child, SCOPES):
                    self.functions.append(child)
                    walk(child, child)
                else:
                    walk(child, inside)
        walk(self.tree, None)
        self._table: symtable.SymbolTable | None = None
        self._chain: dict[int, list[ast.AST | None]] = {}
        self._bound: dict[tuple[str, int], tuple | None] = {}
        self._loaded: dict[tuple[str, int], tuple | None] = {}
        #: the names an import or a call's result binds anywhere: only these start a read of another file
        self.roots = {a.asname or (a.name.split(".")[0] if isinstance(s, ast.Import) else a.name)
                      for s, _ in self.imports for a in s.names} | {s.targets[0].id for s, _ in self.calls}
        grew = True
        while grew:  # and the names given to what one of them holds
            grew = False
            for _, name, parts, _ in self.renames:
                if parts[0] in self.roots and name not in self.roots:
                    self.roots.add(name)
                    grew = True

    def scopes(self) -> dict[tuple[str, int], symtable.SymbolTable]:
        if self._table is None:
            self._table = symtable.symtable(self.text, self.rel, "exec")
            self._scopes = {(c.get_name(), c.get_lineno()): c for c in self._table.get_children()}
        return self._scopes

    def chain(self, line: int) -> list[ast.AST | None]:
        """The functions around `line`, innermost first, then the module (None)."""
        if line not in self._chain:
            around = [f for f in self.functions if f.lineno <= line <= f.end_lineno]
            self._chain[line] = [*sorted(around, key=lambda f: f.end_lineno - f.lineno), None]
        return self._chain[line]

    def bound(self, name: str, line: int) -> tuple[ast.stmt, ast.alias, ast.AST | None] | None:
        """The import statement that binds `name` for a read at `line`: the innermost scope's, the last one there."""
        key = (name, id(self.chain(line)[0]))
        if key not in self._bound:
            self._bound[key] = None
            for scope in self.chain(line):
                hits = [(s, a) for s, inside in self.imports if inside is scope for a in s.names
                        if (a.asname or (a.name.split(".")[0] if isinstance(s, ast.Import) else a.name)) == name]
                if hits:
                    self._bound[key] = (*hits[-1], scope)
                    break
        return self._bound[key]

    def loads_file(self, stmt: ast.Assign) -> str | None:
        """The tracked file `NAME = f(...)` loads: by its path, where `f` is a function of this file that does that
        (`spec_from_file_location` over a path that ends in a constant, or in its argument and ``.py``), or by the name
        `importlib.import_module` or `__import__` is given as a constant."""
        func, args = stmt.value.func, stmt.value.args
        if getattr(func, "attr", "") == "import_module" or getattr(func, "id", "") == "__import__":  # by a name it spells
            named = args and isinstance(args[0], ast.Constant) and isinstance(args[0].value, str)
            return module_file(args[0].value, self.rel) if named else None
        f = self.top.get(func.id)
        if not isinstance(f, ast.FunctionDef):
            return None
        for call in ast.walk(f):
            if isinstance(call, ast.Call) and getattr(call.func, "attr", "") == "spec_from_file_location" and len(call.args) > 1:
                for x in ast.walk(call.args[1]):
                    path = None
                    if isinstance(x, ast.Constant) and isinstance(x.value, str) and x.value.endswith(".py"):
                        path = x.value
                    elif isinstance(x, ast.JoinedStr) and len(x.values) == 2 and getattr(x.values[1], "value", "") == ".py" \
                            and stmt.value.args and isinstance(stmt.value.args[0], ast.Constant):
                        path = f"{stmt.value.args[0].value}.py"
                    if path:
                        hits = [r for r in TEXT if r == path or r.endswith("/" + path)]
                        near = [r for r in hits if r in (path, f"{BARE}/{path}", str(Path(self.rel).parent.parent / path))]
                        if len(near or hits) == 1:
                            return (near or hits)[0]
                for x in args:  # the path as the call's own argument
                    if isinstance(x, ast.Constant) and isinstance(x.value, str) and x.value.endswith(".py"):
                        hits = [r for r in TEXT if r == x.value or r.endswith("/" + x.value)]
                        if len(hits) == 1:
                            return hits[0]
        return None

    def loaded(self, name: str, line: int) -> tuple[ast.Assign, str, ast.AST | None] | None:
        """The assignment that binds `name` to a file loaded by its path, for a read at `line`."""
        key = (name, id(self.chain(line)[0]))
        if key not in self._loaded:
            self._loaded[key] = None
            for scope in self.chain(line):
                hit = next((stmt for stmt, inside in reversed(self.calls) if inside is scope and stmt.targets[0].id == name), None)
                if hit is not None:
                    rel = self.loads_file(hit)
                    self._loaded[key] = (hit, rel, scope) if rel else None
                    break
        return self._loaded[key]

    def renamed(self, name: str, line: int) -> tuple[ast.Assign, list[str], ast.AST | None] | None:
        """The assignment `name = a.b.c` that binds `name` for a read at `line`, with the dotted name's parts."""
        for scope in self.chain(line):
            hit = next(((s, parts, inside) for s, n, parts, inside in reversed(self.renames)
                        if inside is scope and n == name), None)
            if hit is not None:
                return hit
        return None

    def span(self, node: ast.stmt) -> tuple[int, int]:
        """The 1-based lines of `node`, its decorators and the comment lines right above it."""
        start = min([node.lineno, *(d.lineno for d in getattr(node, "decorator_list", []))])
        while start > 1 and self.lines[start - 2].startswith("#"):
            start -= 1
        return start, node.end_lineno

    def reads(self, node: ast.stmt) -> set[str]:
        """The module-scope names the top-level statement `node` reads."""
        if not isinstance(node, DEFS):
            return loads(node)
        table = self.scopes().get((node.name, node.lineno)) or next(
            (t for (name, _), t in self.scopes().items() if name == node.name), None)
        names = globals_of(table) if table is not None else set()
        parts: list[ast.AST | None] = [*node.decorator_list, *getattr(node, "bases", [])]
        for f in ast.walk(node):  # annotations and defaults are read where the function is defined
            if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)):
                a = f.args
                parts += [*a.defaults, *a.kw_defaults, f.returns,
                          *(x.annotation for x in (*a.args, *a.kwonlyargs, *a.posonlyargs, a.vararg, a.kwarg) if x)]
            elif isinstance(f, ast.AnnAssign):
                parts.append(f.annotation)
        if isinstance(node, ast.ClassDef):
            parts += [s for s in node.body if not isinstance(s, DEFS)]
        local = set() if table is None else {s.get_name() for s in table.get_symbols() if s.is_local()}
        for part in parts:
            names |= loads(part) - (local if isinstance(node, ast.ClassDef) else set())
        return {n for n in names if n in self.top or self.bound(n, node.lineno) or not hasattr(builtins, n)}

    def module_reads(self) -> set[str]:
        """Every module-scope name the file reads anywhere."""
        out: set[str] = set()
        for n in self.tree.body:
            out |= self.reads(n) if isinstance(n, DEFS) else loads(n)
        return out


TRACKED = [p for p in subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", "--", "*.py"], check=True,
                                     capture_output=True).stdout.decode().split("\0") if p and (ROOT / p).is_file()]
TEXT = {rel: (ROOT / rel).read_text(encoding="utf-8") for rel in TRACKED}
BY_NAME: dict[str, list[str]] = {}
for _rel in TRACKED:
    BY_NAME.setdefault(Path(_rel).name, []).append(_rel)
_files: dict[str, File | None] = {}


def file(rel: str) -> File | None:
    """The tree's file `rel` as it was before the cut, or None where it doesn't parse."""
    if rel not in _files:
        try:
            _files[rel] = File(rel, TEXT[rel])
        except SyntaxError:
            _files[rel] = None
    return _files[rel]


def module_file(mod: str, at: str) -> str | None:
    """The tracked file a module name written in the file `at` is: by its path, or for a bare name the file beside
    `at`, else the one tracked file of that name where it is in `bare`."""
    if "." in mod:
        rel = mod.replace(".", "/") + ".py"
        return rel if rel in TEXT else None
    beside = str(Path(at).parent / (mod + ".py"))
    if beside in TEXT:
        return beside
    return f"{BARE}/{mod}.py" if BY_NAME.get(mod + ".py") == [f"{BARE}/{mod}.py"] else None


def alias_target(stmt: ast.stmt, a: ast.alias, at: str) -> tuple[str, str | None] | None:
    """What an import's alias binds: (the module's file, None), or (the module's file, the name it takes of it)."""
    if isinstance(stmt, ast.Import):
        rel = module_file(a.name, at)
        return (rel, None) if rel and (a.asname or "." not in a.name) else None
    if stmt.level or not stmt.module:
        return None
    sub = module_file(f"{stmt.module}.{a.name}", at)
    if sub:
        return sub, None
    rel = module_file(stmt.module, at)
    return (rel, a.name) if rel else None


def resolve(rel: str, name: str, seen: frozenset = frozenset()) -> tuple[str, str | None] | None:
    """Where the attribute `name` of the module `rel` is defined: (file, name), or (file, None) for a module."""
    f = file(rel)
    if f is None or (rel, name) in seen:
        return None
    if name in f.top:
        got = f.loaded(name, 10 ** 9)
        if got:
            return got[1], None
        given = f.renamed(name, 10 ** 9)
        if given is not None and given[2] is None:  # another name for a module, perhaps
            cur = resolve(rel, given[1][0], seen | {(rel, name)}) if given[1][0] != name else None
            for attr in given[1][1:]:
                cur = resolve(cur[0], attr, seen | {(rel, name)}) if cur is not None and cur[1] is None else None
            if cur is not None and cur[1] is None:
                return cur
        return rel, name
    for stmt, inside in reversed(f.imports):
        if inside is None:
            for a in stmt.names:
                local = a.asname or (a.name.split(".")[0] if isinstance(stmt, ast.Import) else a.name)
                if local == name:
                    target = alias_target(stmt, a, rel)
                    if target is None or target[1] is None:
                        return target
                    return resolve(target[0], target[1], seen | {(rel, name)})
    return None


# ---- the table: where each definition goes
where: dict[tuple[str, str], str] = {}
ALIAS = {dest: m["alias"] for dest, m in TABLE["modules"].items()}
WHOLE: dict[str, str] = TABLE.get("whole", {})
for dest, m in TABLE["modules"].items():
    taken: dict[str, str] = {}
    for rel, names in m["take"].items():
        f = file(rel) if rel in TEXT else None
        if f is None:
            problems.append(f"{rel}: not a tracked Python file (the table's `take` of {dest})")
            continue
        for name in names:
            if name not in f.top:
                at = [r for r in TRACKED if r.startswith(("benchmarks/network_traces/", "verity/protocols/accounting/certifier/"))
                      and re.search(rf"(?m)^(def|class) {re.escape(name)}\b|^{re.escape(name)}\b.*=", TEXT[r])]
                problems.append(f"{rel}: no top-level definition `{name}`" + (f"; one is in {', '.join(at)}" if at else "")
                                + ": fix the table's `take`")
            if name in taken:
                problems.append(f"{dest}: `{name}` comes from both {taken[name]} and {rel}: give one another module")
            if (rel, name) in where:
                problems.append(f"{rel}: `{name}` is taken twice, by {where[(rel, name)]} and {dest}")
            taken[name], where[(rel, name)] = rel, dest
        for name in names:
            others = [n for n, stmt in f.top.items() if name in f.top and stmt is f.top[name] and (rel, n) not in where]
            if others:
                problems.append(f"{rel}: `{name}` is defined in one statement with {', '.join(f'`{n}`' for n in others)}: "
                                f"take them too, in {dest}")
for rel, dest in WHOLE.items():
    if rel not in TEXT:
        problems.append(f"{rel}: not a tracked Python file (the table's `whole`, to {dest})"
                        + (": the tree has it there already" if dest in TEXT else ""))
if problems:  # the table doesn't fit this tree, so nothing below would read it right
    sys.exit("cut.py refuses, and changed nothing:\n- " + "\n- ".join(dict.fromkeys(problems)))
SOURCES = sorted({rel for rel, _ in where})
MOVED = {name for _, name in where}


def place(rel: str) -> str:
    return WHOLE.get(rel, rel)


ROOTED_AS = BARE.replace("/", ".")
_own = {Path(rel).stem for rel in TEXT if str(Path(rel).parent) == BARE}
#: the modules of `bare` that some file imports from the root: the tree lets a module have one name, so these have that one
ROOTED: set[str] = {Path(dest).stem for dest, m in TABLE["modules"].items() if m.get("rooted")}
for _rel, _text in TEXT.items():
    if ROOTED_AS in _text:
        ROOTED |= set(re.findall(rf"""["']{re.escape(ROOTED_AS)}\.(\w+)["']""", _text))
        for _stmt, _ in (file(_rel).imports if file(_rel) else []):
            for _a in _stmt.names:
                _mod = _a.name if isinstance(_stmt, ast.Import) else f"{_stmt.module}.{_a.name}"
                if _mod.startswith(ROOTED_AS + "."):
                    ROOTED.add(_mod[len(ROOTED_AS) + 1:].split(".")[0])
ROOTED &= _own | {Path(dest).stem for dest in TABLE["modules"]}
#: the files outside `bare` that import a module of it by its bare name: they have the directory on their path
_bare_import = re.compile(rf"(?m)^[ \t]*(?:import|from)[ \t]+({'|'.join(map(re.escape, sorted(_own)))})\b")
BARE_READERS = {rel for rel, text in TEXT.items() if not rel.startswith(BARE + "/")
                and any(module_file(stem, rel) == f"{BARE}/{stem}.py" for stem in set(_bare_import.findall(text)))}
NAMED: dict[str, dict[bool, set[str]]] = {}  # each module of `bare` an import is written for: bare or not -> the files


def import_of(dest: str, at: str, name: str | None = None, asname: str | None = None) -> tuple[str, str, str | None]:
    """(module, name, asname) of the `from module import name [as asname]` that reads `dest`, or a name of it, in `at`.
    A module of `bare` is read by its bare name where no file reads it from the root and `at` has the directory on its
    path (a file of it, or one that imports another of its modules so)."""
    bare = str(Path(dest).parent) == BARE and Path(dest).stem not in ROOTED \
        and (at.startswith(BARE + "/") or at in BARE_READERS)
    if str(Path(dest).parent) == BARE:
        NAMED.setdefault(Path(dest).stem, {}).setdefault(bare, set()).add(at)
    if name is None:
        stem = Path(dest).stem
        return ("" if bare else dotted(dest).rsplit(".", 1)[0]), stem, (asname if asname != stem else None)
    return (Path(dest).stem if bare else dotted(dest)), name, (asname if asname != name else None)


def render(indent: str, group: list[tuple[str, str, str | None]], comment: str) -> list[str]:
    """The import statements for `group`, one a module, wrapped at the width."""
    out: list[str] = []
    by: dict[str, list[str]] = {}
    for module, name, asname in group:
        by.setdefault(module, [])
        item = name + (f" as {asname}" if asname else "")
        if item not in by[module]:
            by[module].append(item)
    for module, items in by.items():
        if module == "":
            out += [f"{indent}import {item}{comment}" for item in items]
            continue
        line = f"{indent}from {module} import {', '.join(items)}"
        if len(line + comment) <= WIDTH:
            out.append(line + comment)
            continue
        if comment:  # a comment holds for its own line only, so each line is a statement of its own
            row: list[str] = []
            for item in items:
                if row and len(f"{indent}from {module} import {', '.join([*row, item])}{comment}") > WIDTH:
                    out.append(f"{indent}from {module} import {', '.join(row)}{comment}")
                    row = []
                row.append(item)
            out.append(f"{indent}from {module} import {', '.join(row)}{comment}")
            continue
        rows, row = [], f"{indent}from {module} import ("
        for i, item in enumerate(items):
            piece = item + ("," if i + 1 < len(items) else ")")
            if len(row) + len(piece) + 1 > WIDTH - len(comment) and not row.endswith("("):
                rows.append(row.rstrip())
                row = indent + " " * 4 + piece
            else:
                row += ("" if row.endswith("(") else " ") + piece
        rows.append(row + comment)
        out += rows
    return out


def code_tokens(text: str) -> list[tokenize.TokenInfo]:
    return [t for t in tokenize.generate_tokens(io.StringIO(text).readline)
            if t.type not in (tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT)]


def redirect(rel: str, text: str, origin: str, self_dest: str | None = None, used_before: set[str] | None = None,
             back: list[tuple[str, str]] | None = None, lost: bool = False) -> tuple[str, dict]:
    """`text`, a file at `rel` whose module names are read as written in `origin`, reading each moved definition from
    its new module and each whole-moved file at its new path. `back` are (new module, name) pairs to import at the
    end of the module-level imports; `used_before` the import names to drop where the text no longer reads them.
    With `lost`, the second pass: what it read through a file that no longer has it, it reads where it is defined."""
    f = File(rel, text)
    toks = code_tokens(text)
    import_lines = {ln for s, _ in f.imports for ln in range(s.lineno, s.end_lineno + 1)}
    edits: list[tuple[tuple[int, int], tuple[int, int], str]] = []
    need: dict[int, list[tuple[str, str, str | None]]] = {}  # id of a binding statement -> the imports to put after it
    loads_at: dict[int, tuple[ast.Assign, ast.AST | None]] = {}
    # a name after a dot is an attribute and takes no name of this file's
    taken_names = {t.string for n, t in enumerate(toks) if t.type == tokenize.NAME and not (n and toks[n - 1].string == ".")}
    count = {"reads": 0, "imports": 0}

    chosen: dict[str, str] = {}

    def alias_for(dest: str, was: str | None = None) -> str:
        """The name this file reads the module `dest` by: the table's, the one it was read by, or the file's own."""
        if dest not in chosen:
            cands = [c for c in (ALIAS.get(dest), was, Path(dest).stem, Path(dest).stem + "_") if c]
            for cand in cands:
                holder = f.bound(cand, 10 ** 9)
                if (cand not in taken_names and cand not in chosen.values()) \
                        or (holder and alias_target(holder[0], holder[1], origin) == (dest, None)):
                    chosen[dest] = cand
                    break
            else:
                problems.append(f"{rel}: {', '.join(f'`{c}`' for c in cands)} are names it already uses: give {dest} another alias")
                chosen[dest] = cands[0]
        return chosen[dest]

    def module_at(name: str, line: int, depth: int = 0) -> tuple[tuple[str, str | None] | None, ast.stmt | None, tuple | None]:
        """(the module `name` is at `line`, the statement that binds it, and that statement with its scope where it is
        a load by path, which the new module's import goes after).  A name given to a module another name holds is
        bound where that other name is: its import, or its load by path."""
        hit = f.bound(name, line)
        if hit:
            cur = alias_target(hit[0], hit[1], origin)
            if cur is not None and cur[1] is not None:  # a name of another file: a module there, perhaps
                there = resolve(*cur)
                cur = there if there is not None and there[1] is None else None
            return cur, hit[0], None
        by_path = f.loaded(name, line)
        if by_path:
            return (by_path[1], None), by_path[0], by_path
        given = f.renamed(name, line) if depth < 4 else None
        if given is not None and given[1][0] != name:
            stmt, parts, _scope = given
            cur, binder, by_path = module_at(parts[0], stmt.lineno, depth + 1)
            for attr in parts[1:]:
                cur = resolve(cur[0], attr) if cur is not None and cur[1] is None else None
            if cur is not None and cur[1] is None:
                return cur, binder, by_path
        return None, None, None

    i = 0
    while i < len(toks):
        t = toks[i]
        if t.type != tokenize.NAME or t.string not in f.roots or (i and toks[i - 1].string == ".") \
                or t.start[0] in import_lines:
            i += 1
            continue
        j = i
        while j + 2 < len(toks) and toks[j + 1].string == "." and toks[j + 2].type == tokenize.NAME:
            j += 2
        if not lost and not any(toks[x].string in MOVED for x in range(i + 2, j + 1, 2)):
            i = j + 1
            continue
        cur, binder, by_path = module_at(t.string, t.start[0])
        k, to, k_to = i, None, i
        while cur is not None and k < j:  # a module: its next attribute
            k += 2
            attr = toks[k].string
            nxt = resolve(cur[0], attr)
            if nxt is not None and nxt in where:
                to, k_to = (where[nxt], nxt[1], None), k
                break
            gone = lost and cur[0] in CHANGED and cur[0] in TEXT and attr in has(cur[0])
            if gone and nxt is None:
                problems.append(f"{rel}:{t.start[0]}: reads `{attr}` of {cur[0]}, which no longer has it "
                                f"(`{''.join(x.string for x in toks[i:k + 1])}`), and nothing says where it is defined")
                break
            if nxt is not None and nxt[1] is None and (gone or to is not None):
                to, k_to = (place(nxt[0]), None, attr), k  # a module: read from the deepest one the chain names
            elif nxt is not None and gone:
                to, k_to = (place(nxt[0]), nxt[1], None), k
            if nxt is None or nxt[1] is not None:
                break
            cur = nxt
        if to is not None:
            dest, name, was = to
            if dest == self_dest:
                new = name
            else:
                a = alias_for(dest, was)
                new = a if name is None else f"{a}.{name}"
                item = import_of(dest, rel, None, a)
                if item not in need.setdefault(id(binder), []):
                    need[id(binder)].append(item)
                if by_path:
                    loads_at[id(binder)] = (by_path[0], by_path[2])
            edits.append((t.start, toks[k_to].end, new))
            count["reads"] += 1
        i = j + 1
    lines = text.split("\n")
    for (r0, c0), (r1, c1), new in sorted(edits, reverse=True):
        if r0 != r1:
            problems.append(f"{rel}:{r0}: a dotted name over two lines reads a moved definition: join it")
            continue
        lines[r0 - 1] = lines[r0 - 1][:c0] + new + lines[r0 - 1][c1:]
    after_toks = code_tokens("\n".join(lines))
    after_reads: list[set[str]] = []

    def still_read(name: str, scope: ast.AST | None) -> bool:
        """Whether the rewritten text reads `name` in `scope`: by scope at module level, by its tokens in a function."""
        if scope is None:
            if not after_reads:
                after_reads.append(File(rel, "\n".join(lines)).module_reads())
            return name in after_reads[0]
        return any(t.type == tokenize.NAME and t.string == name and scope.lineno <= t.start[0] <= scope.end_lineno
                   and t.start[0] not in import_lines and not (n and after_toks[n - 1].string == ".")
                   for n, t in enumerate(after_toks))

    def late(line: int) -> str:
        """The mark a new module-level import at `line` needs: ruff's E402 holds wherever a statement above it is
        neither the docstring, an import nor a change to ``sys.path``."""
        for n in f.tree.body:
            if n.lineno >= line:
                break
            if isinstance(n, (ast.Import, ast.ImportFrom)) or (isinstance(n, ast.Expr) and (
                    isinstance(n.value, ast.Constant) or "sys.path" in ast.unparse(n.value))):
                continue
            return "  # noqa: E402"
        return ""

    binders = [*f.imports, *f.calls, *((s, inside) for s, _, _, inside in f.renames)]
    scope_of = {id(s): inside for s, inside in binders}
    line_of = {id(s): s.lineno for s, _ in binders}
    seen: dict[int, list] = {}
    # one import of a new module a scope, and none in a function for what the module's own level imports: a name a
    # function imports is its local from its first line, so a read above the import there would find it unbound
    for key in sorted(need, key=lambda k: (scope_of[k] is not None, line_of[k])):
        mine = seen.setdefault(id(scope_of[key]), [])
        outer = seen.get(id(None), []) if scope_of[key] is not None else []
        need[key] = [item for item in need[key] if item not in mine and item not in outer]
        mine += need[key]

    # each change to the lines: (first line, last line, the lines in their place); an insertion's last is its first less one
    ops: list[tuple[int, int, list[str]]] = []
    last_top = max((s.end_lineno for s, inside in f.imports if inside is None and s in f.tree.body), default=0)
    top: list[tuple[str, str, str | None]] = [import_of(dest, rel, name) for dest, name in sorted(back or [])]
    for stmt, scope in loads_at.values():  # a file loaded by its path: the new module is imported where the load is
        if scope is None:  # one named by its path can go to the head; a bare name is found only once the load has run
            top += [w for w in need[id(stmt)] if "." in w[0]]
            if after := [w for w in need[id(stmt)] if "." not in w[0]]:
                ops.append((stmt.end_lineno + 1, stmt.end_lineno, render("", after, late(stmt.end_lineno + 1))))
        elif need[id(stmt)]:
            first = lines[stmt.lineno - 1]
            ops.append((stmt.end_lineno + 1, stmt.end_lineno, render(first[:len(first) - len(first.lstrip())], need[id(stmt)], "")))
    for stmt, inside in f.imports:
        keep: list[tuple[str, str, str | None]] = []
        moved: list[tuple[str, str, str | None]] = []
        touched = False
        for a in stmt.names:
            local = a.asname or (a.name.split(".")[0] if isinstance(stmt, ast.Import) else a.name)
            target = alias_target(stmt, a, origin)
            final = resolve(*target) if target and target[1] is not None else target
            same = ("", a.name, a.asname) if isinstance(stmt, ast.Import) else (stmt.module or "", a.name, a.asname)
            if final is not None and final in where:  # a moved definition, by name: from its new module, where still read
                touched = True
                if where[final] != self_dest and still_read(local, inside):
                    moved.append(import_of(where[final], rel, final[1], local))
            elif used_before is not None and inside is None and local in used_before and not still_read(local, None):
                touched = True  # only a definition that left read it
            elif target is not None and target[1] is None and target[0] in WHOLE:  # a file that moves whole
                touched = True
                keep.append(import_of(WHOLE[target[0]], rel, None, a.asname or Path(target[0]).stem))
            elif final is not None and final[1] is not None and final[0] in WHOLE:  # a name of a file that moves whole
                touched = True
                keep.append(import_of(WHOLE[final[0]], rel, final[1], local))
            elif target is not None and target[1] is None and id(stmt) in need and not still_read(local, inside):
                touched = True  # the old module, which nothing here reads any more
            else:
                keep.append(same)
        extra = need.get(id(stmt), [])
        if not touched and not extra:
            continue
        first = lines[stmt.lineno - 1]
        indent = first[:len(first) - len(first.lstrip())]
        if stmt.col_offset != len(indent):
            problems.append(f"{rel}:{stmt.lineno}: an import that doesn't start its line reads a moved definition: split the line")
            continue
        tail = lines[stmt.end_lineno - 1][stmt.end_col_offset:]
        comment = "  " + tail.strip() if tail.strip().startswith("#") else ""
        noqa = "  # noqa: E402" if "E402" in comment else late(stmt.lineno) if inside is None else ""
        new_lines = [line for k, item in enumerate(dict.fromkeys(keep))
                     for line in render(indent, [item], comment if "noqa" in comment or k == 0 else "")] \
            if isinstance(stmt, ast.Import) \
            else render(indent, keep, comment)
        new_lines += render(indent, sorted(dict.fromkeys(moved + extra), key=lambda w: (w[0], w[1], w[2] or "")), noqa)
        ops.append((stmt.lineno, stmt.end_lineno, new_lines))
        count["imports"] += 1
    if top:
        ops.append((last_top + 1, last_top, render("", list(dict.fromkeys(top)), late(last_top + 1))))
    for a, b, new_lines in sorted(ops, key=lambda op: (-op[0], -op[1])):
        lines[a - 1:b] = new_lines
    return "\n".join(lines), count


def tidy_head(rel: str, text: str) -> str:
    """A new module's text with its imports in one order: the plain ones, then the ``from`` ones by module."""
    f = File(rel, text)
    run = [n for n in f.tree.body if isinstance(n, (ast.Import, ast.ImportFrom)) and getattr(n, "module", "") != "__future__"]
    first_other = next((n.lineno for n in f.tree.body[1:] if not isinstance(n, (ast.Import, ast.ImportFrom))), 10 ** 9)
    run = [n for n in run if n.lineno < first_other]
    if not run:
        return text
    plain = sorted({("", a.name, a.asname) for n in run if isinstance(n, ast.Import) for a in n.names}, key=lambda w: w[1])
    froms = sorted({(n.module, a.name, a.asname) for n in run if isinstance(n, ast.ImportFrom) for a in n.names},
                   key=lambda w: (not w[0].split(".")[0] in sys.stdlib_module_names, w[0], w[1]))
    lines = text.split("\n")
    new = [line for w in plain for line in render("", [w], "")] + ([""] if plain and froms else []) + render("", froms, "")
    lines[run[0].lineno - 1:max(n.end_lineno for n in run)] = new
    return "\n".join(lines)


# ---- each new module
new_modules: dict[str, str] = {}
for dest, m in TABLE["modules"].items():
    wanted: list[tuple[str, str, str | None]] = []  # the imports the moved definitions need, as their files wrote them
    body: list[tuple[bool, str, int, int, str]] = []
    origin = None
    for rel, names in m["take"].items():
        f = file(rel)
        if f is None:
            continue
        origin = origin or rel
        nodes = sorted({id(f.top[n]): f.top[n] for n in names if n in f.top}.values(), key=lambda n: n.lineno)
        for node in nodes:
            a, b = f.span(node)
            label = getattr(node, "name", None) or ", ".join(n for n in names if f.top.get(n) is node)
            body.append((isinstance(node, DEFS), rel, a, b, "\n".join(f.lines[a - 1:b])))
            for name in sorted(f.reads(node)):
                if (rel, name) in where:
                    if where[(rel, name)] != dest:
                        wanted.append(import_of(where[(rel, name)], dest, name))
                    continue
                if name in f.top:
                    if m.get("reads_source"):  # a module that stays beside its source and reads what stays of it
                        wanted.append(import_of(rel, dest, name))
                    else:
                        problems.append(f"{rel}: `{label}` goes to {dest} and reads `{name}`, which stays: add `{name}` "
                                        f"to the table, or leave `{label}`")
                    continue
                hit = f.bound(name, node.lineno)
                if hit is None or hit[2] is not None:
                    if name not in ("__file__", "__name__", "__doc__"):
                        notes.append(f"{rel}: `{label}` reads `{name}`, which no module-level statement binds")
                    else:
                        problems.append(f"{rel}: `{label}` goes to {dest} and reads `{name}`, its own file's: leave it")
                    continue
                stmt, al, _ = hit
                written = alias_target(stmt, al, rel)
                defined = resolve(*written) if written is not None and written[1] is not None else None
                if defined is not None and defined[1] is not None and defined[0] != written[0] \
                        and defined[0].startswith("verity/") and (defined[0], defined[1]) not in where:
                    # a name its source read through a module that only imports it: from the module that defines it,
                    # so the new module loads no more than it reads
                    wanted.append(import_of(defined[0], dest, defined[1], al.asname or al.name))
                    continue
                wanted.append(("" if isinstance(stmt, ast.Import) else stmt.module, al.name, al.asname))
    for line in m.get("imports", []):  # what the module's `tail` reads
        for n in ast.parse(line).body:
            wanted += [("" if isinstance(n, ast.Import) else n.module, a.name, a.asname) for a in n.names]
    head = [f'"""{m["doc"]}"""', "from __future__ import annotations", ""]
    by_root = sorted({w for w in wanted if w[0] == ROOTED_AS or w[0].startswith(ROOTED_AS + ".")})
    if by_root and str(Path(dest).parent) == BARE:  # the root on its path first, as the directory's files have it
        wanted += [("", "sys", None), ("pathlib", "Path", None)]
    else:
        by_root = []
    plain = sorted({w for w in wanted if w[0] == ""}, key=lambda w: w[1])
    froms = sorted({w for w in wanted if w[0] != ""} - set(by_root))
    head += [line for w in plain for line in render("", [w], "")] + ([""] if plain and froms else []) + render("", froms, "")
    if by_root:
        head += ["", f"sys.path.append(str(Path(__file__).resolve().parents[{len(Path(dest).parts) - 1}]))"
                     f"  # the root, for {ROOTED_AS}", "", *render("", by_root, "")]
    text = "\n".join(head) + "\n"
    for k, (is_def, rel, a, b, piece) in enumerate(body):
        before = body[k - 1] if k else None
        if is_def or (before and before[0]):
            text += "\n\n"
        elif before is None or before[1] != rel or before[3] + 1 != a:
            text += "\n"
        text += piece + "\n"
    if m.get("tail"):
        text += "\n\n" + m["tail"].rstrip("\n") + "\n"
    try:
        text, _ = redirect(dest, text, origin or dest, self_dest=dest)
        text = tidy_head(dest, text)
        final = File(dest, text)
    except SyntaxError as e:
        problems.append(f"{dest}: what the table takes doesn't parse as a module ({e})")
        continue
    for stmt, _ in final.imports:
        for al in stmt.names:
            target = alias_target(stmt, al, origin or dest)
            if dest.startswith("verity/") and target and target[0].startswith("benchmarks/") and target[0] not in WHOLE:
                problems.append(f"{dest}: it would import {target[0]} ({al.name}), which stays under benchmarks/: move what "
                                f"it reads of it, or leave the definition that reads it")
    new_modules[dest] = text

# ---- every file that reads a moved definition, the files they left first
CHANGED: dict[str, str] = {}
report: dict[str, dict] = {}


def readers() -> set[str]:
    """The files that can read a moved definition: those that bind a file definitions leave, or one that moves whole,
    by an import or a load by its path, and those that bind such a file in turn."""
    carriers, seen, grew = {*SOURCES, *WHOLE}, set(), True
    while grew:
        grew = False
        named = re.compile(r"\b(?:" + "|".join(sorted(re.escape(Path(rel).stem) for rel in carriers)) + r")\b")
        for rel in TRACKED:
            if rel in carriers or rel in seen or not named.search(TEXT[rel]):
                continue
            f = file(rel)
            if f is None:
                continue
            bound = []
            for stmt, inside in f.imports:
                for a in stmt.names:
                    target = alias_target(stmt, a, rel)
                    if target is not None and target[1] is not None:
                        target = resolve(*target) or target
                    bound.append((target[0] if target else None, inside))
            bound += [(f.loads_file(stmt), inside) for stmt, inside in f.calls]
            if any(to in carriers for to, _ in bound):
                seen.add(rel)
                if any(to in carriers and inside is None for to, inside in bound):
                    carriers.add(rel)
                    grew = True
    return carriers | seen


READERS = readers()
for rel in TRACKED:
    text = TEXT[rel]
    back: list[tuple[str, str]] = []
    used_before = None
    if rel in SOURCES:
        f = file(rel)
        gone = sorted({f.span(f.top[name]) for (r, name) in where if r == rel}, reverse=True)
        lines = list(f.lines)
        for a, b in gone:
            del lines[a - 1:b]
        text = re.sub(r"\n{4,}", "\n\n\n", "\n".join(lines))
        for e in TABLE.get("edits", []):  # before what is left is read: an edit may take the last read of a name
            if e["file"] == rel:
                if text.count(e["old"]) != 1:
                    problems.append(f"{rel}: the edit `{e['old'][:60]!r}…` matches {text.count(e['old'])} times, not once: "
                                    f"read the file at this base and fix the table's edit")
                text = text.replace(e["old"], e["new"])
        left = File(rel, text)
        still = left.module_reads()
        back = [(where[(r, name)], name) for (r, name) in where if r == rel and name in still]
        used_before = f.module_reads()
    elif rel not in READERS or file(rel) is None:
        continue
    new, count = redirect(rel, text, rel, used_before=used_before, back=back)
    if new != TEXT[rel]:
        CHANGED[rel] = new
        report[rel] = count | ({"left": len(gone), "back": sorted(n for _, n in back)} if rel in SOURCES else {})

# ---- what was read through a file that no longer has it: read where it is defined
_has: dict[str, set[str]] = {}


def has(rel: str) -> set[str]:
    """The names the module `rel` had at its top before the cut and doesn't bind there after it."""
    if rel not in _has:
        tops = []
        for text in (TEXT[rel], CHANGED[rel]):
            table = symtable.symtable(text, rel, "exec")
            tops.append({x.get_name() for x in table.get_symbols() if x.is_assigned() or x.is_imported() or x.is_namespace()})
        _has[rel] = tops[0] - tops[1]
    return _has[rel]


for rel in sorted(READERS):
    if rel not in WHOLE:
        new, count = redirect(rel, CHANGED.get(rel, TEXT[rel]), rel, lost=True)
        if count["reads"]:
            CHANGED[rel] = new
            report.setdefault(rel, {"reads": 0, "imports": 0})
            report[rel]["through"] = count["reads"]
_has.clear()
for rel in sorted(READERS):
    text = CHANGED.get(rel, TEXT[rel])
    if rel in WHOLE:
        continue
    try:
        f = File(rel, text)
    except SyntaxError:
        continue
    toks = code_tokens(text)
    import_lines = {ln for x, _ in f.imports for ln in range(x.lineno, x.end_lineno + 1)}
    for i, t in enumerate(toks):
        if t.type != tokenize.NAME or t.string not in f.roots or (i and toks[i - 1].string == ".") \
                or t.start[0] in import_lines:
            continue
        j = i
        while j + 2 < len(toks) and toks[j + 1].string == "." and toks[j + 2].type == tokenize.NAME:
            j += 2
        if j == i:
            continue
        hit = f.bound(t.string, t.start[0])
        by_path = None if hit else f.loaded(t.string, t.start[0])
        cur = alias_target(hit[0], hit[1], rel) if hit else (by_path[1], None) if by_path else None
        if cur is not None and cur[1] is not None:
            there = resolve(*cur)
            cur = there if there is not None and there[1] is None else None
        k = i
        while cur is not None and cur[1] is None and k < j:
            k += 2
            if cur[0] in CHANGED and cur[0] in TEXT and toks[k].string in has(cur[0]):
                problems.append(f"{rel}:{t.start[0]}: reads `{toks[k].string}` of {cur[0]}, which no longer has it "
                                f"(`{''.join(x.string for x in toks[i:k + 1])}`): read it where it is defined")
                break
            cur = resolve(cur[0], toks[k].string)

# ---- the edits and files no rule writes
for e in TABLE.get("edits", []):
    if e["file"] in SOURCES:
        continue
    made = new_modules if e["file"] in new_modules else CHANGED  # a new module's text is edited as the cut wrote it
    text = made.get(e["file"], TEXT.get(e["file"]))
    if text is None and (ROOT / e["file"]).is_file():
        text = (ROOT / e["file"]).read_text(encoding="utf-8")
    if text is None or text.count(e["old"]) != 1:
        problems.append(f"{e['file']}: the edit `{e['old'][:60]!r}…` matches {0 if text is None else text.count(e['old'])} "
                        f"times, not once: read the file at this base and fix the table's edit")
        continue
    made[e["file"]] = text.replace(e["old"], e["new"])
for rel, text in TABLE.get("files", {}).items():
    if rel in TEXT:
        problems.append(f"{rel}: the table gives it whole and it exists: name another file")
    new_modules[rel] = text

# ---- what names a file by its path: a suite's declared inputs, a load by path, a document
LISTS = "tests/lint/allowlists/"  # the counted lints' rows, which `allow.py` moves


def text_after(rel: str) -> str | None:
    """A tracked file's text as the cut leaves it."""
    if rel in CHANGED or rel in TEXT:
        return CHANGED.get(rel, TEXT.get(rel))
    try:
        return (ROOT / rel).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


for old, new in WHOLE.items():
    named = subprocess.run(["git", "-C", str(ROOT), "grep", "-l", "-z", "-F", "-e", old, "--", "."], capture_output=True,
                           text=True).stdout.split("\0")
    for rel in sorted({r for r in named if r} | {r for r in CHANGED if old in CHANGED[r]}
                      | {r for r in new_modules if old in new_modules[r]}):
        text = new_modules.get(rel) or text_after(rel)
        if rel.startswith(LISTS) or rel in WHOLE or text is None or old not in text:
            continue
        line = text[:text.index(old)].count("\n") + 1
        problems.append(f"{rel}:{line}: names {old}, which moves whole to {new}: write what it should name after the cut "
                        f"in an `edits` row of the table")


MADE = set(new_modules) | set(WHOLE.values())


def imported_files(rel: str, text: str, held: set[str]) -> set[str]:
    """The files of `held` that the module `rel` imports directly, by any import its text has."""
    try:
        tree = ast.parse(text, filename=rel)
    except SyntaxError:
        return set()
    out = set()
    for stmt in ast.walk(tree):
        if isinstance(stmt, ast.Import):
            mods = [a.name for a in stmt.names]
        elif isinstance(stmt, ast.ImportFrom) and stmt.module and not stmt.level:
            mods = [stmt.module, *(f"{stmt.module}.{a.name}" for a in stmt.names)]
        else:
            continue
        for mod in mods:
            ways = [mod.replace(".", "/") + ".py"] if "." in mod else [str(Path(rel).parent / (mod + ".py")), f"{BARE}/{mod}.py"]
            out |= {way for way in ways[:1 + (ways[0] not in held)] if way in held}
    return out


# a suite's pyproject.toml that names a file by its path names what its tests may read (`[tool.verity.tests] inputs`):
# where such a file comes to import a file the cut makes, the suite reads that one too
AFTER = (set(TEXT) - set(WHOLE)) | MADE
for toml in sorted(subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z", "--", "*pyproject.toml"], capture_output=True,
                                  text=True).stdout.split("\0")):
    text = text_after(toml) if toml else None
    if text is None:
        continue
    quoted = set(re.findall(r'"([A-Za-z0-9_./-]+)"', text))

    def names(path: str) -> bool:
        return any(path == q or path.startswith(q.rstrip("/") + "/") for q in quoted)

    for rel in sorted(q for q in quoted if q in CHANGED and q.endswith(".py")):
        before = imported_files(rel, TEXT[rel], set(TEXT))
        for made in sorted((imported_files(rel, CHANGED[rel], AFTER) - before) & MADE):
            beside = [b for b in before if Path(b).parent == Path(made).parent]
            if names(made) or (beside and not any(names(b) for b in beside)):
                continue  # named, or its directory reaches the suite some other way, as its neighbours there do
            problems.append(f"{toml}: names {rel}, which comes to import {made}: name {made} there too, in an `edits` "
                            f"row of the table (a suite fails on a file its tests read outside its inputs)")

for stem, how in sorted(NAMED.items()):
    if len(how) == 2:
        problems.append(f"{BARE}/{stem}.py would be imported by its bare name ({', '.join(sorted(how[True])[:3])}) and from "
                        f"the root ({', '.join(sorted(how[False])[:3])}), and the tree lets a module have one name: set "
                        f"`rooted` for it in the table, and see that the files of {BARE}/ that import it have the root "
                        f"on their path")
_by_root = re.compile(rf"(?m)^[ \t]*(?:from|import)[ \t]+{re.escape(ROOTED_AS)}\b")
for rel, text in {**CHANGED, **new_modules}.items():
    if not rel.endswith(".py"):
        continue
    if str(Path(rel).parent) == BARE and _by_root.search(text) and not _by_root.search(TEXT.get(rel, "")) \
            and "sys.path" not in text:
        problems.append(f"{rel}: it comes to import a module of {BARE}/ from the root and never puts the root on its "
                        f"path: give it the line the directory's other files have, in a change before the cut")
    try:
        ast.parse(text, filename=rel)
    except SyntaxError as e:
        problems.append(f"{rel}: the cut leaves it unparseable ({e})")
if problems:
    sys.exit("cut.py refuses, and changed nothing:\n- " + "\n- ".join(dict.fromkeys(problems)))

summary = {
    "definitions": len(where),
    "modules": {dest: {"definitions": sum(len(v) for v in TABLE["modules"][dest]["take"].values()) if dest in TABLE["modules"]
                       else None, "lines": text.count("\n")} for dest, text in new_modules.items()},
    "whole": WHOLE,
    "files": {rel: report.get(rel, {}) for rel in sorted(CHANGED)},
    "notes": sorted(set(notes)),
}
if not DRY:
    for rel, text in CHANGED.items():
        (ROOT / rel).write_text(text, encoding="utf-8")
    for rel, text in new_modules.items():
        (ROOT / rel).parent.mkdir(parents=True, exist_ok=True)
        (ROOT / rel).write_text(text, encoding="utf-8")
    for old, new in WHOLE.items():
        subprocess.run(["git", "-C", str(ROOT), "mv", old, new], check=True)
if REPORT:
    REPORT.write_text(json.dumps(summary, indent=1) + "\n")
print(json.dumps({"definitions": summary["definitions"], "modules": summary["modules"], "files changed": len(CHANGED),
                  "reads redirected": sum(v.get("reads", 0) for v in report.values()),
                  "import statements rewritten": sum(v.get("imports", 0) for v in report.values()),
                  "notes": summary["notes"]}, indent=1))
