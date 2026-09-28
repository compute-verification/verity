"""Split the attention family's registry modules into Definitions and vocabulary, ahead of the catalog move.

    uv run python integrations/vllm/tools/migrate/split_attention.py            apply on this tree, checking the invariants
    uv run python integrations/vllm/tools/migrate/split_attention.py --check    exit 1 unless the split is applied (no writes)
    uv run python integrations/vllm/tools/migrate/split_attention.py facts --out F.json
                                                                                 the invariants' facts of this tree

The family's Definition modules (`DEFINITION_MODULES`, and verity#1130's DSA and sparse-MLA modules when present) keep their
paths, so every Definition keeps its module, its id and its descriptor digest; what leaves them is what the frontend reads and no
Definition does (`MOVES`): FlashAttention's launch rules for the `MASKED_FROM` static, now in `registry/attention_vocab.py`.
Every import and use of a moved name is rewritten repo-wide: ``from M import name``, ``alias.name`` for any alias of ``M``
(module-level or function-local), and ``M.name`` in docstrings, comments and Markdown.

A rerun on a later main is the rebase: it applies what is missing and leaves an applied tree as it is. It fails loudly, writing
nothing, on what it can't rewrite: a moved name that only one of its module and `attention_vocab` defines in part, a moved
rule that reads a name its module keeps, ``from M import *``, ``getattr(M, "name")``, a file that names ``M`` and doesn't parse.
Unless ``--no-verify``, it then rebuilds `facts` in a fresh process before and after the rewrite and restores every file when they
differ: each family Definition's id and descriptor digest at every circuit-check binding, pin and served binding
(`targets.attention_spec` on the registered targets), and circuit-check's whole task list (target ids, template statics, pins).
"""
from __future__ import annotations

import argparse
import ast
import builtins
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

PKG = "verity_vllm.program.registry"
REGISTRY = Path("integrations/vllm/verity_vllm/program/registry")
#: the family's Definition modules: after the split each imports only `verity` and other Definition modules
DEFINITION_MODULES = ("fa2_check_inf", "fa3_check_inf", "fa2_softcap", "boolean_attention", "boolean_softcap", "mla")
#: verity#1130's (GLM-5.3's DSA indexer and FA3 sparse MLA): Definitions only, checked when present
OPTIONAL_DEFINITION_MODULES = ("dsa", "boolean_dsa", "sparse_mla", "boolean_sparse_mla")
#: the family's modules this split leaves as they are, and why
LEFT = {
    "hopper": "the B1 Hopper profile: b1's composites re-instantiated over a copy of b1's globals, three of them attention; its "
              "twins and vocabulary (`hopper_wgmma_dot16_total`, `SERVE_HOPPER`, `CALL_NAMES`, `structural_diff`) split with b1's",
    "mla_capture": "capture only (the Triton MLA decode's produce / check / check-bits commands): no Definition",
    "mla_check": "twins only (numpy models of the MLA kernels the captures are held to): no Definition",
}
VOCAB = "attention_vocab"
#: source module -> the top-level names that move to `VOCAB`, in order
MOVES = {"fa2_check_inf": ("KBLOCK_M", "fa2_masked_from"), "fa3_check_inf": ("fa3_masked_from",)}
VOCAB_DOC = '''"""The attention family's vocabulary: FlashAttention's launch rules for the `MASKED_FROM` static of the per-iteration `Check_inf`
chains (`fa2_check_inf.Attention_v5` / `_v11`, `fa2_softcap.AttentionSoftcap_v2`, `fa3_check_inf.Attention_v4`).

The Definitions are generic over `MASKED_FROM`; the lowest key block a served row's kernel guards is a fact of the launch (its tile,
its causality, the sequence's query and key counts), which `targets.attention_spec` reads here to bind it.
"""
'''
WIDTH = 140


class SplitError(RuntimeError):
    """What the script refuses, by name."""


# --- source edits on byte offsets (ast's columns are UTF-8 byte offsets) ------------------------------------------------------

class Source:
    def __init__(self, path: Path, text: str | None = None):
        self.path = path
        self.data = (text if text is not None else path.read_text(encoding="utf-8")).encode()
        self.tree = ast.parse(self.data, filename=str(path))
        self.starts = [0]
        for line in self.data.splitlines(keepends=True):
            self.starts.append(self.starts[-1] + len(line))
        self.edits: list[tuple[int, int, bytes]] = []

    def off(self, line: int, col: int) -> int:
        return self.starts[line - 1] + col

    def span(self, node: ast.AST) -> tuple[int, int]:
        return self.off(node.lineno, node.col_offset), self.off(node.end_lineno, node.end_col_offset)

    def line_span(self, first: int, last: int) -> tuple[int, int]:
        return self.starts[first - 1], self.starts[last]

    def text(self, a: int, b: int) -> str:
        return self.data[a:b].decode()

    def replace(self, a: int, b: int, new: str) -> None:
        self.edits.append((a, b, new.encode()))

    def result(self) -> str:
        out, prev = [], len(self.data)
        for a, b, new in sorted(self.edits, key=lambda e: (e[0], e[1]), reverse=True):
            if b > prev:
                raise SplitError(f"{self.path}: overlapping edits at byte {a}")
            out.append(self.data[b:prev])
            out.append(new)
            prev = a
        out.append(self.data[:prev])
        return b"".join(reversed(out)).decode()


def _wrap_import(head: str, names: list[str], indent: str) -> str:
    """``from X import a, b`` within `WIDTH`, else parenthesised with continuation lines under the first name."""
    one = f"{head} {', '.join(names)}"
    if len(indent) + len(one) <= WIDTH:
        return one
    lines, cur = [], f"{head} ("
    pad = " " * len(cur)
    for i, n in enumerate(names):
        piece = n + (")" if i == len(names) - 1 else ",")
        if len(indent) + len(cur) + len(piece) + 1 > WIDTH and cur.strip() not in (f"{head} (",):
            lines.append(cur.rstrip())
            cur = pad
        cur += piece + " "
    lines.append(cur.rstrip())
    return ("\n" + indent).join(lines)


def _alias_text(a: ast.alias) -> str:
    return a.name if a.asname is None else f"{a.name} as {a.asname}"


def _all_text(names: list[str]) -> str:
    """``__all__ = [...]`` wrapped within `WIDTH`."""
    items = [json.dumps(n) for n in names]
    lines, cur = [], "__all__ = ["
    for i, it in enumerate(items):
        piece = it + ("]" if i == len(items) - 1 else ",")
        if len(cur) + len(piece) + 1 > WIDTH and cur != "__all__ = [":
            lines.append(cur.rstrip())
            cur = " " * len("__all__ = [")
        cur += piece + " "
    if not items:
        cur = "__all__ = []"
    lines.append(cur.rstrip())
    return "\n".join(lines)


# --- the move -------------------------------------------------------------------------------------------------------------------

def _top_level(tree: ast.Module) -> dict[str, ast.stmt]:
    """Top-level names bound by a def, a class or a plain assignment, to their statement."""
    out: dict[str, ast.stmt] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[node.name] = node
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    out[t.id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out[node.target.id] = node
    return out


def _imported(tree: ast.Module) -> dict[str, tuple[ast.stmt, ast.alias]]:
    """Top-level names bound by an import, to the statement and alias."""
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                out[(a.asname or a.name).split(".")[0]] = (node, a)
    return out


def _segment_lines(src: Source, node: ast.stmt) -> tuple[int, int]:
    """The statement's lines with its decorators and the comment lines directly above it."""
    first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    lines = src.data.decode().splitlines()
    while first > 1 and lines[first - 2].lstrip().startswith("#"):
        first -= 1
    return first, node.end_lineno


def _free_names(node: ast.stmt) -> set[str]:
    """Names ``node`` reads that it does not bind itself (parameters, assignments, comprehension targets), annotations aside."""
    loads, stores = set(), set()
    skip = set()
    for n in ast.walk(node):
        for ann in (getattr(n, "returns", None), getattr(n, "annotation", None)):
            if ann is not None:
                skip.update(id(x) for x in ast.walk(ann))
    for n in ast.walk(node):
        if id(n) in skip:
            continue
        if isinstance(n, ast.Name):
            (loads if isinstance(n.ctx, ast.Load) else stores).add(n.id)
        elif isinstance(n, ast.arg):
            stores.add(n.arg)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            stores.add(n.name)
    return loads - stores - set(dir(builtins))


def _module_names(text: str) -> set[str]:
    return set(_top_level(ast.parse(text)))


def move(repo: Path, writes: dict[Path, str]) -> list[str]:
    """Move `MOVES` into `VOCAB`, recording the new texts in ``writes``; returns what it did."""
    vocab_path = repo / REGISTRY / f"{VOCAB}.py"
    vocab_text = writes.get(vocab_path) or (vocab_path.read_text(encoding="utf-8") if vocab_path.exists() else None)
    have = _module_names(vocab_text) if vocab_text is not None else set()
    done, segments, imports, moved_names = [], [], [], []
    for mod, names in MOVES.items():
        path = repo / REGISTRY / f"{mod}.py"
        if not path.exists():
            raise SplitError(f"{path}: the module the split moves {', '.join(names)} out of is missing")
        src = Source(path, writes.get(path))
        top = _top_level(src.tree)
        here = [n for n in names if n in top]
        there = [n for n in names if n in have]
        if not here and len(there) == len(names):
            continue
        if len(here) != len(names) or there:
            raise SplitError(f"{mod}: a partial split ({', '.join(here) or 'none'} in {mod}, {', '.join(there) or 'none'} in {VOCAB}); "
                             f"expected {', '.join(names)} in exactly one of them")
        nodes = []
        for n in names:
            node = top[n]
            if isinstance(node, ast.Assign) and len(node.targets) != 1:
                raise SplitError(f"{mod}.{n}: bound by a multiple assignment; move it by hand")
            if node not in nodes:
                nodes.append(node)
        kept_top = set(top) - set(names)
        imp = _imported(src.tree)
        for node in nodes:
            for free in sorted(_free_names(node) - set(names)):
                if free in imp:
                    stmt, a = imp[free]
                    line = (f"import {_alias_text(a)}" if isinstance(stmt, ast.Import)
                            else f"from {'.' * stmt.level}{stmt.module or ''} import {_alias_text(a)}")
                    if stmt.level:
                        raise SplitError(f"{mod}: moved code reads {free}, a relative import")
                    if line not in imports:
                        imports.append(line)
                elif free in kept_top:
                    raise SplitError(f"{mod}: the moved code reads {free}, which {mod} keeps; move it too or by hand")
        cuts: list[list[int]] = []
        for node in sorted(nodes, key=lambda n: n.lineno):
            first, last = _segment_lines(src, node)
            a, b = src.line_span(first, last)
            segments.append((mod, node.lineno, src.text(a, b).rstrip("\n")))
            # with the blank lines around it, so what was before and after it is two blank lines apart
            while a > 0 and src.data[a - 2:a] == b"\n\n":
                a -= 1
            while b < len(src.data) and src.data[b:b + 1] == b"\n":
                b += 1
            if cuts and a <= cuts[-1][1]:
                cuts[-1][1] = max(b, cuts[-1][1])
            else:
                cuts.append([a, b])
        for a, b in cuts:
            src.replace(a, b, "\n\n" if b < len(src.data) else "")
        # __all__ and backticked mentions of the moved names in what stays
        for node in src.tree.body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "__all__" and isinstance(node.value, (ast.List, ast.Tuple))):
                vals = [e.value for e in node.value.elts if isinstance(e, ast.Constant)]
                if any(v in names for v in vals):
                    a, b = src.span(node)
                    src.replace(a, b, _all_text([v for v in vals if v not in names]))
        text = src.result()
        for n in names:
            text = text.replace(f"`{n}`", f"`{VOCAB}.{n}`")
        writes[path] = re.sub(r"\n{4,}", "\n\n\n", text)
        moved_names += list(names)
        done.append(f"{mod}: {', '.join(names)} -> {VOCAB}")
    if not segments:
        return done
    segments.sort(key=lambda s: (list(MOVES).index(s[0]), s[1]))
    body = "\n\n\n".join(s[2] for s in segments)
    if vocab_text is None:
        head = VOCAB_DOC + "from __future__ import annotations\n"
        if imports:
            head += "\n" + "\n".join(sorted(imports)) + "\n"
        writes[vocab_path] = head + "\n\n" + body + "\n\n\n" + _all_text(moved_names) + "\n"
    else:
        src = Source(vocab_path, vocab_text)
        alls = [n for n in src.tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__"
                                                                              for t in n.targets)]
        if len(alls) != 1:
            raise SplitError(f"{vocab_path}: expected one __all__ to extend")
        node = alls[0]
        vals = [e.value for e in node.value.elts if isinstance(e, ast.Constant)]
        a, b = src.span(node)
        src.replace(a, b, body + "\n\n\n" + _all_text(vals + moved_names))
        missing = [i for i in imports if i not in vocab_text]
        if missing:
            first = next(n for n in src.tree.body if isinstance(n, (ast.Import, ast.ImportFrom)))
            a, _ = src.line_span(first.lineno, first.lineno)
            src.replace(a, a, "\n".join(missing) + "\n")
        writes[vocab_path] = src.result()
    return done


# --- consumers, repo-wide -------------------------------------------------------------------------------------------------------

def _module_of(repo: Path, path: Path) -> list[str] | None:
    """The dotted package of a file under `verity_vllm` (for relative imports), else None."""
    rel = path.relative_to(repo)
    parts = rel.parts
    if parts[:2] != ("integrations", "vllm") or len(parts) < 4 or parts[2] != "verity_vllm":
        return None
    return list(parts[2:-1])


def _resolve_from(node: ast.ImportFrom, package: list[str] | None) -> str | None:
    if node.level == 0:
        return node.module
    if package is None or node.level > len(package):
        return None
    base = package[:len(package) - node.level + 1]
    return ".".join(base + ([node.module] if node.module else []))


MOVED = {f"{PKG}.{m}": set(ns) for m, ns in MOVES.items()}
ACCESSORS = {"getattr", "setattr", "hasattr", "delattr"}


def _scope_of(tree: ast.Module) -> dict[int, ast.AST]:
    """Each node's innermost enclosing function (or the module)."""
    out: dict[int, ast.AST] = {}

    def visit(node: ast.AST, scope: ast.AST) -> None:
        for ch in ast.iter_child_nodes(node):
            out[id(ch)] = scope
            visit(ch, ch if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) else scope)
    visit(tree, tree)
    return out


def rewrite_consumers(repo: Path, files: list[Path], writes: dict[Path, str]) -> list[str]:
    done = []
    names_re = re.compile("|".join(re.escape(m.rpartition(".")[2]) for m in MOVED))
    for path in files:
        if path.suffix != ".py":
            continue
        text = writes.get(path) or path.read_text(encoding="utf-8")
        if not names_re.search(text):
            continue
        try:
            src = Source(path, text)
        except SyntaxError as e:
            raise SplitError(f"{path}: names a module the split edits and does not parse ({e})") from e
        new = _rewrite_file(src, _module_of(repo, path))
        if new != text:
            writes[path] = new
            done.append(str(path.relative_to(repo)))
    return done


def _rewrite_file(src: Source, package: list[str] | None) -> str:
    tree = src.tree
    scope = _scope_of(tree)
    vocab_mod = f"{PKG}.{VOCAB}"
    # 1. ``from M import name``
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        mod = _resolve_from(node, package)
        if mod not in MOVED:
            continue
        if any(a.name == "*" for a in node.names):
            raise SplitError(f"{src.path}:{node.lineno}: `from {mod} import *` (name what it imports first)")
        moved = [a for a in node.names if a.name in MOVED[mod]]
        if not moved:
            continue
        keep = [a for a in node.names if a.name not in MOVED[mod]]
        indent = " " * node.col_offset
        head = f"from {'.' * node.level}{node.module or ''} import"
        lines = ([_wrap_import(head, [_alias_text(a) for a in keep], indent)] if keep else [])
        lines.append(_wrap_import(f"from {vocab_mod} import", [_alias_text(a) for a in moved], indent))
        a, b = src.span(node)
        src.replace(a, b, ("\n" + indent).join(lines))
    # 2. aliases of M: ``from PKG import M [as A]``, ``import PKG.M as A``, ``import PKG.M``
    aliases: dict[str, tuple[str, ast.stmt, ast.alias]] = {}   # bound name -> (module, import, alias); dotted for a plain import
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            base = _resolve_from(node, package)
            for a in node.names:
                full = f"{base}.{a.name}" if base else a.name
                if full in MOVED:
                    aliases[a.asname or a.name] = (full, node, a)
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name in MOVED:
                    aliases[a.asname or a.name] = (a.name, node, a)
    if not aliases:
        return src.result()
    vocab_bound = None                                         # how this file already names `VOCAB` at module level
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and _resolve_from(node, package) == PKG:
            for a in node.names:
                if a.name == VOCAB:
                    vocab_bound = a.asname or a.name
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name == vocab_mod and a.asname:
                    vocab_bound = a.asname
    users: dict[str, list[ast.AST]] = {k: [] for k in aliases}  # every load of an alias
    rewrites: dict[str, list[ast.Attribute]] = {k: [] for k in aliases}
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            dotted = _dotted(node.value)
            for bound, (mod, _imp, _a) in aliases.items():
                if dotted == bound and node.attr in MOVED[mod]:
                    rewrites[bound].append(node)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and node.id in users:
            users[node.id].append(node)
        if (isinstance(node, ast.Call) and len(node.args) >= 2 and isinstance(node.args[1], ast.Constant)
                and (node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)) in ACCESSORS):
            target = _dotted(node.args[0])
            for bound, (mod, _imp, _a) in aliases.items():
                if target == bound and node.args[1].value in MOVED[mod]:
                    raise SplitError(f"{src.path}:{node.lineno}: reaches {mod}.{node.args[1].value} by name; rewrite it by hand")
    # one edit per import statement: the aliases it no longer needs dropped, `VOCAB`'s import added once per scope
    by_import: dict[int, tuple[ast.stmt, list[str]]] = {}
    for bound, (_mod, imp, _a) in aliases.items():
        if rewrites[bound]:
            by_import.setdefault(id(imp), (imp, []))[1].append(bound)
    parents = {id(ch): node for node in ast.walk(tree) for ch in ast.iter_child_nodes(node)}
    scoped: set[int] = set()                                   # scopes that import `VOCAB` (already, or by this rewrite)
    for imp, bounds in sorted(by_import.values(), key=lambda e: e[0].lineno):
        where = scope[id(imp)]
        local = where is not tree
        name = VOCAB if (local or vocab_bound is None) else vocab_bound
        if name == VOCAB and _binds_other(tree, VOCAB, vocab_mod, package):
            raise SplitError(f"{src.path}: binds {VOCAB} to something else; rewrite it by hand")
        dropped = []
        for bound in bounds:
            nodes = rewrites[bound]
            for n in nodes:
                x, y = src.span(n)
                src.replace(x, y, f"{name}.{n.attr}")
            loads = [u for u in users[bound] if not any(u is r.value or _inside(u, r) for r in nodes)]
            plain = isinstance(imp, ast.Import) and aliases[bound][2].asname is None
            if not (loads or (plain and _used_elsewhere(tree, bound.split(".")[0], nodes))):
                dropped.append(aliases[bound][2])
        need_import = (name == VOCAB and id(where) not in scoped
                       and not (not local and vocab_bound is not None) and not _already(tree, imp, scope, package))
        if need_import:
            scoped.add(id(where))
            if not local:
                vocab_bound = VOCAB
        if not (dropped or need_import):
            continue
        indent = " " * imp.col_offset
        base = _resolve_from(imp, package) if isinstance(imp, ast.ImportFrom) else None
        new_import = (f"from {'.' * imp.level}{imp.module or ''} import {VOCAB}" if base == PKG
                      else f"from {PKG} import {VOCAB}")
        x, y = src.span(imp)
        if not dropped:
            src.replace(y, y, "\n" + indent + new_import)
            continue
        rest = [al for al in imp.names if not any(al is d for d in dropped)]
        if isinstance(imp, ast.ImportFrom):
            head = f"from {'.' * imp.level}{imp.module or ''} import"
            lines = [_wrap_import(head, [_alias_text(al) for al in rest], indent)] if rest else []
        else:
            lines = [f"import {', '.join(_alias_text(al) for al in rest)}"] if rest else []
        if need_import:
            lines.append(new_import)
        if lines:
            src.replace(x, y, ("\n" + indent).join(lines))
            continue
        parent = parents[id(imp)]
        block = next(blk for f in ("body", "orelse", "finalbody") for blk in [getattr(parent, f, None) or []]
                     if any(s is imp for s in blk))
        own_lines = (src.text(src.starts[imp.lineno - 1], x).strip() == ""
                     and src.text(y, src.starts[imp.end_lineno]).strip() == "")
        if len(block) == 1:
            src.replace(x, y, "pass")
        elif own_lines:
            src.replace(*src.line_span(imp.lineno, imp.end_lineno), "")
        else:
            raise SplitError(f"{src.path}:{imp.lineno}: an import the rewrite empties shares its line; rewrite it by hand")
    return src.result()


def _dotted(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        head = _dotted(node.value)
        return None if head is None else f"{head}.{node.attr}"
    return None


def _inside(inner: ast.AST, outer: ast.AST) -> bool:
    return any(inner is n for n in ast.walk(outer))


def _used_elsewhere(tree: ast.Module, top: str, rewritten: list[ast.Attribute]) -> bool:
    """A plain ``import PKG.M`` binds `verity_vllm`: does anything but the rewritten attributes read it?"""
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and n.id == top and isinstance(n.ctx, ast.Load):
            if not any(_inside(n, r) for r in rewritten):
                return True
    return False


def _binds_other(tree: ast.Module, name: str, vocab_mod: str, package: list[str] | None) -> bool:
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name == name:
            return True
        if isinstance(n, ast.Name) and n.id == name and isinstance(n.ctx, ast.Store):
            return True
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                if (a.asname or a.name) == name:
                    full = f"{_resolve_from(n, package)}.{a.name}" if isinstance(n, ast.ImportFrom) else a.name
                    if full != vocab_mod:
                        return True
    return False


def _already(tree: ast.Module, imp: ast.stmt, scope: dict[int, ast.AST], package: list[str] | None) -> bool:
    """`VOCAB` is imported in ``imp``'s scope already."""
    where = scope[id(imp)]
    for n in ast.walk(where):
        if isinstance(n, ast.ImportFrom) and scope.get(id(n)) is where and _resolve_from(n, package) == PKG:
            if any(a.name == VOCAB and a.asname is None for a in n.names):
                return True
    return False


TEXT_SUFFIXES = (".py", ".md")


def rewrite_mentions(repo: Path, files: list[Path], writes: dict[Path, str]) -> list[str]:
    """``M.name`` (with any dotted prefix: ``registry.M.name``, ``PKG.M:name``) in any text: docstrings, comments, Markdown."""
    pats = [re.compile(rf"(?<!\w)(?P<pre>(?:\w+\.)*){re.escape(m.rpartition('.')[2])}(?P<sep>[.:])(?P<name>{re.escape(n)})\b")
            for m, ns in MOVED.items() for n in ns]
    done = []
    for path in files:
        if path.suffix not in TEXT_SUFFIXES:
            continue
        text = writes.get(path) or path.read_text(encoding="utf-8")
        new = text
        for rx in pats:
            new = rx.sub(lambda m: f"{m['pre']}{VOCAB}{m['sep']}{m['name']}", new)
        if new != text:
            writes[path] = new
            done.append(str(path.relative_to(repo)))
    return done


def leftovers(repo: Path, files: list[Path], writes: dict[Path, str]) -> list[str]:
    """Every reference to a moved name through its old module that survived: none, or the split failed."""
    out = []
    for path in files:
        if path.suffix != ".py":
            continue
        text = writes.get(path) or path.read_text(encoding="utf-8")
        if not any(m.rpartition(".")[2] in text for m in MOVED):
            continue
        src = Source(path, text)
        package = _module_of(repo, path)
        aliases = {}
        for node in ast.walk(src.tree):
            if isinstance(node, ast.ImportFrom):
                mod = _resolve_from(node, package)
                if mod in MOVED:
                    out += [f"{path}:{node.lineno}: from {mod} import {a.name}" for a in node.names if a.name in MOVED[mod]]
                for a in node.names:
                    if f"{mod}.{a.name}" in MOVED:
                        aliases[a.asname or a.name] = f"{mod}.{a.name}"
            elif isinstance(node, ast.Import):
                aliases.update({a.asname or a.name: a.name for a in node.names if a.name in MOVED})
        for node in ast.walk(src.tree):
            if isinstance(node, ast.Attribute) and _dotted(node.value) in aliases and node.attr in MOVED[aliases[_dotted(node.value)]]:
                out.append(f"{path}:{node.lineno}: {_dotted(node.value)}.{node.attr}")
        for mod, ns in MOVED.items():
            if path == repo / REGISTRY / f"{mod.rpartition('.')[2]}.py":
                out += [f"{path}: still defines {n}" for n in ns if n in _top_level(src.tree)]
    return out


# --- invariants ------------------------------------------------------------------------------------------------------------------

def fact_modules(repo: Path) -> list[str]:
    """The modules whose Definitions the facts cover: the Definition modules present and `hopper` (left, attention-bearing)."""
    present = [m for m in OPTIONAL_DEFINITION_MODULES if (repo / REGISTRY / f"{m}.py").exists()]
    return [*DEFINITION_MODULES, *present, "hopper"]


def family_definitions(repo: Path) -> dict[str, str]:
    """``id -> module`` for every Definition a fact module binds at top level itself (not one it imports)."""
    import importlib

    from verity.ml.circuits.defs import FunctionDefinition, SpecializedDefinition

    out = {}
    for m in fact_modules(repo):
        mod = importlib.import_module(f"{PKG}.{m}")
        tree = ast.parse((repo / REGISTRY / f"{m}.py").read_text(encoding="utf-8"))
        for name in _top_level(tree):
            obj = getattr(mod, name, None)
            if isinstance(obj, FunctionDefinition):
                out[obj.base_id if isinstance(obj, SpecializedDefinition) else obj.id] = m
    return dict(sorted(out.items()))


def digest(fn) -> str:
    """The canonical descriptor digest of ``fn``: its Program's (`verity.ml.circuits.codec.program_digest`), a primitive's entry's."""
    from verity.ml.circuits.codec import canonical_json, definition_entry, program_digest
    from verity.ml.circuits.defs import PrimitiveDefinition
    from verity.ml.circuits.program import Program

    if isinstance(fn, PrimitiveDefinition):
        return hashlib.sha512(canonical_json(definition_entry(fn))).hexdigest()
    return program_digest(Program(fn))


def family_of(fn) -> str:
    return getattr(fn, "base_id", None) or fn.id


def served_bindings() -> dict[str, object]:
    """`targets.attention_spec` on the registered targets at served shapes: the bindings the vocabulary computes (`MASKED_FROM`)."""
    from verity_vllm.program.frontend.target_profile import TargetProfile
    from verity_vllm.program.frontend import targets as TGT

    profiles = {"ada": TargetProfile(), "h100-fa3": TargetProfile(compute_capability=(9, 0), num_sms=132, flash_attn_version=3),
                "h100-guard": TargetProfile(compute_capability=(9, 0), num_sms=132, flash_attn_version=3,
                                            fa3_construction="guard-every-block"),
                "sm120-fa2": TargetProfile(compute_capability=(12, 0), num_sms=188)}
    #: (T, max_seqlen_q, row, own_rows): a first decode, a long decode, a prefill's last and first rows, a chunked prefill's row
    rows = [(1, 1, 0, 1), (300, 1, 0, 1), (130, 130, 129, 130), (130, 130, 0, 130), (300, 17, 16, 17)]
    out = {}
    for pname, prof in profiles.items():
        for NH, KVH, D in ((12, 2, 128), (8, 1, 64)):
            for T, sq, row, own in rows:
                for cap in (None, 50.0):
                    for scale_input in (False, True):
                        label = f"{pname}/NH={NH},KVH={KVH},D={D}/T={T},sq={sq},row={row},own={own}/cap={cap}/scale_input={scale_input}"
                        try:
                            out[label] = TGT.attention_spec(prof, T, NH, KVH, D, sq, row, own, cap=cap, scale_input=scale_input)
                        except (TGT.UnregisteredTarget, KeyError, ValueError) as e:
                            out[label] = f"{type(e).__name__}: {e}"
    return out


def facts(repo: Path, suite: dict | None = None, pins: dict | None = None) -> dict:
    """What the split must not change: the family's Definitions with each one's id and digest at every circuit-check binding, pin
    and served binding, and circuit-check's whole task list.  ``suite`` / ``pins``: `circuit_check.targets.suite()` and
    `circuit_check.checks.load_pins()` when the caller has them already (about a minute to build)."""
    from circuit_check import checks as CK
    from circuit_check import targets as TG
    from verity.ml.circuits.codec import descriptor_id

    TG.load_registries()
    fam = family_definitions(repo)
    suite = TG.suite() if suite is None else suite
    pins = CK.load_pins() if pins is None else pins
    bound = {fid: fn for fid, fn in suite.items() if family_of(fn) in fam}
    pinned = {}
    for key, pin in pins.get("definitions", {}).items():
        if key.split("{")[0] in fam:
            pinned[key] = pin
            if key not in bound:
                bound[key] = TG.definition(key)
    served = {}
    for label, fn in served_bindings().items():
        if isinstance(fn, str):
            served[label] = fn
        else:
            served[label] = fn.id
            bound.setdefault(fn.id, fn)
    definitions = {fid: {"descriptor_id": descriptor_id(fn), "digest": digest(fn)} for fid, fn in sorted(bound.items())}
    templates = [[label, sts] for label, _sub, sts in TG.templates()]
    return {"family": fam, "definitions": definitions, "served": served, "pins": pinned,
            "tasks": {"definitions": sorted(suite), "templates": templates,
                      "pins_sha256": hashlib.sha256(json.dumps(pins, sort_keys=True).encode()).hexdigest()}}


def _facts_in_fresh_process(repo: Path) -> dict:
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "facts.json"
        r = subprocess.run([sys.executable, __file__, "--repo", str(repo), "facts", "--out", str(out)], cwd=repo)
        if r.returncode:
            raise SplitError("the facts did not build (run under `uv run` from the workspace, so circuit_check imports)")
        return json.loads(out.read_text())


def compare(before: dict, after: dict) -> list[str]:
    diffs = []
    for key in ("family", "definitions", "served", "pins", "tasks"):
        a, b = before.get(key), after.get(key)
        if a == b:
            continue
        if isinstance(a, dict) and isinstance(b, dict):
            for k in sorted(set(a) | set(b)):
                if a.get(k) != b.get(k):
                    diffs.append(f"{key}/{k}: {a.get(k)!r} -> {b.get(k)!r}")
        else:
            diffs.append(f"{key}: changed")
    return diffs


# --- driver ----------------------------------------------------------------------------------------------------------------------

#: what the rewrite never edits: the migrations and their tests name the old modules on purpose
EXCLUDED = (".venv/", "integrations/vllm/tools/migrate/", "integrations/vllm/tests/migrate/")


def tracked_files(repo: Path) -> list[Path]:
    out = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=repo, check=True,
                         capture_output=True).stdout.decode().split("\0")
    return [repo / p for p in out if p and not p.startswith(EXCLUDED) and (repo / p).is_file()]


def plan(repo: Path) -> tuple[dict[Path, str], list[str]]:
    """Every file the split writes, with its new text, and what it did; raises `SplitError` on what it refuses."""
    writes: dict[Path, str] = {}
    log = move(repo, writes)
    files = tracked_files(repo)
    files += [p for p in writes if p not in files]
    log += [f"rewrote imports in {f}" for f in rewrite_consumers(repo, files, writes)]
    log += [f"rewrote mentions in {f}" for f in rewrite_mentions(repo, files, writes)]
    for path, text in writes.items():
        if path.suffix == ".py":
            try:
                ast.parse(text, filename=str(path))
            except SyntaxError as e:
                raise SplitError(f"{path}: the rewrite does not parse ({e})") from e
    left = leftovers(repo, files, writes)
    if left:
        raise SplitError("references to a moved name through its old module remain:\n  " + "\n  ".join(left))
    return writes, log


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", type=Path, default=None, help="the repository root (default: this checkout's)")
    ap.add_argument("--check", action="store_true", help="exit 1 unless the split is applied; write nothing")
    ap.add_argument("--no-verify", action="store_true", help="skip the before/after facts (about three minutes)")
    sub = ap.add_subparsers(dest="cmd")
    fp = sub.add_parser("facts", help="write the invariants' facts of this tree")
    fp.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    repo = (a.repo or Path(__file__).resolve().parents[4]).resolve()
    if not (repo / REGISTRY).is_dir():
        print(f"split_attention: {repo / REGISTRY} is not a directory: not a Verity checkout", file=sys.stderr)
        return 2
    if a.cmd == "facts":
        a.out.write_text(json.dumps(facts(repo), indent=1, sort_keys=True) + "\n")
        return 0
    try:
        writes, log = plan(repo)
    except SplitError as e:
        print(f"split_attention: refused: {e}", file=sys.stderr)
        return 1
    if a.check:
        for p in sorted(writes):
            print(f"would write {p.relative_to(repo)}")
        return 1 if writes else 0
    if not writes:
        print("split_attention: already applied; nothing to do")
        return 0
    before = None if a.no_verify else _facts_in_fresh_process(repo)
    originals = {p: (p.read_text(encoding="utf-8") if p.exists() else None) for p in writes}
    for p, text in writes.items():
        p.write_text(text, encoding="utf-8")
    if before is not None:
        try:
            diffs = compare(before, _facts_in_fresh_process(repo))
        except SplitError as e:
            diffs = [str(e)]
        if diffs:
            for p, text in originals.items():
                if text is None:
                    p.unlink()
                else:
                    p.write_text(text, encoding="utf-8")
            print("split_attention: the split changed what it must not; every file restored:\n  " + "\n  ".join(diffs[:50]),
                  file=sys.stderr)
            return 1
    for line in log:
        print(line)
    print(f"split_attention: wrote {len(writes)} files" + ("" if a.no_verify else "; ids, digests and circuit-check's tasks unchanged"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
