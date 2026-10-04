"""Split the attention family's registry modules into Definitions, twins and vocabulary, and rewrite every import of a moved name.

    python integrations/vllm/tools/migrate/split_attention.py               # split, rewrite, verify (two circuit-check passes)
    python integrations/vllm/tools/migrate/split_attention.py --dry-run     # what it would move, and whether it can
    python integrations/vllm/tools/migrate/split_attention.py --no-verify   # split and rewrite without the before/after passes

The family: `fa2_check_inf`, `fa2_softcap`, `fa3_check_inf`, `boolean_attention`, `boolean_softcap`, `hopper`, `mla`, `mla_capture`,
`mla_check`, and #1130's `dsa`, `boolean_dsa`, `sparse_mla`, `boolean_sparse_mla` where they exist. Definitions stay in
`verity_vllm/program/registry/<module>.py`, the catalog's side of the later move. What leaves goes to the integration:

* twins (reference twins, difftests, capture and check helpers) to `verity_vllm/program/twins/<module>.py`;
* vocabulary (what a serving role or a model binds: launch geometry to statics, a model's statics, capture-map names, the
  conformance a profile's evidence sets) to `verity_vllm/program/vocabulary/<module>.py`.

`MOVES` lists every top-level name that leaves a Definition module, `MODULE_MOVES` every module that leaves the registry whole.
A moved name keeps its source text (with the comments directly above it), and its new module imports what it reads. Every import
of a moved name in the repository is rewritten: `from registry.M import X` names the new module, and `A.X` for an `A` bound to
`registry.M` becomes `A_<kind>.X` beside a new `from verity_vllm.program.<kind> import M as A_<kind>` (`_vocab`, `_twins`). A
whole module's dotted name and path are replaced in every tracked text file. A lint allowlist entry for a moved symbol moves
with it.

Verification (unless `--no-verify`): circuit-check's task list (kind, target id with its statics), the Definition each target
and pin resolves to, and the descriptor entry digest of every Definition the suite reaches are computed in a fresh process from
the tree before the split and after it, and must be equal. The family's slice of the before pass is written to
`tools/circuit_check/tests/split_attention_digests.json`, which `tools/circuit_check/tests/test_split_attention.py` regenerates.

Rerunnable: on a tree where the split is done it says so and exits 0, and where #1130's modules arrived after it, it splits only
those. It exits 1 without writing when a listed name or module is missing or would collide, when a moved statement binds a kept
name too, when what stays reads a moved name, or when the work tree has uncommitted changes (`--allow-dirty`).
"""
from __future__ import annotations

import argparse
import ast
import builtins
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import tokenize
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
PKG_DIR = "integrations/vllm"
REGISTRY = "verity_vllm.program.registry"
KINDS = {"twins": "verity_vllm.program.twins", "vocabulary": "verity_vllm.program.vocabulary"}
SUFFIX = {"twins": "twins", "vocabulary": "vocab"}

FAMILY = ("fa2_check_inf", "fa2_softcap", "fa3_check_inf", "boolean_attention", "boolean_softcap", "hopper", "mla")
#: #1130's modules: split where present
OPTIONAL = ("dsa", "boolean_dsa", "sparse_mla", "boolean_sparse_mla")

#: module -> kind -> the top-level names that leave it
MOVES: dict[str, dict[str, tuple[str, ...]]] = {
    "fa2_check_inf": {"vocabulary": ("KBLOCK_M", "fa2_masked_from")},
    "fa3_check_inf": {"vocabulary": ("fa3_masked_from",)},
    "hopper": {"twins": ("_hopper_total", "HOPPER_WGMMA_K", "HOPPER_WGMMA_WIDTH", "HOPPER_WGMMA_ZERO_EXPONENT", "hopper_wgmma_dot16_total",
                         "_fn_key", "_static_repr", "structural_diff"),
               "vocabulary": ("CALL_NAMES", "NAME_MAP", "set_conformance", "bindings")},
    "mla": {"vocabulary": ("GLM47_FLASH",)},
    "sparse_mla": {"vocabulary": ("SPARSE_MLA_GLM53",)},
    "boolean_sparse_mla": {"vocabulary": ("SPARSE_MLA_GLM53",)},
}
#: module -> kind: modules that define no Definition and leave the registry whole
MODULE_MOVES: dict[str, str] = {"mla_check": "twins", "mla_capture": "twins"}

DOCS: dict[tuple[str, str], str] = {
    ("vocabulary", "fa2_check_inf"): "FA2's masking schedule: the lowest key block a splitkv launch runs with `Check_inf`, the `MASKED_FROM` static\n"
                                     "a serving row binds `registry.fa2_check_inf`'s `Attention_v5` / `Attention_v11` at.",
    ("vocabulary", "fa3_check_inf"): "FA3's masking schedule: the tile's `n_block_min_causal_local_mask`, the `MASKED_FROM` static a serving row\n"
                                     "binds `registry.fa3_check_inf`'s `Attention_v4` at.",
    ("twins", "hopper"): "The Hopper k16 step's reference (core's `verity.ml.tc.total` step) and the node-for-node comparison behind\n"
                         "`registry.hopper`'s \"same structure, different bindings\".",
    ("vocabulary", "hopper"): "The Hopper profile's names over `registry.hopper`'s Definitions: TA1's call names to the Hopper ones (for capture\n"
                              "mappings written against TA1's), the binding table, and the conformance its evidence sets.",
    ("vocabulary", "mla"): "GLM-4.7-Flash's statics for `registry.mla`'s Definitions.",
    ("vocabulary", "sparse_mla"): "GLM-5.3's statics for `registry.sparse_mla`'s Definitions.",
    ("vocabulary", "boolean_sparse_mla"): "GLM-5.3's statics for `registry.boolean_sparse_mla`'s Definitions.",
}
PACKAGE_DOCS = {
    "twins": "Twins of the registry's Definitions: numpy, torch and CUDA reference twins, difftests, capture and check helpers.  Each\n"
             "module is named after the `verity_vllm.program.registry` module whose Definitions it twins; no Definition module imports one.",
    "vocabulary": "What a serving role or a model binds from the registry's Definitions: launch geometry to statics, a model's statics,\n"
                  "capture-map names.  Each module is named after the `verity_vllm.program.registry` module whose Definitions it binds; no\n"
                  "Definition module imports one.",
}

PIN = "tools/circuit_check/tests/split_attention_digests.json"
ALLOWLISTS = "integrations/vllm/tests/lint/allowlists"
#: files the rewrite leaves alone: this script and its tests name the old modules on purpose
EXCLUDE = re.compile(r"(^|/)(tools/migrate/|test_split_attention\w*\.py$|split_attention_digests\.json$)")
SKIP_DIRS = frozenset({".git", ".venv", "node_modules", ".lake", "__pycache__", "target", ".mypy_cache", ".pytest_cache"})
TEXT_SUFFIXES = frozenset({".py", ".md", ".toml", ".json", ".sh", ".txt", ".cfg", ".yaml", ".yml", ".ini"})
#: the source dirs of the workspace's packages, so a snapshot process imports this tree whichever venv runs it
SOURCE_DIRS = ("packages/verity/src", "integrations/vllm", "tools/circuit_check/src", "protocols/pouw", "backends/flock/python",
               "backends/numerical/python", "protocols/sampled_proofs", "protocols/one_stage", "protocols/pous",
               "protocols/network_warden", "backends/sp1/python", "tools/research/src", "tools/cluster/src")
WIDTH = 130


class SplitError(Exception):
    """The tree is not one this script can split; nothing was written."""


def mod_path(root: Path, dotted: str) -> Path:
    return root / PKG_DIR / (dotted.replace(".", "/") + ".py")


# --- source text ------------------------------------------------------------------------------------------------------------

def bound_names(stmt: ast.stmt) -> set[str]:
    """The names a top-level statement binds by definition or plain assignment (imports are never moved)."""
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {stmt.name}
    if isinstance(stmt, ast.Assign):
        return {n.id for t in stmt.targets for n in ast.walk(t) if isinstance(n, ast.Name)}
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        return {stmt.target.id}
    return set()


def statement_lines(lines: list[str], stmt: ast.stmt) -> tuple[int, int]:
    """1-based inclusive lines of ``stmt`` with its decorators and the comment lines directly above it."""
    start = min([stmt.lineno, *(d.lineno for d in getattr(stmt, "decorator_list", []))])
    while start > 1 and lines[start - 2].lstrip().startswith("#"):
        start -= 1
    return start, stmt.end_lineno


def loads(node: ast.AST) -> set[str]:
    """Every name ``node`` reads (locals that share a module-level name count too)."""
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}


def local_names(node: ast.AST) -> set[str]:
    """Names ``node`` binds inside itself: assignments, parameters, handlers, imports."""
    out = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    out |= {a.arg for a in ast.walk(node) if isinstance(a, ast.arg)}
    out |= {h.name for h in ast.walk(node) if isinstance(h, ast.ExceptHandler) and h.name}
    out |= {a.asname or a.name.split(".")[0] for st in ast.walk(node) if isinstance(st, (ast.Import, ast.ImportFrom)) for a in st.names}
    return out


def binds(st: ast.Import | ast.ImportFrom, a: ast.alias) -> str:
    return a.asname or (a.name if isinstance(st, ast.ImportFrom) else a.name.split(".")[0])


def top_imports(tree: ast.Module) -> dict[str, ast.stmt]:
    """Top-level import statements by the name each binds."""
    return {binds(st, a): st for st in tree.body if isinstance(st, (ast.Import, ast.ImportFrom)) for a in st.names}


def alias_text(a: ast.alias) -> str:
    return a.name if not a.asname else f"{a.name} as {a.asname}"


def from_import(module: str, aliases: list[str], indent: int = 0) -> str:
    """``from module import a, b as c`` for a statement at column ``indent``; past ``WIDTH`` the names wrap in parentheses,
    each continuation line aligned under the first name."""
    head = f"from {module} import "
    if indent + len(head) + len(", ".join(aliases)) <= WIDTH:
        return head + ", ".join(aliases)
    pad = " " * (indent + len(head) + 1)
    lines, cur, width = [], head + "(", indent
    for i, a in enumerate(aliases):
        piece = a + ("," if i < len(aliases) - 1 else ")")
        if cur.endswith("(") or width + len(cur) + 1 + len(piece) <= WIDTH:
            cur += piece if cur.endswith("(") else " " + piece
        else:
            lines.append(cur)
            cur, width = pad + piece, 0
    return "\n".join([*lines, cur])


def offsets(lines: list[str]) -> list[int]:
    out = [0]
    for ln in lines:
        out.append(out[-1] + len(ln))
    return out


def char_span(lines: list[str], starts: list[int], node: ast.AST) -> tuple[int, int]:
    """Character offsets of ``node`` in the text ``lines`` make up (``ast`` columns are UTF-8 byte offsets)."""
    def at(line: int, col: int) -> int:
        return starts[line - 1] + len(lines[line - 1].encode()[:col].decode())
    return at(node.lineno, node.col_offset), at(node.end_lineno, node.end_col_offset)


def apply_edits(text: str, edits: list[tuple[int, int, str]]) -> str:
    for a, b, new in sorted(edits, key=lambda e: (e[0], e[1]), reverse=True):
        text = text[:a] + new + text[b:]
    return text


def dunder_all(tree: ast.Module) -> ast.Assign | None:
    for st in tree.body:
        if (isinstance(st, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in st.targets)
                and isinstance(st.value, (ast.List, ast.Tuple))):
            return st
    return None


def drop_from_all(text: str, names: set[str]) -> str:
    """``text`` with ``names`` taken out of a literal ``__all__``."""
    tree = ast.parse(text)
    st = dunder_all(tree)
    if st is None:
        return text
    lines = text.splitlines(keepends=True)
    starts = offsets(lines)
    edits = []
    elts = st.value.elts
    for i, e in enumerate(elts):
        if isinstance(e, ast.Constant) and e.value in names:
            a, b = char_span(lines, starts, e)
            after = re.match(r"\s*,[ \t]*", text[b:])
            if after:
                b += after.end()
            else:
                before = re.search(r",\s*$", text[:a])
                if before:
                    a = before.start()
            edits.append((a, b, ""))
    out = apply_edits(text, edits)
    return re.sub(r"\n[ \t]+\n", "\n", out) if edits else out


def prune_imports(text: str, dead: set[str]) -> str:
    """``text`` with the top-level import aliases that bind ``dead`` removed; a statement left with none goes, line and all."""
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    starts = offsets(lines)
    edits = []
    for st in tree.body:
        if not isinstance(st, (ast.Import, ast.ImportFrom)):
            continue
        gone = [a for a in st.names if binds(st, a) in dead]
        if not gone:
            continue
        rest = [a for a in st.names if a not in gone]
        a, b = char_span(lines, starts, st)
        if not rest:
            a, b = starts[st.lineno - 1], starts[st.end_lineno]
            new = ""
        elif isinstance(st, ast.Import):
            new = f"import {', '.join(alias_text(x) for x in rest)}"
        else:
            new = from_import(st.module, [alias_text(x) for x in rest], st.col_offset)
        edits.append((a, b, new))
    return apply_edits(text, edits)


def remove_lines(text: str, spans: list[tuple[int, int]]) -> str:
    """``text`` without the 1-based inclusive line ``spans``, with at most two blank lines where they were and no comment block
    left at the end that the text did not end with."""
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    i = 1
    for a, b in sorted(spans):
        out += lines[i - 1:a - 1]
        before = 0
        while out and out[-1].strip() == "":
            out.pop()
            before += 1
        i = b + 1
        after = 0
        while i <= len(lines) and lines[i - 1].strip() == "":
            i += 1
            after += 1
        out += ["\n"] * min(2, max(before, after))
    out += lines[i - 1:]
    ended_with_comment = bool(lines) and lines[-1].lstrip().startswith("#")
    while out and (out[-1].strip() == "" or (out[-1].lstrip().startswith("#") and not ended_with_comment)):
        out.pop()
    return "".join(out).rstrip("\n") + "\n"


# --- the plan ---------------------------------------------------------------------------------------------------------------

@dataclass
class Plan:
    root: Path
    moves: dict[str, dict[str, tuple[str, ...]]] = field(default_factory=dict)  # pending: module -> kind -> names
    module_moves: dict[str, str] = field(default_factory=dict)                   # pending: module -> kind
    done: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not self.moves and not self.module_moves


def top_level(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return set().union(*(bound_names(s) for s in ast.parse(path.read_text(encoding="utf-8")).body))


def plan(root: Path, moves=MOVES, module_moves=MODULE_MOVES, optional=OPTIONAL) -> Plan:
    """What is left to do on ``root``'s tree; `SplitError` where the tree is neither before nor after the split."""
    p = Plan(root)
    for mod, kinds in moves.items():
        src = mod_path(root, f"{REGISTRY}.{mod}")
        if not src.exists():
            if mod in optional and not any(mod_path(root, f"{KINDS[k]}.{mod}").exists() for k in kinds):
                p.skipped.append(f"{REGISTRY}.{mod} (not on this tree)")
                continue
            raise SplitError(f"{REGISTRY}.{mod} is missing")
        here = top_level(src)
        for kind, names in kinds.items():
            dest = mod_path(root, f"{KINDS[kind]}.{mod}")
            if set(names) <= here and not dest.exists():
                p.moves.setdefault(mod, {})[kind] = names
            elif not set(names) & here and set(names) <= top_level(dest):
                p.done.append(f"{REGISTRY}.{mod} -> {KINDS[kind]}.{mod}")
            else:
                raise SplitError(f"{REGISTRY}.{mod} -> {KINDS[kind]}.{mod} is neither before nor after the split: still there "
                                 f"{sorted(set(names) & here)}, not there {sorted(set(names) - here)}, {dest.name} exists: {dest.exists()}")
    for mod, kind in module_moves.items():
        src, dest = mod_path(root, f"{REGISTRY}.{mod}"), mod_path(root, f"{KINDS[kind]}.{mod}")
        if src.exists() and not dest.exists():
            p.module_moves[mod] = kind
        elif dest.exists() and not src.exists():
            p.done.append(f"{REGISTRY}.{mod} -> {KINDS[kind]}.{mod}")
        else:
            raise SplitError(f"{REGISTRY}.{mod} -> {KINDS[kind]}.{mod}: the first exists: {src.exists()}, the second: {dest.exists()}")
    return p


# --- moving names out of a Definition module -------------------------------------------------------------------------------

def split_module(text: str, mod: str, kinds: dict[str, tuple[str, ...]], external: set[str] = frozenset()) -> tuple[str, dict[str, str]]:
    """``text`` without the moved names, and each kind's new module. ``external``: the module's names other files read, which
    stay importable from it."""
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    moved = {n: k for k, ns in kinds.items() for n in ns}
    segments: dict[str, list[tuple[int, int, ast.stmt]]] = {k: [] for k in kinds}
    for st in tree.body:
        b = bound_names(st)
        if not b & moved.keys():
            continue
        ks = {moved.get(n) for n in b}
        if None in ks or len(ks) > 1:
            raise SplitError(f"{REGISTRY}.{mod}:{st.lineno} binds {sorted(b)}, of which only {sorted(b & moved.keys())} would move together")
        segments[ks.pop()].append((*statement_lines(lines, st), st))
    found = {n for segs in segments.values() for *_, s in segs for n in bound_names(s)}
    if found != set(moved):
        raise SplitError(f"{REGISTRY}.{mod}: no top-level statement binds {sorted(set(moved) - found)}")

    rest = remove_lines(text, [(a, b) for segs in segments.values() for a, b, _ in segs])
    rest = drop_from_all(rest, set(moved))
    rest_tree = ast.parse(rest)
    if loads(rest_tree) & moved.keys():
        raise SplitError(f"{REGISTRY}.{mod}: what stays reads {sorted(loads(rest_tree) & moved.keys())}, which would move")

    imports = top_imports(tree)
    kept = set().union(*(bound_names(s) for s in rest_tree.body)) | set(top_imports(rest_tree))
    out: dict[str, str] = {}
    read_by_moved: set[str] = set()
    for kind, segs in segments.items():
        if not segs:
            continue
        mine = {n for n, k in moved.items() if k == kind}
        reads = set().union(*(loads(s) for *_, s in segs)) - mine - set(dir(builtins))
        locals_ = set().union(*(local_names(s) for *_, s in segs))
        unresolved = sorted(n for n in reads if n not in imports and n not in kept and n not in moved and n not in locals_)
        if unresolved:
            raise SplitError(f"{REGISTRY}.{mod} -> {kind}: the moved code reads {unresolved}, which the module neither defines nor imports")
        reads = {n for n in reads if n in imports or n in kept or n in moved}
        read_by_moved |= reads
        stmts = []
        for st in tree.body:
            if isinstance(st, ast.ImportFrom) and st.module == "__future__":
                continue
            if isinstance(st, (ast.Import, ast.ImportFrom)):
                names = [a for a in st.names if binds(st, a) in reads]
                if not names:
                    continue
                if isinstance(st, ast.Import):
                    stmts += [f"import {alias_text(a)}" for a in names]
                else:
                    stmts.append(from_import(st.module, [alias_text(a) for a in names]))
        local = sorted(n for n in reads if n not in imports and n not in moved)
        if local:
            stmts.append(from_import(f"{REGISTRY}.{mod}", local))
        for k2 in sorted({moved[n] for n in reads if n in moved}):
            stmts.append(from_import(f"{KINDS[k2]}.{mod}", sorted(n for n in reads if moved.get(n) == k2)))
        doc = DOCS.get((kind, mod), f"Moved out of `{REGISTRY}.{mod}`.")
        body, prev = "", None
        for a, b, _ in sorted(segs, key=lambda t: t[0]):
            gap = lines[prev:a - 1] if prev is not None else None
            body += ("" if gap is None else "".join(gap) if all(x.strip() == "" for x in gap) else "\n\n") + "".join(lines[a - 1:b])
            prev = b
        out[kind] = "\n".join([f'"""{doc}\n"""', "", "from __future__ import annotations", "", *stmts, *([""] if stmts else []), "",
                               body.rstrip("\n")]) + "\n"

    still = loads(rest_tree) | {n.value.id for n in ast.walk(rest_tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
    st_all = dunder_all(rest_tree)
    listed = {e.value for e in st_all.value.elts if isinstance(e, ast.Constant)} if st_all else set()
    dead = {n for n in read_by_moved if n in imports and n not in still and n not in listed and n not in external}
    return (prune_imports(rest, dead) if dead else rest), out


def unused_imports(text: str) -> set[str]:
    tree = ast.parse(text)
    return {n for n, st in top_imports(tree).items() if not (isinstance(st, ast.ImportFrom) and st.module == "__future__")} - loads(tree)


# --- rewriting the repository ----------------------------------------------------------------------------------------------

def repo_files(root: Path) -> list[Path]:
    r = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=root, capture_output=True)
    paths = [root / p for p in r.stdout.decode().split("\0") if p] if r.returncode == 0 else [p for p in root.rglob("*") if p.is_file()]
    return sorted(p for p in set(paths) if p.is_file() and p.suffix in TEXT_SUFFIXES and not SKIP_DIRS & set(p.relative_to(root).parts)
                  and not EXCLUDE.search(p.relative_to(root).as_posix()))


def external_reads(files: dict[Path, str], src: str, own: Path) -> set[str]:
    """The names of module ``src`` that files other than ``own`` read (`from src import x`, or ``A.x`` for an ``A`` bound to it)."""
    parent, leaf = src.rsplit(".", 1)
    out: set[str] = set()
    for f, text in files.items():
        if f == own or f.suffix != ".py" or leaf not in text:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        aliases = set()
        for st in ast.walk(tree):
            if isinstance(st, ast.ImportFrom) and st.level == 0 and st.module == src:
                out |= {a.name for a in st.names}
            elif isinstance(st, ast.ImportFrom) and st.level == 0 and st.module == parent:
                aliases |= {a.asname or a.name for a in st.names if a.name == leaf}
            elif isinstance(st, ast.Import):
                aliases |= {a.asname for a in st.names if a.name == src and a.asname}
        out |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in aliases}
    return out


FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)


def scopes(tree: ast.Module) -> dict[ast.AST, ast.AST]:
    """Each node's scope: the innermost function holding it, or the module (a ``def``'s own name is in the scope around it)."""
    out: dict[ast.AST, ast.AST] = {tree: tree}

    def visit(node: ast.AST, scope: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            out[child] = scope
            visit(child, child if isinstance(child, FUNCS) else scope)
    visit(tree, tree)
    return out


def bindings(tree: ast.Module, scope: dict[ast.AST, ast.AST]) -> dict[tuple[ast.AST, str], list[ast.AST]]:
    """(scope, name) -> the nodes that bind the name there (a comprehension's targets count as its function's)."""
    out: dict[tuple[ast.AST, str], list[ast.AST]] = {}
    for n in ast.walk(tree):
        names: list[str] = []
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            names = [n.id]
        elif isinstance(n, ast.arg):
            names = [n.arg]
        elif isinstance(n, ast.ExceptHandler) and n.name:
            names = [n.name]
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [n.name]
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            names = [binds(n, a) for a in n.names]
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            names = list(n.names)
        for name in names:
            out.setdefault((scope[n], name), []).append(n)
    return out


def realign_comments(text: str, lines: list[str], starts: list[int], grown: dict[int, int]) -> list[tuple[int, int, str]]:
    """Edits that keep each grown line's trailing comment in its column, where the spaces before it leave room."""
    if not grown:
        return []
    comments = {}
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.COMMENT and tok.start[0] in grown:
            comments[tok.start[0]] = tok.start[1]
    edits = []
    for line, col in comments.items():
        pad = len(lines[line - 1][:col]) - len(lines[line - 1][:col].rstrip(" "))
        cut = min(grown[line], pad - 2)
        if cut > 0:
            at = starts[line - 1] + col - pad
            edits.append((at, at + cut, ""))
    return edits


def rewrite_source(text: str, moved: dict[str, dict[str, str]], whole: dict[str, str], where: str = "") -> str:
    """``text`` with every reference to a moved name or a moved module rewritten.  ``moved``: source module -> name -> destination
    module; ``whole``: source module -> destination module.  ``A.X`` is rewritten where Python's scoping resolves ``A`` to an
    import of a module with moved names; a scope that binds ``A`` that way and another way too is refused."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text
    lines = text.splitlines(keepends=True)
    starts = offsets(lines)
    edits: list[tuple[int, int, str]] = []
    scope = scopes(tree)
    bound = bindings(tree, scope)
    aliases: dict[tuple[ast.AST, str], str] = {}               # (scope, local name) -> the module with moved names it binds
    binders: dict[tuple[ast.AST, str], list[ast.stmt]] = {}    # the import statements binding it
    parents = {m.rsplit(".", 1)[0] for m in [*moved, *whole]}

    def replace(st: ast.stmt, parts: list[str]) -> None:
        a, b = char_span(lines, starts, st)
        if "#" in text[a:b]:
            raise SplitError(f"{where}:{st.lineno}: a comment inside an import this script would rewrite")
        edits.append((a, b, ("\n" + " " * st.col_offset).join(parts)))

    def bind_alias(local: str, src: str, st: ast.stmt) -> None:
        key = (scope[st], local)
        if aliases.setdefault(key, src) != src:
            raise SplitError(f"{where}:{st.lineno}: {local!r} names both {aliases[key]} and {src}")
        binders.setdefault(key, []).append(st)

    for st in ast.walk(tree):
        if isinstance(st, ast.ImportFrom) and st.level == 0 and st.module in whole:
            replace(st, [from_import(whole[st.module], [alias_text(a) for a in st.names], st.col_offset)])
        elif isinstance(st, ast.ImportFrom) and st.level == 0 and st.module in moved:
            groups: dict[str, list[str]] = {}
            kept = []
            for a in st.names:
                if a.name == "*":
                    raise SplitError(f"{where}:{st.lineno}: `from {st.module} import *`")
                if a.name in moved[st.module]:
                    groups.setdefault(moved[st.module][a.name], []).append(alias_text(a))
                else:
                    kept.append(alias_text(a))
            if groups:
                replace(st, ([from_import(st.module, kept, st.col_offset)] if kept else [])
                        + [from_import(d, ns, st.col_offset) for d, ns in groups.items()])
        elif isinstance(st, ast.ImportFrom) and st.level == 0 and st.module in parents:
            groups, kept = {}, []
            for a in st.names:
                sub = f"{st.module}.{a.name}"
                if sub in whole:
                    groups.setdefault(whole[sub].rsplit(".", 1)[0], []).append(alias_text(a))
                    continue
                kept.append(alias_text(a))
                if sub in moved:
                    bind_alias(a.asname or a.name, sub, st)
            if groups:
                replace(st, ([from_import(st.module, kept, st.col_offset)] if kept else [])
                        + [from_import(d, ns, st.col_offset) for d, ns in groups.items()])
        elif isinstance(st, ast.Import):
            for a in st.names:
                if a.name in moved:
                    if not a.asname:
                        raise SplitError(f"{where}:{st.lineno}: `import {a.name}` without an alias")
                    bind_alias(a.asname, a.name, st)

    def resolve(node: ast.AST, name: str) -> ast.AST | None:
        s = scope[node]
        while (s, name) not in bound:
            if s is tree:
                return None
            s = scope[s]
        return s

    used: dict[tuple[ast.AST, str], set[str]] = {}
    grown: dict[int, int] = {}                                  # line -> characters the alias renames add to it
    locals_ = {local for _, local in aliases}
    for n in ast.walk(tree):
        if not (isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in locals_):
            continue
        key = (resolve(n, n.value.id), n.value.id)
        if key not in aliases or n.attr not in moved[aliases[key]]:
            continue
        others = [b for b in bound[key] if b not in binders[key]]
        if others:
            raise SplitError(f"{where}:{n.lineno}: {n.value.id!r} reads moved {n.attr!r}, but its scope binds {n.value.id!r} on lines "
                             f"{sorted({b.lineno for b in bound[key]})}, not only by importing {aliases[key]}")
        src = aliases[key]
        kind = next(k for k, pkg in KINDS.items() if moved[src][n.attr] == f"{pkg}.{src.rsplit('.', 1)[1]}")
        edits.append((*char_span(lines, starts, n.value), f"{n.value.id}_{SUFFIX[kind]}"))
        grown[n.value.lineno] = grown.get(n.value.lineno, 0) + len(SUFFIX[kind]) + 1
        used.setdefault(key, set()).add(kind)
    edits += realign_comments(text, lines, starts, grown)
    if used:
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
            a.asname or a.name for st in ast.walk(tree) if isinstance(st, (ast.Import, ast.ImportFrom)) for a in st.names}
        for key, kinds in used.items():
            local, leaf = key[1], aliases[key].rsplit(".", 1)[1]
            for st in binders[key]:
                end = text.find("\n", char_span(lines, starts, st)[1])
                end = len(text) if end < 0 else end
                add = ""
                for kind in sorted(kinds):
                    new = f"{local}_{SUFFIX[kind]}"
                    if new in names:
                        raise SplitError(f"{where}: {new!r} is already a name there")
                    add += "\n" + " " * st.col_offset + from_import(KINDS[kind], [f"{leaf} as {new}"], st.col_offset)
                edits.append((end, end, add))
    return apply_edits(text, edits) if edits else text


def rewrite_text(text: str, whole: dict[str, str]) -> str:
    """The dotted names and paths of the modules that moved whole, anywhere in a text file."""
    for src, dest in whole.items():
        text = re.sub(rf"\b{re.escape(src)}\b", dest, text)
        text = re.sub(rf"\b{re.escape(src.replace('.', '/'))}\.py\b", dest.replace(".", "/") + ".py", text)
    return text


def rewrite_allowlists(text: str, moved: dict[str, dict[str, str]]) -> str:
    """A lint allowlist with each entry whose symbol moved naming the symbol's new file."""
    by_file = {f"{s.replace('.', '/')}.py": {n: f"{d.replace('.', '/')}.py" for n, d in names.items()} for s, names in moved.items()}

    def entry(m: re.Match) -> str:
        line = m[0]
        f = re.search(r'"file": "([^"]+)"', line)
        s = re.search(r'"symbol": "([^"]+)"', line)
        if f and s and f[1] in by_file and s[1].split(".")[0] in by_file[f[1]]:
            return line.replace(f[0], f'"file": "{by_file[f[1]][s[1].split(".")[0]]}"', 1)
        return line
    return re.sub(r"\{[^{}\n]*\}", entry, text)


# --- applying ---------------------------------------------------------------------------------------------------------------

def apply(p: Plan) -> list[Path]:
    """Perform ``p`` on its tree; the files written or removed."""
    root = p.root
    files = {f: f.read_text(encoding="utf-8") for f in repo_files(root)}
    moved: dict[str, dict[str, str]] = {}
    whole = {f"{REGISTRY}.{m}": f"{KINDS[k]}.{m}" for m, k in p.module_moves.items()}
    new: dict[Path, str] = {}
    for mod, kinds in p.moves.items():
        src = mod_path(root, f"{REGISTRY}.{mod}")
        names = {n for ns in kinds.values() for n in ns}
        rest, outs = split_module(files[src], mod, kinds, external_reads(files, f"{REGISTRY}.{mod}", src) - names)
        new[src] = rest
        for kind, text in outs.items():
            new[mod_path(root, f"{KINDS[kind]}.{mod}")] = text
        moved[f"{REGISTRY}.{mod}"] = {n: f"{KINDS[k]}.{mod}" for k, ns in kinds.items() for n in ns}
    for kind in {*(k for ks in p.moves.values() for k in ks), *p.module_moves.values()}:
        init = mod_path(root, f"{KINDS[kind]}.__init__")
        if not init.exists():
            new[init] = f'"""{PACKAGE_DOCS[kind]}\n"""\n'
    removed = [mod_path(root, s) for s in whole]
    for s, d in whole.items():
        new[mod_path(root, d)] = files.pop(mod_path(root, s))
    files.update(new)

    out: dict[Path, str] = {}
    for f, text in files.items():
        rel = f.relative_to(root).as_posix()
        t = rewrite_source(text, moved, whole, rel) if f.suffix == ".py" else text
        t = rewrite_text(t, whole)
        if rel.startswith(ALLOWLISTS + "/") and f.suffix == ".json":
            t = rewrite_allowlists(t, moved)
        if f in new and f.suffix == ".py" and any(f == mod_path(root, f"{KINDS[k]}.{m}") for m, ks in p.moves.items() for k in ks):
            t = prune_imports(t, unused_imports(t))
        if t != text or f in new:
            out[f] = t
    for f in removed:
        f.unlink()
    for f, t in out.items():
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(t, encoding="utf-8")
    return sorted([*out, *removed])


# --- verification -----------------------------------------------------------------------------------------------------------

def family_modules(root: Path) -> list[str]:
    return [m for m in (*FAMILY, *OPTIONAL) if mod_path(root, f"{REGISTRY}.{m}").exists()]


def owned_ids(root: Path) -> set[str]:
    """The ids of the Definitions the family's modules define: bound at a module's top level (not imported), with a body (or
    evaluator) that is the module's own code or that a core factory (`verity.ml.boolean.trace.definition`,
    `elementwise.specialized`, ...) built for the module."""
    import importlib

    from verity.ir.defs import FunctionDefinition

    core = {id(v) for name, m in list(sys.modules.items()) if name.split(".")[0] == "verity" and m is not None
            for v in vars(m).values() if isinstance(v, FunctionDefinition)}
    out = set()
    for m in family_modules(root):
        mod = importlib.import_module(f"{REGISTRY}.{m}")
        for name in top_level(mod_path(root, f"{REGISTRY}.{m}")):
            v = getattr(mod, name, None)
            if isinstance(v, FunctionDefinition):
                d = getattr(v, "defn", v)
                where = getattr(getattr(d, "body", None) or getattr(d, "evaluate", None), "__module__", "") or ""
                if where == mod.__name__ or (where.split(".")[0] == "verity" and id(v) not in core):
                    out.add(v.id)
    return out


def family_digests(roots: dict, owned: set[str]) -> dict[str, dict]:
    """Each family target's Definition id, descriptor entry digest and, for a specialisation, its Program's canonical descriptor
    digest."""
    from verity.ir.codec import canonical_json, definition_entry, program_digest
    from verity.ir.defs import FunctionDefinition, SpecializedDefinition
    from verity.ir.program import Program

    out = {}
    for label, fn in sorted(roots.items()):
        if isinstance(fn, FunctionDefinition) and (getattr(fn, "base_id", None) or fn.id) in owned:
            digest = None
            if isinstance(fn, SpecializedDefinition):
                try:
                    digest = program_digest(Program(fn))
                except (TypeError, ValueError):
                    digest = None
            out[label] = {"id": fn.id, "entry": hashlib.sha256(canonical_json(definition_entry(fn))).hexdigest(), "program": digest}
    return out


def snapshot(root: Path) -> dict:
    """circuit-check's task list, the Definition each target and pin resolves to, the descriptor entry digest of every Definition
    they reach, and the family's slice: run in a process that imports ``root``'s tree."""
    from circuit_check import checks as CK
    from circuit_check import cli
    from circuit_check import targets as TG
    from verity.ir.codec import canonical_json, definition_entry
    from verity.ir.defs import FunctionDefinition

    TG.load_registries()
    suite = TG.suite()
    TG.suite = lambda: suite  # type: ignore[assignment]  # the task list and the pins' lookups read this one suite
    tasks = [[k, label] for k, label, _ in cli._tasks([], CK.Options(), True, None)]
    roots: dict = dict(suite)
    for label in CK.load_pins().get("definitions", {}):
        try:
            roots.setdefault(label, TG.definition(label))
        except Exception as e:  # noqa: BLE001  (a pin naming no target is recorded as such, and compared)
            roots.setdefault(label, f"unresolved: {type(e).__name__}")
    for label, sub, sts in TG.templates():
        for i, st in enumerate(sts):
            roots[f"template:{label}#{i}:definition"] = sub.definition(**st)
            roots[f"template:{label}#{i}:target"] = sub.target(**st)
    reach = TG.reachable([f for f in roots.values() if isinstance(f, FunctionDefinition)])
    entries = {fid: hashlib.sha256(canonical_json(definition_entry(fn))).hexdigest() for fid, fn in sorted(reach.items())
               if isinstance(fn, FunctionDefinition)}
    targets = {label: fn if isinstance(fn, str) else fn.id for label, fn in sorted(roots.items())}
    return {"tasks": tasks, "targets": targets, "entries": entries, "family": family_digests(roots, owned_ids(root))}


def run_snapshot(root: Path, out: Path) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(root / d) for d in SOURCE_DIRS] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    r = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--snapshot", str(out), "--root", str(root)], env=env, cwd=root,
                       capture_output=True, text=True)
    if r.returncode:
        raise SplitError(f"the snapshot of {root} failed:\n{r.stderr[-4000:]}")
    return json.loads(out.read_text())


def compare(before: dict, after: dict) -> list[str]:
    bad = []
    if before["tasks"] != after["tasks"]:
        b, a = {tuple(t) for t in before["tasks"]}, {tuple(t) for t in after["tasks"]}
        bad.append(f"circuit-check's task list changed: lost {sorted(b - a)[:10]}, gained {sorted(a - b)[:10]}" + (" (its order)" if a == b else ""))
    for key, what in (("targets", "a target or pin resolves to another Definition"), ("entries", "a descriptor entry digest changed"),
                      ("family", "a family target's digests changed")):
        diff = sorted(k for k in before[key].keys() | after[key].keys() if before[key].get(k) != after[key].get(k))
        if diff:
            bad.append(f"{what}: {diff[:10]} ({len(diff)} in all)")
    return bad


def family_tasks(snap: dict) -> list[list[str]]:
    """circuit-check's tasks over the family's Definitions: a definition task on a family target, a template task whose
    Definitions are the family's."""
    family = snap["family"]
    templates = {label.split(":", 1)[1].rsplit("#", 1)[0] for label in family if label.startswith("template:")}
    return [[k, label] for k, label in snap["tasks"] if (label in family if k == "definition" else label in templates)]


def write_pin(root: Path, snap: dict) -> Path:
    path = root / PIN
    body = {"about": "The attention family's circuit-check targets and pins as the tree before the split built them: each one's "
                     "Definition id, descriptor entry digest and, for a specialisation, its Program's canonical descriptor digest; "
                     "and circuit-check's tasks over them. Written by integrations/vllm/tools/migrate/split_attention.py; "
                     "tools/circuit_check/tests/test_split_attention.py regenerates them.",
            "targets": snap["family"], "tasks": family_tasks(snap)}
    path.write_text(json.dumps(body, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return path


# --- command line -----------------------------------------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=ROOT, help="the repository to split (default: the one holding this script)")
    ap.add_argument("--dry-run", action="store_true", help="say what would move, write nothing")
    ap.add_argument("--no-verify", action="store_true", help="skip the before/after circuit-check passes and the pin")
    ap.add_argument("--allow-dirty", action="store_true", help="split a work tree with uncommitted changes")
    ap.add_argument("--snapshot", type=Path, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    root = a.root.resolve()
    if a.snapshot:
        a.snapshot.write_text(json.dumps(snapshot(root), sort_keys=True))
        return 0
    try:
        p = plan(root)
        for line in p.done:
            print(f"done     {line}")
        for line in p.skipped:
            print(f"skipped  {line}")
        for mod, kinds in p.moves.items():
            for kind, names in kinds.items():
                print(f"move     {REGISTRY}.{mod} -> {KINDS[kind]}.{mod}: {', '.join(names)}")
        for mod, kind in p.module_moves.items():
            print(f"move     {REGISTRY}.{mod} -> {KINDS[kind]}.{mod} (the whole module)")
        if p.empty:
            print("the attention family is split on this tree: nothing to do")
            return 0
        if a.dry_run:
            return 0
        if not a.allow_dirty:
            dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True).stdout.strip()
            if dirty:
                raise SplitError(f"uncommitted changes in {root} (commit them, or pass --allow-dirty):\n{dirty[:2000]}")
        with tempfile.TemporaryDirectory() as tmp:
            before = None if a.no_verify else run_snapshot(root, Path(tmp) / "before.json")
            for f in apply(p):
                print(f"{'wrote' if f.exists() else 'removed':8s} {f.relative_to(root)}")
            if before is not None:
                print(f"pinned   {write_pin(root, before).relative_to(root)} ({len(before['family'])} targets)")
                bad = compare(before, run_snapshot(root, Path(tmp) / "after.json"))
                if bad:
                    print("\n".join(["verification FAILED; the tree is left split (`git checkout -- . && git clean -fd` undoes it):", *bad]))
                    return 1
                print(f"verified {len(before['tasks'])} circuit-check tasks, {len(before['targets'])} targets and pins, "
                      f"{len(before['entries'])} descriptor entries and {len(before['family'])} family targets unchanged")
    except SplitError as e:
        print(f"split_attention: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
