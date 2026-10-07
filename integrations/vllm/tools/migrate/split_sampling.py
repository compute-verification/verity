"""Split the sampling family of `verity_vllm.program.registry` into its Definitions, its twins and its vocabulary.

The repository migration moves every Definition into `verity/ml/circuits/`, which imports only `verity`; before that, each registry module is
split so that its Definition modules import only `verity` and other Definition modules. For the sampling family:

* the Definition modules stay where they are: `sampling`, `sampling_boolean`, `gumbel_boolean`, `topp_word_gates`,
  `topp_keep_builder`, `topp_words`, and the new `topp_geometry` (the split count and lane geometry the Definitions read, out
  of the `topp_split` twin);
* the twins are `sampling_rows`, `sampling_topp_difftest`, `topp_split`, and the new `sampling_twins` (the two test-only
  witnesses, out of `sampling`);
* the vocabulary is the new `sampling_vocab` (`SHARED_GREEDY`, `select_for`, out of `sampling`).

A Definition module that reached a core object through another registry module (`prims.F32Mul`, `b1.const`, `moe.F32GtStrict`)
imports it from the module that defines it, found by following the re-export chain and checked by identity. Every import of a
moved name, repo-wide, is rewritten to its new module, and so is every mention of one in a comment, a docstring or a Markdown
file (string data is left alone: a record may hash it). No Definition changes, so no Definition id or descriptor digest does:
`snapshot` pins both, and circuit-check's task list, on the unsplit tree, and the tests compare them after.

    uv run python integrations/vllm/tools/migrate/split_sampling.py            # on an unsplit tree: snapshot, apply, check
    uv run python integrations/vllm/tools/migrate/split_sampling.py snapshot   # write the two pins from the current tree
    uv run python integrations/vllm/tools/migrate/split_sampling.py apply      # the split itself, then check
    uv run python integrations/vllm/tools/migrate/split_sampling.py check      # the split tree's static invariants

It reruns on a later `main`: it splits, says the tree is already split, or stops and names what it cannot rewrite.
"""
from __future__ import annotations

import argparse
import ast
import builtins
import importlib
import importlib.util
import io
import json
import re
import subprocess
import symtable
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
VLLM = ROOT / "integrations" / "vllm"
PKG = "verity_vllm.program.registry"
REG = VLLM / "verity_vllm" / "program" / "registry"
ALLOWLISTS = VLLM / "tests" / "lint" / "allowlists"
DIGEST_TEST = VLLM / "tests" / "program" / "test_sampling_split.py"
TASKS_TEST = ROOT / "tools" / "verity" / "circuit_check" / "tests" / "test_sampling_split_tasks.py"
WIDTH = 130


class SplitError(RuntimeError):
    pass


@dataclass(frozen=True)
class Move:
    src: str
    dst: str
    names: tuple[str, ...]
    doc: str


MOVES = (
    Move("sampling", "sampling_vocab", ("SHARED_GREEDY", "select_for"), '''"""The sampling family's vocabulary: which stochastic select a target's `sampler_construction` binds.

`SHARED_GREEDY` maps each restated select (`GumbelTokenSelect_v1`, `GumbelTopPTokenSelect_v1`) to the Definition of `registry.sampling`
that computes `temp == 0` once, and `select_for` picks one of the two under a construction. The Definitions are `sampling`'s; this
module names choices among them."""'''),
    Move("sampling", "sampling_twins", ("add_rz_ftz_v1_model", "nv_log1pf_v1_model"), '''"""Test-only witnesses beside the sampling family's Definitions (`registry.sampling`), not evaluators.

`add_rz_ftz_v1_model` is the round-to-nearest model of `add.rz.ftz.f32` that `nv_log1pf` used before its add was made a true
toward-zero one, and `nv_log1pf_v1_model` is `nv_log1pf` under it. The tests compare both with the registered evaluators on every
word; no Definition calls them. Their vectorised forms are `sampling_rows`'s."""'''),
    Move("topp_split", "topp_geometry", ("SPLIT_MAX_BATCH", "SPLIT_MAX_SPLITS", "splits_for", "SPLITS_FOR_V1_WIDTH", "splits_for_v1",
                                         "spt_for", "z_group"), '''"""The launch geometry of vLLM's split top-p pipeline, as the sampling Definitions read it.

`splits_for` is `_apply_topp_split`'s split count (its doubling loop) and `splits_for_v1` its total form, the evaluator of
`registry.lifted`'s `SplitsForSMS{n}_v1`; `spt_for` is the sizePerThread of the kernels' row loads and `z_group` the one of the two
`Z` values a warp lane holds. The keep-word builder (`topp_keep_builder`) reads `spt_for` and `z_group` from here; the exact model of
the pipeline itself is the `topp_split` twin."""'''),
)

#: the family's Definition modules after the split
DEFINITION_MODULES = ("sampling", "sampling_boolean", "gumbel_boolean", "topp_word_gates", "topp_keep_builder", "topp_words", "topp_geometry")
#: its twins (reference models, difftests, test-only witnesses) and its vocabulary
TWIN_MODULES = ("sampling_rows", "sampling_topp_difftest", "topp_split", "sampling_twins")
VOCAB_MODULES = ("sampling_vocab",)
#: other families' modules a Definition module of this family imports, and what it takes from each
OTHER_DEFINITION_MODULES = {
    "ref_prims": "`F32GtStrict_v1` and `F32Sub_v1`, the v1 fp32 compare and subtract the selects and the top-p gates call",
    "b1": "`ArgmaxStep_v1` and `TokenSelect_v1`, the word Definitions `sampling_boolean`'s Boolean ones are traced against (`word=`)",
}
#: imports from a Definition module into a twin that stay: (module, enclosing function, imported module) -> why
EXCEPTIONS = {
    ("sampling", "topp_keep", f"{PKG}.topp_split"): "the evaluator of `TopPMaskWordx{V}_v1` is the exact model of the split top-p "
                                                    "pipeline (`topp_split.topp_keep_row`: numpy and the C-JIT MUFU EX2 and "
                                                    "div.full tables of `program.kernels`); the keep word is that model's output",
}
#: where a Value type re-exported as another module's own (`prims.F32`, equal to core's but a distinct object) is taken from
CORE_TYPES = "verity.ml.circuits.ops.scalar"

#: rewrites no AST rule makes, each anchored on its exact text: (file, old, new, occurrences)
PATCHES = (
    ("integrations/vllm/verity_vllm/program/registry/topp_keep_builder.py",
     "@cache\ndef _ts():\n    from verity_vllm.program.registry import topp_split\n    return topp_split\n\n\n", "", 1),
    ("integrations/vllm/verity_vllm/program/registry/topp_keep_builder.py", "_ts().spt_for(", "spt_for(", 1),
    ("integrations/vllm/verity_vllm/program/registry/topp_keep_builder.py", "_ts().z_group(", "z_group(", 2),
    ("integrations/vllm/verity_vllm/program/registry/topp_keep_builder.py", "from functools import cache\n", "", 1),
    ("integrations/vllm/verity_vllm/program/registry/topp_keep_builder.py",
     "from verity_vllm.program.registry.topp_word_gates import CW,",
     "from verity_vllm.program.registry.topp_geometry import spt_for, z_group\n"
     "from verity_vllm.program.registry.topp_word_gates import CW,", 1),
    ("integrations/vllm/tests/program/test_sampling_rows.py",
     "    for name, domain_ok in ((\"nv_logf\",",
     "    scalar = {\"nv_logf\": SP.nv_logf, \"nv_log1pf\": SP.nv_log1pf, \"nv_log1pf_v1_model\": SP.nv_log1pf_v1_model}\n"
     "    for name, domain_ok in ((\"nv_logf\",", 1),
    ("integrations/vllm/tests/program/test_sampling_rows.py", "want = getattr(SP, name)(x)", "want = scalar[name](x)", 1),
)


# --- source helpers -----------------------------------------------------------------------------------------------------------

def mod_path(name: str) -> Path:
    return REG / f"{name}.py"


def rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix()


def tracked(suffix: str) -> list[Path]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.split("\0")
    return [ROOT / f for f in out if f.endswith(suffix) and (ROOT / f).is_file()]


class Text:
    """A file's text, offsets from AST positions (UTF-8 byte columns), and non-overlapping edits applied together."""

    def __init__(self, text: str):
        self.text = text
        self.lines = text.splitlines(keepends=True)
        self.starts = [0]
        for ln in self.lines:
            self.starts.append(self.starts[-1] + len(ln))
        self.edits: set[tuple[int, int, str]] = set()

    def off(self, line: int, col: int) -> int:
        return self.starts[line - 1] + len(self.lines[line - 1].encode()[:col].decode())

    def span(self, node: ast.AST) -> tuple[int, int]:
        return self.off(node.lineno, node.col_offset), self.off(node.end_lineno, node.end_col_offset)

    def stmt_lines(self, node: ast.stmt) -> tuple[int, int]:
        """The whole lines of a statement that is alone on them (its trailing comment aside)."""
        a, b = self.starts[node.lineno - 1], self.starts[node.end_lineno]
        head = self.lines[node.lineno - 1].encode()[:node.col_offset].decode()
        tail = self.lines[node.end_lineno - 1].encode()[node.end_col_offset:].decode().strip()
        if head.strip() or (tail and not tail.startswith("#")):
            raise SplitError(f"line {node.lineno}: the statement shares its line with another; split it by hand")
        return a, b

    def edit(self, a: int, b: int, new: str) -> None:
        self.edits.add((a, b, new))

    def result(self) -> str:
        out, end = self.text, None
        for a, b, new in sorted(self.edits, key=lambda e: (e[0], e[1], e[2]), reverse=True):
            if end is not None and b > end:
                raise SplitError(f"overlapping edits at offset {a}")
            out = out[:a] + new + out[b:]
            end = a
        return out


def indent_of(t: Text, node: ast.stmt) -> str:
    line = t.lines[node.lineno - 1]
    return line[: len(line) - len(line.lstrip())]


def trailing_comment(t: Text, node: ast.stmt) -> str:
    rest = t.lines[node.end_lineno - 1].encode()[node.end_col_offset:].decode().rstrip("\n")
    return rest if rest.strip().startswith("#") else ""


def render_from(module: str, names: list[tuple[str, str | None]], indent: str = "") -> str:
    parts = [n if a is None or a == n else f"{n} as {a}" for n, a in names]
    one = f"{indent}from {module} import {', '.join(parts)}"
    if len(one) <= WIDTH:
        return one
    lines, cur = [], f"{indent}from {module} import ("
    for i, p in enumerate(parts):
        piece = p + (")" if i == len(parts) - 1 else ",")
        if len(cur) + 1 + len(piece) > WIDTH and not cur.endswith("("):
            lines.append(cur)
            cur = indent + " " * 4 + piece
        else:
            cur += ("" if cur.endswith("(") else " ") + piece
    return "\n".join(lines + [cur])


def top_defined(tree: ast.Module) -> dict[str, ast.stmt]:
    """Names a module defines at top level (def, class, assignment), not the ones it imports."""
    out: dict[str, ast.stmt] = {}
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.setdefault(n.name, n)
        elif isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            for tg in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                for x in ast.walk(tg):
                    if isinstance(x, ast.Name):
                        out.setdefault(x.id, n)
    return out


def top_imports(tree: ast.Module) -> dict[str, tuple[ast.stmt, ast.alias]]:
    out = {}
    for n in tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                out[a.asname or (a.name.split(".")[0] if isinstance(n, ast.Import) else a.name)] = (n, a)
    return out


def import_module_of(node: ast.Import | ast.ImportFrom, alias: ast.alias) -> str:
    """The module an import binding names: `from PKG import mod` names `PKG.mod` when `mod` is a registry module."""
    if isinstance(node, ast.Import):
        return alias.name
    if node.module == PKG and mod_path(alias.name).exists():
        return f"{PKG}.{alias.name}"
    return node.module or ""


def all_names(tree: ast.Module) -> list[str]:
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id == "__all__" for x in n.targets):
            return [e.value for e in n.value.elts if isinstance(e, ast.Constant)]
    return []


def names_used(tree: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}


def free_globals(code: str) -> set[str]:
    """Module-level names `code` reads and does not define."""
    top = symtable.symtable(code, "<module>", "exec")
    defined = {s.get_name() for s in top.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}
    need: set[str] = set()

    def walk(tab, is_top: bool) -> None:
        for s in tab.get_symbols():
            if s.is_referenced() and (is_top or s.is_global()) and s.get_name() not in defined:
                need.add(s.get_name())
        for c in tab.get_children():
            walk(c, False)

    walk(top, True)
    return need - set(dir(builtins)) - {"__name__", "__file__", "__doc__"}


def is_stdlib(module: str) -> bool:
    return module.split(".")[0] in sys.stdlib_module_names


def group_of(module: str) -> int:
    return 0 if is_stdlib(module) else 2 if module.split(".")[0] in ("verity", "verity_vllm") else 1


def enclosing_functions(tree: ast.Module) -> dict[int, str]:
    """id(node) -> the name of the outermost function it is in."""
    out: dict[int, str] = {}
    for f in ast.walk(tree):
        if isinstance(f, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for x in ast.walk(f):
                out.setdefault(id(x), f.name)
    return out


# --- the moves ------------------------------------------------------------------------------------------------------------------

@dataclass
class Chunk:
    name: str
    first: int          # its first line: attached comments and decorators included
    last: int
    text: str


def chunks_of(lines: list[str], tree: ast.Module, names: tuple[str, ...]) -> list[Chunk]:
    defined = top_defined(tree)
    out = []
    for name in names:
        node = defined.get(name)
        if node is None:
            raise SplitError(f"{name} is not defined at the top level of its module")
        if isinstance(node, ast.Assign) and (len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name)):
            raise SplitError(f"{name} is assigned together with other names (line {node.lineno}); split the statement by hand")
        first = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
        while first > 1 and lines[first - 2].lstrip().startswith("#"):
            first -= 1
        out.append(Chunk(name, first, node.end_lineno, "".join(lines[first - 1: node.end_lineno])))
    return out


def destination_imports(need: set[str], src: str, tree: ast.Module, moved: dict[str, str], here: str) -> list[str]:
    """Import blocks (stdlib, third party, first party) for what a destination's code reads: the source's imports, the names
    that stay in the source, or another destination's."""
    imports, defined = top_imports(tree), top_defined(tree)
    stmts: dict[int, tuple[ast.stmt, list[ast.alias]]] = {}
    from_src: list[str] = []
    from_dst: dict[str, list[str]] = {}
    for name in sorted(need):
        if name in moved:
            if moved[name] != here:
                from_dst.setdefault(moved[name], []).append(name)
        elif name in imports:
            node, alias = imports[name]
            stmts.setdefault(id(node), (node, []))[1].append(alias)
        elif name in defined:
            from_src.append(name)
        else:
            raise SplitError(f"{here}: cannot resolve {name!r}: {src} neither imports nor defines it")
    blocks: list[list[str]] = [[], [], []]
    for node, aliases in sorted(stmts.values(), key=lambda v: v[0].lineno):
        if isinstance(node, ast.Import):
            for a in aliases:
                blocks[group_of(a.name)].append(f"import {a.name}" + (f" as {a.asname}" if a.asname else ""))
        else:
            order = {a.name: i for i, a in enumerate(node.names)}
            blocks[group_of(node.module)].append(render_from(node.module, [(a.name, a.asname) for a in sorted(aliases, key=lambda a: order[a.name])]))
    if from_src:
        blocks[2].append(render_from(f"{PKG}.{src}", [(n, None) for n in from_src]))
    for dst, names in sorted(from_dst.items()):
        blocks[2].append(render_from(f"{PKG}.{dst}", [(n, None) for n in names]))
    return ["\n".join(b) for b in blocks if b]


def new_module(move: Move, chunks: list[Chunk], lines: list[str], future: bool, imports: list[str]) -> str:
    head = move.doc + "\n" + ("from __future__ import annotations\n\n" if future else "\n")
    if imports:
        head += "\n\n".join(imports) + "\n\n\n"
    body = []
    for i, c in enumerate(chunks):
        if i:
            between = range(chunks[i - 1].last + 1, c.first)
            adjacent = all(not lines[j - 1].strip() for j in between)
            body.append("\n" * ((len(between) if adjacent else 2) + 1))
        body.append(c.text.rstrip("\n"))
    public = [c.name for c in chunks if not c.name.startswith("_")]
    return head + "".join(body) + "\n\n\n__all__ = [" + ", ".join(f'"{n}"' for n in public) + "]\n"


def drop_from_all(t: Text, tree: ast.Module, drop: set[str]) -> None:
    """Drop `drop` from the module's `__all__`, and a comment line left heading no name."""
    for n in tree.body:
        if isinstance(n, ast.Assign) and any(isinstance(x, ast.Name) and x.id == "__all__" for x in n.targets):
            a, b = t.starts[n.lineno - 1], t.starts[n.end_lineno]
            seg = t.text[a:b]
            for name in sorted(drop):
                seg, k = re.subn(rf'"{re.escape(name)}"\s*,?[ \t]*', "", seg)
                if not k:
                    raise SplitError(f"{name} is not in __all__")
            out = seg.splitlines(keepends=True)
            keep = [ln for i, ln in enumerate(out) if ln.strip() and not
                    (ln.strip().startswith("#") and not any('"' in later.split("#")[0] for later in out[i + 1:]))]
            seg = re.sub(r",\s*\]", "]", "".join(keep))
            t.edit(a, b, seg if seg.endswith("\n") else seg + "\n")
            return


def split_source(src: str, group: list[Move], moved: dict[str, str]) -> dict[Path, str]:
    """The source module without its moved names, and each destination module."""
    text = mod_path(src).read_text()
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    future = any(isinstance(n, ast.ImportFrom) and n.module == "__future__" for n in tree.body)
    out: dict[Path, str] = {}
    chunks: list[Chunk] = []
    for mv in group:
        cs = chunks_of(lines, tree, mv.names)
        chunks += cs
        code = ("from __future__ import annotations\n" if future else "") + "\n".join(c.text for c in cs)
        imports = destination_imports(free_globals(code) - {"annotations"}, src, tree, moved, mv.dst)
        out[mod_path(mv.dst)] = new_module(mv, cs, lines, future, imports)
    t = Text(text)
    regions: list[list[Chunk]] = []
    for c in sorted(chunks, key=lambda c: c.first):
        if regions and all(not lines[i - 1].strip() for i in range(regions[-1][-1].last + 1, c.first)):
            regions[-1].append(c)
        else:
            regions.append([c])
    for r in regions:
        first, last = r[0].first, r[-1].last
        a = first
        while a > 1 and not lines[a - 2].strip():
            a -= 1
        b = last
        while b < len(lines) and not lines[b].strip():
            b += 1
        before, after = first - a, b - last
        t.edit(t.starts[a - 1], t.starts[b], "" if b == len(lines) else "\n" * max(before, after))
    drop_from_all(t, tree, {c.name for c in chunks} & set(all_names(tree)))
    out[mod_path(src)] = prune_and_reimport(t.result(), names_used(tree), moved, src)
    return out


def prune_and_reimport(text: str, before: set[str], moved: dict[str, str], src: str) -> str:
    """Drop the imports only the moved code read, and import back the moved names the source still reads."""
    tree = ast.parse(text)
    used = names_used(tree) | set(all_names(tree))
    t = Text(text)
    for n in tree.body:
        if not isinstance(n, (ast.Import, ast.ImportFrom)) or (isinstance(n, ast.ImportFrom) and n.module == "__future__"):
            continue
        bound = [(a, a.asname or (a.name.split(".")[0] if isinstance(n, ast.Import) else a.name)) for a in n.names]
        keep = [a for a, b in bound if b in used or b not in before]
        if len(keep) == len(n.names):
            continue
        a, b = t.stmt_lines(n)
        if not keep:
            t.edit(a, b, "")
        elif isinstance(n, ast.ImportFrom):
            t.edit(a, b, render_from(n.module, [(x.name, x.asname) for x in keep]) + trailing_comment(t, n) + "\n")
        else:
            raise SplitError(f"{src}:{n.lineno}: an `import` of several modules lost some; prune it by hand")
    text = t.result()
    tree = ast.parse(text)
    back: dict[str, list[str]] = {}
    for name in sorted(free_globals(text) & set(moved)):
        back.setdefault(moved[name], []).append(name)
    if not back:
        return text
    for dst, names in back.items():
        if src in DEFINITION_MODULES and dst not in DEFINITION_MODULES:
            raise SplitError(f"{src}, a Definition module, still reads {names}, which move to {dst}")
    last = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))][-1]
    lead = group_of(last.module if isinstance(last, ast.ImportFrom) else last.names[0].name) != 2
    t = Text(text)
    at = t.starts[last.end_lineno]
    t.edit(at, at, ("\n" if lead else "") + "".join(render_from(f"{PKG}.{d}", [(x, None) for x in ns]) + "\n" for d, ns in sorted(back.items())))
    return t.result()


# --- Definition modules take a re-exported name from the module that defines it ------------------------------------------------------

def module_file(module: str) -> Path | None:
    spec = importlib.util.find_spec(module)
    return Path(spec.origin) if spec and spec.origin and spec.origin.endswith(".py") else None


def owner_of(module: str, name: str, depth: int = 0) -> tuple[str, str]:
    """(module, name) defining what `module.name` is, through `from M import x` and `x = R.y` re-exports."""
    f = module_file(module)
    if f is None or depth > 8:
        raise SplitError(f"cannot follow {module}.{name} to the module that defines it")
    tree = ast.parse(f.read_text())
    pkg = module if f.name == "__init__.py" else module.rpartition(".")[0]
    for n in tree.body:
        if isinstance(n, ast.ImportFrom):
            for a in n.names:
                if (a.asname or a.name) == name:
                    base = n.module if n.level == 0 else ".".join(pkg.split(".")[: len(pkg.split(".")) - n.level + 1] + [n.module or ""]).rstrip(".")
                    return owner_of(base, a.name, depth + 1)
    node = top_defined(tree).get(name)
    if node is None:
        raise SplitError(f"{module} neither defines nor imports {name}")
    if isinstance(node, ast.Assign) and isinstance(node.value, ast.Attribute) and isinstance(node.value.value, ast.Name):
        imp = top_imports(tree).get(node.value.value.id)
        if imp:
            return owner_of(import_module_of(*imp), node.value.attr, depth + 1)
    return module, name


def target_of(module: str, name: str) -> tuple[str, str]:
    """Where a Definition module takes `module.name` from: core, a Definition module of its family, or one of another family's
    it may import; refuses anything else."""
    obj = getattr(importlib.import_module(module), name)
    own, oname = owner_of(module, name)
    allowed = {f"{PKG}.{m}" for m in DEFINITION_MODULES + tuple(OTHER_DEFINITION_MODULES)}
    if own.split(".")[0] == "verity" or own in allowed:
        if getattr(importlib.import_module(own), oname) is not obj:
            raise SplitError(f"{module}.{name} is not the object {own}.{oname}")
        return own, oname
    from verity.ml.circuits.types import Array, Tuple, Value
    if isinstance(obj, (Value, Array, Tuple)) and getattr(importlib.import_module(CORE_TYPES), name, None) == obj:
        return CORE_TYPES, name
    raise SplitError(f"{module}.{name} is {own}'s own; a Definition module may not import it from there")


def rewrite_definition_module(name: str) -> str | None:
    """`name` with each object it reached through another family's module imported from its owner; None if nothing changes."""
    text = mod_path(name).read_text()
    tree = ast.parse(text)
    t = Text(text)
    bound = {**top_defined(tree), **top_imports(tree)}
    bare: dict[str, list[str]] = {}

    def want(owner: str, oname: str, local: str) -> None:
        if oname != local:
            raise SplitError(f"{name}: {local} is {owner}.{oname} under another name")
        imp = top_imports(tree).get(local)
        same = imp is not None and imp[1].name == local and target_of(import_module_of(*imp), local) == (owner, oname) \
            if imp and import_module_of(*imp) != PKG + "." + local else False
        if local in bound and local not in bare.get(owner, []) and not same:
            raise SplitError(f"{name}: {local} is already bound; it cannot also be imported from {owner}")
        if local not in bare.setdefault(owner, []):
            bare[owner].append(local)

    changed = False
    for n in tree.body:
        if not isinstance(n, ast.ImportFrom) or n.level or not (n.module or "").startswith("verity_vllm") or n.module == PKG:
            continue
        keep = []
        for a in n.names:
            owner, oname = target_of(n.module, a.name)
            if owner == n.module:
                keep.append(a)
            else:
                want(owner, oname, a.asname or a.name)
        if len(keep) != len(n.names):
            changed = True
            a_, b_ = t.stmt_lines(n)
            t.edit(a_, b_, render_from(n.module, [(x.name, x.asname) for x in keep]) + trailing_comment(t, n) + "\n" if keep else "")
    drop: dict[int, set[str]] = {}
    for n in tree.body:
        if not (isinstance(n, ast.ImportFrom) and n.module == PKG):
            continue
        for a in n.names:
            local, module = a.asname or a.name, import_module_of(n, a)
            uses = [x for x in ast.walk(tree) if isinstance(x, ast.Attribute) and isinstance(x.value, ast.Name) and x.value.id == local]
            refs = [x for x in ast.walk(tree) if isinstance(x, ast.Name) and x.id == local]
            done = 0
            for u in uses:
                owner, oname = target_of(module, u.attr)
                if owner == module:
                    continue
                if not isinstance(u.ctx, ast.Load):
                    raise SplitError(f"{name}:{u.lineno}: {local}.{u.attr} is assigned")
                want(owner, oname, u.attr)
                t.edit(*t.span(u), u.attr)
                done += 1
            if done:
                changed = True
                if done == len(refs):
                    drop.setdefault(id(n), set()).add(a.name)
    for n in tree.body:
        if id(n) in drop:
            keep = [x for x in n.names if x.name not in drop[id(n)]]
            a_, b_ = t.stmt_lines(n)
            t.edit(a_, b_, render_from(n.module, [(x.name, x.asname) for x in keep]) + trailing_comment(t, n) + "\n" if keep else "")
    if not changed:
        return None
    pending = dict(bare)
    for n in tree.body:
        if isinstance(n, ast.ImportFrom) and n.module in pending and not n.level and not any(a.name == "*" for a in n.names):
            have = [(a.name, a.asname) for a in n.names]
            add = [(x, None) for x in pending.pop(n.module) if (x, None) not in have]
            if add:
                a_, b_ = t.stmt_lines(n)
                t.edit(a_, b_, render_from(n.module, sorted(have + add, key=lambda p: p[0])) + trailing_comment(t, n) + "\n")
    if pending:
        anchors = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom)) and
                   ((n.module if isinstance(n, ast.ImportFrom) else n.names[0].name) or "").split(".")[0] == "verity"]
        anchors = anchors or [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        at = t.starts[anchors[-1].end_lineno]
        t.edit(at, at, "".join(render_from(o, [(x, None) for x in sorted(ns)]) + "\n" for o, ns in sorted(pending.items())))
    return t.result()


# --- consumers, repo-wide ------------------------------------------------------------------------------------------------------------

def scope_imports(scope: ast.AST):
    """The import statements directly in a module or function body (not in a nested function or class)."""
    stack = list(getattr(scope, "body", []))
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            yield n
        elif not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for f in ("body", "orelse", "finalbody", "handlers"):
                stack.extend(getattr(n, f, None) or [])


def may_name(expr: ast.expr | None, names: set[str]) -> bool:
    """Whether an attribute-name expression can evaluate to one of `names` (an f-string by its constant parts, else yes)."""
    if isinstance(expr, ast.JoinedStr):
        pat = "".join(re.escape(v.value) if isinstance(v, ast.Constant) else ".*" for v in expr.values)
        return any(re.fullmatch(pat, n) for n in names)
    return True


def rewrite_consumer(path: Path, text: str, moved: dict[tuple[str, str], str], kept: dict[str, set[str]]) -> tuple[str, list[str]]:
    """`text` with every moved name taken from its new module, and notes on what stayed and why. `kept[src]` are the moved names
    `src` still binds (it imports them back), which a file that assigns `alias.name` keeps reaching through `src`."""
    srcs = {s for s, _ in moved}
    tree = ast.parse(text)
    t = Text(text)
    notes: list[str] = []
    bound = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Load)}
    bound |= {a.asname or a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    names_moved = {name for _s, name in moved}
    mentions_moved = any(isinstance(s, ast.Constant) and s.value in names_moved for s in ast.walk(tree))
    scopes = [tree] + [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    for scope in scopes:
        for n in scope_imports(scope):
            if isinstance(n, ast.Import):
                if any(a.name in {f"{PKG}.{s}" for s in srcs} for a in n.names):
                    raise SplitError(f"{rel(path)}:{n.lineno}: a dotted `import {PKG}.<module>` is not rewritten by this script")
                continue
            if n.level or not n.module:
                continue
            src = n.module[len(PKG) + 1:] if n.module.startswith(PKG + ".") else None
            if src in srcs:
                stay = [a for a in n.names if (src, a.name) not in moved]
                go: dict[str, list[ast.alias]] = {}
                for a in n.names:
                    if (src, a.name) in moved:
                        go.setdefault(moved[(src, a.name)], []).append(a)
                if go:
                    ind, cm = indent_of(t, n), trailing_comment(t, n)
                    out = [render_from(n.module, [(a.name, a.asname) for a in stay], ind) + cm] if stay else []
                    out += [render_from(f"{PKG}.{d}", [(a.name, a.asname) for a in al], ind) + (cm if not stay or "noqa" in cm else "")
                            for d, al in sorted(go.items())]
                    t.edit(*t.stmt_lines(n), "\n".join(out) + "\n")
                continue
            if n.module != PKG:
                continue
            for a in n.names:
                if a.name not in srcs:
                    continue
                local = a.asname or a.name
                attrs = [x for x in ast.walk(scope) if isinstance(x, ast.Attribute) and isinstance(x.value, ast.Name) and x.value.id == local]
                refs = [x for x in ast.walk(scope) if isinstance(x, ast.Name) and x.id == local]
                calls = [x for x in ast.walk(scope) if isinstance(x, ast.Call) and isinstance(x.func, ast.Name)
                         and x.func.id in ("getattr", "hasattr", "setattr", "delattr") and x.args and isinstance(x.args[0], ast.Name)
                         and x.args[0].id == local]
                stored = {x.attr for x in attrs if not isinstance(x.ctx, ast.Load) and (a.name, x.attr) in moved}
                for s in stored:
                    if s not in kept.get(a.name, set()):
                        raise SplitError(f"{rel(path)}: assigns {local}.{s}, which {a.name} no longer binds after the split")
                    notes.append(f"{rel(path)}: kept `{local}.{s}`: the file assigns it, a monkeypatch of {a.name}'s own binding")
                companions: set[str] = set()
                handled: set[int] = set()
                for x in attrs:
                    handled.add(id(x.value))
                    if (a.name, x.attr) in moved and x.attr not in stored:
                        d = moved[(a.name, x.attr)]
                        t.edit(*t.span(x.value), d)
                        companions.add(d)
                live = len([x for x in attrs if not ((a.name, x.attr) in moved and x.attr not in stored)])
                for c in calls:
                    handled.add(id(c.args[0]))
                    lit = c.args[1] if len(c.args) > 1 else None
                    if isinstance(lit, ast.Constant) and isinstance(lit.value, str) and (a.name, lit.value) in moved:
                        d = moved[(a.name, lit.value)]
                        t.edit(*t.span(c.args[0]), d)
                        companions.add(d)
                        continue
                    if not isinstance(lit, ast.Constant) and mentions_moved and may_name(lit, names_moved):
                        raise SplitError(f"{rel(path)}:{c.lineno}: `{c.func.id}({local}, <expr>)` may name a moved name; add a PATCHES entry")
                    live += 1
                bare = [x for x in refs if id(x) not in handled]
                if bare and mentions_moved:
                    raise SplitError(f"{rel(path)}:{bare[0].lineno}: `{local}` is used as a value in a file that names a moved name; "
                                     f"add a PATCHES entry")
                live += len(bare)
                if not companions:
                    continue
                clash = sorted(companions & bound)
                if clash:
                    raise SplitError(f"{rel(path)}: `{clash[0]}` is already bound; the module cannot be imported under that name")
                ind = indent_of(t, n)
                comp = "".join(f"{ind}from {PKG} import {d}\n" for d in sorted(companions))
                a_, b_ = t.stmt_lines(n)
                if live:
                    t.edit(b_, b_, comp)
                elif len(n.names) == 1:
                    t.edit(a_, b_, comp.rstrip("\n") + trailing_comment(t, n) + "\n")
                    notes.append(f"{rel(path)}: `{local}` named only moved names; the file imports {', '.join(sorted(companions))} instead")
                else:
                    keep = [x for x in n.names if x is not a]
                    t.edit(a_, b_, render_from(PKG, [(x.name, x.asname) for x in keep], ind) + trailing_comment(t, n) + "\n" + comp)
    return t.result(), notes


# --- comments, docstrings and Markdown -----------------------------------------------------------------------------------------------

def prose_patterns() -> list[tuple[re.Pattern, str]]:
    out = []
    for mv in MOVES:
        names = "|".join(map(re.escape, mv.names))
        out.append((re.compile(rf"(?<![\w.])((?:verity_vllm\.program\.)?registry\.)?{mv.src}\.({names})\b"), rf"\g<1>{mv.dst}.\g<2>"))
    return out


def rewrite_prose(text: str) -> str:
    """Mentions of a moved name rewritten in comments and docstrings; code and other string data are left alone."""
    pats = prose_patterns()
    toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    tree = ast.parse(text)
    docs = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.body:
            d = n.body[0]
            if isinstance(d, ast.Expr) and isinstance(d.value, ast.Constant) and isinstance(d.value.value, str):
                docs.add((d.lineno, d.col_offset))
    t = Text(text)
    for tok in toks:
        row, col = tok.start
        if tok.type == tokenize.COMMENT or (tok.type == tokenize.STRING and (row, len(t.lines[row - 1][:col].encode())) in docs):
            new = tok.string
            for p, r in pats:
                new = p.sub(r, new)
            if new != tok.string:
                t.edit(t.starts[row - 1] + col, t.starts[tok.end[0] - 1] + tok.end[1], new)
    return t.result()


# --- lint allowlists -----------------------------------------------------------------------------------------------------------------

def move_allowlist_entries() -> list[Path]:
    """Each P-rule allowlist entry keyed on a moved name follows it to its new file, in place (the ratchet keys entries, not their order)."""
    files = {}
    for mv in MOVES:
        for n in mv.names:
            files[(rel(mod_path(mv.src)).removeprefix("integrations/vllm/"), n)] = rel(mod_path(mv.dst)).removeprefix("integrations/vllm/")
    changed = []
    for p in sorted(ALLOWLISTS.glob("*.json")):
        lines = p.read_text().splitlines(keepends=True)
        try:
            i0 = next(i for i, ln in enumerate(lines) if ln.strip() == '"entries": [')
            i1 = next(i for i in range(i0 + 1, len(lines)) if lines[i].strip() in ("]", "],"))
        except StopIteration:
            continue
        hit = False
        for i in range(i0 + 1, i1):
            e = json.loads(lines[i].strip().rstrip(","))
            sym = e.get("symbol", "").split(".")[0]
            key = e.get("detail") if sym == "<module>" and e.get("kind") == "def-name" else sym
            if (e.get("file"), key) in files:
                old = f'"file": {json.dumps(e["file"])}'
                assert lines[i].count(old) == 1, f"{rel(p)}:{i + 1}: cannot rewrite {lines[i].strip()}"
                lines[i] = lines[i].replace(old, f'"file": {json.dumps(files[(e["file"], key)])}')
                hit = True
        if hit:
            write(p, "".join(lines))
            changed.append(p)
    return changed


# --- the boundary --------------------------------------------------------------------------------------------------------------------

def boundary_violations() -> list[str]:
    """Imports in a Definition module of anything but the stdlib, numpy, `verity`, the family's Definition modules and the other
    families' modules it takes Definitions from (`OTHER_DEFINITION_MODULES`), less the listed `EXCEPTIONS`."""
    allowed = {f"{PKG}.{m}" for m in DEFINITION_MODULES + tuple(OTHER_DEFINITION_MODULES)}
    out = []
    for m in DEFINITION_MODULES:
        p = mod_path(m)
        if not p.exists():
            out.append(f"{m}: no such module")
            continue
        tree = ast.parse(p.read_text())
        within = enclosing_functions(tree)
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom):
                mods = [import_module_of(n, a) for a in n.names] if n.module == PKG else [n.module or ""]
            else:
                continue
            for mod in mods:
                if is_stdlib(mod) or mod.split(".")[0] in ("numpy", "verity") or mod in allowed or (m, within.get(id(n)), mod) in EXCEPTIONS:
                    continue
                out.append(f"{m}:{n.lineno}: imports {mod}")
    return out


# --- commands ------------------------------------------------------------------------------------------------------------------------

def state() -> str:
    """'unsplit' or 'split'; raises on a tree that is neither."""
    have = [mod_path(mv.dst).exists() for mv in MOVES]
    if not any(have):
        for mv in MOVES:
            missing = [n for n in mv.names if n not in top_defined(ast.parse(mod_path(mv.src).read_text()))]
            if missing:
                raise SplitError(f"{mv.src} lacks {missing} and {mv.dst} does not exist: neither the unsplit tree nor the split one")
        return "unsplit"
    if all(have):
        for mv in MOVES:
            missing = [n for n in mv.names if n not in top_defined(ast.parse(mod_path(mv.dst).read_text()))]
            if missing:
                raise SplitError(f"{mv.dst} exists but lacks {missing}")
        return "split"
    raise SplitError("some destination modules exist and others do not: a partial split")


def consumers() -> list[Path]:
    me = Path(__file__).resolve()
    return [p for p in tracked(".py") if p.resolve() != me]


_ORIGINAL: dict[Path, str | None] = {}


def write(p: Path, text: str) -> None:
    """Write `p`, remembering what it held so a failed `apply` leaves the tree as it found it."""
    if p not in _ORIGINAL:
        _ORIGINAL[p] = p.read_text() if p.exists() else None
    p.write_text(text)


def roll_back() -> None:
    for p, text in _ORIGINAL.items():
        if text is None:
            p.unlink(missing_ok=True)
        else:
            p.write_text(text)
    _ORIGINAL.clear()


def apply() -> int:
    try:
        return _apply()
    except BaseException:
        roll_back()
        raise


def _apply() -> int:
    if state() == "split":
        print("already split")
        return check()
    moved = {(mv.src, n): mv.dst for mv in MOVES for n in mv.names}
    by_name = {n: mv.dst for mv in MOVES for n in mv.names}
    texts: dict[Path, str] = {}
    for f, old, new, count in PATCHES:
        p = ROOT / f
        text = texts.setdefault(p, p.read_text())
        if text.count(old) != count:
            raise SplitError(f"{f}: the patch anchor occurs {text.count(old)} times, not {count}: {old[:80]!r}")
        texts[p] = text.replace(old, new)
    for p, text in texts.items():
        write(p, text)
    groups: dict[str, list[Move]] = {}
    for mv in MOVES:
        groups.setdefault(mv.src, []).append(mv)
    for src, group in groups.items():
        for p, text in split_source(src, group, by_name).items():
            write(p, text)
    for m in DEFINITION_MODULES:
        new = rewrite_definition_module(m)
        if new is not None:
            write(mod_path(m), new)
    kept = {src: set(top_imports(ast.parse(mod_path(src).read_text()))) for src in groups}
    notes: list[str] = []
    for p in consumers() + [mod_path(mv.dst) for mv in MOVES]:
        text = p.read_text()
        new, n = rewrite_consumer(p, text, moved, kept) if any(s in text for s in groups) else (text, [])
        notes += n
        new = rewrite_prose(new)
        if new != text:
            write(p, new)
    for p in tracked(".md"):
        text = p.read_text()
        new = text
        for pat, r in prose_patterns():
            new = pat.sub(r, new)
        if new != text:
            write(p, new)
    notes += [f"{rel(p)}: entries for moved names follow them" for p in move_allowlist_entries()]
    for n in sorted(set(notes)):
        print(n)
    return check()


def check() -> int:
    if state() != "split":
        raise SplitError("the tree is not split")
    problems = boundary_violations()
    for mv in MOVES:
        left = sorted(set(mv.names) & set(top_defined(ast.parse(mod_path(mv.src).read_text()))))
        if left:
            problems.append(f"{mv.src} still defines {left}")
    moved = {(mv.src, n): mv.dst for mv in MOVES for n in mv.names}
    kept = {mv.src: set(top_imports(ast.parse(mod_path(mv.src).read_text()))) for mv in MOVES}
    for p in consumers() + [mod_path(mv.dst) for mv in MOVES]:
        text = p.read_text()
        if not any(mv.src in text for mv in MOVES):
            continue
        try:
            if rewrite_consumer(p, text, moved, kept)[0] != text:
                problems.append(f"{rel(p)}: still reaches a moved name through its old module")
        except SplitError as e:
            problems.append(str(e))
    for p in problems:
        print("FAIL", p)
    if not problems:
        print(f"split: the Definition modules import only the stdlib, numpy, verity and Definition modules ({len(EXCEPTIONS)} listed "
              f"exception), and nothing reaches a moved name through its old module")
    return 1 if problems else 0


def snapshot() -> int:
    """The pins the invariant tests compare with: circuit-check's task list for the family, then the family's ids and digests."""
    for test in (TASKS_TEST, DIGEST_TEST):
        if subprocess.run([sys.executable, str(test), "--write"], cwd=ROOT).returncode:
            raise SplitError(f"{rel(test)} --write failed")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", nargs="?", default="run", choices=("run", "snapshot", "apply", "check"))
    args = ap.parse_args(argv)
    try:
        if args.command == "snapshot":
            return snapshot()
        if args.command == "check":
            return check()
        if args.command == "run" and state() == "unsplit":
            snapshot()
        return apply()
    except SplitError as e:
        print(f"split_sampling: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
