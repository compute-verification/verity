"""Split the registry's primitives, composites and serving family into Definition modules, twins and vocabulary.

    uv run python integrations/vllm/tools/migrate/split_prims_serve.py            # apply to the checkout
    uv run python integrations/vllm/tools/migrate/split_prims_serve.py --check    # what would change; exit 1 if anything would

The family is `verity_vllm.program.registry`'s hub: `prims`, `ref_prims`, `pad_prims`, `lifted`, `tables`, `value_inputs`,
`operand_widths`, `b1`, `b1_tp2`, `serve3`, `spec`, `pouw_rows` (Definition modules, which keep their names), `catalog` and the
package `__init__` (the registry's loader and its lazy families), and four modules that are not Definitions and leave the
registry whole (`MOVES`).  After the split a Definition module of the family imports only `verity`, the stdlib, numpy and other
Definition modules (`DEFINITION_IMPORTS`), so that the registry directory holds only what the catalog takes:

    registry/X.py's twins       -> verity_vllm/program/twins/X.py        (numpy/C++ reference twins, difftest adapters)
    registry/X.py's vocabulary  -> verity_vllm/program/vocabulary/X.py   (which Definition an identity, a profile or a target binds)
    lifted.py's record rules    -> verity_vllm/program/lifting/workload.py (GP-01's workload, the lag a record attests, the use document)

A move keeps every name, body and docstring; no Definition is touched, so no Definition id or descriptor digest moves.  Every
importer in the repository is rewritten (`from M import x`, `import M as A` with `A.x`, relative imports), and dotted and path
references in text files follow; there are no re-export shims.

Rerunnable: on a tree where the split is already applied it changes nothing; on a later `main` it applies the same split to
whatever the modules hold then, and fails loudly (exit 2, naming the module and the name) where it cannot: a moved name the
module no longer defines, a remaining line of a Definition module that reads a moved name, a Definition module that imports a
non-Definition module after the split, an alias it would have to bind that the importer already uses.
"""
from __future__ import annotations

import argparse
import ast
import io
import re
import subprocess
import sys
import tokenize
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
VLLM = "integrations/vllm"
REG = "verity_vllm.program.registry"


@dataclass(frozen=True)
class Split:
    """Top-level names of a Definition module `src` that move to `dst` (created with `doc`), and the alias an importer that
    reached them as `A.name` through `import src as A` binds for `dst`."""
    src: str
    dst: str
    names: tuple[str, ...]
    alias: str
    doc: str
    carry: tuple[str, ...] = ()                  # section comments of `src` that move with the names (exact text)
    patches: tuple[tuple[str, str], ...] = ()    # (old, new) in the moved code: each applies exactly once
    extra_imports: tuple[str, ...] = ()          # imports the patched code needs


@dataclass(frozen=True)
class Move:
    """A module of the registry that is not a Definition module, moved whole; `short` is how prose names it, before and after."""
    src: str
    dst: str
    short: tuple[str, str]


PACKAGES = {
    "verity_vllm.program.twins": '"""Reference twins of the registered Definitions: numpy and C++ models that reproduce a Definition\'s words fast, and\n'
                                 'the differential-test adapters that compare one with production. A twin is checked against its Definition, never\n'
                                 'the other way round; the Definitions are in `verity_vllm.program.registry`."""\n',
    "verity_vllm.program.vocabulary": '"""Vocabulary: which registered Definition an instance identity, a semantic profile or a declared target binds a\n'
                                      'serving role to. It reads the Definitions (`verity_vllm.program.registry`); no Definition reads it."""\n',
}

SPLITS = (
    Split(f"{REG}.prims", "verity_vllm.program.twins.prims", ("_fa2", "_rms"), "prims_twins",
          '"""The C++ models the MUFU primitives of `verity_vllm.program.registry.prims` were transliterated from, with their tables:\n'
          'FA2\'s (`fa2_model`, `fa2_relation`) and the Triton RMSNorm\'s (`rms_triton_model`, `rms_relation`), loaded once for the\n'
          'twins and the replay driver."""\n',
          carry=("# ---------------------------------------------------------------------------------------------------------\n"
                 "# the C++ models the MUFU primitives were transliterated from, with their tables (twins, the replay driver)\n"
                 "# ---------------------------------------------------------------------------------------------------------\n",)),
    Split(f"{REG}.ref_prims", "verity_vllm.program.twins.ref_prims",
          ("_np_u32", "_np_f32", "np_widen", "np_narrow", "np_roundtrip", "_np_fma", "_np_scalar_fn", "np_dot_ref", "np_linear_ref",
           "_np_seq_sum", "np_rmsnorm_ref", "np_layernorm_ref", "np_gelu_ref", "np_silu_ref", "np_softmax_ref", "np_rope_ref",
           "np_rope_interleaved_ref", "np_relu_ref", "np_eager_attention_head_ref"), "ref_prims_twins",
          '"""numpy twins of the reference kinds of `verity_vllm.program.registry.ref_prims`: fast evaluation over many rows, a\n'
          'testing aid validated against the gate-by-gate IR evaluation in `tests/program/test_ref_prims.py`."""\n',
          carry=("# ---------------------------------------------------------------------------------------------------------\n"
                 "# numpy twins — a TESTING AID (fast evaluation over many rows), validated against the gate-by-gate IR evaluation in\n"
                 "# `test_ref_prims.py`; the registered bodies above are the definition, these are not.\n"
                 "# ---------------------------------------------------------------------------------------------------------\n",),
          patches=(("the registered bodies above are", "the registered bodies of `verity_vllm.program.registry.ref_prims` are"),)),
    Split(f"{REG}.b1", "verity_vllm.program.vocabulary.b1",
          ("FA2_KERNEL_SOURCE", "fa2_kernel_geometry", "profile_attention_consts", "SERVE_FAMILIES", "DEFAULT_SERVE_FAMILY",
           "FA2_SPLITKV_KERNEL_NAMES", "LEGACY_ATTENTION_GEOMETRY", "serve_family", "attention_geometry_static", "serve_program",
           "serve_program_of_identity"), "b1_vocabulary",
          '"""Which B1 serving Definition an instance identity binds (`program.identity`: `case`, `family`, `attention`), and the\n'
          'attention kernel geometry a semantic profile declares; the Definitions are `verity_vllm.program.registry.b1`\'s."""\n',
          patches=(('cfg = globals().get(f"{case}_CONFIG")', 'cfg = getattr(_b1, f"{case}_CONFIG", None)'),),
          extra_imports=("from verity_vllm.program.registry import b1 as _b1",)),
    Split(f"{REG}.lifted", "verity_vllm.program.lifting.workload",
          ("LAG_FIELD", "LAG_DECLARED_FIELDS", "LAG_STATIC_FIELD", "_LAG_STATIC_RE", "_lag_rule", "lag_from_header",
           "declared_lag_of_row", "lifted_vu_use_document", "lift_component", "lifted_workload", "workload_program"),
          "lifting_workload",
          '"""The lifted Program where it meets the rest of the integration, beside its Definitions (`verity_vllm.program.registry.lifted`):\n'
          'GP-01\'s workload composed over lifted components (`lift_component`, `lifted_workload`, `workload_program`), the post-EOS\n'
          'lag a run header attests and a Build declares (`lag_from_header`, `declared_lag_of_row`), and the use document of a lifted\n'
          'VU-partition (`lifted_vu_use_document`)."""\n'),
)

MOVES = (
    Move(f"{REG}.targets", "verity_vllm.program.vocabulary.targets", ("registry.targets", "vocabulary.targets")),
    Move(f"{REG}.conformance", "verity_vllm.program.kernels.conformance", ("registry.conformance", "kernels.conformance")),
    Move(f"{REG}.serve3_reference", "verity_vllm.program.twins.serve3_reference",
         ("registry.serve3_reference", "twins.serve3_reference")),
    Move(f"{REG}.collectives_difftest", "verity_vllm.program.twins.collectives_difftest",
         ("registry.collectives_difftest", "twins.collectives_difftest")),
)

#: the family's Definition modules after the split
DEFINITION_MODULES = tuple(f"{REG}.{m}" for m in (
    "prims", "ref_prims", "pad_prims", "lifted", "tables", "value_inputs", "operand_widths", "b1", "b1_tp2", "serve3", "spec",
    "pouw_rows", "catalog")) + (REG,)
#: what they may import beyond the stdlib, numpy and `verity`: each other, and the other families' Definition modules they call
DEFINITION_IMPORTS = DEFINITION_MODULES + tuple(f"{REG}.{m}" for m in (
    "dense", "fa2_softcap", "fa3_check_inf", "topp_split", "topp_word_gates", "sampling")) + ("verity_pouw.circuit.ncp2",)

#: dead-code keep entries for the modules only tests reach once the split takes them out of a live module
KEEP = {
    "verity_vllm.program.twins.ref_prims": "numpy twins of registry.ref_prims, a testing aid checked by tests/program/test_ref_prims.py",
    "verity_vllm.program.vocabulary.b1": "the B1 identity -> Serve binding (registry.b1's vocabulary until the split); tests/program/test_composition.py",
}

TEXT_SUFFIXES = {".py", ".md", ".json", ".sh", ".toml", ".txt", ".yaml", ".yml", ".cfg", ".ini"}
#: files that name the old paths on purpose
EXCLUDE = (f"{VLLM}/tools/migrate/", f"{VLLM}/tests/migrate/", "tools/circuit_check/tests/fixtures/migration/")


class Refused(Exception):
    pass


def path_of(mod: str) -> str:
    rel = mod.replace(".", "/")
    pkg = f"{VLLM}/{rel}/__init__.py"
    return pkg if mod in PACKAGES or mod == REG else f"{VLLM}/{rel}.py"


def module_of(rel: str) -> str | None:
    parts = Path(rel).with_suffix("").parts
    if "verity_vllm" not in parts:
        return None
    parts = parts[parts.index("verity_vllm"):]
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


class Tree:
    """The checkout as an overlay: reads see earlier writes; `flush` writes, renames and deletes."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.text: dict[str, str] = {}
        self.dirty: set[str] = set()
        self.gone: set[str] = set()

    def exists(self, rel: str) -> bool:
        return rel not in self.gone and (rel in self.text or (self.root / rel).is_file())

    def read(self, rel: str) -> str:
        if rel in self.gone:
            raise FileNotFoundError(rel)
        if rel not in self.text:
            self.text[rel] = (self.root / rel).read_text()
        return self.text[rel]

    def write(self, rel: str, text: str) -> None:
        if not self.exists(rel) or self.read(rel) != text:
            self.text[rel] = text
            self.dirty.add(rel)
            self.gone.discard(rel)

    def move(self, src: str, dst: str) -> None:
        self.write(dst, self.read(src))
        self.gone.add(src)
        self.dirty.discard(src)

    def files(self) -> list[str]:
        out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=self.root, check=True, capture_output=True,
                             text=True).stdout.split()
        return sorted(({r for r in out if (self.root / r).is_file()} | self.dirty) - self.gone)

    def changes(self) -> list[str]:
        return sorted(self.dirty | self.gone)

    def flush(self) -> None:
        for rel in sorted(self.dirty):
            p = self.root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(self.text[rel])
        for rel in sorted(self.gone):
            (self.root / rel).unlink(missing_ok=True)


# --- the moved code ----------------------------------------------------------------------------------------------------------


def _bound(stmt: ast.stmt) -> list[str]:
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [stmt.name]
    if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
        out = []
        for t in (stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]):
            out += [n.id for n in ast.walk(t) if isinstance(n, ast.Name)]
        return out
    return []


def _span(lines: list[str], stmt: ast.stmt) -> tuple[int, int]:
    """[first, last) line indices of a top-level statement with its decorators, the comment lines directly above it, and the
    comment lines directly below it that continue a trailing comment of its last line."""
    first = min([stmt.lineno] + [d.lineno for d in getattr(stmt, "decorator_list", [])]) - 1
    while first > 0 and lines[first - 1].lstrip().startswith("#") and not lines[first - 1].startswith("# ---"):
        first -= 1
    last = stmt.end_lineno
    if "#" in lines[last - 1][stmt.end_col_offset:]:
        while last < len(lines) and lines[last].startswith("#") and not lines[last].startswith("# ---"):
            last += 1
    return first, last


def _merge(lines: list[str], spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Spans separated only by blank lines become one, so statements that sat together stay together, gaps and all."""
    out: list[tuple[int, int]] = []
    for a, b in sorted(spans):
        if out and all(not line.strip() for line in lines[out[-1][1]:a]):
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def _loads(nodes: list[ast.AST]) -> set[str]:
    return {n.id for root in nodes for n in ast.walk(root) if isinstance(n, ast.Name)}


def _import_lines(mod: str, stmt: ast.stmt, keep: set[str]) -> list[str]:
    """`stmt` (a top-level import of module `mod`) restricted to the names in `keep`, absolute."""
    if isinstance(stmt, ast.Import):
        return [f"import {a.name}" + (f" as {a.asname}" if a.asname else "") for a in stmt.names if (a.asname or a.name.split(".")[0]) in keep]
    base = _absolute(mod, stmt)
    names = [a for a in stmt.names if (a.asname or a.name) in keep]
    return [_render_from(base, names, "")] if names else []


def _absolute(me: str, stmt: ast.ImportFrom, is_package: bool = False) -> str:
    if not stmt.level:
        return stmt.module or ""
    parts = me.split(".")
    parts = parts[:len(parts) - stmt.level + (1 if is_package else 0)]
    return ".".join(parts + ([stmt.module] if stmt.module else []))


def _render_from(mod: str, names: list[ast.alias], indent: str, width: int = 130) -> str:
    items = [a.name + (f" as {a.asname}" if a.asname else "") for a in names]
    one = f"{indent}from {mod} import {', '.join(items)}"
    if len(one) <= width:
        return one
    head = f"{indent}from {mod} import ("
    out, cur = [], head
    for i, it in enumerate(items):
        piece = it + (")" if i == len(items) - 1 else ", ")
        if len(cur) + len(piece.rstrip()) > width and cur.strip() not in ("", head.strip()):
            out.append(cur.rstrip())
            cur = " " * len(head) + piece
        else:
            cur += piece
    out.append(cur.rstrip())
    return "\n".join(out)


def _render_all(names: list[str], width: int = 130) -> str:
    out, cur = [], "__all__ = ["
    for i, n in enumerate(names):
        piece = f'"{n}"' + ("]" if i == len(names) - 1 else ", ")
        if len(cur) + len(piece.rstrip()) > width:
            out.append(cur.rstrip())
            cur = " " * len("__all__ = [") + piece
        else:
            cur += piece
    return "\n".join(out + [cur.rstrip()]) if names else "__all__: list[str] = []"


def _module_names(t: ast.Module) -> dict[str, ast.stmt]:
    out = {}
    for s in t.body:
        for n in _bound(s):
            out.setdefault(n, s)
    return out


def _all_list(t: ast.Module) -> ast.Assign | None:
    for s in t.body:
        if isinstance(s, ast.Assign) and any(isinstance(x, ast.Name) and x.id == "__all__" for x in s.targets):
            return s
    return None


def split(tree: Tree, sp: Split, external_attrs: set[str]) -> str:
    """Apply one split; returns "applied" or "already applied"."""
    src_rel, dst_rel = path_of(sp.src), path_of(sp.dst)
    src = tree.read(src_rel)
    t = ast.parse(src)
    names = _module_names(t)
    here = [n for n in sp.names if n in names]
    dst_names = _module_names(ast.parse(tree.read(dst_rel))) if tree.exists(dst_rel) else {}
    if not here and all(n in dst_names for n in sp.names):
        return "already applied"
    if len(here) != len(sp.names):
        raise Refused(f"{sp.src}: does not define {sorted(set(sp.names) - set(here))}" +
                      (f" (and {dst_rel} defines {sorted(set(sp.names) & set(dst_names))})" if dst_names else ""))
    if tree.exists(dst_rel):
        raise Refused(f"{dst_rel} exists already: the split writes it whole")
    lines = src.splitlines(keepends=True)
    moved: list[ast.stmt] = []
    for s in t.body:
        b = _bound(s)
        if b and any(n in sp.names for n in b):
            if not all(n in sp.names for n in b):
                raise Refused(f"{sp.src} line {s.lineno}: binds {b}, of which only some move")
            moved.append(s)
    for s in moved:
        if any(isinstance(n, (ast.Global, ast.Nonlocal)) for n in ast.walk(s)):
            raise Refused(f"{sp.src}: {_bound(s)} rebinds a module global; it cannot leave its module")
    spans = _merge(lines, [_span(lines, s) for s in moved])
    taken = set()
    for a, b in spans:
        taken.update(range(a, b))
    blocks = ["".join(lines[a:b]).rstrip("\n") + "\n" for a, b in spans]
    rest = "".join(line for i, line in enumerate(lines) if i not in taken)
    carried = []
    for c in sp.carry:
        if rest.count(c) != 1:
            raise Refused(f"{sp.src}: the section comment {c.splitlines()[1]!r} is not there exactly once")
        rest = rest.replace(c, "")
        carried.append(c)
    code = "".join(c + "\n\n" for c in carried) + "\n\n".join(blocks)
    for old, new in sp.patches:
        if code.count(old) != 1:
            raise Refused(f"{sp.src}: patch {old!r} does not apply exactly once to the moved code")
        code = code.replace(old, new)
    # what the remaining module still reads of the moved names
    rt = ast.parse(rest)
    left = _loads(rt.body) & set(sp.names)
    if left:
        raise Refused(f"{sp.src}: what stays still reads {sorted(left)}; move them too or keep them")
    # the module's __all__ loses the moved names
    al = _all_list(rt)
    exported: list[str] = []
    if al is not None and isinstance(al.value, (ast.List, ast.Tuple)):
        keep_all = [e.value for e in al.value.elts if isinstance(e, ast.Constant) and e.value not in sp.names]
        exported = [e.value for e in al.value.elts if isinstance(e, ast.Constant) and e.value in sp.names]
        rl = rest.splitlines(keepends=True)
        rest = "".join(rl[:al.lineno - 1]) + _render_all(keep_all) + "\n" + "".join(rl[al.end_lineno:])
        rt = ast.parse(rest)
    # imports: the moved code's, from the module's own imports and from what stays in the module
    mt = ast.parse(code)
    used = _loads(mt.body) - set(sp.names)
    imports: list[str] = []
    from_src: list[str] = []
    rest_names = _module_names(rt)
    for s in t.body:
        if isinstance(s, (ast.Import, ast.ImportFrom)):
            bound = {(a.asname or a.name.split(".")[0]) for a in s.names}
            if isinstance(s, ast.ImportFrom) and s.module == "__future__":
                continue
            imports += _import_lines(sp.src, s, bound & used)
    for n in sorted(used):
        if n in rest_names and not isinstance(rest_names[n], (ast.Import, ast.ImportFrom)):
            from_src.append(n)
    if from_src:
        imports.append(_render_from(sp.src, [ast.alias(n) for n in from_src], ""))
    imports += list(sp.extra_imports)
    # the module's own imports nothing reads any more (here, below, or through it from another file)
    still = _loads(rt.body) | {e for e in (exported or [])}
    if al is not None:
        still |= {e.value for e in al.value.elts if isinstance(e, ast.Constant)}
    rl = rest.splitlines(keepends=True)
    for s in reversed([s for s in rt.body if isinstance(s, (ast.Import, ast.ImportFrom))]):
        if isinstance(s, ast.ImportFrom) and s.module == "__future__":
            continue
        drop = [a for a in s.names if (a.asname or a.name.split(".")[0]) in (used - still) and (a.asname or a.name) not in external_attrs]
        if not drop:
            continue
        keep = [a for a in s.names if a not in drop]
        new = ("" if not keep else (_render_from(_absolute(sp.src, s) if s.level else s.module, keep, "") if isinstance(s, ast.ImportFrom)
                                    else "import " + ", ".join(a.name + (f" as {a.asname}" if a.asname else "") for a in keep)) + "\n")
        rl[s.lineno - 1:s.end_lineno] = [new] if new else []
    rest = re.sub(r"\n{4,}", "\n\n\n", "".join(rl)).rstrip("\n") + "\n"
    dst = (sp.doc + "from __future__ import annotations\n\n" + "\n".join(_sort_imports(imports)) + "\n\n\n" + code.rstrip("\n") + "\n")
    if exported:
        dst += "\n\n" + _render_all(exported) + "\n"
    tree.write(src_rel, rest)
    tree.write(dst_rel, dst)
    return "applied"


def _sort_imports(lines: list[str]) -> list[str]:
    def key(s: str) -> tuple:
        mod = s.split()[1]
        top = mod.split(".")[0]
        group = 0 if top in sys.stdlib_module_names else (2 if top in ("verity", "verity_vllm", "verity_pouw") else 1)
        return (group, s.startswith("from"), mod)
    out, seen = [], set()
    for s in sorted(lines, key=key):
        if s not in seen:
            seen.add(s)
            out.append(s)
    groups, prev = [], None
    for s in out:
        g = key(s)[0]
        if prev is not None and g != prev:
            groups.append("")
        groups.append(s)
        prev = g
    return groups


# --- importers -------------------------------------------------------------------------------------------------------------


def _scopes(t: ast.Module) -> dict[ast.AST, ast.AST]:
    """node -> its enclosing function (or the module)."""
    out: dict[ast.AST, ast.AST] = {}

    def walk(node: ast.AST, scope: ast.AST) -> None:
        for ch in ast.iter_child_nodes(node):
            out[ch] = scope
            walk(ch, ch if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) else scope)
    walk(t, t)
    return out


def _chain(node: ast.AST, scope_of: dict[ast.AST, ast.AST], t: ast.Module) -> list[ast.AST]:
    out, cur = [], scope_of.get(node, t)
    while True:
        out.append(cur)
        if cur is t:
            return out
        cur = scope_of.get(cur, t)


def rewrite_importers(rel: str, text: str) -> str:
    """Every import of a moved module or a moved name in `rel`, and every `A.name` that reads a moved name through a module alias."""
    if "verity_vllm" not in text and "from ." not in text:
        return text
    try:
        t = ast.parse(text)
    except SyntaxError:
        return text
    me = module_of(rel) or ""
    is_pkg = rel.endswith("__init__.py")
    moves = {m.src: m.dst for m in MOVES}
    names = {(sp.src, n): sp for sp in SPLITS for n in sp.names}
    split_srcs = {sp.src for sp in SPLITS}
    lines = text.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))

    def off(lineno: int, col: int) -> int:
        return starts[lineno - 1] + len(lines[lineno - 1].encode()[:col].decode())

    edits: list[tuple[int, int, str]] = []
    scope_of = _scopes(t)
    bindings: dict[tuple[ast.AST, str], list[tuple[str, ast.stmt]]] = {}   # (scope, alias) -> [(split src module, import stmt)]
    for node in ast.walk(t):
        if isinstance(node, ast.ImportFrom):
            mod = _absolute(me, node, is_pkg)
            indent = lines[node.lineno - 1][:node.col_offset]
            if indent.strip():
                continue
            scope = scope_of[node]
            if mod in moves:
                rendered = [_render_from(moves[mod], list(node.names), indent)]
            elif mod in split_srcs:
                stay = [a for a in node.names if (mod, a.name) not in names]
                go: dict[str, list[ast.alias]] = {}
                for a in node.names:
                    if (mod, a.name) in names:
                        go.setdefault(names[(mod, a.name)].dst, []).append(a)
                if not go:
                    continue
                rendered = [_render_from(d, al, indent) for d, al in go.items()] + ([_render_from(mod, stay, indent)] if stay else [])
            else:
                pkg_moves = [a for a in node.names if f"{mod}.{a.name}" in moves]
                for a in node.names:
                    if f"{mod}.{a.name}" in split_srcs:
                        bindings.setdefault((scope, a.asname or a.name), []).append((f"{mod}.{a.name}", node))
                if not pkg_moves:
                    continue
                stay = [a for a in node.names if a not in pkg_moves]
                rendered = []
                for a in pkg_moves:
                    dpkg, _, dname = moves[f"{mod}.{a.name}"].rpartition(".")
                    bound = a.asname or a.name
                    rendered.append(_render_from(dpkg, [ast.alias(dname, None if dname == bound else bound)], indent))
                if stay:
                    rendered.append(_render_from(mod, stay, indent) if not node.level else
                                    indent + "from " + "." * node.level + (node.module or "") + " import "
                                    + ", ".join(a.name + (f" as {a.asname}" if a.asname else "") for a in stay))
            new = ("\n".join(rendered)).lstrip()
            edits.append((off(node.lineno, node.col_offset), off(node.end_lineno, node.end_col_offset), new))
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name in moves:
                    if not a.asname:
                        raise Refused(f"{rel} line {node.lineno}: `import {a.name}` without an alias; rewrite it by hand")
                    a0, b0 = off(node.lineno, node.col_offset), off(node.end_lineno, node.end_col_offset)
                    edits.append((a0, b0, re.sub(re.escape(a.name) + r"(?![\w])", moves[a.name], text[a0:b0])))
                elif a.name in split_srcs:
                    if not a.asname:
                        raise Refused(f"{rel} line {node.lineno}: `import {a.name}` without an alias; rewrite it by hand")
                    bindings.setdefault((scope_of[node], a.asname), []).append((a.name, node))
    # `A.name` through a module alias: each read of `A` belongs to the nearest import of `A` above it in its scope (else the first)
    inserts: dict[ast.stmt, set[Split]] = {}
    other_reads: dict[ast.stmt, int] = {}
    bound_names = {n.id for n in ast.walk(t) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    bound_names |= {(a.asname or a.name.split(".")[0]) for n in ast.walk(t) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    bound_names |= {n.name for n in ast.walk(t) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    for cands in bindings.values():
        cands.sort(key=lambda c: c[1].lineno)
    attr_of = {id(n.value): n for n in ast.walk(t) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)}
    for node in ast.walk(t):
        if not (isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)):
            continue
        for scope in _chain(node, scope_of, t):
            cands = bindings.get((scope, node.id))
            if not cands:
                continue
            above = [c for c in cands if c[1].lineno <= node.lineno]
            src_mod, stmt = above[-1] if above else cands[0]
            attr = attr_of.get(id(node))
            sp = names.get((src_mod, attr.attr)) if attr is not None else None
            if sp is None:
                other_reads[stmt] = other_reads.get(stmt, 0) + 1
            else:
                edits.append((off(node.lineno, node.col_offset), off(node.end_lineno, node.end_col_offset), sp.alias))
                inserts.setdefault(stmt, set()).add(sp)
            break
    for stmt, sps in inserts.items():
        indent = lines[stmt.lineno - 1][:stmt.col_offset]
        new_lines = []
        for sp in sorted(sps, key=lambda s: s.dst):
            dpkg, _, dname = sp.dst.rpartition(".")
            line = f"{indent}from {dpkg} import {dname} as {sp.alias}\n"
            if sp.alias in bound_names and line not in text:
                raise Refused(f"{rel}: binds `{sp.alias}` already; the import of {sp.dst} needs another alias")
            new_lines.append(line)
        if not other_reads.get(stmt) and len(stmt.names) == 1:   # the old alias is read for nothing else: the new import replaces it
            a0, b0 = off(stmt.lineno, stmt.col_offset), off(stmt.end_lineno, stmt.end_col_offset)
            edits.append((a0, b0, "".join(new_lines).strip()))
            continue
        add = "".join(x for x in new_lines if x not in text)
        if add:
            at = starts[stmt.end_lineno]
            if at == len(text) and not text.endswith("\n"):
                add = "\n" + add
            edits.append((at, at, add))
    if not edits:
        return text
    out = text
    for a, b, s in sorted(edits, key=lambda e: (e[0], e[1]), reverse=True):
        out = out[:a] + s + out[b:]
    return out


# --- text ---------------------------------------------------------------------------------------------------------------------


def text_rules() -> list[tuple[re.Pattern[str], str, bool]]:
    """(pattern, replacement, prose only): dotted and path references to moved modules and names."""
    rules = []
    for m in MOVES:
        rules.append((re.compile(re.escape(m.src) + r"(?![\w])"), m.dst, False))
        rules.append((re.compile(re.escape(path_of(m.src)[len(VLLM) + 1:])), path_of(m.dst)[len(VLLM) + 1:], False))
        rules.append((re.compile(r"(?<![\w.])" + re.escape(m.short[0]) + r"(?![\w])"), m.short[1], True))
    for sp in SPLITS:
        for n in sp.names:
            rules.append((re.compile(re.escape(f"{sp.src}.{n}") + r"(?![\w])"), f"{sp.dst}.{n}", False))
    return rules


def _apply(text: str, rules, prose: bool) -> str:
    for pat, new, prose_only in rules:
        if prose or not prose_only:
            text = pat.sub(new, text)
    return text


def rewrite_text(rel: str, text: str, rules, unsure: list[str] | None = None) -> str:
    """A non-Python file: every rule (the prose ones in Markdown only).  A Python file: its comments and statement strings
    (docstrings), and a string literal that is exactly a moved module or name; any other literal naming one is left as it is
    and reported in `unsure` (a string can be data a digest reads: a Vocabulary's notes are)."""
    if not rel.endswith(".py"):
        return _apply(text, rules, rel.endswith(".md"))
    if not any(pat.search(text) for pat, _, _ in rules):
        return text
    try:
        t = ast.parse(text)
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (SyntaxError, tokenize.TokenError):
        return text
    docs = {(n.value.lineno, n.value.col_offset) for n in ast.walk(t)
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)}
    exact = {m.src: m.dst for m in MOVES} | {path_of(m.src)[len(VLLM) + 1:]: path_of(m.dst)[len(VLLM) + 1:] for m in MOVES}
    exact |= {f"{sp.src}.{n}": f"{sp.dst}.{n}" for sp in SPLITS for n in sp.names}
    lines = text.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))
    edits = []
    for tok in toks:
        a, b = starts[tok.start[0] - 1] + tok.start[1], starts[tok.end[0] - 1] + tok.end[1]
        if tok.type == tokenize.COMMENT or (tok.type == tokenize.STRING and tok.start in docs):
            new = _apply(tok.string, rules, True)
        elif tok.type == tokenize.STRING:
            try:
                value = ast.literal_eval(tok.string)
            except (ValueError, SyntaxError):
                value = None
            if isinstance(value, str) and value in exact:
                new = tok.string.replace(value, exact[value])
            else:
                new = tok.string
                if unsure is not None and _apply(tok.string, rules, True) != tok.string:
                    unsure.append(f"{rel}:{tok.start[0]}: {tok.string[:100]}")
        elif tok.type == getattr(tokenize, "FSTRING_MIDDLE", -1):
            new = tok.string
            if unsure is not None and _apply(tok.string, rules, True) != tok.string:
                unsure.append(f"{rel}:{tok.start[0]}: {tok.string[:100]}")
        else:
            continue
        if new != tok.string:
            edits.append((a, b, new))
    for a, b, new in reversed(edits):
        text = text[:a] + new + text[b:]
    return text


def rewrite_allowlist(text: str) -> str:
    """`tests/by_name_allowlist.json`: an entry of a moved name follows it to its new file."""
    out = []
    for line in text.splitlines(keepends=True):
        for sp in SPLITS:
            src_file = path_of(sp.src)[len(VLLM) + 1:]
            m = re.search(r'"symbol": "([^"]+)"', line)
            if f'"file": "{src_file}"' in line and m and m[1] in sp.names:
                line = line.replace(f'"file": "{src_file}"', f'"file": "{path_of(sp.dst)[len(VLLM) + 1:]}"')
        out.append(line)
    return "".join(out)


def rewrite_keep(text: str) -> str:
    """`tests/dead_code_keep.json`: the twins and vocabulary modules only tests import keep a named reason."""
    import json
    doc = json.loads(text)
    mods = doc["modules"]
    changed = False
    for m, why in KEEP.items():
        if m not in mods:
            mods[m] = why
            changed = True
    return json.dumps(doc, indent=1, ensure_ascii=False) + "\n" if changed else text


# --- checks -------------------------------------------------------------------------------------------------------------------


def definition_imports(tree: Tree) -> list[str]:
    """Every import of a Definition module of the family that is not the stdlib, numpy, `verity` or `DEFINITION_IMPORTS`."""
    bad = []
    for mod in DEFINITION_MODULES:
        rel = path_of(mod)
        t = ast.parse(tree.read(rel))
        for n in ast.walk(t):
            targets: list[str] = []
            if isinstance(n, ast.Import):
                targets = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom):
                base = _absolute(mod, n, rel.endswith("__init__.py"))
                targets = [f"{base}.{a.name}" if f"{base}.{a.name}" in DEFINITION_IMPORTS else base for a in n.names]
            for x in targets:
                top = x.split(".")[0]
                if top in sys.stdlib_module_names or top in ("numpy", "verity", "__future__") or x in DEFINITION_IMPORTS:
                    continue
                bad.append(f"{mod} line {n.lineno}: imports {x}")
    return bad


def leftovers(tree: Tree, files: list[str]) -> list[str]:
    """Imports of a moved name from its old module, or `A.name` through an alias of the old module, anywhere."""
    out = []
    for rel in files:
        if not rel.endswith(".py") or rel.startswith(EXCLUDE):
            continue
        text = tree.read(rel)
        if rewrite_importers(rel, text) != text:
            out.append(rel)
    return out


def external_attrs(tree: Tree, files: list[str]) -> set[str]:
    """`A.x` and `from M import x` names other files read of a split module (so the module's own imports of them stay)."""
    srcs = {sp.src for sp in SPLITS}
    out = set()
    for rel in files:
        if not rel.endswith(".py") or rel.startswith(EXCLUDE):
            continue
        text = tree.read(rel)
        if not any(s.rpartition(".")[2] in text for s in srcs):
            continue
        try:
            t = ast.parse(text)
        except SyntaxError:
            continue
        me = module_of(rel) or ""
        aliases = set()
        for n in ast.walk(t):
            if isinstance(n, ast.ImportFrom):
                mod = _absolute(me, n, rel.endswith("__init__.py"))
                if mod in srcs:
                    out |= {a.name for a in n.names}
                aliases |= {a.asname or a.name for a in n.names if f"{mod}.{a.name}" in srcs}
            elif isinstance(n, ast.Import):
                aliases |= {a.asname for a in n.names if a.name in srcs and a.asname}
        out |= {n.attr for n in ast.walk(t) if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in aliases}
    return out


# --- the run ------------------------------------------------------------------------------------------------------------------


def run(root: Path, check: bool = False, log=print) -> list[str]:
    tree = Tree(root)
    for pkg, doc in PACKAGES.items():
        if not tree.exists(path_of(pkg)):
            tree.write(path_of(pkg), doc)
    for m in MOVES:
        s, d = path_of(m.src), path_of(m.dst)
        if tree.exists(s) and not tree.exists(d):
            tree.move(s, d)
            log(f"moved   {m.src} -> {m.dst}")
        elif tree.exists(d) and not tree.exists(s):
            log(f"already {m.src} -> {m.dst}")
        else:
            raise Refused(f"{m.src}: {'both' if tree.exists(s) else 'neither'} {s} and {d} exist")
    ext = external_attrs(tree, tree.files())
    for sp in SPLITS:
        log(f"{split(tree, sp, ext):7.7} {sp.src}: {len(sp.names)} names -> {sp.dst}")
    files = tree.files()
    rules = text_rules()
    unsure: list[str] = []
    for rel in files:
        if rel.startswith(EXCLUDE) or Path(rel).suffix not in TEXT_SUFFIXES:
            continue
        try:
            text = tree.read(rel)
        except (UnicodeDecodeError, FileNotFoundError):
            continue
        new = rewrite_importers(rel, text) if rel.endswith(".py") else text
        new = rewrite_text(rel, new, rules, unsure)
        if rel == f"{VLLM}/tests/by_name_allowlist.json":
            new = rewrite_allowlist(new)
        if rel == f"{VLLM}/tests/dead_code_keep.json":
            new = rewrite_keep(new)
        if new != text:
            tree.write(rel, new)
    for rel in tree.dirty:
        if rel.endswith(".py"):
            try:
                ast.parse(tree.read(rel))
            except SyntaxError as e:
                raise Refused(f"{rel}: the rewrite does not parse: {e}") from None
    left = leftovers(tree, tree.files())
    if left:
        raise Refused(f"importers the rewrite did not settle: {left}")
    bad = definition_imports(tree)
    if bad:
        raise Refused("Definition modules import outside the Definitions after the split:\n  " + "\n  ".join(bad))
    for u in unsure:
        log(f"left    {u}")
    changed = tree.changes()
    if not check:
        tree.flush()
    return changed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="change nothing; exit 1 when the split would change a file")
    ap.add_argument("--root", type=Path, default=REPO)
    a = ap.parse_args(argv)
    try:
        changed = run(a.root, check=a.check)
    except Refused as e:
        print(f"split_prims_serve: refused: {e}", file=sys.stderr)
        return 2
    print(f"{len(changed)} files {'would change' if a.check else 'changed'}")
    for rel in changed:
        print(f"  {rel}")
    return 1 if (a.check and changed) else 0


if __name__ == "__main__":
    sys.exit(main())
