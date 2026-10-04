#!/usr/bin/env python3
"""Split the MoE and FP8 registry modules into Definitions, twins and vocabulary, and rewrite every import of a moved name.

    uv run python integrations/vllm/tools/migrate/split_moe_fp8.py            # apply on this checkout
    uv run python integrations/vllm/tools/migrate/split_moe_fp8.py --check    # exit 1 when anything is left to do, change nothing

The Definitions stay where they are, so every Definition keeps its module, id and descriptor digest.  What leaves the Definition
modules (`MOVES`): the scalar and numpy reference twins of the MoE router and the FP8 kernels, and the NVFP4 device-layout helpers,
to `verity_vllm.program.kernels` beside the other twins; the FP8 vocabulary (which Definition a scale construction binds, the capture
mapping's call names, the batch-invariant MoE config), to `registry.fp8_vocab`.  `EDITS` are the exact text changes that go with it:
numpy-free spellings of the Definition modules' constants (the same f32 words) and docstrings that name a moved function; `KEEP`
holds the dead-code keep entries of the twins only tests reach.

Rerunnable on a later main.  A move whose names are all still in the source applies; one whose names are all in the destination is
already done; anything else (a name in neither, or in both) fails, as does a moved name the remaining source still uses, a free name of
moved code that nothing binds, a `from <source> import *`, `getattr(<source>, "<moved name>")`, or any reference left after the
rewrite.  The rewrite covers `from <source> import <name>` (relative and function-level imports too), `<alias>.<name>` through any
binding of the source module, and `registry.<source>.<name>` / `module:name` and `<source>.<name>` in strings and comments of every
tracked Python file.
"""
from __future__ import annotations

import argparse
import ast
import builtins
import io
import json
import re
import subprocess
import sys
import tokenize
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
LINE = 130
REG = "verity_vllm.program.registry"
KER = "verity_vllm.program.kernels"


@dataclass(frozen=True)
class Move:
    src: str                          # dotted module the names leave
    dst: str                          # dotted module they go to
    names: tuple[str, ...]
    drop: tuple[str, ...] = ()        # comment blocks of the source that go with them (exact text, line by line)


@dataclass(frozen=True)
class Edit:
    path: str                         # relative to the repository
    old: str
    new: str


DESTINATIONS = {
    f"{KER}.moe_twins": '''"""Scalar twins of the MoE router's Definitions (`registry.moe`): the same instruction sequences evaluated directly on one f32
encoding, the probes' and tests' reference for `NvExpf_v1` and `NvSigmoid_v1`."""''',
    f"{KER}.fp8_twins": '''"""CPU twins of the FP8 Definitions (`registry.fp8`) on numpy arrays: the per-token and per-tensor quantization kernels and the
e4m3 conversions, for the binding cross-check and the pod experiments, and the scale products a host committer commits for
`ScaledMmFp8BlockSharedScale_v1`."""''',
    f"{KER}.nvfp4_twins": '''"""The device layouts a Commit opens for the NVFP4 linear (`registry.nvfp4`): the codes are packed two per byte, element 2i in
the low nibble; the scale bytes are the swizzled [ceil(M/128)*128, round_up(K/16, 4)] layout of
`cvt_quant_to_fp4_get_sf_out_offset`."""''',
    f"{REG}.fp8_vocab": '''"""The FP8 vocabulary: which blockwise GEMM Definition a scale construction binds (`registry.fp8`'s v1 and v3 forms), the capture
mapping's call names of the FP8 serving port, and the fused-MoE config under batch invariance (`registry.fp8_moe`)."""''',
}

MOVES = [
    Move(f"{REG}.moe", f"{KER}.moe_twins", ("nv_expf_bits", "nv_sigmoid_bits")),
    Move(f"{REG}.fp8", f"{KER}.fp8_twins",
         ("scale_products", "np_f32_to_e4m3_sat", "np_e4m3_to_f32", "np_bf16_to_f32", "np_dynamic_per_token_quant",
          "np_online_weight_quant"),
         drop=("# ---------------------------------------------------------------------------------------------------------\n"
               "# CPU reference of the two kernels on numpy arrays (for the binding cross-check and the pod experiments)\n"
               "# ---------------------------------------------------------------------------------------------------------\n",)),
    Move(f"{REG}.nvfp4", f"{KER}.nvfp4_twins", ("sf_offset", "unpack_codes")),
    Move(f"{REG}.fp8", f"{REG}.fp8_vocab", ("SHARED_SCALE", "BLOCK_GEMMS", "block_gemm", "CALL_NAMES")),
    Move(f"{REG}.fp8_moe", f"{REG}.fp8_vocab", ("BATCH_INVARIANT_KB", "BATCH_INVARIANT_SWAP")),
]

_R = "integrations/vllm/verity_vllm/program/registry"
EDITS = [
    Edit(f"{_R}/fp8.py", "MIN_SCALE_BITS = bits_of(np.float32(1.0) / (np.float32(448.0) * np.float32(512.0)))",
         "MIN_SCALE_BITS = bits_of(1.0 / (448.0 * 512.0))"),
    Edit(f"{_R}/fp8.py", "C_EPS_QUANT = b1.const(32, bits_of(np.float32(1e-10)))", "C_EPS_QUANT = b1.const(32, bits_of(1e-10))"),
    Edit(f"{_R}/boolean_fp8.py", "F448 = bits_of(np.float32(448.0))\nF_NEG448 = bits_of(np.float32(-448.0))\n"
         "EPS_QUANT = bits_of(np.float32(1e-10))", "F448 = bits_of(448.0)\nF_NEG448 = bits_of(-448.0)\nEPS_QUANT = bits_of(1e-10)"),
    Edit(f"{_R}/boolean_fp8_moe.py", "F1 = bits_of(np.float32(1.0))", "F1 = bits_of(1.0)"),
    Edit(f"{_R}/nvfp4.py", "The device layouts the Commit opens (`sf_offset`, `unpack_codes`): the codes are packed two per byte, element 2i "
         "in the low\nnibble; the scale bytes are the swizzled [ceil(M/128)*128, round_up(K/16, 4)] layout of "
         "`cvt_quant_to_fp4_get_sf_out_offset`.\n",
         "The device layouts the Commit opens (`sf_offset`, `unpack_codes`) are `kernels.nvfp4_twins`.\n"),
]

#: modules a move leaves reachable from the tests alone (`tests/test_no_dead_modules.py`), with the reason each stays
KEEP_FILE = "integrations/vllm/tests/dead_code_keep.json"
KEEP = {f"{KER}.nvfp4_twins": "the NVFP4 linear's device layouts (scale swizzle, packed E2M1 codes), whose offsets and packing "
                              "tests/program/test_nvfp4.py checks (LIVE registry.nvfp4, its Definitions); live once a Commit opens "
                              "an NVFP4 linear"}


class SplitError(RuntimeError):
    pass


# --- files and modules ----------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=None)
def python_files(root: Path) -> tuple[Path, ...]:
    """The tracked Python files (every `*.py` under ``root`` when it is not a git checkout)."""
    try:
        out = subprocess.run(["git", "ls-files", "-z", "--", "*.py"], cwd=root, capture_output=True, text=True, check=True).stdout
        return tuple(root / p for p in out.split("\0") if p and (root / p).is_file())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return tuple(sorted(p for p in root.rglob("*.py") if ".venv" not in p.parts))


def module_of(path: Path) -> str | None:
    """The dotted module a file is, from the packages (`__init__.py`) above it; None for a file outside any package."""
    parts = [] if path.name == "__init__.py" else [path.stem]
    d = path.parent
    while (d / "__init__.py").is_file():
        parts.insert(0, d.name)
        d = d.parent
    return ".".join(parts) if (parts and (path.parent / "__init__.py").is_file()) else None


@lru_cache(maxsize=None)
def package_root(root: Path, top: str) -> Path:
    """The directory holding the top-level package ``top``: the one tracked `<top>/__init__.py` with no package above it."""
    hits = sorted(p.parent.parent for p in python_files(root)
                  if p.name == "__init__.py" and p.parent.name == top and not (p.parent.parent / "__init__.py").exists())
    if len(hits) != 1:
        raise SplitError(f"package {top!r}: expected one directory, found {[str(h) for h in hits]}")
    return hits[0]


def module_path(root: Path, module: str) -> Path:
    return package_root(root, module.split(".")[0]).joinpath(*module.split(".")).with_suffix(".py")


def resolve_from(importer: str | None, is_package: bool, node: ast.ImportFrom) -> str | None:
    """The absolute module an `ImportFrom` names (relative ones against ``importer``)."""
    if not node.level:
        return node.module
    if importer is None:
        return None
    base = importer.split(".") if is_package else importer.split(".")[:-1]
    if node.level > 1:
        base = base[:-(node.level - 1)]
    return ".".join(base + ([node.module] if node.module else []))


# --- source segments --------------------------------------------------------------------------------------------------------------

def defined_names(tree: ast.Module) -> dict[str, ast.stmt]:
    """Top-level name -> the one statement defining it (def, class, or a single-target assignment)."""
    out: dict[str, ast.stmt] = {}
    for st in tree.body:
        names = []
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [st.name]
        elif isinstance(st, ast.Assign):
            names = [t.id for t in st.targets if isinstance(t, ast.Name)]
        elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name):
            names = [st.target.id]
        for n in names:
            out[n] = st
    return out


def segment(lines: list[str], st: ast.stmt) -> tuple[int, int]:
    """[start, end) line indices of ``st`` with its decorators and the comment lines directly above it."""
    start = min([st.lineno] + [d.lineno for d in getattr(st, "decorator_list", [])]) - 1
    while start > 0 and lines[start - 1].lstrip().startswith("#") and not lines[start - 1].startswith("# ---"):
        start -= 1
    return start, st.end_lineno


def _scope_bound(node: ast.AST) -> set[str]:
    """Names a definition binds inside itself (arguments, assignments, imports, nested definitions, comprehension targets)."""
    out: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
        elif isinstance(n, ast.arg):
            out.add(n.arg)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            out.update((a.asname or a.name).split(".")[0] for a in n.names)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n is not node:
            out.add(n.name)
    return out


def free_names(st: ast.stmt) -> set[str]:
    loads = {n.id for n in ast.walk(st) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    own = set(defined_names(ast.Module(body=[st], type_ignores=[])))
    inner = _scope_bound(st) - own if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else {
        n for sub in ast.walk(st) if isinstance(sub, (ast.Lambda, ast.comprehension, ast.ListComp, ast.SetComp, ast.DictComp,
                                                       ast.GeneratorExp)) for n in _scope_bound(sub)}
    return loads - inner - own - set(dir(builtins))


@dataclass
class ImportBinding:
    """How a module binds one name at top level: ``import m as n`` (``name`` None) or ``from m import name as n``."""
    module: str
    name: str | None
    alias: str

    def line(self) -> tuple[str, str]:
        if self.name is None:
            return self.module, (f"import {self.module} as {self.alias}" if self.alias != self.module.split(".")[0]
                                 or "." in self.module else f"import {self.module}")
        return self.module, self.name if self.alias == self.name else f"{self.name} as {self.alias}"


def import_bindings(tree: ast.Module, module: str, is_package: bool = False) -> dict[str, tuple[ImportBinding, ast.stmt]]:
    out = {}
    for st in tree.body:
        if isinstance(st, ast.Import):
            for a in st.names:
                out[a.asname or a.name.split(".")[0]] = (ImportBinding(a.name, None, a.asname or a.name.split(".")[0]), st)
        elif isinstance(st, ast.ImportFrom) and st.module != "__future__":
            m = resolve_from(module, is_package, st)
            for a in st.names:
                out[a.asname or a.name] = (ImportBinding(m, a.name, a.asname or a.name), st)
    return out


def _group(module: str) -> int:
    top = module.split(".")[0]
    return 0 if top in sys.stdlib_module_names else 3 if top == "verity_vllm" else 2 if top == "verity" else 1


def format_from(module: str, names: list[str], indent: str = "") -> list[str]:
    """``from module import names`` on one line, else parenthesised with the names wrapped at `LINE`."""
    one = f"{indent}from {module} import {', '.join(names)}"
    if len(one) <= LINE:
        return [one]
    out, cur, pad = [f"{indent}from {module} import ("], "", indent + "    "
    for n in names:
        piece = f"{n},"
        if cur and len(pad) + len(cur) + 1 + len(piece) > LINE:
            out.append(pad + cur)
            cur = piece
        else:
            cur = f"{cur} {piece}" if cur else piece
    out += [pad + cur, f"{indent})"]
    return out


def import_block(needed: list[ImportBinding]) -> list[str]:
    plain = sorted({b.line()[1] for b in needed if b.name is None}, key=lambda s: (_group(s.split()[1]), s))
    froms: dict[str, set[str]] = {}
    for b in needed:
        if b.name is not None:
            froms.setdefault(b.module, set()).add(b.line()[1])
    lines: list[str] = []
    last = None
    for key, text in sorted([((_group(p.split()[1]), 0, p.split()[1]), [p]) for p in plain]
                            + [((_group(m), 1, m), format_from(m, sorted(ns))) for m, ns in froms.items()]):
        if last is not None and key[0] != last and (last < 2 or key[0] < 2):
            lines.append("")
        lines += text
        last = key[0]
    return lines


# --- the split ---------------------------------------------------------------------------------------------------------------------

@dataclass
class Plan:
    root: Path
    moves: list[Move]
    destinations: dict[str, str]
    edits: list[Edit]
    log: list[str] = field(default_factory=list)
    changed: dict[Path, str] = field(default_factory=dict)       # path -> new text (written by `write`)

    def text(self, p: Path) -> str:
        return self.changed[p] if p in self.changed else p.read_text(encoding="utf-8")

    def put(self, p: Path, s: str) -> None:
        if not p.exists() or p.read_text(encoding="utf-8") != s:
            self.changed[p] = s
        else:
            self.changed.pop(p, None)

    def write(self) -> None:
        for p, s in self.changed.items():
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(s, encoding="utf-8")


def apply_edits(plan: Plan) -> None:
    """The `Edit`s, then the imports of each edited module they leave unused."""
    before: dict[Path, dict[str, int]] = {}
    for e in plan.edits:
        p = plan.root / e.path
        s = plan.text(p)
        before.setdefault(p, _uses(ast.parse(s)))
        if s.count(e.old) == 1:
            plan.put(p, s.replace(e.old, e.new))
            plan.log.append(f"edit {e.path}: {e.old.splitlines()[0][:80]}")
        elif e.new not in s:
            raise SplitError(f"{e.path}: neither the text to replace nor its replacement is there (once): {e.old[:120]!r}")
    for p, uses in before.items():
        mod = module_of(p)
        tree = ast.parse(plan.text(p))
        after = _uses(tree)
        external = external_uses(plan, mod) if mod else set()
        unused = {n for n in import_bindings(tree, mod or "") if uses.get(n) and not after.get(n) and n not in external}
        plan.put(p, prune_imports(plan.text(p), unused))


def keep_modules(plan: Plan, keep: dict[str, str]) -> None:
    """Add the entries of ``keep`` that `KEEP_FILE` lacks, after its last `kernels` entry (else first in its `modules`)."""
    if not keep:
        return
    p = plan.root / KEEP_FILE
    text = plan.text(p)
    new = {m: r for m, r in keep.items() if m not in json.loads(text)["modules"]}
    if not new:
        return
    lines = text.splitlines(keepends=True)
    anchors = [i for i, s in enumerate(lines) if s == ' "modules": {\n']
    if len(anchors) != 1:
        raise SplitError(f"{KEEP_FILE}: expected one ` \"modules\": {{` line, found {len(anchors)}")
    at = max([i for i, s in enumerate(lines) if s.startswith(f'  "{KER}.')], default=anchors[0]) + 1
    add = [f"  {json.dumps(m)}: {json.dumps(r)},\n" for m, r in new.items()]
    if at - 1 != anchors[0] and not lines[at - 1].endswith(",\n"):          # after the last entry
        lines[at - 1] = lines[at - 1].rstrip("\n") + ",\n"
        add[-1] = add[-1].removesuffix(",\n") + "\n"
    lines[at:at] = add
    out = "".join(lines)
    if {**json.loads(text)["modules"], **new} != json.loads(out)["modules"]:
        raise SplitError(f"{KEEP_FILE}: the added entries do not parse back")
    plan.put(p, out)
    plan.log.append(f"keep {', '.join(new)} in {KEEP_FILE}")


def moved_index(moves: list[Move]) -> dict[tuple[str, str], str]:
    out = {}
    for m in moves:
        for n in m.names:
            if (m.src, n) in out:
                raise SplitError(f"{m.src}.{n} moves twice")
            out[(m.src, n)] = m.dst
    return out


def split_modules(plan: Plan) -> list[Move]:
    """Move the names, destination by destination; returns the moves applied by this run."""
    index = moved_index(plan.moves)
    by_dst: dict[str, list[Move]] = {}
    for m in plan.moves:
        by_dst.setdefault(m.dst, []).append(m)
    todo: list[Move] = []
    for dst, moves in by_dst.items():
        dpath = module_path(plan.root, dst)
        in_dst = set(defined_names(ast.parse(plan.text(dpath)))) if dpath.exists() else set()
        state = []
        for m in moves:
            in_src = set(defined_names(ast.parse(plan.text(module_path(plan.root, m.src)))))
            for n in m.names:
                state.append((m, n, n in in_src, n in in_dst))
        if all(s and not d for _, _, s, d in state):
            if dpath.exists():
                raise SplitError(f"{dst} exists but defines none of the names moving there")
            todo += moves
        elif not all(d and not s for _, _, s, d in state):
            raise SplitError(f"{dst}: partially applied or diverged: " + ", ".join(
                f"{m.src}.{n} (source: {s}, destination: {d})" for m, n, s, d in state if not (d and not s)))
        else:
            plan.log.append(f"already split: {dst}")
    if not todo:
        return []

    sources: dict[str, tuple[str, ast.Module]] = {}
    for m in todo:
        if m.src not in sources:
            s = plan.text(module_path(plan.root, m.src))
            sources[m.src] = (s, ast.parse(s))
    removals: dict[str, list[tuple[int, int]]] = {}
    for dst, moves in by_dst.items():
        if moves[0] not in todo:
            continue
        chunks: list[tuple[str, int, int, str]] = []
        needed: list[ImportBinding] = []
        own = {n for m in moves for n in m.names}
        for m in moves:
            text, tree = sources[m.src]
            lines = text.splitlines(keepends=True)
            defs = defined_names(tree)
            binds = import_bindings(tree, m.src)
            for n in sorted(m.names, key=lambda n: defs[n].lineno):
                st = defs[n]
                a, b = segment(lines, st)
                chunks.append((m.src, a, b, "".join(lines[a:b]).rstrip("\n") + "\n"))
                removals.setdefault(m.src, []).append((a, b))
                for f in sorted(free_names(st) - own):
                    if (m.src, f) in index:
                        needed.append(ImportBinding(index[(m.src, f)], f, f))
                    elif f in binds:
                        needed.append(binds[f][0])
                    elif f in defs:
                        needed.append(ImportBinding(m.src, f, f))
                    else:
                        raise SplitError(f"{m.src}.{n} reads {f!r}, which nothing in {m.src} binds")
            for block in m.drop:
                start = text.find(block)
                if start < 0:
                    raise SplitError(f"{m.src}: the comment block to drop is not there: {block.splitlines()[1][:100]!r}")
                a = text.count("\n", 0, start)
                removals[m.src].append((a, a + block.count("\n")))
        body = chunks[0][3]
        for prev, cur in zip(chunks, chunks[1:]):
            body += ("" if (prev[0], prev[2]) == (cur[0], cur[1]) else "\n\n") + cur[3]
        imports = "".join(f"{s}\n" for s in import_block(needed))
        head = plan.destinations[dst].rstrip() + "\nfrom __future__ import annotations\n\n" + (imports + "\n" if imports else "")
        plan.put(module_path(plan.root, dst), head + "\n" + body)
        plan.log.append(f"new {dst}: " + ", ".join(n for m in moves for n in m.names))

    for src, cuts in removals.items():
        text, tree = sources[src]
        plan.put(module_path(plan.root, src), cut_source(plan, src, text, tree, cuts, {n for m in todo if m.src == src for n in m.names}))
    return todo


def _collapse(lines: list[str], at: int) -> int:
    """Leave at most two blank lines around index ``at`` (none at the end of the file); returns the new index."""
    a = at
    while a > 0 and not lines[a - 1].strip():
        a -= 1
    b = at
    while b < len(lines) and not lines[b].strip():
        b += 1
    keep = 0 if b == len(lines) else min(2, b - a)
    lines[a:b] = ["\n"] * keep
    return a


def cut_source(plan: Plan, src: str, text: str, tree: ast.Module, cuts: list[tuple[int, int]], moved: set[str]) -> str:
    lines = text.splitlines(keepends=True)
    before = _uses(tree)
    for a, b in sorted(cuts, reverse=True):
        del lines[a:b]
        _collapse(lines, a)
    out = "".join(lines)
    out = drop_from_all(out, moved)
    tree2 = ast.parse(out)
    still = sorted(n for n in moved if n in _uses(tree2))
    if still:
        raise SplitError(f"{src} still uses the moved names {still}")
    after = _uses(tree2)
    external = external_uses(plan, src)
    unused = {n for n, (b, _) in import_bindings(tree2, src).items() if before.get(n) and not after.get(n) and n not in external}
    return prune_imports(out, unused)


def _uses(tree: ast.Module) -> dict[str, int]:
    out: dict[str, int] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
            out[n.id] = out.get(n.id, 0) + 1
    for st in tree.body:                         # `__all__` keeps a name in use
        if isinstance(st, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in st.targets):
            for c in ast.walk(st.value):
                if isinstance(c, ast.Constant) and isinstance(c.value, str):
                    out[c.value] = out.get(c.value, 0) + 1
    return out


def drop_from_all(text: str, moved: set[str]) -> str:
    tree = ast.parse(text)
    for st in tree.body:
        if not (isinstance(st, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in st.targets)):
            continue
        if not isinstance(st.value, (ast.List, ast.Tuple)):
            raise SplitError("__all__ is not a literal list")
        names = [e.value for e in st.value.elts]
        if not set(names) & moved:
            return text
        keep = [n for n in names if n not in moved]
        lines = text.splitlines(keepends=True)
        open_, close = ("[", "]") if isinstance(st.value, ast.List) else ("(", ")")
        head = f"__all__ = {open_}"
        out, cur = [], head
        for n in keep:
            piece = f'"{n}", '
            if len(cur) + len(piece) > LINE and cur.strip() != head.strip():
                out.append(cur.rstrip() + "\n")
                cur = " " * len(head)
            cur += piece
        out.append(cur.rstrip().rstrip(",") + close + "\n")
        lines[st.lineno - 1:st.end_lineno] = out
        return "".join(lines)
    return text


def _comments(text: str, first: int, last: int) -> list[tuple[int, int, str]]:
    toks = tokenize.generate_tokens(io.StringIO(text).readline)
    return [(t.start[0], t.start[1], t.string) for t in toks if t.type == tokenize.COMMENT and first <= t.start[0] <= last]


def _rewrite_import(text: str, st: ast.ImportFrom | ast.Import, keep: list[ast.alias], extra: list[str]) -> tuple[int, int, list[str]]:
    """The replacement lines of ``st``: its names ``keep`` (none: the statement goes) and then the ``extra`` lines, which carry the
    statement's comment where they write ``{comment}``: a `# noqa` always, any other comment when the whole statement moves."""
    lines = text.splitlines(keepends=True)
    indent = re.match(r"\s*", lines[st.lineno - 1]).group(0)
    comments = _comments(text, st.lineno, st.end_lineno)
    if len(comments) > 1 or (comments and comments[0][0] != st.lineno):
        raise SplitError(f"line {st.lineno}: an import with comments inside it: rewrite it by hand")
    comment, col = (comments[0][2], comments[0][1]) if comments else ("", 0)
    single = st.lineno == st.end_lineno

    def placed(code: str, c: str) -> str:
        if not c:
            return code
        if "(" in code and code.rstrip().endswith("("):
            return f"{code}  {c}"
        return code + " " * (max(2, col - len(code)) if single else 2) + c

    out: list[str] = []
    if keep:
        names = [a.name if not a.asname else f"{a.name} as {a.asname}" for a in keep]
        if isinstance(st, ast.Import):
            out = [placed(f"{indent}import {', '.join(names)}", comment)]
        else:
            mod = "." * st.level + (st.module or "")
            out = format_from(mod, names, indent)
            out[0] = placed(out[0], comment)
    carried = comment if (comment.startswith("# noqa") or not keep) else ""
    for e in extra:
        code = indent + e.replace("{comment}", "").rstrip()
        out.append(placed(code, carried) if "{comment}" in e else code)
    return st.lineno - 1, st.end_lineno, [s + "\n" for s in out]


def prune_imports(text: str, unused: set[str]) -> str:
    if not unused:
        return text
    tree = ast.parse(text)
    edits = []
    for st in tree.body:
        if isinstance(st, (ast.Import, ast.ImportFrom)):
            names = [a for a in st.names if (a.asname or a.name.split(".")[0]) not in unused]
            if len(names) != len(st.names):
                edits.append(_rewrite_import(text, st, names, []))
    lines = text.splitlines(keepends=True)
    for a, b, new in sorted(edits, reverse=True):
        lines[a:b] = new
        if not new and a < len(lines) and not lines[a].strip() and a > 0 and not lines[a - 1].strip():
            del lines[a]                         # a whole import gone: one blank line between the groups around it
    return "".join(lines)


# --- the repository-wide rewrite ---------------------------------------------------------------------------------------------------

def _source_aliases(tree: ast.AST, mod: str | None, is_pkg: bool, sources: set[str]) -> list[tuple[ast.stmt, ast.alias, str, ast.AST]]:
    """(statement, alias, source module, scope) for every binding of a source module as a name, at any depth."""
    out = []

    def visit(scope: ast.AST, body: list[ast.stmt]) -> None:
        for st in body:
            if isinstance(st, ast.Import):
                for a in st.names:
                    if a.name in sources:
                        if not a.asname:
                            raise SplitError(f"line {st.lineno}: `import {a.name}` without an alias: rewrite it by hand")
                        out.append((st, a, a.name, scope))
            elif isinstance(st, ast.ImportFrom):
                m = resolve_from(mod, is_pkg, st)
                for a in st.names:
                    if m and f"{m}.{a.name}" in sources:
                        out.append((st, a, f"{m}.{a.name}", scope))
            for child_body in ("body", "orelse", "finalbody", "handlers"):
                sub = getattr(st, child_body, None)
                if isinstance(sub, list):
                    if child_body == "handlers":
                        for h in sub:
                            visit(scope, h.body)
                    else:
                        visit(st if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)) else scope, sub)

    visit(tree, tree.body)
    return out


def _scope_nodes(scope: ast.AST, alias: str, top: bool):
    """The nodes in ``scope`` where ``alias`` means the binding: all of it, less (for a module-level binding) the functions that
    bind ``alias`` themselves."""
    stack = [scope]
    while stack:
        n = stack.pop()
        yield n
        for c in ast.iter_child_nodes(n):
            if top and isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) and alias in _scope_bound(c):
                continue
            stack.append(c)


def _char_col(line: str, byte_col: int) -> int:
    """`ast` columns count UTF-8 bytes; `tokenize` and slicing count characters."""
    return len(line.encode("utf-8")[:byte_col].decode("utf-8"))


@dataclass
class _StmtEdit:
    st: ast.stmt
    drop: list[ast.alias] = field(default_factory=list)
    extra: list[str] = field(default_factory=list)


def rewrite_file(path: Path, text: str, index: dict[tuple[str, str], str]) -> str:
    sources = {s for s, _ in index}
    names = {n for _, n in index}
    if not any(n in text for n in names) and "*" not in text:
        return text
    tree = ast.parse(text)
    mod = module_of(path)
    is_pkg = path.name == "__init__.py"
    stmts: dict[int, _StmtEdit] = {}                  # id(statement) -> what changes in it
    spans: list[tuple[int, int, int, str]] = []       # (line, byte col, byte end col, new) replacements of an alias

    def edit(st: ast.stmt) -> _StmtEdit:
        return stmts.setdefault(id(st), _StmtEdit(st))

    # `from <source> import <moved>`
    for st in ast.walk(tree):
        if not isinstance(st, ast.ImportFrom):
            continue
        m = resolve_from(mod, is_pkg, st)
        if m not in sources:
            continue
        if any(a.name == "*" for a in st.names):
            raise SplitError(f"{path}:{st.lineno}: `from {m} import *`")
        moving = [a for a in st.names if (m, a.name) in index]
        if not moving:
            continue
        by: dict[str, list[str]] = {}
        for a in moving:
            by.setdefault(index[(m, a.name)], []).append(a.name if not a.asname else f"{a.name} as {a.asname}")
        e = edit(st)
        e.drop += moving
        for dst, ns in by.items():
            new = format_from(dst, ns)
            e.extra += [new[0] + "{comment}"] + new[1:]

    # `<alias>.<moved>` through a binding of the source module
    for st, a, src, scope in _source_aliases(tree, mod, is_pkg, sources):
        alias = a.asname or a.name
        top = scope is tree
        hits = []
        for n in _scope_nodes(scope, alias, top):
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == alias and (src, n.attr) in index:
                hits.append(n)
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("getattr", "setattr", "hasattr", "delattr")
                    and len(n.args) > 1 and isinstance(n.args[0], ast.Name) and n.args[0].id == alias
                    and isinstance(n.args[1], ast.Constant) and (src, n.args[1].value) in index):
                raise SplitError(f"{path}:{n.lineno}: {n.func.id}({alias}, {n.args[1].value!r}) names a moved name by string")
        if not hits:
            continue
        where = tree if top else scope
        bound = _scope_bound(where)
        new_aliases: dict[str, str] = {}
        for n in hits:
            dst = index[(src, n.attr)]
            last = dst.rsplit(".", 1)[1]
            d = new_aliases.setdefault(dst, ("_" + last) if alias.startswith("_") else last)
            if d in bound and not _already_binds(where, d, dst):
                raise SplitError(f"{path}: the name {d!r} for {dst} is already bound to something else")
            spans.append((n.value.lineno, n.value.col_offset, n.value.end_col_offset, d))
        remaining = sum(1 for n in _scope_nodes(scope, alias, top)
                        if isinstance(n, ast.Name) and n.id == alias and isinstance(n.ctx, ast.Load)) - len(hits)
        e = edit(st)
        if remaining == 0:
            e.drop.append(a)
        for dst, d in new_aliases.items():
            if not _already_binds(where, d, dst):
                pkg, last = dst.rsplit(".", 1)
                e.extra.append(f"from {pkg} import {last}" + (f" as {d}" if d != last else "") + "{comment}")

    lines = text.splitlines(keepends=True)
    for line, col, end, new in sorted(spans, reverse=True):
        s = lines[line - 1]
        a, b = _char_col(s, col), _char_col(s, end)
        lines[line - 1] = s[:a] + new + s[b:]
    for e in sorted(stmts.values(), key=lambda e: e.st.lineno, reverse=True):
        a, b, new = _rewrite_import(text, e.st, [x for x in e.st.names if x not in e.drop], e.extra)
        lines[a:b] = new
    return rewrite_text("".join(lines), index)


def _already_binds(scope: ast.AST, d: str, dst: str) -> bool:
    pkg, last = dst.rsplit(".", 1)
    for n in ast.walk(scope):
        if isinstance(n, ast.ImportFrom) and n.module == pkg and any(a.name == last and (a.asname or a.name) == d for a in n.names):
            return True
    return False


def _text_patterns(index: dict[tuple[str, str], str]) -> list[tuple[re.Pattern, str]]:
    out = []
    for (src, n), dst in sorted(index.items()):
        s_short, d_short = src.rsplit(".", 1)[1], dst.rsplit(".", 1)[1]
        s_rel, d_rel = src.removeprefix("verity_vllm.program."), dst.removeprefix("verity_vllm.program.")
        out += [(re.compile(rf"(?<![\w.]){re.escape(src)}[.:]{n}\b"), f"{dst}.{n}"),
                (re.compile(rf"(?<![\w.]){re.escape(s_rel)}[.:]{n}\b"), f"{d_rel}.{n}"),
                (re.compile(rf"(?<![\w.]){re.escape(s_short)}\.{n}\b"), f"{d_short}.{n}")]
    return out


_TEXT_TOKENS = {tokenize.STRING, tokenize.COMMENT, getattr(tokenize, "FSTRING_MIDDLE", tokenize.STRING)}


def _text_spans(text: str):
    """((line, col), (end line, end col), source text) of every string and comment token."""
    lines = text.splitlines(keepends=True)
    for t in tokenize.generate_tokens(io.StringIO(text).readline):
        if t.type in _TEXT_TOKENS:
            (l0, c0), (l1, c1) = t.start, t.end
            s = lines[l0 - 1][c0:] + "".join(lines[l0:l1 - 1]) + lines[l1 - 1][:c1] if l1 > l0 else lines[l0 - 1][c0:c1]
            yield t.start, t.end, s


def rewrite_text(text: str, index: dict[tuple[str, str], str]) -> str:
    """References in strings and comments: `<module>.<name>` / `<module>:<name>`, `registry.<source>.<name>`, `<source>.<name>`."""
    pats = _text_patterns(index)
    edits = []
    for start, end, s in _text_spans(text):
        new = s
        for p, r in pats:
            new = p.sub(r, new)
        if new != s:
            edits.append((start, end, new))
    lines = text.splitlines(keepends=True)
    for (l0, c0), (l1, c1), s in sorted(edits, reverse=True):
        lines[l0 - 1:l1] = [lines[l0 - 1][:c0] + s + lines[l1 - 1][c1:]]
    return "".join(lines)


def residue(path: Path, text: str, index: dict[tuple[str, str], str]) -> list[str]:
    """What still names a moved name at its old home."""
    names = {n for _, n in index}
    if not any(n in text for n in names):
        return []
    sources = {s for s, _ in index}
    out = []
    tree = ast.parse(text)
    mod, is_pkg = module_of(path), path.name == "__init__.py"
    for st in ast.walk(tree):
        if isinstance(st, ast.ImportFrom) and resolve_from(mod, is_pkg, st) in sources:
            m = resolve_from(mod, is_pkg, st)
            out += [f"{path}:{st.lineno}: from {m} import {a.name}" for a in st.names if (m, a.name) in index]
    for st, a, src, scope in _source_aliases(tree, mod, is_pkg, sources):
        alias = a.asname or a.name
        out += [f"{path}:{n.lineno}: {alias}.{n.attr}" for n in _scope_nodes(scope, alias, scope is tree)
                if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == alias and (src, n.attr) in index]
    pats = _text_patterns(index)
    for start, _, s in _text_spans(text):
        out += [f"{path}:{start[0]}: {m.group(0)}" for p, _ in pats for m in p.finditer(s)]
    return out


def external_uses(plan: Plan, src: str) -> set[str]:
    """Names other files import from ``src`` or read through a binding of it (an import the source may not drop)."""
    out: set[str] = set()
    short = src.rsplit(".", 1)[1]
    for p in python_files(plan.root):
        text = plan.text(p)
        if short not in text:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        mod, is_pkg = module_of(p), p.name == "__init__.py"
        if mod == src:
            continue
        for st in ast.walk(tree):
            if isinstance(st, ast.ImportFrom) and resolve_from(mod, is_pkg, st) == src:
                out.update(a.name for a in st.names)
        for st, a, s, scope in _source_aliases(tree, mod, is_pkg, {src}):
            alias = a.asname or a.name
            out.update(n.attr for n in _scope_nodes(scope, alias, scope is tree)
                       if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == alias)
    return out


def run(root: Path = REPO, moves: list[Move] = MOVES, destinations: dict[str, str] = DESTINATIONS, edits: list[Edit] = EDITS,
        keep: dict[str, str] = KEEP) -> Plan:
    """Plan the split of this checkout (`Plan.changed`; `Plan.write` applies it).  Raises `SplitError` on anything unexpected."""
    python_files.cache_clear()
    package_root.cache_clear()
    plan = Plan(root, list(moves), dict(destinations), list(edits))
    index = moved_index(plan.moves)
    missing = sorted({m.dst for m in plan.moves} - set(plan.destinations))
    if missing:
        raise SplitError(f"no docstring for the destinations {missing}")
    apply_edits(plan)
    split_modules(plan)
    keep_modules(plan, keep)
    files = sorted(set(python_files(root)) | {p for p in plan.changed if p.suffix == ".py"})
    for p in files:
        text = plan.text(p)
        try:
            new = rewrite_file(p, text, index)
        except SyntaxError:
            if any(n in text for _, n in index):
                raise SplitError(f"{p}: does not parse and mentions a moved name")
            continue
        if new != text:
            plan.put(p, new)
            plan.log.append(f"rewrite {p.relative_to(root)}")
    left = [r for p in files for r in _safe_residue(p, plan.text(p), index)]
    if left:
        raise SplitError("references left at the old homes:\n" + "\n".join(left))
    return plan


def _safe_residue(p: Path, text: str, index) -> list[str]:
    try:
        return residue(p, text, index)
    except SyntaxError:
        return []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=REPO)
    ap.add_argument("--check", action="store_true", help="change nothing; exit 1 when the split is not complete")
    a = ap.parse_args(argv)
    try:
        plan = run(a.root)
    except SplitError as e:
        print(f"split_moe_fp8: {e}", file=sys.stderr)
        return 2
    for line in plan.log:
        print(line)
    if a.check:
        for p in sorted(plan.changed):
            print(f"would change {p.relative_to(a.root)}")
        return 1 if plan.changed else 0
    plan.write()
    print(f"{len(plan.changed)} files changed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
