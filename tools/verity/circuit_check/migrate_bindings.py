"""Move circuit-check's bindings and pins out of `targets.py` and ``pins/`` into the sidecars of the Definition modules they
bind (`circuit_check.sidecars`), and check that circuit-check's task list is unchanged.

    python tools/verity/circuit_check/migrate_bindings.py [--verify] [--dry-run]

What moves: each element of a flat root function (`flat`), each `_model_roots` entry and each ``TEMPLATE_PARAMS`` entry, each
to the module that defines what it binds; and `_pouw_roots`, split by hand (`POUW`).  An element's module is that of its head
(`head`: ``bind``'s Definition, a ``.bind``'s object, a called function, else the expression itself), resolved at run time:
`sidecars.owner` for a Definition, a function's or class's module, else the module it was read from.  A template's module is
the one that defines the templates.  Each pin moves to the pins file of the module that owns it (`sidecars.pin_owner`).  What
resolves to no module of `sidecars.PACKAGES`, or would make a core or protocol sidecar import circuit-check, stays in
`targets.py` or ``pins/``, and so do `targets.py`'s own root functions (`SHARED`).

A root function's docstring goes with its elements when they all go to one module; one split across modules is split by `DOCS`,
written against that docstring (its digest): a changed docstring fails until `DOCS` follows it, and so does a changed
`_pouw_roots`.  A sidecar that already defines a binding function the move would write is an error (merge it by hand); on a
migrated tree there is nothing to move, so a rebase is: drop the applied commit, rerun this on the new tree.

``--verify`` builds the task list (`TASK_LIST`) before and after, each in a fresh process, and fails unless the targets, their
statics and gate counts, the root errors and every pin are unchanged, and each target's binding units are the same once their
file is dropped from their names.
"""
from __future__ import annotations

import argparse
import ast
import builtins
import hashlib
import importlib
import inspect
import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TARGETS = HERE / "src" / "circuit_check" / "targets.py"
PINS = HERE / "src" / "circuit_check" / "pins"
LEGACY_PINS = PINS.with_name("pins.json")
#: the source roots this tree's packages import from, ahead of any installed copy
PATHS = ("tools/verity/circuit_check/src", ".", "integrations/vllm", "benchmarks/numerical/python", "verity/ml/flock/python")
#: `targets.py`'s own: what no one module owns (every kernel's and vocabulary kind's self-check bindings)
SHARED = ("_kernel_roots", "_vocabulary_bindings", "_vocabulary_roots")
#: split by hand (`POUW`)
BY_HAND = ("_pouw_roots",)
#: packages whose sidecars may import only the core and themselves (`test_boundaries.py`, `test_protocol_boundaries.py`)
NO_CHECKER = ("verity",)

_R = "verity_vllm.program.registry."
#: a root function's docstring split by module, each part one paragraph (`_docstring` wraps it), with the digest of the
#: docstring it splits (`_doc_digest`); what said that a module's other Definitions are static-free (roots of their own,
#: `targets.catalog`) is left out
DOCS: dict[str, tuple[str, dict[str, str]]] = {
    "_boolean_roots": ("12c4223fdfc84cb3", {
        "verity.ml.circuits.ops.boolean.gemm": "The Boolean BF16 coordinate and dot on both steps at K = 64, the GEMM at K = 32, N = 3, the E4M3 "
                                  "coordinate on both FP8 steps at K = 64, and the NVFP4 and MXFP4 coordinates at K = 128.",
        _R + "boolean_silu": "`silu_and_mul`'s rows at I = 8: `SiluMul_v3` and `SiluMul_v4` (v4's word view is the served "
                             "`SiluMul_v2` of `registry.act`); `SiluMulBf16_v3` is static-free.",
        _R + "act": "`silu_and_mul`'s served row, `SiluMul_v2`, at I = 8: the word view of `boolean_silu`'s `SiluMul_v4`.",
        _R + "boolean_activation": "The final-logit softcap's tanh row at N = 8.",
        _R + "boolean_dense": "`registry.dense`'s rows on bits at 8 elements with Gemma-2's constants: `gelu_tanh_and_mul` at "
                              "I = 8, the RMSNorm chain's and the logit softcap's element-wise rows at N = 8.",
        _R + "boolean_moe": "OLMoE's routed sum at TOPK = 8 over 8 columns and its router at 8 experts, 2 slots, 4 values per "
                            "thread, and GLM-5.3's sigmoid `noaux_tc` router at 8 experts, 2 slots, on both of its kernels, on "
                            "bits (its word view is `registry.moe.MoeRouterSigmoidBias_v1`).",
        _R + "moe": "GLM-5.3's sigmoid `noaux_tc` router (`MoeRouterSigmoidBias_v1`, the word view of `boolean_moe`'s) at 8 "
                    "experts, 2 slots, on both of its kernels.",
        _R + "boolean_rope": "vLLM's Boolean RoPE at the kernel rows' `RoPE_v1` binding, which reaches its head and halves, "
                             "and vLLM's GPT-J RoPE on bits at two heads of 8.",
        _R + "mla": "MLA: Triton's one-split decode at one head of two keys over a 16 + 16 latent (a partial block, so its "
                    "masked columns), which reaches the head, the block, the score and every element; the absorb bmm at two "
                    "heads, K = 32, N = 3; the shared-expert combine at 8 elements with GLM-4.7-Flash's "
                    "routed_scaling_factor.",
        _R + "sampling_boolean": "Greedy token selection (`TokenSelect_v2`) at one, two and three levels of its tree, which "
                                 "reach every stage, leaf and merge shape (a full group, a remainder, the first group, the root "
                                 "group); the f32 argmax tree's stages at two and three levels reach its other shapes.  The "
                                 "argmax step is static-free.",
        _R + "boolean_norms": "The served norms on bits at N = 16 with their MUFU statics at the word ids and at mufu's Boolean "
                              "versions (the switch's binding, `verity_vllm.program.boolean`), the fused norm's lanes at two "
                              "elements per thread, and the word Definitions of its constant steps.",
        _R + "boolean_dense_norm": "The rest of Gemma-2's norm chain on bits (over `boolean_dense`'s rows): the mean at N = 16 "
                                   "and at N = 1030 (two chunks, one element per thread), the narrowing at N = 16, and the "
                                   "rsqrt with its MUFU at the word id and at mufu's `RsqrtApprox_v2`.",
        _R + "gumbel_boolean": "The Gumbel select (`GumbelSelectF32_v2`) at two lanes (its family's smallest, and the noisy "
                               "row's: about 600,000 gates each, every lane about 300,000) and at a full group plus a remainder; "
                               "the stream key, setup, lane and their pieces are static-free.",
        _R + "boolean_softcap": "FA2 with logit soft-capping on bits at one head of 17 keys, two blocks (a full one and a "
                                "partial one) with Check_inf in both, its ``TANH`` and ``EX2`` at mufu's Boolean versions, which "
                                "reaches its head and blocks.",
    }),
}

#: `_pouw_roots` split by module: the digest of the function it splits (`_digest`) and each module's function.  The suite
#: checks every binding file's `_pouw_roots` whatever their size (`targets.suite`)
POUW: tuple[str, dict[str, str]] = ("95c09339a3fda91f", {
    "verity.core.protocols.pouw.circuit.cap": '''
def _pouw_roots() -> list[Any]:
    """The prototype of Pearl-C's per-tile cap: its row (pc8's with the salt-dead counts), its word and a one-word tile at
    k = 256 (two promotion windows) on sm_120, the split's units (the strip, the word unit, a tile row's digests and sums,
    the tile and its tally) at the same k, and a tile row's digests and sums at the deployed 64 columns (two chunks)."""
    from verity.ml.circuits.defs import bind
    from verity.core.protocols.pouw.circuit import cap

    out = [*cap.templates(256, "sm120").values()]
    out += [t for r, t in cap.unit_templates(256, "sm120").items()
            if r in ("strip", "word", "row_digest", "row_tally", "tile", "tally")]
    return out + [bind(cap.Pc8CapRowDigestDef, TN=64), bind(cap.Pc8CapRowTallyDef, K=256, TN=64)]
''',
    "verity.core.protocols.pouw.circuit.ncp2": '''
def _pouw_roots() -> list[Any]:
    """`ncp-v2`'s unit templates at k = 64, and the smallest whole call, its weight side and a window of two calls on it, for
    both P's (`ncp-v2`'s word P and `ncp-v2-shift24`); the package's own tests check their template partitions and the Z
    separator."""
    from verity.core.protocols.pouw.circuit import ncp2

    shape = ncp2.Shape(16, 64, 16)
    out = []
    for kind in ("shift24", "word"):
        out += [t for t in ncp2.templates(64, kind).values() if t not in out]
        out += [ncp2.linear(shape, kind), ncp2.weight(shape, kind), ncp2.window(shape, 2, kind)]
    return out
''',
    "verity.core.protocols.pouw.circuit.pc4": '''
def _pouw_roots() -> list[Any]:
    """Pearl-C4's tile check: its unit templates at k = 64 with a one-word tile (small enough to lower whole), and its weight
    side, call side and whole check at one row a side; its hiding tiles and roots (``hidden``: the tile digest,
    `verity.core.protocols.pouw.circuit.leaves`; their rows are C-Flock's ports) at the same shapes; and its roots with the row units cut
    along k (`verity.core.protocols.pouw.circuit.rowk`, ``Shape.row_k``) in two segments at k = 128, plain and hiding."""
    from dataclasses import replace

    from verity.core.protocols.pouw.circuit import pc4

    tile = pc4.Shape(1, 64, 1, tile=(1, 1))
    out = [*pc4.templates(64, tile.tile).values(), pc4.weight(tile), pc4.call(tile), pc4.check(tile)]
    h4 = replace(tile, hidden=True)
    out += [t for t in [*pc4.templates(64, h4.tile, True).values(), pc4.weight(h4), pc4.call(h4), pc4.check(h4)] if t not in out]
    s = pc4.Shape(1, 128, 1, tile=(1, 1), row_k=64)
    cut = [build(replace(s, hidden=h)) for h in (False, True) for build in (pc4.weight, pc4.call, pc4.check)]
    seen = {t.id for t in out}
    return out + [t for t in cut if t.id not in seen and not seen.add(t.id)]
''',
    "verity.core.protocols.pouw.circuit.pc8": '''
def _pouw_roots() -> list[Any]:
    """Pearl-C's tile check as Pearl-C4's at k = 128 (one promotion window): its templates on the H100's and sm_120's
    records and its roots on sm_120's; its hiding templates and roots on sm_120's; its hiding roots with the per-tile cap
    (``Shape.cap``, `verity.core.protocols.pouw.circuit.cap`) at k = 256 (two promotion windows) on sm_120's with
    one row a side; and its roots with the row units cut along k (`verity.core.protocols.pouw.circuit.rowk`,
    ``Shape.row_k``) in two segments at k = 256 on sm_120's, plain and hiding."""
    from dataclasses import replace

    from verity.core.protocols.pouw.circuit import pc8

    tile8 = pc8.Shape(1, 128, 1, tile=(1, 1), dev="sm120")
    out = [*pc8.templates(128, tile8.tile, "h100").values(), *pc8.templates(128, tile8.tile, "sm120").values(),
           pc8.weight(tile8), pc8.call(tile8), pc8.check(tile8)]
    h8 = replace(tile8, hidden=True)
    out += [t for t in [*pc8.templates(128, h8.tile, "sm120", True).values(), pc8.weight(h8), pc8.call(h8), pc8.check(h8)]
            if t not in out]
    split = pc8.Shape(1, 256, 1, tile=(1, 1), dev="sm120", hidden=True, cap=True)
    out += [pc8.call(split), pc8.check(split)]
    s = pc8.Shape(1, 256, 1, tile=(1, 1), dev="sm120", row_k=128)
    cut = [build(replace(s, hidden=h)) for h in (False, True) for build in (pc8.weight, pc8.call, pc8.check)]
    seen = {t.id for t in out}
    return out + [t for t in cut if t.id not in seen and not seen.add(t.id)]
''',
    "verity.core.protocols.pouw.circuit.rowk": '''
def _pouw_roots() -> list[Any]:
    """Both tile checks' row units cut along k (``Shape.row_k``) in two segments: Pearl-C4's pieces at k = 128, and Pearl-C's
    at k = 256 on the H100's and sm_120's records and at k = 64 unpromoted."""
    from verity.core.protocols.pouw.circuit import rowk

    cut = [p for side in ("a", "b") for pieces in (rowk.pieces4(128, 64, side), rowk.pieces8(256, 128, "h100", side),
                                                    rowk.pieces8(256, 128, "sm120", side),
                                                    rowk.pieces8(64, 32, "sm120-unpromoted", side))
           for p in (pieces[0], pieces[1], *pieces[2], pieces[3])]
    seen: set[str] = set()
    return [t for t in cut if t.id not in seen and not seen.add(t.id)]
''',
    _R + "pouw_rows": '''
def _pouw_roots() -> list[Any]:
    """The model Program's rows at a PoUW site, without and with the bias."""
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import pouw_rows

    return [bind(pouw_rows.NcpLinearRow, K=64, N=16), bind(pouw_rows.NcpLinearRowBias, K=64, N=16),
            bind(pouw_rows.NcpLinearCoordinate, K=64)]
''',
})


def _paths() -> list[str]:
    return [str(REPO / p) for p in PATHS]


for _p in reversed(_paths()):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# --- reading targets.py -------------------------------------------------------------------------------------------------------

def _digest(node: ast.AST) -> str:
    return hashlib.sha256(ast.dump(node).encode()).hexdigest()[:16]


def _doc_digest(doc: str | None) -> str:
    return hashlib.sha256(" ".join((doc or "").split()).encode()).hexdigest()[:16]


class Source:
    def __init__(self, text: str):
        self.text = text
        self.lines = text.split("\n")
        self.tree = ast.parse(text)
        self.top: dict[str, ast.stmt] = {}
        self.imports: dict[str, ast.stmt] = {}
        for n in self.tree.body:
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)):
                self.top[n.name] = n
            elif isinstance(n, (ast.Assign, ast.AnnAssign)):
                for t in (n.targets if isinstance(n, ast.Assign) else [n.target]):
                    if isinstance(t, ast.Name):
                        self.top[t.id] = n
            elif isinstance(n, (ast.Import, ast.ImportFrom)):
                for a in n.names:
                    self.imports[(a.asname or a.name).partition(".")[0]] = n

    def seg(self, node: ast.AST, indent: int) -> str:
        """``node``'s source with its first line at ``indent`` and the rest shifted with it."""
        lines = ast.get_source_segment(self.text, node).split("\n")
        shift = indent - node.col_offset
        rest = [(" " * shift + ln) if shift >= 0 else ln[min(-shift, len(ln) - len(ln.lstrip())):] for ln in lines[1:]]
        return "\n".join([" " * indent + lines[0], *rest])

    def comments(self, after: int, before: int) -> list[str]:
        """The comment lines strictly between lines ``after`` and ``before`` (1-based)."""
        return [ln.strip() for ln in self.lines[after:before - 1] if ln.strip().startswith("#")]

    def leading_comments(self, node: ast.stmt) -> int:
        """The first line of the comment block right above ``node``."""
        i = node.lineno - 1
        while i > 0 and self.lines[i - 1].strip().startswith("#"):
            i -= 1
        return i + 1


def _names(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


class Fn:
    """A binding function: its imports, its other local statements, and its elements in return order."""

    def __init__(self, src: Source, node: ast.FunctionDef):
        self.src, self.node, self.name = src, node, node.name
        self.doc = ast.get_docstring(node)
        self.doc_node = node.body[0] if self.doc is not None else None
        self.imports: dict[str, tuple[ast.stmt, ast.alias]] = {}
        self.locals: list[tuple[set[str], ast.stmt]] = []
        lists: dict[str, list[ast.expr]] = {}
        self.elements: list[ast.expr] = []
        for st in node.body[1 if self.doc is not None else 0:]:
            if isinstance(st, (ast.Import, ast.ImportFrom)):
                for a in st.names:
                    self.imports[(a.asname or a.name).partition(".")[0]] = (st, a)
            elif isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name) \
                    and isinstance(st.value, (ast.List, ast.ListComp)):
                lists[st.targets[0].id] = self._elements(st.value, lists)
            elif isinstance(st, ast.AugAssign) and isinstance(st.target, ast.Name) and st.target.id in lists:
                lists[st.target.id] += self._elements(st.value, lists)
            elif isinstance(st, ast.Assign):
                self.locals.append(({n.id for t in st.targets for n in ast.walk(t) if isinstance(n, ast.Name)}, st))
            elif isinstance(st, ast.Return) and st.value is not None:
                self.elements = self._elements(st.value, lists)
            else:
                raise ValueError(f"{self.name}: line {st.lineno} is no import, assignment or return; split it by hand")

    def _elements(self, e: ast.expr, lists: dict[str, list[ast.expr]]) -> list[ast.expr]:
        if isinstance(e, ast.List):
            return list(e.elts)
        if isinstance(e, ast.BinOp) and isinstance(e.op, ast.Add):
            return self._elements(e.left, lists) + self._elements(e.right, lists)
        if isinstance(e, ast.Name) and e.id in lists:
            return list(lists[e.id])
        if isinstance(e, ast.ListComp):
            return [e]
        raise ValueError(f"{self.name}: line {e.lineno} is no list of roots; split it by hand")


# --- owners -------------------------------------------------------------------------------------------------------------------

def head(e: ast.expr) -> ast.expr:
    """What an element binds: ``bind``'s Definition, a ``.bind``'s object, a called function, else the expression itself."""
    if isinstance(e, ast.Starred):
        return head(e.value)
    if isinstance(e, ast.Lambda):
        return head(e.body)
    if isinstance(e, ast.Call):
        f = e.func
        if isinstance(f, ast.Name) and f.id == "bind" and e.args:
            return head(e.args[0])
        if isinstance(f, ast.Attribute) and f.attr == "bind":
            return head(f.value)
        return head(f)
    return e


def _module_of_import(st: ast.stmt, a: ast.alias) -> tuple[Any, str]:
    """The object an import binds and the module it was read from."""
    if isinstance(st, ast.Import):
        return importlib.import_module(a.name if a.asname else a.name.partition(".")[0]), a.name
    try:
        m = importlib.import_module(f"{st.module}.{a.name}")
        return m, m.__name__
    except ImportError:
        return getattr(importlib.import_module(st.module), a.name), st.module


def owner(fn: Fn, e: ast.expr) -> str | None:
    """The module whose sidecar holds the binding with head ``e`` (`head`), or None."""
    from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, SpecializedDefinition

    from circuit_check import sidecars as SC

    attrs: list[str] = []
    while isinstance(e, ast.Attribute):
        attrs.append(e.attr)
        e = e.value
    if not isinstance(e, ast.Name) or e.id not in fn.imports:
        return None
    obj, read_from = _module_of_import(*fn.imports[e.id])
    for a in reversed(attrs):
        if inspect.ismodule(obj):
            read_from = obj.__name__
        obj = getattr(obj, a)
    if isinstance(obj, (CompositeDefinition, PrimitiveDefinition, SpecializedDefinition)):
        mod = SC.owner(obj)
    elif inspect.ismodule(obj):
        mod = obj.__name__
    elif inspect.isfunction(obj) or inspect.isclass(obj):
        mod = obj.__module__
    else:
        mod = read_from
    return mod if mod and mod.partition(".")[0] in SC.PACKAGES and SC.path_of(mod, "py") else None


# --- planning -----------------------------------------------------------------------------------------------------------------

class Piece:
    """One element of a binding function going to one module: its source at indent 8, the comments above it, and the names it
    reads."""

    def __init__(self, text: str, names: set[str], comments: list[str] = ()):
        self.text, self.names, self.comments = text, names, list(comments)


def _comp_pieces(fn: Fn, comp: ast.ListComp) -> list[tuple[str | None, Piece]]:
    """A comprehension over a tuple of Definitions, split by each one's module (one piece if its head is no loop variable)."""
    src = fn.src
    gen = comp.generators[0]
    target = gen.target.id if isinstance(gen.target, ast.Name) else None
    text = src.seg(comp, 9)                                  # under the ``*`` it is written with
    if not (target and isinstance(head(comp.elt), ast.Name) and head(comp.elt).id == target
            and isinstance(gen.iter, (ast.Tuple, ast.List))):
        return [(owner(fn, head(comp.elt)), Piece("        *" + text[9:], _names(comp)))]
    groups: dict[str | None, list[ast.expr]] = {}
    for d in gen.iter.elts:
        groups.setdefault(owner(fn, head(d)), []).append(d)
    if len(groups) == 1:
        return [(next(iter(groups)), Piece("        *" + text[9:], _names(comp)))]
    whole = ast.get_source_segment(src.text, gen.iter)
    out = []
    for mod, ds in groups.items():
        sub = "(" + ", ".join(ast.get_source_segment(src.text, d) for d in ds) + ("," if len(ds) == 1 else "") + ")"
        rest = [comp.elt, gen.target, *gen.ifs, *comp.generators[1:]]
        names = {n for x in rest for n in _names(x)} | {n for d in ds for n in _names(d)}
        out.append((mod, Piece("        *" + text[9:].replace(whole, sub, 1), names)))
    return out


def plan_flat(fn: Fn) -> list[tuple[str | None, Piece]]:
    out: list[tuple[str | None, Piece]] = []
    for e in fn.elements:
        if isinstance(e, ast.ListComp):
            out += _comp_pieces(fn, e)
        else:
            out.append((owner(fn, head(e)), Piece(fn.src.seg(e, 8), _names(e))))
    return out


def plan_models(fn: Fn) -> list[tuple[str | None, Piece]]:
    out = []
    prev = fn.node.body[-1].value.lineno
    for e in fn.elements:
        if not (isinstance(e, ast.Tuple) and len(e.elts) == 2 and isinstance(e.elts[1], ast.Lambda)):
            raise ValueError(f"_model_roots: line {e.lineno} is no (label, lambda) entry")
        out.append((owner(fn, head(e.elts[1])), Piece(fn.src.seg(e, 8), _names(e), fn.src.comments(prev, e.lineno))))
        prev = e.end_lineno
    return out


def plan_templates(src: Source, node: ast.Assign) -> list[tuple[str | None, Piece]]:
    from verity_numerical.bench import templates as TM

    out = []
    prev = node.lineno
    for k, v in zip(node.value.keys, node.value.values):
        text = src.seg(k, 4) + ": " + src.seg(v, v.col_offset).lstrip()
        mod = TM.__name__ if k.value in TM.TEMPLATES else None
        out.append((mod, Piece(text, set(), src.comments(prev, k.lineno))))
        prev = v.end_lineno
    return out


# --- writing a binding function -----------------------------------------------------------------------------------------------

def _import_line(st: ast.stmt, aliases: list[ast.alias]) -> str:
    names = ", ".join(a.name if not a.asname else f"{a.name} as {a.asname}" for a in aliases)
    return f"import {names}" if isinstance(st, ast.Import) else f"from {st.module} import {names}"


def render_fn(fn: Fn, pieces: list[Piece], doc: str | None, package: str | None) -> str:
    """``fn`` holding only ``pieces``, with docstring ``doc`` (written at indent 4), in a sidecar in ``package`` (None:
    `targets.py` itself).  Raises LookupError for a name the pieces read that this function cannot import there."""
    src = fn.src
    need = set().union(*(p.names for p in pieces)) if pieces else set()
    local_stmts: list[ast.stmt] = []
    for names, st in reversed(fn.locals):                  # a local the pieces read, and what it reads
        if names & need:
            local_stmts.insert(0, st)
            need |= _names(st.value)
    by_stmt: dict[int, list[ast.alias]] = {}
    from_targets: list[str] = []
    module_imports: list[ast.stmt] = []
    defined = {n for names, _ in fn.locals for n in names}
    for n in sorted(need):
        if n in fn.imports:
            st, a = fn.imports[n]
            by_stmt.setdefault(id(st), []).append(a)
        elif n in defined or hasattr(builtins, n) or n in _bound_inside(fn) or (package is None and (n in src.top or n in src.imports)):
            continue
        elif n in src.imports:
            module_imports.append(src.imports[n])
        elif n in src.top:
            from_targets.append(n)
        else:
            raise LookupError(f"{fn.name}: a binding reads {n}, which nothing defines")
    if from_targets and package in NO_CHECKER:
        raise LookupError(f"{fn.name}: a binding reads circuit_check.targets' {', '.join(from_targets)}")
    lines = [f"def {fn.name}() -> {ast.get_source_segment(src.text, fn.node.returns)}:"]
    if doc:
        lines.append(doc)
    imports = [_import_line(st, [a for a in st.names if a in by_stmt.get(id(st), [])])
               for st in _ordered_imports(fn) if id(st) in by_stmt]
    imports += [ast.get_source_segment(src.text, st) for st in dict.fromkeys(module_imports)]
    if from_targets:
        imports.append(f"from circuit_check.targets import {', '.join(from_targets)}")
    lines += ["    " + ln for ln in _sorted_imports(imports)]
    if imports:
        lines.append("")
    lines += [src.seg(st, 4) for st in local_stmts]
    lines += ["    return ["]
    for p in pieces:
        lines += [" " * 8 + c for c in p.comments] + [p.text + ","]
    lines += ["    ]"]
    return "\n".join(lines) + "\n"


def _bound_inside(fn: Fn) -> set[str]:
    """Names bound inside the elements themselves: comprehension variables and lambda parameters."""
    out: set[str] = set()
    for n in ast.walk(fn.node):
        if isinstance(n, ast.comprehension):
            out |= _names(n.target)
        elif isinstance(n, ast.Lambda):
            out |= {a.arg for a in n.args.args + n.args.kwonlyargs}
    return out


def _ordered_imports(fn: Fn) -> list[ast.stmt]:
    return list(dict.fromkeys(st for st, _ in fn.imports.values()))


def _sorted_imports(lines: list[str]) -> list[str]:
    """isort's order for one block: ``from __future__``, the stdlib, then the rest, ``import x`` before ``from x``, by module."""
    def key(ln: str) -> tuple:
        mod = ln.split()[1]
        std = mod.partition(".")[0] in sys.stdlib_module_names
        return (not std, mod.lower(), ln.startswith("from"))
    std = sorted((ln for ln in lines if ln.split()[1].partition(".")[0] in sys.stdlib_module_names), key=key)
    rest = sorted((ln for ln in lines if ln not in std), key=key)
    return std + ([""] if std and rest else []) + rest


def render_templates(pieces: list[Piece], comment: list[str]) -> str:
    lines = [*comment, "TEMPLATE_PARAMS = {"]
    for p in pieces:
        lines += ["    " + c for c in p.comments] + [p.text + ","]
    return "\n".join(lines + ["}"]) + "\n"


def _docstring(text: str) -> str:
    return textwrap.fill(text, width=125, initial_indent='    """', subsequent_indent="    ", break_long_words=False,
                         break_on_hyphens=False) + '"""'


# --- the move -----------------------------------------------------------------------------------------------------------------

def _package(mod: str) -> str:
    return mod.partition(".")[0]


def plan(strict: bool = True) -> tuple[dict[str, dict[str, str]], str, dict[str, Any]]:
    """({module: {binding name: its source}}, the new `targets.py`, a summary: where each binding goes, and what stays).  Not
    ``strict``, a docstring or `_pouw_roots` the tables were not written against is reported rather than raised."""
    from circuit_check import targets as TG

    TG.load_registries()
    src = Source(TARGETS.read_text())
    moved: dict[str, dict[str, str]] = {}
    keep: dict[str, str | None] = {}                          # targets.py's binding: its new source, None to drop it
    summary: dict[str, Any] = {"stays": []}

    def put(mod: str, name: str, text: str) -> None:
        moved.setdefault(mod, {})[name] = text

    for name in (*flat(), "_model_roots"):
        if name not in src.top:
            continue
        fn = Fn(src, src.top[name])
        pieces = plan_models(fn) if name == "_model_roots" else plan_flat(fn)
        by: dict[str | None, list[Piece]] = {}
        for mod, p in pieces:
            by.setdefault(mod, []).append(p)
        mods = [m for m in by if m is not None]
        summary.setdefault("elements", {})[name] = [f"{mod}: {' '.join(p.text.split())[:90]}" for mod, p in pieces]
        if not mods:
            summary["stays"] += [f"{name}: {' '.join(p.text.split())[:80]}" for p in by.get(None, [])]
            continue
        docs: dict[str, str] = {}
        if name == "_model_roots":
            docs = {m: _docstring("This module's Definitions in tiny whole-model configurations (`circuit_check.targets.TINY`, "
                                  "the vLLM tests'), and the leaf families they do not reach, at small statics.")
                    for m in mods}
        elif len(mods) == 1 and None not in by and fn.doc_node is not None:
            docs = {mods[0]: src.seg(fn.doc_node, 4)}
        elif fn.doc is not None:
            digest, table = DOCS.get(name, ("", {}))
            if digest != _doc_digest(fn.doc):
                if not strict:
                    summary.setdefault("stale", []).append(f"{name} docstring {_doc_digest(fn.doc)}")
                    continue
                raise ValueError(f"{name}'s docstring is not the one `DOCS` splits ({_doc_digest(fn.doc)}, not {digest!r}): "
                                 "split it again")
            docs = {m: _docstring(table[m]) for m in mods if m in table}
            summary.setdefault("undocumented", []).extend(f"{m}#{name}" for m in mods if m not in table)
        for mod in mods:
            try:
                put(mod, name, render_fn(fn, by[mod], docs.get(mod), _package(mod)))
            except LookupError as e:
                summary["stays"].append(str(e))
                by.setdefault(None, []).extend(by[mod])
        stay = by.get(None, [])
        summary["stays"] += [f"{name}: {p.text.strip()[:80]}" for p in stay]
        keep[name] = render_fn(fn, stay, src.seg(fn.doc_node, 4) if fn.doc_node else None, None) if stay else None
    if "_pouw_roots" in src.top:
        digest, table = POUW
        if digest != _digest(src.top["_pouw_roots"]) and not strict:
            summary.setdefault("stale", []).append(f"_pouw_roots {_digest(src.top['_pouw_roots'])}")
        elif digest != _digest(src.top["_pouw_roots"]):
            raise ValueError(f"_pouw_roots is not the function `POUW` splits ({_digest(src.top['_pouw_roots'])}, not "
                             f"{digest!r}): split it again")
        for mod, text in table.items():
            put(mod, "_pouw_roots", text.strip("\n") + "\n")
        keep["_pouw_roots"] = None
    if "TEMPLATE_PARAMS" in src.top:
        node = src.top["TEMPLATE_PARAMS"]
        comment = [ln.strip() for ln in src.lines[src.leading_comments(node) - 1:node.lineno - 1]]
        by = {}
        for mod, p in plan_templates(src, node):
            by.setdefault(mod, []).append(p)
        for mod in [m for m in by if m is not None]:
            put(mod, "TEMPLATE_PARAMS", render_templates(by[mod], comment))
        if set(by) != {None}:
            keep["TEMPLATE_PARAMS"] = render_templates(by[None], comment) if None in by else None
    summary["modules"] = {m: sorted(v) for m, v in sorted(moved.items())}
    return moved, _rewrite_targets(src, keep), summary


def _rewrite_targets(src: Source, keep: dict[str, str | None]) -> str:
    if not keep:
        return src.text
    lines = list(src.lines)
    for name, text in sorted(keep.items(), key=lambda kv: -src.top[kv[0]].lineno):
        node = src.top[name]
        lo = src.leading_comments(node) if name == "TEMPLATE_PARAMS" else node.lineno
        lines[lo - 1:node.end_lineno] = text.rstrip("\n").split("\n") if text is not None else []
    out = re.sub(r"\n{4,}", "\n\n\n", "\n".join(lines))
    if keep.get("TEMPLATE_PARAMS", "") is None and "__all__" in (top := Source(out).top):
        exported = ast.get_source_segment(out, top["__all__"])
        out = out.replace(exported, exported.replace('"TEMPLATE_PARAMS", ', "", 1), 1)
    return out


def flat() -> tuple[str, ...]:
    """The root functions split element by element: `targets.ROOT_FUNCTIONS` but `SHARED` and `BY_HAND`."""
    from circuit_check import targets as TG

    return tuple(n for n in TG.ROOT_FUNCTIONS if n not in SHARED + BY_HAND)


def render_sidecar(mod: str, bindings: dict[str, str], existing: str | None) -> str:
    from circuit_check import targets as TG

    body = [bindings[n] for n in (*TG.ROOT_FUNCTIONS, "_model_roots", "TEMPLATE_PARAMS") if n in bindings]
    if existing is not None:
        have = Source(existing).top
        if clash := sorted(set(bindings) & set(have)):
            raise ValueError(f"{mod}'s sidecar already defines {', '.join(clash)}: merge it by hand")
        return existing.rstrip("\n") + "\n\n\n" + "\n\n".join(body)
    typing = sorted(n for n in ("Any", "Callable") if any(re.search(rf"\b{n}\b", b.split("\n")[0]) for b in body))
    doc = f"circuit-check's bindings of `{mod}`'s Definitions, in `circuit_check.targets`' forms (`circuit_check.sidecars`)."
    head = [textwrap.fill(doc, width=125, initial_indent='"""', break_long_words=False, break_on_hyphens=False) + '"""',
            "from __future__ import annotations", ""]
    if typing:
        head += [f"from typing import {', '.join(typing)}", ""]
    return "\n".join(head) + "\n\n" + "\n\n".join(body)


def move_pins() -> dict[str, int]:
    """Every pin into the pins file of the module that owns it (`sidecars.pin_owner`), else its file in ``pins/``
    (`sidecars.shared_pin`); {file: pins}.  A branch's `pins.json` (`LEGACY_PINS`, the shared pins before ``pins/``) is
    moved too, its pins over those of the same key elsewhere, and removed."""
    from circuit_check import sidecars as SC

    SC.discover.cache_clear()
    pins = SC.load_pins(PINS)
    if LEGACY_PINS.exists():
        for kind, d in json.loads(LEGACY_PINS.read_text()).items():
            pins.setdefault(kind, {}).update(d)
        LEGACY_PINS.unlink()
    by: dict[Path, dict[str, dict]] = {}
    for kind, d in pins.items():
        for key, v in d.items():
            mod = SC.pin_owner(kind, key)
            dest = (SC.path_of(mod, "json") if mod and _package(mod) in SC.PACKAGES else None) or SC.shared_pin(kind, key)
            by.setdefault(dest, {}).setdefault(kind, {})[key] = v
    for p in SC.pin_files():
        if p not in by and p.exists():
            p.unlink()
    for p, kinds in by.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({k: dict(sorted(v.items())) for k, v in sorted(kinds.items())}, indent=1, sort_keys=True) + "\n")
    SC.discover.cache_clear()
    return {SC.rel(p): sum(len(v) for v in kinds.values()) for p, kinds in sorted(by.items())}


def migrate(dry_run: bool = False) -> dict[str, Any]:
    from circuit_check import sidecars as SC

    moved, targets_text, summary = plan(strict=not dry_run)
    if dry_run:
        return summary
    for mod, bindings in moved.items():
        p = SC.path_of(mod, "py")
        p.write_text(render_sidecar(mod, bindings, p.read_text() if p.exists() else None))
    TARGETS.write_text(targets_text)
    summary["pins"] = move_pins()
    return summary


# --- the invariant ------------------------------------------------------------------------------------------------------------

#: the task list as `cli._tasks` builds it, with each target's binding units: run in a fresh process
TASK_LIST = """
import json, sys
from circuit_check import checks as CK, targets as TG
from verity.ml.circuits.evaluation.reference import resolve

TG.load_registries()
out = {"definitions": [], "templates": []}
for fid, fn in TG.suite().items():
    out["definitions"].append({"id": fid, "as_call": TG.family(resolve(fn)) in TG.call_families(),
                               "gates": getattr(fn, "gates", None), "units": sorted(TG.BINDINGS.get(fid, ()))})
for label, sub, sts in TG.templates():
    out["templates"].append({"id": label, "statics": sts, "units": sorted(TG.BINDINGS.get(label, ()))})
out["root_errors"] = sorted(TG.ROOT_ERRORS)
out["pins"] = CK.load_pins()
sys.stdout.write(json.dumps(out, sort_keys=True, default=str))
"""


def task_list() -> dict:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([*_paths(), *filter(None, [os.environ.get("PYTHONPATH")])])}
    r = subprocess.run([sys.executable, "-c", TASK_LIST], capture_output=True, text=True, cwd=REPO, env=env)
    if r.returncode:
        raise RuntimeError(f"building the task list failed ({r.returncode}):\n{r.stderr[-4000:]}")
    return json.loads(r.stdout)


def _unfiled(t: dict) -> dict:
    return {**t, "units": sorted({u.rpartition("#")[2] for u in t["units"]})}


def differences(before: dict, after: dict) -> list[str]:
    """What changed between two task lists, beyond the binding files their units are named by."""
    out = []
    for kind in ("definitions", "templates"):
        a = {t["id"]: _unfiled(t) for t in before[kind]}
        b = {t["id"]: _unfiled(t) for t in after[kind]}
        out += [f"{kind}: {i} only before" for i in sorted(set(a) - set(b))]
        out += [f"{kind}: {i} only after" for i in sorted(set(b) - set(a))]
        out += [f"{kind}: {i}: {a[i]} became {b[i]}" for i in sorted(set(a) & set(b)) if a[i] != b[i]]
    for key in ("root_errors", "pins"):
        if before[key] != after[key]:
            out.append(f"{key} changed")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--verify", action="store_true", help="build the task list before and after, and fail on a difference")
    ap.add_argument("--dry-run", action="store_true", help="print where each binding would go, and change nothing")
    ap.add_argument("--before", type=Path, help="with --verify: the task list before, read from here if it exists, else written")
    a = ap.parse_args(argv)
    before = None
    if a.verify and not a.dry_run:
        if a.before and a.before.exists():
            before = json.loads(a.before.read_text())
        else:
            before = task_list()
            if a.before:
                a.before.write_text(json.dumps(before, sort_keys=True))
    summary = migrate(dry_run=a.dry_run)
    print(json.dumps(summary, indent=1))
    if before is not None:
        diff = differences(before, task_list())
        for d in diff[:50]:
            print(d, file=sys.stderr)
        if diff:
            print(f"the task list changed ({len(diff)} differences)", file=sys.stderr)
            return 1
        print(f"task list unchanged: {len(before['definitions'])} Definitions, {len(before['templates'])} templates, "
              f"{sum(len(v) for v in before['pins'].values())} pins", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
