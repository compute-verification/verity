"""Split the primitives, composites and serving family's registry modules into Definitions, twins and vocabulary.

The family's Definition modules stay in `verity_vllm/program/registry/` and import only `verity` and other Definition modules
(`DEFINITION_MODULES`; `pouw_rows` is the one exception, `BOUNDARY_EXCEPTIONS`). What is not a Definition leaves the registry for
the integration package: four modules whole (`WHOLE`), and named top-level parts of five more into new modules (`PARTS`):

    registry/targets.py                  -> program/frontend/targets.py           vocabulary: which Definition each target binds
    registry/conformance.py              -> program/frontend/conformance.py       vocabulary: the conformance record and its CLI
    registry/serve3_reference.py         -> program/kernels/serve3_reference.py   twin: serve3's numpy reference
    registry/collectives_difftest.py     -> properties/difftests/collectives_difftest.py   twin: a difftest adapter
    registry/prims.py      _fa2, _rms                           -> program/kernels/prims_twins.py         the C++ models' loaders
    registry/ref_prims.py  np_* and their helpers               -> program/kernels/ref_prims_twins.py     numpy twins
    registry/lifted.py     the lag attestation, the use document -> program/lifting/lifted_records.py     record readers
    registry/lifted.py     lift_component, lifted_workload, ... -> program/lifting/lifted_compose.py     the workload compose
    registry/b1.py         SERVE_FAMILIES, serve_program, ...   -> program/frontend/serve_families.py    vocabulary
    registry/b1_tp2.py     collective_kind, cross_rank_families -> program/frontend/collective_kinds.py  vocabulary

`apply` moves the whole modules (`git mv`), cuts the parts out (each moved statement verbatim, with its comments; the new module's
imports derived from the names it reads), and rewrites every reference in the tracked tree: imports (a statement naming a moved
module or name beside others is split; a module alias `L.moved` reads the new module, imported beside the alias), dotted module
names in code and strings, the short forms (`registry.targets`) in comments and docstrings only (a string literal may be data a
digest reads), and the entries of the lint allowlists, the by-name allowlist and the census roots that name what moved. It then
runs the lints' own scans and moves each allowlist entry the move made stale to the violation it moved to; anything it cannot
pair fails. It is idempotent: on a tree where the split is done it changes nothing, and it fails by name, before changing
anything, on a tree it cannot split (a module in both places or neither, a part half moved, a reference form it cannot rewrite).

`verify --before REV` builds circuit-check's task list, the registered ids and the descriptor digest of every target on REV and on
this tree and fails on any difference; `pin` writes the family's slice of a snapshot to the fixture
`tools/verity/circuit_check/tests/fixtures/split/prims_serve.json`, which `tools/verity/circuit_check/tests/test_split_prims_serve.py` holds
the tree to. `inventory` prints each family module's top-level names by kind and every `verity_vllm` import of a Definition module.

    python integrations/vllm/tools/migrate/split_prims_serve.py plan
    python integrations/vllm/tools/migrate/split_prims_serve.py apply
    python integrations/vllm/tools/migrate/split_prims_serve.py verify --before origin/main
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
import symtable
import sys
import tempfile
import tokenize
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
VLLM = "integrations/vllm"
PKG = "verity_vllm"
REG = f"{PKG}.program.registry"
TOOL = f"{VLLM}/tools/migrate/split_prims_serve.py"
FIXTURE = "tools/verity/circuit_check/tests/fixtures/split/prims_serve.json"

#: the family's Definition modules: they stay, and import only `verity` and Definition modules
DEFINITION_MODULES = ("__init__", "catalog", "prims", "ref_prims", "pad_prims", "lifted", "tables", "value_inputs", "operand_widths",
                      "b1", "b1_tp2", "serve3", "spec", "pouw_rows")
#: a Definition module's import of a Definition module outside the registry, with the reason it stays
BOUNDARY_EXCEPTIONS = {("pouw_rows", "verity.core.protocols.pouw.circuit.ncp2"): "NcpDequant is PoUW's own Definition (`verity.core.protocols.pouw`), which this "
                       "row binds; the catalog will import it from PoUW, not from the integration"}


@dataclass(frozen=True)
class Whole:
    """A registry module that leaves whole for ``package``."""
    name: str
    package: str
    kind: str      # twin | vocabulary
    why: str

    @property
    def old(self) -> str:
        return f"{REG}.{self.name}"

    @property
    def new(self) -> str:
        return f"{self.package}.{self.name}"

    @property
    def old_path(self) -> str:
        return f"{VLLM}/{self.old.replace('.', '/')}.py"

    @property
    def new_path(self) -> str:
        return f"{VLLM}/{self.new.replace('.', '/')}.py"


@dataclass(frozen=True)
class Part:
    """Top-level ``names`` of the Definition module ``src`` that leave it for the new module ``dst``."""
    src: str
    dst: str
    kind: str      # twin | vocabulary | integration
    names: tuple[str, ...]
    doc: str
    why: str
    banners: tuple[str, ...] = ()            # a line of each comment banner that heads only the moved names (dropped)
    patches: tuple[tuple[str, str], ...] = ()  # exact-once replacements in the moved text
    imports: tuple[str, ...] = ()            # the import statements the patches need

    @property
    def old(self) -> str:
        return f"{REG}.{self.src}"

    @property
    def src_path(self) -> str:
        return f"{VLLM}/{self.old.replace('.', '/')}.py"

    @property
    def dst_path(self) -> str:
        return f"{VLLM}/{self.dst.replace('.', '/')}.py"

    @property
    def alias(self) -> str:
        return self.dst.rpartition(".")[2]


WHOLE = (
    Whole("targets", f"{PKG}.program.frontend", "vocabulary", "which Definition and statics each declared target binds (GEMM, "
          "attention, norm, MoE, FP8 records)"),
    Whole("conformance", f"{PKG}.program.frontend", "vocabulary", "the conformance record of the registered library and its CLI"),
    Whole("serve3_reference", f"{PKG}.program.kernels", "twin", "serve3's padded program evaluated in numpy, a reference twin"),
    Whole("collectives_difftest", f"{PKG}.properties.difftests", "twin", "a difftest adapter (torch, NCCL) admission imports by name"),
)

PARTS = (
    Part("prims", f"{PKG}.program.kernels.prims_twins", "twin", ("_fa2", "_rms"),
         '"""The C++ models `registry.prims`\' MUFU and norm primitives were transliterated from, with their tables, each loaded '
         'once:\nthe twins (`kernels.twins`) and the replay driver read them.  The primitives are the definition; these are not."""',
         "loads `kernels.fa2_model` / `kernels.rms_triton_model` (ctypes over the C++ models)",
         banners=("# the C++ models the MUFU primitives were transliterated from",)),
    Part("ref_prims", f"{PKG}.program.kernels.ref_prims_twins", "twin",
         ("_np_u32", "_np_f32", "np_widen", "np_narrow", "np_roundtrip", "_np_fma", "_np_scalar_fn", "np_dot_ref", "np_linear_ref",
          "_np_seq_sum", "np_rmsnorm_ref", "np_layernorm_ref", "np_gelu_ref", "np_silu_ref", "np_softmax_ref", "np_rope_ref",
          "np_rope_interleaved_ref", "np_relu_ref", "np_eager_attention_head_ref"),
         '"""numpy twins of the reference vocabulary (`registry.ref_prims`) -- a TESTING AID (fast evaluation over many rows),\n'
         'validated against the gate-by-gate IR evaluation in `tests/program/test_ref_prims.py`; the registered bodies are the\n'
         'definition, these are not."""',
         "numpy re-implementations of the registered reference bodies; `_np_fma` reads `kernels.derived_rows`",
         banners=("# numpy twins — a TESTING AID",)),
    Part("lifted", f"{PKG}.program.lifting.lifted_records", "twin",
         ("LAG_FIELD", "LAG_DECLARED_FIELDS", "LAG_STATIC_FIELD", "_LAG_STATIC_RE", "_lag_rule", "lag_from_header",
          "declared_lag_of_row", "lifted_vu_use_document"),
         '"""What a record says about a lifted Program (`registry.lifted`): the post-EOS lag a run header ATTESTS and the lag a\n'
         'Build DECLARED (read through GP-01\'s one rule, `pipeline.global_program`), and the (query, use) documents of a\n'
         'lifted VU-partition use (`query.query_artifact`)."""',
         "read run headers, `workload_program.json` and query documents through `pipeline.global_program` / `query.query_artifact`"),
    Part("lifted", f"{PKG}.program.lifting.lifted_compose", "integration",
         ("lift_component", "lifted_workload", "workload_program"),
         '"""The lifted fixed WORKLOAD Program (MIXED BATCH): GP-01\'s compose (`program.workload`), unchanged, over the lifted\n'
         'components of the declared requests (`registry.lifted.lift_request_component`)."""',
         "compose a Workload Program through `program.workload` (GP-01's compose, outside the registry)"),
    Part("b1", f"{PKG}.program.frontend.serve_families", "vocabulary",
         ("FA2_KERNEL_SOURCE", "fa2_kernel_geometry", "profile_attention_consts", "SERVE_FAMILIES", "DEFAULT_SERVE_FAMILY",
          "FA2_SPLITKV_KERNEL_NAMES", "LEGACY_ATTENTION_GEOMETRY", "serve_family", "attention_geometry_static", "serve_program",
          "serve_program_of_identity"),
         '"""Which serving program an instance pins (`registry.b1`): the `Serve` families by `program.identity.family`, the\n'
         'attention geometry a semantic profile declares and the `BN` static it binds, and the bound program of an identity."""',
         "read semantic-profile and `protocol_instance.json` documents to choose a `Serve` Definition and its statics",
         patches=(('cfg = globals().get(f"{case}_CONFIG")', 'cfg = getattr(b1, f"{case}_CONFIG", None)'),),
         imports=(f"from {REG} import b1",)),
    Part("b1_tp2", f"{PKG}.program.frontend.collective_kinds", "vocabulary",
         ("collective_kind", "_COLLECTIVE_KIND", "cross_rank_families"),
         '"""The collective kind of a Definition family or id (`registry.b1_tp2`), and the cross-rank families of a world: what the\n'
         'Commit\'s cross-rank pass and the query compose read instead of spelling a family prefix."""',
         "which families the Commit's cross-rank pass and the query compose treat as collectives"),
)

#: packages this split creates, with their docstrings (the dense family's split creates the same one, with the same text)
NEW_PACKAGES = {f"{PKG}.properties.difftests": '"""Difftest adapters (`verity_vllm.properties.admission`): each runs a registered '
                                               'Definition and the production vLLM kernel it models on the same edge-kind inputs."""\n'}

#: never rewritten: this tool, its tests and fixture name the old modules on purpose, and history stays as it was written
EXCLUDE_PREFIX = (f"{VLLM}/tools/migrate/", FIXTURE)
EXCLUDE_FILES = ("tools/verity/circuit_check/tests/test_split_prims_serve.py", f"{VLLM}/tests/program/test_split_prims_serve.py")
TEXT_SUFFIXES = (".md", ".txt", ".toml", ".sh", ".cfg", ".ini", ".yaml", ".yml", ".rst")
LINT_ALLOWLISTS = f"{VLLM}/tests/lint/allowlists"
ROOTS = f"{VLLM}/tests/census_roots.txt"
KEEP = f"{VLLM}/tests/dead_code_keep.json"
#: census roots for what only the registry walk (`registry.catalog.load()`) reached; the dense family's split declares the same one
ROOT_DIRS = {f"{PKG}.properties.difftests": "the difftest adapters, imported by name (properties-admission --adapter)"}
#: the new modules only tests read: live before only because `catalog.load()` imports every registry module
KEEP_REASONS = {
    f"{PKG}.program.kernels.ref_prims_twins": "numpy twins of registry.ref_prims, checked against the registered bodies by "
                                              "tests/program/test_ref_prims.py (LIVE registry.ref_prims)",
    f"{PKG}.program.lifting.lifted_compose": "the lifted fixed workload Program (GP-01's compose over lifted components) of "
                                             "tests/program/test_lifted_mixed.py / test_lifted_r17.py (LIVE registry.lifted)",
    f"{PKG}.program.frontend.serve_families": "which Serve family and attention geometry an instance pins, read by "
                                              "tests/program/test_composition.py ahead of a production caller (LIVE registry.b1)",
}
BY_NAME = f"{VLLM}/tests/by_name_allowlist.json"
LINE_MAX = 150


class SplitError(RuntimeError):
    """The tree is in a state this split cannot handle.  Raised by `plan` before anything is written; raised by `apply` while
    relocating the allowlists, after the moves are written (`git reset --hard && git clean -fd` restores the tree)."""


# --- text helpers ---------------------------------------------------------------------------------------------------------------

def _git(*args: str, root: Path = REPO) -> str:
    return subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True).stdout


def _tracked(root: Path) -> list[str]:
    return [p for p in _git("ls-files", "-z", root=root).split("\0") if p and (root / p).is_file()]


def _read(path: Path) -> str | None:
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if b"\0" in raw[:8192]:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


class _Offsets:
    """(line, utf-8 column) -> character offset of one text."""

    def __init__(self, text: str):
        self.text = text
        self.lines = text.splitlines(keepends=True)
        self.starts = [0]
        for ln in self.lines:
            self.starts.append(self.starts[-1] + len(ln))

    def at(self, line: int, col: int) -> int:
        ln = self.lines[line - 1] if line - 1 < len(self.lines) else ""
        return self.starts[line - 1] + len(ln.encode("utf-8")[:col].decode("utf-8", "replace"))

    def span(self, node: ast.AST) -> tuple[int, int]:
        return self.at(node.lineno, node.col_offset), self.at(node.end_lineno, node.end_col_offset)


def _apply_edits(text: str, edits: list[tuple[int, int, str]], where: str) -> str:
    edits = sorted(set(edits))
    for (a0, a1, _), (b0, _b1, _) in zip(edits, edits[1:]):
        if b0 < a1:
            raise SplitError(f"{where}: overlapping rewrites at offsets {a0}..{a1} and {b0}")
    for s, e, new in reversed(edits):
        text = text[:s] + new + text[e:]
    return text


def _render_import(indent: str, module: str, names: list[str], comment: str = "") -> str:
    """`from module import a, b as c` (wrapped in parentheses past LINE_MAX), with ``comment`` after it."""
    head = f"{indent}from {module} import "
    one = head + ", ".join(names)
    tail = f"  {comment}" if comment else ""
    if len(one) + len(tail) <= LINE_MAX:
        return one + tail
    lines, cur = [], head + "("
    pad = " " * len(cur)
    for i, n in enumerate(names):
        piece = n + (", " if i < len(names) - 1 else ")")
        if len(cur) + len(piece.rstrip()) > LINE_MAX and cur.strip() not in ("", head.strip() + " ("):
            lines.append(cur.rstrip())
            cur = pad
        cur += piece
    lines.append(cur + tail)
    return "\n".join(lines)


def _alias_text(a: ast.alias) -> str:
    return f"{a.name} as {a.asname}" if a.asname else a.name


def _stmt_comments(text: str, off: _Offsets, node: ast.stmt) -> str:
    """The comments inside a statement's lines (an import's `# noqa`), joined."""
    start = off.starts[node.lineno - 1]
    end = off.starts[node.end_lineno] if node.end_lineno < len(off.starts) else len(text)
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text[start:end]).readline):
            if tok.type == tokenize.COMMENT:
                out.append(tok.string)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    return "  ".join(out)


def _whole_lines(text: str, off: _Offsets, node: ast.stmt, where: str) -> tuple[int, int, str]:
    """The character span of a statement's full lines (its indent through its line end) and its indent; a statement that
    shares a line with another is refused."""
    first = off.lines[node.lineno - 1]
    indent = first[: len(first) - len(first.lstrip())]
    if off.lines[node.lineno - 1][len(indent):].encode()[: max(0, node.col_offset - len(indent.encode()))].strip():
        raise SplitError(f"{where}:{node.lineno}: an import that shares its line with another statement; split it by hand first")
    last = off.lines[node.end_lineno - 1]
    rest = last.encode("utf-8")[node.end_col_offset:].decode("utf-8").strip()
    if rest and not rest.startswith("#"):
        raise SplitError(f"{where}:{node.end_lineno}: an import followed by another statement on its line; split it by hand first")
    s = off.starts[node.lineno - 1]
    e = off.starts[node.end_lineno - 1] + len(last.rstrip("\r\n"))
    return s, e, indent


# --- the parts ------------------------------------------------------------------------------------------------------------------

def _defined(stmt: ast.stmt) -> set[str]:
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {stmt.name}
    if isinstance(stmt, ast.Assign):
        return {n.id for t in stmt.targets for n in ast.walk(t) if isinstance(n, ast.Name)}
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        return {stmt.target.id}
    return set()


def _top_defs(tree: ast.Module) -> dict[str, list[ast.stmt]]:
    out: dict[str, list[ast.stmt]] = {}
    for s in tree.body:
        for n in _defined(s):
            out.setdefault(n, []).append(s)
    return out


def _import_bindings(tree: ast.Module, package: str) -> dict[str, tuple[ast.stmt, ast.alias, str]]:
    """Top-level name -> (statement, alias, absolute module) of the module's top-level imports."""
    out = {}
    for s in tree.body:
        if isinstance(s, ast.Import):
            for a in s.names:
                out[a.asname or a.name.split(".")[0]] = (s, a, a.name)
        elif isinstance(s, ast.ImportFrom):
            mod = _absolute(s, package)
            for a in s.names:
                if a.name != "*":
                    out[a.asname or a.name] = (s, a, mod)
    return out


def _absolute(s: ast.ImportFrom, package: str) -> str:
    if not s.level:
        return s.module or ""
    base = package.split(".")[: len(package.split(".")) - s.level + 1]
    return ".".join([*base, *([s.module] if s.module else [])])


def _globals_read(text: str) -> set[str]:
    """The global names a module's code reads (symtable), and every name in its annotations."""
    out = set()

    def walk(t: symtable.SymbolTable) -> None:
        for s in t.get_symbols():
            if t.get_type() == "module":
                if s.is_referenced() and not (s.is_assigned() or s.is_imported()):
                    out.add(s.get_name())
            elif s.is_referenced() and s.is_global():
                out.add(s.get_name())
        for c in t.get_children():
            walk(c)

    walk(symtable.symtable(text, "<part>", "exec"))
    for node in ast.walk(ast.parse(text)):
        ann = [getattr(node, "returns", None), getattr(node, "annotation", None)]
        for a in ann:
            if a is not None:
                out |= {n.id for n in ast.walk(a) if isinstance(n, ast.Name)}
    return out


def _is_banner(block: list[str]) -> bool:
    return any(re.match(r"^#\s*-{20,}\s*$", ln.strip()) for ln in block)


@dataclass
class _Cut:
    src_text: str
    body: str


def _cut(text: str, part: Part, where: str) -> _Cut:
    """Cut ``part``'s statements out of the module ``text``: the module without them, and their text (in source order, with
    their decorators and the comments attached to them)."""
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    defs = _top_defs(tree)
    moved: set[int] = set()     # 0-based line indexes
    stmts = []
    for name in part.names:
        found = defs.get(name, [])
        if len(found) != 1:
            raise SplitError(f"{where}: {name} is defined {len(found)} times at the top level (expected once)")
        s = found[0]
        if not _defined(s) <= set(part.names):
            raise SplitError(f"{where}:{s.lineno}: the statement defining {name} also defines {sorted(_defined(s) - set(part.names))}")
        stmts.append(s)
    for s in {id(s): s for s in stmts}.values():
        first = (s.decorator_list[0].lineno if getattr(s, "decorator_list", None) else s.lineno) - 1
        last = s.end_lineno - 1
        i = first - 1
        block = []
        while i >= 0 and lines[i].lstrip().startswith("#") and not lines[i][:1].isspace():
            block.append(i)
            i -= 1
        if block and not _is_banner([lines[j] for j in block]):
            first = block[-1]
        j = last + 1
        trail = []
        while j < len(lines) and lines[j].startswith("#"):
            trail.append(j)
            j += 1
        if trail and (j >= len(lines) or not lines[j].strip()) and not _is_banner([lines[k] for k in trail]):
            last = trail[-1]
        moved.update(range(first, last + 1))
    dropped: set[int] = set()
    for banner in part.banners:
        hit = [i for i, ln in enumerate(lines) if ln.startswith(banner)]
        if len(hit) != 1:
            raise SplitError(f"{where}: the banner {banner!r} is on {len(hit)} lines (expected one)")
        lo = hi = hit[0]
        while lo > 0 and lines[lo - 1].startswith("#"):
            lo -= 1
        while hi + 1 < len(lines) and lines[hi + 1].startswith("#"):
            hi += 1
        dropped.update(range(lo, hi + 1))
    gone = moved | dropped

    def blank(i: int) -> bool:
        return not lines[i].strip()

    for i in range(len(lines)):
        if blank(i) and i not in gone:
            up = next((k for k in range(i - 1, -1, -1) if not blank(k)), None)
            down = next((k for k in range(i + 1, len(lines)) if not blank(k)), None)
            if up is not None and up in gone and (down is None or down in gone):
                gone.add(i)
    kept: list[str | None] = []                  # None marks a seam: where a cut was
    for i, ln in enumerate(lines):
        if i not in gone:
            kept.append(ln)
        elif not kept or kept[-1] is not None:
            kept.append(None)
    out: list[str] = []
    i = 0
    while i < len(kept):
        if kept[i] is not None:
            out.append(kept[i])     # type: ignore[arg-type]
            i += 1
            continue
        j = i + 1                   # a seam (or several, with only blank lines between): at most two blank lines stay
        while j < len(kept) and (kept[j] is None or not kept[j].strip()):  # type: ignore[union-attr]
            j += 1
        after = sum(1 for x in kept[i + 1: j] if x is not None)
        before = 0
        while before < len(out) and not out[-1 - before].strip():
            before += 1
        del out[len(out) - before:]
        if j < len(kept) and out:
            out.extend(["\n"] * min(2, before + after))
        i = j
    src_text = "".join(out).rstrip("\n") + "\n"
    runs: list[list[int]] = []
    for i in sorted(moved):
        if runs and all(k in gone for k in range(runs[-1][-1] + 1, i)):
            runs[-1].extend(k for k in range(runs[-1][-1] + 1, i) if blank(k))
            runs[-1].append(i)
        else:
            runs.append([i])
    body = "\n\n".join("".join(lines[k] for k in r).strip("\n") + "\n" for r in runs)
    body = re.sub(r"\n{4,}", "\n\n\n", body)
    for old, new in part.patches:
        if body.count(old) != 1:
            raise SplitError(f"{where}: the patch {old!r} matches {body.count(old)} times in the moved text (expected once)")
        body = body.replace(old, new)
    return _Cut(src_text, body)


def _drop_from_all(text: str, names: set[str], where: str) -> str:
    tree = ast.parse(text)
    for s in tree.body:
        if isinstance(s, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in s.targets):
            before = [e.value for e in s.value.elts]  # type: ignore[attr-defined]
            if not names & set(before):
                return text
            lines = text.splitlines(keepends=True)
            seg = lines[s.lineno - 1: s.end_lineno]
            out = []
            for ln in seg:
                for n in names:
                    ln = re.sub(rf'"{re.escape(n)}",[ \t]*', "", ln)
                    ln = re.sub(rf',[ \t]*"{re.escape(n)}"', "", ln)
                    ln = re.sub(rf'"{re.escape(n)}"\]', "]", ln)
                if not ln.strip():
                    continue
                if ln.strip() == "]" and out:
                    out[-1] = out[-1].rstrip().rstrip(",") + "]\n"
                    continue
                out.append(ln.rstrip() + "\n" if ln.endswith("\n") else ln)
            new = "".join(lines[: s.lineno - 1] + out + lines[s.end_lineno:])
            after = [e.value for s2 in ast.parse(new).body if isinstance(s2, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == "__all__" for t in s2.targets) for e in s2.value.elts]
            if after != [n for n in before if n not in names]:
                raise SplitError(f"{where}: could not drop {sorted(names & set(before))} from __all__ cleanly")
            return new
    return text


def _prune_imports(text: str, candidates: set[str], package: str, where: str) -> str:
    """Drop the top-level imports of ``candidates`` the module no longer reads."""
    tree = ast.parse(text)
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        e.value for s in tree.body if isinstance(s, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in s.targets)
        for e in getattr(s.value, "elts", []) if isinstance(e, ast.Constant)}
    off = _Offsets(text)
    edits = []
    for s in tree.body:
        if not isinstance(s, (ast.Import, ast.ImportFrom)):
            continue
        keep = [a for a in s.names if (a.asname or a.name.split(".")[0]) in used or (a.asname or a.name.split(".")[0]) not in candidates]
        if len(keep) == len(s.names):
            continue
        st, en, indent = _whole_lines(text, off, s, where)
        if not keep:
            nxt = en + 1 if text[en:en + 1] == "\n" else en
            edits.append((st, nxt, ""))
        elif isinstance(s, ast.ImportFrom):
            edits.append((st, en, _render_import(indent, ("." * s.level) + (s.module or ""), [_alias_text(a) for a in keep],
                                                 _stmt_comments(text, off, s))))
        else:
            edits.append((st, en, indent + "import " + ", ".join(_alias_text(a) for a in keep)))
    return _apply_edits(text, edits, where)


_STDLIB = set(sys.stdlib_module_names) | {"__future__"}


def _import_block(lines: list[str]) -> str:
    """Import statements grouped (stdlib, third party, verity) and sorted, a blank line between groups."""
    groups: dict[int, list[str]] = {0: [], 1: [], 2: [], 3: []}
    for ln in dict.fromkeys(lines):
        mod = ln.split()[1]
        top = mod.split(".")[0]
        g = 0 if top == "__future__" else 1 if top in _STDLIB else 3 if top.startswith("verity") else 2
        groups[g].append(ln)
    out = []
    for g in (0, 1, 2, 3):
        if groups[g]:
            out.append("\n".join(sorted(groups[g], key=lambda s: (s.startswith("from "), s.split()[1], s))))
    return "\n\n".join(out)


def _dst_text(cut: _Cut, part: Part, src_tree: ast.Module, moved_to: dict[str, Part], where: str) -> str:
    """The new module: its docstring, the imports its moved code reads, the moved code."""
    package = part.old.rpartition(".")[0]
    bindings = _import_bindings(src_tree, package)
    defs = _top_defs(src_tree)
    own = set(part.names)
    imports = ["from __future__ import annotations"] if any(
        isinstance(s, ast.ImportFrom) and s.module == "__future__" for s in src_tree.body) else []
    imports += list(part.imports)
    patched = {ln.split(" import ")[-1].split(" as ")[-1] for ln in part.imports}
    need = sorted(_globals_read(cut.body) - own - set(dir(builtins)) - patched - {"__file__", "__name__", "__doc__"})
    for n in need:
        if n in bindings:
            s, a, mod = bindings[n]
            if isinstance(s, ast.Import):
                imports.append(f"import {a.name} as {a.asname}" if a.asname else f"import {a.name}")
            else:
                imports.append(f"from {mod} import {_alias_text(a)}")
        elif n in moved_to and moved_to[n] is not part:
            imports.append(f"from {moved_to[n].dst} import {n}")
        elif n in defs:
            imports.append(f"from {part.old} import {n}")
        else:
            raise SplitError(f"{where}: the moved code reads {n!r}, which {part.src} neither defines nor imports")
    merged: dict[str, list[str]] = {}
    plain = []
    for ln in imports:
        if ln.startswith("from "):
            mod, _, names = ln[5:].partition(" import ")
            merged.setdefault(mod, []).extend(x.strip() for x in names.split(","))
        else:
            plain.append(ln)
    rendered = plain + [_render_import("", m, list(dict.fromkeys(ns))) for m, ns in merged.items()]
    return f"{part.doc}\n\n{_import_block(rendered)}\n\n\n{cut.body.rstrip()}\n"


# --- the reference rewrite ------------------------------------------------------------------------------------------------------

WHOLE_BY = {w.name: w for w in WHOLE}
PART_OF: dict[tuple[str, str], Part] = {(p.src, n): p for p in PARTS for n in p.names}
SRCS = {p.src for p in PARTS}


def _module_of_file(rel: str) -> tuple[str, str] | None:
    """(module, package) of a file under integrations/vllm, for its relative imports."""
    if not rel.startswith(f"{VLLM}/{PKG}/") or not rel.endswith(".py"):
        return None
    parts = rel[len(VLLM) + 1: -3].split("/")
    if parts[-1] == "__init__":
        parts.pop()
        return ".".join(parts), ".".join(parts)
    return ".".join(parts), ".".join(parts[:-1])


def _scopes(tree: ast.Module) -> dict[int, ast.AST]:
    """id(node) -> the innermost function (or the module) it is in."""
    out: dict[int, ast.AST] = {}

    def visit(n: ast.AST, scope: ast.AST) -> None:
        for c in ast.iter_child_nodes(n):
            out[id(c)] = scope
            visit(c, c if isinstance(c, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) else scope)

    visit(tree, tree)
    return out


def _bound_names(tree: ast.Module) -> dict[str, set[str]]:
    """name -> what binds it anywhere in the file (an import's module, or `def`/`=`/argument)."""
    out: dict[str, set[str]] = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                out.setdefault(a.asname or a.name.split(".")[0], set()).add(a.name if a.asname else a.name.split(".")[0])
        elif isinstance(n, ast.ImportFrom):
            for a in n.names:
                out.setdefault(a.asname or a.name, set()).add(f"{n.module}.{a.name}" if n.module else a.name)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.setdefault(n.name, set()).add("<def>")
        elif isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
            out.setdefault(n.id, set()).add("<assign>")
        elif isinstance(n, ast.arg):
            out.setdefault(n.arg, set()).add("<arg>")
    return out


def rewrite_python(text: str, rel: str = "<text>") -> str:
    """Every reference of ``text`` (a Python file at ``rel``) to a moved module or name, rewritten to its new home."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return rewrite_text(text, prose=False)
    off = _Offsets(text)
    mf = _module_of_file(rel)
    package = mf[1] if mf else ""
    scopes = _scopes(tree)
    edits: list[tuple[int, int, str]] = []
    alias_src: dict[str, str] = {}                       # alias -> src module (scope-insensitive, as the lints read it)
    binders: dict[str, list[ast.stmt]] = {}              # alias -> the statements binding it to a src module
    split: set[int] = set()                              # ids of the import statements rewritten whole
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            mod = _absolute(n, package) if n.level else (n.module or "")
            if mod == REG:
                for a in n.names:
                    if a.name in SRCS:
                        alias_src[a.asname or a.name] = a.name
                        binders.setdefault(a.asname or a.name, []).append(n)
                if any(a.name in WHOLE_BY for a in n.names):
                    if n.level:
                        raise SplitError(f"{rel}:{n.lineno}: a relative import of a moved module; make it absolute first")
                    _split_import(text, off, n, rel, edits, lambda a: WHOLE_BY[a.name].package if a.name in WHOLE_BY else REG)
                    split.add(id(n))
            src = mod[len(REG) + 1:] if mod.startswith(REG + ".") else None
            if src in SRCS and any((src, a.name) in PART_OF for a in n.names):
                if n.level:
                    raise SplitError(f"{rel}:{n.lineno}: a relative import of a moved name; make it absolute first")
                _split_import(text, off, n, rel, edits, lambda a, src=src: PART_OF[(src, a.name)].dst if (src, a.name) in PART_OF else mod)
                split.add(id(n))
            if src in SRCS and any(a.name == "*" for a in n.names):
                raise SplitError(f"{rel}:{n.lineno}: `from {mod} import *`; name what it imports first")
            if n.level and (src in WHOLE_BY or (n.module or "").split(".")[0] in WHOLE_BY and mod.startswith(REG)):
                raise SplitError(f"{rel}:{n.lineno}: a relative import of a moved module; make it absolute first")
        elif isinstance(n, ast.Import):
            for a in n.names:
                src = a.name[len(REG) + 1:] if a.name.startswith(REG + ".") else None
                if src in SRCS and a.asname:
                    alias_src[a.asname] = src
                    binders.setdefault(a.asname, []).append(n)
    if alias_src:
        bound = _bound_names(tree)
        needs: dict[tuple[int, str], tuple[ast.stmt, Part]] = {}
        renamed: set[int] = set()
        for n in ast.walk(tree):
            target = None
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in alias_src:
                target = (n.value, n.attr)
            elif isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in ("getattr", "hasattr", "setattr", "delattr") \
                    and len(n.args) >= 2 and isinstance(n.args[0], ast.Name) and n.args[0].id in alias_src \
                    and isinstance(n.args[1], ast.Constant) and isinstance(n.args[1].value, str):
                target = (n.args[0], n.args[1].value)
            elif isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ("setattr", "delattr") \
                    and len(n.args) >= 2 and isinstance(n.args[0], ast.Name) and n.args[0].id in alias_src \
                    and isinstance(n.args[1], ast.Constant) and isinstance(n.args[1].value, str):
                target = (n.args[0], n.args[1].value)
            if target is None:
                continue
            name_node, attr = target
            part = PART_OF.get((alias_src[name_node.id], attr))
            if part is None:
                continue
            if bound.get(part.alias, set()) - {f"{part.dst.rpartition('.')[0]}.{part.alias}"}:
                raise SplitError(f"{rel}: `{part.alias}` is already bound ({sorted(bound[part.alias])}); "
                                 f"cannot import {part.dst} under that name")
            s, e = off.span(name_node)
            edits.append((s, e, part.alias))
            renamed.add(id(name_node))
            scope = scopes.get(id(n), tree)
            chain = []
            while scope is not None:
                chain.append(scope)
                scope = scopes.get(id(scope))
            for b in binders[name_node.id]:
                if scopes.get(id(b), tree) in chain:
                    needs.setdefault((id(b), part.dst), (b, part))
        have = {(id(scopes.get(id(n), tree)), f"{n.module}.{a.name}") for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
                for a in n.names if a.asname is None}
        still_read = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and id(n) not in renamed}
        added: dict[int, tuple[ast.stmt, list[str]]] = {}
        for (_bid, dst), (b, part) in needs.items():
            if (id(scopes.get(id(b), tree)), dst) in have:
                continue
            pkg, _, mod = dst.rpartition(".")
            added.setdefault(id(b), (b, []))[1].append(f"from {pkg} import {mod}")
        for b, lines in added.values():
            st, en, indent = _whole_lines(text, off, b, rel)
            alias = b.names[0].asname or b.names[0].name
            if len(b.names) == 1 and id(b) not in split and alias not in still_read:
                edits.append((st, en, "\n".join(indent + ln for ln in lines)))      # the old binder is dead now
            elif (pair := _mergeable(text, off, tree, b, split)) is not None:
                # merge the kept binder with its neighbour from the same module, so the file does not grow (p10's cap)
                first, second = pair
                st, _e, _i = _whole_lines(text, off, first, rel)
                _s, en, _i = _whole_lines(text, off, second, rel)
                merged = _render_import("", first.module or "", [_alias_text(a) for a in first.names + second.names])
                edits.append((st, en, "\n".join([merged] + lines)))
            else:
                edits.append((en, en, "".join(f"\n{indent}{ln}" for ln in lines)))
    text = _apply_edits(text, edits, rel)
    return _rewrite_embedded(rewrite_text(text, prose=False, python=True), rel)


def _embedded(text: str):
    """(token, its value) of each string literal of ``text`` that is itself a Python program naming the registry (a
    subprocess's `-c` code)."""
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return
    for tok in toks:
        if tok.type != tokenize.STRING or REG not in tok.string:
            continue
        try:
            value = ast.literal_eval(tok.string)
            ast.parse(value)
        except (ValueError, SyntaxError, TypeError):
            continue
        if isinstance(value, str) and "import" in value:
            yield tok, value


def _rewrite_embedded(text: str, rel: str) -> str:
    off = _Offsets(text)
    edits = []
    for tok, value in _embedded(text):
        new = rewrite_python(value, f"{rel}<string>")
        if new == value:
            continue
        lit = tok.string
        prefix = len(lit) - len(lit.lstrip("rRbBuU"))
        q = 3 if lit[prefix:prefix + 3] in ('"""', "'''") else 1
        if lit[prefix + q: len(lit) - q] != value:
            raise SplitError(f"{rel}:{tok.start[0]}: a Python string with escapes names a moved module; rewrite it by hand first")
        s = off.at(tok.start[0], len(off.lines[tok.start[0] - 1][: tok.start[1]].encode()))
        e = off.at(tok.end[0], len(off.lines[tok.end[0] - 1][: tok.end[1]].encode()))
        edits.append((s, e, lit[: prefix + q] + new + lit[len(lit) - q:]))
    return _apply_edits(text, edits, rel)


def _mergeable(text: str, off: _Offsets, tree: ast.Module, b: ast.stmt, split: set[int]) -> tuple[ast.ImportFrom, ast.ImportFrom] | None:
    """``b`` and the module-level statement next to it, when both are one-line, comment-free absolute imports from one
    module that no other edit touches and that fit on one line together; else None."""
    if not isinstance(b, ast.ImportFrom) or b.level or b not in tree.body:
        return None
    i = tree.body.index(b)
    for j in (i - 1, i + 1):
        if not 0 <= j < len(tree.body):
            continue
        o = tree.body[j]
        if not (isinstance(o, ast.ImportFrom) and not o.level and o.module == b.module) or {id(o), id(b)} & split:
            continue
        first, second = (o, b) if j < i else (b, o)
        if any(s.lineno != s.end_lineno or _stmt_comments(text, off, s) for s in (first, second)):
            continue
        if second.lineno != first.lineno + 1:
            continue
        merged = _render_import("", first.module or "", [_alias_text(a) for a in first.names + second.names])
        if "\n" not in merged:
            return first, second
    return None


def _split_import(text: str, off: _Offsets, n: ast.ImportFrom, rel: str, edits: list, home) -> None:
    """``n`` rewritten as one statement per module its names now live in (``home(alias)``), in first-appearance order."""
    groups: dict[str, list[str]] = {}
    for a in n.names:
        groups.setdefault(home(a), []).append(_alias_text(a))
    st, en, indent = _whole_lines(text, off, n, rel)
    comment = _stmt_comments(text, off, n)
    out = [_render_import(indent, m, names) for m, names in groups.items()]
    if comment:
        out[-1] += f"  {comment}"
    edits.append((st, en, "\n".join(out)))


def _dotted_rules() -> list[tuple[re.Pattern, str]]:
    rules = []
    for p in PARTS:
        for n in p.names:
            rules.append((re.compile(rf"(?<![\w.]){re.escape(p.old)}([.:]){re.escape(n)}\b"), rf"{p.dst}\g<1>{n}"))
    for w in WHOLE:
        rules.append((re.compile(rf"(?<![\w./])((?:{re.escape(VLLM)}/)?){re.escape(w.old_path[len(VLLM) + 1:])}"),
                      rf"\g<1>{w.new_path[len(VLLM) + 1:]}"))
        rules.append((re.compile(rf"(?<![\w.]){re.escape(w.old)}\b"), w.new))
    return rules


def _prose_rules() -> list[tuple[re.Pattern, str]]:
    rules = []
    for p in PARTS:
        tail = p.dst[len(PKG) + 1:].split(".", 1)[1]                  # lifting.lifted_records
        for n in p.names:
            rules.append((re.compile(rf"(?<![\w./])(?:program\.)?registry\.{re.escape(p.src)}\.{re.escape(n)}\b"), f"{tail}.{n}"))
    for w in WHOLE:
        pkg_tail = w.package[len(PKG) + 1:]                           # program.frontend
        last = w.package.rpartition(".")[2]                           # frontend
        rules.append((re.compile(rf"(?<![\w./])program/registry/{re.escape(w.name)}\.py\b"), f"{pkg_tail.replace('.', '/')}/{w.name}.py"))
        rules.append((re.compile(rf"(?<![\w./])registry/{re.escape(w.name)}\.py\b"), f"{last}/{w.name}.py"))
        rules.append((re.compile(rf"(?<![\w./])program\.registry\.{re.escape(w.name)}\b"), f"{pkg_tail}.{w.name}"))
        rules.append((re.compile(rf"(?<![\w./])registry\.{re.escape(w.name)}\b"), f"{last}.{w.name}"))
    return rules


DOTTED, PROSE = _dotted_rules(), _prose_rules()


def _sub_all(rules, s: str) -> str:
    for rx, rep in rules:
        s = rx.sub(rep, s)
    return s


def rewrite_text(text: str, *, prose: bool, python: bool = False) -> str:
    """Dotted module names and file paths everywhere; with ``prose`` the short forms too (a document); in Python the short
    forms are rewritten only in comments and docstrings."""
    text = _sub_all(DOTTED, text)
    if prose:
        return _sub_all(PROSE, text)
    if not python:
        return text
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text
    docs = {(n.value.lineno, n.value.col_offset) for n in ast.walk(tree)
            if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)}
    off = _Offsets(text)
    edits = []
    for tok in tokenize.generate_tokens(io.StringIO(text).readline):
        if tok.type == tokenize.COMMENT or (tok.type == tokenize.STRING and tok.start in docs):
            new = _sub_all(PROSE, tok.string)
            if new != tok.string:
                s = off.at(tok.start[0], len(off.lines[tok.start[0] - 1][: tok.start[1]].encode()))
                e = off.at(tok.end[0], len(off.lines[tok.end[0] - 1][: tok.end[1]].encode()))
                edits.append((s, e, new))
    return _apply_edits(text, edits, "<prose>")


def remaining_references(text: str, rel: str) -> list[str]:
    """References a rewritten Python file still makes to a moved module or name (none, after `rewrite_python`)."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    out = []
    alias_src = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and not n.level:
            if n.module == REG:
                for a in n.names:
                    if a.name in WHOLE_BY:
                        out.append(f"{rel}:{n.lineno}: imports {REG}.{a.name}")
                    if a.name in SRCS:
                        alias_src[a.asname or a.name] = a.name
            src = (n.module or "")[len(REG) + 1:] if (n.module or "").startswith(REG + ".") else None
            if src in WHOLE_BY:
                out.append(f"{rel}:{n.lineno}: imports from {n.module}")
            for a in n.names:
                if (src, a.name) in PART_OF:
                    out.append(f"{rel}:{n.lineno}: imports {a.name} from {n.module}")
        elif isinstance(n, ast.Import):
            for a in n.names:
                if a.name.startswith(REG + ".") and a.name[len(REG) + 1:] in WHOLE_BY:
                    out.append(f"{rel}:{n.lineno}: imports {a.name}")
                if a.name.startswith(REG + ".") and a.name[len(REG) + 1:] in SRCS and a.asname:
                    alias_src[a.asname] = a.name[len(REG) + 1:]
    for n in ast.walk(tree):
        if not isinstance(n, ast.Attribute):
            continue
        if isinstance(n.value, ast.Name) and (alias_src.get(n.value.id), n.attr) in PART_OF:
            out.append(f"{rel}:{n.lineno}: {n.value.id}.{n.attr}")
        d = _dotted(n)
        if d and d.startswith(REG + ".") and tuple(d[len(REG) + 1:].split(".")) in PART_OF:
            out.append(f"{rel}:{n.lineno}: {d}")
    for tok, value in _embedded(text):
        out += [f"{rel}:{tok.start[0]}: (in a string) {x.split(': ', 1)[1]}" for x in remaining_references(value, rel)]
    return out


def _dotted(node: ast.AST) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        return ".".join([node.id, *reversed(parts)])
    return None


# --- the plan -------------------------------------------------------------------------------------------------------------------

def _state(root: Path) -> tuple[list[Whole], list[Part]]:
    """The moves still to make; raises `SplitError` on a tree that is neither split nor splittable."""
    if not (root / VLLM / PKG / "program" / "registry").is_dir():
        raise SplitError(f"{root}: no {VLLM}/{PKG}/program/registry (not a verity tree, or the registry moved: update this tool)")
    wholes = []
    for w in WHOLE:
        old, new = (root / w.old_path).exists(), (root / w.new_path).exists()
        if old and new:
            raise SplitError(f"{w.name}: both {w.old_path} and {w.new_path} exist; reconcile them by hand")
        if not old and not new:
            raise SplitError(f"{w.name}: neither {w.old_path} nor {w.new_path} exists (moved elsewhere? update WHOLE)")
        if old:
            if not (root / w.new_path).parent.is_dir() and w.package not in NEW_PACKAGES:
                raise SplitError(f"{w.name}: the target package {w.package} does not exist")
            wholes.append(w)
    parts = []
    for p in PARTS:
        src = _read(root / p.src_path)
        if src is None:
            raise SplitError(f"{p.src}: {p.src_path} is missing (moved? update PARTS)")
        defs = _top_defs(ast.parse(src))
        here = [n for n in p.names if n in defs]
        dst = _read(root / p.dst_path)
        there = [n for n in p.names if dst is not None and n in _top_defs(ast.parse(dst))]
        if len(here) == len(p.names) and dst is None:
            parts.append(p)
        elif not here and len(there) == len(p.names):
            continue
        else:
            raise SplitError(f"{p.dst}: half split -- in {p.src}: {here}; in {p.dst_path}: {there if dst is not None else 'no file'}")
    for d in DEFINITION_MODULES:
        if not (root / VLLM / PKG / "program" / "registry" / f"{d}.py").exists():
            raise SplitError(f"Definition module {d} is not in the registry (moved? update DEFINITION_MODULES)")
    return wholes, parts


def absolute_imports(text: str, package: str, where: str) -> str:
    """A module's relative imports, made absolute against ``package`` (the one it lived in)."""
    tree = ast.parse(text)
    off = _Offsets(text)
    edits = []
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.level:
            st, en, indent = _whole_lines(text, off, n, where)
            edits.append((st, en, _render_import(indent, _absolute(n, package), [_alias_text(a) for a in n.names],
                                                 _stmt_comments(text, off, n))))
    return _apply_edits(text, edits, where)


def _excluded(rel: str) -> bool:
    return rel.startswith(EXCLUDE_PREFIX) or rel in EXCLUDE_FILES or rel.endswith("CHANGELOG.md")


def plan(root: Path = REPO) -> dict:
    """What `apply` would do: the moves still to make, the new files and every file whose text it would change (relative path ->
    new text). Raises `SplitError`, having changed nothing, when the tree cannot be split."""
    wholes, parts = _state(root)
    files: dict[str, str] = {}      # rel -> text, after the moves and cuts (rel = the new path of a moved module)
    renames = {w.old_path: w.new_path for w in wholes}
    moved_to = {n: p for p in PARTS for n in p.names}
    by_src: dict[str, list[Part]] = {}
    for p in parts:
        by_src.setdefault(p.src, []).append(p)
    for src, ps in by_src.items():
        rel = ps[0].src_path
        text = _read(root / rel) or ""
        orig_tree = ast.parse(text)
        reads_moved: set[str] = set()
        for p in ps:
            cut = _cut(text, p, rel)
            files[p.dst_path] = _dst_text(cut, p, orig_tree, {n: q for n, q in moved_to.items() if q.src == src}, rel)
            reads_moved |= _globals_read(cut.body)
            text = cut.src_text
        names = {n for p in ps for n in p.names}
        text = _drop_from_all(text, names, rel)
        left = {n.id for n in ast.walk(ast.parse(text)) if isinstance(n, ast.Name)}
        if names & left:
            raise SplitError(f"{rel}: the code that stays reads {sorted(names & left)}, which move; move them together or not at all")
        files[rel] = _prune_imports(text, reads_moved - _external_reads(root, src), ps[0].old.rpartition(".")[0], rel)
    edits: dict[str, str] = {}
    for rel in _tracked(root):
        if _excluded(rel):
            continue
        dest = renames.get(rel, rel)
        text = files.get(dest) if dest in files else _read(root / rel)
        if text is None:
            continue
        if dest != rel:
            text = absolute_imports(text, REG, rel)
        elif "registry" not in text:
            continue
        if rel.endswith(".py"):
            new = rewrite_python(text, dest)
            left = remaining_references(new, dest)
            if left:
                raise SplitError("references this tool cannot rewrite:\n  " + "\n  ".join(left))
        elif rel.endswith(TEXT_SUFFIXES):
            new = rewrite_text(text, prose=True)
        else:
            continue
        if rel == ROOTS:
            new = census_roots(new)
        if new != (_read(root / rel) or "") or dest != rel:
            edits[dest] = new
    for rel, text in files.items():
        if rel not in edits:
            edits[rel] = rewrite_python(text, rel)
    for rel, text in edits.items():
        if rel.endswith(".py"):
            try:
                compile(text, rel, "exec")
            except SyntaxError as e:
                raise SplitError(f"{rel}: the rewrite does not compile: {e}") from None
    return {"wholes": wholes, "parts": parts, "renames": renames, "edits": edits}


def census_roots(text: str) -> str:
    """The census roots with each `ROOT_DIRS` package declared as a directory, in place of its moved modules' own entries."""
    lines = text.splitlines(keepends=True)
    for pkg, why in ROOT_DIRS.items():
        entry = pkg.replace(".", "/") + "/"
        own = {w.new for w in WHOLE if w.package == pkg}
        at = [i for i, ln in enumerate(lines) if ln.split("#")[0].strip() in own]
        have = any(ln.split("#")[0].strip() == entry for ln in lines)
        lines = [ln for i, ln in enumerate(lines) if i not in at]
        if not have:
            lines.insert(at[0] if at else len(lines), f"{entry}      # {why}\n")
    return "".join(lines)


def keep_test_only(root: Path, non_live: list[str], log=print) -> list[str]:
    """`dead_code_keep.json` with a `KEEP_REASONS` entry for each new module no census root reaches; another fails."""
    path = root / KEEP
    data = json.loads(path.read_text(encoding="utf-8"))
    mods = data["modules"]
    new = {p.dst for p in PARTS} | {w.new for w in WHOLE} | set(NEW_PACKAGES)
    dead = [m for m in non_live if m in new and m not in mods]
    unexplained = [m for m in dead if m not in KEEP_REASONS]
    if unexplained:
        raise SplitError(f"the census reaches none of {unexplained}, and KEEP_REASONS names no reason to keep them")
    if not dead:
        return []
    text = path.read_text(encoding="utf-8")
    m = re.search(r'("modules":\s*\{)(\s*)', text)
    if not m:
        raise SplitError(f"{KEEP}: no `modules` object")
    pad = m.group(2)
    add = "".join(f"{json.dumps(d)}: {json.dumps(KEEP_REASONS[d], ensure_ascii=False)},{pad}" for d in dead)
    path.write_text(text[: m.end()] + add + text[m.end():], encoding="utf-8")
    after = json.loads(path.read_text(encoding="utf-8"))["modules"]
    if after != {**mods, **{d: KEEP_REASONS[d] for d in dead}}:
        raise SplitError(f"{KEEP}: adding the keep entries changed the others")
    for d in dead:
        log(f"dead_code_keep: kept {d} (only tests read it)")
    return [KEEP]


def _external_reads(root: Path, src: str) -> set[str]:
    """The names other files read from the registry module ``src`` (an import of it, an alias attribute), which it keeps."""
    out = set()
    rx = re.compile(rf"\b{re.escape(src)}\b")
    for rel in _tracked(root):
        if not rel.endswith(".py") or rel.endswith(f"registry/{src}.py") or _excluded(rel):
            continue
        text = _read(root / rel) or ""
        if "registry" not in text or not rx.search(text):
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        mf = _module_of_file(rel)
        aliases = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom):
                mod = _absolute(n, mf[1]) if (n.level and mf) else (n.module or "")
                if mod == f"{REG}.{src}":
                    out |= {a.name for a in n.names}
                elif mod == REG:
                    aliases |= {a.asname or a.name for a in n.names if a.name == src}
            elif isinstance(n, ast.Import):
                aliases |= {a.asname for a in n.names if a.name == f"{REG}.{src}" and a.asname}
        for n in ast.walk(tree):
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in aliases:
                out.add(n.attr)
    return out


# --- the allowlists -------------------------------------------------------------------------------------------------------------

def _file_map() -> dict[str, str]:
    """Old library file (relative to integrations/vllm) -> where its moved code went: each whole move, each part's source."""
    out = {w.old_path[len(VLLM) + 1:]: w.new_path[len(VLLM) + 1:] for w in WHOLE}
    return out


def _canon(s: str) -> str:
    """A detail or file with every new module name read back as the old one it came from (to pair a moved violation)."""
    for p in PARTS:
        s = s.replace(p.dst_path[len(VLLM) + 1:], p.src_path[len(VLLM) + 1:]).replace(p.dst[len(PKG) + 1:], p.old[len(PKG) + 1:])
        s = s.replace(p.dst, p.old)
    for w in WHOLE:
        s = s.replace(w.new_path[len(VLLM) + 1:], w.old_path[len(VLLM) + 1:]).replace(w.new[len(PKG) + 1:], w.old[len(PKG) + 1:])
        s = s.replace(w.new, w.old)
    return s


_SCAN = r"""
import json, sys
sys.path.insert(0, ".")
from tests.lint import _ratchet as R
names = sys.argv[1:]
out = {}
for name in names:
    mod = __import__(f"tests.lint.test_{name}", fromlist=["scan"])
    vs = list(mod.scan())
    if name == "p10_size":
        out[name] = [[*v.key, v.line] for v in vs]
    else:
        out[name] = [list(v.key) for v in vs]
from tests import test_no_by_name_rules as B
out["by_name"] = [[o.file, o.kind, o.detail] for o in B.scan()]
from tests import dead_code_census as C
g = C.Graph()
C.resolve_roots(g)
out["non_live"] = sorted(C.non_live_modules(g))
print(json.dumps(out))
"""


def _scan(root: Path) -> dict:
    names = sorted(p.stem for p in (root / LINT_ALLOWLISTS).glob("p*.json"))
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, "-c", _SCAN, *names], cwd=root / VLLM, capture_output=True, text=True, env=env)
    if r.returncode:
        raise SplitError(f"the lint scans failed on the rewritten tree:\n{r.stderr[-3000:]}")
    return json.loads(r.stdout)


def _write_entries(path: Path, data: dict, entries: list[dict], sort: bool) -> None:
    """``entries`` into the allowlist file, one JSON object per line, in the file's own layout."""
    text = path.read_text(encoding="utf-8")
    m = re.search(r'("entries":\s*\[)(.*?)(\n\s*\])', text, re.S)
    if not m:
        raise SplitError(f"{path}: no one-entry-per-line `entries` list")
    indent = re.search(r"\n(\s*)\{", m.group(2))
    pad = indent.group(1) if indent else "    "
    if sort:
        entries = sorted(entries, key=lambda e: (e["file"], e["kind"], e["symbol"], e["detail"]))
    body = ",".join(f"\n{pad}{json.dumps(e, ensure_ascii=False)}" for e in entries)
    path.write_text(text[: m.start(2)] + body + text[m.end(2):], encoding="utf-8")
    if json.loads(path.read_text(encoding="utf-8"))["entries"] != entries:
        raise SplitError(f"{path}: rewriting the entries changed their meaning")


def _is_sorted(entries: list[dict]) -> bool:
    keys = [(e["file"], e["kind"], e["symbol"], e["detail"]) for e in entries]
    return keys == sorted(keys)


def _canon_key(k: tuple) -> tuple:
    if k[1] == "module-cycle":
        return (k[0], k[1], k[2], " <-> ".join(sorted(_canon(m) for m in k[3].split(" <-> "))))
    return (_canon(k[0]), k[1], k[2], _canon(k[3]))


def _relocate_counted(name: str, entries: list[dict], found: list[tuple], log=print) -> list[dict]:
    """``entries`` of a counted allowlist after the split: each new violation paired with the stale entry it moved from (a
    module cycle may also have lost members), the rest of the stale counts removed; a new violation nothing pairs fails."""
    from collections import Counter

    key = lambda e: (e["file"], e["kind"], e["symbol"], e["detail"])  # noqa: E731
    got = Counter(found)
    allowed = Counter({key(e): e.get("count", 1) for e in entries})
    extra = Counter({k: got[k] - allowed[k] for k in got if got[k] > allowed[k]})
    stale = Counter({k: allowed[k] - got[k] for k in allowed if allowed[k] > got[k]})
    final = Counter(allowed)
    origin: dict[tuple, tuple] = {}
    for k in sorted(extra):
        ck = _canon_key(k)
        cands = [s for s in sorted(stale) if stale[s] and _canon_key(s) == ck]
        if not cands and k[1] == "module-cycle":
            members = set(ck[3].split(" <-> "))
            cands = [s for s in sorted(stale) if stale[s] and s[1] == "module-cycle" and members < set(_canon_key(s)[3].split(" <-> "))]
        for s in cands:
            n = min(extra[k], stale[s])
            extra[k] -= n
            stale[s] -= n
            final[s] -= n
            final[k] += n
            origin.setdefault(k, s)
            log(f"{name}: moved {s} -> {k}")
    bad = sorted(k for k, n in extra.items() if n)
    if bad:
        raise SplitError(f"{name}: violations the split introduced that no stale entry pairs: {bad}")
    for s, n in stale.items():
        if n:
            final[s] -= n
            log(f"{name}: {'deleted' if not final[s] else 'lowered'} {s}")
    out, placed = [], set()

    def emit(k: tuple, e: dict | None) -> None:
        placed.add(k)
        if final[k] <= 0:
            return
        extra_fields = {x: v for x, v in (e or {}).items() if x not in ("file", "kind", "symbol", "detail", "count")}
        out.append({"file": k[0], "kind": k[1], "symbol": k[2], "detail": k[3], **({"count": final[k]} if final[k] != 1 else {}),
                    **extra_fields})

    for e in entries:
        emit(key(e), e)
        for k in sorted(k for k, s in origin.items() if s == key(e) and k not in allowed):
            emit(k, None)
    left = [k for k in final if final[k] > 0 and k not in placed]
    if left:
        raise SplitError(f"{name}: entries without a place: {left}")
    return out


def relocate_allowlists(root: Path, log=print) -> list[str]:
    """Each lint allowlist entry the split made stale moved to the violation it moved to (same kind and symbol, its file and
    detail read back through `_canon`), lowered (P10: the size it has now) or deleted; a new violation nothing pairs fails."""
    found = _scan(root)
    changed = []
    for path in sorted((root / LINT_ALLOWLISTS).glob("p*.json")):
        name = path.stem
        data = json.loads(path.read_text(encoding="utf-8"))
        entries = data["entries"]
        was_sorted = _is_sorted(entries)
        key = lambda e: (e["file"], e["kind"], e["symbol"], e["detail"])  # noqa: E731
        if name == "p10_size":
            have = {tuple(v[:4]): v[4] for v in found[name]}
            caps = {key(e): e.get("count", 1) for e in entries}
            new = {k: n for k, n in have.items() if k not in caps}
            out = []
            for e in entries:
                k = key(e)
                if k in have:
                    if have[k] > caps[k]:
                        raise SplitError(f"{name}: {k} grew to {have[k]} lines (cap {caps[k]})")
                    out.append({**e, "count": have[k]} if have[k] != caps[k] else e)
                    continue
                cand = [nk for nk in new if (_canon(nk[0]), nk[1], nk[2], nk[3]) == k]
                if len(cand) == 1:
                    nk = cand[0]
                    out.append({"file": nk[0], "kind": nk[1], "symbol": nk[2], "detail": nk[3], "count": new.pop(nk)})
                    log(f"{name}: moved {k[0]} {k[2]} -> {nk[0]}")
                else:
                    log(f"{name}: deleted {k} (now within the limit)")
            if new:
                raise SplitError(f"{name}: new oversized code nothing pairs: {sorted(new)}")
        else:
            out = _relocate_counted(name, entries, [tuple(v) for v in found[name]], log)
        if out != entries:
            _write_entries(path, data, out, was_sorted)
            changed.append(str(path.relative_to(root)))
    return changed + relocate_by_name(root, found["by_name"], log) + keep_test_only(root, found["non_live"], log)


def relocate_by_name(root: Path, offenders: list[list[str]], log=print) -> list[str]:
    """The by-name allowlist's entries for a rule that moved, at its new file (the entry matches file, kind and pattern)."""
    path = root / BY_NAME
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data["entries"]
    hit = lambda e, file: any(o[0] == file and o[1] == e["kind"] and re.search(e["pattern"], o[2]) for o in offenders)  # noqa: E731
    out = []
    for e in entries:
        if hit(e, e["file"]):
            out.append(e)
            continue
        dests = sorted({o[0] for o in offenders if _canon(o[0]) == e["file"] and o[0] != e["file"] and hit(e, o[0])})
        if len(dests) == 1:
            out.append({**e, "file": dests[0]})
            log(f"by_name: moved {e['file']} {e['symbol']} -> {dests[0]}")
        elif e["file"] in {p.src_path[len(VLLM) + 1:] for p in PARTS} | set(_file_map()):
            raise SplitError(f"by_name: the entry {e['file']} {e['symbol']} {e['pattern']!r} matches nothing after the split")
        else:
            out.append(e)
    covered = lambda o: any(e["file"] == o[0] and e["kind"] == o[1] and re.search(e["pattern"], o[2]) for e in out)  # noqa: E731
    new = [o for o in offenders if not covered(o) and _canon(o[0]) != o[0]]
    if new:
        raise SplitError(f"by_name: rules in moved code no entry covers: {new}")
    if out == entries:
        return []
    _write_entries(path, data, out, False)
    return [BY_NAME]


# --- apply ----------------------------------------------------------------------------------------------------------------------

def apply(root: Path = REPO, log=print) -> dict:
    p = plan(root)
    vllm = root / VLLM
    for pkg, doc in NEW_PACKAGES.items():
        init = vllm / pkg.replace(".", "/") / "__init__.py"
        if not init.exists() and any(w.package == pkg for w in p["wholes"]):
            init.parent.mkdir(parents=True, exist_ok=True)
            init.write_text(doc)
            _git("add", str(init.relative_to(root)), root=root)
            log(f"created {init.relative_to(root)}")
    for w in p["wholes"]:
        _git("mv", w.old_path, w.new_path, root=root)
        log(f"moved {w.old_path} -> {w.new_path}")
    for rel, text in sorted(p["edits"].items()):
        path = root / rel
        new_file = not path.exists()
        path.write_text(text, encoding="utf-8")
        if new_file:
            _git("add", rel, root=root)
            log(f"created {rel}")
        else:
            log(f"rewrote {rel}")
    try:
        for rel in relocate_allowlists(root, log):
            log(f"rewrote {rel}")
    except SplitError as e:
        raise SplitError(f"{e} (the moves are written; `git reset --hard && git clean -fd` restores the tree)") from e
    if not p["wholes"] and not p["parts"] and not p["edits"]:
        log("nothing to do: the primitives, composites and serving family is split")
    return p


# --- the invariants -------------------------------------------------------------------------------------------------------------

def _digest(fn) -> str | None:
    """The descriptor digest of a circuit-check target: a specialization's Program digest; a primitive's descriptor entry inside
    a one-Call wrapper (which is not registered)."""
    from verity.ml.circuits import codec
    from verity.ml.circuits.annotations import definition_digest
    from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, SpecializedDefinition, bind
    from verity.ml.circuits.program import Program

    if isinstance(fn, SpecializedDefinition):
        return codec.program_digest(codec.encode_program(Program(fn)))
    if isinstance(fn, PrimitiveDefinition):
        params, ret = fn.signature()
        wrap = CompositeDefinition("SplitDigestWrap", 1, (), lambda S: (params, ret), lambda B, S, *a: B.call(fn, *a), register=False)
        return definition_digest(codec.encode_program(Program(bind(wrap))), fn.id)
    return None


def family_ids() -> dict[str, str]:
    """Each Definition family (`Name_vN`) a family Definition module defines (not one it imports), by that module."""
    import importlib

    from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition

    out: dict[str, str] = {}
    for name in DEFINITION_MODULES:
        mod = importlib.import_module(REG if name == "__init__" else f"{REG}.{name}")
        tree = ast.parse(Path(mod.__file__).read_text(encoding="utf-8"))
        imported = {a.asname or a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
        for k, v in vars(mod).items():
            if k not in imported and isinstance(v, (CompositeDefinition, PrimitiveDefinition)):
                out.setdefault(v.id, name)
    return dict(sorted(out.items()))


def snapshot() -> dict:
    """circuit-check's task list on this tree (kind and label of every target, in order), the registered ids, the root errors,
    pins.json's digest, every target's descriptor digest and family, and the family's own Definition ids."""
    from circuit_check import checks as CK
    from circuit_check import cli
    from circuit_check import targets as TG

    TG.load_registries()
    suite = TG.suite()
    real = TG.suite
    TG.suite = lambda: suite
    try:
        tasks = cli._tasks([], CK.Options(), True, None)
    finally:
        TG.suite = real
    return {"tasks": [[t[0], t[1]] for t in tasks],
            "registered": list(TG.REGISTERED),
            "root_errors": [list(e) for e in TG.ROOT_ERRORS],
            "templates": [[label, sts] for label, _sub, sts in TG.templates()],
            "pins_sha256": hashlib.sha256(CK.PINS.read_bytes()).hexdigest() if CK.PINS.exists() else None,
            "digests": {fid: _digest(fn) for fid, fn in suite.items()},
            "families": {fid: TG.family(fn) for fid, fn in suite.items()},
            "family_ids": family_ids()}


def compare(before: dict, after: dict) -> list[str]:
    out = []
    for key in ("tasks", "registered", "root_errors", "templates", "pins_sha256", "family_ids"):
        b, a = before.get(key), after.get(key)
        if b != a:
            if isinstance(b, list):
                bs, as_ = {json.dumps(x) for x in b}, {json.dumps(x) for x in a}
                out.append(f"{key}: {len(bs - as_)} gone {sorted(bs - as_)[:8]}, {len(as_ - bs)} new {sorted(as_ - bs)[:8]}"
                           + ("" if bs != as_ else ", same entries in another order"))
            else:
                out.append(f"{key}: {b} -> {a}")
    for fid in sorted(set(before["digests"]) | set(after["digests"])):
        b, a = before["digests"].get(fid), after["digests"].get(fid)
        if b != a:
            out.append(f"digest {fid}: {b} -> {a}")
    return out


def _tree_path(tree: Path) -> list[str]:
    """This interpreter's workspace source directories, read in ``tree`` instead (each workspace member's import root)."""
    here = None
    for p in sys.path:
        if p.endswith("."):
            here = p[: -len(".")].rstrip("/")
    if here is None:
        raise SplitError("this interpreter has no workspace on its path (run it with the workspace's environment)")
    out = [str(tree / p[len(here) + 1:]) for p in sys.path if p.startswith(here + "/")]
    import site
    for d in site.getsitepackages():
        for pth in Path(d).glob("*.pth"):
            for ln in pth.read_text().splitlines():
                if ln.startswith(here + "/"):
                    out.append(str(tree / ln[len(here) + 1:]))
    return list(dict.fromkeys(out))


def snapshot_tree(tree: Path, out: Path) -> dict:
    """`snapshot` of ``tree``'s sources, in this interpreter, in a fresh process."""
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(_tree_path(tree))}
    code = (f"import sys, json; sys.path[:0] = {_tree_path(tree)!r}; sys.path.insert(0, {str(Path(__file__).parent)!r}); "
            "import split_prims_serve as S; snap = S.snapshot(); "
            f"bad = sorted(m for m, v in sys.modules.items() if m.startswith(('verity', 'circuit_check')) and getattr(v, '__file__', None) "
            f"and not v.__file__.startswith({str(tree)!r})); "
            "assert not bad, f'modules from outside the tree: {bad[:5]}'; "
            f"open({str(out)!r}, 'w').write(json.dumps(snap, indent=1, sort_keys=True, default=str) + chr(10))")
    with tempfile.TemporaryDirectory() as cwd:                     # an empty directory: nothing on sys.path[0] shadows a module
        subprocess.run([sys.executable, "-c", code], cwd=cwd, check=True, env=env)
    return json.loads(out.read_text())


def verify(before_rev: str, out_dir: Path) -> list[str]:
    """Snapshot ``before_rev`` (in a fresh worktree) and this tree, and compare."""
    out_dir.mkdir(parents=True, exist_ok=True)
    wt = Path(tempfile.mkdtemp(prefix="split-prims-serve-before-"))
    _git("worktree", "add", "--detach", str(wt), before_rev)
    try:
        before = snapshot_tree(wt, out_dir / "before.json")
    finally:
        _git("worktree", "remove", "--force", str(wt))
    after = snapshot_tree(REPO, out_dir / "after.json")
    return compare(before, after)


def pin(snap: dict) -> dict:
    """The fixture: every circuit-check task whose target is of a family Definition, in task order, with its descriptor digest;
    the family's registered ids; and the pins of those targets."""
    from circuit_check import checks as CK

    fam = set(snap["family_ids"])
    tasks = [[k, label] for k, label in snap["tasks"] if k == "definition" and snap["families"].get(label) in fam]
    pins = CK.load_pins()
    labels = {label for _, label in tasks}
    return {"family": "primitives, composites and serving", "tool": TOOL,
            "family_ids": snap["family_ids"],
            "tasks": tasks,
            "digests": {label: snap["digests"][label] for _, label in tasks},
            "registered": sorted(i for i in snap["registered"] if i in fam),
            "pins": {sec: {k: v for k, v in (pins.get(sec) or {}).items() if k.split(" @ ")[0] in labels or k in labels}
                     for sec in sorted(pins)}}


# --- inventory ------------------------------------------------------------------------------------------------------------------

def inventory(root: Path = REPO) -> str:
    """Markdown: for each family module, its top-level names by kind (on this tree: run it before `apply` for the whole picture),
    and every `verity_vllm` import of a Definition module with where it points."""
    import importlib

    from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, SpecializedDefinition

    rows = []
    moved = {(p.src, n): p for p in PARTS for n in p.names}
    for name in [*DEFINITION_MODULES, *[w.name for w in WHOLE]]:
        rel = f"{VLLM}/{REG.replace('.', '/')}/{name}.py"
        if not (root / rel).exists():
            continue
        tree = ast.parse((root / rel).read_text(encoding="utf-8"))
        mod = importlib.import_module(REG if name == "__init__" else f"{REG}.{name}")
        imported = set(_import_bindings(tree, REG))
        kinds: dict[str, list[str]] = {}
        for s in tree.body:
            for n in sorted(_defined(s)):
                if name in WHOLE_BY:
                    k = WHOLE_BY[name].kind
                elif (name, n) in moved:
                    k = f"{moved[(name, n)].kind} -> {moved[(name, n)].dst[len(PKG) + 1:]}"
                elif isinstance(getattr(mod, n, None), (PrimitiveDefinition, CompositeDefinition, SpecializedDefinition)):
                    k = "Definition"
                else:
                    k = "other (Definition-side: statics, encodings, helpers the bodies read)"
                kinds.setdefault(k, []).append(n)
        reexp = sorted(n for n in imported if isinstance(getattr(mod, n, None), (PrimitiveDefinition, CompositeDefinition)))
        rows.append(f"### `registry/{name}.py`\n")
        if name in WHOLE_BY:
            w = WHOLE_BY[name]
            rows.append(f"Moves whole to `{w.new_path[len(VLLM) + 1:]}` ({w.kind}): {w.why}.\n")
        for k, ns in sorted(kinds.items()):
            rows.append(f"- {k} ({len(ns)}): " + ", ".join(f"`{n}`" for n in ns))
        if reexp:
            rows.append(f"- re-exported Definitions of other modules ({len(reexp)}): " + ", ".join(f"`{n}`" for n in reexp))
        if name not in WHOLE_BY:
            imps = []
            for n in ast.walk(tree):
                if isinstance(n, (ast.Import, ast.ImportFrom)):
                    mods = [a.name for a in n.names] if isinstance(n, ast.Import) else [_absolute(n, REG)]
                    for m in mods:
                        if m.startswith(PKG) or m.startswith("verity.core.protocols.pouw"):
                            where = next((f"{p.kind}, moves to `{p.dst[len(PKG) + 1:]}`" for p in PARTS if p.src == name
                                          and any(n.lineno >= s.lineno and n.end_lineno <= s.end_lineno for s in tree.body
                                                  if _defined(s) & set(p.names))), None)
                            imps.append(f"`{m}` (line {n.lineno}{', ' + where if where else ''})")
            rows.append("- `verity_vllm` imports: " + (", ".join(dict.fromkeys(imps)) if imps else "none"))
        rows.append("")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan", help="list what apply would change")
    sub.add_parser("apply", help="move the twins and vocabulary and rewrite every reference")
    s = sub.add_parser("snapshot", help="write circuit-check's task list and digests on this tree")
    s.add_argument("out", type=Path)
    v = sub.add_parser("verify", help="compare the task list and digests of REV with this tree")
    v.add_argument("--before", required=True)
    v.add_argument("--out", type=Path, default=Path(tempfile.gettempdir()) / "split-prims-serve")
    pn = sub.add_parser("pin", help="write the family's fixture from a snapshot")
    pn.add_argument("snapshot", type=Path)
    sub.add_parser("inventory", help="print the family modules' top-level names by kind")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "plan":
            p = plan()
            for w in p["wholes"]:
                print(f"move {w.old_path} -> {w.new_path} ({w.kind})")
            for part in p["parts"]:
                print(f"cut {part.src_path} {', '.join(part.names)} -> {part.dst_path} ({part.kind})")
            for rel in sorted(p["edits"]):
                print(f"rewrite {rel}")
            if not p["wholes"] and not p["parts"] and not p["edits"]:
                print("nothing to do: the primitives, composites and serving family is split")
        elif a.cmd == "apply":
            apply()
        elif a.cmd == "snapshot":
            a.out.write_text(json.dumps(snapshot(), indent=1, sort_keys=True, default=str) + "\n")
        elif a.cmd == "verify":
            diffs = verify(a.before, a.out)
            for d in diffs:
                print(d)
            if diffs:
                print(f"FAIL: {len(diffs)} differences between {a.before} and this tree (snapshots in {a.out})")
                return 1
            snap = json.loads((a.out / "after.json").read_text())
            print(f"ok: {len(snap['tasks'])} circuit-check tasks, {len(snap['registered'])} registered ids, {len(snap['templates'])} "
                  f"templates and {sum(d is not None for d in snap['digests'].values())} target digests are the same on {a.before} "
                  "and this tree")
        elif a.cmd == "pin":
            path = REPO / FIXTURE
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(pin(json.loads(a.snapshot.read_text())), indent=1, sort_keys=True) + "\n")
            print(f"wrote {FIXTURE}")
        elif a.cmd == "inventory":
            print(inventory())
    except SplitError as e:
        print(f"split_prims_serve: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
