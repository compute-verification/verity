"""By-name lint (Draft 3.1 §12.3 phase 6 gate): no new by-name rules in `verity_vllm`.

A *by-name rule* answers a verification question ("is this Value a boundary value?", "which module do we hook?", "which
family is a collective?", "does this model have a qkv bias?") from a model name, a module-path fragment or a Definition
family string instead of from the Program, the query or the runtime correspondence (Draft 3.1 §12.4: an answer of the
form "because the name is in this list" is a structural leak).  The inventory (Appendix A of the ontology draft)
found 98 such decision points, ~68 structural; v2 retires them phase by phase.  This test pins the current set so that it
can only shrink.

What is scanned: every ``*.py`` under ``integrations/vllm/verity_vllm`` and ``verity/ml/definitions`` (the Definitions
that moved from it, keyed by their repository paths) except ``tests/`` directories and
``test_*.py`` files (test fixtures may name models).  Docstrings, comments, f-strings, ``raise``/``assert`` messages,
message-like keyword/dict positions (``hint=``, ``why=``, ``help=``, …) and a module's ``__models__`` marker (the models it
exists for, ``verity_vllm.glue``) are not rule positions and are ignored.

Three kinds of offender (``kind`` in the allowlist):

* ``model-name`` — a model / family name (llama, qwen, gemma, olmoe, mixtral, mistral, smollm, deepseek, phi, pythia,
  gpt-neox) or an HF repo id as a string literal in a rule position (assignment, container element, comparison operand,
  call argument).
* ``path-predicate`` — ``.startswith(`` / ``.endswith(`` / ``in`` / ``not in`` applied to a module path, a module leaf,
  a class name or a Definition family string, or against one of the tables below, in decision logic.
* ``table`` — a module- or class-level constant whose name ends in FAMILIES / LEAVES / PREFIXES / SUFFIX / MODULES /
  CLASSES / INTERIOR / RULES / MARKERS / MEMBERS / PRODUCERS / PLANES / MODELS / FIELDS / _RE … and whose literal
  contents are model, module-path, class or family names (``_DENSE_INTERIOR``, ``MEMBER_RULES``, ``_LINEAR_LEAVES``,
  ``FIRST_NORM_SUFFIX``, ``FAMILY_OF_MODULE``, ``ACQUIRE_CLASSES``, ``HAZARD_LEAVES``, the per-model profile field maps, …).

The allowlist ``by_name_allowlist.json`` beside this file lists every current offender: ``file``, ``kind``, ``symbol``
(enclosing function or the table name), ``pattern`` (a regex over the extracted expression), ``class`` (S = structural
by-name, L = legitimate semantic knowledge, P = policy / serialization vocabulary, from the inventory), ``retires`` (a key
of the file's ``phases`` map: the v2 phase that removes it, or ``stays`` for L/P), ``tag`` (finding tag, if any) and
``inventory`` (Appendix A row).  The test
fails on any offender no entry covers (a new by-name rule) and on any entry that covers nothing (a stale entry: delete it
when the rule is gone, so the list only shrinks truthfully).  Run ``python tests/test_no_by_name_rules.py`` from
``integrations/vllm`` for the per-file / per-kind breakdown.
"""
from __future__ import annotations

import ast
import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]  # integrations/vllm
REPO = ROOT.parents[1]
#: (base, directory): the package, and the Definitions that moved from it to `verity.ml.definitions` (circuits', held to this
#: list as it stood), each file keyed by its path from its base
SCAN = ((ROOT, "verity_vllm"), (REPO, "verity/ml/definitions"))
SKIP_DIRS = frozenset({"tests", "__pycache__", "kernels", "numerics"}); SKIP_FILES = frozenset({"engine_profile.py"})   # relayout: program/kernels/ and engine/engine_profile.py come from the two packages SCAN never covered; 13 by-name rules -- decide (Draft 3.1 §12.4), do not just allowlist
#: program/kernels/ (`numerics/` in `verity.ml.definitions`), and its modules that stay in the package, skipped wherever they go
SKIP_FILES |= {"relations.py", "sampling_rng.py", "libdevice_sampling.py", "rmsnorm_fused_sweep.py", "gemm_target_correspondence.py",
               "compiled_relations.py", "compiled_norm_literal.py"}
#: engine_profile.py's `CASES`, which moved here with 11 of its 13 by-name rules: skipped by path, as its old file is by name
SKIP_PATHS = frozenset({"verity_vllm/engine/profiles/cases.py"})
ALLOWLIST = HERE / "by_name_allowlist.json"
KINDS = ("model-name", "path-predicate", "table")

MODEL_NAME_RE = re.compile(
    r"(?i)(?<![a-z0-9])(llama|qwen\d*|gemma\d*|olmoe|mixtral|mistral|smollm\d*|deepseek|phi\d+|phi(?=[-_ ])|pythia|gpt[-_]?neox)(?![a-z])")
LEAF_RE = re.compile(
    r"(?i)(_proj\b|\bproj\b|layernorm|(?<![a-z])norm\b|embed|lm_head|\blayers\.|\.layers\b|\bexperts\b|\bgate\b|\bdown\b|\bup\b"
    r"|\bqkv\b|\bmlp\b|\battn\b|\bself_attn\b|\bfc[12]\b|\bwte\b|\bw13\b|\bw2\b|\brouter\b|\bmodel\.|\binner\.|\bsampler\b|\bo_proj"
    r"|\bcos_sin_cache\b|_scale\b)")
FAMILY_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9]*_v\d+\b|\b(Gemm|Moe[A-Z][A-Za-z]*|Attention|Embedding[A-Za-z]*|[A-Za-z]*TokenSelect|AllReduce\d*|AllGather\d*"
    r"|RoPE|RMSNorm[A-Za-z]*|BiasAdd|SiluMul|Const|LBot|LStepLive|LSelect|LIsEOS|LEmitNext|Narrow[A-Za-z0-9]+|Square[A-Za-z0-9]+"
    r"|MeanTriton|AddScalar[A-Za-z0-9]+|AddWidened[A-Za-z0-9]+|Rsqrt[A-Za-z0-9]+|ScaleRow[A-Za-z0-9]+|Gumbel[A-Za-z]*|Lifted)\b")
CLASS_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9]*(Linear|LayerNorm|RMSNorm|Embedding|Attention|FusedMoE|LMHead|Sampler|DecoderLayer|Model|MoE|RotaryEmbedding"
    r"|Experts|ForCausalLM)\b")
PLANE_RE = re.compile(r"\b(fa2|fa3)\.m1/|\bmoe/")
TABLE_NAME_RE = re.compile(
    r"^_?[A-Z][A-Z0-9_]*(FAMILIES|LEAVES|PREFIXES|SUFFIX|SUFFIXES|MODULES|CLASSES|INTERIOR|RULES|MARKERS|MEMBERS|PRODUCERS|PLANES"
    r"|MODELS|LEAF|OF_MODULE|OF_KIND|BY_SITE_CLASS|_RE|FUSED|EXPERTS[A-Z0-9_]*|LIFTS|TABLES|PIN|ROOTS|FIELDS|MAPPING|W|_ROW|_REFUSED)$")
#: Table names whose *use* in a predicate is an offender even when the table is defined in another module.
STRONG_TABLE_RE = re.compile(r"^_?[A-Z][A-Z0-9_]*(FAMILIES|LEAVES|PREFIXES|CLASSES|MODULES|INTERIOR|MARKERS|PRODUCERS|OF_MODULE|MODELS|SUFFIX)$")
#: A receiver that names a module path / leaf / family / class (`family.startswith(...)`, `"proj" in leaf`).
PATHY_RE = re.compile(r"(?i)(module|op_path|\bmod\b|\bfam\b|family|families|leaf|leaves|\bcls\b|klass|class|member|\bsite|\bspec|\bhint|\bmc\b|producer|\bmodel\b)")
NOT_PATHY_RE = re.compile(r"(?i)(url|http|dirname|fname|filename|\bfn\b|line|text|argv|\bs\b|\bout\b|\barg\b|\bword|\btok|json|yaml|\.suffix|__module__|__name__|type\()")
#: A string argument to startswith/endswith that is a separator / marker, not a name.
NON_NAME_PREFIXES = (".", "/", "_", "-", "#", "|", "=", "[", "(", "{", "<")

MESSAGE_FUNCS = frozenset({"print", "warn", "warning", "info", "debug", "error", "exception", "log", "format", "write", "fail", "skip",
                           "note", "add_argument", "option", "join", "critical", "echo", "msg", "hint", "explain", "describe", "report",
                           "summarise", "summarize"})
MESSAGE_KEYS = frozenset({"help", "description", "hint", "why", "note", "reason", "detail", "msg", "message", "notes", "explanation",
                          "summary", "text", "doc", "comment", "title", "label", "caption", "what", "how", "problem", "problems",
                          "pending", "meaning", "source", "calibration", "evidence_domain", "provenance", "origin", "basis",
                          "measured", "cite", "citation", "ruling"})


def _vocab(s: str) -> bool:
    return bool(MODEL_NAME_RE.search(s) or LEAF_RE.search(s) or FAMILY_RE.search(s) or CLASS_RE.search(s) or PLANE_RE.search(s))


def _is_str(node: ast.AST) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


@dataclass(frozen=True)
class Offender:
    file: str      # relative to integrations/vllm, posix
    kind: str
    line: int
    symbol: str    # enclosing def / class, or the table name
    detail: str    # the extracted expression the allowlist pattern is matched against

    def key(self) -> tuple[str, str, str]:
        return (self.file, self.kind, self.detail)


class _Finder(ast.NodeVisitor):
    def __init__(self, src: str, rel: str):
        self.lines = src.splitlines()
        self.rel = rel
        self.out: list[Offender] = []
        self.msg_depth = 0
        self.scope: list[str] = []
        self.tables: set[str] = set()

    # --- helpers ---
    def _seg(self, node: ast.AST) -> str:
        try:
            l0, c0, l1, c1 = node.lineno - 1, node.col_offset, node.end_lineno - 1, node.end_col_offset
        except AttributeError:
            return ""
        if l0 == l1:
            return self.lines[l0][c0:c1]
        return " ".join([self.lines[l0][c0:]] + self.lines[l0 + 1:l1] + [self.lines[l1][:c1]])

    def _emit(self, kind: str, node: ast.AST, detail: str, symbol: str | None = None) -> None:
        self.out.append(Offender(self.rel, kind, node.lineno, symbol or (self.scope[-1] if self.scope else "<module>"), detail))

    def _in_msg(self, fn) -> None:
        self.msg_depth += 1
        try:
            fn()
        finally:
            self.msg_depth -= 1

    # --- message contexts (not rule positions) ---
    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        self._in_msg(lambda: self.generic_visit(node))

    def visit_Raise(self, node: ast.Raise) -> None:
        self._in_msg(lambda: self.generic_visit(node))

    def visit_Assert(self, node: ast.Assert) -> None:
        self.visit(node.test)
        if node.msg is not None:
            self._in_msg(lambda: self.visit(node.msg))

    def visit_Expr(self, node: ast.Expr) -> None:
        if _is_str(node.value):          # docstring / bare string statement
            return
        self.generic_visit(node)

    def visit_keyword(self, node: ast.keyword) -> None:
        if node.arg in MESSAGE_KEYS:
            self._in_msg(lambda: self.generic_visit(node))
        else:
            self.generic_visit(node)

    def visit_Dict(self, node: ast.Dict) -> None:
        for k, v in zip(node.keys, node.values):
            if k is not None:
                self.visit(k)
            if _is_str(k) and k.value in MESSAGE_KEYS:
                self._in_msg(lambda v=v: self.visit(v))
            else:
                self.visit(v)

    # --- (b) path predicates ---
    def visit_Call(self, node: ast.Call) -> None:
        fname = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
        if isinstance(node.func, ast.Attribute) and node.func.attr in ("startswith", "endswith"):
            recv = self._seg(node.func.value)
            str_args = [a for a in node.args if _is_str(a)]
            tuple_strs = [e for a in node.args if isinstance(a, ast.Tuple) for e in a.elts if _is_str(e)]
            named = [a.id if isinstance(a, ast.Name) else a.attr for a in node.args if isinstance(a, (ast.Name, ast.Attribute))]
            arg_vocab = (any(_vocab(a.value) for a in str_args) or any(_vocab(e.value) for e in tuple_strs)
                         or any(n in self.tables or STRONG_TABLE_RE.match(n) for n in named))
            pathy = PATHY_RE.search(recv) and not NOT_PATHY_RE.search(recv)
            name_arg = any(not a.value.startswith(NON_NAME_PREFIXES) for a in str_args)
            if arg_vocab or (pathy and name_arg):
                self._emit("path-predicate", node, f"{recv}.{node.func.attr}({', '.join(self._seg(a) for a in node.args)})")
        if fname in MESSAGE_FUNCS or fname.endswith(("Error", "Exception", "Refused", "Fail", "Warning")):
            self.visit(node.func)
            self._in_msg(lambda: [self.visit(a) for a in node.args])
            self._in_msg(lambda: [self.visit(k) for k in node.keywords])
            return
        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        if len(node.ops) == 1 and isinstance(node.ops[0], (ast.In, ast.NotIn)):
            left, right = node.left, node.comparators[0]
            ls, rs = self._seg(left), self._seg(right)
            hit = False
            if isinstance(right, (ast.Name, ast.Attribute)):
                nm = right.id if isinstance(right, ast.Name) else right.attr
                hit = nm in self.tables or bool(STRONG_TABLE_RE.match(nm))
            elif isinstance(right, (ast.Tuple, ast.List, ast.Set)):
                strs = [e.value for e in right.elts if _is_str(e)]
                hit = bool(strs) and any(_vocab(s) for s in strs)
            if _is_str(left) and isinstance(right, (ast.Name, ast.Attribute, ast.Subscript, ast.Call)):
                if _vocab(left.value) and PATHY_RE.search(rs) and not NOT_PATHY_RE.search(rs):
                    hit = True                                        # `"proj" in leaf`
            if hit:
                op = "in" if isinstance(node.ops[0], ast.In) else "not in"
                self._emit("path-predicate", node, f"{ls} {op} {rs}")
        self.generic_visit(node)

    # --- (c) tables ---
    def _table(self, target: ast.AST, value: ast.AST, node: ast.AST) -> None:
        name = target.id if isinstance(target, ast.Name) else target.attr if isinstance(target, ast.Attribute) else None
        if not name or not TABLE_NAME_RE.match(name):
            return
        strs = [sub.value for sub in ast.walk(value) if _is_str(sub)]
        if strs and any(_vocab(s) for s in strs):
            self.tables.add(name)
            self._emit("table", node, name, symbol=name)

    def visit_Assign(self, node: ast.Assign) -> None:
        if not self.scope and any(isinstance(t, ast.Name) and t.id == "__models__" for t in node.targets):
            return
        if not self.scope or self.scope[-1].startswith("class "):
            for t in node.targets:
                self._table(t, node.value, node)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None and (not self.scope or self.scope[-1].startswith("class ")):
            self._table(node.target, node.value, node)
        self.generic_visit(node)

    # --- scopes ---
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        for d in node.decorator_list:
            self.visit(d)
        for a in node.args.defaults + node.args.kw_defaults:
            if a is not None:
                self.visit(a)
        self.scope.append(node.name)
        body = node.body[1:] if node.body and isinstance(node.body[0], ast.Expr) and _is_str(node.body[0].value) else node.body
        for b in body:
            self.visit(b)
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(f"class {node.name}")
        for b in node.body:
            self.visit(b)
        self.scope.pop()

    # --- (a) model names ---
    def visit_Constant(self, node: ast.Constant) -> None:
        v = node.value
        if isinstance(v, str) and self.msg_depth == 0 and MODEL_NAME_RE.search(v) and len(v) <= 120 \
                and not (len(v) > 48 and " " in v.strip()):     # a long string with spaces is prose, not a name
            self._emit("model-name", node, v)


def iter_files() -> list[Path]:
    out: list[Path] = []
    for base, top in SCAN:
        for dirpath, dirnames, filenames in os.walk(base / top):
            dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS)
            out.extend(Path(dirpath) / fn for fn in sorted(filenames) if fn.endswith(".py") and not fn.startswith("test_") and fn not in SKIP_FILES
                       and (Path(dirpath) / fn).relative_to(base).as_posix() not in SKIP_PATHS)
    return out


def scan() -> list[Offender]:
    res: list[Offender] = []
    for p in iter_files():
        rel = p.relative_to(ROOT if p.is_relative_to(ROOT) else REPO).as_posix()
        src = p.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(p))
        f = _Finder(src, rel)
        f.visit(tree)
        res.extend(f.out)
    return res


def load_allowlist() -> list[dict]:
    data = json.loads(ALLOWLIST.read_text(encoding="utf-8"))
    assert isinstance(data, dict) and isinstance(data.get("entries"), list) and isinstance(data.get("phases"), dict), \
        "by_name_allowlist.json: {'phases': {...}, 'entries': [...]} expected"
    for e in data["entries"]:
        for k in ("file", "kind", "symbol", "pattern", "class", "retires"):
            assert k in e, f"allowlist entry missing {k!r}: {e}"
        assert e["kind"] in KINDS, f"allowlist entry with unknown kind: {e}"
        assert e["class"] in ("S", "L", "P"), f"allowlist entry with unknown class: {e}"
        assert e["retires"] in data["phases"], f"allowlist entry with unknown phase key: {e}"
        re.compile(e["pattern"])
    return data["entries"]


def match(offenders: list[Offender], entries: list[dict]) -> tuple[list[Offender], list[dict]]:
    """Returns (offenders no entry covers, entries covering no offender)."""
    compiled = [(e, re.compile(e["pattern"])) for e in entries]
    uncovered: list[Offender] = []
    used = [0] * len(compiled)
    for o in offenders:
        hit = False
        for i, (e, rx) in enumerate(compiled):
            if e["file"] == o.file and e["kind"] == o.kind and rx.search(o.detail):
                used[i] += 1
                hit = True
        if not hit:
            uncovered.append(o)
    stale = [e for (e, _), n in zip(compiled, used) if n == 0]
    return uncovered, stale


def test_every_by_name_rule_is_allowlisted() -> None:
    offenders = scan()
    entries = load_allowlist()
    uncovered, _ = match(offenders, entries)
    assert not uncovered, (
        f"{len(uncovered)} by-name rule(s) not in {ALLOWLIST.name} (Draft 3.1 §12.4: decide from the Program / query / "
        "runtime correspondence, not from a name; if this is an existing rule that moved, update its entry):\n"
        + "\n".join(f"  {o.file}:{o.line} [{o.kind}] {o.symbol}: {o.detail}" for o in uncovered))


@pytest.mark.slow
def test_allowlist_only_shrinks() -> None:
    offenders = scan()
    entries = load_allowlist()
    _, stale = match(offenders, entries)
    assert not stale, (
        f"{len(stale)} allowlist entr(y/ies) match nothing any more -- the by-name rule is gone, delete the entry:\n"
        + "\n".join(f"  {e['file']} [{e['kind']}] {e['symbol']}: /{e['pattern']}/" for e in stale))


def test_a_models_marker_is_not_a_rule_position() -> None:
    src = "__models__ = ('deepseek-ai/DeepSeek-V4-Pro',)\nFAMILIES = ('deepseek',)\n"
    f = _Finder(src, "verity_vllm/x.py")
    f.visit(ast.parse(src))
    assert f.out and {o.line for o in f.out} == {2}, f.out


def test_allowlist_entries_are_unique() -> None:
    entries = load_allowlist()
    keys = Counter((e["file"], e["kind"], e["pattern"]) for e in entries)
    dups = [k for k, n in keys.items() if n > 1]
    assert not dups, "duplicate allowlist entries:\n" + "\n".join(map(str, dups))


def _report(offenders: list[Offender], entries: list[dict]) -> str:
    by_kind = Counter(o.kind for o in offenders)
    by_file = Counter(o.file for o in offenders)
    by_file_kind: dict[str, Counter] = {}
    for o in offenders:
        by_file_kind.setdefault(o.file, Counter())[o.kind] += 1
    points = {(o.file, o.symbol) for o in offenders}
    by_class = Counter(e["class"] for e in entries)
    lines = [f"{len(offenders)} by-name sites in {len(by_file)} files; {len(points)} (file, symbol) decision points; "
             f"allowlist {len(entries)} entries (S {by_class['S']}, L {by_class['L']}, P {by_class['P']})",
             "by kind: " + ", ".join(f"{k} {by_kind[k]}" for k in KINDS), ""]
    for f, n in by_file.most_common():
        kinds = ", ".join(f"{k} {c}" for k, c in sorted(by_file_kind[f].items()))
        lines.append(f"{n:4d}  {f}  ({kinds})")
    return "\n".join(lines)


if __name__ == "__main__":
    offs = scan()
    ents = load_allowlist() if ALLOWLIST.exists() else []
    print(_report(offs, ents))
    if ents:
        unc, st = match(offs, ents)
        print(f"\nuncovered {len(unc)}, stale {len(st)}")
        for o in unc:
            print(f"  NEW  {o.file}:{o.line} [{o.kind}] {o.symbol}: {o.detail}")
        for e in st:
            print(f"  STALE {e['file']} [{e['kind']}] {e['symbol']}: /{e['pattern']}/")
        sys.exit(1 if (unc or st) else 0)
