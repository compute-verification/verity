"""P3: one evaluator interface, and one challenge derivation.

Checked today, against ``allowlists/p03_one_evaluator.json``:

* ``if-ladder``: at least ``LADDER_MIN`` tests of one subject against family or Definition names (``"Gemm_v1"``,
  ``"MoeSum_v1"``, ``"FusedMoE"``; by ``==``, ``in (...)``, ``.startswith``/``.endswith``) among the sibling ``if``
  statements of one block and their ``elif`` branches, or as the cases of one ``match``: per-family dispatch that
  belongs to a registered backend.  Keyed by the subject expression.
* ``rng``: constructing a random generator (``random.Random``, ``random.SystemRandom``, numpy's ``default_rng``,
  ``RandomState``, ``Generator``, ``torch.Generator``): challenge randomness comes from the one challenge module.

The kernels (``verity/ml/definitions/numerics/``) and the modules ``LADDER_HOMES`` names are exempt from ``if-ladder``.
The one challenge module, ``commit/challenge.py``, is the ``RNG_OWNERS`` entry: its legacy derivations keep
``random.Random`` and numpy generators until the re-baseline epoch.
The rule's parametrized self-check (every registered (kernel, Definition) pair against the core reference on seeded
random inputs) is ``tests/program/test_kernel_self_check.py``, not a lint.
"""
from __future__ import annotations

import ast
import re
from functools import lru_cache

from . import _ratchet as R

NAME = "p03_one_evaluator"
RULE = "P3 (one evaluator interface, one challenge derivation)"
LADDER_MIN = 4
#: program/kernels/, and by name the modules of it that stay in the package when its Definitions move to `verity.ml.definitions`
LADDER_HOMES = ("verity/ml/definitions/numerics/", "verity_vllm/program/sampling_rng.py",
                "verity_vllm/program/libdevice_sampling.py", "verity_vllm/program/rmsnorm_fused_sweep.py",
                "verity_vllm/program/gemm_target_correspondence.py", "verity_vllm/program/compiled_relations.py",
                "verity_vllm/program/compiled_norm_literal.py")
FAMILY = re.compile(r"^[A-Z][A-Za-z0-9]*(?:_v[0-9]+)?$")
RNG_OWNERS: frozenset[str] = frozenset({"verity_vllm/commit/challenge.py"})
RNG_CALLS = frozenset({"random.Random", "random.SystemRandom", "numpy.random.default_rng", "numpy.random.RandomState",
                       "numpy.random.Generator", "torch.Generator"})


def _family(node: ast.AST) -> bool:
    return R.is_str(node) and bool(FAMILY.match(node.value))


def _string_test_subject(test: ast.AST) -> str | None:
    """The subject of ``s == "Gemm_v1"``, ``s in ("Gemm_v1", "Gemm_v2")``, ``s.startswith("Moe")`` (possibly under
    ``or``) where every literal is a family or Definition name, else None."""
    if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.Or):
        subs = {_string_test_subject(v) for v in test.values}
        return subs.pop() if len(subs) == 1 and None not in subs else None
    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        left, right, op = test.left, test.comparators[0], test.ops[0]
        if isinstance(op, (ast.Eq, ast.NotEq)):
            if _family(right) and not R.is_str(left):
                return R.text(left)
            if _family(left) and not R.is_str(right):
                return R.text(right)
        if isinstance(op, (ast.In, ast.NotIn)) and isinstance(right, (ast.Tuple, ast.List, ast.Set)) \
                and right.elts and all(_family(e) for e in right.elts):
            return R.text(left)
    if isinstance(test, ast.Call) and isinstance(test.func, ast.Attribute) \
            and test.func.attr in ("startswith", "endswith") and test.args \
            and (_family(test.args[0]) or (isinstance(test.args[0], ast.Tuple) and all(_family(e) for e in test.args[0].elts))):
        return R.text(test.func.value)
    return None


def _block_ladders(block: list[ast.stmt], symbol: str, rel: str, elif_ids: set[int]) -> list[R.Violation]:
    """Ladders among one block's ``if`` statements: sibling ``if``s and their ``elif`` chains testing one subject."""
    counts: dict[str, int] = {}
    first: dict[str, int] = {}
    for st in block:
        if not isinstance(st, ast.If) or id(st) in elif_ids:
            continue
        cur: ast.AST = st
        while isinstance(cur, ast.If):
            s = _string_test_subject(cur.test)
            if s:
                counts[s] = counts.get(s, 0) + 1
                first.setdefault(s, st.lineno)
            if len(cur.orelse) == 1 and isinstance(cur.orelse[0], ast.If):
                elif_ids.add(id(cur.orelse[0]))
                cur = cur.orelse[0]
            else:
                break
    return [R.Violation(rel, "if-ladder", symbol, s, first[s]) for s, n in counts.items() if n >= LADDER_MIN]


def file_violations(src: str, rel: str) -> list[R.Violation]:
    tree = R.parse_src(src)
    aliases = R.aliases(tree)
    out: list[R.Violation] = []
    elif_ids: set[int] = set()
    ladders = not rel.startswith(LADDER_HOMES)
    if ladders:
        out += _block_ladders(tree.body, "<module>", rel, elif_ids)
    for node, symbol in R.walk(tree):
        if ladders:
            inner = (node.name if symbol == "<module>" else f"{symbol}.{node.name}") \
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else symbol
            for field in ("body", "orelse", "finalbody"):
                block = getattr(node, field, None)
                if isinstance(block, list) and block and isinstance(block[0], ast.stmt):
                    out += _block_ladders(block, inner, rel, elif_ids)
        if ladders and isinstance(node, ast.Match):
            n = sum(1 for c in node.cases if isinstance(c.pattern, ast.MatchValue) and _family(c.pattern.value)
                    or isinstance(c.pattern, ast.MatchOr) and all(isinstance(p, ast.MatchValue) and _family(p.value)
                                                                  for p in c.pattern.patterns))
            if n >= LADDER_MIN:
                out.append(R.Violation(rel, "if-ladder", symbol, R.text(node.subject), node.lineno))
        if isinstance(node, ast.Call) and rel not in RNG_OWNERS:
            origin = aliases.resolve(node.func)
            if origin in RNG_CALLS:
                out.append(R.Violation(rel, "rng", symbol, origin, node.lineno))
    return out


@lru_cache(maxsize=None)
def scan() -> tuple[R.Violation, ...]:
    out: list[R.Violation] = []
    for p in R.library_files():
        out += file_violations(R.source(p), R.rel(p))
    return tuple(out)


def test_no_new_violations() -> None:
    R.assert_no_new(NAME, scan(), RULE,
                    "Dispatch on a family through a registry of backends, and derive challenge randomness in the one "
                    "challenge module.")


def test_allowlist_has_no_stale_entries() -> None:
    R.assert_no_stale(NAME, scan(), RULE)


def test_detectors_on_examples() -> None:
    bad = (
        "import random\n"
        "import numpy as np\n"
        "def evaluate(fam, x):\n"
        "    rng = np.random.default_rng(0)\n"
        "    r = random.Random(1)\n"
        "    if fam == 'Gemm_v1':\n"
        "        return 1\n"
        "    elif fam in ('Attention_v1', 'Attention_v2'):\n"
        "        return 2\n"
        "    elif fam.startswith('RMSNorm'):\n"
        "        return 3\n"
        "    elif fam == 'SiluMul_v1' or fam == 'SiluMul_v2':\n"
        "        return 4\n"
        "    return 0\n"
    )
    got = {(v.kind, v.detail) for v in file_violations(bad, "verity_vllm/check/x.py")}
    assert got == {("if-ladder", "fam"), ("rng", "numpy.random.default_rng"), ("rng", "random.Random")}, got
    assert file_violations(bad, "verity_vllm/program/sampling_rng.py") == [
        v for v in file_violations(bad, "verity_vllm/program/sampling_rng.py") if v.kind == "rng"]
    siblings = "def g(family):\n" + "".join(f"    if family == 'F{i}_v1':\n        return {i}\n" for i in range(LADDER_MIN))
    assert [(v.kind, v.symbol, v.detail) for v in file_violations(siblings, "verity_vllm/check/x.py")] == [
        ("if-ladder", "g", "family")]
    good = "def f(kind):\n    if kind == 'a':\n        return 1\n    elif kind == 'b':\n        return 2\n    return BACKENDS[kind]()\n"
    assert file_violations(good, "verity_vllm/check/x.py") == []
