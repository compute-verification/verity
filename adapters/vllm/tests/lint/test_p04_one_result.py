"""P4: every check returns one result type, and one verdict reads it.

Checked today, against ``allowlists/p04_one_result.json``:

* ``verdict-import``: the rule's import contract, "``check.verdict`` imports only ``check.result`` and
  ``properties.record``", the verdict codes ``check.codes`` they carry (core's until the layout move), and the typed stage
  records the rules read (``records``, which imports nothing else in ``verity_vllm``) (``VERDICT_ALLOWED``).  Any other integration import made by the modules ``check/verdict.py``
  absorbs (``_imports.VERDICT_MODULES``: ``check.verdict``, ``check.commit_rules``, ``check.c2_rules``, ``check.gates``) is a violation.
  One entry per (importer, imported) module pair.
* ``g-identifier``: a ``G1`` to ``G8`` token in an identifier (names, attributes, defs, classes, arguments, keywords).
* ``g-literal``: a string literal in code (docstrings excepted) that starts with a ``G1`` to ``G8`` token, as in a
  gate-keyed dict or a reason prefix.  Keyed by the token, counted per function.
* ``reason-prefix``: ``startswith``/``endswith``/``in`` tests on a reason, detail or message field: prose used as a
  code.  The result code is ``CheckResult.code``, a ``VerificationCode``.
"""
from __future__ import annotations

import ast
import re
from functools import lru_cache

from . import _imports as I
from . import _ratchet as R

NAME = "p04_one_result"
RULE = "P4 (one result type, one verdict)"
VERDICT_ALLOWED: tuple[str, ...] = ("verity_vllm.check.result", "verity_vllm.properties.record", "verity_vllm.check.codes",
                                    "verity_vllm.records")
G_TOKEN = re.compile(r"(?<![A-Z0-9])G[1-8](?![0-9])")
G_LEADING = re.compile(r"^\W*(G[1-8])(?![0-9])")
REASONY = re.compile(r"(?i)(reason|\bwhy\b|detail|message|\bmsg\b|\bnote|explan|\bcause|diagnos)")


def verdict_violations() -> list[R.Violation]:
    hits = I.unique_edges(lambda e: e.src in I.VERDICT_MODULES and not I.in_group(e.dst, VERDICT_ALLOWED)
                          and e.dst not in I.VERDICT_MODULES)
    return [R.Violation(e.file, "verdict-import", "<module>", f"{e.src} -> {e.dst}", e.line) for _, e in sorted(hits.items())]


def _identifiers(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        return [node.attr]
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, ast.arg):
        return [node.arg]
    if isinstance(node, ast.keyword) and node.arg:
        return [node.arg]
    if isinstance(node, ast.alias):
        return [node.asname or node.name.split(".")[-1]]
    return []


def file_violations(src: str, rel: str) -> list[R.Violation]:
    tree = R.parse_src(src)
    docs = R.docstring_ids(tree)
    out: list[R.Violation] = []
    for node, symbol in R.walk(tree):
        for ident in _identifiers(node):
            if G_TOKEN.search(ident):
                out.append(R.Violation(rel, "g-identifier", symbol, ident, getattr(node, "lineno", 0)))
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
            m = G_LEADING.match(node.value)
            if m:
                out.append(R.Violation(rel, "g-literal", symbol, m.group(1), node.lineno))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("startswith", "endswith"):
            recv = R.text(node.func.value, 60)
            if REASONY.search(recv):
                out.append(R.Violation(rel, "reason-prefix", symbol, f"{recv}.{node.func.attr}", node.lineno))
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], (ast.In, ast.NotIn)) \
                and R.is_str(node.left) and not isinstance(node.comparators[0], (ast.Tuple, ast.List, ast.Set, ast.Dict)):
            recv = R.text(node.comparators[0], 60)
            if REASONY.search(recv):
                out.append(R.Violation(rel, "reason-prefix", symbol, f"'...' in {recv}", node.lineno))
    return out


@lru_cache(maxsize=None)
def scan() -> tuple[R.Violation, ...]:
    out = verdict_violations()
    for p in R.library_files():
        out += file_violations(R.source(p), R.rel(p))
    return tuple(out)


def test_no_new_violations() -> None:
    R.assert_no_new(NAME, scan(), RULE,
                    "Return CheckResult(name, code, evidence) with a VerificationCode, and read codes, not prose; "
                    "\"G1\"..\"G8\" are not identifiers or keys.")


def test_allowlist_has_no_stale_entries() -> None:
    R.assert_no_stale(NAME, scan(), RULE)


def test_detectors_on_examples() -> None:
    bad = (
        "def verdict(r, G3_ok):\n"
        "    '''G2 docstrings are prose.'''\n"
        "    gates = {'G1': 1, 'G4_bind': 2}\n"
        "    if r.reason.startswith('run root (first 8 bytes'):\n"
        "        return gates['G1']\n"
        "    return 'mismatch' in r['detail']\n"
    )
    got = sorted((v.kind, v.detail) for v in file_violations(bad, "verity_vllm/check/x.py"))
    assert got == [("g-identifier", "G3_ok"), ("g-literal", "G1"), ("g-literal", "G1"), ("g-literal", "G4"),
                   ("reason-prefix", "'...' in r['detail']"), ("reason-prefix", "r.reason.startswith")], got
    good = "LOG1 = 1\ndef f(code):\n    return code is CODES.CHALLENGE_MISMATCH and 'G12' != 'G'\n"
    assert file_violations(good, "verity_vllm/check/x.py") == []
