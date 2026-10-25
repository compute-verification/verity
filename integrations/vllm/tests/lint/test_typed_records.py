"""Typed records: a stage record is read through its ``verity_vllm.records`` type, not as a dict.

Checked today, against ``allowlists/typed_records.json``:

* ``or-empty``: ``x.get(k) or []`` and ``x.get(k) or {}``, counted per package (a directory of ``verity_vllm``), which reads
  a missing field and an empty one alike.  A record's ``load()`` refuses a missing field by name and types an absent
  optional one, so its readers need neither.
* ``schema-literal``: a schema id (``verity-vllm/workload/v2``, ``cb-a/global-match/v1``) written as a string literal
  outside ``verity_vllm/records/``, where each record's ``SCHEMA`` is its one owner.

Not linted: ``x.get(k, [])`` and ``x[k] if k in x else []`` (the same read, rarer), and subscripts of an untyped record.
"""
from __future__ import annotations

import ast
import re
from collections import Counter
from functools import lru_cache
from pathlib import PurePosixPath

from . import _ratchet as R

NAME = "typed_records"
RULE = "Typed records (read a record through its verity_vllm.records type)"
OWNER = "verity_vllm/records/"
OR_EMPTY = ".get(k) or [] / {}"
SCHEMA_ID = re.compile(r"^[a-z][a-z0-9-]*(?:/[a-z0-9][a-z0-9_.+-]*)+/v[0-9]+$")


def _is_get(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get"


def _is_empty(node: ast.AST) -> bool:
    return (isinstance(node, ast.List) and not node.elts) or (isinstance(node, ast.Dict) and not node.keys)


def package(rel: str) -> str:
    return PurePosixPath(rel).parent.as_posix()


def file_violations(src: str, rel: str) -> list[R.Violation]:
    tree = R.parse_src(src)
    docs = R.docstring_ids(tree)
    out: list[R.Violation] = []
    for node, symbol in R.walk(tree):
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
            out += [R.Violation(package(rel), "or-empty", "<package>", OR_EMPTY)
                    for a, b in zip(node.values, node.values[1:]) if _is_get(a) and _is_empty(b)]
        elif R.is_str(node) and id(node) not in docs and SCHEMA_ID.match(node.value) and not rel.startswith(OWNER):
            out.append(R.Violation(rel, "schema-literal", symbol, node.value, node.lineno))
    return out


@lru_cache(maxsize=None)
def scan() -> tuple[R.Violation, ...]:
    out: list[R.Violation] = []
    for p in R.library_files():
        out += file_violations(R.source(p), R.rel(p))
    return tuple(out)


def test_no_new_violations() -> None:
    R.assert_no_new(NAME, scan(), RULE,
                    "Read the record through its type in verity_vllm/records/ (load() refuses a missing field by name; an "
                    "optional field is typed `X | None`), and take its schema id from the record's SCHEMA.  An or-empty "
                    "entry counts a package: the new one is in the lines this change adds there.")


def test_allowlist_has_no_stale_entries() -> None:
    R.assert_no_stale(NAME, scan(), RULE)


def test_detectors_on_examples() -> None:
    bad = (
        "SCHEMA = 'verity-vllm/workload/v2'\n"
        "def f(doc, other):\n"
        "    '''Reads cb-a/global-match/v1.'''\n"
        "    if doc.get('schema') != 'cb-a/global-match/v1':\n"
        "        return [r for r in doc.get('requests') or []] + list((doc.get('x') or {}).items())\n"
        "    return doc.get('a') or other.get('b') or []\n"
    )
    got = Counter((v.file, v.kind, v.symbol, v.detail) for v in file_violations(bad, "verity_vllm/check/match/x.py"))
    assert got == Counter({("verity_vllm/check/match", "or-empty", "<package>", OR_EMPTY): 3,
                           ("verity_vllm/check/match/x.py", "schema-literal", "<module>", "verity-vllm/workload/v2"): 1,
                           ("verity_vllm/check/match/x.py", "schema-literal", "f", "cb-a/global-match/v1"): 1}), got
    good = ("def f(doc):\n"
            "    return doc.get('a') or None, doc.get('b', []), doc['c'] or [], 'a/b', 'cb-a/x/v1-old', 'verity_vllm/check/v1.py'\n")
    assert file_violations(good, "verity_vllm/check/x.py") == []
    assert [v.kind for v in file_violations(bad, "verity_vllm/records/x.py")] == ["or-empty"] * 3
