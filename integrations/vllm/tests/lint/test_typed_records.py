"""Typed records: a stage record is read through its ``verity_vllm.records`` type, not as a dict.

Checked today, against ``allowlists/typed_records.json``:

* ``or-empty``: ``x.get(k) or []`` and ``x.get(k) or {}``, counted per package (a directory of ``verity_vllm``), which reads
  a missing field and an empty one alike.  A record's ``load()`` refuses a missing field by name and types an absent
  optional one, so its readers need neither.
* ``schema-literal``: a schema id (``verity-vllm/workload/v2``, ``cb-a/global-match/v1``) written as a string literal
  outside ``verity_vllm/records/``, where each record's ``SCHEMA`` is its one owner.  A module-level constant named
  ``*_TAG`` or ``*_DOMAIN`` (``DOMAIN_TAG = "verity-vllm/step-rows/v1"``) is a hash's domain-separation tag, which reads
  like a schema id but names no record, so its literal is skipped: a tag stays a plain literal, its bytes in one place.
  The stale check still sees the tags listed before the skip, until a second change deletes their entries: the ratchet
  gate judges a lint change by the lint its base has, so a narrowing lands in two steps.

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
TAG_NAME = re.compile(r"^_?[A-Z][A-Z0-9_]*_(?:TAG|DOMAIN)$")


def _is_get(node: ast.AST) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get"


def _is_empty(node: ast.AST) -> bool:
    return (isinstance(node, ast.List) and not node.elts) or (isinstance(node, ast.Dict) and not node.keys)


def package(rel: str) -> str:
    return PurePosixPath(rel).parent.as_posix()


def tag_ids(tree: ast.Module) -> frozenset[int]:
    """ids of the strings that module-level constants named ``*_TAG`` or ``*_DOMAIN`` hold: domain-separation tags."""
    return frozenset(id(s.value) for s in tree.body if isinstance(s, (ast.Assign, ast.AnnAssign)) and R.is_str(s.value)
                     and all(isinstance(t, ast.Name) and TAG_NAME.match(t.id)
                             for t in (s.targets if isinstance(s, ast.Assign) else [s.target])))


def file_violations(src: str, rel: str, tags: bool = False) -> list[R.Violation]:
    """The violations in ``src``; with ``tags``, a domain-separation tag's literal too (the stale check's view)."""
    tree = R.parse_src(src)
    skip = R.docstring_ids(tree) | (frozenset() if tags else tag_ids(tree))
    out: list[R.Violation] = []
    for node, symbol in R.walk(tree):
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.Or):
            out += [R.Violation(package(rel), "or-empty", "<package>", OR_EMPTY)
                    for a, b in zip(node.values, node.values[1:]) if _is_get(a) and _is_empty(b)]
        elif R.is_str(node) and id(node) not in skip and SCHEMA_ID.match(node.value) and not rel.startswith(OWNER):
            out.append(R.Violation(rel, "schema-literal", symbol, node.value, node.lineno))
    return out


@lru_cache(maxsize=None)
def scan(tags: bool = False) -> tuple[R.Violation, ...]:
    out: list[R.Violation] = []
    for p in R.library_files():
        out += file_violations(R.source(p), R.rel(p), tags)
    return tuple(out)


def test_no_new_violations() -> None:
    R.assert_no_new(NAME, scan(), RULE,
                    "Read the record through its type in verity_vllm/records/ (load() refuses a missing field by name; an "
                    "optional field is typed `X | None`), and take its schema id from the record's SCHEMA.  An or-empty "
                    "entry counts a package: the new one is in the lines this change adds there.")


def test_allowlist_has_no_stale_entries() -> None:
    R.assert_no_stale(NAME, scan(tags=True), RULE)


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


def test_a_domain_separation_tag_is_skipped_and_a_record_schema_id_still_is_not() -> None:
    src = ("DOMAIN_TAG = 'verity-vllm/step-rows/v1'\n"
           "_WEIGHTS_TAG: str = 'verity-vllm/weights-rows/v1'\n"
           "LEAF_DOMAIN = 'verity-vllm/leaf/v1'\n"
           "SCHEMA = 'verity-vllm/workload/v2'\n"
           "TAGS = 'verity-vllm/tags/v1'\n"
           "A_TAG, B = 'verity-vllm/a/v1', 'verity-vllm/b/v1'\n"
           "def f():\n"
           "    ROW_TAG = 'verity-vllm/row/v1'\n")
    got = [(v.symbol, v.detail) for v in file_violations(src, "verity_vllm/commit/x.py")]
    assert got == [("<module>", "verity-vllm/workload/v2"), ("<module>", "verity-vllm/tags/v1"), ("<module>", "verity-vllm/a/v1"),
                   ("<module>", "verity-vllm/b/v1"), ("f", "verity-vllm/row/v1")], got
