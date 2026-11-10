"""P2: value checks read opened values only.

The rule's import contract is "``check.replay`` may not import ``acquire`` or ``commit.committer``".  It applies to
``check/replay/`` and, until they are one determinism module, the determinism value checks
(``_imports.VALUE_CHECK_MODULES``: ``check.replay``, ``check.value_check``, ``check.compiled_value_check``,
``check.oracle_compare``), and it forbids ``verity_vllm.acquire`` and ``verity_vllm.commit.committer``.  Violations
are keyed by (importer, imported) module pair in ``allowlists/p02_value_checks.json``.

A committer object passed in at run time (``committed_reader(com)``) is invisible to an import contract; a pod negative
test (mutate the retained buffer after commit, and the value check must FAIL) covers that half.
"""
from __future__ import annotations

from functools import lru_cache

import pytest

from . import _imports as I
from . import _ratchet as R

NAME = "p02_value_checks"
RULE = "P2 (value checks read opened values only)"


def forbidden(dst: str) -> bool:
    return I.in_group(dst, ("verity_vllm.acquire", "verity_vllm.commit.committer"))


@lru_cache(maxsize=None)
def scan() -> tuple[R.Violation, ...]:
    hits = I.unique_edges(lambda e: I.in_group(e.src, I.VALUE_CHECK_MODULES) and forbidden(e.dst))
    return tuple(R.Violation(e.file, "import", "<module>", f"{e.src} -> {e.dst}", e.line) for _, e in sorted(hits.items()))


@pytest.mark.slow
def test_no_new_violations() -> None:
    R.assert_no_new(NAME, scan(), RULE,
                    "A value check reads opened values (the commit directory and its openings), never acquisition or "
                    "committer state.")


def test_allowlist_has_no_stale_entries() -> None:
    R.assert_no_stale(NAME, scan(), RULE)


def test_contract_modules_exist() -> None:
    known = I.modules()
    assert [m for m in I.VALUE_CHECK_MODULES if m not in known] == []
    assert forbidden("verity_vllm.commit.committer.native_host") and forbidden("verity_vllm.acquire.sources.partial_source")
    assert not forbidden("verity_vllm.commit.binding")
