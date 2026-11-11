"""P5: property logic lives in ``properties/``, and runs cite property records by digest.

The rule's import contract is "``check`` and ``pipeline`` import only ``properties.record`` and
``properties.check_properties``" (``ALLOWED_ENTRY_POINTS``).  Any other import of a ``properties`` module
(``_imports.PROPERTY_MODULES``) by a module in layer ``check|properties`` or ``pipeline`` (``properties`` itself
excepted) is a violation, keyed by (importer, imported) module pair in ``allowlists/p05_properties.json``.

The rule's registry test (every property the verdict cites exists in ``properties.record.REGISTRY``, by a digest
that recomputes) is ``tests/check/test_verdict.py``.  Once ``properties.check_properties`` exists, it joins
``ALLOWED_ENTRY_POINTS``.
"""
from __future__ import annotations

from functools import lru_cache

from . import _imports as I
from . import _ratchet as R

NAME = "p05_properties"
RULE = "P5 (property logic lives in properties/)"
ALLOWED_ENTRY_POINTS: tuple[str, ...] = ("verity_vllm.properties.record",)


def importer(src: str) -> bool:
    return I.layer_of(src) in ("check|properties", "pipeline") and not I.in_group(src, I.PROPERTY_MODULES)


@lru_cache(maxsize=None)
def scan() -> tuple[R.Violation, ...]:
    hits = I.unique_edges(lambda e: importer(e.src) and I.in_group(e.dst, I.PROPERTY_MODULES)
                          and not I.in_group(e.dst, ALLOWED_ENTRY_POINTS))
    return tuple(R.Violation(e.file, "import", "<module>", f"{e.src} -> {e.dst}", e.line) for _, e in sorted(hits.items()))


def test_no_new_violations() -> None:
    R.assert_no_new(NAME, scan(), RULE,
                    "Per-run checks and drivers cite property records; they do not call property harnesses.")


def test_allowlist_has_no_stale_entries() -> None:
    R.assert_no_stale(NAME, scan(), RULE)


def test_contract_modules_exist() -> None:
    known = I.modules()
    assert [m for m in I.PROPERTY_MODULES if m not in known] == []
    assert importer("verity_vllm.pipeline.commit") and importer("verity_vllm.check.match.global_match")
    assert not importer("verity_vllm.properties.census") and not importer("verity_vllm.commit.binding")
