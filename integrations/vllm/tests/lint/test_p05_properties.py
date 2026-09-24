"""P5: property logic lives in ``properties/``, and runs cite property records by digest.

The rule's import contract is "``check`` and ``pipeline`` import only ``properties.record`` and
``properties.check_properties``".  ``properties/`` does not exist yet: its members are the property harnesses still in
``check/`` (``_imports.PROPERTY_MODULES``: ``noninterference``, ``census``, ``kernel_allowlist``, ``golden``,
``holdout``, ``difftest``, ``quarantine_lint``, ``protected``), and neither allowed entry point exists.  So today every
import of a property module by a module bound for ``check`` or ``pipeline`` (interim layers ``check|properties`` and
``pipeline``, property modules themselves excepted) is a violation, keyed by (importer, imported) module pair in
``allowlists/p05_properties.json``.

Once ``properties/`` exists, the imported side becomes ``verity_vllm.properties`` minus ``properties.record`` and
``properties.check_properties``, and the rule's registry test (every property the verdict names exists in
``properties.REGISTRY``) applies.
"""
from __future__ import annotations

from functools import lru_cache

from . import _imports as I
from . import _ratchet as R

NAME = "p05_properties"
RULE = "P5 (property logic lives in properties/)"
ALLOWED_ENTRY_POINTS: tuple[str, ...] = ()      # verity_vllm.properties.record, verity_vllm.properties.check_properties


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
    assert importer("verity_vllm.harness.commit_delta") and importer("verity_vllm.check.global_match")
    assert not importer("verity_vllm.check.census") and not importer("verity_vllm.commit.binding")
