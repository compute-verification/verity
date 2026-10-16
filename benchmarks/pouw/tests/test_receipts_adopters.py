"""The receipt folds' implementations against the one reference (receipts/receipt_conform.py): pfill's twin and header, the
filler scorer's fold table, and E2E's PouwSegCheck word twins. Each test skips until its implementation is in the tree."""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import pytest

POUW = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POUW / "receipts"))

import receipt_conform as C  # noqa: E402


def _load(path: Path, name: str):
    if not path.exists():
        pytest.skip(f"{path.relative_to(POUW)} isn't in this tree")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_pfill_ref_runs_the_reference_folds():
    assert C.pfill_ref(_load(POUW / "decode_capacity" / "pfill_ref.py", "pfill_ref")) > 100


def test_pfill_cuh_folds_with_the_reference_header():
    path = POUW / "decode_capacity" / "pfill.cuh"
    if not path.exists():
        pytest.skip("pfill.cuh isn't in this tree")
    src = path.read_text()
    assert '#include "../receipts/receipt_folds.cuh"' in src
    absorb = re.search(r"void pf_absorb\(.*?\n}\n", src, re.S)[0]
    for fn in ("rf_ram_step", "rf_xin13_step", "rf_li_step", "rf_duo_step"):
        assert fn in absorb, f"pf_absorb doesn't call {fn}"


def test_the_scorers_fold_table_is_the_references():
    sc = _load(POUW / "filler" / "scoring.py", "scoring")
    assert C.scorer_folds(sc.FOLDS, sc.RECEIPT_TAGS) == 5


def test_e2e_segment_runs_the_served_schedules():
    """PouwSegCheck's word twins, and its fold Definitions (PouwDuoAbsorb_v2, PouwXinAbsorb_v2, their finals) on bits."""
    sg = pytest.importorskip("verity.core.protocols.pouw.circuit.segment")
    assert C.e2e_segment(sg) > 10
    assert C.e2e_definitions(sg) == 9
