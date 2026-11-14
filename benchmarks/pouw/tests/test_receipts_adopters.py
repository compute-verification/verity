"""The receipt folds' implementations against the one reference (the receipts' receipt_conform.py): pfill's twin and
pfill.cuh itself, the filler scorer's fold table, and E2E's PouwSegCheck word twins. pfill.cuh's test skips where there
is no g++, and E2E's until its module is in the tree."""
from __future__ import annotations

import shutil
from pathlib import Path

import pfill_ref
import pytest
import scoring

from verity.protocols.accounting.pouw.verifier.receipts import receipt_conform as C

POUW = Path(__file__).resolve().parents[1]


def test_pfill_ref_runs_the_reference_folds():
    assert C.pfill_ref(pfill_ref) > 100


def test_pfill_cuh_runs_the_reference_folds():
    """pf_absorb, pfill's key rules, pf_keys and its lane starts, in its default, LI and XIN13 builds (g++)."""
    if not shutil.which("g++"):
        pytest.skip("no g++ for pfill.cuh's host build")
    assert C.pfill_cuh(POUW / "filler/pfill.cuh") > 500


def test_the_scorers_fold_table_is_the_references():
    assert C.scorer_folds(scoring.FOLDS, scoring.RECEIPT_TAGS) == 5


def test_e2e_segment_runs_the_served_schedules():
    """PouwSegCheck's word twins, and its fold Definitions (PouwDuoAbsorb_v2, PouwXinAbsorb_v2, their finals) on bits."""
    sg = pytest.importorskip("verity.protocols.accounting.pouw.prover.circuit.segment")
    assert C.e2e_segment(sg) > 10
    assert C.e2e_definitions(sg) == 9
