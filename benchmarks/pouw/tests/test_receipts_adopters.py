"""The receipt folds' implementations against the one reference (receipts/receipt_conform.py): pfill's twin and pfill.cuh
itself, the filler scorer's fold table, and E2E's PouwSegCheck word twins. Each test skips until its implementation is in
the tree."""
from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

POUW = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(POUW / "receipts"))

import receipt_conform as C  # noqa: E402


def _find(name: str) -> Path:
    """One of pfill's or the scorer's files: in decode_capacity/ or filler/, whichever this tree keeps it in."""
    for d in ("decode_capacity", "filler"):
        if (POUW / d / name).exists():
            return POUW / d / name
    pytest.skip(f"{name} isn't in this tree")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_pfill_ref_runs_the_reference_folds():
    assert C.pfill_ref(_load(_find("pfill_ref.py"), "pfill_ref")) > 100


def test_pfill_cuh_runs_the_reference_folds():
    """pf_absorb, pfill's key rules, pf_keys and its lane starts, in its default, LI and XIN13 builds (g++)."""
    path = _find("pfill.cuh")
    if not shutil.which("g++"):
        pytest.skip("no g++ for pfill.cuh's host build")
    assert C.pfill_cuh(path) > 500


def test_the_scorers_fold_table_is_the_references():
    sc = _load(_find("scoring.py"), "scoring")
    assert C.scorer_folds(sc.FOLDS, sc.RECEIPT_TAGS) == 5


def test_e2e_segment_runs_the_served_schedules():
    """PouwSegCheck's word twins, and its fold Definitions (PouwDuoAbsorb_v2, PouwXinAbsorb_v2, their finals) on bits."""
    sg = pytest.importorskip("verity.ml.protocols.pouw.circuit.segment")
    assert C.e2e_segment(sg) > 10
    assert C.e2e_definitions(sg) == 9
