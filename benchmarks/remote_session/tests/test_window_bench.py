"""`window_bench.py` at a toy size, so the synthetic window it times stays one both verifiers check as a prover's."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2]))

import window_bench  # noqa: E402


@pytest.mark.slow
def test_the_synthetic_window_is_accepted_by_both_verifiers_and_its_flipped_block_rejected_at_its_draw(tmp_path):
    assert window_bench.main(["--out", str(tmp_path), "--blocks-log2", "8", "--passes", "3", "--samples", "40",
                              "--codec", "p512/k2", "--workers", "2", "--chunks", "8,S", "--subtree-log2", "5"]) == 0
    out = json.loads((tmp_path / "window_bench.json").read_text())
    assert out["agree"] and out["frames"] == 1
    assert [(r["case"], r["chunk"], r["disagreements"]) for r in out["lean"]] == [
        ("negative", 8, []), ("honest", 8, []), ("honest", 40, [])]
    assert out["python_negative"]["failure"] == "payload-path"
    assert out["python_pool"]["verdict"] == {"accepted": 0}
