"""`zk_cost.py overhead` divides by `verity.hardware.rate`, the one peak calculator: on the H200 an e4m3 block's peak is
132 SMs x 4096 MACs at the 1830 MHz tensor clock, the data sheet's 1,979 dense FP8 TFLOPS, and never the 1980 MHz SM clock's
1.0705e15. Toy records of one served type, staged at two sizes and proved warm by the dev arm, give a known marginal second
a block, so its h is fixed by the peak alone."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

from verity import hardware

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import zk_cost as Z

K, FIXED_S, MARGINAL_S = 6144, 2.0, 1e-3


def _records(out: Path, sizes: tuple[int, ...] = (1024, 2048)) -> None:
    rec = out / "records"
    rec.mkdir(parents=True)
    stage = [{"unit": f"fp8-coord:K={K}", "tag": f"fp8-coord-k{K}", "c": c, "ands": 1, "k_log": 24, "full": max(sizes),
              "definition": f"Fp8Coordinate_v1{{K={K}}}", "macs": K, "format": "e4m3", "stage_s": 1.0} for c in sizes]
    lives = lambda c: [{"warm": w, "e2e_s": FIXED_S + MARGINAL_S * c, "prove_total_s": 1.0, "witness_s": None,  # noqa: E731
                        "dense_m": 1} for w in (True, False)]
    prove = [{"unit": f"fp8-coord:K={K}", "tag": f"fp8-coord-k{K}", "arm": "dev", "mode": "warm", "c": c, "k": c,
              "accepted": True, "lives": lives(c), "peak_rss_kb": 1} for c in sizes]
    (rec / "stage.jsonl").write_text("".join(json.dumps(r) + "\n" for r in stage))
    (rec / "prove.jsonl").write_text("".join(json.dumps(r) + "\n" for r in prove))
    (rec / "verify.jsonl").write_text("")
    (rec / "gpu.csv").write_text("name,uuid,driver,memory,compute_cap\nNVIDIA H200,GPU-0,595.91.07,143771 MiB,9.0\n")


def test_the_h200_e4m3_peak_is_the_data_sheets_at_the_tensor_clock():
    card = hardware.load("h200-sxm-141gb")
    peak = hardware.rate(card, fmt="e4m3")
    assert peak == 132 * 4096 * 1830 * 10**6
    assert float(peak) == pytest.approx(9.8945e14, rel=1e-4)
    assert hardware.rate(card, fmt="e4m3", mhz=1980) == 132 * 4096 * 1980 * 10**6


def test_overhead_divides_a_blocks_seconds_by_the_cards_rate(tmp_path):
    _records(tmp_path)
    assert Z.overhead(argparse.Namespace(out=str(tmp_path), card="h200-sxm-141gb", run="r-test")) == 0
    (t,) = json.loads((tmp_path / "overhead.json").read_text())["types"].values()
    peak = float(hardware.rate(hardware.load("h200-sxm-141gb"), fmt="e4m3"))
    assert t["peak_mac_per_s"] == peak
    dev = t["arms"]["dev"]
    assert dev["marginal_s"] == pytest.approx(MARGINAL_S) and dev["fixed_s"] == pytest.approx(FIXED_S)
    assert dev["overhead_marginal"] == pytest.approx(MARGINAL_S / (K / 9.8945e14), rel=1e-4)
    assert dev["overhead_marginal"] == pytest.approx(MARGINAL_S * peak / K, rel=1e-9)


def test_overhead_refuses_a_card_that_isnt_the_runs_gpu(tmp_path):
    _records(tmp_path)
    with pytest.raises(SystemExit, match="ZK_CARD"):
        Z.overhead(argparse.Namespace(out=str(tmp_path), card="rtx-pro-6000-server", run="r-test"))
