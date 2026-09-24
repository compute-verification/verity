"""The pod health check without a GPU: the verdict against the SKU reference, the warnings, the facts-only report and its
rendering, and the DEGRADED exit code.  (The microbenchmark itself runs on a pod: `research pods health POD`.)

    uv run pytest backends/direct/ligero/pod_health_test.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from backends.direct.ligero import pod_health as H  # noqa: E402

REF = {"encode_ms": {"value": 2.0, "pod": "vy-a", "utc": "t"}, "matmul_tflops": {"value": 150.0, "pod": "vy-a", "utc": "t"}}
HOST = {"cpu_quota": 10.2, "nproc": 96, "omp_threads": "10", "load1": 3.0, "disk": "/workspace", "disk_free_gib": 180.0}
GPU = {"name": "NVIDIA GeForce RTX 4090", "clocks.sm": 2520.0, "clocks.max.sm": 3120.0, "clocks.mem": 10501.0, "clocks.max.mem": 10501.0,
       "power.limit": 450.0, "power.default_limit": 450.0, "pcie.link.gen.current": 4.0, "pcie.link.gen.max": 4.0,
       "pcie.link.width.current": 16.0, "pcie.link.width.max": 16.0, "persistence_mode": "Enabled", "temperature.gpu": 41.0}


def test_the_verdict_is_degraded_beyond_twenty_percent_on_either_metric() -> None:
    assert H.judge({"encode_ms": 2.3, "matmul_tflops": 140.0}, REF) == ("OK", {"encode": pytest.approx(1.15), "matmul": pytest.approx(150 / 140)})
    verdict, slow = H.judge({"encode_ms": 30.0, "matmul_tflops": 149.0}, REF)            # the 15x encoder pod
    assert verdict == "DEGRADED" and slow["encode"] == 15.0
    assert H.judge({"encode_ms": 2.0, "matmul_tflops": 120.0}, REF)[0] == "DEGRADED"      # 1.25x slower matmul
    assert H.judge({"encode_ms": 2.0, "matmul_tflops": 150.0}, None) == ("NO-REFERENCE", {})
    assert H.judge(None, REF) == ("NO-BENCH", {})


def test_warnings_name_what_would_skew_a_timing() -> None:
    assert H.warnings(HOST, GPU, []) == []
    bad = H.warnings({**HOST, "omp_threads": None, "load1": 120.0, "disk_free_gib": 5.0},
                     {**GPU, "power.limit": 300.0, "pcie.link.width.current": 8.0, "pcie.link.gen.current": 1.0}, ["4711 python 900"])
    assert bad == ["OMP_NUM_THREADS=unset but the cgroup quota is 10.2 cores (nproc 96): source /workspace/env.sh",
                   "power limit 300 W < default 450 W", "PCIe link x8 of x16", "PCIe gen 1 of 4 under load",
                   "1 other GPU compute process(es): 4711 python 900", "5.0 GiB free on /workspace", "load average 120.0 > 96 cores"]
    assert H.warnings({**HOST, "cpu_quota": None, "omp_threads": None}, None, None) == []    # no quota, no nvidia-smi: nothing to say


def test_a_facts_only_report_renders_and_a_degraded_one_exits_3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(H, "host", lambda: HOST)
    monkeypatch.setattr(H, "gpu_state", lambda: GPU)
    monkeypatch.setattr(H, "gpu_apps", lambda: [])
    ref = tmp_path / "ref.json"
    ref.write_text(json.dumps({"skus": {"rtx-4090": REF}}))
    rep = H.report(None, ref)
    assert (rep["sku"], rep["verdict"], rep["bench"], rep["ref"], rep["warnings"]) == ("rtx-4090", "NO-BENCH", None, REF, [])
    text = H.render(rep)
    assert "cpu quota 10.2 of 96 cores (OMP_NUM_THREADS=10)" in text and "SM 2520/3120 MHz" in text and "PCIe gen4 x16 (max gen4 x16)" in text
    assert "gpu processes: none" in text and text.endswith("NO-BENCH: facts only")

    monkeypatch.setattr(H, "bench", lambda s: ({"encode_ms": 2.6, "matmul_tflops": 151.0, "encode_calls": 9, "matmul_calls": 9}, GPU))
    rep = H.report(10.0, ref)
    assert rep["verdict"] == "DEGRADED" and rep["slowdown"] == {"encode": 1.3, "matmul": 0.993}
    assert "2.600 ms (ref 2.0: 1.30x)" in H.render(rep) and H.render(rep).splitlines()[-1].startswith("DEGRADED")
    monkeypatch.setattr(H, "report", lambda seconds, ref_path=H.REF: rep)
    assert H.main([]) == 3 and "replace this pod" in capsys.readouterr().out
