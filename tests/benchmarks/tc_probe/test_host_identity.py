import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

TC_PROBE = Path(__file__).resolve().parents[3] / "benchmarks" / "tc_probe"
if str(TC_PROBE) not in sys.path:
    sys.path.insert(0, str(TC_PROBE))

import tc_probe  # noqa: E402


@pytest.mark.parametrize(("visible", "asked"), [("3", "3"), ("5,6", "5"), ("GPU-9c8c", "GPU-9c8c"), ("", "0"), (None, "0")])
def test_the_device_record_names_the_card_the_process_holds(monkeypatch, visible, asked):
    seen = []

    def run(argv, **_):
        seen.append(argv[argv.index("-i") + 1])
        return SimpleNamespace(returncode=0, stdout="NVIDIA H200, 595.91.07, 9.0, GPU-9c8c\n")

    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.setattr(subprocess, "run", run)
    assert tc_probe.host_identity()["gpu_uuid"] == "GPU-9c8c" and seen == [asked]
