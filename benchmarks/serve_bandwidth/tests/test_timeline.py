from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
US = 1000


def export(path: Path) -> None:
    """A synthetic nsys export of one GPU at 1000 GB/s peak, sampled every 100 us over a 10 ms trace: a prefill step
    [0, 4) ms (GEMM 3 ms at 20%, attention 1 ms at 50% of peak), host [4, 5) ms at 5%, a decode step [5, 8) ms (GEMM 1 ms
    at 80%, attention 1 ms at 60%, a norm 0.5 ms at 10%, 0.5 ms with no kernel at 0%), idle [8, 10) ms at 0%."""
    db = sqlite3.connect(path)
    db.executescript("""
        create table StringIds (id integer primary key, value text);
        create table TARGET_INFO_GPU (id integer, name text, busLocation text, memoryBandwidth integer);
        create table TARGET_INFO_GPU_METRICS (typeId integer, sourceId integer, typeName text, metricId integer,
                                              metricName text);
        create table GPU_METRICS (timestamp integer, typeId integer, metricId integer, value integer);
        create table NVTX_EVENTS (start integer, end integer, text text, textId integer);
        create table CUPTI_ACTIVITY_KIND_KERNEL (deviceId integer, start integer, end integer, demangledName integer);
    """)
    names = {1: "void cutlass::device_kernel<cutlass::gemm::kernel::GemmUniversal<x>>",
             2: "void flash::flash_fwd_splitkv_kernel<x>", 3: "void vllm::rms_norm_kernel<c10::BFloat16>"}
    db.executemany("insert into StringIds values (?, ?)", names.items())
    db.execute("insert into TARGET_INFO_GPU values (0, 'GB202', '0000:fe:00.0', ?)", (10**12,))
    db.executemany("insert into TARGET_INFO_GPU_METRICS values (0, 0, 'GB202', ?, ?)",
                   [(10, "DRAM Read Bandwidth [Throughput %]"), (11, "DRAM Write Bandwidth [Throughput %]")])
    kernels = [(0, 3000, 1), (3000, 4000, 2), (5000, 6000, 1), (6000, 7000, 2), (7000, 7500, 3)]
    db.executemany("insert into CUPTI_ACTIVITY_KIND_KERNEL values (0, ?, ?, ?)",
                   [(s * US, e * US, n) for s, e, n in kernels])
    db.executemany("insert into NVTX_EVENTS values (?, ?, ?, null)",
                   [(0, 10000 * US, "trace"), (0, 4000 * US, "s0 p512 d0 kv0"), (5000 * US, 8000 * US, "s1 p0 d2 kv100"),
                    (8000 * US, 10000 * US, "idle")])

    def pct(mid_us: float) -> int:
        for (s, e, n), p in zip(kernels, (20, 50, 80, 60, 10)):
            if s <= mid_us < e:
                return p
        return 5 if 4000 <= mid_us < 5000 else 0

    rows = []
    for i in range(1, 101):
        t = i * 100
        rows += [(t * US, 0, 10, pct(t - 50)), (t * US, 0, 11, 1)]
    db.executemany("insert into GPU_METRICS values (?, ?, ?, ?)", rows)
    db.commit()


@pytest.fixture
def result(tmp_path: Path) -> dict:
    export(tmp_path / "x.sqlite")
    (tmp_path / "mix.json").write_text(json.dumps({"weight_bytes_per_step": 10**8, "kv_bytes_per_token": 10**5,
                                                   "steps": []}))
    subprocess.run([sys.executable, str(HERE / "timeline.py"), str(tmp_path / "x.sqlite"), "--mix",
                    str(tmp_path / "mix.json"), "--out", str(tmp_path), "--R", "1000"], check=True)
    return json.loads((tmp_path / "timeline.json").read_text())


def test_each_phase_reads_what_its_samples_carry(result):
    (d,) = result["devices"]
    ph = d["phases"]
    assert ph["prefill"]["seconds"] == pytest.approx(0.004)
    assert ph["prefill"]["read_GB"] == pytest.approx(1.1)
    assert ph["prefill"]["read_of_R"] == pytest.approx(0.275)
    assert ph["prefill"]["idle_read_of_R"] == pytest.approx(0.725)
    assert ph["decode"]["read_GB"] == pytest.approx(1.45)
    assert ph["decode"]["read_of_R"] == pytest.approx(1.45 / 3, abs=1e-4)
    assert ph["host"]["read_of_R"] == pytest.approx(0.05)
    assert ph["idle"]["seconds"] == pytest.approx(0.002)
    assert ph["idle"]["read_GB"] == 0
    assert ph["decode"]["write_of_R"] == pytest.approx(0.01)


def test_reads_split_over_the_kernel_classes_running(result):
    pre, dec = (result["devices"][0]["phases"][p]["by_class"] for p in ("prefill", "decode"))
    assert pre["gemm"]["read_GB"] == pytest.approx(0.6)
    assert pre["attention"]["read_GB"] == pytest.approx(0.5)
    assert pre["gemm"]["read_of_R_while_on"] == pytest.approx(0.2)
    assert dec["gemm"]["read_share"] == pytest.approx(0.8 / 1.45, abs=1e-4)
    assert dec["other"]["read_GB"] == pytest.approx(0.05)
    assert dec["none"]["seconds"] == pytest.approx(0.0005)


def test_modeled_registered_bytes_against_measured_reads(result):
    pre, dec = (result["devices"][0]["phases"][p] for p in ("prefill", "decode"))
    assert pre["steps"] == dec["steps"] == 1
    assert pre["weight_share_modeled"] == pytest.approx(0.1 / 1.1, abs=1e-4)
    assert dec["kv_share_modeled"] == pytest.approx(0.01 / 1.45, abs=1e-4)
    by_kind = result["devices"][0]["by_step_kind"]
    assert by_kind["decode 1-4 reqs"]["read_of_R"] == pytest.approx(1.45 / 3, abs=1e-4)
    assert by_kind["prefill <=512 tokens"]["steps"] == 1
