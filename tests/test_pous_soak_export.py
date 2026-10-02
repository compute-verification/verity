"""The honest-latency soak's plot data (`benchmarks/pous/soak_export.py`, `pous-soak/v1`): the survival interpolation its
vectors pin, the contract's ToR terms, and the spec's keys on a tiny synthetic chunk, its vectors reproduced from the file."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("verity_pous")

ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location("pous_soak_export", ROOT / "benchmarks" / "pous" / "soak_export.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


E = _module()

COLS = ("t_sched", "t_send0", "t_recv", "status", "device_ns", "srv_krx", "srv_urx", "srv_utx", "srv_s1", "srv_s2", "op", "nbytes")
T0 = 1_790_924_000_000_000_000


def _chunk(path: Path) -> Path:
    """Two runs of cell 1 (idle, 10 Hz, 64 KiB, tuned), one with timestamping and its twin: 400 answers each, every tenth a
    ping, the first second of each run inside the warm-up; one answer with no K_tx_resp."""
    rng = np.random.default_rng(7)
    rows, cell, run, ktx, s4, runs = [], [], [], [], [], []
    for j, ts in enumerate((True, False)):
        start = T0 + j * 60_000_000_000
        runs.append({"cell": 1, "pass": 0, "ts": ts, "start_ns": start, "t0_ns": start + 2_000_000_000,
                     "end_ns": start + 44_000_000_000, "start_utc": E._utc(start), "status": "done"})
        for i in range(400):
            t = start + 2_000_000_000 + i * 100_000_000
            op = E.OP_PING if i % 10 == 0 else E.OP_READ
            lat = int(rng.lognormal(math.log(80_000), 0.3))
            krx = t + 5_000 if ts else 0
            s0 = t + 10_000
            s1, s2 = (s0 + 100, s0 + 30_000) if op == E.OP_READ else (0, 0)
            s3 = (s2 or s0) + 4_000
            rows.append([t, t + 1_000, t + 1_000 + lat, 0, 25_000 if op == E.OP_READ else 0, krx, s0, s3, s1, s2, op, 65_536])
            k = s3 + 40_000 if ts and i != 123 else 0
            cell.append(1), run.append(j), ktx.append(k), s4.append(s3 + 45_000)
    steps = np.array([[T0 + k * 1_000_000_000, 0, T0 + k * 1_000_000_000, T0 + k * 1_000_000_000 + 700_000_000, 8, 8, 0, 8]
                      for k in range(120)], np.int64)
    n = 1200
    t_ns = T0 + np.arange(n, dtype=np.int64) * 100_000_000
    cpus = np.arange(80, 94)
    cpustat = np.zeros((n, 14, 10), np.int64)
    cpustat[:, :, 7] = np.arange(n)[:, None] // 50
    meta = {"runs": runs, "cells": E_cells(), "wrong": 0, "schedule_seed": "ab" * 32, "job": "test", "chunk_index": 0,
            "chunk_start_utc": E._utc(T0), "end_utc": E._utc(T0 + 120_000_000_000), "env": {"gpu_uuid": "GPU-7"}}
    np.savez(path, rec=np.array(rows, np.int64), rec_columns=np.array(COLS), rec_cell=np.array(cell, np.int16),
             rec_run=np.array(run, np.int32), rec_k_tx_resp=np.array(ktx, np.int64), rec_s4=np.array(s4, np.int64),
             steps=steps, meta_json=np.array(json.dumps(meta)), sample_t_ns=t_ns, sample_cpus=cpus,
             sample_lastcpu=np.tile([92, 90, 89], (n, 1)), sample_cpustat=cpustat,
             sample_softirq=np.cumsum(np.ones((n, 10, 14), np.int64), axis=0),
             sample_thread_ctxt=np.cumsum(np.ones((n, 3, 2), np.int64), axis=0),
             sample_clock_triple=np.stack([t_ns - 5, t_ns + 1_000, t_ns + 5], axis=1),
             sample_nvml=np.array([[t, 90, 0, 0, 0, 200_000, 50, 0, 0, 0] for t in t_ns[::10]], np.int64),
             sample_nvml_fields=np.array(["t_ns", "sm_util", "mem_util", "sm_mhz", "mem_mhz", "power_mw", "temp_c", "throttle",
                                          "pcie_tx_kbs", "pcie_rx_kbs"]),
             sample_gpus_t_ns=t_ns[::100], sample_gpus_util=np.tile([5, 0, 0, 0, 0, 0, 0, 90], (len(t_ns[::100]), 1)),
             sample_gpus_uuid=np.array([f"GPU-{k}" for k in range(8)]), sample_loadavg=np.ones((len(t_ns[::100]), 3)))
    return path


def E_cells():
    return [{"id": 18 * li + 6 * ri + 2 * si + ci, "load": load, "rate": rate, "kib": kib, "config": cfg}
            for li, load in enumerate(("idle", "decode", "prefill", "mixed")) for ri, rate in enumerate((10, 100, 1000))
            for si, kib in enumerate((64, 144, 384)) for ci, cfg in enumerate(("baseline", "tuned"))]


@pytest.fixture(scope="module")
def doc(tmp_path_factory):
    b = E.Builder()
    b.add(_chunk(tmp_path_factory.mktemp("soak") / "chunk-test.npz"))
    d = b.document(arts=["art:test"], commit="test", generated_at="2026-10-02T00:00:00Z", host="h", gpu="g", extra_notes=[])
    return json.loads(json.dumps(d, allow_nan=False))


def test_survival_interpolation_is_log_log_and_ends_at_max():
    t = list(E.T_US)
    s = [1.0] * 40 + [0.5, 0.25] + [0.0] * (len(t) - 42)
    assert E.survival_at(t, s, 11.0, 0.5) == 1.0
    assert E.survival_at(t, s, 11.0, t[40]) == 0.5
    mid = math.sqrt(t[40] * t[41])
    assert E.survival_at(t, s, 11.0, mid) == pytest.approx(math.sqrt(0.5 * 0.25), rel=1e-12)
    assert E.survival_at(t, s, 11.0, t[41] * 1.01) == 0.25
    assert E.survival_at(t, s, 11.0, 11.0) == 0.25
    assert E.survival_at(t, s, 11.0, 11.000001) == 0.0
    assert len(t) == 201 and t[0] == 1.0 and t[-1] == pytest.approx(1e5)


def test_tor_terms_reproduce_the_contracts_serialization_table():
    for b, pcie, w400, w100 in ((65_536, 2.6, 1.4, 5.6), (147_456, 5.9, 3.1, 12.5), (393_216, 15.7, 8.4, 33.4)):
        r = E.HEADER_BYTES + b
        assert r / E.B_PCIE * 1e6 == pytest.approx(pcie, abs=0.05)
        assert E.tor_terms("tor_400_tuned", r)["t_nic_tx_and_wire_us"] == pytest.approx(1 + max(pcie, w400), abs=0.06)
        assert E.tor_terms("tor_100_default", r)["t_nic_tx_and_wire_us"] == pytest.approx(1 + max(pcie, w100), abs=0.06)
    assert E.tor_terms("tor_400_default", 1000)["t_nic_rx_us"] == 10.0
    assert E.tor_terms("tor_400_tuned", 1000)["t_nic_rx_us"] == 2.0


def test_the_document_has_the_specs_keys(doc):
    assert set(doc) == {"schema", "generated_at_utc", "source_commit", "run", "cells", "metrics", "t_us", "survival", "quantiles",
                        "thresholds", "breakdown", "timeline", "serving", "stalls", "vectors"}
    assert doc["schema"] == "pous-soak/v1"
    assert set(doc["run"]) == {"host", "gpu", "start_utc", "end_utc", "chunks", "answers", "wrong", "schedule_seed", "notes"}
    assert len(doc["cells"]) == 72 and set(doc["cells"][0]) == {"cell", "load", "rate_hz", "block_bytes", "config", "runs",
                                                                "answers", "challenges", "pings", "twin_runs", "twin_answers"}
    c1 = doc["cells"][1]
    assert (c1["runs"], c1["twin_runs"], c1["challenges"] + c1["pings"], c1["twin_answers"]) == (1, 1, 390, 390)
    assert {m["id"] for m in doc["metrics"]} == {"elapsed", "server_user", "server_kernel", "gpu_span", "schedule_lag",
                                                 "tor_400_tuned", "tor_400_default", "tor_100_tuned", "tor_100_default"}
    assert all(set(m) == {"id", "label", "kind", "formula", "params"} for m in doc["metrics"])
    assert len(doc["t_us"]) == 201
    for r in doc["survival"]:
        assert set(r) == {"cell", "metric", "op", "timestamping", "n", "max_us", "s"} and len(r["s"]) == 201
        assert all(a >= b for a, b in zip(r["s"], r["s"][1:], strict=False))
    keys = {(r["metric"], r["op"], r["timestamping"]) for r in doc["survival"]}
    assert ("server_kernel", "challenge", False) not in keys and ("elapsed", "challenge", False) in keys
    assert ("gpu_span", "ping", True) not in keys
    kern = next(r for r in doc["survival"] if (r["metric"], r["op"], r["timestamping"]) == ("server_kernel", "challenge", True))
    assert kern["n"] == 350
    for q in doc["quantiles"]:
        assert set(q) == {"cell", "metric", "op", "timestamping", "n", "p50_us", "p90_us", "p99_us", "p999_us", "p9999_us",
                          "max_us"}
        assert q["p9999_us"] is None
    t = doc["thresholds"][0]
    assert (t["block_bytes"], t["call_ns"]) == (65_536, 241_500)
    assert [r["D"] for r in t["rounds"]] == [0, 1, 2, 3, 4, 5] and t["rounds"][1]["tau_us_at_m1"] == pytest.approx(482.999)
    assert {"cell", "op", "components"} == set(doc["breakdown"][0])
    assert set(doc["breakdown"][0]["components"][0]) == {"id", "label", "formula", "p50_us", "p999_us", "mean_us"}
    run = doc["timeline"]["runs"][0]
    assert set(run) == {"run", "cell", "pass", "start_utc", "timestamping", "n", "p50_us", "p999_us", "max_us", "over_tau"}
    assert set(run["over_tau"]) == {"elapsed", "tor_400_tuned"} and "65536/d0" in run["over_tau"]["elapsed"]
    assert doc["timeline"]["runs"][1]["over_tau"].keys() == {"elapsed"}
    minute = doc["timeline"]["minutes"][0]
    assert set(minute) == {"utc", "steal_ms_per_s", "softirq_per_s", "ctx_switch_per_s", "gpu7_sm_util_pct", "gpu7_power_w",
                           "gpu7_temp_c", "other_gpus_util_pct_max", "loadavg1", "clock_offset_range_ns"}
    assert set(minute["steal_ms_per_s"]) == {"verifier", "responder", "max_other"}
    assert minute["other_gpus_util_pct_max"] == 5 and minute["gpu7_power_w"] == 200.0
    assert set(doc["serving"][0]) == {"cell", "tokens_per_s", "steps_per_s", "step_gpu_ms_p50", "step_gpu_ms_p99"}
    assert doc["serving"][0]["step_gpu_ms_p50"] == pytest.approx(700.0)
    assert len(doc["stalls"]) == 200 and [s["rank"] for s in doc["stalls"][:3]] == [1, 2, 3]
    assert set(doc["stalls"][0]) == {"rank", "utc", "cell", "op", "elapsed_us", "server_kernel_us", "components", "steal_ms",
                                     "vllm_step_overlap", "softirq", "attribution"}
    assert all(s["attribution"] in {"steal", "vllm-step", "gpu", "socket", "unknown"} for s in doc["stalls"])
    els = [s["elapsed_us"] for s in doc["stalls"]]
    assert els == sorted(els, reverse=True)


def test_the_vectors_reproduce_from_the_file_alone(doc):
    rows = {(r["cell"], r["metric"], r["op"], r["timestamping"]): r for r in doc["survival"]}
    assert len(doc["vectors"]) >= 30
    for v in doc["vectors"]:
        i = v["inputs"]
        r = rows[(i["cell"], i["metric"], i["op"], i["timestamping"])]
        got = E.survival_at(doc["t_us"], r["s"], r["max_us"], i["t_us"])
        assert got == pytest.approx(v["expected"]["survival_fraction"], rel=v["tolerance"]["relative"], abs=0)
    assert {v["expected"]["survival_fraction"] for v in doc["vectors"]} >= {0.0, 1.0}
