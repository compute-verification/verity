"""tensor_util.py on a synthetic nsys export: the steps cut at the int8 GEMMs, the wall swept by class, and the losses."""
from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("tensor_util", Path(__file__).resolve().parents[1] / "tensor_util.py")
TU = importlib.util.module_from_spec(_spec)
sys.modules["tensor_util"] = TU
_spec.loader.exec_module(TU)

# one layer: qkv (4 + 2 * 2) * 16 = 128 x 64, o 64 x 64, gate_up 256 x 64, down 64 x 128: 36,864 int8 MACs a token
CONFIG = {"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 1, "num_attention_heads": 4,
          "num_key_value_heads": 2, "head_dim": 16, "vocab_size": 100}
THETA = 8 * 36864 * 1e3  # the step's int8 MACs in 1 ms
MS = 1_000_000


def _export(path: Path, starts_ms: tuple[int, ...] = (0, 10, 20), quant: bool = True) -> None:
    """Three forwards of T 8 every 10 ms after a T 4 warm-up, each: the four linears (quantizer 0.1 ms, or nothing in
    its place without `quant`, then GEMM 0.5 ms), the rope (0.05 ms, one program a token), attention, a 0.05 ms launch
    gap and the activation between them, the LM head, a hashing kernel from 2.5 to 4.0 ms (under the down GEMM and the LM
    head until 3.1), a 1 ms copy, then nothing until the next forward.  And another device's kernel."""
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
    db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INT, end INT, deviceId INT, globalPid INT, demangledName INT,"
               " shortName INT, gridX INT, gridY INT, gridZ INT, streamId INT)")
    db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INT, end INT, bytes INT, deviceId INT, copyKind INT)")
    names = ["exactlin_quant_rows", "exactlin_scaled_mm", "void flash::FlashAttnFwdSm90<int>(int)",
             "triton_poi_fused_mul_silu_0", "nvjet_hsh_128x64_64x4_1x2_h_bz_TNT", "void rows_leaf_kernel<64>(int)",
             "void vllm::rotary_embedding_kernel<c10::BFloat16>(const long *)"]
    db.executemany("INSERT INTO StringIds VALUES (?, ?)", list(enumerate(names)))
    rows = []

    def k(s, e, name, grid=1, dev=0):
        rows.append((s, e, dev, 7 << 24, names.index(name), names.index(name), grid, 1, 1, 7))

    def forward(base: int, t: int) -> None:
        at = base
        for i in range(4):
            if quant:
                k(at, at + MS // 10, "exactlin_quant_rows", grid=t)
            k(at + MS // 10, at + 6 * MS // 10, "exactlin_scaled_mm", grid=16)
            at += 6 * MS // 10
            if i == 0:
                k(at, at + MS // 20, names[6], grid=t)
                k(at + MS // 20, at + MS // 4, names[2])
                at += 3 * MS // 10
            if i == 2:
                k(at, at + 2 * MS // 10, names[3])
                at += 2 * MS // 10
        k(at, at + 2 * MS // 10, names[4])

    forward(-20 * MS, 4)
    for start in starts_ms:
        base = start * MS
        forward(base, 8)
        k(base + 25 * MS // 10, base + 4 * MS, names[5])
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?, ?, ?, 0, 2)", (base + 4 * MS, base + 5 * MS, 1 << 30))
    k(0, MS, "exactlin_scaled_mm", dev=1)
    db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    # GPU metrics every 50 us, mid-interval: Tensor Active 60 while a GEMM runs, else 0; the clock 1900 MHz throughout,
    # in Hz as nsys stores it
    db.execute("CREATE TABLE TARGET_INFO_GPU_METRICS (metricId INT, metricName TEXT)")
    db.execute("CREATE TABLE GPU_METRICS (timestamp INT, metricId INT, value REAL)")
    db.executemany("INSERT INTO TARGET_INFO_GPU_METRICS VALUES (?, ?)",
                   [(1, "Tensor Active [Throughput %]"), (2, "GPC Clock Frequency [MHz]"), (3, "DRAM Read Bandwidth")])
    gemm = [(r[0], r[1]) for r in rows if names[r[4]] == "exactlin_scaled_mm" and r[2] == 0]
    for t in range(25_000, (starts_ms[-1] + 10) * MS, 50_000):
        db.execute("INSERT INTO GPU_METRICS VALUES (?, 1, ?)", (t, 60.0 if any(s <= t < e for s, e in gemm) else 0.0))
        db.execute("INSERT INTO GPU_METRICS VALUES (?, 2, 1900000000)", (t,))
    db.commit()


@pytest.fixture(scope="module")
def doc(tmp_path_factory):
    p = tmp_path_factory.mktemp("nsys") / "report.sqlite"
    _export(p)
    return TU.analyse(str(p), tokens=8, prompt=4, config=CONFIG, theta=THETA, target=0.8)


def test_steps_are_the_periods_between_forwards_of_the_named_T(doc):
    assert [f["tokens"] for f in doc["forwards"]] == [4, 8, 8, 8]
    assert doc["summary"]["steps"] == 2 and doc["device"] == 0
    assert [s["wall_ms"] for s in doc["steps"]] == [10.0, 10.0]


def test_the_sweep_gives_each_instant_to_one_class_and_the_rest_to_the_host():
    iv = [(0, 10, "Commit hashing"), (5, 8, "int8 GEMM"), (12, 14, "memcpy")]
    assert TU.sweep(iv, 0, 20) == {"Commit hashing": 7, "int8 GEMM": 3, "memcpy": 2, TU.HOST_GAP: 8}


def test_wall_by_class(doc):
    by = {r["class"]: r for r in doc["classes"]}
    want = {"int8 GEMM": 2.0, "attention": 0.2, "quantize": 0.4, "activation": 0.2, "bf16 GEMM (LM head)": 0.2,
            "rope": 0.05, "Commit hashing": 0.9, "memcpy": 1.0, TU.GAP_IN: 0.05, TU.GAP_OUT: 5.0}
    assert {c: round(r["exclusive_ms"], 6) for c, r in by.items()} == want
    assert by["Commit hashing"]["kernel_ms"] == pytest.approx(1.5)
    assert by["Commit hashing"]["overlap_with_tensor_ms"] == pytest.approx(0.6)
    assert by["int8 GEMM"]["launches"] == 4 and by["memcpy"]["gb"] == pytest.approx(2 ** 30 / 1e9)


def test_gpu_metrics_by_class(doc):
    by = {r["class"]: r for r in doc["classes"]}
    assert by["int8 GEMM"]["tensor_active"] == 60 and by[TU.GAP_OUT]["tensor_active"] == 0
    assert doc["summary"]["tensor_active"] == pytest.approx(60 * 2 / 10) and doc["summary"]["clock_mhz"] == 1900
    assert doc["summary"]["metric_period_us"] == 50
    assert "SM clock 1900 MHz, Tensor Active 12.0% (sampled every 50 µs" in TU.table(doc)


def test_utilization_and_losses(doc):
    s = doc["summary"]
    assert s["macs"] == {"int8": 8 * 36864, "attn": 2 * 2 * 4 * 16 * (16 - 6), "head": 2 * 64 * 100}
    ideal = 1.0 + 2560 / (THETA / 2) * 1e3 + 12800 / (THETA / 2) * 1e3
    assert s["ideal_ms"] == pytest.approx(ideal) and s["util"] == pytest.approx(ideal / 10)
    by = {r["class"]: r for r in doc["classes"]}
    assert by["int8 GEMM"]["class_util"] == pytest.approx(0.5) and by["int8 GEMM"]["loss_ms"] == pytest.approx(1.0)
    assert sum(r["loss_ms"] for r in doc["classes"]) == pytest.approx(10 - ideal)
    assert doc["classes"][0]["class"] == TU.GAP_OUT
    assert by[TU.GAP_OUT]["util_if_zero"] == pytest.approx(ideal / 5)
    assert by[TU.GAP_IN]["loss_ms"] == pytest.approx(0.05)
    assert "(11.0% at the data sheet's 1,830 MHz)" in TU.table(doc)


def test_rates_are_per_clock_at_the_measured_clock_first(doc):
    """The export's clock is 1,900 MHz against theta's 1,830: every rate scales by 1830 / 1900, and the table leads with
    it, the sheet clock's beside it; without GPU metrics the sheet clock's figure is the headline, labelled."""
    s, by, r = doc["summary"], {r["class"]: r for r in doc["classes"]}, 1830 / 1900
    assert s["util_at_clock"] == pytest.approx(s["util"] * r)
    assert by["int8 GEMM"]["class_util_at_clock"] == pytest.approx(0.5 * r)
    assert s["wall_at_target_at_clock_ms"] == pytest.approx(s["ideal_ms"] * r / 0.8)
    assert s["over_target_at_clock_ms"] == pytest.approx(10 - s["ideal_ms"] * r / 0.8)
    assert by[TU.GAP_OUT]["util_if_zero_at_clock"] == pytest.approx(by[TU.GAP_OUT]["util_if_zero"] * r)
    t = TU.table(doc)
    assert "**tensor utilization 10.6% at the measured SM clock of 1,900 MHz** (11.0% at the data sheet's 1,830 MHz)" in t
    assert "| class util at its SM clock (1,830 MHz) |" in t and "| 48.2% (50.0%) |" in t
    older = {**doc, "summary": {k: v for k, v in s.items() if "at_clock" not in k},
             "classes": [{k: v for k, v in c.items() if "at_clock" not in k} for c in doc["classes"]]}
    assert TU.table(older) == t
    bare = {**doc, "summary": {**s, "clock_mhz": None, "util_at_clock": None},
            "classes": [{**c, "clock_mhz": None, "tensor_active": None} for c in doc["classes"]]}
    t = TU.table(bare)
    assert "**tensor utilization 11.0% at the data sheet's 1,830 MHz** (no measured clock)" in t
    assert "| class util at 1,830 MHz |" in t and "| 50.0% |" in t


def test_steady_drops_a_step_past_one_and_a_half_medians(tmp_path):
    p = tmp_path / "report.sqlite"
    _export(p, starts_ms=(0, 10, 20, 50))
    doc = TU.analyse(str(p), tokens=8, prompt=4, config=CONFIG, theta=THETA, target=0.8, steady=True)
    assert [s["wall_ms"] for s in doc["steps"]] == [10.0, 10.0]
    assert doc["summary"]["dropped"][0]["wall_ms"] == 30.0 and doc["summary"]["wall_ms"] == 10.0
    assert "1 longer step(s) dropped (30 ms)" in TU.table(doc)


def test_a_forward_is_its_gemms_and_ropes_between_lm_heads():
    g = lambda s: (s, s + 1, "exactlin_scaled_mm_wgmma_pp_128x128", 132, 0, 0)  # noqa: E731
    r = lambda s, t: (s, s + 1, "rope_cache_kernel", t, 0, 0)  # noqa: E731
    h = lambda s: (s, s + 1, "matmul_kernel_persistent", 4, 0, 0)  # noqa: E731
    layer = lambda at, t: [g(at), r(at + 1, t), g(at + 2), g(at + 3), g(at + 4)]  # noqa: E731
    two = layer(0, 8) + layer(5, 8) + [h(10)] + layer(11, 4) + layer(16, 4) + [h(21)]
    assert TU.forwards(two, layers=2) == [(0, 8), (11, 4)]
    assert TU.forwards(layer(2, 8) + layer(7, 8) + layer(12, 8) + [h(17)], layers=2) == []
    assert TU.forwards(layer(0, 8) + layer(5, 4) + [h(10)], layers=2) == []


def test_the_cut_needs_no_quantizer_launch(tmp_path):
    """The norms and the activation quantize the linears' inputs themselves, so the forwards launch no quantizer."""
    p = tmp_path / "report.sqlite"
    _export(p, quant=False)
    doc = TU.analyse(str(p), tokens=8, prompt=4, config=CONFIG, theta=THETA, target=0.8)
    assert [f["tokens"] for f in doc["forwards"]] == [4, 8, 8, 8]
    assert [s["wall_ms"] for s in doc["steps"]] == [10.0, 10.0]
    assert "quantize" not in {r["class"] for r in doc["classes"]}
