"""tensor_util.py on a synthetic nsys export: the steps cut at the quantizer, the wall swept by class, and the losses."""
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


def _export(path: Path) -> None:
    """Three forwards of T 8 every 10 ms after a T 4 warm-up, each: the four linears (quantizer 0.1 ms, GEMM 0.5 ms),
    attention and the activation between them, the LM head, a hashing kernel from 2.5 to 4.0 ms (under the down GEMM
    and the LM head until 3.0), a 1 ms copy, then nothing until the next forward.  And another device's kernel."""
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE StringIds (id INTEGER PRIMARY KEY, value TEXT)")
    db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL (start INT, end INT, deviceId INT, globalPid INT, demangledName INT,"
               " shortName INT, gridX INT, gridY INT, gridZ INT, streamId INT)")
    db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INT, end INT, bytes INT, deviceId INT, copyKind INT)")
    names = ["exactlin_quant_rows", "exactlin_scaled_mm", "void flash::FlashAttnFwdSm90<int>(int)",
             "triton_poi_fused_mul_silu_0", "nvjet_hsh_128x64_64x4_1x2_h_bz_TNT", "void rows_leaf_kernel<64>(int)"]
    db.executemany("INSERT INTO StringIds VALUES (?, ?)", list(enumerate(names)))
    rows = []

    def k(s, e, name, grid=1, dev=0):
        rows.append((s, e, dev, 7 << 24, names.index(name), names.index(name), grid, 1, 1, 7))

    def forward(base: int, t: int) -> None:
        at = base
        for i in range(4):
            k(at, at + MS // 10, "exactlin_quant_rows", grid=t)
            k(at + MS // 10, at + 6 * MS // 10, "exactlin_scaled_mm", grid=16)
            at += 6 * MS // 10
            if i == 0:
                k(at, at + 2 * MS // 10, names[2])
                at += 2 * MS // 10
            if i == 2:
                k(at, at + 2 * MS // 10, names[3])
                at += 2 * MS // 10
        k(at, at + 2 * MS // 10, names[4])

    forward(-20 * MS, 4)
    for i in range(3):
        base = i * 10 * MS
        forward(base, 8)
        k(base + 25 * MS // 10, base + 4 * MS, names[5])
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?, ?, ?, 0, 2)", (base + 4 * MS, base + 5 * MS, 1 << 30))
    k(0, MS, "exactlin_scaled_mm", dev=1)
    db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
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
            "Commit hashing": 1.0, "memcpy": 1.0, TU.HOST_GAP: 5.0}
    assert {c: round(r["exclusive_ms"], 6) for c, r in by.items()} == want
    assert by["Commit hashing"]["kernel_ms"] == pytest.approx(1.5)
    assert by["Commit hashing"]["overlap_with_tensor_ms"] == pytest.approx(0.5)
    assert by["int8 GEMM"]["launches"] == 4 and by["memcpy"]["gb"] == pytest.approx(2 ** 30 / 1e9)


def test_utilization_and_losses(doc):
    s = doc["summary"]
    assert s["macs"] == {"int8": 8 * 36864, "attn": 2 * 2 * 4 * 16 * (16 - 6), "head": 2 * 64 * 100}
    ideal = 1.0 + 2560 / (THETA / 2) * 1e3 + 12800 / (THETA / 2) * 1e3
    assert s["ideal_ms"] == pytest.approx(ideal) and s["util"] == pytest.approx(ideal / 10)
    by = {r["class"]: r for r in doc["classes"]}
    assert by["int8 GEMM"]["class_util"] == pytest.approx(0.5) and by["int8 GEMM"]["loss_ms"] == pytest.approx(1.0)
    assert sum(r["loss_ms"] for r in doc["classes"]) == pytest.approx(10 - ideal)
    assert doc["classes"][0]["class"] == TU.HOST_GAP
    assert by[TU.HOST_GAP]["util_if_zero"] == pytest.approx(ideal / 5)
    assert "**tensor utilization 11.0%**" in TU.table(doc)


def test_a_forward_restarts_when_the_quantizer_grid_changes():
    q = lambda s, g: (s, s + 1, "exactlin_quant_rows", g, 0, 0)  # noqa: E731
    assert TU.forwards([q(0, 4), q(1, 4), q(2, 8), q(3, 8), q(4, 8), q(5, 8), q(6, 8)], layers=1) == [(2, 8)]
