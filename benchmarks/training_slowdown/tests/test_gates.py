"""training_slowdown.gates: every price is a Definition's whole gate count, the one circuit-check pins the ANDs of."""
import json
from pathlib import Path

import pytest

from training_slowdown import gates

BOOLEAN = Path(__file__).resolve().parents[3] / "verity" / "definitions" / "ops" / "boolean"


def test_each_and_count_is_circuit_checks_pin():
    for name, (module, ands, _, _) in gates.DEFINITIONS.items():
        pins = json.loads((BOOLEAN / f"{module}.circuit-check.json").read_text())["definitions"]
        assert pins[name]["and"] == ands, name


def test_each_count_is_the_definitions():
    pytest.importorskip("verity.circuits.evaluation.bits")
    assert gates.recompute() == {n: tuple(v[1:]) for n, v in gates.DEFINITIONS.items()}


def test_prices():
    assert gates.per_mac("gemm/bf16") == pytest.approx((7872 + 15049 + 761) / 16)
    assert gates.per_mac("elementwise/bf16") == 2053 + 1894
    assert gates.per_mac("elementwise/f32") == gates.per_mac("gemm/f32") == 12637
    assert gates.per_mac("gather") == 48
    assert gates.per_and("gemm/bf16") == pytest.approx((7872 + 15049 + 761) / 7872)
    assert gates.UNIT == "gates: AND + XOR + NOT, 3.0x a bf16 GEMM's AND count"


def test_kinds_and_assumptions():
    assert gates.kind_of("fwd", "theta", "mm") == ("gemm/bf16", "gemm in bf16 (no dtype)")
    assert gates.kind_of("forward", "updated", "mm", "float32") == ("gemm/f32", "")
    assert gates.kind_of("fwd", "act", "_scaled_dot_product_flash_attention", "bfloat16") == ("gemm/bf16", "")
    assert gates.kind_of("other", "param", "_fused_adamw_") == ("elementwise/f32", "optimizer in f32 (no dtype)")
    assert gates.kind_of("bwd", "grad", "mul") == ("elementwise/bf16", "elementwise in bf16 (no dtype)")
    assert gates.kind_of("fwd", "updated", "embedding")[0] == "gather"
    assert gates.kind_of("other", "param", "linalg_qr")[0] == "elementwise/f32"
    assert gates.kind_of("fwd", "theta", "_scaled_mm") == ("gemm/e4m3", "")
    kind, note = gates.kind_of("fwd", "theta", "_int_mm", "int8")
    assert kind == "gemm/bf16" and "no Definition" in note
    for op in ("mv", "addmv", "addmv_"):   # matrix-vector products, which training_slowdown.capture.structure counts as GEMMs
        assert gates.kind_of("forward", "", op, "bfloat16") == ("gemm/bf16", "")
    assert gates.kind_of("forward", "", "mvlgamma", "bfloat16")[0] == "elementwise/bf16"


def test_a_v23_log_in_gates():
    log = {"by": {"fwd/theta": 100, "other/param": 10, "bwd/grad": 50},
           "ops": {"fwd theta mm [[4, 5], [5, 5]]": [1, 100], "other param _fused_adamw_ [[5]]": [1, 10],
                   "bwd grad mm [[4, 5], [5, 5]] bfloat16": [1, 40]}}
    out = gates.of_log(log)
    assert out["kinds"]["gemm/bf16"]["macs"] == 140
    assert out["kinds"]["elementwise/f32"]["macs"] == 10
    assert out["kinds"]["elementwise/bf16"]["macs"] == 10
    assert out["gates"] == pytest.approx(140 * gates.per_mac("gemm/bf16") + 10 * 12637 + 10 * 3947)
    assert out["assumed"] == {"gemm in bf16 (no dtype)": 100, "optimizer in f32 (no dtype)": 10,
                              "MACs in no op row, as elementwise bf16": 10}
