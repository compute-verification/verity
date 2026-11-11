"""training_slowdown.gates: every price is a Definition's whole gate count, the one circuit-check pins the ANDs of."""
import json
from pathlib import Path

import pytest

from training_slowdown import gates

VERITY = Path(__file__).resolve().parents[3] / "verity"


def test_each_and_count_is_circuit_checks_pin():
    for name, (module, ands, _, _) in gates.DEFINITIONS.items():
        pins = json.loads(VERITY.joinpath(*module.split(".")).with_suffix(".circuit-check.json").read_text())["definitions"]
        assert pins[name]["and"] == ands, name


def test_each_count_is_the_definitions():
    pytest.importorskip("verity.circuits.evaluation.bits")
    assert gates.recompute() == {n: tuple(v[1:]) for n, v in gates.DEFINITIONS.items()}


def test_prices():
    assert gates.per_mac("gemm/bf16") == pytest.approx((7872 + 15049 + 761) / 16)
    assert gates.per_mac("elementwise/bf16") == 2053 + 1894
    assert gates.per_mac("elementwise/f32") == gates.per_mac("gemm/f32") == 12637
    assert gates.per_mac("gemm/tf32") == (6524 + 12120 + 439) / 8 == 2385.375
    assert gates.per_mac("gather") == 48
    assert gates.per_and("gemm/bf16") == pytest.approx((7872 + 15049 + 761) / 7872)
    assert gates.per_and("gemm/tf32") == pytest.approx(19083 / 6524)
    assert gates.UNIT == "gates: AND + XOR + NOT, 3.0x a bf16 GEMM's AND count"


def test_kinds_and_assumptions():
    assert gates.kind_of("fwd", "theta", "mm") == ("gemm/bf16", "gemm in bf16 (no dtype)")
    assert gates.kind_of("forward", "updated", "mm", "float32") == ("gemm/f32", "")
    assert gates.kind_of("other", "param", "mm", "tf32") == ("gemm/tf32", "")
    assert gates.kind_of("fwd", "act", "_scaled_dot_product_flash_attention", "bfloat16") == ("gemm/bf16", "")
    assert gates.kind_of("other", "param", "_fused_adamw_") == ("elementwise/f32", "optimizer in f32 (no dtype)")
    assert gates.kind_of("bwd", "grad", "mul") == ("elementwise/bf16", "elementwise in bf16 (no dtype)")
    assert gates.kind_of("fwd", "updated", "embedding")[0] == "gather"
    assert gates.kind_of("other", "param", "linalg_qr")[0] == "elementwise/f32"
    assert gates.kind_of("fwd", "theta", "_scaled_mm") == ("gemm/e4m3", "")
    for op in ("mv", "addmv", "addmv_"):   # matrix-vector products, which training_slowdown.capture.structure counts as GEMMs
        assert gates.kind_of("forward", "", op, "bfloat16") == ("gemm/bf16", "")
    assert gates.kind_of("forward", "", "mvlgamma", "bfloat16")[0] == "elementwise/bf16"


@pytest.mark.parametrize("phase, role", [("fwd", "theta"), ("bwd", "grad"), ("other", "param"), ("optimizer", "")])
def test_an_op_is_priced_in_the_dtype_it_ran_in_whatever_its_phase_and_role(phase, role):
    """The dtype an op's row names decides its price, never a rule of its class: an optimizer's product in bf16 (Muon's
    Newton-Schulz) is a bf16 GEMM and one in f32 (Shampoo's) an f32 one, a forward scaling in f32 is an f32
    multiply, and an optimizer's elementwise update in bf16 a bf16 one."""
    assert gates.kind_of(phase, role, "mm", "bfloat16") == ("gemm/bf16", "")
    assert gates.kind_of(phase, role, "mm", "float32") == ("gemm/f32", "")
    assert gates.kind_of(phase, role, "mul", "bfloat16") == ("elementwise/bf16", "")
    assert gates.kind_of(phase, role, "mul", "float32") == ("elementwise/f32", "")
    assert gates.kind_of(phase, role, "_linalg_eigh", "float32") == ("elementwise/f32", "")


@pytest.mark.parametrize("op, dtype, what", [
    ("_int_mm", "int32", "a GEMM"), ("mm", "float64", "a GEMM"), ("mul", "float64", "an elementwise"),
    ("eq", "bool", "an elementwise"), ("_foreach_mul_", "bfloat16+float32", "an elementwise"),
    ("_linalg_eigh", "float64", "an elementwise")])
def test_a_dtype_no_definition_prices_is_refused_with_the_op_and_the_dtype(op, dtype, what):
    """Nothing is priced as the cheapest: an op in a dtype with no Definition, and one whose results are in two, has
    no kind, and neither has a log that holds it."""
    with pytest.raises(gates.Unpriced, match=f"^{op} in {dtype.replace('+', '[+]')}: no Definition prices {what} MAC"):
        gates.kind_of("fwd", "theta", op, dtype)
    with pytest.raises(gates.Unpriced, match=f"^{op} in "):
        gates.of_log({"ops": {f"fwd theta {op} [[4, 4]] {dtype}": [1, 16]}})


def test_a_gather_moves_words_of_its_dtypes_bits():
    assert gates.gather_bits("embedding") == (16, "gather of 16-bit words (no dtype)")
    assert [gates.gather_bits("index", d)[0] for d in ("bfloat16", "float32", "int64", "bool")] == [16, 32, 64, 1]
    with pytest.raises(gates.Unpriced, match="^take in complex64: a gather's words"):
        gates.gather_bits("take", "complex64")
    out = gates.of_log({"ops": {"fwd updated embedding [[9, 4], [2, 3]] float32": [1, 10],
                                "fwd updated embedding [[9, 4], [2, 3]] bfloat16": [1, 10]}})
    assert out == {"gates": 10 * 3 * 32 + 10 * 3 * 16, "macs": 20, "assumed": {},
                   "kinds": {"gather": {"macs": 20, "gates": 10 * 3 * 32 + 10 * 3 * 16}}}


def test_a_logs_key_is_its_rows_with_the_dtype_where_the_row_names_one():
    old, new = ["fwd", "theta", "mm", "[[4, 5], [5, 5]]", 2, 200], ["other", "param", "mm", "[[4, 5], [5, 5]]", 1, 100, "float32"]
    assert gates.op_key(old) == "fwd theta mm [[4, 5], [5, 5]]"
    assert gates.op_key(new) == "other param mm [[4, 5], [5, 5]] float32"
    assert [gates.split_key(gates.op_key(r)) for r in (old, new)] == [(*old[:4], None), (*new[:4], "float32")]


def test_a_log_without_dtypes_in_gates():
    log = {"by": {"fwd/theta": 100, "other/param": 10, "bwd/grad": 50, "fwd/updated": 5},
           "ops": {"fwd theta mm [[4, 5], [5, 5]]": [1, 100], "other param _fused_adamw_ [[5]]": [1, 10],
                   "bwd grad mm [[4, 5], [5, 5]] bfloat16": [1, 40], "fwd updated embedding [[9, 4], [2]]": [1, 5]}}
    out = gates.of_log(log)
    assert out["kinds"]["gemm/bf16"]["macs"] == 140
    assert out["kinds"]["elementwise/f32"]["macs"] == 10
    assert out["kinds"]["elementwise/bf16"]["macs"] == 10
    assert out["gates"] == pytest.approx(140 * gates.per_mac("gemm/bf16") + 10 * 12637 + 10 * 3947 + 5 * 48)
    assert out["assumed"] == {"gemm in bf16 (no dtype)": 100, "optimizer in f32 (no dtype)": 10,
                              "gather of 16-bit words (no dtype)": 5, "MACs in no op row, as elementwise bf16": 10}


def test_a_log_that_names_every_dtype_is_priced_by_them_with_no_assumption():
    """An optimizer's products in bf16 and in f32 and a forward's scalings in f32 and in bf16, each at its own
    Definition's price: by `ASSUMED` every product is a bf16 GEMM and every forward scaling a bf16 multiply."""
    ops = {"other param mm [[4, 5], [5, 4]] bfloat16": [5, 400], "other param mm [[4, 5], [5, 4]] float32": [1, 80],
           "fwd theta mul [[2, 4]] float32": [1, 8], "fwd theta mul [[2, 4]] bfloat16": [1, 8]}
    out = gates.of_log({"by": {"other/param": 480, "fwd/theta": 16}, "ops": ops})
    assert out["assumed"] == {} and out["macs"] == 496
    assert out["gates"] == pytest.approx(400 * gates.per_mac("gemm/bf16") + 80 * 12637 + 8 * 12637 + 8 * 3947)
    assumed = gates.of_log({"by": {"other/param": 480, "fwd/theta": 16},
                            "ops": {k.rsplit(" ", 1)[0]: [6, 480] if "mm" in k else [2, 16] for k in ops}})
    assert assumed["gates"] == pytest.approx(480 * gates.per_mac("gemm/bf16") + 16 * 3947) != out["gates"]


def test_a_row_named_tf32_is_priced_at_the_tf32_step():
    """Of two GEMM rows of one shape in a record without dtypes, the one named `tf32` costs the TF32 step's gates a
    MAC and the other bf16's, and nothing is assumed of the named one."""
    log = {"by": {"other/param": 96}, "ops": {"other param mm [[4, 8], [8, 8]]": [1, 32],
                                              "other param mm [[4, 8], [8, 8]] tf32": [2, 64]}}
    out = gates.of_log(log)
    assert out["kinds"] == {"gemm/bf16": {"macs": 32, "gates": 32 * 1480.125},
                            "gemm/tf32": {"macs": 64, "gates": 64 * 2385.375}}
    assert out["assumed"] == {"gemm in bf16 (no dtype)": 32}


def test_tf32_names_a_gemm_alone():
    """TF32 is a GEMM's arithmetic: an elementwise op or a gather a row names `tf32` has no price."""
    with pytest.raises(gates.Unpriced, match="^mul in tf32: no Definition prices an elementwise MAC"):
        gates.kind_of("other", "param", "mul", "tf32")
    with pytest.raises(gates.Unpriced, match="^embedding in tf32: a gather's words"):
        gates.gather_bits("embedding", "tf32")
