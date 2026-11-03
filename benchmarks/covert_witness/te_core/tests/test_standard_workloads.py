"""te_core.standard_workloads builds `training/standard-workloads`: a row per workload and measured dial setting, whose
ρ is te_core's one bound under the row's reading (A where the trace has entries, else B) at the dial's default and the
console's, with each kind's term under both readings, whose rho_inputs give the token form there
(algorithm_policy.page_rho) and whose input_status says which inputs te_core measured; a generation run's row is its
rounds alone, each GEMM on the served codes priced as NVFP4; and the panel keeps to the panels route's limits, a cell
up to 4,096 characters."""
import dataclasses
import json
import math
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from te_core import VERSION, algorithm_policy, console, formalism, policy
from te_core import standard_workloads as SW
from te_core.gates import of_log, per_mac

#: a distillation run's row as the builder measured it from the store: the rows file's row, the run's quantities at its
#: last checkpoint (formalism.Measured), its start's shape, what it read, and the panel's cells it gave
PINNED = Path(__file__).parent / "standard_workloads" / "distill.json"
U, R = 16, 4
#: a served matrix (out U, in 2U) and the GEMMs on it, the activations and a registered matrix in a round of 2 x 3
#: tokens (2 sequences of a 3-token prompt and 2 generated tokens, the last never fed back)
SERVED = {(2 * U, U)}
NOTES = {"args": {"steps": 2, "update": "full", "head": "fp4"}, "served_code_bits": 4.5 * U * 2 * U,
         "served_lora_bits": 16 * R * 3 * U, "rounds": [{"after_step": 1, "prompt": 3, "generated": 2, "sequences": 2},
                                                        {"after_step": 2, "prompt": 3, "generated": 2, "sequences": 2}]}
CODED = ["fwd", "updated", "mm", json.dumps([[8, 2 * U], [2 * U, U]]), 1, 8 * 2 * U * U]
ROUND_LOG = [CODED, ["fwd", "act", "mm", json.dumps([[8, U], [U, 8]]), 1, 8 * U * 8],
             ["fwd", "theta", "mm", json.dumps([[8, U], [U, 2 * U]]), 1, 8 * U * 2 * U],
             ["fwd", "updated", "mul", json.dumps([[8, U], [U]]), 1, 8 * U]]


def record(round_tokens: int = 8, round_log=ROUND_LOG) -> dict:
    """generate.py's layout: training steps 0 and 2, each followed by a round (steps 1 and 3), and two checkpoints."""
    train = [["fwd", "updated", "mm", json.dumps([[64, 2 * U], [2 * U, U]]), 1, 64 * 2 * U * U],
             ["bwd", "grad", "mm", json.dumps([[2 * U, 64], [64, U]]), 2, 2 * 64 * 2 * U * U]]
    steps = [{"step": i, "tokens": round_tokens if i % 2 else 64, "struct": "round" if i % 2 else "train"}
             for i in range(4)]
    return {"steps": steps, "structures": {"train": {"updated": [[1]], "log": train},
                                           "round": {"updated": [], "log": round_log}},
            "setup": {"step": -1, "by": {"other/param": 32}, "macs": 32},
            "tail": {"step": 4, "by": {}, "macs": 0}, "checkpoints": [{"step": 2}, {"step": 4}]}


def build(m: formalism.Measured, fx: dict, monkeypatch) -> tuple[dict, list, dict]:
    monkeypatch.setattr(SW, "traces", lambda arts: {})
    monkeypatch.setattr(SW, "measured", lambda row, traced: (m, fx["shape"], fx["read"]))
    p, details = SW.build({"reading": "A", "rows": [fx["row"]]}, stamp="2026-11-03T00:00:00Z")
    (row,) = p["rows"]
    return p, details, dict(zip(SW.COLUMNS, row))


def test_the_pinned_row_is_its_runs_quantities_priced_by_te_cores_one_bound_at_both_dial_points(monkeypatch):
    fx = json.loads(PINNED.read_text())
    m = formalism.from_json(fx["measured"])
    p, details, cells = build(m, fx, monkeypatch)
    assert m.entries is not None and list(cells.values())[:-1] == fx["cells"][:-1] and cells["te_core"] == VERSION
    assert cells["reading"] == "A" and p["source"].endswith("reading A; islands/standard-workloads)")
    inputs = json.loads(cells["rho_inputs"])
    assert list(inputs) == list(SW.KEYS) == list(json.loads(cells["input_status"]))
    assert set(json.loads(cells["input_status"]).values()) == {"measured"}
    terms = json.loads(cells["rho_terms"])
    at = SW.points(policy.w_fwd(fx["shape"]))
    for name, (c, k) in at.items():
        assert math.isclose(algorithm_policy.page_rho(inputs, c, k)[0], formalism.at(m, c, k)["rho"], rel_tol=1e-4)
        for reading in SW.READINGS:
            want = formalism.at(m, c, k, reading)
            assert terms[reading][name] == {kind: SW.rho_cell(want["bound"]["terms"][kind]) for kind in formalism.KINDS} \
                | {"rho": SW.rho_cell(want["rho"]), "binding": SW.binding(want)}
        assert cells[f"rho_{name}"] == terms["A"][name]["rho"] == SW.rho_cell(formalism.at(m, c, k, "A")["rho"])
    assert cells["binding_kind"] == SW.binding_cell({n: formalism.at(m, c, k, "A") for n, (c, k) in at.items()})
    assert details[0]["reading"] == "A" and details[0]["at"]["A"]["console"]["rho"] == \
        SW.finite(formalism.at(m, *SW.CONSOLE, "A")["rho"])
    assert not SW.refused(p)


def test_a_row_without_entries_is_read_under_b_and_shows_why_te_core_refuses_a(monkeypatch):
    fx = json.loads(PINNED.read_text())
    m = dataclasses.replace(formalism.from_json(fx["measured"]), entries=None)
    p, details, cells = build(m, fx, monkeypatch)
    terms = json.loads(cells["rho_terms"])
    assert cells["reading"] == "B" and p["source"].endswith("reading B; islands/standard-workloads)")
    assert terms["A"].startswith("refused: reading A: no entries")
    assert cells["rho_default"] == terms["B"]["default"]["rho"]
    assert cells["rho_console"] == SW.rho_cell(formalism.at(m, *SW.CONSOLE, "B")["rho"])


def test_the_kinds_a_step_has_no_input_for_are_named_with_their_steps():
    parts = (formalism.Part(3e12, 100, 2.4e12, 1e12, 3, 4e6, 2e6, 3e6), formalism.Part(1e12, 50, 0.8e12, 0, 2),
             formalism.Part(1e9, 0, 0, 0, 1))
    m = formalism.Measured(parts, 15, 8192, formalism.even(1e6), formalism.even(1e6), 1e6)
    assert SW.unmeasured(m) == {kind: f"2 of 5 steps with token work have no traced {field}, counted 0"
                                for kind, field in (("reduction", "partial_bits"), ("update", "grad_bits"))}
    assert SW.unmeasured(dataclasses.replace(m, parts=parts[:1])) == {}


def test_a_row_whose_inputs_dont_give_te_cores_rho_is_refused(monkeypatch):
    fx = json.loads(PINNED.read_text())
    m = formalism.from_json(fx["measured"])
    w_fwd = policy.w_fwd(fx["shape"])
    monkeypatch.setattr(algorithm_policy, "page_rho", lambda d, c, k: (1.01 * formalism.at(m, c, k)["rho"], "target"))
    with pytest.raises(ValueError, match="at default"):
        SW.priced(m, w_fwd)
    with pytest.raises(ValueError, match="w_gates, w_fit_gates not finite"):
        SW.priced(formalism.Measured((formalism.Part(1e9, 0, 0, 0),), 16, 8192, formalism.even(1e6),
                                     formalism.even(1e6), 1e6), w_fwd)


def test_an_input_is_measured_where_te_core_counted_traced_or_read_it_and_assumed_where_it_stubbed_it():
    traced = formalism.TRACED | {"gates": "counted MACs in gates (te_core.gates)", "u1_bits": formalism.TRACED["u_in"],
                                 "delta_bits": formalism.STUBS["delta_bits"], "decoder_gates": formalism.DECODER["none"]}
    parts = (formalism.Part(3e12, 100, 2.4e12, 1e12, 3, 4e6), formalism.Part(1e12, 50, 0.8e12, 0, 2, None))

    def status(labels, override=None):
        return SW.status(formalism.Measured(parts, 15, 8192, formalism.even(1e6), formalism.even(1e6), 1e6, 0.0,
                                            labels), override)

    st = status(traced)
    assert {k for k, v in st.items() if v != "measured"} == {"delta_grad_bits"}
    assert st["delta_grad_bits"] == ("assumed: unmeasured, 2 of 5 steps in structures with no traced Delta_grad, so "
                                     "te_core fails Fit at every k (the traced steps' largest 4e6 bits)")
    st = status(traced | {k: formalism.STUBS[k] for k in ("tau_bits", "u", "token_gates")})
    assert st["tau_bits"] == f"assumed: {formalism.STUBS['tau_bits']}"
    assert st["sigma"] == st["w_gates"] == st["w_fit_gates"] == f"assumed: {formalism.STUBS['token_gates']}"
    assert st["u_points"] == f"assumed: {formalism.STUBS['u']}" and st["u_in_points"] == "measured"
    assert status(traced | {"decoder_gates": formalism.DECODER["witness"]})["d_gates"] == "measured"
    assert status({k: v for k, v in traced.items() if k != "a_bits"})["a_bits"] == "assumed: no label from te_core"
    assert status(traced, {"tau_bits": "assumed: why"})["tau_bits"] == "assumed: why"


def test_a_generation_runs_record_is_its_rounds_with_gemms_on_served_codes_priced_as_nvfp4():
    got = SW.rounds(record(), NOTES, SERVED)
    assert [s["step"] for s in got["steps"]] == [1, 3]
    assert list(got["structures"]) == ["round"]
    st = got["structures"]["round"]
    assert st["log"][0] == [*CODED[:3], f"{CODED[3]} {SW.FP4}", *CODED[4:]]
    assert st["log"][1:] == ROUND_LOG[1:]
    ops = {" ".join(str(x) for x in r[:4]): r[4:6] for r in st["log"]}
    assert of_log({"ops": ops})["kinds"]["gemm/fp4"]["macs"] == CODED[5]
    updated = of_log({"ops": {k: v for k, v in ops.items() if k.startswith("fwd updated")}})["gates"]
    assert st["affected_gates"] == updated > CODED[5] * per_mac("gemm/fp4")
    assert st["delta_grad_bits"] == 0.0
    assert got["setup"]["by"] == got["tail"]["by"] == {} and got["setup"]["macs"] == got["tail"]["macs"] == 0
    assert got["checkpoints"] == [{"step": 4}]


@pytest.mark.parametrize("rec, match", [
    (record(round_tokens=10), "not its prompts' and every generated token but the last"),
    (record() | {"steps": record()["steps"][:3]}, "not generate.py's layout"),
    (record(round_log=[*ROUND_LOG, ["bwd", "grad", "mm", json.dumps([[U, 8], [8, U]]), 1, 8 * U * U]]), "a backward"),
])
def test_a_record_not_in_generate_pys_layout_or_with_a_round_that_trains_is_refused(rec, match):
    with pytest.raises(ValueError, match=match):
        SW.rounds(rec, NOTES, SERVED)


def test_a_rounds_delta_is_the_served_update_and_its_tau_the_prompts_ids_over_the_rounds_tokens():
    tensors = {"embed": ["bfloat16", [1000, U]], "blocks.0.w": ["bfloat16", [U, 2 * U]], "blocks.0.g": ["bfloat16", [U]],
               "norm": ["bfloat16", [U]]}
    lora = SW.round_formalism(NOTES | {"args": NOTES["args"] | {"update": "lora"}}, tensors, 1000)
    assert lora["delta_bits"] == NOTES["served_lora_bits"]
    assert lora["tau_bits"] == policy.id_bits(1000) * 3 / 4
    assert SW.round_formalism(NOTES, tensors, 1000)["delta_bits"] == NOTES["served_code_bits"] + 16 * 2 * U
    bf16_head = NOTES | {"args": NOTES["args"] | {"head": "bf16"}}
    assert SW.round_formalism(bf16_head, tensors, 1000)["delta_bits"] \
        == NOTES["served_code_bits"] + 16 * (2 * U + 1000 * U)
    assert lora["labels"] == SW.ROUND_LABELS


def test_the_route_takes_a_cell_of_4096_characters_and_refuses_a_longer_one_or_a_long_note():
    p = {"note": "n", "columns": ["a", "b"], "rows": [["x" * SW.CELL_MAX, 1.5]]}
    assert not SW.refused(p)
    assert SW.refused(p | {"rows": [["x" * (SW.CELL_MAX + 1), 1.5]]}) == ["row 0's a (4097)"]
    assert SW.refused(p | {"rows": [["x", True]]}) == ["row 0's b (bool)"]
    assert SW.refused(p | {"note": "n" * (console.NOTE_MAX + 1)}) == [f"its note is {console.NOTE_MAX + 1} characters"]
