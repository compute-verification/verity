"""te_core.standard_workloads builds `training/standard-workloads`: a row per workload and measured dial setting, whose
ρ is formalism.published's under the builder's `READING`, A (with its binding kind, what it rests on and its kinds as
published gives them, reading B's forms among their sensitivities; null and "unmeasured" where it has no bound, never
infinite) at the console's default (rho_default and rho_console) and the registered serving block (rho_served), whose
rho_inputs give the dial's ρ at both (page_rho) and whose input_status says which inputs te_core measured (an
unmeasured Δ_grad's steps and whether Fit's verdict rests on it, and the fresh-products kind's input); a generation
run's row is its rounds alone, each GEMM on the served codes priced as NVFP4; it reads traces from te_core 4.4 on; and
the panel keeps to the panels route's limits, a cell up to 4,096 characters, and never replaces a served panel with one
that drops its runs."""
import dataclasses
import json
import math
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from te_core import VERSION, algorithm_policy, annotate, console, formalism, panels, policy
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


def record(round_tokens: int = 6, round_log=ROUND_LOG) -> dict:
    """generate.py's layout: training steps 0 and 2, each followed by a round (steps 1 and 3) whose counted tokens are
    its prompts', and two checkpoints."""
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
    p, details = SW.build({"rows": [fx["row"]]}, stamp="2026-11-03T00:00:00Z")
    (row,) = p["rows"]
    return p, details, dict(zip(SW.COLUMNS, row))


def at(m: formalism.Measured) -> dict[str, dict]:
    return {name: formalism.at(m, c, k) for name, (c, k) in SW.POINTS.items()}


def test_the_pinned_row_is_its_runs_quantities_priced_as_algorithm_policy_prices_a_run_at_both_points(monkeypatch):
    fx = json.loads(PINNED.read_text())
    m = formalism.from_json(fx["measured"])
    p, details, cells = build(m, fx, monkeypatch)
    pinned = dict(zip(SW.COLUMNS, fx["cells"]))
    assert {k: v for k, v in cells.items() if k != "te_core"} == {k: v for k, v in pinned.items() if k != "te_core"}
    assert cells["te_core"] == VERSION
    assert p["source"] == f"python -m te_core.standard_workloads (te_core {VERSION}; islands/standard-workloads)"
    inputs, st = json.loads(cells["rho_inputs"]), json.loads(cells["input_status"])
    assert list(inputs) == list(SW.KEYS) and list(st) == [*SW.KEYS, "products"]
    assert {st[k] for k in SW.KEYS} == {"measured"}
    bounds = {name: formalism.bound(m, c, k) for name, (c, k) in SW.POINTS.items()}
    assert st["products"] == SW.products(bounds) == f"assumed: {bounds['console']['unmeasured']['fresh_products']}"
    pts, got = SW.POINTS, at(m)
    assert pts == {"console": algorithm_policy.CONSOLE, "served": panels.SERVED}
    one = {name: formalism.published(m, c, k, SW.READING) for name, (c, k) in pts.items()}
    for name, (c, k) in pts.items():
        assert math.isclose(algorithm_policy.page_rho(inputs, c, k)[0], got[name]["rho"], rel_tol=1e-4)
        assert cells[f"rho_{name}"] == SW.rho_cell(one[name]["rho"])
    assert cells["rho_default"] == cells["rho_console"]
    b = formalism.bound(m, *pts["console"])
    assert (cells["reading"], cells["binding_kind"], cells["rests_on"]) == \
        (SW.READING, one["console"]["binding_kind"], one["console"]["rests_on"])
    kinds = json.loads(cells["kinds"])
    assert cells["kinds"] == SW.kinds(one["console"], b) and list(kinds["sensitivity"]) == list(SW.SENSITIVITIES)
    for name in SW.SENSITIVITIES:
        assert kinds["sensitivity"][name] == pytest.approx(b["sensitivities"][name], rel=1e-3)
    assert details[0]["at"] == SW.finite(got) and details[0]["published"] == SW.finite(one)
    assert details[0]["bound"]["console"]["terms"] == b["terms"]
    assert not SW.refused(p) and not SW.refused(p | {"note": SW.note(p, "art:" + "0" * 64)})


def test_under_reading_a_a_row_is_publisheds_rho_and_basis_as_is_with_its_terms_null_where_unmeasured(monkeypatch):
    """The pinned row's token kind under reading A (its trace's entries and its macs_log's tiles) is at most 1 at the
    console's and fresh products are unmeasured (an op log can't tell a fresh operand): its ρ, binding kind and basis
    are published's as it gives them, the kinds cell holding the token term, fresh products' null and published's
    partial and why; a measured term above 1 is published with its binding kind at both points; without an op log no
    kind is measured: null and unmeasured; an infinite ρ is refused."""
    fx = json.loads(PINNED.read_text())
    m = formalism.from_json(fx["measured"])
    assert SW.READING == "A"
    c, k = algorithm_policy.CONSOLE
    b = formalism.bound(m, c, k)
    assert b["terms"]["token"] is not None and b["terms"]["token"] <= 1 and b["terms"]["fresh_products"] is None
    one = formalism.published(m, c, k, "A")
    p, _, cells = build(m, fx, monkeypatch)
    assert (cells["reading"], cells["rho_console"], cells["binding_kind"], cells["rests_on"]) == \
        ("A", SW.rho_cell(one["rho"]), one["binding_kind"], one["rests_on"])
    kinds = json.loads(cells["kinds"])
    assert list(kinds["bound"]) == list(formalism.KINDS) and kinds["bound"]["fresh_products"] is None
    assert kinds["bound"]["token"] == pytest.approx(b["terms"]["token"], rel=1e-3)
    assert (kinds.get("partial", False), kinds.get("why")) == (one["partial"], one["why"])
    assert "reading A, one bound over the valid kinds (null: unmeasured)" in p["note"]
    assert (f"c = {algorithm_policy.num(policy.CONSOLE[0], 5)} gates, one full 4,096-in, 1,024-out session of the 45M "
            f"reference with attention") in p["note"]
    bare = dataclasses.replace(m, ops=None)
    _, _, cells = build(bare, fx, monkeypatch)
    assert (cells["rho_default"], cells["rho_console"], cells["rho_served"], cells["rests_on"]) == \
        (None, None, None, "unmeasured")
    assert cells["binding_kind"] == formalism.published(bare, c, k, "A")["binding_kind"]
    assert json.loads(cells["kinds"])["bound"] == {"fresh_products": None, "token": None, "decoding": None}
    real = formalism.published
    above = {"rho": 2.5, "binding_kind": "token", "rests_on": formalism.RESTS_ON_CELL["token"]}
    monkeypatch.setattr(formalism, "published", lambda *a: real(*a) | above)
    _, _, cells = build(m, fx, monkeypatch)
    assert (cells["rho_default"], cells["rho_console"], cells["rho_served"], cells["binding_kind"],
            cells["rests_on"]) == (2.5, 2.5, 2.5, "token", formalism.RESTS_ON_CELL["token"])
    monkeypatch.setattr(formalism, "published", lambda *a: one | {"rho": math.inf})
    with pytest.raises(SystemExit, match="reading A's rho is infinite at console and served"):
        build(m, fx, monkeypatch)


def test_traces_from_te_core_4_4_to_this_one_are_read_and_others_refused(tmp_path, monkeypatch):
    (tmp_path / "traces").mkdir()
    (tmp_path / "manifest.json").write_text(json.dumps({"te_commit": "c0ffee"}))
    (tmp_path / "traces" / "a.json").write_text(json.dumps({"run": "r1", "step": 1, "trace": {"structure": {}}}))
    (tmp_path / "traces" / "b.json").write_text(json.dumps({"run": "r1", "step": 2, "trace": None}))
    monkeypatch.setattr(SW, "fetch", lambda art, *paths: tmp_path)
    for v in (SW.TRACED_SINCE, VERSION):
        monkeypatch.setattr(SW, "version_at", lambda commit, v=v: v)
        assert SW.traces(["art:t"]) == {"r1": {1: {"run": "r1", "step": 1, "trace": {"structure": {}}, "from": "art:t",
                                                   "te_core": v}}}
    for v in ("4.3", "9.0"):
        monkeypatch.setattr(SW, "version_at", lambda commit, v=v: v)
        with pytest.raises(SystemExit, match=f"traced by te_core {v}, not {SW.TRACED_SINCE} to {VERSION}"):
            SW.traces(["art:t"])


def test_a_publish_never_drops_a_served_runs_row_nor_replaces_a_newer_te_cores():
    def panel(runs, te=VERSION):
        return {"columns": SW.COLUMNS, "rows": [[None, None, None, f"{r} art:x (about)", *[None] * 5, te]
                                                for r in runs]}
    mine = panel(["r1", "r2"])
    assert SW.emptied(mine, None) == [] and SW.emptied(mine, panel(["r2", "r1"], "4.4")) == []
    assert SW.emptied(mine, panel(["r1", "r3"])) == ["it drops the served row of r3"]
    assert SW.emptied(mine, panel(["r1"], "9.0")) == [f"the route serves te_core 9.0, newer than {VERSION}"]


def test_a_row_whose_inputs_dont_give_te_cores_rho_is_refused(monkeypatch):
    fx = json.loads(PINNED.read_text())
    m = formalism.from_json(fx["measured"])
    w_fwd = policy.w_fwd(fx["shape"])
    monkeypatch.setattr(algorithm_policy, "page_rho", lambda d, c, k: (1.01 * formalism.at(m, c, k)["rho"], "target"))
    with pytest.raises(ValueError, match="at console"):
        SW.priced(m, w_fwd, at(m))
    with pytest.raises(ValueError, match="w_gates, w_fit_gates not finite"):
        SW.priced(formalism.Measured((formalism.Part(1e9, 0, 0, 0),), 16, 8192, formalism.even(1e6),
                                     formalism.even(1e6), 1e6), w_fwd, {})


def test_a_row_whose_token_dials_miss_its_measured_tokens_is_refused_with_the_value_to_write(monkeypatch):
    """A step of two micro-batches the counter counts one of: a dial read off its count is half the run's tokens."""
    fx = json.loads(PINNED.read_text())
    m = formalism.from_json(fx["measured"])
    assert SW.token_dials(fx["row"]["dials"], m.tokens) == []
    half = fx["row"] | {"dials": fx["row"]["dials"] | {"tokens_per_param": fx["row"]["dials"]["tokens_per_param"] / 2}}
    with pytest.raises(SystemExit, match=r"its dial tokens_per_param is 0\.75547, te_core measured 1\.51095: the dial"):
        build(m, fx | {"row": half}, monkeypatch)
    assert SW.token_dials({"n_params": 10, "tokens": 50.0}, 100.0) == ["its dial tokens is 50, te_core measured 100: "
                                                                        "the dial is the run's"]
    assert SW.token_dials({"n_params": 10, "tokens_per_param": 10.04}, 100.0) == []


def test_an_input_is_measured_where_te_core_counted_traced_or_read_it_and_assumed_where_it_stubbed_it():
    traced = formalism.TRACED | {"gates": "counted MACs in gates (te_core.gates)", "u1_bits": formalism.TRACED["u_in"],
                                 "delta_bits": formalism.STUBS["delta_bits"], "decoder_gates": formalism.DECODER["none"]}
    parts = (formalism.Part(3e12, 100, 2.4e12, 1e12, 3, 4e6), formalism.Part(1e12, 50, 0.8e12, 0, 2, None))

    def status(labels, override=None):
        return SW.status(formalism.Measured(parts, 15, 8192, formalism.even(1e6), formalism.even(1e6), 1e6, 0.0,
                                            labels), override)

    st = status(traced)
    assert {k for k, v in st.items() if v != "measured"} == {"delta_grad_bits"}
    assert st["delta_grad_bits"] == ("assumed: unmeasured on 2 of 5 steps in structures with no traced Delta_grad, so "
                                     "te_core fails Fit at every k; with it at 0, Fit still fails at both points, so "
                                     "neither rho_console nor rho_served rests on it (the traced steps' largest 4e6 "
                                     "bits)")
    st = status(traced | {k: formalism.STUBS[k] for k in ("tau_bits", "u", "token_gates")})
    assert st["tau_bits"] == f"assumed: {formalism.STUBS['tau_bits']}"
    assert st["sigma"] == st["w_gates"] == st["w_fit_gates"] == f"assumed: {formalism.STUBS['token_gates']}"
    assert st["u_points"] == f"assumed: {formalism.STUBS['u']}" and st["u_in_points"] == "measured"
    assert status(traced | {"decoder_gates": formalism.DECODER["witness"]})["d_gates"] == "measured"
    assert status({k: v for k, v in traced.items() if k != "a_bits"})["a_bits"] == "assumed: no label from te_core"
    assert status(traced, {"tau_bits": "assumed: why"})["tau_bits"] == "assumed: why"


def test_an_unmeasured_delta_grad_names_its_steps_and_whether_fits_verdict_at_either_point_rests_on_it():
    """A step with tokens has no Δ_grad where te_core.annotate gives it no traced structure: before the run's first
    update (its structure updates nothing), the program's last (the capture refuses it) or one no traced step shares.
    The label names them, refused unless they number te_core's, and says Fit's verdict at a point rests on them only
    where Fit holds there with their Δ_grad at 0."""
    struct = {"updated": [[1.0, 2, 3, 4, "x"]]}
    rec = {"steps": [{"step": 0, "struct": "first", "tokens": 64}, {"step": 1, "struct": "train", "tokens": 64},
                     {"step": 2, "struct": "odd", "tokens": 64}, {"step": 3, "struct": "sync", "tokens": 0},
                     {"step": 4, "struct": "train", "tokens": 64}, {"step": 5, "struct": "end", "tokens": 64}],
           "structures": {"first": {"updated": []}, "train": struct, "odd": struct, "sync": {"updated": []},
                          "end": struct}}
    gaps = SW.untraced(rec, {0: {"measured": {"delta_grad_bits": 0}}, 1: {"measured": {"delta_grad_bits": 10}}})
    assert gaps == ["step 0: before the run's first update: its structure updates nothing, so te_core.annotate leaves "
                    "out its trace, which measures 0 bits", "step 2: no traced step shares its structure",
                    "step 5: the program's last, its structure having no other step, which te_core.capture refuses to "
                    "trace, the program ending before the step does"]
    got = annotate.with_structures(rec, {"first": struct, "train": struct})
    assert {s["step"] for s in got["steps"] if s["tokens"] and "+traced" not in s["struct"]} == {0, 2, 5}
    parts = (formalism.Part(3e12, 100, 2.4e12, 1e12, 3, 1e3), formalism.Part(1e12, 50, 0.8e12, 0, 3, None))
    m = formalism.Measured(parts, 0.0, 8192, formalism.even(1e6), formalism.even(1e6), 1e3, 0.0, {},
                           delta_t_bits=1e3)
    assert SW.held(m) == ["console", "served"] and not formalism.at(m, *SW.POINTS["console"])["fit"]
    assert SW.status(m, read={"untraced": gaps, "steps": 6})["delta_grad_bits"] == (
        f"assumed: unmeasured on 3 of 6 steps ({'; '.join(gaps)}), so te_core fails Fit at every k; Fit's verdict at "
        f"console and served rests on it, Fit holding there with it at 0 (the traced steps' largest "
        f"{algorithm_policy.num(1e3)} bits)")
    four = dataclasses.replace(m, parts=(parts[0], dataclasses.replace(parts[1], steps=4)))
    got = SW.status(four, read={"untraced": ["a", "b", "c", "d"], "steps": 7})["delta_grad_bits"]
    assert got.startswith("assumed: unmeasured on 4 of 7 steps (a; b; c; and 1 more), so")
    with pytest.raises(ValueError, match="te_core has no Delta_grad for 3 steps, `untraced` names 2"):
        SW.status(m, read={"untraced": gaps[:2], "steps": 6})


def test_the_fresh_products_input_is_measured_at_the_consoles_default_or_assumed_with_te_cores_reason():
    """`products`, the fresh-products kind's input: measured where te_core prices a trace's product rows at the
    console's default, with its reason where the serving block's k leaves them unmeasured (they were classified at
    the console's); assumed where the dominant traced step's stand in for steps of another shape, or with te_core's
    reason where unmeasured (an op log's, and a round untraced)."""
    fx = json.loads(PINNED.read_text())
    m = formalism.from_json(fx["measured"])
    row = {"count": 2, "phase": "forward", "op": "mm", "dims": {"batch": 1, "m": 4, "k": 8, "n": 8},
           "token_dims": ["m"], "macs": 256, "outputs": 32, "affected": [True, True], "widths": [16, 16],
           "rates": [16.0, 16.0], "paid": [True, True], "self": False, "b_c": 32, "k_whole": 8,
           "gates_per_mac": 1480.0, "cell_gates": 23682.0}
    rows = formalism.products_of_trace({"capture": formalism.PRODUCTS_FROM, "k": float(policy.CONSOLE[1]),
                                        "order": "unknown", "tokens": 8, "gates": None, "rows": [row]})

    def bounds(p):
        return {name: formalism.bound(dataclasses.replace(m, products=p), c, k) for name, (c, k) in SW.POINTS.items()}

    b = bounds(rows)
    off = b["served"]["unmeasured"]["fresh_products"]
    assert "fresh_products" not in b["console"]["unmeasured"]
    assert off.startswith(f"unmeasured: the products were classified at k = {policy.CONSOLE[1]:g}")
    assert SW.products(b) == f"measured at the console's default; at the serving block, {off}"
    stand = rows | {"stand_in": {"structures": ["a", "c"], "step_tokens": [4, 8], "traced_tokens": 8,
                                 "why": formalism.STAND_IN}}
    assert SW.products(bounds(stand)) == (
        f"assumed: the run's traced steps differ in shape, so the dominant traced step's products stand in for all of "
        f"them, scaled by tokens (2 traced structures); at the serving block, {off}")
    why = bounds(None)["console"]["unmeasured"]["fresh_products"]
    assert SW.products(bounds(None)) == f"assumed: {why}"
    assert SW.products(bounds(None), rounds=True) == f"assumed: {why}{SW._UNTRACED}"


def test_a_generation_runs_record_is_its_rounds_with_gemms_on_served_codes_priced_as_nvfp4():
    rec = record()
    got = SW.rounds(rec, NOTES, SERVED)
    assert [s["step"] for s in got["steps"]] == [1, 3]
    assert [s["tokens"] for s in got["steps"]] == [2 * (3 + 2 - 1)] * 2 and rec["steps"][1]["tokens"] == 2 * 3
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
    assert set(SW.ROUNDS) <= set(SW.KEYS) and all(v.startswith("assumed: ") for v in SW.ROUNDS.values())


@pytest.mark.parametrize("rec, match", [
    (record(round_tokens=8), "8 tokens counted, not its prompts'"),
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


def test_the_route_takes_a_cell_of_4096_characters_and_refuses_a_longer_one_a_long_note_or_a_long_body():
    p = {"title": "t", "note": "n", "source": "s", "columns": ["a", "b"], "rows": [["x" * console.CELL_MAX, 1.5]]}
    assert not SW.refused(p)
    long = "x" * (console.CELL_MAX + 1)
    assert SW.refused(p | {"rows": [[long, 1.5]]}) == [f"row 0's a is {long!r:.40}"]
    assert SW.refused(p | {"rows": [["x", True]]}) == ["row 0's b is True"]
    assert SW.refused(p | {"note": "n" * (console.NOTE_MAX + 1)}) == \
        [f"its note is {console.NOTE_MAX + 1} characters, over {console.NOTE_MAX}"]
    big = p | {"rows": p["rows"] * (console.BODY_MAX // console.CELL_MAX)}
    assert SW.refused(big) == [f"its body is {len(console.body(big))} bytes, over {console.BODY_MAX}"]
