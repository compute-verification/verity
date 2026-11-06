"""training_slowdown.annotate: a run's annotation through a step as a prefix of its count, and its traced structures."""
from training_slowdown.annotate import annotate, with_structures


def record(n_steps: int) -> dict:
    """A harness record of a toy run: one traced structure, every step the same."""
    struct = {"updated": [[100, i * 100, 100 * 8, 8, 16] for i in range(4)], "log": [], "beta": 2.0, "cut_bits": 800}
    steps = [{"step": t, "tokens": 8, "macs": 3 * 400 * 8, "by": {}, "struct": "s", "weights": {}, "shapes": {}}
             for t in range(n_steps)]
    return {"setup": {"macs": 7, "tokens": 0, "by": {}}, "tail": {"macs": 5, "tokens": 0, "by": {}}, "steps": steps,
            "structures": {"s": struct},
            "checkpoints": [{"step": s, "val_bpb": 1.0 - 0.01 * s} for s in (4, 9)]}


def test_an_annotation_through_a_step_is_a_prefix():
    rec = record(10)
    a4 = annotate(rec, through=4)
    assert a4.macs == 7 + 5 * 9600
    assert annotate(rec).macs == 7 + 10 * 9600 + 5
    assert a4.depth() == 300


def test_with_structures_scales_the_traced_rows():
    """The runtime counter's rows (4 nodes of 100 MACs per token, no cut) give way to a traced structure of 2 nodes of 50
    per token, scaled to each step's tokens; step 0 updates nothing and keeps its own, and |C| is the runtime count."""
    rec = record(3)
    rec["structures"] = {"s": {"updated": [[100, i * 100, 800, 8, 16] for i in range(4)], "log": [["x"]]},
                         "z": {"updated": [], "log": []}}
    rec["steps"][0]["struct"] = "z"
    rec["steps"][2]["tokens"] = 4
    rec["families"] = [{"covers": [1, 2], "beta": {"1": 1.0, "2": 2.0}}]
    traced = {"updated": [[50, 0, 50 * 16, 16, 32], [50, 50, 50 * 16, 16, 16]], "beta": 3.0, "cut_bits": 300}
    out = with_structures(rec, {"s": traced, "z": traced})
    ann = annotate(out)
    assert ann.macs == annotate(rec).macs
    assert [s.updated_macs for s in ann.steps] == [0, 100 * 8, 100 * 4]
    assert [s.beta for s in ann.steps] == [None, 3.0, 3.0] and ann.depth() == 50 and ann.bits() == 16
    assert out["families"] == rec["families"] and "formalism" not in out
    assert out["structures"][out["steps"][1]["struct"]]["log"] == [["x"]]
    assert rec["structures"].keys() == {"s", "z"}


TRACE = {"sigma": 0.9, "affected_share": 0.75, "token_factor": 2.0, "delta_grad_bits": 640, "partial_cut_bits": 320,
         "grad_cut_bits": 640, "tau_ids": 1.125,
         "tau_other_bits": 0.0, "a_bits": 256.0, "u_in": [[0.0, 0.0], [0.0, 512.0], [0.5, 1024.0], [1.0, 1024.0]],
         "u1_bits": 1024.0, "tail_share": 1 / 3, "frozen_tail_share": 0.0, "update_cut_bits": 1024,
         "data_dependent": [], "carried_scalars": []}


def test_with_structures_carries_the_traced_formalism():
    """A traced structure's σ, affected share, token factor, Δ_grad and partial and gradient cuts go to its steps'
    structure, and its per-token quantities to the record's `formalism` block (`trace` every measured field; u is 0 up
    to t, then u_in); training_slowdown.formalism.of_record reads them: σ |C_j| token gates, the affected share's |C_j| affected
    gates, the steps' tokens times the factor, the cuts as the parts' partial and grad bits (the one bound's reduction
    and update kinds), the step before the first update with no affected gates, and τ at ⌈log2 V⌉."""
    from training_slowdown import formalism

    rec = record(3)
    rec["structures"]["z"] = {"updated": [], "log": []}
    rec["steps"][0]["struct"] = "z"
    for s in rec["steps"]:
        s["by"] = {"fwd/x": 1000}
    out = with_structures(rec, {"s": {"updated": [[50, 0, 800, 16, 16]], "cut_bits": 640, "formalism": TRACE}})
    sid = out["steps"][1]["struct"]
    assert {k: out["structures"][sid][k] for k in formalism.TRACED_STRUCTURE} == {
        "sigma": 0.9, "affected_share": 0.75, "token_factor": 2.0, "delta_grad_bits": 640, "partial_cut_bits": 320,
        "grad_cut_bits": 640}
    f = out["formalism"]
    assert f["a_bits"] == 256.0 and f["u1_bits"] == 1024.0 and f["tail_share"] == 1 / 3 and f["trace"] == TRACE
    u_in = tuple(map(tuple, TRACE["u_in"]))
    assert f["u_in"] == TRACE["u_in"] and f["u"] == [[0.0, 0.0], [1 / 3, 0.0], [1 / 3, 512 + 1024 / 3], [0.5, 1024.0],
                                                     [1.0, 1024.0]]
    block = formalism.of_record(out)
    z, s = block["parts"][0], block["parts"][1]
    assert z["affected_gates"] == 0 and z["tokens"] == 8 and z["delta_grad_bits"] is z["partial_bits"] is None
    assert z["token_gates"] == z["gates"]
    assert s["affected_gates"] == 0.75 * s["gates"] and s["token_gates"] == 0.9 * s["gates"]
    assert s["tokens"] == 2 * 16 and s["delta_grad_bits"] == 640 and (s["partial_bits"], s["grad_bits"]) == (320, 640)
    m = formalism.measure({"formalism": block, "output": {"e_bytes": 100}}, {"vocab": 16, "d_model": 8})
    assert m.tau_bits == 1.125 * 4 and m.a_bits == 256.0 and m.tail_share == 1 / 3 and m.u1_bits == 1024.0
    assert m.u_in == u_in and m.u == formalism.with_tail(u_in, 1 / 3)
    assert {k: m.labels[k] for k in ("tau_bits", "affected_gates", "token_gates", "u", "u_in")} == {
        k: formalism.TRACED[k] for k in ("tau_bits", "affected_gates", "token_gates", "u", "u_in")}


def test_a_traced_step_that_updates_nothing_takes_its_measured_delta_grad():
    """A step before the first update, traced (Δ_grad 0, affected share 0), keeps its own U (none) and takes its trace's
    measured quantities and read-backs: its Δ_grad 0 is a measurement, so Fit binds on the updating step's where it
    failed at every k; traced with no updating step traced, it is left as it was."""
    from training_slowdown import formalism

    rec = record(3)
    rec["structures"]["z"] = {"updated": [], "log": []}
    rec["steps"][0]["struct"] = "z"
    for s in rec["steps"]:
        s["by"] = {"fwd/x": 1000}
    zero = {"affected_share": 0.0, "token_factor": 1.0, "delta_grad_bits": 0, "partial_cut_bits": 0, "grad_cut_bits": 0,
            "u1_bits": 0.0}
    idle = {"updated": [], "cut_bits": 0, "formalism": TRACE | zero}
    s_trace = {"updated": [[50, 0, 800, 16, 16]], "cut_bits": 640, "formalism": TRACE}

    def row(traced):
        return {"formalism": formalism.of_record(with_structures(rec, traced)), "output": {"e_bytes": 100}}

    def fit(traced):
        return formalism.measure(row(traced), {"vocab": 16, "d_model": 8}).fit_delta_grad_bits

    out = with_structures(rec, {"s": s_trace, "z": idle})
    z = out["structures"][out["steps"][0]["struct"]]
    assert z["updated"] == [] and (z["delta_grad_bits"], z["affected_share"], z["token_factor"]) == (0, 0.0, 1.0)
    assert [s.updated_macs for s in annotate(out).steps] == [0, 50 * 8, 50 * 8]
    assert out["formalism"]["a_bits"] == TRACE["a_bits"] and out["formalism"]["u1_bits"] == TRACE["u1_bits"]
    parts = row({"s": s_trace, "z": idle})["formalism"]["parts"]
    assert (parts[0]["delta_grad_bits"], parts[0]["affected_gates"], parts[1]["delta_grad_bits"]) == (0, 0, 640)
    assert fit({"s": s_trace, "z": idle}) == 640 and fit({"s": s_trace}) is None
    clip = {"kind": "a comparison of a read-back float", "where": "train.py:9", "steps": [0, 0], "count": 1}
    dirty = dict(idle, formalism=idle["formalism"] | {"data_dependent": [clip]})
    assert formalism.refusal(row({"s": s_trace, "z": dirty})).startswith("data-dependent")
    assert with_structures(rec, {"z": idle}) == with_structures(rec, {})


def test_with_structures_lists_the_read_backs_of_every_traced_structure_its_steps_take():
    """The record's `trace` takes the most-MACs structure's quantities and every taken structure's read-back lists; a
    structure traced before training_slowdown 4.2 (no lists) leaves them out, so training_slowdown.formalism refuses the row as unchecked."""
    from training_slowdown import formalism

    rec = record(3)
    rec["structures"]["t"] = dict(rec["structures"]["s"])
    rec["steps"][2]["struct"] = "t"
    clip = {"kind": "a comparison of a read-back float", "where": "train.py:9", "steps": [0, 2], "count": 3}
    a = {"updated": [[50, 0, 800, 16, 16]], "formalism": TRACE}
    b = {"updated": [[50, 0, 800, 16, 16]], "formalism": TRACE | {"affected_share": 0.5, "data_dependent": [clip]}}
    trace = with_structures(rec, {"s": a, "t": b})["formalism"]["trace"]
    assert trace["affected_share"] == 0.75 and trace["data_dependent"] == [clip] and trace["carried_scalars"] == []
    assert with_structures(rec, {"s": a, "t": a})["formalism"]["trace"] == TRACE
    old = {"updated": a["updated"], "formalism": {k: v for k, v in TRACE.items() if k not in formalism.READ_BACKS}}
    row = {"formalism": with_structures(rec, {"s": a, "t": old})["formalism"], "output": {"e_bytes": 100}}
    assert not set(formalism.READ_BACKS) & set(row["formalism"]["trace"])
    assert formalism.refusal(row).startswith("read-backs unchecked")
