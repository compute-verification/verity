"""The comparator over a `registered_values` fold: program.json states the supplied layout and each parameter's Input nodes, and a
read of the registered `values` canonicalises by field name, as the derived Program's does."""

from __future__ import annotations

import json

import pytest

from verity_vllm.check.match import batch_decomp as bd
from verity_vllm.check.match import program_compare as C
from verity_vllm.check.match import supplied_inputs as SI
from verity_vllm.observe.fold.fold import fold
from verity_vllm.pipeline.match import persist_program
from tests.observe.test_fold_synthetic import memory, toy_log, toy_profile  # noqa: F401 - fixture


def _oracle(tmp_path, name, registered):
    d = tmp_path / name
    d.mkdir()
    r = fold(toy_log(), toy_profile(registered_values=registered), strict=True)
    assert r.report.unsupported == 0 and not r.errors
    persist_program(r, d, profile_name="toy", log_path="toy")
    return str(d)


def _family(spec):
    return spec.split(":", 1)[-1].split("{", 1)[0].rsplit("_v", 1)[0]


def test_a_registered_fold_reads_its_values_by_name_and_its_weights_as_the_constants_fold_does(memory, tmp_path):
    reg_dir, const_dir = _oracle(tmp_path, "reg", True), _oracle(tmp_path, "const", False)
    pj = json.load(open(f"{reg_dir}/program.json"))
    # the empty `run` Record lowers to no Input node: the values are node 1, the weights node 2, though they are parameters 2 and 3
    assert pj["supplied"]["input_nodes"] == {"prompt": [0], "run": [], "values": [1], "weights": [2]}
    assert pj["supplied"]["registration"]["param"] == 2
    assert json.load(open(f"{const_dir}/program.json"))["supplied"] == {}
    R, K = C.load_oracle(reg_dir, "reg"), C.load_oracle(const_dir, "const")
    assert R.supplied_fields == {1: [(0, "eps", 32), (1, "scale_log2_d8", 32)]} and R.supplied_owner == {1: None}
    assert R.input_base == {2: 0} and R.seed_nodes == set()
    assert bd.weights_input_nodes(R, 1) == {2} and SI.values_nodes(R) == {1}
    assert len(R.inst) == len(K.inst)
    supplied = []
    for i, (a, b) in enumerate(zip(R.inst, K.inst)):
        ca, cb = R.canon(i), K.canon(i)
        assert ca[:len(cb)] == cb, (a["spec"], b["spec"])        # the operands both read: the same sources, weights by field name
        extra = ca[len(cb):]
        if extra:
            assert _family(a["spec"]) in ("RMSNormTriton", "RMSNormFusedCuda", "Attention"), a["spec"]
            supplied += [s for op in extra for s in op]
    assert set(supplied) == {("SUPPLIED", "eps", 0, 1), ("SUPPLIED", "scale_log2_d8", 0, 1)}


def test_supplied_parameters_without_their_input_nodes_are_refused(memory, tmp_path):
    reg_dir = _oracle(tmp_path, "reg", True)
    p = f"{reg_dir}/program.json"
    pj = json.load(open(p))
    del pj["supplied"]["input_nodes"]
    json.dump(pj, open(p, "w"))
    with pytest.raises(ValueError, match="not the parameters' Input nodes"):
        C.load_oracle(reg_dir, "reg")


def _build_dir(tmp_path, name, registered):
    d = tmp_path / name
    d.mkdir()
    sup = {"supplied": {"registration": {"param": 1, "name": "values", "fields": []}, "run": []}} if registered else {}
    (d / "result.json").write_text(json.dumps({"program": {"digest": "d", **sup}, "spec_histogram": {}}))
    return str(d)


def test_the_match_folds_under_registered_values_when_the_builds_programs_register_them(tmp_path):
    """The Match's folds bind the value-input Definitions exactly when the Build's Programs register their values (`result.json`
    `program.supplied`), so a `registered_values` row compares like with like; Programs that disagree are refused by name."""
    from verity_vllm.observe.fold.supplied import program_registered_values
    from verity_vllm.pipeline import cli
    from verity_vllm.pipeline.match import Plan
    reg, const = _build_dir(tmp_path, "reg", True), _build_dir(tmp_path, "const", False)
    assert program_registered_values([reg]) and not program_registered_values([const]) and not program_registered_values([])
    with pytest.raises(SystemExit, match="some Programs register"):
        program_registered_values([reg, const])
    wl = tmp_path / "workload.json"
    wl.write_text(json.dumps({"requests": [{"request_id": "r-0", "prompt_token_ids": [1, 2, 3], "max_tokens": 2}]}))
    base = ["--case", "LLAMA32_1B", "--profile", "generic", "--workload", str(wl), "--out", str(tmp_path / "match")]
    for dirs, want in (([reg], True), ([const], False)):
        plan = Plan(cli.parse("match", [*base, *(a for d in dirs for a in ("--program-dir", d))], {}))
        assert all(("--registered-values" in plan.cmd(s)) == want for s in ("fold", "fold_2"))
    a = cli.parse("match", ["--worker", "fold", "--log", "l", "--profile", "generic", "--out", "o", "--registered-values"], {})
    assert a.registered_values is True
