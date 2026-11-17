"""What the Match driver reads of its workload file (`verity_vllm.pipeline.match_workload`): the longest `max_tokens` and the
engine steps it expects, its engine arguments, the MoE construction and the execution label it declares with the
`enforce_eager` that label implies, and the flags a fold of its log is run with."""
from __future__ import annotations

import json

import pytest

from verity_vllm.pipeline import match_workload as MW


def _file(tmp_path, doc):
    p = tmp_path / "workload.json"
    p.write_text(doc if isinstance(doc, str) else json.dumps(doc))
    return str(p)


@pytest.mark.parametrize("text", ["not json", "[" * 100_000, "{}", "[]", "null", "0", '""', "false"])
def test_a_workload_that_cannot_be_read_or_states_nothing_gives_no_steps_and_each_default(tmp_path, text):
    for path in (None, "", str(tmp_path / "absent.json"), str(tmp_path), _file(tmp_path, text)):
        assert MW._workload_max_tokens(path) is None and MW._workload_expected_steps(path) is None, path
        assert MW._workload_engine_args(path) == {} and MW._workload_moe_construction(path) is None, path
        assert MW._workload_execution(path) is None and MW._workload_enforce_eager(path) is True, path
        assert MW.fold_construction_args(path, []) == [], path


@pytest.mark.parametrize("text, kind", [("[1]", "list"), ('[{"max_tokens": 4}]', "list"), ('"x"', "str"), ("7", "int"), ("1.5", "float"),
                                        ("true", "bool")])
def test_a_workload_that_holds_something_other_than_an_object_is_refused_by_name(tmp_path, text, kind):
    path = _file(tmp_path, text)
    readers = (MW._workload_max_tokens, MW._workload_expected_steps, MW._workload_engine_args, MW._workload_moe_construction,
               MW._workload_execution, MW._workload_enforce_eager, lambda p: MW.fold_construction_args(p, []))
    for read in readers:
        with pytest.raises(SystemExit) as e:
            read(path)
        assert str(e.value) == f"workload {path}: its JSON is a {kind}, not an object"


@pytest.mark.parametrize("requests, longest", [
    ([{"max_tokens": 4}, {"max_tokens": "9"}, {"max_tokens": None}, {}], 9),
    ([{"max_tokens": 0}], 0),
    ([{}, {"prompt": "one that states none"}], None),
    ([], None),
])
def test_the_longest_max_tokens_is_over_the_requests_that_state_one(tmp_path, requests, longest):
    w = _file(tmp_path, {"requests": requests, "schema": "a workload"})
    assert MW._workload_max_tokens(w) == longest and MW._workload_expected_steps(w) == longest


@pytest.mark.parametrize("declared, steps", [({"expected_engine_steps": 12}, 12), ({"expected_engine_steps": "12"}, 12),
                                             ({"expected_engine_steps": 0}, 9), ({"expected_engine_steps": None}, 9), ({}, 9)])
def test_the_expected_engine_steps_are_the_declared_ones_else_the_longest_max_tokens(tmp_path, declared, steps):
    w = _file(tmp_path, {"requests": [{"max_tokens": 9}, {"max_tokens": 2}], **declared})
    assert MW._workload_expected_steps(w) == steps and MW._workload_max_tokens(w) == 9
    assert MW._workload_expected_steps(_file(tmp_path, {"expected_engine_steps": 7})) == 7, "declared by a workload of no requests"


def test_the_engine_arguments_are_the_engine_blocks_own_as_a_new_dict(tmp_path):
    args = {"max_model_len": 512, "tensor_parallel_size": 2}
    w = _file(tmp_path, {"engine": {"args": args, "env": {"NCCL_P2P_DISABLE": "1"}}})
    assert MW._workload_engine_args(w) == args and type(MW._workload_engine_args(w)) is dict
    assert MW._workload_engine_args(_file(tmp_path, {"engine": {"env": {"NCCL_P2P_DISABLE": "1"}}})) == {}


def test_a_declared_moe_construction_is_one_the_frontend_names_and_any_other_stops_the_run_with_them(tmp_path):
    for declared in ("indexed-read", "indexed-read-ordered", "padded"):
        assert MW._workload_moe_construction(_file(tmp_path, {"engine": {"target": {"moe_construction": declared}}})) == declared
    assert MW._workload_moe_construction(_file(tmp_path, {"engine": {"target": {"num_sms": 142}}})) is None
    for other in ("paded", ""):
        with pytest.raises(SystemExit) as e:
            MW._workload_moe_construction(_file(tmp_path, {"engine": {"target": {"moe_construction": other}}}))
        assert e.value.code == (f"workload engine.target.moe_construction {other!r} is not one of "
                                "['indexed-read', 'indexed-read-ordered', 'padded']")


@pytest.mark.parametrize("moe, select, values, flags", [
    ("padded", "split", {"a": 1}, ["--moe-construction", "padded", "--sampler-construction", "split", "--registered-values"]),
    ("padded", None, {}, ["--moe-construction", "padded"]),
    (None, "split", None, ["--sampler-construction", "split"]),
    (None, "", True, ["--registered-values"]),
    (None, None, False, []),
])
def test_a_folds_flags_are_the_declared_moe_construction_then_the_programs_select_then_their_registered_values(
        tmp_path, monkeypatch, moe, select, values, flags):
    asked = []

    def the_select(dirs):
        asked.append(("the select", dirs))
        return select

    def the_values(dirs):
        asked.append(("the values", dirs))
        return values

    monkeypatch.setattr(MW, "program_sampler_construction", the_select)
    monkeypatch.setattr(MW, "program_registered_values", the_values)
    dirs = ["/build/rank0", "/build/rank1"]
    w = _file(tmp_path, {"engine": {"target": {"moe_construction": moe}}} if moe else {})
    assert MW.fold_construction_args(w, dirs) == flags
    assert asked == [("the select", dirs), ("the values", dirs)], "each is asked of the Build's Program directories, once"


@pytest.mark.parametrize("doc, label", [
    ({"execution": "compiled", "sweep": {"execution": "eager", "row_id": "m__w__d__tp1__bi-eager"}}, "compiled"),
    ({"execution": "", "sweep": {"execution": "eager", "row_id": "m__compiled"}}, "eager"),
    ({"execution": 3, "sweep": {"execution": "", "row_id": "m__w__bi-eager"}, "row_id": "m__eager"}, "bi-eager"),
    ({"execution": ["eager"], "sweep": {"execution": None, "row_id": ""}, "row_id": "m__eager"}, "eager"),
    ({"sweep": "not a block", "row_id": "m__compiled"}, "compiled"),
    ({"sweep": {"row_id": "a-row-with-no-label"}, "row_id": "m__eager"}, None),
    ({"sweep": {"row_id": 7}}, None),
    ({"row_id": 7}, None),
    ({"requests": []}, None),
])
def test_the_execution_label_is_the_declared_one_then_the_sweeps_then_the_last_field_of_the_row_id(tmp_path, doc, label):
    assert MW._workload_execution(_file(tmp_path, doc)) == label


def test_enforce_eager_is_the_labels_in_the_served_table_and_a_label_it_lacks_stops_the_run_naming_the_ones_it_has(tmp_path):
    assert MW.ENFORCE_EAGER["eager"] is True and MW.ENFORCE_EAGER["compiled"] is False
    for label, eager in MW.ENFORCE_EAGER.items():
        assert MW._workload_enforce_eager(_file(tmp_path, {"execution": label})) is eager, label
        assert MW._workload_enforce_eager(_file(tmp_path, {"row_id": f"m__w__{label}"})) is eager, label
    w = _file(tmp_path, {"sweep": {"execution": "jit"}})
    with pytest.raises(SystemExit) as e:
        MW._workload_enforce_eager(w)
    assert e.value.code == (f"workload {w}: execution label 'jit' is not modelled by the harness "
                            "(known: ['bi-compiled', 'bi-eager', 'bi-eager-fa2', 'compiled', 'eager'])")
