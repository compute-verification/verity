"""Row files (`config.RowConfig`): every file under `workloads/` loads and round-trips, a row outside it is found only in a directory
`$VERITY_WORKLOADS` names (`config.row_file`), the flat v1 layout is refused with its fix, and the generator writes a TP row's `engine.env` NCCL_P2P_DISABLE and `engine.args.disable_custom_all_reduce` on a SYS target only, and a
`--shared-prefix` row's one prefix and prefix caching."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from tests.pipeline import tokenizers_stand_in
from verity_vllm.config import (
    ROOT,
    TARGETS,
    VLLM,
    WORKLOAD_SCHEMA,
    WORKLOADS,
    RowConfig,
    RowEngine,
    RowId,
    UsageError,
    done_source,
    row_file,
    row_path,
    row_target,
    row_world,
    workload_files,
)
from verity_vllm.engine.arrivals import realized_arrive_steps
from verity_vllm.engine.profiles.transport import P2P_SYS_TARGETS
from verity_vllm.pipeline import global_program as GP
from verity_vllm.pipeline import workload as W
from verity_vllm.program import prefix_plan as PP
from verity_vllm.records.workload import Workload

GLM = "glm53__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager"


def is_row_id(name: str) -> bool:
    try:
        return str(RowId.parse(name)) == name
    except ValueError:
        return False


FILES = workload_files()
TP_ROWS = [p for p in FILES if is_row_id(p.stem) and row_world(p.stem) > 1]


def _generate(tmp_path, row_id: str, rc: int = 0, **kw) -> dict:
    out = tmp_path / "w.json"
    a = W.Options(row_id=row_id, hf_repo="org/x", revision="0" * 40, batch=1, max_input=256, max_output=32, case="X",
                  lengths_only=True, check=True, out=str(out), **kw)
    assert W.main(a) == rc
    return json.loads(out.read_text()) if rc == 0 else {}


@pytest.mark.parametrize("p", FILES, ids=lambda p: p.stem)
def test_every_workload_file_is_a_row_config_that_round_trips(p):
    doc = json.loads(p.read_text())
    cfg = RowConfig.of(doc)
    assert cfg.to_json() == doc and list(cfg.to_json()) == list(doc)
    assert cfg.engine.vllm == VLLM
    if is_row_id(p.stem):
        assert RowConfig.load(p.stem) == cfg and row_world(cfg.row) == row_world(p.stem)


def test_a_row_outside_workloads_is_the_one_file_of_its_name_in_a_directory_verity_workloads_names(tmp_path, monkeypatch):
    """A demo keeps its own row beside it (demos/vllm/workloads/), which the integration never names: `row run` finds the
    row's file through `$VERITY_WORKLOADS`, its Build, Match and Commit read it from the tree root (`row_path`), and a row id with
    two files is refused."""
    doc = json.loads((WORKLOADS / f"{GLM}.json").read_text())
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "elsewhere.json").write_text(json.dumps(doc))
    assert row_file("elsewhere") == WORKLOADS / "elsewhere.json" and not row_file("elsewhere").exists()
    monkeypatch.setenv("VERITY_WORKLOADS", f"{tmp_path / 'none'}{os.pathsep}{tmp_path / 'a'}")
    assert row_file("elsewhere") == (tmp_path / "a" / "elsewhere.json").resolve()
    assert RowConfig.load("elsewhere") == RowConfig.of(doc)
    assert (ROOT / row_path("elsewhere")).resolve() == row_file("elsewhere")
    assert row_path(GLM) == f"workloads/{GLM}.json"
    (tmp_path / "a" / f"{GLM}.json").write_text(json.dumps(doc))
    with pytest.raises(UsageError, match=f"row {GLM} has a file in more than one"):
        row_file(GLM)


@pytest.mark.parametrize("p", [p for p in TP_ROWS if done_source(json.loads(p.read_text())) == "public-cap"], ids=lambda p: p.stem)
def test_every_tp_row_done_at_its_caps_declares_the_arrivals_it_is_served_at(p):
    """The TP Commit places the Build's manifest at the declared arrive_steps, and the serving loop admits a request early when the
    engine would idle; a gapped declaration leaves identities unbound (52,005 on GLM-5.3 B8 stochastic)."""
    reqs = json.loads(p.read_text())["requests"]
    declared = [int(r.get("arrive_step") or 0) for r in reqs]
    assert declared == realized_arrive_steps(reqs), (f"declare arrive_steps {realized_arrive_steps(reqs)}, where the serving loop admits "
                                                     f"them (pipeline/workload.py ARRIVE_STEP_RULE)")


@pytest.mark.parametrize("p", [p for p in FILES if is_row_id(p.stem)], ids=lambda p: p.stem)
def test_every_row_pins_its_checkpoint(p):
    cfg = RowConfig.load(p.stem)
    pin = cfg.pin(world=row_world(p.stem))
    assert (pin.model, pin.revision, pin.world) == (cfg.model, cfg.engine.revision, row_world(p.stem))


@pytest.mark.parametrize("key, place", [("engine_args_required", "args"), ("env_required", "env"), ("target", "target"),
                                        ("tokenizer_revision", "revision"), ("checkpoint_revision", "revision")])
def test_the_flat_v1_keys_are_refused_with_their_place_in_the_engine_block(key, place):
    doc = {**json.loads((WORKLOADS / f"{GLM}.json").read_text()), key: {}}
    with pytest.raises(ValueError, match=rf"{key} -> engine\.{place}"):
        RowConfig.of(doc)
    with pytest.raises(ValueError, match="v1 layout"):
        RowEngine.of(doc)


def test_another_schema_vllm_or_engine_field_is_refused():
    doc = json.loads((WORKLOADS / f"{GLM}.json").read_text())
    with pytest.raises(ValueError, match="expected 'verity-vllm/workload/v2'"):
        RowConfig.of({**doc, "schema": "verity-vllm/workload/v1"})
    with pytest.raises(ValueError, match="vLLM d9105ea80 only"):
        RowConfig.of({**doc, "engine": {**doc["engine"], "vllm": "0123456789"}})
    with pytest.raises(ValueError, match="unknown field"):
        RowConfig.of({**doc, "engine": {**doc["engine"], "profile": "derived_GLM53_tp8"}})
    with pytest.raises(ValueError, match="disagrees with engine.vllm"):
        RowConfig.of({**doc, "engine": {**doc["engine"], "target": {**doc["engine"]["target"], "vllm_revision": "0123456789"}}})


def test_a_revision_that_is_not_a_full_sha_is_refused():
    doc = json.loads((WORKLOADS / f"{GLM}.json").read_text())
    with pytest.raises(ValueError, match="full commit sha"):
        RowConfig.of({**doc, "engine": {**doc["engine"], "revision": doc["engine"]["revision"][:10]}})


def test_a_tp_row_without_disable_custom_all_reduce_is_refused():
    doc = json.loads((WORKLOADS / f"{GLM}.json").read_text())
    del doc["engine"]["args"]["disable_custom_all_reduce"]
    with pytest.raises(ValueError, match="disable_custom_all_reduce"):
        RowConfig.of(doc)


def test_glm53s_row_config():
    cfg = RowConfig.load(GLM)
    assert (cfg.model, cfg.case, cfg.engine.revision) == ("zai-org/GLM-5.3", "GLM53", "aca966e4e02791568aa6a4ced368624b3d897f42")
    assert cfg.engine.args["tensor_parallel_size"] == 8 and cfg.engine.args["disable_custom_all_reduce"] is False
    assert cfg.engine.env == {"VLLM_BATCH_INVARIANT": "0"} and cfg.engine.target["moe_backend"] == "triton"


@pytest.mark.parametrize("row_id, p2p", [
    ("x__bf16__l40s__tp2__b1__i256__o32__mixed__greedy__bi-eager", {"NCCL_P2P_DISABLE": "1"}),
    ("x__bf16__rtxpro6000__tp8__b1__i256__o32__mixed__greedy__bi-eager", {"NCCL_P2P_DISABLE": "1"}),
    ("x__bf16__h200__tp8__b1__i256__o32__mixed__greedy__bi-eager", {}),
    ("x__bf16__h100__tp2__b1__i256__o32__mixed__greedy__bi-eager", {}),
])
def test_a_tp_workload_requires_nccl_p2p_disable_on_a_sys_target_only(tmp_path, row_id, p2p):
    doc = _generate(tmp_path, row_id)
    assert doc["schema"] == WORKLOAD_SCHEMA and doc["engine"]["env"] == {**p2p, "VLLM_BATCH_INVARIANT": "1"}


def test_a_tp_workload_requires_the_batch_invariance_its_target_declares(tmp_path):
    """A `--target` without batch invariance (vLLM's default kernels) makes the row's engines run VLLM_BATCH_INVARIANT=0, as its
    Build does (`TargetProfile.env`)."""
    target = json.dumps({"compute_capability": [9, 0], "num_sms": 132, "batch_invariant": False})
    doc = _generate(tmp_path, "x__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager", target=target)
    assert doc["engine"]["env"] == {"VLLM_BATCH_INVARIANT": "0"} and doc["engine"]["target"]["batch_invariant"] is False
    more = json.dumps({**json.loads(target), "fp8_block_gemm": "cutlass"})           # a variable the row's label does not set
    assert _generate(tmp_path, "x__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager", target=more)["engine"]["env"] == {
        "VLLM_BATCH_INVARIANT": "0", "VLLM_USE_DEEP_GEMM": "0"}
    assert "env" not in _generate(tmp_path, "x__fp8__h200__tp1__b1__i256__o32__mixed__greedy__eager", target=more)["engine"]


@pytest.mark.parametrize("row_id, rc", [
    ("x__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager", 0),
    ("x__fp8__h200__tp8__b1__i256__o32__mixed__greedy___eager", 0),    # the label is what follows the id's last `__`
    ("x__fp8__h200__tp8__b1__i256__o32__mixed__greedy__bi-eager", 3),
    ("eager", 0),                                                       # an id without `__` is its own label
    ("x-eager", 3),
])
def test_a_target_is_refused_unless_the_row_ids_execution_label_is_its_own(tmp_path, row_id, rc):
    """`--target` on a row whose id ends in another execution label exits 3.  The label is `config.row_execution`'s, the
    field after the id's last `__`, which is how the generator reads it for the row's `engine.env` too."""
    target = json.dumps({"compute_capability": [9, 0], "num_sms": 132, "batch_invariant": False})
    doc = _generate(tmp_path, row_id, rc=rc, target=target)
    assert rc or doc["sweep"]["execution"] == "eager"


class _CharTok:
    path, sha256 = Path("chars.json"), "0" * 64

    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(map(chr, ids))


def _prefix_row(shared_prefix: int, arrivals: str = "staggered") -> dict:
    return W.plan("x__fp8__h200__tp8__b8__i256__o32__prefix-arrivals__greedy__eager", 8, 256, 32, sampling="greedy", arrivals=arrivals,
                  shared_prefix=shared_prefix)


def test_a_shared_prefix_row_starts_every_longer_prompt_with_one_prefix_and_turns_prefix_caching_on():
    p = _prefix_row(128)
    assert W.check_plan(p) == []
    doc = W.workload_doc(p, hf_repo="org/x", revision="0" * 40, case="X", tok=_CharTok())
    reqs = doc["requests"]
    sharers = [r for r in reqs if r["prompt_len"] > 128]
    assert len(sharers) >= 2 and [r["index"] for r in sharers] == W.prefix_sharers(p)
    assert {tuple(r["prompt_token_ids"][:128]) for r in sharers} == {tuple(sharers[0]["prompt_token_ids"][:128])}
    assert all(r["prompt_token_ids"][:128] != sharers[0]["prompt_token_ids"][:128] for r in reqs if r not in sharers)
    assert all(r.get("shared_prefix_tokens") == (128 if r in sharers else None) for r in reqs)
    assert all(len(r["prompt_token_ids"]) == r["prompt_len"] for r in reqs)
    assert doc["engine"]["args"]["enable_prefix_caching"] is True and doc["sweep"]["shared_prefix"] == 128
    assert [r["arrival_offset_s"] for r in reqs] == p["arrival_offsets_s"] and len({r["request_id"] for r in reqs}) == 8
    lengths = W.workload_doc(p, hf_repo="org/x", revision="0" * 40, case="X", tok=None)["requests"]
    assert lengths == [{k: v for k, v in r.items() if k not in ("prompt_token_ids", "prompt_text", "origin")} for r in reqs]   # but for its prompt
    plain = W.workload_doc(_prefix_row(0), hf_repo="org/x", revision="0" * 40, case="X", tok=_CharTok())
    assert "enable_prefix_caching" not in plain["engine"]["args"] and "shared_prefix" not in plain["sweep"]
    assert not any("shared_prefix_tokens" in r for r in plain["requests"])
    assert [r["prompt_token_ids"] for r in plain["requests"] if r["index"] not in W.prefix_sharers(p)] == [
        r["prompt_token_ids"] for r in reqs if r not in sharers]


def test_a_shared_prefix_no_later_request_could_reuse_fails_the_check():
    assert any("arrives after the first one" in v for v in W.check_plan(_prefix_row(128, arrivals="none")))
    assert any("arrives after the first one" in v for v in W.check_plan(_prefix_row(255)))


def test_the_check_names_a_length_out_of_range_on_either_side_and_a_stochastic_row_short_of_distinct_seeds():
    p = _prefix_row(0)
    assert W.check_plan(p) == []
    lengths = [*p["input_lengths"][:-1], 257]
    assert f"input length outside [{p['min_input']}, 256]: {lengths}" in W.check_plan({**p, "input_lengths": lengths})
    (outputs,) = [k for k in p if k.startswith("output_")]                # the plan's list of the most tokens each request may write
    most = [*p[outputs][:-1], 33]
    assert any(v.startswith("output ") and v.endswith(f" outside [1, 32]: {most}") for v in W.check_plan({**p, outputs: most}))
    short = "stochastic row without B distinct per-request sampling seeds"
    for seeds, refused in (([7] * 8, True), (list(range(7)), True), (list(range(8)), False)):
        assert (short in W.check_plan({**p, "sampling": "topp", "sampling_seeds": seeds})) is refused


#: The plan's key for the most tokens each request may write, and the word the check's messages say for them.
(MOST,) = [k for k in _prefix_row(0) if k.startswith("output_")]
NOUN = MOST.removeprefix("output_")
#: A plan of eight requests, written out, that meets every requirement: lengths in [8, 256] and the most tokens out in [2, 32], each
#: list over every quartile of its range.  One changed value below is one violation.
MEETS = {"batch": 8, "max_input": 256, "max_output": 32, "min_input": 8, "min_output": 2, "sampling": "greedy", "sampling_seeds": [],
         "arrivals": "staggered", "arrival_offsets_s": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], "arrive_steps": [0, 1, 2, 3, 4, 5, 6, 7],
         "input_lengths": [256, 9, 77, 141, 203, 37, 111, 180], MOST: [32, 3, 11, 19, 27, 7, 15, 23]}
REUSE = "), so none could reuse its cached blocks (stagger the arrivals or shorten the prefix)"
#: A changed plan, the one violation it is, and the nearest plan that is none.
VIOLATIONS = [
    ({"batch": 9}, "vector lengths 8/8 != batch 9", {}),
    ({"input_lengths": [256, 9, 77, 141, 203, 37, 111]}, "vector lengths 7/8 != batch 8", {}),
    ({MOST: [32, 3, 11, 19, 27, 7, 15]}, "vector lengths 8/7 != batch 8", {}),
    ({"input_lengths": [257, 9, 77, 141, 203, 37, 111, 180]}, "input length outside [8, 256]: [257, 9, 77, 141, 203, 37, 111, 180]", {}),
    ({MOST: [33, 3, 11, 19, 27, 7, 15, 23]}, f"output {NOUN[:-1]} outside [1, 32]: [33, 3, 11, 19, 27, 7, 15, 23]", {}),
    ({MOST: [32, 0, 3, 11, 19, 27, 7, 15]}, f"output {NOUN[:-1]} outside [1, 32]: [32, 0, 3, 11, 19, 27, 7, 15]",
     {MOST: [32, 1, 3, 11, 19, 27, 7, 15]}),
    ({"input_lengths": [252, 9, 77, 141, 203, 37, 111, 180]}, "no request within 3 tokens of max_input",
     {"input_lengths": [253, 9, 77, 141, 203, 37, 111, 180]}),
    ({"input_lengths": [256, 9, 77, 141, 203, 9, 77, 141]}, "only 5 distinct input lengths (need >= 6)",
     {"input_lengths": [256, 9, 77, 141, 203, 37, 77, 141]}),
    ({"input_lengths": [256, 12, 77, 141, 203, 37, 111, 180]}, "no request near the minimum input length",
     {"input_lengths": [256, 11, 77, 141, 203, 37, 111, 180]}),
    ({"input_lengths": [256, 9, 77, 100, 203, 37, 111, 200]}, "input quartiles covered [0, 1, 3] != [0,1,2,3]",
     {"input_lengths": [256, 9, 77, 132, 203, 37, 111, 200]}),
    ({"input_lengths": [256, 8, 64, 128, 192, 16, 141, 77]}, "only 2 awkward non-power-of-two input lengths (need several >= 3)",
     {"input_lengths": [256, 8, 64, 128, 192, 203, 141, 77]}),
    ({MOST: [32, 3, 11, 19, 27, 3, 11, 19]}, f"only 5 distinct output {NOUN} (need >= 6)", {MOST: [32, 3, 11, 19, 27, 7, 11, 19]}),
    ({MOST: [32, 5, 11, 19, 27, 7, 15, 23]}, f"no very short output {NOUN[:-1]}", {MOST: [32, 4, 11, 19, 27, 7, 15, 23]}),
    ({MOST: [28, 3, 11, 19, 27, 7, 15, 23]}, f"no output {NOUN[:-1]} equal/near max_output", {MOST: [29, 3, 11, 19, 27, 7, 15, 23]}),
    ({MOST: [32, 3, 4, 5, 6, 7, 30, 31]}, f"output {NOUN} not spread across the range (quartiles [0, 3])",
     {MOST: [32, 3, 4, 5, 6, 10, 30, 31]}),
    ({"sampling": "topp", "sampling_seeds": [1, 2, 3, 4, 5, 6, 7, 7]}, "stochastic row without B distinct per-request sampling seeds",
     {"sampling": "topp", "sampling_seeds": [1, 2, 3, 4, 5, 6, 7, 8]}),
    ({"sampling": "topp", "sampling_seeds": [1, 2, 3, 4, 5, 6, 7]}, "stochastic row without B distinct per-request sampling seeds", {}),
    ({"arrival_offsets_s": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]},
     "staggered arrival schedule malformed: [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]",
     {"arrivals": "none", "arrival_offsets_s": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]}),
    ({"arrival_offsets_s": [0.0, 0.2, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7]},
     "staggered arrival schedule malformed: [0.0, 0.2, 0.1, 0.3, 0.4, 0.5, 0.6, 0.7]", {}),
    ({"arrival_offsets_s": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2]},
     "staggered arrival schedule malformed: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2]",
     {"arrival_offsets_s": [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3]}),
    ({"shared_prefix": 100, "arrive_steps": [5, 0, 1, 5, 5, 2, 5, 5]},
     "shared prefix 100: no request longer than it arrives after the first one (arrive steps [5, 5, 5, 5, 5]" + REUSE,
     {"shared_prefix": 100, "arrive_steps": [5, 0, 1, 5, 5, 2, 5, 6]}),
    ({"shared_prefix": 203}, "shared prefix 203: no request longer than it arrives after the first one (arrive steps [0]" + REUSE,
     {"shared_prefix": 202}),
]


@pytest.mark.parametrize("changed,violation,nearest", VIOLATIONS,
                         ids=[f"{n}-{v.split(':')[0].split(' [')[0]}" for n, (_, v, _) in enumerate(VIOLATIONS)])
def test_the_check_names_each_requirement_a_plan_does_not_meet_and_no_other(changed, violation, nearest):
    assert W.check_plan(MEETS) == [] and W.check_plan({**MEETS, **nearest}) == []
    assert W.check_plan({**MEETS, **changed}) == [violation]


def test_the_check_holds_a_batch_under_eight_to_its_ranges_and_schedule_alone_and_names_a_larger_ones_violations_in_order():
    under = [256, 7, 77, 141, 203, 37, 111, 180]                          # a length under the range is in no quartile of it either
    assert W.check_plan({**MEETS, "input_lengths": under}) == [f"input length outside [8, 256]: {under}",
                                                               "input quartiles covered [-1, 0, 1, 2, 3] != [0,1,2,3]"]
    same = {"input_lengths": [256] * 8, MOST: [32] * 8, "arrivals": "none", "arrival_offsets_s": [0.0] * 8}
    assert W.check_plan({**MEETS, **same}) == [
        "only 1 distinct input lengths (need >= 6)", "no request near the minimum input length", "input quartiles covered [3] != [0,1,2,3]",
        "only 0 awkward non-power-of-two input lengths (need several >= 3)", f"only 1 distinct output {NOUN} (need >= 6)",
        f"no very short output {NOUN[:-1]}", f"output {NOUN} not spread across the range (quartiles [3])"]
    seven = {**MEETS, **{k: v[:7] if isinstance(v, list) else v for k, v in same.items()}, "batch": 7, "arrive_steps": [0] * 7}
    assert W.check_plan(seven) == [] and W.check_plan({**seven, "input_lengths": [252] * 7}) == ["no request within 3 tokens of max_input"]
    two = {**MEETS, "batch": 2, "input_lengths": [256, 9], MOST: [32, 3], "arrival_offsets_s": [0.0, 0.0]}
    assert W.check_plan(two) == ["staggered arrival schedule malformed: [0.0, 0.0]"] and W.check_plan({**two, "arrival_offsets_s": [0.0, 0.1]}) == []
    assert W.check_plan({**MEETS, "batch": 1, "input_lengths": [256], MOST: [32], "arrival_offsets_s": [0.5]}) == []   # one request: no schedule
    assert W.check_plan({**MEETS, **same, "max_output": 2, MOST: [2] * 8})[4:] == [f"only 1 distinct output {NOUN} (need >= 6)",
                                                                                 f"output {NOUN} not spread across the range (quartiles [0])"]


@pytest.mark.parametrize("variant,params", [("stoch-t0.8-p0.95", {"temperature": 0.8, "top_p": 0.95, "top_k": 0}),
                                            ("stoch-t0.8-p1", {"temperature": 0.8, "top_p": 1.0, "top_k": 0})])
def test_a_stochastic_rows_plan_seeds_each_request_from_the_row_id_and_its_file_states_the_seeds_and_the_variant(variant, params):
    row = f"x__fp8__h200__tp8__b8__i256__o32__mixed__{variant}__eager"
    p = W.plan(row, 8, 256, 32, sampling=variant, arrivals="none")
    of_row = hashlib.sha256(row.encode()).digest()
    seeds = [int.from_bytes(hashlib.sha256(of_row + b"/sampling/" + str(i).encode()).digest()[:4], "little") % 2 ** 31 for i in range(8)]
    assert p["sampling_seeds"] == seeds and len(set(seeds)) == 8 and p["seed"] == of_row.hex() and p["sampling_params"] == params
    assert W.check_plan(p) == [] and p == W.plan(row, 8, 256, 32, sampling=variant, arrivals="none")
    greedy = {"temperature": 0.0, "top_p": 1.0, "top_k": 0}
    for tok in (_CharTok(), None):                                         # with the prompts' tokens, and the lengths alone
        doc = W.workload_doc(p, hf_repo="org/x", revision="0" * 40, case="X", tok=tok)
        assert [r["seed"] for r in doc["requests"]] == seeds == doc["seeds"] == doc["sweep"]["sampling_seeds"]
        assert doc["variants"] == {"greedy": greedy, variant: params} and doc["sampling"] == {"variant": variant, "seed": None, "n": 1}
        assert doc["sweep"]["lengths_only"] is (tok is None) and all(("prompt_token_ids" in r) is (tok is not None) for r in doc["requests"])
    plain = W.workload_doc(W.plan(row.replace(variant, "greedy"), 8, 256, 32, sampling="greedy", arrivals="none"), hf_repo="org/x",
                           revision="0" * 40, case="X", tok=_CharTok())
    assert plain["seeds"] == [] and plain["variants"] == {"greedy": greedy} and not any("seed" in r for r in plain["requests"])


def test_a_range_of_one_value_plans_that_value_for_every_request():
    row = "x__fp8__h200__tp8__b8__i256__o{}__mixed__greedy__eager"
    for most in (1, 2):                                                         # under the least a request may write, and at it
        p = W.plan(row.format(most), 8, 256, most, sampling="greedy", arrivals="none")
        assert p[MOST] == [most] * 8 and p["min_output"] == most and len(set(p["input_lengths"])) >= 6
    assert W.plan(row.format(32), 8, 1, 32, sampling="greedy", arrivals="none")["input_lengths"] == [1] * 8
    assert [W.plan(row.format(32), 8, longest, 32, sampling="greedy", arrivals="none")["min_input"] for longest in (31, 32)] == [1, 8]


def test_a_batch_of_two_is_staggered_as_a_larger_one_is():
    p = W.plan("x__fp8__h200__tp8__b2__i256__o32__mixed-arrivals__greedy__eager", 2, 256, 32, sampling="greedy", arrivals="staggered")
    assert p["arrival_offsets_s"][0] == 0.0 < p["arrival_offsets_s"][1] and p["arrive_steps"][0] == 0 < p["arrive_steps"][1]
    assert W.check_plan(p) == []


#: What a plan decides: the values of a row's sweep block that the row id alone gives.
PLANNED = ("schema", "seed", "batch", "max_input", "max_output", "min_input", "min_output", "sampling", "sampling_params", "arrivals",
           "input_lengths", MOST, "sampling_seeds", "arrival_offsets_s", "arrive_steps", "seed_rule")


def _made_under_the_generators_length_rule():
    """Each stored row the generator made under its length rule of today: its name, its file and its row id's plan."""
    for p in FILES:
        stored = json.loads(p.read_text())
        sweep = stored.get("sweep") or {}
        if "row_id" not in sweep:
            continue
        plan = W.plan(sweep["row_id"], sweep["batch"], sweep["max_input"], sweep["max_output"], sampling=sweep["sampling"],
                      arrivals=sweep["arrivals"], shared_prefix=sweep.get("shared_prefix", 0))
        if sweep["length_rule"] == plan["length_rule"]:                         # else a row written by hand, or under an earlier rule
            yield p.name, stored, plan


def test_every_stored_workload_made_under_the_generators_length_rule_states_its_row_ids_plan():
    """The generator still plans the stored rows as it planned them: a row's file is what its Builds and Commits were made from."""
    made = 0
    for name, stored, plan in _made_under_the_generators_length_rule():
        assert {k: stored["sweep"][k] for k in PLANNED} == {k: plan[k] for k in PLANNED}, name
        made += 1
    assert made >= 50


#: What a row's file states of its plan outside the sweep block: all but its prose, its engine and what its tokenizer gives.
WRITTEN = ("schema", "model", "case", "seed", "variants", "seeds", "sampling", "serving_profile", "context_class", "length_seed",
           "prompt_seed", "replicate", "length_rule", "expected_engine_steps", "prompt_lengths", "max_tokens")
#: What the generator alone sets of a row's engine: the longest context, a TP row's world, and the prefix cache of a shared prefix.
ENGINE = ("max_model_len", "tensor_parallel_size", "disable_custom_all_reduce", "enable_prefix_caching")
TOKENIZED = ("prompt_token_ids", "prompt_text", "origin")


def _origins(requests: list[dict], told: list[dict]) -> list[tuple[str, list[str], list[str]]]:
    """Each prompt's corpus, its paragraphs and its shared prefix's, as far as ``told`` lists them: a tokenizer of fewer ids to a
    paragraph changes only how many paragraphs a prompt of its length runs over."""
    return [(r["origin"]["corpus"], r["origin"]["paragraphs"][:len(t["origin"]["paragraphs"])],
             r["origin"].get("shared_prefix_paragraphs", [])[:len(t["origin"].get("shared_prefix_paragraphs", []))])
            for r, t in zip(requests, told, strict=True)]


def test_every_stored_workload_made_under_the_generators_length_rule_is_its_plan_as_the_generator_writes_it():
    """The generator still writes the stored rows' plans as it wrote them: the file's own fields and each request's outside what a
    tokenizer gives, a workload of record's requests done only at their public lengths; the context and the world its engine is
    built for; and the paragraphs its prompts start at."""
    made = of_record = worlds = shared = 0
    for name, stored, plan in _made_under_the_generators_length_rule():
        doc = W.workload_doc(plan, hf_repo=stored["model"], revision=stored["engine"]["revision"], case=stored["case"], tok=None)
        assert {k: stored[k] for k in WRITTEN} == {k: doc[k] for k in WRITTEN}, name
        assert [{k: v for k, v in r.items() if k not in TOKENIZED} for r in stored["requests"]] == doc["requests"], name
        assert {k: stored["engine"]["args"].get(k) for k in ENGINE} == {k: doc["engine"]["args"].get(k) for k in ENGINE}, name
        assert stored["sweep"].get("tp") == doc["sweep"].get("tp"), name
        told = W.build_requests(_CharTok(), plan)                               # an id a character: the most ids a paragraph gives
        assert _origins(stored["requests"], told) == _origins(told, told), name
        made, of_record, worlds = made + 1, of_record + ("ignore_eos" in doc["sampling"]), worlds + ("tp" in doc["sweep"])
        shared += any(prefix for *_, prefix in _origins(told, told))
    assert made >= 50 and of_record and worlds and shared                       # each kind of row is among them


def test_a_prompt_is_its_shared_prefix_then_the_corpus_from_its_paragraph_on_a_space_before_each_cut_at_its_length():
    paras = W.corpus_paragraphs()
    at = {f"{para['source']} \N{PILCROW SIGN}{para['index']}": j for j, para in enumerate(paras)}
    text = {name: paras[j]["text"] for name, j in at.items()}
    plan = W.plan("x__fp8__h200__tp8__b8__i4096__o32__prefix-arrivals__greedy__eager", 8, 4096, 32, sampling="greedy", arrivals="staggered",
                  shared_prefix=600)
    told = W.build_requests(_CharTok(), plan)
    for r in told:
        prefix, own = r["origin"].get("shared_prefix_paragraphs", []), r["origin"]["paragraphs"]
        shared = " ".join(text[name] for name in prefix)[:r.get("shared_prefix_tokens", 0)]
        assert r["prompt_text"] == " ".join([shared] * bool(shared) + [text[name] for name in own])[:r["prompt_len"]]
        for names in (prefix, own):                                             # one paragraph after another, around the corpus's end
            assert [at[name] for name in names] == [(at[names[0]] + k) % len(paras) for k in range(len(names))]
    assert {len(r["origin"]["paragraphs"]) > 2 for r in told} == {"shared_prefix_tokens" in r for r in told} == {True, False}


def test_a_batch_larger_than_the_corpus_starts_each_further_prompt_fourteen_paragraphs_after_the_one_it_wraps_to():
    paras = W.corpus_paragraphs()
    at = {f"{para['source']} \N{PILCROW SIGN}{para['index']}": j for j, para in enumerate(paras)}
    batch = len(paras) + 3
    p = W.plan(f"x__fp8__h200__tp8__b{batch}__i256__o32__mixed__greedy__eager", batch, 256, 32, sampling="greedy", arrivals="none")
    first = [at[r["origin"]["paragraphs"][0]] for r in W.build_requests(_CharTok(), p)]
    assert sorted(first[:len(paras)]) == list(range(len(paras)))               # every paragraph starts one prompt, then
    assert first[len(paras):] == [(j + 14) % len(paras) for j in first[:3]]


def test_a_shared_prefix_rows_manifest_requests_state_each_warm_requests_cached_rows_and_their_producers():
    """`verity_vllm.pipeline.global_program.manifest_requests` on a row that serves with the prefix cache: each request's cached
    rows and the Program ids of the requests that produced them, as the prefix plan gives them; a row without it states neither."""
    doc = W.workload_doc(_prefix_row(128), hf_repo="org/x", revision="0" * 40, case="X", tok=_CharTok())
    rows = GP.manifest_requests(Workload.of(doc, "w.json"))
    plan = PP.plan(doc)
    assert [(r["cached"], r["producers"]) for r in rows] == [(p["cached"], [f"r{q}" for q in p["producers"]]) for p in plan]
    assert any(r["cached"] for r in rows) and not all(r["cached"] for r in rows)
    plain = W.workload_doc(_prefix_row(0), hf_repo="org/x", revision="0" * 40, case="X", tok=_CharTok())
    assert not any("cached" in r or "producers" in r for r in GP.manifest_requests(Workload.of(plain, "w.json")))


def test_a_tp_workload_runs_batch_invariant_only_on_a_batch_invariant_row(tmp_path):
    assert _generate(tmp_path, "x__bf16__h200__tp8__b1__i256__o32__mixed__greedy__eager")["engine"]["env"] == {"VLLM_BATCH_INVARIANT": "0"}


def test_a_reason_is_written_for_a_served_switch_and_refused_for_anything_else(tmp_path):
    row = "x__bf16__h100__tp1__b1__i256__o32__mixed__greedy__bi-eager"
    doc = _generate(tmp_path, row, reason=["moe_backend=the Triton experts the rule binds"])
    assert doc["engine"]["reasons"] == {"moe_backend": "the Triton experts the rule binds"}
    assert RowConfig.of(doc).engine.reasons == doc["engine"]["reasons"]
    for bad in (["VLLM_LOGGING_LEVEL=quiet"], ["moe_backend="], ["moe_backend"]):
        _generate(tmp_path, row, rc=2, reason=bad)


def test_a_target_serving_the_int8_attention_states_why_its_backend_is_custom(tmp_path):
    from verity_vllm.engine import served
    from verity_vllm.program.frontend.target_profile import TargetProfile

    row = "x__int8__rtxpro6000__tp1__b1__i256__o32__mixed__greedy__bi-eager"
    t = {"int8_exact": "all", "compute_capability": [12, 0], "num_sms": 188, "async_scheduling": False}
    doc = _generate(tmp_path, row, target=json.dumps(t))
    assert doc["engine"]["args"]["attention_backend"] == "CUSTOM"
    assert doc["engine"]["reasons"] == {"attention_backend": W.INT8_ATTENTION_REASON}
    target = TargetProfile.from_json({**TargetProfile.default().to_json(), **t})
    assert served.declared({"engine": doc["engine"]}, target)["attention_backend"]["reason"] == W.INT8_ATTENTION_REASON
    own = _generate(tmp_path, row, target=json.dumps(t), reason=["attention_backend=the row's own"])
    assert own["engine"]["reasons"] == {"attention_backend": "the row's own"}
    linears = _generate(tmp_path, row, target=json.dumps({**t, "int8_exact": "linears"}))
    assert "reasons" not in linears["engine"] and "attention_backend" not in linears["engine"]["args"]


@pytest.mark.parametrize("target", sorted(TARGETS))
def test_a_tp_workload_states_disable_custom_all_reduce_for_every_target(tmp_path, target):
    args = _generate(tmp_path, f"x__bf16__{target}__tp2__b1__i256__o32__mixed__greedy__bi-eager")["engine"]["args"]
    assert args["disable_custom_all_reduce"] is (target in P2P_SYS_TARGETS)
    assert "disable_custom_all_reduce" not in _generate(tmp_path, f"x__bf16__{target}__tp1__b1__i256__o32__mixed__greedy__bi-eager")[
        "engine"]["args"]


def test_every_tp_row_disables_p2p_and_the_custom_all_reduce_exactly_on_a_sys_target():
    """What each TP row's engines run with (`row_tp.TpRow.p2p_env`, `custom_all_reduce_off`): NCCL_P2P_DISABLE=1 and the custom
    all-reduce off on every L40S and RTX PRO 6000 row, on no H box's."""
    assert GLM in {p.stem for p in TP_ROWS}
    for p in TP_ROWS:
        eng = RowConfig.load(p).engine
        sys_box = row_target(p.stem) in P2P_SYS_TARGETS
        assert eng.env.get("NCCL_P2P_DISABLE") == ("1" if sys_box else None), p.name
        assert eng.args["disable_custom_all_reduce"] is sys_box, p.name


def test_the_generator_logs_what_it_did_each_message_at_its_level_and_keeps_its_exit_codes(tmp_path, monkeypatch, caplog):
    """What `verity_vllm.pipeline.workload` says goes to the log: the `--check` line and the written file's note at INFO, a failed
    self-check without `--check` at WARNING (the file is still written), and a refusal at ERROR beside its exit code."""
    row = "x__bf16__h100__tp1__b1__i256__o32__mixed__greedy__bi-eager"
    opts = dict(row_id=row, hf_repo="org/x", revision="0" * 40, batch=1, max_input=256, max_output=32, case="X", out=str(tmp_path / "w.json"))
    a = W.Options(**opts)
    plan = W.plan(row, 1, 256, 32, sampling=a.sampling, arrivals=a.arrivals, shared_prefix=0)
    bad = ["a request never arrives", "a request has no output"]

    def said() -> list[tuple[str, str]]:
        got = [(r.levelname, r.getMessage()) for r in caplog.records if r.name == "verity_vllm.pipeline.workload"]
        caplog.clear()
        return got

    def checked(record: tuple[str, str], verdict: str) -> bool:
        """The `--check` line: the row and its plan's lengths first, the verdict last."""
        level, line = record
        return (level == "INFO" and line.startswith(f"[check] {row}: B=1 lens={plan['input_lengths']} ")
                and line.endswith(f" steps={plan['arrive_steps']} -> {verdict}"))

    doc = _generate(tmp_path, row)
    note = ("INFO", f"{tmp_path / 'w.json'}: {doc['note'][:220]}")
    check, wrote = said()
    assert checked(check, "PASS") and wrote == note
    _generate(tmp_path, row, rc=2, reason=["moe_backend"])
    check, refused = said()
    assert checked(check, "PASS")
    assert refused == ("ERROR", f"--reason ['moe_backend']: a reason is SWITCH=TEXT for a switch of {sorted(W.served.DEFAULTS)}")
    _generate(tmp_path, "x-eager", rc=3, target=json.dumps({"compute_capability": [9, 0], "num_sms": 132, "batch_invariant": False}))
    assert said()[-1] == ("ERROR", "row id execution suffix 'x-eager' != the target's execution label 'eager'")
    monkeypatch.setattr(W, "find_tokenizer", lambda repo, revision: None)
    assert W.main(W.Options(**opts)) == 2
    assert said() == [("ERROR", f"tokenizer.json for org/x@{'0' * 40} not found under {W.hf_home()} — pass --tokenizer or use --lengths-only")]
    monkeypatch.setattr(W, "check_plan", lambda plan: bad)
    (tmp_path / "w.json").unlink()
    _generate(tmp_path, row, rc=1)
    (check,) = said()
    assert checked(check, "FAIL: " + "; ".join(bad)) and not (tmp_path / "w.json").exists()
    assert W.main(W.Options(**opts, lengths_only=True)) == 0 and (tmp_path / "w.json").exists()
    assert said() == [("WARNING", "§12 self-check: " + "; ".join(bad)), note]


REV = "ab" * 20


def _snapshot(hf_home: Path, revision: str = REV, tokenizer: str | None = '{"model": "chars"}') -> Path:
    """A snapshot of `org/x` in the hub under `hf_home`, with a tokenizer file or none."""
    snap = hf_home / "hub" / "models--org--x" / "snapshots" / revision
    snap.mkdir(parents=True)
    if tokenizer is not None:
        (snap / "tokenizer.json").write_text(tokenizer)
    return snap


def test_a_models_tokenizer_is_the_one_in_the_hubs_snapshot_of_the_revision_under_hf_home_or_the_checkpoints_files(tmp_path, monkeypatch):
    hub = tmp_path / "hf"
    monkeypatch.setenv("HF_HOME", str(hub))
    assert W.hf_home() == hub and W.find_tokenizer("org/x", REV) is None                      # no snapshot of the model at all
    _snapshot(hub, "00" + REV[2:])
    snap = _snapshot(hub, tokenizer=None)
    assert W.find_tokenizer("org/x", REV) is None                                             # the revision's snapshot holds no tokenizer
    (snap / "tokenizer.json").write_text("{}")
    assert W.find_tokenizer("org/x", REV) == W.find_tokenizer("org/x", REV[:7]) == snap / "tokenizer.json"
    assert W.find_tokenizer("org/y", REV) is None and W.find_tokenizer("org/x", "cd" * 20) is None
    monkeypatch.delenv("HF_HOME")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(W, "CHECKPOINTS", tmp_path / "checkpoints.json")
    cache = tmp_path / "home" / ".cache" / "huggingface"
    assert W.hf_home() == cache                                                               # no checkpoints file to read
    for text, home in (("{", cache), ("{}", cache), (json.dumps({"hf_home": str(tmp_path / "nowhere")}), cache),
                       (json.dumps({"hf_home": str(hub)}), hub)):
        (tmp_path / "checkpoints.json").write_text(text)
        assert W.hf_home() == home
    assert W.find_tokenizer("org/x", REV) == snap / "tokenizer.json"


def test_a_models_case_is_the_one_named_then_the_adapters_own_for_its_repository_then_the_checkpoints_files_role(tmp_path, monkeypatch):
    from verity_vllm.engine.profiles.cases import CASES
    own = next(k for k, v in CASES.items() if v.get("repo"))
    assert W.case_for(CASES[own]["repo"], REV, None) == own and W.case_for(CASES[own]["repo"], REV, "MINE") == "MINE"
    monkeypatch.setattr(W, "CHECKPOINTS", tmp_path / "checkpoints.json")
    assert W.case_for("org/x", REV, None) is None                                             # no checkpoints file to read
    (tmp_path / "checkpoints.json").write_text("{")
    assert W.case_for("org/x", REV, None) is None
    (tmp_path / "checkpoints.json").write_text(json.dumps({"checkpoints": [
        {"role": "two words", "repo": "org/x", "revision": REV}, {"repo": "org/x", "revision": REV},
        {"role": "OTHER", "repo": "org/x", "revision": "cd" * 20}, {"role": "X_1B", "repo": "org/x", "revision": REV}]}))
    assert W.case_for("org/x", REV, None) == W.case_for("org/x", REV[:7], None) == "X_1B"     # a role that is one identifier
    assert W.case_for("org/x", "cd" * 20, None) == "OTHER" and W.case_for("org/x", "ef" * 20, None) is None
    assert W.case_for("org/y", REV, None) is None


def test_the_generator_writes_each_prompt_in_the_ids_of_the_models_tokenizer_and_says_which_file_it_read(tmp_path, monkeypatch):
    """`workload.main` with a tokenizer: `tokenizer.json` of the revision's snapshot under `$HF_HOME`, or the snapshot or file
    `--tokenizer` names; read with the `tokenizers` library (a stand-in here), no special token added or dropped."""
    tokenizer = tokenizers_stand_in.install(monkeypatch)
    snap = _snapshot(tmp_path)
    monkeypatch.setenv("HF_HOME", str(tmp_path))
    out = tmp_path / "rows" / "of" / "w.json"                                                  # in a directory that is made for it
    opts = W.Options(row_id="x__fp8__h200__tp8__b8__i256__o32__prefix-arrivals__greedy__eager", hf_repo="org/x", revision=REV, batch=8,
                     max_input=256, max_output=32, case="X", arrivals="staggered", out=str(out))
    assert W.main(opts) == 0
    doc = json.loads(out.read_text())
    assert out.read_text() == json.dumps(doc, indent=1, ensure_ascii=False) + "\n" and not out.read_text().isascii()
    reqs = doc["requests"]
    assert [len(r["prompt_token_ids"]) for r in reqs] == [r["prompt_len"] for r in reqs] == doc["sweep"]["input_lengths"]
    assert all(r["prompt_text"] == "".join(map(chr, r["prompt_token_ids"])) for r in reqs) and len({r["prompt_text"] for r in reqs}) == 8
    assert tokenizer.asked == {("encode", False), ("decode", False)}
    sha = hashlib.sha256(b'{"model": "chars"}').hexdigest()
    assert doc["tokenizer_note"].startswith(f"prompt_token_ids from tokenizer.json sha256 {sha[:16]} (") and doc["sweep"]["lengths_only"] is False
    monkeypatch.setenv("HF_HOME", str(tmp_path / "elsewhere"))
    for named in (snap, snap / "tokenizer.json"):
        out.unlink()
        assert W.main(W.Options(**{**vars(opts), "tokenizer": str(named)})) == 0 and json.loads(out.read_text()) == doc
    with pytest.raises(FileNotFoundError, match=f"^tokenizer.json not found at {tmp_path / 'tokenizer.json'}$"):
        W.Tok(tmp_path)
