"""Row files (`config.RowConfig`): every file under `workloads/` loads and round-trips, a row outside it is found only in a directory
`$VERITY_WORKLOADS` names (`config.row_file`), the flat v1 layout is refused with its fix, and the generator writes a TP row's `engine.env` NCCL_P2P_DISABLE and `engine.args.disable_custom_all_reduce` on a SYS target only, and a
`--shared-prefix` row's one prefix and prefix caching."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

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
