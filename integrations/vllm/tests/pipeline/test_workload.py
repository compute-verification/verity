"""Row files (`config.RowConfig`): every file under `workloads/` loads and round-trips, the flat v1 layout is refused with its fix,
and the generator writes a TP row's `engine.env` NCCL_P2P_DISABLE and `engine.args.disable_custom_all_reduce` on a SYS target only."""
from __future__ import annotations

import json

import pytest

from verity_vllm.config import TARGETS, VLLM, WORKLOAD_SCHEMA, WORKLOADS, RowConfig, RowEngine, RowId, row_target, row_world
from verity_vllm.engine.profiles.transport import P2P_SYS_TARGETS
from verity_vllm.pipeline import workload as W

GLM = "glm53__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager"


def is_row_id(name: str) -> bool:
    try:
        return str(RowId.parse(name)) == name
    except ValueError:
        return False


FILES = sorted(WORKLOADS.glob("*.json"))
TP_ROWS = [p for p in FILES if is_row_id(p.stem) and row_world(p.stem) > 1]


def _generate(tmp_path, row_id: str) -> dict:
    out = tmp_path / "w.json"
    a = W.Options(row_id=row_id, hf_repo="org/x", revision="0" * 40, batch=1, max_input=256, max_output=32, case="X",
                  lengths_only=True, check=True, out=str(out))
    assert W.main(a) == 0
    return json.loads(out.read_text())


@pytest.mark.parametrize("p", FILES, ids=lambda p: p.stem)
def test_every_workload_file_is_a_row_config_that_round_trips(p):
    doc = json.loads(p.read_text())
    cfg = RowConfig.of(doc)
    assert cfg.to_json() == doc and list(cfg.to_json()) == list(doc)
    assert cfg.engine.vllm == VLLM
    if is_row_id(p.stem):
        assert RowConfig.load(p.stem) == cfg and row_world(cfg.row) == row_world(p.stem)


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
