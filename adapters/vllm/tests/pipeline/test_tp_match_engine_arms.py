"""`verity_vllm.pipeline.tp.match`'s two engine arms.  `--arm capture` builds the TP engine with the rank worker's extension, asks
every rank for its probe, warms up, installs the collective recorder, runs the workload and closes the recorder; `--arm control`
builds the same engine with no extension and asks the ranks nothing.  Each writes `topology.json` (the engine's kwargs, the
process's NCCL, vLLM and cuBLAS variables with a credential by name only, the ranks' probes) and `tokens.json` (each request's
generated ids and what the recorder said at its close).  CPU-only: `main` runs over the stand-in LLM of two ranks of
`tests.pipeline.test_tp_target`."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from tests.pipeline import test_tp_target as TT
from verity_vllm import settings
from verity_vllm.engine import engine_profile as prof
from verity_vllm.engine import vllm_adapter as va
from verity_vllm.pipeline import cli
from verity_vllm.pipeline.tp import match as M
from verity_vllm.pipeline.tp.rank_worker import EXTENSION

stand_ins, vllm = TT.stand_ins, TT.vllm
FAMILIES = ("NCCL_", "VLLM_", "CUBLAS")
DUMMY = "dummy"


def _arm(tmp_path: Path, monkeypatch, stand_ins, arm: str, *extra: str) -> tuple[TT._LLM, Path, list[list[str]]]:
    """(the LLM, the arm's directory, the rpcs made before each run of requests) after `tp-match --arm ARM`."""
    llm = TT._LLM()
    stand_ins(llm, None)
    before: list[list[str]] = []
    run = va.run_requests

    def run_requests(eng: Any, reqs: Any, sampling: Any) -> Any:
        before.append(llm.methods())
        return run(eng, reqs, sampling)

    monkeypatch.setattr(va, "run_requests", run_requests)
    out = tmp_path / arm
    args = ["--arm", arm, "--workload", str(tmp_path / "w.json"), "--out", str(out), "--case", "C", *extra]
    assert M.main(cli.parse("tp-match", args, environ={})) == 0
    return llm, out, before


def _read(out: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    return json.loads((out / "topology.json").read_text()), json.loads((out / "tokens.json").read_text())


def test_the_capture_arm_builds_the_engine_with_the_extension_and_records_between_the_warm_up_and_the_close(tmp_path, monkeypatch, stand_ins):
    llm, out, before = _arm(tmp_path, monkeypatch, stand_ins, "capture", "--engine-arg", "max_num_batched_tokens=2048", "--store-steps", "3")
    assert llm.kw == {"replay": False, "seed": prof.ENGINE_SEED, "enforce_eager": True, "tensor_parallel_size": 2, "enable_prefix_caching": False,
                      "max_num_seqs": 8, "max_num_batched_tokens": 2048, "worker_extension_cls": EXTENSION}
    assert llm.rpcs == [("tp2_probe", {}), ("tp2_coll_install", {"out_dir": str(out), "store_steps": 3}), ("tp2_coll_close", {})]
    assert before == [["tp2_probe"], ["tp2_probe", "tp2_coll_install"]]         # the warm-up runs before the recorder is installed
    topo, tok = _read(out)
    answers = llm.collective_rpc("tp2_probe")
    assert (topo["arm"], topo["tp"], topo["engine_kwargs"], topo["ranks"]) == ("capture", 2, llm.kw, answers)
    assert sorted(topo) == ["arm", "build_s", "engine_kwargs", "env", "ranks", "tp"] and isinstance(topo["build_s"], float)
    assert (tok["arm"], tok["tp"], tok["case"], tok["model"], tok["recorder"]) == ("capture", 2, "C", "org/model", answers)
    assert tok["timing"] == {"wall_s": 0.5, "engine_steps": 2} and tok["argv"] == sys.argv and isinstance(tok["run_s"], float)
    assert tok["requests"] == [{"request_id": "q0", "prompt_token_ids": [1, 2, 3], "max_tokens": 2, "generated_token_ids": [5, 6]}]


def test_the_control_arm_builds_the_same_engine_with_no_extension_and_asks_the_ranks_nothing(tmp_path, monkeypatch, stand_ins):
    llm, out, before = _arm(tmp_path, monkeypatch, stand_ins, "control", "--tp", "4", "--max-num-seqs", "3")
    assert llm.kw == {"replay": False, "seed": prof.ENGINE_SEED, "enforce_eager": True, "tensor_parallel_size": 4, "enable_prefix_caching": False,
                      "max_num_seqs": 3}
    assert llm.rpcs == [] and before == [[], []]
    topo, tok = _read(out)
    assert (topo["arm"], topo["tp"], topo["ranks"], tok["arm"], tok["tp"], tok["recorder"]) == ("control", 4, None, "control", 4, None)
    assert tok["requests"][0]["generated_token_ids"] == [5, 6]


def test_without_a_warm_up_the_workload_is_the_only_run(tmp_path, monkeypatch, stand_ins):
    llm, _, before = _arm(tmp_path, monkeypatch, stand_ins, "capture", "--no-warmup")
    assert before == [["tp2_probe", "tp2_coll_install"]] and llm.methods() == ["tp2_probe", "tp2_coll_install", "tp2_coll_close"]


def test_the_engines_kwargs_are_recorded_without_the_model_and_the_tokenizer(tmp_path, monkeypatch, stand_ins):
    llm, out, _ = _arm(tmp_path, monkeypatch, stand_ins, "control", "--engine-arg", 'model="org/model"', "--engine-arg", 'tokenizer="org/tok"')
    kw = llm.kw
    assert kw is not None and (kw["model"], kw["tokenizer"]) == ("org/model", "org/tok")
    assert _read(out)[0]["engine_kwargs"] == {k: v for k, v in kw.items() if k not in ("model", "tokenizer")}


def test_a_request_the_run_returned_no_tokens_for_generated_none(tmp_path, monkeypatch, stand_ins):
    llm = TT._LLM()
    stand_ins(llm, None)
    monkeypatch.setattr(va, "run_requests", lambda eng, reqs, sampling: ({}, {"wall_s": 0.5}))
    out = tmp_path / "control"
    assert M.main(cli.parse("tp-match", ["--arm", "control", "--workload", str(tmp_path / "w.json"), "--out", str(out), "--no-warmup"], environ={})) == 0
    assert _read(out)[1]["requests"] == [{"request_id": "q0", "prompt_token_ids": [1, 2, 3], "max_tokens": 2, "generated_token_ids": []}]


def test_the_topology_records_the_three_families_of_the_environment_in_its_order_and_a_credential_by_name_only(tmp_path, monkeypatch, stand_ins):
    for name, value in [("NCCL_ALGO", "Ring"), ("HF_TOKEN", DUMMY), ("VLLM_API_KEY", DUMMY), ("CUBLAS_WORKSPACE_CONFIG", ":4096:8"),
                        ("NCCL_DUMMY_TOKEN", DUMMY), ("OTHER_VLLM_NAME", "no family's")]:
        monkeypatch.setenv(name, value)
    _, out, _ = _arm(tmp_path, monkeypatch, stand_ins, "control", "--no-warmup")
    env = _read(out)[0]["env"]
    assert list(env) == [k for k in os.environ if k.startswith(FAMILIES)] and "HF_TOKEN" not in env and "OTHER_VLLM_NAME" not in env
    assert (env["NCCL_ALGO"], env["CUBLAS_WORKSPACE_CONFIG"]) == ("Ring", ":4096:8")
    assert (env["VLLM_API_KEY"], env["NCCL_DUMMY_TOKEN"]) == (settings.REDACTED, settings.REDACTED)
    assert env["VLLM_BATCH_INVARIANT"] == os.environ["VLLM_BATCH_INVARIANT"] and DUMMY not in (out / "topology.json").read_text()
