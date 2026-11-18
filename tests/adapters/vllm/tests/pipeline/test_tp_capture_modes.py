"""`verity_vllm.pipeline.tp.capture` in each of its modes: what `tp-capture` asks of every rank, and in what order, for a capture
(the observer, with the parameter hashes and the collective recorder beside it when asked), for a census (one profiler session a
rank and no observer) and for a control (no rank asked anything), with the live inventory and the warm-up before any of it; and
what its options make of the snapshot steps, the ranks' Builds, the weights' check and the bound on each rpc.  CPU-only: `main`
runs over the stand-in LLM of two ranks of `tests.pipeline.test_tp_target`, here with each rpc and each run of the requests on
one list."""
from __future__ import annotations

import gzip
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.pipeline import test_tp_target as TT
from verity_vllm.pipeline import cli
from verity_vllm.pipeline.tp import capture as TPC

stand_ins, vllm = TT.stand_ins, TT.vllm

EXTENSION = "verity_vllm.pipeline.tp.rank_worker.TP2CaptureWorkerExtension"
SAMPLING = {"variant": "greedy"}
RUN = ("run", ["q0"], SAMPLING)
WARM = ("run", ["throwaway of q0 under seed 0x5eed7a9b"], SAMPLING)
STEP_BOUND = "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS"


class _LLM(TT._LLM):
    """The stand-in LLM with every rpc's name and every run of requests on ``ev`` in the order they happened, each rpc's bound on
    ``timeouts``, and ``answers`` (by rpc name) added to what each rank answers."""

    def __init__(self, **answers: dict[str, Any]) -> None:
        super().__init__()
        self.answers = answers
        self.ev: list[Any] = []
        self.timeouts: list[float | None] = []

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        name = method if isinstance(method, str) else method.__name__
        self.ev.append(name)
        self.timeouts.append(timeout)
        return [{**row, **self.answers.get(name, {})} for row in super().collective_rpc(method, timeout, args, kwargs)]

    def asked(self, name: str) -> dict[str, Any]:
        """The keyword arguments of the one rpc of that name."""
        [kw] = [kw for method, kw in self.rpcs if method == name]
        return kw


def _capture(tmp_path: Path, monkeypatch, stand_ins, llm: _LLM, *extra: str, tokens: dict | None = None,
             patches: tuple[tuple[Any, str, Any], ...] = ()) -> Path:
    """`tp-capture` over ``llm`` with no declared target and the options ``extra``; the engine answers ``tokens`` for the requests
    (default: the workload's accepted ones)."""
    from verity_vllm.engine import vllm_adapter as va
    stand_ins(llm, None)

    def run_requests(eng, reqs, sampling):
        llm.ev.append(("run", [r["request_id"] for r in reqs], sampling))
        return ({"q0": [5, 6]} if tokens is None else tokens), {"wall_s": 0.5, "engine_steps": 2}

    def throwaway_requests(eng, like, seed):
        return [{**r, "request_id": f"throwaway of {r['request_id']} under seed {seed:#x}"} for r in like]

    for owner, name, value in [(va, "run_requests", run_requests), (va, "throwaway_requests", throwaway_requests), *patches]:
        monkeypatch.setattr(owner, name, value)
    monkeypatch.setenv(STEP_BOUND, "")                 # so the test's end takes the driver's pin out again
    monkeypatch.delenv(STEP_BOUND)
    out = tmp_path / "capture"
    TPC.main(cli.parse("tp-capture", ["--workload", str(tmp_path / "w.json"), "--out", str(out), "--case", "C", *extra], environ={}))
    return out


def _doc(out: Path, name: str) -> dict:
    return json.loads((out / name).read_text())


# ---------------------------------------------------------------------------------------------------------------- the three modes


def test_a_capture_installs_the_observer_after_the_warm_up_and_closes_it_after_the_run(tmp_path, monkeypatch, stand_ins, capsys):
    llm = _LLM()
    out = _capture(tmp_path, monkeypatch, stand_ins, llm)
    assert llm.ev == ["tp2_probe", WARM, "tp2_install", RUN, "tp2_close"]
    assert llm.kw["worker_extension_cls"] == EXTENSION
    assert list(_doc(out, "ranks.json")) == ["install", "close"]
    assert [r["pid"] for r in _doc(out, "topology.json")["ranks"]] == [100, 101]
    tokens = _doc(out, "tokens.json")
    assert (tokens["mode"], tokens["observer"], tokens["tp"], tokens["case"], tokens["model"]) == ("capture", True, 2, "C", "org/model")
    said = capsys.readouterr().out
    assert "[tp2] warm-up (uninstrumented, 1 throwaway requests): 0.50s, 2 steps\n" in said
    assert "[tp2] observer installed on 2 ranks: rank 0 pid 100 roots 1 pid=p; rank 1 pid 101 roots 1 pid=p\n" in said
    assert re.search(r"\[tp2\] capture run \d+\.\d\ds, 2 engine steps\n", said)
    assert "[tp2] rank 1: 1 events, 2 steps, snapshot 0 B, errors 0\n" in said and "ERRORS" not in said
    assert said.endswith(f"[tp2] done -> {out}\n")


def test_no_warmup_runs_the_requests_once(tmp_path, monkeypatch, stand_ins, capsys):
    llm = _LLM()
    _capture(tmp_path, monkeypatch, stand_ins, llm, "--no-warmup")
    assert llm.ev == ["tp2_probe", "tp2_install", RUN, "tp2_close"] and "warm-up" not in capsys.readouterr().out


def test_the_parameter_hashes_and_the_collective_recorder_go_in_after_the_observer_and_the_recorder_closes_first(
        tmp_path, monkeypatch, stand_ins, capsys):
    """Both are the capture's own: the ranks hash their shards with the observer installed, the collective recorder is installed
    last and closed first, so the observer is on for all the recorder does, and each is on the ranks' record."""
    llm = _LLM(tp2_param_hashes={"sha256": "ab"}, tp2_coll_install={"recorder": "on"}, tp2_coll_close={"rows": 9})
    out = _capture(tmp_path, monkeypatch, stand_ins, llm, "--no-warmup", "--param-hashes", "--record-collectives", "--store-steps", "3")
    assert llm.ev == ["tp2_probe", "tp2_install", "tp2_param_hashes", "tp2_coll_install", RUN, "tp2_coll_close", "tp2_close"]
    assert llm.asked("tp2_param_hashes") == {"out_dir": str(out)}
    assert llm.asked("tp2_coll_install") == {"out_dir": str(out), "store_steps": 3}
    assert llm.asked("tp2_coll_close") == {} and llm.asked("tp2_close") == {}
    ranks = _doc(out, "ranks.json")
    assert list(ranks) == ["install", "param_hashes", "coll_install", "coll_close", "close"]
    assert [[r[k] for r in ranks[name]] for name, k in (("param_hashes", "sha256"), ("coll_install", "recorder"), ("coll_close", "rows"))] == [
        ["ab", "ab"], ["on", "on"], [9, 9]]
    said = capsys.readouterr().out
    assert "[tp2] param shard hashes: [{'rank': 0," in said
    assert "[tp2] collective recorder installed (same run as the observer): [{'rank': 0," in said
    assert "[tp2] collective recorder closed: [{'rank': 0," in said


def test_the_collective_recorder_stores_every_steps_tensors_unless_told_how_many(tmp_path, monkeypatch, stand_ins):
    llm = _LLM()
    _capture(tmp_path, monkeypatch, stand_ins, llm, "--no-warmup", "--record-collectives")
    assert llm.asked("tp2_coll_install")["store_steps"] == 1 << 30 and "tp2_param_hashes" not in llm.ev


@pytest.mark.parametrize("extra", [(), ("--param-hashes", "--record-collectives")], ids=["alone", "with the capture's own options"])
def test_a_census_profiles_every_rank_over_the_run_and_installs_no_observer(tmp_path, monkeypatch, stand_ins, capsys, extra):
    """The ranks still load the extension, whose methods the profiler's two calls are; the options that belong to a capture ask
    the ranks nothing."""
    llm = _LLM(tp2_profiler_stop={"n_kernels": 7, "distinct": 3, "collective_kernels": {"ncclAllReduce": 2}})
    out = _capture(tmp_path, monkeypatch, stand_ins, llm, "--census", "--no-warmup", *extra)
    assert llm.ev == ["tp2_probe", "tp2_profiler_start", RUN, "tp2_profiler_stop"]
    assert llm.kw["worker_extension_cls"] == EXTENSION
    assert llm.asked("tp2_profiler_start") == {} and llm.asked("tp2_profiler_stop") == {"out_dir": str(out)}
    ranks = _doc(out, "ranks.json")
    assert list(ranks) == ["profiler_start", "profiler_stop"] and [r["n_kernels"] for r in ranks["profiler_stop"]] == [7, 7]
    tokens = _doc(out, "tokens.json")
    assert (tokens["mode"], tokens["observer"], _doc(out, "topology.json")["mode"]) == ("census", False, "census")
    said = capsys.readouterr().out
    assert "[tp2] rank 1: 7 device kernels, 3 distinct; collectives {'ncclAllReduce': 2}\n" in said
    assert re.search(r"\[tp2\] census run \d+\.\d\ds, 2 engine steps\n", said)


def test_a_control_builds_the_engine_without_the_extension_and_asks_no_rank_anything(tmp_path, monkeypatch, stand_ins, capsys):
    llm = _LLM()
    out = _capture(tmp_path, monkeypatch, stand_ins, llm, "--no-observer", "--param-hashes", "--record-collectives")
    assert llm.ev == [WARM, RUN] and "worker_extension_cls" not in llm.kw
    assert not (out / "ranks.json").exists() and _doc(out, "topology.json")["ranks"] is None
    tokens = _doc(out, "tokens.json")
    assert (tokens["mode"], tokens["observer"]) == ("control", False)
    assert "ranks=[]" in capsys.readouterr().out


def test_every_rpc_and_every_engine_step_wait_as_long_as_the_command_line_says(tmp_path, monkeypatch, stand_ins):
    llm = _LLM()
    _capture(tmp_path, monkeypatch, stand_ins, llm, "--rpc-timeout", "7", "--param-hashes", "--record-collectives", "--inventory-out",
             str(tmp_path / "inventory.json"))
    assert llm.timeouts == [7.0] * 7 and os.environ[STEP_BOUND] == "7"
    census = _LLM(tp2_profiler_stop={"n_kernels": 0, "distinct": 0, "collective_kernels": {}})
    _capture(tmp_path, monkeypatch, stand_ins, census, "--rpc-timeout", "9", "--census")
    assert census.timeouts == [9.0] * 3 and os.environ[STEP_BOUND] == "9"


def test_without_a_bound_on_the_command_line_each_waits_half_an_hour(tmp_path, monkeypatch, stand_ins):
    llm = _LLM()
    _capture(tmp_path, monkeypatch, stand_ins, llm)
    assert llm.timeouts == [1800.0] * 3 and os.environ[STEP_BOUND] == "1800"


# ------------------------------------------------------------------------------------------- the engine and what the install is handed


@pytest.mark.parametrize("extra, max_num_seqs", [((), 8), (("--max-num-seqs", "3"), 3), (("--max-num-seqs", "3", "--engine-arg", "max_num_seqs=5"), 5)],
                         ids=["by default", "the option", "an engine argument has the last word"])
def test_the_engine_is_the_profiles_eager_one_without_prefix_caching_for_the_world_asked(tmp_path, monkeypatch, stand_ins, extra, max_num_seqs):
    from verity_vllm.engine import engine_profile as prof

    llm = _LLM()
    out = _capture(tmp_path, monkeypatch, stand_ins, llm, "--no-warmup", "--tp", "4", *extra)
    assert llm.kw == {"replay": False, "seed": prof.ENGINE_SEED, "enforce_eager": True, "tensor_parallel_size": 4, "enable_prefix_caching": False,
                      "max_num_seqs": max_num_seqs, "worker_extension_cls": EXTENSION}
    assert (_doc(out, "topology.json")["tp"], _doc(out, "tokens.json")["tp"]) == (4, 4)


@pytest.mark.parametrize("extra, switches", [
    ((), (True, True, True)), (("--no-reenter",), (False, True, True)), (("--no-triton",), (True, False, True)),
    (("--no-module-hooks",), (True, True, False)), (("--no-reenter", "--no-triton", "--no-module-hooks"), (False, False, False)),
], ids=["all on", "no reentry", "no triton", "no module hooks", "none"])
def test_the_install_is_handed_the_observers_switches_as_the_command_line_sets_them(tmp_path, monkeypatch, stand_ins, extra, switches):
    llm = _LLM()
    _capture(tmp_path, monkeypatch, stand_ins, llm, "--no-warmup", *extra)
    install = llm.asked("tp2_install")
    assert (install["reenter"], install["triton"], install["module_hooks"]) == switches


def test_the_install_is_handed_the_run_it_records(tmp_path, monkeypatch, stand_ins):
    """The workload, the case, the checkpoint, the engine's arguments in full (a copy of the dict the engine was built from) and
    the environment it was built in, where the rank writes, the command line, and the build the header names."""
    from verity_vllm.engine import engine_profile as prof

    llm, built = _LLM(), {}
    out = _capture(tmp_path, monkeypatch, stand_ins, llm, "--no-warmup", "--wheel-sha256", "w" * 64, "--vllm-commit", "c0ffee",
                   patches=((prof, "engine_kwargs", lambda cp, **kw: built.update(kw) or built),))
    install = llm.asked("tp2_install")
    assert (install["out_dir"], install["workload"], install["case"], install["cp"]) == (str(out), TT.WORKLOAD, "C", {"repo": "org/model"})
    assert install["engine_kwargs"] == built == llm.kw and install["engine_kwargs"] is not built
    # the profile's flags (the stand-in's: batch invariant) under what the workload's execution, eager, sets
    assert install["env_flags"] == {"VLLM_BATCH_INVARIANT": "0"} == _doc(out, "topology.json")["env_flags"] and install["argv"] == sys.argv
    assert (install["wheel_sha256"], install["vllm_commit"], install["target"]) == ("w" * 64, "c0ffee", None)


def test_the_models_and_the_tokenizers_paths_go_to_the_engine_and_the_ranks_and_into_no_document(tmp_path, monkeypatch, stand_ins, capsys):
    from verity_vllm.engine import engine_profile as prof

    llm = _LLM()
    inventory = tmp_path / "inventory.json"
    out = _capture(tmp_path, monkeypatch, stand_ins, llm, "--no-warmup", "--inventory-out", str(inventory),
                   patches=((prof, "engine_kwargs", lambda cp, **kw: {**kw, "model": "/weights/m", "tokenizer": "/weights/t"}),))
    assert (llm.kw["model"], llm.kw["tokenizer"]) == ("/weights/m", "/weights/t") and llm.asked("tp2_install")["engine_kwargs"] == llm.kw
    public = {k: v for k, v in llm.kw.items() if k not in ("model", "tokenizer")}
    assert len(public) == len(llm.kw) - 2
    for doc in (_doc(out, "topology.json"), _doc(out, "tokens.json"), json.loads(inventory.read_text())):
        assert doc["engine_kwargs"] == public
    said = capsys.readouterr().out
    assert f"[tp2] mode=capture tp=2 case=C kwargs={public}\n" in said and "/weights" not in said


# ------------------------------------------------------------------------------------------------- what the ranks' close reports


@pytest.mark.parametrize("tail, printed", [(700, 16 + 1500), (0, 16 + len('["error 0: ", "error 1: ", "error 2: "]'))], ids=["long", "short"])
def test_a_rank_that_closed_with_errors_has_its_first_three_printed_in_at_most_1500_characters(
        tmp_path, monkeypatch, stand_ins, capsys, tail, printed):
    errors = [f"error {k}: " + "x" * tail for k in range(4)]
    llm = _LLM(tp2_close={"n_errors": 4, "errors": errors})
    _capture(tmp_path, monkeypatch, stand_ins, llm, "--no-warmup")
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.startswith("[tp2]   ERRORS: ")]
    assert lines == ["[tp2]   ERRORS: " + json.dumps(errors[:3])[:1500]] * 2 and "error 2: " in lines[0] and len(lines[0]) == printed
    assert [(r["n_errors"], r["errors"]) for r in _doc(tmp_path / "capture", "ranks.json")["close"]] == [(4, errors)] * 2


# ------------------------------------------------------------------------------------------------------ the tokens beside the record


@pytest.mark.parametrize("tokens, accepted, equal, all_equal", [
    ({"q0": [5, 6]}, [5, 6], True, True), ({"q0": [5, 7]}, [5, 6], False, False), ({}, [5, 6], False, False),
    ({"q0": [5, 6]}, None, None, False), ({"q0": []}, [], True, True),
], ids=["the accepted tokens", "another token", "no tokens for the request", "no accepted record", "none accepted, none generated"])
def test_tokens_json_says_per_request_whether_the_engine_generated_the_workloads_accepted_tokens(
        tmp_path, monkeypatch, stand_ins, tokens, accepted, equal, all_equal):
    from verity_vllm.engine import vllm_adapter as va

    workload = json.loads(json.dumps(TT.WORKLOAD))
    workload["accepted"] = {} if accepted is None else {"q0": accepted}
    out = _capture(tmp_path, monkeypatch, stand_ins, _LLM(), "--no-warmup", tokens=tokens,
                   patches=((va, "load_workload", lambda path, max_tokens: workload),))
    doc = _doc(out, "tokens.json")
    assert doc["requests"] == [{"request_id": "q0", "prompt_token_ids": [1, 2, 3], "max_tokens": 2, "generated_token_ids": tokens.get("q0", []),
                                "accepted_record_token_ids": accepted, "equals_accepted": equal}]
    assert doc["all_equal_accepted"] is all_equal and doc["timing"] == {"wall_s": 0.5, "engine_steps": 2}


@pytest.mark.parametrize("tokens, equal", [
    ({"q0": [5, 6], "q1": [8]}, [True, True]), ({"q0": [5, 7], "q1": [8]}, [False, True]), ({"q0": [5, 6], "q1": [9]}, [True, False]),
], ids=["both", "the first differs", "the last differs"])
def test_the_run_equals_the_accepted_record_only_where_every_request_does(tmp_path, monkeypatch, stand_ins, tokens, equal):
    from verity_vllm.engine import vllm_adapter as va

    workload = json.loads(json.dumps(TT.WORKLOAD))
    workload["requests"].append({"request_id": "q1", "prompt_token_ids": [4], "max_tokens": 1})
    workload["accepted"] = {"q0": [5, 6], "q1": [8]}
    llm = _LLM()
    out = _capture(tmp_path, monkeypatch, stand_ins, llm, "--no-warmup", tokens=tokens,
                   patches=((va, "load_workload", lambda path, max_tokens: workload),))
    doc = _doc(out, "tokens.json")
    assert [(r["request_id"], r["equals_accepted"]) for r in doc["requests"]] == list(zip(["q0", "q1"], equal))
    assert doc["all_equal_accepted"] is all(equal) and ("run", ["q0", "q1"], SAMPLING) in llm.ev


# ---------------------------------------------------------------------------------------------------------------- the snapshot steps


@pytest.mark.parametrize("given, max_tokens, steps", [
    ("all", 5, [0, 1, 2, 3, 4]), (" All ", 2, [0, 1]), ("all", None, [0, 1, 2, 3]), ("all", 0, [0, 1, 2, 3]), (None, 5, []), ("", 5, []),
    ("3", 5, [3]), ("5,3,,3, 1 ", 2, [1, 3, 5]),
], ids=["all", "all in any case", "all of an unknown length", "all of no tokens", "none given", "empty", "one", "a list in any order"])
def test_the_snapshot_steps_are_every_step_of_the_longest_request_or_the_ones_listed(given, max_tokens, steps):
    assert TPC.parse_steps(given, max_tokens) == steps


@pytest.mark.parametrize("extra, steps", [((), [0, 1, 2]), (("--snapshot-steps", ""), []), (("--snapshot-steps", "4,1,4"), [1, 4])],
                         ids=["by default every step", "empty: none", "a list"])
def test_the_ranks_are_handed_the_snapshot_steps_of_the_workloads_longest_request(tmp_path, monkeypatch, stand_ins, extra, steps):
    from verity_vllm.engine import vllm_adapter as va

    workload = json.loads(json.dumps(TT.WORKLOAD))
    workload["requests"].append({"request_id": "q1", "prompt_token_ids": [4], "max_tokens": 3})
    llm = _LLM()
    _capture(tmp_path, monkeypatch, stand_ins, llm, "--no-warmup", "--snapshot-max-bytes", "64", *extra,
             patches=((va, "load_workload", lambda path, max_tokens: workload),))
    install = llm.asked("tp2_install")
    assert (install["snapshot_steps"], install["snapshot_max_bytes"]) == (steps, 64)


# ------------------------------------------------------------------------------------------------- the ranks' Builds and the weights


def _build(root: Path, rank: int, programs: dict[str, dict]) -> str:
    """A rank's Build dir: one component Program dir a name, each with its instances and the ``result.json`` given."""
    d = root / f"build_rank{rank}"
    for name, result in programs.items():
        (d / name).mkdir(parents=True)
        with gzip.open(d / name / "instances.json.gz", "wt") as f:
            json.dump([], f)
        (d / name / "result.json").write_text(json.dumps(result))
    return str(d)


def test_each_ranks_install_is_handed_the_tensors_its_build_registered_after_the_load(tmp_path, monkeypatch, stand_ins):
    b0 = _build(tmp_path, 0, {"build_request": {"post_load_tensors": {"parameters": ["w.b", "w.a"]}},
                              "build_request_LP4_T2": {"post_load_tensors": {"parameters": ["w.c", "w.a"]}}})
    b1 = _build(tmp_path, 1, {"build_request": {"post_load_tensors": {}}, "build_request_LP4_T2": {}})
    llm = _LLM()
    _capture(tmp_path, monkeypatch, stand_ins, llm, "--no-warmup", "--build-rank", b0, "--build-rank", b1)
    assert llm.asked("tp2_install")["post_load_tensors"] == {"0": ["w.a", "w.b", "w.c"], "1": []}


def test_without_a_build_no_rank_is_handed_a_tensor_to_register(tmp_path, monkeypatch, stand_ins):
    llm = _LLM()
    _capture(tmp_path, monkeypatch, stand_ins, llm, "--no-warmup")
    assert llm.asked("tp2_install")["post_load_tensors"] == {}


@pytest.mark.parametrize("given, tp", [(1, 2), (3, 2), (2, 4)])
def test_a_build_for_some_ranks_and_not_others_is_refused_before_anything_is_loaded_or_written(tmp_path, monkeypatch, stand_ins, given, tp):
    llm = _LLM()
    builds = [flag for r in range(given) for flag in ("--build-rank", _build(tmp_path, r, {"build_request": {}}))]
    with pytest.raises(SystemExit) as refused:
        _capture(tmp_path, monkeypatch, stand_ins, llm, "--tp", str(tp), *builds)
    assert str(refused.value) == f"--build-rank given {given} time(s) for a world of {tp}: one Build dir per rank, in rank order"
    assert llm.kw == {} and llm.ev == [] and not (tmp_path / "capture").exists()


@pytest.mark.parametrize("extra, checked", [(("--verify-weights",), True), ((), False)], ids=["asked", "not asked"])
def test_the_weights_are_checked_against_the_checkpoint_before_the_engine_is_built_when_asked(tmp_path, monkeypatch, stand_ins, extra, checked):
    from verity_vllm.engine import engine_profile as prof

    llm = _LLM()
    seen: list[tuple[dict, dict]] = []
    _capture(tmp_path, monkeypatch, stand_ins, llm, "--no-warmup", *extra,
             patches=((prof, "verify_local_weights", lambda cp: seen.append((cp, dict(llm.kw)))),))
    assert seen == ([({"repo": "org/model"}, {})] if checked else [])
    assert llm.asked("tp2_install")["weights_verified"] is checked


# ------------------------------------------------------------------------------------------------------- the topology and the inventory


class _Executor:
    pass


class _Inner:
    model_executor = _Executor()


class _Core:
    engine_core = _Inner()


class _Bare:
    """An engine core that is its own inner core."""

    engine_core = None
    model_executor = _Executor()


@pytest.mark.parametrize("core, core_class, executor_class", [
    (_Core(), "_Core", "_Executor"), (_Bare(), "_Bare", "_Executor"), (SimpleNamespace(), "SimpleNamespace", None), (None, None, None),
], ids=["a core around the engine's", "the engine's own core", "a core with no executor", "no core"])
def test_the_topology_names_the_engines_core_and_its_executor_by_class(tmp_path, monkeypatch, stand_ins, capsys, core, core_class, executor_class):
    llm = _LLM()
    llm.llm_engine = SimpleNamespace(engine_core=core)
    out = _capture(tmp_path, monkeypatch, stand_ins, llm, "--no-warmup")
    topo = _doc(out, "topology.json")
    here = {"_Core": __name__, "_Bare": __name__, "_Executor": __name__, "SimpleNamespace": "types"}
    named = [None if c is None else f"{here[c]}.{c}" for c in (core_class, executor_class)]
    assert [topo["engine_core_class"], topo["executor_class"]] == named
    assert f"; executor={named[1]}; in-process runner: none: division by zero; ranks=[(100, None, None), (101, None, None)]\n" in capsys.readouterr().out
    assert (topo["mode"], topo["tp"], topo["driver_pid"], topo["in_process_runner"]) == ("capture", 2, os.getpid(), "none: division by zero")
    assert topo["engine_kwargs"]["tensor_parallel_size"] == 2 and topo["engine_kwargs"]["enable_prefix_caching"] is False


def test_an_engine_with_a_runner_in_this_process_is_recorded_as_one(tmp_path, monkeypatch, stand_ins, capsys):
    from verity_vllm.engine import handle

    llm = _LLM()
    out = _capture(tmp_path, monkeypatch, stand_ins, llm, "--no-warmup", "--tp", "1", patches=((handle, "model_runner", lambda llm: object()),))
    assert _doc(out, "topology.json")["in_process_runner"] is True and "; in-process runner: True; ranks=" in capsys.readouterr().out
    assert llm.kw["tensor_parallel_size"] == 1


def test_the_live_inventory_is_every_ranks_and_is_written_before_the_warm_up(tmp_path, monkeypatch, stand_ins, capsys):
    from verity_vllm.engine import meta_inventory as MI

    llm = _LLM(rank_inventory={"modules": ["m"]})
    path = tmp_path / "made" / "for" / "it" / "inventory.json"
    _capture(tmp_path, monkeypatch, stand_ins, llm, "--inventory-out", str(path), "--engine-arg", "max_num_batched_tokens=2048")
    assert llm.ev == ["tp2_probe", "rank_inventory", WARM, "tp2_install", RUN, "tp2_close"]
    doc = json.loads(path.read_text())
    assert (doc["schema"], doc["source"], doc["repo"], doc["revision"], doc["tp"]) == (MI.SCHEMA, "live", "org/model", None, 2)
    assert doc["engine_kwargs"] == _doc(tmp_path / "capture", "topology.json")["engine_kwargs"] and doc["engine_kwargs"]["max_num_batched_tokens"] == 2048
    assert list(doc["ranks"]) == ["0", "1"] and [r["modules"] for r in doc["ranks"].values()] == [["m"], ["m"]]
    assert f"[tp2] live inventory of ranks ['0', '1'] -> {path}\n" in capsys.readouterr().out


def test_no_inventory_is_read_unless_a_path_is_given(tmp_path, monkeypatch, stand_ins):
    llm = _LLM()
    _capture(tmp_path, monkeypatch, stand_ins, llm, "--no-warmup")
    assert "rank_inventory" not in llm.ev
