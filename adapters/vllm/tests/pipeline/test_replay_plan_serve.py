"""The body of the slim bundle's planner process (`verity_vllm.pipeline.replay_plan`: `_serve`, `serve`, `main`) in this process,
on stand-ins for the steps that read Programs: what it loads, what it sends the Commit once it has prepared, the sample it takes when
the Commit's facts arrive (the replay's own, read only) and the committed reads it sends back, each once."""
from __future__ import annotations

import json
import multiprocessing as mp
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from verity_vllm.check.replay import driver as SR
from verity_vllm.commit import partition
from verity_vllm.engine import vllm_adapter as va
from verity_vllm.pipeline import c2_replay as C2
from verity_vllm.pipeline import replay_plan as RP

REQS = [{"request_id": "a", "index": 0, "prompt_token_ids": [5, 6]},
        {"request_id": "b", "index": 1, "prompt_token_ids": [5, 6, 7], "arrive_step": 1, "cached": 16, "cached_from": ["a"]}]
SAMPLING = {"max_tokens": 4}
LABELS = {"a": "r0", "b": "r1"}
LOADED = {"r0": {"instances": {"rows": ["row0"]}, "dir": "/programs/r0", "result": {"T": 3}},
          "r1": {"instances": {"rows": ["row1"]}, "dir": "/programs/r1", "result": {"T": 1}}}
PROGRAMS = {"r0": LOADED["r0"]["instances"], "r1": LOADED["r1"]["instances"]}
DIRS = {"r0": "/programs/r0", "r1": "/programs/r1"}
WARM = {"r1": {"cached": 16, "producers": ["r0"], "prompt": [5, 6, 7]}}
MANIFEST = {"manifest_digest": "m" * 8}
BINDING_MAP = {"binding": "map"}
READS = [(1, "x", 0, 4), (1, "x", 0, 4), (2, "y", 4, 6)]
PLAN = {"plan": True, "picked": 3}


class Index:
    def embed_op_path(self) -> str:
        return "model/embed"


class Steps:
    """The planner's steps that read Programs, each a stand-in that keeps what it was given."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        self.seen: dict[str, Any] = {}
        self.read: list[bytes] = []
        self.manifest = tmp_path / "manifest.json"
        self.manifest.write_text(json.dumps(MANIFEST))
        self.binding_map = tmp_path / "binding_map.json"
        self.binding_map.write_text(json.dumps(BINDING_MAP))
        monkeypatch.setattr(va, "load_workload", self.load_workload)
        monkeypatch.setattr(partition, "require_partition", self.require_partition)
        monkeypatch.setattr(C2, "programs_of_record", self.programs_of_record)
        monkeypatch.setattr(SR, "populate", self.populate)
        monkeypatch.setattr(SR, "sampled_replay", self.sampled_replay)

    def load_workload(self, path: str) -> dict[str, Any]:
        self.seen["workload"] = path
        return {"requests": REQS, "sampling": SAMPLING}

    def require_partition(self, man: dict[str, Any], programs_root: str | None, program_dir: str | None) -> dict[str, Any]:
        self.seen["partition"] = (man, programs_root, program_dir)
        return man

    def programs_of_record(self, sources: Any, labels: dict[str, str], *, definitions: bool = True) -> tuple[dict[str, Any], dict[str, str]]:
        self.seen["programs"] = (sources, labels, definitions)
        return LOADED, {"r0": "d0", "r1": "d1"}

    def populate(self, programs: dict[str, Any], ranks: list[int], arrivals: dict[str, int], executed_prefix: dict[str, Any],
                 ring_slots: int | None, *, memo: dict[Any, Any], **kw: Any) -> None:
        self.seen["populate"] = (programs, ranks, arrivals, executed_prefix, ring_slots, kw)
        self.memo = memo
        memo[("index", id(programs["r0"]))] = Index()

    def sampled_replay(self, programs: dict[str, Any], bmap: dict[str, Any], committed: Any, weights: Any, **kw: Any) -> dict[str, Any]:
        self.seen["replay"] = (programs, bmap, weights, kw)
        self.read = [committed(*r) for r in READS]
        return dict(PLAN)

    def args(self, **given: Any) -> dict[str, Any]:
        """The Commit's options the planner is sent (`RP.ARGS`)."""
        args = {k: None for k in RP.ARGS} | {"workload": "workload.json", "program_dir": "/programs", "program_digest": "d0",
                                             "required_manifest": str(self.manifest), "replay_per_stratum": 2, "replay_max": 50,
                                             "window_slots": 4, "window_mb": 1}
        return args | given

    def facts(self) -> dict[str, Any]:
        """What the Commit sends once its run is committed (`Planner.plan`)."""
        return {"binding_map": str(self.binding_map), "arrivals": {"r0": 0, "r1": 2}, "seed": 7, "seed_source": "coins",
                "seed_challenge": "challenge", "sr_kw": {"executed_prefix": {"r0": {"n_exec": 2}}}}


def served(steps: Steps, **given: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """`_serve` run to its end over a pipe whose other end already holds the Commit's facts: the two messages it sent."""
    commit, planner = mp.Pipe()
    commit.send(steps.facts())
    RP._serve(planner, SimpleNamespace(**steps.args(**given)))
    ready, plan = commit.recv(), commit.recv()
    assert not commit.poll()
    commit.close()
    planner.close()
    return ready, plan


def test_the_planner_loads_the_programs_of_record_and_says_what_the_linkage_reads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    steps = Steps(monkeypatch, tmp_path)
    ready, _plan = served(steps)
    assert steps.seen["workload"] == "workload.json" and steps.seen["partition"] == (MANIFEST, None, "/programs")
    sources, labels, definitions = steps.seen["programs"]
    assert isinstance(sources, C2.ProgramSources)
    assert (sources.args.program_digest, sources.manifest, sources.reqs, sources.sampling) == ("d0", MANIFEST, REQS, SAMPLING)
    assert labels == LABELS and definitions is False, "a plan evaluates no row, so it checks no Definitions"
    programs, ranks, arrivals, executed_prefix, ring, kw = steps.seen["populate"]
    assert (programs, ranks, arrivals, executed_prefix, ring) == (PROGRAMS, [0], {"r0": 0, "r1": 1}, {}, 4)
    assert kw == {"attention_paths": None, "tp": None, "program_dirs": DIRS, "manifest": MANIFEST, "cached_prefixes": WARM}
    assert ready.pop("prepare_s") >= 0
    assert ready == {"program_steps": {"r0": 4, "r1": 2}, "embed_op_path": "model/embed"}


def test_the_sample_is_the_replays_own_read_only_and_each_read_comes_back_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    steps = Steps(monkeypatch, tmp_path)
    _ready, plan = served(steps)
    programs, bmap, weights, kw = steps.seen["replay"]
    assert (programs, bmap) == (PROGRAMS, BINDING_MAP)
    assert weights is C2.no_weights, "a plan evaluates no row and reads no weight"
    with pytest.raises(RuntimeError, match="no weights provider"):
        weights("model.w", 0, 4)
    assert kw.pop("memo") is steps.memo, "the populations the planner enumerated before the seed existed"
    assert kw == {"arrivals": {"r0": 0, "r1": 2}, "per_stratum": 2, "seed": 7, "seed_source": "coins", "seed_challenge": "challenge",
                  "max_evaluate": 50, "targets": [], "replay_k": None, "replay_draw": "family", "ring_slots": 4, "manifest": MANIFEST,
                  "program_dirs": DIRS, "plan": True, "cached_prefixes": WARM, "executed_prefix": {"r0": {"n_exec": 2}}}
    assert steps.read == [bytes(4), bytes(4), bytes(2)], "a recording reader answers zeros of the read's length"
    assert plan == {"reads": [(1, "x", 0, 4), (2, "y", 4, 6)], "plan": PLAN | {"reads_made": 3}}


def test_without_a_manifest_or_a_window_the_planner_checks_no_partition_and_bounds_no_ring(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    steps = Steps(monkeypatch, tmp_path)
    served(steps, required_manifest=None, window_mb=None, replay_target=["attention"], replay_draw="rows", replay_k=3)
    assert "partition" not in steps.seen and steps.seen["programs"][0].manifest is None
    assert steps.seen["populate"][4] is None and steps.seen["populate"][5]["manifest"] is None
    kw = steps.seen["replay"][3]
    assert (kw["ring_slots"], kw["manifest"], kw["targets"], kw["replay_draw"], kw["replay_k"]) == (None, None, ["attention"], "rows", 3)


def test_a_failure_while_preparing_reaches_the_commit_by_name_and_the_connection_is_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    steps = Steps(monkeypatch, tmp_path)

    def missing(path: str) -> dict[str, Any]:
        raise FileNotFoundError(f"no workload at {path}")

    monkeypatch.setattr(va, "load_workload", missing)
    commit, planner = mp.Pipe()
    commit.send(steps.args())
    RP.serve(planner)
    assert commit.recv() == {"error": "FileNotFoundError: no workload at workload.json"} and planner.closed
    commit.close()


def test_the_command_serves_the_connection_it_is_handed_and_exits_0(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    steps = Steps(monkeypatch, tmp_path)
    commit, planner = mp.Pipe()
    commit.send(steps.args())
    commit.send(steps.facts())
    fd = os.dup(planner.fileno())
    planner.close()
    assert RP.main(RP.Options(fd=str(fd))) == 0
    ready, plan = commit.recv(), commit.recv()
    assert ready["embed_op_path"] == "model/embed" and plan["plan"]["reads_made"] == 3
    with pytest.raises(OSError):
        os.fstat(fd)
    commit.close()
