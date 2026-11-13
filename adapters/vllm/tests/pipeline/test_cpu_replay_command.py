"""`cpu_replay.replay` and `cpu_replay.main`, the `commit-replay` command, on stand-ins for each step of the replay: the order of
the steps, what a source that is already a slim keep changes, the line it logs for the weights, the line it prints and ends on, and
its exit codes.  A row's verdict quotes the last line of the command's log, so that line stays a print and its text is held here."""
from __future__ import annotations

import contextlib
import json
import logging
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import verity_vllm.pipeline.cpu_replay as CPU
from verity_vllm.check import weights_of_record as WR
from verity_vllm.check.replay.driver import PoolDoesNotFit

ROOT = "ab" * 64
OPTIONS = CPU.Options(bundle="B", row_dir="R", checkpoint_dir="CK", checkpoints_manifest="checkpoints.json", model="m", revision="r", threads=3)
OPENED = {"equal": True, "opened": 2, "servable": 2, "committed_root": ROOT, "opened_by_source": {"checkpoint": 2}, "unopened_n": 0,
          "wall_s": 0.5, "why": None}
Lines = Callable[[], list[tuple[str, str]]]
ORDER = ["open_source", "of_record", "replay_args", "manifest_of", "CheckpointIndex", "open_weights", "worker_peak", "touched", "run",
         "engine_of_record", "decide", "keep", "record"]


@pytest.fixture
def logged(caplog: pytest.LogCaptureFixture) -> Lines:
    """What `cpu_replay`'s logger took so far: each record's level and text."""
    caplog.set_level(logging.INFO, logger=CPU.__name__)
    return lambda: [(r.levelname, r.getMessage()) for r in caplog.records if r.name == CPU.__name__]


class Replay:
    """The replay's steps, each a stand-in that records its call; `decide` decides the run as ``passed`` says."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, passed: bool = True, wc: dict[str, Any] = OPENED, kept: bool = False) -> None:
        self.calls: list[str] = []
        self.seen: dict[str, Any] = {}
        self.bdir = tmp_path / "commit" / "replay_bundle_p3"
        (self.bdir / "weights").mkdir(parents=True)
        (self.bdir / "weights" / "tree.json").write_text(json.dumps({"names": [], "extra": {"model.rope": 0}}))
        self.row: dict[str, Any] = {"pair": 3, "arm": "delta", "commit": {"weight_registration": {"root": ROOT}, "value_correspondence": {}}}
        self.verdict: dict[str, Any] = {"commit_pass": None}
        self.ctx = {"pair": 3, "requests": [{"request_id": "a"}], "sampling": {"temperature": 0.0}, "tokens": {"a": [1]}, "stop_of_record": {},
                    "ids_full": None, "code": "c0de", "vc_result": True}
        self.b: dict[str, Any] = {"context": self.ctx, "path": str(self.bdir)} | ({"keep": {"picks": []}} if kept else {})
        self.store, self.run = object(), object()
        self.words: dict[str, np.ndarray] = {"w": np.arange(4, dtype=np.uint16)}

        def step(name: str, result: Any = None) -> Any:
            def call(*args: Any, **kw: Any) -> Any:
                self.calls.append(name)
                self.seen[name] = (args, kw)
                return result
            return call

        def run(site: Any, cache: dict) -> None:
            step("run")(site, cache)
            site.weights_of("")("w", 0, 2)
            site.reads["picked"] = 1

        def decide(row: dict[str, Any], *rest: Any) -> None:
            step("decide")(row, *rest)
            row["commit"].update(commit_pass=passed, commit_fail_reason=None if passed else "the sampled replay failed")
            self.verdict["commit_pass"] = passed

        @contextlib.contextmanager
        def touched(store: Any) -> Any:
            step("touched")(store)
            yield "touched"

        monkeypatch.setattr(CPU, "open_source", step("open_source", self.b))
        monkeypatch.setattr(CPU, "of_record", step("of_record", ([self.row], self.row, self.verdict, self.store, self.run)))
        monkeypatch.setattr(CPU, "replay_args", step("replay_args", SimpleNamespace()))
        monkeypatch.setattr(CPU, "manifest_of", step("manifest_of", ("manifest.json", {"partition": 1})))
        monkeypatch.setattr(WR, "CheckpointIndex", step("CheckpointIndex", "ck"))
        monkeypatch.setattr(CPU, "open_weights", step("open_weights", (self.words, wc)))
        monkeypatch.setattr(CPU, "worker_peak", step("worker_peak", 123))
        monkeypatch.setattr(CPU, "kept_sample", step("kept_sample"))
        monkeypatch.setattr(CPU.SD, "touched", touched)
        monkeypatch.setattr(CPU.C2, "run", run)
        monkeypatch.setattr(CPU, "engine_of_record", step("engine_of_record"))
        monkeypatch.setattr(CPU, "decide", decide)
        monkeypatch.setattr(CPU, "keep", step("keep"))
        monkeypatch.setattr(CPU, "record", step("record"))


def test_the_replay_runs_its_steps_in_order_on_the_record_and_returns_its_verdict(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                                    capsys: pytest.CaptureFixture[str], logged: Lines) -> None:
    r = Replay(tmp_path, monkeypatch)
    assert CPU.replay(OPTIONS) is True
    assert r.calls == ORDER
    assert r.seen["open_source"] == (("B",), {}) and r.seen["of_record"] == ((r.b, r.bdir.parent), {})
    assert r.seen["CheckpointIndex"] == (("CK",), {"checkpoints_manifest": "checkpoints.json", "model": "m", "revision": "r"})
    assert r.seen["open_weights"] == ((r.bdir, {"names": [], "extra": {"model.rope": 0}}, ROOT, "ck"), {"threads": 3}), \
        "the tensors open against the weights root the run registered"
    site, cache = r.seen["run"][0]
    assert cache == {} and (site.pair, site.com, site.rc, site.row, site.out) == (3, r.store, r.run, r.row, r.bdir.parent)
    assert (site.args.required_manifest, site.args.replay_worker_peak, site.manifest) == ("manifest.json", 123, {"partition": 1})
    assert (site.reqs, site.sampling, site.tokens, site.vc_result) == (r.ctx["requests"], r.ctx["sampling"], r.ctx["tokens"], True)
    assert site.versions == {"git_head": "c0de"} and site.components is r.row["commit"]["value_correspondence"]["components"]
    assert site.weights_check() == OPENED and site.weights_check() is not OPENED
    assert r.seen["engine_of_record"] == ((r.row, r.ctx, r.bdir.parent / "sampled_replay_p3.json"), {})
    assert r.seen["decide"][0][1:] == (r.ctx, OPENED, r.b, {"w"}, r.words, {"model.rope": 0})
    assert r.seen["keep"] == ((r.bdir.parent, r.b, r.store, r.run, "touched", {"picked": 1}, r.row["commit"]), {"opened": {"w"}}), \
        "the keep takes the reads the replay made and the tensors it read"
    out, rows, verdict, b, wall = r.seen["record"][0]
    assert (out, rows, verdict, b) == (r.bdir.parent, [r.row], r.verdict, r.b) and wall >= 0
    weights = (f"[replay] WEIGHTS (pair 3): OPENED 2/2 registered tensors against the committed weights root {ROOT[:16]} "
               "({'checkpoint': 2}; unopened 0) in 0.5 s")
    assert logged() == [("INFO", weights)]
    assert capsys.readouterr().out.splitlines() == ["[replay] COMMIT PASS (pair 3): every check on the record; verdict commit_pass True"]


def test_a_failed_replay_ends_on_the_reason_and_returns_false(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                              capsys: pytest.CaptureFixture[str], logged: Lines) -> None:
    wc = OPENED | {"equal": False, "opened": 1, "unopened_n": 1, "why": "tensor root != the committed tensor root"}
    r = Replay(tmp_path, monkeypatch, passed=False, wc=wc)
    assert CPU.replay(OPTIONS) is False and r.calls == ORDER
    weights = (f"[replay] WEIGHTS (pair 3): FAIL 1/2 registered tensors against the committed weights root {ROOT[:16]} "
               "({'checkpoint': 2}; unopened 1) in 0.5 s -- tensor root != the committed tensor root")
    assert logged() == [("WARNING", weights)]
    assert capsys.readouterr().out.splitlines() == ["[replay] COMMIT FAIL (pair 3): the sampled replay failed; verdict commit_pass False"]


def test_a_slim_keep_is_held_to_its_sample_and_written_no_second_time(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    r = Replay(tmp_path, monkeypatch, kept=True)
    assert CPU.replay(OPTIONS) is True
    assert r.calls == [*ORDER[:7], "kept_sample", *ORDER[7:11], "record"], "the sample is checked before the replay; no keep is cut from a keep"
    site, keep = r.seen["kept_sample"][0]
    assert site is r.seen["run"][0][0] and keep == {"picks": []}


@pytest.mark.parametrize("commit", [{}, {"weight_registration": None}, {"weight_registration": {}}, {"weight_registration": {"root": None}}])
def test_a_run_that_registered_no_weights_opens_its_tensors_against_no_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                            commit: dict[str, Any]) -> None:
    r = Replay(tmp_path, monkeypatch)
    r.row["commit"] = {"value_correspondence": {}} | commit
    assert CPU.replay(OPTIONS) is True
    assert r.seen["open_weights"] == ((r.bdir, {"names": [], "extra": {"model.rope": 0}}, None, "ck"), {"threads": 3})


@pytest.mark.parametrize("tree", [{"names": []}, {"names": [], "extra": None}, {"names": [], "extra": {}}])
def test_a_tree_that_lists_no_tensor_as_the_bundle_s_decides_the_run_with_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                               tree: dict[str, Any]) -> None:
    r = Replay(tmp_path, monkeypatch)
    (r.bdir / "weights" / "tree.json").write_text(json.dumps(tree))
    assert CPU.replay(OPTIONS) is True
    assert r.seen["open_weights"][0][1] == tree and r.seen["decide"][0][-1] == {}


@pytest.mark.parametrize(("verdict", "code"), [(True, 0), (False, 3), (None, 3)])
def test_the_command_exits_by_the_verdict(monkeypatch: pytest.MonkeyPatch, verdict: bool | None, code: int) -> None:
    warmed = []
    monkeypatch.setattr(CPU, "warm_vml", lambda: warmed.append(True))
    monkeypatch.setattr(CPU, "replay", lambda o: verdict)
    assert CPU.main(OPTIONS) == code and warmed == [True]


@pytest.mark.parametrize("refusal", [CPU.Refused("the bundle is not this record's"), PoolDoesNotFit("no worker fits the cgroup")])
def test_a_refused_replay_ends_on_its_reason_and_exits_2(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
                                                         refusal: Exception) -> None:
    def replay(o: CPU.Options) -> bool:
        print("[replay] a line before the refusal", flush=True)
        raise refusal

    monkeypatch.setattr(CPU, "warm_vml", lambda: None)
    monkeypatch.setattr(CPU, "replay", replay)
    assert CPU.main(OPTIONS) == 2
    assert capsys.readouterr().out.splitlines()[-1] == f"[replay] REFUSED: {refusal}"
