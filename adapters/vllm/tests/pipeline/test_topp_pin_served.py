"""The served mode of the top-p pin check: what it serves, what it reads, and when it passes.

A GPU and vLLM run the mode for real.  Here the engine is a stand-in: it answers each request with the tokens a case gives
it, and launches vLLM's three split kernels (stand-ins in the module of `tests.fake_vllm`) at the split counts the case gives
it.  The launch proxies are the real ones (`commit.committer.topp_splits`), so each case shows the mode's verdict on what
the proxies record.  That a device's serve gives those tokens and those counts is what a GPU says.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("torch")

import torch
from verity.definitions.registry.topp_geometry import registered_splits

from tests import fake_vllm
from verity_vllm.commit.committer import topp_splits
from verity_vllm.engine import engine_profile, topp_pin
from verity_vllm.engine import vllm_adapter as va
from verity_vllm.pipeline import cli
from verity_vllm.pipeline import topp_pin_check as tc

NUM_SM = 142
S = registered_splits(NUM_SM)
SHA = "ab" * 32
SAMPLING = {"temperature": 0.8, "top_p": 0.9, "seed": 11}


class _Kernel:
    """Stands in for one of vLLM's split kernels: `kernel[grid](**kw)` launches nothing."""

    def __getitem__(self, grid: tuple[int]) -> Any:
        return lambda **kw: None


def _ids(n: int) -> list[str]:
    """The workload's request ids in its order, which is not their sorted order."""
    return [f"r{i}" for i in reversed(range(n))]


def _workload(n: int) -> dict[str, Any]:
    return {"requests": [{"request_id": rid, "prompt_token_ids": [int(rid[1:])]} for rid in _ids(n)], "case": "a-case",
            "engine_args_required": {"seed": 7}, "execution": "an-execution", "row_id": "a-row", "sha256": SHA,
            "sampling": dict(SAMPLING)}


def _tokens(n: int) -> dict[str, list[int]]:
    return {rid: [100 + int(rid[1:]), int(rid[1:])] for rid in _ids(n)}


class _Case:
    """One serve of the stand-in engine: `n` requests, with what the engine does when it serves them."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, n: int = 3, *, alone: dict[str, list[int] | None] | None = None,
                 s_together: tuple[int, ...] = (S,), s_alone: dict[str, tuple[int, ...]] | None = None,
                 s_alone_else: tuple[int, ...] = (S,), pinned: bool = True) -> None:
        self.n, self.alone = n, alone or {}
        self.log: list[tuple[Any, ...]] = []
        self.s_together, self.s_alone, self.s_alone_else = s_together, s_alone or {}, s_alone_else
        self.module = fake_vllm.install(monkeypatch)[topp_splits.TOPP_SPLIT_MODULE]
        self.kernels = {name: _Kernel() for name in topp_splits.TOPP_SPLIT_KERNELS}
        for name, kernel in self.kernels.items():
            setattr(self.module, name, kernel)
        monkeypatch.setattr(engine_profile, "apply_env", lambda: self.log.append(("env",)))
        monkeypatch.setattr(va, "load_workload", self._load)
        monkeypatch.setattr(va, "target_of_workload", lambda wl: ("target of", wl["row_id"]))
        monkeypatch.setattr(va, "build_engine", self._build)
        monkeypatch.setattr(va, "run_requests", self._run)
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i: SimpleNamespace(multi_processor_count=NUM_SM + i))
        monkeypatch.setattr(topp_pin, "pinned", lambda: pinned)

    def _load(self, path: str) -> dict[str, Any]:
        self.log.append(("load", path))
        return _workload(self.n)

    def _build(self, **kw: Any) -> str:
        self.log.append(("build", kw))
        return "an engine"

    def _proxied(self) -> bool:
        return all(isinstance(getattr(self.module, name), topp_splits._SplitKernelProxy) for name in self.kernels)

    def _run(self, engine: str, group: list[dict[str, Any]], sampling: dict[str, Any]) -> tuple[dict[str, list[int]], dict[str, Any]]:
        ids = [r["request_id"] for r in group]
        self.log.append(("serve", engine, ids, sampling, self._proxied()))
        together = len(ids) > 1
        for s in self.s_together if together else self.s_alone.get(ids[0], self.s_alone_else):
            for name in self.kernels:
                getattr(self.module, name)[(len(ids) * s,)](S=s)
        tokens: dict[str, list[int]] = {}
        for rid in ids:
            mine = _tokens(self.n)[rid] if together or rid not in self.alone else self.alone[rid]
            if mine is not None:
                tokens[rid] = mine
        return tokens, {"requests": {}}

    def served(self, out: Path) -> dict[str, Any]:
        return tc.served("a-workload.json", out)


def _passing(n: int = 3) -> dict[str, Any]:
    return {"ok": True, "pinned": True, "row": "a-row", "workload_sha256": SHA, "sampling": {"temperature": 0.8, "top_p": 0.9},
            "num_sm": NUM_SM, "registered_splits": S, "S_together": [S], "S_alone": [S], "requests": n, "differ": [],
            "tokens_together": _tokens(n), "tokens_alone": _tokens(n)}


def test_a_serve_whose_tokens_and_split_counts_hold_passes(tmp_path, monkeypatch, capsys):
    doc = _Case(monkeypatch).served(tmp_path)
    assert S == 32 and doc == _passing()
    assert (tmp_path / "served.json").read_text() == json.dumps(_passing(), indent=1) + "\n"
    said = {k: v for k, v in _passing().items() if k not in ("tokens_together", "tokens_alone")}
    assert capsys.readouterr().out == json.dumps(said) + "\n"


@pytest.mark.parametrize(("n", "seqs"), [(3, 8), (8, 8), (9, 9)])
def test_the_engine_is_built_for_the_workload_once_the_engine_environment_is_applied(tmp_path, monkeypatch, n, seqs):
    case = _Case(monkeypatch, n)
    assert case.served(tmp_path) == _passing(n)
    assert case.log[:3] == [("env",), ("load", "a-workload.json"),
                            ("build", {"max_num_seqs": seqs, "case": "a-case", "engine_args": {"seed": 7},
                                       "target": ("target of", "a-row"), "execution": "an-execution"})]


def test_the_requests_are_served_together_and_then_each_alone_through_the_proxies(tmp_path, monkeypatch):
    case = _Case(monkeypatch)
    case.served(tmp_path)
    assert case.log[3:] == [("serve", "an engine", ids, SAMPLING, True) for ids in (["r2", "r1", "r0"], ["r2"], ["r1"], ["r0"])]
    assert {name: getattr(case.module, name) for name in case.kernels} == case.kernels


def test_a_request_whose_tokens_differ_alone_fails_and_is_named(tmp_path, monkeypatch):
    doc = _Case(monkeypatch, alone={"r2": [102, 9], "r0": [100]}).served(tmp_path)
    alone = {"r2": [102, 9], "r1": [101, 1], "r0": [100]}
    assert doc == {**_passing(), "ok": False, "differ": ["r0", "r2"], "tokens_alone": alone}
    assert (tmp_path / "served.json").read_text() == json.dumps(doc, indent=1) + "\n" and list(doc["tokens_alone"]) == list(alone)


def test_a_request_with_no_tokens_alone_fails_and_is_named(tmp_path, monkeypatch):
    doc = _Case(monkeypatch, alone={"r1": None}).served(tmp_path)
    assert doc == {**_passing(), "ok": False, "differ": ["r1"], "tokens_alone": {"r0": [100, 0], "r2": [102, 2]}}


@pytest.mark.parametrize(("splits", "together", "alone"), [
    ({"s_together": (S, S // 2)}, [S // 2, S], [S]),
    ({"s_alone": {"r1": (S // 2,)}}, [S], [S // 2, S]),
    ({"s_alone": {"r0": (2 * S, S)}}, [S], [S, 2 * S]),
    ({"s_together": (S // 2,), "s_alone_else": (S // 2,)}, [S // 2], [S // 2]),
    ({"s_together": (), "s_alone_else": ()}, [], []),
    ({"s_together": ()}, [], [S]),
    ({"s_alone_else": ()}, [S], []),
])
def test_a_launch_at_another_split_count_or_no_launch_fails_with_the_tokens_equal(tmp_path, monkeypatch, splits, together, alone):
    doc = _Case(monkeypatch, **splits).served(tmp_path)
    assert doc == {**_passing(), "ok": False, "S_together": together, "S_alone": alone}


def test_a_process_that_is_not_pinned_fails_whatever_it_served(tmp_path, monkeypatch):
    doc = _Case(monkeypatch, pinned=False).served(tmp_path)
    assert doc == {**_passing(), "ok": False, "pinned": False}


@pytest.mark.parametrize(("case", "code"), [({}, 0), ({"alone": {"r1": [1]}}, 1), ({"s_together": (2 * S,)}, 1), ({"pinned": False}, 1)])
def test_the_command_exits_with_the_served_verdict(tmp_path, monkeypatch, case, code):
    world, out = _Case(monkeypatch, **case), tmp_path / "made" / "here"
    with pytest.raises(SystemExit) as e:
        tc.main(cli.parse("topp-pin-check", ["served", "--workload", "a-workload.json", "--out", str(out)], {}))
    assert e.value.code == code and json.loads((out / "served.json").read_text())["ok"] is (code == 0)
    assert world.log[1] == ("load", "a-workload.json")
