"""`replay_bundle.defer` at the C2 site of a deferred run: the slim bundle of a planned replay's reads, a plan that fails, a store
over its bound and a write that was started beside the checks, with the line each logs.  CPU only; sealing and opening a
bundle are `test_replay_bundle.py`."""
from __future__ import annotations

import logging
import re
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import verity_vllm.pipeline.replay_bundle as RB
from tests.check.opened import commit_steps
from tests.commit.test_store_dump import STEPS
from tests.pipeline.test_c2_replay_run import Spans
from verity_vllm.check import oracle_compare as OC
from verity_vllm.commit import store_dump as SD

MEMBER = "model.layers.0.self_attn.qkv_proj"
Lines = Callable[[], list[str]]


@pytest.fixture
def logged(caplog: pytest.LogCaptureFixture) -> Lines:
    """The lines `replay_bundle` has logged so far in this test."""
    caplog.set_level(logging.INFO, logger=RB.__name__)
    return lambda: [r.getMessage() for r in caplog.records if r.name == RB.__name__]


class Site:
    """One committed run at its C2 site: the committer, its run commitment, the run's row and the Commit's deferred contexts."""

    def __init__(self, out: Path, **args: Any) -> None:
        torch = pytest.importorskip("torch")
        self.out, self.tl = out, Spans()
        self.row: dict[str, Any] = {"commit": {}}
        self.deferred: dict[int, dict[str, Any]] = {}
        self.com, self.rc = commit_steps(STEPS)
        registered = self.rc.weight_registration
        self.com.weights_tree, self.com.model = {"root": registered["root"] if registered else None, "names": []}, torch.nn.Module()
        self.args = SimpleNamespace(**{"replay_deferred": "slim", "retain": "host", "required_manifest": None, "openings": 64} | args)

    def defer(self, **kw: Any) -> bool:
        return RB.defer(self.out, 1, self.com, self.rc, self.args, tl=self.tl, arm="instrumented", versions={}, reqs=[], sampling={},
                        tokens={}, stop_of_record=None, ids_full=None, declaration=None, required_classes=[], row=self.row,
                        deferred=self.deferred, **kw)

    def store_bytes(self) -> int:
        return sum(p.stat().st_size for p in (RB.partial(self.out, 1) / "store").rglob("*.bin"))

    def nothing_deferred(self) -> bool:
        return (not RB.partial(self.out, 1).exists() and self.row == {"commit": {}} and self.deferred == {}
                and self.tl.calls == [("start", "validate.replay_bundle", {"pair": 1}),
                                      ("end", "validate.replay_bundle", {"pair": 1, "ok": False})])


def test_a_planned_replay_defers_to_a_slim_bundle_of_the_reads_it_made(tmp_path: Path, logged: Lines) -> None:
    site, full = Site(tmp_path / "slim"), Site(tmp_path / "full", replay_deferred="full")

    def plan() -> dict[str, Any]:
        assert OC.OpenedReader(site.com, site.rc)(0, MEMBER, 10, 20) == b"\x07" * 10
        return {"replay": {"picked": 1, "unresolved_n": 0, "unresolved": []}, "linkage": {"pairs": 0}}
    assert site.defer(plan=plan) and full.defer()
    slim = site.deferred[1]["slim"]
    assert {k: v for k, v in slim.items() if k not in ("bytes", "plan_s")} == {"picked": 1, "unresolved_n": 0, "unresolved": [],
                                                                               "linkage": {"pairs": 0}}
    assert 0 < slim["bytes"] < SD.nbytes(site.com) and 0 < site.store_bytes() < full.store_bytes()
    assert full.deferred[1]["slim"] is None
    assert (site.row["commit"]["sampled_replay"], site.row["commit"]["boundary_linkage"]) == RB.deferred_records(RB.path(site.out, 1))
    assert site.tl.calls == [("start", "validate.replay_bundle", {"pair": 1}), ("end", "validate.replay_bundle", {"pair": 1})]
    planned, deferred, deferred_full = logged()
    assert re.fullmatch(rf"\[delta\] REPLAY BUNDLE \(pair 1\): slim -- 1 picks \+ linkage read {slim['bytes'] >> 20} MiB of the store in "
                        r"[0-9.e-]+ s \(unresolved 0\)", planned), planned
    assert deferred == f"[delta] C2 SAMPLED REPLAY + LINKAGE (pair 1): DEFERRED -- store and weights tree written to {RB.partial(site.out, 1)}"
    assert deferred_full == deferred.replace(str(site.out), str(full.out))


@pytest.mark.parametrize("planned", [{}, {"replay": None}, {"replay": {}}])
def test_a_plan_with_no_replay_of_its_own_defers_to_a_slim_bundle_that_names_no_pick(tmp_path: Path, logged: Lines,
                                                                                    planned: dict[str, Any]) -> None:
    """A plan may hold no `replay` (absent, null or empty): the slim bundle then says its picks and what it left unresolved as
    unknown."""
    site = Site(tmp_path)
    assert site.defer(plan=lambda: planned | {"linkage": {"pairs": 0}})
    slim = site.deferred[1]["slim"]
    assert list(slim) == ["bytes", "picked", "unresolved_n", "unresolved", "linkage", "plan_s"]
    assert {k: v for k, v in slim.items() if k != "plan_s"} == {"bytes": 0, "picked": None, "unresolved_n": None, "unresolved": None,
                                                                "linkage": {"pairs": 0}}
    assert re.fullmatch(r"\[delta\] REPLAY BUNDLE \(pair 1\): slim -- None picks \+ linkage read 0 MiB of the store in [0-9.e-]+ s "
                        r"\(unresolved None\)", logged()[0]), logged()[0]


def test_a_plan_that_fails_defers_nothing_and_the_replay_runs_in_process(tmp_path: Path, logged: Lines) -> None:
    site = Site(tmp_path)

    def plan() -> dict[str, Any]:
        raise RuntimeError("no Programs of record")
    assert site.defer(plan=plan) is False and site.nothing_deferred()
    assert logged() == ["[delta] REPLAY BUNDLE (pair 1): no plan (RuntimeError('no Programs of record')); the C2 replay runs in this process"]


def test_a_store_over_its_bound_defers_nothing_and_the_replay_runs_in_process(tmp_path: Path, logged: Lines) -> None:
    site = Site(tmp_path, replay_deferred="full", replay_bundle_cap_gb=0)
    assert site.defer() is False and site.nothing_deferred()
    assert logged() == [("[delta] REPLAY BUNDLE (pair 1): REFUSED -- 0.0 GB is over the 0 GB cap (--replay-bundle-cap-gb); the C2 replay runs "
                          "in this process")]


def test_a_write_started_beside_the_checks_is_the_one_defer_takes(tmp_path: Path, logged: Lines) -> None:
    site = Site(tmp_path, replay_deferred="full", replay_bundle_overlap=1)
    started = RB.start(site.out, 1, site.com, site.rc, site.args, None)
    assert started is not None and site.defer(started=started)
    assert site.deferred[1]["engine_of_record"] == started.engine and site.deferred[1][RB.HASHED] == started.hashed and site.store_bytes() > 0
    assert RB.start(site.out, 1, site.com, site.rc, site.args, SimpleNamespace(entries=["pouw:ncp-v2"])) is None, "a live-only run defers nothing"
    taken, deferred = logged()
    assert re.fullmatch(r"\[delta\] REPLAY BUNDLE \(pair 1\): store written in [0-9.]+ s beside the checks before the C2 site "
                        r"\(waited [0-9.]+ s here\)", taken), taken
    assert deferred == f"[delta] C2 SAMPLED REPLAY + LINKAGE (pair 1): DEFERRED -- store and weights tree written to {RB.partial(site.out, 1)}"


def flags(**more: Any) -> SimpleNamespace:
    """The Commit flags `refuse` reads, for a slim deferral it takes."""
    return SimpleNamespace(**{"replay_deferred": "slim", "program_dir": "/p", "programs_root": None, "arms": None, "only_arm": None,
                              "compiled_taps": "off", "vu_export_dir": None, "serving_rows": None} | more)


def test_a_slim_deferral_starts_its_planner_on_the_commit_s_flags_and_no_other_does(monkeypatch: pytest.MonkeyPatch) -> None:
    """`prepare` starts the slim bundle's planner on the Commit's own flags, once they are ones a deferral runs with; a full
    deferral and a Commit that defers nothing start none."""
    started: list[Any] = []

    def planner(args: Any) -> str:
        started.append(args)
        return "the planner"
    monkeypatch.setattr(RB, "Planner", planner)
    slim = flags()
    assert RB.prepare(slim) == "the planner" and started == [slim]
    assert RB.prepare(flags(replay_deferred="full")) is None and RB.prepare(flags(replay_deferred=None)) is None
    with pytest.raises(SystemExit, match="--replay-deferred: no --program-dir / --programs-root"):
        RB.prepare(flags(program_dir=None))
    assert started == [slim]
