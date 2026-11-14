"""`replay_bundle.defer` at the C2 site of a deferred run: the slim bundle of a planned replay's reads, a plan that fails, a store
over its bound and a write that was started beside the checks, with the line each logs, and when the placed identities go
with the bundle.  CPU only; sealing and opening a bundle are `test_replay_bundle.py`."""
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
from verity_vllm.pipeline import replay_plan
from verity_vllm.records import schema_ids

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
        """`defer` for a run of no request, or of the `reqs`, `tokens` and `ids_full` given."""
        run: dict[str, Any] = {"reqs": [], "tokens": {}, "ids_full": None} | kw
        return RB.defer(self.out, 1, self.com, self.rc, self.args, tl=self.tl, arm="instrumented", versions={}, sampling={},
                        stop_of_record=None, declaration=None, required_classes=[], row=self.row, deferred=self.deferred, **run)

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
    assert logged() == [("[delta] REPLAY BUNDLE (pair 1): REFUSED -- 0.0 GB is over the 0 GB bound (--replay-bundle-cap-gb); the C2 replay runs "
                          "in this process")]


@pytest.mark.parametrize("tokens, kept", [({"a": [5, 6, 7], "b": [8]}, False), ({"a": [5, 6], "b": [8]}, True),
                                          ({"a": [5, 6, 7], "b": []}, True), ({"a": [5, 6, 7, 9], "b": [8]}, False)])
def test_the_placed_identities_go_with_the_bundle_only_when_a_request_stopped_early(tmp_path: Path, tokens: dict[str, list[int]],
                                                                                    kept: bool) -> None:
    """The replay reconciles what was not executed against the identities at the steps they were placed at, and reads them only
    for a run in which a request stopped early: one whose served tokens are fewer than its `max_tokens`.  A request that
    served as many, or more, is not one."""
    site = Site(tmp_path, replay_deferred="full")
    reqs = [{"request_id": "a", "max_tokens": 3}, {"request_id": "b", "max_tokens": 1}]
    identities = [{"member": MEMBER, "engine_step": 0}]
    assert site.defer(reqs=reqs, tokens=tokens, ids_full=identities)
    context = site.deferred[1]
    assert (context["requests"], context["tokens"], context["ids_full"]) == (reqs, tokens, identities if kept else None)


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


NO_TREE = "replay bundle: the committer kept no weights tree for the run's committed weights root (register_weights)"


@pytest.mark.parametrize("overlap", [1, 0], ids=["beside-the-checks", "at-the-site"])
@pytest.mark.parametrize("tree", [None, {}, {"root": "ab" * 64, "names": []}], ids=["none", "empty", "another-root"])
def test_a_committer_that_kept_no_tree_of_the_run_s_committed_weights_is_refused_where_the_write_is_taken(tmp_path: Path,
                                                                                                         tree: dict[str, Any] | None,
                                                                                                         overlap: int) -> None:
    """A bundle carries the committed weights tree: a committer that kept none, or one of another root than the run registered, is
    refused by name.  `start` reads the tree before it returns and keeps the refusal for its writer, so it is raised where the
    write is taken, by `defer`, as it is for a write made there; a write started beside the checks has then written nothing."""
    site = Site(tmp_path, replay_deferred="full", replay_bundle_overlap=overlap)
    site.com.weights_tree = tree
    started = RB.start(site.out, 1, site.com, site.rc, site.args, None)
    assert (started is not None) == bool(overlap)
    with pytest.raises(ValueError, match=f"^{re.escape(NO_TREE)}$"):
        site.defer(started=started)
    assert site.row == {"commit": {}} and site.deferred == {} and (started is None or not RB.partial(site.out, 1).exists())


def test_a_registered_tensor_of_another_size_than_its_tree_entry_is_refused_by_name(tmp_path: Path) -> None:
    """The bundle carries the bytes of the registered buffers, read from the model: one that holds another number of bytes than the
    tree registers for it is refused with both; a tensor the tree names that is not a buffer is not read at all."""
    torch = pytest.importorskip("torch")
    site = Site(tmp_path, replay_deferred="full")
    cos = torch.arange(4, dtype=torch.int32)
    site.com.model.register_buffer("cos", cos)
    site.com.weights_tree["names"] = [["w", "bfloat16", [2], 4], ["cos", "int32", [4], 12]]
    with pytest.raises(ValueError, match="^replay bundle: cos holds 16 B, registered 12 B$"):
        RB.weights_extra(site.com, site.rc)
    site.com.weights_tree["names"][1][3] = 16
    tree, extra = RB.weights_extra(site.com, site.rc)
    assert tree is site.com.weights_tree and [(i, name, bytes(raw)) for i, name, raw in extra] == [(1, "cos", cos.numpy().tobytes())]


def test_a_partial_bundle_an_earlier_commit_left_is_removed_before_the_store_is_written(tmp_path: Path) -> None:
    """A Commit that ended before its bundle was sealed leaves the partial dir; the next write of the pair's bundle starts from
    none, so the bundle holds the files of its own write and no other."""
    again, fresh = Site(tmp_path / "again", replay_deferred="full"), Site(tmp_path / "fresh", replay_deferred="full")
    left = RB.partial(again.out, 1) / "store" / "raw" / "left.bin"
    left.parent.mkdir(parents=True)
    left.write_bytes(b"an earlier Commit's")
    assert again.defer() and fresh.defer()

    def files(site: Site) -> set[str]:
        return {f.relative_to(RB.partial(site.out, 1)).as_posix() for f in RB.partial(site.out, 1).rglob("*") if f.is_file()}
    assert not left.exists() and files(again) == files(fresh) and "weights/tree.json" in files(again)


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
    monkeypatch.setattr(replay_plan, "Planner", planner)
    slim = flags()
    assert RB.prepare(slim) == "the planner" and started == [slim]
    assert RB.prepare(flags(replay_deferred="full")) is None and RB.prepare(flags(replay_deferred=None)) is None
    with pytest.raises(SystemExit, match="--replay-deferred: no --program-dir / --programs-root"):
        RB.prepare(flags(program_dir=None))
    assert started == [slim]


def test_the_bundle_reads_its_schema_id_from_the_records_package() -> None:
    """The record's id is written once, in `records.schema_ids`: its writer here and a reader elsewhere cannot hold two."""
    assert RB.SCHEMA is schema_ids.REPLAY_BUNDLE_SCHEMA
