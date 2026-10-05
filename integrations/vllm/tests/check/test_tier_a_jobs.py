"""Tier A on forked workers (`replay.tier_a(..., jobs=N)`, `check/replay/tier_a_pool.py`) is the sequential tier: at one
seed the same instances, child members, counts and verdict, so the same report bytes apart from `jobs`, `seconds` and
`max_rss_gb`.  On the adversarial toy (`tests/check/adversarial.py`) at 5 seeds, over the three ways a stratum's draw goes:
whole instances, child units (members drawn per candidate), and skips that a stratum refills from its order."""
from __future__ import annotations

import copy
import gc
import json
import os
import signal
from collections import defaultdict

import pytest

from tests.check import adversarial as G
from verity_vllm.check.replay import fork, replay

TIMING = ("jobs", "seconds", "max_rss_gb")
CASES = {"instances": dict(k=1), "child units": dict(k=1, max_arg_leaves=100), "refills": dict(k=2, holes=True),
         "child refills": dict(k=2, max_arg_leaves=100, holes=True)}


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    rec = G.ToyRecord(tmp_path_factory.mktemp("tier_a_jobs"))
    try:
        rec.bake()
        yield rec
    finally:
        rec.restore()


def _bytes(report: dict) -> str:
    def strip(x):
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items() if k not in TIMING}
        return [strip(v) for v in x] if isinstance(x, list) else x
    return json.dumps(replay._jsonable(strip(report)), indent=1)


def _holed(store):
    """The store without the snapshots of every third event: those instances, and their consumers, are skipped."""
    holed = copy.copy(store)
    holed.by_event = defaultdict(list, {e: v for e, v in store.by_event.items() if e % 3})
    return holed


def _refilled(report: dict) -> list[str]:
    """Granularities of the strata that evaluated more candidates than they need (a skip was refilled)."""
    seen: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for e in report["instances"]:
        seen[(e["spec"], e["regime"])].append(e)
    return [es[0]["granularity"] for (spec, rg), es in seen.items()
            if len(es) > min(report["k"], report["coverage"][spec]["regimes"][rg]["instances"])]


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("case", list(CASES))
def test_jobs_4_gives_the_jobs_1_report(toy, case, seed):
    kw = dict(CASES[case])
    store = _holed(toy.store) if kw.pop("holes", False) else toy.store
    one = replay.tier_a(toy.genuine, store, toy.weights, seed=seed, jobs=1, **kw)
    four = replay.tier_a(toy.genuine, store, toy.weights, seed=seed, jobs=4, **kw)
    assert (one["jobs"], four["jobs"]) == (1, 4)
    assert _bytes(four) == _bytes(one)
    granularity = {c["granularity"] for c in one["coverage"].values()}
    child = {g for g in granularity if g and g.startswith("child:")}
    assert any(rc["instances"] > kw["k"] for c in one["coverage"].values() for rc in c["regimes"].values())   # a draw to make
    assert bool(child) == ("max_arg_leaves" in kw)
    if "refills" in case:
        refilled = _refilled(one)
        assert refilled and (not child or child & set(refilled))
    else:
        assert one["status"] == "pass" and not _refilled(one)


def test_a_worker_killed_mid_task_fails_the_tier_by_name(toy, monkeypatch, capsys):
    """A SIGKILLed worker (the cgroup's OOM killer) loses its task: tier A raises `WorkerDied` with the reason rather than wait
    for the task or report without it."""
    parent, evaluate = os.getpid(), replay.Replayer.evaluate_instance

    def killed(self, m):
        if os.getpid() != parent:
            os.kill(os.getpid(), signal.SIGKILL)
        return evaluate(self, m)

    monkeypatch.setattr(replay.Replayer, "evaluate_instance", killed)
    monkeypatch.setattr(fork, "oom_kills", lambda: 0)
    monkeypatch.setattr(fork, "POOL_POLL_S", 0.01)
    with pytest.raises(fork.WorkerDied, match="exited mid-task"):
        replay.tier_a(toy.genuine, toy.store, toy.weights, k=1, seed=0, jobs=4)
    assert "exited mid-task" in capsys.readouterr().err
    assert gc.get_freeze_count() == 0


def test_replay_refuses_fewer_than_one_job(tmp_path, capsys):
    from verity_vllm.pipeline import cli

    with pytest.raises(SystemExit) as e:
        cli.main(["replay", "--log", str(tmp_path / "log.jsonl.gz"), "--profile", "unused", "--tier", "a", "--seed", "1", "--jobs", "0",
                  "--out", str(tmp_path / "out.json")])
    assert e.value.code == 2 and "--jobs must be >= 1" in capsys.readouterr().err and not (tmp_path / "out.json").exists()
