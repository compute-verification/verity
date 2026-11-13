"""A rebuilt candidate's check carries the passes of the checks it replaces (`research.vqueue_slim`'s ``after``): the
simulated queue of `test_vqueue.World`, each candidate's check a real `vqueue_slim.start` of three shards on a fake
driver, each task keyed by the content of the members that meet its inputs.  A drop cancels its check's shards, each
having passed some of its tasks; each run's record reaches the store a while after (most within `CARRY_WAIT_S`, some
later, some left running by `KEEP_RUNS`), and the loop's log gets the drop's line.  A rebuild launches as the loop's
ticks find it (`judge`, every `POLL_S`), and its check reuses only the packs of the runs its launch names that the store
holds then (`check.carried_packs`), as though newest-selection and `GATHER_BYTES` had left every one of them out."""
from __future__ import annotations

import contextlib
import os
import random
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from research import vqueue_slim
from research.landing import Landing
from research.vqueue import PENDING, Drop, meets
from research.vqueue_loop import stamp
from research.vqueue_slim import CARRY_ENV, CARRY_WAIT_S, judge, replaced, start
from test_vqueue import MAIN, SEEDS, TICK_S, World, ticks

#: how often the loop's ticks judge a waiting check, within one of the world's ticks
POLL_S = 60.0
SHARDS = 3


class Store:
    """The store's attempts as `vqueue_slim.left` reads them: a run's, once its record was uploaded."""

    def __init__(self, w: CarryWorld):
        self.w = w

    def get_attempt(self, rid: str) -> dict | None:
        up = self.w.upload.get(rid)
        return {"id": rid} if up is not None and up <= self.w.clock else None


def driver(w: CarryWorld) -> Landing:
    """The loop's `Landing` on the world's clock and store, whose launch of each shard registers one run, which keeps the runs
    the environment it hands train.sh names (`CARRY_ENV`); every candidate is sent the same arguments."""
    def launch(repo, commit, machine, agreement) -> tuple[list[str], str]:
        rid = f"r-{len(w.named)}"
        w.named[rid] = [r for r in dict(os.environ).get(CARRY_ENV, "").split(",") if r]
        return [rid], ""

    q = SimpleNamespace(shadow=False, log=lambda text: None, store=Store(w), workdir=Path("/nonexistent"), auth=())
    d = Landing(q, on="m1,m2,m3", launch=launch, now=lambda: w.clock)
    d.beside_args = lambda sha, base: ({"base": base}, [], "")
    return d


class CarryWorld(World):
    """`World`, its candidates' checks run as shards whose passes a rebuild carries; `step` checks the invariant at each
    launch and `World`'s own at each tick."""

    def __init__(self, seed: int, **kw):
        super().__init__(seed, **kw)
        self.crng = random.Random(seed + 1_000_003)   # apart from the world's own stream, which scripts its queue
        self.clock = 0.0
        self.driver = driver(self)
        self.content: dict[str, dict[int, str]] = {MAIN: {}}
        self.tasks = {**{f"suite:{m}": (f"pkg{m}/",) for m in self.marks}, "shared": ("shared/",), "docs": ("docs/",)}
        self.lines: list[dict] = []
        self.named: dict[str, list[str]] = {}
        self.upload: dict[str, float | None] = {}
        self.passes: dict[str, set[tuple]] = {}
        self.checks: dict[str, list[str]] = {}
        self.waiting: dict[str, dict] = {}
        self.after: dict[str, tuple[float, dict[str, float]]] = {}

    def task_key(self, task: str, sha: str) -> tuple:
        return task, frozenset((m, h) for m, h in self.content[sha].items() if meets(self.marks[m].files, self.tasks[task]))

    def passes_at(self, task: str, sha: str) -> bool:
        return not any(task == f"suite:{m}" for m in self.tree[sha] & (set(self.bad) | set(self.erring)))

    def shard_tasks(self, k: int) -> list[str]:
        return sorted(self.tasks)[k::SHARDS]

    def events(self) -> str | None:
        was = self.main
        if (moved := super().events()) is not None:
            self.content[moved] = self.content[was]
        return moved

    def apply(self, a) -> None:
        if isinstance(a, Drop):
            self.cancel(a.sha)
        super().apply(a)
        if type(a).__name__ == "Build":
            sha = f"q{self.made - 1}"
            c = self.cands[sha]
            self.content[sha] = {**self.content[c.base], **dict(zip(c.members, c.heads))}
            after = replaced(self.lines, c.members, self.clock)
            rec = start(self.driver, sha, c.base, shards=SHARDS, **({"after": after} if after else {}))
            assert rec is not None, f"seed {self.seed}: {sha} registered no run"
            self.after[sha] = (self.clock, after)
            if "wait" in rec:
                assert after and not rec["ids"], rec
                self.waiting[sha] = rec
            else:
                self.launched(sha, rec)

    def cancel(self, sha: str) -> None:
        """The loop's drop of ``sha``: its check's shards stopped, each with the passes it had, its record uploaded a while
        after (now and then never, or by a run `KEEP_RUNS` left running, much later); a check still waiting never launches."""
        c, ids, kept = self.cands[sha], [], {}
        self.waiting.pop(sha, None)
        ended = c.verdict.state != PENDING and c.verdict.ended or c.verdict.state not in (PENDING, "fail")
        for k, rid in enumerate(self.checks.get(sha, ())):
            ids.append(rid)
            self.passes[rid] = {self.task_key(t, sha) for t in self.shard_tasks(k)
                                if self.passes_at(t, sha) and (ended or self.crng.random() < 0.5)}
            roll = self.crng.random()
            if ended:
                self.upload[rid] = self.clock - 1
            elif roll < 0.65:
                self.upload[rid] = self.clock + self.crng.randint(5, int(CARRY_WAIT_S) - 1)
            elif roll < 0.8:
                self.upload[rid] = self.clock + self.crng.randint(int(CARRY_WAIT_S) + 1, 500)
            elif roll < 0.95:
                self.upload[rid] = self.clock + self.crng.randint(600, 1500)
                kept[rid] = "left running: /data/vqueue/live/keep-runs names it"
            else:
                self.upload[rid] = None
        self.lines.append({"action": "drop", "at": stamp(self.clock), "sha": sha, "members": list(c.members),
                           **({"beside_cancelled": ids} if ids else {}), **({"beside_cancel_failed": kept} if kept else {})})

    def launched(self, sha: str, rec: dict) -> None:
        """``sha``'s check launched: its shards' runs, and the invariant: it reruns no task whose key a pass of a run it
        replaces holds, of each run whose record was uploaded within `CARRY_WAIT_S` of its cancel, and it waited no longer
        than that past the last cancel and a tick of the loop."""
        built, after = self.after[sha]
        ids = rec["ids"]
        assert len(ids) == SHARDS and "wait" not in rec, rec
        self.checks[sha] = ids
        named = self.named[ids[0]]
        assert all(self.named[i] == named for i in ids), f"seed {self.seed}: {sha}'s shards name different runs"
        listed = {k for r in named if (up := self.upload.get(r)) is not None and up <= self.clock for k in self.passes[r]}
        assert self.clock <= max(after.values(), default=built) + BOUND + POLL_S, (
            f"seed {self.seed}: {sha} launched {self.clock - built:.0f} s after its build, past its bound")
        for rid, at in sorted(after.items()):
            if (up := self.upload.get(rid)) is None or up > at + BOUND:
                self.seen["launched without a late record"] += up is not None and up > self.clock
                continue
            for task in self.tasks:
                if (k := self.task_key(task, sha)) in self.passes[rid]:
                    assert k in listed, (f"seed {self.seed}: {sha} reruns {task} at {sorted(k[1])}, which {rid} passed "
                                         f"and uploaded {up - at:.0f} s after its cancel; its launch "
                                         f"{self.clock - at:.0f} s after it named {named}")
                    self.seen["a pass carried"] += 1
        self.seen["waited for a record"] += self.clock > built
        self.seen["launched at once beside a drop"] += bool(after) and self.clock == built

    def step(self) -> None:
        self.clock = (self.tick + 1) * TICK_S
        super().step()
        for k in range(1, int(TICK_S // POLL_S)):
            self.clock = self.tick * TICK_S + k * POLL_S
            for sha, rec in list(self.waiting.items()):
                got = judge(self.driver, rec)
                assert got.state == PENDING and "verdict" not in rec, (sha, got)
                if rec["ids"]:
                    del self.waiting[sha]
                    self.launched(sha, rec)


BOUND = CARRY_WAIT_S


@pytest.fixture(autouse=True)
def every_step_declared(monkeypatch):
    monkeypatch.setattr(vqueue_slim, "declared_steps", lambda driver, sha, steps: steps)


def settle(seeds=SEEDS) -> Counter[str]:
    seen: Counter[str] = Counter()
    for seed in seeds:
        w = CarryWorld(seed)
        while not w.done():
            assert w.tick < ticks(w), f"seed {seed}: not settled after {ticks(w)} ticks"
            w.step()
        assert not w.waiting, f"seed {seed}: a check still waits once the queue settled: {sorted(w.waiting)}"
        seen.update(w.seen)
    return seen


def test_a_rebuild_reruns_no_task_a_shard_it_replaces_passed_once_its_record_is_in_within_the_bound():
    """Over every seed: a check replacing dropped candidates' checks reruns no task whose key a pass of theirs holds, of
    each run whose record reached the store within `CARRY_WAIT_S` of its cancel; it launches within that bound and a tick
    of the loop after the last cancel; and `World`'s own invariants hold, the queue settling as it does without the wait.
    Each situation happens in some seed."""
    seen = settle()
    for what in ("a pass carried", "waited for a record", "launched at once beside a drop", "launched without a late record",
                 "landed as built", "bisected"):
        assert seen[what], f"no seed exercised: {what}"


def test_a_rebuild_launched_without_waiting_reruns_passes_the_shards_it_replaces_left():
    """The invariant has teeth: no wait (a bound of 0 in `vqueue_slim`) launches before records the store gets in time."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(vqueue_slim, "CARRY_WAIT_S", 0)
        with pytest.raises(AssertionError, match="reruns"):
            settle()


def test_a_rebuild_naming_no_run_reruns_passes_the_shards_it_replaces_left():
    """Nor does waiting alone carry them: a launch that names no run lists none of their packs."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(vqueue_slim, "_carrying", lambda runs: contextlib.nullcontext())
        with pytest.raises(AssertionError, match="reruns"):
            settle()
