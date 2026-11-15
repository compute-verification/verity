"""A snapshot queue's properties, checked through `snapshots.Driver` over seeded random scenarios (`harness.random_scenario`),
each event order a seed.  The queue here is the contract's small fake (`fake_vqueue`); `run`, `_live` and `SEEDS` take any
snapshot queue the driver runs, `research.vqueue`'s `decide` among them.

* soundness: `main` moves only to a tree built on it whose every listed gating task passed on that tree, by a result the
  world delivered for that tree, and which holds no change that breaks a task;
* liveness: once nothing is left to deliver, every PR still ready and reviewed has landed or been ejected, each ejection
  naming a task that failed on a tree holding it.  ``exact`` adds that, with no false fails and no interacting pairs,
  exactly the PRs that break nothing land and each ejected PR breaks the task its ejection names, which a queue that ejects
  every member when no failure names one, as the fake does, doesn't keep;
* an infra outcome never ejects a PR.

Each check also runs against a queue broken on purpose, which it must catch, so a pass isn't a pass for the wrong reason."""
from __future__ import annotations

import random
import types
from dataclasses import dataclass

import fake_vqueue as vq
import pytest

from harness import MAIN0, final_heads, outcomes, random_scenario, run
from verity_landq import model as lq
from verity_landq import snapshots

SEEDS = range(300)


def drive(s, q=vq, limit: int = 20_000):
    d = snapshots.Driver(q)
    return run(s, d.step, d.start(MAIN0), limit)


def _live(r, exact: bool) -> list[str]:
    why = []
    if r.pending:
        why.append(f"never quiesced: {r.pending} events left after {r.steps}")
    if r.state.stack:
        why.append(f"candidates still in flight: {r.state.stack}")
    landed, ejected = outcomes(r)
    spec = {p.pr: p for p in r.scenario.prs}
    for pr, head in sorted(final_heads(r).items()):
        if landed.get(pr) != head and (pr not in ejected or ejected[pr].head != head):
            why.append(f"#{pr} at {head} neither landed nor ejected")
    for pr, ej in sorted(ejected.items()):
        failed = [e for e in r.log if isinstance(e, lq.Result) and e.status == lq.FAIL and e.tree == ej.tree
                  and e.task == ej.task]
        if not failed or (pr, ej.head) not in r.trees[ej.tree]:
            why.append(f"#{pr} ejected naming {ej.task} on {ej.tree}, which no delivered fail of a tree holding it says")
        if exact and ej.task not in spec[pr].breaks:
            why.append(f"#{pr} ejected for {ej.task}, which it doesn't break")
    if exact:
        for pr, head in sorted(final_heads(r).items()):
            if not spec[pr].breaks and landed.get(pr) != head:
                why.append(f"#{pr} breaks nothing and didn't land")
            if spec[pr].breaks and landed.get(pr) == head:
                why.append(f"#{pr} breaks {sorted(spec[pr].breaks)} and landed")
    return why


def broken(decide, **more):
    """The fake queue with ``decide`` in place of its own, and ``more`` beside its types."""
    q = types.SimpleNamespace(**{k: getattr(vq, k) for k in dir(vq) if not k.startswith("_")}, **more)
    q.decide = decide
    return q


@dataclass(frozen=True)
class Rerun:
    sha: str
    keys: tuple[str, ...]


@pytest.mark.parametrize("flaky,pairs,infra", [(0, 0, 0.0), (0, 0, 0.3), (3, 0, 0.0), (2, 2, 0.2)])
def test_main_moves_only_to_a_tree_that_passed_there(flaky, pairs, infra) -> None:
    moves = ejects = 0
    for seed in SEEDS:
        r = drive(random_scenario(random.Random(seed), flaky=flaky, pairs=pairs, infra=infra))
        assert not r.unsound, (seed, r.unsound[:2])
        moves += len(r.mains) - 1
        ejects += sum(isinstance(d, lq.Eject) for d in r.decisions)
    assert moves > len(SEEDS) and ejects > len(SEEDS) // 4, (moves, ejects)


@pytest.mark.parametrize("flaky,pairs,infra", [(0, 0, 0.0), (0, 0, 0.3), (3, 0, 0.2), (0, 2, 0.2), (2, 2, 0.2)])
def test_every_pr_lands_or_is_ejected_naming_a_task_that_failed_on_a_tree_holding_it(flaky, pairs, infra) -> None:
    for seed in SEEDS:
        r = drive(random_scenario(random.Random(seed), flaky=flaky, pairs=pairs, infra=infra))
        assert not _live(r, exact=False), (seed, _live(r, exact=False)[:3])


def test_an_infra_outcome_never_ejects() -> None:
    retries = pages = 0
    for seed in SEEDS:
        r = drive(random_scenario(random.Random(seed), bad=0.0, infra=0.5))
        assert not [d for d in r.decisions if isinstance(d, lq.Eject)], seed
        landed, _ = outcomes(r)
        assert all(landed.get(pr) == head for pr, head in final_heads(r).items()), seed
        retries += sum(isinstance(d, lq.Retry) for d in r.decisions)
        pages += sum(isinstance(d, lq.Page) for d in r.decisions)
    assert retries > len(SEEDS) and pages > 0, (retries, pages)


def test_soundness_catches_a_queue_that_advances_on_a_fail() -> None:
    def advances_on_a_fail(s):
        c = s.candidates[0] if s.candidates else None
        return [vq.Advance(c.sha)] if c is not None and c.verdict.state == vq.FAIL else vq.decide(s)

    q = broken(advances_on_a_fail)
    caught = next((s for s in SEEDS if drive(random_scenario(random.Random(s)), q).unsound), None)
    assert caught is not None, "the soundness check passed a queue that advances on a fail"


def test_the_infra_check_catches_a_queue_that_ejects_on_infra() -> None:
    def ejects_on_infra(s):
        c = s.candidates[0] if s.candidates else None
        if c is None or c.verdict.state != vq.INFRA:
            return vq.decide(s)
        return [vq.Drop(c.sha, "infra"), *(vq.Eject(n, h, "infra", c.verdict.infra) for n, h in zip(c.members, c.heads))]

    q = broken(ejects_on_infra)
    caught = next((s for s in SEEDS if any(isinstance(d, lq.Eject) for d in
                                           drive(random_scenario(random.Random(s), bad=0.0, infra=0.5), q).decisions)),
                  None)
    assert caught is not None, "the infra check passed a queue that ejects on an infra outcome"


def test_liveness_catches_a_queue_that_reruns_a_failed_batch_instead_of_ejecting() -> None:
    def reruns(s):
        c = s.candidates[0] if s.candidates else None
        if c is not None and c.verdict.state == vq.FAIL:
            return [Rerun(c.sha, tuple(f.key for f in c.verdict.failed))]
        return vq.decide(s)

    q = broken(reruns, Rerun=Rerun)
    caught = next((s for s in SEEDS if _live(drive(random_scenario(random.Random(s), bad=0.5), q, limit=2_000),
                                             exact=False)), None)
    assert caught is not None, "the liveness check passed a queue that reruns a failed batch whole, forever"
