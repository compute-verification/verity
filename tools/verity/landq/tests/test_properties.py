"""The queue model's three properties over seeded random scenarios (`harness.random_scenario`), each event order a seed:

* soundness: `main` moves only to a tree built on it whose every listed gating task passed on that tree, by a result the
  world delivered for that tree, and which holds no change that breaks a task;
* liveness: once nothing is left to deliver, every PR still ready and reviewed has landed or been ejected, each ejection
  naming a task that failed on a tree holding it; with no false fails and no interacting pairs, exactly the PRs that break
  nothing land, and each ejected PR breaks the task its ejection names;
* an infra outcome never ejects a PR.

Each property also runs against a model broken on purpose, which it must catch, so a pass isn't a pass for the wrong reason."""
from __future__ import annotations

import random

import pytest

from harness import final_heads, outcomes, random_scenario, run
from verity_landq import model as lq

SEEDS = range(300)


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


@pytest.mark.parametrize("flaky,pairs,infra", [(0, 0, 0.0), (0, 0, 0.3), (3, 0, 0.0), (2, 2, 0.2)])
def test_main_moves_only_to_a_tree_that_passed_there(flaky, pairs, infra) -> None:
    moves = ejects = 0
    for seed in SEEDS:
        r = run(random_scenario(random.Random(seed), flaky=flaky, pairs=pairs, infra=infra))
        assert not r.unsound, (seed, r.unsound[:2])
        moves += len(r.mains) - 1
        ejects += sum(isinstance(d, lq.Eject) for d in r.decisions)
    assert moves > len(SEEDS) and ejects > len(SEEDS) // 4, (moves, ejects)


@pytest.mark.parametrize("infra", [0.0, 0.3])
def test_every_pr_lands_or_is_ejected_naming_the_task_it_breaks(infra) -> None:
    for seed in SEEDS:
        r = run(random_scenario(random.Random(seed), infra=infra))
        assert not _live(r, exact=True), (seed, _live(r, exact=True)[:3])


@pytest.mark.parametrize("flaky,pairs", [(3, 0), (0, 2), (2, 2)])
def test_with_false_fails_and_pairs_every_pr_still_lands_or_is_ejected(flaky, pairs) -> None:
    for seed in SEEDS:
        r = run(random_scenario(random.Random(seed), flaky=flaky, pairs=pairs, infra=0.2))
        assert not _live(r, exact=False), (seed, _live(r, exact=False)[:3])


def test_an_infra_outcome_never_ejects() -> None:
    retries = pages = 0
    for seed in SEEDS:
        r = run(random_scenario(random.Random(seed), bad=0.0, infra=0.5))
        assert not [d for d in r.decisions if isinstance(d, lq.Eject)], seed
        landed, _ = outcomes(r)
        assert all(landed.get(pr) == head for pr, head in final_heads(r).items()), seed
        retries += sum(isinstance(d, lq.Retry) for d in r.decisions)
        pages += sum(isinstance(d, lq.Page) for d in r.decisions)
    assert retries > len(SEEDS) and pages > 0, (retries, pages)


def test_infra_noise_changes_no_outcome() -> None:
    """The same PRs with and without infra outcomes (so in another order) land and eject the same PRs at the same heads."""
    for seed in SEEDS:
        quiet = run(random_scenario(random.Random(seed), churn=False))
        noisy = run(random_scenario(random.Random(seed), churn=False, infra=0.4))
        assert quiet.scenario.prs == noisy.scenario.prs
        (ql, qe), (nl, ne) = outcomes(quiet), outcomes(noisy)
        assert (ql, {p: e.head for p, e in qe.items()}) == (nl, {p: e.head for p, e in ne.items()}), seed


def test_soundness_catches_a_model_reading_another_trees_results(monkeypatch) -> None:
    """A verdict that takes a pass of the task on any tree in flight moves `main` to a breaking tree in some scenario."""
    def any_tree(results, c):
        if c.tree is None or c.tasks is None:
            return lq.PENDING
        mine = [results.get((c.tree, t)) for t in c.tasks]
        if any(o is not None and o.status == lq.FAIL for o in mine):
            return lq.FAIL
        passed = {t for (_, t), o in results.items() if o.status == lq.PASS}
        return lq.PASS if c.tasks <= passed else lq.PENDING

    monkeypatch.setattr(lq, "_verdict", any_tree)
    caught = next((s for s in SEEDS if run(random_scenario(random.Random(s))).unsound), None)
    assert caught is not None, "the soundness check passed a model that reads another tree's results"


def test_the_infra_check_catches_a_model_that_ejects_on_infra(monkeypatch) -> None:
    real = lq._Tx.result

    def as_fail(tx, e):
        real(tx, lq.Result(e.tree, e.task, lq.FAIL if e.status == lq.INFRA else e.status, e.inputs))

    monkeypatch.setattr(lq._Tx, "result", as_fail)
    monkeypatch.setitem(lq._HANDLERS, lq.Result, as_fail)
    caught = next((s for s in SEEDS if any(isinstance(d, lq.Eject) for d in
                                           run(random_scenario(random.Random(s), bad=0.0, infra=0.5)).decisions)), None)
    assert caught is not None, "the infra check passed a model that ejects on an infra outcome"


def test_liveness_catches_a_model_that_retries_a_failed_batch_instead_of_bisecting(monkeypatch) -> None:
    real = lq._Tx.attribute

    def no_bisect(tx, i):
        before = len(tx.groups)
        done = real(tx, i)
        if len(tx.groups) > before:
            del tx.groups[:2]
        return done

    monkeypatch.setattr(lq._Tx, "attribute", no_bisect)
    caught = next((s for s in SEEDS if _live(run(random_scenario(random.Random(s), bad=0.5), limit=2_000), exact=True)),
                  None)
    assert caught is not None, "the liveness check passed a model that retries a failed batch whole, forever"
