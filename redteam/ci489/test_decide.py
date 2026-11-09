"""`research.vqueue.decide` (ci's W3a) on crafted snapshots and in a small world that applies its actions as the loop does.

Run against ci's branch: `VQRED_VQUEUE=<checkout of cursor/ci-vqueue-core>/tools/research/src/research/vqueue.py pytest
redteam/ci489/test_decide.py`. The world gives each candidate a verdict from a scripted oracle over the PRs its tree holds.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import pytest

MAIN = "m0"


def pr(vq, n: int, *files: str):
    return vq.Ready(n, f"h{n}", frozenset(files or (f"pkg{n}/mod.py",)), False, n)


@dataclass
class World:
    """`main`, the candidates' refs, the recorded outcomes and the PRs' fates, moved only by `decide`'s actions."""
    vq: object
    ready: list
    oracle: object                     # (vq, PRs in the candidate's tree, the candidate) -> Verdict
    main: str = MAIN
    cands: dict = field(default_factory=dict)
    trees: dict = field(default_factory=lambda: {MAIN: frozenset()})
    history: list = field(default_factory=list)
    landed: list = field(default_factory=list)
    ejected: list = field(default_factory=list)
    acts: list = field(default_factory=list)
    built: int = 0

    def snapshot(self):
        gone = {*self.landed, *(e.number for e in self.ejected)}
        return self.vq.Snapshot(self.main, tuple(r for r in self.ready if r.number not in gone), tuple(self.cands.values()),
                                tuple(self.history))

    def tick(self) -> list:
        vq = self.vq
        acts = vq.decide(self.snapshot())
        self.acts.append(acts)
        for a in acts:
            if isinstance(a, vq.Advance):
                self.main = a.sha
                self.landed += list(self.cands[a.sha].members)
            elif isinstance(a, vq.Drop):
                self.cands.pop(a.sha, None)
                if a.outcome is not None:
                    self.history.append(a.outcome)
            elif isinstance(a, vq.Eject):
                self.ejected.append(a)
            elif isinstance(a, vq.Retry):
                self.cands[a.sha] = replace(self.cands[a.sha], tries=self.cands[a.sha].tries + 1)
            elif isinstance(a, vq.Build):
                self.built += 1
                sha = f"c{self.built}"
                self.trees[sha] = self.trees[a.base] | set(a.members)
                self.cands[sha] = vq.Candidate(sha, a.base, a.members, tuple(f"h{m}" for m in a.members))
        for sha, c in list(self.cands.items()):
            if c.verdict.state == vq.PENDING:
                self.cands[sha] = replace(c, verdict=self.oracle(vq, self.trees[sha], c))
        return acts

    def run(self, ticks: int = 300) -> "World":
        for _ in range(ticks):
            self.tick()
        return self


def red_on_base(vq, tree, c):
    """`main` already fails a test no PR touches (as `test_hawkeye_close.py` failed at `checked`'s tip tonight)."""
    return vq.Verdict(vq.FAIL, (vq.Failed("test-file:benchmarks/numerical/tests/reference/test_hawkeye_close.py", "sha512:red",
                                          frozenset({"benchmarks/numerical/tests/reference/test_hawkeye_close.py"})),))


def test_a_red_base_ejects_every_pr_as_a_possible_flake(vq):
    """FINDING (innocent ejects, blocks the cut-over; top's 07:09Z point 1): with a task red on `main` that no PR touches, every
    candidate fails it; no member meets its inputs, so each batch is bisected down to single PRs, each fails alone twice, and
    each is ejected "if a test flaked, delete it". After the world runs, every PR is ejected and none landed. FIX: ci's
    `base_key` extension (due 08:30Z): a failure whose key equals the base's same task's key blames nobody; the loop must give
    `decide` the base's verdict (`ready:<main sha>`), and wait (not eject) while it is pending."""
    w = World(vq, [pr(vq, n) for n in range(1, 9)], red_on_base).run()
    assert w.landed == []
    assert sorted(e.number for e in w.ejected) == list(range(1, 9))
    assert all("if a test flaked, delete it" in e.why for e in w.ejected)


def test_a_fail_with_no_failed_task_bisects_and_ejects_everyone(vq):
    """FINDING (contract, fix before the cut-over): `decide` takes `Verdict(FAIL, failed=())` as a failure nobody's inputs meet:
    it bisects and ejects every PR as a possible flake. graph2's `state` is `fail` when only off-gate tasks failed
    (circuit-check's past-range red today), and ci drops off-gate failures in W3; a loop that keeps graph2's state and filters
    `failed` hands `decide` exactly this. FIX: the loop derives the state from the gating tasks alone (pass only when every
    gating task passed, nothing pending or blocked); `decide` treats FAIL with no failed task as a contract breach (Ring, no
    eject)."""
    w = World(vq, [pr(vq, n) for n in range(1, 5)], lambda vq, tree, c: vq.Verdict(vq.FAIL)).run()
    assert w.landed == [] and sorted(e.number for e in w.ejected) == [1, 2, 3, 4]


def test_an_infra_verdict_one_pr_causes_stops_the_queue_for_good(vq):
    """FINDING (liveness, blocks the cut-over): a PR whose tree makes a task end in error every time (an undeclared read, a
    hang to its timeout, an untraced program: graph2's `infra`) holds the lowest candidate at INFRA. `decide` retries it and
    rings @infra every tick from the third try, forever: it never drops, bisects or ejects, so no PR lands, those batched with
    the culprit included, and the candidate above is never read. Today the same PR is a red with a culprit (suites.py's
    guard). FIX: an INFRA verdict whose infra keys repeat after one Retry is bisected like a FAIL (suspects by the infra
    entries' inputs, else all members), and @infra is rung once, not every tick."""
    culprit = 3

    def oracle(vq, tree, c):
        if culprit in tree:
            return vq.Verdict(vq.INFRA, infra=(vq.Failed("test-file:pkg3/tests/test_reads.py", "sha512:k3",
                                                          frozenset({"pkg3/tests/test_reads.py", "pkg3/mod.py"})),))
        return vq.Verdict(vq.PASS)

    w = World(vq, [pr(vq, n) for n in range(1, 13)], oracle).run(200)
    assert w.landed == [] and w.ejected == []
    late = w.acts[-1]
    assert {type(a).__name__ for a in late} == {"Retry", "Ring"}
    assert sum(isinstance(a, vq.Ring) for acts in w.acts for a in acts) > 190


def test_a_candidate_pending_forever_is_never_rung(vq):
    """FINDING (liveness, fix before the cut-over): nothing in `decide` or the snapshot knows how long a candidate has been
    pending, so a check graph2 never ends (a listing wedged in `listing`, a verdict label lost, a stuck serve) holds the queue
    with no action at all: no Retry, no Ring. FIX: the snapshot carries each candidate's enqueue time and `now`; `decide`
    retries a candidate pending past a bound (W5's wall time ×2) and rings @infra once."""
    w = World(vq, [pr(vq, n) for n in range(1, 13)], lambda vq, tree, c: vq.Verdict(vq.PENDING))
    w.run(3)
    assert len(w.cands) == 2
    for _ in range(500):
        assert w.tick() == []


def test_a_single_pr_retest_is_the_same_tree_so_graph2_answers_it_from_the_cache(vq):
    """FINDING (innocent ejects, fix before the cut-over): after a PR fails alone, `decide` builds it alone again on the same
    base: the same tree, so the same keys, and graph2 caches fails (`test_graph2.py::
    test_a_cached_fail_answers_a_retest_of_the_same_key_without_running`). The "retest to rule out a flake" never reruns the
    test, and a flake's first run ejects the PR. FIX: rerun the failing keys uncached (a graph2 call to add) or drop the
    retest and call the eject what it is."""
    one = pr(vq, 7)
    failed = vq.Failed("test-file:t/test_x.py", "sha512:x", frozenset({"t/test_x.py"}))
    c1 = vq.Candidate("c1", MAIN, (7,), ("h7",), vq.Verdict(vq.FAIL, (failed,)))
    acts = vq.decide(vq.Snapshot(MAIN, (one,), (c1,)))
    builds = [a for a in acts if isinstance(a, vq.Build)]
    assert builds == [vq.Build((7,), MAIN, 2)]
    assert not any(isinstance(a, vq.Eject) for a in acts)


def test_failed_alone_twice_counts_a_different_tasks_failure_before_the_re_mark(vq):
    """FINDING (innocent ejects, after): "failed alone twice" is any earlier outcome of the PR alone at the same head that
    failed, whatever failed then: an owner who re-marks a head ejected as a possible flake (the window is 48 h) is ejected at the
    next single failure, of another task too, with no second run. FIX: count only an earlier failure of the same task, and
    only outcomes after the PR's newest mark (the snapshot's `Ready` gains its mark time)."""
    earlier = vq.Outcome((5,), ("h5",), vq.FAIL, (5,))
    other = vq.Failed("test-file:t/test_other.py", "sha512:o", frozenset({"t/test_other.py"}))
    c1 = vq.Candidate("c1", MAIN, (5,), ("h5",), vq.Verdict(vq.FAIL, (other,)))
    acts = vq.decide(vq.Snapshot(MAIN, (pr(vq, 5),), (c1,), (earlier,)))
    assert [a.number for a in acts if isinstance(a, vq.Eject)] == [5]
    assert "failed alone twice" in next(a.why for a in acts if isinstance(a, vq.Eject))


def test_a_pr_that_touches_a_red_tasks_inputs_is_ejected_for_it(vq):
    """FINDING (innocent ejects, fix before the cut-over): a PR that edits a file a red-on-`main` test imports (a refactor, a
    docstring) changes that task's key, so even with `base_key` the failure isn't inherited, and the PR is ejected at once,
    though `main` fails the same test the same way. top's rule is that no task fails on the candidate that passed on its
    base. FIX: a failure is inherited when the base failed the same task and the candidate's failing tests (`detail.
    failed_tests`) are a subset of the base's; a task without that detail (Lean, circuits) blames the PR as now."""
    red = vq.Failed("test-file:t/test_x.py", "sha512:changed-by-the-pr", frozenset({"t/test_x.py", "src/x.py"}))
    touching = pr(vq, 2, "src/x.py")
    c1 = vq.Candidate("c1", MAIN, (1, 2, 3), ("h1", "h2", "h3"), vq.Verdict(vq.FAIL, (red,)))
    acts = vq.decide(vq.Snapshot(MAIN, (pr(vq, 1), touching, pr(vq, 3)), (c1,)))
    assert [a.number for a in acts if isinstance(a, vq.Eject)] == [2]


def test_an_env_break_is_ejected_as_a_flake_after_bisecting(vq):
    """FINDING (innocent ejects, after): graph2's failed entries carry each task's own `inputs` and `reads`, never its env's
    spec (uv.lock, pyproject.toml, lean-toolchain, lake-manifest.json), which is in every task's key. A PR that bumps uv.lock
    and breaks tests meets no failing task's inputs, so its batch is bisected, its innocent batchmates wait out the halvings,
    and it is ejected as a possible flake. FIX: `Failed.inputs` adds the env's spec paths (the listed graph's `envs`), and
    the transitive closure the contract promises (the loop's to compute from the graph at `art`)."""
    def oracle(vq, tree, c):
        if 4 in tree:
            return vq.Verdict(vq.FAIL, (vq.Failed("test-file:t/test_np.py", "sha512:np", frozenset({"t/test_np.py"})),))
        return vq.Verdict(vq.PASS)

    ready = [pr(vq, n) for n in range(1, 7)]
    ready[3] = pr(vq, 4, "uv.lock")
    w = World(vq, ready, oracle).run()
    assert [e.number for e in w.ejected] == [4]
    assert "if a test flaked, delete it" in w.ejected[0].why
    assert w.built > 4
