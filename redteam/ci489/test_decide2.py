"""`research.vqueue.decide` from core2 on (W3e: top's red-main rule, `Rerun`, timeouts, `tests`/`base_tests`), on crafted
snapshots and in a world that applies its actions as the loop does, setting what the loop sets: `Verdict.ended`, `now`,
`Candidate.enqueued`, and each failed task's `base_key`, `tests` and `base_tests` from graph2's entries for candidate and base.

    VQRED_VQUEUE=<checkout of cursor/ci-vqueue-w3>/tools/research/src/research/vqueue.py pytest redteam/ci489/test_decide2.py

A test named FIXED is a finding of `test_decide.py` that this code no longer has, kept as its regression check.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import pytest

MAIN = "m0"
HAWKEYE = "benchmarks/numerical/tests/reference/test_hawkeye_close.py"


@pytest.fixture(autouse=True)
def _contract(vq):
    if not hasattr(vq, "Rerun"):
        pytest.skip("this decide predates core2 (test_decide.py covers it)")


def pr(vq, n: int, *files: str, lean: bool = False):
    return vq.Ready(n, f"h{n}", frozenset(files or (f"pkg{n}/mod.py",)), lean, n)


@dataclass
class World:
    """`main`, the candidates, the outcomes and the PRs' fates, moved only by `decide`'s actions; a minute a tick."""
    vq: object
    ready: list
    oracle: object                     # (vq, PRs in the candidate's tree, the candidate, its base's tree) -> Verdict
    main: str = MAIN
    cands: dict = field(default_factory=dict)
    trees: dict = field(default_factory=lambda: {MAIN: frozenset()})
    history: list = field(default_factory=list)
    landed: list = field(default_factory=list)
    ejected: list = field(default_factory=list)
    acts: list = field(default_factory=list)
    built: int = 0
    now: float = 1_000_000.0
    kw: dict = field(default_factory=dict)

    def snapshot(self):
        gone = {*self.landed, *(e.number for e in self.ejected)}
        return self.vq.Snapshot(self.main, tuple(r for r in self.ready if r.number not in gone), tuple(self.cands.values()),
                                tuple(self.history), now=self.now, **self.kw)

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
                c = self.cands[a.sha]
                self.cands[a.sha] = replace(c, tries=c.tries + 1, verdict=vq.Verdict())
            elif isinstance(a, vq.Rerun):
                c = self.cands[a.sha]
                self.cands[a.sha] = replace(c, reruns=c.reruns + 1, verdict=vq.Verdict())
            elif isinstance(a, vq.Build):
                self.built += 1
                sha = f"c{self.built}"
                self.trees[sha] = self.trees[a.base] | set(a.members)
                self.cands[sha] = vq.Candidate(sha, a.base, a.members, tuple(f"h{m}" for m in a.members), enqueued=self.now)
        for sha, c in list(self.cands.items()):
            if c.verdict.state == vq.PENDING:
                self.cands[sha] = replace(c, verdict=self.oracle(vq, self.trees[sha], c, self.trees.get(c.base, frozenset())))
        self.now += 60
        return acts

    def run(self, ticks: int = 300) -> "World":
        for _ in range(ticks):
            self.tick()
        return self

    def said(self, kind) -> list:
        return [a for acts in self.acts for a in acts if isinstance(a, kind)]


def failing(vq, task: str, inputs: set[str], key: str, base_key: str | None, tests=(), base_tests=None):
    return vq.Failed(task, key, frozenset(inputs), base_key=base_key, tests=frozenset(tests),
                     base_tests=None if base_tests is None else frozenset(base_tests))


def red_main(vq, tests=("test_hawkeye_close.py::test_saturation[4090-18-0]",)):
    """`main` fails hawkeye, which no PR touches: same key on candidate and base, and graph2 names the failing test."""
    def oracle(vq, tree, c, base):
        return vq.Verdict(vq.FAIL, (failing(vq, f"test-file:{HAWKEYE}", {HAWKEYE}, "sha512:red", "sha512:red", tests,
                                            tests),), ended=True)
    return oracle


def test_fixed_a_red_base_advances_every_pr_and_rings_its_red_once(vq):
    """FIXED (top's 08:1xZ rule; at 2fe8e8f19c nothing landed until a fix did): with `main` red on a task no PR touches, all
    eight PRs land and nobody is ejected; Red is said once for the task and key."""
    w = World(vq, [pr(vq, n) for n in range(1, 9)], red_main(vq)).run()
    assert sorted(w.landed) == list(range(1, 9)) and w.ejected == []
    assert len({(f.task, f.key) for r in w.said(vq.Red) for f in r.failed}) == 1


def test_fixed_a_fix_for_one_of_two_reds_lands(vq):
    """FIXED (at 2fe8e8f19c a fix for one of two reds never landed): #9 fixes red A; its candidate still fails red B at the
    base's key, which is inherited, so it advances, and so does everyone else."""
    a_in = {"a/tests/test_a.py", "a/mod.py"}

    def oracle(vq, tree, c, base):
        b = failing(vq, f"test-file:{HAWKEYE}", {HAWKEYE}, "sha512:b", "sha512:b", ("t::b",), ("t::b",))
        a = () if 9 in tree else (failing(vq, "test-file:a/tests/test_a.py", a_in, "sha512:a",
                                          "sha512:a" if 9 not in base else None, ("t::a",), ("t::a",)),)
        return vq.Verdict(vq.FAIL, (*a, b), ended=True)

    w = World(vq, [*(pr(vq, n) for n in range(1, 9)), pr(vq, 9, "a/mod.py")], oracle).run()
    assert sorted(w.landed) == list(range(1, 10)) and w.ejected == []


def test_a_faulted_red_on_main_ejects_a_pr_that_touches_its_inputs(vq):
    """FINDING (innocent ejects, still open; fix before the cut-over): a faulted task (an undeclared read; 45 of 49 failed
    entries on ready:5e3d5a7e) has no test ids, so on a red `main` it is inherited only at the base's key. A PR touching any of
    its inputs changes the key and is ejected for a red `main` already had: for a task reading most of the tree (the root
    suite), that is nearly every PR. FIX: the loop gives a faulted entry its undeclared paths as its `tests`
    (`undeclared:<path>`), and the base's as `base_tests`, so a PR that adds no undeclared read inherits it."""
    def oracle(vq, tree, c, base):
        key = "sha512:changed" if 2 in tree else "sha512:red"
        return vq.Verdict(vq.FAIL, (failing(vq, "test-file:tests/test_repository.py", {"tests/test_repository.py", "src/x.py"},
                                            key, "sha512:red", (), None),), ended=True)

    w = World(vq, [pr(vq, 1), pr(vq, 2, "src/x.py"), pr(vq, 3)], oracle).run(60)
    assert [e.number for e in w.ejected] == [2]
    assert sorted(w.landed) == [1, 3]


def test_fixed_a_pr_touching_a_red_tests_inputs_inherits_by_its_failing_tests(vq):
    """FIXED (at 2fe8e8f19c it was ejected): with graph2 naming the failing tests, a PR that edits a red test's inputs but
    fails only what `main` fails inherits the failure and lands."""
    def oracle(vq, tree, c, base):
        key = "sha512:changed" if 2 in tree else "sha512:red"
        return vq.Verdict(vq.FAIL, (failing(vq, "test-file:t/test_x.py", {"t/test_x.py", "src/x.py"}, key, "sha512:red",
                                            ("t/test_x.py::a",), ("t/test_x.py::a",)),), ended=True)

    w = World(vq, [pr(vq, 1), pr(vq, 2, "src/x.py"), pr(vq, 3)], oracle).run(60)
    assert w.ejected == [] and sorted(w.landed) == [1, 2, 3]


@pytest.mark.parametrize("together", [False, True])
def test_a_pair_that_fails_only_together_ejects_one_of_them_and_lands_the_rest(vq, together):
    """Tonight's #7584 (lock frozen reasons starting `completeness:`) and #8146 (a Lean-check rule refusing them): each passes
    alone; together the Security Lean check fails. Both meet the failing task's inputs. Apart, the later one, built on the
    other, is the one member meeting them and is ejected. Together, both are suspects: bisected, the first lands alone and
    the second, built on it, is ejected. Nobody else is ejected, and nothing stalls. HOLDS, provided the failing check is a
    gating task of the candidate (the comment says which task runs the rule)."""
    lock_in = {"Security/lean-audit.json", "tools/verity/lean/check.py"}

    def oracle(vq, tree, c, base):
        if {7584, 8146} <= tree:
            return vq.Verdict(vq.FAIL, (failing(vq, "lean-lock:Security/lean-audit.json", lock_in, f"sha512:{len(tree)}",
                                                "sha512:base", (), None),), ended=True)
        return vq.Verdict(vq.PASS, ended=True)

    first = 0 if together else 10
    pair = [vq.Ready(7584, "h7584", frozenset({"Security/lean-audit.json"}), True, first),
            vq.Ready(8146, "h8146", frozenset({"tools/verity/lean/check.py"}), True, first + 1)]
    w = World(vq, [*(pr(vq, n) for n in range(1, 6)), *pair], oracle, kw={"lean": 3}).run(80)
    assert [e.number for e in w.ejected] in ([8146], [7584])
    assert sorted(w.landed) == sorted([1, 2, 3, 4, 5, *({7584, 8146} - {w.ejected[0].number})])


def _outage(vq, tree, c, base):
    err = vq.Failed("test-file:t/test_store.py", "sha512:k", frozenset({"t/test_store.py"}),
                    **({"tree": True} if "tree" in vq.Failed.__dataclass_fields__ else {}))
    return vq.Verdict(vq.INFRA, infra=(err,), ended=True)


def test_fixed_an_infra_outage_ejects_nobody(vq):
    """FIXED at a2a1368d98 (W3e's `Snapshot.tree_ejects`, off by default: the loop's `--tree-ejects`); a finding at
    3b9df5d6dc and core3 df980a2f28, which bisected and ejected lone members for an outage graph2 marks `tree`. Off, infra
    with `tree` entries is retried at every reading and rung, and nothing is bisected or ejected."""
    if "tree_ejects" not in vq.Snapshot.__dataclass_fields__:
        pytest.skip("no tree_ejects switch: this decide ejects for an outage graph2 marks tree")
    w = World(vq, [pr(vq, n) for n in range(1, 7)], _outage).run(120)
    assert w.landed == [] and w.ejected == [] and w.said(vq.Ring)


def test_with_tree_ejects_on_an_outage_marked_tree_ejects_lone_members(vq):
    """Why `--tree-ejects` stays off until graph2's `tree` means the same error repeated (and `trees` bounds when a pass
    counts: test_graph2's outage test): on, every lone member is ejected for errors its change didn't cause."""
    if "tree_ejects" not in vq.Snapshot.__dataclass_fields__:
        pytest.skip("no tree_ejects switch")
    w = World(vq, [pr(vq, n) for n in range(1, 7)], _outage, kw={"tree_ejects": True}).run(120)
    assert w.landed == [] and len(w.ejected) >= 1


def test_with_tree_ejects_off_a_pr_whose_change_errs_stops_the_queue(vq):
    """FINDING (liveness, before the cut-over or accepted with a run-book): with `tree_ejects` off, infra never bisects or
    ejects, so a PR whose own change makes a task err on every node (a hang, an out-of-memory kill: errors graph2 marks
    `tree`) is never taken out. The lowest candidate holding it is retried at every reading and @infra is rung, not its
    owner; every PR batched or stacked above it waits until a person removes it. FIX: with `tree_ejects` off, still bisect a
    candidate whose infra persists over `rings` readings, ejecting nobody: the members without the error land, and the lone
    erring PR waits, rung to its owner."""
    if "tree_ejects" not in vq.Snapshot.__dataclass_fields__:
        pytest.skip("no tree_ejects switch")

    def oracle(vq, tree, c, base):
        if 3 not in tree:
            return vq.Verdict(vq.PASS, ended=True)
        err = vq.Failed("test-file:pkg3/tests/test_big.py", "sha512:big", frozenset({"pkg3/mod.py"}), tree=True)
        return vq.Verdict(vq.INFRA, infra=(err,), ended=True)

    w = World(vq, [pr(vq, n, f"pkg{n}/mod.py") for n in range(1, 21)], oracle,
              kw={"batch": 16, "depth": 3, "lean": 3}).run(300)
    assert w.ejected == [] and w.landed == []
    assert w.built == 2 and len(w.said(vq.Retry)) > 100


def test_a_red_main_on_a_whole_tree_step_of_the_run_beside_ejects_every_pr(vq):
    """FINDING (innocent ejects, before the cut-over or a run-book check): the run beside a check keys `preflight-lints` on
    the whole tree (`vqueue_slim.WHOLE`), so with `main` red on it the candidate's key never equals the base's and the
    failure is never inherited. Bisected, each lone member fails alone again on its rerun, and the flake rule ejects it,
    though the failure names no file it touched: all 8 PRs go. `main` turns red on it only through what the run beside
    didn't judge (a red `main` at the cut-over, a landing by the old path, a lint that reads beyond the tree). FIX: before
    ejecting a lone member whose failures meet none of its files, judge the base (its run beside, or graph2's check of
    `main`): a failure the base has too is a Red; and at the cut-over, check that `main` passes the run beside."""
    def oracle(vq, tree, c, base):
        f = vq.Failed("step preflight-lints", f"sha512:{sorted(tree)}", frozenset({"docs/README.md"}),
                      base_key=f"sha512:{sorted(base)}")
        return vq.Verdict(vq.FAIL, (f,), ended=True)

    w = World(vq, [pr(vq, n) for n in range(1, 9)], oracle).run(200)
    assert sorted(e.number for e in w.ejected) == list(range(1, 9)) and w.landed == []


def test_fixed_a_pending_candidate_is_retried_then_rung(vq):
    """FIXED (at 2fe8e8f19c nothing timed out): a check that never ends is retried after `pending_s` and rung after twice that."""
    w = World(vq, [pr(vq, 1)], lambda vq, tree, c, base: vq.Verdict(vq.PENDING)).run(500)
    assert len(w.said(vq.Retry)) == 1 and len(w.said(vq.Ring)) >= 1 and w.landed == w.ejected == []


def test_fixed_a_lone_failure_is_rerun_before_it_is_ejected(vq):
    """FIXED (at 2fe8e8f19c the retest rebuilt the same tree, answered from graph2's cache): a lone failure is rerun by key;
    a test that passes on the rerun lands its PR."""
    runs = {"n": 0}

    def oracle(vq, tree, c, base):
        runs["n"] += 1
        if runs["n"] == 1:
            return vq.Verdict(vq.FAIL, (failing(vq, "test-file:t/test_flaky.py", {"t/test_flaky.py"}, "sha512:f", None,
                                                ("t/test_flaky.py::x",), None),), ended=True)
        return vq.Verdict(vq.PASS, ended=True)

    w = World(vq, [pr(vq, 7)], oracle).run(20)
    assert [a.keys for a in w.said(vq.Rerun)] == [("sha512:f",)]
    assert w.landed == [7] and w.ejected == []


def test_fixed_a_fail_naming_no_task_rings_and_ejects_nobody(vq):
    """FIXED (at 2fe8e8f19c it bisected and ejected everyone)."""
    w = World(vq, [pr(vq, n) for n in range(1, 5)], lambda vq, tree, c, base: vq.Verdict(vq.FAIL, ended=True)).run(40)
    assert w.ejected == [] and w.said(vq.Ring)
