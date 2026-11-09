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


def test_an_infra_outage_ejects_lone_members_as_the_tree(vq):
    """FINDING (innocent ejects, fix before the cut-over): during an outage (every check ends in errors graph2 saw on two
    nodes, which graph2 marks `tree` for any exhausted error since it retries elsewhere), `decide` bisects and ejects lone
    members for errors their change didn't cause, at w3 after one retry whatever `tree` says, at core3 on graph2's `tree`.
    FIX: graph2's `tree` needs the same error on two nodes on which other tasks passed meanwhile; `decide` ejects for infra
    only when the base's check passed in the same window."""
    out = {"n": 0}

    def oracle(vq, tree, c, base):
        out["n"] += 1
        err = vq.Failed("test-file:t/test_store.py", "sha512:k", frozenset({"t/test_store.py"}),
                        **({"tree": True} if "tree" in vq.Failed.__dataclass_fields__ else {}))
        return vq.Verdict(vq.INFRA, infra=(err,), ended=True)

    w = World(vq, [pr(vq, n) for n in range(1, 7)], oracle).run(120)
    assert w.landed == [] and len(w.ejected) >= 1
    assert all("t/test_store.py" in str(e.failed) for e in w.ejected)


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
