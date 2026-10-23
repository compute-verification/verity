"""drive's per-PR landing in shadow (`--per-pr-shadow N`, parallel landing's step 3): each ready PR, oldest mark first and N at a
time, gets a check of `main` merged with its head; a new head restarts only that check; a pass on an unmoved `main` would land,
and on a moved one the new landing commit is planned by its keys first.  It pushes no `main` or `next`, ejects nothing, posts
nothing, and leaves the trains' records alone.  On `test_queue`'s throwaway origin, with drive's launches, run states and plans
injected (`test_queue_drive.Harness`)."""
from __future__ import annotations

import contextlib
import json
import os
import subprocess
from datetime import datetime, timezone

import pytest

from research import merge, queue_stats
from research.drive import launch_vars, plan_launch, plan_state, undeferred
from research.queue import main as queue_main
from test_queue import World, _git
from test_queue_drive import Harness, _check, _tip

ENV = ("CHECK_SLOT_RANK", "TRAIN_ORDINARY", "TRAIN_QUIET")
MARKS = {1: "2026-10-20T10:00:00Z", 2: "2026-10-20T11:00:00Z", 3: "2026-10-20T09:00:00Z"}


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


class Shadow(Harness):
    """The harness with the shadow on (``n`` checks at once), each launch's train.sh environment recorded, its runs' check slots
    (`took[RID]`, the `slot.json` a fetch of RID finds) and its plans injected: a plan of COMMIT stays running until
    `plans[COMMIT]` holds its (state, docs, log tail), and `judged[COMMIT]` is what judging a plan of it says."""

    def __init__(self, world: World, monkeypatch, n: int = 2, trains: int = 0, **kw):
        kw.setdefault("marked_at", lambda pr, head: MARKS.get(pr))
        super().__init__(world, monkeypatch, **kw)
        self.d.per_pr_shadow, self.d.max_running, self.d.train_size = n, trains, 1
        self.took: dict[str, str] = {}
        self.d.fetch = self.fetch
        self.envs: dict[str, tuple] = {}
        self.planned: list[dict] = []
        self.plans: dict[str, tuple] = {}
        self.judged: dict[str, tuple] = {}
        self.d.plan_launch = self.plan_launch
        self.d.plan_state = lambda p: self.plans.get(p["commit"], ("running", [], ""))
        self.d.judge_plan = lambda doc, commit, base: self.judged[commit]

    def launch(self, repo, commit, machine, agreement):
        ids = super().launch(repo, commit, machine, agreement)
        self.envs[commit] = tuple(launch_vars().get(k) for k in ENV)
        return ids

    def fetch(self, rid, names):
        if rid not in self.took:
            return None
        d = self.world.tmp / "fetched" / rid
        d.mkdir(parents=True, exist_ok=True)
        (d / "slot.json").write_text(json.dumps({"state": "held", "took": self.took[rid]}))
        return d, {}

    def plan_launch(self, repo, commit, base, runs, machines, out):
        self.planned.append({"commit": commit, "base": base, "runs": runs, "machines": machines})
        return {"dir": str(out), "pid": 0, "machines": machines}

    def shadow(self) -> dict:
        return self.q.load_state()["drive"]["per_pr"]

    def rec(self, pr: int) -> dict:
        return self.shadow()["prs"][str(pr)]

    def check_of(self, pr: int) -> dict:
        return self.rec(pr)["checks"][-1]


def _main(world: World) -> str:
    return _git(world.origin, "rev-parse", "refs/heads/main")


def _branches(world: World) -> set[str]:
    return set(_git(world.origin, "for-each-ref", "--format=%(refname:short)", "refs/heads/").split())


def _parents(repo, commit: str) -> list[str]:
    return _git(repo, "rev-list", "--parents", "-n", "1", commit).split()[1:]


def test_ready_prs_get_one_check_each_of_their_merge_onto_main_oldest_mark_first_up_to_the_cap(world, monkeypatch):
    """The shadow's N slots go to the oldest ready marks, each checking exactly `main` merged with that head (pushed as
    per-pr/N, which train.sh checks out of origin), each an ordinary quiet job (no `--priority`, no rank, no post: it never
    takes a lane's or a train's slot); the next one starts when one of the N frees."""
    h = Shadow(world, monkeypatch, n=2)
    heads = {n: world.change({f"f{n}.txt": f"{n}\n"}) for n in (1, 2, 3)}
    for n, head in heads.items():
        world.gh.put(n, head)
    main = _main(world)
    assert "per-PR shadow: 2 checking, 0 planning, 0 would land" in h.tick()
    sh = h.shadow()
    assert set(sh["prs"]) == {"3", "1"}                                  # marks 09:00 and 10:00; #2's 11:00 waits, unrecorded
    assert [c for c, *_ in h.launched] == [h.check_of(3)["commit"], h.check_of(1)["commit"]]
    for n in (3, 1):
        commit = h.check_of(n)["commit"]
        assert _parents(world.origin, commit) == [main, heads[n]] and h.check_of(n)["main"] == main
        assert _git(world.origin, "rev-parse", f"refs/heads/per-pr/{n}") == commit
        assert h.envs[commit] == (None, "1", "1")
    assert "CHECK_SLOT_RANK" not in os.environ and h.q.load_state()["drive"]["launched"] == {}
    h.tick()
    assert len(h.launched) == 2                                          # both still run: the cap holds
    _check(world, monkeypatch, h.check_of(3)["commit"], ("false",))      # #3's check fails, and its slot frees
    h.tick()
    assert h.rec(3)["phase"] == "failed" and h.check_of(2)["commit"] == h.launched[-1][0] and len(h.launched) == 3
    assert h.posts == [] and h.q.load_state().get("ejected", {}) == {}  # a failure is logged, never ejected or told
    assert any("#3's check of" in x and "would be ejected" in x and "[not ejected]" in x for x in world.logs)


def test_a_new_head_restarts_only_its_own_prs_check(world, monkeypatch):
    h = Shadow(world, monkeypatch, n=2)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"b.txt": "2\n"}))
    h.tick()
    one, two = h.check_of(1), h.check_of(2)
    new = world.change({"a.txt": "1 fixed\n"})
    world.gh.put(1, new)
    h.tick()
    assert h.cancelled == one["ids"]                                      # #1's run alone
    assert h.check_of(2) == two and h.rec(2)["phase"] == "checking"       # #2's check runs on, untouched
    assert h.rec(1)["head"] == new and _parents(world.origin, h.check_of(1)["commit"]) == [_main(world), new]
    assert h.launched[-1][0] == h.check_of(1)["commit"] and len(h.launched) == 3
    (old,) = h.shadow()["done"]
    assert old["head"] != new and old["ended"]["why"] == "new-head" and old["checks"] == [one]
    world.gh.put(2, world.change({"b.txt": "2 again\n"}), ready=False)    # a head not yet marked: its check stops, none starts
    h.tick()
    assert h.cancelled == [*one["ids"], *two["ids"]] and "2" not in h.shadow()["prs"] and len(h.launched) == 3


def test_a_pass_on_the_main_it_was_built_on_would_land_without_pushing_main_or_ejecting(world, monkeypatch):
    h = Shadow(world, monkeypatch, n=1)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.tick()
    main, commit = _main(world), h.check_of(1)["commit"]
    assert _git(world.origin, "log", "-1", "--format=%cn", commit) == "research queue per-PR shadow" != \
        _git(world.origin, "log", "-1", "--format=%cn", _tip(world))     # never `next`'s commit, whose attempts are the trains'
    h.clock += 1800
    h.took[h.check_of(1)["ids"][0]] = "2026-10-20T12:05:00Z"             # when a slot took it, behind the priority checks
    _check(world, monkeypatch, commit)
    assert "1 would land" in h.tick()
    took = datetime(2026, 10, 20, 12, 5, tzinfo=timezone.utc).timestamp()
    assert h.rec(1)["took"] == h.check_of(1)["took"] == took
    wl = h.rec(1)["would_land"]
    assert h.rec(1)["phase"] == "would-land" and wl["commit"] == commit and wl["main"] == main and wl["via"].startswith("check r")
    assert wl["at"] - h.rec(1)["sampled"] == 1800 and h.planned == []
    assert _main(world) == main and h.posts == [] and h.q.load_state().get("ejected", {}) == {}
    assert any("would land #1" in x and "[not pushed]" in x for x in world.logs)
    h.tick()
    assert h.rec(1)["phase"] == "would-land" and len(h.launched) == 1    # once it would land, nothing more runs for it


def test_on_a_moved_main_the_new_landing_commit_is_planned_by_its_keys_and_would_land_once_its_parts_pass(world, monkeypatch):
    h = Shadow(world, monkeypatch, n=1)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.tick()
    first = h.check_of(1)["commit"]
    other = world.change({"c.txt": "outside the queue\n"})
    _git(world.dev, "push", "-q", "origin", f"{other}:refs/heads/main")
    _check(world, monkeypatch, first)
    h.tick()
    (plan,) = h.planned
    assert h.rec(1)["phase"] == "planning" and _parents(h.d.repo, plan["commit"])[0] == other
    assert plan["base"] == other and plan["runs"] == [h.check_of(1)["ended"]["attempt"]] and plan["machines"] == ["m1"]
    h.tick()
    assert h.rec(1)["phase"] == "planning" and len(h.planned) == 1        # the plan runs: nothing more
    h.plans[plan["commit"]] = ("ended", [{"kind": "check-plan/v1"}], "")
    h.judged[plan["commit"]] = (["r1", "r2"], None, False)
    h.tick()
    wl = h.rec(1)["would_land"]
    assert wl["commit"] == plan["commit"] and wl["main"] == other and wl["via"].startswith("plan cites r1, r2")
    assert _main(world) == other and _branches(world) == {"main", "next", "per-pr/1"} and h.posts == []


def test_a_plan_its_keys_dont_pass_is_checked_again_and_one_main_overtook_is_planned_again(world, monkeypatch):
    h = Shadow(world, monkeypatch, n=1)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.tick()
    _git(world.dev, "push", "-q", "origin", f"{world.change({'c.txt': 'm1\n'})}:refs/heads/main")
    _check(world, monkeypatch, h.check_of(1)["commit"])
    h.tick()
    (plan,) = h.planned
    h.plans[plan["commit"]] = ("ended", [{"kind": "check-plan/v1"}], "")
    h.judged[plan["commit"]] = ([], "3 of 40 parts have no cited pass under their key: lean-audit:pkg must-run", False)
    h.tick()
    assert h.rec(1)["plans"][-1]["result"] == "refused" and h.rec(1)["phase"] == "checking"
    again = h.check_of(1)
    assert len(h.rec(1)["checks"]) == 2 and again["main"] == _main(world) and h.launched[-1][0] == again["commit"]
    moved = world.change({"c.txt": "m2\n"}, base="origin/main")
    _git(world.dev, "push", "-q", "origin", f"{moved}:refs/heads/main")
    _check(world, monkeypatch, again["commit"])
    h.tick()
    second = h.planned[-1]
    h.plans[second["commit"]] = ("ended", [{"kind": "check-plan/v1"}], "")
    h.judged[second["commit"]] = (["r9"], None, True)
    third = world.change({"c.txt": "m3\n"}, base="origin/main")           # main moves while the plan runs
    _git(world.dev, "push", "-q", "origin", f"{third}:refs/heads/main")
    h.tick()
    assert h.rec(1)["phase"] == "planning" and h.planned[-1]["base"] == third and len(h.planned) == 3


def test_the_shadow_never_touches_next_main_or_the_trains_records(world, monkeypatch, tmp_path):
    """The trains decide exactly as they do without the shadow: the same `next`, the same checks, the same landing; the shadow
    writes only `drive["per_pr"]` and its per-pr/N branches, cancels only its own runs, and its records end when the trains land
    their heads, deleting their branches."""
    def run(w: World, n: int) -> tuple[Shadow, list[str]]:
        h = Shadow(w, monkeypatch, n=n, trains=2)
        w.gh.put(1, w.change({"a.txt": "1\n"}))
        w.gh.put(2, w.change({"b.txt": "2\n"}))
        h.tick()
        trees = [_git(w.origin, "rev-parse", f"{r}^{{tree}}") for r in ("main", "next")]
        return h, trees

    (tmp_path / "plain").mkdir()
    plain, plain_trees = run(World(tmp_path / "plain"), 0)
    h, trees = run(world, 2)
    launched = h.q.load_state()["drive"]["launched"]
    chain = [m.commit for m in h.q.chain()]
    assert trees == plain_trees and [m.pr for m in h.q.chain()] == [m.pr for m in plain.q.chain()] == [1, 2]
    assert len(launched) == len(plain.q.load_state()["drive"]["launched"]) == 2 and set(launched) <= set(chain)
    ours = {h.check_of(n)["commit"] for n in (1, 2)}
    assert {h.envs[c][1:] for c in launched} == {(None, None)} and {h.envs[c] for c in ours} == {(None, "1", "1")}
    assert not ours & set(launched) and "per_pr" not in plain.q.load_state()["drive"]
    assert _branches(world) == {"main", "next", "per-pr/1", "per-pr/2"} and _branches(plain.world) == {"main", "next"}
    _check(world, monkeypatch, h.check_of(2)["commit"], ("false",))      # the shadow's #2 fails: the trains don't hear of it
    h.tick()
    assert [m.pr for m in h.q.chain()] == [1, 2] and h.posts == [] and h.q.load_state().get("ejected", {}) == {}
    before, trains = _main(world), {rid for r in h.q.load_state()["drive"]["launched"].values() for rid in r["ids"]}
    live, failed = h.check_of(1)["ids"], h.check_of(2)["ids"]
    _check(world, monkeypatch, _tip(world))                              # the trains' tip passes and lands both
    assert h.tick().startswith("landed #2") and _main(world) == _tip(world) != before
    assert set(h.cancelled) - trains == {*live, *failed}                  # its own runs only: #2's failed one, #1's at the landing
    assert {r["pr"]: r["ended"]["why"] for r in h.shadow()["done"]} == {1: "landed", 2: "landed"} and h.shadow()["prs"] == {}
    assert _branches(world) == {"main", "next"}


def test_off_by_default_and_a_dry_run_launches_and_pushes_nothing(world, monkeypatch):
    h = Shadow(world, monkeypatch, n=0)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    h.tick()
    assert "per_pr" not in h.q.load_state()["drive"] and h.launched == []
    h.d.per_pr_shadow = 2
    h.tick(dry_run=True)
    assert h.launched == [] and _branches(world) == {"main", "next"} and any("would check #1 as" in x for x in world.logs)


def test_a_conflict_with_main_is_recorded_and_takes_no_slot(world, monkeypatch):
    h = Shadow(world, monkeypatch, n=1)
    world.gh.put(3, world.change({"a.txt": "mine\n"}))                   # the oldest mark
    _git(world.dev, "push", "-q", "origin", f"{world.change({'a.txt': 'main s\n'})}:refs/heads/main")
    world.gh.put(1, world.change({"b.txt": "1\n"}))
    h.tick()
    assert h.rec(3)["phase"] == "conflict" and "conflicts with `main`" in h.rec(3)["conflict"]["why"] and "a.txt" in h.rec(3)["conflict"]["why"]
    assert h.rec(1)["phase"] == "checking" and len(h.launched) == 1


def test_a_shadow_failure_is_logged_and_the_trains_tick_goes_on(world, monkeypatch):
    h = Shadow(world, monkeypatch, n=1, trains=1)
    world.gh.put(1, world.change({"a.txt": "1\n"}))

    def broken():
        raise RuntimeError("labels unreadable")

    h.d._admitted = broken
    assert h.tick().startswith("launched") and len(h.launched) == 1
    assert any("per-PR shadow: failed this tick (RuntimeError: labels unreadable)" in x for x in world.logs)
    assert _git(h.q.workdir, "rev-parse", "HEAD") == _tip(world)        # the clone is left where the trains left it


def test_the_plan_is_judged_by_drives_rule_the_deferred_steps_aside(world, monkeypatch):
    """`merge.plan_refusal`, the `--verdicts` gate's, on the plan less the deferred steps' parts (`undeferred`): a build step
    those parts alone read goes with them, and one a part left reads stays."""
    h = Shadow(world, monkeypatch, n=1)
    h.d.defer = frozenset({"circuit-check", "pytest"})
    monkeypatch.setattr(merge, "pack_refusal", lambda *a: None)
    commit = _main(world)
    tree = _git(world.origin, "rev-parse", f"{commit}^{{tree}}")
    found = [{"attempt": "r20261020-000000-aaaa", "pack_sha256": "ab"}]

    def doc(**states):
        parts = [{"part": "suite:x", "step": "pytest", "state": states.get("x", "passed"), "found": found},
                 {"part": "suite:lean", "step": "lean-suites", "state": "passed", "found": found},
                 {"part": "circuit-check:t", "step": "circuit-check", "state": states.get("t", "passed"), "found": found},
                 {"part": "lean-audit:pkg", "step": "lean-audit", "state": states.get("pkg", "passed"), "found": found},
                 {"part": "preflight-lock", "step": "preflight-lock", "state": "implied", "implied_by": ["suite:x", "suite:lean"]},
                 {"part": "flock-circuit-build", "step": "flock-circuit-build", "state": "implied",
                  "implied_by": ["circuit-check:t"]}]
        return {"kind": "check-plan/v1", "commit": commit, "tree": tree, "parts": parts}

    h.q._clone()
    judge = type(h.d).judge_plan
    assert judge(h.d, doc(), commit, commit) == (["r20261020-000000-aaaa"], None, True)
    assert judge(h.d, doc(t="must-run", x="must-run"), commit, commit) == (["r20261020-000000-aaaa"], None, False)
    cited, why, full = judge(h.d, doc(pkg="must-run"), commit, commit)
    assert cited == [] and "lean-audit:pkg must-run" in why and not full
    left = undeferred(doc(), h.d.defer)
    assert [p["part"] for p in left["parts"]] == ["suite:lean", "lean-audit:pkg", "preflight-lock"]
    assert left["parts"][-1]["implied_by"] == ["suite:lean"]


def test_the_plan_job_runs_detached_on_each_machine_until_one_is_fully_passed_and_leaves_its_exit_last(world, tmp_path, monkeypatch):
    """`plan_launch`'s job, with a stand-in for `check.py --plan` that passes on the second machine only."""
    fake = tmp_path / "plan.py"
    fake.write_text("import json, sys\non = sys.argv[sys.argv.index('--on') + 1]\n"
                    "print(json.dumps({'kind': 'check-plan/v1', 'on': on, 'argv': sys.argv[1:]}))\nsys.exit(0 if on == 'm2' else 1)\n")
    monkeypatch.setattr(merge, "PLAN", ("python3", str(fake)))
    commit = _main(world)
    q = world.queue()
    q._clone()
    p = plan_launch(q.workdir, commit, commit, ["r1"], ["m1", "m2", "m3"], tmp_path / "out")
    with contextlib.suppress(ChildProcessError):                         # its end, not a clock
        os.waitpid(p["pid"], 0)
    state, docs, _ = plan_state(p)
    assert state == "ended" and [d["on"] for d in docs] == ["m1", "m2"] and (tmp_path / "out" / "rc").read_text() == "0\n"
    assert docs[0]["argv"][:6] == [commit, "--json", "--packs-only", "--base", commit, "--from"]
    assert not (tmp_path / "out" / "tree").exists() and str(tmp_path / "out" / "tree") not in _git(q.workdir, "worktree", "list")
    gone = subprocess.Popen(["true"])
    gone.wait()
    (tmp_path / "out" / "rc").unlink()
    assert plan_state({**p, "pid": gone.pid})[0] == "lost"


def _rec(pr: int, marked: float, sampled: float, would: float | None, landed: float | None, **kw) -> dict:
    return {"pr": pr, "head": f"{pr:08d}", "marked": marked, "sampled": sampled, "phase": "would-land" if would else "checking",
            "took": kw.get("took", sampled + 900), "checks": [{}] * kw.get("checks", 1), "plans": kw.get("plans", []),
            **({"would_land": {"at": would, "via": "check r", **kw.get("wl", {})}} if would else {}),
            **({"ended": {"at": landed, "why": "landed"}} if landed else {})}


def test_stats_per_pr_reads_the_shadows_records_from_drives_state_file(tmp_path, capsys):
    t0 = datetime(2026, 10, 21, tzinfo=timezone.utc).timestamp()
    h = 3600.0
    state = {"drive": {"per_pr": {
        "done": [_rec(1, t0, t0 + 0.5 * h, t0 + 1.0 * h, t0 + 6 * h),
                 _rec(2, t0, t0 + 1 * h, t0 + 3.0 * h, t0 + 9 * h, checks=2, wl={"revert": "circuit-check failed"},
                      plans=[{"result": "refused", "why": "lean-audit:x must-run"}, {"result": "passed"}])],
        "prs": {"3": _rec(3, t0, t0 + 2 * h, None, None), "4": _rec(4, t0 - 90 * h, t0 - 80 * h, None, None)}}}}
    rows = queue_stats.shadow_rows(state, datetime.fromtimestamp(t0, timezone.utc),
                                   datetime.fromtimestamp(t0 + 24 * h, timezone.utc))
    assert sorted(r["pr"] for r in rows) == [1, 2, 3]                    # #4 was sampled before the window
    s = queue_stats.shadow_summary(rows)
    assert s["sampled"] == 3 and s["would-land"] == 2 and s["checking"] == 1 and s["rechecked"] == 1 and s["would_revert"] == 1
    assert s["spans"]["path"] == {"n": 2, "median_h": 1.0, "p90_h": 1.75} and s["spans"]["trains"]["median_h"] == 7.5
    assert s["spans"]["slot"]["median_h"] == 0.25 and s["spans"]["queue"]["median_h"] == 0.75
    assert s["under_1h"] == 1 and s["plans"] == {"passed": 1, "refused": 1} and s["plan_refusals_naming_lean_audit"] == 1
    workdir = tmp_path / "q" / "verity"
    workdir.parent.mkdir(parents=True)
    (workdir.parent / "verity.state.json").write_text(json.dumps(state))
    assert queue_main(["stats", "--per-pr", "--workdir", str(workdir), "--until", "2026-10-22T00:00:00Z"]) == 0
    out = capsys.readouterr().out
    assert "the per-PR shadow sampled 3 PR(s): 2 would land, 0 failed, 0 conflict with `main`, 1 still going" in out
    assert "first check's slot -> would land (drive's rule)" in out and "median  1.00 h  p90  1.75 h  (n=2)" in out
    assert "first launch -> its slot (an ordinary job's wait)" in out


def test_the_shadow_yields_its_held_slots_and_launches_nothing_while_a_train_waits_for_one(world, monkeypatch):
    """CI-141: 21 Oct the shadow's ordinary jobs took vy-nebius-2's two check slots while nothing waited, then held them for an
    hour while trains queued behind. While a train's check waits for a slot, each shadow check holding one is cancelled and its
    PR queued again, and no new shadow check launches; once no train waits, the shadow checks it again."""
    h = Shadow(world, monkeypatch, n=2, trains=2)
    world.gh.put(1, world.change({"a.txt": "1\n"}))
    world.gh.put(2, world.change({"b.txt": "2\n"}))
    h.tick()
    ours = {n: h.check_of(n)["ids"] for n in (1, 2)}
    trains = {r for v in h.q.load_state()["drive"]["launched"].values() for r in v["ids"]}
    waiting = {"on": True}
    h.d._slot_state = lambda rid: ({"state": "waiting"} if rid in trains and waiting["on"] else
                                   {"state": "held", "took": "2026-10-20T12:00:00Z"} if rid not in trains else {})
    h.took.update({r: "2026-10-20T12:00:00Z" for ids in ours.values() for r in ids})
    before = len(h.launched)
    h.tick()
    assert set(h.cancelled) >= {r for ids in ours.values() for r in ids}
    assert [h.rec(n)["phase"] for n in (1, 2)] == ["queued", "queued"] and all(h.check_of(n).get("yielded") for n in (1, 2))
    assert len(h.launched) == before and any("yields its slot" in x for x in world.logs)
    waiting["on"] = False
    h.tick()
    assert len(h.launched) == before + 2 and [h.rec(n)["phase"] for n in (1, 2)] == ["checking", "checking"]
