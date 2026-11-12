"""`research.vqueue_records`: the records run of a verified-queue candidate with a member that changes Lean, a lock or a spec,
before its check, launched as a lock run (`Landing.launch_locks`) and judged as one (`lock_outcome`, `resend_locks`), and the
records commit of what it regenerated, a lock commit (`lock_message`) vouched for by the merge gate
(`merge.regenerated_parent`), on `test_landing`'s fakes: `Harness`'s launches, run states, evidence and clock, and
`_live_run`'s run dir, whose Lean report holds each package's report as lean_audit.py writes it."""
from __future__ import annotations

import json
import re

import pytest

from research import merge
from research.landing import CI, LAUNCH_LOST_S, LEAN_STEP, LOST_AFTER_S, NOT_STARTED
from research.vqueue import FAIL, INFRA, PASS, PENDING, RecordsRun
from research.vqueue_records import LOCK_SUBJECT, commit, judge, lock_message, start
from test_landing import Harness, _live_run, _steps_run
from test_queue import World, _git

#: the candidate's lock: two guarantees and a named assumption, each with its owner, and a guarantee with none
LOCKED = {"guarantees": {"T": {"owner": "@proofs"}, "U": {"owner": "@memory-accounting"}, "V": {}},
          "assumptions": {"H": {"owner": "@lean"}}, "reads": {}}
#: a candidate changing `pkg`'s Lean and its lock
CHANGE = {"pkg/A.lean": "theorem a : True := trivial\n", "pkg/lean-audit.json": json.dumps(LOCKED)}
#: what the records run regenerated of `pkg`'s lock
REGENERATED = json.dumps({**LOCKED, "reads": {"M": "2"}}) + "\n"


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def _harness(world: World, monkeypatch, files: dict | None = None) -> tuple[Harness, str]:
    """`Harness`, and candidate ``sha`` (``files``, else `CHANGE`, one commit above `main`) in the queue's clone."""
    h = Harness(world, monkeypatch)
    h.q._clone()
    sha = world.change(files or CHANGE)
    _git(h.q.workdir, "fetch", "-q", "origin", f"refs/dev/{sha}")
    return h, sha


def _report(d, rid: str, packages: dict[str, dict], **extra) -> None:
    """Run dir ``d``'s Lean report (`_live_run`'s) with a report of each package of ``packages`` as lean_audit.py writes one,
    settled on its regenerated record, with the review and `statement_changes` given (default none), and ``extra``."""
    doc = json.loads((d / merge.AUDIT).read_text())
    doc["packages"] = [{"package": pkg, "failures": [], "settled": True, "review": [], "statement_changes": False,
                        **({"regenerated": doc["regenerated"][pkg]} if pkg in doc["regenerated"] else {}), **rep}
                       for pkg, rep in packages.items()]
    (d / merge.AUDIT).write_text(json.dumps(doc | extra))


def _passes(h: Harness, rid: str, sha: str, records: dict[str, str | None] | None = None,
            reports: dict[str, dict] | None = None, **extra):
    """Records run ``rid`` of ``sha`` passed, having regenerated ``records`` ({package: its lock's text}, default `pkg`'s
    `REGENERATED`), each package's report ``reports``'s: its run dir."""
    records = {"pkg": REGENERATED} if records is None else records
    h.run_state[rid] = ("done", 0, None)
    h.evidence = [_steps_run(h.world.tmp, rid, {LEAN_STEP: "passed"})]
    d, h.d.fetch, _ = _live_run(h.world.tmp, rid, sha, records)
    _report(d, rid, {pkg: (reports or {}).get(pkg, {}) for pkg in {*records, *(reports or {})}}, **extra)
    return d


def _vouched(h: Harness, made: str, rid: str) -> str | None:
    return merge.regenerated_parent(h.q.workdir, made, h.q.store, tool="check", kinds=merge.produced(h.q.workdir, "check"),
                                    command=merge.declared_command(h.q.workdir, "check"), deferred=rid, fetch=h.d.fetch)


# -- start


def test_start_launches_a_lock_run_on_the_candidate_on_the_queues_machines_and_keeps_its_record(world, monkeypatch):
    """The records run is a lock run (`launch_locks`: train.sh's `--deferred lean-audit` at the after-landing rank), on the
    candidate's own tree and the queue's machines; its record is a launch's (`_started`) with the candidate and when it
    launched."""
    h, sha = _harness(world, monkeypatch)
    asked, launch = [], h.d.launch_locks
    h.d.launch_locks = lambda repo, tip, machines: asked.append((repo, tip, machines)) or launch(repo, tip, machines)
    rec = start(h.d, sha)
    (rid,) = h.run_state
    assert asked == [(h.q.workdir, sha, "m1")] and h.launched == [(sha, "m1", False)]
    assert rec == {"ids": [rid], "tip": sha, "at": h.clock}
    assert any(x == f"research queue vqueue: train.sh {sha[:12]} m1 --deferred {LEAN_STEP}, the records run of candidate "
                    f"{sha[:12]}: {rid}" for x in world.logs), world.logs


def test_start_launches_nothing_in_a_dry_run_and_none_without_the_queues_machines_or_when_it_registered_no_run(world,
                                                                                                                 monkeypatch):
    h, sha = _harness(world, monkeypatch)
    h.d.dry_run = True
    assert start(h.d, sha) is None and h.launched == []
    assert any(x.startswith(f"research queue vqueue: would launch train.sh {sha[:12]} m1") for x in world.logs), world.logs
    h.d.dry_run = False
    h.d.launch_locks = lambda repo, tip, machines: ([], "train: no slot")
    assert start(h.d, sha) is None and any(x.endswith("registered no run: train: no slot") for x in world.logs)
    h.d.on = ""
    with pytest.raises(ValueError, match=r"the queue's machines: .*\(`research\.queue\.flags_at\(repo, main\)\.on`\)"):
        start(h.d, sha)


# -- judge


def test_a_records_run_is_pending_while_it_runs_and_passes_once_its_records_say_the_same(world, monkeypatch):
    """Passed (exited 0, lean-audit passed in its steps.jsonl), it passes; the verdict is kept with the run that passed, and
    read from there."""
    h, sha = _harness(world, monkeypatch)
    rec = start(h.d, sha)
    (rid,) = rec["ids"]
    assert judge(h.d, rec) == RecordsRun(PENDING) and "verdict" not in rec
    _passes(h, rid, sha, reports={"pkg": {"review": ["guarantee T: printed differently, the same statement"]}})
    assert judge(h.d, rec) == RecordsRun(PASS)
    assert rec["verdict"] == {"state": PASS, "why": "", "run": rid}
    h.run_state.clear()                                                  # kept: no state is read again
    h.d.fetch = None
    assert judge(h.d, rec) == RecordsRun(PASS)


def test_a_records_run_that_failed_refuses_its_records_with_why(world, monkeypatch):
    h, sha = _harness(world, monkeypatch)
    rec = start(h.d, sha)
    (rid,) = rec["ids"]
    h.run_state[rid] = ("failed", 1, None)
    h.evidence = [_steps_run(h.world.tmp, rid, {LEAN_STEP: "failed"})]
    got = judge(h.d, rec)
    assert (got.state, got.why) == (FAIL, f"it failed: {rid} failed rc=1") and rec["verdict"]["state"] == FAIL
    assert len(h.launched) == 1


def test_a_records_run_that_exited_0_without_passing_lean_audit_fails(world, monkeypatch):
    h, sha = _harness(world, monkeypatch)
    rec = start(h.d, sha)
    (rid,) = rec["ids"]
    h.run_state[rid] = ("done", 0, None)
    h.evidence = [_steps_run(h.world.tmp, rid, {LEAN_STEP: "skipped"})]
    assert judge(h.d, rec) == RecordsRun(FAIL, f"it failed: {rid} exited 0 without passing {LEAN_STEP}")


def test_a_records_run_its_machine_failed_runs_again_once_elsewhere_and_its_resend_failing_refuses(world, monkeypatch):
    """Killed (or lost, or out of disk or memory) is its machine's fault (`machine_fault`): it is launched again once, on the
    queue's other machines, and is pending; its resend failing too refuses."""
    h, sha = _harness(world, monkeypatch)
    h.d.on = "m1,m2"
    rec = start(h.d, sha)
    (rid,) = rec["ids"]
    h.run_state[rid] = ("failed", -9, None, None, "m1")
    assert judge(h.d, rec) == RecordsRun(PENDING) and h.launched == [(sha, "m1,m2", False), (sha, "m2", False)]
    (again,) = rec["ids"]
    assert again != rid and rec["resend"] and "verdict" not in rec
    assert any(f"the records run of candidate {sha[:12]} was killed, a machine's fault: again, once, on m2: {again}" in x
               for x in world.logs), world.logs
    h.run_state[again] = ("failed", -9, None, None, "m2")
    got = judge(h.d, rec)
    assert got.state == FAIL and again in got.why and len(h.launched) == 2


@pytest.mark.parametrize("rows, log, why", [
    ([{"step": LEAN_STEP, "status": "passed"}, {"step": "lean-suites", "status": "failed", "rc": 1}], "",
     f"failed after its {LEAN_STEP} passed"),
    ([{"step": LEAN_STEP, "status": "failed", "rc": 137}], "", "killed"),
    ([{"step": LEAN_STEP, "status": "failed", "rc": 1, "log": f"{LEAN_STEP}.log"}], "error: No space left on device\n",
     f"failed in {LEAN_STEP} by its machine (No space left on device)"),
], ids=["after-audit", "killed-step", "enospc"])
def test_a_records_run_that_exited_1_by_its_machines_fault_runs_again_once_elsewhere(world, monkeypatch, rows, log, why):
    """A records run that exited 1 failed by its machine when its run dir says so (`machine_fault`): lean-audit passed and a
    step after it failed, a step was killed, or a failed step's log says its machine ran out of disk."""
    h, sha = _harness(world, monkeypatch)
    h.d.on = "m1,m2"
    rec = start(h.d, sha)
    (rid,) = rec["ids"]
    h.run_state[rid] = ("failed", 1, None, None, "m1")
    (d := h.world.tmp / rid).mkdir()
    (d / "steps.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    if log:
        (d / f"{LEAN_STEP}.log").write_text(log)
    h.evidence = [d]
    assert judge(h.d, rec) == RecordsRun(PENDING) and h.launched == [(sha, "m1,m2", False), (sha, "m2", False)]
    assert any(f"the records run of candidate {sha[:12]} was {why}, a machine's fault: again, once, on m2: {rec['ids'][0]}"
               in x for x in world.logs), world.logs


def test_a_records_run_lost_runs_again_once_elsewhere_and_its_resend_lost_refuses(world, monkeypatch):
    """A records run whose state can't be read for `LOST_AFTER_S` since its launch is lost (`Landing._lost`), a machine's
    fault as a killed one is: it runs again once elsewhere, and its resend lost too refuses."""
    h, sha = _harness(world, monkeypatch)
    h.d.on = "m1,m2"
    rec = start(h.d, sha)
    (rid,) = rec["ids"]
    h.run_state[rid] = ("unknown", None, None, None, "m1")
    assert judge(h.d, rec) == RecordsRun(PENDING) and len(h.launched) == 1
    h.clock += LOST_AFTER_S + 1
    assert judge(h.d, rec) == RecordsRun(PENDING) and h.launched[1:] == [(sha, "m2", False)]
    (again,) = rec["ids"]
    assert any(f"the records run of candidate {sha[:12]} was lost, a machine's fault: again, once, on m2: {again}" in x
               for x in world.logs), world.logs
    h.run_state[again] = ("unknown", None, None, None, "m2")
    h.clock += LOST_AFTER_S + 1
    assert judge(h.d, rec) == RecordsRun(FAIL, f"it failed: {again} lost on its resend") and len(h.launched) == 2


def test_a_records_run_that_never_ran_or_registered_no_run_is_infra(world, monkeypatch, tmp_path):
    h, sha = _harness(world, monkeypatch)
    assert judge(h.d, None) == RecordsRun(INFRA, "it registered no run")
    rec = start(h.d, sha)
    (rid,) = rec["ids"]
    h.run_state[rid] = ("failed", NOT_STARTED, None)
    assert judge(h.d, rec) == RecordsRun(INFRA, f"it never ran or was stopped ({rid})")
    h.d.launch_locks = lambda repo, tip, machines: {"dir": str(tmp_path / "launch"), "pid": 0}
    h.d.launched = lambda launch: (None, "train: abc on m1")
    detached = start(h.d, sha)
    assert detached["ids"] == [] and judge(h.d, detached) == RecordsRun(PENDING)
    h.clock += LAUNCH_LOST_S + 1
    got = judge(h.d, detached)
    assert got.state == INFRA and got.why.startswith("it registered no run: no run ids") and "launch" not in detached


def test_a_passed_records_run_whose_report_cant_be_read_yet_is_pending_with_why_and_judged_again(world, monkeypatch):
    h, sha = _harness(world, monkeypatch)
    rec = start(h.d, sha)
    (rid,) = rec["ids"]
    _passes(h, rid, sha)
    fetch, h.d.fetch = h.d.fetch, lambda r, names: None
    got = judge(h.d, rec)
    assert got.state == PENDING and f"{rid} passed, but its Lean report ({merge.AUDIT}) can't be read yet" in got.why
    assert "verdict" not in rec
    h.d.fetch = fetch
    assert judge(h.d, rec) == RecordsRun(PASS)


def test_a_passed_records_run_that_regenerated_nothing_passes_and_makes_no_records_commit(world, monkeypatch):
    """Its candidate's records were fresh (`regenerated_locks` finds none): no commit is made, and the loop checks the
    candidate as built."""
    h, sha = _harness(world, monkeypatch)
    rec = start(h.d, sha)
    (rid,) = rec["ids"]
    _passes(h, rid, sha, {})
    assert judge(h.d, rec) == RecordsRun(PASS)
    refs = _git(h.q.workdir, "for-each-ref")
    assert commit(h.d, sha, rid, (1,)) is None and _git(h.q.workdir, "for-each-ref") == refs


def test_regenerated_records_that_change_what_a_guarantee_says_pass_and_their_records_commit_names_each_change(
        world, monkeypatch):
    """A package whose regenerated record changes what a guarantee says (its report's `statement_changes`: a guarantee's
    statement, a definition one reads, a listed premise) refuses nothing: the run passes, and its records commit carries every
    record it regenerated and names each change, its details aside, under its package with the owners the candidate's lock
    gives what it names, in the order it names them, each once, and `CI` for one naming nothing the lock gives an owner.  The
    spec DM shows each change once it lands."""
    lib = {"guarantees": {"W": {"owner": "@proofs"}}, "reads": {}}
    h, sha = _harness(world, monkeypatch, {**CHANGE, "lib/lean-audit.json": json.dumps(lib)})
    rec = start(h.d, sha)
    (rid,) = rec["ids"]
    review = ["guarantee U: changed", "  before:", "    theorem U : 1 = 1", "  after:", "    theorem U : 2 = 2",
              "definition Pkg.d (Pkg.M), read by T, U: changed", "assumption H: changed (its statement or a definition it reads)",
              "guarantee V: changed", "definition Pkg.e (Pkg.M): new"]
    _passes(h, rid, sha, {"pkg": REGENERATED, "lib": json.dumps({**lib, "reads": {"N": "1"}}) + "\n"},
            reports={"pkg": {"review": review, "statement_changes": True},
                     "lib": {"review": ["guarantee W: printed differently, the same statement"]}})
    assert judge(h.d, rec) == RecordsRun(PASS) and rec["verdict"] == {"state": PASS, "why": "", "run": rid}
    made, files, told = commit(h.d, sha, rid, (1,))
    assert files == ["lib/lean-audit.json", "pkg/lean-audit.json"] and told is None
    named = ["pkg: guarantee U: changed (@memory-accounting)",
             "pkg: definition Pkg.d (Pkg.M), read by T, U: changed (@proofs, @memory-accounting)",
             "pkg: assumption H: changed (its statement or a definition it reads) (@lean)",
             f"pkg: guarantee V: changed ({CI})", f"pkg: definition Pkg.e (Pkg.M): new ({CI})"]
    body = _git(h.q.workdir, "log", "-1", "--format=%B", made)
    assert body == lock_message("#1", rid, f"queue/{sha[:12]}", sha, [], files, named).strip(), body
    assert body.startswith(f"{LOCK_SUBJECT}#1\n\nWhat these records change of what a guarantee says, which the spec DM shows "
                           "once they land:\n" + "".join(f"  {n}\n" for n in named) + f"\nChecked: {rid} "), body
    assert re.search(rf"^Checked: {rid} ", body, re.M) and re.search(r"^Regenerated: \S", body, re.M)
    assert (_git(h.q.workdir, "show", f"{made}:pkg/lean-audit.json"), _vouched(h, made, rid)) == (REGENERATED.strip(), rid)


def test_a_records_commit_is_held_while_its_lean_report_cant_be_read_to_name_its_changes(world, monkeypatch):
    """Never a records commit naming none of the changes it makes: held, with why, in words that stay the same tick to tick."""
    import research.vqueue_records as R

    h, sha = _harness(world, monkeypatch)
    rid = start(h.d, sha)["ids"][0]
    _passes(h, rid, sha, reports={"pkg": {"review": ["guarantee T: changed"], "statement_changes": True}})
    refs = _git(h.q.workdir, "for-each-ref")
    monkeypatch.setattr(R, "lean_report", lambda *a: (_ for _ in ()).throw(OSError(f"gone: {h.world.tmp}/scratch-{len(refs)}")))
    got = commit(h.d, sha, rid, (1,))
    assert got == (f"{rid}'s Lean report ({merge.AUDIT}) or a lock it names can't be read (OSError), so a records commit "
                   "couldn't name what its records change of what a guarantee says") and got == commit(h.d, sha, rid, (1,)), got
    monkeypatch.setattr(R, "lean_report", lambda *a: None)
    assert commit(h.d, sha, rid, (1,)).startswith(f"{rid}'s Lean report ({merge.AUDIT}) or a lock it names can't be read "
                                                  "(QueueError)")
    assert _git(h.q.workdir, "for-each-ref") == refs


def test_a_records_commit_is_held_while_a_lock_whose_owners_it_names_cant_be_read(world, monkeypatch):
    """A lock of the candidate's that isn't JSON says no owner: its records commit is held, not made naming `CI` for them."""
    h, sha = _harness(world, monkeypatch, {**CHANGE, "pkg/lean-audit.json": "{"})
    rid = start(h.d, sha)["ids"][0]
    _passes(h, rid, sha, reports={"pkg": {"review": ["guarantee T: changed"], "statement_changes": True}})
    refs = _git(h.q.workdir, "for-each-ref")
    assert commit(h.d, sha, rid, (1,)) == (f"{rid}'s Lean report ({merge.AUDIT}) or a lock it names can't be read "
                                           "(JSONDecodeError), so a records commit couldn't name what its records change of "
                                           "what a guarantee says")
    assert _git(h.q.workdir, "for-each-ref") == refs


# -- commit


def test_the_records_commit_is_a_lock_commit_on_the_candidate_which_the_merge_gate_vouches_for(world, monkeypatch):
    """The records the run regenerated, committed on top of the candidate as a lock commit (`lock_message`, `queue.IDENTITY`),
    in a scratch index: the clone's checkout, index and refs stay as they are.  The gate vouches for it against the run, and
    post_train reads its trailers."""
    h, sha = _harness(world, monkeypatch)
    rid = start(h.d, sha)["ids"][0]
    _passes(h, rid, sha)
    wd = h.q.workdir
    head, status, refs = (_git(wd, *a) for a in (("rev-parse", "HEAD"), ("status", "--porcelain"), ("for-each-ref",)))
    made, files, told = commit(h.d, sha, rid, (1, 2))
    assert files == ["pkg/lean-audit.json"] and _git(wd, "rev-parse", f"{made}^@") == sha and told is None
    assert _git(wd, "show", f"{made}:pkg/lean-audit.json") == REGENERATED.strip()
    assert _git(wd, "diff", "--name-only", sha, made) == "pkg/lean-audit.json"
    body = _git(wd, "log", "-1", "--format=%B", made)
    assert body == lock_message("#1 #2", rid, f"queue/{sha[:12]}", sha, [], files).strip(), body
    assert body.startswith(f"Lock records regenerated for #1 #2\n\nChecked: {rid} (the lock run of `queue/{sha[:12]}` "
                           f"{sha[:12]}, passed by regenerating them)\nRegenerated: pkg/lean-audit.json"), body
    assert re.search(rf"^Checked: {rid} ", body, re.M) and re.search(r"^Regenerated: \S", body, re.M)
    assert _git(wd, "log", "-1", "--format=%an <%ae>", made) == "research queue <queue@research.invalid>"
    assert _vouched(h, made, rid) == rid
    assert (_git(wd, "rev-parse", "HEAD"), _git(wd, "status", "--porcelain"), _git(wd, "for-each-ref")) == (head, status, refs)


def test_a_records_commit_carries_the_bookkeeping_its_records_run_repaired_and_tells_its_owners(world, monkeypatch):
    h, sha = _harness(world, monkeypatch)
    rid = start(h.d, sha)["ids"][0]
    d = _passes(h, rid, sha)
    caps = "UNWITNESSED = {\"pkg\": 2}\n"
    (d / "lean-audit" / "repaired" / "tests").mkdir(parents=True)
    (d / "lean-audit" / "repaired" / "tests" / "test_lean_packages.py").write_text(caps)
    warned = ["pkg: bookkeeping: H (@lean): listed under `open` on LEAN-1, due 2026-11-14 (open.unlisted)"]
    doc = json.loads((d / merge.AUDIT).read_text())
    (d / merge.AUDIT).write_text(json.dumps(doc | {"bookkeeping": warned, "repaired": {
        "tests/test_lean_packages.py": f"/workspace/research/runs/{rid}/lean-audit/repaired/tests/test_lean_packages.py"}}))
    made, files, told = commit(h.d, sha, rid, (1,))
    assert files == ["pkg/lean-audit.json", "tests/test_lean_packages.py"]
    body = _git(h.q.workdir, "log", "-1", "--format=%B", made)
    assert body == lock_message("#1", rid, f"queue/{sha[:12]}", sha, warned, files).strip(), body
    assert f"  {warned[0]}\n" in body and _vouched(h, made, rid) == rid
    assert told == (f"the lock run {rid} of `queue/{sha[:12]}` {sha[:12]} repaired bookkeeping that #1 brought, in "
                    f"{made[:12]}, each a warning for the next PR to fix: {warned[0]}", ("lean",))


def test_a_records_commit_is_held_while_its_records_cant_be_read_or_the_gate_wont_vouch_and_none_is_made_of_the_same(world,
                                                                                                                 monkeypatch):
    """Held, with why, in words that stay the same tick to tick (the loop rings each once); records the candidate holds
    already make no commit, as none regenerated makes none."""
    import research.vqueue_records as R

    h, sha = _harness(world, monkeypatch)
    rid = start(h.d, sha)["ids"][0]
    d = _passes(h, rid, sha)
    (d / "lean-audit" / "audit-pkg" / "lean-audit.json").unlink()
    got = commit(h.d, sha, rid, (1,))
    assert isinstance(got, str) and got.startswith("its regenerated lock records can't be read: Unreadable"), got
    _passes(h, rid2 := start(h.d, sha)["ids"][0], sha, {"pkg": json.dumps(LOCKED)})
    assert commit(h.d, sha, rid2, (1,)) is None
    _passes(h, rid3 := start(h.d, sha)["ids"][0], f"{'0' * 40}")          # it ran on another tree
    got = commit(h.d, sha, rid3, (1,))
    assert isinstance(got, str) and got.startswith("the merge gate refuses its lock commit"), got
    monkeypatch.setattr(R, "regenerated_locks", lambda driver, rid, scratch: (
        {}, f"its regenerated lock records can't be read: OSError: [Errno 28] No space left on device: '{scratch / rid / 'pkg'}'"))
    got = commit(h.d, sha, rid, (1,))
    assert got == commit(h.d, sha, rid, (1,)) and got.endswith(f"device: '<scratch>/{rid}/pkg'"), got   # rung once, not each tick


def test_a_records_commit_is_held_while_no_lean_report_says_whether_lean_audit_regenerated_records_unless_it_was_skipped(
        world, monkeypatch):
    """Records neither the store nor the run dir on its node holds hold the records commit, with why, while lean-audit may
    have regenerated some (`lean_audit_ran`); a run whose steps.jsonl says lean-audit was skipped regenerated none, and makes
    no commit."""
    h, sha = _harness(world, monkeypatch)
    rid = start(h.d, sha)["ids"][0]
    _passes(h, rid, sha)
    h.d.fetch = lambda r, names: None
    refs = _git(h.q.workdir, "for-each-ref")
    assert commit(h.d, sha, rid, (1,)) == (f"its {merge.AUDIT}, which says whether it regenerated lock records, can't be read "
                                           "from the store or from its run dir on its node")
    h.evidence = [_steps_run(h.world.tmp / "skipped", rid, {LEAN_STEP: "skipped"})]
    assert commit(h.d, sha, rid, (1,)) is None and _git(h.q.workdir, "for-each-ref") == refs


def test_lean_audit_ran_reads_the_stored_result_and_takes_a_store_it_cant_read_as_ran(monkeypatch, tmp_path):
    """`lean_audit_ran` of a run whose result is in the store: lean-audit skipped or absent there regenerated nothing; a run
    the store can't be read of may have."""
    import types

    from research.vqueue_records import lean_audit_ran

    class Store:
        def get_attempt(self, rid):
            if rid == "unreadable":
                raise OSError("the store is down")
            return {"outputs": {"result": "sha512:r"}}

    monkeypatch.setattr(merge, "step_status", lambda store, rid, step: rid)
    d = types.SimpleNamespace(q=types.SimpleNamespace(store=Store()), repo=tmp_path, evidence=lambda repo, ids: [])
    assert [lean_audit_ran(d, r) for r in ("passed", "skipped", "absent", "unreadable")] == [True, False, False, True]
