"""`research.vqueue_records`: the records run of a verified-queue candidate whose own failures are all stale Lean lock records,
launched as drive launches its lock run (`Driver.launch_locks`) and judged as drive judges it (`Driver._lock_outcome`,
`Driver._resend_locks`), and the records commit of what it regenerated, in drive's lock commit's form (`drive.lock_message`)
and vouched for by the merge gate (`merge.regenerated_parent`), on drive's own fakes: `Harness`'s launches, run states,
evidence and clock, and `_live_run`'s run dir, whose Lean report holds each package's report as lean_audit.py writes it."""
from __future__ import annotations

import json
import re

import pytest

from research import merge
from research.drive import CI, LAUNCH_LOST_S, LEAN_STEP, LOCK_SUBJECT, NOT_STARTED, lock_message
from research.queue import Member
from research.vqueue import FAIL, INFRA, PASS, PENDING, RecordsRun
from research.vqueue_records import SHOWN, commit, judge, start
from test_queue import World, _git
from test_queue_drive import Harness, _live_run, _steps_run

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
    """Drive's harness, and candidate ``sha`` (``files``, else `CHANGE`, one commit above `main`) in the queue's clone."""
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


def test_start_launches_drives_lock_run_on_the_candidate_on_drives_machines_and_keeps_its_record(world, monkeypatch):
    """The records run is drive's lock run (`launch_locks`: train.sh's `--deferred lean-audit` at the after-landing rank), on
    the candidate's own tree and drive's machines; its record is drive's (`_started`) with the candidate, when it launched
    and the packages graph2 found stale."""
    h, sha = _harness(world, monkeypatch)
    asked, launch = [], h.d.launch_locks
    h.d.launch_locks = lambda repo, tip, machines: asked.append((repo, tip, machines)) or launch(repo, tip, machines)
    rec = start(h.d, sha, ["pkg", "lib"])
    (rid,) = h.run_state
    assert asked == [(h.q.workdir, sha, "m1")] and h.launched == [(sha, "m1", False)]
    assert rec == {"ids": [rid], "tip": sha, "at": h.clock, "packages": ["lib", "pkg"]}
    assert any(x == f"research queue vqueue: train.sh {sha[:12]} m1 --deferred {LEAN_STEP}, the records run of candidate "
                    f"{sha[:12]}: {rid}" for x in world.logs), world.logs


def test_start_launches_nothing_in_a_dry_run_and_none_without_drives_machines_or_when_it_registered_no_run(world, monkeypatch):
    h, sha = _harness(world, monkeypatch)
    h.d.dry_run = True
    assert start(h.d, sha, ["pkg"]) is None and h.launched == []
    assert any(x.startswith(f"research queue vqueue: would launch train.sh {sha[:12]} m1") for x in world.logs), world.logs
    h.d.dry_run = False
    h.d.launch_locks = lambda repo, tip, machines: ([], "train: no slot")
    assert start(h.d, sha, ["pkg"]) is None and any(x.endswith("registered no run: train: no slot") for x in world.logs)
    h.d.on = ""
    with pytest.raises(ValueError, match="drive's machines"):
        start(h.d, sha, ["pkg"])


# -- judge


def test_a_records_run_is_pending_while_it_runs_and_passes_once_it_regenerated_each_stale_record_saying_the_same(world,
                                                                                                                monkeypatch):
    """Passed (exited 0, lean-audit passed in its steps.jsonl), with a record regenerated for each package graph2 found stale
    and none changing what a guarantee says, it passes; the verdict is kept with the run that passed, and read from there."""
    h, sha = _harness(world, monkeypatch)
    rec = start(h.d, sha, ["pkg"])
    (rid,) = rec["ids"]
    assert judge(h.d, rec) == RecordsRun(PENDING) and "verdict" not in rec
    _passes(h, rid, sha, reports={"pkg": {"review": ["guarantee T: printed differently, the same statement"]}})
    assert judge(h.d, rec) == RecordsRun(PASS)
    assert rec["verdict"] == {"state": PASS, "why": "", "owners": [], "run": rid}
    h.run_state.clear()                                                  # kept: no state is read again
    h.d.fetch = None
    assert judge(h.d, rec) == RecordsRun(PASS)


def test_a_records_run_that_failed_refuses_its_records_with_why(world, monkeypatch):
    h, sha = _harness(world, monkeypatch)
    rec = start(h.d, sha, ["pkg"])
    (rid,) = rec["ids"]
    h.run_state[rid] = ("failed", 1, None)
    h.evidence = [_steps_run(h.world.tmp, rid, {LEAN_STEP: "failed"})]
    got = judge(h.d, rec)
    assert (got.state, got.why) == (FAIL, f"it failed: {rid} failed rc=1") and rec["verdict"]["state"] == FAIL
    assert len(h.launched) == 1


def test_a_records_run_that_exited_0_without_passing_lean_audit_fails(world, monkeypatch):
    h, sha = _harness(world, monkeypatch)
    rec = start(h.d, sha, ["pkg"])
    (rid,) = rec["ids"]
    h.run_state[rid] = ("done", 0, None)
    h.evidence = [_steps_run(h.world.tmp, rid, {LEAN_STEP: "skipped"})]
    assert judge(h.d, rec) == RecordsRun(FAIL, f"it failed: {rid} exited 0 without passing {LEAN_STEP}")


def test_a_records_run_its_machine_failed_runs_again_once_elsewhere_and_its_resend_failing_refuses(world, monkeypatch):
    """Killed (or lost, or out of disk or memory) is its machine's fault, as drive's lock run's is: it is launched again once,
    on drive's other machines, and is pending; its resend failing too refuses."""
    h, sha = _harness(world, monkeypatch)
    h.d.on = "m1,m2"
    rec = start(h.d, sha, ["pkg"])
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


def test_a_records_run_that_never_ran_or_registered_no_run_is_infra(world, monkeypatch, tmp_path):
    h, sha = _harness(world, monkeypatch)
    assert judge(h.d, None) == RecordsRun(INFRA, "it registered no run")
    rec = start(h.d, sha, ["pkg"])
    (rid,) = rec["ids"]
    h.run_state[rid] = ("failed", NOT_STARTED, None)
    assert judge(h.d, rec) == RecordsRun(INFRA, f"it never ran or was stopped ({rid})")
    h.d.launch_locks = lambda repo, tip, machines: {"dir": str(tmp_path / "launch"), "pid": 0}
    h.d.launched = lambda launch: (None, "train: abc on m1")
    detached = start(h.d, sha, ["pkg"])
    assert detached["ids"] == [] and judge(h.d, detached) == RecordsRun(PENDING)
    h.clock += LAUNCH_LOST_S + 1
    got = judge(h.d, detached)
    assert got.state == INFRA and got.why.startswith("it registered no run: no run ids") and "launch" not in detached


def test_a_passed_records_run_whose_report_cant_be_read_yet_is_pending_with_why_and_judged_again(world, monkeypatch):
    h, sha = _harness(world, monkeypatch)
    rec = start(h.d, sha, ["pkg"])
    (rid,) = rec["ids"]
    _passes(h, rid, sha)
    fetch, h.d.fetch = h.d.fetch, lambda r, names: None
    got = judge(h.d, rec)
    assert got.state == PENDING and f"{rid} passed, but its Lean report ({merge.AUDIT}) can't be read yet" in got.why
    assert "verdict" not in rec
    h.d.fetch = fetch
    assert judge(h.d, rec) == RecordsRun(PASS)


def test_a_passed_records_run_that_regenerated_no_record_of_a_package_graph2_found_stale_refuses(world, monkeypatch):
    h, sha = _harness(world, monkeypatch)
    rec = start(h.d, sha, ["lib", "pkg"])
    (rid,) = rec["ids"]
    _passes(h, rid, sha)
    assert judge(h.d, rec) == RecordsRun(FAIL, f"{rid} passed without regenerating the records of lib, which graph2 found "
                                                "stale")


def test_regenerated_records_that_change_what_a_guarantee_says_are_refused_naming_the_review_and_rung_to_their_owners(
        world, monkeypatch):
    """A package whose regenerated record changes what a guarantee says (its report's `statement_changes`: a guarantee's
    statement, a definition one reads, a listed premise) is a claim change: refused, its review's lines named, and the owners
    the candidate's lock gives what they name are the ones to ring, in the order the review names them, each once."""
    h, sha = _harness(world, monkeypatch)
    rec = start(h.d, sha, ["pkg"])
    (rid,) = rec["ids"]
    review = ["guarantee U: changed", "  before:", "    theorem U : 1 = 1", "  after:", "    theorem U : 2 = 2",
              "definition Pkg.d (Pkg.M), read by T, U: changed", "assumption H: changed (its statement or a definition it reads)",
              "guarantee T: printed differently, the same statement"]
    _passes(h, rid, sha, reports={"pkg": {"review": review, "statement_changes": True}})
    got = judge(h.d, rec)
    assert (got.state, got.owners) == (FAIL, ("@memory-accounting", "@proofs", "@lean")), got
    assert got.why == (f"regenerating the records of pkg ({rid}) changes what a guarantee says, which the change's own records "
                       "must say for its owners to review: guarantee U: changed; definition Pkg.d (Pkg.M), read by T, U: "
                       "changed; assumption H: changed (its statement or a definition it reads); guarantee T: printed "
                       "differently, the same statement")
    assert rec["verdict"] == {"state": FAIL, "why": got.why, "owners": list(got.owners), "run": rid}
    assert judge(h.d, rec) == got


def test_a_statement_change_naming_nothing_its_lock_gives_an_owner_rings_ci_and_a_long_review_is_cut(world, monkeypatch):
    h, sha = _harness(world, monkeypatch)
    rec = start(h.d, sha, ["pkg"])
    (rid,) = rec["ids"]
    review = [f"definition Pkg.d{i} (Pkg.M): new" for i in range(SHOWN + 2)] + ["guarantee V: changed"]
    _passes(h, rid, sha, reports={"pkg": {"review": review, "statement_changes": True}})
    got = judge(h.d, rec)
    assert got.state == FAIL and got.owners == (CI,), got
    assert got.why.endswith(f"definition Pkg.d{SHOWN - 1} (Pkg.M): new; and 3 more in its Lean report"), got.why


# -- commit


def test_the_records_commit_is_drives_lock_commit_on_the_candidate_which_the_merge_gate_vouches_for(world, monkeypatch):
    """The records the run regenerated, committed on top of the candidate as drive commits its lock run's (`lock_message`,
    `queue.IDENTITY`), in a scratch index: the clone's checkout, index and refs stay as they are.  The gate vouches for it
    against the run, post_train reads its trailers, and drive reverts it with the PRs its subject names (`_lock_commits`)."""
    h, sha = _harness(world, monkeypatch)
    rid = start(h.d, sha, ["pkg"])["ids"][0]
    _passes(h, rid, sha)
    wd = h.q.workdir
    head, status, refs = (_git(wd, *a) for a in (("rev-parse", "HEAD"), ("status", "--porcelain"), ("for-each-ref",)))
    made, files = commit(h.d, sha, rid, (1, 2))
    assert files == ["pkg/lean-audit.json"] and _git(wd, "rev-parse", f"{made}^@") == sha
    assert _git(wd, "show", f"{made}:pkg/lean-audit.json") == REGENERATED.strip()
    assert _git(wd, "diff", "--name-only", sha, made) == "pkg/lean-audit.json"
    body = _git(wd, "log", "-1", "--format=%B", made)
    assert body == lock_message("#1 #2", rid, f"queue/{sha[:12]}", sha, [], files).strip(), body
    assert body.startswith(f"{LOCK_SUBJECT}#1 #2\n\nChecked: {rid} (the lock run of `queue/{sha[:12]}` {sha[:12]}, passed "
                           "by regenerating them)\nRegenerated: pkg/lean-audit.json"), body
    assert re.search(rf"^Checked: {rid} ", body, re.M) and re.search(r"^Regenerated: \S", body, re.M)
    assert _git(wd, "log", "-1", "--format=%an <%ae>", made) == "research queue <queue@research.invalid>"
    assert _vouched(h, made, rid) == rid
    assert (_git(wd, "rev-parse", "HEAD"), _git(wd, "status", "--porcelain"), _git(wd, "for-each-ref")) == (head, status, refs)
    assert h.d._lock_commits(f"{sha}^", made, [Member(1, sha, "lane-1", sha)]) == [made]
    assert h.d._lock_commits(f"{sha}^", made, [Member(3, sha, "lane-3", sha)]) == []


def test_a_records_commit_carries_the_bookkeeping_its_records_run_repaired_as_drives_does(world, monkeypatch):
    h, sha = _harness(world, monkeypatch)
    rid = start(h.d, sha, ["pkg"])["ids"][0]
    d = _passes(h, rid, sha)
    caps = "UNWITNESSED = {\"pkg\": 2}\n"
    (d / "lean-audit" / "repaired" / "tests").mkdir(parents=True)
    (d / "lean-audit" / "repaired" / "tests" / "test_lean_packages.py").write_text(caps)
    warned = ["pkg: bookkeeping: H (@lean): listed under `open` on LEAN-1, due 2026-11-14 (open.unlisted)"]
    doc = json.loads((d / merge.AUDIT).read_text())
    (d / merge.AUDIT).write_text(json.dumps(doc | {"bookkeeping": warned, "repaired": {
        "tests/test_lean_packages.py": f"/workspace/research/runs/{rid}/lean-audit/repaired/tests/test_lean_packages.py"}}))
    made, files = commit(h.d, sha, rid, (1,))
    assert files == ["pkg/lean-audit.json", "tests/test_lean_packages.py"]
    body = _git(h.q.workdir, "log", "-1", "--format=%B", made)
    assert body == lock_message("#1", rid, f"queue/{sha[:12]}", sha, warned, files).strip(), body
    assert f"  {warned[0]}\n" in body and _vouched(h, made, rid) == rid


def test_a_records_commit_is_refused_when_its_records_cant_be_read_are_the_candidates_or_the_gate_wont_vouch(world,
                                                                                                           monkeypatch):
    h, sha = _harness(world, monkeypatch)
    rid = start(h.d, sha, ["pkg"])["ids"][0]
    d = _passes(h, rid, sha)
    (d / "lean-audit" / "audit-pkg" / "lean-audit.json").unlink()
    got = commit(h.d, sha, rid, (1,))
    assert isinstance(got, str) and got.startswith("its regenerated lock records can't be read: Unreadable"), got
    _passes(h, rid2 := start(h.d, sha, ["pkg"])["ids"][0], sha, {"pkg": json.dumps(LOCKED)})
    assert commit(h.d, sha, rid2, (1,)) == f"the records {rid2} regenerated are those {sha[:12]} holds"
    _passes(h, rid3 := start(h.d, sha, ["pkg"])["ids"][0], f"{'0' * 40}")          # it ran on another tree
    got = commit(h.d, sha, rid3, (1,))
    assert isinstance(got, str) and got.startswith("the merge gate refuses its lock commit"), got
    _passes(h, rid4 := start(h.d, sha, ["pkg"])["ids"][0], sha, {})
    assert commit(h.d, sha, rid4, (1,)) == f"{rid4}'s Lean report names no record it regenerated"
