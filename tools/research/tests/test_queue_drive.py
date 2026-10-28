"""`research queue drive`: what a tick reads of a failed run (the files it implicates, a kill), what it tells and to whom, and the
branch sweep, on `test_queue`'s throwaway origin with the launch, the runs' states and the merge-thread post injected.  `Harness` is
the drive `test_drive_review` and `test_drive_stale` build on."""
from __future__ import annotations

import pytest

from research import merge
from research.drive import Driver
from test_queue import World, _git


class Harness:
    """A drive on ``world`` whose Lean-check launches are recorded (`launched`) and whose after-landing launches are recorded
    (`posted`) as `r-post-N`, each running until a test sets its state in `run_state`."""

    def __init__(self, world: World, monkeypatch, **kw):
        self.world, self.launched, self.posted, self.posts, self.after, self.cancelled = world, [], [], [], [], []
        self.run_state: dict[str, tuple] = {}
        self.clock = 1_000_000.0
        monkeypatch.setattr(merge, "caught_up", lambda repo, remote, tool, since="", log=print, runs=None:
                            self.after.append((since, runs)))
        monkeypatch.setattr(merge, "_sweep_after", lambda *a: None)
        monkeypatch.setattr(merge, "sweep_branches", lambda *a, **k: (0, 0))
        self.q = world.queue(**kw)
        self.d = Driver(self.q, on="m1", launch=self.launch, states=lambda repo, ids: {i: self.run_state[i] for i in ids},
                        post=lambda text, to=(): self.posts.append(text) or True, now=lambda: self.clock,
                        cancel=lambda repo, ids, reason: self.cancelled.extend(ids), evidence=lambda repo, ids: self.evidence)
        self.evidence: list = []
        self.d.launch_postland = self.launch_post
        self.d.lean_runs = lambda commits, since: {}

    def launch(self, repo, commit, machine, agreement):
        rid = f"r20261012-0000{len(self.launched):02d}-abcd"
        self.launched.append((commit, machine, agreement))
        self.run_state[rid] = ("running", None, None)
        return [rid], ""

    def launch_post(self, repo, commit, machine, steps, **kw):
        rid = f"r-post-{len(self.posted)}"
        self.posted.append((commit, steps, kw))
        self.run_state[rid] = ("running", None, None)
        return [rid], ""

    def tick(self, **kw) -> str:
        return self.d.tick(**kw)


def _tip(world: World) -> str:
    return _git(world.origin, "rev-parse", "refs/heads/next")


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def _failed_run(tmp_path, test: str, trace: str, steps: str = '{"step": "lean-suites", "status": "failed"}\n'):
    """A check's run dir whose suite `pkg` failed ``test``, its log naming ``trace``."""
    import json

    d = tmp_path / "run"
    (d / "suites").mkdir(parents=True)
    (d / "steps.jsonl").write_text(steps)
    (d / "suites-lean.json").write_text(json.dumps([{"name": "pkg", "suite": "pkg", "status": "failed", "failed_tests": [test],
                                                     "log": "/workspace/research/runs/r/suites/pkg.log"}]))
    (d / "suites" / "pkg.log").write_text(f"E       NameError: name '_once' is not defined\n\n{trace}: NameError\n")
    return d


LEAN_AUDIT_LOG = """lean-audit: verity/core/primitives/world: PASS reused (eabc707c8940): every file its audit reads is unchanged
AUDIT /workspace/research/cache/verity-check/lean-audit-scratch-2957380-6l2lanb_/Security/Proofs: FAIL; built WITHOUT a sandbox
  build: exit 1
    warning: ../Proofs/Flock/Recursive/PartTie.lean:168:47: Used `tac1 <;> tac2`
    error: ../Proofs/Flock/Claims/NumbersRadius.lean:100:2: Tactic `rfl` failed: The left-hand side
    error: build failed
AUDIT /workspace/research/cache/verity-check/lean-audit-scratch-2957380-6l2lanb_/Security: FAIL; built WITHOUT a sandbox
    error: Definitions/Claims/Sampling.lean:512:20: Type mismatch
"""


def _lean_run(tmp_path, log: str = LEAN_AUDIT_LOG):
    d = tmp_path / "run"
    d.mkdir(parents=True)
    (d / "steps.jsonl").write_text('{"step": "lean-audit", "status": "failed", "rc": 1, "log": "lean-audit.log"}\n')
    (d / "lean-audit.log").write_text(log)
    return d


def test_a_lean_audit_failure_names_the_lean_files_its_build_errors_are_in(tmp_path):
    """A train that fails in lean-audit isn't told as naming no file, to be attributed by hand."""
    from research.drive import implicated, lean_errors

    assert lean_errors(LEAN_AUDIT_LOG) == {"Security/Proofs/Flock/Claims/NumbersRadius.lean",
                                           "Security/Definitions/Claims/Sampling.lean"}
    assert implicated(_lean_run(tmp_path)) == lean_errors(LEAN_AUDIT_LOG)
    assert implicated(_lean_run(tmp_path / "b", "AUDIT /x/lean-audit-scratch-1/Security: FAIL\n    error: build failed\n")) is None


def _killed_run(tmp_path, name: str):
    d = tmp_path / name
    d.mkdir(parents=True)
    (d / "steps.jsonl").write_text('{"step": "circuit-check", "status": "failed", "rc": -9, "log": "circuit-check.log"}\n')
    (d / "circuit-check.log").write_text("")
    return d


LINT_LOG = """..........F........                                                      [100%]
=================================== FAILURES ===================================
___________________________ test_markdown_size_caps ____________________________

>       assert not offenders, "maintained documents are small:\\n" + "\\n".join(offenders)
E       AssertionError: maintained documents are small:
E         tools/research/src/research/store/README.md: 48 KB > 48 KB, 17 bytes over; move detail to the notes store

tests/test_repository.py:180: AssertionError
=========================== short test summary info ============================
FAILED tests/test_repository.py::test_markdown_size_caps - AssertionError: ma...
1 failed, 18 passed in 17.49s
"""


def _lint_run(tmp_path, log: str = LINT_LOG):
    d = tmp_path / "run"
    d.mkdir(parents=True)
    (d / "steps.jsonl").write_text('{"step": "preflight-lock", "status": "passed"}\n'
                                   '{"step": "preflight-lints", "status": "failed", "rc": 1, "log": "preflight-lints.log"}\n')
    (d / "preflight-lints.log").write_text(log)
    return d


def test_a_failed_pytest_step_points_at_its_tests_and_the_paths_its_failures_name(tmp_path):
    from research.drive import implicated

    assert implicated(_lint_run(tmp_path)) == {"tests/test_repository.py", "tools/research/src/research/store/README.md"}
    assert implicated(_lint_run(tmp_path / "b", "error: lock is stale\n")) is None


def test_a_failure_outside_the_suites_or_without_a_failing_test_points_at_no_file(tmp_path):
    from research.drive import implicated, touches

    d = _failed_run(tmp_path, "tests/test_x.py::test_y", 'File "/x/site/pkg/conftest.py", line 3')
    assert implicated(d) >= {"pkg/tests/test_x.py", "/x/site/pkg/conftest.py"}
    assert touches({"conftest.py", "pkg/conftest.py"}, {"/x/site/pkg/conftest.py"}) == {"pkg/conftest.py"}
    (d / "steps.jsonl").write_text('{"step": "circuit-check", "status": "failed"}\n')
    assert implicated(d) is None


def test_a_guard_failed_suite_with_no_failing_test_points_at_its_outside_reads_and_their_readers(tmp_path):
    """A suite can fail with every test passing — the read guard alone — and then "the failure names no file" would leave
    attribution to an agent's archaeology. The suite's record names the files
    read outside the inputs and the test module that read each (`outside`, path -> module; a child's reads in
    `child_outside`), and those implicate the PR that added the reading test or changed the file read."""
    import json

    from research.drive import implicated, touches

    d = tmp_path / "run"
    (d / "suites").mkdir(parents=True)
    (d / "steps.jsonl").write_text('{"step": "lean-suites", "status": "failed"}\n')
    record = {"name": "verity-lean-audit", "suite": "tools/verity/lean", "status": "failed", "failed_tests": [],
              "outside": {"tools/verity/check/lean_slot.py": "tools/verity/lean/tests/test_lean_changed.py",
                          "tools/verity/check/slot.py": "*"},
              "child_outside": {"tools/verity/lean/tests/test_cache.py": ["elsewhere/data.txt"]}, "log": None}
    (d / "suites-lean.json").write_text(json.dumps([record]))
    got = implicated(d)
    assert got == {"tools/verity/check/lean_slot.py", "tools/verity/check/slot.py",
                   "tools/verity/lean/tests/test_lean_changed.py", "tools/verity/lean/tests/test_cache.py", "elsewhere/data.txt"}
    assert touches({"tools/verity/lean/tests/test_lean_changed.py", "a.txt"}, got) == {"tools/verity/lean/tests/test_lean_changed.py"}
    (d / "suites-lean.json").write_text(json.dumps([{**record, "outside": {}, "child_outside": {}}]))
    assert implicated(d) is None


def test_what_a_running_check_has_failed_is_read_from_its_steps_and_its_suites_failures(tmp_path):
    import json

    from research.drive import failing

    assert failing(tmp_path) is None
    (tmp_path / "steps.jsonl").write_text('{"step": "pytest", "status": "passed"}\n{"step": "circuit-check", "status": "failed"}\n')
    (tmp_path / "suites-lean.failed.json").write_text(json.dumps([{"name": "repository", "failed_tests": ["a", "b", "c"]}]))
    assert failing(tmp_path) == "step circuit-check; suite repository (a, b ...)"


def test_status_fetches_only_paths_under_the_run_dir_beside_its_records(tmp_path):
    from research import remote

    for bad in ("../x", "/etc/passwd", "a//b", ".fetched", "suites/../x", ""):
        with pytest.raises(remote.Refused, match="paths of files under the run dir"):
            remote.fetch(tmp_path, also=(bad,))


def test_forces_cold_reads_the_paths_that_move_most_circuit_check_keys():
    from research.queue import forces_cold

    assert forces_cold(["verity/ml/circuits/library.py"]) and forces_cold(["uv.lock"]) and forces_cold(["x/y.circuit-check.py"])
    assert forces_cold(["tools/verity/circuit_check/src/circuit_check/cache.py"])
    assert forces_cold(["integrations/vllm/verity_vllm/pipeline/serving_rows.py"])          # else hundreds of targets rerun
    assert not forces_cold(["integrations/vllm/tests/pipeline/test_row.py", "tools/research/src/research/drive.py"])
    assert not forces_cold(["Security/Proofs/X.lean", "verity/ml/circuits/README.md", "verity/ml/circuits/tests/test_x.py"])
    assert not forces_cold([])


def test_train_sh_runs_with_the_queues_credentials_and_git_never_prompts(tmp_path, monkeypatch):
    """train.sh's `git fetch` in the queue's clone, which has no credential helper of its own, mustn't wait on a password at
    the tty, and drive with it."""
    from research.drive import train_launch

    (tmp_path / "tools" / "verity" / "check").mkdir(parents=True)
    (tmp_path / "tools" / "verity" / "check" / "train.sh").write_text(
        'echo "prompt=$GIT_TERMINAL_PROMPT n=$GIT_CONFIG_COUNT $GIT_CONFIG_KEY_1=$GIT_CONFIG_VALUE_1 $GIT_CONFIG_KEY_2=$GIT_CONFIG_VALUE_2"\n'
        'read -r line && echo "stdin: $line"\necho "RUN r20261012-060000-aaaa"\n')
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "safe.directory")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "*")
    ids, tail = train_launch(tmp_path, "c" * 40, "m1", False, auth=("-c", "credential.helper=", "-c", "credential.helper=!h get"))
    assert ids == ["r20261012-060000-aaaa"]
    assert "prompt=0 n=3 credential.helper= credential.helper=!h get" in tail and "stdin:" not in tail


def test_a_conflict_is_told_to_ci_and_the_lane_that_owns_the_pr(world, monkeypatch):
    """An ejection post names the PR's owner too, not only @ci, so the owner doesn't hear of a conflict through a relay.  The
    owner's handle comes from main's registry: the lane the body names, the agent id it carries, the ready mark's author, or
    the agent whose id ends in the branch's suffix."""
    import dataclasses
    import json

    agent = "bc-b8aaadaa-aefd-503e-9863-5eabb3a28c79"
    reg = {"handles": [{"handle": "circuits", "holder": f"the circuits coordinator ({agent})"},
                       {"handle": "infra", "holder": "infra (bc-17cc41f1-6227-5fab-bc64-3fa2f224558b)"}]}
    main = world.change({".agents/skills/using-slack/registry.json": json.dumps(reg)})
    _git(world.dev, "push", "-q", "origin", f"{main}:refs/heads/main")
    h = Harness(world, monkeypatch)
    sent = []
    h.d.post = lambda text, to: sent.append((text, to)) or True
    world.gh.put(1, world.change({"a.txt": "one\n"}))
    world.gh.put(2, world.change({"a.txt": "two\n"}), body="Lane `catalog`.")
    world.gh.prs[2] = dataclasses.replace(world.gh.prs[2], branch="cursor/catalog-8c79")
    h.tick()
    assert [to for _, to in sent] == [("@ci", "@circuits")]
    assert sent[0][0].startswith("@ci @circuits queue: #2 at ")
    q = h.q
    for pr, want in ((dataclasses.replace(world.gh.prs[2], body=f"[agent]({agent})", branch=""), "circuits"),
                     (dataclasses.replace(world.gh.prs[2], body="Lane `infra`.", branch=""), "infra"),
                     (dataclasses.replace(world.gh.prs[2], branch="cursor/x-0000"), None)):
        assert q.handle(pr) == want


def test_a_pr_waiting_on_a_contained_pr_that_didnt_merge_is_told_once_per_head(world, monkeypatch):
    """A PR that sits off `next` behind a contained PR that didn't merge is told so, rather than waiting unseen."""
    h = Harness(world, monkeypatch)
    sent = []
    h.d.post = lambda text, to: sent.append((text, to)) or True
    world.gh.put(1, world.change({"a.txt": "one\n"}))
    two = world.change({"a.txt": "two\n"})
    world.gh.put(2, two)
    world.gh.put(3, world.change({"b.txt": "three\n"}, base=two))
    h.tick()
    h.tick()
    told = [text for text, _ in sent if "#3 at" in text]
    assert told == [f"@ci queue: #3 at {world.gh.prs[3].head[:8]} is out of `next`: contains #2, which didn't merge"]


def test_a_notice_reaches_the_merge_thread_without_naming_its_sender_and_a_refused_post_is_retried(world, monkeypatch):
    """`research msg send` refuses a `--to` naming the sender, so drive posting as ci to @ci would post no notice at all. The
    sender is also selectable (`queue eject --by lean`), so its handle, not always @ci, must be dropped."""
    from research import drive

    argv = []
    monkeypatch.setattr(drive.subprocess, "run", lambda cmd, **kw: argv.append(cmd) or
                        drive.subprocess.CompletedProcess(cmd, 0))
    assert drive.merge_thread_post("x", ("@ci", "@memory-accounting"))
    assert argv[-1][argv[-1].index("--as") + 1] == "ci"
    assert argv[-1][argv[-1].index("--to"):argv[-1].index("--quiet")] == ["--to", "@memory-accounting"]
    assert drive.merge_thread_post("x", ("@ci",)) and "--to" not in argv[-1]
    assert drive.merge_thread_post("x", ("@ci", "@lean"), sender="lean")
    assert argv[-1][argv[-1].index("--as") + 1] == "lean"
    assert argv[-1][argv[-1].index("--to"):argv[-1].index("--quiet")] == ["--to", "@ci"]
    assert all("--report" in cmd for cmd in argv)                       # a report, held to no post length

    h = Harness(world, monkeypatch)
    logged = []
    h.q.log = logged.append
    ok = [False]
    h.d.post = lambda text, to: ok[0]
    state = {"told": []}
    h.d.refuse(state, "k", "#1 conflicts", "circuits")
    assert state["told"] == [] and "could not post" in logged[-1]
    ok[0] = True
    h.d.refuse(state, "k", "#1 conflicts", "circuits")
    assert state["told"] == ["k"] and logged[-1].startswith("research queue drive: told @ci @circuits")


@pytest.mark.parametrize("to, sender", [(("@ci",), "ci"), (("@ci", "@circuits"), "ci"), (("@ci", "@lean"), "lean")])
def test_every_notice_drive_sends_passes_msg_sends_own_refusals(monkeypatch, to, sender):
    """A train's failure for @ci alone, starting `@ci queue:`, is refused by `msg send` as a post that wakes only its author.  A
    stand-in for `msg send` would hide that refusal, so here each post goes through `msg send` itself."""
    from research import drive, slack

    monkeypatch.setattr(slack, "acting_as", lambda ns, reg, name: slack.name_entry(ns, reg, name))

    def run(cmd, **kw):
        assert cmd[1:5] == ["-m", "research", "msg", "send"]
        return drive.subprocess.CompletedProcess(cmd, slack.main([*cmd[4:], "--dry-run"]))

    monkeypatch.setattr(drive.subprocess, "run", run)
    assert drive.merge_thread_post(f"{' '.join(to)} queue: b22a0c54d29e on `next` (#1 #2) failed its check", to, sender=sender)


def _steps_run(tmp_path, name: str, steps: dict[str, str]):
    import json

    d = tmp_path / name
    d.mkdir(parents=True)
    (d / "steps.jsonl").write_text("".join(json.dumps({"step": k, "status": v}) + "\n" for k, v in steps.items()))
    return d


def test_a_collection_error_names_its_test_module(tmp_path):
    """A suite that fails collecting one PR's test module (one reading a vector only another PR adds) names that module, so
    the fallback on the suite's inputs doesn't eject every PR touching them instead of the one."""
    import json

    from research.drive import implicated

    d = tmp_path / "run"
    (d / "suites").mkdir(parents=True)
    (d / "steps.jsonl").write_text('{"step": "pytest", "status": "failed"}\n')
    (d / "suites.json").write_text(json.dumps([{"name": "filler", "suite": "benchmarks/pouw/filler", "status": "failed",
                                                "failed_tests": [], "log": "/x/suites/filler.log"}]))
    (d / "suites" / "filler.log").write_text("ERROR tests/test_capacity_of.py - KeyError: 'VF3r-16'\n1 error in 2.1s\n")
    assert "benchmarks/pouw/filler/tests/test_capacity_of.py" in implicated(d)


def test_lean_errors_reads_the_audits_orphan_and_layers_failures(tmp_path):
    """An orphan file or an out-of-layer import names a file drive matches, so it ejects the one PR rather than resending the
    train."""
    from research.drive import lean_errors

    log = ("AUDIT /w/lean-audit-scratch-1-x/verity/core/service/flock: FAIL  14588 declarations in 159 modules\n"
           "  orphan: Flock/Test/KnownRead.lean is not imported by the roots (Flock, FlockVerify) nor under `exempt`: nothing builds"
           " or audits it\n  note: 267 definitions of the spec (VStar) are read by no guarantee\n"
           "AUDIT /w/lean-audit-scratch-1-x/verity/core/protocols/pouw/lean/check: FAIL  944 declarations in 13 modules\n"
           "  layers: PouwCheck.Credit imports Flock.Program, outside PouwCheck.Credit's layer (Std.Data.HashMap)\n")
    assert lean_errors(log) == {"verity/core/service/flock/Flock/Test/KnownRead.lean",
                                "verity/core/protocols/pouw/lean/check/PouwCheck/Credit.lean"}


def test_lean_errors_reads_lakes_list_of_failed_modules(tmp_path):
    """With the build's error lines scrolled out of the step's log, drive still finds the Lean files from Lake's list of
    failed modules, rather than falling back on every PR touching Lean."""
    from research.drive import lean_errors

    log = ("AUDIT /w/lean-audit-scratch-1-x/Security/Proofs: FAIL; built WITHOUT a sandbox\n  build: exit 1\n"
           "    Some required targets logged failures:\n    - Proofs.Flock.Claims.NumbersRadius\n    error: build failed\n"
           "AUDIT /w/lean-audit-scratch-1-x/Security: FAIL\n    Some required targets logged failures:\n"
           "    - Definitions.Claims.Sampling\n")
    assert lean_errors(log) == {"Security/Proofs/Flock/Claims/NumbersRadius.lean", "Security/Definitions/Claims/Sampling.lean"}


def test_a_lean_check_whose_replay_was_killed_is_a_machines_fault(tmp_path):
    """A Security/Proofs replay OOM-killed (exit 137) fails lean-audit with rc 1; that is the machine's fault, and the fallback
    on Lean sources mustn't eject PRs whose heads passed."""
    from research.drive import killed

    d = tmp_path / "run"
    d.mkdir()
    (d / "steps.jsonl").write_text('{"step": "lean-audit", "status": "failed", "rc": 1, "log": "lean-audit.log"}\n')
    (d / "lean-audit.log").write_text("AUDIT /x/lean-audit-scratch-1/Security/Proofs: FAIL\n  replay: killed (exit 137; out of "
                                      "memory?), which is not a kernel verdict: rerun it alone\n")
    assert killed(d)
    (d / "lean-audit.log").write_text("AUDIT /x/lean-audit-scratch-1/Security/Proofs: FAIL\n    error: build failed\n")
    assert not killed(d)


def test_a_circuit_check_whose_only_failures_are_killed_targets_is_a_machines_fault(tmp_path):
    """Targets OOM-killed (as PoUS's SHA-512 targets can be, at 150 GiB or so) fail circuit-check with rc 1; that is the
    machine's fault, and drive mustn't blame circuit-check's inputs and revert every landing since the last verified `main`."""
    from research.drive import killed

    d = tmp_path / "run"
    d.mkdir()
    step = '{"step": "circuit-check", "status": "failed", "rc": 1, "log": "circuit-check.log"}\n'
    (d / "steps.jsonl").write_text(step)
    died = ("FAIL circuit-check/killed PousSha512Head_v1: its process died (signal 9: the OOM killer?) at a peak of 164.16 GiB\n"
            "FAIL circuit-check/killed PousSha512Mid_v1{M=1}: its process died (signal 9: the OOM killer?) at a peak of 164.16 GiB\n")
    tail = "circuit-check: 5211 target(s), 2 new failure(s), 3 known\nwarn dead-gates/ir: 258100 over 309 target(s)\n"
    (d / "circuit-check.log").write_text(died + tail)
    assert killed(d)
    (d / "circuit-check.log").write_text(died + "FAIL partition/unchecked PousSha512Ctr_v1: an output bit is free\n" + tail)
    assert not killed(d), "a real failure beside the killed ones is the code's"
    (d / "circuit-check.log").write_text(died + "FAIL circuit_check.known lists Foo, which no longer fails: drop it\n" + tail)
    assert not killed(d)
    (d / "circuit-check.log").write_text(died + tail)
    (d / "steps.jsonl").write_text(step + '{"step": "pytest", "status": "failed", "rc": 1, "log": "pytest.log"}\n')
    assert not killed(d), "another failed step is judged on its own"


def test_drive_sweeps_the_repositorys_branches_once_a_day(world, monkeypatch):
    """Something deletes landed PRs' branches and idle ones, or the repository's branches pile up by the thousand."""
    from research.drive import BRANCH_SWEEP_S

    h = Harness(world, monkeypatch)
    calls = []
    monkeypatch.setattr(merge, "sweep_branches", lambda *a, **k: calls.append(a) or (2, 1))
    h.tick()
    h.tick()
    assert len(calls) == 1 and any("branch sweep: 2 deleted (on main), 1 archived" in x for x in world.logs)
    h.clock += BRANCH_SWEEP_S + 1
    h.tick()
    assert len(calls) == 2


def test_a_branch_sweep_at_its_limit_sweeps_again_the_next_tick(world, monkeypatch):
    """A sweep pushes at most `SWEEP_LIMIT` deletions; the rest go on the ticks after, not a day later."""
    from research.merge import SWEEP_LIMIT

    h = Harness(world, monkeypatch)
    calls = []
    monkeypatch.setattr(merge, "sweep_branches", lambda *a, **k: calls.append(a) or ((SWEEP_LIMIT, 0) if len(calls) == 1 else (3, 0)))
    h.tick()
    assert any(f"at its limit of {SWEEP_LIMIT}, so the next tick sweeps again" in x for x in world.logs)
    h.tick()
    h.tick()
    assert len(calls) == 2


def _live_run(tmp_path, rid: str, commit: str, records: dict[str, str | None]):
    """A check still running, as `merge.fetch_run` reads its run dir from its node: (that dir, a fetch of it, the fetches made).
    Its Lean report names a regenerated lock for each package of ``records``, holding its text (None: the report names it, and
    the file isn't there)."""
    import json

    d = tmp_path / "live" / rid
    (d / "lean-audit").mkdir(parents=True)
    regen = {}
    for pkg, text in records.items():
        rel = f"lean-audit/audit-{pkg.replace('/', '-')}/lean-audit.json"
        regen[pkg] = f"/workspace/research/runs/{rid}/{rel}"
        if text is not None:
            (d / rel).parent.mkdir(parents=True, exist_ok=True)
            (d / rel).write_text(text)
    (d / "lean-audit" / "audit.json").write_text(json.dumps({"passed": True, "regenerated": regen}))
    fetched: list[tuple[str, list[str]]] = []

    def fetch(r, names):
        fetched.append((r, names))
        return (d, {"dir": str(d), "tree_sha": commit}) if r == rid else None
    return d, fetch, fetched


def test_the_after_landing_launch_gives_train_sh_no_cache_with_circuit_checks_cache(tmp_path):
    """Drive's after-landing run reuses no suite or Lean pass (`after_landing_no_cache`) but circuit-check's per-target ones: a
    cold `circuit-check --all` (~4 h) would outlast every after-landing run, so `main` would never be verified."""
    from research.drive import train_launch

    (tmp_path / "tools" / "verity" / "check").mkdir(parents=True)
    (tmp_path / "tools" / "verity" / "check" / "train.sh").write_text('echo "argv: $*"\necho "RUN r20261027-120000-aaaa"\n')
    extra = {"agreement": False, "no_cache": True, "circuit_cache": True, "no_regenerate": True}    # `start_postland`'s
    ids, tail = train_launch(tmp_path, "c" * 40, "m1", extra.pop("agreement"), deferred="pytest,circuit-check", **extra)
    assert ids == ["r20261027-120000-aaaa"]
    assert f"argv: {'c' * 40} m1 --deferred pytest,circuit-check --no-cache --circuit-cache --no-regenerate" in tail, tail
