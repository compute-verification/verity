"""`research.landing`, what the verified queue's loop reads of a run and a tree to judge a candidate: the files a failed run
implicates and whether its machine killed it, what a running check has failed so far, the run beside a candidate's check
(`train.sh`'s launch, its credentials and check slot), the jobs `main` waits for, the runs' states and evidence read several at
a time, the merge-thread post, and the notice to the owners of PRs stacked on a landed one; on `test_queue`'s throwaway origin
with the launch, the runs' states and the merge-thread post injected.  `Harness` is the `Landing` `test_vqueue_slim` builds
on."""
from __future__ import annotations

import ast
import dataclasses
import json
import subprocess
import threading
from pathlib import Path

import pytest

from research import landing, merge
from research.landing import JOBS, SKIPPABLE, SLIM_JUDGED, SLIM_STEPS, Landing, gating
from research.policy import SCHEMA as POLICY_SCHEMA
from research.queue import PR, Member
from test_queue import World, _git

SRC = Path(__file__).resolve().parents[1] / "src" / "research"
#: the modules of the verified queue's loop, which runs on the control pod, whose TMPDIR is an NFS volume
LOOP_MODULES = ("queue.py", "landing.py", "vqueue.py", "vqueue_loop.py", "vqueue_slim.py")


class Harness:
    """A `Landing` on ``world`` whose launches of a run beside a check are recorded (`posted`) as `r-post-N`, each running until a
    test sets its state in `run_state`; its posts, cancels, evidence and clock injected."""

    def __init__(self, world: World, monkeypatch, **kw):
        self.world, self.posted, self.posts, self.after, self.cancelled = world, [], [], [], []
        self.run_state: dict[str, tuple] = {}
        self.clock = 1_000_000.0
        monkeypatch.setattr(merge, "caught_up", lambda repo, remote, tool, since="", log=print, runs=None:
                            self.after.append((since, runs)))
        monkeypatch.setattr(merge, "_sweep_after", lambda *a, **k: None)
        self.q = world.queue(**kw)
        self.q.now = lambda: self.clock
        self.evidence: list = []
        self.d = Landing(self.q, on="m1", launch=self.launch_post, states=lambda repo, ids: {i: self.run_state[i] for i in ids},
                         post=lambda text, to=(): self.posts.append(text) or True, now=lambda: self.clock,
                         cancel=lambda repo, ids, reason: self.cancelled.extend(ids), evidence=lambda repo, ids: self.evidence)

    def launch_post(self, repo, commit, machine, steps, **kw):
        rid = f"r-post-{len(self.posted)}"
        self.posted.append((commit, steps, kw))
        self.run_state[rid] = ("running", None, None)
        return [rid], ""


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


#: a ci.toml with a job for each of the check's jobs, and its policy taking circuit-check off the merge gate as in the tree
CI_TOML = 'schema = "ci/v0"\n' + "".join(f'[[job]]\nname = "{s}"\nowner = "ci"\ngroup = 1\n' for s in JOBS)

POLICY_TOML = 'schema = "policy/v0"\noff_gate = ["circuit-check"]\n'

#: the files `_move_main` writes for a tree declaring `CI_TOML` under `POLICY_TOML`
GATED_TREE = {"tools/verity/check/ci.toml": CI_TOML, "tools/verity/check/policy.toml": POLICY_TOML}

LIVE, EXPIRED = "2099-01-01T00:00:00Z", "1970-01-01T00:00:00Z"   # the harness's clock reads 1970-01-12


def _policy(*allow: tuple[str, str, str], off_gate: tuple[str, ...] = ("circuit-check",)) -> str:
    """policy.toml's text: ``off_gate`` and an allowance (id, owner, expires) for each of ``allow``."""
    return (f'schema = "{POLICY_SCHEMA}"\noff_gate = {json.dumps(list(off_gate))}\n'
            + "".join(f'[[allow]]\nid = "{i}"\nowner = "{o}"\nexpires = "{e}"\nreason = "red; its fix is landing"\n'
                      for i, o, e in allow))


#: the preflights a check runs before any step (check.py's `preflight`), which the run beside the executor's check judges
PREFLIGHTS = tuple(s for s in SLIM_JUDGED if s not in SLIM_STEPS)


def _main(world: World) -> str:
    return _git(world.origin, "rev-parse", "main")


def _move_main(world: World, files: dict[str, str]) -> None:
    """A commit of ``files`` onto `main`."""
    _git(world.dev, "push", "-q", "origin", f"{world.change(files)}:refs/heads/main")


def _slim(h: Harness, tmp_path, **logs: str) -> Path:
    """The newest run beside the executor's check (`r-post-N`, of `SLIM_STEPS` but one it skips) ended: each preflight and each
    of its steps passed, but each ``logs`` names (``ratchet_gate=LOG``), which failed with that log, its exit 1 then; after a
    failed preflight it ran no step, as check.py starts none."""
    rid, failed = f"r-post-{len(h.posted) - 1}", {k.replace("_", "-"): v for k, v in logs.items()}
    d = tmp_path / rid
    d.mkdir(parents=True)
    rows = []
    for s in (*PREFLIGHTS, *(() if failed.keys() & set(PREFLIGHTS) else h.posted[-1][1].split(","))):
        rows.append({"step": s, "status": "failed", "rc": 1, "log": f"{s}.log"} if s in failed else {"step": s, "status": "passed"})
        if s in failed:
            (d / f"{s}.log").write_text(failed[s])
    (d / "steps.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    h.run_state[rid], h.evidence = ("done", 1 if failed else 0, None), [d]
    return d


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



# -- what a run implicates


def test_a_lean_audit_failure_names_the_lean_files_its_build_errors_are_in(tmp_path):
    """A train that fails in lean-audit isn't told as naming no file, to be attributed by hand."""
    from research.landing import implicated, lean_errors

    assert lean_errors(LEAN_AUDIT_LOG) == {"Security/Proofs/Flock/Claims/NumbersRadius.lean",
                                           "Security/Definitions/Claims/Sampling.lean"}
    assert implicated(_lean_run(tmp_path)) == lean_errors(LEAN_AUDIT_LOG)
    assert implicated(_lean_run(tmp_path / "b", "AUDIT /x/lean-audit-scratch-1/Security: FAIL\n    error: build failed\n")) is None


def test_a_failed_pytest_step_points_at_its_tests_and_the_paths_its_failures_name(tmp_path):
    from research.landing import implicated

    assert implicated(_lint_run(tmp_path)) == {"tests/test_repository.py", "tools/research/src/research/store/README.md"}
    assert implicated(_lint_run(tmp_path / "b", "error: lock is stale\n")) is None


def test_a_failure_outside_the_suites_or_without_a_failing_test_points_at_no_file(tmp_path):
    from research.landing import implicated, touches

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

    from research.landing import implicated, touches

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


def test_a_collection_error_names_its_test_module(tmp_path):
    """A suite that fails collecting one PR's test module (one reading a vector only another PR adds) names that module, so
    the fallback on the suite's inputs doesn't eject every PR touching them instead of the one."""
    import json

    from research.landing import implicated

    d = tmp_path / "run"
    (d / "suites").mkdir(parents=True)
    (d / "steps.jsonl").write_text('{"step": "pytest", "status": "failed"}\n')
    (d / "suites.json").write_text(json.dumps([{"name": "filler", "suite": "benchmarks/pouw/filler", "status": "failed",
                                                "failed_tests": [], "log": "/x/suites/filler.log"}]))
    (d / "suites" / "filler.log").write_text("ERROR tests/test_capacity_of.py - KeyError: 'VF3r-16'\n1 error in 2.1s\n")
    assert "benchmarks/pouw/filler/tests/test_capacity_of.py" in implicated(d)


def test_lean_errors_reads_the_audits_orphan_and_layers_failures(tmp_path):
    """An orphan file or an out-of-layer import names a file the loop matches, so it ejects the one PR rather than every PR
    touching Lean."""
    from research.landing import lean_errors

    flock = "verity/core/service/flock"
    log = (f"AUDIT /w/lean-audit-scratch-1-x/{flock}: FAIL  14588 declarations in 159 modules\n"
           "  orphan: Flock/Test/KnownRead.lean is not imported by the roots (Flock, FlockVerify) nor under `exempt`: nothing builds"
           " or audits it\n  note: 267 definitions of the spec (VStar) are read by no guarantee\n"
           "AUDIT /w/lean-audit-scratch-1-x/verity/core/protocols/pouw/lean/check: FAIL  944 declarations in 13 modules\n"
           "  layers: PouwCheck.Credit imports Flock.Program, outside PouwCheck.Credit's layer (Std.Data.HashMap)\n")
    assert lean_errors(log) == {f"{flock}/Flock/Test/KnownRead.lean",
                                "verity/core/protocols/pouw/lean/check/PouwCheck/Credit.lean"}


def test_lean_errors_reads_lakes_list_of_failed_modules(tmp_path):
    """With the build's error lines scrolled out of the step's log, the loop still finds the Lean files from Lake's list of
    failed modules, rather than falling back on every PR touching Lean."""
    from research.landing import lean_errors

    log = ("AUDIT /w/lean-audit-scratch-1-x/Security/Proofs: FAIL; built WITHOUT a sandbox\n  build: exit 1\n"
           "    Some required targets logged failures:\n    - Proofs.Flock.Claims.NumbersRadius\n    error: build failed\n"
           "AUDIT /w/lean-audit-scratch-1-x/Security: FAIL\n    Some required targets logged failures:\n"
           "    - Definitions.Claims.Sampling\n")
    assert lean_errors(log) == {"Security/Proofs/Flock/Claims/NumbersRadius.lean", "Security/Definitions/Claims/Sampling.lean"}


def test_a_lean_audit_whose_replay_was_killed_is_a_machines_fault(tmp_path):
    """A Security/Proofs replay OOM-killed (exit 137) fails lean-audit with rc 1; that is the machine's fault, and the fallback
    on Lean sources mustn't revert landings that broke nothing."""
    from research.landing import killed

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
    machine's fault, and the loop mustn't blame circuit-check's inputs and eject every PR of the candidate."""
    from research.landing import killed

    d = tmp_path / "run"
    d.mkdir()
    step = '{"step": "circuit-check", "status": "failed", "rc": 1, "log": "circuit-check.log"}\n'
    (d / "steps.jsonl").write_text(step)
    died = ("FAIL circuit-check/killed PousSha512Head_v1: its process died (signal 9: the OOM killer?) at a peak of 164.16 GiB\n"
            "FAIL circuit-check/killed PousSha512Mid_v1{M=1}: its process died (signal 9: the OOM killer?) at a peak of 164.16 GiB\n")
    tail = "circuit-check: 5211 target(s), 2 failure(s), 0 unchecked\nwarn dead-gates/ir: 258100 over 309 target(s)\n"
    (d / "circuit-check.log").write_text(died + tail)
    assert killed(d)
    (d / "circuit-check.log").write_text(died + "FAIL partition/unchecked PousSha512Ctr_v1: an output bit is free\n" + tail)
    assert not killed(d), "a real failure beside the killed ones is the code's"
    (d / "circuit-check.log").write_text(died + tail)
    (d / "steps.jsonl").write_text(step + '{"step": "pytest", "status": "failed", "rc": 1, "log": "pytest.log"}\n')
    assert not killed(d), "another failed step is judged on its own"


def test_what_a_running_check_has_failed_is_read_from_its_steps_and_its_suites_failures(tmp_path):
    import json

    from research.landing import failing

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
    assert not forces_cold(["integrations/vllm/tests/pipeline/test_row.py", "tools/research/src/research/landing.py"])
    assert not forces_cold(["Security/Proofs/X.lean", "verity/ml/circuits/README.md", "verity/ml/circuits/tests/test_x.py"])
    assert not forces_cold([])



# -- the run beside a candidate's check


def test_train_sh_runs_with_the_queues_credentials_and_git_never_prompts(tmp_path, monkeypatch):
    """train.sh's `git fetch` in the queue's clone, which has no credential helper of its own, mustn't wait on a password at
    the tty, and the loop with it."""
    from research.landing import train_launch

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


def test_train_launch_gives_train_sh_the_steps_and_flags_it_is_given(tmp_path):
    from research.landing import train_launch

    (tmp_path / "tools" / "verity" / "check").mkdir(parents=True)
    (tmp_path / "tools" / "verity" / "check" / "train.sh").write_text('echo "argv: $*"\necho "RUN r20261027-120000-aaaa"\n')
    extra = {"agreement": False, "no_cache": True, "circuit_cache": True, "no_regenerate": True}
    ids, tail = train_launch(tmp_path, "c" * 40, "m1", extra.pop("agreement"), deferred="pytest,circuit-check", **extra)
    assert ids == ["r20261027-120000-aaaa"]
    assert f"argv: {'c' * 40} m1 --deferred pytest,circuit-check --no-cache --circuit-cache --no-regenerate" in tail, tail
    ids, tail = train_launch(tmp_path, "c" * 40, "m1", False, deferred="ratchet-gate", base="b" * 40, ratchet_base="a" * 40)
    assert f"argv: {'c' * 40} m1 --deferred ratchet-gate --base {'b' * 40} --ratchet-base {'a' * 40}" in tail, tail


def test_the_run_beside_a_candidates_check_takes_a_check_slot_ahead_of_unranked_priority_checks(tmp_path, monkeypatch):
    """The run beside a candidate's check asks for its check slot at `SLOT_RANK` (CHECK_SLOT_RANK, which check.py passes on
    as `slot.py --rank`), whatever the loop's own environment says: a ranked priority waiter takes the next free slot ahead
    of the unranked, so a candidate never waits on a slot for a `train.sh` launched by hand."""
    import contextlib
    import os
    import tempfile
    import types

    from research.landing import SLOT_RANK, launched

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setenv("CHECK_SLOT_RANK", "7")
    repo = tmp_path / "clone"
    (repo / "tools" / "verity" / "check").mkdir(parents=True)
    (repo / "tools" / "verity" / "check" / "train.sh").write_text(
        'echo "rank=[$CHECK_SLOT_RANK]"\necho "RUN r20261103-140000-aaaa"\n')
    d = Landing(types.SimpleNamespace(auth=(), workdir=repo), on="m1")

    def tail(job: dict) -> str:
        with contextlib.suppress(ChildProcessError):                     # its end, not a clock
            os.waitpid(job["pid"], 0)
        return launched(job)[1]

    assert f"rank=[{SLOT_RANK}]" in tail(d.launch(repo, "c" * 40, "m1", "lean-agreement,ratchet-gate", base="b" * 40))


def test_main_waits_for_the_steps_on_the_gate_of_the_commits_own_ci_toml(world, monkeypatch):
    """`main` waits at a commit for every job of the check but those a run skips by name and the jobs that commit's own policy
    takes off the merge gate (circuit-check); for every such job when its ci.toml or policy can't be read."""
    h = Harness(world, monkeypatch)
    h.q._clone()
    h.q.fetch()
    every = [s for s in JOBS if s not in SKIPPABLE]
    assert gating(h.q.workdir, _main(world)) == every
    _move_main(world, GATED_TREE)
    h.q.fetch()
    assert gating(h.q.workdir, _main(world)) == [s for s in every if s != "circuit-check"]
    _move_main(world, {"tools/verity/check/policy.toml": "not toml ["})
    h.q.fetch()
    assert gating(h.q.workdir, _main(world)) == every
    _move_main(world, {"tools/verity/check/policy.toml": POLICY_TOML, "tools/verity/check/ci.toml": "not toml ["})
    h.q.fetch()
    assert gating(h.q.workdir, _main(world)) == every



# -- reading runs


def test_run_states_reads_runs_in_parallel_and_keeps_each_state_with_its_run(tmp_path, monkeypatch):
    """Reading about 40 runs one `research status` (an ssh round trip) at a time took a tick 33 min. Several are read at once, and each answer stays with the run it is about, whatever order they finish in."""
    seen: set[int] = set()
    lock = threading.Lock()

    def run(argv, **kw):
        rid = argv[argv.index("status") + 1]
        with lock:
            seen.add(threading.get_ident())
        state = {"r1": "running", "r2": "done", "r3": "failed"}.get(rid)
        if state is None:
            return subprocess.CompletedProcess(argv, 1, "", "no such run")
        d = tmp_path / rid
        d.mkdir(exist_ok=True)
        rc = 0 if state == "done" else (1 if state == "failed" else None)
        return subprocess.CompletedProcess(argv, 0, json.dumps({"runs": [{"state": state, "rc": rc, "dir": str(d)}]}), "")

    monkeypatch.setattr(landing.subprocess, "run", run)
    ids = ["r3", "r1", "nope", "r2"]
    out = landing.run_states(tmp_path, ids)
    assert list(out) == ids
    assert out["r1"][0] == "running" and out["r2"][:2] == ("done", 0) and out["r3"][:2] == ("failed", 1)
    assert out["nope"] == ("unknown", None, None, None)
    assert seen and threading.get_ident() not in seen, "read on worker threads, not one after another on the caller's"


def test_each_keeps_order_and_runs_a_single_item_inline():
    assert landing._each(lambda x: x * 2, [3, 1, 2]) == [6, 2, 4]
    me = threading.get_ident()
    assert landing._each(lambda _: threading.get_ident(), ["only"]) == [me]
    assert landing._each(lambda x: x, []) == []


def test_evidence_drops_runs_it_cannot_fetch_and_keeps_the_rest_in_order(tmp_path, monkeypatch):
    got = {"r1": tmp_path / "r1", "r3": tmp_path / "r3"}
    for d in got.values():
        d.mkdir()
    monkeypatch.setattr(landing, "_fetch", lambda repo, rid, names: got.get(rid))
    assert landing.evidence(Path("."), ["r3", "r2", "r1"]) == [got["r3"], got["r1"]]



# -- what the loop tells


def test_a_notice_reaches_the_merge_thread_without_naming_its_sender(monkeypatch):
    """`research msg send` refuses a `--to` naming the sender, so the loop posting to its own handle would post no notice at
    all.  The sender is also selectable (`queue eject --by lean`), so its handle, not always the loop's own, must be dropped."""

    argv = []
    monkeypatch.setattr(landing.subprocess, "run", lambda cmd, **kw: argv.append(cmd) or
                        landing.subprocess.CompletedProcess(cmd, 0))
    assert landing.merge_thread_post("x", ("@ci", "@memory-accounting"))
    assert argv[-1][argv[-1].index("--as") + 1] == "ci"
    assert argv[-1][argv[-1].index("--to"):argv[-1].index("--quiet")] == ["--to", "@memory-accounting"]
    assert landing.merge_thread_post("x", ("@ci",)) and "--to" not in argv[-1]
    assert landing.merge_thread_post("x", ("@ci", "@lean"), sender="lean")
    assert argv[-1][argv[-1].index("--as") + 1] == "lean"
    assert argv[-1][argv[-1].index("--to"):argv[-1].index("--quiet")] == ["--to", "@ci"]
    assert all("--report" in cmd for cmd in argv)                       # a report, held to no post length


@pytest.mark.parametrize("to, sender", [(("@lean",), "lean"), (("@lean", "@circuits"), "lean"), (("@lean", "@infra"), "infra")])
def test_every_notice_the_loop_sends_passes_msg_sends_own_refusals(monkeypatch, to, sender):
    """A notice for its sender alone, starting `@SENDER queue:`, is refused by `msg send` as a post that wakes only its author.  A
    stand-in for `msg send` would hide that refusal, so here each post goes through `msg send` itself."""
    from research import slack

    monkeypatch.setattr(slack, "acting_as", lambda ns, reg, name: slack.name_entry(ns, reg, name))

    def run(cmd, **kw):
        assert cmd[1:5] == ["-m", "research", "msg", "send"]
        return landing.subprocess.CompletedProcess(cmd, slack.main([*cmd[4:], "--dry-run"]))

    monkeypatch.setattr(landing.subprocess, "run", run)
    assert landing.merge_thread_post(f"{' '.join(to)} queue: b22a0c54d29e on `next` (#1 #2) failed its check", to, sender=sender)


def test_the_owner_of_each_unmarked_pr_stacked_on_a_landed_one_hears_once_to_merge_main(world, monkeypatch):
    """A PR whose branch contains a landed PR's head, based on `main` or on that PR's branch, needs `main`
    merged in; one marked ready is built as it is, and a merge would cost it its mark.  One based on another branch
    that carries the landed PR (a long-lived one) takes `main` through that branch, and isn't told."""
    gh = world.gh
    gh.put(1, h1 := world.change({"a.txt": "1\n"}))
    gh.prs[1] = dataclasses.replace(gh.prs[1], branch="cursor/w-1")
    gh.put(2, world.change({"b.txt": "2\n"}, base=h1), ready=False)                                  # stacked, unmarked
    gh.put(3, world.change({"c.txt": "3\n"}, base=h1))                                                # stacked, marked
    gh.prs[4] = PR(4, world.change({"d.txt": "4\n"}, base=h1), "change 4", "Lane `lane-4`.", base="cursor/w-1",
                   branch="cursor/w-4")                                                               # on #1's branch
    _git(world.origin, "update-ref", "refs/pull/4/head", gh.prs[4].head)
    gh.put(5, world.change({"e.txt": "5\n"}), ready=False)                                           # not stacked
    gh.prs[6] = PR(6, world.change({"f.txt": "6\n"}, base=h1), "change 6", "Lane `lane-6`.", base="remote",
                   branch="cursor/w-6")                                                               # on a long-lived branch
    _git(world.origin, "update-ref", "refs/pull/6/head", gh.prs[6].head)
    h = Harness(world, monkeypatch)
    h.q.handle = lambda pr: pr.owner
    _git(world.dev, "push", "-q", "origin", f"{h1}:refs/heads/main")
    h.q._clone()
    h.q.fetch()
    tip = _main(world)
    state = {"told": []}
    h.d._stacked(state, [Member(1, h1, "lane-1", tip)], tip)
    assert sorted(h.posts) == [f"@lane-2 queue: #2's base #1 landed in main at {tip[:12]}; merge main into #2",
                               f"@lane-4 queue: #4's base #1 landed in main at {tip[:12]}; merge main into #4"]
    assert set(state["told"]) == {"stacked:2<1", "stacked:4<1"}
    h.posts.clear()
    h.d._stacked(state, [Member(1, h1, "lane-1", tip)], tip)
    assert h.posts == [], "once per (PR, base)"



# -- the loop's clone and temporary directories


def test_the_queues_clone_gets_the_record_drivers_it_lacks_and_plain_git_then_merges_two_records_with_merge_py(world, monkeypatch):
    """A queue clone seeded from a copy without the drivers (or with another value) gets every `RECORD_DRIVER` key when the
    queue brings it to the remote's state (`Queue._clone`), logged once, and a later one only reads them; then a git without `git_env`'s settings merges two PRs' lean-audit.json
    records in that clone through merge.py rather than line by line."""
    import json
    import os
    from pathlib import Path

    from research.clone import RECORD_DRIVER

    lean = Path(__file__).resolve().parents[2] / "verity" / "lean"
    tools = {f"tools/verity/lean/{f}": (lean / f).read_text() for f in ("merge.py", "check.py")}
    base = world.change({**tools, ".gitattributes": "lean-audit.json merge=lean-audit\n"})
    _git(world.dev, "push", "-q", "origin", f"{base}:refs/heads/main")
    wd = world.tmp / "q" / "verity"
    _git(world.tmp, "clone", "-q", str(world.origin), str(wd))
    _git(wd, "config", "--local", "merge.lean-caps.driver", "true")
    h = Harness(world, monkeypatch)
    assert h.q.workdir == wd

    def said() -> list[str]:
        return [x for x in world.logs if "set the record merge drivers" in x]

    h.q._clone()
    (line,) = said()
    assert line.endswith(", ".join(RECORD_DRIVER)), line
    assert {k: _git(wd, "config", "--local", "--get", k) for k in RECORD_DRIVER} == RECORD_DRIVER
    h.q._clone()
    assert said() == [line]

    rec = json.loads((world.dev / "pkg" / "lean-audit.json").read_text())
    a, b = (world.change({"pkg/lean-audit.json": json.dumps({**rec, "guarantees": {**rec["guarantees"], g: "1"}})}, base=base)
            for g in "AB")
    for k in [k for k in os.environ if k.startswith("GIT_CONFIG")]:
        monkeypatch.delenv(k)
    _git(wd, "fetch", "-q", "origin", f"refs/dev/{a}", f"refs/dev/{b}")
    tree = _git(wd, f"--attr-source={base}", "merge-tree", "--write-tree", "--no-messages", a, b)
    assert set(json.loads(_git(wd, "show", f"{tree}:pkg/lean-audit.json"))["guarantees"]) == {"T", "A", "B"}


def _unguarded(path: Path) -> list[int]:
    out = []
    for node in ast.walk(ast.parse(path.read_text())):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "TemporaryDirectory"
                and not any(k.arg == "ignore_cleanup_errors" and isinstance(k.value, ast.Constant) and k.value.value is True
                            for k in node.keywords)):
            out.append(node.lineno)
    return out


def test_every_temporary_directory_in_the_loops_code_ignores_cleanup_errors():
    """The loop runs on the control pod, whose TMPDIR is an NFS volume: removing a file a just-ended subprocess held open leaves
    an `.nfsXXXX` silly-rename behind, so `TemporaryDirectory`'s cleanup raises `OSError: [Errno 39] Directory not empty` and
    the whole tick fails over a leftover file."""
    bad = {m: lines for m in LOOP_MODULES if (lines := _unguarded(SRC / m))}
    assert not bad, (f"tempfile.TemporaryDirectory without ignore_cleanup_errors=True at {bad}: pass it, since the loop's TMPDIR "
                     "is NFS and a silly-renamed file fails the cleanup")
