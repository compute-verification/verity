"""`research.landing`, what the verified queue's loop reads of runs and trees: the files a failed run implicates and whether
its machine killed it, what a running check has failed so far, train.sh's launch (its credentials, flags and check slot), the
keywords of a check of a candidate, the jobs `main` waits for, the runs' states and evidence read several at a time, the
merge-thread post, and the notice to the owners of PRs stacked on a landed one; on `test_queue`'s throwaway origin with the
launch, the runs' states and the merge-thread post injected.  `Harness` is the `Landing` the loop's tests build on, and the
helpers below write the run dirs a check leaves."""
from __future__ import annotations

import ast
import contextlib
import dataclasses
import json
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

import pytest

from research import landing, merge, remote
from research.landing import AFTER_LANDING, AFTER_LANDING_OPTIONAL, SLIM_JUDGED, SLIM_STEPS, Landing, gating
from research.policy import SCHEMA as POLICY_SCHEMA
from research.queue import PR, Member
from test_queue import World, _git

SRC = Path(__file__).resolve().parents[1] / "src" / "research"
#: the modules of the verified queue's loop, which runs on the control pod, whose TMPDIR is an NFS volume
LOOP_MODULES = ("queue.py", "landing.py", "vqueue.py", "vqueue_loop.py", "vqueue_slim.py", "vqueue_records.py")


class Harness:
    """A `Landing` (`make`) on ``world`` whose records-run launches are recorded (`launched`) and whose launches of a check or
    of the run beside one (`launch_postland`) are recorded (`posted`) as `r-post-N`, each running until a test sets its state
    in `run_state`; its posts, cancels, evidence and clock injected."""

    make = Landing

    def __init__(self, world: World, monkeypatch, **kw):
        self.world, self.launched, self.posted, self.posts, self.after, self.cancelled = world, [], [], [], [], []
        self.run_state: dict[str, tuple] = {}
        self.clock = 1_000_000.0
        monkeypatch.setattr(merge, "caught_up", lambda repo, remote, tool, since="", log=print, runs=None:
                            self.after.append((since, runs)))
        monkeypatch.setattr(merge, "_sweep_after", lambda *a, **k: None)
        monkeypatch.setattr(merge, "sweep_branches", lambda *a, **k: (0, 0))
        self.q = world.queue(**kw)
        self.q.now = lambda: self.clock
        self.d = self.make(self.q, on="m1", launch=self.launch, states=lambda repo, ids: {i: self.run_state[i] for i in ids},
                           post=lambda text, to=(): self.posts.append(text) or True, now=lambda: self.clock,
                           cancel=lambda repo, ids, reason: self.cancelled.extend(ids), evidence=lambda repo, ids: self.evidence)
        self.evidence: list = []
        self.d.launch_postland = self.launch_post

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


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


#: a ci.toml with a job for each step of a check, and its policy taking circuit-check off the merge gate as in the tree
CI_TOML = 'schema = "ci/v0"\n' + "".join(f'[[job]]\nname = "{s}"\nowner = "ci"\ngroup = 1\n' for s in AFTER_LANDING)
POLICY_TOML = 'schema = "policy/v0"\noff_gate = ["circuit-check"]\n'
#: the files `_move_main` writes for a tree declaring `CI_TOML` under `POLICY_TOML`
GATED_TREE = {"tools/verity/check/ci.toml": CI_TOML, "tools/verity/check/policy.toml": POLICY_TOML}
#: the preflights a check runs before any step (check.py's `preflight`), which the run beside a candidate's check judges
PREFLIGHTS = tuple(s for s in SLIM_JUDGED if s not in SLIM_STEPS)


def _policy(*allow: tuple[str, str, str], off_gate: tuple[str, ...] = ("circuit-check",)) -> str:
    """policy.toml's text: ``off_gate`` and an allowance (id, owner, expires) for each of ``allow``."""
    return (f'schema = "{POLICY_SCHEMA}"\noff_gate = {json.dumps(list(off_gate))}\n'
            + "".join(f'[[allow]]\nid = "{i}"\nowner = "{o}"\nexpires = "{e}"\nreason = "red; its fix is landing"\n'
                      for i, o, e in allow))


def _main(world: World) -> str:
    return _git(world.origin, "rev-parse", "main")


def _move_main(world: World, files: dict[str, str]) -> None:
    """A commit of ``files`` onto `main`, made outside the queue."""
    _git(world.dev, "push", "-q", "origin", f"{world.change(files)}:refs/heads/main")


def _slim(h: Harness, tmp_path, **logs: str) -> Path:
    """The newest run beside a check (`r-post-N`, of `SLIM_STEPS` but one it skips) ended: each preflight and each of its steps
    passed, but each ``logs`` names (``ratchet_gate=LOG``), which failed with that log, its exit 1 then; after a failed
    preflight it ran no step, as check.py starts none."""
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
    d = tmp_path / "run"
    (d / "suites").mkdir(parents=True)
    (d / "steps.jsonl").write_text(steps)
    (d / "suites-lean.json").write_text(json.dumps([{"name": "pkg", "suite": "pkg", "status": "failed", "failed_tests": [test],
                                                     "log": "/workspace/research/runs/r/suites/pkg.log"}]))
    (d / "suites" / "pkg.log").write_text(f"E       NameError: name '_once' is not defined\n\n{trace}: NameError\n")
    return d


#: the analysis's packages, as an audit's log names each after its scratch directory
SPEC, PROOFS = "security", "security/Proofs"
LEAN_AUDIT_LOG = f"""lean-audit: verity/hardware/world: PASS reused (eabc707c8940): every file its audit reads is unchanged
AUDIT /workspace/research/cache/verity-check/lean-audit-scratch-2957380-6l2lanb_/{PROOFS}: FAIL; built WITHOUT a sandbox
  build: exit 1
    warning: ../Proofs/Flock/Recursive/PartTie.lean:168:47: Used `tac1 <;> tac2`
    error: ../Proofs/Flock/Claims/NumbersRadius.lean:100:2: Tactic `rfl` failed: The left-hand side
    error: build failed
AUDIT /workspace/research/cache/verity-check/lean-audit-scratch-2957380-6l2lanb_/{SPEC}: FAIL; built WITHOUT a sandbox
    error: Definitions/Claims/Sampling.lean:512:20: Type mismatch
"""


def _lean_run(tmp_path, log: str = LEAN_AUDIT_LOG):
    d = tmp_path / "run"
    d.mkdir(parents=True)
    (d / "steps.jsonl").write_text('{"step": "lean-audit", "status": "failed", "rc": 1, "log": "lean-audit.log"}\n')
    (d / "lean-audit.log").write_text(log)
    return d


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


def _steps_run(tmp_path, name: str, steps: dict[str, str]):
    d = tmp_path / name
    d.mkdir(parents=True)
    (d / "steps.jsonl").write_text("".join(json.dumps({"step": k, "status": v}) + "\n" for k, v in steps.items()))
    return d


def _live_run(tmp_path, rid: str, commit: str, records: dict[str, str | None]):
    """A check still running, as `merge.fetch_run` reads its run dir from its node: (that dir, a fetch of it, the fetches made).
    Its Lean report names a regenerated lock for each package of ``records``, holding its text (None: the report names it, and
    the file isn't there)."""
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
    """A check that fails in lean-audit isn't told as naming no file, to be attributed by hand."""
    from research.landing import implicated, lean_errors

    assert lean_errors(LEAN_AUDIT_LOG) == {"security/Proofs/Flock/Claims/NumbersRadius.lean",
                                           "security/Definitions/Claims/Sampling.lean"}
    assert implicated(_lean_run(tmp_path)) == lean_errors(LEAN_AUDIT_LOG)
    assert implicated(_lean_run(tmp_path / "b", f"AUDIT /x/lean-audit-scratch-1/{SPEC}: FAIL\n    error: build failed\n")) is None


def test_a_failed_pytest_step_points_at_its_tests_and_the_paths_its_failures_name(tmp_path):
    from research.landing import implicated

    assert implicated(_lint_run(tmp_path)) == {"tests/test_repository.py", "tools/research/src/research/store/README.md"}
    assert implicated(_lint_run(tmp_path / "b", "error: lock is stale\n")) is None


def test_a_failure_outside_the_suites_or_without_a_failing_test_points_at_no_file(tmp_path):
    """A frame in the check's tree (job.json's `run.cwd`) points at its repository path, one outside it at nothing."""
    from research.landing import implicated

    d = _failed_run(tmp_path, "tests/test_x.py::test_y", 'File "/x/site/pkg/conftest.py", line 3\n'
                                                         'File "/usr/lib/python3.12/runpy.py", line 9')
    assert implicated(d) == {"pkg/tests/test_x.py"}
    (d / "job.json").write_text(json.dumps({"run": {"cwd": "/x/site"}}))
    assert implicated(d) == {"pkg/tests/test_x.py", "pkg/conftest.py"}
    (d / "steps.jsonl").write_text('{"step": "circuit-check", "status": "failed"}\n')
    assert implicated(d) is None


def test_a_runs_tree_root_is_its_job_jsons_run_cwd_and_a_malformed_job_json_raises(tmp_path):
    """No job.json, or one naming no absolute `run.cwd`, gives no root; a job.json that isn't a JSON object raises rather than
    reading as a run with no root."""
    from research.landing import tree_root

    assert tree_root(tmp_path) is None
    for doc, root in (({"run": {"cwd": "/x/site/"}}, "/x/site"), ({"run": {"cwd": "site"}}, None), ({"run": None}, None),
                      ({"environment": {}}, None)):
        (tmp_path / "job.json").write_text(json.dumps(doc))
        assert tree_root(tmp_path) == root, doc
    (tmp_path / "job.json").write_text('{"run": {"cwd": "/x/si')
    with pytest.raises(ValueError):
        tree_root(tmp_path)
    (tmp_path / "job.json").write_text("[]")
    with pytest.raises(AttributeError):
        tree_root(tmp_path)


def test_a_guard_failed_suite_with_no_failing_test_points_at_its_outside_reads_and_their_readers(tmp_path):
    """A suite can fail with every test passing — the read guard alone — and then "the failure names no file" would leave
    attribution to an agent's archaeology. The suite's record names the files
    read outside the inputs and the test module that read each (`outside`, path -> module; a child's reads in
    `child_outside`), and those implicate the PR that added the reading test or changed the file read."""
    from research.landing import implicated

    d = tmp_path / "run"
    (d / "suites").mkdir(parents=True)
    (d / "steps.jsonl").write_text('{"step": "lean-suites", "status": "failed"}\n')
    record = {"name": "verity-lean-audit", "suite": "tools/verity/lean", "status": "failed", "failed_tests": [],
              "outside": {"tools/verity/check/lean_slot.py": "tools/verity/lean/tests/test_lean_changed.py",
                          "tools/verity/check/slot.py": "*"},
              "child_outside": {"tools/verity/lean/tests/test_cache.py": ["elsewhere/data.txt"]}, "log": None}
    (d / "suites-lean.json").write_text(json.dumps([record]))
    assert implicated(d) == {"tools/verity/check/lean_slot.py", "tools/verity/check/slot.py",
                             "tools/verity/lean/tests/test_lean_changed.py", "tools/verity/lean/tests/test_cache.py",
                             "elsewhere/data.txt"}
    (d / "suites-lean.json").write_text(json.dumps([{**record, "outside": {}, "child_outside": {}}]))
    assert implicated(d) is None


def test_a_collection_error_names_its_test_module(tmp_path):
    """A suite that fails collecting one PR's test module (one reading a vector only another PR adds) names that module, so
    the fallback on the suite's inputs doesn't eject every PR touching them instead of the one."""
    from research.landing import implicated

    d = tmp_path / "run"
    (d / "suites").mkdir(parents=True)
    (d / "steps.jsonl").write_text('{"step": "pytest", "status": "failed"}\n')
    (d / "suites.json").write_text(json.dumps([{"name": "filler", "suite": "benchmarks/pouw/filler", "status": "failed",
                                                "failed_tests": [], "log": "/x/suites/filler.log"}]))
    (d / "suites" / "filler.log").write_text("ERROR tests/test_capacity_of.py - KeyError: 'VF3r-16'\n1 error in 2.1s\n")
    assert "benchmarks/pouw/filler/tests/test_capacity_of.py" in implicated(d)


def test_lean_errors_reads_the_audits_orphan_and_layers_failures(tmp_path):
    """An orphan file or an out-of-layer import names a file a landing's change is matched to, so the one PR is named rather
    than every PR touching Lean."""
    from research.landing import lean_errors

    flock = "verity/protocols/flock/verifier"
    log = (f"AUDIT /w/lean-audit-scratch-1-x/{flock}: FAIL  14588 declarations in 159 modules\n"
           "  orphan: Flock/Test/KnownRead.lean is not imported by the roots (Flock, FlockVerify) nor under `exempt`: nothing builds"
           " or audits it\n  note: 267 definitions of the spec (VStar) are read by no guarantee\n"
           "AUDIT /w/lean-audit-scratch-1-x/verity/protocols/pouw/verifier/lean/check: FAIL  944 declarations in 13 modules\n"
           "  layers: PouwCheck.Credit imports Flock.Program, outside PouwCheck.Credit's layer (Std.Data.HashMap)\n")
    assert lean_errors(log) == {f"{flock}/Flock/Test/KnownRead.lean",
                                "verity/protocols/pouw/verifier/lean/check/PouwCheck/Credit.lean"}


def test_lean_errors_reads_lakes_list_of_failed_modules(tmp_path):
    """With the build's error lines scrolled out of the step's log, the Lean files are still found from Lake's list of failed
    modules, rather than falling back on every PR touching Lean."""
    from research.landing import lean_errors

    log = (f"AUDIT /w/lean-audit-scratch-1-x/{PROOFS}: FAIL; built WITHOUT a sandbox\n  build: exit 1\n"
           "    Some required targets logged failures:\n    - Proofs.Flock.Claims.NumbersRadius\n    error: build failed\n"
           f"AUDIT /w/lean-audit-scratch-1-x/{SPEC}: FAIL\n    Some required targets logged failures:\n"
           "    - Definitions.Claims.Sampling\n")
    assert lean_errors(log) == {"security/Proofs/Flock/Claims/NumbersRadius.lean", "security/Definitions/Claims/Sampling.lean"}


def test_a_lean_audit_whose_replay_was_killed_is_a_machines_fault(tmp_path):
    """A security/Proofs replay OOM-killed (exit 137) fails lean-audit with rc 1; that is the machine's fault, and the fallback
    on Lean sources mustn't blame landings that broke nothing."""
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
    machine's fault, and circuit-check's inputs mustn't be blamed for it."""
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
    assert killed(_killed_run(tmp_path, "sigkill")), "a step the machine killed outright"


def test_what_a_running_check_has_failed_is_read_from_its_steps_and_its_suites_failures(tmp_path):
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

    assert forces_cold(["verity/definitions/circuits/library.py"]) and forces_cold(["uv.lock"]) and forces_cold(["x/y.circuit-check.py"])
    assert forces_cold(["tools/verity/circuit_check/src/circuit_check/cache.py"])
    assert forces_cold(["adapters/vllm/verity_vllm/pipeline/serving_rows.py"])          # else hundreds of targets rerun
    assert not forces_cold(["adapters/vllm/tests/pipeline/test_row.py", "tools/research/src/research/landing.py"])
    assert not forces_cold(["security/Proofs/X.lean", "verity/circuits/README.md", "verity/circuits/tests/test_x.py"])
    assert not forces_cold([])


# -- train.sh's launch


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


def test_train_launch_gives_train_sh_the_steps_and_flags_of_a_candidates_check(tmp_path):
    """A candidate's check, or the run beside it, is `train.sh SHA MACHINES` with the upstream build, its steps, records
    regenerated never, ratchet-gate's base and its shard, in train.sh's order."""
    from research.landing import train_launch

    (tmp_path / "tools" / "verity" / "check").mkdir(parents=True)
    (tmp_path / "tools" / "verity" / "check" / "train.sh").write_text('echo "argv: $*"\necho "RUN r20261027-120000-aaaa"\n')
    ids, tail = train_launch(tmp_path, "c" * 40, "m1", True, deferred="lean-agreement,ratchet-gate", no_regenerate=True,
                             base="b" * 40, shard="2/3")
    assert ids == ["r20261027-120000-aaaa"]
    assert f"argv: {'c' * 40} m1 --agreement --deferred lean-agreement,ratchet-gate --no-regenerate --base {'b' * 40} " \
           "--shard 2/3" in tail, tail
    ids, tail = train_launch(tmp_path, "c" * 40, "m1", False)
    assert f"argv: {'c' * 40} m1\n" in tail + "\n", tail


def test_train_launch_gives_train_sh_no_cache_circuit_checks_cache_and_ratchet_gates_own_base(tmp_path):
    """`train.sh --no-cache` reuses no suite or Lean pass, `--circuit-cache` keeps circuit-check's per-target ones (a cold
    `circuit-check --all` outlasts every check), and `--ratchet-base` is the commit ratchet-gate judges above when it isn't
    the `--base`."""
    from research.landing import train_launch

    (tmp_path / "tools" / "verity" / "check").mkdir(parents=True)
    (tmp_path / "tools" / "verity" / "check" / "train.sh").write_text('echo "argv: $*"\necho "RUN r20261027-120000-aaaa"\n')
    ids, tail = train_launch(tmp_path, "c" * 40, "m1", False, deferred="pytest,circuit-check", no_cache=True,
                             circuit_cache=True, no_regenerate=True)
    assert ids == ["r20261027-120000-aaaa"]
    assert f"argv: {'c' * 40} m1 --deferred pytest,circuit-check --no-cache --circuit-cache --no-regenerate" in tail, tail
    ids, tail = train_launch(tmp_path, "c" * 40, "m1", False, deferred="ratchet-gate", base="b" * 40, ratchet_base="a" * 40)
    assert f"argv: {'c' * 40} m1 --deferred ratchet-gate --base {'b' * 40} --ratchet-base {'a' * 40}" in tail, tail


def test_the_records_run_gives_train_sh_bookkeeping_warn_when_its_train_sh_takes_it(tmp_path):
    """The records run repairs bookkeeping and warns (`--bookkeeping-warn`) when the queue's clone's train.sh takes the flag
    (`takes_bookkeeping`); a train.sh that predates it would refuse every records run, so it isn't given it."""
    from research.landing import takes_bookkeeping, train_launch

    sh = tmp_path / "tools" / "verity" / "check" / "train.sh"
    sh.parent.mkdir(parents=True)
    assert not takes_bookkeeping(tmp_path)
    sh.write_text('# --bookkeeping-warn\necho "argv: $*"\necho "RUN r20261027-120000-aaaa"\n')
    assert takes_bookkeeping(tmp_path)
    ids, tail = train_launch(tmp_path, "c" * 40, "m1", False, deferred="lean-audit", bookkeeping_warn=True)
    assert ids == ["r20261027-120000-aaaa"] and f"argv: {'c' * 40} m1 --deferred lean-audit --bookkeeping-warn" in tail, tail
    sh.write_text('echo "argv: $*"\n')
    assert not takes_bookkeeping(tmp_path)


def test_a_candidates_check_and_its_records_run_take_a_check_slot_ahead_of_unranked_priority_checks(tmp_path, monkeypatch):
    """Each part of a candidate's check and its records run ask for their check slot at `POSTLAND_SLOT_RANK`
    (CHECK_SLOT_RANK, which check.py passes on as `slot.py --rank`), whatever the loop's own environment says: a ranked
    priority waiter takes the next free slot ahead of the unranked, so `main` never waits on a slot for a `train.sh`
    launched by hand."""
    import types

    from research.landing import POSTLAND_SLOT_RANK, launched

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setenv("CHECK_SLOT_RANK", "7")
    repo = tmp_path / "clone"
    (repo / "tools" / "verity" / "check").mkdir(parents=True)
    (repo / "tools" / "verity" / "check" / "train.sh").write_text(
        'echo "rank=[$CHECK_SLOT_RANK] argv=[$*]"\necho "RUN r20261103-140000-aaaa"\n')
    d = Landing(types.SimpleNamespace(auth=(), workdir=repo), on="m1")

    def tail(job: dict) -> str:
        with contextlib.suppress(ChildProcessError):                     # its end, not a clock
            os.waitpid(job["pid"], 0)
        return launched(job)[1]

    ranked = f"rank=[{POSTLAND_SLOT_RANK}]"
    assert f"{ranked} argv=[{'c' * 40} m1 --deferred pytest --shard 1/3]" in tail(
        d.launch_postland(repo, "c" * 40, "m1", "pytest", shard="1/3"))
    assert f"{ranked} argv=[{'c' * 40} m1 --deferred lean-audit]" in tail(d.launch_locks(repo, "c" * 40, "m1"))


def test_a_landings_launches_carry_the_queues_credentials_the_loops_train_sh_and_bookkeeping_where_train_sh_takes_it(
        tmp_path, monkeypatch):
    """A `Landing` on `train_launch` launches with the queue's credentials (``q.auth``: `launch` as `launch_postland` and
    `launch_locks`), a candidate's check with the loop's own train.sh when given one (``train_from``), and the records run with
    `--bookkeeping-warn` where the clone's train.sh takes it (`takes_bookkeeping`)."""
    import types

    from research.landing import launched

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.delenv("GIT_CONFIG_COUNT", raising=False)
    repo, own, rid = tmp_path / "clone", tmp_path / "own" / "train.sh", "r20261110-100000-aaaa"
    for path, who in ((repo / "tools" / "verity" / "check" / "train.sh", "clone"), (own, "own")):
        path.parent.mkdir(parents=True)
        path.write_text(f'# --bookkeeping-warn\necho "{who} n=$GIT_CONFIG_COUNT argv=[$*]"\necho "RUN {rid}"\n')
    d = Landing(types.SimpleNamespace(auth=("-c", "credential.helper="), workdir=repo), on="m1")

    def tail(job: dict) -> str:
        with contextlib.suppress(ChildProcessError):                     # its end, not a clock
            os.waitpid(job["pid"], 0)
        return launched(job)[1]

    assert d.launch(repo, "c" * 40, "m1", False) == ([rid], f"clone n=1 argv=[{'c' * 40} m1]\nRUN {rid}")
    assert f"own n=1 argv=[{'c' * 40} m1 --deferred pytest]" in tail(
        d.launch_postland(repo, "c" * 40, "m1", "pytest", train_from=own))
    assert f"clone n=1 argv=[{'c' * 40} m1 --deferred lean-audit --bookkeeping-warn]" in tail(
        d.launch_locks(repo, "c" * 40, "m1"))


def test_a_landing_on_an_injected_launch_launches_a_candidates_check_and_its_records_run_through_it(world):
    """With ``launch`` injected, `launch_postland` and `launch_locks` launch through it, on the machines asked for."""
    got = []
    d = Landing(world.queue(), on="m1", launch=lambda repo, commit, machine, agreement: got.append(
        (commit, machine, agreement)) or (["r1"], ""))
    assert d.launch_postland(d.repo, "c" * 40, "m2", "pytest", shard="1/3") == d.launch_locks(d.repo, "d" * 40, "m3") == (
        ["r1"], "")
    assert got == [("c" * 40, "m2", False), ("d" * 40, "m3", False)]


def test_detach_returns_before_train_sh_prints_its_run_ids_and_launched_reads_them_or_an_exit_without_them(tmp_path, monkeypatch):
    """`detach`'s job, with a stand-in for train.sh that prints its `RUN` line only once the test lets it."""
    from research.landing import LAUNCHES, detach, launched

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    gate, rid = tmp_path / "gate", "r20261026-093000-abcd"
    os.mkfifo(gate)
    job = detach(tmp_path, ["sh", "-c", f'echo "train: abc on m1"; cat "$1" > /dev/null; echo "RUN {rid}"', "sh", str(gate)],
                 dict(os.environ))
    assert job["dir"].startswith(str(tmp_path / LAUNCHES)) and launched(job)[0] is None
    with open(gate, "w"):
        pass
    with contextlib.suppress(ChildProcessError):                         # its end, not a clock
        os.waitpid(job["pid"], 0)
    assert launched(job) == ([rid], f"train: abc on m1\nRUN {rid}")
    job = detach(tmp_path, ["sh", "-c", "echo 'train: no run launched' >&2; exit 1"], dict(os.environ))
    with contextlib.suppress(ChildProcessError):
        os.waitpid(job["pid"], 0)
    assert launched(job) == ([], "train: no run launched") and (Path(job["dir"]) / "rc").read_text() == "1\n"
    gone = subprocess.Popen(["true"])
    gone.wait()
    (Path(job["dir"]) / "rc").unlink()
    assert launched({**job, "pid": gone.pid})[0] == []


def test_a_detached_train_launch_runs_its_own_copy_of_train_sh(tmp_path, monkeypatch):
    """Bash reads a script as it runs, and a later tick resets and checks out the queue's clone, so
    a train.sh that `main` changed meanwhile would change under a running launch.  It runs a copy in its launch directory, from
    the clone, and a rewrite of the clone's train.sh doesn't reach it."""
    from research.landing import LAUNCHES, LEAN_STEP, launched, train_launch

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    repo, gate, rid = tmp_path / "clone", tmp_path / "gate", "r20261026-093000-abcd"
    script = repo / "tools" / "verity" / "check" / "train.sh"
    script.parent.mkdir(parents=True)
    script.write_text(f'echo "train: $0 in $PWD"\ncat {gate} > /dev/null\necho "RUN {rid}"\n')
    os.mkfifo(gate)
    job = train_launch(repo, "c0ffee", "m1", False, deferred=LEAN_STEP, detached=True)
    copy = Path(job["dir"]) / "train.sh"
    assert job["dir"].startswith(str(tmp_path / LAUNCHES)) and copy.read_text() == script.read_text()
    with open(script, "w") as f:                                           # in place, as a checkout of a changed train.sh
        f.write('echo "RUN r20261026-093000-ffff"\n' * 8)
    with open(gate, "w"):
        pass
    with contextlib.suppress(ChildProcessError):                         # its end, not a clock
        os.waitpid(job["pid"], 0)
    ids, tail = launched(job)
    assert ids == [rid] and f"train: {copy} in {repo}" in tail, tail


def test_a_detached_train_launch_given_a_train_sh_runs_a_copy_of_it_in_the_clone(tmp_path, monkeypatch):
    """The loop launches with its own train.sh (`script_from`): a copy of that one runs, in the clone, the clone's ignored."""
    from research.landing import LEAN_STEP, launched, train_launch

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    repo, own, rid = tmp_path / "clone", tmp_path / "own" / "train.sh", "r20261026-093000-abcd"
    script = repo / "tools" / "verity" / "check" / "train.sh"
    for path, text in ((script, 'echo "RUN r20261026-093000-ffff"\n'), (own, f'echo "train: own in $PWD"\necho "RUN {rid}"\n')):
        path.parent.mkdir(parents=True)
        path.write_text(text)
    job = train_launch(repo, "c0ffee", "m1", False, deferred=LEAN_STEP, detached=True, script_from=own)
    assert (Path(job["dir"]) / "train.sh").read_text() == own.read_text()
    with contextlib.suppress(ChildProcessError):                         # its end, not a clock
        os.waitpid(job["pid"], 0)
    ids, tail = launched(job)
    assert ids == [rid] and f"train: own in {repo}" in tail, tail


def test_a_train_launch_given_a_deselect_list_passes_it_to_train_sh(tmp_path, monkeypatch):
    from research.landing import train_launch

    got = []
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: got.append(argv) or subprocess.CompletedProcess(argv, 0, "", ""))
    train_launch(tmp_path, "c0ffee", "m1", False, deferred="pytest", shard="1/3", deselect="/data/vqueue/live/deselect.sent")
    train_launch(tmp_path, "c0ffee", "m1", False, deferred="pytest")
    assert got[0][got[0].index("--deselect") + 1] == "/data/vqueue/live/deselect.sent" and "--deselect" not in got[1]


def test_plan_shards_asks_the_planner_for_its_own_n_at_most_max_shards_or_exactly_n(tmp_path, monkeypatch):
    """`plan_shards` hands `PLAN_JOB` the steps, and `--max-shards M` (drive's `auto:MAX`) or `--n N` (the loop's `auto:N`)."""
    asked = []
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(landing, "detach", lambda repo, argv, env: asked.append(argv) or {"dir": "job", "pid": 1})
    got = [landing.plan_shards(tmp_path, "t" * 40, "pytest,lean-build", **kw) for kw in ({}, {"max_shards": 4}, {"n": 6})]
    assert [a[:6] for a in asked] == [["sh", "-c", landing.PLAN_JOB, "sh", g["out"], "t" * 40] for g in got]
    assert [a[6:] for a in asked] == [["--deferred", "pytest,lean-build", *more] for more in ([], ["--max-shards", "4"],
                                                                                             ["--n", "6"])]
    assert all(Path(g["out"]).parent == tmp_path / landing.LAUNCHES and g["dir"] == "job" for g in got)


def _plan_doc(n: int) -> dict:
    return {"kind": "check-shard-plan/v1", "n": n, "digest": "d" * 64, "predicted_s": 3600.0}


def test_a_check_of_a_candidate_sends_the_upstream_build_only_when_what_lean_agreement_checks_changed(world, monkeypatch):
    """`beside_args`: lean-agreement runs (`--agreement`) only when the merges above the base change what it checks, as
    check.py finds against that base, and is skipped by name else; lean-audit fails on a record difference
    (`--no-regenerate`); ratchet-gate judges the merges above `--base`: the base given, the tip's first parent when the tip is
    the base, and none for a root."""
    asked, changed = [], [False]
    monkeypatch.setattr(merge, "required_steps", lambda repo, base, commit, tool: asked.append((base, commit, tool)) or (
        {"lean-agreement": ["verity/protocols/flock/prover/"]} if changed[0] else {}))
    h = Harness(world, monkeypatch)
    h.q._clone()
    base, sha = _main(world), world.change({"f.txt": "1\n"})
    _git(h.q.workdir, "fetch", "-q", "origin", f"refs/dev/{sha}")
    assert h.d.beside_args(sha, base) == ({"base": base, "agreement": False, "no_regenerate": True}, ["lean-agreement"],
                                          f" --base {base[:12]} --no-regenerate")
    changed[0] = True
    assert h.d.beside_args(sha, base) == ({"base": base, "agreement": True, "no_regenerate": True}, [],
                                          f" --base {base[:12]} --agreement --no-regenerate")
    assert asked == [(base, sha, "check")] * 2
    assert h.d.beside_args(sha, sha)[0]["base"] == _git(h.q.workdir, "rev-parse", f"{sha}^1") == base
    root = _git(h.q.workdir, "rev-list", "--max-parents=0", base)
    assert h.d.beside_args(root, root) == ({"base": None, "agreement": True, "no_regenerate": True}, [],
                                           " --agreement --no-regenerate")


def test_main_waits_for_the_steps_on_the_gate_of_the_commits_own_ci_toml(world, monkeypatch):
    """`main` waits at a commit for every step of a check but those a run skips by name and the jobs that commit's own policy
    takes off the merge gate (circuit-check); for every such step when its ci.toml or policy can't be read."""
    h = Harness(world, monkeypatch)
    h.q._clone()
    h.q.fetch()
    every = [s for s in AFTER_LANDING if s not in AFTER_LANDING_OPTIONAL]
    assert gating(h.q.workdir, _main(world)) == h.d._gating(_main(world)) == every
    _move_main(world, GATED_TREE)
    h.q.fetch()
    assert gating(h.q.workdir, _main(world)) == h.d._gating(_main(world)) == [s for s in every if s != "circuit-check"]
    _move_main(world, {"tools/verity/check/policy.toml": "not toml ["})
    h.q.fetch()
    assert gating(h.q.workdir, _main(world)) == every
    _move_main(world, {"tools/verity/check/policy.toml": POLICY_TOML, "tools/verity/check/ci.toml": "not toml ["})
    h.q.fetch()
    assert gating(h.q.workdir, _main(world)) == every


# -- reading runs


def _shard(tmp_path, rid: str, steps: dict[str, str], live: dict[str, str] | None = None, preflight: bool = False,
           state: str = "running"):
    """A shard's run dir: ``steps`` ended in its steps.jsonl, and its live record (`LIVE`, the check's ``state``) naming the
    preflight's steps, those and ``live``'s, still to end (pending, running); with ``preflight``, the preflight's alone, its
    groups not yet planned."""
    from research.landing import LIVE, PREFLIGHT

    d = _steps_run(tmp_path, rid, steps)
    planned = {s: "running" if preflight else "passed" for s in PREFLIGHT} | ({} if preflight else {**steps, **(live or {})})
    (d / LIVE).write_text(json.dumps({"schema": "check-live/v1", "state": state,
                                      "steps": [{"step": s, "state": v} for s, v in planned.items()]}))
    return d


def test_a_step_more_than_one_shard_holds_passed_only_once_every_shard_that_holds_it_passed_it(tmp_path):
    """A sharded check's shards hold parts of one step (lean-audit's packages, lean-suites' suites, lean-build beside them): it
    passed once every shard that holds it passed it (or skipped it, or failed it only on what an allowance covers), none having
    it still to run by its live record, so another shard's pass doesn't pass a part that runs on, failed or was stopped; a
    twin's pass passes it for its shard, and twins' other words rank whatever their order (`RANK`); a step an ended check never
    started was stopped; and nothing passed while a shard's plan isn't known."""
    from research.landing import PREFLIGHT, shard_passed, shard_steps

    one = _shard(tmp_path, "r1", {"lean-build": "passed", "lean-suites": "passed", "pytest": "passed"})
    two = _shard(tmp_path, "r2", {"lean-build": "passed"}, {"lean-suites": "running", "lean-agreement": "pending"})
    view = shard_steps([[one], [two]], [True, False])
    assert view[1] == {**dict.fromkeys(PREFLIGHT, "passed"), "lean-build": "passed", "lean-suites": None, "lean-agreement": None}
    assert shard_passed(view) == {*PREFLIGHT, "lean-build", "pytest"}

    failed, stopped, skipped = (shard_steps([[one], [_shard(tmp_path / s, "r2", {"lean-suites": s})]], [True, False])
                                for s in ("failed", "stopped", "skipped"))
    assert "lean-suites" not in shard_passed(failed) and "lean-suites" in shard_passed(failed, {"lean-suites"})
    assert "lean-suites" not in shard_passed(stopped, {"lean-suites"})
    assert "lean-suites" in shard_passed(skipped)
    twins = [_shard(tmp_path / "t", "r2", {"lean-suites": "stopped"}), _shard(tmp_path / "t", "r2b", {"lean-suites": "passed"})]
    assert "lean-suites" in shard_passed(shard_steps([[one], twins], [True, False])), "a twin's pass passes its shard"
    twins[1] = _shard(tmp_path / "u", "r2b", {}, {"lean-suites": "running"})
    assert shard_steps([[one], twins], [True, False])[1]["lean-suites"] is None, "one twin still runs it"
    for a, b in (("stopped", "failed"), ("failed", "stopped"), ("skipped", "stopped"), ("stopped", "skipped")):
        pair = [_shard(tmp_path / f"{a}-{b}", "r2", {"lean-suites": a}), _shard(tmp_path / f"{a}-{b}", "r2b", {"lean-suites": b})]
        assert shard_steps([[one], pair], [True, False])[1]["lean-suites"] == ("failed" if "failed" in (a, b) else "skipped")
    closed = shard_steps([[one], [_shard(tmp_path / "c", "r2", {"lean-build": "failed"}, {"lean-suites": "pending"}, state="failed")]],
                         [True, False])
    assert closed[1]["lean-suites"] == "stopped", "its group stopped on lean-build's failure: it never starts"
    assert "lean-suites" not in shard_passed(closed, {"lean-suites"})

    early, bare = _shard(tmp_path / "e", "r2", {}, preflight=True), _steps_run(tmp_path / "b", "r2", {"lean-build": "passed"})
    assert shard_steps([[one], [early]], [True, False]) is None and shard_steps([[one], [bare]], [True, False]) is None
    assert shard_steps([[one], [bare]], [True, True])[1] == {"lean-build": "passed"}, "a twin exited 0: its steps.jsonl"
    assert shard_steps([[bare]], [False]) == [{"lean-build": "passed"}], "an unsharded run: its steps.jsonl"


def _row(d: Path, state: str) -> tuple[Path, dict]:
    """A read of run dir ``d`` (`landing.read_run`): it, and its `research status --json` row in ``state``."""
    d.mkdir(parents=True, exist_ok=True)
    return d, {"state": state, "rc": {"done": 0, "failed": 1}.get(state), "dir": str(d)}


def test_run_states_reads_runs_in_parallel_and_keeps_each_state_with_its_run(tmp_path):
    """Reading about 40 runs one `research status` (an ssh round trip) at a time took a tick 33 min. Several are read at once, and each answer stays with the run it is about, whatever order they finish in."""
    seen: set[int] = set()
    lock = threading.Lock()

    def read(rid, names):
        with lock:
            seen.add(threading.get_ident())
        state = {"r1": "running", "r2": "done", "r3": "failed"}.get(rid)
        return None if state is None else _row(tmp_path / rid, state)

    ids = ["r3", "r1", "nope", "r2"]
    out = landing.Reads(read=read, node=str).states(tmp_path, ids)
    assert list(out) == ids
    assert out["r1"][0] == "running" and out["r2"][:2] == ("done", 0) and out["r3"][:2] == ("failed", 1)
    assert out["nope"] == ("unknown", None, None, None)
    assert seen and threading.get_ident() not in seen, "read on worker threads, not one after another on the caller's"


def test_reads_keep_order_and_read_a_single_run_inline(tmp_path):
    me, at = threading.get_ident(), []
    one = landing.Reads(read=lambda rid, names: at.append(threading.get_ident()) or _row(tmp_path / rid, "done"), node=str)
    assert list(one.states(tmp_path, ["only"])) == ["only"] and at == [me]
    assert one.states(tmp_path, []) == {} and one.evidence(tmp_path, []) == []


def test_evidence_drops_runs_it_cannot_fetch_and_keeps_the_rest_in_order_and_fetches_a_failed_suites_log(tmp_path):
    """`Reads.evidence`: the runs' `EVIDENCE` first, then of a run whose suite failed, that suite's log with its guard's report
    and its children's reads."""
    asked = []
    failed = _failed_run(tmp_path, "test_x", "t.py:1")

    def read(rid, names):
        asked.append((rid, names))
        return {"r1": _row(tmp_path / "r1", "done"), "run": (failed, {"state": "failed", "dir": str(failed)})}.get(rid)

    assert landing.Reads(read=read, node=str).evidence(Path("."), ["run", "r2", "r1"]) == [failed, tmp_path / "r1"]
    assert sorted(asked) == sorted([("run", landing.EVIDENCE), ("r2", landing.EVIDENCE), ("r1", landing.EVIDENCE),
                                    ("run", ("suites/pkg.log", "suites/pkg.json", "suites/pkg.reads"))])


def _remote_run(root: Path, rid: str, state: str = "running") -> Path:
    """The local dir of a run launched on a node, its status.json as last fetched in ``state``."""
    d = root / rid
    d.mkdir(parents=True)
    (d / "remote.json").write_text(json.dumps({"host": "h", "port": 22, "machine": "m1", "remote_run_dir": f"/runs/{rid}"}))
    (d / "status.json").write_text(json.dumps({"state": state}))
    return d


class Node:
    """`remote.fetch` from a node: each fetch recorded (the run, the files asked), the run's status.json then the next of its
    ``states`` (the last once they run out; None: the fetch fails)."""

    def __init__(self, monkeypatch, **states):
        self.fetches, self.states = [], {r: list(s) for r, s in states.items()}
        monkeypatch.setattr(remote, "fetch", self.fetch)

    def fetch(self, d: Path, *, also=(), **kw) -> dict:
        self.fetches.append((d.name, also))
        st = self.states[d.name]
        if (state := st.pop(0) if len(st) > 1 else st[0]) is None:
            return {"last_fetch_error": {"rc": 255}}
        (d / "status.json").write_text(json.dumps({"state": state, **({"rc": 0} if state == "done" else {})}))
        return {}


def test_a_run_is_read_in_process_with_the_files_asked_and_not_the_nodes_memory_or_a_checks_timings(tmp_path, monkeypatch):
    """`read_run` reads a run as `research status RUN --json --also NAME...` does, in the loop's process and without the two ssh
    round trips of the command's that nothing it judges reads: the node's memory and holds, and a check's timings against the
    machine's newest finished check, about 5 s of each running run's 7."""
    from research import cli
    from research.checkview import CHECK_FILES

    monkeypatch.setattr(cli, "_read_holds", lambda *a: pytest.fail("read the node's memory and holds"))
    monkeypatch.setattr(cli, "_check_view", lambda *a: pytest.fail("read a check's timings"))
    monkeypatch.setattr(landing.subprocess, "run", lambda *a, **k: pytest.fail("ran a subprocess"))
    d = _remote_run(tmp_path, "r1")
    node = Node(monkeypatch, r1=["running"])
    got = landing.read_run("r1", ("steps.jsonl.progress", "check-live.json"), tmp_path)
    assert got is not None and got[0] == d and got[1]["state"] == "running" and Path(got[1]["dir"]) == d
    assert node.fetches == [("r1", tuple(dict.fromkeys(("steps.jsonl.progress", "check-live.json", *CHECK_FILES))))]
    assert landing.read_run("nope", (), tmp_path) is None


def test_an_ended_run_is_fetched_until_a_fetch_begun_after_its_end_brought_the_files_asked_and_then_never(tmp_path, monkeypatch):
    """A fetch copies a run dir's files one after another, its live record and spans before status.json, so the fetch that
    finds a run ended may hold what it wrote before its end; one begun after brings its last, after which nothing changes, so
    no later read of those files fetches them (each tick refetched every ended run, 1–2 s each).  A file not yet asked is
    fetched once more; a failed fetch brings nothing."""
    _remote_run(tmp_path, "r1")
    node = Node(monkeypatch, r1=["running", "done"])
    names = ("steps.jsonl.progress", "check-live.json")
    assert landing.read_run("r1", names, tmp_path)[1]["state"] == "running" and len(node.fetches) == 1
    assert landing.read_run("r1", names, tmp_path)[1]["state"] == "done"
    assert len(node.fetches) == 3, "the fetch that found it ended, and one begun after"
    assert landing.read_run("r1", names, tmp_path)[1]["state"] == "done"
    assert landing.read_run("r1", names[:1], tmp_path)[1]["state"] == "done"
    assert len(node.fetches) == 3, "its files as a fetch begun after its end brought them"
    landing.read_run("r1", ("suites/pkg.log",), tmp_path)
    landing.read_run("r1", ("suites/pkg.log", *names), tmp_path)
    assert [also[0] for _, also in node.fetches[3:]] == ["suites/pkg.log"]

    _remote_run(tmp_path, "r2", "done")
    flaky = Node(monkeypatch, r2=[None, "done"])
    for _ in range(3):
        landing.read_run("r2", names, tmp_path)
    assert len(flaky.fetches) == 2, "a fetch that failed, then the one that brought them"


def test_a_tick_reads_each_run_once_and_again_only_for_files_it_has_not_fetched(tmp_path):
    """Inside `Reads.tick` the judging of every candidate reads each run once, all of them at once first, however many times a
    judge asks for its state and evidence (two rounds a candidate, one after another, while an early pass reads its runs too);
    a log not yet fetched is read alone, and a read that fails keeps the tick's earlier one.  Outside a tick each ask reads
    afresh."""
    asked, gone = [], set()

    def read(rid, names):
        asked.append((rid, names))
        return None if rid in gone else _row(tmp_path / rid, "running")

    reads = landing.Reads(read=read, node=str)
    with reads.tick(["r1", "r2", "r1"]):
        assert sorted(asked) == [("r1", landing.EVIDENCE), ("r2", landing.EVIDENCE)], "read at once before any judge asks"
        gone.add("r1")
        assert reads.states(tmp_path, ["r1", "r2"])["r1"][0] == "running"
        assert reads.evidence(tmp_path, ["r2", "r1"]) == [tmp_path / "r2", tmp_path / "r1"]
        reads([("r1", ("suites/a.log",))])
        reads([("r1", ("suites/a.log",)), ("r2", landing.PROGRESS)])
        assert reads.states(tmp_path, ["r1"])["r1"][0] == "running", "its failed read keeps the tick's earlier one"
    assert asked[2:] == [("r1", ("suites/a.log",))]
    del asked[:]
    reads.states(tmp_path, ["r2"])
    reads.states(tmp_path, ["r2"])
    assert asked == [("r2", (*landing.PROGRESS, landing.LIVE))] * 2, "outside a tick each ask reads afresh"


def test_a_slow_node_holds_no_more_than_its_share_of_the_readers(tmp_path):
    """A node that answers slowly (a stalled ssh, 120 s to a probe's timeout) held every reader its runs reached, the runs
    of every other node waiting behind it.  Its runs hold at most `NODE_JOBS` readers, and every other node's runs are read
    while they do: the slow node's reads end only once the last of the others' has.  Its runs come first and outnumber the
    readers, so a pool with no share per node puts every reader on it; a read finding more of its node's in flight than its
    share answers at once, so that pool fails here rather than hangs."""
    slow = [f"s{k}" for k in range(landing.STATUS_JOBS + 2)]
    fast = [f"f{k}" for k in range(landing.STATUS_JOBS)]
    lock, flying, over, others_done, done = threading.Lock(), {"slow": 0, "fast": 0}, [], threading.Event(), []

    def node(rid):
        return "slow" if rid.startswith("s") else "fast"

    def read(rid, names):
        n = node(rid)
        with lock:
            flying[n] += 1
            if flying[n] > landing.NODE_JOBS:
                over.append((rid, flying[n]))
            free = n == "fast" or bool(over)
        try:
            if not free:
                others_done.wait()
            return _row(tmp_path / rid, "running")
        finally:
            with lock:
                flying[n] -= 1
                if n == "fast":
                    done.append(rid)
                    if len(done) == len(fast):
                        others_done.set()

    ids = [*slow, *fast]
    out = landing.Reads(read=read, node=node).states(tmp_path, ids)
    assert not over, f"more of one node's runs read at once than its share: {over}"
    assert list(out) == ids and all(s[0] == "running" for s in out.values())
    assert sorted(done) == sorted(fast)


def test_the_evidence_of_an_attempt_is_fetched_once(world, monkeypatch):
    """`_fetched` reads the run dirs of the runs an attempt names (its ids, space-separated) once, and each attempt's apart."""
    h = Harness(world, monkeypatch)
    asked = []
    h.d.evidence = lambda repo, ids: asked.append((repo, ids)) or [Path(i) for i in ids]
    assert h.d._fetched("r1 r2") == h.d._fetched("r1 r2") == [Path("r1"), Path("r2")]
    assert h.d._fetched("r3") == [Path("r3")]
    assert asked == [(h.q.workdir, ["r1", "r2"]), (h.q.workdir, ["r3"])]


def test_a_failed_task_is_allowed_by_its_suites_row_or_by_each_circuit_failure_it_names_and_no_other_kind_is(world, monkeypatch):
    """`Landing._task_allowed` under the tip's policy: a lean-suites task's failed suite by its report row (an expired allowance
    covers nothing, and is named), a circuit-check task's when each failure it names, fewer than `CIRCUIT_NAMED`, has an
    allowance, and nothing of any other kind."""
    later, gone = "2030-06-01T00:00:00Z", "1970-01-01T00:00:01Z"
    _move_main(world, {"tools/verity/check/policy.toml": _policy(("suite pkg", "ci", later), ("suite old", "ci", gone),
                                                                 ("circuit-check fold/F01 Mul", "ci", later))})
    h = Harness(world, monkeypatch)
    h.q._clone()
    h.q.fetch()
    tip, d = _main(world), h.d
    suites, row = {"kind": landing.SUITES_KIND, "id": "lean-suites:x"}, {"name": "pkg", "failed_tests": [], "beyond_tests": True}
    assert d._task_allowed(tip, suites, {}, row).covered
    old = d._task_allowed(tip, suites, {}, {**row, "name": "old"})
    assert not old.covered and [a.id for a in old.expired] == ["suite old"]
    circuit, mul = {"kind": "circuit-binding", "id": "circuit-binding:x"}, "fold/F01 Mul{n=3}"
    assert d._task_allowed(tip, circuit, {"detail": {"failures": [mul]}}).covered
    for failures in ([mul, "fold/F02 Add"], [mul] * landing.CIRCUIT_NAMED, []):
        assert not d._task_allowed(tip, circuit, {"detail": {"failures": failures}}).covered, failures
    assert not d._task_allowed(tip, {"kind": "lean-module", "id": "lean-module:x"}, {"detail": {"failures": [mul]}}).covered


def test_a_paths_owner_is_its_longest_prefix_in_mains_registry_and_none_where_the_registry_cant_say(world, monkeypatch):
    """`Landing._path_owner`, whom the loop's blame tells: the handle whose `owns` in `main`'s registry is the path's longest
    prefix; none for a path nobody owns, or when the registry isn't JSON, holds no list of handles or a handle without a name."""
    reg = {"handles": [{"handle": "ci", "owns": ["tools/research/"]}, {"handle": "infra", "owns": ["tools/research/src/x/"]}]}
    _move_main(world, {".agents/skills/using-slack/registry.json": json.dumps(reg)})
    h = Harness(world, monkeypatch)
    h.q._clone()
    h.q.fetch()
    assert [h.d._path_owner(p) for p in ("tools/research/src/x/a.py", "tools/research/README.md", "verity/a.py")] == [
        "infra", "ci", None]
    for text in ("not json [", json.dumps({"handles": "ci"}), json.dumps({"handles": [{"owns": ["tools/"]}]})):
        _move_main(world, {".agents/skills/using-slack/registry.json": text})
        h.q.fetch()
        assert Landing(h.q, on="m1")._path_owner("tools/research/README.md") is None, text


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


def test_a_refused_notice_is_told_again_and_one_told_is_not(world, monkeypatch):
    """`tell` keeps what it told (``state["told"]``) only once the post went through, so the next tick tells a refused one
    again; a notice is told unless every one of its keys was; with --dry-run it only logs."""
    h = Harness(world, monkeypatch)
    logged, ok, sent = [], [False], []
    h.q.log = logged.append
    h.d.post = lambda text, to: sent.append((text, to)) or ok[0]
    state = {"told": []}
    h.d.tell(state, ["k"], "#1 conflicts", "circuits")
    assert state["told"] == [] and "could not post" in logged[-1]
    ok[0] = True
    h.d.tell(state, ["k"], "#1 conflicts", "circuits", "lean")
    assert state["told"] == ["k"] and logged[-1].startswith("research queue drive: told @circuits @lean")
    assert sent[-1] == ("@circuits @lean queue: #1 conflicts", ("@circuits", "@lean"))
    h.d.tell(state, ["k"], "#1 conflicts", "circuits")
    assert len(sent) == 2
    h.d.tell(state, ["k", "k3"], "#3 conflicts", "circuits")
    assert sent[-1] == ("@circuits queue: #3 conflicts", ("@circuits",)) and state["told"] == ["k", "k3"]
    h.d.dry_run = True
    h.d.tell(state, ["k2"], "#2 conflicts", None)
    assert len(sent) == 3 and state["told"] == ["k", "k3"] and logged[-1].endswith("would tell @ci: #2 conflicts [dry run]")


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


def test_after_members_reach_main_they_are_caught_up_with_their_run_their_stacked_prs_told_and_the_sweep_keeps_the_closed(
        world, monkeypatch):
    """What follows members reaching `main` (`Landing._on_main`, the loop's after-steps): the PRs stacked on them told
    (`_stacked`), each member's commit and the tip caught up from ``since`` with ``attempt`` as their run (`merge.caught_up`),
    and the PR sweep keeping the branches ``state["stale_closed"]`` names (`merge._sweep_after`)."""
    gh = world.gh
    gh.put(1, h1 := world.change({"a.txt": "1\n"}))
    gh.put(2, world.change({"b.txt": "2\n"}, base=h1), ready=False)
    h = Harness(world, monkeypatch)
    h.q.handle = lambda pr: pr.owner
    swept = []
    monkeypatch.setattr(merge, "_sweep_after", lambda repo, into, remote, keep=(): swept.append((into, remote, keep)))
    since = _main(world)
    above = world.change({"z.txt": "z\n"}, base=h1)
    _git(world.dev, "push", "-q", "origin", f"{above}:refs/heads/main")
    h.q._clone()
    h.q.fetch()
    tip = _main(world)
    state = {"told": [], "stale_closed": {"cursor/w-9": 9}}
    h.d._on_main(state, [Member(1, h1, "lane-1", h1)], since, tip, "r1 r2")
    assert h.after == [(since, {h1: "r1 r2", tip: "r1 r2"})]
    assert swept == [("main", "origin", {"cursor/w-9": 9})]
    assert state["told"] == ["stacked:2<1"]
    assert h.posts == [f"@lane-2 queue: #2's base #1 landed in main at {tip[:12]}; merge main into #2"]


# -- the verdict carry's judgement and a check's auto plan

#: a PR's new head and two older heads, for `_carried_from`, and who marked each head ready
NEW, OLD, OLDER = "n" * 40, "o" * 40, "p" * 40
MARKED = {NEW: "marker", OLD: "old-marker"}


@pytest.mark.parametrize("labels, why, owner, carries, got", [
    ({OLD: [("rev", "land", "T1")]}, "no land verdict", "lane-1", True, (OLD, "only merges", True)),
    ({OLD: [("rev", "land", "T1")]}, "no land verdict", "lane-1", False, (OLD, "edited-merge: only merges", False)),
    ({OLD: [("@LANE-1", "land", "T1"), ("Marker", "land", "T2"), ("old-marker", "land", "T3")]}, "no land verdict", "lane-1",
     True, None),
    ({OLD: [("rev", "land", "T1")]}, "held by top (wait)", "lane-1", True,
     (OLD, f"held by top (wait) at {NEW[:12]}", False)),
    ({OLD: [("rev", "land", "T1")], OLDER: [("@top", "hold", "T2", "wait")]}, "no land verdict", "lane-1", True,
     (OLD, f"top's hold at {OLDER[:12]} (wait) is newer", False)),
    ({OLD: [("rev", "land", "T1")], OLDER: [("lane-1", "hold", "T2", "mine")]}, "no land verdict", "lane-1", True,
     (OLD, f"lane-1's hold at {OLDER[:12]} (mine) is newer", False)),
    ({OLDER: [("rev", "land", "T1")], OLD: [("top", "hold", "T1")]}, "no land verdict", "lane-1", True,
     (OLDER, f"top's hold at {OLD[:12]} (no reason) is newer", False)),
    ({OLD: [("top", "hold", "T1", "x"), ("rev", "land", "T2")]}, "no land verdict", "lane-1", True,
     (OLD, f"held by top (x) at {OLD[:12]}", False)),
    ({OLDER: [("top", "hold", "T1", "x"), ("rev", "land", "T2")]}, "no land verdict", "lane-1", True,
     (OLDER, f"held by top (x) at {OLDER[:12]}", False)),
    ({OLD: [("rev", "land", "T2"), ("rev", "hold", "T1", "x")]}, "no land verdict", "lane-1", True, (OLD, "only merges", True)),
    ({OLD: [("rev", "land", "T1")]}, "no land verdict", "ci", True, (OLD, "@ci is among its owners, so its verdict wouldn't count",
                                                                     False)),
    ({OLDER: [("rev", "land", "T1")], OLD: [("rev2", "land", "T2")]}, "no land verdict", "lane-1", True,
     (OLD, "only merges", True)),
], ids=["carries", "no-carry", "owners-only", "held-here", "newer-hold", "owners-hold", "tie-hold", "standing-hold",
        "standing-hold-older", "reviewers-newest", "ci-owned", "newest-land"])
def test_the_verdict_carry_takes_the_newest_land_at_an_older_head_but_for_a_hold_or_an_owners_own(world, labels, why, owner,
                                                                                                 carries, got):
    """`Landing._carried_from`: of each reviewer's newest verdict at each older head, a hold by anyone or a land by none of the
    PR's owners nor the head's ready mark's author (without case or a leading `@`), the newest decides, a hold over a land of
    the same time.  A hold at the new head (``why``) or a newer one at an older head keeps it, as a hold standing at the land's
    own head does; `CI` among its owners carries nothing; else the land carries when `delta` says so, of the new head over the
    land's, trusting the landed tips and the trusted heads."""
    from research.carry import Delta
    from research.store.model import Label

    asked = []
    d = Landing(q := world.queue(), on="", ready_by=lambda pr, head: MARKED.get(head),
                reviews=lambda pr, head: [Label(f"pr/{pr}@{head}", "reviewed", v, by, ts, *ref)
                                          for by, v, ts, *ref in labels.get(head, ())])
    q.handle = lambda pr: None
    d.delta = lambda old, new, landed, trusted: asked.append((old, new, landed, trusted)) or Delta(
        "merge-only" if carries else "edited-merge", "only merges")
    assert d._carried_from(Member(1, NEW, owner, ""), [OLDER, OLD], why, ["m" * 40], {"t" * 40}) == got
    assert asked == ([(got[0], NEW, ["m" * 40], {"t" * 40})] if got and got[1] in ("only merges", "edited-merge: only merges")
                     else [])


def test_a_landing_reads_verdicts_and_ready_marks_from_the_store_and_whose_verdict_doesnt_count(world):
    """By default a `Landing` reads a PR head's verdicts and who marked it from the queue's store (`StoreLabels`); a PR's
    owners (`_owners`) are its merge's `Owner`, the handle of its open PR's lane (else of that `Owner`) and who marked its
    head."""
    from research.queue import label_target
    from research.store import vocab

    world.with_store_remote()
    head = world.change({"a.txt": "1\n"})
    world.gh.put(1, head, body="Lane `lane-9`.")
    world.store.add_label(label_target(1, head), vocab.REVIEWED, "land", "reviewer", ts="2026-11-10T00:00:00Z")
    world.store.add_label(label_target(1, head), vocab.READY, "true", "marker", ts="2026-11-10T00:00:00Z")
    d = Landing(q := world.queue(), on="")
    q.handle = lambda pr: f"handle-of-{pr.owner}"
    assert [(lab.by, lab.value) for lab in d.reviews(1, head)] == [("reviewer", "land")] and d.ready_by(1, head) == "marker"
    assert d._owners(Member(1, head, "lane-1", "")) == {"lane-1", "handle-of-lane-9", "marker"}
    assert d._owner(Member(2, head, "lane-2", "")) == "handle-of-lane-2"


def test_a_landings_delta_carries_a_merge_of_a_landed_tip_and_not_of_an_untrusted_one(world):
    """`Landing.delta` is `carry.delta` of the new head over the old in the queue's clone, trusting the landed tips given."""
    from test_queue import _merge

    old = world.change({"a.txt": "1\n"})
    _move_main(world, {"m.txt": "moved\n"})
    new = _merge(world, old, main := _main(world))
    q = world.queue()
    q._clone()
    _git(q.workdir, "fetch", "-q", "origin", f"refs/dev/{old}", f"refs/dev/{new}")
    d = Landing(q, on="")
    assert d.delta(old, new, [main], set()).carries
    assert d.delta(old, new, [], set()).kind == "untrusted-parent"


def test_a_landings_delta_trusts_a_merged_in_head_that_isnt_landed_only_when_it_is_given(world):
    """`Landing.delta` hands `carry.trust` the heads given beside the landed tips: a new head that merges another PR's head,
    landed nowhere, carries only when that head is one of them (another PR's marked head with a land verdict and no hold)."""
    from test_queue import _merge

    other, old = world.change({"b.txt": "2\n"}), world.change({"a.txt": "1\n"})
    new = _merge(world, old, other)
    q = world.queue()
    q._clone()
    _git(q.workdir, "fetch", "-q", "origin", f"refs/dev/{old}", f"refs/dev/{new}")
    d = Landing(q, on="")
    assert d.delta(old, new, [_main(world)], {other}).carries
    assert d.delta(old, new, [_main(world)], set()).kind == "untrusted-parent"


def test_an_auto_plan_is_planned_in_the_background_in_a_directory_of_its_own_and_a_landing_launches_it(tmp_path, monkeypatch):
    """`plan_shards` detaches `PLAN_JOB` on the tip's check of the steps given, at most ``max_shards`` shards, into a new
    directory under `LAUNCHES` that its record names (``out``); a `Landing` plans with it and reads the plan with `planned`."""
    import types

    from research.landing import LAUNCHES, PLAN_JOB, plan_shards, planned

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    jobs = []
    monkeypatch.setattr(landing, "detach", lambda repo, argv, env: jobs.append((repo, argv, env["GIT_TERMINAL_PROMPT"])) or
                        {"dir": "job", "pid": 1})
    one, two = plan_shards(tmp_path, "c" * 40, "pytest,ratchet-gate", 3), plan_shards(tmp_path, "c" * 40, "pytest")
    assert one["out"] != two["out"] and all(Path(r["out"]).parent == tmp_path / LAUNCHES for r in (one, two))
    assert jobs == [(tmp_path, ["sh", "-c", PLAN_JOB, "sh", one["out"], "c" * 40, "--deferred", "pytest,ratchet-gate",
                                "--max-shards", "3"], "0"),
                    (tmp_path, ["sh", "-c", PLAN_JOB, "sh", two["out"], "c" * 40, "--deferred", "pytest"], "0")]
    d = Landing(types.SimpleNamespace(auth=(), workdir=tmp_path), on="m1")
    assert (d.plan_shards, d.planned) == (plan_shards, planned)


@pytest.mark.parametrize("rc, plan, got", [
    ("0", {"kind": "check-shard-plan/v1", "n": 1}, ({"kind": "check-shard-plan/v1", "n": 1}, None)),
    ("0", {"kind": "check-shard-plan/v1", "n": 0}, (None, "`check.py --shard-plan` exited 0: planned")),
    ("0", {"kind": "check-shard-plan/v1", "n": "2"}, (None, "`check.py --shard-plan` exited 0: planned")),
    ("0", {"kind": "check-shard-plan/v0", "n": 2}, (None, "`check.py --shard-plan` exited 0: planned")),
    ("0", "not json", (None, "`check.py --shard-plan` exited 0: JSONDecodeError: Expecting value: line 1 column 1 (char 0)")),
    ("0", "", (None, "`check.py --shard-plan` exited 0: JSONDecodeError: Expecting value: line 1 column 1 (char 0)")),
    ("1", None, (None, "`check.py --shard-plan` exited 1: planned")),
    (None, None, (None, None)),
    ("gone", None, (None, "its process is gone with no exit written (a restart of the pod)")),
], ids=["one-shard", "no-shards", "n-not-int", "other-kind", "not-json", "empty", "failed", "running", "gone"])
def test_planned_takes_a_plan_of_a_shard_or_more_once_its_job_exited_0_and_says_why_there_is_none(tmp_path, rc, plan, got):
    """`planned`: the plan its directory's `check-shard-plan.json` holds (check.py's `AUTO_PLAN`, the name each shard's run
    sends it under) once its job exited 0 and it is a `check-shard-plan/v1` of at least one shard; else why there is none,
    or (None, None) while the job runs."""
    from research.landing import AUTO_PLAN, planned

    job, out = tmp_path / "job", tmp_path / "plan"
    for d in (job, out):
        d.mkdir()
    pid = os.getpid()
    if rc == "gone":
        (p := subprocess.Popen(["true"])).wait()
        pid = p.pid
    elif rc is not None:
        (job / "rc").write_text(f"{rc}\n")
        (job / "log").write_text("planned\n")
    if plan is not None:
        (out / AUTO_PLAN).write_text(plan if isinstance(plan, str) else json.dumps(plan))
    assert AUTO_PLAN == "check-shard-plan.json"
    assert planned({"dir": str(job), "pid": pid, "out": str(out)}) == got


#: `git` as on the control pod's NFS TMPDIR, where `worktree remove` stops at a directory a silly-renamed file (one something
#: still held when git unlinked it) keeps from emptying, and exits 255
NFS_GIT = """#!/bin/sh
if [ "$1 $2" = "worktree remove" ]; then
  eval "t=\\${$#}"; echo "error: failed to delete '$t': Directory not empty" >&2; exit 255
fi
exec GIT "$@"
"""


@pytest.mark.parametrize("removal", ["git", "rm", "none"])
@pytest.mark.parametrize("rc", [0, 2])
def test_the_plan_job_plans_in_a_worktree_of_the_tip_and_planned_reads_its_plan(tmp_path, rc, removal):
    """`PLAN_JOB`, as `detach` runs it (here to its end): `check.py --shard-plan` in a worktree of the tip under TRAIN_TREES,
    its venv kept beside, the plan written into the plan's directory, the worktree removed after; `planned` gives the plan,
    or why there is none, and keeps the plan's directory (its shards send the file) but not the job's.  The job exits as
    check.py did however the worktree goes: by `git worktree remove`; by `rm -rf` once that fails as on the control pod
    (`NFS_GIT`); or not at all, when `rm -rf` fails too."""
    from research.landing import AUTO_PLAN, DETACH_JOB, PLAN_JOB, planned

    repo, out, job, trees, bin_ = (tmp_path / x for x in ("repo", "plan", "job", "trees", "bin"))
    for d in (repo, out, job, bin_):
        d.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "tip")
    tip = _git(repo, "rev-parse", "HEAD")
    doc = {"kind": "check-shard-plan/v1", "n": 3, "digest": "d" * 64}
    (bin_ / "uv").write_text("#!/bin/sh\n"
                             f'printf "%s\\n" "$PWD" "$UV_PROJECT_ENVIRONMENT" "$(git rev-parse HEAD)" "$@" > {tmp_path}/argv\n'
                             'while [ "$1" != --out ]; do shift; done\n'
                             f"echo '{json.dumps(doc)}' > \"$2\"\necho 'shard plan: N = 3'\nexit {rc}\n")
    if removal != "git":
        (bin_ / "git").write_text(NFS_GIT.replace("GIT", shutil.which("git")))
    if removal == "none":
        (bin_ / "rm").write_text("#!/bin/sh\nfor a; do t=$a; done\necho \"rm: cannot remove '$t': Directory not empty\" >&2\n"
                                 "exit 1\n")
    for f in bin_.iterdir():
        f.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_}:/usr/bin:/bin", "TRAIN_TREES": str(trees)}
    subprocess.run(["sh", "-c", DETACH_JOB, "sh", str(job), "sh", "-c", PLAN_JOB, "sh", str(out), tip, "--deferred", "pytest",
                    "--max-shards", "3"], cwd=repo, env=env, check=True)
    cwd, venv, head, *argv = (tmp_path / "argv").read_text().splitlines()
    assert (cwd, venv, head) == (str(trees / "plan-plan"), str(trees / "plan-venv"), tip)
    assert argv == ["run", "-q", "python", "tools/verity/check/check.py", "--shard-plan", "--out", str(out / AUTO_PLAN),
                    "--deferred", "pytest", "--max-shards", "3"]
    left = removal == "none"
    assert (trees / "plan-plan").exists() == left and len(_git(repo, "worktree", "list").splitlines()) == 1 + left, \
        "the worktree is removed, and pruned, unless nothing can remove it"
    got = planned({"dir": str(job), "pid": None, "out": str(out)})
    why = "`check.py --shard-plan` exited 2: shard plan: N = 3" + \
        f"\nrm: cannot remove '{trees / 'plan-plan'}': Directory not empty" * left
    assert got == ((doc, None) if rc == 0 else (None, why))
    assert not job.exists() and (out / AUTO_PLAN).is_file()


# -- the loop's modules


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
