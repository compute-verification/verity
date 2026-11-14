"""`check.py --deferred STEP,...`, the after-landing run `research queue drive` launches on `main` (`train.sh --deferred`): only
those ci.toml jobs and what they need, inside a recorded run that is no check, so no gate reads it."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


CHECK = _module("repo_check_for_deferred", ROOT / "tools" / "verity" / "check" / "check.py")


@pytest.fixture(autouse=True)
def _provers(provers):
    return provers(CHECK)


def _main(monkeypatch, tmp_path, argv):
    got = []
    monkeypatch.setenv("RESEARCH_RUN_ID", "r20261018-000000-abcd")
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(tmp_path))
    monkeypatch.delenv("VERITY_SKIP_STORE", raising=False)
    monkeypatch.setattr(CHECK.SUITES, "foreign", lambda members, env: None)
    monkeypatch.setattr(CHECK, "run", lambda out, n, cache, keep_going, only, *packs: got.append(only) or (True, []))
    return CHECK.main(argv), got[0] if got else None


def test_a_deferred_run_runs_only_its_steps_and_what_they_need_inside_a_recorded_run(monkeypatch, tmp_path):
    rc, only = _main(monkeypatch, tmp_path, ["--deferred", "pytest,flock-rust-tests", "--keep-going", "--no-cache"])
    assert rc == 0 and {p.partition(":")[0] for p in only} == {"pytest", "flock-rust-tests", "flock-circuit-build"}
    doc = json.loads((tmp_path / "result.json").read_text())
    assert doc["kind"] == CHECK.DEFERRED_KIND == "check-deferred/v1" and "shard" not in doc
    assert doc["deferred"]["deferred"] == ["pytest", "flock-rust-tests"]


def test_a_recorded_check_runs_every_job_on_the_gate_and_passes_without_circuit_check(monkeypatch, tmp_path):
    """circuit-check and ratchet-gate are off the gate (policy.toml's `off_gate`): a recorded check (`--record`, the declared
    command in a run) leaves them out, and its `check/v1` passes on the jobs on the gate alone."""
    monkeypatch.setenv("CHECK_CACHE", str(tmp_path / "cache" / "check"))
    monkeypatch.setenv("VERITY_TEST_CACHE", str(tmp_path / "cache" / "tests"))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rc, only = _main(monkeypatch, run_dir, ["--keep-going"])
    off = {"circuit-check", "ratchet-gate"}
    assert rc == 0 and only == [p for p in CHECK.shard_parts() if p.partition(":")[0] not in off] == CHECK.gated_parts()
    assert {p.partition(":")[0] for p in only} == {j.name for j in CHECK.JOBS} - off
    doc = json.loads((run_dir / "result.json").read_text())
    assert doc["kind"] == "check/v1" and doc["validation"]["status"] == "passed" and "shard" not in doc


def test_circuit_check_runs_in_the_after_landing_run_and_the_nightly_cold_check(monkeypatch, tmp_path):
    """Off the gate is not unrun: drive's after-landing run passes every job (`--deferred`, drive's `AFTER_LANDING`), and the
    nightly `audit-main` (`--no-cache`) checks every part cold.  Outside a recorded run, `verity check` runs the whole check
    (test_declaration's `test_check_of_parts_runs_only_their_jobs_and_lean_audit_on_every_package`)."""
    every = ",".join(j.name for j in CHECK.JOBS)
    rc, only = _main(monkeypatch, tmp_path, ["--deferred", every, "--keep-going", "--no-cache"])
    assert rc == 0 and only == CHECK.shard_parts() and {"circuit-check", "ratchet-gate"} <= {p.partition(":")[0] for p in only}
    assert json.loads((tmp_path / "result.json").read_text())["kind"] == CHECK.DEFERRED_KIND
    assert _main(monkeypatch, tmp_path, ["--no-cache"]) == (0, None)


def test_a_deferred_run_is_refused_with_a_part_or_a_step_ci_toml_doesnt_declare(monkeypatch, tmp_path, capsys):
    assert _main(monkeypatch, tmp_path, ["--deferred", "pytest", "tests"]) == (2, None)
    assert _main(monkeypatch, tmp_path, ["--deferred", "pytest,nope"]) == (2, None)
    assert "no job nope" in capsys.readouterr().err
    assert _main(monkeypatch, tmp_path, ["--deferred", "pytest", "--shard", "3/3"]) == (2, None)
    assert "--shard 3/3: that shard holds none of those steps" in capsys.readouterr().err


def test_the_lock_runs_bookkeeping_flag_reaches_its_lean_audit_and_nothing_else_takes_it(monkeypatch, tmp_path, capsys):
    """`--bookkeeping-warn` (drive's lock run: `train.sh --deferred lean-audit --bookkeeping-warn`) reaches `run`, whose
    lean-audit step then repairs bookkeeping (`lean_steps`); a check that isn't a deferred, unsharded run that regenerates
    records refuses it, so a PR's own check and the after-landing run fail on bookkeeping as before."""
    got = []
    monkeypatch.setenv("RESEARCH_RUN_ID", "r20261018-000000-abcd")
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(tmp_path))
    monkeypatch.delenv("VERITY_SKIP_STORE", raising=False)
    monkeypatch.setattr(CHECK.SUITES, "foreign", lambda members, env: None)
    monkeypatch.setattr(CHECK, "run", lambda *a, bookkeeping=False: got.append(bookkeeping) or (True, []))
    assert CHECK.main(["--deferred", "lean-audit", "--keep-going", "--bookkeeping-warn"]) == 0 and got == [True]
    assert CHECK.main(["--deferred", "lean-audit", "--keep-going"]) == 0 and got == [True, False]
    for argv in (["--bookkeeping-warn"], ["--deferred", "lean-audit", "--no-regenerate", "--bookkeeping-warn"],
                 ["--deferred", "lean-audit", "--shard", "1/3", "--bookkeeping-warn"]):
        assert CHECK.main(argv) == 2 and len(got) == 2, argv
    assert "--bookkeeping-warn is drive's lock run" in capsys.readouterr().err


def test_a_sharded_deferred_run_runs_those_steps_parts_in_its_shard_alone(monkeypatch, tmp_path):
    """Drive's after-landing run as N runs at once (`after_landing_shards`): shard K runs the deferred steps' parts that ci.toml's
    plan with N shards puts in shard K, and is recorded as a deferred run naming its shard, so the N together cover each step."""
    steps = "pytest,circuit-check,flock-rust-tests"
    rc, only = _main(monkeypatch, tmp_path, ["--deferred", steps, "--shard", "3/3", "--keep-going"])
    whole = CHECK.deferred_parts(steps)
    assert rc == 0 and only == [p for p in CHECK.shard("3/3", gated=False)["parts"] if p in whole]
    assert {p.partition(":")[0] for p in only} == {"circuit-check", "flock-rust-tests", "flock-circuit-build"}
    doc = json.loads((tmp_path / "result.json").read_text())
    assert doc["kind"] == CHECK.DEFERRED_KIND and (doc["deferred"]["k"], doc["deferred"]["n"]) == (3, 3)
    assert doc["deferred"]["deferred"] == steps.split(",")
    rc, only = _main(monkeypatch, tmp_path, ["--deferred", steps, "--shard", "1/3", "--keep-going"])
    assert rc == 0 and {p.partition(":")[0] for p in only} == {"pytest"}


def test_every_shard_of_every_plan_runs_a_step_of_a_check_whose_lean_audit_another_check_lent(monkeypatch, tmp_path):
    """The loop's check of a candidate that takes lean-audit's verdict from the check below (`vqueue_slim.start`'s
    ``audit_from``) defers every step but lean-audit, and a shard holding none of a `--deferred`'s steps is refused (exit 2),
    which the loop took for the machines failing and retried.  So every shard of every plan holds a step that each such check
    sends, every job but lean-audit (`AFTER_LANDING`, those off the gate too): lean-agreement, which a run may skip by name,
    doesn't count."""
    steps = ",".join(j.name for j in CHECK.JOBS if j.name not in ("lean-audit", "lean-agreement"))
    for n in CHECK.DECLARATION.plans():
        for k in range(1, n + 1):
            rc, only = _main(monkeypatch, tmp_path, ["--deferred", steps, "--shard", f"{k}/{n}", "--keep-going"])
            assert rc == 0 and only, f"shard {k}/{n} holds none of {steps}"


def test_the_shards_of_a_deferred_run_together_run_every_part_ratchet_gate_in_the_last_unless_named():
    """ratchet-gate was in no shard of ci.toml's plans: drive's sharded after-landing run never ran it, so drive took each passing
    run as one that passed without ratchet-gate, verified nothing and ended no red record, while `main` moved on its gate.  A
    plan that doesn't name it runs it in its last shard."""
    for n, plan in CHECK.DECLARATION.plans().items():
        got = [CHECK.shard(f"{k}/{n}", gated=False)["parts"] for k in range(1, n + 1)]
        assert sorted(p for s in got for p in s) == sorted(CHECK.DECLARATION.all_parts(CHECK.JOBS, CHECK._parts_of(plan))), n
        named = [k for k, s in enumerate(plan) if "ratchet-gate" in s]
        assert "ratchet-gate" in got[named[0] if named else -1]
        assert all("ratchet-gate" not in CHECK.shard(f"{k}/{n}")["parts"] for k in range(1, n + 1))


def test_record_of_a_deferred_run_launches_it_without_the_check_tool_and_without_lean(monkeypatch):
    """An after-landing run recorded as `--tool check` would be a check of `main`'s tip that `research merge` reads.  It carries
    its base's commit, above which ratchet-gate judges every merge."""
    real, launched, sent = subprocess.run, [], {}

    def run(cmd, *a, **k):
        if cmd[:4] == [sys.executable, "-m", "research", "run"]:
            launched.append(cmd)
            sent.update({Path(cmd[i + 1]).name: Path(cmd[i + 1]).read_text() for i, x in enumerate(cmd) if x == "--send"})
            return subprocess.CompletedProcess(cmd, 0)
        return real(cmd, *a, **k)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(CHECK, "lean_deps_url_file", lambda d: (_ for _ in ()).throw(AssertionError("Lean dependencies sent")))
    monkeypatch.setattr(CHECK, "landable", lambda base: True)
    monkeypatch.setattr(CHECK, "slot_wrap", lambda priority: ["slot", *(["--priority"] if priority else [])])
    monkeypatch.setattr(CHECK, "verdict_url_file", lambda d, **k: None)
    assert CHECK.record(["--base", "HEAD", "--on", "vy-x", "--priority", "--keep-going", "--deferred", "pytest,circuit-check"]) == 0
    [cmd] = launched
    head, tail = cmd[:cmd.index("--")], cmd[cmd.index("--") + 1:]
    assert "--tool" not in head and head[head.index("--send") + 1].endswith(CHECK.SHARD_PARTS)
    assert tail[:2] == ["slot", "--priority"] and tail[-3:] == ["--keep-going", "--deferred", "pytest,circuit-check"]
    assert sent[CHECK.BASE_FILE] == f"{CHECK._rev('HEAD')}\n" and CHECK.RATCHET_BASE_FILE not in sent
    launched.clear()
    sent.clear()
    assert CHECK.record(["--base", "HEAD^", "--ratchet-base", "HEAD", "--on", "vy-x", "--deferred", "ratchet-gate"]) == 0
    [cmd] = launched
    assert "--ratchet-base" not in cmd and "HEAD" not in cmd[cmd.index("--") + 1:]
    assert sent[CHECK.BASE_FILE] == f"{CHECK._rev('HEAD^')}\n" and sent[CHECK.RATCHET_BASE_FILE] == f"{CHECK._rev('HEAD')}\n"


def _pack(tmp_path: Path, monkeypatch) -> Path:
    """A pack of one circuit-check target's pass, one suite's and one Lean package's, as a recorded check's pack holds them."""
    monkeypatch.setenv("CHECK_CACHE", str(tmp_path / "earned" / "check"))
    monkeypatch.setenv("VERITY_TEST_CACHE", str(tmp_path / "earned" / "tests"))
    for p in (CHECK.cache_dir() / "circuit-check" / "v1" / "d" / "t.json", CHECK.cache_dir() / "lean-audit" / "k.json",
              CHECK.VERDICTS.roots()["tests"] / "suite" / "k.json"):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
    CHECK.VERDICTS.export(pack := tmp_path / "gathered.tar.gz")
    return pack


@pytest.mark.parametrize("sent", [False, True])
@pytest.mark.parametrize("circuit_cache", [False, True])
def test_the_after_landing_run_reuses_circuit_checks_passes_alone(monkeypatch, tmp_path, circuit_cache, sent, capsys):
    """Drive's after-landing run (`train.sh --deferred STEPS --no-cache --circuit-cache`): its suites, Lean and steps reuse no
    pass, and circuit-check its per-target ones, imported from the recorded checks' packs sent or a pack given (their entries,
    beside the suites' passes its fresh suites are compared with) and packed again as the run stores them; `--no-cache` alone,
    the nightly `audit-main`, imports only the suites' passes and packs nothing.  The packs' URLs leave its inputs either way,
    and a run sent none fetches none and says nothing of them."""
    pack = _pack(tmp_path, monkeypatch)
    monkeypatch.setenv("CHECK_CACHE", str(tmp_path / "here" / "check"))
    monkeypatch.setenv("VERITY_TEST_CACHE", str(tmp_path / "here" / "tests"))
    run_dir, got = tmp_path / "run", {}
    (run_dir / "inputs").mkdir(parents=True)
    urls = run_dir / "inputs" / CHECK.VERDICT_URLS
    if sent:
        urls.write_text(json.dumps([{"run": "r-earned", "sha256": hashlib.sha256(pack.read_bytes()).hexdigest(),
                                     "bytes": pack.stat().st_size, "url": pack.as_uri()}]))
    monkeypatch.setenv("RESEARCH_RUN_ID", "r20261027-000000-abcd")
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(run_dir))
    monkeypatch.delenv("VERITY_SKIP_STORE", raising=False)
    monkeypatch.delenv(CHECK.VERDICT_URLS_ENV, raising=False)
    monkeypatch.setattr(CHECK.SUITES, "foreign", lambda members, env: None)

    def run(out, n, cache, keep_going, only, packs, reused, regenerate, circuit_cache=None, live=None, splits=None):
        got.update(cache=cache, circuit_cache=circuit_cache, packs=packs is not None, reused=reused is not None)
        return True, []

    monkeypatch.setattr(CHECK, "run", run)
    argv = ["--deferred", "pytest,circuit-check", "--keep-going", "--no-cache", *(["--circuit-cache"] if circuit_cache else []),
            "--no-regenerate", *([] if sent else ["--verdicts-in", str(pack)])]
    assert CHECK.main(argv) == 0
    assert got == {"cache": False, "circuit_cache": circuit_cache, "packs": circuit_cache, "reused": circuit_cache}
    assert sorted(CHECK.VERDICTS.entries()) == [*(["check/circuit-check/v1/d/t.json"] if circuit_cache else []), "tests/suite/k.json"]
    doc = json.loads((run_dir / "result.json").read_text())
    only = [CHECK.CIRCUIT_ENTRIES, CHECK.COMPARED] if circuit_cache else CHECK.COMPARED
    assert [x.get("only") for x in doc["verdicts"]["in"]] == [only]
    assert not urls.exists() and ("recorded checks' packs" in capsys.readouterr().out) == sent


def test_record_sends_the_commit_its_suites_lend_from_only_when_it_is_the_runs_base_or_a_commit_below_it(monkeypatch, capsys):
    """The verified queue names the commit whose check a launched run leaves suites to (`TRAIN_SUITES_FROM`): a `--deferred`
    run that reuses passes carries it (`LEND_FILE`) when it is the run's own base, the candidate a candidate is built on, or
    a commit below that, as an earlier tip of `main` is below the first parent that the check of `main`'s own tip has as its
    base.  Named for any other run (the commit itself, one not below its base, an uncached run), or not named, nothing is
    sent and every suite runs."""
    real, sent = subprocess.run, {}

    def run(cmd, *a, **k):
        if cmd[:4] == [sys.executable, "-m", "research", "run"]:
            sent.update({Path(cmd[i + 1]).name: Path(cmd[i + 1]).read_text() for i, x in enumerate(cmd) if x == "--send"})
            return subprocess.CompletedProcess(cmd, 0)
        return real(cmd, *a, **k)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(CHECK, "landable", lambda base: True)
    monkeypatch.setattr(CHECK, "slot_wrap", lambda priority: ["slot"])
    monkeypatch.setattr(CHECK, "verdict_url_file", lambda d, **k: None)
    head, below = CHECK._rev("HEAD"), CHECK._rev("HEAD^")

    def launch(*argv: str) -> str | None:
        sent.clear()
        assert CHECK.record(["--on", "vy-x", "--priority", "--keep-going", *argv]) == 0
        return sent.get(CHECK.LEND_FILE)

    monkeypatch.delenv("TRAIN_SUITES_FROM", raising=False)
    assert launch("--base", "HEAD^", "--deferred", "pytest") is None
    monkeypatch.setenv("TRAIN_SUITES_FROM", below)
    assert launch("--base", "HEAD^", "--deferred", "pytest") == f"{below}\n" and "lends no suite" not in capsys.readouterr().out
    assert launch("--base", "HEAD^", "--deferred", "pytest", "--no-cache") is None
    assert "lends no suite: only a --deferred run that reuses passes" in capsys.readouterr().out
    monkeypatch.setenv("TRAIN_SUITES_FROM", head)
    assert launch("--base", "HEAD^", "--deferred", "pytest") is None
    assert f"TRAIN_SUITES_FROM {head} lends no suite: it is neither the run's base HEAD^ nor a commit below it" in \
        capsys.readouterr().out
    assert launch("--base", "HEAD", "--deferred", "pytest") is None             # the commit itself, whatever its base
    monkeypatch.setenv("TRAIN_SUITES_FROM", "no-such-ref")
    assert launch("--base", "HEAD^", "--deferred", "pytest") is None
    older = CHECK._rev("HEAD^^")
    monkeypatch.setenv("TRAIN_SUITES_FROM", older)
    assert launch("--base", "HEAD^", "--deferred", "pytest") == f"{older}\n"    # an earlier tip, below its first parent
    assert launch("--base", "HEAD^", "--deferred", "pytest", "--no-cache") is None
    assert launch("--base", "HEAD^^^", "--deferred", "pytest") is None          # above its base
    assert f"TRAIN_SUITES_FROM {older} lends no suite: it is neither the run's base HEAD^^^ nor" in capsys.readouterr().out


def test_a_run_sent_a_lender_hands_it_to_both_suite_steps_unless_uncached(tmp_path):
    """The commit a run was sent as its suites' lender (`LEND_FILE`, among its inputs) reaches `suites.py --lend-from` in the
    pytest and lean-suites steps alike; a run sent none, and an uncached run, which reuses nothing, pass no such flag."""
    def suites(**kw) -> dict[str, str]:
        return {s[0]: " ".join(s[1]) for g in CHECK.groups(tmp_path, 16, **kw) for s in g if s[0] in ("pytest", "lean-suites")}

    assert suites() and not any("--lend-from" in argv for argv in suites().values())
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs" / CHECK.LEND_FILE).write_text("0123abcd\n")
    got = suites()
    assert set(got) == {"pytest", "lean-suites"} and all("--lend-from 0123abcd" in argv for argv in got.values()), got
    assert not any("--lend-from" in argv for argv in suites(cache=False).values())
