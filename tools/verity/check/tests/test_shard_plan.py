"""The shard planner (`shards.py`) and `check.py --shard auto`: tasks packed longest first, N the smallest that reaches the
floor (or the cap's bound), a build kept with what reads it, an auto plan's shards read from the one plan and combined by the
gate's own reading (`research.merge.shard_set`).  Synthetic costs."""
from __future__ import annotations

import importlib.util
import json
import os
import shlex
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


CHECK = _module("repo_check_for_shard_plan", ROOT / "tools" / "verity" / "check" / "check.py")
S = CHECK.SHARDS


@pytest.fixture(autouse=True)
def _provers(provers):
    return provers(CHECK)


def _ts(costs: dict[str, float]) -> list:
    return [S.Task(p, float(s), "measured") for p, s in costs.items()]


def test_lpt_packs_the_longest_first_onto_the_shard_where_it_ends_soonest():
    ts = _ts({"a": 5, "b": 4, "c": 3, "d": 3, "e": 3})
    assert S.lpt(ts, {}, 2) == [{"parts": ["a", "d"], "builds": [], "seconds": 8.0},
                                {"parts": ["b", "c", "e"], "builds": [], "seconds": 10.0}]
    assert [s["parts"] for s in S.lpt(ts, {}, 5)] == [["a"], ["b"], ["c"], ["d"], ["e"]]
    assert S.lpt(ts, {}, 1)[0]["seconds"] == 18.0


def test_the_plans_n_stops_at_the_floor_and_the_cap_bounds_it():
    ts = _ts({"long": 100, **{f"t{i}": 10 for i in range(10)}})
    doc = S.plan(ts, {}, max_shards=8)
    assert (doc["n"], doc["floor_s"], doc["predicted_s"], doc["floor"]) == (2, 100.0, 100.0, ["long"]), \
        "a third shard can't beat the longest task"
    assert S.plan(ts, {}, max_shards=1)["n"] == 1
    even = _ts({f"t{i}": 10 for i in range(40)})
    capped = S.plan(even, {}, max_shards=4)
    assert (capped["n"], capped["predicted_s"], capped["floor_s"]) == (4, 100.0, 10.0), "total / max binds: N is the cap"
    assert S.plan(even, {}, max_shards=4, n=2)["n"] == 2


def test_an_unmeasured_part_costs_its_kinds_median_then_every_measured_parts():
    got = {t.part: (t.seconds, t.source) for t in S.tasks(
        ["pytest:a", "lean-suites:b", "pytest:new", "lean-audit:P", "lean-audit:Q", "x"],
        {"pytest:a": 10, "lean-suites:b": 30, "lean-audit:P": 7})}
    assert got == {"pytest:a": (10, "measured"), "lean-suites:b": (30, "measured"), "pytest:new": (20, "median"),
                   "lean-audit:P": (7, "measured"), "lean-audit:Q": (7, "median"), "x": (10, "median-all")}
    assert S.tasks(["y"], {}) == [S.Task("y", S.DEFAULT_S, "default")]
    assert S.priced(["circuit-check:1/2", "circuit-check:2/2", "pytest:a", "lean-audit:P"], {"a": 3}, {"P": 4},
                    {"circuit-check": 10}) == {"circuit-check:1/2": 5, "circuit-check:2/2": 5, "pytest:a": 3, "lean-audit:P": 4}


def test_a_shard_plan_keeps_each_part_with_its_build_and_charges_a_shard_that_runs_the_build_beside_its_parts():
    needs = {"lean-suites": ["lean-build"], "lean-agreement": ["lean-build"], "lean-build": [], "pytest": []}
    ts = _ts({"lean-build": 30, "lean-suites:a": 50, "lean-suites:b": 50, "lean-agreement": 40, "pytest:x": 60, "pytest:y": 60})
    doc = S.plan(ts, needs, max_shards=3, n=3)
    shards = doc["shards"]
    assert sorted(p for s in shards for p in s["parts"]) == sorted(t.part for t in ts), "every part once"
    [owner] = [s for s in shards if "lean-build" in s["parts"]]
    for s in shards:
        reads = [p for p in s["parts"] if p.partition(":")[0] in ("lean-suites", "lean-agreement")]
        assert bool(reads) == ("lean-build" in s["parts"] or "lean-build" in s["builds"])
        assert "lean-build" not in s["parts"] or "lean-build" not in s["builds"]
        own = sum(doc["tasks"][p][0] for p in s["parts"])
        assert s["seconds"] == own + 30 * ("lean-build" in s["builds"]), "a build beside the parts is charged"
    assert owner["builds"] == []
    assert doc["floor_s"] == 80.0 and doc["floor"][0] == "lean-build" and doc["floor"][1].startswith("lean-suites:")
    assert S.builds(["a", "b", "c"], {"c": ["b"], "b": ["a"]}) == {"a": [], "b": ["a"], "c": ["a", "b"]}, "deepest first"
    assert S.builds(["j:x", "j:y", "k"], {"j:x": ["j:y"], "j": ["k"]}) == {"j:x": ["k", "j:y"], "j:y": ["k"], "k": []}, \
        "a part's own needs (a spec's provers) and its job's"
    assert S.builds(["a", "b"], {"a": ["b"], "b": ["a"]}) == {"a": ["b"], "b": ["a"]}


def test_a_shard_plan_holds_the_readers_of_a_costly_build_on_its_shard_rather_than_running_it_again():
    costly = S.CHEAP_S * 4
    ts = _ts({"proofs": costly, "spec": costly / 2, "small": 10, "x": costly, "y": costly})
    doc = S.plan(ts, {"spec": ["proofs"], "small": ["proofs"]}, max_shards=3, n=3)
    [holder] = [s for s in doc["shards"] if "proofs" in s["parts"]]
    assert holder["parts"] == ["proofs", "spec", "small"] and not any(s["builds"] for s in doc["shards"])
    assert (doc["floor_s"], doc["floor"]) == (costly * 1.5 + 10, ["proofs", "spec", "small"])


def test_a_specs_audit_is_held_with_its_provers_however_cheap_they_are_and_a_cheap_jobs_build_runs_beside():
    """A part's own needs (a spec's audit's provers, which it audits from one build of theirs) hold it on their shard at any
    cost, where a cheap build its job needs runs again beside a reader on another shard."""
    ts = _ts({"lean-audit:p": 10, "lean-audit:s": 100, "lean-audit:t": 100, "lean-build": 10, "lean-suites:a": 100,
              "lean-suites:b": 100})
    needs = {"lean-audit:s": ["lean-audit:p"], "lean-audit:t": ["lean-audit:p"], "lean-suites": ["lean-build"]}
    shards = S.plan(ts, needs, max_shards=4, n=4)["shards"]
    [holder] = [s for s in shards if "lean-audit:p" in s["parts"]]
    assert holder["parts"] == ["lean-audit:p", "lean-audit:s", "lean-audit:t"] and holder["builds"] == []
    assert any(s["builds"] == ["lean-build"] for s in shards), "the cheap build runs again beside a reader"


def _write(tmp_path: Path, doc: dict) -> Path:
    f = tmp_path / CHECK.AUTO_PLAN
    f.write_text(json.dumps(doc))
    return f


def test_an_auto_plan_of_this_tree_runs_each_gated_part_in_one_shard_and_a_stale_plan_is_refused(tmp_path):
    doc = CHECK.auto_plan(({}, {}, {}), 4, n=3)
    want = CHECK.auto_parts()
    assert sorted(doc["all"]) == sorted(want) and not any(p.partition(":")[0] in CHECK.OFF_GATE for p in want)
    assert {"pytest", "lean-suites"}.isdisjoint(want), "the test steps are split by suite"
    assert all(S.suite_part(p)[1] in CHECK.AFTER_LEAN for p in want if p.startswith("lean-suites:"))
    assert [p for p in want if S.suite_part(p)[1] == "verity-core-composition"] == [
        f"lean-suites:verity-core-composition:{k}/3" for k in (1, 2, 3)], "a suite that declares parts by test id is those parts"
    needs = CHECK.shard_needs()
    specs = [f"lean-audit:{p}" for p in ("security", "security/Properties")]
    for spec in specs:
        assert needs[spec] == ["lean-audit:security/Proofs"], "a spec's audit audits its provers beside it"
    for proofs in (S.CHEAP_S * 2, S.CHEAP_S / 2):
        got = CHECK.auto_plan(({}, {"security/Proofs": proofs, "security": 1.0, "security/Properties": 1.0}, {}), 4, n=3)
        [s] = [s for s in got["shards"] if "lean-audit:security/Proofs" in s["parts"]]
        assert set(specs) <= set(s["parts"]), "held on its shard, not built again"
        assert not any(b.startswith("lean-audit:") for s in got["shards"] for b in s["builds"])
    f = _write(tmp_path, doc)
    got = [CHECK.shard(f"auto:{k}/3", plan_file=f) for k in (1, 2, 3)]
    assert sorted(p for g in got for p in g["parts"]) == sorted(want)
    assert all(g["all"] == want and g["plan"] == S.summary(doc) and g["plan"]["digest"] == doc["digest"] for g in got)
    with pytest.raises(CHECK.DECLARATION.DeclarationError, match="it has 3 shards, not 4"):
        CHECK.shard("auto:1/4", plan_file=f)
    stale = json.loads(json.dumps(doc))
    stale["shards"][0]["parts"].pop()
    stale["digest"] = S.digest(stale["shards"])
    with pytest.raises(CHECK.DECLARATION.DeclarationError, match="isn't this tree's|aren't this tree's"):
        CHECK.shard("auto:1/3", plan_file=_write(tmp_path, stale))
    forged = {**doc, "digest": "0" * 64}
    with pytest.raises(CHECK.DECLARATION.DeclarationError, match="its digest isn't its shards'"):
        CHECK.shard("auto:1/3", plan_file=_write(tmp_path, forged))
    os.unlink(f)
    with pytest.raises(CHECK.DECLARATION.DeclarationError, match="no auto plan"):
        CHECK.shard("auto:1/3")


def test_an_auto_shard_runs_its_suites_alone_and_the_build_they_need(tmp_path):
    assert (CHECK.FLOCK_LEAN / "lakefile.toml").exists()
    suite = next(s.name for s in CHECK.SUITES.discover() if s.name not in CHECK.AFTER_LEAN)
    lean, other = CHECK.AFTER_LEAN[0], CHECK.AFTER_LEAN[1]
    gs = CHECK.groups(tmp_path, 16, only=[f"pytest:{suite}", f"lean-suites:{lean}"])
    by = {name: argv for g in gs for name, argv, _ in g}
    assert set(by) == {"pytest", "lean-build", "lean-suites"}
    assert by["pytest"][2:by["pytest"].index("--json")] == [suite]
    words = shlex.split(by["lean-suites"][2])
    assert lean in words and other not in words


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "-c", "commit.gpgsign=false", *args], cwd=repo,
                          check=True, capture_output=True, text=True).stdout.strip()


COMMAND = ("uv", "run", "python", "tools/verity/check/check.py")
CI = ('schema = "ci/v0"\n' + "".join(f'[[job]]\nname = "{n}"\nowner = "ci"\ngroup = {g}\n{extra}'
                                     for n, g, extra in (("pytest", 1, ""), ("lean-build", 2, ""),
                                                         ("lean-suites", 2, 'needs = ["lean-build"]\n'))))


def _auto_shards(tmp_path: Path, monkeypatch, edit=None) -> tuple:
    """(the store, the scratch tree, its commit, the run ids) of an auto plan's two shards of a scratch tree, each written by
    `check.write_result` as a passing shard and stored as a passing attempt; ``edit(k, shard)`` changes a shard's record first."""
    from research.store.local import LocalStore

    tree = tmp_path / "tree"
    (tree / "tools" / "verity" / "check").mkdir(parents=True)
    (tree / "tools" / "verity" / "check" / "ci.toml").write_text(CI)
    _git(tree, "init", "-q", "-b", "main")
    _git(tree, "add", "-A")
    _git(tree, "commit", "-q", "-m", "t")
    head = _git(tree, "rev-parse", "HEAD")
    monkeypatch.setattr(CHECK, "ROOT", tree)
    store = LocalStore(tmp_path / "store", config_path=tmp_path / "no-remote.toml")
    doc = S.plan(_ts({"lean-build": 100, "lean-suites:a": 50, "lean-suites:b": 50, "pytest:x": 80, "pytest:y": 70}),
                 {"lean-suites": ["lean-build"]}, max_shards=2, n=2)
    assert any(s["builds"] == ["lean-build"] for s in doc["shards"]), "one shard runs the build beside its parts"
    ids = []
    for k, s in enumerate(doc["shards"], 1):
        rid = f"r20261031-000000-{k:04x}"
        monkeypatch.setenv("RESEARCH_RUN_ID", rid)
        sh = {"k": k, "n": 2, "parts": s["parts"], "all": doc["all"], "plan": S.summary(doc)}
        if edit:
            edit(k, sh)
        jobs = {p.partition(":")[0] for p in [*s["parts"], *s["builds"]]}
        done = [{"step": "preflight-lock", "status": "passed", "seconds": 1.0},
                *({"step": j, "status": "passed", "rc": 0, "seconds": 2.0} for j in ("pytest", "lean-build", "lean-suites") if j in jobs)]
        run = tmp_path / f"run-{k}"
        run.mkdir()
        CHECK.write_result(run, True, done, shard=sh)
        meta = json.loads((run / "result.json").read_text())
        art = store.put_artifact(meta["kind"], meta)
        store.put_attempt({
            "schema": "research/attempt/v0.1", "id": rid, "tool": {"name": "check", "version": "1"}, "derivation": "drv:" + "0" * 64,
            "reuse_key": None, "argv": [*COMMAND, "--shard", f"auto:{k}/2"], "params": {}, "key_params": [], "inputs": {},
            "conditions": {}, "key_conditions": [], "source": {"commit": head, "dirty": False}, "closure": None, "execution": {},
            "cost": None, "status": {"state": "done", "rc": 0}, "outputs": {"result": art}, "telemetry": None})
        ids.append(rid)
    return store, tree, head, ids, doc


def test_combine_accepts_an_auto_plans_shards_with_the_builds_they_ran_beside_their_parts(tmp_path, monkeypatch):
    from research.merge import shard_set

    store, tree, head, ids, doc = _auto_shards(tmp_path, monkeypatch)
    steps, why = shard_set(store, tree, head, ids, "check", COMMAND)
    assert why is None
    assert [s["step"] for s in steps] == ["preflight-lock", *doc["all"]]
    owner = next(k for k, s in enumerate(doc["shards"], 1) if "lean-build" in s["parts"])
    assert next(s for s in steps if s["step"] == "lean-build")["run_id"] == ids[owner - 1], "the build's part is its owner's"


def test_combine_refuses_auto_shards_of_two_plans_and_takes_a_hand_plans_builds_beside_its_parts(tmp_path, monkeypatch):
    """A hand-written plan's shard runs lean-build beside lean-suites' parts as an auto plan's does: the build's verdict is its
    own shard's."""
    from research.merge import shard_set

    def other_plan(k, sh):
        if k == 2:
            sh["plan"] = {**sh["plan"], "digest": "f" * 64}

    store, tree, head, ids, _ = _auto_shards(tmp_path / "two", monkeypatch, other_plan)
    assert shard_set(store, tree, head, ids, "check", COMMAND) == (None, "the shards are of different plans (their N, every part "
                                                                  "they say an unsharded check covers, or their auto plan's digest)")

    def no_plan(k, sh):
        del sh["plan"]

    store, tree, head, ids, doc = _auto_shards(tmp_path / "hand", monkeypatch, no_plan)
    steps, why = shard_set(store, tree, head, ids, "check", COMMAND)
    owner = next(k for k, s in enumerate(doc["shards"], 1) if "lean-build" in s["parts"])
    assert why is None and next(s for s in steps if s["step"] == "lean-build")["run_id"] == ids[owner - 1]


def test_record_sends_an_auto_shard_its_plan_as_the_runs_input_and_refuses_one_without(tmp_path, monkeypatch):
    """Drive's `after_landing_shards = "auto"`: `--record --shard auto:K/N --shard-plan-file PLAN --on M` launches shard K of
    PLAN with PLAN sent as the run's `AUTO_PLAN` input, where the run's own `--shard auto:K/N` finds it, and with the parts of
    that shard alone; an auto shard without a plan, a plan without an auto shard or --on, and a plan of other steps are
    refused before anything launches."""
    import research.store.local

    real, launched = subprocess.run, []

    def run(cmd, *a, **k):
        if cmd[:4] == [sys.executable, "-m", "research", "run"]:
            sent = {Path(cmd[i + 1]).name: Path(cmd[i + 1]).read_text() for i, x in enumerate(cmd) if x == "--send"}
            launched.append((cmd, sent))
            return subprocess.CompletedProcess(cmd, 0)
        return real(cmd, *a, **k)

    monkeypatch.setattr(research.store.local, "LocalStore", lambda: None)
    monkeypatch.setattr(CHECK, "landable", lambda base: True)
    monkeypatch.setattr(CHECK, "lean_deps_url_file", lambda d: None)
    monkeypatch.setattr(CHECK, "verdict_url_file", lambda d, **k: None)
    monkeypatch.setattr(subprocess, "run", run)
    steps = "pytest,circuit-check"
    doc = CHECK.auto_plan(({}, {}, {}), 4, deferred=steps, n=2)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(doc))
    shard = ["--deferred", steps, "--shard", "auto:2/2"]
    assert CHECK.record(["--base", "HEAD", "--on", "vy-x", *shard, "--shard-plan-file", str(plan)]) == 0
    [(cmd, sent)] = launched
    assert cmd[-len(shard):] == shard and "--shard-plan-file" not in cmd, cmd
    assert json.loads(sent[CHECK.AUTO_PLAN]) == doc and json.loads(sent[CHECK.SHARD_PARTS]) == doc["shards"][1]["parts"]
    launched.clear()
    for argv in (["--on", "vy-x", *shard], ["--on", "vy-x", "--deferred", steps, "--shard", "2/2", "--shard-plan-file", str(plan)],
                 [*shard, "--shard-plan-file", str(plan)], ["--on", "vy-x", "--deferred", "pytest", "--shard", "auto:2/2",
                                                            "--shard-plan-file", str(plan)]):
        assert CHECK.record(["--base", "HEAD", *argv]) == 2 and not launched, argv
