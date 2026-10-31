"""`check.py --shard K/N`: ci.toml's plans run every part of an unsharded check exactly once (but ratchet-gate's, off the gate and
in no plan), each shard holds what its jobs need, a shard runs only its parts (lean-audit on its packages), and its result names
what it ran."""
from __future__ import annotations

import importlib.util
import json
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


CHECK = _module("repo_check_for_shards", ROOT / "tools" / "verity" / "check" / "check.py")
D = CHECK.DECLARATION
LA = _module("repo_lean_audit_for_shards", ROOT / "tools" / "verity" / "lean" / "lean_audit.py")
PACKAGES = LA.packages(CHECK.tracked_files(ROOT))
SECURITY = ("lean-audit:Security", "lean-audit:Security/Proofs")


def _resolved(n: int, packages=PACKAGES) -> list[list[str]]:
    return D.resolve(D.plans()[n], CHECK.JOBS, {"lean-audit": packages})


def test_the_plans_are_the_documented_ones():
    assert D.plans() == {
        3: ((*SECURITY, "pytest"),
            ("lean-build", "lean-unit-cut", "lean-audit:*", "lean-suites", "lean-agreement", "lean-changed"),
            ("flock-circuit-build", "circuit-check", "flock-rust-tests")),
        4: ((*SECURITY, "circuit-check:1/4"),
            ("lean-build", "lean-unit-cut", "lean-audit:*", "lean-suites", "lean-agreement", "lean-changed", "circuit-check:2/4"),
            ("pytest", "circuit-check:3/4"), ("flock-circuit-build", "flock-rust-tests", "circuit-check:*")),
        5: (SECURITY, ("lean-audit:*", "lean-changed"), ("lean-build", "lean-unit-cut", "lean-suites", "lean-agreement"),
            ("pytest",), ("flock-circuit-build", "circuit-check", "flock-rust-tests"))}


@pytest.mark.parametrize("n", sorted(D.plans()))
def test_every_plan_runs_every_part_of_this_tree_exactly_once_and_the_same_way_each_time(n):
    """Every part but ratchet-gate's, which is off the gate and in no plan: circuit-check, off the gate too, stays planned."""
    every = D.all_parts(CHECK.JOBS, {"lean-audit": PACKAGES})
    assert len(PACKAGES) >= 4 and all(every.count(s) == 1 for s in SECURITY) and "lean-audit" not in every
    assert {p.partition(":")[0] for p in every} == {j.name for j in CHECK.JOBS}
    shards = _resolved(n)
    flat = [p for s in shards for p in s]
    planned = [p for p in every if p != "ratchet-gate"]
    assert len(shards) == n and all(shards) and sorted(flat) == sorted(planned) and len(set(flat)) == len(flat)
    assert _resolved(n) == shards == _resolved(n, list(reversed(PACKAGES))), "deterministic, whatever order git lists"
    assert all(s == [p for p in every if p in s] for s in shards), "each shard's parts in the unsharded check's order"


def test_a_package_the_plan_doesnt_name_goes_to_the_shard_with_the_rest():
    new = [*PACKAGES, "protocols/new/lean"]
    for n in D.plans():
        [holder] = [s for s in _resolved(n, new) if "lean-audit:protocols/new/lean" in s]
        assert "lean-audit:verity/core/service/flock" in holder and not set(SECURITY) & set(holder)


def test_a_plan_naming_a_package_the_tree_lacks_stops_the_shard():
    with pytest.raises(D.DeclarationError, match="Proofs, which this tree doesn't have"):
        _resolved(3, [p for p in PACKAGES if not p.endswith("Proofs")])


JOBS = [D.Job("a", "ci", 1), D.Job("b", "ci", 1, needs=("a",)), D.Job("lean-audit", "lean", 2), D.Job("c", "ci", 3)]


def _plans(shards: str):
    return D.parse_plans(f'schema = "ci/v0"\n[[plan]]\nshards = {shards}\n', JOBS)


def test_a_well_formed_plan_parses_and_resolves():
    plan = _plans('[["a", "b", "lean-audit:x"], ["lean-audit:*", "c"]]')[2]
    assert D.resolve(plan, JOBS, {"lean-audit": ["y", "x", "z"]}) == [["a", "b", "lean-audit:x"], ["lean-audit:y", "lean-audit:z", "c"]]


def test_a_plan_may_leave_out_a_job_off_the_gate_but_no_job_on_it():
    jobs = [*JOBS, D.Job("d", "ci", 3, gate=False)]

    def plans(shards: str):
        return D.parse_plans(f'schema = "ci/v0"\n[[plan]]\nshards = {shards}\n', jobs)

    left_out = plans('[["a", "b", "lean-audit:*"], ["c"]]')[2]
    assert D.resolve(left_out, jobs, {"lean-audit": ["x"]}) == [["a", "b", "lean-audit:x"], ["c"]]
    kept = plans('[["a", "b", "lean-audit:*"], ["c", "d"]]')[2]
    assert D.resolve(kept, jobs, {"lean-audit": ["x"]}) == [["a", "b", "lean-audit:x"], ["c", "d"]]
    with pytest.raises(D.DeclarationError, match="no shard runs c"):
        plans('[["a", "b", "lean-audit:*"], ["d"]]')
    with pytest.raises(D.DeclarationError, match="d is in more than one shard"):
        plans('[["a", "b", "lean-audit:*", "d"], ["c", "d"]]')


@pytest.mark.parametrize("shards, why", [
    ('[["a", "b", "lean-audit:*"]]', "two or more"),
    ('[["a", "b"], []]', "two or more"),
    ('[["a", "b", "lean-audit:*"], ["x"]]', "'x' names no declared job"),
    ('[["a", "b", "lean-audit:*"], ["c:1"]]', "only lean-audit, circuit-check split"),
    ('[["a", "b", "lean-audit:*"], ["lean-audit:"]]', "only lean-audit, circuit-check split"),
    ('[["a", "b", "lean-audit:*"], ["a", "c"]]', "a is in more than one shard"),
    ('[["a", "b", "lean-audit:*"], ["lean-audit", "c"]]', "lean-audit is in more than one shard"),
    ('[["a", "b"], ["lean-audit:*"]]', "no shard runs c"),
    ('[["a", "b", "lean-audit:x"], ["c"]]', "exactly one lean-audit:*"),
    ('[["a", "b", "lean-audit:*"], ["lean-audit:*", "c"]]', "exactly one lean-audit:*"),
    ('[["a", "b", "lean-audit:*", "lean-audit:x"], ["lean-audit:x", "c"]]', "named once each"),
    ('[["a", "lean-audit:*"], ["b", "c"]]', "shard 2 runs b, which needs a in the same shard"),
    ('[["a", "a", "b", "lean-audit:*"], ["c"]]', "names a selector twice"),
    ('[["a", "b", 1], ["c"]]', "a selector must be a string"),
])
def test_a_plan_that_could_leave_a_part_out_or_run_it_twice_is_refused(shards, why):
    with pytest.raises(D.DeclarationError, match=why):
        _plans(shards)


def test_a_shard_holds_the_job_its_job_runs_after():
    """lean-suites runs `after` lean-build, in another group, on the verifier lean-build built: a shard without it would run
    the suites on no verifier, so the plan is refused."""
    jobs = [*JOBS, D.Job("d", "ci", 4, after="a")]
    plan = 'schema = "ci/v0"\n[[plan]]\nshards = {}\n'
    assert D.parse_plans(plan.format('[["a", "b", "d", "lean-audit:*"], ["c"]]'), jobs)
    with pytest.raises(D.DeclarationError, match="shard 2 runs d, which needs a in the same shard"):
        D.parse_plans(plan.format('[["a", "b", "lean-audit:*"], ["c", "d"]]'), jobs)


def test_two_plans_of_one_size_and_a_need_outside_its_group_are_refused():
    plan = '[[plan]]\nshards = [["a", "b", "lean-audit:*"], ["c"]]\n'
    with pytest.raises(D.DeclarationError, match="another plan has 2 shards"):
        D.parse_plans('schema = "ci/v0"\n' + plan + plan, JOBS)
    with pytest.raises(D.DeclarationError, match="needs \\['c'\\], which aren't earlier jobs of its group"):
        D.parse('schema = "ci/v0"\n[[job]]\nname = "c"\nowner = "ci"\ngroup = 1\n[[job]]\nname = "a"\nowner = "ci"\ngroup = 2\n'
                'needs = ["c"]\n', {"ci"})


def test_the_jobs_that_read_lean_builds_or_the_rust_builds_output_need_it():
    assert {j.name: j.needs for j in CHECK.JOBS if j.needs} == {
        "flock-rust-tests": ("flock-circuit-build",), "lean-unit-cut": ("lean-build",), "lean-agreement": ("lean-build",)}
    assert {j.name: j.after for j in CHECK.JOBS if j.after} == {"lean-suites": "lean-build"}


@pytest.mark.parametrize("spec", ["0/3", "4/3", "1/2", "1/3/4", "x", "3"])
def test_a_shard_no_plan_declares_is_refused(spec):
    with pytest.raises(D.DeclarationError, match="ci.toml's plans are K/3, K/4, K/5"):
        CHECK.shard(spec)


@pytest.mark.parametrize("n", sorted(D.plans()))
def test_a_shard_runs_its_parts_alone_in_their_groups_order_and_lean_audit_on_its_packages(tmp_path, n):
    for k, planned in enumerate(_resolved(n), 1):
        parts = [p for p in planned if p.partition(":")[0] not in CHECK.OFF_GATE]
        assert CHECK.shard(f"{k}/{n}") == {"k": k, "n": n, "parts": parts, "all": D.all_parts(D.gated(CHECK.JOBS), {"lean-audit": PACKAGES})}
        gs = CHECK.groups(tmp_path, 16, only=parts)
        ran = [name for g in gs for name, _, _ in g]
        assert sorted(ran) == sorted({p.partition(":")[0] for p in parts})
        for g in gs:
            names = [name for name, _, _ in g]
            assert names == [j.name for j in CHECK.JOBS if j.name in names], "in ci.toml's order"
            for j in CHECK.JOBS:
                if j.name in names:
                    assert all(need in names[:names.index(j.name)] for need in j.needs)
        audit = [argv for g in gs for name, argv, _ in g if name == "lean-audit"]
        want = [p.partition(":")[2] for p in parts if p.startswith("lean-audit:")]
        assert [[argv[i + 1] for i, x in enumerate(argv) if x == "--package"] for argv in audit] == ([want] if want else [])


@pytest.mark.parametrize("n", sorted(D.plans()))
def test_a_shard_skips_circuit_check_which_its_plan_names_and_the_shards_cover_every_job_on_the_gate(n):
    def cc(parts):
        return [p for p in parts if p.partition(":")[0] == "circuit-check"]

    shards = [CHECK.shard(f"{k}/{n}") for k in range(1, n + 1)]
    assert any(cc(s) for s in _resolved(n)), "the plan keeps it, so putting it back on the gate is one line"
    assert not any(cc(s["parts"]) for s in shards)
    assert sorted(p for s in shards for p in s["parts"]) == sorted(shards[0]["all"]) == sorted(CHECK.gated_parts())
    assert not cc(shards[0]["all"])


def test_a_shard_without_the_other_half_gets_every_core(tmp_path, monkeypatch):
    """circuit-check gets its cores whatever the memory: its own admission by measured peaks bounds the memory, and a flat 160 GB
    per worker would hold a cold `--all` in a 560 GB slot to 5 workers for over 2 h."""
    monkeypatch.setattr(CHECK, "memory_gb", lambda: 160)

    def jobs(n, k):
        gs = CHECK.groups(tmp_path, 16, only=_resolved(n)[k - 1])
        return {name: argv[argv.index("--jobs") + 1] for g in gs for name, argv, _ in g if name in ("pytest", "circuit-check")}

    assert jobs(5, 4) == {"pytest": "16"} and jobs(5, 5) == {"circuit-check": "16"} and jobs(3, 1) == {"pytest": "16"}
    assert {name: argv[argv.index("--jobs") + 1] for g in CHECK.groups(tmp_path, 16) for name, argv, _ in g
            if name in ("pytest", "circuit-check")} == {"pytest": "8", "circuit-check": "8"}


def test_lean_audit_audits_only_the_packages_it_is_given():
    assert LA.select(PACKAGES, None) == PACKAGES
    assert LA.select(PACKAGES, ["verity/core/protocols/pous/lean", "Security"]) == ["Security", "verity/core/protocols/pous/lean"]
    with pytest.raises(ValueError, match="no audited package nowhere"):
        LA.select(PACKAGES, ["nowhere"])


def test_a_shards_lean_slot_counts_only_its_own_packages(tmp_path, monkeypatch):
    class Audit:
        FLAGS = ()
        packages = staticmethod(lambda files: PACKAGES)
        key = staticmethod(lambda p, files, flags: p.replace("/", "-"))

    monkeypatch.setenv("CHECK_CACHE", str(tmp_path))
    monkeypatch.setattr(CHECK, "_lean_audit", Audit)
    (tmp_path / "lean-audit").mkdir()
    for p in PACKAGES[1:]:
        (tmp_path / "lean-audit" / f"{Audit.key(p, (), ())}.json").write_text("{}")
    assert CHECK.lean_audit_misses() and CHECK.lean_audit_misses(only=PACKAGES[:1])
    assert not CHECK.lean_audit_misses(only=PACKAGES[1:])
    cmd = ["python", "lean_audit.py", "--package", PACKAGES[1]]
    assert CHECK.lean_slot(cmd) == cmd and CHECK.lean_slot(["python", "lean_audit.py"])[:len(CHECK.LEAN_SLOT_WRAP)] == CHECK.LEAN_SLOT_WRAP


def test_a_shards_result_names_its_plan_commit_config_and_each_steps_parts(tmp_path, monkeypatch):
    from research import result as RES
    from research.merge import SHARD_KIND

    monkeypatch.setenv("RESEARCH_RUN_ID", "r20260927-010000-abcd")
    sh = CHECK.shard("2/5")
    done = [{"step": "preflight-lock", "status": "passed", "seconds": 1.0},
            {"step": "lean-audit", "status": "passed", "rc": 0, "seconds": 2.0},
            {"step": "lean-changed", "status": "passed", "rc": 0, "seconds": 3.0}]
    CHECK.write_result(tmp_path, True, done, shard=sh)
    r = RES.check(tmp_path, "r20260927-010000-abcd")
    assert r.valid, r.errors
    assert CHECK.SHARD_KIND == SHARD_KIND == r.doc["kind"] != "check/v1"
    head = subprocess.run(["git", "rev-parse", "HEAD", "HEAD:tools/verity/check"], cwd=ROOT, capture_output=True, text=True).stdout.split()
    assert r.doc["shard"] == {**sh, "commit": head[0], "config": head[1]}
    assert [s.get("parts") for s in r.doc["steps"]] == [None, [p for p in sh["parts"] if p.startswith("lean-audit:")], ["lean-changed"]]
    CHECK.write_result(tmp_path, True, done[:1])
    assert json.loads((tmp_path / "result.json").read_text())["kind"] == "check/v1", "an unsharded check is as it was"


def _combine_tree(tmp_path: Path, monkeypatch, steps: list[dict], store) -> tuple[Path, str, list]:
    """A clean one-commit checkout with a check tool, `shard_set` answering ``steps`` for the shards r1 and r2 (and refusing
    any other set), ``store`` as the store, and a recorded run's id: (tree, its commit, what `shard_set` was asked)."""
    import research.merge
    import research.store.local

    tree = tmp_path / "tree"
    (tree / "tools" / "verity" / "check").mkdir(parents=True)
    (tree / "tools" / "verity" / "check" / "tool.py").write_text(
        "from research.store.tool import Tool\nCHECK = Tool(name='check', version='1', command=['run', 'check'])\n")
    for args in (["init", "-q", "-b", "main"], ["add", "-A"], ["commit", "-q", "-m", "t"]):
        subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "-c", "commit.gpgsign=false", *args], cwd=tree, check=True,
                       capture_output=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tree, capture_output=True, text=True).stdout.strip()
    asked = []

    def shard_set(store, repo, commit, ids, tool, command):
        asked.append((repo, commit, ids, tool, command))
        return (steps, None) if ids == ["r1", "r2"] else (None, "shard 2/2 is not among them")

    monkeypatch.setattr(CHECK, "ROOT", tree)
    monkeypatch.setattr(research.merge, "shard_set", shard_set)
    monkeypatch.setattr(research.store.local, "LocalStore", store)
    monkeypatch.setenv("RESEARCH_RUN_ID", "r20261004-000000-ffff")
    return tree, head, asked


def test_combine_records_the_steps_the_gates_reading_of_the_shards_gives_and_fails_what_it_refuses(tmp_path, monkeypatch):
    """Inside a recorded run, a `check/v1` of `shard_set`'s steps (which the gate compares with its own reading); a failed one when
    the set is refused or the checkout is dirty; outside a run it only says."""
    from research import result as RES

    class Store:
        remote = None

    steps = [{"step": "preflight-lock", "status": "passed", "shards": ["r1", "r2"]},
             {"step": "pytest", "status": "passed", "shard": 1, "run_id": "r1", "seconds": 1.0}]
    tree, head, asked = _combine_tree(tmp_path, monkeypatch, steps, Store)
    run = tmp_path / "run"
    run.mkdir()
    assert CHECK.combine(["r1", "r2"], run) == 0
    assert asked == [(tree, head, ["r1", "r2"], "check", ("run", "check"))]
    r = RES.check(run, "r20261004-000000-ffff")
    assert r.valid, r.errors
    assert (r.doc["kind"], r.validation_status, r.doc["steps"], r.doc["shards"]) == ("check/v1", "passed", steps, ["r1", "r2"])
    assert CHECK.combine(["r1"], run) == 1
    doc = json.loads((run / "result.json").read_text())
    assert doc["validation"] == {"status": "failed", "checker": "tools/verity/check/check.py --combine", "detail": "shard 2/2 is not among them"}
    (tree / "tools" / "verity" / "check" / "tool.py").write_text("# dirty\n" + (tree / "tools" / "verity" / "check" / "tool.py").read_text())
    assert CHECK.combine(["r1", "r2"], None) == 1


def test_circuit_check_s_parts_are_its_shards_and_a_check_runs_only_those_it_holds(tmp_path):
    """circuit-check's parts are its `shards`, ``circuit-check:K/4``: a check that holds all four (unsharded, the 3- and 5-pod
    plans) runs `--all` whole; the 4-pod plan gives each pod one (in the after-landing run's shards, as a recorded check skips
    circuit-check), run as ``--shard K/4`` into ``CC_SHARDS/K``; a check that holds some but not all runs each in turn and fails
    when any does."""
    [cc] = [j for j in CHECK.JOBS if j.name == "circuit-check"]
    assert cc.shards == 4 and D.job_parts(cc, {}) == ["1/4", "2/4", "3/4", "4/4"]
    every = D.all_parts(CHECK.JOBS, {"lean-audit": PACKAGES})
    assert [p for p in every if p.startswith("circuit-check")] == [f"circuit-check:{k}/4" for k in range(1, 5)]

    def argv(only):
        return next(argv for g in CHECK.groups(tmp_path, 16, only=only) for name, argv, _ in g if name == "circuit-check")

    for whole in (argv(None), argv(_resolved(3)[2]), argv(_resolved(5)[4])):
        assert whole[2:4] == ["circuit_check", "--all"] and "--shard" not in whole and "--shard-dir" not in whole
    for k, parts in enumerate(_resolved(4), 1):
        assert [p for p in parts if p.startswith("circuit-check")] == [f"circuit-check:{k}/4"]
        after_landing, gated = CHECK.shard(f"{k}/4", gated=False)["parts"], CHECK.shard(f"{k}/4")["parts"]
        assert [p for p in after_landing if p.startswith("circuit-check")] == [f"circuit-check:{k}/4"]
        assert not [p for p in gated if p.startswith("circuit-check")]
        a = argv(parts)
        assert a[2:4] == ["circuit_check", "--all"] and a[-4:] == ["--shard", f"{k}/4", "--shard-dir", str(tmp_path / CHECK.CC_SHARDS / str(k))]
    script = argv(["circuit-check:1/4", "circuit-check:3/4"])
    assert script[:2] == ["bash", "-c"] and script[2].startswith("rc=0; ") and script[2].endswith("; exit $rc")
    runs = [shlex.split(c.removesuffix(" || rc=$?")) for c in script[2].removeprefix("rc=0; ").removesuffix("; exit $rc").split("; ")]
    assert [r[-4:] for r in runs] == [["--shard", f"{k}/4", "--shard-dir", str(tmp_path / CHECK.CC_SHARDS / str(k))] for k in (1, 3)]
    assert CHECK.circuit_shards(None) is None and CHECK.circuit_shards(["pytest"]) is None
    assert CHECK.circuit_shards([f"circuit-check:{k}/4" for k in range(1, 5)]) is None


def test_combine_joins_circuit_check_s_shards_and_fails_unless_they_join_to_the_unsharded_pass(tmp_path, monkeypatch):
    """With circuit-check's parts in more than one shard, the combine reads each shard's `CC_SHARDS` from its Attempt's run
    files and gives them to `circuit-check --combine`, and fails when that fails (a by-calls root whose callee another shard
    checked, a known failure that no longer fails) or refuses them; a shard that held every part ran `--all` whole, and nothing
    is joined."""
    from research import result as RES

    fetched, joined, rc = [], [], [0]

    class Store:
        remote = None

        def get_attempt(self, rid):
            return {"outputs": {"run_files": f"art:{rid}"} if rid != "lost" else {}}

        def fetch(self, art, to=None, *, paths=None):
            fetched.append((art, paths))
            return to

    def combine(dirs, out):
        joined.append([d.relative_to(out.parent) for d in dirs])
        why = {1: "FAIL by-calls/callee-failed Root_v1\n", 2: "circuit-check --combine: refused: shard(s) 4 of 4 missing\n"}
        return subprocess.CompletedProcess([], rc[0], "", why.get(rc[0], ""))

    monkeypatch.setattr(CHECK, "circuit_combine", combine)
    steps = [{"step": "preflight-lock", "status": "passed", "shards": ["r1", "r2"]}] + [
        {"step": f"circuit-check:{k}/4", "status": "passed", "shard": 1 + (k > 2), "run_id": "r1" if k <= 2 else "r2"}
        for k in range(1, 5)]
    _combine_tree(tmp_path, monkeypatch, steps, Store)
    run = tmp_path / "run"
    run.mkdir()
    assert CHECK.combine(["r1", "r2"], run) == 0
    assert fetched == [("art:r1", [CHECK.CC_SHARDS]), ("art:r2", [CHECK.CC_SHARDS])]
    assert joined == [[Path(r, CHECK.CC_SHARDS, str(k)) for r, k in (("r1", 1), ("r1", 2), ("r2", 3), ("r2", 4))]]
    assert RES.check(run, "r20261004-000000-ffff").validation_status == "passed"
    for rc[0], why in ((1, "circuit-check --combine of its 4 shards exited 1: FAIL by-calls/callee-failed Root_v1"),
                       (2, "circuit-check --combine of its 4 shards refused them: circuit-check --combine: refused: shard(s) 4 of 4 missing")):
        assert CHECK.combine(["r1", "r2"], run) == 1
        doc = json.loads((run / "result.json").read_text())
        assert doc["validation"]["status"] == "failed" and doc["validation"]["detail"] == why and doc["steps"] == steps
    joined.clear()
    assert CHECK.circuit_join(Store(), [{**s, "run_id": "r1"} for s in steps], run) is None and not joined
    assert "lost published no run files" in CHECK.circuit_join(Store(), [{**s, "run_id": "lost"} for s in steps[:2]] + steps[3:], run)


def test_record_sends_a_shard_only_what_its_own_steps_read(tmp_path, monkeypatch):
    import research.store.local

    real, launched, urls = subprocess.run, [], []

    def run(cmd, *a, **k):
        if cmd[:4] == [sys.executable, "-m", "research", "run"]:
            launched.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)
        return real(cmd, *a, **k)

    class Store:
        def fetch(self, art):
            (tmp_path / art[4:]).write_text("")
            return tmp_path / art[4:]

    def url_file(d):
        urls.append(d)
        return None

    (tmp_path / "upstream.json").write_text(json.dumps({"upstream": {"art": "art:u"}, "inputs": {"art": "art:i"}}))
    monkeypatch.setattr(CHECK, "UPSTREAM_PIN", tmp_path / "upstream.json")
    monkeypatch.setattr(research.store.local, "LocalStore", Store)
    monkeypatch.setattr(CHECK, "landable", lambda base: True)
    monkeypatch.setattr(CHECK, "lean_deps_url_file", url_file)
    monkeypatch.setattr(subprocess, "run", run)
    sent = {}
    for args in (["--shard", "3/5"], ["--shard", "1/5"], ["--shard", "4/5"], ["--combine", "r1", "r2"], []):
        launched.clear(), urls.clear()
        assert CHECK.record(["--base", "HEAD", "--agreement", *args]) == 0
        [cmd] = launched
        assert cmd[-len(args):] == args if args else True
        sent[" ".join(args)] = ("flock_upstream=art:u" in cmd, bool(urls))
    assert sent == {"--shard 3/5": (True, False), "--shard 1/5": (False, True), "--shard 4/5": (False, False),
                    "--combine r1 r2": (False, False), "": (True, True)}
    assert CHECK.record(["--base", "HEAD", "--shard", "9/5"]) == 2


def test_a_recorded_shard_tells_the_preflight_its_parts(monkeypatch):
    """The preflight asks for the Lean dependencies of the packages a shard audits, the only ones it carries URLs for."""
    monkeypatch.setattr(CHECK, "gather_verdicts", lambda d, given=None: None)
    real, parts = subprocess.run, []

    def run(cmd, *a, **k):
        if cmd[:4] == [sys.executable, "-m", "research", "run"]:
            sent = [Path(cmd[i + 1]) for i, x in enumerate(cmd) if x == "--send" and cmd[i + 1].endswith(CHECK.SHARD_PARTS)]
            parts.append([json.loads(p.read_text()) for p in sent])
            return subprocess.CompletedProcess(cmd, 0)
        return real(cmd, *a, **k)

    monkeypatch.setattr(CHECK, "landable", lambda base: True)
    monkeypatch.setattr(CHECK, "lean_deps_url_file", lambda d: None)
    monkeypatch.setattr(subprocess, "run", run)
    for args in (["--shard", "1/5"], ["--shard", "4/5"], ["--combine", "r1", "r2"], []):
        assert CHECK.record(["--base", "HEAD", "--on", "pod", *args]) == 0
    assert parts == [[list(SECURITY)], [["pytest"]], [[]], []], "an unsharded check sends none"


def _records_run(tmp_path, monkeypatch, rc: int = 0) -> tuple[Path, list[list[str]]]:
    """`check.py --cc-records` in a recorded run, circuit-check stubbed: it stores one record in its ``--cache`` and exits
    ``rc``.  (the run dir, each circuit-check it ran)."""
    real, ran = subprocess.run, []

    def run(cmd, *a, **k):
        if cmd[1:3] == ["-m", "circuit_check"]:
            ran.append(cmd)
            rec = Path(cmd[cmd.index("--cache") + 1]) / "v9" / "abc" / "e1.json"
            rec.parent.mkdir(parents=True)
            rec.write_text('{"key": "k"}')
            Path(cmd[cmd.index("--out") + 1]).write_text("{}")
            return subprocess.CompletedProcess(cmd, rc)
        return real(cmd, *a, **k)

    run_dir = tmp_path / "records-run"
    run_dir.mkdir()
    monkeypatch.setenv("RESEARCH_RUN_ID", "r20261030-000000-cc00")
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(run_dir))
    monkeypatch.delenv("VERITY_SKIP_STORE", raising=False)
    monkeypatch.setattr(subprocess, "run", run)
    assert CHECK.main(["--keep-going", "--cc-records"]) == (0 if rc == 0 else rc)
    monkeypatch.setattr(subprocess, "run", real)
    return run_dir, ran


def test_the_records_step_builds_circuit_checks_records_once_and_tars_them_with_the_tree_they_were_built_from(tmp_path, monkeypatch):
    """`--cc-records`: one `circuit-check --all --records-only` into a fresh cache, tarred under $OUT with this tree's id, its
    sha256 in a valid result; a failed build leaves no tarball, and the result says it failed."""
    import hashlib
    import tarfile

    from research import result as RES

    run_dir, ran = _records_run(tmp_path, monkeypatch)
    [cc] = ran
    assert cc[3:6] == ["--all", "--records-only", "-q"] and "--catalog-units" not in cc and "--shard" not in cc
    r = RES.check(run_dir, "r20261030-000000-cc00")
    assert r.valid and r.validation_status == "passed" and r.doc["kind"] == CHECK.CC_RECORDS_KIND, r.errors
    tar, tree = run_dir / CHECK.CC_RECORDS, CHECK._rev("HEAD^{tree}")
    assert r.doc["records"] == {"path": CHECK.CC_RECORDS, "sha256": hashlib.sha256(tar.read_bytes()).hexdigest(), "tree": tree}
    with tarfile.open(tar) as t:
        assert t.extractfile(CHECK.CC_RECORDS_TREE).read().decode() == tree + "\n"
        assert t.extractfile("cache/v9/abc/e1.json").read() == b'{"key": "k"}'
    (tmp_path / "failed").mkdir()
    run_dir, ran = _records_run(tmp_path / "failed", monkeypatch, rc=1)
    assert len(ran) == 1 and not (run_dir / CHECK.CC_RECORDS).exists()
    doc = json.loads((run_dir / "result.json").read_text())
    assert doc["validation"]["status"] == "failed" and "records" not in doc
    assert CHECK.main(["--cc-records", "--no-cache"]) == 2, "it takes no other flag"


def test_a_shard_takes_the_records_artifact_only_with_its_digest_and_this_tree(tmp_path, monkeypatch):
    """`--cc-records-from ART@SHA256`: the tarball fetched from ART goes into circuit-check's cache only when its bytes have
    that sha256 and it was built from this tree; anything else is refused by name, and nothing goes in."""
    import hashlib
    import io
    import tarfile

    run_dir, _ = _records_run(tmp_path, monkeypatch)
    tar = run_dir / CHECK.CC_RECORDS
    sha = hashlib.sha256(tar.read_bytes()).hexdigest()
    fetched = []

    def fetch(art, d, src=tar):
        fetched.append(art)
        (got := d / "got.tar.gz").write_bytes(src.read_bytes())
        return got

    cache = tmp_path / "check" / "circuit-check"
    other = tmp_path / "other.tar.gz"
    with tarfile.open(other, "w:gz") as t:
        info = tarfile.TarInfo(CHECK.CC_RECORDS_TREE)
        info.size = len(b"0" * 40 + b"\n")
        t.addfile(info, io.BytesIO(b"0" * 40 + b"\n"))
    osha = hashlib.sha256(other.read_bytes()).hexdigest()
    assert CHECK.cc_records_in(f"art:r1@{'0' * 64}", cache, fetch) == f"its bytes' sha256 is {sha}, not {'0' * 64}"
    why = CHECK.cc_records_in(f"art:r2@{osha}", cache, lambda art, d: fetch(art, d, other))
    assert why.startswith("it was built from tree 000000000000, and this tree is ")
    assert CHECK.cc_records_in("art:r3", cache, fetch) == "art:r3 is not ART@SHA256"
    assert "can't be fetched (OSError: gone)" in CHECK.cc_records_in(f"art:r4@{sha}", cache, lambda art, d: (_ for _ in ()).throw(OSError("gone")))
    assert not cache.exists() or not list(cache.rglob("*.json")), "nothing refused goes in"
    assert CHECK.cc_records_in(f"art:r5@{sha}", cache, fetch) is None
    assert (cache / "v9" / "abc" / "e1.json").read_text() == '{"key": "k"}' and fetched == ["art:r1", "art:r2", "art:r5"]
    assert [p.name for p in cache.parent.iterdir()] == ["circuit-check"], "its scratch is gone"


def test_each_shard_holding_circuit_check_starts_from_the_records_and_one_refused_runs_nothing(tmp_path, monkeypatch, capsys):
    """Drive gives every shard of its after-landing run the records step's artifact: each that holds a part of circuit-check
    unpacks it into circuit-check's cache before its steps; a shard without one ignores it; a refusal exits 2 before any step,
    as does an artifact where circuit-check keeps no cache."""
    monkeypatch.setenv("RESEARCH_RUN_ID", "r20261030-000000-abcd")
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(tmp_path))
    monkeypatch.setenv("CHECK_CACHE", str(tmp_path / "cache" / "check"))
    monkeypatch.delenv("VERITY_SKIP_STORE", raising=False)
    monkeypatch.setattr(CHECK.SUITES, "foreign", lambda members, env: None)
    ran, took, why = [], [], [None]
    monkeypatch.setattr(CHECK, "run", lambda out, n, cache, keep_going, only, *a: ran.append(only) or (True, []))
    monkeypatch.setattr(CHECK, "cc_records_in", lambda spec, cache: took.append((spec, cache)) or why[0])
    art = f"art:abc@{'1' * 64}"
    steps = ",".join(j.name for j in CHECK.JOBS)
    for k in range(1, 5):
        argv = ["--deferred", steps, "--shard", f"{k}/4", "--keep-going", "--no-cache", "--circuit-cache", "--cc-records-from", art]
        assert CHECK.main(argv) == 0
    assert took == [(art, CHECK.cache_dir() / "circuit-check")] * 4 and len(ran) == 4
    assert all(f"circuit-check:{k}/4" in only for k, only in enumerate(ran, 1))
    took.clear()
    assert CHECK.main(["--deferred", "pytest", "--shard", "3/4", "--cc-records-from", art]) == 0 and not took
    why[0] = "it was built from tree 000000000000, and this tree is 111111111111"
    assert CHECK.main(["--deferred", steps, "--shard", "2/4", "--cc-records-from", art]) == 2 and len(ran) == 5
    assert f"--cc-records-from {art} refused: {why[0]}" in capsys.readouterr().err
    assert CHECK.main(["--deferred", steps, "--shard", "2/4", "--no-cache", "--cc-records-from", art]) == 2 and len(ran) == 5
    assert "circuit-check reuses no cache here" in capsys.readouterr().err


def test_record_of_the_records_step_launches_it_alone_with_circuit_checks_memory(monkeypatch):
    """`--record --cc-records`: recorded without the check tool (no gate reads it), in a train slot's memory as circuit-check
    takes, with no upstream build, Lean dependencies or gathered verdicts."""
    real, launched = subprocess.run, []

    def run(cmd, *a, **k):
        if cmd[:4] == [sys.executable, "-m", "research", "run"]:
            launched.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)
        return real(cmd, *a, **k)

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(CHECK, "landable", lambda base: True)
    monkeypatch.setattr(CHECK, "lean_deps_url_file", lambda d: (_ for _ in ()).throw(AssertionError("Lean dependencies sent")))
    monkeypatch.setattr(CHECK, "gather_verdicts", lambda d, given=None: (_ for _ in ()).throw(AssertionError("verdicts gathered")))
    monkeypatch.setattr(CHECK, "slot_wrap", lambda priority: ["slot"])
    assert CHECK.record(["--base", "HEAD", "--agreement", "--on", "vy-x", "--priority", "--keep-going", "--cc-records"]) == 0
    [cmd] = launched
    head, tail = cmd[:cmd.index("--")], cmd[cmd.index("--") + 1:]
    assert "--tool" not in head and "flock_upstream=" not in " ".join(head)
    assert f"{CHECK.SLOT.MIN_MEM_ENV}={CHECK.TRAIN_GB}" in head and tail[0] == "slot" and tail[-2:] == ["--keep-going", "--cc-records"]
