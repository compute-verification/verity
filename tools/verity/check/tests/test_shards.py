"""`check.py --shard K/N`: ci.toml's plans run every part of an unsharded check exactly once, each shard holds what its jobs
need, a shard runs only its parts (lean-audit on its packages), and its result names what it ran."""
from __future__ import annotations

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


CHECK = _module("repo_check_for_shards", ROOT / "tools" / "verity" / "check" / "check.py")
D = CHECK.DECLARATION
LA = _module("repo_lean_audit_for_shards", ROOT / "tools" / "verity" / "lean" / "lean_audit.py")
PACKAGES = LA.packages(CHECK.tracked_files(ROOT))
SECURITY = ("lean-audit:Security", "lean-audit:Security/Proofs")


def _resolved(n: int, packages=PACKAGES) -> list[list[str]]:
    return D.resolve(D.plans()[n], CHECK.JOBS, {"lean-audit": packages})


def test_the_plans_are_the_documented_ones():
    assert D.plans() == {
        3: ((*SECURITY, "pytest", "ratchet-gate"),
            ("lean-build", "lean-unit-cut", "lean-audit:*", "lean-suites", "lean-agreement", "lean-changed"),
            ("flock-circuit-build", "circuit-check", "flock-rust-tests")),
        5: (SECURITY, ("lean-audit:*", "lean-changed"), ("lean-build", "lean-unit-cut", "lean-suites", "lean-agreement"),
            ("pytest", "ratchet-gate"), ("flock-circuit-build", "circuit-check", "flock-rust-tests"))}


@pytest.mark.parametrize("n", sorted(D.plans()))
def test_every_plan_runs_every_part_of_this_tree_exactly_once_and_the_same_way_each_time(n):
    every = D.all_parts(CHECK.JOBS, {"lean-audit": PACKAGES})
    assert len(PACKAGES) >= 4 and all(every.count(s) == 1 for s in SECURITY) and "lean-audit" not in every
    assert {p.partition(":")[0] for p in every} == {j.name for j in CHECK.JOBS}
    shards = _resolved(n)
    flat = [p for s in shards for p in s]
    assert len(shards) == n and all(shards) and sorted(flat) == sorted(every) and len(set(flat)) == len(flat)
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


@pytest.mark.parametrize("shards, why", [
    ('[["a", "b", "lean-audit:*"]]', "two or more"),
    ('[["a", "b"], []]', "two or more"),
    ('[["a", "b", "lean-audit:*"], ["x"]]', "'x' names no declared job"),
    ('[["a", "b", "lean-audit:*"], ["c:1"]]', "only lean-audit split"),
    ('[["a", "b", "lean-audit:*"], ["lean-audit:"]]', "only lean-audit split"),
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
    with pytest.raises(D.DeclarationError, match="ci.toml's plans are K/3, K/5"):
        CHECK.shard(spec)


@pytest.mark.parametrize("n", sorted(D.plans()))
def test_a_shard_runs_its_parts_alone_in_their_groups_order_and_lean_audit_on_its_packages(tmp_path, n):
    for k, parts in enumerate(_resolved(n), 1):
        assert CHECK.shard(f"{k}/{n}") == {"k": k, "n": n, "parts": parts, "all": D.all_parts(CHECK.JOBS, {"lean-audit": PACKAGES})}
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


def test_a_shard_without_the_other_half_gets_every_core(tmp_path, monkeypatch):
    """circuit-check gets its cores whatever the memory: its own admission by measured peaks bounds the memory, and a flat 160 GB
    per worker held a cold `--all` in a 560 GB slot to 5 workers for over 2 h (f7b5, 10 Oct)."""
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


def test_combine_records_the_steps_the_gates_reading_of_the_shards_gives_and_fails_what_it_refuses(tmp_path, monkeypatch):
    """Inside a recorded run, a `check/v1` of `shard_set`'s steps (which the gate compares with its own reading); a failed one when
    the set is refused or the checkout is dirty; outside a run it only says."""
    import research.merge
    import research.store.local
    from research import result as RES

    tree = tmp_path / "tree"
    (tree / "tools" / "verity" / "check").mkdir(parents=True)
    (tree / "tools" / "verity" / "check" / "tool.py").write_text(
        "from research.store.tool import Tool\nCHECK = Tool(name='check', version='1', command=['run', 'check'])\n")
    for args in (["init", "-q", "-b", "main"], ["add", "-A"], ["commit", "-q", "-m", "t"]):
        subprocess.run(["git", "-c", "user.email=t@t", "-c", "user.name=t", "-c", "commit.gpgsign=false", *args], cwd=tree, check=True,
                       capture_output=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tree, capture_output=True, text=True).stdout.strip()
    steps = [{"step": "preflight-lock", "status": "passed", "shards": ["r1", "r2"]},
             {"step": "pytest", "status": "passed", "shard": 1, "run_id": "r1", "seconds": 1.0}]
    asked = []

    def shard_set(store, repo, commit, ids, tool, command):
        asked.append((repo, commit, ids, tool, command))
        return (steps, None) if ids == ["r1", "r2"] else (None, "shard 2/2 is not among them")

    class Store:
        remote = None

    monkeypatch.setattr(CHECK, "ROOT", tree)
    monkeypatch.setattr(research.merge, "shard_set", shard_set)
    monkeypatch.setattr(research.store.local, "LocalStore", Store)
    monkeypatch.setenv("RESEARCH_RUN_ID", "r20261004-000000-ffff")
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
    assert parts == [[list(SECURITY)], [["pytest", "ratchet-gate"]], [[]], []], "an unsharded check sends none"
