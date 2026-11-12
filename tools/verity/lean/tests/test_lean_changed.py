"""`lean_changed.py --records --update`: the regenerated record survives where the run keeps it, and a node
whose Lean slots grant no `build` slot takes the `--dev-cache` path.

In a recorded run the fast path never writes ROOT (the commit's shared source tree), and the slot tree's copy of the
updated `lean-audit.json` is restored to the committed bytes by the next requirer iteration's `sync`, so without an
explicit keep the record its author must commit survives nowhere the run keeps. These tests stub the slot, the tree sync
and check.py itself, and check the keep under `out/record/` (recorded and not), the ROOT write-back (unrecorded
only), which path a refused `build` slot takes, and how a run finds and repairs a tree another run left mid-way. Two
stand-in packages that read each other's builds check that one `--update` run regenerates and keeps both records."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lean_changed", HERE.parent / "lean_changed.py")
LC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(LC)

STORED = LC.stored


@pytest.fixture(autouse=True)
def no_store(monkeypatch):
    """No test asks this machine's store (`stored`, through `research`) unless it says so."""
    monkeypatch.setattr(LC, "stored", lambda *a, **k: None)


OLD = json.dumps({"guarantees": {"G": {"owner": "@lean"}}}) + "\n"
NEW = json.dumps({"guarantees": {"G": {"owner": "@lean", "type_hash": "new"}}}) + "\n"


@pytest.fixture()
def fast(tmp_path, monkeypatch):
    """`audit_records` against a stand-in package `pkg`, a held-open slot tree and a fake check.py whose `--update`
    writes `NEW` over the tree's record; `sync` restores the committed record, as the real one does from ROOT's files."""
    root = tmp_path / "root"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "lean-audit.json").write_text(OLD)
    cache = tmp_path / "cache"
    tree = cache / LC.RECORDS / "local"
    (tree / "pkg").mkdir(parents=True)
    monkeypatch.setattr(LC, "ROOT", root)
    monkeypatch.setattr(LC.LA.COMMON, "cache_dir", lambda: cache)
    monkeypatch.setattr(LC.LA.COMMON, "tracked_files", lambda _: ["pkg/lean-audit.json"])
    monkeypatch.setattr(LC.LA, "packages", lambda _: ["pkg"])
    monkeypatch.setattr(LC.LA, "deps_key", lambda _: "k")
    monkeypatch.setattr(LC, "requirers", lambda *_: ["pkg/Req"])
    monkeypatch.setattr(LC, "records_tree", lambda: tree)
    monkeypatch.setattr(LC, "link", lambda *a, **k: True)
    monkeypatch.setattr(LC, "unlink", lambda *a, **k: None)
    monkeypatch.setattr(LC, "label", lambda _: None)
    monkeypatch.setattr(LC.LEAN_SLOT, "hold", lambda *a, **k: None)
    monkeypatch.setattr(LC, "reused", lambda *a, **k: {})
    monkeypatch.setattr(LC, "passed", lambda *a, **k: [])
    monkeypatch.setattr(LC, "git", lambda *a: None)

    def sync(_closure, _files, t):
        (t / "pkg" / "lean-audit.json").write_text(OLD)
        return 0

    monkeypatch.setattr(LC, "sync", sync)

    calls = []

    def run(argv, **kw):
        calls.append((argv, kw.get("cwd")))
        if "--update" in argv:
            (Path(argv[-1]) / "lean-audit.json").write_text(NEW)
        return type("P", (), {"returncode": 0})()

    monkeypatch.setattr(LC.subprocess, "run", run)
    return root, tree, calls


def test_a_slot_tree_linked_to_a_removed_directory_is_made_again(fast, tmp_path, monkeypatch):
    """A slot's tree that links to a directory removed since (a node's cache moved to a larger volume, then cleaned) is made
    again at the link's target, where `mkdir` on the link itself raised FileExistsError and failed every run on that slot."""
    target = tmp_path / "volume" / "lean-records_build-1"
    link = tmp_path / "cache" / LC.RECORDS / "build-1"
    link.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(LC, "records_tree", lambda: link)

    def sync(_closure, _files, t):
        (t / "pkg").mkdir(parents=True, exist_ok=True)
        (t / "pkg" / "lean-audit.json").write_text(OLD)
        return 0

    monkeypatch.setattr(LC, "sync", sync)
    assert LC.audit_records("pkg", tmp_path / "deps", out=tmp_path / "out") == 0
    assert link.is_symlink() and target.is_dir(), "the link stays and its target is made"


def test_a_recorded_update_keeps_the_regenerated_record_in_the_run_outputs(fast, tmp_path, monkeypatch):
    root, _, _ = fast
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(run_dir))
    monkeypatch.setenv("RESEARCH_RUN_ID", "r-test")
    assert LC.audit_records("pkg", tmp_path / "deps", update=True) == 0
    kept = run_dir / "lean-audit" / "record" / "lean-audit.json"
    assert kept.read_text() == NEW, "the regenerated record is kept under the run's outputs"
    assert (root / "pkg" / "lean-audit.json").read_text() == OLD, "a recorded run never writes the shared source tree"


def test_an_unrecorded_update_writes_the_record_back_into_the_checkout(fast, tmp_path, monkeypatch):
    root, _, _ = fast
    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
    monkeypatch.delenv("RESEARCH_RUN_ID", raising=False)
    out = tmp_path / "out"
    assert LC.audit_records("pkg", tmp_path / "deps", update=True, out=out) == 0
    assert (root / "pkg" / "lean-audit.json").read_text() == NEW, "the unrecorded copy-back still writes the checkout"
    assert (out / "record" / "lean-audit.json").read_text() == NEW, "and the keep is written either way"


def test_no_record_change_keeps_nothing(fast, tmp_path, monkeypatch):
    _, tree, _ = fast
    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
    monkeypatch.delenv("RESEARCH_RUN_ID", raising=False)

    def run(argv, **_):
        return type("P", (), {"returncode": 0})()

    monkeypatch.setattr(LC.subprocess, "run", run)
    out = tmp_path / "out"
    assert LC.audit_records("pkg", tmp_path / "deps", update=True, out=out) == 0
    assert not (out / "record").exists(), "an unchanged record keeps nothing"


def refuse(*_a, **_k):
    raise LC.LEAN_SLOT.Refused("lean-slots lists no pool 'build', the only one this request takes")


def test_a_refused_build_slot_takes_the_dev_cache_path_on_the_checkout_and_its_requirers(fast, tmp_path, monkeypatch):
    root, tree, calls = fast
    monkeypatch.setattr(LC.LEAN_SLOT, "hold", refuse)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(run_dir))
    monkeypatch.setenv("RESEARCH_RUN_ID", "r-test")
    said = []
    assert LC.audit_records("pkg", tmp_path / "deps", update=True, log=said.append) == 0
    assert [(a[a.index("--out") + 2:], cwd) for a, cwd in calls] == [([str(root / "pkg")], root),
                                                                       ([str(root / "pkg/Req")], root)]
    assert all("--dev-cache" in a and "--no-replay" in a for a, _ in calls), "both steps take the --dev-cache path"
    assert "--update" in calls[0][0] and "--update" not in calls[1][0], "a requirer is checked, never updated"
    assert calls[1][0][calls[1][0].index("--out") + 1] == str(run_dir / "lean-audit" / "requirer-pkg_Req")
    assert any("taking the --dev-cache path" in s for s in said)
    assert not (tree / ".lock").exists(), "the slot tree is never touched"
    assert (run_dir / "lean-audit" / "record" / "lean-audit.json").read_text() == NEW, "the same keep as the slot path"
    result = json.loads((run_dir / "result.json").read_text())
    assert result["kind"] == LC.FAST_KIND and result["workload_fingerprint"]["path"] == "dev-cache"


def test_a_build_slot_takes_the_slot_path(fast, tmp_path, monkeypatch):
    _, tree, calls = fast
    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
    said = []
    assert LC.audit_records("pkg", tmp_path / "deps", out=tmp_path / "out", log=said.append) == 0
    assert [a[-1] for a, _ in calls] == [str(tree / "pkg"), str(tree / "pkg/Req")]
    assert not any("--dev-cache" in a for a, _ in calls)
    assert any("taking the slot path" in s for s in said)


def test_the_slot_path_reads_the_node_s_artifact_cache_and_fills_it_only_from_a_package_that_passed(fast, tmp_path,
                                                                                                   monkeypatch):
    _, tree, calls = fast
    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
    put = []

    def run(argv, **kw):
        calls.append((argv, kw.get("cwd")))
        _reported(argv, True)
        return type("P", (), {"returncode": 1 if argv[-1].endswith("Req") else 0})()

    def fake_put(self, reports, since=None, log=print):
        put.append((self.store, reports, since))
        return 3

    monkeypatch.setattr(LC.subprocess, "run", run)
    monkeypatch.setattr(LC.LA.LakeArtifacts, "put", fake_put)
    said = []
    assert LC.audit_records("pkg", tmp_path / "deps", out=tmp_path / "out", log=said.append) == 1
    store = (tmp_path / "lean-artifacts").resolve()
    assert [a[a.index("--artifacts") + 1] for a, _ in calls] == [str(store)] * 2
    assert [(s, p) for s, p, _ in put] == [(store, {tree / "pkg": {"package": str(tree / "pkg"), "sandbox": True}})], \
        "the package that passed, with check.py's report on it; a requirer that failed puts nothing"
    assert isinstance(put[0][2], float) and any("3 module version(s) new in the artifact cache" in s for s in said)


def _reported(argv: list[str], sandbox: bool) -> None:
    """check.py's report on the package ``argv`` names, under its `--out`, built in the sandbox or not."""
    out = Path(argv[argv.index("--out") + 1])
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.json").write_text(json.dumps({"packages": [{"package": argv[-1], "sandbox": sandbox}]}))


def test_a_records_build_outside_the_sandbox_puts_nothing_in_the_artifact_cache(fast, tmp_path, monkeypatch):
    """Records runs fill the store check's audit stages from, and check.py builds without the sandbox where `unshare` is
    missing, where compile-time code could leave in the build what it chose: such a package puts nothing there, though it
    passed."""
    _, tree, _ = fast
    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
    fake = LC.subprocess.run
    monkeypatch.setattr(LC.subprocess, "run", lambda argv, **kw: _reported(argv, False) or fake(argv, **kw))
    said = []
    assert LC.audit_records("pkg", tmp_path / "deps", out=tmp_path / "out", log=said.append) == 0
    assert [s for s in said if "artifact cache" in s] == [s for q in ("pkg", "pkg/Req") for s in (
        f"lean-audit: {tree / q}: its report doesn't say it was built in the sandbox, so none of its modules go into the "
        "artifact cache", f"lean-records: {q}: 0 module version(s) new in the artifact cache")]
    assert not (tmp_path / "lean-artifacts" / "keys").exists()


def test_dev_cache_takes_that_path_even_where_a_build_slot_is_granted(fast, tmp_path, monkeypatch):
    root, tree, calls = fast
    monkeypatch.setattr(LC.LEAN_SLOT, "hold", lambda *a, **k: pytest.fail("--dev-cache takes no build slot"))
    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
    said = []
    assert LC.audit_records("pkg", tmp_path / "deps", out=tmp_path / "out", log=said.append, dev_cache=True) == 0
    assert [a[-1] for a, _ in calls] == [str(root / "pkg"), str(root / "pkg/Req")]
    assert all("--dev-cache" in a for a, _ in calls) and any("asks for it; taking the --dev-cache path" in s for s in said)
    for argv in (["--dev-cache"], ["--records", "pkg", "--dev-cache", "--tree", str(tree)]):
        with pytest.raises(SystemExit):
            LC.main(argv)


def test_the_dev_cache_path_never_builds_in_a_shipped_tree(fast, tmp_path, monkeypatch):
    root, _, calls = fast
    monkeypatch.setattr(LC.LEAN_SLOT, "hold", refuse)
    (root / "READY.json").write_text("{}\n")
    said = []
    assert LC.audit_records("pkg", tmp_path / "deps", out=tmp_path / "out", log=said.append) == 2
    assert not calls and any("--cwd clone" in s for s in said)


S, P = "s", "s/props"


def record(h: str, proved_in: list[str] | None = None) -> str:
    return json.dumps({**({"proved_in": proved_in} if proved_in else {}), "guarantees": {"G": {"type_hash": h}}}) + "\n"


#: each package's record as the fake check.py's build has it, which `--update` writes
BUILT = {S: record("new", ["proofs"]), P: record("new")}
#: the record check.py writes for `s/proofs` when it fails there on a record difference
PROOFS = json.dumps({"assumptions": {}}) + "\n"


#: the stand-in packages of `pair`: `s`, whose `proved_in` names `s/proofs`, which requires `s` and `s/props` by path
PAIR = {"s/lakefile.toml": 'name = "s"\n', "s/lean-audit.json": record("old", ["proofs"]), "s/S.lean": "",
        "s/proofs/lakefile.toml": 'name = "proofs"\n', "s/proofs/lean-audit.json": "{}\n",
        "s/proofs/lake-manifest.json": json.dumps({"packages": [{"type": "path", "dir": ".."},
                                                                {"type": "path", "dir": "../props"}]}),
        "s/props/lakefile.toml": 'name = "props"\n', "s/props/lean-audit.json": record("old"), "s/props/P.lean": "",
        "s/props/lake-manifest.json": json.dumps({"packages": [{"type": "path", "dir": ".."}]})}


@pytest.fixture()
def pair(tmp_path, monkeypatch):
    """`audit_records` on two stand-in packages each of whose records read the other's build, as `Security`'s and
    `Security/Properties`' do (`s` names `s/proofs` under `proved_in`, which requires `s` and `s/props` by path), with the
    real `sync`, requirers and pass cache, and a fake check.py over a fixed build: a package's records pass when its lock is
    `BUILT`'s, fail on a record difference otherwise (`regenerated` in the report names `BUILT`'s, which check.py wrote
    there), and `--update` writes `BUILT`'s; a package in the returned set fails for another reason (with `writes` in it,
    `--update` writes its record all the same, as check.py does), and with `proofs` in it `s/proofs`, audited beside `s`,
    fails on a record difference. Each call is (package, `--update`, both locks as it saw them)."""
    root = tmp_path / "root"
    files = dict(PAIR)
    for f, text in files.items():
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        (root / f).write_text(text)
    cache = tmp_path / "cache"
    monkeypatch.setattr(LC, "ROOT", root)
    monkeypatch.setattr(LC.LA, "ROOT", root)
    monkeypatch.setattr(LC.LA.COMMON, "cache_dir", lambda: cache)
    monkeypatch.setattr(LC.LA.COMMON, "tracked_files", lambda _: sorted(files))
    monkeypatch.setattr(LC, "records_tree", lambda: cache / LC.RECORDS / "local")
    monkeypatch.setattr(LC.LEAN_SLOT, "hold", lambda *a, **k: None)
    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
    monkeypatch.delenv("RESEARCH_RUN_ID", raising=False)
    calls, other, real = [], set(), LC.subprocess.run

    def run(argv, **kw):
        if argv[1:2] != [str(LC.AUDIT_PY)]:
            return real(argv, **kw)
        at, out, update = Path(argv[-1]), Path(argv[argv.index("--out") + 1]), "--update" in argv
        p = os.path.relpath(at, kw["cwd"])
        calls.append((p, update, {q: (Path(kw["cwd"]) / q / LC.LOCK).read_text() for q in (S, P)}))
        fails, regen = [], None
        if p in other:
            fails = ["fresh.rejected: fresh: the kernel rejected a declaration"]
            if update and "writes" in other:
                (at / LC.LOCK).write_text(BUILT[p])
        elif update:
            (at / LC.LOCK).write_text(BUILT[p])
        elif (at / LC.LOCK).read_text() != BUILT[p]:
            fails, regen = [f"guarantees.reads-changed: {p}: what G reads changed"], out / "audit-x" / LC.LOCK
            regen.parent.mkdir(parents=True, exist_ok=True)
            regen.write_text(BUILT[p])
        reps = [{"package": str(at.resolve()), "failures": fails, **({"regenerated": str(regen)} if regen else {})}]
        if p == S and "proofs" in other:
            (side := out / "audit-proofs" / LC.LOCK).parent.mkdir(parents=True, exist_ok=True)
            side.write_text(PROOFS)
            reps.append({"package": str((at / "proofs").resolve()), "regenerated": str(side),
                         "failures": ["assumptions.unread: proofs: an assumption no guarantee reads"]})
        elif p == S:
            reps.append({"package": str((at / "proofs").resolve()), "failures": []})
        out.mkdir(parents=True, exist_ok=True)
        (out / "audit.json").write_text(json.dumps({"passed": not any(r["failures"] for r in reps), "packages": reps}) + "\n")
        return type("P", (), {"returncode": 1 if any(r["failures"] for r in reps) else 0})()

    monkeypatch.setattr(LC.subprocess, "run", run)
    return root, calls, other


@pytest.mark.parametrize("hold", [lambda *a, **k: None, refuse], ids=["slot", "dev-cache"])
def test_one_update_regenerates_a_requirer_failing_only_on_record_differences_and_its_commit_runs_nothing(
        pair, tmp_path, monkeypatch, hold):
    """A change to what both `Security`'s and `Security/Properties`' guarantees read moves both locks: one `--records
    Security --update` run regenerates and keeps both, each audited on the records regenerated before it, and keeps each
    pass under the regenerated locks, so a commit of both records needs no run of either package. Each package's
    `inputs_key` is the commit's, though the `--dev-cache` path's check.py rewrites the locks in the checkout."""
    root, calls, _ = pair
    monkeypatch.setattr(LC.LEAN_SLOT, "hold", hold)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(run_dir))
    monkeypatch.setenv("RESEARCH_RUN_ID", "r-pair")
    files = LC.LA.COMMON.tracked_files(root)
    before = {p: LC.inputs_key(p, files) for p in (S, P)}
    said = []
    assert LC.audit_records(S, tmp_path / "deps", update=True, log=said.append) == 0
    doc = json.loads((run_dir / "result.json").read_text())
    assert {p: doc["records"][p]["inputs_key"] for p in (S, P)} == before
    assert [(p, up) for p, up, _ in calls] == [(S, True), (P, False), (P, True)], "the requirer is updated after it fails"
    assert calls[1][2] == {S: BUILT[S], P: record("old")}, "the requirer is audited on PKG's regenerated record"
    out = run_dir / "lean-audit"
    kept = {S: out / "record" / S / LC.LOCK, P: out / "record" / P / LC.LOCK}
    assert {p: k.read_text() for p, k in kept.items()} == BUILT and (out / "record" / LC.LOCK).read_text() == BUILT[S]
    for p in (S, P):
        assert sum(f"commit it over {p}/{LC.LOCK}" in s for s in said) == 1, f"one line says what to commit {p}'s record over"
    assert any(f"{P} requires {S}, and every failure of its records is a record difference" in s for s in said)
    assert json.loads((run_dir / "result.json").read_text())["validation"]["status"] == "passed"
    for p, k in kept.items():
        (root / p / LC.LOCK).write_bytes(k.read_bytes())
    assert {LC.inputs_key(p, files): p for p in (S, P)}.items() <= doc["passes"].items()
    calls.clear()
    for p in (S, P):
        said.clear()
        assert LC.audit_records(p, tmp_path / "deps", log=said.append) == 0
        assert sum("reused, nothing built" in s for s in said) == 2, f"--records {p} at the commit of both records"
    assert calls == []


def outcome_line(said: list[str]) -> dict:
    """The run's last line, `OUTCOME_LINE` and JSON."""
    assert said[-1].startswith(LC.OUTCOME_LINE), said[-1]
    return json.loads(said[-1][len(LC.OUTCOME_LINE):])


def test_a_requirer_failing_for_another_reason_fails_and_without_update_none_is_updated(pair, tmp_path):
    root, calls, other = pair
    other.add(P)
    said = []
    out = tmp_path / "out"
    assert LC.audit_records(S, tmp_path / "deps", update=True, out=out, log=said.append) == 1
    assert [(p, up) for p, up, _ in calls] == [(S, True), (P, False)], "no --update on a requirer that fails otherwise"
    assert not (out / "record" / P).exists() and (out / "record" / S / LC.LOCK).read_text() == BUILT[S]
    assert any(f"{P} requires {S}" in s and "not only on record differences" in s for s in said)
    assert (root / P / LC.LOCK).read_text() == record("old")
    got = outcome_line(said)
    assert (got["outcome"], got["records"][S]["outcome"], got["records"][P]["outcome"]) == (LC.FAILED, LC.REGENERATED,
                                                                                            LC.FAILED)
    calls.clear()
    said.clear()
    (root / S / LC.LOCK).write_text(record("old", ["proofs"]))
    out = tmp_path / "out2"
    assert LC.audit_records(S, tmp_path / "deps", out=out, log=said.append) == 1
    assert [(p, up) for p, up, _ in calls] == [(S, False), (P, False)], "without --update no package is updated"
    assert (out / "record" / S / LC.LOCK).read_text() == BUILT[S], "the record check.py wrote for S's differences is kept"
    assert (root / S / LC.LOCK).read_text() == record("old", ["proofs"]), "and never written into the checkout"
    assert outcome_line(said)["outcome"] == LC.FAILED, "the requirer's other failure is the run's"


def test_a_run_whose_only_failures_are_record_differences_exits_0_as_regenerated_with_each_record_kept(pair, tmp_path,
                                                                                                       monkeypatch):
    """A run that found only record differences, here without `--update`, so that check.py fails each package and names
    the record that settles it, exits 0 with `outcome: regenerated`, each record
    kept under `record/`, the requirer audited on the record kept before it, and `validation.status` passed, which the
    records gate reads as it does for an `--update` run's kept records; a commit of both records then needs no run."""
    root, calls, _ = pair
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(run_dir))
    monkeypatch.setenv("RESEARCH_RUN_ID", "r-regen")
    said = []
    assert LC.audit_records(S, tmp_path / "deps", log=said.append) == 0
    assert [(p, up) for p, up, _ in calls] == [(S, False), (P, False)], "nothing is audited again with --update"
    assert calls[1][2][S] == BUILT[S], "the requirer is audited on the record kept for PKG"
    out = run_dir / "lean-audit"
    kept = {S: out / "record" / S / LC.LOCK, P: out / "record" / P / LC.LOCK}
    assert {p: k.read_text() for p, k in kept.items()} == BUILT and (out / "record" / LC.LOCK).read_text() == BUILT[S]
    assert {p: (root / p / LC.LOCK).read_text() for p in (S, P)} == {S: record("old", ["proofs"]), P: record("old")}
    doc = json.loads((run_dir / "result.json").read_text())
    assert (doc["outcome"], doc["validation"]["status"]) == (LC.REGENERATED, "passed")
    files = LC.LA.COMMON.tracked_files(root)
    for p, k in kept.items():
        assert doc["records"][p] == {"outcome": LC.REGENERATED, "inputs_key": LC.inputs_key(p, files),
                                     "record": f"lean-audit/record/{p}/{LC.LOCK}",
                                     "record_sha256": hashlib.sha256(k.read_bytes()).hexdigest()}
    assert doc["records"]["s/proofs"] == {"outcome": LC.PASS, "inputs_key": LC.inputs_key("s/proofs", files)}, \
        "a package audited beside one has its own outcome"
    assert outcome_line(said) == {"outcome": LC.REGENERATED, "package": S, "records": doc["records"],
                                  "passes": doc["passes"]}
    for p, k in kept.items():
        (root / p / LC.LOCK).write_bytes(k.read_bytes())
    assert {LC.inputs_key(p, files): p for p in (S, P)}.items() <= doc["passes"].items(), \
        "a commit of the kept records has, for each package, a key the run passed it on"
    calls.clear()
    said.clear()
    assert LC.audit_records(S, tmp_path / "deps", log=said.append) == 0 and calls == []
    doc = json.loads((run_dir / "result.json").read_text())
    assert (doc["outcome"], doc["workload_fingerprint"]["path"]) == (LC.PASS, "reused")
    assert {p: (r["outcome"], "reused" in r) for p, r in doc["records"].items()} == {S: (LC.PASS, True), P: (LC.PASS, True)}


def test_a_requirer_that_differs_beside_another_package_is_regenerated_too(pair, tmp_path):
    """`settles` re-audits a requirer with `--update` only when it alone failed; when a package audited beside it (its
    `proved_in` one) differs too, both records check.py wrote are kept, and the run still ends `regenerated`."""
    root, calls, other = pair
    other.add("proofs")
    said = []
    out = tmp_path / "out"
    assert LC.audit_records(P, tmp_path / "deps", update=True, out=out, log=said.append) == 0
    assert [(p, up) for p, up, _ in calls] == [(P, True), (S, False)], "no --update on a requirer that didn't fail alone"
    assert {q: (out / "record" / q / LC.LOCK).read_text() for q in (P, S, "s/proofs")} == {**BUILT, "s/proofs": PROOFS}
    got = outcome_line(said)
    assert got["outcome"] == LC.REGENERATED
    assert {q: r["outcome"] for q, r in got["records"].items()} == {P: LC.REGENERATED, S: LC.REGENERATED,
                                                                    "s/proofs": LC.REGENERATED}


def test_a_package_failing_for_another_reason_keeps_no_record_though_update_wrote_one(pair, tmp_path):
    """check.py `--update` writes the build's record even when the package fails for another reason (here the kernel
    rejects a declaration). The run keeps that record nowhere: `record/` holds only `regenerated` packages', the records
    gate's to commit, and no checkout is written; its requirer, which differs only in its record, is still kept."""
    root, calls, other = pair
    other |= {S, "writes"}
    said = []
    out = tmp_path / "out"
    assert LC.audit_records(S, tmp_path / "deps", update=True, out=out, log=said.append) == 1
    assert [(p, up) for p, up, _ in calls] == [(S, True), (P, False), (P, True)]
    assert not (out / "record" / S / LC.LOCK).exists() and not (out / "record" / LC.LOCK).exists()
    assert (out / "record" / P / LC.LOCK).read_text() == BUILT[P]
    assert (root / S / LC.LOCK).read_text() == record("old", ["proofs"]), "the failed record is not written back"
    assert not any(f"commit it over {S}/{LC.LOCK}" in s for s in said) and any(f"{S}: its records fail" in s for s in said)
    got = outcome_line(said)["records"]
    assert (got[S], got[P]["outcome"]) == ({"outcome": LC.FAILED, "inputs_key": got[S]["inputs_key"]}, LC.REGENERATED)


def unwitnessed(h: str) -> str:
    """`s/props`' lock whose guarantee G, which reads `S.DrawIsCoin`, has an `unwitnessed` entry: its statement's hash."""
    return json.dumps({"guarantees": {"G": {"type_hash": h}}, "unwitnessed": {"G": h}}) + "\n"


@pytest.mark.parametrize("hold", [lambda *a, **k: None, refuse], ids=["slot", "dev-cache"])
def test_a_rename_a_requirers_unwitnessed_entry_reads_regenerates_it_in_one_update_moved_run(pair, tmp_path, monkeypatch,
                                                                                            hold):
    """A PR renames `DrawIsCoin` in Security, and statements of Security/Properties read it: 8 of its `unwitnessed` entries
    hold such a statement's hash, which the build under the new name no longer has. check.py fails that as
    `witnesses.unwitnessed-changed`, no record difference, unless `--moved` says the text under the old names is the
    record's, and it reads `--moved` only with `--update`. So `--records Security --update --moved` audits the requirer
    with `--update --moved` from the first, and both records are regenerated; with the requirer's first audit left without
    them, as before, it failed there, never reached `regenerates`, and the PR could not pass its records."""
    root, calls, _ = pair
    monkeypatch.setattr(LC.LEAN_SLOT, "hold", hold)
    (root / P / LC.LOCK).write_text(unwitnessed("old"))
    moves = tmp_path / "MOVES.json"
    moves.write_text(json.dumps({"S.DrawIsCoin": "S.SampleIsCoin"}))
    fake, mine = LC.subprocess.run, []

    def run(argv, **kw):
        if argv[1:2] != [str(LC.AUDIT_PY)] or os.path.relpath(argv[-1], kw["cwd"]) != P:
            return fake(argv, **kw)
        at, out, update = Path(argv[-1]), Path(argv[argv.index("--out") + 1]), "--update" in argv
        moved = json.loads(Path(argv[argv.index("--moved") + 1]).read_text()) if "--moved" in argv else None
        mine.append((update, moved))
        if moved is not None and not update:
            return type("P", (), {"returncode": 2})()                      # check.py: --moved goes with --update
        entry = json.loads((at / LC.LOCK).read_text())["unwitnessed"]["G"]
        same = entry == "new" or (entry == "old" and (moved or {}).get("S.DrawIsCoin") == "S.SampleIsCoin")
        fails = [] if same else ["witnesses.unwitnessed-changed: G changed since `unwitnessed` listed it"]
        if update and same:
            (at / LC.LOCK).write_text(unwitnessed("new"))
        out.mkdir(parents=True, exist_ok=True)
        (out / "audit.json").write_text(json.dumps({"passed": not fails, "packages": [
            {"package": str(at.resolve()), "failures": fails}]}) + "\n")
        return type("P", (), {"returncode": 1 if fails else 0})()

    monkeypatch.setattr(LC.subprocess, "run", run)
    said, out = [], tmp_path / "out"
    assert LC.audit_records(S, tmp_path / "deps", update=True, moved=moves, out=out, log=said.append) == 0
    assert [(p, up) for p, up, _ in calls] == [(S, True)] and mine == [(True, {"S.DrawIsCoin": "S.SampleIsCoin"})], \
        "the requirer's one audit takes --update --moved"
    assert {q: (out / "record" / q / LC.LOCK).read_text() for q in (S, P)} == {S: BUILT[S], P: unwitnessed("new")}
    got = outcome_line(said)
    assert (got["outcome"], got["records"][P]["outcome"]) == (LC.REGENERATED, LC.REGENERATED)
    assert any(f"{P} requires {S}: check.py --update --moved on it from its first audit" in s for s in said)

    calls.clear()
    mine.clear()
    said.clear()
    for q, text in ((S, record("old", ["proofs"])), (P, unwitnessed("old"))):
        (root / q / LC.LOCK).write_text(text)
    assert LC.audit_records(S, tmp_path / "deps", update=True, out=tmp_path / "out2", log=said.append) == 1
    assert mine == [(False, None)], "without --moved the requirer fails on its first audit and is not updated"
    assert outcome_line(said)["records"][P]["outcome"] == LC.FAILED


def test_a_package_beside_two_that_fails_otherwise_in_one_ends_failed_and_its_record_goes(pair, tmp_path):
    """A package audited beside two others takes the worse outcome. A report that fails without naming the record that
    settles it (`regenerated`) is `failed` though an earlier report's record was kept, and that record is removed."""
    root, _, _ = pair
    out = tmp_path / "out"
    kept = out / "record" / "s/proofs" / LC.LOCK
    kept.parent.mkdir(parents=True)
    kept.write_text(PROOFS)
    for d, rep in [(out, {"failures": ["assumptions.unread: x"], "regenerated": "r"}),
                   (out / "requirer-s_props", {"failures": ["fresh.rejected: x"]})]:
        d.mkdir(parents=True, exist_ok=True)
        (d / "audit.json").write_text(json.dumps({"packages": [{"package": str(root / "s/proofs"), **rep}]}))
    said = []
    got = LC.verdicts({S: LC.REGENERATED}, {f"s/proofs/{LC.LOCK}": PROOFS.encode()}, out, root,
                      LC.LA.COMMON.tracked_files(root), log=said.append)
    assert got["s/proofs"]["outcome"] == LC.FAILED and "record" not in got["s/proofs"] and not kept.exists()
    assert any("s/proofs: its records fail beside another package" in s for s in said)


def test_a_run_that_runs_nothing_is_refused_with_a_result(pair, tmp_path, monkeypatch):
    _, calls, _ = pair
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(run_dir))
    monkeypatch.setenv("RESEARCH_RUN_ID", "r-refused")
    said = []
    assert LC.audit_records("nothing-matches", tmp_path / "deps", log=said.append) == 2 and calls == []
    doc = json.loads((run_dir / "result.json").read_text())
    assert (doc["outcome"], doc["validation"]["status"], doc["records"]) == (LC.REFUSED, "failed", {})
    assert outcome_line(said) == {"outcome": LC.REFUSED, "package": "nothing-matches", "records": {}, "passes": {}}


def test_a_lock_that_cant_be_read_names_no_prover_and_says_so(capsys):
    """`provers` reads a lock that isn't JSON, or isn't an object, as naming no `proved_in` (the audit then fails its policy),
    as `lean_audit.provers` does, and says so on stderr."""
    for bad in (b"not json", b"[1]"):
        assert LC.provers("s", {"s/lean-audit.json": bad}.get) == []
        assert "s/lean-audit.json can't be read" in capsys.readouterr().err
    assert LC.provers("s", {"s/lean-audit.json": b'{"proved_in": ["../p", 1]}'}.get) == ["p"]
    assert LC.provers("s", {}.get) == [] and capsys.readouterr().err == ""


def test_the_inputs_key_at_a_commit_is_read_from_git_alone_and_is_the_runs(tmp_path, monkeypatch, capsys):
    """`--inputs-key PKG --head SHA` (`keys_at`) reads the commit's objects alone, so in a clone with no worktree checked out
    it gives the key a records run of that commit computes in its checkout (`inputs_key`), over the package's closure (the
    packages its `proved_in` names, and those they require by path) and the checker's files."""
    root = tmp_path / "root"
    for f, text in {**PAIR, "tools/verity/lean/check.py": "x\n", "tools/verity/lean/tests/t.py": "", "README.md": ""}.items():
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        (root / f).write_text(text)
    git = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", "-C", str(root)]
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run([*git, "add", "-A"], check=True)
    subprocess.run([*git, "commit", "-qm", "c"], check=True)
    sha = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", "--no-checkout", str(root), str(clone)], check=True)
    monkeypatch.setattr(LC, "ROOT", root)
    files = LC.LA.COMMON.tracked_files(root)
    want = {p: LC.inputs_key(p, files) for p in (S, P, "s/proofs")}
    assert len(set(want.values())) == 3
    assert LC.keys_at([S, "props", "s/proofs"], sha, clone) == want
    assert LC.main(["--inputs-key", S, "props", "--head", sha, "--repo", str(clone)]) == 0
    assert json.loads(capsys.readouterr().out) == {S: want[S], P: want[P]}
    (root / P / "P.lean").write_text("theorem t : True := trivial\n")
    assert LC.inputs_key(S, files) != want[S], "a package `s/proofs` requires is in the closure of `s`"
    assert LC.keys_at([S], sha, clone) == {S: want[S]}, "the commit's key reads the commit, not the checkout"
    assert LC.main(["--inputs-key", "s/p", "--head", sha, "--repo", str(clone)]) == 2, "`s/p` names two packages"
    assert LC.main(["--inputs-key", S, "--head", "0" * 40, "--repo", str(clone)]) == 2


def test_the_store_lookup_takes_a_passed_result_on_the_same_inputs_and_fails_closed(tmp_path, monkeypatch):
    """`stored` asks `research data select` for a `lean-fast/v1` result whose `passes` holds the key for the package and
    takes a passed one; with no `research`, a failed or unreadable answer, or none for this package, it finds nothing, and
    says why when it couldn't ask."""
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    answer, argv, code = tmp_path / "answer.json", tmp_path / "argv", tmp_path / "code"
    (bin_ / "research").write_text(f'#!/bin/sh\necho "$@" > {argv}\ncat {answer}\necho no index >&2\nexit "$(cat {code})"\n')
    (bin_ / "research").chmod(0o755)
    code.write_text("0")
    monkeypatch.setenv("PATH", str(bin_), prepend=os.pathsep)

    def row(status="passed", pkg=S, run="r1"):
        return {"id": "art:x", "manifest": {"meta": {"run_id": run, "commit": "c" * 40, "validation": {"status": status},
                                                     "passes": {"k1": pkg}}}}

    said: list[str] = []
    answer.write_text(json.dumps([row("failed", run="r0"), row()]))
    assert STORED(S, "k1", said.append) == {"run": "r1", "commit": "c" * 40}
    assert argv.read_text().split() == ["data", "select", "--kind", LC.FAST_KIND, "--where", f"passes.k1={S}", "--json"]
    answer.write_text(json.dumps([row(pkg=P)]))
    assert STORED(S, "k1", said.append) is None and said == [], "none for this package is an answer, not a failure"
    answer.write_text("not json")
    assert STORED(S, "k1", said.append) is None and "couldn't be asked" in said.pop() and said == []
    answer.write_text(json.dumps([row()]))
    code.write_text("3")
    assert STORED(S, "k1", said.append) is None and "exit 3: no index" in said.pop() and said == []
    monkeypatch.setenv("PATH", str(tmp_path / "nowhere"))
    assert STORED(S, "k1", said.append) is None and "no `research` on PATH" in said.pop() and said == []


def test_a_pass_the_store_holds_on_the_same_inputs_is_reused_and_says_where_it_came_from(pair, tmp_path, monkeypatch):
    """A records run elsewhere passed both packages on these inputs (`stored`, standing in for the store): nothing is built,
    the run passes on the `reused` path, and each record names the run and commit it reuses."""
    root, calls, _ = pair
    src, asked = {"run": "r-other", "commit": "c" * 40}, []
    monkeypatch.setattr(LC, "stored", lambda p, k, log: asked.append((p, k)) or src)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(run_dir))
    monkeypatch.setenv("RESEARCH_RUN_ID", "r-reuse")
    assert LC.audit_records(S, tmp_path / "deps", log=[].append) == 0 and calls == []
    doc = json.loads((run_dir / "result.json").read_text())
    assert (doc["outcome"], doc["workload_fingerprint"]["path"], doc["passes"]) == (LC.PASS, "reused", {})
    assert {p: r["reused_from"] for p, r in doc["records"].items()} == {S: src, P: src}
    files = LC.LA.COMMON.tracked_files(root)
    assert asked == [(S, LC.inputs_key(S, files)), (P, LC.inputs_key(P, files))]


def module(lib: Path, name: str, olean: bytes = b"olean\x02rest", trace: str | None = None, hashed: str = "aa") -> Path:
    """A module's outputs under ``lib`` as Lake leaves them: `.olean`, `.ilean`, their `.hash` files and the trace."""
    base = lib / name
    base.parent.mkdir(parents=True, exist_ok=True)
    base.with_suffix(".olean").write_bytes(olean)
    base.with_suffix(".ilean").write_text("{}")
    base.with_name(f"{base.name}.olean.hash").write_text(hashed)
    base.with_name(f"{base.name}.ilean.hash").write_text("bb")
    t = base.with_suffix(".trace")
    t.write_text(trace if trace is not None else json.dumps({"outputs": {"o": ["aa.olean"], "i": "bb.ilean", "c": "cc.c"}}))
    return t


def test_damaged_finds_each_module_an_interrupted_build_left_half_written(tmp_path):
    lib = tmp_path / "lib"
    module(lib, "P/Good")
    bad = [module(lib, "P/NoOlean"), module(lib, "P/NotOlean", olean=b"\0\0\0"), module(lib, "P/Stale", hashed="zz"),
           module(lib, "P/Cut", trace='{"outputs": {"o": ["aa.ol')]
    (lib / "P" / "NoOlean.olean").unlink()
    assert LC.damaged(lib) == sorted(bad)
    assert LC.damaged(tmp_path / "none") == []


def test_damaged_reads_each_output_a_module_system_trace_names_beside_it(tmp_path):
    """A `module` file's trace names three oleans (`o`), `.ir` (`r`) and `.ir.sig` (`rs`) beside its `.ilean`, as Lean 4.34's
    Lake writes them, all but the `.ilean` in the olean format; each is checked as the `.olean` is."""
    lib = tmp_path / "lib"
    outs = {"o": ["a1.olean", "a2.olean.server", "a3.olean.private"], "i": "b1.ilean", "r": "c1.ir", "rs": "c2.ir.sig",
            "c": "d1.c", "m": True}

    def ported(name: str) -> Path:
        base = lib / name
        base.parent.mkdir(parents=True, exist_ok=True)
        for n in [*outs["o"], outs["i"], outs["r"], outs["rs"]]:
            digest, _, ext = n.partition(".")
            f = base.with_name(f"{base.name}.{ext}")
            f.write_bytes(b"x" if ext == "ilean" else b"olean\x02rest")
            f.with_name(f"{f.name}.hash").write_text(digest)
            os.utime(f, ns=(10**18, 10**18))
        t = base.with_suffix(".trace")
        t.write_text(json.dumps({"outputs": outs}))
        return t

    ported("P/Good")
    bad = [ported("P/NoPrivate"), ported("P/NotServer"), ported("P/NotIr"), ported("P/StaleIr"), ported("P/NoSig")]
    (lib / "P" / "NoPrivate.olean.private").unlink()
    for f in (lib / "P" / "NotServer.olean.server", lib / "P" / "NotIr.ir"):
        f.write_bytes(b"\0\0\0")
        os.utime(f, ns=(10**18, 10**18))
    (lib / "P" / "StaleIr.ir.hash").write_text("zz")
    (lib / "P" / "NoSig.ir.sig").unlink()
    assert LC.damaged(lib) == sorted(bad)


def torn(lib: Path, name: str) -> Path:
    """A module whose `.olean` was written after its `.hash`, the trace and the `.hash` still agreeing: what a build killed
    after `lean` wrote its outputs leaves, and a `.ltar` restore from Lake's cache over a stale `.hash`."""
    t = module(lib, name)
    h = lib / f"{name}.olean.hash"
    os.utime(h, ns=(10**18, 10**18))
    os.utime(lib / f"{name}.olean", ns=(10**18 + 1, 10**18 + 1))
    return t


def test_an_output_newer_than_its_hash_is_half_written_and_loses_every_hash_with_its_trace(tmp_path):
    """Lake reads an import's hash from its `.hash` without hashing the `.olean`, so a module rebuilt against the torn one
    records the old hash for the new bytes; the trace alone going would leave the `.hash` that a `.ltar` restore keeps."""
    tree = tmp_path / "tree"
    lib = tree / "pkg" / ".lake" / "build" / "lib" / "lean"
    good = module(lib, "P/Good")
    os.utime(lib / "P" / "Good.olean", ns=(10**18, 10**18))
    os.utime(lib / "P" / "Good.olean.hash", ns=(10**18, 10**18))
    bad = torn(lib, "P/Torn")
    ir = tree / "pkg" / ".lake" / "build" / "ir" / "P"
    ir.mkdir(parents=True)
    (ir / "Torn.c").write_text("c")
    (ir / "Torn.c.hash").write_text("cc")
    (ir / "Good.c.hash").write_text("cc")
    assert LC.damaged(lib) == [bad], "an output written after its `.hash`; one written at the same time is not"
    said = []
    LC.repair(tree, ["pkg"], {}, None, said.append)
    assert not bad.exists() and not list(lib.glob("P/Torn.*.hash")) and not (ir / "Torn.c.hash").exists()
    assert (lib / "P" / "Torn.olean").exists() and (ir / "Torn.c").exists(), "the outputs stay for Lake to replace"
    assert good.exists() and (lib / "P" / "Good.olean.hash").exists() and (ir / "Good.c.hash").exists()
    assert any("1 module(s) half-written by an earlier build, rebuilt: P/Torn" in s for s in said)


def test_a_module_without_a_trace_loses_the_hash_files_it_kept(tmp_path):
    """A tree can hold an `.olean` newer than its `.hash` and no `.trace`, as a `repair` that drops only the trace leaves it;
    `damaged` reads only traces, so it goes unflagged, and the next `.ltar` restore would keep the stale `.hash` beside a
    new trace."""
    tree = tmp_path / "tree"
    lib = tree / "pkg" / ".lake" / "build" / "lib" / "lean"
    good = module(lib, "P/Good")
    module(lib, "P/W1").unlink()
    ir = tree / "pkg" / ".lake" / "build" / "ir" / "P"
    ir.mkdir(parents=True)
    for name in ("W1", "Good"):
        (ir / f"{name}.c").write_text("c")
        (ir / f"{name}.c.hash").write_text("cc")
    assert LC.damaged(lib) == [], "no trace to read"
    said = []
    LC.repair(tree, ["pkg"], {}, None, said.append)
    assert not list(lib.glob("P/W1.*.hash")) and not (ir / "W1.c.hash").exists()
    assert (lib / "P" / "W1.olean").exists() and (ir / "W1.c").exists(), "the outputs stay for Lake to replace"
    assert good.exists() and (lib / "P" / "Good.olean.hash").exists() and (ir / "Good.c.hash").exists()
    assert any("1 module(s) without a trace kept their .hash files, dropped: P/W1" in s for s in said)


def test_a_tree_left_mid_way_is_repaired_and_said_so_and_a_clean_run_clears_the_mark(fast, tmp_path, monkeypatch):
    _, tree, _ = fast
    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
    (tree / LC.DIRTY).write_text(json.dumps({"run": "r-old", "pid": 7, "package": "pkg", "phase": "deps", "at": "T"}))
    lib = tree / "pkg" / ".lake" / "build" / "lib" / "lean"
    module(lib, "P/Good")
    half = module(lib, "P/Half")
    (lib / "P" / "Half.olean").unlink()
    (tree / "pkg" / ".lake" / "packages" / "dep").mkdir(parents=True)
    dropped, said = [], []
    monkeypatch.setattr(LC, "unlink", lambda p, t, links: dropped.append(p))
    assert LC.audit_records("pkg", tmp_path / "deps", out=tmp_path / "out", log=said.append) == 0
    assert any("left mid-way by r-old (pid 7, pkg in phase deps, T)" in s for s in said)
    assert any("1 module(s) half-written by an earlier build, rebuilt: P/Half" in s for s in said)
    assert not half.exists() and (lib / "P" / "Good.trace").exists(), "only the half-written module loses its trace"
    assert dropped == ["pkg"], "the dependencies the run was taking are taken again"
    assert not (tree / LC.DIRTY).exists(), "a run that ends without a killed check.py clears the mark"


def test_a_killed_check_leaves_the_mark_for_the_next_run(fast, tmp_path, monkeypatch):
    _, tree, calls = fast
    monkeypatch.delenv("RESEARCH_RUN_DIR", raising=False)
    monkeypatch.setenv("RESEARCH_RUN_ID", "r-killed")

    def run(argv, **_):
        return type("P", (), {"returncode": -9 if argv[-1].endswith("Req") else 0})()

    monkeypatch.setattr(LC.subprocess, "run", run)
    said = []
    assert LC.audit_records("pkg", tmp_path / "deps", out=tmp_path / "out", log=said.append) == -9
    left = json.loads((tree / LC.DIRTY).read_text())
    assert (left["run"], left["package"], left["phase"], left["killed"]) == ("r-killed", "pkg/Req", "check", -9)
    monkeypatch.setattr(LC.subprocess, "run", lambda argv, **_: type("P", (), {"returncode": 0})())
    said.clear()
    assert LC.audit_records("pkg", tmp_path / "deps", out=tmp_path / "out", log=said.append) == 0
    assert any("left mid-way by r-killed" in s and "check.py exit -9" in s for s in said)
    assert not (tree / LC.DIRTY).exists()


def test_after_a_run_left_mid_way_dependencies_off_their_pinned_checkout_are_taken_again(tmp_path, monkeypatch):
    tree = tmp_path / "tree"
    (tree / "pkg" / ".lake" / "packages" / "dep" / ".git").mkdir(parents=True)
    (tree / "pkg" / "lake-manifest.json").write_text(json.dumps({"packages": [{"type": "git", "name": "dep", "rev": "abc"}]}))
    monkeypatch.setattr(LC, "git_head", lambda d: "def")
    links, said = {"pkg": "k"}, []
    LC.repair(tree, ["pkg"], links, None, said.append)
    assert links == {"pkg": "k"} and not said, "a tree no run left mid-way keeps its dependencies"
    LC.repair(tree, ["pkg"], links, {"run": "r-old", "package": "other", "phase": "check"}, said.append)
    assert links == {} and not (tree / "pkg" / ".lake" / "packages").exists()
    assert any("dep is not the checkout its manifest pins (abc)" in s for s in said)


@pytest.fixture()
def trying(tmp_path, monkeypatch):
    """`try_file` against nested stand-in packages `Sec` and `Sec/Proofs` (whose modules sit under `srcDir = ".."`, as
    Security/Proofs' do), a slot tree, no sandbox, and `build` and check.py recorded instead of run."""
    root = tmp_path / "root"
    (root / "Sec" / "Proofs" / "Flock").mkdir(parents=True)
    (root / "Sec" / "Proofs" / "lakefile.toml").write_text('name = "p"\nsrcDir = ".."\n')
    (root / "Sec" / "Proofs" / "Flock" / "X.lean").write_text("import Mathlib\n")
    lake = tmp_path / "lake"
    lake.write_text("")
    cache = tmp_path / "cache"
    tree = cache / LC.RECORDS / "local"
    monkeypatch.setattr(LC, "ROOT", root)
    monkeypatch.setattr(LC.LA.COMMON, "cache_dir", lambda: cache)
    monkeypatch.setattr(LC.LA.COMMON, "tracked_files", lambda _: ["Sec/Proofs/Flock/X.lean"])
    monkeypatch.setattr(LC.LA, "packages", lambda _: ["Sec", "Sec/Proofs"])
    monkeypatch.setattr(LC.LA, "deps_key", lambda _: "k")
    monkeypatch.setattr(LC, "closure", lambda p: [p, "Sec"])
    monkeypatch.setattr(LC, "records_tree", lambda: tree)
    monkeypatch.setattr(LC, "sync", lambda *_: 1)
    monkeypatch.setattr(LC, "sandboxed", lambda: False)
    monkeypatch.setattr(LC.shutil, "which", lambda _: str(lake))
    monkeypatch.setattr(LC.LEAN_SLOT, "hold", lambda *a, **k: None)
    calls = {"build": [], "run": [], "link": [], "lean_rc": 0, "deps_rc": 0, "lean_text": ""}

    def link(pkg, t, deps, links, log=print, copy=False):
        calls["link"].append((pkg, copy))
        links[pkg] = "k"
        return True

    def build(pkg, t, lake_, targets=None, sandbox=True, cmd=None):
        calls["build"].append((pkg, t, targets, sandbox, cmd))
        return (calls["lean_rc"], calls["lean_text"]) if cmd else (calls["deps_rc"], "")

    monkeypatch.setattr(LC, "link", link)
    monkeypatch.setattr(LC, "build", build)
    monkeypatch.setattr(LC.subprocess, "run", lambda argv, **kw: calls["run"].append(argv))
    return tree, calls


def test_try_builds_the_files_imports_then_runs_lean_on_it_in_its_owning_package_and_never_check(trying, tmp_path):
    tree, calls = trying
    calls["lean_rc"] = 1
    said = []
    assert LC.try_file("Sec/Proofs/Flock/X.lean", tmp_path / "deps", log=said.append) == 1, "lean's exit status"
    assert calls["build"] == [("Sec/Proofs", tree, ["+Proofs.Flock.X:deps"], False, None),
                              ("Sec/Proofs", tree, None, False, ["env", "lean", ".lake/axiom-probe/Flock/X.lean"])]
    assert calls["link"] == [("Sec/Proofs", True)], "the owning package's dependencies, a copy as --records takes"
    assert calls["run"] == [], "no check.py, audit or --update"
    assert not (tree / LC.DIRTY).exists() and any("lean exit 1" in s for s in said)
    assert LC.main(["--try", "Sec/Proofs/Flock/X.lean", "--tree", str(tree)]) == 1
    with pytest.raises(SystemExit):
        LC.main(["--try", "Sec/Proofs/Flock/X.lean", "--records", "Sec"])


def test_try_gives_lean_the_packages_lean_options_as_its_build_does(trying, tmp_path):
    """`lake env lean` passes no `[leanOptions]`: without them a try under `autoImplicit = false` takes an unknown name for
    an auto-bound variable and passes what the package's build refuses. They're read from the slot tree's package."""
    tree, calls = trying
    (tree / "Sec" / "Proofs").mkdir(parents=True)
    (tree / "Sec" / "Proofs" / "lakefile.toml").write_text(
        'name = "p"\nsrcDir = ".."\n\n[leanOptions]\nautoImplicit = false\nbackward.proofsInPublic = true\n')
    said = []
    assert LC.try_file("Sec/Proofs/Flock/X.lean", tmp_path / "deps", log=said.append) == 0
    opts = ["-Dweak.autoImplicit=false", "-Dweak.backward.proofsInPublic=true"]
    assert calls["build"][1][4] == ["env", "lean", *opts, ".lake/axiom-probe/Flock/X.lean"]
    assert any(f"then lake env lean {' '.join(opts)} on .lake/axiom-probe/Flock/X.lean" in s for s in said), said


def test_try_elaborates_the_files_probed_copy_and_ends_with_the_theorems_beyond_the_standard_axioms(trying, tmp_path,
                                                                                                     monkeypatch):
    """Lean's exit 0 isn't proved: the last line names the theorem resting on `sorry`, and an error on an appended
    `#print axioms` line leaves the file's status 0."""
    tree, calls = trying
    x = LC.ROOT / "Sec" / "Proofs" / "Flock" / "X.lean"
    x.write_text("namespace F\ntheorem holey : 1 = 2 := sorry\ntheorem clean : True := trivial\nend F\n")
    seen = []

    def build(pkg, t, lake_, targets=None, sandbox=True, cmd=None):
        if cmd is None:
            return 0, ""
        seen.append((t / pkg / cmd[2]).read_text())
        return 1, (".lake/axiom-probe/Flock/X.lean:8:14: error: Unknown constant `F.clean`\n"
                   "'F.holey' depends on axioms: [sorryAx]\n")

    monkeypatch.setattr(LC, "build", build)
    said = []
    assert LC.try_file("Sec/Proofs/Flock/X.lean", tmp_path / "deps", log=said.append) == 0
    assert seen[0].endswith("end F\n\n#print axioms F.holey\n#print axioms F.clean\n")
    assert not (tree / "Sec" / "Proofs" / ".lake" / "axiom-probe" / "Flock" / "X.lean").exists()
    assert "lean exit 0" in said[-2] and said[-1] == (
        "lean-try: Sec/Proofs/Flock/X.lean: axiom probe: 1 of the 1 it read use axioms beyond propext, Classical.choice "
        "and Quot.sound: F.holey (sorryAx); no answer for 1 (an error on its `#print axioms` line above): F.clean")


def test_try_forgets_both_modules_an_import_clash_names_so_the_next_try_rebuilds_them(trying, tmp_path):
    """A module can fail on one tree with one import replayed beside a fresh other, each built against another version of a
    shared dependency, while it elaborates from source everywhere else."""
    tree, calls = trying
    proofs = tree / "Sec" / "Proofs" / ".lake" / "build" / "lib" / "lean"
    sec = tree / "Sec" / ".lake" / "build" / "lib" / "lean"
    stale, fresh, other = module(proofs, "Proofs/Pouw/W1BudgetAbort"), module(sec, "Defs/RowSeed"), module(proofs, "P/Other")
    calls["lean_rc"] = 1
    calls["lean_text"] = ("Flock/X.lean:1:0: error: import Proofs.Pouw.W1BudgetAbort failed, environment already contains "
                          "'Pouw.PearlC.W1.Instr.exec.eq_10' from Defs.RowSeed\n")
    said = []
    assert LC.try_file("Sec/Proofs/Flock/X.lean", tmp_path / "deps", log=said.append) == 1, "lean's status, unchanged"
    assert not stale.exists() and not fresh.exists() and other.exists()
    assert not list(proofs.glob("Proofs/Pouw/W1BudgetAbort.*.hash")) and not list(sec.glob("Defs/RowSeed.*.hash"))
    assert any("import clash between Proofs.Pouw.W1BudgetAbort and Defs.RowSeed" in s for s in said)
    calls["lean_text"] = "Flock/X.lean:3:0: error: unknown identifier 'x'\n"
    said.clear()
    assert LC.try_file("Sec/Proofs/Flock/X.lean", tmp_path / "deps", log=said.append) == 1
    assert other.exists() and not any("clash" in s for s in said), "any other error forgets nothing"


def test_try_with_siblings_compiles_the_file_then_imports_it_beside_its_built_siblings(trying, tmp_path, monkeypatch):
    tree, calls = trying
    (tree / "Sec" / "Proofs" / "Flock").mkdir(parents=True)
    (tree / "Sec" / "Proofs" / "lakefile.toml").write_text('name = "p"\nsrcDir = ".."\n\n[[lean_lib]]\nname = "Proofs"\n')
    (tree / "Sec" / "Proofs" / "Flock" / "X.lean").write_text("module\n")
    (tree / "Sec" / "Proofs" / "Y.lean").write_text("module\n")
    olean = tree / "Sec" / "Proofs" / LC.SIBLINGS.LIB / "Proofs" / "Y.olean"
    olean.parent.mkdir(parents=True)
    olean.write_bytes(b"olean")
    os.utime(olean, (2_000_000_000.0,) * 2)
    ran = []

    def run(argv, cwd):
        ran.append(argv)
        return 1, "error: import Proofs.Y failed, environment already contains 'f' from Proofs.Flock.X\n", 2.0, 700

    monkeypatch.setattr(LC.SIBLINGS, "run", run)
    said = []
    assert LC.try_file("Sec/Proofs/Flock/X.lean", tmp_path / "deps", log=said.append, siblings=True) == 1
    lean = calls["build"][1][4]
    assert lean[:3] == ["env", "lean", "-R"] and lean[3].startswith(".lake/axiom-probe/siblings-") and lean[4] == "-o"
    assert lean[5].endswith("/Proofs/Flock/X.olean") and lean[6] == f"{lean[3]}/Proofs/Flock/X.lean", \
        "the probed copy, at its module's path under -R"
    assert ran[0][1:4] == ["env", "lean", "--setup"]
    assert any("siblings: Proofs.Flock.X and Proofs.Y both declare 'f'" in s for s in said), said
    assert not list((tree / "Sec" / "Proofs" / LC.SIBLINGS.COPIES).iterdir()), "the scratch build is removed"
    assert LC.try_file("Sec/Proofs/Flock/X.lean", tmp_path / "deps", log=lambda _: None) == 0 and len(ran) == 1, \
        "no check without --siblings"
    with pytest.raises(SystemExit):
        LC.main(["--records", "Sec", "--siblings"])


def test_try_runs_no_lean_when_the_imports_fail_to_build(trying, tmp_path):
    _, calls = trying
    calls["deps_rc"] = 3
    assert LC.try_file("Sec/Proofs/Flock/X.lean", tmp_path / "deps", log=lambda _: None) == 3
    assert [c[4] for c in calls["build"]] == [None], "lean never runs on a file whose imports don't build"


def test_try_keeps_dependencies_no_records_run_took_and_leaves_them_out_of_links(trying, tmp_path):
    tree, calls = trying
    (tree / "Sec" / "Proofs" / ".lake" / "packages" / "mathlib").mkdir(parents=True)
    (tree / LC.LINKS).write_text(json.dumps({"Sec": "k"}))
    assert LC.try_file("Sec/Proofs/Flock/X.lean", tmp_path / "deps", log=lambda _: None) == 0
    assert calls["link"] == [] and LC.read_links(tree) == {"Sec": "k"}, "a later --records run never drops them"


def test_a_tree_s_builds_and_links_follow_a_layout_move_before_the_packages_are_synced(tmp_path, monkeypatch):
    """A slot's tree is keyed by package paths, and a move renames tracked files only: the build of `Old/Proofs` would stay
    there while `new/Proofs` compiled every module again and took its dependencies again. Under the tree's lock the build
    directories and their `LINKS` entries move with their packages, and the former directory's stale copies go: where the
    step takes the lock itself, and in `hold_tree` for every other run, before its caller reads the tree."""
    def a_tree(name: str) -> Path:
        tree = tmp_path / name
        for pkg, built in (("Old", "spec"), ("Old/Proofs", "proofs"), ("code", "code")):
            (tree / pkg / ".lake" / "build").mkdir(parents=True)
            (tree / pkg / ".lake" / "build" / "olean").write_text(built)
            (tree / pkg / "A.lean").write_text("-- a copy the last run synced\n")
        LC.write_links(tree, {"Old": "k0", "Old/Proofs": "k1", "code": "k2"})
        return tree

    def carried(tree: Path) -> bool:
        return (LC.read_links(tree) == {"new": "k0", "new/Proofs": "k1", "code": "k2"} and not (tree / "Old").exists()
                and [(tree / p / ".lake" / "build" / "olean").read_text() for p in ("new", "new/Proofs", "code")]
                == ["spec", "proofs", "code"])

    monkeypatch.setattr(LC._moved(), "recorded", lambda root: [{"Older/": "Old/"}, {"Old/": "new/"}])
    checkout = tmp_path / "checkout"
    for f in ("new/lakefile.toml", "new/A.lean", "new/Proofs/lakefile.toml", "code/lakefile.lean", "README.md"):
        (checkout / f).parent.mkdir(parents=True, exist_ok=True)
        (checkout / f).write_text("")
    monkeypatch.setattr(LC, "ROOT", checkout)
    tree, said = a_tree("step"), []
    LC.carry_builds(tree, said.append)
    assert carried(tree) and len(said) == 2 and "new/Proofs: its build directory moved with it from Old/Proofs" in said[1]
    LC.carry_builds(tree, said.append)
    assert carried(tree) and len(said) == 2, "nothing to carry the next time"

    tree, said = a_tree("records"), []
    with open(tree / ".lock", "w") as held:
        assert LC.hold_tree(held, tree, said.append) and carried(tree) and len(said) == 2
    (checkout / "new" / "Proofs" / "lakefile.toml").unlink()
    tree = a_tree("no-package")
    with open(tree / ".lock", "w") as held:
        assert LC.hold_tree(held, tree, said.append)
    assert (tree / "Old" / "Proofs" / ".lake").is_dir() and (tree / "new" / ".lake").is_dir(), \
        "a build whose path the checkout has no package at stays, and its directory with it"


def test_try_refuses_without_a_build_slot_naming_the_machines_whose_slots_have_one(trying, tmp_path, monkeypatch):
    _, calls = trying
    monkeypatch.setattr(LC.LEAN_SLOT, "hold", refuse)
    files = {"a": "build 3\naudit 2\n", "b": "check 4\naudit 4\n", "c": "# NAME N\nbuild 2   # a note\n"}

    def read(ssh):
        if ssh[-1] not in files:
            raise OSError("exit 255: Permission denied (publickey).")
        return files[ssh[-1]]

    peers = {name: ["-i", "KEY", name] for name in ("c", "b", "a", "self")}
    monkeypatch.setattr(LC.LEAN_SLOT.SLOT, "peers", lambda: peers)
    monkeypatch.setattr(LC, "slots_there", read)
    said = []
    assert LC.try_file("Sec/Proofs/Flock/X.lean", tmp_path / "deps", log=said.append) == 2
    assert calls["build"] == [], "nothing runs without a build slot"
    [line] = [s for s in said if "nothing ran" in s]
    assert "have `build`: a, c (`verity lean try Sec/Proofs/Flock/X.lean --on MACHINE`)" in line, line
    assert line.endswith("; not read: self"), "a machine it can't read is named, never counted as having none"
    assert LC.pool_machines("build", {}, read) == ([], {})
    peers.clear()
    said.clear()
    assert LC.try_file("Sec/Proofs/Flock/X.lean", tmp_path / "deps", log=said.append) == 2
    assert any("no machine in" in s and "lean_slot.py --status" in s for s in said), "none read: how to list a machine's pools"


def test_slots_there_reads_the_path_every_machine_keeps_whatever_lean_slots_says_here(monkeypatch, tmp_path):
    monkeypatch.setattr(LC.LEAN_SLOT, "SLOTS", tmp_path / "lean-slots")
    ran = []

    def run(argv, **kw):
        ran.append(argv)
        return subprocess.CompletedProcess(argv, 255 if len(ran) > 1 else 0, "build 2\n", "Permission denied (publickey).\n")

    monkeypatch.setattr(LC.subprocess, "run", run)
    assert LC.slots_there(["-i", "KEY", "research@peer"]) == "build 2\n"
    assert ran[0][-2:] == ["research@peer", f"cat {LC.LEAN_SLOT.SLOTS_PATH}"] and "BatchMode=yes" in ran[0]
    with pytest.raises(OSError, match="exit 255: Permission denied"):
        LC.slots_there(["research@peer"])


def test_try_refuses_a_file_in_no_package_and_a_sandbox_without_dependencies(trying, tmp_path, monkeypatch):
    tree, calls = trying
    said = []
    assert LC.try_file("Elsewhere/X.lean", tmp_path / "deps", log=said.append) == 2
    assert LC.try_file("Sec/Proofs/Flock/X.lean", tmp_path / "deps", tree=tmp_path / "other", log=said.append) == 2
    monkeypatch.setattr(LC, "sandboxed", lambda: True)
    monkeypatch.setattr(LC, "link", lambda *a, **k: False)
    assert LC.try_file("Sec/Proofs/Flock/X.lean", tmp_path / "deps", log=said.append) == 2
    assert calls["build"] == [] and any("--records Sec/Proofs" in s for s in said)


def test_build_builds_the_files_modules_on_the_slot_tree_and_reports_each_files_first_error_as_json(trying, tmp_path,
                                                                                                     monkeypatch, capsys):
    """The ready mark's preflight reads `--build`'s last line: a Lean error at a position in a tracked file fails that file,
    named from the repository's root, whichever package of the build's closure holds it."""
    tree, calls = trying
    (slots := tmp_path / "lean-slots").write_text("build 1\n")
    monkeypatch.setattr(LC.LEAN_SLOT, "SLOTS", slots)
    (LC.ROOT / "Sec" / "Proofs" / "Flock" / "Y.lean").write_text("import Proofs.Flock.X\n")
    monkeypatch.setattr(LC.LA.COMMON, "tracked_files", lambda _: ["Sec/Proofs/Flock/X.lean", "Sec/Proofs/Flock/Y.lean",
                                                                   "Sec/Defs/A.lean"])
    calls["deps_rc"] = 1
    monkeypatch.setattr(LC, "build", lambda pkg, t, lake_, targets=None, sandbox=True, cmd=None: (
        calls["build"].append((pkg, targets)) or 1,
        "✖ [3/9] Building Proofs.Flock.Y\nerror: ././../Proofs/Flock/Y.lean:2:4: unknown identifier 'foo'\n  more\n"
        "error: ./../Defs/A.lean:1:0: unexpected token\nerror: ././../Proofs/Flock/Y.lean:5:0: a later one\n"
        "error: Lean exited with code 1\n"))
    assert LC.main(["--build", "Sec/Proofs/Flock/X.lean", "Sec/Proofs/Flock/Y.lean", "Elsewhere/Z.lean"]) == 1
    assert calls["build"] == [("Sec/Proofs", ["+Proofs.Flock.X", "+Proofs.Flock.Y"])] and calls["run"] == []
    doc = json.loads(capsys.readouterr().out.splitlines()[-1].removeprefix(LC.BUILD_LINE))
    assert doc["built"] == {"Sec/Proofs": 1} and doc["failed"] == {
        "Sec/Proofs/Flock/Y.lean": "error: ././../Proofs/Flock/Y.lean:2:4: unknown identifier 'foo'",
        "Sec/Defs/A.lean": "error: ./../Defs/A.lean:1:0: unexpected token"}
    assert list(doc["skipped"]) == ["Elsewhere/Z.lean"] and not (tree / LC.DIRTY).exists()


def test_build_builds_nothing_off_a_node_or_where_the_node_keeps_no_dependencies(trying, tmp_path, monkeypatch, capsys):
    """An agent VM has no Lean slots, and a node without the package's dependencies would fetch Mathlib cold: both are
    skipped with why, so the preflight passes the build to its next node, or refuses naming why."""
    _, calls = trying
    monkeypatch.setattr(LC.LEAN_SLOT, "SLOTS", tmp_path / "absent")
    assert LC.main(["--build", "Sec/Proofs/Flock/X.lean"]) == 2
    doc = json.loads(capsys.readouterr().out.splitlines()[-1].removeprefix(LC.BUILD_LINE))
    assert "not a node" in doc["skipped"]["*"] and calls["build"] == [] and calls["link"] == []
    (slots := tmp_path / "lean-slots").write_text("build 1\n")
    monkeypatch.setattr(LC.LEAN_SLOT, "SLOTS", slots)
    monkeypatch.setattr(LC, "link", lambda *a, **k: False)
    assert LC.main(["--build", "Sec/Proofs/Flock/X.lean"]) == 2
    doc = json.loads(capsys.readouterr().out.splitlines()[-1].removeprefix(LC.BUILD_LINE))
    assert "Mathlib" in doc["skipped"]["Sec/Proofs"] and doc["built"] == {} and calls["build"] == []
    with pytest.raises(SystemExit):
        LC.main(["--build", "Sec/Proofs/Flock/X.lean", "--try", "Sec/Proofs/Flock/X.lean"])


def test_build_with_no_wait_builds_nothing_when_every_build_slot_is_taken(trying, tmp_path, monkeypatch, capsys):
    """The preflight builds on a node it can fall back from with `--no-wait`: a full `build` pool is a skip naming why, at once,
    so the next node gets the build before the step's deadline; without it the slot is waited for, as before."""
    _, calls = trying
    (slots := tmp_path / "lean-slots").write_text("build 1\n")
    monkeypatch.setattr(LC.LEAN_SLOT, "SLOTS", slots)
    asked = []

    def hold(pool, **kw):
        asked.append((pool, kw))
        if not kw.get("queue", True):
            raise LC.LEAN_SLOT.Refused("all 1 Lean audit slot(s) of pool build are taken, and this request doesn't wait in line")

    monkeypatch.setattr(LC.LEAN_SLOT, "hold", hold)
    assert LC.main(["--no-wait", "--build", "Sec/Proofs/Flock/X.lean"]) == 2
    doc = json.loads(capsys.readouterr().out.splitlines()[-1].removeprefix(LC.BUILD_LINE))
    assert doc == {"built": {}, "failed": {}, "skipped": {"*": "no `build` Lean slot here (all 1 Lean audit slot(s) of pool "
                                                               "build are taken, and this request doesn't wait in line)"}}
    assert calls["build"] == [] and asked == [("build", {"or_any": False, "queue": False})]
    assert LC.main(["--build", "Sec/Proofs/Flock/X.lean"]) == 0
    assert asked[1] == ("build", {"or_any": False, "queue": True}) and calls["build"]
    with pytest.raises(SystemExit):
        LC.main(["--no-wait", "--try", "Sec/Proofs/Flock/X.lean"])


def test_module_of_reads_the_packages_src_dir(trying):
    root = LC.ROOT
    assert LC.module_of("Sec/Proofs", "Sec/Proofs/Flock/X.lean") == "Proofs.Flock.X"
    (root / "Sec" / "lakefile.lean").write_text('package sec where\n  srcDir := "src"\n')
    assert LC.module_of("Sec", "Sec/src/A/B.lean") == "A.B"
    (root / "Plain").mkdir()
    assert LC.module_of("Plain", "Plain/A/B.lean") == "A.B"


def test_the_sandbox_is_handed_the_resolved_lake_paths_of_a_linked_slot_tree(tmp_path, monkeypatch):
    """A slot tree is a link into the node's cache root, and mountinfo names a mount by its resolved path, which
    `sandbox.sh` keeps writable: the `.lake` paths (and their `packages`, kept read-only) are the resolved ones."""
    real = tmp_path / "cache-root" / "build-2"
    (real / "Sec" / ".lake" / "packages").mkdir(parents=True)
    (real / "Dep").mkdir()
    link = tmp_path / "build-2"
    link.symlink_to(real)
    monkeypatch.setattr(LC, "closure", lambda p: [p, "Dep"])
    argv = LC.in_sandbox("Sec", link, ["lake", "build"])
    box = argv.index(str(LC.SANDBOX))
    assert argv[box + 1:argv.index("--", box)] == [str(real / "Sec" / ".lake"), str(real / "Dep" / ".lake")]
    assert argv[argv.index("ro") + 1:argv.index("--", argv.index("ro"))] == [str(real / "Sec" / ".lake" / "packages")]


def test_the_records_facts_cache_is_outside_every_slot_tree_s_build_roots(tmp_path, monkeypatch):
    """The records step keeps its facts cache (`facts_dir`) beside its slot trees, so no sandboxed build in any slot's tree
    can write it: neither `check.py`'s (`writable`, path dependencies included) nor this module's (`in_sandbox`, the
    closure's `.lake`s). check.py refuses an overlapping `--facts-cache` (`build_roots_over`); this keeps that from firing."""
    monkeypatch.setattr(LC.LA.COMMON, "cache_dir", lambda: tmp_path / "cache")
    monkeypatch.setattr(LC, "closure", lambda p: [p, "Security"])
    for slot in ("build/1", "build/2", None):
        if slot:
            monkeypatch.setenv(LC.LEAN_SLOT.HELD, slot)
        else:
            monkeypatch.delenv(LC.LEAN_SLOT.HELD, raising=False)
        tree = LC.records_tree()
        pkgs = [tree / "Security", tree / "Security" / "Proofs"]
        for p in pkgs:
            p.mkdir(parents=True, exist_ok=True)
        (pkgs[1] / "lake-manifest.json").write_text(json.dumps({"packages": [{"name": "s", "type": "path", "dir": ".."}]}))
        assert LC.AUDIT.build_roots_over(LC.facts_dir(), pkgs) == []
        box = LC.in_sandbox("Security/Proofs", tree, ["lake", "build"])
        rw = box[box.index(str(LC.SANDBOX)) + 1:box.index("--", box.index(str(LC.SANDBOX)))]
        assert rw and not any(LC.facts_dir().resolve().is_relative_to(d) or Path(d).is_relative_to(LC.facts_dir().resolve())
                              for d in rw)


def test_records_takes_a_packages_lake_name_as_well_as_its_path(tmp_path):
    """`--records network_check` names the package whose `lakefile.toml` says `name = "network_check"`: matching it by
    path alone refused it as matching no package, after the run had waited for its slot."""
    pkgs = ["verity/protocols/certifier/verifier/lean", "verity/protocols/inference_only/verifier/nci/lean", "Security"]
    for p, n in zip(pkgs, ["network_check", "nci_check", "security"]):
        (tmp_path / p).mkdir(parents=True)
        (tmp_path / p / "lakefile.toml").write_text(f'name = "{n}"\n\n[[require]]\nname = "mathlib"\n')
    assert LC.matching("network_check", pkgs, tmp_path) == ["verity/protocols/certifier/verifier/lean"]
    assert LC.matching("verity/protocols/certifier/verifier/lean/", pkgs, tmp_path) == ["verity/protocols/certifier/verifier/lean"]
    assert LC.matching("network/lean", pkgs, tmp_path) == ["verity/protocols/certifier/verifier/lean"]
    assert LC.matching("network", pkgs, tmp_path) == pkgs[:2]
    assert LC.matching("network_check", pkgs) == []
