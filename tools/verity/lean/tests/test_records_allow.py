"""`lean_changed.py --records --allow CODE:ENTRY`: a failure a lane's fast path takes as known is still a failure.

An allowed failure stays in check.py's report and the run exits `ALLOWED_RC` as `allowed`, which is no pass: no pass of that
package is kept, and the record `--update` regenerated for it goes under `allowed-record/`, never `record/`, which the
records gate reads. An allowance names a failure by its code and entry, only a code of the closed list `ALLOWABLE` may be
named, one that matches no failure fails the run, and nothing but `--records` takes one: check.py has no such option
(that no caller of either file in the check, the merge queue or the ready mark's preflight names it is the root suite's
`tests/test_lean_allowances.py`).

The stand-in packages are `s`, whose `proved_in` names `s/proofs`, and `s/props`, which requires `s`, with the real `sync`,
requirers and pass cache, and a fake check.py whose reports are check.py's own (`Failure`, `reported`)."""
from __future__ import annotations

import importlib.util
import json
import os
import re
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lean_changed_allow", HERE.parent / "lean_changed.py")
LC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(LC)
AUDIT = LC.AUDIT

S, P, PROOFS = "s", "s/props", "s/proofs"
#: the two failures of the closed list, as check.py makes them: a statement reading the proofs tree (its entry the module
#: read), and a guarantee restated since `unwitnessed` listed it (its entry the guarantee)
READS = AUDIT.Failure("trees.reads", "Proofs.Sampled", "trees: G reads D in Proofs.Sampled, a module of the proofs tree")
CHANGED = AUDIT.Failure("witnesses.unwitnessed-changed", "G", "witnesses: G changed since `unwitnessed` listed it")
REJECTED = AUDIT.Failure("replay.rejected", "G", "replay: the kernel rejected G")
DIFFERS = AUDIT.Failure("guarantees.reads-changed", "G", "guarantees: what G reads changed")


def record(h: str, proved_in: list[str] | None = None) -> str:
    return json.dumps({**({"proved_in": proved_in} if proved_in else {}), "guarantees": {"G": {"type_hash": h}}}) + "\n"


#: each package's record as the fake check.py's build has it, which `--update` writes
BUILT = {S: record("new", ["proofs"]), P: record("new")}
OLD = {S: record("old", ["proofs"]), P: record("old")}
FILES = {"s/lakefile.toml": 'name = "s"\n', "s/S.lean": "",
         "s/proofs/lakefile.toml": 'name = "proofs"\n', "s/proofs/lean-audit.json": "{}\n",
         "s/proofs/lake-manifest.json": json.dumps({"packages": [{"type": "path", "dir": ".."},
                                                                 {"type": "path", "dir": "../props"}]}),
         "s/props/lakefile.toml": 'name = "props"\n', "s/props/P.lean": "",
         "s/props/lake-manifest.json": json.dumps({"packages": [{"type": "path", "dir": ".."}]})}


def refuse(*_a, **_k):
    raise LC.LEAN_SLOT.Refused("no `build` pool here")


@pytest.fixture()
def lane(tmp_path, monkeypatch):
    """`audit_records` on `s` and its requirer `s/props`, each lock `BUILT`'s unless a test writes `OLD`'s, in a recorded
    run. The fake check.py fails a package on each failure of ``standing[package]``, which no `--update` cures, and on a
    record difference while its lock is not `BUILT`'s, which `--update` writes (as check.py does, whatever else fails) and
    which the report names (`regenerated`) when it is the package's only failure; ``beside[package]`` is {another package:
    its failures} reported in that package's audit. Returns (root, the run's directory, calls: each (package, `--update`),
    standing, beside)."""
    root, cache, run_dir = tmp_path / "root", tmp_path / "cache", tmp_path / "run"
    files = {**FILES, **{f"{p}/{LC.LOCK}": text for p, text in BUILT.items()}}
    for f, text in files.items():
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        (root / f).write_text(text)
    run_dir.mkdir()
    monkeypatch.setattr(LC, "ROOT", root)
    monkeypatch.setattr(LC.LA, "ROOT", root)
    monkeypatch.setattr(LC.LA.COMMON, "cache_dir", lambda: cache)
    monkeypatch.setattr(LC.LA.COMMON, "tracked_files", lambda _: sorted(files))
    monkeypatch.setattr(LC, "records_tree", lambda: cache / LC.RECORDS / "local")
    monkeypatch.setattr(LC.LEAN_SLOT, "hold", lambda *a, **k: None)
    monkeypatch.setattr(LC, "stored", lambda *a, **k: None)
    monkeypatch.setenv("RESEARCH_RUN_DIR", str(run_dir))
    monkeypatch.setenv("RESEARCH_RUN_ID", "r-allow")
    calls, standing, beside, real = [], {}, {}, LC.subprocess.run

    def run(argv, **kw):
        if argv[1:2] != [str(LC.AUDIT_PY)]:
            return real(argv, **kw)
        at, out, update = Path(argv[-1]), Path(argv[argv.index("--out") + 1]), "--update" in argv
        p = os.path.relpath(at, kw["cwd"])
        calls.append((p, update))
        fails, regen = list(standing.get(p, [])), {}
        if update:
            (at / LC.LOCK).write_text(BUILT[p])
        elif (at / LC.LOCK).read_text() != BUILT[p]:
            fails.append(DIFFERS)
            if len(fails) == 1:
                (kept := out / "audit-x" / LC.LOCK).parent.mkdir(parents=True, exist_ok=True)
                kept.write_text(BUILT[p])
                regen = {"regenerated": str(kept)}
        reps = [AUDIT.reported({"package": str(at.resolve()), "failures": fails, **regen}),
                *(AUDIT.reported({"package": str((Path(kw["cwd"]) / q).resolve()), "failures": list(fs)})
                  for q, fs in beside.get(p, {}).items())]
        out.mkdir(parents=True, exist_ok=True)
        (out / "audit.json").write_text(json.dumps({"passed": not any(r["failures"] for r in reps), "packages": reps}) + "\n")
        return type("P", (), {"returncode": 1 if any(r["failures"] for r in reps) else 0})()

    monkeypatch.setattr(LC.subprocess, "run", run)
    return root, run_dir, calls, standing, beside


def result(run_dir: Path) -> dict:
    return json.loads((run_dir / "result.json").read_text())


def outcome_line(said: list[str]) -> dict:
    assert said[-1].startswith(LC.OUTCOME_LINE), said[-1]
    return json.loads(said[-1][len(LC.OUTCOME_LINE):])


def passes_kept() -> set[str]:
    """The inputs keys under `lean-records/passes/`."""
    return {p.stem for p in LC.passes_dir().glob("*.json")}


def keys(root: Path, p: str) -> set[str]:
    """``p``'s inputs keys with its own lock as the checkout has it and as the build would write it."""
    files = LC.LA.COMMON.tracked_files(root)
    return {LC.inputs_key(p, files), LC.inputs_key(p, files, LC.read_here, {f"{p}/{LC.LOCK}": BUILT[p].encode()})}


@pytest.mark.parametrize("hold", [lambda *a, **k: None, refuse], ids=["slot", "dev-cache"])
def test_an_allowed_failure_stays_in_the_report_and_the_run_ends_allowed_with_its_own_exit_and_no_pass(lane, tmp_path,
                                                                                                       monkeypatch, hold):
    """The one failure of `s` is one `--allow` names. check.py still fails and reports it (`failures`, by code and entry);
    the run is `allowed`, a new outcome with its own exit status, neither a pass nor `regenerated`; and no pass of `s` is
    kept, under `lean-records/passes/` or in `result.json`'s `passes`, so the next run audits it again. Its requirer, whose
    audit is clean, keeps its own pass."""
    root, run_dir, calls, standing, _ = lane
    monkeypatch.setattr(LC.LEAN_SLOT, "hold", hold)
    standing[S] = [READS]
    said = []
    rc = LC.audit_records(S, tmp_path / "deps", log=said.append, allow=["trees.reads:Proofs.Sampled"])
    assert rc == LC.ALLOWED_RC == 3 and rc not in (0, 1, 2), "its own exit status, not a pass's, check.py's or a refusal's"
    rep = json.loads((run_dir / "lean-audit" / "audit.json").read_text())["packages"][0]
    assert (rep["failures"], rep["failure_codes"], rep["failure_entries"]) == ([str(READS)], ["trees.reads"],
                                                                               ["Proofs.Sampled"]), "still a failure there"
    doc = result(run_dir)
    assert (doc["outcome"], doc["validation"]["status"]) == (LC.ALLOWED, "failed") == ("allowed", "failed")
    assert (doc["allow"], doc["stale_allowances"]) == (["trees.reads:Proofs.Sampled"], [])
    assert doc["workload_fingerprint"]["allow"] == ["trees.reads:Proofs.Sampled"]
    assert doc["records"][S] == {"outcome": LC.ALLOWED, "inputs_key": doc["records"][S]["inputs_key"],
                                 "allowed": ["trees.reads:Proofs.Sampled"]}
    assert doc["records"][P]["outcome"] == LC.PASS
    assert outcome_line(said) == {"outcome": LC.ALLOWED, "package": S, "records": doc["records"], "passes": doc["passes"],
                                  "allow": doc["allow"], "stale": []}
    assert not keys(root, S) & (set(doc["passes"]) | passes_kept()), "no pass of the package with the allowed failure"
    assert set(doc["passes"].values()) == {P} and keys(root, P) & set(doc["passes"]) and set(doc["passes"]) <= passes_kept()
    assert any(f"exit {LC.ALLOWED_RC}" in s and "no pass" in s for s in said)
    calls.clear()
    assert LC.audit_records(S, tmp_path / "deps", log=said.append, allow=["trees.reads:Proofs.Sampled"]) == LC.ALLOWED_RC
    assert calls == [(S, False)], "the next run audits the allowed package again, and reuses its requirer's pass"


def test_the_outcomes_are_ordered_and_allowed_is_between_regenerated_and_failed():
    assert LC.OUTCOMES == (LC.PASS, LC.REGENERATED, LC.ALLOWED, LC.FAILED, LC.REFUSED)
    assert LC.overall({"a": {"outcome": LC.REGENERATED}, "b": {"outcome": LC.ALLOWED}}) == LC.ALLOWED
    assert LC.overall({"a": {"outcome": LC.FAILED}, "b": {"outcome": LC.ALLOWED}}) == LC.FAILED


def test_without_the_allowance_the_same_failure_fails_the_run(lane, tmp_path):
    _, run_dir, _, standing, _ = lane
    standing[S] = [READS]
    said = []
    assert LC.audit_records(S, tmp_path / "deps", log=said.append) == 1
    doc = result(run_dir)
    assert doc["outcome"] == LC.FAILED and "allow" not in doc and "allowed" not in doc["records"][S]
    assert outcome_line(said).keys() == {"outcome", "package", "records", "passes"}, "a run given no --allow says none"


@pytest.mark.parametrize("hold", [lambda *a, **k: None, refuse], ids=["slot", "dev-cache"])
def test_update_keeps_the_record_of_an_allowed_package_outside_record_and_says_where(lane, tmp_path, monkeypatch, hold):
    """`s` fails on a record difference and on an allowed failure. `--update` settles the difference and the allowed
    failure is the only one left, so the run regenerates the record it can and keeps it under `allowed-record/`, which the
    records gate does not read; `record/` holds only its requirer's, whose only failure was a record difference, audited on
    the record regenerated for `s`. A line says where each is."""
    root, run_dir, calls, standing, _ = lane
    monkeypatch.setattr(LC.LEAN_SLOT, "hold", hold)
    for p, text in OLD.items():
        (root / p / LC.LOCK).write_text(text)
    standing[S] = [CHANGED]
    said = []
    assert LC.audit_records(S, tmp_path / "deps", update=True, log=said.append,
                            allow=["witnesses.unwitnessed-changed:G"]) == LC.ALLOWED_RC
    assert calls == [(S, True), (P, False), (P, True)]
    out = run_dir / "lean-audit"
    kept = out / "allowed-record" / S / LC.LOCK
    assert kept.read_text() == BUILT[S]
    assert not (out / "record" / S / LC.LOCK).exists() and not (out / "record" / LC.LOCK).exists(), \
        "nothing of the allowed package is in the directory the records gate reads"
    assert (out / "record" / P / LC.LOCK).read_text() == BUILT[P]
    assert sorted(str(f.relative_to(out / "record")) for f in (out / "record").rglob(LC.LOCK)) == [f"{P}/{LC.LOCK}"]
    assert any(str(kept) in s and "outside record/" in s for s in said), "it prints where the record is"
    assert any(str(out / "record" / P / LC.LOCK) in s and f"commit it over {P}/{LC.LOCK}" in s for s in said)
    doc = result(run_dir)
    assert doc["outcome"] == LC.ALLOWED
    assert (doc["records"][S]["outcome"], doc["records"][S]["record"]) == (LC.ALLOWED,
                                                                          f"lean-audit/allowed-record/{S}/{LC.LOCK}")
    assert (doc["records"][P]["outcome"], doc["records"][P]["record"]) == (LC.REGENERATED,
                                                                          f"lean-audit/record/{P}/{LC.LOCK}")
    assert set(doc["passes"].values()) == {P}, "the requirer's pass is kept, the allowed package's is not"
    if hold is not refuse:
        assert (root / S / LC.LOCK).read_text() == OLD[S], "and the run writes no record of it into the checkout"


def test_an_unrecorded_run_writes_no_record_of_an_allowed_package_into_the_checkout(lane, tmp_path, monkeypatch):
    """Outside a recorded run the slot path writes each regenerated record back into the checkout, where the next commit
    takes it. It writes the requirer's, which is `regenerated`, and not the allowed package's, which is under
    `allowed-record/` for the lane to commit knowingly."""
    root, _, _, standing, _ = lane
    monkeypatch.delenv("RESEARCH_RUN_DIR")
    monkeypatch.delenv("RESEARCH_RUN_ID")
    for p, text in OLD.items():
        (root / p / LC.LOCK).write_text(text)
    standing[S] = [CHANGED]
    out = tmp_path / "out"
    assert LC.audit_records(S, tmp_path / "deps", update=True, out=out, log=lambda _s: None,
                            allow=["witnesses.unwitnessed-changed:G"]) == LC.ALLOWED_RC
    assert {p: (root / p / LC.LOCK).read_text() for p in (S, P)} == {S: OLD[S], P: BUILT[P]}
    assert (out / "allowed-record" / S / LC.LOCK).read_text() == BUILT[S]
    assert not (out / "record" / S / LC.LOCK).exists() and not (out / "record" / LC.LOCK).exists()


def test_a_requirer_failing_on_record_differences_beside_an_allowed_failure_is_audited_again_with_update(lane, tmp_path):
    """The requirer's first audit, without `--update`, fails on a record difference and on an allowed failure: it is
    audited again with `--update`, which settles the difference, and ends `allowed` with its record outside `record/`.
    `s`, whose only failure was a record difference, is `regenerated` as ever, with its pass and its record in `record/`."""
    root, run_dir, calls, standing, _ = lane
    for p, text in OLD.items():
        (root / p / LC.LOCK).write_text(text)
    standing[P] = [READS]
    said = []
    assert LC.audit_records(S, tmp_path / "deps", update=True, log=said.append,
                            allow=["trees.reads:Proofs.Sampled"]) == LC.ALLOWED_RC
    assert calls == [(S, True), (P, False), (P, True)]
    assert any(f"{P} requires {S}" in s and "or one --allow names" in s for s in said)
    out = run_dir / "lean-audit"
    assert (out / "allowed-record" / P / LC.LOCK).read_text() == BUILT[P] and not (out / "record" / P).exists()
    assert (out / "record" / S / LC.LOCK).read_text() == BUILT[S] == (out / "record" / LC.LOCK).read_text()
    doc = result(run_dir)
    assert {q: r["outcome"] for q, r in doc["records"].items()} == {S: LC.REGENERATED, P: LC.ALLOWED}
    assert set(doc["passes"].values()) == {S}


def test_an_allowance_never_covers_a_record_difference_which_takes_update(lane, tmp_path):
    """Without `--update` a package failing on a record difference beside an allowed failure fails: check.py names no
    record that settles it, and the run says which run does."""
    root, run_dir, calls, standing, _ = lane
    (root / S / LC.LOCK).write_text(OLD[S])
    standing[S] = [READS]
    said = []
    assert LC.audit_records(S, tmp_path / "deps", log=said.append, allow=["trees.reads:Proofs.Sampled"]) == 1
    assert any(f"run `--records {S} --update` with the same --allow" in s for s in said)
    doc = result(run_dir)
    assert (doc["outcome"], doc["records"][S]["outcome"]) == (LC.FAILED, LC.FAILED)
    assert not (run_dir / "lean-audit" / "record").exists() and not (run_dir / "lean-audit" / "allowed-record").exists()
    assert not keys(root, S) & set(doc["passes"])


def test_a_failure_no_allowance_names_beside_an_allowed_one_fails_the_run(lane, tmp_path):
    root, run_dir, _, standing, _ = lane
    standing[S] = [READS, REJECTED]
    assert LC.audit_records(S, tmp_path / "deps", update=True, log=lambda _s: None,
                            allow=["trees.reads:Proofs.Sampled"]) == 1
    doc = result(run_dir)
    assert (doc["outcome"], doc["records"][S]["outcome"], doc["stale_allowances"]) == (LC.FAILED, LC.FAILED, [])
    assert not keys(root, S) & set(doc["passes"])


def test_a_sampled_mismatch_beside_an_allowed_failure_is_still_compiled_again_and_audited_once_more(lane, tmp_path,
                                                                                                    monkeypatch):
    """check.py's sampled recompile can fail a package (a module the slot tree kept compiles to other outputs, which
    Lean's nondeterminism does) whose only other failure is one `--allow` names. The run then forgets the module and
    audits the package once more, as it does beside a record difference, and the second audit's failures are the
    package's: the allowed one alone, so the run ends `allowed`, the first audit's reports apart from the verdict.
    Without the allowance that failure is one like any other, and nothing is audited again."""
    root, run_dir, calls, standing, _ = lane
    sampled = AUDIT.Failure("sampled", "S.M", "sampled: S.M: the build's S/M.olean differs from what its source compiles to")
    fake, left = LC.subprocess.run, []

    def run(argv, **kw):
        got = fake(argv, **kw)
        if argv[1:2] == [str(LC.AUDIT_PY)] and sampled in standing.get(S, []) and Path(argv[-1]).name == S:
            doc = json.loads((f := Path(argv[argv.index("--out") + 1]) / "audit.json").read_text())
            doc["packages"][0]["sampled"] = {"mismatched": {"S.M": doc["packages"][0]["package"]}}
            f.write_text(json.dumps(doc) + "\n")
            standing[S] = left
        return got

    monkeypatch.setattr(LC.subprocess, "run", run)
    standing[S], left[:] = [READS, sampled], [READS]
    said = []
    assert LC.audit_records(S, tmp_path / "deps", log=said.append, allow=["trees.reads:Proofs.Sampled"]) == LC.ALLOWED_RC
    assert calls == [(S, False), (S, False), (P, False)], "one audit more of the package, never a third"
    aside = json.loads((run_dir / "lean-audit" / "mismatched" / "s" / "audit.json").read_text())["packages"][0]
    assert aside["failure_codes"] == ["trees.reads", "sampled"]
    doc = result(run_dir)
    assert doc["records"][S]["outcome"] == LC.ALLOWED and doc["stale_allowances"] == [] and not keys(root, S) & set(doc["passes"])
    assert any("audits s once more" in s for s in said)
    calls.clear()
    standing[S], left[:] = [READS, sampled], [READS, sampled]
    assert LC.audit_records(S, tmp_path / "deps", log=lambda _s: None, rebuild=True) == 1
    assert calls[:2] == [(S, False), (P, False)], "without the allowance the other failure keeps it from a second audit"


def test_an_allowance_names_a_failure_by_its_code_and_entry_never_its_text(lane, tmp_path):
    """The same code about another entry is not the failure allowed, nor is another code about the same entry; and a report
    that lists no entries (an older check.py's) matches nothing."""
    _, run_dir, _, standing, _ = lane
    standing[S] = [READS]
    for allow in (["trees.reads:Proofs.Other"], ["witnesses.unwitnessed-changed:Proofs.Sampled"],
                  ["trees.reads:" + str(READS)]):
        assert LC.audit_records(S, tmp_path / "deps", log=lambda _s: None, allow=allow) == 1
        doc = result(run_dir)
        assert (doc["outcome"], doc["stale_allowances"]) == (LC.FAILED, allow), allow
        assert doc["records"][S] == {"outcome": LC.FAILED, "inputs_key": doc["records"][S]["inputs_key"]}, allow
    allow = frozenset({("trees.reads", "Proofs.Sampled")})
    rep = AUDIT.reported({"failures": [READS]})
    assert LC.failures_of(rep) == [("trees.reads", "Proofs.Sampled")]
    rep = AUDIT.reported({"failures": [AUDIT.Failure("trees.reads", "Proofs.Sampled", "a reworded message")]})
    assert all(f in allow for f in LC.failures_of(rep)), "the text is not what is matched"
    for old in ({"failures": [str(READS)]}, {"failures": [str(READS)], "failure_codes": ["trees.reads"]},
                {"failures": [str(READS), "x"], "failure_codes": ["trees.reads"], "failure_entries": ["Proofs.Sampled"]}):
        assert not any(f in allow for f in LC.failures_of(old)), old


def test_the_allowable_codes_are_a_closed_list_of_failures_check_makes_about_one_entry():
    assert LC.ALLOWABLE == {"trees.reads", "witnesses.unwitnessed-changed"}
    assert not LC.ALLOWABLE & AUDIT.RECORD_DIFFERENCE, "a record difference is --update's to settle, never allowed"
    source = (HERE.parent / "check.py").read_text()
    for code in LC.ALLOWABLE:
        made = re.findall(rf'Failure\(\s*"{re.escape(code)}",\s*(\w+)', source)
        assert made and "None" not in made, f"check.py makes {code} about one entry, which an allowance names"


def codes() -> list[str]:
    """Every failure code check.py's source makes."""
    return sorted(set(re.findall(r'Failure\(\s*"([a-z0-9.-]+)"', (HERE.parent / "check.py").read_text())))


def test_every_other_code_is_refused_before_anything_runs():
    """No code but the closed list's may be named: every other code check.py makes is refused, the record differences, the
    build, the replay, the axioms, the unchecked sources (`sorry`, `native_decide`, a kernel bypass) and the malformed lock
    among them."""
    made = set(codes())
    never = {"build", "lean", "replay.rejected", "replay.axioms", "replay.killed", "guarantees.axioms", "fresh.rejected",
             "unchecked.source", "policy.invalid", "policy.missing", "trees.form", "tamper", "witnesses.missing",
             "trees.imports", "reads.outside-meaning"} | AUDIT.RECORD_DIFFERENCE
    assert LC.ALLOWABLE <= made and len(made) > 100 and never - AUDIT.RECORD_DIFFERENCE <= made
    for code in sorted((made | never) - LC.ALLOWABLE):
        got, why = LC.allowances([f"{code}:G"])
        assert not got and why and f"--allow {code}:G" in why and "trees.reads, witnesses.unwitnessed-changed" in why, code
        assert ("a record difference, which only --update settles" in why) == (code in AUDIT.RECORD_DIFFERENCE), code
    assert LC.allowances(["trees.reads:M", "witnesses.unwitnessed-changed:G", "trees.reads:M"]) == (
        frozenset({("trees.reads", "M"), ("witnesses.unwitnessed-changed", "G")}), None)


@pytest.mark.parametrize("allow", ["guarantees.reads-changed:G", "build:Security", "replay.axioms:G", "unchecked.source:F",
                                   "policy.invalid:x", "trees.reads", "trees.reads:", ":Proofs.Sampled",
                                   "trees: G reads D in Proofs.Sampled"])
def test_an_allowance_that_may_not_be_given_refuses_the_run_with_the_reason(lane, tmp_path, allow):
    """A code outside the closed list, or an argument that is not a code and an entry (a failure's text, say), refuses the
    run before check.py starts: exit 2, `refused`, with a line naming the argument and the codes that may be allowed."""
    _, run_dir, calls, standing, _ = lane
    standing[S] = [READS]
    said = []
    assert LC.audit_records(S, tmp_path / "deps", update=True, log=said.append,
                            allow=["trees.reads:Proofs.Sampled", allow]) == 2
    assert calls == [] and not passes_kept()
    assert result(run_dir)["outcome"] == LC.REFUSED == outcome_line(said)["outcome"]
    assert any(f"--allow {allow}: " in s and "trees.reads, witnesses.unwitnessed-changed" in s for s in said)


def test_an_allowance_that_matches_no_failure_fails_the_run(lane, tmp_path):
    """A stale allowance is a red of its own: the run's audits pass, and the run fails naming the allowance. A second
    allowance that does match changes nothing of that, and the clean requirer's pass is still kept."""
    root, run_dir, calls, standing, _ = lane
    said = []
    assert LC.audit_records(S, tmp_path / "deps", log=said.append, allow=["trees.reads:Proofs.Sampled"]) == 1
    assert calls == [(S, False), (P, False)]
    doc = result(run_dir)
    assert (doc["outcome"], doc["stale_allowances"], doc["validation"]["status"]) == (
        LC.FAILED, ["trees.reads:Proofs.Sampled"], "failed")
    assert {q: r["outcome"] for q, r in doc["records"].items()} == {S: LC.PASS, P: LC.PASS}
    assert outcome_line(said)["stale"] == ["trees.reads:Proofs.Sampled"]
    assert any("--allow trees.reads:Proofs.Sampled: no failure of this run has that code and entry" in s for s in said)
    assert "matched no failure" in doc["validation"]["detail"]

    for f in LC.passes_dir().glob("*.json"):
        f.unlink()
    standing[S] = [READS]
    said.clear()
    assert LC.audit_records(S, tmp_path / "deps", log=said.append,
                            allow=["trees.reads:Proofs.Sampled", "witnesses.unwitnessed-changed:G"]) == 1
    doc = result(run_dir)
    assert (doc["outcome"], doc["stale_allowances"]) == (LC.FAILED, ["witnesses.unwitnessed-changed:G"])
    assert doc["records"][S]["outcome"] == LC.ALLOWED and set(doc["passes"].values()) == {P}
    assert any("the failures this run found that may be allowed: trees.reads:Proofs.Sampled" in s for s in said)


def test_an_allowance_given_to_a_run_that_builds_nothing_is_stale(lane, tmp_path):
    """Every package's records passed on these inputs, so nothing runs and no failure is found: an allowance given to that
    run matches none and fails it."""
    _, run_dir, calls, _, _ = lane
    assert LC.audit_records(S, tmp_path / "deps", log=lambda _s: None) == 0
    calls.clear()
    said = []
    assert LC.audit_records(S, tmp_path / "deps", log=said.append, allow=["trees.reads:Proofs.Sampled"]) == 1
    assert calls == []
    doc = result(run_dir)
    assert (doc["outcome"], doc["workload_fingerprint"]["path"], doc["stale_allowances"]) == (
        LC.FAILED, "reused", ["trees.reads:Proofs.Sampled"])


def test_a_package_with_an_allowed_failure_in_a_prover_audited_beside_it_keeps_no_pass(lane, tmp_path):
    """check.py audits `s/proofs` beside `s`. Its allowed failure is in the audit of `s`, so `s` ends `allowed` with no pass
    though its own report is clean, and the prover's own record says what was allowed."""
    root, run_dir, _, _, beside = lane
    beside[S] = {PROOFS: [READS]}
    assert LC.audit_records(S, tmp_path / "deps", log=lambda _s: None,
                            allow=["trees.reads:Proofs.Sampled"]) == LC.ALLOWED_RC
    doc = result(run_dir)
    assert {q: r["outcome"] for q, r in doc["records"].items()} == {S: LC.ALLOWED, P: LC.PASS, PROOFS: LC.ALLOWED}
    assert doc["records"][PROOFS]["allowed"] == ["trees.reads:Proofs.Sampled"] and "allowed" not in doc["records"][S]
    assert not keys(root, S) & (set(doc["passes"]) | passes_kept()) and set(doc["passes"].values()) == {P}


def test_a_package_another_audit_reports_an_allowed_failure_of_loses_its_pass_and_its_record_leaves_record(lane, tmp_path):
    """`s` passes its own audit with a regenerated record, and its requirer's audit, which audits it beside, reports an
    allowed failure of it: no pass of `s` is kept, and its record moves out of `record/`."""
    root, run_dir, _, _, beside = lane
    (root / S / LC.LOCK).write_text(OLD[S])
    beside[P] = {S: [CHANGED]}
    said = []
    assert LC.audit_records(S, tmp_path / "deps", update=True, log=said.append,
                            allow=["witnesses.unwitnessed-changed:G"]) == LC.ALLOWED_RC
    doc, out = result(run_dir), run_dir / "lean-audit"
    assert {q: r["outcome"] for q, r in doc["records"].items()} == {S: LC.ALLOWED, P: LC.ALLOWED}
    assert doc["passes"] == {} and not passes_kept()
    assert (out / "allowed-record" / S / LC.LOCK).read_text() == BUILT[S]
    assert not (out / "record" / S / LC.LOCK).exists() and not (out / "record" / LC.LOCK).exists()
    assert doc["records"][S]["record"] == f"lean-audit/allowed-record/{S}/{LC.LOCK}"
    assert any(f"{S}: another package's audit reports a failure of it that --allow names" in s for s in said)


def test_a_killed_check_is_never_allowed(lane, tmp_path, monkeypatch):
    """check.py killed mid-way leaves reports no one can trust: the package fails whatever its report lists."""
    _, run_dir, _, standing, _ = lane
    standing[S] = [READS]
    fake = LC.subprocess.run

    def run(argv, **kw):
        got = fake(argv, **kw)
        return type("P", (), {"returncode": 137})() if argv[1:2] == [str(LC.AUDIT_PY)] else got

    monkeypatch.setattr(LC.subprocess, "run", run)
    assert LC.audit_records(S, tmp_path / "deps", log=lambda _s: None, allow=["trees.reads:Proofs.Sampled"]) == 137
    assert result(run_dir)["records"][S]["outcome"] == LC.FAILED


@pytest.mark.parametrize("argv", [[], ["--build", "s/S.lean"], ["--no-wait", "--build", "s/S.lean"], ["--try", "s/S.lean"],
                                  ["--inputs-key", "s"]],
                         ids=["step", "build", "build-no-wait", "try", "inputs-key"])
def test_allow_goes_with_records_and_with_nothing_else(monkeypatch, capsys, argv):
    """No mode but `--records` takes an allowance, `--build` (the ready mark's preflight), `--try` and `--inputs-key` among
    them: `--allow` beside any of them, or with no mode, is a usage error, before anything runs."""
    for name in ("run", "audit_records", "try_file", "build_files", "keys_at"):
        monkeypatch.setattr(LC, name, lambda *a, **k: pytest.fail(f"{name} ran"), raising=False)
    with pytest.raises(SystemExit) as e:
        LC.main([*argv, "--allow", "trees.reads:Proofs.Sampled"])
    assert e.value.code == 2 and "--allow goes with --records" in capsys.readouterr().err


def test_records_takes_each_allow_from_the_command_line(monkeypatch):
    got = {}
    monkeypatch.setattr(LC, "audit_records", lambda *a, **k: got.update(k) or 0)
    assert LC.main(["--records", "s", "--update", "--allow", "trees.reads:M", "--allow", "witnesses.unwitnessed-changed:G"]) == 0
    assert got["allow"] == ["trees.reads:M", "witnesses.unwitnessed-changed:G"]
    assert LC.main(["--records", "s"]) == 0 and got["allow"] == []


def test_check_py_takes_no_allowance_and_the_fast_path_gives_it_none(tmp_path, capsys):
    """check.py, which the Lean check, the merge queue's Lean tasks and `--records` itself run, has no `--allow`: an
    allowance never changes what it reports or how it exits, and `--records` never hands it one."""
    with pytest.raises(SystemExit) as e:
        AUDIT.main(["--allow", "trees.reads:Proofs.Sampled", str(tmp_path)])
    assert e.value.code == 2 and "unrecognized arguments: --allow" in capsys.readouterr().err
    for update in (False, True):
        for own in (False, True):
            for dev in (False, True):
                argv = LC.check_argv(tmp_path / "p", tmp_path / "out", update, None, own, dev=dev)
                assert not any(a.startswith("--allow") for a in argv), argv
