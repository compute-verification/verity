"""W2c review tests (CONTRACT 16.5): first-stage ordering (Q2), the Bound-compatible sampling
plan (Q3), the JSON-only row cross-check (D2) and the review fixes (D4)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from verity_vllm import commit, firststage, relations, rowcheck, sampling, synthetic, verify
from verity_vllm.capture import CaptureReader, ru_tree_id

TOY_PLUGIN = "verity_vllm.synthetic:install"


def _production_record(run_dir: Path, path: Path) -> Path:
    reader = CaptureReader(run_dir)
    with open(path, "w") as handle:
        for rid in reader.request_ids():
            b = reader.boundary(rid)
            handle.write(json.dumps({
                "request_id": rid,
                "prompt_token_ids": b.prompt_token_ids,
                "output_token_ids": b.output_token_ids,
                "finish_reason": b.finish_reason,
                "sampling_params": b.sampling_params,
            }) + "\n")
    return path


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    root = tmp_path_factory.mktemp("w2c")
    run = synthetic.build_toy_run(root / "good", seed=5, requests=2, prompt_len=4, gen=3)
    production = _production_record(run.run_dir, root / "production.jsonl")
    return {"root": root, "run": run, "production": production}


def _drop_tree(run_dir: Path, dest: Path, tree_id: str) -> Path:
    """A bundle whose commitments.json lacks one replay-unit root: the shape of "the challenge
    was drawn before this step's root was written"."""
    shutil.copytree(run_dir, dest)
    path = dest / "commitments.json"
    doc = json.loads(path.read_text())
    assert tree_id in doc["trees"]
    del doc["trees"][tree_id]
    path.write_text(json.dumps(doc, sort_keys=True))
    return dest


# --- Q2: the whole request is the first-stage unit ---------------------------------------------


def test_good_bundle_is_first_stage_complete(toy):
    reader = toy["run"].reader()
    report = firststage.first_stage_problems(reader, commit.load_commitments(reader))
    assert report.ok, report.problems
    assert report.requests == 2 and report.steps == sum(len(reader.boundary(r).steps) for r in reader.request_ids())


def test_all_step_roots_precede_the_challenge(toy):
    """Every step of every request has a committed root, the challenge binds the digest of that
    complete set, and the effective seed changes if any root changes: no sample can be fixed
    before the last step's root exists."""
    reader = toy["run"].reader()
    commitments = commit.load_commitments(reader)
    for rid in reader.request_ids():
        for step in reader.boundary(rid).steps:
            assert ru_tree_id(rid, step.step) in commitments["trees"]
    seed_full, digest_full = verify.effective_seed(3, commitments)
    partial = json.loads(json.dumps(commitments))
    last = max((t for t in partial["trees"] if t.startswith("ru/")), key=lambda t: int(t.rsplit("/", 1)[1]))
    del partial["trees"][last]
    seed_partial, digest_partial = verify.effective_seed(3, partial)
    assert digest_full != digest_partial and seed_full != seed_partial


def test_challenge_referencing_uncommitted_root_is_refused(toy):
    reader = toy["run"].reader()
    commitments = commit.load_commitments(reader)
    rid = reader.request_ids()[0]
    last_step = reader.boundary(rid).steps[-1].step
    victim = ru_tree_id(rid, last_step)
    incomplete = json.loads(json.dumps(commitments))
    del incomplete["trees"][victim]
    sample = [cid for cid in reader.ru(rid, last_step).ids()][:2]
    assert sample
    # the guard
    with pytest.raises(firststage.FirstStageError, match="not in the committed set"):
        firststage.assert_sample_committed(sample, reader, incomplete)
    assert firststage.assert_sample_committed(sample, reader, commitments) == {"vus": 2, "trees": 1}
    # the pre-check names the step
    report = firststage.first_stage_problems(reader, incomplete)
    assert any(victim in p and "no committed root" in p for p in report.problems), report.problems
    # and the checker never turns such a VU into `exact`
    rep = relations.check(toy["run"].description, reader, incomplete, sample)
    assert {r.status for r in rep.results} == {"auth_failed"}, [(r.status, r.detail) for r in rep.results]


def test_verify_fails_structurally_when_a_step_root_is_missing(toy):
    reader = toy["run"].reader()
    rid = reader.request_ids()[-1]
    victim = ru_tree_id(rid, reader.boundary(rid).steps[-1].step)
    dest = _drop_tree(toy["run"].run_dir, toy["root"] / "missing-step", victim)
    result = verify.verify(
        dest, seed=7, sample_size=6, out_dir=toy["root"] / "out-missing-step", case="TOY",
        plugins=[TOY_PLUGIN], torch_mode="never", negatives=None, audit=False,
    )
    assert result.verdict == "FAIL"
    assert any(p.startswith("first-stage:") and victim in p for p in result.structural_problems), result.structural_problems
    report = json.loads((Path(toy["root"]) / "out-missing-step" / "report.json").read_text())
    assert report["first_stage"]["ok"] is False


def test_committed_step_metadata_must_match_boundary(toy):
    reader = toy["run"].reader()
    commitments = json.loads(json.dumps(commit.load_commitments(reader)))
    rid = reader.request_ids()[0]
    tid = ru_tree_id(rid, reader.boundary(rid).steps[0].step)
    commitments["trees"][tid]["phase"] = "decode" if commitments["trees"][tid]["phase"] == "prefill" else "prefill"
    report = firststage.first_stage_problems(reader, commitments)
    assert any(tid in p and "differ from the boundary" in p for p in report.problems)


# --- Q3: Bound-compatible plan --------------------------------------------------------------------


def test_bernoulli_plan_is_deterministic_independent_and_bound_to_roots(toy):
    reader = toy["run"].reader()
    desc = toy["run"].description
    commitments = commit.load_commitments(reader)
    seed = sampling.bernoulli_seed(7, commitments)
    a = sampling.bernoulli_plan(desc, reader, q=1.0, s=0.5, seed=seed)
    b = sampling.bernoulli_plan(desc, reader, q=1.0, s=0.5, seed=seed)
    assert a.selected == b.selected and a.selected_requests == sorted(reader.request_ids())
    population = sum(a.vus_per_request.values())
    assert population > 0 and a.expected_vus == pytest.approx(0.5 * population)
    # monotone in s: raising the rate only adds units (per-unit thresholds)
    c = sampling.bernoulli_plan(desc, reader, q=1.0, s=0.9, seed=seed)
    assert set(a.selected) <= set(c.selected) and len(c.selected) > len(a.selected)
    # s = 1 selects everything committed; s = 0 nothing
    assert len(sampling.bernoulli_plan(desc, reader, q=1.0, s=1.0, seed=seed).selected) == population
    assert sampling.bernoulli_plan(desc, reader, q=1.0, s=0.0, seed=seed).selected == []
    # q < 1 drops whole requests, never parts of them
    d = sampling.bernoulli_plan(desc, reader, q=0.5, s=1.0, seed=seed)
    for rid in reader.request_ids():
        n = sum(1 for cid in d.selected if f"/r={rid}/" in cid)
        assert n in (0, d.vus_per_request[rid])
    # the seed is a function of the roots
    other = json.loads(json.dumps(commitments))
    first = next(t for t in other["trees"] if t.startswith("ru/"))
    other["trees"][first]["root"] = "00" * 32
    assert sampling.bernoulli_seed(7, other) != seed
    doc = a.to_json()
    assert doc["mode"] == "bernoulli" and doc["q"] == 1.0 and doc["s"] == 0.5
    assert doc["population"]["vus"] == population


def test_verify_records_sampling_plan_and_bound_sample(toy):
    out = toy["root"] / "out-bound"
    result = verify.verify(
        toy["run"].run_dir, seed=7, sample_size=6, out_dir=out, case="TOY",
        production_record=toy["production"], plugins=[TOY_PLUGIN], torch_mode="never",
        negatives=None, bound_s=0.25,
    )
    assert result.verdict == "PARTIAL" and result.table1["first_blocker"].startswith("negative")
    assert result.report.all_exact()
    plan = json.loads((out / "sampling_plan.json").read_text())
    modes = [p["mode"] for p in plan["plans"]]
    assert modes == ["stratified", "bernoulli"]
    assert plan["plans"][0]["bound_compatible"] is False
    assert plan["plans"][1]["s"] == 0.25 and plan["plans"][1]["q"] == 1.0
    assert plan["bound_to"]["commitments_digest"] == result.challenge.commitments_digest
    bound = json.loads((out / "bound_report.json").read_text())
    assert bound["sample_coverage"]["ok"] is True
    statuses = bound["check"]["statuses"]
    assert set(statuses) <= {"exact"} and sum(statuses.values()) == plan["plans"][1]["selected"]["vus"]
    report = json.loads((out / "report.json").read_text())
    assert report["bound_sample"]["challenge_digest"] == bound["challenge_digest"]
    assert report["bound_sample"]["challenge_digest"] != result.challenge.digest().hex()
    # the stratified sample and the tables did not change because the bound plan was drawn
    plain = verify.verify(
        toy["run"].run_dir, seed=7, sample_size=6, out_dir=toy["root"] / "out-plain", case="TOY",
        production_record=toy["production"], plugins=[TOY_PLUGIN], torch_mode="never", negatives=None,
    )
    assert plain.sample == result.sample and plain.challenge == result.challenge
    assert json.loads((toy["root"] / "out-plain" / "sampling_plan.json").read_text())["plans"][0] == plan["plans"][0]
    assert not (toy["root"] / "out-plain" / "bound_report.json").exists()


# --- D2: the JSON-only cross-check on an adapter-shaped row -------------------------------------


def _adapter_row(toy, dest: Path, verify_out: Path) -> Path:
    """Lay a toy verify run out like `fixtures/results/<id>/` (bundle/, production/, challenge.json, ...)."""
    dest.mkdir()
    bundle = dest / "bundle"
    bundle.mkdir()
    for name in ("manifest.json", "description.json", "required.json", "commitments.json"):
        shutil.copyfile(toy["run"].run_dir / name, bundle / name)
    (dest / "production").mkdir()
    shutil.copyfile(toy["production"], dest / "production" / "production.jsonl")
    ch = json.loads((verify_out / "challenge.json").read_text())
    doc = dict(ch["challenge"])
    doc["seed"] = ch["seed"]
    doc["effective_seed"] = ch["effective_seed"]
    doc["adjacent_vus"] = [p["consumer"] for p in ch["adjacent_pairs"]]
    (dest / "challenge.json").write_text(json.dumps(doc))
    report = json.loads((verify_out / "report.json").read_text())
    (dest / "verify.json").write_text(json.dumps({
        "profile_id": report["profile_id"],
        "challenge_digest": report["challenge"]["digest"],
        "sample_size": report["challenge"]["sample_size"],
        "report": report["check"],
        "sample_coverage": report["sample_coverage"],
        "required_coverage": report["required_coverage"],
    }))
    for name in ("table1.jsonl", "table2.jsonl", "table3.jsonl"):
        shutil.copyfile(verify_out / name, dest / name)
    return dest


def test_rowcheck_agrees_with_a_good_row(toy):
    out = toy["root"] / "out-row"
    verify.verify(
        toy["run"].run_dir, seed=11, sample_size=8, out_dir=out, case="TOY",
        production_record=toy["production"], plugins=[TOY_PLUGIN], torch_mode="never", negatives=None,
    )
    row = _adapter_row(toy, toy["root"] / "row-good", out)
    report = rowcheck.check_row(rowcheck.locate(row))
    by_name = {c["name"]: c for c in report["checks"]}
    for name in ("profile_id", "challenge.digest", "boundary_roots_from_production", "ru_namespace",
                 "sample", "report_vs_challenge", "verdict", "steps_cover_positions"):
        assert by_name[name]["status"] == "agree", by_name[name]
    assert by_name["seed_derivation"]["detail"].startswith("mixed")
    # the finding, reproduced: nothing in the row hashes production.jsonl
    assert by_name["production_record_bound"]["status"] == "disagree"


def test_rowcheck_detects_a_production_record_that_differs_from_the_boundary(toy):
    out = toy["root"] / "out-row2"
    verify.verify(
        toy["run"].run_dir, seed=11, sample_size=8, out_dir=out, case="TOY",
        production_record=toy["production"], plugins=[TOY_PLUGIN], torch_mode="never", negatives=None,
    )
    row = _adapter_row(toy, toy["root"] / "row-tampered", out)
    path = row / "production" / "production.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["output_token_ids"][-1] = (records[0]["output_token_ids"][-1] + 1) % 32
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    report = rowcheck.check_row(rowcheck.locate(row))
    by_name = {c["name"]: c for c in report["checks"]}
    assert by_name["boundary_roots_from_production"]["status"] == "disagree"
    assert by_name["boundary_roots_from_production"]["data"]["matched"] == 1


def test_rowcheck_detects_a_challenge_not_bound_to_the_documents(toy):
    out = toy["root"] / "out-row3"
    verify.verify(
        toy["run"].run_dir, seed=11, sample_size=8, out_dir=out, case="TOY",
        production_record=toy["production"], plugins=[TOY_PLUGIN], torch_mode="never", negatives=None,
    )
    row = _adapter_row(toy, toy["root"] / "row-rebound", out)
    desc = json.loads((row / "bundle" / "description.json").read_text())
    desc["nodes"][0]["doc"] = "edited after the challenge"
    (row / "bundle" / "description.json").write_text(json.dumps(desc))
    report = rowcheck.check_row(rowcheck.locate(row), namespace=False)
    by_name = {c["name"]: c for c in report["checks"]}
    assert by_name["challenge.description_digest"]["status"] == "disagree"
    assert by_name["required.description_digest"]["status"] == "disagree"


# --- D4: review fixes --------------------------------------------------------------------------------


def test_unstored_weights_are_missing_not_a_crash(tmp_path):
    """Weights referenced by checkpoint sha (not copied) must degrade a weight-input VU to
    `missing`, never abort the whole check (relations._open_verified used to propagate
    CaptureError)."""
    run = synthetic.build_toy_run(tmp_path / "noweights", seed=1, store_weights=False)
    reader = run.reader()
    commitments = commit.load_commitments(reader)
    rid = reader.request_ids()[0]
    sample = [c for c in reader.ru(rid, 0).ids() if "op=embed_tokens" in c][:1]
    sample += [c for c in reader.ru(rid, 0).ids() if "residual" in c][:1]
    report = relations.check(run.description, reader, commitments, sample)
    statuses = {r.canonical_id: (r.status, r.detail or "") for r in report.results}
    emb = next(v for k, v in statuses.items() if "embed" in k)
    res = next(v for k, v in statuses.items() if "residual" in k)
    assert emb[0] == "missing" and "not stored" in emb[1]
    assert res[0] == "exact"
