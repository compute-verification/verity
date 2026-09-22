"""verity_vllm.challenge: the staged, verifier-owned challenge (GUIDANCE C1/C1a/C1b) and its
third-party re-derivation from a bundle (`python -m verity_vllm.bundle rederive`)."""

from __future__ import annotations

import json
from fractions import Fraction

import pytest

from verity_vllm import bundle, challenge as C, commit as vcommit, hashing, relations, synthetic
from verity_vllm.capture import CaptureReader
from verity_vllm.description import Description

# --- the sources ----------------------------------------------------------------------------------


def test_security_source_commits_then_reveals_and_refuses_a_seed():
    src = C.make_source("security")
    op = src.opening()
    assert op["mode"] == "security" and "seed" not in op and op["seed_commitment"]
    with pytest.raises(C.ChallengeError):
        C.make_source("security", seed=3)
    r = src.randomness(b"\x01" * 32, C.PURPOSE_Q)
    assert len(r) == 32
    rev = src.reveal()
    assert C.seed_commitment(bytes.fromhex(rev["seed"])) == op["seed_commitment"]
    assert rev["draws"] == [{"phase_digest": "01" * 32, "purpose": C.PURPOSE_Q.decode(), "r": r.hex()}]


def test_one_draw_per_purpose_per_session():
    src = C.make_source("diagnostic", seed=1)
    src.randomness(b"\x01" * 32, C.PURPOSE_Q)
    with pytest.raises(C.ChallengeError, match="already drawn"):
        src.randomness(b"\x02" * 32, C.PURPOSE_Q)
    src.randomness(b"\x02" * 32, C.PURPOSE_S)  # a different purpose is fine


def test_diagnostic_source_is_reproducible_and_labelled():
    a, b = C.make_source("diagnostic", seed=7), C.make_source("diagnostic", seed=7)
    assert a.randomness(b"\x00" * 32, C.PURPOSE_Q) == b.randomness(b"\x00" * 32, C.PURPOSE_Q)
    assert a.opening()["mode"] == "diagnostic" and "NOT a security" in a.opening()["warning"]
    with pytest.raises(C.ChallengeError):
        C.make_source("diagnostic")
    assert C.verdict_label("PASS", "diagnostic") == "PASS (diagnostic challenge)"
    assert C.verdict_label("PASS", "security") == "PASS"
    assert C.verdict_label("FAIL", "diagnostic") == "FAIL"


# --- the staged session over a synthetic capture ------------------------------------------------


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    root = tmp_path_factory.mktemp("challenge")
    run = synthetic.build_toy_run(root / "run", seed=3)
    reader = CaptureReader(run.run_dir)
    return {"root": root, "run": run, "reader": reader, "description": Description(reader.description),
            "commitments": vcommit.load_commitments(reader)}


def _staged(toy, source, q="1", s="1/2"):
    reader, commitments, description = toy["reader"], toy["commitments"], toy["description"]
    policy = C.Policy.parse(q, s)
    statement = {"profile_id": reader.profile_id, "description_digest": description.digest().hex(),
                 "policy": policy.to_json(), "source": source.opening()}
    sess = C.StagedSession(source, policy)
    sess.accept_statement(statement)
    selected = sess.challenge_replay(reader.request_ids())
    interiors, units = C.interiors_of(commitments, reader, selected)
    sess.accept_interiors(interiors)
    resolve = C.make_resolver(reader)
    coverage = C.make_coverage(description, reader, [], 4, selected)
    certified, cov = sess.challenge_sample(units, resolve, coverage=coverage)
    transcript = sess.finish()
    ctx = dict(statement=statement, interiors=interiors, request_ids=reader.request_ids(), units=units,
               resolve_ids=resolve, coverage=coverage)
    return transcript, certified, cov, ctx


def test_staging_is_visible_and_rederives(toy):
    transcript, certified, cov, ctx = _staged(toy, C.make_source("security"))
    phases = [p["phase"] for p in transcript["phases"]]
    assert phases == ["statement", "replay", "interior", "sample"]
    # r_q keyed by T_0 only; r_s by T_2 (after the selected interiors)
    seed = bytes.fromhex(transcript["reveal"]["seed"])
    t0 = bytes.fromhex(transcript["phases"][0]["digest"])
    t2 = bytes.fromhex(transcript["phases"][2]["digest"])
    assert transcript["replay_challenge"]["r_q"] == C._HmacSource.derive(seed, t0, C.PURPOSE_Q).hex()
    assert transcript["sample_challenge"]["r_s"] == C._HmacSource.derive(seed, t2, C.PURPOSE_S).hex()
    assert set(certified) and all(toy["reader"].locate(cid) is not None for cid in certified)
    assert transcript["sample_challenge"]["coverage_seed"] == C.coverage_seed(bytes.fromhex(transcript["sample_challenge"]["r_s"]))
    r = C.reverify(transcript, **ctx)
    assert r["ok"], r["problems"]
    # every certified id is a leaf of a selected replay unit and each is drawn with probability s
    n = ctx["units"].count
    assert 0 < len(certified) < n


@pytest.mark.parametrize("mutation", [
    lambda t: t["sample_challenge"]["certified_sample"].pop(0),
    lambda t: t["sample_challenge"]["selected_ranks"].__setitem__(0, t["sample_challenge"]["selected_ranks"][0] + 1),
    lambda t: t["sample_challenge"].__setitem__("r_s", "00" * 32),
    lambda t: t["replay_challenge"].__setitem__("r_q", "11" * 32),
    lambda t: t["reveal"].__setitem__("seed", "22" * 32),
    lambda t: t["sample_challenge"]["coverage_sample"].pop(-1),
    lambda t: t["phases"][0].__setitem__("digest", "33" * 32),
    lambda t: t["reveal"]["draws"].append({**t["reveal"]["draws"][-1], "r": "55" * 32}),
    lambda t: t["reveal"]["draws"][-1].__setitem__("phase_digest", "66" * 32),
])
def test_reverify_rejects_every_substitution(toy, mutation):
    transcript, _, _, ctx = _staged(toy, C.make_source("diagnostic", seed=11))
    t = json.loads(json.dumps(transcript))
    mutation(t)
    assert not C.reverify(t, **ctx)["ok"]


def test_reverify_rejects_other_documents(toy):
    transcript, _, _, ctx = _staged(toy, C.make_source("diagnostic", seed=11))
    assert not C.reverify(transcript, **{**ctx, "statement": {"profile_id": "other"}})["ok"]
    assert not C.reverify(transcript, **{**ctx, "interiors": {**ctx["interiors"], "ru/x/0": {"root": "00"}}})["ok"]


def test_core_sampler_matches_the_core_module():
    """The adapter's draw is the core protocol's draw (in-process or via the 3.12 subprocess)."""
    seed, digest = b"\x07" * 32, b"\x09" * 32
    picked = C.core_bernoulli_subset(seed, C.CORE_STAGE_S, digest, 1000, Fraction(1, 10))
    assert picked == tuple(sorted(set(picked))) and all(0 <= i < 1000 for i in picked)
    assert 60 <= len(picked) <= 140  # Binomial(1000, 0.1): far outside this is a broken sampler
    try:
        from veritor.core import VerificationLimits
        from veritor.protocol import challenge as core
    except Exception:  # noqa: BLE001  (Python < 3.12 without the core on the path)
        return
    assert picked == tuple(core.bernoulli_subset(seed, C.CORE_STAGE_S, digest, 1000, Fraction(1, 10), VerificationLimits()))


# --- the bundle: pack a staged verification, re-derive it CPU-only ------------------------------


def test_bundle_rederive(toy):
    reader, commitments, description = toy["reader"], toy["commitments"], toy["description"]
    root = toy["root"]
    transcript, certified, cov, ctx = _staged(toy, C.make_source("security"))
    sample = list(dict.fromkeys(cov + certified))
    ch = relations.make_challenge(commitments, description, sample, reader.required if reader.has("required.json") else None)
    vdir = root / "verify"
    vdir.mkdir()
    digests = {p["phase"]: p["digest"] for p in transcript["phases"]}
    (vdir / "challenge.json").write_text(json.dumps({
        **ch.to_json(),
        "challenge": {"mode": "security", "source": "InteractiveVerifierSource", "policy": transcript["policy"], "phase_digests": digests,
                      "transcript": "challenge_transcript.json"},
        "certified_sample": certified, "coverage_sample": cov, "coverage_size": 4, "adjacent_vus": [],
    }))
    (vdir / "challenge_transcript.json").write_text(json.dumps(transcript))
    (vdir / "table1.jsonl").write_text(json.dumps({"result": "PASS", "challenge": {"mode": "security"}}) + "\n")
    prod = root / "production.jsonl"
    with open(prod, "w") as h:
        for rid in reader.request_ids():
            b = reader.boundary(rid)
            h.write(json.dumps({"request_id": rid, "prompt_token_ids": b.prompt_token_ids, "output_token_ids": b.output_token_ids}) + "\n")
    # the statement must bind the production record the bundle ships, so put its sha where verify.py does
    transcript["statement"]["bindings"] = {"output": {"production_record_sha256": hashing.sha256_hex(prod.read_bytes())}}
    (vdir / "challenge_transcript.json").write_text(json.dumps(transcript))
    tar = bundle.pack(reader.run_dir, root / "TOY-staged.tar.gz", production_record=prod, verify_dir=vdir)
    doc = bundle.read_manifest(tar)
    rec = doc["recorded_verification"]
    assert rec["challenge_mode"] == "security" and rec["seed"] is None and rec["transcript"] == "verify/challenge_transcript.json"
    assert "rederive" in doc["reverify"]["command"] and "--transcript" in doc["reverify"]["equivalent"]
    # T_0 covers the statement as recorded; the recorded statement here was edited after the session, so the
    # re-derivation must say so -- then redo it properly
    out = bundle.rederive_transcript(tar, out_dir=root / "rederive-edited")
    assert not out["ok"] and any("T_0" in p for p in out["transcript"]["problems"])
    ctx["statement"]["bindings"] = transcript["statement"]["bindings"]
    transcript2, certified2, cov2, _ = _staged_with_statement(toy, ctx["statement"])
    sample2 = list(dict.fromkeys(cov2 + certified2))
    ch2 = relations.make_challenge(commitments, description, sample2, reader.required if reader.has("required.json") else None)
    (vdir / "challenge.json").write_text(json.dumps({
        **ch2.to_json(),
        "challenge": {"mode": "security", "source": "InteractiveVerifierSource", "policy": transcript2["policy"],
                      "phase_digests": {p["phase"]: p["digest"] for p in transcript2["phases"]}, "transcript": "challenge_transcript.json"},
        "certified_sample": certified2, "coverage_sample": cov2, "coverage_size": 4, "adjacent_vus": [],
    }))
    (vdir / "challenge_transcript.json").write_text(json.dumps(transcript2))
    tar2 = bundle.pack(reader.run_dir, root / "TOY-staged-2.tar.gz", production_record=prod, verify_dir=vdir)
    out2 = bundle.rederive_transcript(tar2, out_dir=root / "rederive")
    assert out2["ok"], out2
    assert out2["statement"]["rederived_from_bundle"] and out2["challenge_digest"]["agrees"]
    assert out2["population"]["certified"] == len(certified2) and out2["verdict_label"] == "PASS"
    assert (root / "rederive" / "rederive.json").exists()


def _staged_with_statement(toy, statement):
    reader, commitments, description = toy["reader"], toy["commitments"], toy["description"]
    source = C.make_source("security")
    policy = C.Policy.parse("1", "1/2")
    trees = commitments.get("trees", {})
    from verity_vllm.capture import WEIGHTS_TREE_ID

    full = {
        "profile_id": reader.profile_id,
        "manifest_digest": hashing.sha256_hex((reader.run_dir / "manifest.json").read_bytes()),
        "description_digest": description.digest().hex(),
        "required_digest": __import__("verity_vllm.required", fromlist=["required_digest"]).required_digest(reader.required).hex() if reader.has("required.json") else None,
        "boundary_roots": {tid: {"root": rec["root"], "leaf_count": rec["leaf_count"], "ids_digest": rec.get("ids_digest")}
                           for tid, rec in sorted(trees.items()) if rec.get("kind") == "boundary"},
        "weights_root": {k: trees[WEIGHTS_TREE_ID].get(k) for k in ("root", "leaf_count", "ids_digest", "weights_sha256")} if WEIGHTS_TREE_ID in trees else None,
        "replay_units_declared": sorted(tid for tid, rec in trees.items() if rec.get("kind") == "ru"),
        "bindings": statement["bindings"],
        "policy": policy.to_json(),
        "source": source.opening(),
    }
    sess = C.StagedSession(source, policy)
    sess.accept_statement(full)
    selected = sess.challenge_replay(reader.request_ids())
    interiors, units = C.interiors_of(commitments, reader, selected)
    sess.accept_interiors(interiors)
    certified, cov = sess.challenge_sample(units, C.make_resolver(reader), coverage=C.make_coverage(description, reader, [], 4, selected))
    return sess.finish(), certified, cov, None
