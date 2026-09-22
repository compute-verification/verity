"""[R8 C-codec] Interchange exactness with C-builder's `verity-aot/hints/v1`: for every corpus hint in out/gen/r8/aot-codec/hints
(26 runs, extracted on the pod), from_builder_hints(phi, to_builder_hints(phi, h)) == h bit-exact, plus the synthetic escape paths.
Skipped when the builder tree is not importable (their worktree is read-only for this lane; see builder_adapter.load_builder)."""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from .. import builder_adapter as ba
from .. import facts
from .test_codecs import synth_hint, synth_phi

ART = Path(__file__).resolve().parents[4] / "out" / "gen" / "r8" / "aot-codec"


def _builder_or_skip():
    try:
        return ba.load_builder()
    except ba.BuilderUnavailable as e:
        pytest.skip(str(e))


def corpus_points() -> list[tuple[str, dict, dict]]:
    inv = ART / "inventory.jsonl"
    if not inv.exists():
        return []
    out = []
    for line in inv.read_text().splitlines():
        x = json.loads(line)
        hp = ART / "hints" / f"{x['run_id']}.json"
        if not hp.exists():
            continue
        h = json.loads(hp.read_text())["hint"]
        phi0 = {"schema": "aot-codec/phi/v1", "run_id": x["run_id"], "model": x["model"], "engine": x["engine"],
                "requests": [{"request_id": f"r{i}", "prompt_len": q["prompt_len"], "max_tokens": q["max_tokens"]} for i, q in enumerate(h["requests"])]}
        out.append((x["run_id"], phi0, h))
    return out


@pytest.mark.parametrize("run_id,phi0,h", corpus_points(), ids=[p[0] for p in corpus_points()])
def test_corpus_hint_roundtrips_through_v1(run_id, phi0, h):
    _builder_or_skip()
    hj = ba.to_builder_hints(phi0, h)
    back = ba.from_builder_hints(facts.phi_variant(phi0, h, "phi1a"), hj)
    assert facts.canonical_json(back) == facts.canonical_json(facts.strip_private(h)), run_id
    # what their predictor needs to be told for this run (result, recorded in REPORT.md)
    n_ov = len(hj["steps"])
    print(f"{run_id}: v1 served={len(hj['served'])} step-overrides={n_ov} naive_bits={ba.naive_bits(phi0, hj)}")


def test_synthetic_paths_roundtrip_through_v1():
    _builder_or_skip()
    rng = random.Random(23)
    for k in range(8):
        phi0, reqs = synth_phi(rng, rng.randint(1, 6), rng.choice([64, 2112]), 200, 12, 0)  # simultaneous arrivals: no idle gaps
        g = [max(1, q["max_tokens"] - rng.randint(0, q["max_tokens"] - 1)) for q in reqs]
        # an EOS stop at max_tokens-1 is structurally a full-length run (placeholder row): v1 carries rows, so it comes back as (max_tokens,
        # length) -- the schedule is identical; keep the token-level distinction out of this synthetic set
        g = [q["max_tokens"] if gi == q["max_tokens"] - 1 else gi for gi, q in zip(g, reqs)]
        h = synth_hint(phi0, reqs, g)
        hj = ba.to_builder_hints(phi0, h)
        back = ba.from_builder_hints(facts.phi_variant(phi0, h, "phi1a"), hj)
        assert facts.canonical_json(back) == facts.canonical_json(facts.strip_private(h))
        assert len(hj["served"]) == sum(1 for q in h["requests"] if q["n_generated"] != q["max_tokens"])
        # served = rows sampled = n_generated + 1 for every EOS stop (async placeholder)
        assert all(s["served"] == h["requests"][s["request"]]["n_generated"] + 1 for s in hj["served"])


def test_abort_not_expressible_in_v1():
    _builder_or_skip()
    rng = random.Random(29)
    phi0, reqs = synth_phi(rng, 2, 2112, 50, 6, 0)
    h = synth_hint(phi0, reqs, [3, reqs[1]["max_tokens"]])
    h["requests"][0]["finish_reason"] = "abort"
    with pytest.raises(ba.NotExpressibleInV1):
        ba.to_builder_hints(phi0, h)


@pytest.mark.parametrize("rid,overrides,served,step_overrides", [("eos4", {}, 2, 0), ("preempt4", {"num_gpu_blocks": 20}, 0, 199), ("thr4", {"long_prefill_token_threshold": 20}, 0, 1)])
def test_captured_records_roundtrip_through_v1(rid, overrides, served, step_overrides):
    """The three records captured by this lane (out/gen/r8/aot-codec/gen/*) through C-builder's v1 schema + predictor v2: EOS stops are two
    `served` entries (n_generated + 1 rows each) and no step override; the preemption run needs 199 step overrides (their predictor has no
    KV-pool model; v1's step vocabulary can carry the re-prefill rows); the chunk-cap run needs 1.  Recorded in REPORT.md §5."""
    _builder_or_skip()
    from pathlib import Path

    log = Path(__file__).resolve().parents[4] / f"out/gen/r8/aot-codec/gen/{rid}/log.jsonl.gz"
    if not log.exists():
        pytest.skip(f"{rid} record not present")
    r = facts.load_log_only(log, rid)
    r["phi_overrides"] = overrides
    h, _ = facts.extract_hint(r)
    phi0 = facts.phi_from_record(r)
    hj = ba.to_builder_hints(phi0, h)
    back = ba.from_builder_hints(facts.phi_variant(phi0, h, "phi1a"), hj)
    assert facts.canonical_json(back) == facts.canonical_json(facts.strip_private(h))
    assert (len(hj["served"]), len(hj["steps"])) == (served, step_overrides)
