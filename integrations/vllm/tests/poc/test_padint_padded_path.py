"""[PAD-INT] The padded path's bookkeeping, independent of any GPU record (run on the pod: `pytest tests/poc/test_padint_padded_path.py`).

* the realized-liveness flag chain from committed tokens (EOS-inclusive transcripts, horizon requests, an EOS-only request);
* the registered `PadGuard_v1{EOS}` relation at the three instance kinds (live / stop-evidence / inactive), and that the
  transparent checker REJECTS a live relation claimed on an inactive slot and an altered padded output (negatives N1a/N3a);
* the D_MAX convention (lead's decision on serve3-spec §15 Q1): `Serve_v3{…, D_MAX = STEPS}` has `STEPS + 1` designated token
  slots at positions `t = LP-1+k`, the SAME positions the certified `Serve_v1{…, STEPS}` manifest lists, with one guard
  per sampling event and one k/v mux per (step, layer) — the L3 address space `padint_pad_statements.py` uses (TINY dims);
* `padint_linkage_check` on a synthetic record: passes when the attached census names both digests and carries inactive
  units, rejects a relabelled-live slot, a missing inactive proof, and a served transcript extended past EOS.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "zk" / "bench", ROOT / "src", ROOT, ROOT / "vllm-poc", ROOT / "scripts"):
    sys.path.insert(0, str(p))

from padint_pad_statements import flag_chain, named_calls, request_liveness  # noqa: E402

EOS = 0


# ---------------------------------------------------------------------------
# flag chain
# ---------------------------------------------------------------------------


def test_flag_chain_eos_terminated_is_live_prefix_then_absorbed():
    # 5 produced tokens, EOS-inclusive: events 0..4 live, event 4 is the stop evidence, 5..15 inactive
    live_in, live_out = flag_chain([7, 8, 9, 10, EOS], 16)
    assert live_in == [1] * 5 + [0] * 11
    assert live_out == [1] * 4 + [0] * 12
    assert live_in[1:] == live_out[:-1]  # the chain is a value carried step to step


def test_flag_chain_horizon_request_is_all_live():
    live_in, live_out = flag_chain(list(range(1, 17)), 16)
    assert live_in == [1] * 16 and live_out == [1] * 16


def test_flag_chain_eos_only_request_leaves_every_decode_step_inactive():
    live_in, live_out = flag_chain([EOS], 16)
    assert live_in == [1] + [0] * 15 and live_out == [0] * 16


def test_request_liveness_reports_present_slots_and_stop_event():
    b = {"request_id": "r1", "prompt_token_ids": list(range(32)),
         "output_records": [{"token_id": t, "present": True} for t in (5, 6, EOS)] + [{"token_id": 0, "present": False}] * 13,
         "finish_reason": "stop", "output_token_ids": [5, 6, EOS]}
    lv = request_liveness(b, 16)
    assert lv["n_present"] == 3 and lv["stop_event"] == 2
    assert lv["live_events"] == [0, 1, 2] and lv["inactive_events"] == list(range(3, 16))
    assert lv["inactive_steps"] == list(range(3, 16)) and lv["live_steps"] == [1, 2]
    assert not lv["anomalies"]


# ---------------------------------------------------------------------------
# the guard relation and its negatives (transparent checker; no host)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def guard():
    from pad_relations import PadTranslated
    from verity.ir.defs import bind
    from verity_vllm.registry import serve3

    return PadTranslated(bind(serve3.PadGuard, EOS=EOS), f"PadGuard<EOS={EOS}>", [1, 32])


def test_guard_relation_live_stop_inactive(guard):
    assert guard.oracle_outputs([1, 42]) == [42, 1]      # live: token passes, chain stays live
    assert guard.oracle_outputs([1, EOS]) == [EOS, 0]    # stop evidence: raw sample = EOS, chain absorbs
    assert guard.oracle_outputs([0, 42]) == [0, 0]       # inactive: padded slot 0, independent of y
    assert guard.oracle_outputs([0, 0]) == [0, 0]


def _checked(guard, live_in, y, tok_out, live_out) -> bool:
    from pad_relations import PadBatcher, make_statement
    from veritor.protocol.proofs.statement import Witness
    from veritor.protocol.proofs.transparent import TransparentBackend

    b = PadBatcher(guard, "test", [32, 1])
    b.add("u", [live_in, y], [tok_out, live_out])
    statement = make_statement(guard.gate_set.id, bytes.fromhex(str(guard.gate_set.digest)), 32, [guard.program], [o for o, _ in b.prepared])
    witness = Witness(tuple(w for _, w in b.prepared))
    checker = TransparentBackend(guard.gate_set, witness_format=2)
    try:
        return bool(checker.verify(statement, checker.prove(statement, witness)))
    except Exception:  # noqa: BLE001
        return False


def test_transparent_checker_accepts_honest_instances(guard):
    assert _checked(guard, 1, 42, 42, 1)
    assert _checked(guard, 1, EOS, EOS, 0)
    assert _checked(guard, 0, 0, 0, 0)


def test_negative_live_relation_claimed_on_inactive_slot_is_rejected(guard):
    assert not _checked(guard, 0, 0, 0, 1)      # N1a: live_out = 1 with live_in = 0
    assert not _checked(guard, 0, 42, 42, 1)    # the live pattern (y, 1) on an inactive slot


def test_negative_altered_padded_output_is_rejected(guard):
    assert not _checked(guard, 0, 0, 17, 0)     # N3a: padded slot 0 -> 17
    assert not _checked(guard, 1, EOS, EOS, 1)  # the chain kept live past the EOS


# ---------------------------------------------------------------------------
# D_MAX convention + address space (TINY dims)
# ---------------------------------------------------------------------------


def test_dmax_equals_steps_gives_the_certified_slot_positions_and_one_guard_per_event():
    from verity.ir import Program
    from verity_vllm.registry import serve3

    TINY = {"model": "tiny", "revision": None, "H": 32, "NL": 2, "NH": 2, "KVH": 1, "D": 16, "I": 32, "V": 64,
            "QKV": 64, "GU": 64, "EPS": 1e-5, "ROPE_THETA": 1e4, "MAX_POS": 16, "BIAS": True,
            "tie_word_embeddings": True, "profile": {}, "EOS_ID": EOS}  # tests/ir/test_serve3.py's TINY + EOS_ID
    LP, STEPS = 4, 3
    D_MAX = STEPS  # lead's decision: D_MAX := STEPS = max_tokens - 1
    prog = Program(serve3.serve_v3(TINY, LP, D_MAX))
    named = named_calls(prog)
    guards = sorted(k for k in named if k.endswith("/guard"))
    assert len(guards) == D_MAX + 1 == STEPS + 1  # ell_out = STEPS + 1 = max_tokens designated slots
    # slot k <-> position t = LP-1+k: k=0 is the prefill-produced token, k=1..STEPS the decode steps
    assert f"t={LP - 1}/guard" in named
    for s in range(1, D_MAX + 1):
        assert f"s={s}/step/t={LP - 1 + s}/guard" in named
        for L in range(TINY["NL"]):
            assert f"s={s}/step/t={LP - 1 + s}/kmux.{L}" in named and f"s={s}/step/t={LP - 1 + s}/vmux.{L}" in named
    positions = [LP - 1 + k for k in range(D_MAX + 1)]
    assert positions == list(range(LP - 1, LP + STEPS))  # == the Serve_v1{LP, STEPS} manifest's designated positions
    # designation = token slots only: the return type has D_MAX+1 i32 token slots + the internal live vector
    rets = prog.fn.returns if hasattr(prog.fn, "returns") else None
    if rets is not None:
        assert len([r for r in rets if getattr(r, "width", 32) == 32]) >= D_MAX + 1


def test_structural_addresses_of_a_decoded_descriptor_match_the_hand_built_calls(tmp_path):
    """The derived program arrives as a descriptor (no node names); `structural_named_calls` resolves the hand-built
    names' node-index paths in the DECODED circuit and cross-checks callee id + gate interval call-by-call."""
    import gzip

    from padint_pad_statements import padded_program, structural_named_calls
    from verity.ir import Program, encode_program, program_digest
    from verity_vllm.registry import serve3

    TINY = {"model": "tiny", "revision": None, "H": 32, "NL": 2, "NH": 2, "KVH": 1, "D": 16, "I": 32, "V": 64,
            "QKV": 64, "GU": 64, "EPS": 1e-5, "ROPE_THETA": 1e4, "MAX_POS": 16, "BIAS": True,
            "tie_word_embeddings": True, "profile": {}, "EOS_ID": EOS}
    LP, D_MAX = 4, 3
    ref = Program(serve3.serve_v3(TINY, LP, D_MAX))
    desc = encode_program(ref)
    path = tmp_path / "descriptor.json.gz"
    with gzip.open(path, "wb") as f:
        f.write(json.dumps(desc).encode())
    # `padded_program("derived:...")` is B0-shaped in its label only; the decode itself is dimension-agnostic
    prog, digest, label, fn_id = padded_program(f"derived:{path}", LP, D_MAX)
    assert digest == program_digest(desc) == program_digest(encode_program(ref))
    assert fn_id == ref.fn.id and fn_id.startswith("Serve_v3{")  # the Root[...] wrapper is stripped
    assert label.startswith("DERIVED padded program")
    assert not any(getattr(n, "name", None) for n in prog.circuit.root_scope.spec.body.nodes)  # the codec carries no names
    named, report = structural_named_calls(prog, ref)
    assert report["n_mismatches"] == 0 and report["cross_checked_calls"] == len(named) > 0
    hand = named_calls(ref)
    assert set(named) == set(hand)
    for key in (f"t={LP - 1}/guard", f"s={D_MAX}/step/t={LP - 1 + D_MAX}/guard", f"s=1/step/t={LP}/kmux.0"):
        assert (named[key].id, named[key].lo, named[key].hi) == (hand[key].id, hand[key].lo, hand[key].hi)
    # a structurally different program (one fewer step) is caught by the cross-check, not silently re-addressed
    other = Program(serve3.serve_v3(TINY, LP, D_MAX - 1))
    with pytest.raises((IndexError, AssertionError, KeyError)):
        _named, rep = structural_named_calls(other, ref)
        assert rep["n_mismatches"] == 0


# ---------------------------------------------------------------------------
# linkage check on a synthetic record
# ---------------------------------------------------------------------------


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _synthetic_record(tmp: Path, *, n_present: int = 5, slots: int = 16, inactive_units: int | None = None):
    pad_digest, cert_digest = "ab" * 32, "cd" * 32
    rec = tmp / "REC"
    (rec / "proofs" / "proof-cov" / "pad-guard-b16").mkdir(parents=True)
    (rec / "manifest.json").write_text(json.dumps({"inputs": {"program_digest": cert_digest}}))
    toks = list(range(10, 10 + n_present - 1)) + [EOS]
    (rec / "served_tokens.json").write_text(json.dumps({"requests": [{"request_id": "r1", "served_token_ids": toks, "finish_reason": "stop"}]}))
    proof = b"proof-bytes"
    (rec / "proofs" / "proof-cov" / "pad-guard-b16" / "proof-compressed.bin").write_bytes(proof)
    n_inact = (slots - n_present) if inactive_units is None else inactive_units
    (rec / "proofs" / "proof-cov" / "pad-guard-b16" / "census.json").write_text(json.dumps(
        {"program_digest": pad_digest, "certified_program_digest": cert_digest, "inactive_units": n_inact, "live_units": n_present - 1}))
    cs = {"batches": [{"name": "pad-guard-b16", "relation": "pad-guard/eos0-padint", "attached": True, "attached_dir": "proofs/proof-cov/pad-guard-b16",
                       "inactive_units": n_inact,
                       "modes": {"compressed": {"attached": {"file": "proofs/proof-cov/pad-guard-b16/proof-compressed.bin", "sha256": _sha(proof)},
                                                "verified_cpu": {"ok": True}}}}]}
    (rec / "proofs" / "proof-cov" / "cov_summary.json").write_text(json.dumps(cs))
    live_in, live_out = flag_chain(toks, slots)
    census = {"guard": [{"request_id": "r1", "event": g, "t": 31 + g, "live_in": live_in[g], "y": (toks[g] if g < n_present else 0),
                         "tok_out": (toks[g] if g < n_present else 0), "live_out": live_out[g],
                         "kind": "inactive" if not live_in[g] else ("stop-evidence" if toks[g] == EOS else "live")} for g in range(slots)],
              "rowmux": []}
    statements = {"program": {"padded": {"digest": pad_digest, "D_MAX": slots - 1, "EOS_ID": EOS, "LP": 32},
                              "certified": {"digest": cert_digest}}, "census": census}
    return rec, statements


def test_linkage_check_passes_on_a_consistent_record(tmp_path):
    from padint_negatives import padint_linkage_check

    rec, st = _synthetic_record(tmp_path)
    assert padint_linkage_check(rec, st) == []


def test_linkage_check_rejects_relabelled_live_slot(tmp_path):
    from padint_negatives import padint_linkage_check

    rec, st = _synthetic_record(tmp_path)
    forged = json.loads(json.dumps(st["census"]))
    forged["guard"][7].update({"live_in": 1, "kind": "live"})
    problems = padint_linkage_check(rec, st, census_override=forged)
    assert any("claimed liveness" in p and "[7]" in p for p in problems)


def test_linkage_check_rejects_missing_inactive_proof_and_all_live_evidence(tmp_path):
    from padint_negatives import padint_linkage_check

    rec, st = _synthetic_record(tmp_path)
    (rec / "proofs" / "proof-cov" / "pad-guard-b16" / "proof-compressed.bin").unlink()
    problems = padint_linkage_check(rec, st)
    assert any("missing" in p for p in problems) and any("inactive units" in p for p in problems)
    rec2, st2 = _synthetic_record(tmp_path / "b", inactive_units=0)
    assert any("inactive units" in p for p in padint_linkage_check(rec2, st2))


def test_linkage_check_rejects_served_transcript_extended_past_eos(tmp_path):
    from padint_negatives import padint_linkage_check

    rec, st = _synthetic_record(tmp_path)
    doc = json.loads((rec / "served_tokens.json").read_text())
    doc["requests"][0]["served_token_ids"].append(99)
    (rec / "served_tokens.json").write_text(json.dumps(doc))
    problems = padint_linkage_check(rec, st)
    assert any("served transcript" in p for p in problems)  # the chain absorbs at EOS: the extra token sits at an inactive slot


def test_linkage_check_rejects_a_foreign_program_digest(tmp_path):
    from padint_negatives import padint_linkage_check

    rec, st = _synthetic_record(tmp_path)
    st["program"]["padded"]["digest"] = "ef" * 32
    assert any("program_digest" in p for p in padint_linkage_check(rec, st))
