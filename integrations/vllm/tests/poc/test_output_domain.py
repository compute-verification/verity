"""GUIDANCE §32.5 (lane OUT16): the unconditional 16-bit output-domain check.

The pinned B0 token-output convention (§31.2, approved §32.5): native token IDs in a 32-bit
storage container with a DECLARED mathematical domain of b = 16 bits, made real by one
unconditional comparison per opened output leaf — the container bits above the declared domain
(the high 16-bit word) equal zero — inside the every-audit output-linkage check, independent of
both sampling draws.  The adversarial cases demanded by §32.5 / walk-16bit-smallest-change.md §3:

* a committed output leaf with a high container bit set is REJECTED even when the lie is
  committed consistently (boundary record leaf == interior sampler leaf, so the pre-existing
  equality linkage passes and no sampled relation runs);
* an unused-but-in-domain code (B0: 49,152..65,535; here: out-of-vocab but < 2^16) is ACCEPTED
  by the type check — exact vocabulary membership is NOT required for the W_out bound
  (§31.2/§32.5); vocabulary faults stay with the sampled argmax/embedding relations;
* on a description without a declaration nothing is enforced: pre-declaration records verify
  exactly as before (the check is declaration-gated, additive).
"""

from __future__ import annotations

import numpy as np
import pytest

from verity_vllm import commit, relations, synthetic
from verity_vllm.capture import Boundary, OutputRecord
from verity_vllm.description import Description, DescriptionError
from verity_vllm.negatives import PatchedReader


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    return synthetic.build_toy_run(tmp_path_factory.mktemp("toy-domain") / "run", seed=3)


def _consistent_claim(toy, rid: str, value: int):
    """The strongest adversary of walkthrough-b0.md §5: the claimed token ``value`` committed
    consistently in the boundary ``output_token`` leaf AND the interior ``sampler.select`` leaf
    (trees rebuilt, so every opening authenticates).  Returns (patched reader, commitments, t)."""
    reader = toy.reader()
    boundary = reader.boundary(rid)
    present = [r for r in boundary.output_records if r.present]
    record = present[1]  # a middle slot: no stop-token / presence-prefix interaction
    t = record.t
    mutated = Boundary.from_json(boundary.to_json())
    k = boundary.output_records.index(record)
    mutated.output_records[k] = OutputRecord(t=t, present=True, token_id=value, step=record.step)
    mutated.output_token_ids[1] = value
    sampler_cid = toy.value_id(rid, t, "sampler.select")
    raw = np.array(value, dtype=np.int32).tobytes()
    patched = PatchedReader(reader, values={sampler_cid: raw}, boundaries={rid: mutated})
    commitments = commit.build_commitments(patched, write=False)
    return patched, commitments, t


def test_toy_sampler_declares_the_pinned_16_bit_domain(toy):
    node = toy.description.node("sampler.select")
    assert node.declared_domain == 16
    assert node.dtype == "int32"  # the 32-bit storage container stays (§31.2)


def test_honest_run_passes_linkage_and_domain_checks(toy):
    reader = toy.reader()
    for rid in reader.request_ids():
        linkage = relations.check_output_linkage(toy.description, reader, toy.commitments, rid)
        assert linkage.ok, linkage.problems
        domains = relations.check_output_domains(toy.description, reader, toy.commitments, rid)
        assert domains.ok and domains.checked == len(reader.boundary(rid).sampled_positions())


@pytest.mark.parametrize(
    "value",
    [
        0x0001_00C6,  # walkthrough-b0.md §5's example: low half 198, high half 1
        1 << 16,  # the smallest out-of-domain code
        0x7FFF_FFFF,  # max int32
        -7,  # negative int32: container 0xFFFF_FFF9, high word 0xFFFF
    ],
)
def test_high_bits_set_is_rejected_on_every_audit(toy, value):
    """REJECTED: a committed output leaf outside the declared 16-bit domain, however consistent.

    Both copies of the value authenticate and equal each other, so the pre-existing linkage
    equality passes and no sampled relation runs — only the unconditional §32.5 comparison
    rejects, on both the boundary leaf and the interior sampler leaf.
    """
    patched, commitments, t = _consistent_claim(toy, "req-1", value)
    linkage = relations.check_output_linkage(toy.description, patched, commitments, "req-1")
    assert not linkage.ok
    domain_problems = [p for p in linkage.problems if "declared 16-bit domain" in p]
    assert len(domain_problems) == 2, linkage.problems  # boundary leaf + interior sampler leaf
    assert all(f"t={t}" in p for p in domain_problems)
    # the boundary-only every-request check (run for requests outside J too) rejects as well
    domains = relations.check_output_domains(toy.description, patched, commitments, "req-1")
    assert not domains.ok and len(domains.problems) == 1


def test_unused_but_in_domain_code_is_accepted_by_the_type_check(toy):
    """ACCEPTED: an out-of-vocabulary code inside the declared 16-bit domain (§31.2/§32.5).

    60,000 > the toy's vocabulary (32) — the B0 analogue of codes 49,152..65,535 — but
    60,000 < 2^16: the type check passes; catching it is the sampled argmax/embedding
    relations' job (probability-of-detection), not the unconditional bound machinery's.
    """
    value = 60_000
    assert value >= toy.description.symbols["V"] and value < 1 << 16
    patched, commitments, _ = _consistent_claim(toy, "req-1", value)
    linkage = relations.check_output_linkage(toy.description, patched, commitments, "req-1")
    assert linkage.ok, linkage.problems
    domains = relations.check_output_domains(toy.description, patched, commitments, "req-1")
    assert domains.ok


def test_without_a_declaration_nothing_is_enforced(toy):
    """Pre-declaration behavior is preserved bit-for-bit: no declared_domain, no domain check.

    This is the WALK finding the change closes (walkthrough-b0.md §5: "nothing enforces that a
    committed token is < V or < 2^16") — and the regression guard that old records (whose
    descriptions lack the field) verify exactly as before.
    """
    doc = toy.description.to_json()
    for node in doc["nodes"]:
        node.pop("declared_domain", None)
    undeclared = Description(doc)
    assert undeclared.node("sampler.select").declared_domain is None
    patched, commitments, _ = _consistent_claim(toy, "req-1", 0x0001_00C6)
    linkage = relations.check_output_linkage(undeclared, patched, commitments, "req-1")
    assert linkage.ok, linkage.problems  # the consistent lie passes: only the declaration rejects it
    domains = relations.check_output_domains(undeclared, patched, commitments, "req-1")
    assert domains.ok and domains.checked == 0


def test_domain_check_does_not_need_interior_commitments(toy):
    """check_output_domains reads boundary leaves only: it runs for requests outside J."""
    reader = toy.reader()
    # commitments stripped to phase-0 shape: boundary + weights trees only, no replay units
    phase0 = dict(toy.commitments)
    phase0["trees"] = {
        tid: rec
        for tid, rec in toy.commitments["trees"].items()
        if rec.get("kind") in ("boundary", "weights")
    }
    for rid in reader.request_ids():
        report = relations.check_output_domains(toy.description, reader, phase0, rid)
        assert report.ok and report.checked > 0


def test_declared_domain_validation():
    base = {
        "op": "sampler.select", "layer": "sampler", "op_type": "sampler", "dtype": "int32",
        "shape": [], "relation": "argmax.lowest_index.v1",
    }  # fmt: skip
    from verity_vllm.description import Node

    assert Node.from_json({**base, "declared_domain": 16}).declared_domain == 16
    assert Node.from_json(base).declared_domain is None
    with pytest.raises(DescriptionError, match="outside 1..32"):
        Node.from_json({**base, "declared_domain": 33})
    with pytest.raises(DescriptionError, match="outside 1..32"):
        Node.from_json({**base, "declared_domain": 0})
    with pytest.raises(DescriptionError, match="integer container"):
        Node.from_json({**base, "dtype": "bfloat16", "declared_domain": 16})
    with pytest.raises(DescriptionError, match="must be an int"):
        Node.from_json({**base, "declared_domain": "16"})


def test_declaration_round_trips_and_moves_the_digest(toy):
    doc = toy.description.to_json()
    again = Description(doc)
    assert again.node("sampler.select").declared_domain == 16
    assert again.digest() == toy.description.digest()
    stripped = toy.description.to_json()
    for node in stripped["nodes"]:
        node.pop("declared_domain", None)
    # the declaration is digest-binding protocol metadata (bound at T_0 via bindings.architecture)
    assert Description(stripped).digest() != toy.description.digest()
