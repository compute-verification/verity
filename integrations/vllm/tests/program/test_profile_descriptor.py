"""The descriptor profiler's attribution is exhaustive (parts sum to the definition bytes) and its two dominant
terms follow their closed forms."""

from __future__ import annotations

from verity.ml.circuits import REGISTRY

from verity_vllm.program import profile_descriptor as PD
from verity.ml.circuits.codec import SCHEMA_V0, decode_program, descriptor_size, encode_program, program_digest
from verity_vllm.program.registry import b1

BN = b1.BN


def test_b0_small_profile_is_exhaustive_and_matches_the_formulas():
    LP, STEPS = 4, 1
    desc = PD.build("B0", LP, STEPS)
    prof = PD.profile(desc)
    assert prof["descriptor_bytes"] == descriptor_size(desc)
    assert prof["program_digest"] == program_digest(desc)
    # every definition's parts (incl. `structure`) sum to its bytes; families sum to the definitions field
    fam_total = sum(f["bytes"] for f in prof["families"].values())
    assert fam_total < prof["top_level"]["definitions"] < fam_total + 200 * prof["definitions"]
    for f in prof["families"].values():
        assert sum(f["parts"].values()) == f["bytes"]
    # (1) the root's KV-cache concats (v1): at every position T >= 2 the `kc`/`vc` argument is an alias of the
    # previous position's argument plus the one new row — 2 parts, NL * 2 (k, v) * (P - 1) concats — instead of
    # the v0 re-materialised T single-position parts (NL * 2 * sum_{T=2}^{P} T, quadratic in P)
    C = b1.B0_CONFIG
    P = LP + STEPS  # positions attended (T = 1..P)
    root = prof["concat_by_family"]["Root"]
    assert root["concats"] == C["NL"] * 2 * (P - 1)
    assert root["parts"] == 2 * root["concats"]
    assert root["alias_parts"] == C["NL"] * 2 * (P - 2)  # T = 2 concatenates two single rows; T >= 3 aliases T - 1
    assert root["all_parts_distinct_nodes"] == root["concats"]
    v0 = PD.profile(encode_program(decode_program(desc, REGISTRY), schema=SCHEMA_V0))
    assert v0["concat_by_family"]["Root"]["parts"] == C["NL"] * 2 * sum(range(2, P + 1))
    assert v0["program_digest"] != prof["program_digest"] and v0["descriptor_bytes"] > prof["descriptor_bytes"]
    # (2) AttentionHead{T}: the zero-row padding (Clear_OOB_MN) is a zero-stride `strided` part — one reference to
    # the body's single `Const16[0]` gate repeated D * (BN - nvis(T)) times — so a padded K (or V) block is a
    # 2-part concat [rows, zeros] and there are no single-leaf zero parts left (before the fix: 2 * D * (BN - nvis)
    # + 4 + D single-leaf parts per specialisation, 5.16 MB of the 12.3 MB B0 descriptor; descriptor-codec.md §2)
    ah = prof["concat_by_family"]["AttentionHead_v1"]
    padded = sum(1 for T in range(1, P + 1) if T % BN)
    assert ah["concats"] == 2 * padded and ah["parts"] == 4 * padded
    assert ah["single_leaf_parts"] == 0
    assert prof["kinds"]["node.refs.strided"]["count"] >= 2 * P  # l0/o0 broadcasts of the single Const32[0] gate
    assert root["bytes"] < 0.05 * prof["descriptor_bytes"]  # the root's KV concats are no longer the top offender
    assert prof["explicit"]["lists"] == 0
    assert set(prof["top_level"]) >= {"callees", "types"}  # v1 interning tables are accounted for
