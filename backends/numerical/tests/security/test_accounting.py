"""Tests for the soundness calculator: paper figures reproduced, parameterisations for the two targets, refusals."""

from __future__ import annotations

from fractions import Fraction

import pytest

from verity_numerical.security.accounting import (
    BABYBEAR, GOLDILOCKS, MERSENNE61, AccountingError, CandidateA, CandidateB, CheckerLayer, Field,
    LigeroParams, account_candidate_a, account_candidate_b, data_parallel_layer_error, fractional_gkr_error,
    hash_collision_bound, ligero_error, ligero_queries_for, log2, logup_gkr_error, logup_identity_error,
    per_proof_target, random_combination_error, schwartz_zippel_error, security_bits, sumcheck_error,
    union_over_proofs,
)

GL = Field.prime("Goldilocks", GOLDILOCKS)
M61 = Field.prime("Mersenne61", MERSENNE61)
BB = Field.prime("BabyBear", BABYBEAR)

# One reference Ligero parameter set per target: RS rate 1/4, k = 2^12, l = 3840 payload symbols per row.
LIGERO_80 = LigeroParams(n=2**14, k=2**12, l=3840, t=121)
LIGERO_128 = LigeroParams(n=2**14, k=2**12, l=3840, t=192)


# ---------------------------------------------------------------------------------------------------------------------
# Figures checked against the papers
# ---------------------------------------------------------------------------------------------------------------------

def test_sumcheck_error_is_sum_of_degrees_over_field():
    """LFKN92 as stated in Haboeck 2022/1530 Theorem 2 and Frigo-shelat 2024/2010 Theorem 1: v*d/|F|."""
    assert sumcheck_error([3] * 20, M61) == Fraction(60, 2**61 - 1)
    assert round(log2(sumcheck_error([3] * 20, M61)), 2) == round(log2(60) - 61, 2)
    # a batch of L+1 sumchecks combined with random scalars adds 1/|F| (Haboeck (10))
    assert random_combination_error(M61, 5) == Fraction(1, 2**61 - 1)
    assert random_combination_error(M61, 5, "powers") == Fraction(4, 2**61 - 1)


def test_ligero_section_5_3_figure():
    """Ligero 2022/1608 section 5.3: with e = k and n = 3k the query terms are 3*(2/3)^t; the Appendix C field
    term is (n+3)/|F|^sigma (section 5.3 prints n+4; the Appendix C decomposition (n+1) + 1 + 1 is used)."""
    k = 1024
    p = LigeroParams(n=3 * k, k=k, l=k // 2, t=69, e=k, zk=False)
    F = Field.prime("p30", 2**30 - 35)   # the paper's implementation used a 30-bit prime field
    terms = ligero_error(p, F, F)
    assert terms["irs_query"] == Fraction(2, 3) ** 69
    assert terms["quadratic_query"] == Fraction(2, 3) ** 69
    assert terms["linear_query"] <= Fraction(2, 3) ** 69           # (k + l)/n <= 2/3 for l <= k
    assert terms["irs_field"] + terms["linear_field"] + terms["quadratic_field"] == Fraction(3 * k + 3, F.size)
    # t = lambda / log2(3/2) makes (2/3)^t = 2^-lambda: lambda = 40 -> t ~ 69 (the paper's 2^-40 setting)
    assert -41 < log2(Fraction(2, 3) ** 69) < -40


def test_ligero_theorem_4_7_bound():
    """Ligero Theorem 4.7 (e < (n-k)/4): (d+2)/|F|^sigma + (1-e/n)^t + 2((e+2k)/n)^t."""
    p = LigeroParams(n=4096, k=1024, l=512, t=100, e=700, sigma=2, zk=False)
    terms = ligero_error(p, GL, GL, analysis="theorem_4_7")
    assert terms["field"] == Fraction(4096 - 1024 + 1 + 2, GOLDILOCKS**2)
    assert terms["irs_query"] == Fraction(4096 - 700, 4096) ** 100
    assert terms["linear_query"] == terms["quadratic_query"] == Fraction(700 + 2048, 4096) ** 100


def test_ligero_parameter_constraints_are_enforced():
    with pytest.raises(AccountingError, match="n > 2k"):
        ligero_error(LigeroParams(n=2048, k=1024, l=512, t=50), GL, GL)          # rate 1/2: (2k/n)^t = 1
    with pytest.raises(AccountingError, match="k > l \\+ t"):
        ligero_error(LigeroParams(n=4096, k=1024, l=1000, t=50), GL, GL)         # no zero-knowledge
    with pytest.raises(AccountingError, match="e < d/2"):
        ligero_error(LigeroParams(n=4096, k=1024, l=512, t=50, e=2000), GL, GL)
    with pytest.raises(AccountingError, match="distinct evaluation points"):
        ligero_error(LigeroParams(n=2**31, k=2**29, l=2**28, t=50), Field.prime("tiny", 2**31 - 1), GL)


def test_logup_identity_term_matches_theorem_4():
    """Haboeck 2022/1530 Theorem 4, first term: ((M+1)|H| - 1)/(|F| - |H|) for M columns and a table of length |H|."""
    M, H = 8, 2**12
    assert logup_identity_error(M * H, H, GL) == Fraction((M + 1) * H - 1, GOLDILOCKS - H)
    # the characteristic must exceed the number of looked-up elements (Lemma 5)
    with pytest.raises(AccountingError, match="char"):
        logup_identity_error(2**31, 16, BB)


def test_fractional_gkr_matches_papini_haboeck_proposition_1():
    """Exact sum (2(n-1)+1)/|F| + sum_{k<n} 3k/|F| is at most the paper's simplified n(3n+1)/(2|F|)."""
    for n in (1, 2, 5, 17, 24):
        exact = fractional_gkr_error(n, GL)
        assert exact == Fraction(2 * (n - 1) + 1 + 3 * n * (n - 1) // 2, GOLDILOCKS)
        assert exact <= Fraction(n * (3 * n + 1), 2 * GOLDILOCKS)
    lk = logup_gkr_error(num_lookups=12_582_912, table_rows=65536, F=GL.extension(2))
    assert lk["hypercube_dim"] == 24                                    # 2^24 >= N + |T| = 12648448
    assert lk["identity"] == Fraction(12_582_912 + 65536 - 1, GOLDILOCKS**2 - 65536)


def test_data_parallel_layer_degrees_and_rounds():
    """Thaler section 7.3 order: bind (g1, l1, r1) then the b copy variables; degrees 2, 2, 2, 3."""
    acc = data_parallel_layer_error(s_out=8, s_in=7, b=19, F=GL.extension(3))
    assert acc["rounds"] == 8 + 2 * 7 + 19
    assert acc["sumcheck"] == Fraction(2 * 8 + 3 * 19 + 4 * 7, GOLDILOCKS**3)
    assert acc["alpha"] == Fraction(1, GOLDILOCKS**3)
    assert schwartz_zippel_error(27, GL.extension(3)) == Fraction(27, GOLDILOCKS**3)


def test_hash_term_is_generic_birthday_bound():
    assert hash_collision_bound(256, 2**64) == Fraction(1, 2**128)
    assert hash_collision_bound(256, 2**60) == Fraction(1, 2**136)


# ---------------------------------------------------------------------------------------------------------------------
# Target parameterisations, K = 1536, B in {1, 64, 4096}
# ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("B", [1, 64, 4096])
def test_two_to_minus_80_parameterisation(B):
    """Goldilocks trace field, quadratic-extension challenge field (~2^128), Ligero rate 1/4 with t = 121."""
    Fc = GL.extension(2)
    a = account_candidate_a(CandidateA(K=1536, B=B, trace_field=GL, challenge_field=Fc, ligero=LIGERO_80))
    b = account_candidate_b(CandidateB(K=1536, B=B, trace_field=GL, challenge_field=Fc, ligero=LIGERO_80))
    for budget in (a, b):
        assert budget.meets(-80), budget.table()
        assert security_bits(budget) >= 80
        comps = budget.by_component()
        assert log2(comps["encoding_opening"]) < -80          # the query terms dominate at this field size
        assert log2(comps["lookup"]) < -100
    assert a.parameters["units"] == B * 96 and a.parameters["copy_bits_b"] == max(0, (B * 96 - 1).bit_length())


@pytest.mark.parametrize("B", [1, 64, 4096])
def test_two_to_minus_128_parameterisation(B):
    """Goldilocks trace field, CUBIC-extension challenge field (~2^192), Ligero rate 1/4 with t = 192, adversary
    hash work 2^60.  A quadratic extension cannot reach 2^-128: the logUp identity term alone is ~2^-102 at
    B = 4096 and ~2^-110 at B = 1 (table-size bound)."""
    Fc3 = GL.extension(3)
    a = account_candidate_a(CandidateA(K=1536, B=B, trace_field=GL, challenge_field=Fc3, ligero=LIGERO_128,
                                       hash_queries=2**60))
    b = account_candidate_b(CandidateB(K=1536, B=B, trace_field=GL, challenge_field=Fc3, ligero=LIGERO_128,
                                       hash_queries=2**60))
    for budget in (a, b):
        assert budget.meets(-128), budget.table()
        assert budget.statistical_total() <= Fraction(1, 2**129)
    Fc2 = GL.extension(2)
    a2 = account_candidate_a(CandidateA(K=1536, B=B, trace_field=GL, challenge_field=Fc2, ligero=LIGERO_128))
    assert not a2.meets(-128)
    assert log2(a2.by_component()["lookup"]) > -112


def test_two_to_minus_128_reference_decomposition_b4096():
    """The figures quoted in the PROTOCOL.md documents and the lane report (K = 1536, B = 4096)."""
    Fc3 = GL.extension(3)
    a = account_candidate_a(CandidateA(K=1536, B=4096, trace_field=GL, challenge_field=Fc3, ligero=LIGERO_128,
                                       hash_queries=2**60))
    b = account_candidate_b(CandidateB(K=1536, B=4096, trace_field=GL, challenge_field=Fc3, ligero=LIGERO_128,
                                       hash_queries=2**60))
    ca, cb = a.by_component(), b.by_component()
    assert a.parameters["units"] == 393216 and a.parameters["copy_bits_b"] == 19
    assert -187 < log2(ca["batching"]) < -186
    assert -184 < log2(ca["sumcheck"]) < -183
    assert -166 < log2(ca["lookup"]) < -164 and -166 < log2(cb["lookup"]) < -164
    assert -131 < log2(ca["encoding_opening"]) < -130 and -131 < log2(cb["encoding_opening"]) < -130
    assert ca["encoding_opening"] < cb["encoding_opening"]      # B combines constraints by powers (Schwartz-Zippel)
    assert log2(ca["commitment_hash"]) == -136
    assert -131 < log2(a.total()) <= -128 and -131 < log2(b.total()) <= -128
    assert cb["batching"] == 0 and cb["sumcheck"] == 0        # no sumcheck in B's helper-column lookups
    assert b.parameters["commitment_rounds"] == 2 and a.parameters["committed_trace_elements"] < b.parameters["committed_trace_elements"]


def test_ligero_queries_for_targets():
    assert ligero_queries_for(-82, 2**14, 2**12, 3840) == 121
    assert ligero_queries_for(-130, 2**14, 2**12, 3840) == 192


def test_candidate_b_gkr_lookup_mode_has_one_commitment_round():
    Fc = GL.extension(2)
    b = account_candidate_b(CandidateB(K=1536, B=64, trace_field=GL, challenge_field=Fc, ligero=LIGERO_80,
                                       lookup_mode="gkr"))
    assert b.parameters["commitment_rounds"] == 1 and b.by_component()["lookup"] > 0
    assert b.meets(-80)


def test_small_field_study_limits():
    """A 31-bit trace field: a degree-5 extension (~2^155) misses 2^-128 at B = 4096 (lookup terms ~2^-127);
    degree 6 reaches it.  logUp's characteristic condition caps the total lookups per table below 2^31."""
    a5 = account_candidate_a(CandidateA(K=1536, B=4096, trace_field=BB, challenge_field=BB.extension(5),
                                        ligero=LIGERO_128, hash_queries=2**60))
    assert not a5.meets(-128) and -128 < log2(a5.by_component()["lookup"]) < -126
    Fc = BB.extension(6)
    a = account_candidate_a(CandidateA(K=1536, B=4096, trace_field=BB, challenge_field=Fc, ligero=LIGERO_128,
                                       hash_queries=2**60))
    assert a.meets(-128)
    with pytest.raises(AccountingError, match="char"):
        account_candidate_a(CandidateA(K=1536, B=2**20, trace_field=BB, challenge_field=Fc, ligero=LIGERO_128))


# ---------------------------------------------------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------------------------------------------------

def test_refuses_bits_of_security_from_a_field_size():
    with pytest.raises(AccountingError, match="not of a field size"):
        security_bits(GL)
    with pytest.raises(AccountingError, match="not of a field size"):
        security_bits(GL.extension(2))
    with pytest.raises(AccountingError, match="not of a field size"):
        security_bits(2**128)
    assert not hasattr(GL, "bits")


def test_layer_shape_is_a_parameter_not_a_constant():
    Fc = GL.extension(3)
    depth1 = account_candidate_a(CandidateA(K=1536, B=64, trace_field=GL, challenge_field=Fc, ligero=LIGERO_128,
                                            layers=(CheckerLayer(8, 9),), inputs_per_unit=350, hash_queries=2**60))
    assert depth1.parameters["checker_layers"] == 1 and depth1.meets(-128)
    with pytest.raises(AccountingError, match="extension"):
        account_candidate_a(CandidateA(K=1536, B=64, trace_field=GL, challenge_field=M61, ligero=LIGERO_128))


def test_sub_batches_are_union_bounded_mechanically():
    """A 4096-instance batch proved as 16 sub-batches of 256 does not get to quote the per-proof error: every soundness
    term is multiplied by 16, so a per-proof 2^-130.2 becomes 2^-126.2 and no longer meets 2^-128; the parameters
    record the count."""
    Fc3 = GL.extension(3)
    one = account_candidate_b(CandidateB(K=1536, B=256, trace_field=GL, challenge_field=Fc3, ligero=LIGERO_128,
                                         hash_queries=2**60))
    sixteen = union_over_proofs(one, 16)
    assert sixteen.parameters["n_proofs"] == 16
    assert sixteen.total() == 16 * one.total()
    assert sixteen.zk_total() == 16 * one.zk_total()
    assert one.meets(-128)
    assert log2(sixteen.total()) == pytest.approx(log2(one.total()) + 4)
    with pytest.raises(AccountingError):
        union_over_proofs(one, 0)


def test_per_proof_target_is_derived_from_the_application_budget():
    """With an application budget of 2^-128 spent over 1024 proofs and no other error, each proof may spend 2^-138;
    other protocol errors reduce what is left; an over-spent budget is refused."""
    assert per_proof_target(-128, 1024) == pytest.approx(-138)
    assert per_proof_target(-128, 1) == pytest.approx(-128)
    assert per_proof_target(-128, 2, other_errors=Fraction(1, 2**129)) == pytest.approx(-130)
    with pytest.raises(AccountingError):
        per_proof_target(-128, 4, other_errors=Fraction(1, 2**127))


# ---------------------------------------------------------------------------------------------------------------------
# Coordinator decisions 2026-09-21: conditioned challenge space, depth, census, operating points, step 0
# ---------------------------------------------------------------------------------------------------------------------

from verity_numerical.bench import contract
from verity_numerical.security.accounting import (
    DepthOptions, census_estimate_a, census_estimate_b, e2e_sweep, first_configuration, hm_commitment_hiding,
    redraw_terms, rounds_and_depth_a, rounds_and_depth_b,
)

Fc3 = GL.extension(3)


def _a(B=4096, **kw):
    return CandidateA(K=1536, B=B, trace_field=GL, challenge_field=Fc3, ligero=LIGERO_128, hash_queries=2**60, **kw)


def _b(B=4096, **kw):
    return CandidateB(K=1536, B=B, trace_field=GL, challenge_field=Fc3, ligero=LIGERO_128, hash_queries=2**60, **kw)


def test_conditioned_challenge_space_vs_whole_field_theorem():
    """T_range16 at B = 4096: N = 18,087,936 lookups, |T| = 65,536, F_c = Goldilocks^3.  The conditioned bound
    d/(|F| - |T|) exceeds the whole-field d/|F| by exactly d|T|/(|F|(|F| - |T|)) ~ 2^-344, a relative 2^-176."""
    N, T = 46 * 393216, 65536
    cond = logup_identity_error(N, T, Fc3, "conditioned")
    whole = logup_identity_error(N, T, Fc3, "whole_field")
    d = N + T - 1
    assert cond == Fraction(d, Fc3.size - T) and whole == Fraction(d, Fc3.size)
    assert cond - whole == Fraction(d * T, Fc3.size * (Fc3.size - T))
    assert -344 < log2(cond - whole) < -343
    assert cond / whole - 1 == Fraction(T, Fc3.size - T) and -177 < log2(cond / whole - 1) < -175
    assert logup_identity_error(N, T, Fc3, exclude_poles=False) == whole      # alias kept
    with pytest.raises(AccountingError, match="challenge_space"):
        logup_identity_error(N, T, Fc3, "other")


def test_redraw_terms_are_public_and_not_soundness():
    rd = redraw_terms(65536, Fc3, slots=2)
    assert rd["redraw"] == Fraction(65536, Fc3.size) and rd["abort"] == Fraction(65536, Fc3.size) ** 2
    bud = account_candidate_a(_a())
    assert bud.completeness_total() > 0 and all(t.kind == "completeness" for t in bud.terms if t.component == "completeness")
    # the redraw terms never enter the soundness total
    assert bud.total() == sum(t.error for t in bud.soundness_terms())
    assert -175 < log2(bud.completeness_total()) < -174


def test_sequential_depth_candidate_a_b4096():
    """Baseline (known techniques only): 1 (all z) + 325 (T_range16 chain: 1 + sum_{k=1}^{24} (k + 1)) + 2 (Ligero)."""
    d = rounds_and_depth_a(_a())
    assert d["depth.lookup_chain"] == 1 + sum(k + 1 for k in range(1, 25)) == 325
    assert d["depth.checker_chain"] == sum(r + 1 for r in (41, 38, 35, 40)) + 1 == 159
    assert d["rounds.sequential_depth"] == 328 and d["rounds.protocol"] == 3607
    # queries on the last computed layer put that layer (40 rounds + alpha) and gamma after the lookups
    assert rounds_and_depth_a(_a(depth=DepthOptions(queries_on_inputs=False)))["rounds.sequential_depth"] == 370
    # proposals
    assert rounds_and_depth_a(_a(depth=DepthOptions(variables_per_round=2)))["rounds.sequential_depth"] == 184
    assert rounds_and_depth_a(_a(depth=DepthOptions(table_shard_bits=5)))["rounds.sequential_depth"] == 213
    assert rounds_and_depth_a(_a(depth=DepthOptions(variables_per_round=2, table_shard_bits=5)))["rounds.sequential_depth"] == 123
    assert rounds_and_depth_a(_a(depth=DepthOptions(variables_per_round=3, table_shard_bits=5)))["rounds.sequential_depth"] == 93
    # step 0 adds one verifier message before com_1
    assert rounds_and_depth_a(_a(coin_commitment_bits=160))["rounds.sequential_depth"] == 329
    # the sharded budget still meets the target and pays (2^s - 1)|T| multiplicities per sharded table
    sh = account_candidate_a(_a(depth=DepthOptions(table_shard_bits=5)))
    assert sh.meets(-128) and sh.parameters["committed_trace_elements"] - account_candidate_a(_a()).parameters["committed_trace_elements"] == 5_616_742


def test_sequential_depth_candidate_b():
    assert rounds_and_depth_b(_b())["rounds.sequential_depth"] == 3
    assert rounds_and_depth_b(_b(coin_commitment_bits=160))["rounds.sequential_depth"] == 4
    assert rounds_and_depth_b(_b(lookup_mode="gkr"))["rounds.sequential_depth"] == 328
    for B in (1, 64):
        assert rounds_and_depth_b(_b(B=B))["rounds.sequential_depth"] == 3


def test_e2e_sweep_is_depth_times_rtt_plus_given_compute():
    s = e2e_sweep(328, prover_seconds=0.0)
    assert s["t.e2e.rtt_0ms"] == 0 and s["t.e2e.rtt_1ms"] == pytest.approx(0.328) and s["t.e2e.rtt_50ms"] == pytest.approx(16.4)
    assert e2e_sweep(3, prover_seconds=2.0)["t.e2e.rtt_10ms"] == pytest.approx(2.03)
    assert set(s) == {f"t.e2e.rtt_{r}ms" for r in ("0", "0.1", "1", "10", "50")}


def test_census_estimates_use_contract_names():
    for est in (census_estimate_a(_a()), census_estimate_b(_b())):
        for name in est:
            if name.startswith("estimate."):
                continue
            assert name in contract.UNITS, name
    a, b = census_estimate_a(_a()), census_estimate_b(_b())
    assert a["census.committed_elements"] == 100_858_462 and b["census.committed_elements"] == 404_874_820
    assert a["census.bytes_encoded"] == 26_269 * 2**14 * 8 and b["census.bytes_encoded"] == 105_440 * 2**14 * 8
    assert a["rounds.sequential_depth"] == 328 and b["rounds.sequential_depth"] == 3
    assert b["census.committed_elements"] / a["census.committed_elements"] == pytest.approx(4.01, abs=0.01)


@pytest.mark.parametrize("target,ext,t", [(-80, 2, 120), (-96, 2, 144), (-112, 3, 167), (-128, 3, 191)])
def test_first_configuration_per_operating_point(target, ext, t):
    """Which configuration first meets each operating point (Goldilocks trace field, n = 2^14, k = 2^12, l = 3840,
    adversary 2^60 hash evaluations): quadratic extension up to 2^-96, cubic from 2^-112; B in {1, 64, 4096}."""
    def mk_a(Ft, Fc, lig, hash_queries, B):
        return account_candidate_a(CandidateA(K=1536, B=B, trace_field=Ft, challenge_field=Fc, ligero=lig, hash_queries=hash_queries))

    def mk_b(Ft, Fc, lig, hash_queries, B):
        return account_candidate_b(CandidateB(K=1536, B=B, trace_field=Ft, challenge_field=Fc, ligero=lig, hash_queries=hash_queries))

    for B in (1, 64, 4096):
        for mk in (mk_a, mk_b):
            d, tt, bud = first_configuration(target, mk, GL, 2**14, 2**12, 3840, B=B)
            assert (d, tt) == (ext, t), (target, B, d, tt)
            assert bud.meets(target)
    if ext == 3:
        with pytest.raises(AccountingError, match="lookup"):
            first_configuration(target, mk_a, GL, 2**14, 2**12, 3840, extension_degrees=(2,), B=4096)


def test_coin_commitment_step0_terms_and_label():
    hv = account_candidate_a(_a())
    zk = account_candidate_a(_a(coin_commitment_bits=160))
    assert hv.parameters["zk_class"].startswith("COMPLETE_HVZK") and zk.parameters["zk_class"].startswith("COMPLETE_ZK")
    hm = [t for t in zk.terms if "coin commitments" in t.name]
    assert len(hm) == 1 and hm[0].kind == "statistical" and hm[0].error == hm_commitment_hiding(3607, 160)
    assert zk.meets(-128) and zk.total() - hv.total() == hm[0].error
    assert -149 < log2(hm[0].error) < -148
    leaf = [t for t in zk.terms if "column commitments" in t.name][0]
    assert leaf.kind == "zk" and leaf.error == Fraction(2**14, 2**160)


def test_candidate_b_powers_combination_terms():
    b = account_candidate_b(_b())
    terms = {t.name.split()[1]: t.error for t in b.terms if t.component == "encoding_opening"}
    n_lin, n_quad = b.parameters["linear_constraints"], b.parameters["quadratic_constraints"]
    assert terms["linear_field"] == Fraction(n_lin - 1, Fc3.size)
    assert terms["quadratic_field"] == Fraction(-(-n_quad // 3840) - 1, Fc3.size)
    assert -166 < log2(terms["linear_field"]) < -164
    with pytest.raises(AccountingError, match="n_linear"):
        ligero_error(LIGERO_128, GL, Fc3, combination="powers")
