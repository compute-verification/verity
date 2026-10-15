import Definitions.Pous
import Models.Pous.Assumptions
import Proofs.Pous.Statements
import Proofs.Pous

/-!
# Axiom audit

`lake env lean CheckAxioms.lean`. Every headline theorem of the package must depend only on `propext`,
`Classical.choice` and `Quot.sound`. Named assumptions are hypotheses, never `axiom` declarations, so they cannot
appear here. `tools/lean/audit.py` checks every declaration; this list is the headline subset, one line each.
-/

-- sanity: the honest prover is accepted
#print axioms Pous.honest_sim_accepted
#print axioms Pous.honest_seq_accepted
-- sanity: storing `m` raw blocks passes with probability at least `(m/B)^k`
#print axioms Pous.raw_sim_accepted
#print axioms Pous.raw_seq_accepted
-- sanity: the requirement forces `k ≥ 86`
#print axioms Pous.k_ge_86
-- sanity: in the plain model, coins that fit in the state break every scheme
#print axioms Pous.plain_short_coins_insecure
-- sanity: degenerate requirements are excluded (`W` parked in `pp`; `ε` large enough to void the audit)
#print axioms Pous.junk_correct
#print axioms Pous.junk_spaceBound
#print axioms Pous.junk_not_meets
#print axioms Pous.not_meets_of_lt_eps
#print axioms Pous.not_meets_eps_one
-- accounting: the report's operating point
#print axioms Pous.Accounting.p₀_at_report
#print axioms Pous.Accounting.report_k214
-- the proved guarantees (`Security/Proofs/Pous.lean`)
#print axioms Pous.SecurityProofs.KFloor
#print axioms Pous.SecurityProofs.Theorem1ii
#print axioms Pous.SecurityProofs.Theorem1iiTimed
#print axioms Pous.SecurityProofs.Theorem1iTimed
#print axioms Pous.SecurityProofs.Theorem1iiSeq
#print axioms Pous.SecurityProofs.Theorem1iiiTimed
#print axioms Pous.SecurityProofs.FibreCount
#print axioms Pous.SecurityProofs.A1CompressionCard
#print axioms Pous.SecurityProofs.A1CompressionPr
#print axioms Pous.SecurityProofs.A3Prime
#print axioms Pous.SecurityProofs.A3PrimeLam
#print axioms Pous.SecurityProofs.A6RederiveUntimed
#print axioms Pous.SecurityProofs.A6RederiveTimed
#print axioms Pous.SecurityProofs.B1IndepSet
#print axioms Pous.SecurityProofs.B2Stacking
#print axioms Pous.SecurityProofs.B3Chain
#print axioms Pous.SecurityProofs.FischClaim13
#print axioms Pous.SecurityProofs.FischClaim14
#print axioms Pous.SecurityProofs.B5ExPostFactoBounded
#print axioms Pous.SecurityProofs.B5ExPostFacto
#print axioms Pous.SecurityProofs.TransposeMixing
#print axioms Pous.SecurityProofs.ReverseNeedsAllChildren
#print axioms Pous.SecurityProofs.N10Refuted
#print axioms Pous.SecurityProofs.SampledAuditProd
#print axioms Pous.SecurityProofs.SampledAuditSeq
#print axioms Pous.SecurityProofs.SampledAuditSim
#print axioms Pous.SecurityProofs.StackedAuditSeq
#print axioms Pous.SecurityProofs.StackedAuditOnePercent
#print axioms Pous.SecurityProofs.DenseMeets
-- the one-way bound on Fisch's stacked graph
#print axioms PousBridge.b2ToB5
#print axioms PousBridge.b5_stacked
-- the two-directional band certificate (`X = 3`) and the band graph's depth robustness
#print axioms PousP3Column.columnHard_of_schedule
#print axioms PousP3Column.band_columnHard_twoWay
#print axioms PousP3Column.band_twoWay_220_12
#print axioms PousP3Column.band_twoWay_220_12_d11
#print axioms PousP3Column.band_twoWay_220_16
#print axioms PousP3Column.band_depthRobust
-- the dense scheme at the deployment point: one 32 MiB segment, and 448 segments (14 GB), random-oracle `H`
#print axioms PousDenseReCert.dense_meets_64
#print axioms PousDenseReCert.seg_meets_14
-- the 14 GB theorem's global indices are the implementation's per-segment tags `salt ‖ u64 s`, relabeled injectively
#print axioms PousDenseReCert.domainSep_equiv
#print axioms PousDenseReCert.tag_injective_14
-- the 14 GB theorem for the implementation's tagged keying, and the oracle-game transfer it rests on
#print axioms PousDenseReCert.segTag_meets_14
#print axioms PousDenseReCert.segTag_auditSecure
-- the overwrite chain: P3 with a concrete `H`, the band, the dense chain scheme and the public encoder, no hypothesis
#print axioms Pous.SecurityProofs.P3ConcreteMeets64
#print axioms Pous.SecurityProofs.P3ConcreteMeets14GB
#print axioms Pous.SecurityProofs.P3ConcreteMeets64D12
#print axioms Pous.SecurityProofs.BandChainMeets64
#print axioms Pous.SecurityProofs.BandChainMeets64D5
#print axioms Pous.SecurityProofs.BandChainMeets64D12
#print axioms Pous.SecurityProofs.BandMultiMeets14
#print axioms Pous.SecurityProofs.BandMultiMeets14K106
#print axioms Pous.SecurityProofs.BandMultiMeetsFamily
#print axioms Pous.SecurityProofs.ChainDenseMeets64
#print axioms Pous.SecurityProofs.BandMultiMeetsFamilyD2
#print axioms Pous.SecurityProofs.BandMultiMeetsFamilyD1
#print axioms Pous.SecurityProofs.BandMultiMeetsFamilyD0
#print axioms Pous.SecurityProofs.BandChainMeets64Rounds2
#print axioms Pous.SecurityProofs.BandChainMeets64Rounds1
#print axioms Pous.SecurityProofs.BandPubMeets64
#print axioms Pous.SecurityProofs.BandPubMeets14
#print axioms Pous.SecurityProofs.BandPubMeets14Global
#print axioms Pous.SecurityProofs.BandPubMeetsFamily
#print axioms Pous.SecurityProofs.BandPubMeetsFamilyD2
#print axioms Pous.SecurityProofs.BandPubMeetsFamilyD1
#print axioms Pous.SecurityProofs.ChainDenseMeets111
#print axioms Pous.SecurityProofs.BandGradedMeetsFamily
#print axioms Pous.SecurityProofs.PerAnswer
#print axioms Pous.SecurityProofs.ContinuousAudit
#print axioms Pous.SecurityProofs.BackgroundRebuildExcluded
#print axioms Pous.SecurityProofs.ContinuousComplete
#print axioms Pous.SecurityProofs.ContinuousTimeAverage
#print axioms Pous.SecurityProofs.BandFamilyContinuousTimeAverage
-- secure erasure (the 4 Oct volatile-memory model): the digest, the reduction, the band family and 96 GiB
#print axioms Pous.SecurityProofs.DigestAU
#print axioms Pous.SecurityProofs.SecureErasureMeets
#print axioms Pous.SecurityProofs.SecureErasureMeetsFamily
#print axioms Pous.SecurityProofs.SecureErasureMeets96GiB
-- the abstract model `m1p` at P2's parameters, no hypothesis: the Meets rows, the slack rows and the error split
#print axioms Pous.SecurityProofs.P2MeetsM1pB19Uncond
#print axioms Pous.SecurityProofs.P2MeetsM1pFamilyUncond
#print axioms Pous.SecurityProofs.P2MeetsM1pW8192Uncond
#print axioms Pous.SecurityProofs.P2SlackB19Uncond
#print axioms Pous.SecurityProofs.P2SlackFamilyUncond
#print axioms Pous.SecurityProofs.P2ErrorSplitB19Uncond
#print axioms Pous.SecurityProofs.P2SlackFamilyFreeBlocks
#print axioms Pous.SecurityProofs.P2SlackFamilyWide
#print axioms Pous.SecurityProofs.P2SlackFamilyFreeBlocksWide
#print axioms PousP2p.not_meets_m1p_b19_k85
#print axioms PousMargin.not_slack_raw
-- P2 in the ring cost model (`Security/Definitions/Pous/P2/Cost.lean`): `MeetsOn` at `L = Q` gives `Meets`,
-- the non-vacuity pins (honest storage is accepted; recomputing the roots breaks `MeetsOn`), and dropping low bits
-- breaks `k = 88`
#print axioms PousP2Ring.meets_of_meetsOn
#print axioms PousP2Ring.complete
#print axioms PousP2Ring.tight
#print axioms PousP2Ring.tight_p16448
#print axioms PousP2Ring.not_meetsOn_b19_k88
-- P2 per round, at every block count from `2^19` to `2^25`: the graded bound, today's whole-audit form derived from
-- it, and its falsification tests
#print axioms Pous.SecurityProofs.P2RingAuditSecureOfRoundSecure
#print axioms Pous.SecurityProofs.P2RingGradedOfRoundSecure
#print axioms PousP2Rounds.floor_043
#print axioms PousP2Rounds.floor_043_eps
#print axioms PousP2Rounds.round_tight
#print axioms PousP2Rounds.round_trunc
#print axioms PousP2Rounds.round_trunc_b19
