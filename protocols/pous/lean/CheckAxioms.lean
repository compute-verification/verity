import Pous
import PousProofs

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
-- the proved pinned statements (`PousProofs/Pinned.lean`)
#print axioms PousProofs.KFloor
#print axioms PousProofs.Theorem1ii
#print axioms PousProofs.Theorem1iiTimed
#print axioms PousProofs.Theorem1iTimed
#print axioms PousProofs.Theorem1iiSeq
#print axioms PousProofs.Theorem1iiiTimed
#print axioms PousProofs.FibreCount
#print axioms PousProofs.A1CompressionCard
#print axioms PousProofs.A1CompressionPr
#print axioms PousProofs.A3Prime
#print axioms PousProofs.A3PrimeLam
#print axioms PousProofs.A6RederiveUntimed
#print axioms PousProofs.A6RederiveTimed
#print axioms PousProofs.B1IndepSet
#print axioms PousProofs.B2Stacking
#print axioms PousProofs.B3Chain
#print axioms PousProofs.FischClaim13
#print axioms PousProofs.FischClaim14
#print axioms PousProofs.B5ExPostFactoBounded
#print axioms PousProofs.B5ExPostFacto
#print axioms PousProofs.TransposeMixing
#print axioms PousProofs.ReverseNeedsAllChildren
#print axioms PousProofs.N10Refuted
#print axioms PousProofs.SampledAuditProd
#print axioms PousProofs.SampledAuditSeq
#print axioms PousProofs.SampledAuditSim
#print axioms PousProofs.StackedAuditSeq
#print axioms PousProofs.StackedAuditOnePercent
#print axioms PousProofs.DenseMeets
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
#print axioms PousProofs.P3ConcreteMeets64
#print axioms PousProofs.P3ConcreteMeets14GB
#print axioms PousProofs.P3ConcreteMeets64D12
#print axioms PousProofs.BandChainMeets64
#print axioms PousProofs.BandChainMeets64D5
#print axioms PousProofs.BandChainMeets64D12
#print axioms PousProofs.BandMultiMeets14
#print axioms PousProofs.BandMultiMeets14K106
#print axioms PousProofs.BandMultiMeetsFamily
#print axioms PousProofs.ChainDenseMeets64
#print axioms PousProofs.BandPubMeets64
#print axioms PousProofs.BandPubMeets14
#print axioms PousProofs.BandPubMeets14Global
#print axioms PousProofs.ChainDenseMeets111
-- the abstract model `m1p` at P2's parameters, no hypothesis: the Meets rows, the slack rows and the error split
#print axioms PousProofs.P2MeetsM1pB19Uncond
#print axioms PousProofs.P2MeetsM1pFamilyUncond
#print axioms PousProofs.P2MeetsM1pW8192Uncond
#print axioms PousProofs.P2SlackB19Uncond
#print axioms PousProofs.P2SlackFamilyUncond
#print axioms PousProofs.P2ErrorSplitB19Uncond
#print axioms PousP2p.not_meets_m1p_b19_k85
#print axioms PousMargin.not_slack_raw
-- P2 in the ring cost model (`Pous/P2/Cost.lean`): `MeetsOn` at `L = Q` gives `Meets`, the non-vacuity pins
-- (honest storage is accepted; recomputing the roots breaks `MeetsOn`), and dropping low bits breaks `k = 88`
#print axioms PousP2Ring.meets_of_meetsOn
#print axioms PousP2Ring.complete
#print axioms PousP2Ring.tight
#print axioms PousP2Ring.tight_p16448
#print axioms PousP2Ring.not_meetsOn_b19_k88
