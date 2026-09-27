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
