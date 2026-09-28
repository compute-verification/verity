import Definitions.NetworkCertifier
import Models.NetworkCertifier.Assumptions
import Models.NetworkCertifier.Guarantees
import Proofs.NetworkCertifier
open NetTiming

-- The 35 statements' proofs (22 of them `lean-audit.json`'s `guarantees`, 13 lemmas), then the 16 witnesses.
#print axioms SecurityProofs.CardFree
#print axioms SecurityProofs.CardLeftPacked
#print axioms SecurityProofs.FillInjective
#print axioms SecurityProofs.ExistsFillOfLeftPacked
#print axioms SecurityProofs.CardCountsEqLe
#print axioms SecurityProofs.CardCountsLeLe
#print axioms SecurityProofs.ReleasedSelf
#print axioms SecurityProofs.SumReleasedLe
#print axioms SecurityProofs.EncardFifoGridsLeOfAvailable
#print axioms SecurityProofs.EncardFifoGrids
#print axioms SecurityProofs.EncardLeProdOfBranching
#print axioms SecurityProofs.EncardAcceptedRunsLe
#print axioms SecurityProofs.EncardObsDecodableLe
#print axioms SecurityProofs.EncardDecodableLe
#print axioms SecurityProofs.NoFaultOfAccepted
#print axioms SecurityProofs.RecordedRowEqFill
#print axioms SecurityProofs.EncardAccSeqsLeAdvice
#print axioms SecurityProofs.EncardAccSeqsLeSync
#print axioms SecurityProofs.EncardAccSeqsLeFixedClock
#print axioms SecurityProofs.EncardAccSeqsLeBlocks
#print axioms SecurityProofs.EncardAccSeqsLeBlocksCounts
#print axioms SecurityProofs.EncardDecodableLeBlocks
#print axioms SecurityProofs.EncardAccSeqsLeConstantRate
#print axioms SecurityProofs.EncardAccSeqsLeConstantRateFixedClock
#print axioms SecurityProofs.ExpectedRowCausal
#print axioms SecurityProofs.EncardAccSeqsLeConstantRateWindows
#print axioms SecurityProofs.EncardDecodableLeConstantRate
#print axioms SecurityProofs.EncardIngressObsDecodableLe
#print axioms SecurityProofs.EncardIngressDecodableLe
#print axioms SecurityProofs.EncardIngressObsDecodableLeContents
#print axioms SecurityProofs.EncardIngressDecodableLeContents
#print axioms SecurityProofs.EncardAccSeqsLeConstantRatePerWindow
#print axioms SecurityProofs.EncardDecodableLeConstantRatePerWindow
#print axioms SecurityProofs.EncardIngressObsDecodableLePerWindow
#print axioms SecurityProofs.EncardIngressObsDecodableLeContentsPerWindow
#print axioms witnessCertifier_assumptions
#print axioms witnessCertifier_jitter_varies
#print axioms witnessProgram_conditions
#print axioms witnessProgram_timing_matters
#print axioms joint_witness
#print axioms joint_witness_bound
#print axioms zP_accepts_one
#print axioms zP_constOp_accepts_one
#print axioms ingress_witness
#print axioms ingress_witness_tight
#print axioms allocated_witness
#print axioms allocated_witness_tight
#print axioms toy_leftPacked
#print axioms constantRate_not_framesRead_counts
#print axioms constantRate_not_windowBlocks
#print axioms constantRate_joint_witness

-- The certifier program agrees with these definitions (`Code`, four guarantees in `lean-audit.json`).
#print axioms SecurityProofs.CodeExpectedRow
#print axioms SecurityProofs.CodeAcceptedConstantRate
#print axioms SecurityProofs.CodeStatusComplete
#print axioms SecurityProofs.CodeSyncSet

-- Proved and audited, not guarantees: no record in `lean-audit.json`, so not citable as guarantees.
#print axioms occ_fill
#print axioms leftPacked_occ_fill
#print axioms leftPacked_fifoGrid
