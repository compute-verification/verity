import NetTiming
open NetTiming

-- The 31 pins (`lean-audit.json`'s `pins`): the 22 guarantees' proofs, then the nine witnesses.
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
#print axioms witnessWarden_assumptions
#print axioms witnessWarden_jitter_varies
#print axioms witnessProgram_conditions
#print axioms witnessProgram_timing_matters
#print axioms joint_witness
#print axioms joint_witness_bound
#print axioms zP_accepts_one
#print axioms zP_constOp_accepts_one
#print axioms toy_leftPacked

-- Proved and audited, not pinned: no record in `lean-audit.json`, so not citable as pinned.
#print axioms occ_fill
#print axioms leftPacked_occ_fill
#print axioms leftPacked_fifoGrid
