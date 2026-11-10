module

public import Proofs.Flock.Sampled.Defs
public import Proofs.Flock.Sampled.Statements
public import Proofs.Flock.Sampled.NonVacuity
public import Proofs.Flock.Sampled.Pinned
public import Proofs.Flock.Sampled.Proofs
import Mathlib.Algebra.Order.Module.Field
import Mathlib.Basic.ENNReal.Real

/-! Sampling with prover-chosen batching, the partition hidden: its model, its statements, their non-vacuity and
their proofs. -/

@[expose] public section
