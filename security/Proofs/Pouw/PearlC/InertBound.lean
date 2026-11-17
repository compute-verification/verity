module

public import Proofs.Pouw.Fp8Atom.Fp32Proofs
public import Proofs.Pouw.PearlC.FreezeP
public import Proofs.Pouw.PearlC.InertBound.IntForm
public import Proofs.Pouw.PearlC.InertBound.SaltDeadStep
public import Proofs.Pouw.PearlC.InertBound.SaltDead
import Mathlib.Algebra.Order.Floor.Semifield
import Mathlib.Tactic.Positivity.Finset
public import Proofs.Pouw.PearlC.InertBound.Pins

/-! The cap circuit's Lean pieces: its integer tests and their 32-bit words, `_inert_bound_constants` at pinned shapes,
and the salt-dead two-ends lemma. -/

@[expose] public section
