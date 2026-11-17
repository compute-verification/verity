module

public import Proofs.Flock.Claims.RootFindCorrect
public import Proofs.Flock.Claims.RootFindCost
import Mathlib.Algebra.Order.Field.Basic
import Mathlib.Data.Nat.Totient
import Mathlib.Data.Sym.Sym2.Init
import Mathlib.Tactic.ContinuousFunctionalCalculus
import Mathlib.Tactic.NormNum.GCD
import Mathlib.Tactic.Positivity.Finset

/-! A deterministic root finder over a finite field of characteristic 2, Berlekamp's trace method, with its operation
count. -/

@[expose] public section
