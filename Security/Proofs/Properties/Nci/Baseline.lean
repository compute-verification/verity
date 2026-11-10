module

public import Properties.Nci.Baseline
import Mathlib.Analysis.SpecialFunctions.Pow.NNReal
import Mathlib.Data.NNRat.Floor
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.Measurability.Init
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Proofs.Properties.Nci.BaselineSteps
import Proofs.Properties.Nci.DescAgree

/-!
# θ's provenance check's property, proved

`BaselineProvenance` (`Properties.Nci.Baseline`) of `NciCheck.Baseline.evaluate`, from its steps
(`Proofs.Properties.Nci.BaselineSteps`) and, for each θ entry naming a node, the policy check's θ fold
(`DescAgree.nodeThetaFold_ok`), which `NciCheck.Baseline.evaluate` runs first.
-/

@[expose] public section

namespace NciCheck.Properties.SecurityProofs

open Lean (Json)

theorem BaselineProvenance : Properties.BaselineProvenance := by
  intro j tr out h
  refine BaselineSteps.provenanceHeld_of j tr out h fun i h1 t ht => ?_
  obtain ⟨hdesc, -⟩ := BaselineSteps.evaluate_ok j tr out h
  obtain ⟨i', P, reads, mask, own, h1', -, -, -, -, -, -, hfold, -⟩ := DescAgree.descEvaluate_ok j _ hdesc
  rw [h1, Except.ok.injEq] at h1'
  subst h1'
  obtain ⟨k, hk, -⟩ := (DescAgree.nodeThetaFold_ok P i.theta mask hfold).2.1 t ht
  exact ⟨k, hk⟩

end NciCheck.Properties.SecurityProofs
