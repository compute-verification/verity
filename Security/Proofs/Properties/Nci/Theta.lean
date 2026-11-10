import Proofs.Properties.Nci.ThetaSteps
import Proofs.Properties.Nci.DescAgree

/-!
# θ's provenance check's property, proved

`ThetaProvenance` (`Properties.Nci.Theta`) of `NciCheck.Theta.evaluate`, from its steps
(`Proofs.Properties.Nci.ThetaSteps`) and, for each θ entry naming a node, the policy check's θ fold
(`DescAgree.nodeThetaFold_ok`), which `NciCheck.Theta.evaluate` runs first.
-/

namespace NciCheck.Properties.SecurityProofs

open Lean (Json)

theorem ThetaProvenance : Properties.ThetaProvenance := by
  intro j tr out h
  refine ThetaSteps.provenanceHeld_of j tr out h fun i h1 t ht => ?_
  obtain ⟨hdesc, -⟩ := ThetaSteps.evaluate_ok j tr out h
  obtain ⟨i', P, reads, mask, own, h1', -, -, -, -, -, -, hfold, -⟩ := DescAgree.descEvaluate_ok j _ hdesc
  rw [h1, Except.ok.injEq] at h1'
  subst h1'
  obtain ⟨k, hk, -⟩ := (DescAgree.nodeThetaFold_ok P i.theta mask hfold).2.1 t ht
  exact ⟨k, hk⟩

end NciCheck.Properties.SecurityProofs
