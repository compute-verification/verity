module

public import Proofs.Properties.Verity.SampledTwo.Projection
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.Sampling.KS

/-!
# The coupling at the two-stage form, assembled

`coupling_of` proves `CouplingOf`: the projection's bound and the majority's give `CouplingTargetTwo` with the two
bounds' sum as its `terms`. At each commitment and strategy whose claim fails on the vocabulary's transcript, the
statement is the one the coupling is given and the prover is the frame's, on the tapes of the period's play.

1. The projection's bound puts the sampled verdict under the frame's run accepting, plus `bound`.
2. The run's accept over the tape and the run's coins is its accept over those and the extractor's coins.
3. That splits on whether the extracted witness is in the session's relation: the half outside it is `PrBad`, and the
   half inside it the majority's piece bounds by `bind`, where the vocabulary's transcript is the one the claim fails
   on.

The scope clause and the published `δ` clause are the coupling's hypotheses at the same statement and prover.
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims
open scoped ENNReal

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **The coupling from the projection and the majority**: `CouplingOf` holds at every vocabulary, composite,
session of record and committed files. -/
theorem coupling_of (V : Vocab d α) (k : Verity.Registration.Composite) (A : Live₂) (θ : ByteArray)
    (F : Array ByteArray) : CouplingOf V k A θ F := by
  intro x Fr X bound bind hScope hProj hMaj hX hdelta c σ hI
  refine ⟨x c σ, TapeOf V θ F c σ, inferInstance, inferInstance, (Fr c σ).P, hScope c σ, ?_, hdelta c σ⟩
  have hacc : Claims.Pr (fun s : TapeOf V θ F c σ × Coins A (x c σ) =>
        (Claims.run FlockDeploy.sha512 (Fr c σ).P A.L.V (x c σ) s.1 s.2).out = .acc) ≤
      PrBad FlockDeploy.sha512 A.L.V A.R A.L.E (x c σ) (Fr c σ).P +
        Claims.Pr (fun s : TapeOf V θ F c σ × Coins A (x c σ) × A.L.E.C (x c σ) =>
          (Claims.run FlockDeploy.sha512 (Fr c σ).P A.L.V (x c σ) s.1 s.2.1).out = .acc ∧
            Claims.Rel.ext A.R (x c σ) (A.L.E.eval FlockDeploy.sha512 (Fr c σ).P (x c σ) s.1 s.2.1 s.2.2).1) := by
    refine (Claims.Pr_drop (α := TapeOf V θ F c σ) (β := Coins A (x c σ)) (γ := A.L.E.C (x c σ))
      (fun ω r => (Claims.run FlockDeploy.sha512 (Fr c σ).P A.L.V (x c σ) ω r).out = .acc)).symm.le.trans ?_
    refine Claims.Sampling.Pr_or_le
      (q := fun s : TapeOf V θ F c σ × Coins A (x c σ) × A.L.E.C (x c σ) =>
        (Claims.run FlockDeploy.sha512 (Fr c σ).P A.L.V (x c σ) s.1 s.2.1).out = .acc ∧
          ¬ Claims.Rel.ext A.R (x c σ) (A.L.E.eval FlockDeploy.sha512 (Fr c σ).P (x c σ) s.1 s.2.1 s.2.2).1)
      (r := fun s : TapeOf V θ F c σ × Coins A (x c σ) × A.L.E.C (x c σ) =>
        (Claims.run FlockDeploy.sha512 (Fr c σ).P A.L.V (x c σ) s.1 s.2.1).out = .acc ∧
          Claims.Rel.ext A.R (x c σ) (A.L.E.eval FlockDeploy.sha512 (Fr c σ).P (x c σ) s.1 s.2.1 s.2.2).1)
      (fun s hs => ?_) le_rfl le_rfl
    by_cases he : Claims.Rel.ext A.R (x c σ) (A.L.E.eval FlockDeploy.sha512 (Fr c σ).P (x c σ) s.1 s.2.1 s.2.2).1
    · exact Or.inr ⟨hs, he⟩
    · exact Or.inl ⟨hs, he⟩
  have hm := hMaj c σ (hX c σ) (by rw [← hX c σ]; exact hI)
  calc Verity.Game.prob (fun o => V.Verdict k .sampled o) (V.game θ F c) σ
      ≤ ENNReal.ofReal (Claims.Pr fun s : TapeOf V θ F c σ × Coins A (x c σ) =>
          (Claims.run FlockDeploy.sha512 (Fr c σ).P A.L.V (x c σ) s.1 s.2).out = .acc) + bound c σ :=
        hProj c σ
    _ ≤ (ENNReal.ofReal (PrBad FlockDeploy.sha512 A.L.V A.R A.L.E (x c σ) (Fr c σ).P) + bind c σ) + bound c σ :=
        add_le_add ((ENNReal.ofReal_le_ofReal hacc).trans (ENNReal.ofReal_add_le.trans (add_le_add le_rfl hm)))
          le_rfl
    _ = ENNReal.ofReal (PrBad FlockDeploy.sha512 A.L.V A.R A.L.E (x c σ) (Fr c σ).P) + (bound c σ + bind c σ) := by
        ring

end Verity.SampledTwo
