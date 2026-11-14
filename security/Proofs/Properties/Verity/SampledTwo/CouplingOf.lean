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

/-- **The coupling at one commitment and strategy, from the projection and the majority**: `CouplingTwoAt` at
the two bounds' sum, at the frame's prover. -/
theorem coupling_at (V : Vocab d α) (k : Verity.Registration.Composite) (A : Live₂) (θ : ByteArray)
    (F : Array ByteArray) {c : ByteArray} {σ : Verity.Game.Strategy (V.game θ F c)} (x : Msg)
    (Fr : Frame V A x θ F c σ) (tr : Verity.Circuit.Trace (V.circuitOf θ F)) (bound bind : ℝ≥0∞)
    (hScope : A.L.Scope x ∨ Verity.Game.prob (fun o => V.Verdict k .sampled o) (V.game θ F c) σ = 0)
    (hProj : ProjectionBound V k A x θ F c σ Fr bound) (hMaj : MajorityBound V A x θ F c σ Fr tr bind)
    (hX : V.extract θ F c σ = tr)
    (hdelta : V.finderCR .sampled θ F c σ → V.premises .sampled θ F c σ →
      ENNReal.ofReal (A.errAt x Fr.P) + (bound + bind) ≤ V.delta θ .sampled (V.cost θ F c σ))
    (hI : ¬ Accounting.IntegrityHolds (V.partsOf θ F) (V.Opens θ F c) (V.extract θ F c σ) (V.certOf θ).ε) :
    CouplingTwoAt V k A θ F c σ (bound + bind) := by
  rcases hScope with hs | h0
  swap
  · exact .inl h0
  refine .inr ⟨x, TapeOf V θ F c σ, inferInstance, inferInstance, Fr.P, hs, ?_, hdelta⟩
  have hacc : Claims.Pr (fun s : TapeOf V θ F c σ × Coins A x =>
        (Claims.run FlockDeploy.sha512 Fr.P A.L.V x s.1 s.2).out = .acc) ≤
      PrBad FlockDeploy.sha512 A.L.V A.R A.L.E x Fr.P +
        Claims.Pr (fun s : TapeOf V θ F c σ × Coins A x × A.L.E.C x =>
          (Claims.run FlockDeploy.sha512 Fr.P A.L.V x s.1 s.2.1).out = .acc ∧
            Claims.Rel.ext A.R x (A.L.E.eval FlockDeploy.sha512 Fr.P x s.1 s.2.1 s.2.2).1) := by
    refine (Claims.Pr_drop (α := TapeOf V θ F c σ) (β := Coins A x) (γ := A.L.E.C x)
      (fun ω r => (Claims.run FlockDeploy.sha512 Fr.P A.L.V x ω r).out = .acc)).symm.le.trans ?_
    refine Claims.Sampling.Pr_or_le
      (q := fun s : TapeOf V θ F c σ × Coins A x × A.L.E.C x =>
        (Claims.run FlockDeploy.sha512 Fr.P A.L.V x s.1 s.2.1).out = .acc ∧
          ¬ Claims.Rel.ext A.R x (A.L.E.eval FlockDeploy.sha512 Fr.P x s.1 s.2.1 s.2.2).1)
      (r := fun s : TapeOf V θ F c σ × Coins A x × A.L.E.C x =>
        (Claims.run FlockDeploy.sha512 Fr.P A.L.V x s.1 s.2.1).out = .acc ∧
          Claims.Rel.ext A.R x (A.L.E.eval FlockDeploy.sha512 Fr.P x s.1 s.2.1 s.2.2).1)
      (fun s hs => ?_) le_rfl le_rfl
    by_cases he : Claims.Rel.ext A.R x (A.L.E.eval FlockDeploy.sha512 Fr.P x s.1 s.2.1 s.2.2).1
    · exact Or.inr ⟨hs, he⟩
    · exact Or.inl ⟨hs, he⟩
  have hm := hMaj hX (by rw [← hX]; exact hI)
  calc Verity.Game.prob (fun o => V.Verdict k .sampled o) (V.game θ F c) σ
      ≤ ENNReal.ofReal (Claims.Pr fun s : TapeOf V θ F c σ × Coins A x =>
          (Claims.run FlockDeploy.sha512 Fr.P A.L.V x s.1 s.2).out = .acc) + bound :=
        hProj
    _ ≤ (ENNReal.ofReal (PrBad FlockDeploy.sha512 A.L.V A.R A.L.E x Fr.P) + bind) + bound :=
        add_le_add ((ENNReal.ofReal_le_ofReal hacc).trans (ENNReal.ofReal_add_le.trans (add_le_add le_rfl hm)))
          le_rfl
    _ = ENNReal.ofReal (PrBad FlockDeploy.sha512 A.L.V A.R A.L.E x Fr.P) + (bound + bind) := by
        ring

/-- **The coupling from the projection and the majority**: `CouplingOf` holds at every vocabulary, composite,
session of record and committed files. -/
theorem coupling_of (V : Vocab d α) (k : Verity.Registration.Composite) (A : Live₂) (θ : ByteArray)
    (F : Array ByteArray) : CouplingOf V k A θ F := by
  intro x Fr tr bound bind hScope hProj hMaj hX hdelta c σ hI
  exact coupling_at V k A θ F (x c σ) (Fr c σ) (tr c σ) (bound c σ) (bind c σ) (hScope c σ) (hProj c σ) (hMaj c σ)
    (hX c σ) (hdelta c σ) hI

end Verity.SampledTwo
