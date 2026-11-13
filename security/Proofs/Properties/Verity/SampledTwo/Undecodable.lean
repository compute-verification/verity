module

public import Proofs.Properties.Verity.SampledTwo.WiredScope

/-!
# The coupling at a commitment that does not decode

At a commitment `c` whose bytes carry no statement (`stmtOf c = none`) no `JoinsAt` holds, since its `stmtX` fails;
`JoinsAt`'s `wiring` and `caseCommits` don't read `c`, and from them alone the sampled verdict at `c` never accepts
(`prob_zero_of_stmtOf_none`): an accepted play's case carries a commitment whose record reads back, so its record
prints, and `stmtOf c` is that print.

`couplingTwo_of_stmtOf_none` is `CouplingTargetTwo`'s body at such a `c`, at any `terms`, from those two alone, and
`joinedCouplingOfLaw_decoded` is the two-stage leg's coupling with `CouplesAt` taken only where the commitment carries
a statement.
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims
open scoped ENNReal

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-- **The sampled verdict never accepts at a commitment that does not decode**, from the composite's wiring at `θ`
and the instance's cases carrying the play's commitment. -/
theorem prob_zero_of_stmtOf_none {V : Vocab d α} {k : Verity.Registration.Composite} {θ : ByteArray}
    {F : Array ByteArray} {c : ByteArray} {C : CaseOf}
    (hwire : ∀ o : Verity.Exchange, o.params = θ → V.Verdict k .sampled o → FormAccepts C o)
    (hcc : CaseCommits C) (hn : stmtOf c = none) (σ : Verity.Game.Strategy (V.game θ F c)) :
    Verity.Game.prob (fun o => V.Verdict k .sampled o) (V.game θ F c) σ = 0 := by
  refine prob_play_eq_zero (V.families θ) θ F c _ (fun o ho hv => ?_) σ
  obtain ⟨hθ, -, hc⟩ := ho
  obtain ⟨kk, hk, hAcc⟩ := hwire o hθ hv
  obtain ⟨cm, i, hcm, -, hrb, -⟩ := input_of_accepts hAcc
  have hkc : kk.commitment = c := (hcc o kk hk).trans hc
  obtain ⟨s, hs', -, -⟩ := VStar.Recursive.SampledTwo.readBack_ok hrb
  have hsx : stmtOf c = recordMsg cm.record := by
    rw [← hkc]
    unfold stmtOf
    simp only [hcm]
  rw [hsx] at hn
  unfold recordMsg at hn
  rw [hs'] at hn
  cases hn

/-- **`CouplingTargetTwo`'s body at a commitment that does not decode**, at any `terms`, from the composite's wiring
at `θ` and `CaseCommits`: the sampled verdict never accepts there. -/
theorem couplingTwo_of_stmtOf_none {V : Vocab d α} {k : Verity.Registration.Composite} {A : Live₂} {θ : ByteArray}
    {F : Array ByteArray} {c : ByteArray} {C : CaseOf} (t : ℝ≥0∞)
    (hwire : ∀ o : Verity.Exchange, o.params = θ → V.Verdict k .sampled o → FormAccepts C o)
    (hcc : CaseCommits C) (hn : stmtOf c = none) (σ : Verity.Game.Strategy (V.game θ F c)) :
    CouplingTwoAt V k A θ F c σ t :=
  .inl (prob_zero_of_stmtOf_none hwire hcc hn σ)

/-- **The two-stage leg's coupling, with `CouplesAt` only where the commitment decodes**: `joinedCouplingAt` at each
commitment that carries a statement, and `couplingTwo_of_stmtOf_none` at each that carries none, from the composite's
wiring at `θ` and `CaseCommits`, which read no commitment. -/
theorem joinedCouplingOfLaw_decoded (V : Vocab d α) (k : Verity.Registration.Composite) (A : Live₂) (θ : ByteArray)
    (F : Array ByteArray) (x : (c : ByteArray) → Verity.Game.Strategy (V.game θ F c) → Msg) (C : CaseOf)
    (Ms : SessionReading) (tab : ℕ → String) (ℓ₁ ℓ₂ : String) (Dg : Verity.Exchange → ℕ → Option Msg) (q : ℕ)
    (P : Sessions.Deployment) (fs₁ fs₂ : ByteArray → List (String × ℕ)) (cut : List Verity.Release → ℕ)
    (P' : Verity.Exchange → ℕ → Claims.Prover Unit)
    (s : (c : ByteArray) → Verity.Game.Strategy (V.game θ F c) → Claims.Sampling.AuditStatement)
    (Pt : (c : ByteArray) → Verity.Game.Strategy (V.game θ F c) → Claims.Sampling.Partition)
    (dd : (c : ByteArray) → Verity.Game.Strategy (V.game θ F c) → Claims.Sampling.Design.Two)
    (readX : (c : ByteArray) → Verity.Game.Strategy (V.game θ F c) →
      Claims.Sampling.Wit → Verity.Circuit.Trace (V.circuitOf θ F))
    (openOf : (c : ByteArray) → Verity.Game.Strategy (V.game θ F c) →
      Claims.Sampling.Wit → Fin (V.circuitOf θ F).N → Option Hash.RowLeaf.Opening)
    (dColl : (c : ByteArray) → Verity.Game.Strategy (V.game θ F c) → ℕ)
    (hHolds : CoinMap.CoinMapHolds (d := d) (α := α))
    (hwire : ∀ o : Verity.Exchange, o.params = θ → V.Verdict k .sampled o → FormAccepts C o)
    (hcc : CaseCommits C)
    (h : ∀ c σ, (∃ x', stmtOf c = some x') →
      CouplesAt V k A (x c σ) θ F c σ C Ms tab ℓ₁ ℓ₂ Dg q P fs₁ fs₂ cut P' (s c σ) (Pt c σ) (dd c σ)
        (readX c σ) (openOf c σ) (dColl c σ)) :
    CouplingTargetTwo V k A θ F fun c σ =>
      joinBound V A (x c σ) θ F c σ C Ms tab ℓ₁ ℓ₂ Dg q P fs₁ fs₂ cut +
        bindBound V A (x c σ) θ F c σ
          (joinFrame V A (x c σ) θ F c σ C (joinSessions A C Ms tab ℓ₁ ℓ₂) (joinLocate A tab ℓ₁ ℓ₂) Dg)
          (openOf c σ) (dColl c σ) := by
  intro c σ hI
  cases hs : stmtOf c with
  | none => exact couplingTwo_of_stmtOf_none _ hwire hcc hs σ
  | some x' =>
    exact joinedCouplingAt V k A θ F (x c σ) C Ms tab ℓ₁ ℓ₂ Dg q P fs₁ fs₂ cut P' (s c σ) (Pt c σ) (dd c σ)
      (readX c σ) (openOf c σ) (dColl c σ) hHolds (h c σ ⟨x', hs⟩) hI

end Verity.SampledTwo
