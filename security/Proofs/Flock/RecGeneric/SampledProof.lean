module

public import Proofs.Flock.RecGeneric.Sampled
public import Proofs.Flock.SampledGeneric.JointTransfers
public import Proofs.Flock.Claims.Bridge.Transport

/-!
# B1's proof

`jointOf` is judged (`jointOf_judged`: its verdict is the judged run's, and `E`'s coins are unread there) and knowledge
sound at `err` (`jointOf_knowledgeSound`: acceptance with a failed extraction is `PrBad`'s event at the claims prover
the strategy plays). With these, `GenericSampledKSJoint` gives B1 (`Flock.SecurityProofs.GenericSampledKSClaims`). The
recursive audit is B1 at `Comp.err` once L1 holds (`recursiveSampledKS_of_L1`).
-/

@[expose] public section

namespace Claims.RecGeneric

open Claims FlockSoundness FlockSoundness.Game
open scoped ENNReal

variable {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
variable {n : ℕ} {Reg Dh : Type} {Hc : List UInt8 → Dh}

theorem rel_ext_iff (Rl : Rel W) (x : Msg) (e : Option W) : Rl.ext x e ↔ ∃ w, e = some w ∧ Rl x w := by
  cases e with
  | none => simp [Rel.ext]
  | some w => simp [Rel.ext]

theorem Ex_unit (f : Unit → ℝ) : Ex f = f () := by
  simp [Ex]

theorem ind_congr_iff {α β : Type} {p : α → Prop} {q : β → Prop} {a : α} {b : β} (h : p a ↔ q b) :
    FlockSoundness.Game.ind p a = FlockSoundness.Game.ind q b := by
  by_cases hp : p a
  · simp [FlockSoundness.Game.ind, hp, h.1 hp]
  · have hq : ¬ q b := fun hq => hp (h.2 hq)
    simp [FlockSoundness.Game.ind, hp, hq]

/-- **`jointOf`'s verdict is the judged session's**: it accepts as often as the session, `E`'s coins unread. -/
theorem jointOf_judged (H : Msg → Msg) (V : Verifier Coin) (Rl : Rel W) (E : Claims.Extractor Coin W V)
    (stmt : Finset (Fin n) → Reg → Msg) (B : FlockSoundness.SampledGeneric.Binding Reg Hc)
    (reads : Fin n → Finset B.Pos) (opens : Finset (Fin n) → Reg → Fin n → W → B.Pos → B.Opening) :
    (jointOf H V Rl E stmt B reads opens).Judged := by
  intro S R τ
  simp only [jointOf, proofSystemOf, sessionOf, prob, proverOf]
  refine Finset.sum_congr rfl fun r₀ _ => ?_
  congr 1
  by_cases h : (run H τ.1 V (stmt S R) () r₀).out = .acc
  · simp only [h, decide_true, ↓reduceIte, mul_one, Finset.sum_const, Finset.card_univ, nsmul_eq_mul]
    exact ENNReal.mul_inv_cancel (by exact_mod_cast Fintype.card_ne_zero) (ENNReal.natCast_ne_top _)
  · simp only [h, decide_false, Bool.false_eq_true, ↓reduceIte, mul_zero, Finset.sum_const_zero]

/-- **`jointOf` is knowledge sound at `err`**: at a sampled proof block, acceptance with a failed extraction is
`PrBad`'s event at the claims prover the strategy plays, so its probability is at most `err` there. -/
theorem jointOf_knowledgeSound (H : Msg → Msg) (V : Verifier Coin) (Rl : Rel W) (E : Claims.Extractor Coin W V)
    (stmt : Finset (Fin n) → Reg → Msg) (B : FlockSoundness.SampledGeneric.Binding Reg Hc)
    (reads : Fin n → Finset B.Pos) (opens : Finset (Fin n) → Reg → Fin n → W → B.Pos → B.Opening)
    (err : Msg → Prover Unit → ℝ)
    (hks : ∀ S R (P : Prover Unit), PrBad H V Rl E (stmt S R) P ≤ err (stmt S R) P) :
    (jointOf H V Rl E stmt B reads opens).KnowledgeSound
      (fun S R τ _ => ENNReal.ofReal (err (stmt S R) (proverOf τ))) := by
  intro S R τ u hu
  rw [← ENNReal.ofReal_toReal (ne_top_of_le_ne_top ENNReal.one_ne_top (prob_le_one _ _ _))]
  refine ENNReal.ofReal_le_ofReal (le_trans (le_of_eq ?_) (hks S R (proverOf τ)))
  rw [prob_toReal]
  unfold PrBad
  rw [Claims.Bridge.Pr_eq_Ex, Claims.Bridge.Ex_prod, Ex_unit, Claims.Bridge.Ex_prod]
  simp only [jointOf, proofSystemOf, expect, Ex]
  congr 1
  refine Finset.sum_congr rfl fun r₀ _ => ?_
  congr 1
  refine Finset.sum_congr rfl fun c _ => ind_congr_iff ?_
  simp only [FlockSoundness.SampledGeneric.JointExtractor.Fails, hu, ↓reduceIte, rel_ext_iff, proverOf,
    Option.some.injEq, exists_eq_left']
  exact and_congr ⟨of_decide_eq_true, decide_eq_true⟩ Iff.rfl

end Claims.RecGeneric

open Claims Claims.RecGeneric FlockSoundness FlockSoundness.Game in
/-- **B1** (`Flock.Guarantees.GenericSampledKSClaims`): `GenericSampledKSJoint` at `jointOf`, judged and knowledge
sound at `err`, its cost a count. -/
theorem Flock.SecurityProofs.GenericSampledKSClaims : Flock.Guarantees.GenericSampledKSClaims := by
  intro Coin W _ _ _ V Rl E H err n Reg Dh Hc L B Correct stmt reads opens M hks hS hM σ hCR
  exact Flock.SecurityProofs.GenericSampledKSJoint n Reg Dh Hc L (proofSystemOf H V stmt) B Correct
    (jointOf H V Rl E stmt B reads opens) (fun S R τ _ => ENNReal.ofReal (err (stmt S R) (proverOf τ))) M
    (jointOf_judged H V Rl E stmt B reads opens) hS (fun _ _ _ => Nat.cast_nonneg _)
    (jointOf_knowledgeSound H V Rl E stmt B reads opens err hks) hM σ hCR

namespace Claims.RecGeneric

/-- **The recursive audit through the sampled compiler, from L1**: L1 at every sampled statement bounds `PrBad` at
`Comp.err` on every claims prover, which is B1's hypothesis. -/
theorem recursiveSampledKS_of_L1 (hL1 : CompositionKS) : Flock.Guarantees.RecursiveSampledKS := by
  intro Coin W _ _ _ H I Sv K mI Mo n Reg Dh Hc L B Correct stmt reads opens M hWF hKS hLC hBd hAt hS hM σ hCR
  exact Flock.SecurityProofs.GenericSampledKSClaims Coin W K.V (K.R I) (K.E I Sv) H
    (fun x P => K.err I Sv H P x mI Mo) n Reg Dh Hc L B Correct stmt reads opens M
    (fun S R P => hL1 Coin W H I Sv K hWF hKS hLC hBd (stmt S R) (hAt S R).1 (hAt S R).2.1 (hAt S R).2.2.1
      (hAt S R).2.2.2 Unit P mI Mo)
    hS hM σ hCR

end Claims.RecGeneric
