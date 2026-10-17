import Mathlib
import Definitions.Pouw.Neekon
import Models.Pouw.Assumptions.Neekon
import Models.Pouw.Guarantees.Neekon

/-!
# Dither's Theorem B in Neekon's two-stage definition: the proof

`Pouw.SecurityProofs.Neekon.TheoremB : Pouw.Guarantees.Neekon.TheoremB`. The general step is
`neekonSecure_of_floor`: a complete `Solve`, a time floor `F` on every adversary accepted with probability at least
`δ`, and `time(Solve) ≤ (1 + ε) F` give `NeekonSecure ε δ η` with `M' = Solve`. For Dither, acceptance is the event that
`M₂` outputs the transcript (`dither_acceptance`), so H is that floor at `F = (1 − εH) tHonest`.

`transcript_eq_dPrefix` reads Dither's transcript as Theorem K's targets: the re-encoded design's prefix words
(`dPrefix`) at shape `(n, 32, 7, K, m)` and weights `wRe`, with `B − 64` and `A` committed.
-/

namespace Pouw.SecurityProofs.Neekon

open Pouw.RelativeSecurity Pouw.Neekon Pouw.Neekon.Assumptions Finset

variable {Mc : Machine}

/-- An event that always holds has probability 1 on a nonempty finite type. -/
theorem prob_of_forall {Ω : Type*} [Fintype Ω] (h₀ : Nonempty Ω) {p : Ω → Prop} (h : ∀ ω, p ω) : prob p = 1 := by
  classical
  unfold prob
  rw [Finset.filter_true_of_mem fun ω _ => h ω, Finset.card_univ]
  exact div_self (Nat.cast_ne_zero.mpr Fintype.card_ne_zero)

/-- A weaker event has at least the probability. -/
theorem prob_le_of_imp {Ω : Type*} [Fintype Ω] {p q : Ω → Prop} (h : ∀ ω, p ω → q ω) : prob p ≤ prob q := by
  classical
  unfold prob
  exact div_le_div_of_nonneg_right
    (by exact_mod_cast Finset.card_le_card (Finset.monotone_filter_right _ fun a _ => h a)) (Nat.cast_nonneg _)

/-- **A per-run floor gives a time floor**: if the runs of `M₂` that are good (for H, correct) and charged below `F`
(`max(|M₂|, cost)`, the form K's floors bound) have probability at most `κ < δ`, while its good runs have probability
at least `δ`, then `time(M₂) ≥ F`. With `F = (1 − εH) tHonest` this is H's conclusion for `M₂`, so a one-draw floor of
K's kind for the runs in its scope gives H there. -/
theorem le_time_of_runFloor {I O : Type} [Fintype I] (M₂ : Mc.Alg I O) {good : I → Prop} {F κ δ : ℝ}
    (hfloor : prob (fun ω => good ω ∧ Mc.charge M₂ ω < F) ≤ κ) (hκδ : κ < δ) (hgood : δ ≤ prob good) :
    F ≤ Mc.time M₂ := by
  by_contra hlt
  have hcharge : ∀ ω, Mc.charge M₂ ω < F := fun ω =>
    (max_le (Mc.size_le_time M₂) (Mc.cost_le_time M₂ ω)).trans_lt (not_le.mp hlt)
  have := prob_le_of_imp (p := good) (q := fun ω => good ω ∧ Mc.charge M₂ ω < F) fun ω hω => ⟨hω, hcharge ω⟩
  linarith

/-- **Security from a time floor**: if `Solve` is complete, every adversary accepted with probability at least `δ` has
`time(M) ≥ F`, and `Solve` costs at most `(1 + ε) F`, then the protocol is `NeekonSecure ε δ η` with `M' = Solve`. -/
theorem neekonSecure_of_floor (P : PoUW Mc) {ε δ η F : ℝ} (hc : Complete P η)
    (hfloor : ∀ (Ω : Type) [Fintype Ω] (M : Adversary P Ω), δ ≤ M.acceptance → F ≤ M.time)
    (hS : Mc.time P.solve ≤ (1 + ε) * F) (hε : 0 ≤ 1 + ε) : NeekonSecure P ε δ η := by
  intro Ω _ M hM
  exact ⟨P.solve, hS.trans (mul_le_mul_of_nonneg_left (hfloor Ω M hM) hε), hc⟩

section Dither

variable {m K n : ℕ} {solve : Mc.Alg (DSeed m K n × DInst m K n) (DTrans m K n × Matrix (Fin m) (Fin n) Z32)}

/-- Dither's acceptance is the event that `M₂` outputs the transcript of the instance `M₁` fixed. -/
theorem dither_acceptance {Ω : Type} [Fintype Ω] (M : Adversary (dither Mc m K n solve) Ω) :
    M.acceptance = prob fun ω : DSeed m K n × Ω => Mc.eval M.M₂ ω = transcript ω.1 M.inst := by
  unfold Adversary.acceptance
  congr 1
  ext ω
  simp [dither]

/-- Dither is complete at `η = 0` when `Solve` computes the transcript and the edited product. -/
theorem dither_complete (hsolve : ∀ σ x, Mc.eval solve (σ, x) = (transcript σ x, editA σ x * x.2)) :
    Complete (dither Mc m K n solve) 0 := by
  intro x
  rw [sub_zero]
  refine (prob_of_forall (Ω := (dither Mc m K n solve).Seed) ⟨(∅ : DSeed m K n)⟩ fun σ => ?_).symm.le
  simp only [dither, decide_eq_true_eq]
  exact ⟨congrArg Prod.snd (hsolve σ x), congrArg Prod.fst (hsolve σ x)⟩

end Dither

theorem TheoremB : Pouw.Guarantees.Neekon.TheoremB := by
  intro Mc m K n solve εH δ tHonest tStrip ht hεH hsolve htime hH
  have h1 : 0 < 1 - εH := sub_pos.mpr hεH
  have hstrip : 0 ≤ 1 + tStrip / tHonest := by
    rw [show 1 + tStrip / tHonest = (tHonest + tStrip) / tHonest by field_simp]
    exact div_nonneg ((Mc.time_nonneg solve).trans htime) ht.le
  refine neekonSecure_of_floor _ (dither_complete hsolve) (F := (1 - εH) * tHonest) ?_ ?_ ?_
  · intro Ω _ M hM
    rw [dither_acceptance] at hM
    exact hH M.inst Ω M.M₂ hM
  · rw [add_sub_cancel]
    calc Mc.time (dither Mc m K n solve).solve = Mc.time solve := rfl
      _ ≤ tHonest + tStrip := htime
      _ = (1 + tStrip / tHonest) / (1 - εH) * ((1 - εH) * tHonest) := by field_simp
  · rw [add_sub_cancel]
    exact div_nonneg hstrip h1.le

section Design

variable {m K n : ℕ}

/-- The design's committed coin-side constants at Dither's instance: `⌊x/s⌋_{(ℓ, o), i} = A(i, (ℓ, o))`. -/
def vqOf (x : DInst m K n) : Fin K → Fin 32 → Fin m → Z32 := fun ℓ o i => x.1 i (ℓ, o)

/-- The design's committed U-side constants at Dither's instance: `pc(j, ℓ, o) = B((ℓ, o), j) − 64`. -/
def pcOf (x : DInst m K n) : Fin n → Fin K → Fin 32 → Z32 := fun j ℓ o => x.2 (ℓ, o) j - 64

theorem dX_eval (vq : Fin K → Fin 32 → Fin m → Z32) (ℓ : Fin K) (o : Fin 32) (i : Fin m) (σ : DSeed m K n) :
    (dX (n := n) (np := 7) vq ℓ o i).eval (bitInd σ) = vq ℓ o i + bitInd σ (.inr (ℓ, o, i)) := by
  simp [dX, BitAff.eval, Fintype.sum_sum_type]

theorem dP_eval (pc : Fin n → Fin K → Fin 32 → Z32) (j : Fin n) (ℓ : Fin K) (o : Fin 32) (σ : DSeed m K n) :
    (dP (n' := m) pc wRe j ℓ o).eval (bitInd σ) = pc j ℓ o + ∑ p, wRe p * bitInd σ (.inl (j, o, p)) := by
  simp [dP, BitAff.eval, Fintype.sum_sum_type, Fintype.sum_prod_type, ite_and]

/-- **Dither's transcript is Theorem K's targets' table**: with σ read as the re-encoded design's bits, `T(i, j, ℓ)` is
the committed prefix word `dPrefix` of cell `(j, i)` through block `ℓ`, on the design of shape `(n, 32, 7, K, m)` at
the weights `wRe`, with `B − 64` (`pcOf`) and `A` (`vqOf`) committed: `B̃ᵀ` is the design's `P̃` (U on its side, `U`'s
column `o` serving every block) and `Ãᵀ` its `X` (the coins `b(ℓ, o, i)` are `R`). -/
theorem transcript_eq_dPrefix (σ : DSeed m K n) (x : DInst m K n) (i : Fin m) (j : Fin n) (ℓ : Fin K) :
    transcript σ x i j ℓ = dPrefix (pcOf x) (vqOf x) wRe j i ℓ.val (bitInd σ) := by
  unfold transcript dPrefix dIncr
  refine Finset.sum_congr rfl fun ℓ' _ => ?_
  simp only [Fin.le_iff_val_le_val]
  split_ifs
  · refine Finset.sum_congr rfl fun o _ => ?_
    rw [dP_eval, dX_eval]
    simp only [editA, editB, noiseR, noiseS, noiseU, pcOf, vqOf, Matrix.add_apply, Matrix.of_apply]
    ring
  · rfl

end Design

end Pouw.SecurityProofs.Neekon
