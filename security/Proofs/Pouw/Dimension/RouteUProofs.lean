module

public import Definitions.Pouw.Assumptions.Dimension.RouteU
public import Definitions.Pouw.Assumptions.Dimension.Mixing
import Mathlib.LinearAlgebra.FiniteDimensional.Lemmas
import Mathlib.Tactic.Positivity.Finset
import Proofs.Pouw.Dimension.CausalProofs
import Proofs.Pouw.NCP.GameProofs
import Proofs.Pouw.NCP.RelationWitness
import Proofs.Pouw.NCP.RouteUProofs
import Proofs.Pouw.NCP.RouteUWordProofs

/-!
# Theorem D on route U's checked words: proofs

* `wordUSubV0`, `wordCUSubV0`: the shift is a function of `F₁` alone (`NCP.RouteUWordProofs.checkedUPos_sub`).
* `combUV0Iff`, `secondDifferenceU`, `indepModShift` (generic), `indepModU`.
* `pricedCountU`, `algBoundU`, `ttNCPAlgU`, `algBoundCausalU`: `count_le` on the route-U family, whose independence
  modulo `V₀` is `indepModShift` applied to the unbiased family's (`indep_sub`, `indep_subC`).
* `algBoundWrappedU`, `algBoundWrappedUUnsigned`: the wrap is the identity for a signed `R` and `k ≤ 2^15`, or a
  permutation matrix and `k ≤ 2^16` (`wrapped_eq_wordU`, `NCP.RouteUWordProofs.int7Int32UPos`).
* `algWitnessU`, `algSharpU`, `algWrappedWitnessU`: one priced instruction at 16 per route-U word.
-/

@[expose] public section

namespace Pouw.Dimension.RouteUProofs

open Finset Pouw.NCP Pouw.NCP.Relation Pouw.Dimension Pouw.Dimension.Assumptions Pouw.NCP.RouteUWordProofs

theorem f1Of_eq {N m n k : ℕ} {ω ω' : Noise N m n k} (h : ω.2 = ω'.2) : f1Of ω = f1Of ω' := by
  funext l j
  simp only [f1Of, h]

theorem shift_mem_fnF {N m n k : ℕ} (φ : Noise N m n k → ℚ) (s : Mat → ℤ)
    (hφ : ∀ ω, φ ω = (s (f1Of ω) : ℚ)) : φ ∈ fnF N m n k :=
  Submodule.subset_span fun ω ω' h => by rw [hφ, hφ, f1Of_eq h]

/-- **The shift is free.** -/
theorem wordUSubV0 : WordUSubV0 := by
  intro β k m n N R A B u p
  refine Submodule.mem_sup_right (shift_mem_fnF _ (fun F => shiftU β k m n R B F u p) fun ω => ?_)
  simp only [Pi.sub_apply, wordU, word]
  rw [← Int.cast_sub, checkedUPos_sub]

/-- **The shift is free, causal inputs.** -/
theorem wordCUSubV0 : WordCUSubV0 := by
  intro β k m n N R A B u p
  refine Submodule.mem_sup_right (shift_mem_fnF _ (fun F => shiftU β k m n R B F u p) fun ω => ?_)
  simp only [Pi.sub_apply, wordCU, wordC]
  rw [← Int.cast_sub, checkedUPos_sub]

theorem comb_sub_mem {G : Type} (V : Submodule ℚ (G → ℚ)) (U : Finset ℕ) (M : ℕ) (c : ℕ → ℕ → ℚ)
    (t t' : ℕ → ℕ → G → ℚ) (h : ∀ u p, t' u p - t u p ∈ V) :
    (∑ u ∈ U, ∑ p ∈ range M, c u p • t' u p) - (∑ u ∈ U, ∑ p ∈ range M, c u p • t u p) ∈ V := by
  rw [← sum_sub_distrib]
  refine V.sum_mem fun u _ => ?_
  rw [← sum_sub_distrib]
  exact V.sum_mem fun p _ => by rw [← smul_sub]; exact V.smul_mem _ (h u p)

/-- **Combinations agree modulo `V₀`.** -/
theorem combUV0Iff : CombUV0Iff := by
  intro β k m n N R A B U c
  have h := comb_sub_mem (V0 N m n k) U (nonFinal k m n) c (word k m n N R A B) (wordU β k m n N R A B)
    (wordUSubV0 β k m n N R A B)
  constructor
  · intro hU
    have := (V0 N m n k).sub_mem hU h
    rwa [sub_sub_cancel] at this
  · intro hw
    have := (V0 N m n k).add_mem h hw
    rwa [sub_add_cancel] at this

/-- **Independence modulo a subspace is invariant under shifts in it.** -/
theorem indepModShift : IndepModShift := by
  intro G ι _ V t t' hsh hind c hc
  refine hind c ?_
  have hd : ∑ i, c i • t' i - ∑ i, c i • t i ∈ V := by
    rw [← sum_sub_distrib]
    exact V.sum_mem fun i _ => by rw [← smul_sub]; exact V.smul_mem _ (hsh i)
  have := V.sub_mem hc hd
  rwa [sub_sub_cancel] at this

/-- **The four-point evaluation lemma on route U.** -/
theorem secondDifferenceU : SecondDifferenceU := fun β k m n N R A B U c hU hc =>
  secondDifference k m n N R A B U c hU ((combUV0Iff β k m n N R A B U c).1 hc)

/-- **Route U's words are independent modulo `V₀`.** -/
theorem indepModU : IndepModU := by
  intro β k m n N R A B U c hind hU hc
  exact indep_mod A B hind hU c ((combUV0Iff β k m n N R A B U c).1 hc)

/-- The non-final route-U words, indexed by `U ×ˢ range (nonFinal k m n)`, are linearly independent modulo `V₀`. -/
theorem indep_subU (β : ℤ) {k m n N : ℕ} {R : Mat} (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat)
    {U : Finset ℕ} (hind : CheckpointIndependent k 16 R) (hU : U ⊆ range N)
    (c : ↥(U ×ˢ range (nonFinal k m n)) → ℚ)
    (hc : ∑ x, c x • wordU β k m n N R A B x.1.1 x.1.2 ∈ V0 N m n k) : ∀ x, c x = 0 :=
  indepModShift (Noise N m n k) ↥(U ×ˢ range (nonFinal k m n)) (V0 N m n k)
    (fun x => word k m n N R A B x.1.1 x.1.2) (fun x => wordU β k m n N R A B x.1.1 x.1.2)
    (fun x => wordUSubV0 β k m n N R A B x.1.1 x.1.2) (indep_sub A B hind hU) c hc

theorem indep_subCU (β : ℤ) {k m n N : ℕ} {R : Mat} {A : Noise N m n k → ℕ → Mat} (hA : Causal A) (B : ℕ → Mat)
    {U : Finset ℕ} (hind : CheckpointIndependent k 16 R) (hU : U ⊆ range N)
    (c : ↥(U ×ˢ range (nonFinal k m n)) → ℚ)
    (hc : ∑ x, c x • wordCU β k m n N R A B x.1.1 x.1.2 ∈ V0 N m n k) : ∀ x, c x = 0 :=
  indepModShift (Noise N m n k) ↥(U ×ˢ range (nonFinal k m n)) (V0 N m n k)
    (fun x => wordC k m n N R A B x.1.1 x.1.2) (fun x => wordCU β k m n N R A B x.1.1 x.1.2)
    (fun x => wordCUSubV0 β k m n N R A B x.1.1 x.1.2) (indep_subC hA B hind hU) c hc

/-- **The priced count on route U.** -/
theorem pricedCountU : PricedCountU := by
  intro β k m n N R A B U L P hind hU hL hout
  have hcount := count_le P L
    (fun x : ↥(U ×ˢ range (nonFinal k m n)) => wordU β k m n N R A B x.1.1 x.1.2)
    (fun c hc => indep_subU β A B hind hU c (span1_le_V0 hL hc))
    (fun x => hout x.1.1 (mem_product.1 x.2).1 x.1.2 (mem_range.1 (mem_product.1 x.2).2))
  rw [Fintype.card_coe, card_product, card_range, nonFinal] at hcount
  calc U.card * (m * n) * (3 * k / 16 - 1) = U.card * ((3 * k / 16 - 1) * (m * n)) := by ring
    _ ≤ nPriced P := hcount

/-- **Theorem D on route U.** -/
theorem algBoundU : AlgBoundU := by
  intro β k m n N R A B U L P hind hU hL hH hout
  have := pricedCountU β k m n N R A B U L P hind hU hL hout
  have := h16_cost hH
  omega

/-- TT_NCP's event fails for every cost at Theorem D's bound, for `g₀ ≥ 16/(3k)` (`ttNCPAlg`'s arithmetic). -/
theorem not_event_of_bound {k m n : ℕ} {U : Finset ℕ} {C g₀ : ℝ} (h16 : 16 ∣ k) (hk : 0 < k)
    (hb : ((16 * (U.card * (m * n) * (3 * k / 16 - 1)) : ℕ) : ℝ) ≤ C) (hg : 16 / (3 * k : ℝ) ≤ g₀) (hg1 : g₀ < 1) :
    ¬ (C / (1 - g₀) < ∑ _u ∈ U, 3 * ((m * k * n : ℕ) : ℝ)) := by
  intro hev
  obtain ⟨k', rfl⟩ := h16
  have hk' : 1 ≤ k' := by omega
  have h3 : 3 * (16 * k') / 16 - 1 = 3 * k' - 1 := by omega
  rw [h3] at hb
  have hbR : (16 : ℝ) * (U.card * (m * n) * (3 * k' - 1)) ≤ C := by
    push_cast [Nat.cast_sub (by omega : 1 ≤ 3 * k')] at hb
    linarith
  rw [sum_const, nsmul_eq_mul] at hev
  have hpos : 0 < 1 - g₀ := by linarith
  rw [div_lt_iff₀ hpos] at hev
  push_cast at hev hg
  have h48 : 16 ≤ g₀ * (3 * (16 * (k' : ℝ))) := (div_le_iff₀ (by positivity)).1 hg
  have hX : (0 : ℝ) ≤ U.card * (m * n) := by positivity
  nlinarith [mul_le_mul_of_nonneg_left h48 hX]

/-- **TT_NCP in AlgM_16 on route U.** -/
theorem ttNCPAlgU : TTNCPAlgU := by
  intro β k m n N R A B U L P g₀ h16 hk hind hU hL hH hout hg hg1
  exact not_event_of_bound h16 hk (by exact_mod_cast algBoundU β k m n N R A B U L P hind hU hL hH hout) hg hg1

/-- **Theorem D with causal inputs on route U.** -/
theorem algBoundCausalU : AlgBoundCausalU := by
  intro β k m n N R A B U L P hind hA hU hL hH hout
  have hcount := count_le P L
    (fun x : ↥(U ×ˢ range (nonFinal k m n)) => wordCU β k m n N R A B x.1.1 x.1.2)
    (fun c hc => indep_subCU β hA B hind hU c (span1_le_V0 hL hc))
    (fun x => hout x.1.1 (mem_product.1 x.2).1 x.1.2 (mem_range.1 (mem_product.1 x.2).2))
  rw [Fintype.card_coe, card_product, card_range, nonFinal] at hcount
  have hc : U.card * (m * n) * (3 * k / 16 - 1) ≤ nPriced P := by
    calc U.card * (m * n) * (3 * k / 16 - 1) = U.card * ((3 * k / 16 - 1) * (m * n)) := by ring
      _ ≤ nPriced P := hcount
  have := h16_cost hH
  omega

/-- In the int32-exact range, the wrapped non-final route-U word is the word. -/
theorem wrapped_eq_wordU {β : ℤ} {k m n N : ℕ} {R : Mat} {A : (Fin k → Fin n → Fin 64) → ℕ → Mat} {B : ℕ → Mat}
    {U : Finset ℕ} (hβ1 : 127 ≤ β) (hβ2 : β ≤ 129) (hR : SignedPerm k R ∧ k ≤ 2 ^ 15 ∨ Unsigned k R ∧ k ≤ 2 ^ 16)
    (hA : ∀ F, ∀ u ∈ U, ∀ i, ∀ l < k, InInt7 (A F u i l)) (hB : ∀ u ∈ U, ∀ l < k, ∀ j, InInt7 (B u l j))
    {u p : ℕ} (hu : u ∈ U) (hp : p < nonFinal k m n) :
    (fun ω => (wrap32 (checkedUPos β k m n N R (A ω.2) B ω u p) : ℚ)) = wordU β k m n N R A B u p := by
  have hpos : p < positions k m n :=
    lt_of_lt_of_le hp (Nat.mul_le_mul_right _ (Nat.sub_le _ _))
  funext ω
  rw [Pouw.Proofs.wrap32_of_mem (int7Int32UPos β k m n N R (A ω.2) B ω u p hβ1 hβ2 hR (hA ω.2 u hu) (hB u hu) hpos)]
  rfl

/-- **Theorem D for route U's protocol words.** -/
theorem algBoundWrappedU : AlgBoundWrappedU := by
  intro β k m n N R A B U L P hβ1 hβ2 hind hR hk hU hA hB hL hH hout
  refine algBoundU β k m n N R A B U L P hind hU hL hH fun u hu p hp => ?_
  rw [← wrapped_eq_wordU hβ1 hβ2 (Or.inl ⟨hR, hk⟩) hA hB hu hp]
  exact hout u hu p hp

/-- **Theorem D for route U's protocol words, permutation-matrix `R`.** -/
theorem algBoundWrappedUUnsigned : AlgBoundWrappedUUnsigned := by
  intro β k m n N R A B U L P hβ1 hβ2 hind hR hk hU hA hB hL hH hout
  refine algBoundU β k m n N R A B U L P hind hU hL hH fun u hu p hp => ?_
  rw [← wrapped_eq_wordU hβ1 hβ2 (Or.inr ⟨hR, hk⟩) hA hB hu hp]
  exact hout u hu p hp

/-! ## The honest program on route U -/

/-- One priced instruction at price `c` per non-final route-U word of the units in `U`. -/
noncomputable def honestU (β : ℤ) (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat)
    (U : Finset ℕ) (c : ℕ) : List (Instr (Noise N m n k)) :=
  U.toList.flatMap fun u => (List.range (nonFinal k m n)).map fun p => Instr.priced c (wordU β k m n N R A B u p)

theorem mem_honestU {β : ℤ} {k m n N : ℕ} {R : Mat} {A : (Fin k → Fin n → Fin 64) → ℕ → Mat} {B : ℕ → Mat}
    {U : Finset ℕ} {c : ℕ} {ins : Instr (Noise N m n k)} :
    ins ∈ honestU β k m n N R A B U c ↔
      ∃ u ∈ U, ∃ p < nonFinal k m n, Instr.priced c (wordU β k m n N R A B u p) = ins := by
  simp [honestU, List.mem_flatMap, List.mem_map]

theorem honestU_cost (β : ℤ) (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat)
    (U : Finset ℕ) (c : ℕ) :
    cost (honestU β k m n N R A B U c) = c * (U.card * (m * n) * (3 * k / 16 - 1)) := by
  have h1 : ∀ u, cost ((List.range (nonFinal k m n)).map fun p => Instr.priced c (wordU β k m n N R A B u p)) =
      c * nonFinal k m n := by
    intro u
    simp only [cost, List.map_map]
    rw [show (Instr.price ∘ fun p => Instr.priced c (wordU β k m n N R A B u p)) = fun _ => c from rfl,
      List.map_const', List.sum_replicate, List.length_range, smul_eq_mul, mul_comm]
  rw [honestU, cost_flatMap]
  simp only [h1]
  rw [List.map_const', List.sum_replicate, length_toList, smul_eq_mul, nonFinal]
  ring

/-- **Theorem D on route U's hypotheses hold jointly, and the bound is attained.** -/
theorem algWitnessU : AlgWitnessU := by
  intro β k hk h16 m n N A B U _
  obtain ⟨R, hadm, hind⟩ := RelationWitness.independentWitness k hk h16
  refine ⟨R, honestU β k m n N R A B U 16, hadm, hind, coords_V0, ?_, fun u hu p hp =>
    mem_slots_of_priced (mem_honestU.2 ⟨u, hu, p, hp, rfl⟩) _, honestU_cost _ _ _ _ _ _ _ _ _ _⟩
  intro ins hins _
  obtain ⟨u, _, p, _, rfl⟩ := mem_honestU.1 hins
  exact le_rfl

theorem honestU_h16 (β : ℤ) (k m n N : ℕ) (R : Mat) (A : (Fin k → Fin n → Fin 64) → ℕ → Mat) (B : ℕ → Mat)
    (U : Finset ℕ) : H16 (honestU β k m n N R A B U 16) := by
  intro ins hins _
  obtain ⟨u, _, p, _, rfl⟩ := mem_honestU.1 hins
  exact le_rfl

/-- **Theorem D on route U is sharp.** -/
theorem algSharpU : AlgSharpU := by
  intro β k m n N R A B U g₀ h16 hk _ hne hmn hg
  refine ⟨honestU β k m n N R A B U 16, honestU_h16 β k m n N R A B U,
    fun u hu p hp => mem_slots_of_priced (mem_honestU.2 ⟨u, hu, p, hp, rfl⟩) _, ?_⟩
  rw [honestU_cost]
  obtain ⟨k', rfl⟩ := h16
  have hk' : 1 ≤ k' := by omega
  have h3 : 3 * (16 * k') / 16 - 1 = 3 * k' - 1 := by omega
  rw [h3, sum_const, nsmul_eq_mul]
  have hX : (0 : ℝ) < U.card * (m * n) := by
    have := Nat.cast_pos (α := ℝ).2 (Nat.mul_pos (card_pos.2 hne) hmn)
    push_cast at this
    exact this
  push_cast [Nat.cast_sub (by omega : 1 ≤ 3 * k')] at hg ⊢
  have hk'R : (1 : ℝ) ≤ k' := by exact_mod_cast hk'
  have hg' : g₀ * (3 * (16 * (k' : ℝ))) < 16 := (lt_div_iff₀ (by positivity)).1 hg
  have hpos : 0 < 1 - g₀ := by
    have : (16 : ℝ) / (3 * (16 * k')) ≤ 1 / 3 := by
      rw [div_le_iff₀ (by positivity)]
      nlinarith
    linarith
  rw [div_lt_iff₀ hpos]
  nlinarith [mul_lt_mul_of_pos_left hg' hX]

/-- **`AlgBoundWrappedU`'s hypotheses hold jointly, and its bound is attained.** -/
theorem algWrappedWitnessU : AlgWrappedWitnessU := by
  intro β k hβ1 hβ2 hk h16 hk16 m n N A B U _ hA hB
  have hU := NCP.RouteUProofs.shiftBy_unsigned (show 24 < k by omega)
  refine ⟨GameProofs.shiftBy 24 k, honestU β k m n N (GameProofs.shiftBy 24 k) A B U 16,
    GameProofs.shiftBy24_admissible hk h16, hU, NCP.RouteUProofs.unsigned_signed hU,
    RelationWitness.indep_of_fresh (RelationWitness.fresh hk h16), coords_V0, ?_, fun u hu p hp => ?_,
    honestU_cost _ _ _ _ _ _ _ _ _ _⟩
  · intro ins hins _
    obtain ⟨u, _, p, _, rfl⟩ := mem_honestU.1 hins
    exact le_rfl
  · rw [wrapped_eq_wordU hβ1 hβ2 (Or.inr ⟨hU, hk16⟩) hA hB hu hp]
    exact mem_slots_of_priced (mem_honestU.2 ⟨u, hu, p, hp, rfl⟩) _

end Pouw.Dimension.RouteUProofs
