import Proofs.Nci.Cover.Additive
import Mathlib.Data.Fin.VecNotation
import Mathlib.Tactic.NormNum

/-!
# The cover theorem: its hypotheses can hold together

`witness`: one small circuit with a counted product that satisfies `UpdateHardness`, the upstream certificate with
`k < b D`, and has a cover by inference blocks, so the hypotheses the bridge to a compliant declaration rests on
(`CoverTraining`) are not contradictory. `inference_sanity`: a circuit with no products, such as inference, satisfies
`UpdateHardness` and the certificate at every depth, and the bounds then say nothing, as they should.

The circuit (`Witness.wC`, 14 gates, `θ = {0}`) is one layer's worth of training in miniature. Gate 0 is the registered
token, gates 1–5 are the updated parameters' bits, one each, read only by that parameter's product. The upstream
products `(0, l)`, `l = 1, …, 4`, are gates 6–9, each the token times parameter `l`; gate 10 combines 6–8 and the
activation gates 11 and 12 each combine 9 and 10. The counted product `(1, 5)` is gate 13, reading both activation gates
and parameter 5. The activation region of use 1 is gates 6–12. Every product reads its parameter's own input gate,
outside `θ` and the region, so `ConditionalUpdateHardness`, and with it `UpdateHardness`, holds at every block size with
`b = 1` (`conditionalUpdateHardness_of_feeds`); the certificate holds at depth `D = 4`, so `k = 3 < b D`; and seven
blocks of `s = 4` gates, each handed at most three bits outside `θ`, cover the circuit.
-/

namespace Nci.Cover.SecurityProofs

namespace Witness

/-- **Conditional update hardness from private feeds**: if each product reads, among its gates' predecessors, input
gates `F l` of its updated parameter `l`, none of them registered or in an activation region, with disjoint `F l` for
distinct parameters and at least `b` gates each, then `ConditionalUpdateHardness` holds at every block size. A set of
gates that computes a product's gates is handed that product's feed. -/
theorem conditionalUpdateHardness_of_feeds {C : Circuit} {J L : Type} [DecidableEq J] [DecidableEq L]
    (M : MulBlocks C J L) (A : J → Finset (Fin C.N)) (F : L → Finset (Fin C.N)) (s : ℕ) {b : ℝ}
    (hread : ∀ p ∈ M.Pall, ∀ x ∈ F p.2, C.pred x = ∅ ∧ ∃ g ∈ M.m p, x ∈ C.pred g)
    (hout : ∀ p ∈ M.Pall, Disjoint (F p.2) (C.θ ∪ M.uses.biUnion A))
    (hdisj : ∀ p ∈ M.Pall, ∀ q ∈ M.Pall, p.2 ≠ q.2 → Disjoint (F p.2) (F q.2))
    (hb : ∀ p ∈ M.Pall, b ≤ (F p.2).card) :
    Assumptions.ConditionalUpdateHardness M A s b := by
  intro S _ B hB hBS
  have hsub : (B.image Prod.snd).biUnion F ⊆ C.inputs S \ (C.θ ∪ M.uses.biUnion A) := by
    intro x hx
    obtain ⟨l, hl, hx⟩ := Finset.mem_biUnion.1 hx
    obtain ⟨p, hp, rfl⟩ := Finset.mem_image.1 hl
    obtain ⟨hx0, g, hg, hxg⟩ := hread p (hB hp) x hx
    have hxS : x ∈ S := pred_subset_of_mem_computed (hBS p hp hg) hxg
    exact Finset.mem_sdiff.2 ⟨Finset.mem_filter.2 ⟨hxS, Or.inl hx0⟩,
      fun h => Finset.disjoint_left.1 (hout p (hB hp)) hx h⟩
  have hcard : (((B.image Prod.snd).biUnion F).card : ℝ) = ∑ l ∈ B.image Prod.snd, ((F l).card : ℝ) := by
    rw [Finset.card_biUnion, Nat.cast_sum]
    intro l hl l' hl' hne
    obtain ⟨p, hp, rfl⟩ := Finset.mem_image.1 hl
    obtain ⟨q, hq, rfl⟩ := Finset.mem_image.1 hl'
    exact hdisj p (hB hp) q (hB hq) hne
  calc b * (B.image Prod.snd).card = ∑ _l ∈ B.image Prod.snd, b := by
        rw [Finset.sum_const, nsmul_eq_mul, mul_comm]
    _ ≤ ∑ l ∈ B.image Prod.snd, ((F l).card : ℝ) := Finset.sum_le_sum fun l hl => by
        obtain ⟨p, hp, rfl⟩ := Finset.mem_image.1 hl
        exact hb p (hB hp)
    _ = ((B.image Prod.snd).biUnion F).card := hcard.symm
    _ ≤ (C.inputs S \ (C.θ ∪ M.uses.biUnion A)).card := by exact_mod_cast Finset.card_le_card hsub

local instance decInferenceBlock (C : Circuit) (s k : ℕ) (I : Finset (Fin C.N)) :
    Decidable (C.InferenceBlock s k I) :=
  inferInstanceAs (Decidable (I.card ≤ s ∧ (C.inputs I \ C.θ).card ≤ k))

local instance decCovers (C : Circuit) (𝓘 : Finset (Finset (Fin C.N))) : Decidable (C.Covers 𝓘) :=
  inferInstanceAs (Decidable (∀ g, (C.pred g).Nonempty → ∃ I ∈ 𝓘, g ∈ C.computed I))

/-- The witness's wiring: the token 0 and the parameter bits 1–5 are input gates; 6–9 are the upstream products, 10
combines 6–8, the activation gates 11 and 12 combine 9 and 10, and 13 is the counted product. -/
def wPred : Fin 14 → Finset (Fin 14) :=
  ![∅, ∅, ∅, ∅, ∅, ∅, {0, 1}, {0, 2}, {0, 3}, {0, 4}, {6, 7, 8}, {9, 10}, {9, 10}, {5, 11, 12}]

/-- The witness circuit, with the token registered. -/
abbrev wC : Circuit where
  N := 14
  pred := wPred
  pred_lt := by decide
  θ := {0}
  θ_input := by decide

/-- The products' gates: `(0, l)` is gate `5 + l` for `l = 1, …, 4`, and `(1, 5)` is gate 13. -/
def wm : ℕ × ℕ → Finset (Fin 14)
  | (0, 1) => {6}
  | (0, 2) => {7}
  | (0, 3) => {8}
  | (0, 4) => {9}
  | (1, 5) => {13}
  | _ => ∅

/-- The products' output gates. -/
def wo : ℕ × ℕ → Fin 14
  | (0, 1) => 6
  | (0, 2) => 7
  | (0, 3) => 8
  | (0, 4) => 9
  | _ => 13

/-- The witness's products: four upstream ones and the counted `(1, 5)`. -/
abbrev wM : MulBlocks wC ℕ ℕ where
  Pall := {(0, 1), (0, 2), (0, 3), (0, 4), (1, 5)}
  P := {(1, 5)}
  P_subset := by decide
  m := wm
  m_nonempty := by decide
  m_computed := by decide
  m_disjoint := by
    have h : ∀ p ∈ ({(0, 1), (0, 2), (0, 3), (0, 4), (1, 5)} : Finset (ℕ × ℕ)),
        ∀ q ∈ ({(0, 1), (0, 2), (0, 3), (0, 4), (1, 5)} : Finset (ℕ × ℕ)), p ≠ q → Disjoint (wm p) (wm q) := by
      decide
    exact fun p hp q hq => h p hp q hq
  o := wo
  m_outputs := by decide

/-- Use 1's activation region: the upstream products, gate 10 and the activation gates. -/
def wA : ℕ → Finset (Fin 14) := fun j => if j = 1 then {6, 7, 8, 9, 10, 11, 12} else ∅

/-- Each parameter's input gate. -/
def wF : ℕ → Finset (Fin 14)
  | 1 => {1}
  | 2 => {2}
  | 3 => {3}
  | 4 => {4}
  | 5 => {5}
  | _ => ∅

/-- The cover: one block for each upstream product, one for gate 10, one for the activation and one for the counted
product, each of four gates. -/
def wI : Finset (Finset (Fin 14)) :=
  {{0, 1, 6, 5}, {0, 2, 7, 5}, {0, 3, 8, 5}, {0, 4, 9, 5}, {6, 7, 8, 10}, {9, 10, 11, 12}, {5, 11, 12, 13}}

theorem wM_uses : wM.uses = {1} := by decide

theorem wM_P_nonempty : wM.P.Nonempty := ⟨(1, 5), by decide⟩

theorem wM_conditional (s : ℕ) : Assumptions.ConditionalUpdateHardness wM wA s 1 := by
  have hcard : ∀ p ∈ wM.Pall, (wF p.2).card = 1 := by decide
  refine conditionalUpdateHardness_of_feeds wM wA wF s (by decide) (by decide) (by decide) fun p hp => ?_
  rw [hcard p hp, Nat.cast_one]

theorem wM_upstream : wM.Upstream wA 4 where
  disjoint := by
    rw [wM_uses, Finset.coe_singleton]
    exact Set.pairwiseDisjoint_singleton _ _
  disjoint_θ := by decide
  upstream := by
    intro j hj
    rw [wM_uses, Finset.mem_singleton] at hj
    subst hj
    refine ⟨{(0, 1), (0, 2), (0, 3), (0, 4)}, by decide, by decide, by decide, ?_⟩
    intro p hp _
    obtain rfl : p = (1, 5) := Finset.mem_singleton.1 hp
    refine ⟨13, by decide, 11, by decide, ?_⟩
    intro q hq
    simp only [Finset.mem_insert, Finset.mem_singleton] at hq
    rcases hq with rfl | rfl | rfl | rfl
    · exact ⟨[6, 10, 11], ⟨rfl, rfl, by decide⟩, by decide⟩
    · exact ⟨[7, 10, 11], ⟨rfl, rfl, by decide⟩, by decide⟩
    · exact ⟨[8, 10, 11], ⟨rfl, rfl, by decide⟩, by decide⟩
    · exact ⟨[9, 11], ⟨rfl, rfl, by decide⟩, by decide⟩

theorem wI_blocks : ∀ I ∈ wI, wC.InferenceBlock 4 3 I := by
  intro I hI
  simp only [wI, Finset.mem_insert, Finset.mem_singleton] at hI
  rcases hI with rfl | rfl | rfl | rfl | rfl | rfl | rfl <;> decide

theorem wI_covers : wC.Covers wI := by decide

end Witness

open Witness in
/-- **Non-vacuity**: some circuit with a counted product satisfies `UpdateHardness` and the upstream certificate with
`k < b D`, and has a cover by inference blocks. -/
theorem witness :
    ∃ (C : Circuit) (M : MulBlocks C ℕ ℕ) (A : ℕ → Finset (Fin C.N)) (s k D : ℕ) (b : ℝ)
      (𝓘 : Finset (Finset (Fin C.N))),
      M.P.Nonempty ∧ 0 < b ∧ Assumptions.UpdateHardness M s b ∧ M.Upstream A D ∧ (k : ℝ) < b * D ∧
      (∀ I ∈ 𝓘, C.InferenceBlock s k I) ∧ C.Covers 𝓘 :=
  ⟨wC, wM, wA, 4, 3, 4, 1, wI, wM_P_nonempty, one_pos, updateHardness_of_conditional wM wA (wM_conditional 4),
    wM_upstream, by norm_num, wI_blocks, wI_covers⟩

/-- **Honest inference**: with no products, `UpdateHardness` and the upstream certificate hold at every block size, rate
and depth. -/
theorem inference_sanity (C : Circuit) (M : MulBlocks C ℕ ℕ) (hM : M.Pall = ∅) (A : ℕ → Finset (Fin C.N))
    (s D : ℕ) (b : ℝ) : Assumptions.UpdateHardness M s b ∧ M.Upstream A D := by
  have hP : M.P = ∅ := Finset.subset_empty.1 (hM ▸ M.P_subset)
  have huses : M.uses = ∅ := by simp [MulBlocks.uses, hP]
  refine ⟨fun S _ B hB _ => ?_, ⟨?_, ?_, ?_⟩⟩
  · rw [hM, Finset.subset_empty] at hB
    subst hB
    simp
  · rw [huses, Finset.coe_empty]
    exact Set.pairwiseDisjoint_empty
  · intro j hj
    simp [huses] at hj
  · intro j hj
    simp [huses] at hj

end Nci.Cover.SecurityProofs
