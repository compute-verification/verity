import Proofs.Nci.Cover.FanOutValue
import Proofs.Nci.Cover.FanOutWitness

/-!
# The read tier from wiring: its hypotheses can hold together

`fanOut_value_witness`: `FanOutWitness`'s circuit and keyed token structure (`f = 2`, `d = 2`) satisfy
`ConditionalUpdateHardness` over the values' gates, `ColumnsIncompressible` with `k < bcol · V · (E + 1)` at `E = 1`,
and the per-value cut at one leg (`g = 1`), so `δ = d − E − (g − 1) = 1`, with the same cover by inference blocks.

The counted product (gate 14) reads the root (gates 12–13), its activation, whose key path is the root alone. Each leg's
exit reaches gate 14 through gate 12 and through gate 13, so a cut inside the root needs both. The product's parameter
(gate 7) lies outside every value. The block that spends the root is handed both of its gates, `f` of them.
-/

namespace Nci.Cover.SecurityProofs

namespace FanOutWitness

theorem fK_pathGates : fK.pathGates 2 = {12, 13} := by
  ext z
  rw [Keyed.mem_pathGates]
  constructor
  · rintro ⟨w, -, hw, hz⟩
    obtain rfl : w = 2 := Follows.eq_of_key_none (fun _ => rfl) hw
    exact hz
  · intro hz
    exact ⟨2, Finset.mem_univ _, Follows.refl 2, hz⟩

/-- Each leg has an exit, outside the root, read by both of the root's gates. -/
theorem fK_legExit : ∀ q ∈ ({0, 1} : Finset (Fin 3)), ∃ h ∈ fC.exits (fgates q),
    h ∈ fC.pred 12 ∧ h ∈ fC.pred 13 ∧ h ∉ ({12, 13} : Finset (Fin 15)) := by
  decide

theorem fK_cut : fK.PerValueCut 1 := by
  intro p hp
  obtain rfl : p = (1, 1) := Finset.mem_singleton.1 hp
  refine ⟨2, by decide, Follows.refl 2, ?_⟩
  intro G hG hG1 Z hZ hcut
  change Z ⊆ fK.pathGates 2 at hZ
  rw [fK_pathGates] at hZ
  obtain ⟨q, hq⟩ := Finset.card_pos.1 hG1
  obtain ⟨h, hh, h12, h13, hhr⟩ := fK_legExit q (hG hq)
  have htail : ∀ y ∈ ({12, 13} : Finset (Fin 15)), ∀ x ∈ [h, y, 14].tail,
      x ∈ fK.pathGates (fK.act (1, 1)) ∪ fM.m (1, 1) := by
    intro y hy x hx
    show x ∈ fK.pathGates 2 ∪ fm (1, 1)
    rw [fK_pathGates]
    simp only [List.tail_cons, List.mem_cons, List.not_mem_nil, or_false] at hx
    rcases hx with rfl | rfl
    · exact Finset.mem_union_left _ hy
    · exact Finset.mem_union_right _ (by decide)
  have hmem : ∀ y ∈ ({12, 13} : Finset (Fin 15)), h ∈ fC.pred y → y ∈ Z := by
    intro y hy hhy
    have hy14 : y ∈ fC.pred 14 := by
      simp only [Finset.mem_insert, Finset.mem_singleton] at hy
      rcases hy with rfl | rfl <;> decide
    obtain ⟨z, hz, hzπ⟩ := hcut q hq h hh [h, y, 14] ⟨rfl, rfl, .cons_cons hhy (.cons_cons hy14 (.singleton _))⟩
      (htail y hy)
    have hz' := hZ hz
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hzπ
    rcases hzπ with rfl | rfl | rfl
    · exact absurd hz' hhr
    · exact hz
    · exact absurd hz' (by decide)
  have hsub : ({12, 13} : Finset (Fin 15)) ⊆ Z := by
    intro y hy
    simp only [Finset.mem_insert, Finset.mem_singleton] at hy
    rcases hy with rfl | rfl
    · exact hmem 12 (by decide) h12
    · exact hmem 13 (by decide) h13
  exact Finset.card_le_card hsub

theorem fM_conditional (s : ℕ) :
    Assumptions.ConditionalUpdateHardness fM (fun _ => fK.vals.biUnion fK.gates) s 1 :=
  Witness.conditionalUpdateHardness_of_feeds fM _ (fun _ => {7}) s (by decide) (by decide) (by decide)
    fun _ _ => by simp

end FanOutWitness

open FanOutWitness in
/-- **Non-vacuity of the read tier from wiring**: some circuit with a counted product and a keyed token structure of
width `f = 2` satisfies `ConditionalUpdateHardness` over the values' gates, `ColumnsIncompressible` with
`k < bcol · V · (E + 1)` and the per-value cut at `g` legs, with `1 ≤ E` and `δ = d − E − (g − 1) ≥ 1`, and has a cover
by inference blocks, one of which holds a lookup, one a root, and one spends a value (`FanOutWitness.fC`: `s = 6`,
`k = 5`, `E = 1`, `g = 1`, `d = 2`, `V = 3`, `b = bcol = 1`, four blocks). -/
theorem fanOut_value_witness :
    ∃ (C : Circuit) (M : MulBlocks C ℕ ℕ) (d aOut c V f : ℕ) (K : Keyed M Unit (Fin 3) d aOut c V f) (s k E g : ℕ)
      (b bcol : ℝ) (𝓘 : Finset (Finset (Fin C.N))),
      M.P.Nonempty ∧ 0 < b ∧ 2 ≤ f ∧ 1 ≤ E ∧ 1 ≤ d - E - (g - 1) ∧
      Assumptions.ConditionalUpdateHardness M (fun _ => K.vals.biUnion K.gates) s b ∧
      Assumptions.ColumnsIncompressible K s bcol ∧ K.PerValueCut g ∧ (k : ℝ) < bcol * V * (E + 1) ∧
      (∀ I ∈ 𝓘, C.InferenceBlock s k I) ∧ C.Covers 𝓘 ∧
      (∃ I ∈ 𝓘, (K.lookups ∩ K.heldSet I).Nonempty) ∧ (∃ I ∈ 𝓘, (K.roots ∩ K.heldSet I).Nonempty) ∧
      (∃ I ∈ 𝓘, ∃ u, K.Spent I u) :=
  ⟨fC, fM, 2, 1, 1, 3, 2, fK, 6, 5, 1, 1, 1, 1, fI, ⟨(1, 1), by decide⟩, one_pos, le_rfl, le_rfl, le_rfl,
    fM_conditional 6, fK_columns 6, fK_cut, by norm_num, fI_blocks, fI_covers,
    ⟨{0, 1, 2, 3, 8, 9}, by decide, 0, by decide⟩, ⟨{8, 9, 10, 11, 12, 13}, by decide, 2, by decide⟩,
    ⟨{0, 1, 7, 12, 13, 14}, by decide, 2, by decide, 2, Or.inl ⟨(1, 1), by decide, rfl⟩, Follows.refl 2⟩⟩

end Nci.Cover.SecurityProofs
