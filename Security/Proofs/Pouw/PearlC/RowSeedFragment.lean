import Definitions.Pouw.Assumptions.PearlC.TTOutRowSeed
import Definitions.Pouw.Guarantees.PearlC

/-!
# The fragment argument: the counting theorem (staging, nothing pinned)

The fragment argument. A skip
program omits `mma.sync m16n8k32` instructions, each a slot `(b, J, t)` (16-row block `b`, 8-column group `J`, atom
`t`), and routes at most 5 of a slot's 128 words to generic cores. Its words stay correct only if each word's omitted
atoms form a skip set (`SkipSetU`).

* **The counting theorem** `fragSkip_count_le`: if every word stays correct and each slot's atom is undebited in the
  slot's unpatched words, then `11·#slots ≤ Σ_i fragCount i`. A slot has at least 11 rows whose 8 words are all
  unpatched (`fragCleanRows_card`), and each `(i, J, t)` lies in at most one slot (`fragSlots_card_le`).
* **The saving** `fragSkip_saving_le`: at 4,096 units a slot, at most `(4096/11)·Σ_i fragCount i`, about `372` units
  per unit of `fragCount`.

The draw side is in `RowSeedSkip.lean`: `ttOutRowSeed_skipClass` bounds a skip-class program's saving under the named
assumption `Assumptions.FragDraw` at `Λ = 100`, (C4) and (C6), through the union bound `qtree_union_bound`.
-/

namespace Pouw.PearlC

open Finset Pouw.Fp8Atom

/-! ## The counting theorem -/

/-- The rows of 16-row block `b`. -/
def fragRows (b : ℕ) : Finset ℕ := Ico (16 * b) (16 * b + 16)

/-- The columns of 8-column group `J`. -/
def fragCols (J : ℕ) : Finset ℕ := Ico (8 * J) (8 * J + 8)

/-- The rows of a slot's block none of whose words in the slot's column group is patched. -/
def fragCleanRows (patch : Finset (ℕ × ℕ)) (b J : ℕ) : Finset ℕ :=
  (fragRows b).filter fun i => ∀ j ∈ fragCols J, (i, j) ∉ patch

theorem fragCleanRows_card (patch : Finset (ℕ × ℕ)) (b J : ℕ) (hp : patch.card ≤ 5) :
    11 ≤ (fragCleanRows patch b J).card := by
  classical
  have hsplit := card_filter_add_card_filter_not (s := fragRows b) (fun i => ∀ j ∈ fragCols J, (i, j) ∉ patch)
  have hblock : (fragRows b).card = 16 := by simp [fragRows]
  have hdirty : ((fragRows b).filter fun i => ¬ ∀ j ∈ fragCols J, (i, j) ∉ patch).card ≤ patch.card := by
    calc _ ≤ (patch.image Prod.fst).card := by
            apply card_le_card
            intro i hi
            simp only [mem_filter, not_forall, not_not] at hi
            obtain ⟨_, j, _, hij⟩ := hi
            exact mem_image.mpr ⟨(i, j), hij, rfl⟩
      _ ≤ patch.card := card_image_le
  unfold fragCleanRows
  omega

/-- **The counting theorem, generically.** Slots `(b, J, t)` (a 16-row block, an 8-column group, an atom), each with at
most 5 patched words, where every unpatched word `(i, j)` of the slot has `ok i j t`: then `11·#slots` is at most the
number of `(i, J, t)` with `ok i j t` for all 8 columns `j` of `J`. -/
theorem fragSlots_card_le (ok : ℕ → ℕ → ℕ → Prop) [∀ i j t, Decidable (ok i j t)] (m nJ kT : ℕ)
    (slots : Finset (ℕ × ℕ × ℕ)) (patch : ℕ × ℕ × ℕ → Finset (ℕ × ℕ))
    (hb : ∀ s ∈ slots, 16 * s.1 + 16 ≤ m) (hJ : ∀ s ∈ slots, s.2.1 < nJ) (ht : ∀ s ∈ slots, s.2.2 < kT)
    (hp : ∀ s ∈ slots, (patch s).card ≤ 5)
    (hok : ∀ s ∈ slots, ∀ i ∈ fragRows s.1, ∀ j ∈ fragCols s.2.1, (i, j) ∉ patch s → ok i j s.2.2) :
    11 * slots.card ≤
      ∑ i ∈ range m, ((range nJ ×ˢ range kT).filter fun Jt => ∀ j ∈ fragCols Jt.1, ok i j Jt.2).card := by
  classical
  set G : ℕ × ℕ × ℕ → Finset ℕ := fun s => fragCleanRows (patch s) s.1 s.2.1
  have hrow : ∀ i ∈ range m, (slots.filter fun s => i ∈ G s).card ≤
      ((range nJ ×ˢ range kT).filter fun Jt => ∀ j ∈ fragCols Jt.1, ok i j Jt.2).card := by
    intro i _
    refine card_le_card_of_injOn (fun s => (s.2.1, s.2.2)) (fun s hs => ?_) (fun s hs s' hs' h => ?_)
    · obtain ⟨hs, hi⟩ := mem_filter.mp (mem_coe.mp hs)
      obtain ⟨hib, hij⟩ := mem_filter.mp hi
      exact mem_coe.mpr (mem_filter.mpr ⟨mem_product.mpr ⟨mem_range.mpr (hJ s hs), mem_range.mpr (ht s hs)⟩,
        fun j hj => hok s hs i hib j hj (hij j hj)⟩)
    · obtain ⟨_, hi⟩ := mem_filter.mp (mem_coe.mp hs)
      obtain ⟨_, hi'⟩ := mem_filter.mp (mem_coe.mp hs')
      have h1 := (mem_filter.mp hi).1
      have h1' := (mem_filter.mp hi').1
      simp only [fragRows, mem_Ico] at h1 h1'
      simp only [Prod.mk.injEq] at h
      obtain ⟨b, J, t⟩ := s
      obtain ⟨b', J', t'⟩ := s'
      simp only at h h1 h1' ⊢
      obtain ⟨rfl, rfl⟩ := h
      have : b = b' := by omega
      rw [this]
  have hsub : ∀ s ∈ slots, G s ⊆ range m := fun s hs i hi => by
    have := (mem_filter.mp hi).1
    simp only [fragRows, mem_Ico] at this
    exact mem_range.mpr (by have := hb s hs; omega)
  calc 11 * slots.card = ∑ _s ∈ slots, 11 := by rw [sum_const, smul_eq_mul, mul_comm]
    _ ≤ ∑ s ∈ slots, (G s).card := sum_le_sum fun s hs => fragCleanRows_card _ _ _ (hp s hs)
    _ = ∑ s ∈ slots, ∑ i ∈ range m, if i ∈ G s then 1 else 0 := by
        refine sum_congr rfl fun s hs => ?_
        rw [sum_boole, Nat.cast_id, filter_mem_eq_inter, inter_eq_right.mpr (hsub s hs)]
    _ = ∑ i ∈ range m, ∑ s ∈ slots, if i ∈ G s then 1 else 0 := sum_comm
    _ = ∑ i ∈ range m, (slots.filter fun s => i ∈ G s).card := by
        refine sum_congr rfl fun i _ => ?_
        rw [sum_boole, Nat.cast_id]
    _ ≤ _ := sum_le_sum hrow

/-- **The counting theorem**: a skip program omits slots `(b, J, t)`, the `mma.sync` at atom `t` over the fragment of
16-row block `b` and 8-column group `J`, each with at most 5 patched words. If every word stays
correct (its omitted atoms form a skip set) and each slot's atom is undebited in its unpatched words, then
`11·#slots ≤ Σ_i fragCount i`: each such slot has at least 11 rows whose 8 words are all unpatched, and each
`(i, J, t)` lies in at most one slot. -/
theorem fragSkip_count_le (p : Pipeline) (G : ℕ) {m n k : ℕ} (A : Matrix (Fin m) (Fin k) Code)
    (B : Matrix (Fin n) (Fin k) Code) (slots : Finset (ℕ × ℕ × ℕ)) (patch : ℕ × ℕ × ℕ → Finset (ℕ × ℕ))
    (hb : ∀ s ∈ slots, 16 * s.1 + 16 ≤ m) (hJ : ∀ s ∈ slots, s.2.1 < n / 8) (ht : ∀ s ∈ slots, s.2.2 < k / 32)
    (hp : ∀ s ∈ slots, (patch s).card ≤ 5)
    (hcorrect : ∀ (i : Fin m) (j : Fin n), SkipSetU p G (rowAtoms (A i) (B j)) (omittedAt slots patch i j))
    (hundebited : ∀ s ∈ slots, ∀ (i : Fin m) (j : Fin n), (i : ℕ) / 16 = s.1 → (j : ℕ) / 8 = s.2.1 →
      ((i : ℕ), (j : ℕ)) ∉ patch s → SkipP.flagsAt p G k (A i) (B j) s.2.2 = false) :
    11 * slots.card ≤ ∑ i : Fin m, fragCount p G A B i := by
  classical
  let ok : ℕ → ℕ → ℕ → Prop := fun i j t =>
    ∀ (hi : i < m) (hj : j < n), undebitedSkippable p G (A ⟨i, hi⟩) (B ⟨j, hj⟩) t
  have key := fragSlots_card_le ok m (n / 8) (k / 32) slots patch hb hJ ht hp (by
    intro s hs i hi j hj hij hi' hj'
    simp only [fragRows, fragCols, mem_Ico] at hi hj
    refine ⟨⟨omittedAt slots patch i j, ?_, hcorrect ⟨i, hi'⟩ ⟨j, hj'⟩⟩, ?_⟩
    · exact mem_image.mpr ⟨s, mem_filter.mpr ⟨hs, by omega, by omega, hij⟩, rfl⟩
    · exact hundebited s hs ⟨i, hi'⟩ ⟨j, hj'⟩ (by simp only; omega) (by simp only; omega) hij)
  refine key.trans ?_
  rw [← Fin.sum_univ_eq_sum_range]
  refine sum_le_sum fun i _ => card_le_card fun Jt hJt => ?_
  obtain ⟨hmem, hall⟩ := mem_filter.mp hJt
  refine mem_filter.mpr ⟨hmem, fun j hj => ?_⟩
  have := hall j (by simp only [fragCols, mem_Ico]; omega) i.isLt j.isLt
  exact this

/-- **The saving bound** (`fragSkip_saving_le`): at 4,096 units per omitted instruction, the undebited slots of a skip
program whose words stay correct save at most `(4096/11)·Σ_i fragCount i`. -/
theorem fragSkip_saving_le (p : Pipeline) (G : ℕ) {m n k : ℕ} (A : Matrix (Fin m) (Fin k) Code)
    (B : Matrix (Fin n) (Fin k) Code) (slots : Finset (ℕ × ℕ × ℕ)) (patch : ℕ × ℕ × ℕ → Finset (ℕ × ℕ))
    (hb : ∀ s ∈ slots, 16 * s.1 + 16 ≤ m) (hJ : ∀ s ∈ slots, s.2.1 < n / 8) (ht : ∀ s ∈ slots, s.2.2 < k / 32)
    (hp : ∀ s ∈ slots, (patch s).card ≤ 5)
    (hcorrect : ∀ (i : Fin m) (j : Fin n), SkipSetU p G (rowAtoms (A i) (B j)) (omittedAt slots patch i j))
    (hundebited : ∀ s ∈ slots, ∀ (i : Fin m) (j : Fin n), (i : ℕ) / 16 = s.1 → (j : ℕ) / 8 = s.2.1 →
      ((i : ℕ), (j : ℕ)) ∉ patch s → SkipP.flagsAt p G k (A i) (B j) s.2.2 = false) :
    (4096 * slots.card : ℚ) ≤ 4096 / 11 * ∑ i : Fin m, (fragCount p G A B i : ℚ) := by
  have h := fragSkip_count_le p G A B slots patch hb hJ ht hp hcorrect hundebited
  have h' : (11 * slots.card : ℚ) ≤ ∑ i : Fin m, (fragCount p G A B i : ℚ) := by exact_mod_cast h
  nlinarith

end Pouw.PearlC
