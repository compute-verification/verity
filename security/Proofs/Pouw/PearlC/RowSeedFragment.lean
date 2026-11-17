module

public import Definitions.Pouw.Assumptions.PearlC.TTOutRowSeed

/-! The fragment argument's counting theorem, staged. -/

@[expose] public section

namespace Pouw.PearlC

open Finset Pouw.Fp8Atom

def fragRows (b : ℕ) : Finset ℕ := Ico (16 * b) (16 * b + 16)

def fragCols (J : ℕ) : Finset ℕ := Ico (8 * J) (8 * J + 8)

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
