import Definitions.Pouw.Dimension.Adaptive
import Proofs.Pouw.Dimension.PointwiseProofs

/-!
# adaptive-dw: proofs

* **Chunk provenance** (`run_chunks_le`). By `good_prefix`, every slot after every prefix holds an initial chunk value
  or one of `writes P` new ones; so the target chunk values no initial chunk holds number at most `writes P`.
* **The lift** (`adaptiveDW_chunks`): the pointwise bound, summed over the draws.
* **The obstruction** (`holds_flat`, `not_adaptiveDW`, `card_levels_le`, `not_adaptiveDW_levels`). Writing each
  distinct target value once into one slot (`fill`) holds them all at `w` writes a value. Row-step chunks
  `base i + c i t · 1` number at most `Σ_i |{c i t}|`, so adaptive-dw fails above `w` times that.
-/

namespace Pouw.Dimension.Pointwise

open Finset

variable {A V : Type} {w : ℕ}

/-- Write the chunk value `y` into slot `a₀`, word by word. -/
def fill (a₀ : A) (y : Fin w → V) : List (Step A V w) := (List.finRange w).map fun p => .write a₀ p (y p)

theorem writes_append (P Q : List (Step A V w)) : writes (P ++ Q) = writes P + writes Q := by
  simp [writes]

theorem writes_fill (a₀ : A) (y : Fin w → V) : writes (fill a₀ y) = w := by
  have h : (Step.cost ∘ fun p => (Step.write a₀ p (y p) : Step A V w)) = fun _ => 1 := funext fun _ => rfl
  simp [writes, fill, h]

theorem writes_flatMap_fill (a₀ : A) (l : List (Fin w → V)) : writes (l.flatMap (fill a₀)) = w * l.length := by
  induction l with
  | nil => simp [writes]
  | cons y l ih => rw [List.flatMap_cons, writes_append, writes_fill, ih, List.length_cons]; ring

variable [DecidableEq A]

open Classical in
theorem run_chunks_le [DecidableEq V] [Inhabited V] (m₀ : A → Fin w → V) (P : List (Step A V w))
    (Y : Finset (Fin w → V)) (hY : Holds m₀ P Y) : (Y.filter fun y => ∀ a, m₀ a ≠ y).card ≤ writes P := by
  obtain ⟨L, -, hcov⟩ := good_prefix m₀ P
  calc (Y.filter fun y => ∀ a, m₀ a ≠ y).card ≤ ((range (writes P)).image L).card := by
        apply card_le_card
        intro y hy
        rw [mem_filter] at hy
        obtain ⟨i, a, hia⟩ := hY y hy.1
        rcases hcov i a with ⟨b, hb⟩ | ⟨k, hk, hkL⟩
        · exact absurd (hb.trans hia) (hy.2 b)
        · exact mem_image.2 ⟨k, mem_range.2 hk, hkL.trans hia⟩
    _ ≤ (range (writes P)).card := card_image_le
    _ = writes P := card_range _

open Classical in
theorem adaptiveDW_chunks [DecidableEq V] [Inhabited V] (Ω : Type) [Fintype Ω] (m₀ : A → Fin w → V)
    (Y : Ω → Finset (Fin w → V)) : AdaptiveDW Ω m₀ Y (∑ g, ((Y g).filter fun y => ∀ a, m₀ a ≠ y).card) :=
  fun P hP => sum_le_sum fun g _ => run_chunks_le m₀ (P g) (Y g) (hP g)

theorem fill_aux (a₀ : A) (y : Fin w → V) (ps : List (Fin w)) (m : A → Fin w → V) (p : Fin w) :
    ((ps.map fun q => (Step.write a₀ q (y q) : Step A V w)).foldl (fun m st => st.apply m) m) a₀ p =
      if p ∈ ps then y p else m a₀ p := by
  induction ps generalizing m with
  | nil => simp
  | cons q ps ih =>
    rw [List.map_cons, List.foldl_cons, ih]
    by_cases hp : p ∈ ps
    · simp [hp]
    · by_cases hpq : p = q
      · subst hpq
        simp [Step.apply]
      · simp [hp, hpq, Step.apply]

theorem fill_run (a₀ : A) (y : Fin w → V) (m : A → Fin w → V) :
    ((fill a₀ y).foldl (fun m st => st.apply m) m) a₀ = y := by
  funext p
  rw [fill, fill_aux]
  simp [List.mem_finRange]

/-- **Any finite set of chunk values is held by a run of `w` writes a value.** -/
theorem holds_flat (a₀ : A) (m₀ : A → Fin w → V) (Y : Finset (Fin w → V)) :
    Holds m₀ (Y.toList.flatMap (fill a₀)) Y ∧ writes (Y.toList.flatMap (fill a₀)) = w * Y.card := by
  refine ⟨fun y hy => ?_, by rw [writes_flatMap_fill, Finset.length_toList]⟩
  obtain ⟨l₁, l₂, h⟩ := List.append_of_mem (Finset.mem_toList.2 hy)
  refine ⟨(l₁.flatMap (fill a₀) ++ fill a₀ y).length, a₀, ?_⟩
  rw [h, List.flatMap_append, List.flatMap_cons, ← List.append_assoc, List.take_left, run, List.foldl_append,
    fill_run]

/-- **No adaptive-dw above `w·Σ_g |Y g|`**: a run that writes each distinct target value once holds them all. -/
theorem not_adaptiveDW (a₀ : A) (Ω : Type) [Fintype Ω] (m₀ : A → Fin w → V) (Y : Ω → Finset (Fin w → V)) (B : ℕ)
    (hB : w * ∑ g, (Y g).card < B) : ¬ AdaptiveDW Ω m₀ Y B := by
  intro h
  have h₁ := h (fun g => (Y g).toList.flatMap (fill a₀)) fun g => (holds_flat a₀ m₀ (Y g)).1
  have h₂ : ∑ g, writes ((Y g).toList.flatMap (fill a₀)) = w * ∑ g, (Y g).card := by
    rw [Finset.mul_sum]
    exact sum_congr rfl fun g _ => (holds_flat a₀ m₀ (Y g)).2
  omega

omit [DecidableEq A] in
/-- **Column-free increments collapse**: row-step chunks `base i + c i t · 1` number at most `Σ_i |{c i t}|`. -/
theorem card_levels_le [DecidableEq V] [Add V] {ι T : Type} (rows : Finset ι) (steps : Finset T)
    (base : ι → Fin w → V) (c : ι → T → V) :
    ((rows ×ˢ steps).image fun it => base it.1 + fun _ => c it.1 it.2).card ≤
      ∑ i ∈ rows, (steps.image (c i)).card := by
  calc _ ≤ (rows.biUnion fun i => (steps.image (c i)).image fun x => base i + fun _ => x).card := by
        apply card_le_card
        intro y hy
        obtain ⟨⟨i, t⟩, hit, rfl⟩ := mem_image.1 hy
        rw [mem_product] at hit
        exact mem_biUnion.2 ⟨i, hit.1, mem_image.2 ⟨c i t, mem_image.2 ⟨t, hit.2, rfl⟩, rfl⟩⟩
    _ ≤ ∑ i ∈ rows, ((steps.image (c i)).image fun x => base i + fun _ => x).card := card_biUnion_le
    _ ≤ _ := sum_le_sum fun i _ => card_image_le

/-- **The walk's obstruction**: when every draw's targets are row-step chunks `base g i + c g i t · 1`, adaptive-dw
fails at any bound above `w·Σ_g Σ_i |{c g i t}|`. -/
theorem not_adaptiveDW_levels [DecidableEq V] [Add V] {ι T : Type} (a₀ : A) (Ω : Type) [Fintype Ω]
    (m₀ : A → Fin w → V) (rows : Finset ι) (steps : Finset T) (base : Ω → ι → Fin w → V) (c : Ω → ι → T → V)
    (B : ℕ) (hB : w * ∑ g, ∑ i ∈ rows, (steps.image (c g i)).card < B) :
    ¬ AdaptiveDW Ω m₀ (fun g => (rows ×ˢ steps).image fun it => base g it.1 + fun _ => c g it.1 it.2) B := by
  refine not_adaptiveDW a₀ Ω m₀ _ B (lt_of_le_of_lt ?_ hB)
  exact Nat.mul_le_mul_left w (sum_le_sum fun g _ => card_levels_le rows steps (base g) (c g))

end Pouw.Dimension.Pointwise
