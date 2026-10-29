import Definitions.Pouw.Dimension.Adaptive
import Proofs.Pouw.Dimension.PointwiseProofs

/-!
# adaptive-dw: proofs

* **Chunk provenance** (`run_chunks_le`). By `good_prefix`, every slot after every prefix holds an initial chunk value
  or one of `writes P` new ones; so the target chunk values no initial chunk holds number at most `writes P`.
* **The lifts** (`adaptiveDW_chunks`, `adaptiveDW_ball`): the pointwise bounds (`run_chunks_le`, `run_ball_le`),
  summed over the draws.
* **Tables** (`run_ball_table_le`, `adaptiveDW_ball_table`). An initial chunk within `r y` of a target is within it of
  no other, so any initial memory costs the balls at most `R + 1` a slot.
* **The obstruction** (`holds_flat`, `not_adaptiveDW`, `card_levels_le`, `not_adaptiveDW_levels`). Writing each
  distinct target value once into one slot (`fill`) holds them all at `w` writes a value. Row-step chunks
  `base i + c i t · 1` number at most `Σ_i |{c i t}|`, so adaptive-dw fails above `w` times that.
* **Repeats on a `false` bit** (`card_rows_le`, `card_image_le_ones`, `not_adaptiveDW_ones`). A row whose chunk
  repeats at every step whose bit is `false` takes at most `1 + #true bits` values, so adaptive-dw fails above `w`
  times their sum over rows and draws.
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

/-- **The balls' lift**: when each draw's targets sit further than `r g y` from every initial chunk and pairwise
further apart than the sum of their radii, adaptive-dw holds at `Σ_g Σ_y (r g y + 1)`. -/
theorem adaptiveDW_ball [DecidableEq V] [Inhabited V] (Ω : Type) [Fintype Ω] (m₀ : A → Fin w → V)
    (Y : Ω → Finset (Fin w → V)) (r : Ω → (Fin w → V) → ℕ)
    (hI : ∀ g, ∀ y ∈ Y g, ∀ a, r g y < hammingDist (m₀ a) y)
    (hYY : ∀ g, ∀ y ∈ Y g, ∀ y' ∈ Y g, y ≠ y' → r g y + r g y' < hammingDist y y') :
    AdaptiveDW Ω m₀ Y (∑ g, ∑ y ∈ Y g, (r g y + 1)) :=
  fun P hP => sum_le_sum fun g _ => run_ball_le m₀ (P g) (Y g) (hP g) (r g) (hI g) (hYY g)

/-- **Balls against any initial memory**: targets pairwise further apart than the sum of their radii, each radius at
most `R`, need `Σ_y (r y + 1)` writes less `R + 1` a slot. An initial chunk lies within `r y` of at most one target,
so dropping those targets leaves `run_ball_le`'s hypotheses. -/
theorem run_ball_table_le [Fintype A] [DecidableEq V] [Inhabited V] (m₀ : A → Fin w → V) (P : List (Step A V w))
    (Y : Finset (Fin w → V)) (hY : ∀ y ∈ Y, ∃ i a, run m₀ (P.take i) a = y) (r : (Fin w → V) → ℕ) (R : ℕ)
    (hR : ∀ y ∈ Y, r y ≤ R) (hYY : ∀ y ∈ Y, ∀ y' ∈ Y, y ≠ y' → r y + r y' < hammingDist y y') :
    ∑ y ∈ Y, (r y + 1) ≤ writes P + Fintype.card A * (R + 1) := by
  classical
  let far : (Fin w → V) → Prop := fun y => ∀ a, r y < hammingDist (m₀ a) y
  have h1 : ∑ y ∈ Y.filter far, (r y + 1) ≤ writes P :=
    run_ball_le m₀ P _ (fun y hy => hY y (mem_filter.1 hy).1) r (fun y hy => (mem_filter.1 hy).2)
      fun y hy y' hy' => hYY y (mem_filter.1 hy).1 y' (mem_filter.1 hy').1
  have hnear : ∀ y ∈ Y.filter (¬ far ·), ∃ a, hammingDist (m₀ a) y ≤ r y := fun y hy => by
    have := (mem_filter.1 hy).2
    simp only [far, not_forall, not_lt] at this
    exact this
  have hcard : (Y.filter (¬ far ·)).card ≤ Fintype.card A :=
    calc _ ≤ (univ.biUnion fun a => Y.filter fun y => hammingDist (m₀ a) y ≤ r y).card := by
          apply card_le_card
          intro y hy
          obtain ⟨a, ha⟩ := hnear y hy
          exact mem_biUnion.2 ⟨a, mem_univ _, mem_filter.2 ⟨(mem_filter.1 hy).1, ha⟩⟩
      _ ≤ ∑ a, (Y.filter fun y => hammingDist (m₀ a) y ≤ r y).card := card_biUnion_le
      _ ≤ ∑ _a : A, 1 := sum_le_sum fun a _ => card_le_one.2 fun y hy y' hy' => by
          by_contra hne
          rw [mem_filter] at hy hy'
          have := hammingDist_triangle y (m₀ a) y'
          rw [hammingDist_comm y (m₀ a)] at this
          have := hYY y hy.1 y' hy'.1 hne
          omega
      _ = Fintype.card A := by simp
  have h2 : ∑ y ∈ Y.filter (¬ far ·), (r y + 1) ≤ Fintype.card A * (R + 1) :=
    calc _ ≤ ∑ _y ∈ Y.filter (¬ far ·), (R + 1) := sum_le_sum fun y hy => by
          have := hR y (mem_filter.1 hy).1
          omega
      _ = (Y.filter (¬ far ·)).card * (R + 1) := by rw [sum_const, smul_eq_mul]
      _ ≤ _ := Nat.mul_le_mul_right _ hcard
  calc ∑ y ∈ Y, (r y + 1) = ∑ y ∈ Y.filter far, (r y + 1) + ∑ y ∈ Y.filter (¬ far ·), (r y + 1) :=
        (sum_filter_add_sum_filter_not Y far _).symm
    _ ≤ _ := Nat.add_le_add h1 h2

/-- **The tables' lift**: over the draws, adaptive-dw holds at `Σ_g Σ_y (r g y + 1)` less `R + 1` a slot a draw, for
any initial memory. -/
theorem adaptiveDW_ball_table [Fintype A] [DecidableEq V] [Inhabited V] (Ω : Type) [Fintype Ω]
    (m₀ : A → Fin w → V) (Y : Ω → Finset (Fin w → V)) (r : Ω → (Fin w → V) → ℕ) (R : ℕ)
    (hR : ∀ g, ∀ y ∈ Y g, r g y ≤ R)
    (hYY : ∀ g, ∀ y ∈ Y g, ∀ y' ∈ Y g, y ≠ y' → r g y + r g y' < hammingDist y y') :
    AdaptiveDW Ω m₀ Y (∑ g, ∑ y ∈ Y g, (r g y + 1) - Fintype.card Ω * (Fintype.card A * (R + 1))) := by
  intro P hP
  have h := sum_le_sum fun g (_ : g ∈ univ) => run_ball_table_le m₀ (P g) (Y g) (hP g) (r g) R (hR g) (hYY g)
  rw [sum_add_distrib, sum_const, card_univ, smul_eq_mul] at h
  omega

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

/-- Row-step values number at most the sum over rows of each row's distinct values. -/
theorem card_rows_le {X ι T : Type} [DecidableEq X] (rows : Finset ι) (steps : Finset T) (F : ι → T → X) :
    ((rows ×ˢ steps).image fun it => F it.1 it.2).card ≤ ∑ i ∈ rows, (steps.image (F i)).card := by
  calc _ ≤ (rows.biUnion fun i => steps.image (F i)).card := by
        apply card_le_card
        intro y hy
        obtain ⟨⟨i, t⟩, hit, rfl⟩ := mem_image.1 hy
        rw [mem_product] at hit
        exact mem_biUnion.2 ⟨i, hit.1, mem_image.2 ⟨t, hit.2, rfl⟩⟩
    _ ≤ _ := card_biUnion_le

omit [DecidableEq A] in
/-- **Column-free increments collapse**: row-step chunks `base i + c i t · 1` number at most `Σ_i |{c i t}|`. -/
theorem card_levels_le [DecidableEq V] [Add V] {ι T : Type} (rows : Finset ι) (steps : Finset T)
    (base : ι → Fin w → V) (c : ι → T → V) :
    ((rows ×ˢ steps).image fun it => base it.1 + fun _ => c it.1 it.2).card ≤
      ∑ i ∈ rows, (steps.image (c i)).card := by
  refine (card_rows_le rows steps fun i t => base i + fun _ => c i t).trans (sum_le_sum fun i _ => ?_)
  calc (steps.image fun t => base i + fun _ => c i t).card
        = ((steps.image (c i)).image fun x => base i + fun _ => x).card := by rw [image_image]; rfl
    _ ≤ _ := card_image_le

/-- **A sequence that repeats at every step whose bit is `false` takes at most `1 + #true bits` values.** -/
theorem card_image_le_ones {X : Type} [DecidableEq X] (f : ℕ → X) (b : ℕ → Bool) (k : ℕ)
    (hf : ∀ t < k, b t = false → f (t + 1) = f t) :
    ((range (k + 1)).image f).card ≤ 1 + ((range k).filter fun t => b t).card := by
  induction k with
  | zero => simp
  | succ k ih =>
    have ih' := ih fun t ht => hf t (by omega)
    rw [range_add_one (n := k + 1), image_insert]
    cases hb : b k with
    | true =>
      have hf' : ((range (k + 1)).filter fun t => b t) = insert k ((range k).filter fun t => b t) := by
        rw [range_add_one, filter_insert]; simp [hb]
      have hk : k ∉ (range k).filter fun t => b t := by simp
      rw [hf', card_insert_of_notMem hk]
      calc _ ≤ ((range (k + 1)).image f).card + 1 := card_insert_le _ _
        _ ≤ _ := by omega
    | false =>
      have hm : f k ∈ (range (k + 1)).image f := mem_image_of_mem f (mem_range.2 (Nat.lt_succ_self k))
      rw [hf k (Nat.lt_succ_self k) hb, insert_eq_of_mem hm]
      exact ih'.trans (by gcongr; omega)

/-- **The walk's obstruction**: when every draw's targets are row-step chunks `base g i + c g i t · 1`, adaptive-dw
fails at any bound above `w·Σ_g Σ_i |{c g i t}|`. -/
theorem not_adaptiveDW_levels [DecidableEq V] [Add V] {ι T : Type} (a₀ : A) (Ω : Type) [Fintype Ω]
    (m₀ : A → Fin w → V) (rows : Finset ι) (steps : Finset T) (base : Ω → ι → Fin w → V) (c : Ω → ι → T → V)
    (B : ℕ) (hB : w * ∑ g, ∑ i ∈ rows, (steps.image (c g i)).card < B) :
    ¬ AdaptiveDW Ω m₀ (fun g => (rows ×ˢ steps).image fun it => base g it.1 + fun _ => c g it.1 it.2) B := by
  refine not_adaptiveDW a₀ Ω m₀ _ B (lt_of_le_of_lt ?_ hB)
  exact Nat.mul_le_mul_left w (sum_le_sum fun g _ => card_levels_le rows steps (base g) (c g))

/-- **The cancel's obstruction**: when every row's chunk at step `t + 1` repeats its chunk at step `t` whenever the
bit `b g i t` is `false`, adaptive-dw fails at any bound above `w·Σ_g Σ_i (1 + #{t < k : b g i t})`. -/
theorem not_adaptiveDW_ones [DecidableEq V] {ι : Type} (a₀ : A) (Ω : Type) [Fintype Ω] (m₀ : A → Fin w → V)
    (rows : Finset ι) (k : ℕ) (F : Ω → ι → ℕ → Fin w → V) (b : Ω → ι → ℕ → Bool)
    (hF : ∀ g i t, t < k → b g i t = false → F g i (t + 1) = F g i t) (B : ℕ)
    (hB : w * ∑ g, ∑ i ∈ rows, (1 + ((range k).filter fun t => b g i t).card) < B) :
    ¬ AdaptiveDW Ω m₀ (fun g => (rows ×ˢ range (k + 1)).image fun it => F g it.1 it.2) B := by
  refine not_adaptiveDW a₀ Ω m₀ _ B (lt_of_le_of_lt ?_ hB)
  exact Nat.mul_le_mul_left w (sum_le_sum fun g _ => (card_rows_le rows _ (F g)).trans
    (sum_le_sum fun i _ => card_image_le_ones (F g i) (b g i) k (hF g i)))

end Pouw.Dimension.Pointwise
