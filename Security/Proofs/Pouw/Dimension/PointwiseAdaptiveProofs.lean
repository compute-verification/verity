import Proofs.Pouw.Dimension.PointwiseProofs

/-!
# The chunk model against a program that branches on the coins (proofs; draft model, pending review)

At each coin outcome `ω` such a program is a run `P ω` from the memory `m₀ ω` that holds the targets `Y ω` at some
point, as in `adaptive_pairs_le`. Each bound reads only the targets an outcome realizes, so it sums over any outcomes.

* **Chunk provenance** (`run_chunks_le`, `adaptive_chunks_le`). By `good_prefix`, every slot after every prefix holds
  an initial chunk or one of `writes P` new ones, so the targets no initial chunk holds number at most `writes P`.
* **Balls against any initial memory** (`run_ball_table_le`, `adaptive_ball_table_le`). An initial chunk within `r y`
  of a target is within it of no other, so `run_ball_le` holds with no hypothesis on `m₀`, less `R + 1` a slot. Its
  count is at most `⌊(w − 1)/2⌋ + 1` a target, about half of the target's `w` words.
* **The attained side** (`pathRun_le_card`, `adaptive_attained`, `adaptive_levels_attained`,
  `adaptive_ones_attained`). The path run (`pathRun_attained`) holds any targets at `w` writes a distinct value.
  Row-step chunks `base i + c i t · 1` number at most `Σ_i |{c i t}|`, and a row whose chunk repeats at every step
  whose coin bit is `false` takes at most `1 + #true bits` values.
-/

namespace Pouw.Dimension.Pointwise

open Finset

variable {A V : Type} {w : ℕ}

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

theorem pathLen_le [DecidableEq V] (x : Fin w → V) (ys : List (Fin w → V)) : pathLen x ys ≤ w * ys.length := by
  induction ys generalizing x with
  | nil => simp [pathLen]
  | cons y ys ih =>
    have h : hammingDist x y ≤ w := (hammingDist_le_card_fintype).trans_eq (Fintype.card_fin w)
    have := ih y
    simp only [pathLen, List.length_cons, Nat.mul_succ]
    omega

variable [DecidableEq A]

open Classical in
/-- **A run writes at least its targets' new chunk values**, whatever the program. -/
theorem run_chunks_le [DecidableEq V] [Inhabited V] (m₀ : A → Fin w → V) (P : List (Step A V w))
    (Y : Finset (Fin w → V)) (hY : ∀ y ∈ Y, ∃ i a, run m₀ (P.take i) a = y) :
    (Y.filter fun y => ∀ a, m₀ a ≠ y).card ≤ writes P := by
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
/-- **Branching on the coins, chunk by chunk**: summed over any outcomes, a program writes at least each outcome's new
target chunk values. -/
theorem adaptive_chunks_le [DecidableEq V] [Inhabited V] {Ω : Type} (O : Finset Ω) (m₀ : Ω → A → Fin w → V)
    (P : Ω → List (Step A V w)) (Y : Ω → Finset (Fin w → V))
    (hY : ∀ ω ∈ O, ∀ y ∈ Y ω, ∃ i a, run (m₀ ω) ((P ω).take i) a = y) :
    ∑ ω ∈ O, ((Y ω).filter fun y => ∀ a, m₀ ω a ≠ y).card ≤ ∑ ω ∈ O, writes (P ω) :=
  sum_le_sum fun ω hω => run_chunks_le (m₀ ω) (P ω) (Y ω) (hY ω hω)

/-- **Balls against any initial memory**: targets pairwise further apart than the sum of their radii, each radius at
most `R`, need `Σ_y (r y + 1)` writes less `R + 1` a slot. -/
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

/-- **Branching on the coins, ball by ball**: summed over any outcomes, `run_ball_table_le`'s count, less `R + 1` a
slot an outcome, whatever each outcome's initial memory holds. -/
theorem adaptive_ball_table_le [Fintype A] [DecidableEq V] [Inhabited V] {Ω : Type} (O : Finset Ω)
    (m₀ : Ω → A → Fin w → V) (P : Ω → List (Step A V w)) (Y : Ω → Finset (Fin w → V)) (r : Ω → (Fin w → V) → ℕ)
    (R : ℕ) (hY : ∀ ω ∈ O, ∀ y ∈ Y ω, ∃ i a, run (m₀ ω) ((P ω).take i) a = y)
    (hR : ∀ ω ∈ O, ∀ y ∈ Y ω, r ω y ≤ R)
    (hYY : ∀ ω ∈ O, ∀ y ∈ Y ω, ∀ y' ∈ Y ω, y ≠ y' → r ω y + r ω y' < hammingDist y y') :
    ∑ ω ∈ O, ∑ y ∈ Y ω, (r ω y + 1) ≤ ∑ ω ∈ O, writes (P ω) + O.card * (Fintype.card A * (R + 1)) := by
  have h := sum_le_sum fun ω hω =>
    run_ball_table_le (m₀ ω) (P ω) (Y ω) (hY ω hω) (r ω) R (hR ω hω) (hYY ω hω)
  rwa [sum_add_distrib, sum_const, smul_eq_mul] at h

/-- **Any targets are held at `w` writes a distinct value**: the path run through them in one slot. -/
theorem pathRun_le_card [DecidableEq V] (m₀ : A → Fin w → V) (a : A) (Y : Finset (Fin w → V)) :
    (∀ y ∈ Y, ∃ i, run m₀ ((pathRun a (m₀ a) Y.toList).take i) a = y) ∧
      writes (pathRun a (m₀ a) Y.toList) ≤ w * Y.card := by
  obtain ⟨h1, h2⟩ := pathRun_attained m₀ a Y.toList
  exact ⟨fun y hy => h1 y (mem_toList.2 hy), h2 ▸ (pathLen_le _ _).trans_eq (by rw [length_toList])⟩

/-- **The attained side over the outcomes**: path runs hold every outcome's targets at `w` writes a distinct value. -/
theorem adaptive_attained [DecidableEq V] {Ω : Type} (O : Finset Ω) (m₀ : Ω → A → Fin w → V) (a : A)
    (Y : Ω → Finset (Fin w → V)) :
    ∃ P : Ω → List (Step A V w), (∀ ω ∈ O, ∀ y ∈ Y ω, ∃ i a, run (m₀ ω) ((P ω).take i) a = y) ∧
      ∑ ω ∈ O, writes (P ω) ≤ w * ∑ ω ∈ O, (Y ω).card := by
  refine ⟨fun ω => pathRun a (m₀ ω a) (Y ω).toList, fun ω _ y hy => ?_, ?_⟩
  · obtain ⟨i, hi⟩ := (pathRun_le_card (m₀ ω) a (Y ω)).1 y hy
    exact ⟨i, a, hi⟩
  · rw [mul_sum]
    exact sum_le_sum fun ω _ => (pathRun_le_card (m₀ ω) a (Y ω)).2

/-- **The walk's obstruction**: when every outcome's targets are row-step chunks `base ω i + c ω i t · 1`, some program
holds them all at `w·Σ_ω Σ_i |{c ω i t}|` writes. -/
theorem adaptive_levels_attained [DecidableEq V] [Add V] {Ω ι T : Type} (O : Finset Ω) (m₀ : Ω → A → Fin w → V)
    (a : A) (rows : Finset ι) (steps : Finset T) (base : Ω → ι → Fin w → V) (c : Ω → ι → T → V) :
    ∃ P : Ω → List (Step A V w),
      (∀ ω ∈ O, ∀ y ∈ (rows ×ˢ steps).image (fun it => base ω it.1 + fun _ => c ω it.1 it.2),
        ∃ i a, run (m₀ ω) ((P ω).take i) a = y) ∧
      ∑ ω ∈ O, writes (P ω) ≤ w * ∑ ω ∈ O, ∑ i ∈ rows, (steps.image (c ω i)).card := by
  obtain ⟨P, hP, hw⟩ :=
    adaptive_attained O m₀ a fun ω => (rows ×ˢ steps).image fun it => base ω it.1 + fun _ => c ω it.1 it.2
  exact ⟨P, hP, hw.trans (Nat.mul_le_mul_left w (sum_le_sum fun ω _ => card_levels_le rows steps (base ω) (c ω)))⟩

/-- **The cancel's obstruction**: when every row's chunk at step `t + 1` repeats its chunk at step `t` whenever the
coin bit `b ω i t` is `false`, some program holds every outcome's row-step chunks at
`w·Σ_ω Σ_i (1 + #{t < k : b ω i t})` writes. -/
theorem adaptive_ones_attained [DecidableEq V] {Ω ι : Type} (O : Finset Ω) (m₀ : Ω → A → Fin w → V) (a : A)
    (rows : Finset ι) (k : ℕ) (F : Ω → ι → ℕ → Fin w → V) (b : Ω → ι → ℕ → Bool)
    (hF : ∀ ω i t, t < k → b ω i t = false → F ω i (t + 1) = F ω i t) :
    ∃ P : Ω → List (Step A V w),
      (∀ ω ∈ O, ∀ y ∈ (rows ×ˢ range (k + 1)).image (fun it => F ω it.1 it.2),
        ∃ i a, run (m₀ ω) ((P ω).take i) a = y) ∧
      ∑ ω ∈ O, writes (P ω) ≤ w * ∑ ω ∈ O, ∑ i ∈ rows, (1 + ((range k).filter fun t => b ω i t).card) := by
  obtain ⟨P, hP, hw⟩ := adaptive_attained O m₀ a fun ω => (rows ×ˢ range (k + 1)).image fun it => F ω it.1 it.2
  exact ⟨P, hP, hw.trans (Nat.mul_le_mul_left w (sum_le_sum fun ω _ => (card_rows_le rows _ (F ω)).trans
    (sum_le_sum fun i _ => card_image_le_ones (F ω i) (b ω i) k (hF ω i))))⟩

end Pouw.Dimension.Pointwise
