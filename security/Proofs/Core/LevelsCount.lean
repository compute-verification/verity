module

public import Definitions.Core.LevelsCount
public import Proofs.Core.LevelsTwo

/-!
# The count at δ over levels

`Levels.countAt_le_iff`: the count at `δ` is the least count bound, so at most `K` exactly when every set of more than
`K` of the last level's blocks is missed with probability below `δ`. `Levels.isCountBound_one_iff`: at one level that
is the level's own design's count over its blocks. `Levels.countAt_anti` and `Levels.shareAt_anti`: both are
antitone in `δ`.

Two levels: `Levels.missProb_two_le_prod` bounds the miss by `∏_{X ∈ B₁} (r_X (1 − m_X) + m_X)` when the first design
misses each set `R` of outer blocks with probability at most `∏_{X ∈ R} r_X` (with equality for independent outer
draws, a Bernoulli level), `m_X` the miss of `T`'s blocks inside `X`. `Levels.countAt_two_le`: when the design inside
each outer block `X` misses `j` of its blocks with probability at most `a_X(j)`, a `K` that bounds `∑_X j_X` over every
`j` with `δ ≤ ∏_X (r_X (1 − a_X(j_X)) + a_X(j_X))` bounds the count at `δ`.
-/

@[expose] public section

namespace Verity

open Game
open scoped ENNReal

namespace Levels

variable {G : Type} [Fintype G] [DecidableEq G]

theorem isCountBound_countAt (L : Levels G) (δ : ℝ≥0∞) : L.IsCountBound δ (L.countAt δ) :=
  fun _ hT hδ => Finset.le_sup (f := Finset.card) (Finset.mem_filter.2 ⟨Finset.mem_powerset.2 hT, hδ⟩)

/-- **The count at `δ` is the least count bound.** -/
theorem countAt_le_iff (L : Levels G) (δ : ℝ≥0∞) (K : ℕ) : L.countAt δ ≤ K ↔ L.IsCountBound δ K := by
  refine ⟨fun h T hT hδ => (L.isCountBound_countAt δ T hT hδ).trans h, fun h => Finset.sup_le fun T hT => ?_⟩
  obtain ⟨hT, hδ⟩ := Finset.mem_filter.1 hT
  exact h T (Finset.mem_powerset.1 hT) hδ

/-- The count curve's form: at most `K` exactly when every set of more than `K` blocks is missed with probability
below `δ`. -/
theorem countAt_le_iff_lt (L : Levels G) (δ : ℝ≥0∞) (K : ℕ) :
    L.countAt δ ≤ K ↔ ∀ T ⊆ L.blocks, K + 1 ≤ T.card → L.design.missProb T < δ := by
  rw [countAt_le_iff]
  refine ⟨fun h T hT hK => lt_of_not_ge fun hδ => by have := h T hT hδ; omega, fun h T hT hδ => ?_⟩
  by_contra hK
  exact absurd hδ (not_le.2 (h T hT (by omega)))

theorem blocks_one (L : Levels G) {l : Level G} (hL : L.levels = [l]) : L.blocks = l.blocks.parts := by
  simp [blocks, hL]

theorem blocks_two (L : Levels G) {l₁ l₂ : Level G} (hL : L.levels = [l₁, l₂]) : L.blocks = l₂.blocks.parts := by
  simp [blocks, hL]

/-- **One level**: a count bound of the levels is one of the level's own design over its blocks. -/
theorem isCountBound_one_iff (L : Levels G) {l : Level G} (hL : L.levels = [l]) (δ : ℝ≥0∞) (K : ℕ) :
    L.IsCountBound δ K ↔ ∀ T ⊆ l.blocks.parts, δ ≤ (l.design Finset.univ).missProb T → T.card ≤ K := by
  simp only [IsCountBound, L.blocks_one hL, L.missProb_one hL]

theorem isCountBound_anti (L : Levels G) {δ δ' : ℝ≥0∞} (h : δ ≤ δ') {K : ℕ} (hK : L.IsCountBound δ K) :
    L.IsCountBound δ' K :=
  fun T hT hδ => hK T hT (h.trans hδ)

/-- **Antitone in `δ`**: a larger `δ` admits fewer sets. -/
theorem countAt_anti (L : Levels G) {δ δ' : ℝ≥0∞} (h : δ ≤ δ') : L.countAt δ' ≤ L.countAt δ :=
  (L.countAt_le_iff δ' _).2 (L.isCountBound_anti h (L.isCountBound_countAt δ))

theorem shareAt_anti (L : Levels G) {δ δ' : ℝ≥0∞} (h : δ ≤ δ') : L.shareAt δ' ≤ L.shareAt δ := by
  unfold shareAt
  gcongr
  exact L.countAt_anti h

/-- **Two levels, through each outer block's pass probability**: when the first design misses each set `R` of outer
blocks with probability at most `∏_{X ∈ R} r_X`, the sample misses `T` with probability at most
`∏_{X ∈ B₁} (r_X (1 − m_X) + m_X)`, `m_X` the miss of `T`'s blocks inside `X`. -/
theorem missProb_two_le_prod (L : Levels G) {l₁ l₂ : Level G} (hL : L.levels = [l₁, l₂]) (r : Finset G → ℝ≥0∞)
    (h₁ : ∀ R ⊆ l₁.blocks.parts, (l₁.design Finset.univ).missProb R ≤ ∏ X ∈ R, r X) (T : Finset (Finset G)) :
    L.design.missProb T ≤ ∏ X ∈ l₁.blocks.parts,
      (r X * (1 - (l₂.design X).missProb (T.filter (· ⊆ X))) + (l₂.design X).missProb (T.filter (· ⊆ X))) := by
  classical
  have hi := L.inside
  rw [hL] at hi
  simp only [Level.InsideBelow, Finset.mem_singleton, forall_eq, and_true] at hi
  set m := fun X => (l₂.design X).missProb (T.filter (· ⊆ X))
  rw [L.missProb_two hL, SamplingDesign.avg_prod_draw (l₁.design Finset.univ) l₁.blocks.parts
    (fun ω b hb => (hi.1 ω b hb).1) m (fun X _ => SamplingDesign.missProb_le_one _ _),
    Finset.prod_add]
  refine Finset.sum_le_sum fun R hR => ?_
  rw [Finset.prod_mul_distrib, mul_assoc]
  exact mul_le_mul_left (h₁ R (Finset.mem_powerset.1 hR)) _

theorem pass_mono {r a a' : ℝ≥0∞} (hr : r ≤ 1) (h : a ≤ a') (ha' : a' ≤ 1) : r * (1 - a) + a ≤ r * (1 - a') + a' := by
  have hsplit : 1 - a = (1 - a') + (a' - a) := (tsub_add_tsub_cancel ha' h).symm
  calc r * (1 - a) + a = r * (1 - a') + (r * (a' - a) + a) := by rw [hsplit, mul_add, add_assoc]
    _ ≤ r * (1 - a') + ((a' - a) + a) := by gcongr; exact mul_le_of_le_one_left zero_le hr
    _ = r * (1 - a') + a' := by rw [tsub_add_cancel_of_le h]

theorem card_eq_sum_filter (L : Levels G) {l₁ l₂ : Level G} (hL : L.levels = [l₁, l₂]) {T : Finset (Finset G)}
    (hT : T ⊆ l₂.blocks.parts) : T.card = ∑ X ∈ l₁.blocks.parts, (T.filter (· ⊆ X)).card := by
  classical
  have hn := L.nested
  rw [hL] at hn
  simp only [Level.NestedBelow, and_true] at hn
  simp only [Finset.card_filter]
  rw [Finset.sum_comm, Finset.card_eq_sum_ones]
  refine Finset.sum_congr rfl fun b hb => ?_
  obtain ⟨X, hX, hbX⟩ := hn.2 b (hT hb)
  obtain ⟨g, hg⟩ := l₂.blocks.nonempty_of_mem_parts (hT hb)
  rw [← Finset.card_filter, eq_comm, Finset.card_eq_one]
  refine ⟨X, Finset.ext fun Y => ?_⟩
  simp only [Finset.mem_filter, Finset.mem_singleton]
  exact ⟨fun ⟨hY, hbY⟩ => l₁.blocks.eq_of_mem_parts hY hX (hbY hg) (hbX hg), fun h => h ▸ ⟨hX, hbX⟩⟩

/-- **Two levels, a computable count bound**: when the first design misses each set `R` of outer blocks with
probability at most `∏_{X ∈ R} r_X`, `r_X ≤ 1`, and the design inside each outer block `X` misses any `j` of its blocks
with probability at most `a_X(j) ≤ 1`, every `K` bounding `∑_X j_X` over each `j` with
`δ ≤ ∏_X (r_X (1 − a_X(j_X)) + a_X(j_X))` bounds the count at `δ`. -/
theorem countAt_two_le (L : Levels G) {l₁ l₂ : Level G} (hL : L.levels = [l₁, l₂]) (r : Finset G → ℝ≥0∞)
    (hr : ∀ X, r X ≤ 1) (h₁ : ∀ R ⊆ l₁.blocks.parts, (l₁.design Finset.univ).missProb R ≤ ∏ X ∈ R, r X)
    (a : Finset G → ℕ → ℝ≥0∞) (ha : ∀ X j, a X j ≤ 1)
    (h₂ : ∀ X ∈ l₁.blocks.parts, ∀ S ⊆ l₂.blocks.parts, (∀ b ∈ S, b ⊆ X) →
      (l₂.design X).missProb S ≤ a X S.card)
    (δ : ℝ≥0∞) (K : ℕ)
    (hK : ∀ j : Finset G → ℕ, δ ≤ ∏ X ∈ l₁.blocks.parts, (r X * (1 - a X (j X)) + a X (j X)) →
      ∑ X ∈ l₁.blocks.parts, j X ≤ K) :
    L.countAt δ ≤ K := by
  classical
  rw [countAt_le_iff]
  intro T hT hδ
  rw [L.blocks_two hL] at hT
  rw [L.card_eq_sum_filter hL hT]
  refine hK (fun X => (T.filter (· ⊆ X)).card) ((hδ.trans (L.missProb_two_le_prod hL r h₁ T)).trans ?_)
  refine Finset.prod_le_prod fun X hX => pass_mono (hr X) ?_ (ha X _)
  exact h₂ X hX _ ((Finset.filter_subset _ _).trans hT) fun b hb => (Finset.mem_filter.1 hb).2

end Levels

end Verity
