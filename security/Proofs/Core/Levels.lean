module

public import Definitions.Core.Levels
public import Proofs.Core.Game.Prob
public import Proofs.Core.Game.Tape

/-!
# Levels: the level-by-level sample's miss probability

`SamplingDesign.missProb_bind`: a set `T` escapes `D.bind F` with probability `E_{ω ∼ D}[∏_{u ∈ D.draw ω} missProb
(F u) T]`, since each sampled `u` is drawn on coins of its own. From it:

* one level is its design (`Levels.prCoin_one`: every event of the sample has the probability it has under `D₁`;
  `Levels.missProb_one`, `Levels.miss_one`);
* two levels give `missProb D T = E_{Q₁ ∼ D₁}[∏_{X ∈ Q₁} missProb D₂^X (T_X)]`, `T_X` the blocks of `T` inside `X`
  (`Levels.missProb_two`), and in general the first level's draw and the sample below each of its blocks
  (`Levels.missProb_cons`);
* the expectation over a design's draw of a product, written as a sum over the sets it misses
  (`SamplingDesign.avg_prod_draw`): `E[∏_{u ∈ Q} e_u] = Σ_{R ⊆ A} missProb R · ∏_{u ∈ R}(1 − e_u) · ∏_{u ∈ A \ R} e_u`
  for draws inside `A` and `e ≤ 1`, the form of `Claims.Sampling.Design.Two.Pa`.
-/

@[expose] public section

namespace Verity

open Game
open scoped ENNReal

namespace SamplingDesign

theorem missProb_bind {U V : Type} [Fintype U] [DecidableEq U] [DecidableEq V] (D : SamplingDesign U)
    (F : U → SamplingDesign V) (T : Finset V) :
    (D.bind F).missProb T = avg fun ω => ∏ u ∈ D.draw ω, (F u).missProb T := by
  show prCoin (fun ω : D.Ω × ((u : U) → (F u).Ω) =>
      Disjoint ((D.draw ω.1).biUnion fun u => (F u).draw (ω.2 u)) T) =
    avg fun ω => ∏ u ∈ D.draw ω, prCoin fun c => Disjoint ((F u).draw c) T
  simp only [Finset.disjoint_biUnion_left]
  calc prCoin (fun ω : D.Ω × ((u : U) → (F u).Ω) => ∀ u ∈ D.draw ω.1, Disjoint ((F u).draw (ω.2 u)) T)
      = avg fun ω₁ => prCoin fun ω₂ : (u : U) → (F u).Ω => ∀ u ∈ D.draw ω₁, Disjoint ((F u).draw (ω₂ u)) T := by
        unfold prCoin
        exact Tape.avg_prod _
    _ = _ := congrArg avg (funext fun ω₁ =>
        prCoin_pi_forall (D.draw ω₁) fun u (c : (F u).Ω) => Disjoint ((F u).draw c) T)

theorem bind_const_draw {U : Type} [Fintype U] [DecidableEq U] (D : SamplingDesign U)
    (ω : (D.bind fun u => const {u}).Ω) : (D.bind fun u => const {u}).draw ω = D.draw ω.1 :=
  Finset.biUnion_singleton_eq_self

/-- Drawing from `D` and then each sampled unit alone is `D`: every event of the draw has its probability under `D`. -/
theorem prCoin_bind_const {U : Type} [Fintype U] [DecidableEq U] (D : SamplingDesign U) (P : Finset U → Prop) :
    prCoin (fun ω => P ((D.bind fun u => const {u}).draw ω)) = prCoin fun ω => P (D.draw ω) := by
  classical
  simp only [bind_const_draw]
  show prCoin (fun ω : D.Ω × ((u : U) → (const ({u} : Finset U)).Ω) => P (D.draw ω.1)) = _
  unfold prCoin
  rw [Tape.avg_prod]
  refine congrArg avg (funext fun ω => ?_)
  exact avg_const (C := (u : U) → (const ({u} : Finset U)).Ω) (if P (D.draw ω) then (1 : ℝ≥0∞) else 0)

/-- A design whose draws satisfy `p` misses `T` exactly when it misses `T`'s members satisfying `p`. -/
theorem missProb_filter {U : Type} (D : SamplingDesign U) (p : U → Prop) [DecidablePred p]
    (h : ∀ ω, ∀ u ∈ D.draw ω, p u) (T : Finset U) : D.missProb (T.filter p) = D.missProb T := by
  unfold missProb
  congr 1
  funext ω
  refine propext ⟨fun hd => Finset.disjoint_left.2 fun u hu hT => Finset.disjoint_left.1 hd hu
    (Finset.mem_filter.2 ⟨hT, h ω u hu⟩), fun hd => Finset.disjoint_left.2 fun u hu hT =>
    Finset.disjoint_left.1 hd hu (Finset.mem_filter.1 hT).1⟩

/-- **The expectation of a product over the draw, by the sets it misses**: for draws inside `A` and `e ≤ 1` on `A`,
`E[∏_{u ∈ Q} e_u] = Σ_{R ⊆ A} missProb R · ∏_{u ∈ R}(1 − e_u) · ∏_{u ∈ A \ R} e_u`. -/
theorem avg_prod_draw {U : Type} [DecidableEq U] (D : SamplingDesign U) (A : Finset U) (hA : ∀ ω, D.draw ω ⊆ A)
    (e : U → ℝ≥0∞) (he : ∀ u ∈ A, e u ≤ 1) :
    avg (fun ω => ∏ u ∈ D.draw ω, e u) =
      ∑ R ∈ A.powerset, D.missProb R * ((∏ u ∈ R, (1 - e u)) * ∏ u ∈ A \ R, e u) := by
  classical
  have point : ∀ S ⊆ A, ∏ u ∈ S, e u =
      ∑ R ∈ A.powerset, (if Disjoint S R then (1 : ℝ≥0∞) else 0) * ((∏ u ∈ R, (1 - e u)) * ∏ u ∈ A \ R, e u) := by
    intro S hS
    simp only [ite_mul, one_mul, zero_mul]
    rw [← Finset.sum_filter]
    have hfilter : A.powerset.filter (fun R => Disjoint S R) = (A \ S).powerset := by
      ext R
      simp only [Finset.mem_filter, Finset.mem_powerset, Finset.subset_sdiff]
      exact and_congr_right fun _ => disjoint_comm
    rw [hfilter]
    have hsplit : ∀ R ∈ (A \ S).powerset, ∏ u ∈ A \ R, e u = (∏ u ∈ S, e u) * ∏ u ∈ (A \ S) \ R, e u := by
      intro R hR
      rw [Finset.mem_powerset] at hR
      have hdisj : Disjoint S ((A \ S) \ R) :=
        Finset.disjoint_left.2 fun u hu h' => (Finset.mem_sdiff.1 (Finset.mem_sdiff.1 h').1).2 hu
      have hunion : A \ R = S ∪ ((A \ S) \ R) := by
        ext u
        simp only [Finset.mem_sdiff, Finset.mem_union]
        constructor
        · rintro ⟨huA, huR⟩
          by_cases huS : u ∈ S
          · exact Or.inl huS
          · exact Or.inr ⟨⟨huA, huS⟩, huR⟩
        · rintro (huS | ⟨⟨huA, -⟩, huR⟩)
          · exact ⟨hS huS, fun huR => (Finset.mem_sdiff.1 (hR huR)).2 huS⟩
          · exact ⟨huA, huR⟩
      rw [hunion, Finset.prod_union hdisj]
    have hone : ∏ u ∈ A \ S, ((1 - e u) + e u) = 1 :=
      Finset.prod_eq_one fun u hu => tsub_add_cancel_of_le (he u (Finset.mem_sdiff.1 hu).1)
    rw [Finset.prod_add] at hone
    calc ∏ u ∈ S, e u = (∏ u ∈ S, e u) * 1 := (mul_one _).symm
      _ = (∏ u ∈ S, e u) * ∑ t ∈ (A \ S).powerset, (∏ u ∈ t, (1 - e u)) * ∏ u ∈ (A \ S) \ t, e u := by
          rw [hone]
      _ = _ := by
          rw [Finset.mul_sum]
          refine Finset.sum_congr rfl fun R hR => ?_
          rw [hsplit R hR]
          ring
  calc avg (fun ω => ∏ u ∈ D.draw ω, e u)
      = avg (fun ω => ∑ R ∈ A.powerset,
          (if Disjoint (D.draw ω) R then (1 : ℝ≥0∞) else 0) * ((∏ u ∈ R, (1 - e u)) * ∏ u ∈ A \ R, e u)) :=
        congrArg avg (funext fun ω => point _ (hA ω))
    _ = ∑ R ∈ A.powerset, D.missProb R * ((∏ u ∈ R, (1 - e u)) * ∏ u ∈ A \ R, e u) := by
        unfold missProb prCoin avg
        simp only [Finset.mul_sum, Finset.sum_mul]
        rw [Finset.sum_comm]
        refine Finset.sum_congr rfl fun R _ => Finset.sum_congr rfl fun ω _ => ?_
        by_cases h : Disjoint (D.draw ω) R <;> simp [h]

end SamplingDesign

namespace Level

variable {G : Type} [Fintype G] [DecidableEq G]

theorem prCoin_sampleBelow_one (l : Level G) (X : Finset G) (P : Finset (Finset G) → Prop) :
    prCoin (fun ω => P ((sampleBelow [l] X).draw ω)) = prCoin fun ω => P ((l.design X).draw ω) :=
  SamplingDesign.prCoin_bind_const (l.design X) P

end Level

namespace Levels

variable {G : Type} [Fintype G] [DecidableEq G]

/-- **One level is its design**: every event of the sample has the probability it has under `D₁`. -/
theorem prCoin_one (L : Levels G) {l : Level G} (hL : L.levels = [l]) (P : Finset (Finset G) → Prop) :
    prCoin (fun ω => P (L.design.draw ω)) = prCoin fun ω => P ((l.design Finset.univ).draw ω) := by
  obtain ⟨levels, hn, hi⟩ := L
  dsimp only at hL
  subst hL
  exact Level.prCoin_sampleBelow_one l Finset.univ P

/-- **One level's miss is its design's.** -/
theorem missProb_one (L : Levels G) {l : Level G} (hL : L.levels = [l]) (T : Finset (Finset G)) :
    L.design.missProb T = (l.design Finset.univ).missProb T :=
  L.prCoin_one hL fun S => Disjoint S T

theorem miss_one (L : Levels G) {l : Level G} (hL : L.levels = [l]) (K : ℕ) :
    L.design.miss K = (l.design Finset.univ).miss K := by
  simp only [SamplingDesign.miss, L.missProb_one hL]

/-- The first level's draw, then the sample of the levels below at each of its blocks. -/
theorem missProb_cons (L : Levels G) {l : Level G} {ls : List (Level G)} (hL : L.levels = l :: ls)
    (T : Finset (Finset G)) :
    L.design.missProb T =
      avg fun ω => ∏ X ∈ (l.design Finset.univ).draw ω, (Level.sampleBelow ls X).missProb T := by
  obtain ⟨levels, hn, hi⟩ := L
  dsimp only at hL
  subst hL
  exact SamplingDesign.missProb_bind _ _ T

/-- **Two levels**: `missProb D T = E_{Q₁ ∼ D₁}[∏_{X ∈ Q₁} missProb D₂^X (T_X)]`, `T_X` the blocks of `T` inside
`X`. -/
theorem missProb_two (L : Levels G) {l₁ l₂ : Level G} (hL : L.levels = [l₁, l₂]) (T : Finset (Finset G)) :
    L.design.missProb T = avg fun ω : (l₁.design Finset.univ).Ω =>
      ∏ X ∈ (l₁.design Finset.univ).draw ω, (l₂.design X).missProb (T.filter (· ⊆ X)) := by
  have hi := L.inside
  rw [hL] at hi
  simp only [Level.InsideBelow, Finset.mem_singleton, forall_eq, and_true] at hi
  obtain ⟨h₁, h₂⟩ := hi
  rw [L.missProb_cons hL]
  refine congrArg avg (funext fun ω => Finset.prod_congr rfl fun X hX => ?_)
  rw [SamplingDesign.missProb_filter (l₂.design X) (· ⊆ X)
    (fun ω' b hb => (h₂ X (h₁ ω X hX).1 ω' b hb).2) T]
  exact Level.prCoin_sampleBelow_one l₂ X fun S => Disjoint S T

end Levels

end Verity
