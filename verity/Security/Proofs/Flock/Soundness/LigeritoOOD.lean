import Proofs.Flock.Soundness.LigeritoLevels

/-!
# Out-of-domain binding (PROTOCOL.md §13.3 step 6)

After a level's folds, the prover commits the next level's split table and answers two
out-of-domain samples: the table's multilinear extension at two uniform points of `F^μ`
(`μ = logCols + lanes`). Two distinct tables agree at one uniform point with probability at most
`μ/|F|` (a nonzero multilinear polynomial), and at two independent points with probability at most
`(μ/|F|)²`. So, over the `L(L − 1)/2` unordered pairs of the next level's list, two distinct list
elements both match the answers with probability at most `pairs(L)·(μ/|F|)²`
(`value_twoOod_collide_le`), and at most one list element survives the samples.
-/

namespace FlockSoundness

open Game Model Accounting
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-- A table's multilinear extension at a point given as a list (as the out-of-domain contexts
store it). -/
def mleL {μ : ℕ} (g : Fin (2 ^ μ) → F) (zs : List F) : F :=
  ∑ y : Fin (2 ^ μ), eqList zs y.val * g y

/-- **One point.** Two distinct tables' extensions agree at a uniform point of `F^μ` with
probability at most `μ/|F|`. -/
theorem prCoin_mle_eq_le [CharP F 2] {μ : ℕ} (g h : Fin (2 ^ μ) → F) (hgh : g ≠ h) :
    prCoin (fun z : Fin μ → F => mleL g (List.ofFn z) = mleL h (List.ofFn z)) ≤
      (μ : ℝ≥0∞) / Fintype.card F := by
  obtain ⟨y₀, hy₀⟩ : ∃ y, g y ≠ h y := by
    by_contra hall; push Not at hall; exact hgh (funext hall)
  refine (prCoin_mono fun z hz => ?_).trans
    (prCoin_eqSum_le (fun y => g y - h y) y₀ (sub_ne_zero.2 hy₀))
  simp only [mleL, eqList_ofFn] at hz
  simp only [sub_mul, Finset.sum_sub_distrib, sub_eq_zero]
  have hc : ∀ w : Fin (2 ^ μ) → F, ∑ y, w y * eqAt z y.val = ∑ y, eqAt z y.val * w y :=
    fun w => Finset.sum_congr rfl fun y _ => mul_comm _ _
  rw [hc g, hc h]
  exact hz

open Classical in
/-- Two independent uniform points. -/
theorem avg_avg_and {α : Type} [Fintype α] (P : α → Prop) :
    avg (fun z₁ : α => avg fun z₂ : α =>
        (if P z₁ ∧ P z₂ then (1 : ℝ≥0∞) else 0)) = prCoin P * prCoin P := by
  classical
  simp only [prCoin, avg, Finset.sum_mul, Finset.mul_sum]
  refine Finset.sum_congr rfl fun x _ => Finset.sum_congr rfl fun y _ => ?_
  by_cases hx : P x <;> by_cases hy : P y <;> simp [hx, hy, mul_comm, mul_left_comm]

/-- Two distinct elements of `S` both match at two points. -/
def OodCollide {μ : ℕ} (S : Finset (Fin (2 ^ μ) → F)) (z₁ z₂ : List F) : Prop :=
  ∃ g ∈ S, ∃ h ∈ S, g ≠ h ∧ mleL g z₁ = mleL h z₁ ∧ mleL g z₂ = mleL h z₂

open Classical in
/-- **Two points, the list.** Over the unordered pairs of `S`, two distinct elements agree at two
independent uniform points with probability at most `(|S| choose 2)·(μ/|F|)²`. -/
theorem avg_avg_collide_le [CharP F 2] {μ : ℕ} (S : Finset (Fin (2 ^ μ) → F)) :
    avg (fun z₁ : Fin μ → F => avg fun z₂ : Fin μ → F =>
        (if OodCollide S (List.ofFn z₁) (List.ofFn z₂) then (1 : ℝ≥0∞) else 0)) ≤
      (S.card.choose 2 : ℝ≥0∞) * ((μ : ℝ≥0∞) / Fintype.card F) ^ 2 := by
  classical
  let E : Finset (Fin (2 ^ μ) → F) → (Fin μ → F) → Prop := fun P z =>
    ∀ g ∈ P, ∀ h ∈ P, mleL g (List.ofFn z) = mleL h (List.ofFn z)
  have hind : ∀ z₁ z₂ : Fin μ → F,
      (if OodCollide S (List.ofFn z₁) (List.ofFn z₂) then (1 : ℝ≥0∞) else 0) ≤
        ∑ P ∈ S.powersetCard 2, (if E P z₁ ∧ E P z₂ then (1 : ℝ≥0∞) else 0) := by
    intro z₁ z₂
    split_ifs with hc
    · obtain ⟨g, hg, h, hh, hgh, h1, h2⟩ := hc
      have hP : ({g, h} : Finset _) ∈ S.powersetCard 2 := by
        rw [Finset.mem_powersetCard]
        refine ⟨?_, Finset.card_pair hgh⟩
        intro x hx
        rcases Finset.mem_insert.1 hx with rfl | hx
        · exact hg
        · rw [Finset.mem_singleton.1 hx]; exact hh
      refine le_trans ?_ (Finset.single_le_sum (f := fun P => if E P z₁ ∧ E P z₂ then (1 : ℝ≥0∞) else 0)
        (fun _ _ => zero_le) hP)
      rw [if_pos]
      have pair : ∀ z : Fin μ → F, mleL g (List.ofFn z) = mleL h (List.ofFn z) → E {g, h} z := by
        intro z hz x hx y hy
        simp only [Finset.mem_insert, Finset.mem_singleton] at hx hy
        rcases hx with hx | hx <;> rcases hy with hy | hy <;> subst hx <;> subst hy
        · rfl
        · exact hz
        · exact hz.symm
        · rfl
      exact ⟨pair z₁ h1, pair z₂ h2⟩
    · exact zero_le
  have hpair : ∀ P ∈ S.powersetCard 2, avg (fun z₁ : Fin μ → F => avg fun z₂ : Fin μ → F =>
      (if E P z₁ ∧ E P z₂ then (1 : ℝ≥0∞) else 0)) ≤ ((μ : ℝ≥0∞) / Fintype.card F) ^ 2 := by
    intro P hP
    rw [Finset.mem_powersetCard, Finset.card_eq_two] at hP
    obtain ⟨_, g, h, hgh, rfl⟩ := hP
    have hb := (prCoin_mono (p := E {g, h}) fun z hz => hz g (by simp) h (by simp)).trans
      (prCoin_mle_eq_le g h hgh)
    calc avg (fun z₁ : Fin μ → F => avg fun z₂ : Fin μ → F =>
            (if E {g, h} z₁ ∧ E {g, h} z₂ then (1 : ℝ≥0∞) else 0))
        = prCoin (E {g, h}) * prCoin (E {g, h}) := by
          convert avg_avg_and (E {g, h}) using 4
          funext z₂
          congr
      _ ≤ ((μ : ℝ≥0∞) / Fintype.card F) * ((μ : ℝ≥0∞) / Fintype.card F) := mul_le_mul' hb hb
      _ = _ := (sq _).symm
  calc avg (fun z₁ : Fin μ → F => avg fun z₂ : Fin μ → F =>
          (if OodCollide S (List.ofFn z₁) (List.ofFn z₂) then (1 : ℝ≥0∞) else 0))
      ≤ avg (fun z₁ : Fin μ → F => avg fun z₂ : Fin μ → F =>
          ∑ P ∈ S.powersetCard 2, (if E P z₁ ∧ E P z₂ then (1 : ℝ≥0∞) else 0)) :=
        avg_mono fun z₁ => avg_mono fun z₂ => hind z₁ z₂
    _ = ∑ P ∈ S.powersetCard 2, avg (fun z₁ : Fin μ → F => avg fun z₂ : Fin μ → F =>
          (if E P z₁ ∧ E P z₂ then (1 : ℝ≥0∞) else 0)) := by
        simp only [avg_sum]
    _ ≤ ∑ _P ∈ S.powersetCard 2, ((μ : ℝ≥0∞) / Fintype.card F) ^ 2 := Finset.sum_le_sum hpair
    _ = (S.card.choose 2 : ℝ≥0∞) * ((μ : ℝ≥0∞) / Fintype.card F) ^ 2 := by
        rw [Finset.sum_const, Finset.card_powersetCard, nsmul_eq_mul]

/-! ## In the game -/

theorem twoOod_eq (A : Arith F K) (dim split : ℕ) (s : K) (q : Quad K) :
    twoOod A dim split s q =
      Game.coin (Fin dim → F) fun z₁ => Game.send F fun y₁ => Game.send (K × K) fun m₁ =>
        Game.coin F fun β₁ => Game.coin (Fin dim → F) fun z₂ => Game.send F fun y₂ =>
          Game.send (K × K) fun m₂ => Game.coin F fun β₂ =>
            Game.ret (s + A.embed β₁ * A.embed y₁ + A.embed β₂ * A.embed y₂,
              (q.addScaled (fromMsg m₁ (A.embed y₁)) (A.embed β₁)).addScaled
                (fromMsg m₂ (A.embed y₂)) (A.embed β₂),
              [⟨List.ofFn z₁, β₁, some split⟩, ⟨List.ofFn z₂, β₂, some split⟩]) := rfl

open Classical in
/-- **Out-of-domain binding.** Over the two samples of `twoOod` on a table with `μ` variables, two
distinct elements of `S` (the next level's list) both match the samples with probability at most
`(|S| choose 2)·(μ/|F|)²`, whatever the prover answers. -/
theorem value_twoOod_collide_le [CharP F 2] (A : Arith F K) {μ : ℕ}
    (S : Finset (Fin (2 ^ μ) → F)) (split : ℕ) (s : K) (q : Quad K) :
    value (fun out : K × Quad K × List (OodCtx F) =>
        ∃ o₁ o₂, out.2.2 = [o₁, o₂] ∧ OodCollide S o₁.z o₂.z) (twoOod A μ split s q) ≤
      (S.card.choose 2 : ℝ≥0∞) * ((μ : ℝ≥0∞) / Fintype.card F) ^ 2 := by
  rw [twoOod_eq]
  refine (value_coin_le_avg _ _ _ (fun z₁ => avg fun z₂ : Fin μ → F =>
      if OodCollide S (List.ofFn z₁) (List.ofFn z₂) then (1 : ℝ≥0∞) else 0) fun z₁ => ?_).trans
    (avg_avg_collide_le S)
  rw [value_send]; refine iSup_le fun y₁ => ?_
  rw [value_send]; refine iSup_le fun m₁ => ?_
  refine value_coin_le_of_forall _ _ _ _ fun β₁ => ?_
  refine value_coin_le_avg _ _ _ _ fun z₂ => ?_
  rw [value_send]; refine iSup_le fun y₂ => ?_
  rw [value_send]; refine iSup_le fun m₂ => ?_
  refine value_coin_le_of_forall _ _ _ _ fun β₂ => ?_
  rw [value_ret]
  split_ifs with h h'
  · exact le_rfl
  · obtain ⟨o₁, o₂, ho, hc⟩ := h
    simp only [List.cons.injEq, and_true] at ho
    obtain ⟨rfl, rfl⟩ := ho
    exact absurd hc h'
  · exact zero_le
  · exact zero_le

end FlockSoundness
