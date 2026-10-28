import Mathlib.Algebra.BigOperators.Group.Finset.Sigma
import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Algebra.Order.BigOperators.Group.Finset
import Mathlib.Data.Fintype.BigOperators
import Definitions.Pous
import Definitions.Pous.Assumptions
import Proofs.Pous.Statements
import Proofs.Pous.Sanity
import Proofs.Pous.Accounting.Numbers

/-!
# B3: hash-chain sequentiality in the parallel-query model

The draft `PousProofs.b3_chain`, stated verbatim (with its `chain`) and proved.

Write `c_j = (j, x_j)` for the `j`-th chain query, so `x_{j+1} = H(c_j)`. For `1 ≤ i ≤ D`, charge step `i` to the
queries the program makes in rounds up to and including the first round that asks `c_{i-1}` (all rounds, if it never
does); call that list `pre c_{i-1}`. Changing `H` at `c_{i-1}` changes neither `c_{i-1}` nor `pre c_{i-1}`, while
`x_i` runs over all of `Fin m`; so `c_i ∈ pre c_{i-1}` has probability at most `E[#(queries of tweak i)]/m`.
The tweaks of different steps differ, so the charges sum to at most the work, `≤ D·q`. If no step is charged, the
first rounds asking `c_0, …, c_D` strictly increase, which `D` rounds cannot fit: `c_D` is never asked, and the
output hits `x_{D+1} = H(c_D)` with probability `1/m`.

Only the total work enters, so the same proof gives `(Q + 1)/m` for `OracleProg.Bounded D Q` (`b3_chain_bounded`).
-/

namespace PousPebbling

open Pous
open scoped ENNReal

/-- The index-tweaked hash chain from `x₀`: `x_{i+1} = H(i, x_i)` (verbatim from `PousProofs.chain`). -/
def chain {m D : ℕ} (H : Fin (D + 1) × Fin m → Fin m) (x₀ : Fin m) : ℕ → Fin m
  | 0 => x₀
  | i + 1 => H (⟨i % (D + 1), Nat.mod_lt _ (Nat.succ_pos D)⟩, chain H x₀ i)

/-! ### Queries of a program along the path it takes -/

section Queries

variable {M : IdealModel} {α : Type}

/-- Every query, in order, on the path taken when queries are answered by `f`. -/
def allQ (f : M.Qry → M.Ans) : OracleProg M α → List M.Qry
  | .ret _ => []
  | .round qs k => qs ++ allQ f (k (qs.map f))

/-- The queries of the rounds up to and including the first round that asks `p` (all of them if none does). -/
def pre [DecidableEq M.Qry] (p : M.Qry) (f : M.Qry → M.Ans) : OracleProg M α → List M.Qry
  | .ret _ => []
  | .round qs k => qs ++ if p ∈ qs then [] else pre p f (k (qs.map f))

theorem length_allQ (f : M.Qry → M.Ans) (P : OracleProg M α) : (allQ f P).length = P.work f := by
  induction P with
  | ret a => rfl
  | round qs k ih => simp [allQ, OracleProg.work, ih]

theorem work_le_depth_mul_width (f : M.Qry → M.Ans) (P : OracleProg M α) : P.work f ≤ P.depth f * P.width f := by
  induction P with
  | ret a => simp [OracleProg.work]
  | round qs k ih =>
    simp only [OracleProg.work, OracleProg.depth, OracleProg.width]
    have h1 := ih (qs.map f)
    have h2 : (k (qs.map f)).depth f * (k (qs.map f)).width f ≤
        (k (qs.map f)).depth f * max qs.length ((k (qs.map f)).width f) :=
      Nat.mul_le_mul_left _ (le_max_right _ _)
    have h3 : qs.length ≤ max qs.length ((k (qs.map f)).width f) := le_max_left _ _
    rw [Nat.succ_mul]
    omega

theorem mem_allQ_of_mem_pre [DecidableEq M.Qry] {p x : M.Qry} (f : M.Qry → M.Ans) (P : OracleProg M α)
    (h : x ∈ pre p f P) : x ∈ allQ f P := by
  induction P with
  | ret a => simp [pre] at h
  | round qs k ih =>
    simp only [pre, allQ, List.mem_append] at h ⊢
    rcases h with h | h
    · exact Or.inl h
    · split_ifs at h with hp
      · simp at h
      · exact Or.inr (ih _ h)

/-- `pre p` does not look at the answer to `p`. -/
theorem pre_congr [DecidableEq M.Qry] (p : M.Qry) {f g : M.Qry → M.Ans} (hfg : ∀ x, x ≠ p → f x = g x)
    (P : OracleProg M α) : pre p f P = pre p g P := by
  induction P with
  | ret a => rfl
  | round qs k ih =>
    simp only [pre]
    split_ifs with hp
    · rfl
    · have : qs.map f = qs.map g := List.map_congr_left fun x hx => hfg x fun h => hp (h ▸ hx)
      rw [this, ih]

/-- A program that never asks `p` does not depend on the answer to `p`. -/
theorem run_congr [DecidableEq M.Qry] (p : M.Qry) {f g : M.Qry → M.Ans} (hfg : ∀ x, x ≠ p → f x = g x)
    (P : OracleProg M α) (hp : p ∉ pre p f P) : P.run g = P.run f := by
  induction P with
  | ret a => rfl
  | round qs k ih =>
    simp only [pre, List.mem_append, not_or] at hp
    obtain ⟨hq, hrest⟩ := hp
    simp only [hq, ↓reduceIte] at hrest
    have : qs.map f = qs.map g := List.map_congr_left fun x hx => hfg x fun h => hq (h ▸ hx)
    simp only [OracleProg.run]
    rw [← this]
    exact ih _ hrest

/-- If `c (i+1)` is never asked by the first round that asks `c i`, for every `i < k`, and `c k` is asked at all,
then the first rounds asking `c 0, …, c k` strictly increase, so there are at least `k + 1` rounds. -/
theorem depth_ge [DecidableEq M.Qry] (f : M.Qry → M.Ans) (P : OracleProg M α) :
    ∀ (c : ℕ → M.Qry) (k : ℕ), (∀ i < k, c (i + 1) ∉ pre (c i) f P) → c k ∈ allQ f P → k + 1 ≤ P.depth f := by
  induction P with
  | ret a => intro c k _ hk; simp [allQ] at hk
  | round qs cont ih =>
    intro c k hc hk
    simp only [OracleProg.depth]
    have hq : ∀ i < k, c (i + 1) ∉ qs := fun i hi h => hc i hi (by simp only [pre]; exact List.mem_append_left _ h)
    cases k with
    | zero => omega
    | succ k =>
      have hk' : c (k + 1) ∈ allQ f (cont (qs.map f)) := by
        simp only [allQ, List.mem_append] at hk
        exact hk.resolve_left (hq k (by omega))
      have := ih (qs.map f) (fun i => c (i + 1)) k (fun i hi => by
        have h1 := hc (i + 1) (by omega)
        have h2 : c (i + 1) ∉ qs := hq i (by omega)
        simp only [pre, List.mem_append, not_or, h2, ↓reduceIte] at h1
        exact h1.2) hk'
      omega

end Queries

/-! ### Resampling one point of a random function -/

/-- If `p H` and `s H` do not change when `H` is changed at `p H`, then `H (p H) ∈ s H` has probability exactly
`E|s H| / |Y|`. -/
theorem card_resample {X Y : Type} [Fintype X] [DecidableEq X] [Fintype Y] [DecidableEq Y]
    (p : (X → Y) → X) (s : (X → Y) → Finset Y)
    (hp : ∀ H v, p (Function.update H (p H) v) = p H) (hs : ∀ H v, s (Function.update H (p H) v) = s H) :
    (Finset.univ.filter fun H => H (p H) ∈ s H).card * Fintype.card Y = ∑ H, (s H).card := by
  rw [← Finset.card_sigma, ← Finset.card_univ (α := Y), ← Finset.card_product]
  symm
  refine Finset.card_nbij' (fun q => (Function.update q.1 (p q.1) q.2, q.1 (p q.1)))
    (fun q => ⟨Function.update q.1 (p q.1) q.2, q.1 (p q.1)⟩) ?_ ?_ ?_ ?_
  · rintro ⟨H, v⟩ hv
    simp only [Finset.coe_sigma, Set.mem_sigma_iff, Finset.coe_univ, Set.mem_univ, Finset.mem_coe,
      true_and] at hv
    simp only [Finset.coe_product, Finset.coe_filter, Finset.mem_univ, true_and, Finset.coe_univ,
      Set.mem_prod, Set.mem_ofPred_eq, Set.mem_univ, and_true]
    rw [hp, Function.update_self, hs]
    exact hv
  · rintro ⟨H, w⟩ hw
    simp only [Finset.coe_product, Finset.coe_filter, Finset.mem_univ, true_and, Finset.coe_univ,
      Set.mem_prod, Set.mem_ofPred_eq, Set.mem_univ, and_true] at hw
    simp only [Finset.coe_sigma, Set.mem_sigma_iff, Finset.coe_univ, Set.mem_univ, Finset.mem_coe, true_and]
    rw [hs]
    exact hw
  · rintro ⟨H, v⟩ _
    simp only [hp, Function.update_self, Function.update_idem, Function.update_eq_self]
  · rintro ⟨H, w⟩ _
    simp only [hp, Function.update_self, Function.update_idem, Function.update_eq_self]

theorem update_agree {X Y : Type} [DecidableEq X] (H : X → Y) (p : X) (v : Y) :
    ∀ x, x ≠ p → Function.update H p v x = H x :=
  fun _ hx => Function.update_of_ne hx _ _

/-! ### The chain -/

section Chain

variable {m D : ℕ}

/-- The tweak of step `j`. -/
def tw (D j : ℕ) : Fin (D + 1) := ⟨j % (D + 1), Nat.mod_lt _ (Nat.succ_pos D)⟩

/-- The `j`-th chain query `(j, x_j)`. -/
def cpt (H : Fin (D + 1) × Fin m → Fin m) (x₀ : Fin m) (j : ℕ) : Fin (D + 1) × Fin m := (tw D j, chain H x₀ j)

theorem chain_succ (H : Fin (D + 1) × Fin m → Fin m) (x₀ : Fin m) (j : ℕ) :
    chain H x₀ (j + 1) = H (cpt H x₀ j) := rfl

theorem tw_val {j : ℕ} (hj : j ≤ D) : (tw D j : ℕ) = j := Nat.mod_eq_of_lt (by omega)

theorem chain_update (H : Fin (D + 1) × Fin m → Fin m) (x₀ : Fin m) {j : ℕ} (hj : j ≤ D) (v : Fin m) :
    ∀ i ≤ j, chain (Function.update H (cpt H x₀ j) v) x₀ i = chain H x₀ i
  | 0, _ => rfl
  | i + 1, hi => by
    rw [chain_succ, chain_succ]
    have ih := chain_update H x₀ hj v i (by omega)
    have hne : cpt H x₀ i ≠ cpt H x₀ j := by
      intro h
      have := congrArg (fun z => (z.1 : ℕ)) h
      simp only [cpt, tw_val (show i ≤ D by omega), tw_val hj] at this
      omega
    simp only [cpt] at ih ⊢
    rw [ih]
    exact Function.update_of_ne hne _ _

theorem cpt_update (H : Fin (D + 1) × Fin m → Fin m) (x₀ : Fin m) {j : ℕ} (hj : j ≤ D) (v : Fin m) :
    cpt (Function.update H (cpt H x₀ j) v) x₀ j = cpt H x₀ j := by
  show (tw D j, chain (Function.update H (cpt H x₀ j) v) x₀ j) = (tw D j, chain H x₀ j)
  rw [chain_update H x₀ hj v j le_rfl]

end Chain

/-! ### The bound -/

/-- The random oracle's queries are its domain; instance search does not unfold `IdealModel.randomOracle`. -/
instance instDecidableEqRandomOracleQry (X Y : Type) [Fintype X] [DecidableEq X] [Fintype Y] [Nonempty Y] :
    DecidableEq (IdealModel.randomOracle X Y).Qry :=
  inferInstanceAs (DecidableEq X)

/-- The counting form of B3, for any work bound `W`: `m · #{H | success} ≤ |Ω| · (W + 1)`. -/
theorem b3_count (m D W : ℕ) [NeZero m] (x₀ : Fin m)
    (P : OracleProg (IdealModel.randomOracle (Fin (D + 1) × Fin m) (Fin m)) (Fin m))
    (hd : ∀ H : Fin (D + 1) × Fin m → Fin m, P.depth H ≤ D) (hw : ∀ H : Fin (D + 1) × Fin m → Fin m, P.work H ≤ W) :
    (Finset.univ.filter fun H : Fin (D + 1) × Fin m → Fin m => P.run H = chain H x₀ (D + 1)).card * m ≤
      Fintype.card (Fin (D + 1) × Fin m → Fin m) * (W + 1) := by
  let Pr : (Fin (D + 1) × Fin m → Fin m) → ℕ → List (Fin (D + 1) × Fin m) := fun H j => pre (cpt H x₀ j) H P
  -- `s j H`: the values of `x_{j+1}` that step `j + 1` is charged for (the output, for `j = D`)
  let s : ℕ → (Fin (D + 1) × Fin m → Fin m) → Finset (Fin m) := fun j H =>
    if j < D then ((Pr H j).toFinset.filter (fun x => x.1 = tw D (j + 1))).image Prod.snd
    else if cpt H x₀ D ∈ Pr H D then ∅ else {P.run H}
  have hPr : ∀ j ≤ D, ∀ H v, Pr (Function.update H (cpt H x₀ j) v) j = Pr H j := by
    intro j hj H v
    simp only [Pr, cpt_update H x₀ hj v]
    exact pre_congr _ (update_agree H _ v) P
  have hs : ∀ j ≤ D, ∀ H v, s j (Function.update H (cpt H x₀ j) v) = s j H := by
    intro j hj H v
    rcases Nat.lt_or_ge j D with hjD | hjD
    · simp only [s, hjD, ↓reduceIte, hPr j hj H v]
    · obtain rfl : j = D := le_antisymm hj hjD
      simp only [s, lt_irrefl, ↓reduceIte, hPr j hj H v, cpt_update H x₀ hj v]
      split_ifs with h
      · rfl
      · exact congrArg _
          (run_congr (cpt H x₀ j) (fun x hx => (update_agree H _ v x hx).symm) P h)
  have hcard : ∀ j ≤ D,
      (Finset.univ.filter fun H : Fin (D + 1) × Fin m → Fin m => H (cpt H x₀ j) ∈ s j H).card * m =
        ∑ H, (s j H).card := by
    intro j hj
    have := card_resample (fun H => cpt H x₀ j) (s j) (fun H v => cpt_update H x₀ hj v) (hs j hj)
    rwa [Fintype.card_fin] at this
  have hcover : (Finset.univ.filter fun H : Fin (D + 1) × Fin m → Fin m => P.run H = chain H x₀ (D + 1)) ⊆
      (Finset.range (D + 1)).biUnion
        (fun j => Finset.univ.filter fun H : Fin (D + 1) × Fin m → Fin m => H (cpt H x₀ j) ∈ s j H) := by
    intro H hH
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hH
    simp only [Finset.mem_biUnion, Finset.mem_range, Finset.mem_filter, Finset.mem_univ, true_and]
    by_contra hno
    push Not at hno
    have hstep : ∀ i < D, cpt H x₀ (i + 1) ∉ pre (cpt H x₀ i) H P := by
      intro i hi hmem
      apply hno i (by omega)
      simp only [s, hi, ↓reduceIte, Finset.mem_image, Finset.mem_filter, List.mem_toFinset]
      exact ⟨cpt H x₀ (i + 1), ⟨hmem, rfl⟩, rfl⟩
    have hD : cpt H x₀ D ∉ Pr H D := by
      intro hmem
      have h1 := depth_ge H P (cpt H x₀) D hstep (mem_allQ_of_mem_pre H P hmem)
      have h2 := hd H
      omega
    apply hno D (by omega)
    simp only [s, lt_irrefl, ↓reduceIte, hD, Finset.mem_singleton]
    rw [hH]
    rfl
  have hper : ∀ H : Fin (D + 1) × Fin m → Fin m, ∑ j ∈ Finset.range (D + 1), (s j H).card ≤ W + 1 := by
    intro H
    rw [Finset.sum_range_succ]
    have hlast : (s D H).card ≤ 1 := by
      simp only [s, lt_irrefl, ↓reduceIte]
      split_ifs <;> simp
    let A : Finset (Fin (D + 1) × Fin m) := List.toFinset (α := Fin (D + 1) × Fin m) (allQ H P)
    have hj : ∀ j ∈ Finset.range D, (s j H).card ≤ (A.filter fun x => x.1 = tw D (j + 1)).card := by
      intro j hj
      rw [Finset.mem_range] at hj
      simp only [s, hj, ↓reduceIte]
      refine Finset.card_image_le.trans (Finset.card_le_card (Finset.filter_subset_filter _ ?_))
      intro x hx
      rw [List.mem_toFinset] at hx
      exact List.mem_toFinset.2 (mem_allQ_of_mem_pre H P hx)
    have hdisj : ((Finset.range D : Finset ℕ) : Set ℕ).PairwiseDisjoint
        (fun j => A.filter fun x => x.1 = tw D (j + 1)) := by
      intro i hi j hj hij
      simp only [Finset.coe_range, Set.mem_Iio] at hi hj
      refine Finset.disjoint_filter.2 fun x _ h1 h2 => hij ?_
      have := congrArg Fin.val (h1.symm.trans h2)
      rw [tw_val (by omega), tw_val (by omega)] at this
      omega
    have hsum : ∑ j ∈ Finset.range D, (s j H).card ≤ W := by
      calc ∑ j ∈ Finset.range D, (s j H).card
          ≤ ∑ j ∈ Finset.range D, (A.filter fun x => x.1 = tw D (j + 1)).card := Finset.sum_le_sum hj
        _ = ((Finset.range D).biUnion fun j => A.filter fun x => x.1 = tw D (j + 1)).card :=
            (Finset.card_biUnion hdisj).symm
        _ ≤ A.card := Finset.card_le_card (Finset.biUnion_subset.2 fun j _ => Finset.filter_subset _ _)
        _ ≤ (allQ H P).length := List.toFinset_card_le _
        _ = P.work H := length_allQ H P
        _ ≤ W := hw H
    omega
  calc (Finset.univ.filter fun H : Fin (D + 1) × Fin m → Fin m => P.run H = chain H x₀ (D + 1)).card * m
      ≤ (∑ j ∈ Finset.range (D + 1),
          (Finset.univ.filter fun H : Fin (D + 1) × Fin m → Fin m => H (cpt H x₀ j) ∈ s j H).card) * m := by
        gcongr
        exact (Finset.card_le_card hcover).trans Finset.card_biUnion_le
    _ = ∑ j ∈ Finset.range (D + 1), ∑ H, (s j H).card := by
        rw [Finset.sum_mul]
        exact Finset.sum_congr rfl fun j hj => hcard j (Nat.lt_succ_iff.1 (Finset.mem_range.1 hj))
    _ = ∑ H, ∑ j ∈ Finset.range (D + 1), (s j H).card := Finset.sum_comm
    _ ≤ ∑ _H : Fin (D + 1) × Fin m → Fin m, (W + 1) := Finset.sum_le_sum fun H _ => hper H
    _ = Fintype.card (Fin (D + 1) × Fin m → Fin m) * (W + 1) := by simp

/-- The probability form of `b3_count`. -/
theorem b3_pr (m D W : ℕ) [NeZero m] (x₀ : Fin m)
    (P : OracleProg (IdealModel.randomOracle (Fin (D + 1) × Fin m) (Fin m)) (Fin m))
    (hd : ∀ H : Fin (D + 1) × Fin m → Fin m, P.depth H ≤ D) (hw : ∀ H : Fin (D + 1) × Fin m → Fin m, P.work H ≤ W) :
    pr (fun H : Fin (D + 1) × Fin m → Fin m => P.run H = chain H x₀ (D + 1)) ≤ ((W + 1 : ℕ) : ℝ≥0∞) / m := by
  have key := b3_count m D W x₀ P hd hw
  have hΩ : (Fintype.card (Fin (D + 1) × Fin m → Fin m) : ℝ≥0∞) ≠ 0 := by
    exact_mod_cast (Fintype.card_ne_zero : Fintype.card (Fin (D + 1) × Fin m → Fin m) ≠ 0)
  have hm : (m : ℝ≥0∞) ≠ 0 := by exact_mod_cast NeZero.ne m
  have key' : ((Finset.univ.filter fun H : Fin (D + 1) × Fin m → Fin m => P.run H = chain H x₀ (D + 1)).card :
      ℝ≥0∞) / (Fintype.card (Fin (D + 1) × Fin m → Fin m)) ≤ ((W + 1 : ℕ) : ℝ≥0∞) / m := calc
    ((Finset.univ.filter fun H : Fin (D + 1) × Fin m → Fin m => P.run H = chain H x₀ (D + 1)).card : ℝ≥0∞) /
        (Fintype.card (Fin (D + 1) × Fin m → Fin m))
      = ((Finset.univ.filter fun H : Fin (D + 1) × Fin m → Fin m => P.run H = chain H x₀ (D + 1)).card * m :
          ℝ≥0∞) / (Fintype.card (Fin (D + 1) × Fin m → Fin m) * m) :=
        (ENNReal.mul_div_mul_right _ _ hm (ENNReal.natCast_ne_top m)).symm
    _ ≤ (Fintype.card (Fin (D + 1) × Fin m → Fin m) * ((W + 1 : ℕ) : ℝ≥0∞)) /
          (Fintype.card (Fin (D + 1) × Fin m → Fin m) * m) := by
        apply ENNReal.div_le_div_right
        exact_mod_cast key
    _ = ((W + 1 : ℕ) : ℝ≥0∞) / m := ENNReal.mul_div_mul_left _ _ hΩ (ENNReal.natCast_ne_top _)
  unfold pr
  convert key' using 4
  exact Finset.filter_congr_decidable _ _ _

/-- B3. Hash-chain sequentiality: `D` rounds of at most `q` random-oracle queries output the end `x_{D+1}` of the
tweaked chain with probability at most `(D·q + 1)/m` (verbatim from `PousProofs.b3_chain`). -/
theorem b3_chain (m D q : ℕ) [NeZero m] (x₀ : Fin m)
    (P : OracleProg (IdealModel.randomOracle (Fin (D + 1) × Fin m) (Fin m)) (Fin m)) (hP : P.RoundBounded D q) :
    pr (fun H : Fin (D + 1) × Fin m → Fin m => P.run H = chain H x₀ (D + 1)) ≤ ((D * q + 1 : ℕ) : ℝ≥0∞) / m :=
  b3_pr m D (D * q) x₀ P (fun H => (hP H).1)
    (fun H => (work_le_depth_mul_width H P).trans (Nat.mul_le_mul (hP H).1 (hP H).2))

/-- B3 in the audit game's `(D, Q)` model (`OracleProg.Bounded`, not drafted): probability at most `(Q + 1)/m`. -/
theorem b3_chain_bounded (m D Q : ℕ) [NeZero m] (x₀ : Fin m)
    (P : OracleProg (IdealModel.randomOracle (Fin (D + 1) × Fin m) (Fin m)) (Fin m)) (hP : P.Bounded D Q) :
    pr (fun H : Fin (D + 1) × Fin m → Fin m => P.run H = chain H x₀ (D + 1)) ≤ ((Q + 1 : ℕ) : ℝ≥0∞) / m :=
  b3_pr m D Q x₀ P (fun H => (hP H).1) (fun H => (hP H).2)

end PousPebbling
