import Properties.Pouw.PeriodJ10R
import Proofs.Accounting.Served
import Proofs.Accounting.ComposePeriod.Honest
import Proofs.Flock.Soundness.Audit.StratifiedProj

/-!
# The tile coordinate is the sampling block's stratified draw at the tile stratum

`TileStratumEscape`, proved, and its join to the law's side:

* `miss_eq`: a uniform permutation's `K`-prefix (`Pouw.Window.subsetDraw`) misses `B` with probability exactly
  `C(N − |B|, K)/C(N, K)`, by `Properties.Served.SecurityProofs.miss_ge`'s induction through
  `Equiv.Perm.decomposeFin` with equality at each step: the first image misses `B` with probability `1 − |B|/N`, and
  given it misses, the rest is a uniform permutation of `N − 1` places that must miss a set of exactly `|B|` of them.
* `tileStratumEscape`: the tile coordinate (`tileDraw`) misses `B` exactly when the permutation's `k s_T`-prefix misses
  the stratum's places whose proof blocks are in `B` (`disjoint_tileDraw_iff`), `|B.filter (σ · = s_T)|` of them
  (`card_tilesIn`), so `miss_eq` is `TileStratumEscape`.
* `tileStratumEscape_eq_proj`: that is `FlockSoundness.Audit.Law.stratified_proj_escape`'s factor, so the tile
  coordinate at a uniform permutation misses every `B` exactly as `Law.stratified σ k hk`'s coordinate `s_T` does, and
  `tileStratumEscape_eq_exp`: with `m_s + k_s ≤ n_s`, it misses `B` with `exp(−stratumRisk σ k s_T m_s)`, stratum
  `s_T`'s exact risk (`Law.stratified_proj_escape_eq_exp`).
-/

namespace Properties.Pouw.PeriodJ10R.SecurityProofs

open Properties.Pouw.PeriodJ10R Pouw.RelativeSecurity
open Accounting.SecurityProofs.Compose.Budget (forall_mem_subsetDraw rest card_le_rest prob_mem)
open Accounting.SecurityProofs.TS.Stages (prob_prod)
open Properties.Served.SecurityProofs (card_rest_le)

/-- **A uniform permutation's `K`-prefix misses `B` with the hypergeometric factor, exactly**:
`C(N − |B|, K)/C(N, K)` for `K ≤ N`. -/
theorem miss_eq : ∀ (K N : ℕ) (B : Finset (Fin N)), K ≤ N →
    prob (fun π : Equiv.Perm (Fin N) => ∀ v ∈ Pouw.Window.subsetDraw K π, v ∉ B) =
      ((N - B.card).choose K : ℝ) / (N.choose K : ℝ)
  | 0, N, B, _ => by
    classical
    rw [prob_eq_card, Finset.filter_true_of_mem fun π _ =>
      (forall_mem_subsetDraw (K := 0) π fun v => v ∉ B).mpr fun i hi => absurd hi (Nat.not_lt_zero _),
      Finset.card_univ, Nat.choose_zero_right, Nat.choose_zero_right, div_self (Nat.cast_ne_zero.2 Fintype.card_ne_zero)]
    norm_num
  | K + 1, 0, _, h => absurd h (by omega)
  | K + 1, n + 1, B, h => by
    classical
    have hK : K ≤ n := Nat.le_of_succ_le_succ h
    rw [prob_equiv Equiv.Perm.decomposeFin, prob_prod]
    have hstep : ∀ p : Fin (n + 1),
        prob (fun e : Equiv.Perm (Fin n) =>
            ∀ v ∈ Pouw.Window.subsetDraw (K + 1) (Equiv.Perm.decomposeFin.symm (p, e)), v ∉ B) =
          ((n - B.card).choose K : ℝ) / (n.choose K : ℝ) * (if p ∈ B then 0 else 1) := by
      intro p
      by_cases hp : p ∈ B
      · rw [if_pos hp, mul_zero]
        refine Accounting.SecurityProofs.ComposePeriod.Honest.prob_none fun e he => ?_
        have h0 := (forall_mem_subsetDraw _ _).mp he 0 (Nat.succ_pos K)
        rw [Equiv.Perm.decomposeFin_symm_apply_zero] at h0
        exact h0 hp
      rw [if_neg hp, mul_one]
      have hc : (rest B p).card = B.card := le_antisymm (card_rest_le B p) (card_le_rest B p hp)
      rw [← hc, ← miss_eq K n (rest B p) hK]
      congr 1
      funext e
      refine propext ⟨fun he => (forall_mem_subsetDraw _ _).mpr fun j hj => ?_, fun he =>
        (forall_mem_subsetDraw _ _).mpr fun i hi => ?_⟩
      · have hmiss := (forall_mem_subsetDraw _ _).mp he j.succ (by rw [Fin.val_succ]; omega)
        rw [Equiv.Perm.decomposeFin_symm_apply_succ] at hmiss
        simpa [rest] using hmiss
      · rcases Fin.eq_zero_or_eq_succ i with rfl | ⟨j, rfl⟩
        · rw [Equiv.Perm.decomposeFin_symm_apply_zero]
          exact hp
        · have hj : j.val < K := by simpa using hi
          have hmiss := (forall_mem_subsetDraw _ _).mp he j hj
          rw [Equiv.Perm.decomposeFin_symm_apply_succ]
          simpa [rest] using hmiss
    refine (congrArg expect (funext hstep)).trans ?_
    rw [expect_mul_left]
    have hnot : expect (fun p : Fin (n + 1) => if p ∈ B then (0 : ℝ) else 1) =
        1 - (B.card : ℝ) / ((n + 1 : ℕ) : ℝ) := by
      have hsum := prob_add_not (fun p : Fin (n + 1) => p ∈ B)
      rw [prob_mem] at hsum
      have hind : prob (fun p : Fin (n + 1) => ¬p ∈ B) =
          expect (fun p : Fin (n + 1) => if p ∈ B then (0 : ℝ) else 1) := by
        rw [prob_eq_expect]
        exact congrArg expect (funext fun p => by by_cases hp : p ∈ B <;> simp [hp])
      linarith
    rw [hnot]
    have hb : B.card ≤ n + 1 := (Finset.card_le_univ B).trans_eq (Fintype.card_fin _)
    have hCn : (0 : ℝ) < (n.choose K : ℝ) := by exact_mod_cast Nat.choose_pos hK
    have hCn1 : (0 : ℝ) < ((n + 1).choose (K + 1) : ℝ) := by exact_mod_cast Nat.choose_pos h
    have hn1 : (0 : ℝ) < ((n + 1 : ℕ) : ℝ) := by exact_mod_cast Nat.succ_pos n
    rcases Nat.lt_or_ge n B.card with hbn | hbn
    · have h0 : n + 1 - B.card = 0 := by omega
      have hB1 : (B.card : ℝ) = ((n + 1 : ℕ) : ℝ) := by exact_mod_cast (show B.card = n + 1 by omega)
      rw [h0, Nat.choose_eq_zero_of_lt (show 0 < K + 1 by omega), Nat.cast_zero, zero_div, hB1,
        div_self hn1.ne', sub_self, mul_zero]
    · obtain ⟨a, ha⟩ : ∃ a, n = B.card + a := ⟨n - B.card, by omega⟩
      have hm : n + 1 - B.card = a + 1 := by omega
      have hnb : n - B.card = a := by omega
      have hsub : 1 - (B.card : ℝ) / ((n + 1 : ℕ) : ℝ) = ((a + 1 : ℕ) : ℝ) / ((n + 1 : ℕ) : ℝ) := by
        rw [eq_div_iff hn1.ne', sub_mul, one_mul, div_mul_cancel₀ _ hn1.ne']
        have hcast : ((n + 1 : ℕ) : ℝ) = (B.card : ℝ) + ((a + 1 : ℕ) : ℝ) := by
          exact_mod_cast (show n + 1 = B.card + (a + 1) by omega)
        rw [hcast]
        ring
      have key : (a + 1).choose (K + 1) * (n.choose K * (n + 1)) =
          a.choose K * (a + 1) * (n + 1).choose (K + 1) := by
        calc (a + 1).choose (K + 1) * (n.choose K * (n + 1))
            = (a + 1).choose (K + 1) * ((n + 1) * n.choose K) := by ring
          _ = (a + 1).choose (K + 1) * ((n + 1).choose (K + 1) * (K + 1)) := by
            rw [Nat.add_one_mul_choose_eq n K]
          _ = (a + 1) * a.choose K * (n + 1).choose (K + 1) := by
            rw [Nat.add_one_mul_choose_eq a K]
            ring
          _ = a.choose K * (a + 1) * (n + 1).choose (K + 1) := by ring
      rw [hm, hnb, hsub, div_mul_div_comm, div_eq_div_iff (mul_pos hCn hn1).ne' hCn1.ne']
      exact_mod_cast key.symm

/-- The tile stratum's places whose proof blocks are in `B`. -/
noncomputable def tilesIn {nB mB : ℕ} (σ : Fin nB → Fin mB) (sT : Fin mB) (B : Finset (Fin nB)) :
    Finset (Fin (tileCount σ sT)) :=
  Finset.univ.filter fun t => tileAt σ sT t ∈ B

/-- **The tile coordinate misses `B` exactly when the permutation's `k s_T`-prefix misses `B`'s places.** -/
theorem disjoint_tileDraw_iff {nB mB : ℕ} (σ : Fin nB → Fin mB) (k : Fin mB → ℕ)
    (hk : ∀ s, k s ≤ (stratum σ s).card) (sT : Fin mB) (π : Equiv.Perm (Fin (tileCount σ sT)))
    (B : Finset (Fin nB)) :
    Disjoint (tileDraw σ k hk sT π) B ↔ ∀ v ∈ Pouw.Window.subsetDraw (k sT) π, v ∉ tilesIn σ sT B := by
  rw [Finset.disjoint_left, forall_mem_subsetDraw]
  constructor
  · intro h i hi hT
    exact h (Finset.mem_image.2 ⟨⟨i.val, hi⟩, Finset.mem_univ _, rfl⟩) (Finset.mem_filter.1 hT).2
  · intro h a ha hB
    obtain ⟨j, -, rfl⟩ := Finset.mem_image.1 ha
    exact h (Fin.castLE (hk sT) j) j.isLt (Finset.mem_filter.2 ⟨Finset.mem_univ _, hB⟩)

/-- **`B` has `|B.filter (σ · = s_T)|` places in the tile stratum**: `tileAt` is a bijection onto the stratum. -/
theorem card_tilesIn {nB mB : ℕ} (σ : Fin nB → Fin mB) (sT : Fin mB) (B : Finset (Fin nB)) :
    (tilesIn σ sT B).card = (B.filter (σ · = sT)).card := by
  refine Finset.card_bij (fun t _ => tileAt σ sT t) (fun t ht => ?_) (fun a _ b _ h => ?_) (fun u hu => ?_)
  · have hs : tileAt σ sT t ∈ stratum σ sT := Finset.orderEmbOfFin_mem _ _ t
    exact Finset.mem_filter.2 ⟨(Finset.mem_filter.1 ht).2, (Finset.mem_filter.1 hs).2⟩
  · exact ((stratum σ sT).orderEmbOfFin rfl).injective h
  · obtain ⟨hB, hσ⟩ := Finset.mem_filter.1 hu
    have hs : u ∈ (stratum σ sT : Set (Fin nB)) := Finset.mem_filter.2 ⟨Finset.mem_univ _, hσ⟩
    rw [← Finset.range_orderEmbOfFin (stratum σ sT) rfl] at hs
    obtain ⟨t, ht⟩ := hs
    refine ⟨t, Finset.mem_filter.2 ⟨Finset.mem_univ _, ?_⟩, ht⟩
    show (stratum σ sT).orderEmbOfFin rfl t ∈ B
    rw [ht]
    exact hB

/-- **`TileStratumEscape`**: the tile coordinate at a uniform permutation of the stratum misses `B` with the stratum's
hypergeometric factor. -/
theorem tileStratumEscape : TileStratumEscape := by
  intro nB mB σ k hk sT B
  have h : (fun π : Equiv.Perm (Fin (tileCount σ sT)) => Disjoint (tileDraw σ k hk sT π) B) =
      fun π => ∀ v ∈ Pouw.Window.subsetDraw (k sT) π, v ∉ tilesIn σ sT B :=
    funext fun π => propext (disjoint_tileDraw_iff σ k hk sT π B)
  rw [h, miss_eq (k sT) (tileCount σ sT) _ (hk sT), card_tilesIn]

/-- **The tile coordinate is the stratified draw's coordinate `s_T`**: at a uniform permutation of the stratum it misses
every `B` exactly as `Law.stratified σ k hk`'s coordinate `s_T` does (`Law.stratified_proj_escape`), so the period's
tile coin is the sampling block's draw restricted to the tile stratum. -/
theorem tileStratumEscape_eq_proj {nB mB : ℕ} (σ : Fin nB → Fin mB) (k : Fin mB → ℕ)
    (hk : ∀ s, k s ≤ (stratum σ s).card) (sT : Fin mB) (B : Finset (Fin nB)) :
    ENNReal.ofReal (prob fun π : Equiv.Perm (Fin (tileCount σ sT)) => Disjoint (tileDraw σ k hk sT π) B) =
      FlockSoundness.Game.prCoin
        (fun ω : (FlockSoundness.Audit.Law.stratified σ k hk).Ω => Disjoint (ω sT).1 B) := by
  have hpos : (0 : ℝ) < ((tileCount σ sT).choose (k sT) : ℝ) := by exact_mod_cast Nat.choose_pos (hk sT)
  rw [tileStratumEscape σ k hk sT B, ENNReal.ofReal_div_of_pos hpos, ENNReal.ofReal_natCast,
    ENNReal.ofReal_natCast]
  exact (FlockSoundness.Audit.Law.stratified_proj_escape σ k hk sT B).symm

/-- **The tile coordinate's escape is stratum `s_T`'s exact risk**: with `m_s + k_s ≤ n_s` it misses `B` with
`exp(−stratumRisk σ k s_T m_s)` (`Law.stratified_proj_escape_eq_exp`). -/
theorem tileStratumEscape_eq_exp {nB mB : ℕ} (σ : Fin nB → Fin mB) (k : Fin mB → ℕ)
    (hk : ∀ s, k s ≤ (stratum σ s).card) (sT : Fin mB) (B : Finset (Fin nB))
    (hc : (B.filter (σ · = sT)).card + k sT ≤ tileCount σ sT) :
    ENNReal.ofReal (prob fun π : Equiv.Perm (Fin (tileCount σ sT)) => Disjoint (tileDraw σ k hk sT π) B) =
      ENNReal.ofReal (Real.exp (-FlockSoundness.Audit.Law.stratumRisk σ k sT (B.filter (σ · = sT)).card)) := by
  rw [tileStratumEscape_eq_proj]
  exact FlockSoundness.Audit.Law.stratified_proj_escape_eq_exp σ k hk sT B hc

end Properties.Pouw.PeriodJ10R.SecurityProofs
