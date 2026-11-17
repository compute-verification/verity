module

public import Properties.Pouw.MonitoringPeriodJ10R
public import Proofs.ResourceAccounting.Served
public import Proofs.Compose.ComposeMonitoringPeriod.Honest
public import Proofs.Flock.Soundness.Audit.StratifiedProj

/-! The tile coordinate is the sampling block's stratified sample at the tile stratum. -/

@[expose] public section

open Claims renaming Ex → expect

namespace Properties.Pouw.MonitoringPeriodJ10R.SecurityProofs

open Properties.Pouw.MonitoringPeriodJ10R Pouw.RelativeSecurity
open ResourceAccounting.SecurityProofs.Compose.Budget (forall_mem_subsetDraw rest card_le_rest prob_mem)
open ResourceAccounting.SecurityProofs.TS.Stages (prob_prod)
open Properties.Served.SecurityProofs (card_rest_le)

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
        refine ResourceAccounting.SecurityProofs.ComposeMonitoringPeriod.Honest.prob_none fun e he => ?_
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

noncomputable def tilesIn {nB mB : ℕ} (σ : Fin nB → Fin mB) (sT : Fin mB) (B : Finset (Fin nB)) :
    Finset (Fin (tileCount σ sT)) :=
  Finset.univ.filter fun t => tileAt σ sT t ∈ B

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

theorem tileStratumAcceptProb : TileStratumAcceptProb := by
  intro nB mB σ k hk sT B
  have h : (fun π : Equiv.Perm (Fin (tileCount σ sT)) => Disjoint (tileDraw σ k hk sT π) B) =
      fun π => ∀ v ∈ Pouw.Window.subsetDraw (k sT) π, v ∉ tilesIn σ sT B :=
    funext fun π => propext (disjoint_tileDraw_iff σ k hk sT π B)
  rw [h, miss_eq (k sT) (tileCount σ sT) _ (hk sT), card_tilesIn]

theorem tileStratumAcceptProb_eq_proj {nB mB : ℕ} (σ : Fin nB → Fin mB) (k : Fin mB → ℕ)
    (hk : ∀ s, k s ≤ (stratum σ s).card) (sT : Fin mB) (B : Finset (Fin nB)) :
    ENNReal.ofReal (prob fun π : Equiv.Perm (Fin (tileCount σ sT)) => Disjoint (tileDraw σ k hk sT π) B) =
      FlockSoundness.Game.prCoin
        (fun ω : (FlockSoundness.Audit.SamplingDesign.stratified σ k hk).Ω => Disjoint (ω sT).1 B) := by
  have hpos : (0 : ℝ) < ((tileCount σ sT).choose (k sT) : ℝ) := by exact_mod_cast Nat.choose_pos (hk sT)
  rw [tileStratumAcceptProb σ k hk sT B, ENNReal.ofReal_div_of_pos hpos, ENNReal.ofReal_natCast,
    ENNReal.ofReal_natCast]
  exact (FlockSoundness.Audit.SamplingDesign.stratified_proj_escape σ k hk sT B).symm

theorem tileStratumAcceptProb_eq_exp {nB mB : ℕ} (σ : Fin nB → Fin mB) (k : Fin mB → ℕ)
    (hk : ∀ s, k s ≤ (stratum σ s).card) (sT : Fin mB) (B : Finset (Fin nB))
    (hc : (B.filter (σ · = sT)).card + k sT ≤ tileCount σ sT) :
    ENNReal.ofReal (prob fun π : Equiv.Perm (Fin (tileCount σ sT)) => Disjoint (tileDraw σ k hk sT π) B) =
      ENNReal.ofReal (Real.exp (-FlockSoundness.Audit.SamplingDesign.stratumRisk σ k sT (B.filter (σ · = sT)).card)) := by
  rw [tileStratumAcceptProb_eq_proj]
  exact FlockSoundness.Audit.SamplingDesign.stratified_proj_escape_eq_exp σ k hk sT B hc

end Properties.Pouw.MonitoringPeriodJ10R.SecurityProofs
