module

public import Properties.Pouw.MonitoringPeriodJ10R
public import Proofs.ResourceAccounting.Served
public import Proofs.Compose.ComposeMonitoringPeriod.Honest
public import Proofs.Spot.Sampling.StratifiedProj

/-!
# The tile coordinate is the sampling block's stratified sample at the tile stratum

`TileStratumAcceptProb`, proved, and its join to the sampling design's side:

* `miss_eq` (`ResourceAccounting.SecurityProofs.Compose.Budget.miss_eq`, the core's
  `Verity.SamplingDesign.missProb_permPrefix_eq`): a uniform permutation's `K`-prefix misses `B` with probability exactly
  `C(N − |B|, K)/C(N, K)`.
* `tileStratumAcceptProb`: the tile coordinate (`tileDraw`) misses `B` exactly when the permutation's `k s_T`-prefix misses
  the stratum's places whose proof blocks are in `B` (`disjoint_tileDraw_iff`), `|B.filter (σ · = s_T)|` of them
  (`card_tilesIn`), so `miss_eq` is `TileStratumAcceptProb`.
* `tileStratumAcceptProb_eq_proj`: that is `Spot.Sampling.SamplingDesign.stratified_proj_escape`'s factor, so the tile
  coordinate at a uniform permutation misses every `B` exactly as `SamplingDesign.stratified σ k hk`'s coordinate `s_T` does, and
  `tileStratumAcceptProb_eq_exp`: with `m_s + k_s ≤ n_s`, it misses `B` with `exp(−stratumRisk σ k s_T m_s)`, stratum
  `s_T`'s exact risk (`SamplingDesign.stratified_proj_escape_eq_exp`).
-/

@[expose] public section

open Claims renaming Ex → expect

namespace Properties.Pouw.MonitoringPeriodJ10R.SecurityProofs

open Properties.Pouw.MonitoringPeriodJ10R Pouw.RelativeSecurity
open ResourceAccounting.SecurityProofs.Compose.Budget (forall_mem_permPrefix miss_eq)
open ResourceAccounting.SecurityProofs.TS.Stages (prob_prod)

/-- The tile stratum's places whose proof blocks are in `B`. -/
noncomputable def tilesIn {nB mB : ℕ} (σ : Fin nB → Fin mB) (sT : Fin mB) (B : Finset (Fin nB)) :
    Finset (Fin (tileCount σ sT)) :=
  Finset.univ.filter fun t => tileAt σ sT t ∈ B

/-- **The tile coordinate misses `B` exactly when the permutation's `k s_T`-prefix misses `B`'s places.** -/
theorem disjoint_tileDraw_iff {nB mB : ℕ} (σ : Fin nB → Fin mB) (k : Fin mB → ℕ)
    (hk : ∀ s, k s ≤ (stratum σ s).card) (sT : Fin mB) (π : Equiv.Perm (Fin (tileCount σ sT)))
    (B : Finset (Fin nB)) :
    Disjoint (tileDraw σ k hk sT π) B ↔ ∀ v ∈ (Verity.SamplingDesign.permPrefix (tileCount σ sT) (k sT)).draw π, v ∉ tilesIn σ sT B := by
  rw [Finset.disjoint_left, forall_mem_permPrefix]
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

/-- **`TileStratumAcceptProb`**: the tile coordinate at a uniform permutation of the stratum misses `B` with the stratum's
hypergeometric factor. -/
theorem tileStratumAcceptProb : TileStratumAcceptProb := by
  intro nB mB σ k hk sT B
  have h : (fun π : Equiv.Perm (Fin (tileCount σ sT)) => Disjoint (tileDraw σ k hk sT π) B) =
      fun π : Equiv.Perm (Fin (tileCount σ sT)) =>
        ∀ v ∈ (Verity.SamplingDesign.permPrefix (tileCount σ sT) (k sT)).draw π, v ∉ tilesIn σ sT B :=
    funext fun π => propext (disjoint_tileDraw_iff σ k hk sT π B)
  rw [h, miss_eq (k sT) (tileCount σ sT) _ (hk sT), card_tilesIn]

/-- **The tile coordinate is the stratified sample's coordinate `s_T`**: at a uniform permutation of the stratum it misses
every `B` exactly as `SamplingDesign.stratified σ k hk`'s coordinate `s_T` does (`SamplingDesign.stratified_proj_escape`), so the period's
tile coin is the sampling block's sample restricted to the tile stratum. -/
theorem tileStratumAcceptProb_eq_proj {nB mB : ℕ} (σ : Fin nB → Fin mB) (k : Fin mB → ℕ)
    (hk : ∀ s, k s ≤ (stratum σ s).card) (sT : Fin mB) (B : Finset (Fin nB)) :
    ENNReal.ofReal (prob fun π : Equiv.Perm (Fin (tileCount σ sT)) => Disjoint (tileDraw σ k hk sT π) B) =
      FlockSoundness.Game.prCoin
        (fun ω : (Spot.Sampling.SamplingDesign.stratified σ k hk).Ω => Disjoint (ω sT).1 B) := by
  have hpos : (0 : ℝ) < ((tileCount σ sT).choose (k sT) : ℝ) := by exact_mod_cast Nat.choose_pos (hk sT)
  rw [tileStratumAcceptProb σ k hk sT B, ENNReal.ofReal_div_of_pos hpos, ENNReal.ofReal_natCast,
    ENNReal.ofReal_natCast]
  exact (Spot.Sampling.SamplingDesign.stratified_proj_escape σ k hk sT B).symm

/-- **The tile coordinate's escape is stratum `s_T`'s exact risk**: with `m_s + k_s ≤ n_s` it misses `B` with
`exp(−stratumRisk σ k s_T m_s)` (`SamplingDesign.stratified_proj_escape_eq_exp`). -/
theorem tileStratumAcceptProb_eq_exp {nB mB : ℕ} (σ : Fin nB → Fin mB) (k : Fin mB → ℕ)
    (hk : ∀ s, k s ≤ (stratum σ s).card) (sT : Fin mB) (B : Finset (Fin nB))
    (hc : (B.filter (σ · = sT)).card + k sT ≤ tileCount σ sT) :
    ENNReal.ofReal (prob fun π : Equiv.Perm (Fin (tileCount σ sT)) => Disjoint (tileDraw σ k hk sT π) B) =
      ENNReal.ofReal (Real.exp (-Spot.Sampling.SamplingDesign.stratumRisk σ k sT (B.filter (σ · = sT)).card)) := by
  rw [tileStratumAcceptProb_eq_proj]
  exact Spot.Sampling.SamplingDesign.stratified_proj_escape_eq_exp σ k hk sT B hc

end Properties.Pouw.MonitoringPeriodJ10R.SecurityProofs
