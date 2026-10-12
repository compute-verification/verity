import Proofs.Pouw.PearlC.DeviceCapGamma
import Models.Pouw.Assumptions.PearlC.DeviceChainCap
import Models.Pouw.Guarantees.PearlC

/-!
# The cap on the chain's share: what it admits, that it asserts less, and γ (the TT_OUT restatements lane's file)

Staged; nothing pinned.
* **What it admits:** `chainCap_flagged_le`. A debit within `ρ·chainCreditDev` has at most `ρ·m·k·n/32` flagged
  atom-words, a `ρ` share of the unit's atom-words, at every shape and every G, for a non-negative add price. So a family
  whose flagged share exceeds `ρ` is rejected at every unit width. The per-unit cap bounds it only by
  `ρ·creditDev/(m·k·n)`.
* **It asserts less than the per-unit cap at the same `ρ`,** per unit (`ttOutPearlCDevChainCap_of_cap`) and per tile
  (`ttOutTilePearlCDevChainCap_of_cap`), for non-negative prices and `0 ≤ ρ ≤ 1`. `chainCreditDev ≤ creditDev`, so a unit
  or tile within the chain cap is within the per-unit one, and the credit is the same.
* **γ at any record:** per unit at `γ = 1 − (399/400)/ω` with `wrefDev = ω·(creditDev − ρ·chainCreditDev)`
  (`pearlCGammaDevChainCapAt`). That is a little better than the per-unit cap's, since an admitted unit's debit is at most
  `ρ` times the chain, not the whole credit. Per tile, with the same fixed credit as `pearlCTilesDev`, it is the per-unit
  cap's γ (`pearlCSampledDevChainCapAt`).
* **v2 at the chain cap 1/1,000**, at any record with `G = 0` and `Prices.sm120`:
  - per unit, `26921/7493750` (0.35925%) at 8192³ and `367519/103656250` (0.35456%) at 16384³;
  - per audit tile, `216793/59950000` (0.36162%) and `2950127/829250000` (0.35576%), as the 1/1,000 row.
-/

namespace Pouw.PearlC

open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

theorem chainCreditDev_le (d : PearlCDevice) (hbf : 0 ≤ d.prices.bf16) (s : Shape) :
    chainCreditDev d s ≤ creditDev d s := by
  unfold chainCreditDev creditDev
  have h1 : (0 : ℚ) ≤ (s.m : ℚ) * s.k * Params.pi.r := by positivity
  have h2 : (0 : ℚ) ≤ 2 * d.prices.bf16 * Params.pi.r * s.m * s.n := by positivity
  have h3 : (0 : ℚ) ≤ (d.prices.costs.fs : ℚ) * s.m * s.k := by positivity
  simp only
  linarith

/-- **An admitted unit has at most a `ρ` share of its atom-words flagged**, at every shape: a debit within
`ρ·chainCreditDev` has `32·atoms ≤ ρ·m·k·n`. -/
theorem chainCap_flagged_le (d : PearlCDevice) (hadd : 0 ≤ d.prices.add) (ρ : ℚ) (s : Shape) (x : Debit)
    (h : debitDev d x ≤ ρ * chainCreditDev d s) : 32 * (x.atoms : ℚ) ≤ ρ * ((s.m : ℚ) * s.k * s.n) := by
  unfold debitDev chainCreditDev at h
  have he : (0 : ℚ) ≤ (d.prices.costs.fs : ℚ) * x.elems := by positivity
  have ha : (0 : ℚ) ≤ x.atoms := by positivity
  by_cases hG : d.G = 0
  · simp only [hG, ite_true, add_zero] at h
    linarith
  · simp only [hG, ite_false] at h
    have hGpos : (0 : ℚ) < d.G := by exact_mod_cast Nat.pos_of_ne_zero hG
    set mkn : ℚ := (s.m : ℚ) * s.k * s.n
    set a : ℚ := d.prices.add / d.G
    have ha0 : 0 ≤ a := div_nonneg hadd hGpos.le
    have hsplit : mkn * d.prices.add / (32 * d.G) = mkn * a / 32 := by
      simp only [a]
      field_simp
    rw [hsplit] at h
    have key : (32 + a) * (32 * x.atoms) ≤ (32 + a) * (ρ * mkn) := by nlinarith
    exact le_of_mul_le_mul_left key (by linarith)

section
variable {Q R S : Type}

open Classical in
theorem chainCap_credit_sum_le (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) (hadd : 0 ≤ d.prices.add)
    (hbf : 0 ≤ d.prices.bf16) (hρ0 : 0 ≤ ρ) (hρ1 : ρ ≤ 1) (U : Workload) (H : Q → R) (s : S) (τ : Transcript) :
    ∑ u ∈ correctSet (pearlCProtocolDevChainCap d sem ρ) U H s τ,
        (pearlCProtocolDevChainCap d sem ρ).credit H s U u (actOf τ u) ≤
      ∑ u ∈ correctSet (pearlCProtocolDev d sem ρ) U H s τ, (pearlCProtocolDev d sem ρ).credit H s U u (actOf τ u) := by
  have hsub : correctSet (pearlCProtocolDevChainCap d sem ρ) U H s τ ⊆
      correctSet (pearlCProtocolDev d sem ρ) U H s τ := by
    intro u hu
    unfold correctSet at hu ⊢
    rw [mem_filter] at hu ⊢
    obtain ⟨hu, c, hc, hact, hcap, hw⟩ := hu
    refine ⟨hu, c, hc, hact, ?_, hw⟩
    have h1 : debitDev d (sem.unitDebitDev d H s U u c.act) ≤ ρ * chainCreditDev d (U.layout.shape u) := hcap
    show debitDev d (sem.unitDebitDev d H s U u c.act) ≤ ρ * creditDev d (U.layout.shape u)
    have := mul_le_mul_of_nonneg_left (chainCreditDev_le d hbf (U.layout.shape u)) hρ0
    linarith
  refine sum_le_sum_of_subset_of_nonneg hsub fun u hu _ => ?_
  unfold correctSet at hu
  rw [mem_filter] at hu
  obtain ⟨_, c, hc, _, hcap, _⟩ := hu
  have hact : actOf τ u = c.act := by simp [actOf, hc]
  rw [hact]
  have h1 : debitDev d (sem.unitDebitDev d H s U u c.act) ≤ ρ * creditDev d (U.layout.shape u) := hcap
  have h2 := creditDev_nonneg d hadd hbf (U.layout.shape u)
  show (0 : ℝ) ≤ ((creditDev d (U.layout.shape u) - debitDev d (sem.unitDebitDev d H s U u c.act) : ℚ) : ℝ)
  have : (0 : ℚ) ≤ creditDev d (U.layout.shape u) - debitDev d (sem.unitDebitDev d H s U u c.act) := by nlinarith
  exact_mod_cast this

/-- **The chain cap asserts less than the per-unit cap at the same `ρ`.** -/
theorem ttOutPearlCDevChainCap_of_cap [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) (hadd : 0 ≤ d.prices.add) (hbf : 0 ≤ d.prices.bf16)
    (hρ0 : 0 ≤ ρ) (hρ1 : ρ ≤ 1) (h : TTOutPearlCDev CM d sem ρ) : TTOutPearlCDevChainCap CM d sem ρ := by
  refine ⟨h.1, fun U hU T hT q hq A hA => le_trans (Pouw.pr_mono ?_) (h.2 U hU T hT q hq A hA)⟩
  intro x hx
  exact lt_of_lt_of_le hx (chainCap_credit_sum_le d sem ρ hadd hbf hρ0 hρ1 U x.1 x.2 _)

open Classical in
theorem chainCap_tile_credit_sum_le (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) (hadd : 0 ≤ d.prices.add)
    (hbf : 0 ≤ d.prices.bf16) (hρ0 : 0 ≤ ρ) (hρ1 : ρ ≤ 1) (U : Workload) (H : Q → R) (s : S) (τ : Transcript) :
    ∑ g ∈ goodTiles (pearlCProtocolDevChainCap d sem ρ) (pearlCTilesDevChainCap d sem ρ) U H s τ,
        (pearlCTilesDevChainCap d sem ρ).credit H s U g
          (actOf τ (((pearlCTilesDevChainCap d sem ρ).tiling U.layout).unit g)) ≤
      ∑ g ∈ goodTiles (pearlCProtocolDev d sem ρ) (pearlCTilesDev d sem ρ) U H s τ,
        (pearlCTilesDev d sem ρ).credit H s U g (actOf τ (((pearlCTilesDev d sem ρ).tiling U.layout).unit g)) := by
  set tl := auditTiling 64 64 U.layout
  have hsub : goodTiles (pearlCProtocolDevChainCap d sem ρ) (pearlCTilesDevChainCap d sem ρ) U H s τ ⊆
      goodTiles (pearlCProtocolDev d sem ρ) (pearlCTilesDev d sem ρ) U H s τ := by
    intro g hg
    unfold goodTiles at hg ⊢
    rw [mem_filter] at hg ⊢
    obtain ⟨hg, c, hc, hcap, hw⟩ := hg
    refine ⟨hg, c, hc, ?_, hw⟩
    have h1 : sem.tileDebitDev d H s U (tl.unit g) (sem.passRows (U.layout.shape (tl.unit g)) c.act (tl.rows g))
        (tl.cols g) c.act ≤ ρ * chainCreditDev d (U.layout.shape (tl.unit g)) *
          tileShare (U.layout.shape (tl.unit g)) (sem.passRows (U.layout.shape (tl.unit g)) c.act (tl.rows g))
            (tl.cols g) := hcap
    show sem.tileDebitDev d H s U (tl.unit g) (sem.passRows (U.layout.shape (tl.unit g)) c.act (tl.rows g))
        (tl.cols g) c.act ≤ ρ * creditDev d (U.layout.shape (tl.unit g)) *
          tileShare (U.layout.shape (tl.unit g)) (sem.passRows (U.layout.shape (tl.unit g)) c.act (tl.rows g))
            (tl.cols g)
    have hsh := tileShare_nonneg (U.layout.shape (tl.unit g))
      (sem.passRows (U.layout.shape (tl.unit g)) c.act (tl.rows g)) (tl.cols g)
    have := mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left
      (chainCreditDev_le d hbf (U.layout.shape (tl.unit g))) hρ0) hsh
    linarith
  refine sum_le_sum_of_subset_of_nonneg hsub fun g _ _ => ?_
  simp only [pearlCTilesDevChainCap, pearlCTilesDev]
  have hsh := tileShare_nonneg (U.layout.shape (tl.unit g))
    (sem.passRows (U.layout.shape (tl.unit g)) (actOf τ (tl.unit g)) (tl.rows g)) (tl.cols g)
  have h2 := creditDev_nonneg d hadd hbf (U.layout.shape (tl.unit g))
  have : (0 : ℚ) ≤ (1 - ρ) * creditDev d (U.layout.shape (tl.unit g)) *
      tileShare (U.layout.shape (tl.unit g))
        (sem.passRows (U.layout.shape (tl.unit g)) (actOf τ (tl.unit g)) (tl.rows g)) (tl.cols g) :=
    mul_nonneg (mul_nonneg (by linarith) h2) hsh
  exact_mod_cast this

/-- **The chain cap per tile asserts less than the per-unit cap per tile at the same `ρ`.** -/
theorem ttOutTilePearlCDevChainCap_of_cap [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) (hadd : 0 ≤ d.prices.add) (hbf : 0 ≤ d.prices.bf16)
    (hρ0 : 0 ≤ ρ) (hρ1 : ρ ≤ 1) (h : TTOutTilePearlCDev CM d sem ρ) : TTOutTilePearlCDevChainCap CM d sem ρ := by
  refine ⟨h.1, fun U hU T hT q hq A hA => le_trans (Pouw.pr_mono ?_) (h.2 U hU T hT q hq A hA)⟩
  intro x hx
  exact lt_of_lt_of_le hx (chainCap_tile_credit_sum_le d sem ρ hadd hbf hρ0 hρ1 U x.1 x.2 _)

end

/-- TT_OUT with the chain cap gives `G_γ` on the one-shape domain at `γ = 1 − (399/400)/ω`, where
`wrefDev = ω·(creditDev − ρ·chainCreditDev)`: an admitted unit's debit is at most `ρ·chainCreditDev`. -/
theorem pearlCGammaDevChainCapAt {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) (s : Shape) (ω γ : ℚ) (hω0 : 0 < ω)
    (hω : wrefDev d s = ω * (creditDev d s - ρ * chainCreditDev d s)) (hγ : γ = 1 - (1 - 1 / 400) / ω)
    (hTT : TTOutPearlCDevChainCap CM d sem ρ) :
    Gγ CM (pearlCProtocolDevChainCap d sem ρ) (pearlCDomainDevAt d s) (γ : ℝ) εPearlC := by
  have hW : ∀ U : Workload, U.InDomain (pearlCProtocolDevChainCap d sem ρ) (pearlCDomainDevAt d s) →
      ∀ (H : Q → R) (s' : S), ∀ u < U.layout.N, ∀ act : Codes,
        (pearlCProtocolDevChainCap d sem ρ).actOK (U.layout.shape u) act →
        (pearlCProtocolDevChainCap d sem ρ).capOK H s' U u act →
          (pearlCProtocolDevChainCap d sem ρ).Wref U.layout u ≤
            (ω : ℝ) * (pearlCProtocolDevChainCap d sem ρ).credit H s' U u act := by
    intro U hU H s' u hu act _ hcap
    have hsh : U.layout.shape u = s := hU.2.2 u hu
    have hc : debitDev d (sem.unitDebitDev d H s' U u act) ≤ ρ * chainCreditDev d (U.layout.shape u) := hcap
    show (wrefDev d (U.layout.shape u) : ℝ) ≤
      (ω : ℝ) * ((creditDev d (U.layout.shape u) - debitDev d (sem.unitDebitDev d H s' U u act) : ℚ) : ℝ)
    rw [hsh] at hc ⊢
    rw [hω]
    have : ω * (creditDev d s - ρ * chainCreditDev d s) ≤
        ω * (creditDev d s - debitDev d (sem.unitDebitDev d H s' U u act)) :=
      mul_le_mul_of_nonneg_left (by linarith) hω0.le
    exact_mod_cast this
  have key := gammaFromTTOut CM (pearlCProtocolDevChainCap d sem ρ) (pearlCDomainDevAt d s) (1 / 400) (ω : ℝ)
    εPearlC (by exact_mod_cast hω0) hW (ttOut_mono_domain (fun _ h => h.1) hTT)
  have e : (γ : ℝ) = 1 - (1 - 1 / 400) / (ω : ℝ) := by rw [hγ]; push_cast; ring
  rw [e]
  exact key

/-- TT_OUT per tile with the chain cap gives `GγSampled` on the one-shape domain at `γ = 1 − (399/400)/ω`, where
`wrefDev = ω·(1 − ρ)·creditDev ≥ 0`: the tiles' credit and `W_ref` are `pearlCTilesDev`'s. -/
theorem pearlCSampledDevChainCapAt {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) (s : Shape) (ω γ : ℚ) (hω0 : 0 < ω)
    (hw0 : 0 ≤ wrefDev d s) (hω : wrefDev d s = ω * ((1 - ρ) * creditDev d s)) (hγ : γ = 1 - (1 - 1 / 400) / ω)
    (hTT : TTOutTilePearlCDevChainCap CM d sem ρ) :
    GγSampled CM (pearlCProtocolDevChainCap d sem ρ) (pearlCTilesDevChainCap d sem ρ) (pearlCDomainDevAt d s) (γ : ℝ)
      εPearlC := by
  have hW : ∀ U : Workload, U.InDomain (pearlCProtocolDevChainCap d sem ρ) (pearlCDomainDevAt d s) →
      ∀ (H : Q → R) (s' : S), ∀ g < ((pearlCTilesDevChainCap d sem ρ).tiling U.layout).NT, ∀ act : Codes,
        (pearlCTilesDevChainCap d sem ρ).capOK H s' U g act →
        (pearlCTilesDevChainCap d sem ρ).Wcred U.layout g act ≤
          (ω : ℝ) * (pearlCTilesDevChainCap d sem ρ).credit H s' U g act := by
    intro U hU H s' g hg act _
    have hu := auditTiling_unit_lt 64 64 U.layout hg
    have hsh : U.layout.shape ((auditTiling 64 64 U.layout).unit g) = s := hU.2.2 _ hu
    simp only [pearlCTilesDevChainCap, pearlCTilesDev]
    rw [hsh, hω]
    push_cast
    exact le_of_eq (by ring)
  have hWc : ∀ U : Workload, U.InDomain (pearlCProtocolDevChainCap d sem ρ) (pearlCDomainDevAt d s) →
      ∀ g < ((pearlCTilesDevChainCap d sem ρ).tiling U.layout).NT, ∀ act : Codes,
        (pearlCTilesDevChainCap d sem ρ).Wcred U.layout g act ≤ (pearlCTilesDevChainCap d sem ρ).Wref U.layout g := by
    intro U hU g hg act
    have hu := auditTiling_unit_lt 64 64 U.layout hg
    have hsh : U.layout.shape ((auditTiling 64 64 U.layout).unit g) = s := hU.2.2 _ hu
    simp only [pearlCTilesDevChainCap, pearlCTilesDev]
    rw [hsh]
    exact_mod_cast mul_le_mul_of_nonneg_left (tileShare_mono _ (passRows_subset sem _ _ _) _) hw0
  have key := gammaSampled CM (pearlCProtocolDevChainCap d sem ρ) (pearlCTilesDevChainCap d sem ρ)
    (pearlCDomainDevAt d s) (1 / 400) (ω : ℝ) εPearlC (by norm_num) (by exact_mod_cast hω0) hW hWc
    (ttOutTile_mono_domain (fun _ h => h.1) hTT)
  have e : (γ : ℝ) = 1 - (1 - 1 / 400) / (ω : ℝ) := by rw [hγ]; push_cast; ring
  rw [e]
  exact key

/-- **v2 at the chain cap 1/1,000, 8192³**, per unit, at a record with no promotion and sm_120's prices:
`γ = 26921/7493750`. -/
theorem pearlCGammaUnpromotedChainCap1000_8192 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (hG : d.G = 0) (hp : d.prices = Prices.sm120)
    (hTT : TTOutPearlCDevChainCap CM d sem (1 / 1000)) :
    Gγ CM (pearlCProtocolDevChainCap d sem (1 / 1000)) (pearlCDomainDevAt d sh8192) ((26921 / 7493750 : ℚ) : ℝ)
      εPearlC :=
  pearlCGammaDevChainCapAt CM d sem _ _ (1049125 / 1047976) _ (by norm_num)
    (by rw [wrefDev, creditDev, chainCreditDev, hG, hp]; norm_num [Prices.sm120, Params.pi, sh8192]) (by norm_num) hTT

/-- **v2 at the chain cap 1/1,000, 16384³**, per unit, at a record with no promotion and sm_120's prices:
`γ = 367519/103656250`. -/
theorem pearlCGammaUnpromotedChainCap1000_16384 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (hG : d.G = 0) (hp : d.prices = Prices.sm120)
    (hTT : TTOutPearlCDevChainCap CM d sem (1 / 1000)) :
    Gγ CM (pearlCProtocolDevChainCap d sem (1 / 1000)) (pearlCDomainDevAt d sh16384)
      ((367519 / 103656250 : ℚ) : ℝ) εPearlC :=
  pearlCGammaDevChainCapAt CM d sem _ _ (2073125 / 2070952) _ (by norm_num)
    (by rw [wrefDev, creditDev, chainCreditDev, hG, hp]; norm_num [Prices.sm120, Params.pi, sh16384]) (by norm_num) hTT

/-- **v2 at the chain cap 1/1,000 per audit tile, 8192³**: `GγSampled` at `γ = 216793/59950000`. -/
theorem pearlCSampledUnpromotedChainCap1000_8192 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (hG : d.G = 0) (hp : d.prices = Prices.sm120)
    (hTT : TTOutTilePearlCDevChainCap CM d sem (1 / 1000)) :
    GγSampled CM (pearlCProtocolDevChainCap d sem (1 / 1000)) (pearlCTilesDevChainCap d sem (1 / 1000))
      (pearlCDomainDevAt d sh8192) ((216793 / 59950000 : ℚ) : ℝ) εPearlC :=
  pearlCSampledDevChainCapAt CM d sem _ _ (1049125 / 1047951) _ (by norm_num)
    (by rw [wrefDev, creditDev, hG, hp]; norm_num [Prices.sm120, Params.pi, sh8192])
    (by rw [wrefDev, creditDev, hG, hp]; norm_num [Prices.sm120, Params.pi, sh8192]) (by norm_num) hTT

/-- **v2 at the chain cap 1/1,000 per audit tile, 16384³**: `GγSampled` at `γ = 2950127/829250000`. -/
theorem pearlCSampledUnpromotedChainCap1000_16384 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (hG : d.G = 0) (hp : d.prices = Prices.sm120)
    (hTT : TTOutTilePearlCDevChainCap CM d sem (1 / 1000)) :
    GγSampled CM (pearlCProtocolDevChainCap d sem (1 / 1000)) (pearlCTilesDevChainCap d sem (1 / 1000))
      (pearlCDomainDevAt d sh16384) ((2950127 / 829250000 : ℚ) : ℝ) εPearlC :=
  pearlCSampledDevChainCapAt CM d sem _ _ (2073125 / 2070927) _ (by norm_num)
    (by rw [wrefDev, creditDev, hG, hp]; norm_num [Prices.sm120, Params.pi, sh16384])
    (by rw [wrefDev, creditDev, hG, hp]; norm_num [Prices.sm120, Params.pi, sh16384]) (by norm_num) hTT

end Pouw.PearlC

theorem Pouw.SecurityProofs.PearlC.GammaUnpromotedChainCap1000_8192 : Pouw.Guarantees.PearlC.GammaUnpromotedChainCap1000_8192 := @Pouw.PearlC.pearlCGammaUnpromotedChainCap1000_8192
