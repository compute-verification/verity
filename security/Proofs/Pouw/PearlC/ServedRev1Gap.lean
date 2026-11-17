module

public import Proofs.Pouw.PearlC.TileInertBoundRev1Gap

/-! v1 under rev1 at the served linears, at any batch. -/

@[expose] public section

namespace Pouw.PearlC

open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

section
variable {Q R S : Type}

theorem shape_eq_mk_of_nk {s : Shape} {n k : ℕ} (h : s.n = n ∧ s.k = k) : s = ⟨s.m, n, k⟩ := by
  obtain ⟨m, n', k'⟩ := s
  obtain ⟨h1, h2⟩ := h
  simp only at h1 h2
  subst h1 h2
  rfl

theorem pearlCGapDevRev1KNK [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ c : ℚ) (n k : ℕ) (ω g : ℚ) (hω0 : 0 < ω)
    (hω : ∀ m : ℕ, wrefDevRev1K d c ⟨m, n, k⟩ = ω * ((1 - ρ) * creditDevRev1 d ⟨m, n, k⟩))
    (hg : g = 1 - (1 - 1 / 400) / ω) (hTT : TTOutPearlCDevRev1 CM d sem ρ) :
    GGap CM (pearlCProtocolDevRev1K d sem ρ c) (pearlCDomainDevNK d n k) (g : ℝ) δPearlC := by
  have hW : ∀ U : Workload, U.InDomain (pearlCProtocolDevRev1K d sem ρ c) (pearlCDomainDevNK d n k) →
      ∀ (H : Q → R) (s' : S), ∀ u < U.layout.N, ∀ act : Codes,
        (pearlCProtocolDevRev1K d sem ρ c).actOK (U.layout.shape u) act →
        (pearlCProtocolDevRev1K d sem ρ c).inertBoundOK H s' U u act →
          (pearlCProtocolDevRev1K d sem ρ c).Wref U.layout u ≤
            (ω : ℝ) * (pearlCProtocolDevRev1K d sem ρ c).credit H s' U u act := by
    intro U hU H s' u hu act _ hcap
    have hsh := shape_eq_mk_of_nk (hU.2.2 u hu)
    have hc : sem.unitInertWorkRev1 d H s' U u act ≤ ρ * creditDevRev1 d (U.layout.shape u) := hcap
    show (wrefDevRev1K d c (U.layout.shape u) : ℝ) ≤
      (ω : ℝ) * ((creditDevRev1 d (U.layout.shape u) - sem.unitInertWorkRev1 d H s' U u act : ℚ) : ℝ)
    rw [hsh] at hc ⊢
    rw [hω]
    have : ω * ((1 - ρ) * creditDevRev1 d ⟨(U.layout.shape u).m, n, k⟩) ≤
        ω * (creditDevRev1 d ⟨(U.layout.shape u).m, n, k⟩ - sem.unitInertWorkRev1 d H s' U u act) :=
      mul_le_mul_of_nonneg_left (by linarith) hω0.le
    exact_mod_cast this
  have key := gapFromTTOut CM (pearlCProtocolDevRev1K d sem ρ c) (pearlCDomainDevNK d n k) (1 / 400) (ω : ℝ)
    δPearlC (by exact_mod_cast hω0) hW (ttOut_mono_domain (fun _ h => h.1) (ttOut_wref _ hTT))
  have e : (g : ℝ) = 1 - (1 - 1 / 400) / (ω : ℝ) := by rw [hg]; push_cast; ring
  rw [e]
  exact key

theorem pearlCSampledDevRev1KNK_gen [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ c : ℚ) (n k : ℕ) (ω : ℚ) (g₀ : ℝ)
    (hg₀ : g₀ < 1) (hω0 : 0 < ω) (hw0 : ∀ m : ℕ, 0 ≤ wrefDevRev1K d c ⟨m, n, k⟩)
    (hω : ∀ m : ℕ, wrefDevRev1K d c ⟨m, n, k⟩ = ω * ((1 - ρ) * creditDevRev1 d ⟨m, n, k⟩))
    (hTT : TTOutTile CM (pearlCProtocolDevRev1 d sem ρ) (pearlCTilesDevRev1 d sem ρ) (pearlCDomainDev d) g₀ δPearlC) :
    GGapSampled CM (pearlCProtocolDevRev1K d sem ρ c) (pearlCTilesDevRev1K d sem ρ c) (pearlCDomainDevNK d n k)
      (1 - (1 - g₀) / (ω : ℝ)) δPearlC := by
  have hW : ∀ U : Workload, U.InDomain (pearlCProtocolDevRev1K d sem ρ c) (pearlCDomainDevNK d n k) →
      ∀ (H : Q → R) (s' : S), ∀ g < ((pearlCTilesDevRev1K d sem ρ c).tiling U.layout).NT, ∀ act : Codes,
        (pearlCTilesDevRev1K d sem ρ c).inertBoundOK H s' U g act →
        (pearlCTilesDevRev1K d sem ρ c).Wcred U.layout g act ≤
          (ω : ℝ) * (pearlCTilesDevRev1K d sem ρ c).credit H s' U g act := by
    intro U hU H s' g hg act _
    have hu := auditTiling_unit_lt 64 64 U.layout hg
    have hsh := shape_eq_mk_of_nk (hU.2.2 _ hu)
    simp only [pearlCTilesDevRev1K, pearlCTilesDevRev1]
    rw [hsh, hω]
    push_cast
    exact le_of_eq (by ring)
  have hWc : ∀ U : Workload, U.InDomain (pearlCProtocolDevRev1K d sem ρ c) (pearlCDomainDevNK d n k) →
      ∀ g < ((pearlCTilesDevRev1K d sem ρ c).tiling U.layout).NT, ∀ act : Codes,
        (pearlCTilesDevRev1K d sem ρ c).Wcred U.layout g act ≤ (pearlCTilesDevRev1K d sem ρ c).Wref U.layout g := by
    intro U hU g hg act
    have hu := auditTiling_unit_lt 64 64 U.layout hg
    have hsh := shape_eq_mk_of_nk (hU.2.2 _ hu)
    simp only [pearlCTilesDevRev1K, pearlCTilesDevRev1]
    rw [hsh]
    exact_mod_cast mul_le_mul_of_nonneg_left (tileShare_mono _ (passRows_subset sem _ _ _) _) (hw0 _)
  exact gapSampled CM (pearlCProtocolDevRev1K d sem ρ c) (pearlCTilesDevRev1K d sem ρ c)
    (pearlCDomainDevNK d n k) g₀ (ω : ℝ) δPearlC hg₀ (by exact_mod_cast hω0) hW hWc
    (ttOutTile_mono_domain (fun _ h => h.1) (ttOutTile_wref _ _ _ hTT))

theorem pearlCSampledDevRev1KNK [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ c : ℚ) (n k : ℕ) (ω g : ℚ) (hω0 : 0 < ω)
    (hw0 : ∀ m : ℕ, 0 ≤ wrefDevRev1K d c ⟨m, n, k⟩)
    (hω : ∀ m : ℕ, wrefDevRev1K d c ⟨m, n, k⟩ = ω * ((1 - ρ) * creditDevRev1 d ⟨m, n, k⟩))
    (hg : g = 1 - (1 - 1 / 400) / ω) (hTT : TTOutTilePearlCDevRev1 CM d sem ρ) :
    GGapSampled CM (pearlCProtocolDevRev1K d sem ρ c) (pearlCTilesDevRev1K d sem ρ c) (pearlCDomainDevNK d n k)
      (g : ℝ) δPearlC := by
  have key := pearlCSampledDevRev1KNK_gen CM d sem ρ c n k ω (1 / 400) (by norm_num) hω0 hw0 hω hTT
  have e : (g : ℝ) = 1 - (1 - 1 / 400) / (ω : ℝ) := by rw [hg]; push_cast; ring
  rw [e]
  exact key

theorem pearlCSampledDevRev1KNK_of_inert_bound [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ ρ' c : ℚ) (n k : ℕ) (ω g : ℚ)
    (hadd : 0 ≤ d.prices.add) (hbf : 0 ≤ d.prices.bf16) (hle : ρ' ≤ ρ) (hρ1 : ρ < 1) (hω0 : 0 < ω)
    (hw0 : ∀ m : ℕ, 0 ≤ wrefDevRev1K d c ⟨m, n, k⟩)
    (hω : ∀ m : ℕ, wrefDevRev1K d c ⟨m, n, k⟩ = ω * ((1 - ρ) * creditDevRev1 d ⟨m, n, k⟩))
    (hg : g = 1 - (1 - 1 / 400) / ω) (hTT : TTOutTilePearlCDevRev1 CM d sem ρ) :
    GGapSampled CM (pearlCProtocolDevRev1K d sem ρ' c) (pearlCTilesDevRev1K d sem ρ' c) (pearlCDomainDevNK d n k)
      (g : ℝ) δPearlC := by
  have hr : (0 : ℚ) < 1 - ρ := by linarith
  have hr' : (0 : ℚ) < 1 - ρ' := by linarith
  have key := pearlCSampledDevRev1KNK_gen CM d sem ρ' c n k (ω * ((1 - ρ) / (1 - ρ'))) _
    (by
      have : (0 : ℝ) < (((1 - ρ) / (1 - ρ') : ℚ) : ℝ) := by exact_mod_cast div_pos hr hr'
      nlinarith)
    (mul_pos hω0 (div_pos hr hr')) hw0
    (fun m => by rw [hω]; field_simp)
    (ttOutTilePearlCDevRev1_inert_bound_transfer CM d sem ρ ρ' (1 / 400) δPearlC hadd hbf hle hρ1 hTT)
  have e : (g : ℝ) = 1 - (1 - (1 - (1 - 1 / 400) * (((1 - ρ) / (1 - ρ') : ℚ) : ℝ))) /
      ((ω * ((1 - ρ) / (1 - ρ')) : ℚ) : ℝ) := by
    rw [hg]
    have h1 : ((1 - ρ : ℚ) : ℝ) ≠ 0 := by exact_mod_cast hr.ne'
    have h2 : ((1 - ρ' : ℚ) : ℝ) ≠ 0 := by exact_mod_cast hr'.ne'
    have h3 : ((ω : ℚ) : ℝ) ≠ 0 := by exact_mod_cast hω0.ne'
    push_cast at h1 h2 ⊢
    field_simp
    ring
  rw [e]
  exact key

end

section Served

variable {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
  (sem : PearlCSem Q R S)

theorem wrefDevRev1K_sm120v1LoopCast8p72_n6144k4096 (m : ℕ) :
    wrefDevRev1K (devSm120v1 Prices.sm120Loop) (218 / 25 - 8953 / 1000) ⟨m, 6144, 4096⟩ =
      2266417 / 2265828 * creditDevRev1 (devSm120v1 Prices.sm120Loop) ⟨m, 6144, 4096⟩ := by
  rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
    show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
    show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
  norm_num [Prices.sm120Loop, Params.pi]
  ring

theorem wrefDevRev1K_sm120v1LoopCast8p72_n6144k4096_nonneg (m : ℕ) :
    0 ≤ wrefDevRev1K (devSm120v1 Prices.sm120Loop) (218 / 25 - 8953 / 1000) ⟨m, 6144, 4096⟩ := by
  rw [wrefDevRev1K_sm120v1LoopCast8p72_n6144k4096]
  refine mul_nonneg (by norm_num) (creditDevRev1_nonneg _ ?_ ?_ _ (by norm_num) (Or.inr ?_))
  · rw [show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]; norm_num [Prices.sm120Loop]
  · rw [show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]; norm_num [Prices.sm120Loop]
  · rw [show (devSm120v1 Prices.sm120Loop).G = 4 from rfl]; norm_num

theorem pearlCGapSm120v1LoopCast8p72Rev1InertShare1000_n6144k4096
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 6144 4096) ((851373343 / 226641700000 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevRev1KNK CM _ sem _ _ _ _ (2266417 / 2265828 / (1 - 1 / 1000)) _ (by norm_num)
    (fun m => by rw [wrefDevRev1K_sm120v1LoopCast8p72_n6144k4096 m]; field_simp) (by norm_num) hTT

theorem pearlCSampledSm120v1LoopCast8p72Rev1InertShare1000_n6144k4096
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 6144 4096) ((851373343 / 226641700000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1KNK CM _ sem _ _ _ _ (2266417 / 2265828 / (1 - 1 / 1000)) _ (by norm_num)
    wrefDevRev1K_sm120v1LoopCast8p72_n6144k4096_nonneg
    (fun m => by rw [wrefDevRev1K_sm120v1LoopCast8p72_n6144k4096 m]; field_simp) (by norm_num) hTT

theorem wrefDevRev1K_sm120v1LoopCast8p72_n4096k4096 (m : ℕ) :
    wrefDevRev1K (devSm120v1 Prices.sm120Loop) (218 / 25 - 8953 / 1000) ⟨m, 4096, 4096⟩ =
      1519141 / 1518552 * creditDevRev1 (devSm120v1 Prices.sm120Loop) ⟨m, 4096, 4096⟩ := by
  rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
    show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
    show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
  norm_num [Prices.sm120Loop, Params.pi]
  ring

theorem wrefDevRev1K_sm120v1LoopCast8p72_n4096k4096_nonneg (m : ℕ) :
    0 ≤ wrefDevRev1K (devSm120v1 Prices.sm120Loop) (218 / 25 - 8953 / 1000) ⟨m, 4096, 4096⟩ := by
  rw [wrefDevRev1K_sm120v1LoopCast8p72_n4096k4096]
  refine mul_nonneg (by norm_num) (creditDevRev1_nonneg _ ?_ ?_ _ (by norm_num) (Or.inr ?_))
  · rw [show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]; norm_num [Prices.sm120Loop]
  · rw [show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]; norm_num [Prices.sm120Loop]
  · rw [show (devSm120v1 Prices.sm120Loop).G = 4 from rfl]; norm_num

theorem pearlCGapSm120v1LoopCast8p72Rev1InertShare1000_n4096k4096
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 4096 4096) ((295006781 / 75957050000 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevRev1KNK CM _ sem _ _ _ _ (1519141 / 1518552 / (1 - 1 / 1000)) _ (by norm_num)
    (fun m => by rw [wrefDevRev1K_sm120v1LoopCast8p72_n4096k4096 m]; field_simp) (by norm_num) hTT

theorem pearlCSampledSm120v1LoopCast8p72Rev1InertShare1000_n4096k4096
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 4096 4096) ((295006781 / 75957050000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1KNK CM _ sem _ _ _ _ (1519141 / 1518552 / (1 - 1 / 1000)) _ (by norm_num)
    wrefDevRev1K_sm120v1LoopCast8p72_n4096k4096_nonneg
    (fun m => by rw [wrefDevRev1K_sm120v1LoopCast8p72_n4096k4096 m]; field_simp) (by norm_num) hTT

theorem wrefDevRev1K_sm120v1LoopCast8p72_n28672k4096 (m : ℕ) :
    wrefDevRev1K (devSm120v1 Prices.sm120Loop) (218 / 25 - 8953 / 1000) ⟨m, 28672, 4096⟩ =
      10486453 / 10485864 * creditDevRev1 (devSm120v1 Prices.sm120Loop) ⟨m, 28672, 4096⟩ := by
  rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
    show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
    show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
  norm_num [Prices.sm120Loop, Params.pi]
  ring

theorem wrefDevRev1K_sm120v1LoopCast8p72_n28672k4096_nonneg (m : ℕ) :
    0 ≤ wrefDevRev1K (devSm120v1 Prices.sm120Loop) (218 / 25 - 8953 / 1000) ⟨m, 28672, 4096⟩ := by
  rw [wrefDevRev1K_sm120v1LoopCast8p72_n28672k4096]
  refine mul_nonneg (by norm_num) (creditDevRev1_nonneg _ ?_ ?_ _ (by norm_num) (Or.inr ?_))
  · rw [show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]; norm_num [Prices.sm120Loop]
  · rw [show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]; norm_num [Prices.sm120Loop]
  · rw [show (devSm120v1 Prices.sm120Loop).G = 4 from rfl]; norm_num

theorem pearlCGapSm120v1LoopCast8p72Rev1InertShare1000_n28672k4096
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 28672 4096) ((1863165467 / 524322650000 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevRev1KNK CM _ sem _ _ _ _ (10486453 / 10485864 / (1 - 1 / 1000)) _ (by norm_num)
    (fun m => by rw [wrefDevRev1K_sm120v1LoopCast8p72_n28672k4096 m]; field_simp) (by norm_num) hTT

theorem pearlCSampledSm120v1LoopCast8p72Rev1InertShare1000_n28672k4096
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 28672 4096) ((1863165467 / 524322650000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1KNK CM _ sem _ _ _ _ (10486453 / 10485864 / (1 - 1 / 1000)) _ (by norm_num)
    wrefDevRev1K_sm120v1LoopCast8p72_n28672k4096_nonneg
    (fun m => by rw [wrefDevRev1K_sm120v1LoopCast8p72_n28672k4096 m]; field_simp) (by norm_num) hTT

theorem wrefDevRev1K_sm120v1LoopCast8p72_n4096k14336 (m : ℕ) :
    wrefDevRev1K (devSm120v1 Prices.sm120Loop) (218 / 25 - 8953 / 1000) ⟨m, 4096, 14336⟩ =
      31303841 / 31291472 * creditDevRev1 (devSm120v1 Prices.sm120Loop) ⟨m, 4096, 14336⟩ := by
  rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
    show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
    show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
  norm_num [Prices.sm120Loop, Params.pi]
  ring

theorem wrefDevRev1K_sm120v1LoopCast8p72_n4096k14336_nonneg (m : ℕ) :
    0 ≤ wrefDevRev1K (devSm120v1 Prices.sm120Loop) (218 / 25 - 8953 / 1000) ⟨m, 4096, 14336⟩ := by
  rw [wrefDevRev1K_sm120v1LoopCast8p72_n4096k14336]
  refine mul_nonneg (by norm_num) (creditDevRev1_nonneg _ ?_ ?_ _ (by norm_num) (Or.inr ?_))
  · rw [show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]; norm_num [Prices.sm120Loop]
  · rw [show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]; norm_num [Prices.sm120Loop]
  · rw [show (devSm120v1 Prices.sm120Loop).G = 4 from rfl]; norm_num

theorem pearlCGapSm120v1LoopCast8p72Rev1InertShare1000_n4096k14336
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 4096 14336) ((3045273083 / 782596025000 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevRev1KNK CM _ sem _ _ _ _ (31303841 / 31291472 / (1 - 1 / 1000)) _ (by norm_num)
    (fun m => by rw [wrefDevRev1K_sm120v1LoopCast8p72_n4096k14336 m]; field_simp) (by norm_num) hTT

theorem pearlCSampledSm120v1LoopCast8p72Rev1InertShare1000_n4096k14336
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 4096 14336) ((3045273083 / 782596025000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1KNK CM _ sem _ _ _ _ (31303841 / 31291472 / (1 - 1 / 1000)) _ (by norm_num)
    wrefDevRev1K_sm120v1LoopCast8p72_n4096k14336_nonneg
    (fun m => by rw [wrefDevRev1K_sm120v1LoopCast8p72_n4096k14336 m]; field_simp) (by norm_num) hTT

theorem wrefDevRev1K_sm120v1LoopCast8p72_n24576k4096 (m : ℕ) :
    wrefDevRev1K (devSm120v1 Prices.sm120Loop) (218 / 25 - 8953 / 1000) ⟨m, 24576, 4096⟩ =
      8991901 / 8991312 * creditDevRev1 (devSm120v1 Prices.sm120Loop) ⟨m, 24576, 4096⟩ := by
  rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
    show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
    show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
  norm_num [Prices.sm120Loop, Params.pi]
  ring

theorem wrefDevRev1K_sm120v1LoopCast8p72_n24576k4096_nonneg (m : ℕ) :
    0 ≤ wrefDevRev1K (devSm120v1 Prices.sm120Loop) (218 / 25 - 8953 / 1000) ⟨m, 24576, 4096⟩ := by
  rw [wrefDevRev1K_sm120v1LoopCast8p72_n24576k4096]
  refine mul_nonneg (by norm_num) (creditDevRev1_nonneg _ ?_ ?_ _ (by norm_num) (Or.inr ?_))
  · rw [show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]; norm_num [Prices.sm120Loop]
  · rw [show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]; norm_num [Prices.sm120Loop]
  · rw [show (devSm120v1 Prices.sm120Loop).G = 4 from rfl]; norm_num

theorem wrefDevRev1K_sm120v1LoopCast8p72_n4096k12288 (m : ℕ) :
    wrefDevRev1K (devSm120v1 Prices.sm120Loop) (218 / 25 - 8953 / 1000) ⟨m, 4096, 12288⟩ =
      13433021 / 13427720 * creditDevRev1 (devSm120v1 Prices.sm120Loop) ⟨m, 4096, 12288⟩ := by
  rw [wrefDevRev1K, wrefDevRev1, creditDevRev1, firstAddDev, creditDev,
    show (devSm120v1 Prices.sm120Loop).G = 4 from rfl,
    show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]
  norm_num [Prices.sm120Loop, Params.pi]
  ring

theorem wrefDevRev1K_sm120v1LoopCast8p72_n4096k12288_nonneg (m : ℕ) :
    0 ≤ wrefDevRev1K (devSm120v1 Prices.sm120Loop) (218 / 25 - 8953 / 1000) ⟨m, 4096, 12288⟩ := by
  rw [wrefDevRev1K_sm120v1LoopCast8p72_n4096k12288]
  refine mul_nonneg (by norm_num) (creditDevRev1_nonneg _ ?_ ?_ _ (by norm_num) (Or.inr ?_))
  · rw [show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]; norm_num [Prices.sm120Loop]
  · rw [show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]; norm_num [Prices.sm120Loop]
  · rw [show (devSm120v1 Prices.sm120Loop).G = 4 from rfl]; norm_num

end Served

end Pouw.PearlC
