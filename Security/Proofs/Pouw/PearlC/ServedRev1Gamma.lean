import Proofs.Pouw.PearlC.TileCapRev1Gamma

/-!
# v1 under rev1 at the served linears, at any batch (compute accounting's file)

* **One linear at any `m`:** `pearlCDomainDevNK d n k` holds the units of one linear `m × n × k` at any mix of batch
  sizes, decode and prefill in one layout. Every term of `creditDevRev1` and `wrefDevRev1K` is linear in `m`, so
  `ω = wrefDevRev1K/((1 − ρ)·creditDevRev1)` is one number per `(n, k)`, and `pearlCGammaDevRev1KNK`,
  `pearlCSampledDevRev1KNK` and `pearlCSampledDevRev1KNK_of_cap` are `KernelWrefGamma`'s and `TileCapRev1Gamma`'s
  one-shape theorems on it.
* **γ at the linears a served pass credits** at v1's record with FP32 at 8.376 and the honest cast at 8.72, at the cap
  1/1,000: Llama-3.1-8B's `(n, k)` = (6144, 4096) qkv, (4096, 4096) o, (28672, 4096) gate_up, (4096, 14336) down, and
  Qwen3-8B's (6144, 4096), (4096, 4096), (24576, 4096) gate_up, (4096, 12288) down. Per unit and per audit tile from
  TT_OUT at 1/1,000, and per unit from TT_OUT at 1/400 at the same γ: 0.35535% (Llama's gate_up) to 0.38912% (Llama's
  down). Per tile from TT_OUT per tile at 1/400 the γ is 1/400's (`TileCapRev1Gamma`): 0.50496% to 0.53869%.
-/

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

/-- `pearlCGammaDevRev1KAt` on one linear at any `m`: TT_OUT rev1 at a device record gives `G_γ` against the kernel's
`W_ref` at `γ = 1 − (399/400)/ω`, where `wrefDevRev1K d c = ω·(1 − ρ)·creditDevRev1` at every `m × n × k`. -/
theorem pearlCGammaDevRev1KNK [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ c : ℚ) (n k : ℕ) (ω γ : ℚ) (hω0 : 0 < ω)
    (hω : ∀ m : ℕ, wrefDevRev1K d c ⟨m, n, k⟩ = ω * ((1 - ρ) * creditDevRev1 d ⟨m, n, k⟩))
    (hγ : γ = 1 - (1 - 1 / 400) / ω) (hTT : TTOutPearlCDevRev1 CM d sem ρ) :
    Gγ CM (pearlCProtocolDevRev1K d sem ρ c) (pearlCDomainDevNK d n k) (γ : ℝ) εPearlC := by
  have hW : ∀ U : Workload, U.InDomain (pearlCProtocolDevRev1K d sem ρ c) (pearlCDomainDevNK d n k) →
      ∀ (H : Q → R) (s' : S), ∀ u < U.layout.N, ∀ act : Codes,
        (pearlCProtocolDevRev1K d sem ρ c).actOK (U.layout.shape u) act →
        (pearlCProtocolDevRev1K d sem ρ c).capOK H s' U u act →
          (pearlCProtocolDevRev1K d sem ρ c).Wref U.layout u ≤
            (ω : ℝ) * (pearlCProtocolDevRev1K d sem ρ c).credit H s' U u act := by
    intro U hU H s' u hu act _ hcap
    have hsh := shape_eq_mk_of_nk (hU.2.2 u hu)
    have hc : sem.unitDebitRev1 d H s' U u act ≤ ρ * creditDevRev1 d (U.layout.shape u) := hcap
    show (wrefDevRev1K d c (U.layout.shape u) : ℝ) ≤
      (ω : ℝ) * ((creditDevRev1 d (U.layout.shape u) - sem.unitDebitRev1 d H s' U u act : ℚ) : ℝ)
    rw [hsh] at hc ⊢
    rw [hω]
    have : ω * ((1 - ρ) * creditDevRev1 d ⟨(U.layout.shape u).m, n, k⟩) ≤
        ω * (creditDevRev1 d ⟨(U.layout.shape u).m, n, k⟩ - sem.unitDebitRev1 d H s' U u act) :=
      mul_le_mul_of_nonneg_left (by linarith) hω0.le
    exact_mod_cast this
  have key := gammaFromTTOut CM (pearlCProtocolDevRev1K d sem ρ c) (pearlCDomainDevNK d n k) (1 / 400) (ω : ℝ)
    εPearlC (by exact_mod_cast hω0) hW (ttOut_mono_domain (fun _ h => h.1) (ttOut_wref _ hTT))
  have e : (γ : ℝ) = 1 - (1 - 1 / 400) / (ω : ℝ) := by rw [hγ]; push_cast; ring
  rw [e]
  exact key

/-- `pearlCSampledDevRev1KAt_gen` on one linear at any `m`. -/
theorem pearlCSampledDevRev1KNK_gen [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ c : ℚ) (n k : ℕ) (ω : ℚ) (γ₀ : ℝ)
    (hγ₀ : γ₀ < 1) (hω0 : 0 < ω) (hw0 : ∀ m : ℕ, 0 ≤ wrefDevRev1K d c ⟨m, n, k⟩)
    (hω : ∀ m : ℕ, wrefDevRev1K d c ⟨m, n, k⟩ = ω * ((1 - ρ) * creditDevRev1 d ⟨m, n, k⟩))
    (hTT : TTOutTile CM (pearlCProtocolDevRev1 d sem ρ) (pearlCTilesDevRev1 d sem ρ) (pearlCDomainDev d) γ₀ εPearlC) :
    GγSampled CM (pearlCProtocolDevRev1K d sem ρ c) (pearlCTilesDevRev1K d sem ρ c) (pearlCDomainDevNK d n k)
      (1 - (1 - γ₀) / (ω : ℝ)) εPearlC := by
  have hW : ∀ U : Workload, U.InDomain (pearlCProtocolDevRev1K d sem ρ c) (pearlCDomainDevNK d n k) →
      ∀ (H : Q → R) (s' : S), ∀ g < ((pearlCTilesDevRev1K d sem ρ c).tiling U.layout).NT, ∀ act : Codes,
        (pearlCTilesDevRev1K d sem ρ c).capOK H s' U g act →
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
  exact gammaSampled CM (pearlCProtocolDevRev1K d sem ρ c) (pearlCTilesDevRev1K d sem ρ c)
    (pearlCDomainDevNK d n k) γ₀ (ω : ℝ) εPearlC hγ₀ (by exact_mod_cast hω0) hW hWc
    (ttOutTile_mono_domain (fun _ h => h.1) (ttOutTile_wref _ _ _ hTT))

/-- `pearlCSampledDevRev1KAt` on one linear at any `m`: TT_OUT rev1 per tile at a device record gives `GγSampled`
against the kernel's `W_ref` at `γ = 1 − (399/400)/ω`, where `wrefDevRev1K d c = ω·(1 − ρ)·creditDevRev1 ≥ 0` at every
`m × n × k`. -/
theorem pearlCSampledDevRev1KNK [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ c : ℚ) (n k : ℕ) (ω γ : ℚ) (hω0 : 0 < ω)
    (hw0 : ∀ m : ℕ, 0 ≤ wrefDevRev1K d c ⟨m, n, k⟩)
    (hω : ∀ m : ℕ, wrefDevRev1K d c ⟨m, n, k⟩ = ω * ((1 - ρ) * creditDevRev1 d ⟨m, n, k⟩))
    (hγ : γ = 1 - (1 - 1 / 400) / ω) (hTT : TTOutTilePearlCDevRev1 CM d sem ρ) :
    GγSampled CM (pearlCProtocolDevRev1K d sem ρ c) (pearlCTilesDevRev1K d sem ρ c) (pearlCDomainDevNK d n k)
      (γ : ℝ) εPearlC := by
  have key := pearlCSampledDevRev1KNK_gen CM d sem ρ c n k ω (1 / 400) (by norm_num) hω0 hw0 hω hTT
  have e : (γ : ℝ) = 1 - (1 - 1 / 400) / (ω : ℝ) := by rw [hγ]; push_cast; ring
  rw [e]
  exact key

/-- `pearlCSampledDevRev1KAt_of_cap` on one linear at any `m`: the sampled γ at a cap `ρ' ≤ ρ < 1` from TT_OUT rev1 per
tile at `ρ` is the γ at `ρ`, `1 − (399/400)/ω` where `wrefDevRev1K d c = ω·(1 − ρ)·creditDevRev1 ≥ 0` at every
`m × n × k`. -/
theorem pearlCSampledDevRev1KNK_of_cap [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ ρ' c : ℚ) (n k : ℕ) (ω γ : ℚ)
    (hadd : 0 ≤ d.prices.add) (hbf : 0 ≤ d.prices.bf16) (hle : ρ' ≤ ρ) (hρ1 : ρ < 1) (hω0 : 0 < ω)
    (hw0 : ∀ m : ℕ, 0 ≤ wrefDevRev1K d c ⟨m, n, k⟩)
    (hω : ∀ m : ℕ, wrefDevRev1K d c ⟨m, n, k⟩ = ω * ((1 - ρ) * creditDevRev1 d ⟨m, n, k⟩))
    (hγ : γ = 1 - (1 - 1 / 400) / ω) (hTT : TTOutTilePearlCDevRev1 CM d sem ρ) :
    GγSampled CM (pearlCProtocolDevRev1K d sem ρ' c) (pearlCTilesDevRev1K d sem ρ' c) (pearlCDomainDevNK d n k)
      (γ : ℝ) εPearlC := by
  have hr : (0 : ℚ) < 1 - ρ := by linarith
  have hr' : (0 : ℚ) < 1 - ρ' := by linarith
  have key := pearlCSampledDevRev1KNK_gen CM d sem ρ' c n k (ω * ((1 - ρ) / (1 - ρ'))) _
    (by
      have : (0 : ℝ) < (((1 - ρ) / (1 - ρ') : ℚ) : ℝ) := by exact_mod_cast div_pos hr hr'
      nlinarith)
    (mul_pos hω0 (div_pos hr hr')) hw0
    (fun m => by rw [hω]; field_simp)
    (ttOutTilePearlCDevRev1_cap_transfer CM d sem ρ ρ' (1 / 400) εPearlC hadd hbf hle hρ1 hTT)
  have e : (γ : ℝ) = 1 - (1 - (1 - (1 - 1 / 400) * (((1 - ρ) / (1 - ρ') : ℚ) : ℝ))) /
      ((ω * ((1 - ρ) / (1 - ρ')) : ℚ) : ℝ) := by
    rw [hγ]
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

/-- **v1 under rev1 at the cap 1/1,000, `m × 6144 × 4096` at any `m` (Llama-3.1-8B's and Qwen3-8B's qkv)**, FP32 at
8.376, the honest cast at 8.72: `γ = 851373343/226641700000` (0.37565%). -/
theorem pearlCGammaSm120v1LoopCast8p72Rev1Cap1000_n6144k4096
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 1000)) :
    Gγ CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 6144 4096) ((851373343 / 226641700000 : ℚ) : ℝ) εPearlC :=
  pearlCGammaDevRev1KNK CM _ sem _ _ _ _ (2266417 / 2265828 / (1 - 1 / 1000)) _ (by norm_num)
    (fun m => by rw [wrefDevRev1K_sm120v1LoopCast8p72_n6144k4096 m]; field_simp) (by norm_num) hTT

/-- **v1 under rev1 at the cap 1/1,000 from rev1 at 1/400, `m × 6144 × 4096` at any `m` (Llama-3.1-8B's and Qwen3-8B's
qkv)**, FP32 at 8.376, the honest cast at 8.72: `γ = 851373343/226641700000` (0.37565%). -/
theorem pearlCGammaSm120v1LoopCast8p72Rev1Cap1000_n6144k4096_of_cap400
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    Gγ CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 6144 4096) ((851373343 / 226641700000 : ℚ) : ℝ) εPearlC :=
  pearlCGammaSm120v1LoopCast8p72Rev1Cap1000_n6144k4096 CM sem
    (ttOutPearlCDevRev1_cap_anti CM _ sem (1 / 400) (1 / 1000)
      (by rw [show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]; norm_num [Prices.sm120Loop])
      (by rw [show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]; norm_num [Prices.sm120Loop])
      (by norm_num) (by norm_num) hTT)

/-- **v1 under rev1 per audit tile at the cap 1/1,000, `m × 6144 × 4096` at any `m` (Llama-3.1-8B's and Qwen3-8B's
qkv)**, FP32 at 8.376, the honest cast at 8.72: `γ = 851373343/226641700000` (0.37565%). -/
theorem pearlCSampledSm120v1LoopCast8p72Rev1Cap1000_n6144k4096
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 1000)) :
    GγSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 6144 4096) ((851373343 / 226641700000 : ℚ) : ℝ) εPearlC :=
  pearlCSampledDevRev1KNK CM _ sem _ _ _ _ (2266417 / 2265828 / (1 - 1 / 1000)) _ (by norm_num)
    wrefDevRev1K_sm120v1LoopCast8p72_n6144k4096_nonneg
    (fun m => by rw [wrefDevRev1K_sm120v1LoopCast8p72_n6144k4096 m]; field_simp) (by norm_num) hTT

/-- **v1 under rev1 per audit tile at the cap 1/1,000 from rev1 per tile at 1/400, `m × 6144 × 4096` at any `m`
(Llama-3.1-8B's and Qwen3-8B's qkv)**, FP32 at 8.376, the honest cast at 8.72: `γ = 476159143/90656680000` (0.52523%),
the γ at 1/400. -/
theorem pearlCSampledSm120v1LoopCast8p72Rev1Cap1000_n6144k4096_of_cap400
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GγSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 6144 4096) ((476159143 / 90656680000 : ℚ) : ℝ) εPearlC :=
  pearlCSampledDevRev1KNK_of_cap CM _ sem (1 / 400) (1 / 1000) _ _ _
    (2266417 / 2265828 / (1 - 1 / 400)) _
    (by rw [show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]; norm_num [Prices.sm120Loop])
    (by rw [show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]; norm_num [Prices.sm120Loop])
    (by norm_num) (by norm_num) (by norm_num) wrefDevRev1K_sm120v1LoopCast8p72_n6144k4096_nonneg
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

/-- **v1 under rev1 at the cap 1/1,000, `m × 4096 × 4096` at any `m` (Llama-3.1-8B's and Qwen3-8B's o)**, FP32 at 8.376,
the honest cast at 8.72: `γ = 295006781/75957050000` (0.38839%). -/
theorem pearlCGammaSm120v1LoopCast8p72Rev1Cap1000_n4096k4096
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 1000)) :
    Gγ CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 4096 4096) ((295006781 / 75957050000 : ℚ) : ℝ) εPearlC :=
  pearlCGammaDevRev1KNK CM _ sem _ _ _ _ (1519141 / 1518552 / (1 - 1 / 1000)) _ (by norm_num)
    (fun m => by rw [wrefDevRev1K_sm120v1LoopCast8p72_n4096k4096 m]; field_simp) (by norm_num) hTT

/-- **v1 under rev1 at the cap 1/1,000 from rev1 at 1/400, `m × 4096 × 4096` at any `m` (Llama-3.1-8B's and Qwen3-8B's
o)**, FP32 at 8.376, the honest cast at 8.72: `γ = 295006781/75957050000` (0.38839%). -/
theorem pearlCGammaSm120v1LoopCast8p72Rev1Cap1000_n4096k4096_of_cap400
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    Gγ CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 4096 4096) ((295006781 / 75957050000 : ℚ) : ℝ) εPearlC :=
  pearlCGammaSm120v1LoopCast8p72Rev1Cap1000_n4096k4096 CM sem
    (ttOutPearlCDevRev1_cap_anti CM _ sem (1 / 400) (1 / 1000)
      (by rw [show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]; norm_num [Prices.sm120Loop])
      (by rw [show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]; norm_num [Prices.sm120Loop])
      (by norm_num) (by norm_num) hTT)

/-- **v1 under rev1 per audit tile at the cap 1/1,000, `m × 4096 × 4096` at any `m` (Llama-3.1-8B's and Qwen3-8B's o)**,
FP32 at 8.376, the honest cast at 8.72: `γ = 295006781/75957050000` (0.38839%). -/
theorem pearlCSampledSm120v1LoopCast8p72Rev1Cap1000_n4096k4096
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 1000)) :
    GγSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 4096 4096) ((295006781 / 75957050000 : ℚ) : ℝ) εPearlC :=
  pearlCSampledDevRev1KNK CM _ sem _ _ _ _ (1519141 / 1518552 / (1 - 1 / 1000)) _ (by norm_num)
    wrefDevRev1K_sm120v1LoopCast8p72_n4096k4096_nonneg
    (fun m => by rw [wrefDevRev1K_sm120v1LoopCast8p72_n4096k4096 m]; field_simp) (by norm_num) hTT

/-- **v1 under rev1 per audit tile at the cap 1/1,000 from rev1 per tile at 1/400, `m × 4096 × 4096` at any `m`
(Llama-3.1-8B's and Qwen3-8B's o)**, FP32 at 8.376, the honest cast at 8.72: `γ = 163445381/30382820000` (0.53795%), the
γ at 1/400. -/
theorem pearlCSampledSm120v1LoopCast8p72Rev1Cap1000_n4096k4096_of_cap400
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GγSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 4096 4096) ((163445381 / 30382820000 : ℚ) : ℝ) εPearlC :=
  pearlCSampledDevRev1KNK_of_cap CM _ sem (1 / 400) (1 / 1000) _ _ _
    (1519141 / 1518552 / (1 - 1 / 400)) _
    (by rw [show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]; norm_num [Prices.sm120Loop])
    (by rw [show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]; norm_num [Prices.sm120Loop])
    (by norm_num) (by norm_num) (by norm_num) wrefDevRev1K_sm120v1LoopCast8p72_n4096k4096_nonneg
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

/-- **v1 under rev1 at the cap 1/1,000, `m × 28672 × 4096` at any `m` (Llama-3.1-8B's gate_up)**, FP32 at 8.376, the
honest cast at 8.72: `γ = 1863165467/524322650000` (0.35535%). -/
theorem pearlCGammaSm120v1LoopCast8p72Rev1Cap1000_n28672k4096
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 1000)) :
    Gγ CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 28672 4096) ((1863165467 / 524322650000 : ℚ) : ℝ) εPearlC :=
  pearlCGammaDevRev1KNK CM _ sem _ _ _ _ (10486453 / 10485864 / (1 - 1 / 1000)) _ (by norm_num)
    (fun m => by rw [wrefDevRev1K_sm120v1LoopCast8p72_n28672k4096 m]; field_simp) (by norm_num) hTT

/-- **v1 under rev1 at the cap 1/1,000 from rev1 at 1/400, `m × 28672 × 4096` at any `m` (Llama-3.1-8B's gate_up)**,
FP32 at 8.376, the honest cast at 8.72: `γ = 1863165467/524322650000` (0.35535%). -/
theorem pearlCGammaSm120v1LoopCast8p72Rev1Cap1000_n28672k4096_of_cap400
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    Gγ CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 28672 4096) ((1863165467 / 524322650000 : ℚ) : ℝ) εPearlC :=
  pearlCGammaSm120v1LoopCast8p72Rev1Cap1000_n28672k4096 CM sem
    (ttOutPearlCDevRev1_cap_anti CM _ sem (1 / 400) (1 / 1000)
      (by rw [show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]; norm_num [Prices.sm120Loop])
      (by rw [show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]; norm_num [Prices.sm120Loop])
      (by norm_num) (by norm_num) hTT)

/-- **v1 under rev1 per audit tile at the cap 1/1,000, `m × 28672 × 4096` at any `m` (Llama-3.1-8B's gate_up)**, FP32 at
8.376, the honest cast at 8.72: `γ = 1863165467/524322650000` (0.35535%). -/
theorem pearlCSampledSm120v1LoopCast8p72Rev1Cap1000_n28672k4096
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 1000)) :
    GγSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 28672 4096) ((1863165467 / 524322650000 : ℚ) : ℝ) εPearlC :=
  pearlCSampledDevRev1KNK CM _ sem _ _ _ _ (10486453 / 10485864 / (1 - 1 / 1000)) _ (by norm_num)
    wrefDevRev1K_sm120v1LoopCast8p72_n28672k4096_nonneg
    (fun m => by rw [wrefDevRev1K_sm120v1LoopCast8p72_n28672k4096 m]; field_simp) (by norm_num) hTT

/-- **v1 under rev1 per audit tile at the cap 1/1,000 from rev1 per tile at 1/400, `m × 28672 × 4096` at any `m`
(Llama-3.1-8B's gate_up)**, FP32 at 8.376, the honest cast at 8.72: `γ = 1059055667/209729060000` (0.50496%), the γ at
1/400. -/
theorem pearlCSampledSm120v1LoopCast8p72Rev1Cap1000_n28672k4096_of_cap400
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GγSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 28672 4096) ((1059055667 / 209729060000 : ℚ) : ℝ) εPearlC :=
  pearlCSampledDevRev1KNK_of_cap CM _ sem (1 / 400) (1 / 1000) _ _ _
    (10486453 / 10485864 / (1 - 1 / 400)) _
    (by rw [show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]; norm_num [Prices.sm120Loop])
    (by rw [show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]; norm_num [Prices.sm120Loop])
    (by norm_num) (by norm_num) (by norm_num) wrefDevRev1K_sm120v1LoopCast8p72_n28672k4096_nonneg
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

/-- **v1 under rev1 at the cap 1/1,000, `m × 4096 × 14336` at any `m` (Llama-3.1-8B's down)**, FP32 at 8.376, the honest
cast at 8.72: `γ = 3045273083/782596025000` (0.38912%). -/
theorem pearlCGammaSm120v1LoopCast8p72Rev1Cap1000_n4096k14336
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 1000)) :
    Gγ CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 4096 14336) ((3045273083 / 782596025000 : ℚ) : ℝ) εPearlC :=
  pearlCGammaDevRev1KNK CM _ sem _ _ _ _ (31303841 / 31291472 / (1 - 1 / 1000)) _ (by norm_num)
    (fun m => by rw [wrefDevRev1K_sm120v1LoopCast8p72_n4096k14336 m]; field_simp) (by norm_num) hTT

/-- **v1 under rev1 at the cap 1/1,000 from rev1 at 1/400, `m × 4096 × 14336` at any `m` (Llama-3.1-8B's down)**, FP32
at 8.376, the honest cast at 8.72: `γ = 3045273083/782596025000` (0.38912%). -/
theorem pearlCGammaSm120v1LoopCast8p72Rev1Cap1000_n4096k14336_of_cap400
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    Gγ CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 4096 14336) ((3045273083 / 782596025000 : ℚ) : ℝ) εPearlC :=
  pearlCGammaSm120v1LoopCast8p72Rev1Cap1000_n4096k14336 CM sem
    (ttOutPearlCDevRev1_cap_anti CM _ sem (1 / 400) (1 / 1000)
      (by rw [show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]; norm_num [Prices.sm120Loop])
      (by rw [show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]; norm_num [Prices.sm120Loop])
      (by norm_num) (by norm_num) hTT)

/-- **v1 under rev1 per audit tile at the cap 1/1,000, `m × 4096 × 14336` at any `m` (Llama-3.1-8B's down)**, FP32 at
8.376, the honest cast at 8.72: `γ = 3045273083/782596025000` (0.38912%). -/
theorem pearlCSampledSm120v1LoopCast8p72Rev1Cap1000_n4096k14336
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 1000)) :
    GγSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 4096 14336) ((3045273083 / 782596025000 : ℚ) : ℝ) εPearlC :=
  pearlCSampledDevRev1KNK CM _ sem _ _ _ _ (31303841 / 31291472 / (1 - 1 / 1000)) _ (by norm_num)
    wrefDevRev1K_sm120v1LoopCast8p72_n4096k14336_nonneg
    (fun m => by rw [wrefDevRev1K_sm120v1LoopCast8p72_n4096k14336 m]; field_simp) (by norm_num) hTT

/-- **v1 under rev1 per audit tile at the cap 1/1,000 from rev1 per tile at 1/400, `m × 4096 × 14336` at any `m`
(Llama-3.1-8B's down)**, FP32 at 8.376, the honest cast at 8.72: `γ = 1686307883/313038410000` (0.53869%), the γ at
1/400. -/
theorem pearlCSampledSm120v1LoopCast8p72Rev1Cap1000_n4096k14336_of_cap400
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GγSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 4096 14336) ((1686307883 / 313038410000 : ℚ) : ℝ) εPearlC :=
  pearlCSampledDevRev1KNK_of_cap CM _ sem (1 / 400) (1 / 1000) _ _ _
    (31303841 / 31291472 / (1 - 1 / 400)) _
    (by rw [show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]; norm_num [Prices.sm120Loop])
    (by rw [show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]; norm_num [Prices.sm120Loop])
    (by norm_num) (by norm_num) (by norm_num) wrefDevRev1K_sm120v1LoopCast8p72_n4096k14336_nonneg
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

/-- **v1 under rev1 at the cap 1/1,000, `m × 24576 × 4096` at any `m` (Qwen3-8B's gate_up)**, FP32 at 8.376, the honest
cast at 8.72: `γ = 800902843/224797525000` (0.35628%). -/
theorem pearlCGammaSm120v1LoopCast8p72Rev1Cap1000_n24576k4096
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 1000)) :
    Gγ CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 24576 4096) ((800902843 / 224797525000 : ℚ) : ℝ) εPearlC :=
  pearlCGammaDevRev1KNK CM _ sem _ _ _ _ (8991901 / 8991312 / (1 - 1 / 1000)) _ (by norm_num)
    (fun m => by rw [wrefDevRev1K_sm120v1LoopCast8p72_n24576k4096 m]; field_simp) (by norm_num) hTT

/-- **v1 under rev1 at the cap 1/1,000 from rev1 at 1/400, `m × 24576 × 4096` at any `m` (Qwen3-8B's gate_up)**, FP32 at
8.376, the honest cast at 8.72: `γ = 800902843/224797525000` (0.35628%). -/
theorem pearlCGammaSm120v1LoopCast8p72Rev1Cap1000_n24576k4096_of_cap400
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    Gγ CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 24576 4096) ((800902843 / 224797525000 : ℚ) : ℝ) εPearlC :=
  pearlCGammaSm120v1LoopCast8p72Rev1Cap1000_n24576k4096 CM sem
    (ttOutPearlCDevRev1_cap_anti CM _ sem (1 / 400) (1 / 1000)
      (by rw [show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]; norm_num [Prices.sm120Loop])
      (by rw [show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]; norm_num [Prices.sm120Loop])
      (by norm_num) (by norm_num) hTT)

/-- **v1 under rev1 per audit tile at the cap 1/1,000, `m × 24576 × 4096` at any `m` (Qwen3-8B's gate_up)**, FP32 at
8.376, the honest cast at 8.72: `γ = 800902843/224797525000` (0.35628%). -/
theorem pearlCSampledSm120v1LoopCast8p72Rev1Cap1000_n24576k4096
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 1000)) :
    GγSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 24576 4096) ((800902843 / 224797525000 : ℚ) : ℝ) εPearlC :=
  pearlCSampledDevRev1KNK CM _ sem _ _ _ _ (8991901 / 8991312 / (1 - 1 / 1000)) _ (by norm_num)
    wrefDevRev1K_sm120v1LoopCast8p72_n24576k4096_nonneg
    (fun m => by rw [wrefDevRev1K_sm120v1LoopCast8p72_n24576k4096 m]; field_simp) (by norm_num) hTT

/-- **v1 under rev1 per audit tile at the cap 1/1,000 from rev1 per tile at 1/400, `m × 24576 × 4096` at any `m`
(Qwen3-8B's gate_up)**, FP32 at 8.376, the honest cast at 8.72: `γ = 454893643/89919010000` (0.50589%), the γ at 1/400.
-/
theorem pearlCSampledSm120v1LoopCast8p72Rev1Cap1000_n24576k4096_of_cap400
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GγSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 24576 4096) ((454893643 / 89919010000 : ℚ) : ℝ) εPearlC :=
  pearlCSampledDevRev1KNK_of_cap CM _ sem (1 / 400) (1 / 1000) _ _ _
    (8991901 / 8991312 / (1 - 1 / 400)) _
    (by rw [show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]; norm_num [Prices.sm120Loop])
    (by rw [show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]; norm_num [Prices.sm120Loop])
    (by norm_num) (by norm_num) (by norm_num) wrefDevRev1K_sm120v1LoopCast8p72_n24576k4096_nonneg
    (fun m => by rw [wrefDevRev1K_sm120v1LoopCast8p72_n24576k4096 m]; field_simp) (by norm_num) hTT

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

/-- **v1 under rev1 at the cap 1/1,000, `m × 4096 × 12288` at any `m` (Qwen3-8B's down)**, FP32 at 8.376, the honest
cast at 8.72: `γ = 74663501/19190030000` (0.38907%). -/
theorem pearlCGammaSm120v1LoopCast8p72Rev1Cap1000_n4096k12288
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 1000)) :
    Gγ CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 4096 12288) ((74663501 / 19190030000 : ℚ) : ℝ) εPearlC :=
  pearlCGammaDevRev1KNK CM _ sem _ _ _ _ (13433021 / 13427720 / (1 - 1 / 1000)) _ (by norm_num)
    (fun m => by rw [wrefDevRev1K_sm120v1LoopCast8p72_n4096k12288 m]; field_simp) (by norm_num) hTT

/-- **v1 under rev1 at the cap 1/1,000 from rev1 at 1/400, `m × 4096 × 12288` at any `m` (Qwen3-8B's down)**, FP32 at
8.376, the honest cast at 8.72: `γ = 74663501/19190030000` (0.38907%). -/
theorem pearlCGammaSm120v1LoopCast8p72Rev1Cap1000_n4096k12288_of_cap400
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    Gγ CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 4096 12288) ((74663501 / 19190030000 : ℚ) : ℝ) εPearlC :=
  pearlCGammaSm120v1LoopCast8p72Rev1Cap1000_n4096k12288 CM sem
    (ttOutPearlCDevRev1_cap_anti CM _ sem (1 / 400) (1 / 1000)
      (by rw [show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]; norm_num [Prices.sm120Loop])
      (by rw [show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]; norm_num [Prices.sm120Loop])
      (by norm_num) (by norm_num) hTT)

/-- **v1 under rev1 per audit tile at the cap 1/1,000, `m × 4096 × 12288` at any `m` (Qwen3-8B's down)**, FP32 at 8.376,
the honest cast at 8.72: `γ = 74663501/19190030000` (0.38907%). -/
theorem pearlCSampledSm120v1LoopCast8p72Rev1Cap1000_n4096k12288
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 1000)) :
    GγSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 4096 12288) ((74663501 / 19190030000 : ℚ) : ℝ) εPearlC :=
  pearlCSampledDevRev1KNK CM _ sem _ _ _ _ (13433021 / 13427720 / (1 - 1 / 1000)) _ (by norm_num)
    wrefDevRev1K_sm120v1LoopCast8p72_n4096k12288_nonneg
    (fun m => by rw [wrefDevRev1K_sm120v1LoopCast8p72_n4096k12288 m]; field_simp) (by norm_num) hTT

/-- **v1 under rev1 per audit tile at the cap 1/1,000 from rev1 per tile at 1/400, `m × 4096 × 12288` at any `m`
(Qwen3-8B's down)**, FP32 at 8.376, the honest cast at 8.72: `γ = 41346101/7676012000` (0.53864%), the γ at 1/400. -/
theorem pearlCSampledSm120v1LoopCast8p72Rev1Cap1000_n4096k12288_of_cap400
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 400)) :
    GγSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevNK (devSm120v1 Prices.sm120Loop) 4096 12288) ((41346101 / 7676012000 : ℚ) : ℝ) εPearlC :=
  pearlCSampledDevRev1KNK_of_cap CM _ sem (1 / 400) (1 / 1000) _ _ _
    (13433021 / 13427720 / (1 - 1 / 400)) _
    (by rw [show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]; norm_num [Prices.sm120Loop])
    (by rw [show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]; norm_num [Prices.sm120Loop])
    (by norm_num) (by norm_num) (by norm_num) wrefDevRev1K_sm120v1LoopCast8p72_n4096k12288_nonneg
    (fun m => by rw [wrefDevRev1K_sm120v1LoopCast8p72_n4096k12288 m]; field_simp) (by norm_num) hTT

end Served

end Pouw.PearlC
