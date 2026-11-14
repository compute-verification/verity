module

public import Proofs.Pouw.PearlC.ServedRev1Gap
public import Definitions.Pouw.PearlC.DeviceWithin

/-!
# v1 under rev1 on a served step's layout, every linear at once (compute accounting's file)

* **A set of linears at any `m`:** `pearlCDomainDevShapes d nks` holds a layout whose units are any linears in `nks`,
  each at any mix of batch sizes, as a served step's layout is. `pearlCGapDevRev1KShapes`,
  `pearlCSampledDevRev1KShapes` and `pearlCSampledDevRev1KShapes_of_cap` are `ServedRev1Gap`'s one-linear theorems
  with `ω` any bound on the ratio `wrefDevRev1K/((1 − ρ)·creditDevRev1)` over `nks`.
* **Any set of shapes:** `pearlCGapDevRev1KOn` and `pearlCSampledDevRev1KOn_gen` are the same on the domain restricted
  to any shapes `Ss` (`pearlCDomainDevOn`). The theorems on `nks` are them at the shapes whose `(n, k)` is in `nks`, and
  the guarantees over every device record (`Proofs.Pouw.PearlC.DevGap`) are them at W1.
* **The g is the max of the per-linear g.** `Wins` compares the adversary's online cost with the honest work of its
  correct units `K`, and the adversary picks `K`. The reduction turns TT_OUT's bound on credit into one on `W_ref`
  through `∑_K W_ref ≤ ω·∑_K credit`, so `ω` must bound the credit-weighted mediant of the per-unit ratios over every
  `K` the adversary may pick. That includes a `K` inside the worst linear's units, where the mediant is that linear's
  ratio, so the least such `ω` is the max of the per-linear ratios over `nks`, and no weighting by the layout's mix of
  linears holds against an adversary that is correct only on the worst one. `g = 1 − (1 − g₀)/ω` increases with `ω`, so
  at that `ω` it is the max over `nks` of the per-linear g.
* **The gap at the served models** at v1's record with FP32 at 8.376 and the honest cast at 8.72, at the cap 1/1,000:
  Llama-3.1-8B's four linears (6144, 4096) qkv, (4096, 4096) o, (28672, 4096) gate_up, (4096, 14336) down, and
  Qwen3-8B's (6144, 4096), (4096, 4096), (24576, 4096) gate_up, (4096, 12288) down. Each model's max is its down
  projection's: per unit and per audit tile from TT_OUT at 1/1,000, and per unit from TT_OUT at 1/400, 0.38912%
  (Llama-3.1-8B) and 0.38907% (Qwen3-8B); per tile from TT_OUT per tile at 1/400, 0.53869% and 0.53864%.
-/

@[expose] public section

namespace Pouw.PearlC

open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

section
variable {Q R S : Type}

theorem mul_share_le_cast {w C t ω ρ : ℚ} (h : w ≤ ω * ((1 - ρ) * C)) (ht : 0 ≤ t) :
    ((w * t : ℚ) : ℝ) ≤ (ω : ℝ) * (((1 - ρ) * C * t : ℚ) : ℝ) := by
  have : w * t ≤ ω * ((1 - ρ) * C * t) := by
    have := mul_le_mul_of_nonneg_right h ht
    linarith [show ω * ((1 - ρ) * C) * t = ω * ((1 - ρ) * C * t) by ring]
  exact_mod_cast this

/-- TT_OUT rev1 at a device record gives `G_g` against the kernel's `W_ref` on the domain restricted to the shapes `Ss`,
at `g = 1 − (399/400)/ω`, where `wrefDevRev1K d c ≤ ω·(1 − ρ)·creditDevRev1` at every shape `Ss` holds of. -/
theorem pearlCGapDevRev1KOn [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ c : ℚ) (Ss : Shape → Prop) (ω : ℚ)
    (hω0 : 0 < ω) (hω : ∀ s, Ss s → wrefDevRev1K d c s ≤ ω * ((1 - ρ) * creditDevRev1 d s))
    (hTT : TTOutPearlCDevRev1 CM d sem ρ) :
    GGap CM (pearlCProtocolDevRev1K d sem ρ c) (pearlCDomainDevOn d Ss) ((1 - (1 - 1 / 400) / ω : ℚ) : ℝ)
      δPearlC := by
  have hW : ∀ U : Workload, U.InDomain (pearlCProtocolDevRev1K d sem ρ c) (pearlCDomainDevOn d Ss) →
      ∀ (H : Q → R) (s' : S), ∀ u < U.layout.N, ∀ act : Codes,
        (pearlCProtocolDevRev1K d sem ρ c).actOK (U.layout.shape u) act →
        (pearlCProtocolDevRev1K d sem ρ c).capOK H s' U u act →
          (pearlCProtocolDevRev1K d sem ρ c).Wref U.layout u ≤
            (ω : ℝ) * (pearlCProtocolDevRev1K d sem ρ c).credit H s' U u act := by
    intro U hU H s' u hu act _ hcap
    have hc : sem.unitDebitRev1 d H s' U u act ≤ ρ * creditDevRev1 d (U.layout.shape u) := hcap
    have h1 : wrefDevRev1K d c (U.layout.shape u) ≤ ω * ((1 - ρ) * creditDevRev1 d (U.layout.shape u)) :=
      hω _ (hU.2.2 u hu)
    show (wrefDevRev1K d c (U.layout.shape u) : ℝ) ≤
      (ω : ℝ) * ((creditDevRev1 d (U.layout.shape u) - sem.unitDebitRev1 d H s' U u act : ℚ) : ℝ)
    have : ω * ((1 - ρ) * creditDevRev1 d (U.layout.shape u)) ≤
        ω * (creditDevRev1 d (U.layout.shape u) - sem.unitDebitRev1 d H s' U u act) :=
      mul_le_mul_of_nonneg_left (by linarith) hω0.le
    exact_mod_cast h1.trans this
  have key := gapFromTTOut CM (pearlCProtocolDevRev1K d sem ρ c) (pearlCDomainDevOn d Ss) (1 / 400) (ω : ℝ)
    δPearlC (by exact_mod_cast hω0) hW (ttOut_mono_domain (fun _ h => h.1) (ttOut_wref _ hTT))
  have e : (((1 - (1 - 1 / 400) / ω : ℚ)) : ℝ) = 1 - (1 - 1 / 400) / (ω : ℝ) := by push_cast; ring
  rw [e]
  exact key

/-- `pearlCGapDevRev1KNK` on a set of linears at any `m`: TT_OUT rev1 at a device record gives `G_g` against the
kernel's `W_ref` at `g = 1 − (399/400)/ω`, where `wrefDevRev1K d c ≤ ω·(1 − ρ)·creditDevRev1` at every `m × n × k` with
`(n, k) ∈ nks`. The least such `ω` is the max of the per-linear ratios, and its g the max of the per-linear g. -/
theorem pearlCGapDevRev1KShapes [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ c : ℚ) (nks : Finset (ℕ × ℕ)) (ω g : ℚ)
    (hω0 : 0 < ω)
    (hω : ∀ nk ∈ nks, ∀ m : ℕ,
      wrefDevRev1K d c ⟨m, nk.1, nk.2⟩ ≤ ω * ((1 - ρ) * creditDevRev1 d ⟨m, nk.1, nk.2⟩))
    (hg : g = 1 - (1 - 1 / 400) / ω) (hTT : TTOutPearlCDevRev1 CM d sem ρ) :
    GGap CM (pearlCProtocolDevRev1K d sem ρ c) (pearlCDomainDevShapes d nks) (g : ℝ) δPearlC := by
  subst hg
  exact pearlCGapDevRev1KOn CM d sem ρ c (fun s => (s.n, s.k) ∈ nks) ω hω0 (fun s hs => hω _ hs s.m) hTT

/-- TT_OUT rev1 per tile, at any `g₀`, gives `GGapSampled` against the kernel's `W_ref` on the domain restricted to the
shapes `Ss`, at `g = 1 − (1 − g₀)/ω`, where `0 ≤ wrefDevRev1K d c ≤ ω·(1 − ρ)·creditDevRev1` at every shape `Ss` holds
of. -/
theorem pearlCSampledDevRev1KOn_gen [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ c : ℚ) (Ss : Shape → Prop) (ω : ℚ)
    (g₀ : ℝ) (hg₀ : g₀ < 1) (hω0 : 0 < ω) (hw0 : ∀ s, Ss s → 0 ≤ wrefDevRev1K d c s)
    (hω : ∀ s, Ss s → wrefDevRev1K d c s ≤ ω * ((1 - ρ) * creditDevRev1 d s))
    (hTT : TTOutTile CM (pearlCProtocolDevRev1 d sem ρ) (pearlCTilesDevRev1 d sem ρ) (pearlCDomainDev d) g₀ δPearlC) :
    GGapSampled CM (pearlCProtocolDevRev1K d sem ρ c) (pearlCTilesDevRev1K d sem ρ c) (pearlCDomainDevOn d Ss)
      (1 - (1 - g₀) / (ω : ℝ)) δPearlC := by
  have hW : ∀ U : Workload, U.InDomain (pearlCProtocolDevRev1K d sem ρ c) (pearlCDomainDevOn d Ss) →
      ∀ (H : Q → R) (s' : S), ∀ g < ((pearlCTilesDevRev1K d sem ρ c).tiling U.layout).NT, ∀ act : Codes,
        (pearlCTilesDevRev1K d sem ρ c).capOK H s' U g act →
        (pearlCTilesDevRev1K d sem ρ c).Wcred U.layout g act ≤
          (ω : ℝ) * (pearlCTilesDevRev1K d sem ρ c).credit H s' U g act := by
    intro U hU H s' g hg act _
    have hu := auditTiling_unit_lt 64 64 U.layout hg
    have h1 : wrefDevRev1K d c (U.layout.shape ((auditTiling 64 64 U.layout).unit g)) ≤
        ω * ((1 - ρ) * creditDevRev1 d (U.layout.shape ((auditTiling 64 64 U.layout).unit g))) :=
      hω _ (hU.2.2 _ hu)
    simp only [pearlCTilesDevRev1K, pearlCTilesDevRev1]
    exact mul_share_le_cast h1 (tileShare_nonneg _ _ _)
  have hWc : ∀ U : Workload, U.InDomain (pearlCProtocolDevRev1K d sem ρ c) (pearlCDomainDevOn d Ss) →
      ∀ g < ((pearlCTilesDevRev1K d sem ρ c).tiling U.layout).NT, ∀ act : Codes,
        (pearlCTilesDevRev1K d sem ρ c).Wcred U.layout g act ≤ (pearlCTilesDevRev1K d sem ρ c).Wref U.layout g := by
    intro U hU g hg act
    have hu := auditTiling_unit_lt 64 64 U.layout hg
    have h0 : 0 ≤ wrefDevRev1K d c (U.layout.shape ((auditTiling 64 64 U.layout).unit g)) := hw0 _ (hU.2.2 _ hu)
    simp only [pearlCTilesDevRev1K, pearlCTilesDevRev1]
    exact_mod_cast mul_le_mul_of_nonneg_left (tileShare_mono _ (passRows_subset sem _ _ _) _) h0
  exact gapSampled CM (pearlCProtocolDevRev1K d sem ρ c) (pearlCTilesDevRev1K d sem ρ c)
    (pearlCDomainDevOn d Ss) g₀ (ω : ℝ) δPearlC hg₀ (by exact_mod_cast hω0) hW hWc
    (ttOutTile_mono_domain (fun _ h => h.1) (ttOutTile_wref _ _ _ hTT))

/-- `pearlCSampledDevRev1KNK_gen` on a set of linears at any `m`. -/
theorem pearlCSampledDevRev1KShapes_gen [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ c : ℚ) (nks : Finset (ℕ × ℕ)) (ω : ℚ)
    (g₀ : ℝ) (hg₀ : g₀ < 1) (hω0 : 0 < ω)
    (hw0 : ∀ nk ∈ nks, ∀ m : ℕ, 0 ≤ wrefDevRev1K d c ⟨m, nk.1, nk.2⟩)
    (hω : ∀ nk ∈ nks, ∀ m : ℕ,
      wrefDevRev1K d c ⟨m, nk.1, nk.2⟩ ≤ ω * ((1 - ρ) * creditDevRev1 d ⟨m, nk.1, nk.2⟩))
    (hTT : TTOutTile CM (pearlCProtocolDevRev1 d sem ρ) (pearlCTilesDevRev1 d sem ρ) (pearlCDomainDev d) g₀ δPearlC) :
    GGapSampled CM (pearlCProtocolDevRev1K d sem ρ c) (pearlCTilesDevRev1K d sem ρ c) (pearlCDomainDevShapes d nks)
      (1 - (1 - g₀) / (ω : ℝ)) δPearlC :=
  pearlCSampledDevRev1KOn_gen CM d sem ρ c (fun s => (s.n, s.k) ∈ nks) ω g₀ hg₀ hω0 (fun s hs => hw0 _ hs s.m)
    (fun s hs => hω _ hs s.m) hTT

/-- `pearlCSampledDevRev1KNK` on a set of linears at any `m`: TT_OUT rev1 per tile at a device record gives
`GGapSampled` against the kernel's `W_ref` at `g = 1 − (399/400)/ω`, where `0 ≤ wrefDevRev1K d c ≤
ω·(1 − ρ)·creditDevRev1` at every `m × n × k` with `(n, k) ∈ nks`. -/
theorem pearlCSampledDevRev1KShapes [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ c : ℚ) (nks : Finset (ℕ × ℕ)) (ω g : ℚ)
    (hω0 : 0 < ω) (hw0 : ∀ nk ∈ nks, ∀ m : ℕ, 0 ≤ wrefDevRev1K d c ⟨m, nk.1, nk.2⟩)
    (hω : ∀ nk ∈ nks, ∀ m : ℕ,
      wrefDevRev1K d c ⟨m, nk.1, nk.2⟩ ≤ ω * ((1 - ρ) * creditDevRev1 d ⟨m, nk.1, nk.2⟩))
    (hg : g = 1 - (1 - 1 / 400) / ω) (hTT : TTOutTilePearlCDevRev1 CM d sem ρ) :
    GGapSampled CM (pearlCProtocolDevRev1K d sem ρ c) (pearlCTilesDevRev1K d sem ρ c) (pearlCDomainDevShapes d nks)
      (g : ℝ) δPearlC := by
  have key := pearlCSampledDevRev1KShapes_gen CM d sem ρ c nks ω (1 / 400) (by norm_num) hω0 hw0 hω hTT
  have e : (g : ℝ) = 1 - (1 - 1 / 400) / (ω : ℝ) := by rw [hg]; push_cast; ring
  rw [e]
  exact key

/-- `pearlCSampledDevRev1KNK_of_cap` on a set of linears at any `m`: the sampled g at a cap `ρ' ≤ ρ < 1` from TT_OUT
rev1 per tile at `ρ` is the g at `ρ`, `1 − (399/400)/ω` where `0 ≤ wrefDevRev1K d c ≤ ω·(1 − ρ)·creditDevRev1` at every
`m × n × k` with `(n, k) ∈ nks`. -/
theorem pearlCSampledDevRev1KShapes_of_cap [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ ρ' c : ℚ) (nks : Finset (ℕ × ℕ)) (ω g : ℚ)
    (hadd : 0 ≤ d.prices.add) (hbf : 0 ≤ d.prices.bf16) (hle : ρ' ≤ ρ) (hρ1 : ρ < 1) (hω0 : 0 < ω)
    (hw0 : ∀ nk ∈ nks, ∀ m : ℕ, 0 ≤ wrefDevRev1K d c ⟨m, nk.1, nk.2⟩)
    (hω : ∀ nk ∈ nks, ∀ m : ℕ,
      wrefDevRev1K d c ⟨m, nk.1, nk.2⟩ ≤ ω * ((1 - ρ) * creditDevRev1 d ⟨m, nk.1, nk.2⟩))
    (hg : g = 1 - (1 - 1 / 400) / ω) (hTT : TTOutTilePearlCDevRev1 CM d sem ρ) :
    GGapSampled CM (pearlCProtocolDevRev1K d sem ρ' c) (pearlCTilesDevRev1K d sem ρ' c) (pearlCDomainDevShapes d nks)
      (g : ℝ) δPearlC := by
  have hr : (0 : ℚ) < 1 - ρ := by linarith
  have hr' : (0 : ℚ) < 1 - ρ' := by linarith
  have key := pearlCSampledDevRev1KShapes_gen CM d sem ρ' c nks (ω * ((1 - ρ) / (1 - ρ'))) _
    (by
      have : (0 : ℝ) < (((1 - ρ) / (1 - ρ') : ℚ) : ℝ) := by exact_mod_cast div_pos hr hr'
      nlinarith)
    (mul_pos hω0 (div_pos hr hr')) hw0
    (fun nk hnk m => by
      have e : ω * ((1 - ρ) / (1 - ρ')) * ((1 - ρ') * creditDevRev1 d ⟨m, nk.1, nk.2⟩) =
          ω * ((1 - ρ) * creditDevRev1 d ⟨m, nk.1, nk.2⟩) := by field_simp
      rw [e]
      exact hω nk hnk m)
    (ttOutTilePearlCDevRev1_cap_transfer CM d sem ρ ρ' (1 / 400) δPearlC hadd hbf hle hρ1 hTT)
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

/-- A per-linear ratio `r ≤ r_max` bounds the linear's work by `ω = r_max/(1 − ρ)` against `(1 − ρ)·credit`. -/
theorem ratio_le_max_div {r rmax ρ C : ℚ} (hρ : ρ < 1) (hr : r ≤ rmax) (hC : 0 ≤ C) :
    r * C ≤ rmax / (1 - ρ) * ((1 - ρ) * C) := by
  have h : rmax / (1 - ρ) * ((1 - ρ) * C) = rmax * C := by
    have : (1 - ρ) ≠ 0 := by linarith
    field_simp
  rw [h]
  exact mul_le_mul_of_nonneg_right hr hC

end

section Served

variable {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
  (sem : PearlCSem Q R S)

theorem creditDevRev1_sm120v1Loop_nonneg (m n k : ℕ) (hk : 0 < k) (h128 : 128 ∣ k) :
    0 ≤ creditDevRev1 (devSm120v1 Prices.sm120Loop) ⟨m, n, k⟩ :=
  creditDevRev1_nonneg _
    (by rw [show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]; norm_num [Prices.sm120Loop])
    (by rw [show (devSm120v1 Prices.sm120Loop).prices = Prices.sm120Loop from rfl]; norm_num [Prices.sm120Loop])
    _ hk (Or.inr (by rw [show (devSm120v1 Prices.sm120Loop).G = 4 from rfl]; exact h128))

theorem wrefDevRev1K_sm120v1LoopCast8p72_llama31_8b_nonneg :
    ∀ nk ∈ llama31_8bLinears, ∀ m : ℕ,
      0 ≤ wrefDevRev1K (devSm120v1 Prices.sm120Loop) (218 / 25 - 8953 / 1000) ⟨m, nk.1, nk.2⟩ := by
  intro nk hnk m
  simp only [llama31_8bLinears, mem_insert, mem_singleton] at hnk
  rcases hnk with rfl | rfl | rfl | rfl
  · exact wrefDevRev1K_sm120v1LoopCast8p72_n6144k4096_nonneg m
  · exact wrefDevRev1K_sm120v1LoopCast8p72_n4096k4096_nonneg m
  · exact wrefDevRev1K_sm120v1LoopCast8p72_n28672k4096_nonneg m
  · exact wrefDevRev1K_sm120v1LoopCast8p72_n4096k14336_nonneg m

/-- Llama-3.1-8B's largest ratio `wrefDevRev1K/creditDevRev1` is its down projection's, `31303841/31291472`. -/
theorem wrefDevRev1K_sm120v1LoopCast8p72_llama31_8b_le (ρ : ℚ) (hρ : ρ < 1) :
    ∀ nk ∈ llama31_8bLinears, ∀ m : ℕ,
      wrefDevRev1K (devSm120v1 Prices.sm120Loop) (218 / 25 - 8953 / 1000) ⟨m, nk.1, nk.2⟩ ≤
        31303841 / 31291472 / (1 - ρ) * ((1 - ρ) * creditDevRev1 (devSm120v1 Prices.sm120Loop) ⟨m, nk.1, nk.2⟩) := by
  intro nk hnk m
  simp only [llama31_8bLinears, mem_insert, mem_singleton] at hnk
  rcases hnk with rfl | rfl | rfl | rfl
  · rw [wrefDevRev1K_sm120v1LoopCast8p72_n6144k4096 m]
    exact ratio_le_max_div hρ (by norm_num) (creditDevRev1_sm120v1Loop_nonneg _ _ _ (by norm_num) (by norm_num))
  · rw [wrefDevRev1K_sm120v1LoopCast8p72_n4096k4096 m]
    exact ratio_le_max_div hρ (by norm_num) (creditDevRev1_sm120v1Loop_nonneg _ _ _ (by norm_num) (by norm_num))
  · rw [wrefDevRev1K_sm120v1LoopCast8p72_n28672k4096 m]
    exact ratio_le_max_div hρ (by norm_num) (creditDevRev1_sm120v1Loop_nonneg _ _ _ (by norm_num) (by norm_num))
  · rw [wrefDevRev1K_sm120v1LoopCast8p72_n4096k14336 m]
    exact ratio_le_max_div hρ (by norm_num) (creditDevRev1_sm120v1Loop_nonneg _ _ _ (by norm_num) (by norm_num))

/-- **v1 under rev1 at the cap 1/1,000 on Llama-3.1-8B's served layout**, its four linears at any `m` in one layout,
FP32 at 8.376, the honest cast at 8.72: `g = 3045273083/782596025000` (0.38912%), the down projection's, the max of
the four. -/
theorem pearlCGapSm120v1LoopCast8p72Rev1Cap1000_llama31_8b
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 1000)) :
    GGap CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevShapes (devSm120v1 Prices.sm120Loop) llama31_8bLinears)
      ((3045273083 / 782596025000 : ℚ) : ℝ) δPearlC :=
  pearlCGapDevRev1KShapes CM _ sem _ _ _ (31303841 / 31291472 / (1 - 1 / 1000)) _ (by norm_num)
    (wrefDevRev1K_sm120v1LoopCast8p72_llama31_8b_le _ (by norm_num)) (by norm_num) hTT

/-- **v1 under rev1 per audit tile at the cap 1/1,000 on Llama-3.1-8B's served layout**, FP32 at 8.376, the honest
cast at 8.72: `g = 3045273083/782596025000` (0.38912%), the down projection's, the max of the four. -/
theorem pearlCSampledSm120v1LoopCast8p72Rev1Cap1000_llama31_8b
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevShapes (devSm120v1 Prices.sm120Loop) llama31_8bLinears)
      ((3045273083 / 782596025000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1KShapes CM _ sem _ _ _ (31303841 / 31291472 / (1 - 1 / 1000)) _ (by norm_num)
    wrefDevRev1K_sm120v1LoopCast8p72_llama31_8b_nonneg
    (wrefDevRev1K_sm120v1LoopCast8p72_llama31_8b_le _ (by norm_num)) (by norm_num) hTT

theorem wrefDevRev1K_sm120v1LoopCast8p72_qwen3_8b_nonneg :
    ∀ nk ∈ qwen3_8bLinears, ∀ m : ℕ,
      0 ≤ wrefDevRev1K (devSm120v1 Prices.sm120Loop) (218 / 25 - 8953 / 1000) ⟨m, nk.1, nk.2⟩ := by
  intro nk hnk m
  simp only [qwen3_8bLinears, mem_insert, mem_singleton] at hnk
  rcases hnk with rfl | rfl | rfl | rfl
  · exact wrefDevRev1K_sm120v1LoopCast8p72_n6144k4096_nonneg m
  · exact wrefDevRev1K_sm120v1LoopCast8p72_n4096k4096_nonneg m
  · exact wrefDevRev1K_sm120v1LoopCast8p72_n24576k4096_nonneg m
  · exact wrefDevRev1K_sm120v1LoopCast8p72_n4096k12288_nonneg m

/-- Qwen3-8B's largest ratio `wrefDevRev1K/creditDevRev1` is its down projection's, `13433021/13427720`. -/
theorem wrefDevRev1K_sm120v1LoopCast8p72_qwen3_8b_le (ρ : ℚ) (hρ : ρ < 1) :
    ∀ nk ∈ qwen3_8bLinears, ∀ m : ℕ,
      wrefDevRev1K (devSm120v1 Prices.sm120Loop) (218 / 25 - 8953 / 1000) ⟨m, nk.1, nk.2⟩ ≤
        13433021 / 13427720 / (1 - ρ) * ((1 - ρ) * creditDevRev1 (devSm120v1 Prices.sm120Loop) ⟨m, nk.1, nk.2⟩) := by
  intro nk hnk m
  simp only [qwen3_8bLinears, mem_insert, mem_singleton] at hnk
  rcases hnk with rfl | rfl | rfl | rfl
  · rw [wrefDevRev1K_sm120v1LoopCast8p72_n6144k4096 m]
    exact ratio_le_max_div hρ (by norm_num) (creditDevRev1_sm120v1Loop_nonneg _ _ _ (by norm_num) (by norm_num))
  · rw [wrefDevRev1K_sm120v1LoopCast8p72_n4096k4096 m]
    exact ratio_le_max_div hρ (by norm_num) (creditDevRev1_sm120v1Loop_nonneg _ _ _ (by norm_num) (by norm_num))
  · rw [wrefDevRev1K_sm120v1LoopCast8p72_n24576k4096 m]
    exact ratio_le_max_div hρ (by norm_num) (creditDevRev1_sm120v1Loop_nonneg _ _ _ (by norm_num) (by norm_num))
  · rw [wrefDevRev1K_sm120v1LoopCast8p72_n4096k12288 m]
    exact ratio_le_max_div hρ (by norm_num) (creditDevRev1_sm120v1Loop_nonneg _ _ _ (by norm_num) (by norm_num))

/-- **v1 under rev1 per audit tile at the cap 1/1,000 on Qwen3-8B's served layout**, FP32 at 8.376, the honest cast at
8.72: `g = 74663501/19190030000` (0.38907%), the down projection's, the max of the four. -/
theorem pearlCSampledSm120v1LoopCast8p72Rev1Cap1000_qwen3_8b
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120Loop) sem (1 / 1000)) :
    GGapSampled CM (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevShapes (devSm120v1 Prices.sm120Loop) qwen3_8bLinears)
      ((74663501 / 19190030000 : ℚ) : ℝ) δPearlC :=
  pearlCSampledDevRev1KShapes CM _ sem _ _ _ (13433021 / 13427720 / (1 - 1 / 1000)) _ (by norm_num)
    wrefDevRev1K_sm120v1LoopCast8p72_qwen3_8b_nonneg
    (wrefDevRev1K_sm120v1LoopCast8p72_qwen3_8b_le _ (by norm_num)) (by norm_num) hTT

end Served

end Pouw.PearlC
