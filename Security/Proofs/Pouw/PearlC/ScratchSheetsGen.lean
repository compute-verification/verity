import Proofs.Pouw.PearlC
import Proofs.Pouw.Deadline

/-!
Scratch (CMP-392 plan, never landed): PoUW's gap statements with the device record as a parameter, and today's
statements at numbers as their corollaries.
-/

namespace Pouw.PearlC

open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

/-- The domain restricted to the shapes `Ss` holds of (one shape, one linear, a set of linears). -/
def pearlCDomainDevOn (d : PearlCDevice) (Ss : Shape → Prop) (L : Layout) : Prop :=
  pearlCDomainDev d L ∧ ∀ u < L.N, Ss (L.shape u)

/-- What the gap statements ask of a device record: on the shapes `Ss`, the kernel's honest work is non-negative and
within `ω` of the credit left under the cap `ρ`. -/
def KernelWithin (d : PearlCDevice) (ρ c : ℚ) (Ss : Shape → Prop) (ω : ℚ) : Prop :=
  0 < ω ∧ ∀ s, Ss s → 0 ≤ wrefDevRev1K d c s ∧ wrefDevRev1K d c s ≤ ω * ((1 - ρ) * creditDevRev1 d s)

/-- The gap per unit, at any device record. -/
def GapDevRev1K : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (dv : W1.W1Device) (salt : S → List ℕ)
    (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ c : ℚ) (Ss : Shape → Prop) (ω : ℚ),
    KernelWithin d ρ c Ss ω → TTOutPearlCDevRev1W1 dv salt d sem ρ →
    GGap (W1.w1Priced dv salt) (pearlCProtocolDevRev1K d sem ρ c) (pearlCDomainDevOn d Ss)
      ((1 - (1 - 1 / 400) / ω : ℚ) : ℝ) δPearlC

/-- The gap per audit tile, at any device record. -/
def SampledDevRev1K : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (dv : W1.W1Device) (salt : S → List ℕ)
    (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ c : ℚ) (Ss : Shape → Prop) (ω : ℚ),
    KernelWithin d ρ c Ss ω → TTOutTilePearlCDevRev1W1 dv salt d sem ρ →
    GGapSampled (W1.w1Priced dv salt) (pearlCProtocolDevRev1K d sem ρ c) (pearlCTilesDevRev1K d sem ρ c)
      (pearlCDomainDevOn d Ss) ((1 - (1 - 1 / 400) / ω : ℚ) : ℝ) δPearlC

theorem gapDevRev1K : GapDevRev1K := by
  intro Q R S _ _ _ _ dv salt d sem ρ c Ss ω hK hTT
  obtain ⟨hω0, hω⟩ := hK
  have hW : ∀ U : Workload, U.InDomain (pearlCProtocolDevRev1K d sem ρ c) (pearlCDomainDevOn d Ss) →
      ∀ (H : Q → R) (s' : S), ∀ u < U.layout.N, ∀ act : Codes,
        (pearlCProtocolDevRev1K d sem ρ c).actOK (U.layout.shape u) act →
        (pearlCProtocolDevRev1K d sem ρ c).capOK H s' U u act →
          (pearlCProtocolDevRev1K d sem ρ c).Wref U.layout u ≤
            (ω : ℝ) * (pearlCProtocolDevRev1K d sem ρ c).credit H s' U u act := by
    intro U hU H s' u hu act _ hcap
    have hc : sem.unitDebitRev1 d H s' U u act ≤ ρ * creditDevRev1 d (U.layout.shape u) := hcap
    have h1 : wrefDevRev1K d c (U.layout.shape u) ≤ ω * ((1 - ρ) * creditDevRev1 d (U.layout.shape u)) :=
      (hω _ (hU.2.2 u hu)).2
    show (wrefDevRev1K d c (U.layout.shape u) : ℝ) ≤
      (ω : ℝ) * ((creditDevRev1 d (U.layout.shape u) - sem.unitDebitRev1 d H s' U u act : ℚ) : ℝ)
    have : ω * ((1 - ρ) * creditDevRev1 d (U.layout.shape u)) ≤
        ω * (creditDevRev1 d (U.layout.shape u) - sem.unitDebitRev1 d H s' U u act) :=
      mul_le_mul_of_nonneg_left (by linarith) hω0.le
    exact_mod_cast h1.trans this
  have key := gapFromTTOut (W1.w1Priced dv salt) (pearlCProtocolDevRev1K d sem ρ c) (pearlCDomainDevOn d Ss)
    (1 / 400) (ω : ℝ) δPearlC (by exact_mod_cast hω0) hW
    (ttOut_mono_domain (fun _ h => h.1) (ttOut_wref _ (TTOutW1.ttOutPearlCDevRev1_of_w1 hTT)))
  have e : (((1 - (1 - 1 / 400) / ω : ℚ)) : ℝ) = 1 - (1 - 1 / 400) / (ω : ℝ) := by push_cast; ring
  rw [e]
  exact key

theorem sampledDevRev1K : SampledDevRev1K := by
  intro Q R S _ _ _ _ dv salt d sem ρ c Ss ω hK hTT
  obtain ⟨hω0, hω⟩ := hK
  have hTT' : TTOutTilePearlCDevRev1 (W1.w1Priced dv salt) d sem ρ := TTOutW1.ttOutTilePearlCDevRev1_of_w1 hTT
  have hW : ∀ U : Workload, U.InDomain (pearlCProtocolDevRev1K d sem ρ c) (pearlCDomainDevOn d Ss) →
      ∀ (H : Q → R) (s' : S), ∀ g < ((pearlCTilesDevRev1K d sem ρ c).tiling U.layout).NT, ∀ act : Codes,
        (pearlCTilesDevRev1K d sem ρ c).capOK H s' U g act →
        (pearlCTilesDevRev1K d sem ρ c).Wcred U.layout g act ≤
          (ω : ℝ) * (pearlCTilesDevRev1K d sem ρ c).credit H s' U g act := by
    intro U hU H s' g hg act _
    have hu := auditTiling_unit_lt 64 64 U.layout hg
    have h1 : wrefDevRev1K d c (U.layout.shape ((auditTiling 64 64 U.layout).unit g)) ≤
        ω * ((1 - ρ) * creditDevRev1 d (U.layout.shape ((auditTiling 64 64 U.layout).unit g))) :=
      (hω _ (hU.2.2 _ hu)).2
    simp only [pearlCTilesDevRev1K, pearlCTilesDevRev1]
    exact mul_share_le_cast h1 (tileShare_nonneg _ _ _)
  have hWc : ∀ U : Workload, U.InDomain (pearlCProtocolDevRev1K d sem ρ c) (pearlCDomainDevOn d Ss) →
      ∀ g < ((pearlCTilesDevRev1K d sem ρ c).tiling U.layout).NT, ∀ act : Codes,
        (pearlCTilesDevRev1K d sem ρ c).Wcred U.layout g act ≤ (pearlCTilesDevRev1K d sem ρ c).Wref U.layout g := by
    intro U hU g hg act
    have hu := auditTiling_unit_lt 64 64 U.layout hg
    have h0 : 0 ≤ wrefDevRev1K d c (U.layout.shape ((auditTiling 64 64 U.layout).unit g)) :=
      (hω _ (hU.2.2 _ hu)).1
    simp only [pearlCTilesDevRev1K, pearlCTilesDevRev1]
    exact_mod_cast mul_le_mul_of_nonneg_left (tileShare_mono _ (passRows_subset sem _ _ _) _) h0
  have key := gapSampled (W1.w1Priced dv salt) (pearlCProtocolDevRev1K d sem ρ c) (pearlCTilesDevRev1K d sem ρ c)
    (pearlCDomainDevOn d Ss) (1 / 400) (ω : ℝ) δPearlC (by norm_num) (by exact_mod_cast hω0) hW hWc
    (ttOutTile_mono_domain (fun _ h => h.1) (ttOutTile_wref _ _ _ hTT'))
  have e : (((1 - (1 - 1 / 400) / ω : ℚ)) : ℝ) = 1 - (1 - 1 / 400) / (ω : ℝ) := by push_cast; ring
  rw [e]
  exact key

/-! ## Today's statements at numbers, from the general ones and one fact about the record -/

/-- The one fact the 16384³ statements take from the sm_120 record: `W_ref = 5883173/5882584 · credit` there. -/
theorem sm120v1Loop_within_16384 :
    KernelWithin (devSm120v1 Prices.sm120Loop) (1 / 1000) (218 / 25 - 8953 / 1000) (fun s => s = sh16384)
      (5883173 / 5882584 / (1 - 1 / 1000)) := by
  refine ⟨by norm_num, ?_⟩
  rintro s rfl
  refine ⟨wrefDevRev1K_sm120v1LoopCast8p72_16384_nonneg, le_of_eq ?_⟩
  rw [wrefDevRev1K_sm120v1LoopCast8p72_16384]
  ring

theorem lit_16384 :
    (1058166877 / 294158650000 : ℚ) = 1 - (1 - 1 / 400) / (5883173 / 5882584 / (1 - 1 / 1000)) := by norm_num

/-- The listed per-unit statement, as a corollary. -/
theorem gap_16384_of_general : Pouw.Guarantees.PearlC.GapSm120v1LoopCast8p72Rev1Cap1000_16384 := by
  intro Q R S _ _ _ _ sha fetch salt sem hTT
  rw [lit_16384]
  exact gapDevRev1K (W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 1000)
    (218 / 25 - 8953 / 1000) (fun s => s = sh16384) _ sm120v1Loop_within_16384 hTT

theorem sampled_16384_of_general {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (sha fetch : ℚ)
    (salt : S → List ℕ) (sem : PearlCSem Q R S)
    (hTT : TTOutTilePearlCDevRev1W1 (W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem
      (1 / 1000)) :
    GGapSampled (W1.w1Priced (W1.W1Device.sm120Loop sha fetch) salt)
      (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh16384)
      ((1058166877 / 294158650000 : ℚ) : ℝ) δPearlC := by
  rw [lit_16384]
  exact sampledDevRev1K (W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 1000)
    (218 / 25 - 8953 / 1000) (fun s => s = sh16384) _ sm120v1Loop_within_16384 hTT

end Pouw.PearlC

/-- The listed by-deadline statement, as a corollary of the general sampled statement and the general deadline one. -/
theorem byDeadline_16384_of_general : Pouw.Guarantees.PearlCSm120Rev1Cap1000ByDeadline_16384 :=
  fun Q R S _ _ _ _ sha fetch salt sem hTT =>
    Pouw.SecurityProofs.PearlCSampledByDeadline Q R S
      (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt) _ _ _ _ _
      (Pouw.PearlC.sampled_16384_of_general sha fetch salt sem hTT)

#print axioms Pouw.PearlC.gapDevRev1K
#print axioms Pouw.PearlC.sampledDevRev1K
#print axioms Pouw.PearlC.gap_16384_of_general
#print axioms byDeadline_16384_of_general
