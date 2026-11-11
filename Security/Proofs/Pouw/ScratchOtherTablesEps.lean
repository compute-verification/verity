import Proofs.Pouw.Advantage
import Proofs.Pouw.PearlC
import Proofs.Pouw.PearlC.DeviceFp4Gap
import Proofs.Pouw.PearlC.TTOutFp4W1
import Proofs.Pouw.PearlC.W1Tabulate

/-!
Scratch (CMP-400 plan, never landed): PoUW's two ε statements at numbers with the device record as a parameter, and
today's two listed statements as their corollaries.

`pearlCDomainDevOn`, `KernelWithin` and the proof of the gap at any record are compute-accounting's scratch forms
(`GapDevRev1K`), restated here so that this file stands alone on `main`. The FP4 twins (`pearlCDomainFp4On`,
`Fp4Within`) and the ε forms are this file's.
-/

namespace Pouw.PearlC

open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom Pouw.PearlC.Fp4Dev

/-- `ε = ω/(1 − g₀) − 1` at `g₀ = 1/400`, as a real. -/
theorem eps_cast (ω : ℚ) : (((ω / (1 - 1 / 400) - 1 : ℚ)) : ℝ) = (ω : ℝ) / (1 - 1 / 400) - 1 := by
  first
    | (push_cast; ring)
    | (push_cast; done)
    | (norm_num; done)

/-! ## Pearl-C (FP8), the served form `pearlCProtocolDevRev1K` -/

/-- The domain restricted to the shapes `Ss` holds of. -/
def pearlCDomainDevOn (d : PearlCDevice) (Ss : Shape → Prop) (L : Layout) : Prop :=
  pearlCDomainDev d L ∧ ∀ u < L.N, Ss (L.shape u)

/-- What the statements ask of a device record: on the shapes `Ss`, the kernel's honest work is non-negative and
within `ω` of the credit left under the cap `ρ`. -/
def KernelWithin (d : PearlCDevice) (ρ c : ℚ) (Ss : Shape → Prop) (ω : ℚ) : Prop :=
  0 < ω ∧ ∀ s, Ss s → 0 ≤ wrefDevRev1K d c s ∧ wrefDevRev1K d c s ≤ ω * ((1 - ρ) * creditDevRev1 d s)

/-- **The advantage at any device record**: `ε = ω/(1 − 1/400) − 1`, a function of `ω` and not a number. -/
def EpsDevRev1K : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (dv : W1.W1Device) (salt : S → List ℕ)
    (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ c : ℚ) (Ss : Shape → Prop) (ω : ℚ),
    KernelWithin d ρ c Ss ω → TTOutPearlCDevRev1W1 dv salt d sem ρ →
    Gε (W1.w1Priced dv salt) (pearlCProtocolDevRev1K d sem ρ c) (pearlCDomainDevOn d Ss)
      ((ω / (1 - 1 / 400) - 1 : ℚ) : ℝ) δPearlC

/-- The gap at any device record, at the real `1 − (1 − 1/400)/ω` (the proof of compute-accounting's `gapDevRev1K`). -/
theorem gapDevRev1K_real {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (dv : W1.W1Device)
    (salt : S → List ℕ) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ c : ℚ) (Ss : Shape → Prop) (ω : ℚ)
    (hK : KernelWithin d ρ c Ss ω) (hTT : TTOutPearlCDevRev1W1 dv salt d sem ρ) :
    GGap (W1.w1Priced dv salt) (pearlCProtocolDevRev1K d sem ρ c) (pearlCDomainDevOn d Ss)
      (1 - (1 - 1 / 400) / (ω : ℝ)) δPearlC := by
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
  exact gapFromTTOut (W1.w1Priced dv salt) (pearlCProtocolDevRev1K d sem ρ c) (pearlCDomainDevOn d Ss)
    (1 / 400) (ω : ℝ) δPearlC (by exact_mod_cast hω0) hW
    (ttOut_mono_domain (fun _ h => h.1) (ttOut_wref _ (TTOutW1.ttOutPearlCDevRev1_of_w1 hTT)))

theorem epsDevRev1K : EpsDevRev1K := by
  intro Q R S _ _ _ _ dv salt d sem ρ c Ss ω hK hTT
  have hω : (0 : ℝ) < (ω : ℝ) := by exact_mod_cast hK.1
  have hg₀ : (1 / 400 : ℝ) < 1 := by norm_num
  have hG : GGap (W1.w1Priced dv salt) (pearlCProtocolDevRev1K d sem ρ c) (pearlCDomainDevOn d Ss)
      (Pouw.reductionGap (1 / 400) (ω : ℝ)) δPearlC := gapDevRev1K_real dv salt d sem ρ c Ss ω hK hTT
  have hE := (gGap_iff_gε _ _ _ (Pouw.Proofs.reductionGap_lt_one hg₀ hω) _).1 hG
  rw [Pouw.Proofs.reductionGap_div hg₀ hω] at hE
  rw [eps_cast]
  exact hE

/-- The one fact the 8192³ statement takes from the served record: `W_ref = 8921455/8919688 · credit` there. An
honest witness's conclusion for `KernelWithin`. -/
theorem sm120v1Loop_within_8192 :
    KernelWithin (devSm120v1 Prices.sm120Loop) (1 / 1000) (218 / 25 - 8953 / 1000) (fun s => s = sh8192)
      (8921455 / 8919688 / (1 - 1 / 1000)) := by
  refine ⟨by norm_num, ?_⟩
  rintro s rfl
  refine ⟨wrefDevRev1K_sm120v1LoopCast8p72_8192_nonneg, le_of_eq ?_⟩
  rw [wrefDevRev1K_sm120v1LoopCast8p72_8192]
  ring

theorem lit_eps_8192 :
    (1648180439 / 444424569561 : ℚ) = 8921455 / 8919688 / (1 - 1 / 1000) / (1 - 1 / 400) - 1 := by norm_num

/-- The listed statement, as a corollary of the general one and one fact about the record. -/
theorem eps_8192_of_general : Pouw.Guarantees.PearlC.EpsSm120v1LoopCast8p72Rev1Cap1000_8192 := by
  intro Q R S _ _ _ _ sha fetch salt sem hTT
  rw [lit_eps_8192]
  exact epsDevRev1K (W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop) sem (1 / 1000)
    (218 / 25 - 8953 / 1000) (fun s => s = sh8192) _ sm120v1Loop_within_8192 hTT

/-! ## Pearl-C4 (FP4) -/

/-- The FP4 domain restricted to the shapes `Ss` holds of. -/
def pearlCDomainFp4On {Q R S : Type} (sem : Fp4Sem Q R S) (Ss : Shape → Prop) (L : Layout) : Prop :=
  pearlCDomainFp4 sem L ∧ ∀ u < L.N, Ss (L.shape u)

/-- What the FP4 statements ask of an FP4 record: on the shapes `Ss`, the honest work is non-negative and within `ω`
of the credit left under the cap `ρ`. -/
def Fp4Within (d : Fp4Device) (ρ : ℚ) (Ss : Shape → Prop) (ω : ℚ) : Prop :=
  0 < ω ∧ ∀ s, Ss s → 0 ≤ wrefFp4 d s ∧ wrefFp4 d s ≤ ω * ((1 - ρ) * creditFp4 d s)

/-- **Pearl-C4's advantage at any FP4 record**: `ε = ω/(1 − 1/400) − 1`. -/
def EpsFp4 : Prop :=
  ∀ {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (dv : W1.W1Device) (salt : S → List ℕ)
    (d : Fp4Device) (sem : Fp4Sem Q R S) (ρ : ℚ) (Ss : Shape → Prop) (ω : ℚ),
    Fp4Within d ρ Ss ω → TTOutFp4W1 dv salt d sem ρ →
    Gε (W1.w1Priced dv salt) (pearlCProtocolFp4 d sem ρ) (pearlCDomainFp4On sem Ss)
      ((ω / (1 - 1 / 400) - 1 : ℚ) : ℝ) δPearlC

/-- The gap at any FP4 record (`main`'s `pearlCGapFp4At` with the record's fact as an inequality on a set of shapes). -/
theorem gapFp4_real {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (dv : W1.W1Device)
    (salt : S → List ℕ) (d : Fp4Device) (sem : Fp4Sem Q R S) (ρ : ℚ) (Ss : Shape → Prop) (ω : ℚ)
    (hK : Fp4Within d ρ Ss ω) (hTT : TTOutFp4W1 dv salt d sem ρ) :
    GGap (W1.w1Priced dv salt) (pearlCProtocolFp4 d sem ρ) (pearlCDomainFp4On sem Ss)
      (1 - (1 - 1 / 400) / (ω : ℝ)) δPearlC := by
  obtain ⟨hω0, hω⟩ := hK
  have hW : ∀ U : Workload, U.InDomain (pearlCProtocolFp4 d sem ρ) (pearlCDomainFp4On sem Ss) →
      ∀ (H : Q → R) (s' : S), ∀ u < U.layout.N, ∀ act : Codes,
        (pearlCProtocolFp4 d sem ρ).actOK (U.layout.shape u) act →
        (pearlCProtocolFp4 d sem ρ).capOK H s' U u act →
          (pearlCProtocolFp4 d sem ρ).Wref U.layout u ≤
            (ω : ℝ) * (pearlCProtocolFp4 d sem ρ).credit H s' U u act := by
    intro U hU H s' u hu act _ hcap
    have hc : sem.unitDebit d H s' U u act ≤ ρ * creditFp4 d (U.layout.shape u) := hcap
    have h1 : wrefFp4 d (U.layout.shape u) ≤ ω * ((1 - ρ) * creditFp4 d (U.layout.shape u)) :=
      (hω _ (hU.2.2 u hu)).2
    show (wrefFp4 d (U.layout.shape u) : ℝ) ≤
      (ω : ℝ) * ((creditFp4 d (U.layout.shape u) - sem.unitDebit d H s' U u act : ℚ) : ℝ)
    have : ω * ((1 - ρ) * creditFp4 d (U.layout.shape u)) ≤
        ω * (creditFp4 d (U.layout.shape u) - sem.unitDebit d H s' U u act) :=
      mul_le_mul_of_nonneg_left (by linarith) hω0.le
    exact_mod_cast h1.trans this
  exact gapFromTTOut (W1.w1Priced dv salt) (pearlCProtocolFp4 d sem ρ) (pearlCDomainFp4On sem Ss) (1 / 400)
    (ω : ℝ) δPearlC (by exact_mod_cast hω0) hW
    (ttOut_mono_domain (fun _ h => h.1) (TTOutFp4W1.ttOutFp4_of_w1 hTT))

theorem epsFp4 : EpsFp4 := by
  intro Q R S _ _ _ _ dv salt d sem ρ Ss ω hK hTT
  have hω : (0 : ℝ) < (ω : ℝ) := by exact_mod_cast hK.1
  have hg₀ : (1 / 400 : ℝ) < 1 := by norm_num
  have hG : GGap (W1.w1Priced dv salt) (pearlCProtocolFp4 d sem ρ) (pearlCDomainFp4On sem Ss)
      (Pouw.reductionGap (1 / 400) (ω : ℝ)) δPearlC := gapFp4_real dv salt d sem ρ Ss ω hK hTT
  have hE := (gGap_iff_gε _ _ _ (Pouw.Proofs.reductionGap_lt_one hg₀ hω) _).1 hG
  rw [Pouw.Proofs.reductionGap_div hg₀ hω] at hE
  rw [eps_cast]
  exact hE

/-- The one fact the 8192³ statement takes from the FP4 record. An honest witness's conclusion for `Fp4Within`. -/
theorem fp4Sm120_within_8192 :
    Fp4Within devFp4Sm120 (1 / 400) (fun s => s = sh8192) (3441658000 / 3425534301) := by
  refine ⟨by norm_num, ?_⟩
  rintro s rfl
  exact ⟨by norm_num [wrefFp4, creditFp4, devFp4Sm120, Fp4Prices.sm120, Params.pi, sh8192],
    le_of_eq (by norm_num [wrefFp4, creditFp4, devFp4Sm120, Fp4Prices.sm120, Params.pi, sh8192])⟩

theorem lit_eps_fp4_8192 :
    (9875013901 / 1366788186099 : ℚ) = 3441658000 / 3425534301 / (1 - 1 / 400) - 1 := by norm_num

/-- The listed statement, as a corollary of the general one and one fact about the record. -/
theorem epsFp4_8192_of_general : Pouw.Guarantees.PearlC.EpsFp4Sm120_8192 := by
  intro Q R S _ _ _ _ sha fetch salt sem hTT
  rw [lit_eps_fp4_8192]
  exact epsFp4 (W1.W1Device.fp4Sm120 sha fetch) salt devFp4Sm120 sem (1 / 400) (fun s => s = sh8192) _
    fp4Sm120_within_8192 hTT

/-! ## The conclusion is not a tautology -/

/-- `G_ε` is one named `Prop`, and it fails at a protocol whose rules and cap always pass (`W1Tabulate.tabP`, bound
here by `∃` since it is declared in the proofs tree): the empty program wins there at no cost. -/
theorem gε_nontrivial (dv : W1.W1Device) {ε : ℝ} (hε : 0 ≤ ε) :
    ∃ P : Protocol Unit Unit Unit,
      ¬ Gε (W1.w1Priced dv (fun _ : Unit => []) : CostModel Unit Unit Unit) P (fun _ => True) ε δPearlC :=
  ⟨W1Tabulate.tabP, fun h =>
    W1Tabulate.schema_not_gGap dv (ε / (1 + ε)) (δ := δPearlC) (by norm_num [δPearlC])
      ((gε_iff_gGap _ _ _ hε _).1 h)⟩

end Pouw.PearlC

#print axioms Pouw.PearlC.epsDevRev1K
#print axioms Pouw.PearlC.eps_8192_of_general
#print axioms Pouw.PearlC.sm120v1Loop_within_8192
#print axioms Pouw.PearlC.epsFp4
#print axioms Pouw.PearlC.epsFp4_8192_of_general
#print axioms Pouw.PearlC.fp4Sm120_within_8192
#print axioms Pouw.PearlC.gε_nontrivial
