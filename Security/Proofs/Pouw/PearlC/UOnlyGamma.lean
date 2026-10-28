import Proofs.Pouw.PearlC.DeviceCapGamma
import Definitions.Pouw.Assumptions.PearlC.TTOutUOnly
import Definitions.Pouw.Guarantees.PearlC
import Proofs.Pouw.PearlC.TTOutW1Variants

/-!
# TT_OUT over U alone: it asserts more than the (C̃, U) row, and γ from it

Staged; nothing pinned.
* `correctSet_subset_correctSetU`: a unit correct under (C̃, U) binding is correct under U alone.
* `ttOutPearlCDevRev1_of_uOnly`: TT_OUT over U gives the (C̃, U) row at the same record, for non-negative prices and
  `ρ ≤ 1`. So U-only is the stronger conjecture.
* `gammaFromTTOutU` and `pearlCGammaDevUOnlyAt`: TT_OUT over U gives `G_γ^U` at `γ = 1 − (399/400)/ω`, the same ω as
  rev1, so γ is rev1's: 0.51056% (v1) and 0.36162% (v2 at the cap 1/1,000) at 8192³ at sm_120's prices
  (`pearlCGammaUOnlySm120Rev1_8192`, `pearlCGammaUOnlyUnpromotedCap1000_8192`).
* `gammaU_implies_gamma`: `G_γ^U` gives `G_γ`, since a unit the (C̃, U) verifier accepts is one the U-only verifier
  accepts.
-/

namespace Pouw.PearlC

open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

section
variable {Q R S : Type}

open Classical in
theorem correctSet_subset_correctSetU (P : Protocol Q R S) (U : Workload) (H : Q → R) (s : S) (τ : CommitLog) :
    correctSet P U H s τ ⊆ correctSetU P U H s τ := by
  intro u hu
  unfold correctSet at hu
  unfold correctSetU
  rw [mem_filter] at hu ⊢
  obtain ⟨hu, c, hc, hact, hcap, hB, hw⟩ := hu
  exact ⟨hu, c, hc, hact, hcap, hB, fun i hi j hj => (hw i hi j hj).2⟩

/-- `G_γ^U` gives `G_γ` when `W_ref` is non-negative. -/
theorem gammaU_implies_gamma [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (P : Protocol Q R S) (D : UnitLayout → Prop) (γ : ℝ) (η : ℕ → ℕ → ℝ) (hW : ∀ L u, 0 ≤ P.Wref L u)
    (h : GγU CM P D γ η) : Gγ CM P D γ η := by
  intro U hU T hT q hq A hA
  refine le_trans (Pouw.pr_mono ?_) (h U hU T hT q hq A hA)
  intro x hx
  unfold Wins at hx
  unfold WinsU
  exact lt_of_lt_of_le hx (sum_le_sum_of_subset_of_nonneg (correctSet_subset_correctSetU P U _ _ _)
    fun u _ _ => hW _ u)

theorem gammaFromTTOutU [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (P : Protocol Q R S) (D : UnitLayout → Prop) (γ₀ Ω : ℝ) (ε : ℕ → ℕ → ℝ) (hΩ : 0 < Ω)
    (hW : ∀ U : Workload, U.InDomain P D → ∀ (H : Q → R) (s : S), ∀ u < U.layout.N, ∀ act : Codes,
      P.actOK (U.layout.shape u) act → P.capOK H s U u act → P.Wref U.layout u ≤ Ω * P.credit H s U u act)
    (hTT : TTOutU CM P D γ₀ ε) : GγU CM P D (1 - (1 - γ₀) / Ω) ε := by
  intro U hU T hT q hq A hA
  classical
  refine le_trans (Pouw.pr_mono ?_) (hTT.2 U hU T hT q hq A hA)
  intro x hwin
  unfold WinsU at hwin
  have hsum : ∑ u ∈ correctSetU P U x.1 x.2 (A.transcript U x.1 x.2), P.Wref U.layout u ≤
      Ω * ∑ u ∈ correctSetU P U x.1 x.2 (A.transcript U x.1 x.2),
        P.credit x.1 x.2 U u (actOf (A.transcript U x.1 x.2) u) := by
    rw [mul_sum]
    refine sum_le_sum fun u hu => ?_
    unfold correctSetU at hu
    rw [mem_filter] at hu
    obtain ⟨hu, c, hc, hcor⟩ := hu
    have hact : actOf (A.transcript U x.1 x.2) u = c.act := by simp [actOf, hc]
    rw [hact]
    exact hW U hU x.1 x.2 u (mem_range.mp hu) c.act hcor.1 hcor.2.1
  have hden : 1 - (1 - (1 - γ₀) / Ω) = (1 - γ₀) / Ω := by ring
  rw [hden, div_div_eq_mul_div] at hwin
  have hscale : T * Ω / (1 - γ₀) = Ω * (T / (1 - γ₀)) := by ring
  rw [hscale] at hwin
  exact lt_of_mul_lt_mul_left (lt_of_lt_of_le hwin hsum) hΩ.le

/-- rev1's credit is non-negative on an in-domain shape: at least one promotion group when the record promotes. -/
theorem creditDevRev1_nonneg (d : PearlCDevice) (hadd : 0 ≤ d.prices.add) (hbf : 0 ≤ d.prices.bf16) (s : Shape)
    (hk : 0 < s.k) (hg : d.G = 0 ∨ 32 * d.G ∣ s.k) : 0 ≤ creditDevRev1 d s := by
  unfold creditDevRev1 creditDev firstAddDev
  have h1 : (0 : ℚ) ≤ (s.m : ℚ) * s.k * s.n := by positivity
  have h2 : (0 : ℚ) ≤ (s.m : ℚ) * s.k * Params.pi.r := by positivity
  have h3 : (0 : ℚ) ≤ 2 * d.prices.bf16 * Params.pi.r * s.m * s.n := by positivity
  have h4 : (0 : ℚ) ≤ (d.prices.costs.fs : ℚ) * s.m * s.k := by positivity
  by_cases hG : d.G = 0
  · simp only [hG, ite_true]
    linarith
  · have hle : (32 * d.G : ℚ) ≤ s.k := by exact_mod_cast Nat.le_of_dvd hk (hg.resolve_left hG)
    have hpos : (0 : ℚ) < 32 * d.G := by
      have : 0 < d.G := Nat.pos_of_ne_zero hG
      positivity
    simp only [hG, ite_false]
    have ha : (0 : ℚ) ≤ d.prices.add * s.m * s.n := by positivity
    have key : d.prices.add * s.m * s.n ≤ (s.m : ℚ) * s.k * s.n * d.prices.add / (32 * d.G) := by
      rw [le_div_iff₀ hpos]
      nlinarith
    linarith

open Classical in
/-- **TT_OUT over U asserts more than the (C̃, U) row** at the same record, for non-negative prices and `ρ ≤ 1`. -/
theorem ttOutPearlCDevRev1_of_uOnly [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) (hadd : 0 ≤ d.prices.add) (hbf : 0 ≤ d.prices.bf16)
    (hρ1 : ρ ≤ 1) (h : TTOutPearlCDevUOnly CM d sem ρ) : TTOutPearlCDevRev1 CM d sem ρ := by
  refine ⟨h.1, fun U hU T hT q hq A hA => le_trans (Pouw.pr_mono ?_) (h.2 U hU T hT q hq A hA)⟩
  intro x hx
  refine lt_of_lt_of_le hx (sum_le_sum_of_subset_of_nonneg (correctSet_subset_correctSetU _ U _ _ _) ?_)
  intro u hu _
  unfold correctSetU at hu
  rw [mem_filter] at hu
  obtain ⟨hur, c, hc, _, hcap, _⟩ := hu
  obtain ⟨_, _, hk, _, hg, _⟩ := hU.2 u (mem_range.mp hur)
  have hcr := creditDevRev1_nonneg d hadd hbf (U.layout.shape u) hk hg
  have hact : actOf (A.transcript U x.1 x.2) u = c.act := by simp [actOf, hc]
  rw [hact]
  have h1 : sem.unitDebitRev1 d x.1 x.2 U u c.act ≤ ρ * creditDevRev1 d (U.layout.shape u) := hcap
  have h0 := unitDebitRev1_nonneg sem d hadd x.1 x.2 U u c.act
  show (0 : ℝ) ≤ ((creditDevRev1 d (U.layout.shape u) - sem.unitDebitRev1 d x.1 x.2 U u c.act : ℚ) : ℝ)
  have : (0 : ℚ) ≤ creditDevRev1 d (U.layout.shape u) - sem.unitDebitRev1 d x.1 x.2 U u c.act := by
    nlinarith
  exact_mod_cast this

end

/-- TT_OUT over U at a device record gives `G_γ^U` on the one-shape domain at `γ = 1 − (399/400)/ω`, where
`wrefDevRev1 = ω·(1 − ρ)·creditDevRev1`. -/
theorem pearlCGammaDevUOnlyAt {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) (s : Shape) (ω γ : ℚ) (hω0 : 0 < ω)
    (hω : wrefDevRev1 d s = ω * ((1 - ρ) * creditDevRev1 d s)) (hγ : γ = 1 - (1 - 1 / 400) / ω)
    (hTT : TTOutPearlCDevUOnly CM d sem ρ) :
    GγU CM (pearlCProtocolDevRev1 d sem ρ) (pearlCDomainDevAt d s) (γ : ℝ) εPearlC := by
  have hW : ∀ U : Workload, U.InDomain (pearlCProtocolDevRev1 d sem ρ) (pearlCDomainDevAt d s) →
      ∀ (H : Q → R) (s' : S), ∀ u < U.layout.N, ∀ act : Codes,
        (pearlCProtocolDevRev1 d sem ρ).actOK (U.layout.shape u) act →
        (pearlCProtocolDevRev1 d sem ρ).capOK H s' U u act →
          (pearlCProtocolDevRev1 d sem ρ).Wref U.layout u ≤
            (ω : ℝ) * (pearlCProtocolDevRev1 d sem ρ).credit H s' U u act := by
    intro U hU H s' u hu act _ hcap
    have hsh : U.layout.shape u = s := hU.2.2 u hu
    have hc : sem.unitDebitRev1 d H s' U u act ≤ ρ * creditDevRev1 d (U.layout.shape u) := hcap
    show (wrefDevRev1 d (U.layout.shape u) : ℝ) ≤
      (ω : ℝ) * ((creditDevRev1 d (U.layout.shape u) - sem.unitDebitRev1 d H s' U u act : ℚ) : ℝ)
    rw [hsh] at hc ⊢
    rw [hω]
    have : ω * ((1 - ρ) * creditDevRev1 d s) ≤ ω * (creditDevRev1 d s - sem.unitDebitRev1 d H s' U u act) :=
      mul_le_mul_of_nonneg_left (by linarith) hω0.le
    exact_mod_cast this
  have hmono : TTOutU CM (pearlCProtocolDevRev1 d sem ρ) (pearlCDomainDevAt d s) (1 / 400) εPearlC :=
    ⟨hTT.1, fun U hU => hTT.2 U ⟨hU.1, hU.2.1⟩⟩
  have key := gammaFromTTOutU CM (pearlCProtocolDevRev1 d sem ρ) (pearlCDomainDevAt d s) (1 / 400) (ω : ℝ) εPearlC
    (by exact_mod_cast hω0) hW hmono
  have e : (γ : ℝ) = 1 - (1 - 1 / 400) / (ω : ℝ) := by rw [hγ]; push_cast; ring
  rw [e]
  exact key

/-- **v1 under rev1 with U-only binding, 8192³**, at sm_120's prices, cap 1/400: `γ = 16223/3177500`. -/
theorem pearlCGammaUOnlySm120Rev1_8192 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (hG : d.G = 4) (hp : d.prices = Prices.sm120)
    (hTT : TTOutPearlCDevUOnly CM d sem (1 / 400)) :
    GγU CM (pearlCProtocolDevRev1 d sem (1 / 400)) (pearlCDomainDevAt d sh8192) ((16223 / 3177500 : ℚ) : ℝ)
      εPearlC :=
  pearlCGammaDevUOnlyAt CM d sem _ _ (31775 / 31692) _ (by norm_num)
    (by rw [wrefDevRev1, creditDevRev1, firstAddDev, creditDev, hG, hp]; norm_num [Prices.sm120, Params.pi, sh8192])
    (by norm_num) hTT

/-- **v2 at the cap 1/1,000 with U-only binding, 8192³**, at sm_120's prices: `γ = 216793/59950000`. -/
theorem pearlCGammaUOnlyUnpromotedCap1000_8192 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (hG : d.G = 0) (hp : d.prices = Prices.sm120)
    (hTT : TTOutPearlCDevUOnly CM d sem (1 / 1000)) :
    GγU CM (pearlCProtocolDevRev1 d sem (1 / 1000)) (pearlCDomainDevAt d sh8192) ((216793 / 59950000 : ℚ) : ℝ)
      εPearlC :=
  pearlCGammaDevUOnlyAt CM d sem _ _ (1049125 / 1047951) _ (by norm_num)
    (by rw [wrefDevRev1, creditDevRev1, firstAddDev, creditDev, hG, hp]; norm_num [Prices.sm120, Params.pi, sh8192])
    (by norm_num) hTT

end Pouw.PearlC

theorem Pouw.SecurityProofs.PearlC.GammaUOnlySm120Rev1_8192 : Pouw.Guarantees.PearlC.GammaUOnlySm120Rev1_8192 :=
  fun sha fetch salt d sem hG hp hTT =>
    @Pouw.PearlC.pearlCGammaUOnlySm120Rev1_8192 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) d sem hG hp
      (Pouw.PearlC.TTOutW1.ttOutPearlCDevUOnly_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.GammaUOnlyUnpromotedCap1000_8192 : Pouw.Guarantees.PearlC.GammaUOnlyUnpromotedCap1000_8192 :=
  fun sha fetch salt d sem hG hp hTT =>
    @Pouw.PearlC.pearlCGammaUOnlyUnpromotedCap1000_8192 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120 sha fetch) salt) d sem hG hp
      (Pouw.PearlC.TTOutW1.ttOutPearlCDevUOnly_of_w1 hTT)
