import Proofs.Pouw.PearlC.DeviceGamma
import Definitions.Pouw.Assumptions.PearlC.TTOutRev1
import Definitions.Pouw.Guarantees.PearlC
import Proofs.Pouw.PearlC.TTOutW1

/-!
# rev1: it asserts less than the granted TT_OUT, and γ from it

Staged; nothing pinned.
* **rev1 is weaker:** at any device record whose add and BF16 prices are non-negative, and any cap `0 ≤ ρ ≤ 1`,
  `TTOutPearlCDev` gives `TTOutPearlCDevRev1` (`ttOutPearlCDevRev1_of_granted`), and likewise per tile
  (`ttOutTilePearlCDevRev1_of_granted`). A unit or tile good under rev1 is good under the granted rows (rev1's debit is
  larger and its cap stricter), and its rev1 credit is at most the granted one. These are implications only: their
  premise, the granted row, is falsified by the first-promotion skip (rating D) at the H100 and sm_120 records
  (`TTOutRev1.lean`), so they carry no evidence for rev1, whose rating B rests on its own review alone.
* **The H100 record in the granted form:** `creditDevRev1_h100`, `wrefDevRev1_h100`.
* **γ at any record,** on the one-shape domain where `wrefDevRev1 = ω·(1 − ρ)·creditDevRev1`: `pearlCGammaDevRev1At`
  and `pearlCSampledDevRev1At`, at `γ = 1 − (399/400)/ω`.
* **The H100 headline under rev1**, cap `1/400`: `140237/26090000` (0.53751%) at 8192³ and `89359/17230000` (0.51862%)
  at 16384³, per unit (`pearlCGammaH100Rev1_8192`, `_16384`) and per audit tile (`pearlCSampledH100Rev1_8192`,
  `_16384`). The granted values were `281273/52340000` (0.53740%) and `536953/103540000` (0.51860%): the credit and
  `W_ref` drop by the same add.
-/

namespace Pouw.PearlC

open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

theorem firstAddDev_nonneg (d : PearlCDevice) (hadd : 0 ≤ d.prices.add) (s : Shape) : 0 ≤ firstAddDev d s := by
  unfold firstAddDev
  split_ifs <;> positivity

theorem creditDevRev1_le (d : PearlCDevice) (hadd : 0 ≤ d.prices.add) (s : Shape) :
    creditDevRev1 d s ≤ creditDev d s := by
  unfold creditDevRev1
  linarith [firstAddDev_nonneg d hadd s]

theorem creditDev_nonneg (d : PearlCDevice) (hadd : 0 ≤ d.prices.add) (hbf : 0 ≤ d.prices.bf16) (s : Shape) :
    0 ≤ creditDev d s := by
  unfold creditDev
  split_ifs <;> positivity

section
variable {Q R S : Type}

theorem unitDebitRev1_ge (sem : PearlCSem Q R S) (d : PearlCDevice) (hadd : 0 ≤ d.prices.add) (H : Q → R) (s : S)
    (U : Workload) (u : ℕ) (act : Codes) :
    debitDev d (sem.unitDebitDev d H s U u act) ≤ sem.unitDebitRev1 d H s U u act := by
  unfold PearlCSem.unitDebitRev1
  have : (0 : ℚ) ≤ d.prices.add * ∑ i, ∑ j, (laterZeroPromos d (sem.unitChain H s U u act).1
      (sem.unitChain H s U u act).2 i j : ℚ) := by positivity
  linarith

theorem unitDebitRev1_nonneg (sem : PearlCSem Q R S) (d : PearlCDevice) (hadd : 0 ≤ d.prices.add) (H : Q → R) (s : S)
    (U : Workload) (u : ℕ) (act : Codes) : 0 ≤ sem.unitDebitRev1 d H s U u act := by
  have h0 : 0 ≤ debitDev d (sem.unitDebitDev d H s U u act) := by
    unfold debitDev
    split_ifs <;> positivity
  linarith [unitDebitRev1_ge sem d hadd H s U u act]

theorem tileDebitRev1_ge (sem : PearlCSem Q R S) (d : PearlCDevice) (hadd : 0 ≤ d.prices.add) (H : Q → R) (s : S)
    (U : Workload) (u : ℕ) (rows cols : Finset ℕ) (act : Codes) :
    sem.tileDebitDev d H s U u rows cols act ≤ sem.tileDebitRev1 d H s U u rows cols act := by
  classical
  unfold PearlCSem.tileDebitRev1
  simp only
  have : (0 : ℚ) ≤ d.prices.add * ∑ e ∈ (univ : Finset (Fin (U.layout.shape u).m × Fin (U.layout.shape u).n)).filter
      (fun e => (e.1 : ℕ) ∈ rows ∧ (e.2 : ℕ) ∈ cols),
      (laterZeroPromos d (sem.unitChain H s U u act).1 (sem.unitChain H s U u act).2 e.1 e.2 : ℚ) := by positivity
  linarith

end

theorem creditDevRev1_h100 (s : Shape) : creditDevRev1 devH100 s = creditOfRev1 Params.pi Costs.adopted s := by
  rw [creditDevRev1, creditOfRev1, creditDev_h100]
  simp only [firstAddDev, firstAddOf, devH100, Prices.h100, Params.pi]
  norm_num

theorem wrefDevRev1_h100 (s : Shape) : wrefDevRev1 devH100 s = wrefOfRev1 Params.pi Costs.adopted s := by
  simp only [wrefDevRev1, wrefOfRev1, creditDevRev1_h100]
  rfl

section
variable {Q R S : Type}

open Classical in
/-- Under rev1 the correct units' credit is at most the granted protocol's, on any transcript. -/
theorem rev1_credit_sum_le (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) (hadd : 0 ≤ d.prices.add)
    (hbf : 0 ≤ d.prices.bf16) (hρ0 : 0 ≤ ρ) (hρ1 : ρ ≤ 1) (U : Workload) (H : Q → R) (s : S) (τ : Transcript) :
    ∑ u ∈ correctSet (pearlCProtocolDevRev1 d sem ρ) U H s τ,
        (pearlCProtocolDevRev1 d sem ρ).credit H s U u (actOf τ u) ≤
      ∑ u ∈ correctSet (pearlCProtocolDev d sem ρ) U H s τ, (pearlCProtocolDev d sem ρ).credit H s U u (actOf τ u) := by
  have hsub : correctSet (pearlCProtocolDevRev1 d sem ρ) U H s τ ⊆ correctSet (pearlCProtocolDev d sem ρ) U H s τ := by
    intro u hu
    unfold correctSet at hu ⊢
    rw [mem_filter] at hu ⊢
    obtain ⟨hu, c, hc, hact, hcap, hw⟩ := hu
    refine ⟨hu, c, hc, hact, ?_, hw⟩
    have h1 : sem.unitDebitRev1 d H s U u c.act ≤ ρ * creditDevRev1 d (U.layout.shape u) := hcap
    show debitDev d (sem.unitDebitDev d H s U u c.act) ≤ ρ * creditDev d (U.layout.shape u)
    have := mul_le_mul_of_nonneg_left (creditDevRev1_le d hadd (U.layout.shape u)) hρ0
    linarith [unitDebitRev1_ge sem d hadd H s U u c.act]
  calc ∑ u ∈ correctSet (pearlCProtocolDevRev1 d sem ρ) U H s τ,
          (pearlCProtocolDevRev1 d sem ρ).credit H s U u (actOf τ u)
      ≤ ∑ u ∈ correctSet (pearlCProtocolDevRev1 d sem ρ) U H s τ,
          (pearlCProtocolDev d sem ρ).credit H s U u (actOf τ u) := by
        refine sum_le_sum fun u _ => ?_
        show ((creditDevRev1 d (U.layout.shape u) - sem.unitDebitRev1 d H s U u (actOf τ u) : ℚ) : ℝ) ≤
          ((creditDev d (U.layout.shape u) - debitDev d (sem.unitDebitDev d H s U u (actOf τ u)) : ℚ) : ℝ)
        have h1 := creditDevRev1_le d hadd (U.layout.shape u)
        have h2 := unitDebitRev1_ge sem d hadd H s U u (actOf τ u)
        exact_mod_cast (by linarith)
    _ ≤ ∑ u ∈ correctSet (pearlCProtocolDev d sem ρ) U H s τ, (pearlCProtocolDev d sem ρ).credit H s U u (actOf τ u) := by
        refine sum_le_sum_of_subset_of_nonneg hsub fun u hu _ => ?_
        unfold correctSet at hu
        rw [mem_filter] at hu
        obtain ⟨_, c, hc, _, hcap, _⟩ := hu
        have hact : actOf τ u = c.act := by simp [actOf, hc]
        rw [hact]
        have h1 : debitDev d (sem.unitDebitDev d H s U u c.act) ≤ ρ * creditDev d (U.layout.shape u) := hcap
        have h2 := creditDev_nonneg d hadd hbf (U.layout.shape u)
        show (0 : ℝ) ≤ ((creditDev d (U.layout.shape u) - debitDev d (sem.unitDebitDev d H s U u c.act) : ℚ) : ℝ)
        have : (0 : ℚ) ≤ creditDev d (U.layout.shape u) - debitDev d (sem.unitDebitDev d H s U u c.act) := by
          nlinarith
        exact_mod_cast this

/-- **rev1 asserts less than the granted TT_OUT at the same record**, for non-negative prices and `0 ≤ ρ ≤ 1`. An
implication only: its premise `TTOutPearlCDev` is falsified by the first-promotion skip (rating D) at the H100 and
sm_120 records, so it carries no evidence for rev1. -/
theorem ttOutPearlCDevRev1_of_granted [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) (hadd : 0 ≤ d.prices.add) (hbf : 0 ≤ d.prices.bf16)
    (hρ0 : 0 ≤ ρ) (hρ1 : ρ ≤ 1) (h : TTOutPearlCDev CM d sem ρ) : TTOutPearlCDevRev1 CM d sem ρ := by
  refine ⟨h.1, fun U hU T hT q hq A hA => le_trans (Pouw.pr_mono ?_) (h.2 U hU T hT q hq A hA)⟩
  intro x hx
  exact lt_of_lt_of_le hx (rev1_credit_sum_le d sem ρ hadd hbf hρ0 hρ1 U x.1 x.2 _)

open Classical in
/-- Under rev1 the good tiles' credit is at most the granted tiles', on any transcript. -/
theorem rev1_tile_credit_sum_le (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) (hadd : 0 ≤ d.prices.add)
    (hbf : 0 ≤ d.prices.bf16) (hρ0 : 0 ≤ ρ) (hρ1 : ρ ≤ 1) (U : Workload) (H : Q → R) (s : S) (τ : Transcript) :
    ∑ g ∈ goodTiles (pearlCProtocolDevRev1 d sem ρ) (pearlCTilesDevRev1 d sem ρ) U H s τ,
        (pearlCTilesDevRev1 d sem ρ).credit H s U g
          (actOf τ (((pearlCTilesDevRev1 d sem ρ).tiling U.layout).unit g)) ≤
      ∑ g ∈ goodTiles (pearlCProtocolDev d sem ρ) (pearlCTilesDev d sem ρ) U H s τ,
        (pearlCTilesDev d sem ρ).credit H s U g (actOf τ (((pearlCTilesDev d sem ρ).tiling U.layout).unit g)) := by
  set tl := auditTiling 64 64 U.layout
  have hsub : goodTiles (pearlCProtocolDevRev1 d sem ρ) (pearlCTilesDevRev1 d sem ρ) U H s τ ⊆
      goodTiles (pearlCProtocolDev d sem ρ) (pearlCTilesDev d sem ρ) U H s τ := by
    intro g hg
    unfold goodTiles at hg ⊢
    rw [mem_filter] at hg ⊢
    obtain ⟨hg, c, hc, hcap, hw⟩ := hg
    refine ⟨hg, c, hc, ?_, hw⟩
    have h1 : sem.tileDebitRev1 d H s U (tl.unit g) (sem.passRows (U.layout.shape (tl.unit g)) c.act (tl.rows g))
        (tl.cols g) c.act ≤ ρ * creditDevRev1 d (U.layout.shape (tl.unit g)) *
          tileShare (U.layout.shape (tl.unit g)) (sem.passRows (U.layout.shape (tl.unit g)) c.act (tl.rows g))
            (tl.cols g) := hcap
    show sem.tileDebitDev d H s U (tl.unit g) (sem.passRows (U.layout.shape (tl.unit g)) c.act (tl.rows g))
        (tl.cols g) c.act ≤ ρ * creditDev d (U.layout.shape (tl.unit g)) *
          tileShare (U.layout.shape (tl.unit g)) (sem.passRows (U.layout.shape (tl.unit g)) c.act (tl.rows g))
            (tl.cols g)
    have hsh := tileShare_nonneg (U.layout.shape (tl.unit g))
      (sem.passRows (U.layout.shape (tl.unit g)) c.act (tl.rows g)) (tl.cols g)
    have := mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left
      (creditDevRev1_le d hadd (U.layout.shape (tl.unit g))) hρ0) hsh
    linarith [tileDebitRev1_ge sem d hadd H s U (tl.unit g)
      (sem.passRows (U.layout.shape (tl.unit g)) c.act (tl.rows g)) (tl.cols g) c.act]
  calc ∑ g ∈ goodTiles (pearlCProtocolDevRev1 d sem ρ) (pearlCTilesDevRev1 d sem ρ) U H s τ,
          (pearlCTilesDevRev1 d sem ρ).credit H s U g (actOf τ (tl.unit g))
      ≤ ∑ g ∈ goodTiles (pearlCProtocolDevRev1 d sem ρ) (pearlCTilesDevRev1 d sem ρ) U H s τ,
          (pearlCTilesDev d sem ρ).credit H s U g (actOf τ (tl.unit g)) := by
        refine sum_le_sum fun g _ => ?_
        simp only [pearlCTilesDevRev1, pearlCTilesDev]
        have hsh := tileShare_nonneg (U.layout.shape (tl.unit g))
          (sem.passRows (U.layout.shape (tl.unit g)) (actOf τ (tl.unit g)) (tl.rows g)) (tl.cols g)
        have := mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left
          (creditDevRev1_le d hadd (U.layout.shape (tl.unit g))) (by linarith : (0 : ℚ) ≤ 1 - ρ)) hsh
        exact_mod_cast this
    _ ≤ ∑ g ∈ goodTiles (pearlCProtocolDev d sem ρ) (pearlCTilesDev d sem ρ) U H s τ,
          (pearlCTilesDev d sem ρ).credit H s U g (actOf τ (tl.unit g)) := by
        refine sum_le_sum_of_subset_of_nonneg hsub fun g _ _ => ?_
        simp only [pearlCTilesDev]
        have hsh := tileShare_nonneg (U.layout.shape (tl.unit g))
          (sem.passRows (U.layout.shape (tl.unit g)) (actOf τ (tl.unit g)) (tl.rows g)) (tl.cols g)
        have h2 := creditDev_nonneg d hadd hbf (U.layout.shape (tl.unit g))
        have : (0 : ℚ) ≤ (1 - ρ) * creditDev d (U.layout.shape (tl.unit g)) *
            tileShare (U.layout.shape (tl.unit g))
              (sem.passRows (U.layout.shape (tl.unit g)) (actOf τ (tl.unit g)) (tl.rows g)) (tl.cols g) :=
          mul_nonneg (mul_nonneg (by linarith) h2) hsh
        exact_mod_cast this

/-- **rev1 per tile asserts less than the granted TT_OUT per tile at the same record.** An implication only: its premise
`TTOutTilePearlCDev` is falsified by the first-promotion skip (rating D) at the H100 and sm_120 records, so it carries
no evidence for rev1. -/
theorem ttOutTilePearlCDevRev1_of_granted [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) (hadd : 0 ≤ d.prices.add) (hbf : 0 ≤ d.prices.bf16)
    (hρ0 : 0 ≤ ρ) (hρ1 : ρ ≤ 1) (h : TTOutTilePearlCDev CM d sem ρ) : TTOutTilePearlCDevRev1 CM d sem ρ := by
  refine ⟨h.1, fun U hU T hT q hq A hA => le_trans (Pouw.pr_mono ?_) (h.2 U hU T hT q hq A hA)⟩
  intro x hx
  exact lt_of_lt_of_le hx (rev1_tile_credit_sum_le d sem ρ hadd hbf hρ0 hρ1 U x.1 x.2 _)

end

/-- TT_OUT rev1 at a device record gives `G_γ` on the one-shape domain at `γ = 1 − (399/400)/ω`, where
`wrefDevRev1 = ω·(1 − ρ)·creditDevRev1`. -/
theorem pearlCGammaDevRev1At {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) (s : Shape) (ω γ : ℚ) (hω0 : 0 < ω)
    (hω : wrefDevRev1 d s = ω * ((1 - ρ) * creditDevRev1 d s)) (hγ : γ = 1 - (1 - 1 / 400) / ω)
    (hTT : TTOutPearlCDevRev1 CM d sem ρ) :
    Gγ CM (pearlCProtocolDevRev1 d sem ρ) (pearlCDomainDevAt d s) (γ : ℝ) εPearlC := by
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
  have key := gammaFromTTOut CM (pearlCProtocolDevRev1 d sem ρ) (pearlCDomainDevAt d s) (1 / 400) (ω : ℝ) εPearlC
    (by exact_mod_cast hω0) hW (ttOut_mono_domain (fun _ h => h.1) hTT)
  have e : (γ : ℝ) = 1 - (1 - 1 / 400) / (ω : ℝ) := by rw [hγ]; push_cast; ring
  rw [e]
  exact key

/-- TT_OUT rev1 per tile at a device record gives `GγSampled` on the one-shape domain at `γ = 1 − (399/400)/ω`, where
`wrefDevRev1 = ω·(1 − ρ)·creditDevRev1 ≥ 0`. -/
theorem pearlCSampledDevRev1At {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) (s : Shape) (ω γ : ℚ) (hω0 : 0 < ω)
    (hw0 : 0 ≤ wrefDevRev1 d s) (hω : wrefDevRev1 d s = ω * ((1 - ρ) * creditDevRev1 d s))
    (hγ : γ = 1 - (1 - 1 / 400) / ω) (hTT : TTOutTilePearlCDevRev1 CM d sem ρ) :
    GγSampled CM (pearlCProtocolDevRev1 d sem ρ) (pearlCTilesDevRev1 d sem ρ) (pearlCDomainDevAt d s) (γ : ℝ)
      εPearlC := by
  have hW : ∀ U : Workload, U.InDomain (pearlCProtocolDevRev1 d sem ρ) (pearlCDomainDevAt d s) →
      ∀ (H : Q → R) (s' : S), ∀ g < ((pearlCTilesDevRev1 d sem ρ).tiling U.layout).NT, ∀ act : Codes,
        (pearlCTilesDevRev1 d sem ρ).capOK H s' U g act →
        (pearlCTilesDevRev1 d sem ρ).Wcred U.layout g act ≤ (ω : ℝ) * (pearlCTilesDevRev1 d sem ρ).credit H s' U g act := by
    intro U hU H s' g hg act _
    have hu := auditTiling_unit_lt 64 64 U.layout hg
    have hsh : U.layout.shape ((auditTiling 64 64 U.layout).unit g) = s := hU.2.2 _ hu
    simp only [pearlCTilesDevRev1]
    rw [hsh, hω]
    push_cast
    exact le_of_eq (by ring)
  have hWc : ∀ U : Workload, U.InDomain (pearlCProtocolDevRev1 d sem ρ) (pearlCDomainDevAt d s) →
      ∀ g < ((pearlCTilesDevRev1 d sem ρ).tiling U.layout).NT, ∀ act : Codes,
        (pearlCTilesDevRev1 d sem ρ).Wcred U.layout g act ≤ (pearlCTilesDevRev1 d sem ρ).Wref U.layout g := by
    intro U hU g hg act
    have hu := auditTiling_unit_lt 64 64 U.layout hg
    have hsh : U.layout.shape ((auditTiling 64 64 U.layout).unit g) = s := hU.2.2 _ hu
    simp only [pearlCTilesDevRev1]
    rw [hsh]
    exact_mod_cast mul_le_mul_of_nonneg_left (tileShare_mono _ (passRows_subset sem _ _ _) _) hw0
  have key := gammaSampled CM (pearlCProtocolDevRev1 d sem ρ) (pearlCTilesDevRev1 d sem ρ) (pearlCDomainDevAt d s)
    (1 / 400) (ω : ℝ) εPearlC (by norm_num) (by exact_mod_cast hω0) hW hWc
    (ttOutTile_mono_domain (fun _ h => h.1) hTT)
  have e : (γ : ℝ) = 1 - (1 - 1 / 400) / (ω : ℝ) := by rw [hγ]; push_cast; ring
  rw [e]
  exact key

/-- **The H100 headline under rev1, 8192³**, worst case under the cap `1/400`: `γ = 140237/26090000` (0.53751%). -/
theorem pearlCGammaH100Rev1_8192 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (sem : PearlCSem Q R S) (hTT : TTOutPearlCH100Rev1 CM sem (1 / 400)) :
    Gγ CM (pearlCProtocolDevRev1 devH100 sem (1 / 400)) (pearlCDomainDevAt devH100 sh8192)
      ((140237 / 26090000 : ℚ) : ℝ) εPearlC :=
  pearlCGammaDevRev1At CM devH100 sem _ _ (65225 / 65037) _ (by norm_num)
    (by norm_num [wrefDevRev1, creditDevRev1, firstAddDev, creditDev, devH100, Prices.h100, Params.pi,
      Costs.adopted, sh8192]) (by norm_num) hTT

/-- **The H100 headline under rev1, 16384³**, worst case under the cap `1/400`: `γ = 89359/17230000` (0.51862%). -/
theorem pearlCGammaH100Rev1_16384 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (sem : PearlCSem Q R S) (hTT : TTOutPearlCH100Rev1 CM sem (1 / 400)) :
    Gγ CM (pearlCProtocolDevRev1 devH100 sem (1 / 400)) (pearlCDomainDevAt devH100 sh16384)
      ((89359 / 17230000 : ℚ) : ℝ) εPearlC :=
  pearlCGammaDevRev1At CM devH100 sem _ _ (43075 / 42959) _ (by norm_num)
    (by norm_num [wrefDevRev1, creditDevRev1, firstAddDev, creditDev, devH100, Prices.h100, Params.pi,
      Costs.adopted, sh16384]) (by norm_num) hTT

/-- **The H100 headline under rev1 per audit tile, 8192³**: `GγSampled` at `γ = 140237/26090000`. -/
theorem pearlCSampledH100Rev1_8192 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCH100Rev1 CM sem (1 / 400)) :
    GγSampled CM (pearlCProtocolDevRev1 devH100 sem (1 / 400)) (pearlCTilesDevRev1 devH100 sem (1 / 400))
      (pearlCDomainDevAt devH100 sh8192) ((140237 / 26090000 : ℚ) : ℝ) εPearlC :=
  pearlCSampledDevRev1At CM devH100 sem _ _ (65225 / 65037) _ (by norm_num)
    (by norm_num [wrefDevRev1, creditDevRev1, firstAddDev, creditDev, devH100, Prices.h100, Params.pi,
      Costs.adopted, sh8192])
    (by norm_num [wrefDevRev1, creditDevRev1, firstAddDev, creditDev, devH100, Prices.h100, Params.pi,
      Costs.adopted, sh8192]) (by norm_num) hTT

/-- **The H100 headline under rev1 per audit tile, 16384³**: `GγSampled` at `γ = 89359/17230000`. -/
theorem pearlCSampledH100Rev1_16384 {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    (CM : CostModel Q R S) (sem : PearlCSem Q R S) (hTT : TTOutTilePearlCH100Rev1 CM sem (1 / 400)) :
    GγSampled CM (pearlCProtocolDevRev1 devH100 sem (1 / 400)) (pearlCTilesDevRev1 devH100 sem (1 / 400))
      (pearlCDomainDevAt devH100 sh16384) ((89359 / 17230000 : ℚ) : ℝ) εPearlC :=
  pearlCSampledDevRev1At CM devH100 sem _ _ (43075 / 42959) _ (by norm_num)
    (by norm_num [wrefDevRev1, creditDevRev1, firstAddDev, creditDev, devH100, Prices.h100, Params.pi,
      Costs.adopted, sh16384])
    (by norm_num [wrefDevRev1, creditDevRev1, firstAddDev, creditDev, devH100, Prices.h100, Params.pi,
      Costs.adopted, sh16384]) (by norm_num) hTT

end Pouw.PearlC

theorem Pouw.SecurityProofs.PearlC.GammaH100Rev1_16384 : Pouw.Guarantees.PearlC.GammaH100Rev1_16384 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGammaH100Rev1_16384 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.h100 sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCH100Rev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.GammaH100Rev1_8192 : Pouw.Guarantees.PearlC.GammaH100Rev1_8192 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCGammaH100Rev1_8192 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.h100 sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutPearlCH100Rev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.SampledH100Rev1_16384 : Pouw.Guarantees.PearlC.SampledH100Rev1_16384 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCSampledH100Rev1_16384 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.h100 sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutTilePearlCH100Rev1_of_w1 hTT)

theorem Pouw.SecurityProofs.PearlC.SampledH100Rev1_8192 : Pouw.Guarantees.PearlC.SampledH100Rev1_8192 :=
  fun sha fetch salt sem hTT =>
    @Pouw.PearlC.pearlCSampledH100Rev1_8192 _ _ _ _ _ _ _ (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.h100 sha fetch) salt) sem
      (Pouw.PearlC.TTOutW1.ttOutTilePearlCH100Rev1_of_w1 hTT)
