import Proofs.Pouw.PearlC.TileGamma

/-!
# TT_OUT and γ at a device record

Staged; nothing pinned.
* **The H100 record is the granted statement on a narrower domain:** `pearlCProtocolDev_h100` and
  `pearlCTilesDev_h100`, and `pearlCDomainDev_h100` (the granted domain with `n ≥ 64`, `64 ∣ n`), so
  `TTOutPearlC CM · Costs.adopted` gives `TTOutPearlCDev CM devH100` (`ttOutPearlCDev_h100`), and likewise per tile
  (`ttOutTilePearlCDev_h100`).
* **γ at any device record,** on the one-shape domain where `wrefDev = ω·(1 − ρ)·creditDev`: `G_γ` from TT_OUT
  (`pearlCGammaDevAt`) and `GγSampled` from TT_OUT per tile (`pearlCSampledDevAt`), both at `γ = 1 − (399/400)/ω`.
  `pearlCTilesDev_total`: the audit tiles' `W_ref` is the layout's `∑_u wrefDev`.
-/

namespace Pouw.PearlC

open Finset Assumptions Pouw.Fp8Atom

/-- The record at the Hopper step, promotion every 4 atoms and the H100's prices is `devH100`: every twin is today's
definition at `hopperE4m3K32` and 4, by `rfl`. -/
theorem devAt_hopper : devAt hopperE4m3K32 4 Prices.h100 = devH100 := rfl

theorem creditDev_h100 (s : Shape) : creditDev devH100 s = creditOf Params.pi Costs.adopted s := by
  simp only [creditDev, creditOf, devH100, Prices.h100, Params.pi, Costs.adopted]
  norm_num
  ring

theorem debitDev_h100 (x : Debit) : debitDev devH100 x = debitOf Params.pi Costs.adopted x := by
  simp only [debitDev, debitOf, devH100, Prices.h100, Params.pi, Costs.adopted]
  norm_num

theorem wrefDev_h100 (s : Shape) : wrefDev devH100 s = wrefOf Params.pi Costs.adopted s := by
  simp only [wrefDev, wrefOf, creditDev_h100]
  rfl

section
variable {Q R S : Type}

theorem unitDebitDev_h100 (sem : PearlCSem Q R S) (H : Q → R) (s : S) (U : Workload) (u : ℕ) (act : Codes) :
    sem.unitDebitDev devH100 H s U u act = sem.unitDebit H s U u act := rfl

theorem unitUsefulDev_h100 (sem : PearlCSem Q R S) (H : Q → R) (s : S) (U : Workload) (u : ℕ) (act : Codes) :
    sem.unitUsefulDev devH100 H s U u act = sem.unitUseful H s U u act := rfl

theorem tileDebitDev_h100 (sem : PearlCSem Q R S) (H : Q → R) (s : S) (U : Workload) (u : ℕ) (rows cols : Finset ℕ)
    (act : Codes) : sem.tileDebitDev devH100 H s U u rows cols act = sem.tileDebit Costs.adopted H s U u rows cols act := by
  simp only [PearlCSem.tileDebitDev, PearlCSem.tileDebit, debitDev_h100]
  rfl

/-- The H100 record's protocol is the granted one. -/
theorem pearlCProtocolDev_h100 (sem : PearlCSem Q R S) (ρ : ℚ) :
    pearlCProtocolDev devH100 sem ρ = pearlCProtocol sem Costs.adopted ρ := by
  simp only [pearlCProtocolDev, pearlCProtocol, unitDebitDev_h100, unitUsefulDev_h100, creditDev_h100, debitDev_h100,
    wrefDev_h100]
  rfl

/-- The H100 record's domain is the granted one with `n`'s floor (`pearlCDomainDev`). -/
theorem pearlCDomainDev_h100 (L : Layout) : pearlCDomainDev devH100 L ↔
    pearlCDomain Params.pi L ∧ ∀ u < L.N, 64 ≤ (L.shape u).n ∧ 64 ∣ (L.shape u).n := by
  simp only [pearlCDomainDev, pearlCDomain, Shape.Tiled, devH100, Params.pi]
  constructor
  · intro h
    refine ⟨fun u hu => ?_, fun u hu => (h u hu).2.1⟩
    obtain ⟨h1, h2, h3, _, h5, h6⟩ := h u hu
    exact ⟨h1, by have := h2.1; omega, h3, h5.resolve_left (by norm_num), h6, trivial⟩
  · intro h u hu
    obtain ⟨h1, _, h3, h4, h5, _⟩ := h.1 u hu
    exact ⟨h1, h.2 u hu, h3, h4, Or.inr h4, h5⟩

/-- The H100 record's tiles are the granted delta's. -/
theorem pearlCTilesDev_h100 (sem : PearlCSem Q R S) (ρ : ℚ) :
    pearlCTilesDev devH100 sem ρ = pearlCTiles sem Costs.adopted ρ := by
  simp only [pearlCTilesDev, pearlCTiles, tileDebitDev_h100, creditDev_h100, wrefDev_h100]

/-- The granted `TTOutPearlC` gives TT_OUT at the H100 record, whose domain is the granted one's with `n`'s floor. -/
theorem ttOutPearlCDev_h100 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (sem : PearlCSem Q R S) (ρ : ℚ) (h : TTOutPearlC CM sem Costs.adopted ρ) : TTOutPearlCDev CM devH100 sem ρ := by
  unfold TTOutPearlCDev
  unfold TTOutPearlC at h
  rw [pearlCProtocolDev_h100]
  exact ttOut_mono_domain (D := pearlCDomain Params.pi) (fun L hL => ((pearlCDomainDev_h100 L).1 hL).1) h

theorem ttOutTilePearlCDev_h100 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (sem : PearlCSem Q R S) (ρ : ℚ) (h : TTOutTilePearlC CM sem Costs.adopted ρ) :
    TTOutTilePearlCDev CM devH100 sem ρ := by
  unfold TTOutTilePearlCDev
  unfold TTOutTilePearlC at h
  rw [pearlCProtocolDev_h100, pearlCTilesDev_h100]
  exact ttOutTile_mono_domain (D := pearlCDomain Params.pi) (fun L hL => ((pearlCDomainDev_h100 L).1 hL).1) h

end

/-- TT_OUT at a device record gives `G_γ` on the one-shape domain at `γ = 1 − (399/400)/ω`, where
`wrefDev = ω·(1 − ρ)·creditDev`: a unit within the cap is credited at least `(1 − ρ)·creditDev`. -/
theorem pearlCGammaDevAt {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) (s : Shape) (ω γ : ℚ) (hω0 : 0 < ω)
    (hω : wrefDev d s = ω * ((1 - ρ) * creditDev d s)) (hγ : γ = 1 - (1 - 1 / 400) / ω)
    (hTT : TTOutPearlCDev CM d sem ρ) :
    Gγ CM (pearlCProtocolDev d sem ρ) (pearlCDomainDevAt d s) (γ : ℝ) εPearlC := by
  have hW : ∀ U : Workload, U.InDomain (pearlCProtocolDev d sem ρ) (pearlCDomainDevAt d s) → ∀ (H : Q → R) (s' : S),
      ∀ u < U.layout.N, ∀ act : Codes, (pearlCProtocolDev d sem ρ).actOK (U.layout.shape u) act →
        (pearlCProtocolDev d sem ρ).capOK H s' U u act →
          (pearlCProtocolDev d sem ρ).Wref U.layout u ≤ (ω : ℝ) * (pearlCProtocolDev d sem ρ).credit H s' U u act := by
    intro U hU H s' u hu act _ hcap
    have hsh : U.layout.shape u = s := hU.2.2 u hu
    have hc : debitDev d (sem.unitDebitDev d H s' U u act) ≤ ρ * creditDev d (U.layout.shape u) := hcap
    show (wrefDev d (U.layout.shape u) : ℝ) ≤
      (ω : ℝ) * ((creditDev d (U.layout.shape u) - debitDev d (sem.unitDebitDev d H s' U u act) : ℚ) : ℝ)
    rw [hsh] at hc ⊢
    rw [hω]
    have : ω * ((1 - ρ) * creditDev d s) ≤ ω * (creditDev d s - debitDev d (sem.unitDebitDev d H s' U u act)) :=
      mul_le_mul_of_nonneg_left (by linarith) hω0.le
    exact_mod_cast this
  have key := gammaFromTTOut CM (pearlCProtocolDev d sem ρ) (pearlCDomainDevAt d s) (1 / 400) (ω : ℝ) εPearlC
    (by exact_mod_cast hω0) hW (ttOut_mono_domain (fun _ h => h.1) hTT)
  have e : (γ : ℝ) = 1 - (1 - 1 / 400) / (ω : ℝ) := by rw [hγ]; push_cast; ring
  rw [e]
  exact key

/-- TT_OUT per tile at a device record gives `GγSampled` on the one-shape domain at `γ = 1 − (399/400)/ω`, where
`wrefDev = ω·(1 − ρ)·creditDev ≥ 0`. -/
theorem pearlCSampledDevAt {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S)
    (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) (s : Shape) (ω γ : ℚ) (hω0 : 0 < ω) (hw0 : 0 ≤ wrefDev d s)
    (hω : wrefDev d s = ω * ((1 - ρ) * creditDev d s)) (hγ : γ = 1 - (1 - 1 / 400) / ω)
    (hTT : TTOutTilePearlCDev CM d sem ρ) :
    GγSampled CM (pearlCProtocolDev d sem ρ) (pearlCTilesDev d sem ρ) (pearlCDomainDevAt d s) (γ : ℝ) εPearlC := by
  have hW : ∀ U : Workload, U.InDomain (pearlCProtocolDev d sem ρ) (pearlCDomainDevAt d s) → ∀ (H : Q → R) (s' : S),
      ∀ g < ((pearlCTilesDev d sem ρ).tiling U.layout).NT, ∀ act : Codes, (pearlCTilesDev d sem ρ).capOK H s' U g act →
        (pearlCTilesDev d sem ρ).Wcred U.layout g act ≤ (ω : ℝ) * (pearlCTilesDev d sem ρ).credit H s' U g act := by
    intro U hU H s' g hg act _
    have hu := auditTiling_unit_lt 64 64 U.layout hg
    have hsh : U.layout.shape ((auditTiling 64 64 U.layout).unit g) = s := hU.2.2 _ hu
    simp only [pearlCTilesDev]
    rw [hsh, hω]
    push_cast
    exact le_of_eq (by ring)
  have hWc : ∀ U : Workload, U.InDomain (pearlCProtocolDev d sem ρ) (pearlCDomainDevAt d s) →
      ∀ g < ((pearlCTilesDev d sem ρ).tiling U.layout).NT, ∀ act : Codes,
        (pearlCTilesDev d sem ρ).Wcred U.layout g act ≤ (pearlCTilesDev d sem ρ).Wref U.layout g := by
    intro U hU g hg act
    have hu := auditTiling_unit_lt 64 64 U.layout hg
    have hsh : U.layout.shape ((auditTiling 64 64 U.layout).unit g) = s := hU.2.2 _ hu
    simp only [pearlCTilesDev]
    rw [hsh]
    exact_mod_cast mul_le_mul_of_nonneg_left (tileShare_mono _ (passRows_subset sem _ _ _) _) hw0
  have key := gammaSampled CM (pearlCProtocolDev d sem ρ) (pearlCTilesDev d sem ρ) (pearlCDomainDevAt d s) (1 / 400)
    (ω : ℝ) εPearlC (by norm_num) (by exact_mod_cast hω0) hW hWc (ttOutTile_mono_domain (fun _ h => h.1) hTT)
  have e : (γ : ℝ) = 1 - (1 - 1 / 400) / (ω : ℝ) := by rw [hγ]; push_cast; ring
  rw [e]
  exact key

/-- On a layout of nonempty units, the audit tiles' `W_ref` at a device adds up to the layout's `∑_u wrefDev`. -/
theorem pearlCTilesDev_total {Q R S : Type} (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) (L : Layout)
    (hpos : ∀ u < L.N, 0 < (L.shape u).m ∧ 0 < (L.shape u).n) :
    tileWrefTotal (pearlCTilesDev d sem ρ) L = ∑ u ∈ range L.N, (wrefDev d (L.shape u) : ℝ) := by
  let F : ℕ × ℕ × ℕ → ℝ := fun b => ((wrefDev d (L.shape b.1) * tileShare (L.shape b.1)
    (Ico (b.2.1 * 64) (min ((b.2.1 + 1) * 64) (L.shape b.1).m))
    (Ico (b.2.2 * 64) (min ((b.2.2 + 1) * 64) (L.shape b.1).n)) : ℚ) : ℝ)
  have hdef : tileWrefTotal (pearlCTilesDev d sem ρ) L =
      ∑ g ∈ range (gridTiles 64 64 L).length, F ((gridTiles 64 64 L).getD g (0, 0, 0)) := rfl
  rw [hdef, sum_range_getD, gridTiles, sum_map_flatMap, sum_map_range]
  refine sum_congr rfl fun u hu => ?_
  obtain ⟨hm, hn⟩ := hpos u (mem_range.mp hu)
  rw [sum_map_flatMap, sum_map_range]
  have inner : ∀ bi, (((List.range (((L.shape u).n + 64 - 1) / 64)).map fun bj => (u, bi, bj)).map F).sum
      = ∑ bj ∈ range (((L.shape u).n + 64 - 1) / 64), F (u, bi, bj) := fun bi => by
    rw [List.map_map]
    exact sum_map_range _ _
  rw [sum_congr rfl fun bi _ => inner bi]
  rw [← unit_blocks (wrefDev d (L.shape u)) _ _ hm hn, Rat.cast_sum]
  refine sum_congr rfl fun bi _ => ?_
  rw [Rat.cast_sum]
  rfl

end Pouw.PearlC
