module

public import Definitions.Pouw.FpG1.H1
public import Proofs.Pouw.FpG1
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Algebra.Order.Ring.Star
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.Int.Star
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init

/-! FP8-g1 and W4A8-g1 on H1: g1's leaves are H1's. -/

@[expose] public section

namespace Pouw.FpG1

open Finset

theorem h1_act_int {e : Entry} (h : Grids.h1.On .act e) : -126 ≤ e.int ∧ e.int ≤ 127 :=
  act_int_of_on (by decide) h

theorem h1_wt_served {e : Entry} (h : Grids.h1.On .wt e) {U : ℤ} (hU : -16 ≤ U ∧ U ≤ 16) :
    -124 ≤ e.int + U ∧ e.int + U ≤ 125 := by
  obtain ⟨h1, h2⟩ := wt_int_of_on (W := 108) (by decide) (by decide) h
  push_cast at h1 h2
  exact ⟨by omega, by omega⟩

namespace Leaves

theorem fp8_on_act {g : Grids} {v : Leaves} (h : v.Fp8 g) (i l : ℕ) (b : Coin) : g.On .act ((v.act i l).withCoin b) := by
  obtain ⟨R, hv, hR⟩ := h.1 i l
  rw [hv]
  exact hR

theorem fp8_on_wt {g : Grids} {v : Leaves} (h : v.Fp8 g) (l j : ℕ) (b : Coin) : g.On .wt ((v.wt l j).withCoin b) := by
  obtain ⟨R, hv, hR⟩ := h.2 l j
  rw [hv]
  exact hR

theorem w4a8_on_act {g : Grids} {v : Leaves} (h : v.W4a8 g) (i l : ℕ) (b : Coin) :
    g.On .act ((v.act i l).withCoin b) := by
  obtain ⟨R, hv, hR⟩ := h.1 i l
  rw [hv]
  exact hR

theorem w4a8_on_wt {g : Grids} {v : Leaves} (h : v.W4a8 g) (l j : ℕ) (b : Coin) :
    g.On .wt ((v.wt l j).withCoin b) := by
  obtain ⟨m, q, hv, hm⟩ := h.2 l j
  rw [hv]
  exact hm

theorem fp8_inH1 {v : Leaves} (h : v.Fp8 Grids.h1) : v.InH1 := by
  refine ⟨fun i l => ?_, fun l j => ?_⟩
  · obtain ⟨R, hv, hR⟩ := h.1 i l
    have hb : -((120 : ℕ) : ℤ) ≤ R ∧ R ≤ ((120 : ℕ) : ℤ) := (mem_grid_iff.1 hR).1
    rw [hv]
    show -126 ≤ R ∧ R ≤ 126
    omega
  · obtain ⟨R, hv, hR⟩ := h.2 l j
    have hb : -((104 : ℕ) : ℤ) ≤ R ∧ R ≤ ((104 : ℕ) : ℤ) := (mem_grid_iff.1 hR).1
    rw [hv]
    show -110 ≤ R ∧ R ≤ 110
    omega

theorem w4a8_inH1 {v : Leaves} (h : v.W4a8 Grids.h1) : v.InH1 := by
  refine ⟨fun i l => ?_, fun l j => ?_⟩
  · obtain ⟨R, hv, h1, h2⟩ := h.1 i l
    rw [hv]
    exact ⟨h1, h2⟩
  · obtain ⟨m, q, hv, hm1, hms, hq⟩ := h.2 l j
    have hb : -((12 : ℕ) : ℤ) ≤ q ∧ q ≤ ((12 : ℕ) : ℤ) := (mem_grid_iff.1 hq).1
    have hms' : m ≤ 9 := hms
    have hq1 : (-12 : ℤ) ≤ q := by omega
    have hq2 : q ≤ 12 := by omega
    have hm0 : (0 : ℤ) ≤ m := by omega
    have hm9 : (m : ℤ) ≤ 9 := by omega
    have k1 : (m : ℤ) * q ≤ m * 12 := mul_le_mul_of_nonneg_left hq2 hm0
    have k2 : (m : ℤ) * (-12) ≤ m * q := mul_le_mul_of_nonneg_left hq1 hm0
    rw [hv]
    show -110 ≤ (m : ℤ) * q ∧ (m : ℤ) * q ≤ 110
    constructor <;> linarith

end Leaves

theorem encodeFp8_fp8 (g : Grids) (sa sw : ℕ → ℚ) (x y : ℕ → ℕ → ℚ) : (encodeFp8 g sa sw x y).Fp8 g :=
  ⟨fun i l => ⟨rn (e4m3Grid g.act) (x i l / sa i), rfl, fp8_on (sa i) (x i l)⟩,
    fun l j => ⟨rn (e4m3Grid g.wt) (y l j / sw j), rfl, fp8_on (sw j) (y l j)⟩⟩

theorem zero_mem_e2m1x2Grid : (0 : ℤ) ∈ e2m1x2Grid :=
  mem_grid_iff.2 ⟨⟨by omega, by omega⟩, by decide +kernel⟩

theorem encodeW4a8_w4a8 (sa sw : ℕ → ℚ) (m : ℕ → ℕ → ℕ) (x y : ℕ → ℕ → ℚ) :
    (encodeW4a8 Grids.h1 sa sw m x y).W4a8 Grids.h1 := by
  refine ⟨fun i l => ⟨max (-126) (min 126 ⌊x i l / sa i⌋), rfl, le_max_left _ _, max_le (by norm_num) (min_le_left _ _)⟩,
    fun l j => ⟨max 1 (min Grids.h1.scale (m l j)), _, rfl, le_max_left _ _, max_le (by decide) (min_le_left _ _), ?_⟩⟩
  exact rn_mem ⟨0, zero_mem_e2m1x2Grid⟩ _

/-- Every activation the level `0` and every weight `103`: in H1's ranges, but odd and above 16, so no E4M3
integer. -/
def offGrid : Leaves := ⟨fun _ _ => .fp8 0, fun _ _ => .fp8 103⟩

theorem offGrid_inH1 : offGrid.InH1 :=
  ⟨fun _ _ => ⟨show (-126 : ℤ) ≤ 0 by decide, show (0 : ℤ) ≤ 126 by decide⟩,
    fun _ _ => ⟨show (-110 : ℤ) ≤ 103 by decide, show (103 : ℤ) ≤ 110 by decide⟩⟩

theorem offGrid_not_fp8 : ¬ offGrid.Fp8 Grids.h1 := by
  rintro ⟨-, h⟩
  obtain ⟨R, hR, hmem⟩ := h 0 0
  have e : (103 : ℤ) = R := Level.fp8.inj hR
  subst e
  have hP : E4M3Nat (103 : ℤ).natAbs := (mem_grid_iff.1 hmem).2
  exact absurd hP (by decide +kernel)

end Pouw.FpG1
