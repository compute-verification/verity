module

public import Definitions.Pouw.FpG1
public import Definitions.Pouw.FpG1.Gemm
public import Proofs.Pouw.Exact
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
import Mathlib.RingTheory.WittVector.IsPoly
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
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init

/-! The arithmetic behind `Definitions.Pouw.FpG1`: g1's grids and ranges, the coin's rank-one terms and the
epilogue's exactness. -/

@[expose] public section

namespace Pouw.FpG1

open Finset Pouw.Proofs

theorem mem_grid_iff {P : ℕ → Prop} [DecidablePred P] {B : ℕ} {z : ℤ} :
    z ∈ grid P B ↔ (-(B : ℤ) ≤ z ∧ z ≤ B) ∧ P z.natAbs := by
  simp only [grid, Finset.mem_filter, Finset.mem_Icc]

theorem e2m1x2Grid_eq : e2m1x2Grid = {-12, -8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8, 12} := by
  decide +kernel

theorem card_e4m3Grid_120 : (e4m3Grid 120).card = 79 := by
  decide +kernel

theorem card_e4m3Grid_60 : (e4m3Grid 60).card = 63 := by
  decide +kernel

theorem e4m3Nat_even {n : ℕ} (h : E4M3Nat n) (h16 : 16 ≤ n) : 2 ∣ n := by
  obtain ⟨_, e, -, hd, hlt⟩ := h
  cases e with
  | zero =>
    simp only [pow_zero, Nat.div_one] at hlt
    omega
  | succ e => exact (dvd_pow_self 2 (Nat.succ_ne_zero e)).trans hd

theorem fp8_int_emod_two {B : ℕ} {R : ℤ} (hR : R ∈ e4m3Grid B) (h16 : 16 ≤ R.natAbs) (b : Coin) :
    (Entry.fp8 R b).int % 2 = (b : ℕ) := by
  have hP : E4M3Nat R.natAbs := (mem_grid_iff.1 hR).2
  have h2 : ((2 : ℕ) : ℤ) ∣ R := Int.ofNat_dvd_left.2 (e4m3Nat_even hP h16)
  obtain ⟨c, hc⟩ := h2
  have hb := b.isLt
  simp only [Entry.int]
  omega

theorem zero_mem_e4m3Grid (B : ℕ) : (0 : ℤ) ∈ e4m3Grid B := by
  refine mem_grid_iff.2 ⟨⟨by omega, by omega⟩, ?_⟩
  decide +kernel

theorem nearest_nonempty {G : Finset ℤ} (hG : G.Nonempty) (v : ℚ) : (nearest G v).Nonempty := by
  obtain ⟨g, hg, hmin⟩ := G.exists_min_image (fun g : ℤ => |v - g|) hG
  exact ⟨g, Finset.mem_filter.2 ⟨hg, hmin⟩⟩

theorem rn_spec {G : Finset ℤ} (hG : G.Nonempty) (v : ℚ) : rn G v ∈ nearest G v := by
  have hn := nearest_nonempty hG v
  unfold rn
  split
  · exact Finset.min'_mem _ _
  · contradiction

theorem rn_mem {G : Finset ℤ} (hG : G.Nonempty) (v : ℚ) : rn G v ∈ G :=
  (Finset.mem_filter.1 (rn_spec hG v)).1

theorem rn_nearest {G : Finset ℤ} (hG : G.Nonempty) (v : ℚ) : ∀ g ∈ G, |v - rn G v| ≤ |v - g| :=
  (Finset.mem_filter.1 (rn_spec hG v)).2

theorem fp8_on {B : ℕ} (s x : ℚ) : rn (e4m3Grid B) (x / s) ∈ e4m3Grid B :=
  rn_mem ⟨0, zero_mem_e4m3Grid B⟩ _

theorem act_int_of_on {g : Grids} (hg : g.act ≤ 126) {e : Entry} (h : g.On .act e) :
    -126 ≤ e.int ∧ e.int ≤ 127 := by
  cases e with
  | fp8 R b =>
    have hR : R ∈ e4m3Grid g.act := h
    obtain ⟨⟨h1, h2⟩, -⟩ := mem_grid_iff.1 hR
    have hb := b.isLt
    simp only [Entry.int]
    exact ⟨by omega, by omega⟩
  | fp4 _ _ _ => exact False.elim h
  | int8 R b =>
    have hR : -126 ≤ R ∧ R ≤ 126 := h
    obtain ⟨h1, h2⟩ := hR
    have hb := b.isLt
    simp only [Entry.int]
    exact ⟨by omega, by omega⟩

theorem wt_int_of_on {g : Grids} {W : ℕ} (hw : g.wt ≤ W) (hs : 12 * g.scale ≤ W) {e : Entry} (h : g.On .wt e) :
    -(W : ℤ) ≤ e.int ∧ e.int ≤ W + 1 := by
  cases e with
  | fp8 R b =>
    have hR : R ∈ e4m3Grid g.wt := h
    obtain ⟨⟨h1, h2⟩, -⟩ := mem_grid_iff.1 hR
    have hb := b.isLt
    simp only [Entry.int]
    exact ⟨by omega, by omega⟩
  | fp4 m q b =>
    have h' : 1 ≤ m ∧ m ≤ g.scale ∧ q ∈ e2m1x2Grid := h
    obtain ⟨hm1, hms, hq⟩ := h'
    obtain ⟨⟨hq1, hq2⟩, -⟩ := mem_grid_iff.1 hq
    have hb := b.isLt
    have hq1' : -12 ≤ q := by omega
    have hq2' : q ≤ 12 := by omega
    have hm0 : (0 : ℤ) ≤ m := by omega
    have hms' : (m : ℤ) ≤ g.scale := by omega
    have hs' : 12 * (g.scale : ℤ) ≤ W := by omega
    have hb' : (0 : ℤ) ≤ ((b : ℕ) : ℤ) ∧ ((b : ℕ) : ℤ) ≤ 1 := ⟨by omega, by omega⟩
    have k1 : (m : ℤ) * q ≤ m * 12 := mul_le_mul_of_nonneg_left hq2' hm0
    have k2 : (m : ℤ) * (-12) ≤ m * q := mul_le_mul_of_nonneg_left hq1' hm0
    simp only [Entry.int]
    constructor <;> linarith [hb'.1, hb'.2]
  | int8 _ _ => exact False.elim h

theorem sm120_act_int {e : Entry} (h : Grids.sm120.On .act e) : -126 ≤ e.int ∧ e.int ≤ 127 :=
  act_int_of_on (by decide) h

theorem sm120_wt_int {e : Entry} (h : Grids.sm120.On .wt e) : -60 ≤ e.int ∧ e.int ≤ 61 := by
  obtain ⟨h1, h2⟩ := wt_int_of_on (W := 60) (by decide) (by decide) h
  exact ⟨by omega, by omega⟩

theorem sm120_wt_noise {e : Entry} (h : Grids.sm120.On .wt e) : InNoise e.int := by
  obtain ⟨h1, h2⟩ := sm120_wt_int h
  exact ⟨by omega, by omega⟩

theorem dc_act_int {e : Entry} (h : Grids.dc.On .act e) : -126 ≤ e.int ∧ e.int ≤ 127 :=
  act_int_of_on (by decide) h

theorem dc_wt_int {e : Entry} (h : Grids.dc.On .wt e) : -120 ≤ e.int ∧ e.int ≤ 121 := by
  obtain ⟨h1, h2⟩ := wt_int_of_on (W := 120) (by decide) (by decide) h
  exact ⟨by omega, by omega⟩

theorem int8_of_on {g : Grids} (hg : g = Grids.sm120 ∨ g = Grids.dc) {s : Side} {e : Entry} (h : g.On s e) :
    InInt8 e.int := by
  rcases hg with rfl | rfl <;> cases s
  · obtain ⟨h1, h2⟩ := sm120_act_int h; exact ⟨by omega, by omega⟩
  · obtain ⟨h1, h2⟩ := sm120_wt_int h; exact ⟨by omega, by omega⟩
  · obtain ⟨h1, h2⟩ := dc_act_int h; exact ⟨by omega, by omega⟩
  · obtain ⟨h1, h2⟩ := dc_wt_int h; exact ⟨by omega, by omega⟩

theorem centred_eq (cx cw : ℚ) (k : ℕ) (A W : Mat) (i j : ℕ) :
    centred cx cw k A W i j = (mm k A W i j : ℚ) - rowTerm cx cw k A i - colTerm cx k W j := by
  simp only [centred, rowTerm, colTerm, rowSum, colSum, mm]
  push_cast
  induction k with
  | zero => simp
  | succ k ih =>
    simp only [Finset.sum_range_succ]
    push_cast
    linear_combination ih

theorem model_eq_centred (sa sw : ℕ → ℚ) (k : ℕ) (EA EW : ℕ → ℕ → Entry) (i j : ℕ) (cx cw : ℚ)
    (hA : ∀ l < k, (EA i l).offset = cx) (hW : ∀ l < k, (EW l j).offset = cw) :
    model sa sw k EA EW i j = sa i * sw j * centred cx cw k (ints EA) (ints EW) i j := by
  unfold model centred
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun l hl => ?_
  have hl' := Finset.mem_range.1 hl
  simp only [Entry.served, ints, hA l hl', hW l hl']
  ring

theorem model_eq (sa sw : ℕ → ℚ) (k : ℕ) (EA EW : ℕ → ℕ → Entry) (i j : ℕ) (cx cw : ℚ)
    (hA : ∀ l < k, (EA i l).offset = cx) (hW : ∀ l < k, (EW l j).offset = cw) :
    model sa sw k EA EW i j = sa i * sw j *
      ((mm k (ints EA) (ints EW) i j : ℚ) - rowTerm cx cw k (ints EA) i - colTerm cx k (ints EW) j) := by
  rw [model_eq_centred sa sw k EA EW i j cx cw hA hW, centred_eq]

theorem epilogue_eq (ex ew : ℤ) (k : ℕ) (A W : Mat) (i j : ℕ) :
    (epilogue ex ew k A W i j : ℚ) = 4 * centred ((ex : ℚ) / 2) ((ew : ℚ) / 2) k A W i j := by
  rw [centred_eq]
  simp only [epilogue, rowTerm, colTerm]
  push_cast
  ring

theorem model_eq_epilogue (sa sw : ℕ → ℚ) (k : ℕ) (EA EW : ℕ → ℕ → Entry) (i j : ℕ) (ex ew : ℤ)
    (hA : ∀ l < k, (EA i l).offset = (ex : ℚ) / 2) (hW : ∀ l < k, (EW l j).offset = (ew : ℚ) / 2) :
    model sa sw k EA EW i j = sa i * sw j / 4 * (epilogue ex ew k (ints EA) (ints EW) i j : ℚ) := by
  rw [model_eq_centred sa sw k EA EW i j _ _ hA hW, epilogue_eq]
  ring

theorem mul_bounds_g1 {a w : ℤ} (ha : -127 ≤ a ∧ a ≤ 127) (hw : -121 ≤ w ∧ w ≤ 121) :
    -15367 ≤ a * w ∧ a * w ≤ 15367 := by
  obtain ⟨ha1, ha2⟩ := ha; obtain ⟨hw1, hw2⟩ := hw
  constructor <;> nlinarith [mul_nonneg (sub_nonneg.2 ha1) (sub_nonneg.2 hw1),
    mul_nonneg (sub_nonneg.2 ha2) (sub_nonneg.2 hw2), mul_nonneg (sub_nonneg.2 ha1) (sub_nonneg.2 hw2),
    mul_nonneg (sub_nonneg.2 ha2) (sub_nonneg.2 hw1)]

theorem epilogueW_eq {ex ew : ℤ} (hex : ex = 0 ∨ ex = 1) (hew : ew = 0 ∨ ew = 1) {k : ℕ} (hk : k ≤ 32768)
    {A W : Mat} {i j : ℕ} (hA : ∀ l < k, -127 ≤ A i l ∧ A i l ≤ 127) (hW : ∀ l < k, -121 ≤ W l j ∧ W l j ≤ 121) :
    InInt32 (mm k A W i j) ∧ mmW k A W i j = mm k A W i j ∧ InInt32 (epilogue ex ew k A W i j) ∧
      epilogueW ex ew k A W i j = epilogue ex ew k A W i j := by
  have hk' : (k : ℤ) ≤ 32768 := by exact_mod_cast hk
  obtain ⟨m1, m2⟩ := sum_bounds k (fun l => A i l * W l j) (-15367) 15367
    fun l hl => mul_bounds_g1 (hA l hl) (hW l hl)
  obtain ⟨r1, r2⟩ := sum_bounds k (fun l => A i l) (-127) 127 hA
  obtain ⟨c1, c2⟩ := sum_bounds k (fun l => W l j) (-121) 121 hW
  have hmm : InInt32 (mm k A W i j) := by
    unfold InInt32 mm
    constructor <;> nlinarith [m1, m2]
  have hep : InInt32 (epilogue ex ew k A W i j) := by
    unfold InInt32 epilogue rowSum colSum mm
    rcases hex with rfl | rfl <;> rcases hew with rfl | rfl <;> constructor <;> linarith [m1, m2, r1, r2, c1, c2]
  refine ⟨hmm, mmW_eq_of_mem hmm, hep, ?_⟩
  refine eq_of_cast_eq (wrap32_mem _) hep ?_
  simp only [epilogueW, epilogue, wrap32_cast, Int.cast_sub, Int.cast_add, Int.cast_mul, accW_cast, cast_mmW_apply,
    cast_mm_apply, rowSum, colSum, Int.cast_sum]

theorem model_eq_epilogueW {g : Grids} (hga : g.act ≤ 126) (hgw : g.wt ≤ 120) (hgs : 12 * g.scale ≤ 120)
    (sa sw : ℕ → ℚ) {k : ℕ} (hk : k ≤ 32768) (EA EW : ℕ → ℕ → Entry) (i j : ℕ) {ex ew : ℤ}
    (hex : ex = 0 ∨ ex = 1) (hew : ew = 0 ∨ ew = 1)
    (hA : ∀ l < k, g.On .act (EA i l) ∧ (EA i l).offset = (ex : ℚ) / 2)
    (hW : ∀ l < k, g.On .wt (EW l j) ∧ (EW l j).offset = (ew : ℚ) / 2) :
    model sa sw k EA EW i j = sa i * sw j / 4 * (epilogueW ex ew k (ints EA) (ints EW) i j : ℚ) := by
  have hA' : ∀ l < k, -127 ≤ ints EA i l ∧ ints EA i l ≤ 127 := fun l hl => by
    obtain ⟨h1, h2⟩ := act_int_of_on hga (hA l hl).1
    show -127 ≤ (EA i l).int ∧ (EA i l).int ≤ 127
    exact ⟨by omega, by omega⟩
  have hW' : ∀ l < k, -121 ≤ ints EW l j ∧ ints EW l j ≤ 121 := fun l hl => by
    obtain ⟨h1, h2⟩ := wt_int_of_on hgw hgs (hW l hl).1
    show -121 ≤ (EW l j).int ∧ (EW l j).int ≤ 121
    exact ⟨by omega, by omega⟩
  rw [(epilogueW_eq hex hew hk hA' hW').2.2.2]
  exact model_eq_epilogue sa sw k EA EW i j ex ew (fun l hl => (hA l hl).2) (fun l hl => (hW l hl).2)

theorem sm120_exact (sa sw : ℕ → ℚ) {k : ℕ} (hk : k ≤ 32768) (EA EW : ℕ → ℕ → Entry) (i j : ℕ) {ex ew : ℤ}
    (hex : ex = 0 ∨ ex = 1) (hew : ew = 0 ∨ ew = 1)
    (hA : ∀ l < k, Grids.sm120.On .act (EA i l) ∧ (EA i l).offset = (ex : ℚ) / 2)
    (hW : ∀ l < k, Grids.sm120.On .wt (EW l j) ∧ (EW l j).offset = (ew : ℚ) / 2) :
    model sa sw k EA EW i j = sa i * sw j / 4 * (epilogueW ex ew k (ints EA) (ints EW) i j : ℚ) :=
  model_eq_epilogueW (by decide) (by decide) (by decide) sa sw hk EA EW i j hex hew hA hW

theorem dc_exact (sa sw : ℕ → ℚ) {k : ℕ} (hk : k ≤ 32768) (EA EW : ℕ → ℕ → Entry) (i j : ℕ) {ex ew : ℤ}
    (hex : ex = 0 ∨ ex = 1) (hew : ew = 0 ∨ ew = 1)
    (hA : ∀ l < k, Grids.dc.On .act (EA i l) ∧ (EA i l).offset = (ex : ℚ) / 2)
    (hW : ∀ l < k, Grids.dc.On .wt (EW l j) ∧ (EW l j).offset = (ew : ℚ) / 2) :
    model sa sw k EA EW i j = sa i * sw j / 4 * (epilogueW ex ew k (ints EA) (ints EW) i j : ℚ) :=
  model_eq_epilogueW (by decide) (by decide) (by decide) sa sw hk EA EW i j hex hew hA hW

theorem offset_fp8 (R : ℤ) (b : Coin) : (Entry.fp8 R b).offset = ((1 : ℤ) : ℚ) / 2 := by
  simp [Entry.offset]

theorem offset_fp4 (m : ℕ) (q : ℤ) (b : Coin) : (Entry.fp4 m q b).offset = ((1 : ℤ) : ℚ) / 2 := by
  simp [Entry.offset]

theorem offset_int8 (R : ℤ) (b : Coin) : (Entry.int8 R b).offset = ((0 : ℤ) : ℚ) / 2 := by
  simp [Entry.offset]

end Pouw.FpG1
