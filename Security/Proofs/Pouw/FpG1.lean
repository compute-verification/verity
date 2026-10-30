import Definitions.Pouw.FpG1
import Proofs.Pouw.Exact

/-!
# FP8-g1 and W4A8-g1: ranges, the coin's rank-one terms and the epilogue's exactness

The arithmetic behind `Definitions.Pouw.FpG1`:

* the grids: `e2m1x2Grid` is `steps.py`'s `e2m1x2` (`e2m1x2Grid_eq`), and E4M3's integers to 120 and to 60 are 79 and
  63 levels (`card_e4m3Grid_120`, `card_e4m3Grid_60`), all by `decide`; every E4M3 integer of magnitude at least 16 is
  even (`e4m3Nat_even`), so above 16 an FP8-g1 entry's low bit is its coin (`fp8_int_emod_two`); the RN encoder lands
  on its grid (`rn_mem`, `fp8_on`);
* the ranges: an entry on a device's g1 grids, its coin added, is an s8 value, an activation in `[−126, 127]` and a
  weight in `[−60, 61]` on sm_120 (inside co32c's 7-bit weights `[−63, 63]`, and so inside H1's, to ±110) and
  `[−120, 121]` on B200 and H200 (`sm120_act_int`, `sm120_wt_int`, `sm120_wt_noise`, `dc_act_int`, `dc_wt_int`);
* the GEMM: the centred product is the integer GEMM's exact product less the coin's two rank-one terms
  (`centred_eq`), so the served model's exact product is the scales times that (`model_eq`), and four times the
  centred product is the epilogue's integer (`epilogue_eq`, `model_eq_epilogue`);
* int32: for `k ≤ 32,768` and operands in `[−127, 127]` and `[−121, 121]` the s32 accumulator and the int32 epilogue
  never wrap (`epilogueW_eq`), so on a device's g1 grids the kernel's int32 epilogue, times `s_a(i)·s_w(j)/4`, is the
  served model's exact product (`model_eq_epilogueW`, `sm120_exact`, `dc_exact`).

g1's windows are the record's. The record's headlines (`Definitions.Accounting.Guarantees.Headline`) quantify over every
value type and every proof layer, and their conclusion reads neither, so they cover every window a g1 prover commits
by their own quantifier over committed values. `Definitions.Accounting.Guarantees.FpG1` states them at g1's windows on H1,
where the premise that the committed leaves are g1's narrows the windows and leaves the bound the record's, and
`Proofs.Accounting.Headline.FpG1` shows that an honest g1 window meets every premise. What
these lemmas add is that g1's transcript is the record's: every entry plus its coin is an s8 value inside the record's
operand ranges, and the int8 GEMM the record's MMA computes is the served model's exact product up to the coin's two
rank-one terms. The record's premises read at g1's values as they do at any other:

* the luck premises (`OperandHMu`, `ReceiptLuckAt`, `WholeLuckAt`) are Props of the receipt game over every level a
  prover commits, and nothing checks g1's grids, so a g1 prover is one of the record's provers;
* the machine premises read no operand values: A3 (`SheetTensor`) and A6 (`SheetWords`) price every MMA form, QMMA
  included, per instruction and for any data, as the measured headline's `FacetCeilings` does at its measured rates,
  and A7's verdicts on the units outside that menu read no alphabet. That is why they hold at g1's values, not
  that `isa_escape.py`'s dense alphabets contain g1's entries: its decomposition verdicts are worst cases over an
  alphabet, which don't carry to a subset of it.

A3 sums the types' MACs as "the types sharing the pipe's time", and on sm_120 QMMA does share IMMA's tensor pipe: timed
together, `QMMA.16832` E4M3 and `IMMA.16832` give `t + s = 1.0002` and `0.9994`, so a run that
splits its words between them is charged in full, for g1 as for the record.
-/

namespace Pouw.FpG1

open Finset Pouw.Proofs

/-! ## The grids -/

theorem mem_grid_iff {P : ℕ → Prop} [DecidablePred P] {B : ℕ} {z : ℤ} :
    z ∈ grid P B ↔ (-(B : ℤ) ≤ z ∧ z ≤ B) ∧ P z.natAbs := by
  simp only [grid, Finset.mem_filter, Finset.mem_Icc]

theorem e2m1x2Grid_eq : e2m1x2Grid = {-12, -8, -6, -4, -3, -2, -1, 0, 1, 2, 3, 4, 6, 8, 12} := by
  decide +kernel

theorem card_e4m3Grid_120 : (e4m3Grid 120).card = 79 := by
  decide +kernel

theorem card_e4m3Grid_60 : (e4m3Grid 60).card = 63 := by
  decide +kernel

/-- Every E4M3 integer of magnitude at least 16 is even. -/
theorem e4m3Nat_even {n : ℕ} (h : E4M3Nat n) (h16 : 16 ≤ n) : 2 ∣ n := by
  obtain ⟨_, e, -, hd, hlt⟩ := h
  cases e with
  | zero =>
    simp only [pow_zero, Nat.div_one] at hlt
    omega
  | succ e => exact (dvd_pow_self 2 (Nat.succ_ne_zero e)).trans hd

/-- Above 16, an FP8-g1 entry's low bit is its coin. -/
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

/-- The RN encoder lands on its grid. -/
theorem rn_mem {G : Finset ℤ} (hG : G.Nonempty) (v : ℚ) : rn G v ∈ G :=
  (Finset.mem_filter.1 (rn_spec hG v)).1

/-- The RN encoder's level is a nearest one. -/
theorem rn_nearest {G : Finset ℤ} (hG : G.Nonempty) (v : ℚ) : ∀ g ∈ G, |v - rn G v| ≤ |v - g| :=
  (Finset.mem_filter.1 (rn_spec hG v)).2

/-- FP8-g1's RN encoder gives an entry on E4M3's integers to `B`. -/
theorem fp8_on {B : ℕ} (s x : ℚ) : rn (e4m3Grid B) (x / s) ∈ e4m3Grid B :=
  rn_mem ⟨0, zero_mem_e4m3Grid B⟩ _

/-! ## Ranges -/

/-- An activation on a device's g1 grids whose E4M3 bound is at most 126 is in `[−126, 127]`, its coin added. -/
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

/-- A weight on a device's g1 grids is in `[−W, W + 1]`, its coin added, when E4M3's bound for weights and twelve
times the largest block scale are at most `W`. -/
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

/-- On sm_120 a g1 weight, its coin added, is in `[−60, 61]`: inside co32c's 7-bit weights `[−63, 63]`
(`sm120_wt_noise`), and so inside H1's (co32c-u16, to ±110). -/
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

/-- Every g1 operand entry, on either device's grids at either side, is an s8 value. -/
theorem int8_of_on {g : Grids} (hg : g = Grids.sm120 ∨ g = Grids.dc) {s : Side} {e : Entry} (h : g.On s e) :
    InInt8 e.int := by
  rcases hg with rfl | rfl <;> cases s
  · obtain ⟨h1, h2⟩ := sm120_act_int h; exact ⟨by omega, by omega⟩
  · obtain ⟨h1, h2⟩ := sm120_wt_int h; exact ⟨by omega, by omega⟩
  · obtain ⟨h1, h2⟩ := dc_act_int h; exact ⟨by omega, by omega⟩
  · obtain ⟨h1, h2⟩ := dc_wt_int h; exact ⟨by omega, by omega⟩

/-! ## The GEMM and the coin's rank-one terms -/

/-- **The centred product is the integer GEMM's exact product less the coin's two rank-one terms.** -/
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

/-- With every activation of row `i` centred by `c_x` and every weight of column `j` by `c_w`, the served model's
exact product is the scales times the centred product of the entries' integers. -/
theorem model_eq_centred (sa sw : ℕ → ℚ) (k : ℕ) (EA EW : ℕ → ℕ → Entry) (i j : ℕ) (cx cw : ℚ)
    (hA : ∀ l < k, (EA i l).offset = cx) (hW : ∀ l < k, (EW l j).offset = cw) :
    model sa sw k EA EW i j = sa i * sw j * centred cx cw k (ints EA) (ints EW) i j := by
  unfold model centred
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun l hl => ?_
  have hl' := Finset.mem_range.1 hl
  simp only [Entry.served, ints, hA l hl', hW l hl']
  ring

/-- **The served model's exact product is the scales times the int8 GEMM's exact product less the coin's two rank-one
terms**, one of row `i` alone and one of column `j` alone. -/
theorem model_eq (sa sw : ℕ → ℚ) (k : ℕ) (EA EW : ℕ → ℕ → Entry) (i j : ℕ) (cx cw : ℚ)
    (hA : ∀ l < k, (EA i l).offset = cx) (hW : ∀ l < k, (EW l j).offset = cw) :
    model sa sw k EA EW i j = sa i * sw j *
      ((mm k (ints EA) (ints EW) i j : ℚ) - rowTerm cx cw k (ints EA) i - colTerm cx k (ints EW) j) := by
  rw [model_eq_centred sa sw k EA EW i j cx cw hA hW, centred_eq]

/-- **The epilogue's correction is exact**: its integer is four times the centred product at `c_x = e_x/2`,
`c_w = e_w/2`. -/
theorem epilogue_eq (ex ew : ℤ) (k : ℕ) (A W : Mat) (i j : ℕ) :
    (epilogue ex ew k A W i j : ℚ) = 4 * centred ((ex : ℚ) / 2) ((ew : ℚ) / 2) k A W i j := by
  rw [centred_eq]
  simp only [epilogue, rowTerm, colTerm]
  push_cast
  ring

/-- The served model's exact product is the epilogue's integer times `s_a(i)·s_w(j)/4`. -/
theorem model_eq_epilogue (sa sw : ℕ → ℚ) (k : ℕ) (EA EW : ℕ → ℕ → Entry) (i j : ℕ) (ex ew : ℤ)
    (hA : ∀ l < k, (EA i l).offset = (ex : ℚ) / 2) (hW : ∀ l < k, (EW l j).offset = (ew : ℚ) / 2) :
    model sa sw k EA EW i j = sa i * sw j / 4 * (epilogue ex ew k (ints EA) (ints EW) i j : ℚ) := by
  rw [model_eq_centred sa sw k EA EW i j _ _ hA hW, epilogue_eq]
  ring

/-! ## int32 -/

theorem mul_bounds_g1 {a w : ℤ} (ha : -127 ≤ a ∧ a ≤ 127) (hw : -121 ≤ w ∧ w ≤ 121) :
    -15367 ≤ a * w ∧ a * w ≤ 15367 := by
  obtain ⟨ha1, ha2⟩ := ha; obtain ⟨hw1, hw2⟩ := hw
  constructor <;> nlinarith [mul_nonneg (sub_nonneg.2 ha1) (sub_nonneg.2 hw1),
    mul_nonneg (sub_nonneg.2 ha2) (sub_nonneg.2 hw2), mul_nonneg (sub_nonneg.2 ha1) (sub_nonneg.2 hw2),
    mul_nonneg (sub_nonneg.2 ha2) (sub_nonneg.2 hw1)]

/-- **Neither the s32 accumulator nor the int32 epilogue wraps** for `k ≤ 32,768`, activations in `[−127, 127]`,
weights in `[−121, 121]` and `e_x, e_w ∈ {0, 1}`: `|epilogue| ≤ 32,768·61,965 < 2^31`. -/
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

/-- **On a device's g1 grids the kernel's int32 epilogue, times `s_a(i)·s_w(j)/4`, is the served model's exact
product**, for `k ≤ 32,768`, when every activation of row `i` is centred by `e_x/2` and every weight of column `j` by
`e_w/2` (FP8-g1: `e_x = e_w = 1`; W4A8-g1 beside Dither: `e_x = 0`, `e_w = 1`). -/
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

/-- An FP8-g1 entry is centred by `½ = e/2` at `e = 1`, a Dither entry by `0 = e/2` at `e = 0`. -/
theorem offset_fp8 (R : ℤ) (b : Coin) : (Entry.fp8 R b).offset = ((1 : ℤ) : ℚ) / 2 := by
  simp [Entry.offset]

theorem offset_fp4 (m : ℕ) (q : ℤ) (b : Coin) : (Entry.fp4 m q b).offset = ((1 : ℤ) : ℚ) / 2 := by
  simp [Entry.offset]

theorem offset_int8 (R : ℤ) (b : Coin) : (Entry.int8 R b).offset = ((0 : ℤ) : ℚ) / 2 := by
  simp [Entry.offset]

end Pouw.FpG1
