import Mathlib.Tactic.Linarith
import Mathlib.Tactic.FieldSimp
import Mathlib.Tactic.Positivity
import Mathlib.Tactic.Ring
import Definitions.Pouw.PearlC.DeviceRev1
import Definitions.Pouw.PearlC.DevicePrices

/-!
# The cap circuit's integer tests are rev1's per-tile cap

`cap.py`'s `Pc8TileCap` decides rev1's `capOK` (`pearlCTilesDevRev1`) on three integer tests over 32-bit words: the
debit `L` against its largest passing value, `(N/g)·L + (G·fs·|C|/g)·elems` against a table over the passing rows'
count, and the count's equality with each row of the table. This file restates the scheme's integers, the constants
and the tests as `cap.py` writes them, and proves the tests decide the cap.

* `IntPrices`, `creditInt`: `pearl_c_device.Prices` and `PearlC(forming="v1").credit_of(Shape(m, k, n))`, so
  `creditDevRev1 = creditInt` (`creditDevRev1_eq`).
* `capConstants`, `CapDomain`, `CapFits`: `_cap_constants` (with `_promoted`'s domain), and its refusals.
* `capTests`: the tests on integers; `capTests_iff`: they hold iff `RHO·(N·L + elem·elems) ≤ |R|·|C|·P`, and
  `cap_iff_int`: iff the debit is at most `ρ·creditDevRev1·tileShare` (ρ = 1/RHO).
* `addW`, `mulW`, `leW`, `boundW`, `capTestsW`: the same tests on 32-bit words (`I32Add`, `_mul`, `I32Le`, the
  `I32Eq` selection of the bound); `capTestsW_eq`: under `_cap_constants`' bounds they are `capTests`, since no sum
  they compute leaves 31 bits (`capSums_fit`).
-/

namespace Pouw.PearlC.Cap

open Pouw.PearlC

/-- A device's prices as the scheme's integers (`pearl_c_device.Prices`): `fadd`, `bf16_mac`, `forming_credited` and
`forming_a_only`. -/
structure IntPrices where
  fadd : ℕ
  bf16 : ℕ
  fs : ℕ
  qa : ℕ

/-- The Lean record's prices. -/
def IntPrices.toPrices (p : IntPrices) : Prices := ⟨p.fadd, p.bf16, ⟨p.fs, p.qa⟩⟩

/-- `SM120_PRICES`. -/
def IntPrices.sm120 : IntPrices := ⟨8, 2, 40, 1⟩

/-- `H100_PRICES`. -/
def IntPrices.h100 : IntPrices := ⟨32, 2, 64, 4⟩

theorem IntPrices.sm120_toPrices : IntPrices.sm120.toPrices = Prices.sm120 := by
  simp [IntPrices.toPrices, IntPrices.sm120, Prices.sm120]

theorem IntPrices.h100_toPrices : IntPrices.h100.toPrices = Prices.h100 := by
  simp [IntPrices.toPrices, IntPrices.h100, Prices.h100, Costs.adopted]

/-- `PearlC(forming="v1", device).credit_of(Shape(m, k, n))` at promotion period `g`: `wref` (the chain with its
promotion adds less the first of every word, `A′·F_B` at `R = 32`, the peel at `PEEL_K = 64` and the forming) less `qa`
per element of A. -/
def creditInt (p : IntPrices) (g m k n : ℕ) : ℕ :=
  let chainUnits := m * n * (k + p.fadd * (k / (32 * g)))
  let firstAdds := p.fadd * m * n
  chainUnits - firstAdds + m * k * 32 + p.bf16 * 64 * m * n + (p.fs + p.qa) * m * k - p.qa * m * k

/-- `_cap_constants`' outputs: `atom`, `zero`, `n/g`, `elem/g`, `L`'s largest passing value and the table over `|R|`. -/
structure CapConstants where
  atom : ℕ
  zero : ℕ
  nG : ℕ
  elemG : ℕ
  lMax : ℕ
  table : ℕ → ℕ

/-- **`_cap_constants(k, tm, tn, n, rho, dev)`** at the device's `g` and prices `p`: `big_p = g·credit_of(Shape(1, k,
n))`, and `table[rows] = int(Fraction(rows·tn·big_p, rho·gg))` (`Nat` division is `int` of a nonnegative fraction). -/
def capConstants (p : IntPrices) (g k tm tn n rho : ℕ) : CapConstants :=
  let bigP := g * creditInt p g 1 k n
  let atom := 32 * g + p.fadd
  let zeroPromo := g * p.fadd
  let elem := g * p.fs * tn
  let gg := Nat.gcd n elem
  let table := fun rows => rows * tn * bigP / (rho * gg)
  let lMax := table tm / (n / gg)
  ⟨atom, zeroPromo, n / gg, elem / gg, lMax, table⟩

/-- `_promoted`'s and `_cap_constants`' domain: a promoted device (`g ≥ 1`), whole promotion windows of `32·g`, at least
two of them, `0 < tn ≤ n` with `tn ∣ n`, `tm > 0` and `rho > 0`. -/
structure CapDomain (g k tm tn n rho : ℕ) : Prop where
  g_pos : 0 < g
  window : 32 * g ∣ k
  two : 2 ≤ k / (32 * g)
  tn_pos : 0 < tn
  tn_le : tn ≤ n
  tn_dvd : tn ∣ n
  tm_pos : 0 < tm
  rho_pos : 0 < rho

instance instDecidableCapDomain (g k tm tn n rho : ℕ) : Decidable (CapDomain g k tm tn n rho) :=
  decidable_of_iff (0 < g ∧ 32 * g ∣ k ∧ 2 ≤ k / (32 * g) ∧ 0 < tn ∧ tn ≤ n ∧ tn ∣ n ∧ 0 < tm ∧ 0 < rho)
    ⟨fun ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ => ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩,
     fun ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩ => ⟨h1, h2, h3, h4, h5, h6, h7, h8⟩⟩

/-- `_cap_constants`' refusals, negated: none of its three sums passes `2^31 − 1`. -/
def CapFits (p : IntPrices) (g k tm tn n rho : ℕ) : Prop :=
  let c := capConstants p g k tm tn n rho
  let words := tm * tn * (k / 32)
  c.atom * words + c.zero * tm * tn * (k / (32 * g)) ≤ 2 ^ 31 - 1 ∧
    c.nG * c.lMax + c.elemG * tm * k ≤ 2 ^ 31 - 1 ∧ c.table tm ≤ 2 ^ 31 - 1

instance instDecidableCapFits (p : IntPrices) (g k tm tn n rho : ℕ) : Decidable (CapFits p g k tm tn n rho) := by
  unfold CapFits; infer_instance

/-- **The cap's tests on integers**: `debit = atom·atoms + zero·zeros` at most `l_max`, and
`lhs = (n/g)·debit + (elem/g)·elems` at most the table at the passing rows' count. -/
def capTests (c : CapConstants) (atoms zeros elems rows : ℕ) : Bool :=
  let debit := c.atom * atoms + c.zero * zeros
  let lhs := c.nG * debit + c.elemG * elems
  decide (debit ≤ c.lMax) && decide (lhs ≤ c.table rows)

/-! ## 32-bit words -/

/-- `I32Add`: the sum mod `2^32`. -/
def addW (a b : ℕ) : ℕ := (a + b) % 2 ^ 32

/-- **`_mul`**: `c·x mod 2^32` for a constant `c`, double and add from the top bit (the recursion on `c / 2` runs the
higher bits first). -/
def mulW (c x : ℕ) : ℕ :=
  if c = 0 then 0 else
    let h := mulW (c / 2) x
    let d := addW h h
    if c % 2 = 1 then addW d x else d
decreasing_by omega

/-- A word as a signed 32-bit integer. -/
def toInt32 (w : ℕ) : ℤ := if w < 2 ^ 31 then (w : ℤ) else (w : ℤ) - 2 ^ 32

/-- `I32Le`: signed comparison of two words. -/
def leW (a b : ℕ) : Bool := decide (toInt32 a ≤ toInt32 b)

/-- The bound: `table[0]`, then for `r = 1..tm` `SelectI32(I32Eq(rows, r), table[r], bound)`. -/
def boundW (c : CapConstants) (tm rows : ℕ) : ℕ :=
  (List.range' 1 tm).foldl (fun b r => if rows = r then c.table r else b) (c.table 0)

/-- **The cap's tests on words** (`_tile_cap_body`): `debit = atom·atoms + zero·zeros` and
`lhs = (n/g)·debit + (elem/g)·elems` by `_mul` and `I32Add`, then `I32Le(debit, l_max) ∧ I32Le(lhs, bound)`. -/
def capTestsW (c : CapConstants) (tm atoms zeros elems rows : ℕ) : Bool :=
  let debit := addW (mulW c.atom atoms) (mulW c.zero zeros)
  let lhs := addW (mulW c.nG debit) (mulW c.elemG elems)
  leW debit c.lMax && leW lhs (boundW c tm rows)

theorem mulW_eq (c x : ℕ) : mulW c x = c * x % 2 ^ 32 := by
  induction c using Nat.strong_induction_on with
  | _ c ih =>
    rw [mulW]
    by_cases h0 : c = 0
    · simp [h0]
    · simp only [h0, ↓reduceIte, addW]
      rw [ih (c / 2) (by omega)]
      have hc : c = 2 * (c / 2) + c % 2 := (Nat.div_add_mod c 2).symm
      have e : c * x = 2 * (c / 2 * x) + c % 2 * x := by
        rw [show c * x = (2 * (c / 2) + c % 2) * x by rw [← hc]]; ring
      split_ifs with h1
      · rw [h1, one_mul] at e
        omega
      · rw [show c % 2 = 0 by omega, zero_mul] at e
        omega

theorem toInt32_of_lt {w : ℕ} (h : w < 2 ^ 31) : toInt32 w = w := by
  unfold toInt32; split_ifs; rfl

theorem leW_of_lt {a b : ℕ} (ha : a < 2 ^ 31) (hb : b < 2 ^ 31) : leW a b = decide (a ≤ b) := by
  simp [leW, toInt32_of_lt ha, toInt32_of_lt hb]

theorem foldl_select (c : CapConstants) (rows : ℕ) : ∀ (len s b : ℕ),
    (List.range' s len).foldl (fun b r => if rows = r then c.table r else b) b =
      if s ≤ rows ∧ rows < s + len then c.table rows else b
  | 0, s, b => by simp
  | len + 1, s, b => by
    rw [List.range'_succ, List.foldl_cons, foldl_select c rows len (s + 1)]
    split_ifs <;> first | rfl | (subst_vars; rfl) | (exfalso; omega)

theorem boundW_eq (c : CapConstants) {tm rows : ℕ} (h : rows ≤ tm) : boundW c tm rows = c.table rows := by
  unfold boundW
  rw [foldl_select]
  split_ifs with h1
  · rfl
  · have : rows = 0 := by omega
    rw [this]

/-! ## The tests decide the cap -/

/-- **The integer tests are the scaled cap**: on the domain and with `rows ≤ tm`, `capTests` holds iff
`RHO·(N·L + G·fs·tn·elems) ≤ rows·tn·P`, with `L = (32·G + add)·atoms + G·add·zeros` and `P = G·credit_of(1, k, N)`.
The first test follows from the second (the table grows with `rows`); it is there to bound `L` on words. -/
theorem capTests_iff (p : IntPrices) {g k tm tn n rho : ℕ} (hD : CapDomain g k tm tn n rho)
    {atoms zeros elems rows : ℕ} (hr : rows ≤ tm) :
    capTests (capConstants p g k tm tn n rho) atoms zeros elems rows = true ↔
      rho * (n * ((32 * g + p.fadd) * atoms + g * p.fadd * zeros) + g * p.fs * tn * elems) ≤
        rows * tn * (g * creditInt p g 1 k n) := by
  have hn : 0 < n := lt_of_lt_of_le hD.tn_pos hD.tn_le
  have hgg : 0 < Nat.gcd n (g * p.fs * tn) := Nat.gcd_pos_of_pos_left _ hn
  obtain ⟨a, ha⟩ := Nat.gcd_dvd_left n (g * p.fs * tn)
  obtain ⟨b, hb⟩ := Nat.gcd_dvd_right n (g * p.fs * tn)
  simp only [capTests, Bool.and_eq_true, decide_eq_true_eq]
  dsimp only [capConstants]
  generalize g * creditInt p g 1 k n = P
  generalize (32 * g + p.fadd) * atoms + g * p.fadd * zeros = L
  generalize Nat.gcd n (g * p.fs * tn) = gg at hgg ha hb ⊢
  have hna : n / gg = a := by rw [ha, Nat.mul_div_cancel_left a hgg]
  have heb : g * p.fs * tn / gg = b := by rw [hb, Nat.mul_div_cancel_left b hgg]
  have hapos : 0 < a := Nat.pos_of_ne_zero (by rintro rfl; omega)
  have hrg : 0 < rho * gg := Nat.mul_pos hD.rho_pos hgg
  rw [hna, heb, ha, hb, Nat.le_div_iff_mul_le hrg,
    show rho * (gg * a * L + gg * b * elems) = (a * L + b * elems) * (rho * gg) by ring]
  constructor
  · exact fun h => h.2
  · intro h
    refine ⟨?_, h⟩
    rw [Nat.le_div_iff_mul_le hapos, Nat.le_div_iff_mul_le hrg]
    calc L * a * (rho * gg) ≤ (a * L + b * elems) * (rho * gg) := by
          apply Nat.mul_le_mul_right; rw [mul_comm]; exact Nat.le_add_right _ _
      _ ≤ rows * tn * P := h
      _ ≤ tm * tn * P := by gcongr

/-- `credit_of` is `creditDevRev1`: on whole promotion windows of a promoted record at prices `p`, rev1's credit of a
unit is `m` times `credit_of(Shape(1, k, n))`. -/
theorem creditDevRev1_eq (p : IntPrices) {d : PearlCDevice} {g : ℕ} (hG : d.G = g) (hg : 0 < g)
    (hp : d.prices = p.toPrices) (sh : Shape) (hk : 32 * g ∣ sh.k) (hk1 : 32 * g ≤ sh.k) :
    creditDevRev1 d sh = (sh.m : ℚ) * (creditInt p g 1 sh.k sh.n : ℚ) := by
  obtain ⟨q, hq⟩ := hk
  have hq' : sh.k / (32 * g) = q := by rw [hq, Nat.mul_div_cancel_left q (by omega)]
  have hq1 : 1 ≤ q := by
    rcases Nat.eq_zero_or_pos q with h | h
    · rw [h, mul_zero] at hq; omega
    · exact h
  have h1 : p.fadd * 1 * sh.n ≤ 1 * sh.n * (sh.k + p.fadd * q) := by
    have : p.fadd ≤ sh.k + p.fadd * q := by nlinarith
    calc p.fadd * 1 * sh.n = sh.n * p.fadd := by ring
      _ ≤ sh.n * (sh.k + p.fadd * q) := Nat.mul_le_mul_left _ this
      _ = 1 * sh.n * (sh.k + p.fadd * q) := by ring
  have h2 : p.qa * 1 * sh.k ≤ 1 * sh.n * (sh.k + p.fadd * q) - p.fadd * 1 * sh.n + 1 * sh.k * 32 +
      p.bf16 * 64 * 1 * sh.n + (p.fs + p.qa) * 1 * sh.k := by
    have : p.qa * 1 * sh.k ≤ (p.fs + p.qa) * 1 * sh.k := by gcongr; omega
    omega
  simp only [creditDevRev1, creditDev, firstAddDev, creditInt, hq', hG, hp, IntPrices.toPrices, Params.pi,
    Nat.ne_of_gt hg, ↓reduceIte]
  rw [Nat.cast_sub h2]
  push_cast [Nat.cast_sub h1]
  rw [hq]
  push_cast
  field_simp
  ring

/-- **rev1's per-tile cap is the scaled integer inequality**: at `ρ = 1/RHO`, the debit
`debitDev ⟨atoms, 0⟩ + fs·(|C|/N)·elems + add·zeros` is at most `ρ·creditDevRev1·tileShare` iff
`RHO·(N·L + G·fs·|C|·elems) ≤ |R|·|C|·P` (`m` cancels). -/
theorem cap_iff_int (p : IntPrices) {d : PearlCDevice} {g rho : ℕ} (hG : d.G = g) (hg : 0 < g)
    (hp : d.prices = p.toPrices) (sh : Shape) (hk : 32 * g ∣ sh.k) (hk1 : 32 * g ≤ sh.k) (hm : 0 < sh.m)
    (hn : 0 < sh.n) (hrho : 0 < rho) (rows cols : Finset ℕ) (atoms zeros elems : ℕ) :
    debitDev d ⟨atoms, 0⟩ + (d.prices.costs.fs : ℚ) * (cols.card / sh.n) * elems + d.prices.add * zeros ≤
        1 / (rho : ℚ) * creditDevRev1 d sh * tileShare sh rows cols ↔
      rho * (sh.n * ((32 * g + p.fadd) * atoms + g * p.fadd * zeros) + g * p.fs * cols.card * elems) ≤
        rows.card * cols.card * (g * creditInt p g 1 sh.k sh.n) := by
  rw [creditDevRev1_eq p hG hg hp sh hk hk1, ← Nat.cast_le (α := ℚ)]
  have hpos : (0 : ℚ) < rho * g * sh.n := by positivity
  rw [← mul_le_mul_iff_of_pos_right hpos]
  have hm' : (sh.m : ℚ) ≠ 0 := by positivity
  have hn' : (sh.n : ℚ) ≠ 0 := by positivity
  have hg' : (g : ℚ) ≠ 0 := by positivity
  have hr' : (rho : ℚ) ≠ 0 := by positivity
  simp only [debitDev, tileShare, hG, hp, IntPrices.toPrices, Nat.ne_of_gt hg, ↓reduceIte]
  push_cast
  apply le_iff_le_of_cmp_eq_cmp
  congr 1
  · field_simp
    ring
  · field_simp

/-! ## On 32-bit words -/

/-- The word tests are the integer tests when the debit, `l_max`, the bound and (where the first test passes) the left
side stay below `2^31`. -/
theorem capTestsW_eq_of (c : CapConstants) {tm atoms zeros elems rows : ℕ} (hr : rows ≤ tm)
    (h1 : c.atom * atoms + c.zero * zeros < 2 ^ 31) (h2 : c.lMax < 2 ^ 31) (h3 : c.table rows < 2 ^ 31)
    (h4 : c.atom * atoms + c.zero * zeros ≤ c.lMax →
      c.nG * (c.atom * atoms + c.zero * zeros) + c.elemG * elems < 2 ^ 31) :
    capTestsW c tm atoms zeros elems rows = capTests c atoms zeros elems rows := by
  simp only [capTestsW, capTests, mulW_eq, addW]
  generalize hd : c.atom * atoms + c.zero * zeros = debit at h1 h4
  have ha : c.atom * atoms < 2 ^ 32 := by omega
  have hz : c.zero * zeros < 2 ^ 32 := by omega
  rw [Nat.mod_eq_of_lt ha, Nat.mod_eq_of_lt hz, hd, Nat.mod_eq_of_lt (by omega), leW_of_lt h1 h2]
  by_cases h : debit ≤ c.lMax
  · have h5 := h4 h
    have hnd : c.nG * debit < 2 ^ 32 := by omega
    have hne : c.elemG * elems < 2 ^ 32 := by omega
    rw [Nat.mod_eq_of_lt hnd, Nat.mod_eq_of_lt hne, Nat.mod_eq_of_lt (by omega), boundW_eq c hr,
      leW_of_lt h5 h3]
  · simp [h]

/-- **No sum the cap computes leaves 31 bits** under `_cap_constants`' bounds, with the counts a tile of `tm` rows and
`tn` columns can reach (`k/32` flags and `k/(32·G)` promotions per word, `k` elements per row): the debit, `l_max`, the
bound at any `rows ≤ tm`, and the left side wherever the first test passes. -/
theorem capSums_fit (p : IntPrices) {g k tm tn n rho : ℕ} (hF : CapFits p g k tm tn n rho)
    {atoms zeros elems rows : ℕ} (ha : atoms ≤ tm * tn * (k / 32)) (hz : zeros ≤ tm * tn * (k / (32 * g)))
    (he : elems ≤ tm * k) (hr : rows ≤ tm) :
    let c := capConstants p g k tm tn n rho
    c.atom * atoms + c.zero * zeros < 2 ^ 31 ∧ c.lMax < 2 ^ 31 ∧ c.table rows < 2 ^ 31 ∧
      (c.atom * atoms + c.zero * zeros ≤ c.lMax → c.nG * (c.atom * atoms + c.zero * zeros) + c.elemG * elems < 2 ^ 31) := by
  intro c
  obtain ⟨f1, f2, f3⟩ := hF
  have hc : capConstants p g k tm tn n rho = c := rfl
  rw [hc] at f1 f2 f3
  have hd : c.atom * atoms + c.zero * zeros ≤ c.atom * (tm * tn * (k / 32)) + c.zero * tm * tn * (k / (32 * g)) := by
    rw [mul_assoc c.zero, mul_assoc c.zero]
    exact Nat.add_le_add (Nat.mul_le_mul_left _ ha) (Nat.mul_le_mul_left _ hz)
  have ht : c.table rows ≤ c.table tm := by
    show rows * tn * _ / _ ≤ tm * tn * _ / _
    exact Nat.div_le_div_right (Nat.mul_le_mul_right _ (Nat.mul_le_mul_right _ hr))
  have hl : c.lMax ≤ c.table tm := Nat.div_le_self _ _
  refine ⟨by omega, by omega, by omega, fun h => ?_⟩
  have : c.nG * (c.atom * atoms + c.zero * zeros) + c.elemG * elems ≤ c.nG * c.lMax + c.elemG * tm * k := by
    rw [mul_assoc c.elemG]
    exact Nat.add_le_add (Nat.mul_le_mul_left _ h) (Nat.mul_le_mul_left _ he)
  omega

/-- **The circuit's tests decide the cap**: on `_cap_constants`' domain and within its bounds, with the counts a tile can
reach, the word tests (`capTestsW`) hold iff `RHO·(N·L + G·fs·tn·elems) ≤ rows·tn·P`. -/
theorem capTestsW_iff (p : IntPrices) {g k tm tn n rho : ℕ} (hD : CapDomain g k tm tn n rho)
    (hF : CapFits p g k tm tn n rho) {atoms zeros elems rows : ℕ} (ha : atoms ≤ tm * tn * (k / 32))
    (hz : zeros ≤ tm * tn * (k / (32 * g))) (he : elems ≤ tm * k) (hr : rows ≤ tm) :
    capTestsW (capConstants p g k tm tn n rho) tm atoms zeros elems rows = true ↔
      rho * (n * ((32 * g + p.fadd) * atoms + g * p.fadd * zeros) + g * p.fs * tn * elems) ≤
        rows * tn * (g * creditInt p g 1 k n) := by
  obtain ⟨h1, h2, h3, h4⟩ := capSums_fit p hF ha hz he hr
  rw [capTestsW_eq_of _ hr h1 h2 h3 h4, capTests_iff p hD hr]

end Pouw.PearlC.Cap
