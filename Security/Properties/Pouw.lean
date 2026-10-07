import PouwWindow

/-!
# PoUW's window: what its Lean code (`PouwWindow`) does

The properties of the executable window (`verity/experimental/core/protocols/pouw/window/`, the port of
`verity/experimental/protocols/pouw/window.py`), stated against its functions as they run. Each covers every input the
function takes: a function on Python's integers (`floorPoint?`, `kPrime`) either answers or refuses, and the refusal is
exactly the inputs off its domain. Each is a `Prop`, and `Properties.Proofs.Pouw` proves
every one, on Lean core.

The numbers are the served ones: the 3-bit grid (`gridBits`), tiles of 4096 multiply-adds (an FP8 tile), ε = 1/2, and δ
from 1/10 to 2⁻⁴⁰, over tile counts up to 2⁴⁰.

What a property assumes is in its statement. The draw's (`DrawMisses`) is not: the miss probability it bounds is the
one of `K′` indices drawn uniformly without replacement from `[0, N)`, which is what the proof service's live draw is
under its named assumptions (the operating system's bytes are uniform, `UniformRandomBytes`, and the service draws a
uniform subset of them); nothing here models the draw.
-/

namespace Properties.Pouw.Window

open PouwWindow

/-! ## The grid -/

/-- A point of the `bits`-bit grid: a positive integer with at most `bits + 1` significant bits. -/
def IsPoint (bits n : Nat) : Prop := ∃ m e, n = m * 2 ^ e ∧ 0 < m ∧ m < 2 ^ (bits + 1)

/-- `onGrid` decides `IsPoint`. -/
def OnGridMeans : Prop := ∀ bits n, onGrid bits n = true ↔ IsPoint bits n

/-- `floorPoint n` is the largest point at most `n ≥ 1`. -/
def FloorPointGreatest : Prop :=
  ∀ bits n, 1 ≤ n →
    IsPoint bits (floorPoint bits n) ∧ floorPoint bits n ≤ n ∧ ∀ g, IsPoint bits g → g ≤ n → g ≤ floorPoint bits n

/-- `pad n` is the least point at least `n ≥ 1`. -/
def PadLeast : Prop :=
  ∀ bits n, 1 ≤ n → IsPoint bits (pad bits n) ∧ n ≤ pad bits n ∧ ∀ g, IsPoint bits g → n ≤ g → pad bits n ≤ g

/-- `point` lists the grid in increasing order from 1, and `level` is its inverse on the grid. -/
def PointEnumerates : Prop :=
  ∀ bits, point bits 0 = 1 ∧ (∀ ℓ, point bits ℓ < point bits (ℓ + 1)) ∧ (∀ ℓ, IsPoint bits (point bits ℓ)) ∧
    (∀ ℓ, level bits (point bits ℓ) = ℓ) ∧ ∀ g, IsPoint bits g → point bits (level bits g) = g

/-- A window of `real ≥ 1` real tiles registers the level of the largest point at most `real`: it is credited that
point and pads to the least point above `real`. -/
def BucketCredits : Prop :=
  ∀ bits real, 1 ≤ real →
    let w := bucket bits real
    w = Work.at bits w.level ∧ w.credited = floorPoint bits real ∧ w.credited ≤ real ∧ real < w.n ∧
      w.n = pad bits (real + 1)

/-- The grid's cost: from `2^bits` real tiles up, the credit drops less than a share `2^-bits` of the real tiles, and
the padded count is at most `1 + 2^-bits` times the credit. -/
def BucketCost : Prop :=
  ∀ bits real, 2 ^ bits ≤ real →
    real * (2 ^ bits - 1) < (bucket bits real).credited * 2 ^ bits ∧
      (bucket bits real).n * 2 ^ bits ≤ (bucket bits real).credited * (2 ^ bits + 1)

/-- On the served grid: from 8 real tiles up, the credit is more than 7/8 of them and the padded count at most 9/8 of the
credit. -/
def BucketCost3 : Prop :=
  ∀ real, 8 ≤ real →
    7 * real < 8 * (bucket gridBits real).credited ∧ 8 * (bucket gridBits real).n ≤ 9 * (bucket gridBits real).credited

/-- `levels lo hi` (for `1 ≤ lo ≤ hi`) counts the levels a tile count in `[lo, hi]` registers, which are consecutive
from `lo`'s. -/
def LevelsCount : Prop :=
  ∀ bits lo hi, 1 ≤ lo → lo ≤ hi →
    0 < levels bits lo hi ∧
      ∀ ℓ, (∃ r, lo ≤ r ∧ r ≤ hi ∧ (bucket bits r).level = ℓ) ↔
        (bucket bits lo).level ≤ ℓ ∧ (ℓ : Int) < (bucket bits lo).level + levels bits lo hi

/-- The served grid's sizes: 304 levels for 1 to 2⁴⁰ tiles, 161 for 2¹⁰ to 2³⁰; the window's one public output, its
level, so takes at most 9 bits, within a proof unit's 16. -/
def LevelsServed : Prop :=
  levels gridBits 1 (2 ^ 40) = 304 ∧ levels gridBits (2 ^ 10) (2 ^ 30) = 161 ∧ 304 ≤ 2 ^ 9

/-- A window of 8192 real tiles registers level 87: credited 8192, padded to 9216. -/
def Bucket8192 : Prop := bucket gridBits 8192 = ⟨87, 8192, 9216⟩

/-- Python's grid functions on any integer: each answers what the function on naturals does, and refuses exactly off
its domain (`n ≥ 1` for a tile count, a grid point for `level`, `ℓ ≥ 0` for a level; `levels` needs both ends). -/
def GridRefusals : Prop :=
  ∀ bits (n lo : Int),
    (floorPoint? bits n).isOk = decide (1 ≤ n) ∧ (pad? bits n).isOk = decide (1 ≤ n) ∧
    (bucket? bits n).isOk = decide (1 ≤ n) ∧ (level? bits n).isOk = (decide (1 ≤ n) && onGrid bits n.toNat) ∧
    (point? bits n).isOk = decide (0 ≤ n) ∧ (Work.at? bits n).isOk = decide (0 ≤ n) ∧
    (levels? bits lo n).isOk = decide (1 ≤ lo ∧ 1 ≤ n) ∧
    (1 ≤ n → floorPoint? bits n = .ok (floorPoint bits n.toNat) ∧ pad? bits n = .ok (pad bits n.toNat) ∧
      bucket? bits n = .ok (bucket bits n.toNat)) ∧
    (1 ≤ n → onGrid bits n.toNat = true → level? bits n = .ok (level bits n.toNat)) ∧
    (0 ≤ n → point? bits n = .ok (point bits n.toNat) ∧ Work.at? bits n = .ok (Work.at bits n.toNat)) ∧
    (1 ≤ lo → 1 ≤ n → levels? bits lo n = .ok (levels bits lo.toNat n.toNat))

/-! ## The draw (L3) -/

/-- `kPrime` answers exactly on its domain. -/
def KPrimeRefuses : Prop :=
  ∀ eps δ (credited n w : Int), (kPrime eps δ credited n w).isOk = decide (KPrimeDomain eps δ credited n w)

/-- On its domain, `kPrime` is the least `K ≥ 1` with `(1 − p)^K ≤ δ`, `p = ε·credited/(n·w)`, capped at `n`: exact in
the rationals, so no rounding moves it by one. -/
def KPrimeLeast : Prop :=
  ∀ eps δ (credited n w : Int), KPrimeDomain eps δ credited n w →
    let q := 1 - missShare eps credited n w
    ∃ k, kPrime eps δ credited n w = .ok k ∧ 1 ≤ k ∧ k ≤ n.toNat ∧ (k < n.toNat → q ^ k ≤ δ) ∧
      ∀ j, 1 ≤ j → j < k → δ < q ^ j

/-- The chance that `k` indices drawn uniformly without replacement from `[0, N)` all miss a set of `B` of them. -/
def missProb (N B k : Nat) : Rat :=
  ((List.range k).map fun i => ((N - B - i : Nat) : Rat) / ((N - i : Nat) : Rat)).foldl (· * ·) 1

/-- L3: a wrong set of `B` of the window's `n` tiles that holds a share `ε` of its credited work (each tile does at most
`w`, so `ε·credited ≤ B·w`) is missed by all `K′` drawn tiles with chance at most `δ`. -/
def DrawMisses : Prop :=
  ∀ eps δ (credited n w : Int) (B k : Nat), KPrimeDomain eps δ credited n w → kPrime eps δ credited n w = .ok k →
    B ≤ n.toNat → eps * credited ≤ B * w → missProb n.toNat B k ≤ δ

/-- K′ at level ℓ of the served grid, at ε = 1/2, for FP8 tiles of 4096 (`sampling`'s law; 0 if refused). -/
def servedK (δ : Rat) (ℓ : Nat) : Nat :=
  let w := Work.at gridBits ℓ
  ((kPrime (1 / 2) δ (w.credited * 4096) w.n 4096).toOption).getD 0

/-- The served draw's sizes: 4 tiles for the 8192-tile window at δ = 1/10, and over every level for 1 to 2⁴⁰ tiles
(0 to 303) at most 5 at δ = 1/10 and 48 at δ = 2⁻⁴⁰. -/
def KPrimeServed : Prop :=
  (kPrime (1 / 2) (1 / 10) (8192 * 4096) 9216 4096).toOption = some 4 ∧
    ((List.range 304).map (servedK (1 / 10))).foldl max 0 = 5 ∧
    ((List.range 304).map (servedK (1 / 2 ^ 40))).foldl max 0 = 48

/-- Exactness where it shows: level 6 (7 tiles credited of 8) at ε = 1, δ = 1/512 has `(1/8)^3 = δ` exactly, and K′ is
3; a binary64 `log δ / log(1 − p)` lands just above 3 and gives 4. -/
def KPrimeExactAtEquality : Prop := (kPrime 1 (1 / 512) 7 8 1).toOption = some 3

/-! ## The advice and the window statement (L2, L4) -/

/-- One call or filler entry as the window statement accepts it, with its range `[lo, hi)`: both sides present with
their kept rows ascending rows of the side, a non-empty range, which is exactly its kept tiles for a call and at most
them for the filler. -/
def EntryOK (e : Raw) (lo hi : Nat) (call : Bool) : Prop :=
  (∃ a b, e.a = some a ∧ e.b = some b ∧ a.ok = true ∧ b.ok = true) ∧ e.lo = some lo ∧ e.hi = some hi ∧ lo < hi ∧
    if call then hi - lo = e.tiles else hi - lo ≤ e.tiles

/-- The window statement, declared: `bound` entries, `calls ≥ 1` calls, then one filler, then empty entries; the calls'
and the filler's ranges tile `[0, work.n)` in order; and the level is the bucket of where the filler starts, which is
the calls' tile count. -/
def WindowOK (es : List Raw) (work : Work) (bound : Int) (bits : Nat) : Prop :=
  (es.length : Int) = bound ∧
    ∃ calls, 1 ≤ calls ∧ calls < es.length ∧
      (∀ i x, es[i]? = some x →
        x.kind = some (if i < calls then .call else if i = calls then .filler else .none)) ∧
      ∃ cut : Nat → Nat, cut 0 = 0 ∧ cut (calls + 1) = work.n ∧ bucket bits (cut calls) = work ∧
        ∀ i x, i ≤ calls → es[i]? = some x → EntryOK x (cut i) (cut (i + 1)) (decide (i < calls))

/-- `windowProblem` refuses exactly what `WindowOK` does not hold of. -/
def WindowAccepts : Prop := ∀ es work bound bits, windowProblem es work bound bits = none ↔ WindowOK es work bound bits

/-- An accepted window is credited the bucket of its calls' kept tiles, and nothing of the filler's. -/
def WindowCreditsTheCalls : Prop :=
  ∀ es work bound bits, windowProblem es work bound bits = none →
    ∃ calls, (es.take calls).all (·.kind == some .call) = true ∧ (es[calls]?.bind (·.kind)) = some .filler ∧
      work = bucket bits ((es.take calls).map Raw.tiles).sum

/-- On an accepted window, `tileAt` maps `[0, work.n)` one to one into the entries' kept tiles (`none` from `work.n`
up), and onto each call's: every tile a call keeps is drawable. -/
def TileAtCovers : Prop :=
  ∀ es work bound bits, windowProblem es work bound bits = none →
    (∀ t, (tileAt es t).isSome = decide (t < work.n)) ∧
    (∀ t t', (tileAt es t).isSome → tileAt es t = tileAt es t' → t = t') ∧
    (∀ t e i j, tileAt es t = some (e, i, j) →
      ∃ x a b, es[e]? = some x ∧ x.kind ≠ some .none ∧ x.a = some a ∧ x.b = some b ∧ i ∈ a.keep ∧ j ∈ b.keep) ∧
    ∀ e x a b, es[e]? = some x → x.kind = some .call → x.a = some a → x.b = some b →
      ∀ i ∈ a.keep, ∀ j ∈ b.keep, ∃ t, tileAt es t = some (e, i, j)

/-! ## What is public -/

/-- `PUBLIC` is the draft's six items, each of one of the four kinds a protocol may make public, and a registration
carries the ones that do not come from the outcome. -/
def PublicItems : Prop :=
  publicItems.map (·.item) = ["verdict", "work", "check", "sampling", "drawn", "root_W"] ∧
    publicItems.all (fun x => ["output", "structure", "coins", "commitment"].contains x.kind) = true ∧
    registered = (publicItems.filter (·.source != "outcome")).map (·.item)

end Properties.Pouw.Window
