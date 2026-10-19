import PouwWindow

/-!
# PoUW's window: what its Lean code (`PouwWindow`) does

The properties of the executable window (`verity/core/protocols/pouw/lean/window/`, the port of
`verity/core/protocols/pouw/window.py`), stated against its functions as they run. Each covers every input the
function takes: a function on Python's integers (`floorPoint?`, `kPrime`) either answers or refuses, and a property says
what an answer is, so its domain comes from the answer. That the function answers every input on its domain is
completeness, proved beside it. Each is a `Prop`, and `Properties.Proofs.Pouw` proves every one, on Lean core.

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

/-- What `floor_point(n)` answers is the largest point at most `n`. -/
def FloorPointGreatest : Prop :=
  ∀ bits (n : Int) g, floorPoint? bits n = .ok g →
    IsPoint bits g ∧ g ≤ n.toNat ∧ ∀ g', IsPoint bits g' → g' ≤ n.toNat → g' ≤ g

/-- What `pad(n)` answers is the least point at least `n`. -/
def PadLeast : Prop :=
  ∀ bits (n : Int) g, pad? bits n = .ok g → IsPoint bits g ∧ n.toNat ≤ g ∧ ∀ g', IsPoint bits g' → n.toNat ≤ g' → g ≤ g'

/-- `point` lists the grid in increasing order from 1, and `level` is its inverse on the grid. -/
def PointEnumerates : Prop :=
  ∀ bits, point bits 0 = 1 ∧ (∀ ℓ, point bits ℓ < point bits (ℓ + 1)) ∧ (∀ ℓ, IsPoint bits (point bits ℓ)) ∧
    (∀ ℓ, level bits (point bits ℓ) = ℓ) ∧ ∀ g, IsPoint bits g → point bits (level bits g) = g

/-- The work `credit(real, n)` answers, for a window of `n` tiles (its shape's) whose masked sum is `real` credited tiles,
registers the level of the largest point at most `real`: it is credited that point of its `n` tiles, never the filler's,
so fewer than `n`. -/
def CreditCredits : Prop :=
  ∀ bits (real n : Int) w, credit? bits real n = .ok w →
    w = Work.at bits w.level n.toNat ∧ w.credited = floorPoint bits real.toNat ∧ w.credited ≤ real.toNat ∧
      w.credited < w.n

/-- The grid's cost for a window `w` whose masked sum is `real` credited tiles: from `2^bits` up, the credit drops less
than a share `2^-bits` of them. The padding (empty slots, refused rows and the filler) is the shape's, and costs K′ but
no credit. -/
def CostBounded (bits real : Nat) (w : Work) : Prop :=
  2 ^ bits ≤ real → real * (2 ^ bits - 1) < w.credited * 2 ^ bits

/-- The work `credit(real, n)` answers costs what `CostBounded` says. -/
def CreditCost : Prop := ∀ bits (real n : Int) w, credit? bits real n = .ok w → CostBounded bits real.toNat w

/-- The served grid's cost for a window `w` whose masked sum is `real`: from 8 credited tiles up, the credit is more
than 7/8 of them. -/
def CostBounded3 (real : Nat) (w : Work) : Prop := 8 ≤ real → 7 * real < 8 * w.credited

/-- On the served grid, the work `credit(real, n)` answers costs what `CostBounded3` says. -/
def CreditCost3 : Prop := ∀ (real n : Int) w, credit? gridBits real n = .ok w → CostBounded3 real.toNat w

/-- `k` counts the levels a masked sum in `[lo, hi]` registers, which are consecutive from `lo`'s, when `lo ≤ hi`. -/
def LevelsCounted (bits lo hi : Nat) (k : Int) : Prop :=
  lo ≤ hi → 0 < k ∧
    ∀ ℓ, (∃ r, lo ≤ r ∧ r ≤ hi ∧ level bits (floorPoint bits r) = ℓ) ↔
      level bits (floorPoint bits lo) ≤ ℓ ∧ (ℓ : Int) < level bits (floorPoint bits lo) + k

/-- What `levels(lo, hi)` answers counts what `LevelsCounted` says. -/
def LevelsCount : Prop :=
  ∀ bits (lo hi : Int) k, levels? bits lo hi = .ok k → LevelsCounted bits lo.toNat hi.toNat k

/-- The served grid's sizes: 304 levels for 1 to 2⁴⁰ tiles, 161 for 2¹⁰ to 2³⁰; the window's one public output, its
level, so takes at most 9 bits, within a proof unit's 16. -/
def LevelsServed : Prop :=
  levels gridBits 1 (2 ^ 40) = 304 ∧ levels gridBits (2 ^ 10) (2 ^ 30) = 161 ∧ levels gridBits 1 (2 ^ 40) ≤ 2 ^ 9

/-- A full window of 8192 tiles, every one credited, and its filler registers level 87: credited 8192 of its 8193. -/
def Credit8192 : Prop := credit gridBits 8192 8193 = ⟨87, 8192, 8193⟩

/-! Python's grid functions on any integer answer only on their domain, and what they answer is the function on
naturals. -/

/-- `floor_point(n)` answers only for a tile count `n ≥ 1`, and answers `floorPoint` of it. -/
def FloorPointRefuses : Prop := ∀ bits (n : Int) g, floorPoint? bits n = .ok g → 1 ≤ n ∧ g = floorPoint bits n.toNat

/-- `pad(n)` answers only for a tile count `n ≥ 1`, and answers `pad` of it. -/
def PadRefuses : Prop := ∀ bits (n : Int) g, pad? bits n = .ok g → 1 ≤ n ∧ g = pad bits n.toNat

/-- `credit(real, n)` answers only for a masked sum `1 ≤ real < n`, and answers `credit` of it. -/
def CreditRefuses : Prop :=
  ∀ bits (real n : Int) w, credit? bits real n = .ok w → 1 ≤ real ∧ real < n ∧ w = credit bits real.toNat n.toNat

/-- `level(n)` answers only for a grid point `n`, and answers `level` of it. -/
def LevelRefuses : Prop :=
  ∀ bits (n : Int) ℓ, level? bits n = .ok ℓ → 1 ≤ n ∧ onGrid bits n.toNat = true ∧ ℓ = level bits n.toNat

/-- `point(ℓ)` answers only for a level `ℓ ≥ 0`, and answers `point` of it. -/
def PointRefuses : Prop := ∀ bits (ℓ : Int) g, point? bits ℓ = .ok g → 0 ≤ ℓ ∧ g = point bits ℓ.toNat

/-- `Work.at(ℓ, n)` answers only for a level `ℓ ≥ 0`, and answers `Work.at` of it. -/
def WorkAtRefuses : Prop :=
  ∀ bits (ℓ : Int) (n : Nat) w, Work.at? bits ℓ n = .ok w → 0 ≤ ℓ ∧ w = Work.at bits ℓ.toNat n

/-- `levels(lo, hi)` answers only for tile counts `lo, hi ≥ 1`, and answers `levels` of them. -/
def LevelsRefuses : Prop :=
  ∀ bits (lo hi : Int) k, levels? bits lo hi = .ok k → 1 ≤ lo ∧ 1 ≤ hi ∧ k = levels bits lo.toNat hi.toNat

/-! ## The draw (L3) -/

/-- `kPrime` answers only on its domain. -/
def KPrimeRefuses : Prop :=
  ∀ eps δ (credited n w : Int) k, kPrime eps δ credited n w = .ok k → KPrimeDomain eps δ credited n w

/-- What `kPrime` answers is the least `K ≥ 1` with `(1 − p)^K ≤ δ`, `p = ε·credited/(n·w)`, capped at `n`: exact in the
rationals, so no rounding moves it by one. -/
def KPrimeLeast : Prop :=
  ∀ eps δ (credited n w : Int) k, kPrime eps δ credited n w = .ok k →
    let q := 1 - missShare eps credited n w
    1 ≤ k ∧ k ≤ n.toNat ∧ (k < n.toNat → q ^ k ≤ δ) ∧ ∀ j, 1 ≤ j → j < k → δ < q ^ j

/-- The chance that `k` indices drawn uniformly without replacement from `[0, N)` all miss a set of `B` of them. -/
def missProb (N B k : Nat) : Rat :=
  ((List.range k).map fun i => ((N - B - i : Nat) : Rat) / ((N - i : Nat) : Rat)).foldl (· * ·) 1

/-- A wrong set of `B` of the window's `n` tiles that holds a share `ε` of its credited work (each tile does at most `w`,
so `ε·credited ≤ B·w`) is missed by all `k` drawn tiles with chance at most `δ`. -/
def MissBounded (eps δ : Rat) (credited n w : Int) (B k : Nat) : Prop :=
  B ≤ n.toNat → eps * credited ≤ B * w → missProb n.toNat B k ≤ δ

/-- L3: the `K′` `kPrime` answers bounds every wrong set's miss as `MissBounded` says. -/
def DrawMisses : Prop :=
  ∀ eps δ (credited n w : Int) (B k : Nat), kPrime eps δ credited n w = .ok k → MissBounded eps δ credited n w B k

/-- K′ at level ℓ of the served grid, at ε = 1/2, for FP8 tiles of 4096 (`sampling`'s law; 0 if refused), for the
largest full window registering ℓ: `point (ℓ + 1) − 1` tiles, every one credited, and the filler, so `N = point (ℓ + 1)`.
K′ grows with N at a fixed credit, so this is the most a full window at ℓ draws; a window with more padding draws more. -/
def servedK (δ : Rat) (ℓ : Nat) : Nat :=
  let w := Work.at gridBits ℓ (point gridBits (ℓ + 1))
  ((kPrime (1 / 2) δ (w.credited * 4096) w.n 4096).toOption).getD 0

/-- The served draw's sizes for full windows: 4 tiles for 9215 tiles and the filler (credited 8192 of 9216) at
δ = 1/10, and over every level for 1 to 2⁴⁰ tiles (0 to 303) at most 5 at δ = 1/10 and 48 at δ = 2⁻⁴⁰. -/
def KPrimeServed : Prop :=
  (kPrime (1 / 2) (1 / 10) (8192 * 4096) 9216 4096).toOption = some 4 ∧
    ((List.range 304).map (servedK (1 / 10))).foldl max 0 = 5 ∧
    ((List.range 304).map (servedK (1 / 2 ^ 40))).foldl max 0 = 48

/-- Exactness where it shows: level 6 (7 tiles credited of 8) at ε = 1, δ = 1/512 has `(1/8)^3 = δ` exactly, and K′ is
3; a binary64 `log δ / log(1 − p)` lands just above 3 and gives 4. -/
def KPrimeExactAtEquality : Prop := (kPrime 1 (1 / 512) 7 8 1).toOption = some 3

/-! ## What is public -/

/-- `PUBLIC` is the draft's six items, the window's shape and its fixed length, each of one of the four kinds a protocol
may make public, and a registration carries the ones that do not come from the outcome. -/
def PublicItems : Prop :=
  publicItems.map (·.item) = ["verdict", "work", "check", "sampling", "shape", "length", "drawn", "root_W"] ∧
    publicItems.all (fun x => ["output", "structure", "coins", "commitment"].contains x.kind) = true ∧
    registered = (publicItems.filter (·.source != "outcome")).map (·.item)

end Properties.Pouw.Window
