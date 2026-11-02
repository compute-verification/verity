import Definitions.Pouw.Assumptions.Dimension.H100Floor
import Definitions.Pouw.Dimension.Pointwise
import Definitions.Pouw.Dimension.Ssn

/-!
# What writing distinct words costs: the guarantees (trusted; draft semantics, pending review)

**Draft semantics, pending review**, except `DistinctWritesMH100Floor`. Each other statement reads definitions still
marked pending review: the served layouts (`Pouw.Dimension.Ssn`) or the chunk model (`Pouw.Dimension.Pointwise`).
`MH100w` (`Pouw.Dimension.H100`) is reviewed against the H200's measurements in `H100W.lean`. Each is proved under its
own name in `Pouw.SecurityProofs.Writes`, as `Pouw.SecurityProofs.Writes.DistinctWritesMH100Floor :
Pouw.Guarantees.Writes.DistinctWritesMH100Floor`; there is no single `Writes.G`.

* **What each model covers.** `MH100w` prices one straight-line program fixed before the coins, whose registers are
  functions of the coins and whose `move` is free. It charges a load from a computed address as `other`, which a device
  copying a whole chunk from a computed address doesn't pay, and it has no store to a computed address, which a device
  has and which a free `move` would misprice. Its bounds are about programs that do neither. The chunk model prices one
  run at a time, so the run may depend on the coins: every word write costs one, whatever produced it, and a whole-chunk
  copy is free at any address, read or written.
* `DistinctWritesMH100Floor`: under `Floor c`, `c` per target.
* `SsnServedDistinct`, `SsnServedCharged` and `SsnServedShift`: served sparse noise's committed words in the bit model
  on the full coin grid, for the 4-lane layout to 28,559 columns and the 5-lane layout to 371,292.
* `RunPairs` and `RunBall`: what one run must write to hold its target chunks at some point.
-/

namespace Pouw.Guarantees.Writes

open Finset Pouw.Dimension.H100 Pouw.Dimension.Ssn Pouw.Dimension.Pointwise

/-- **The distinct-writes bound at any floor** (`MH100w`): under `Floor c`, for every
set `S` of grid points and free values `F` containing the constants, a well-formed `MH100w` program on free inputs that
matches on `S` every target of a finite set `T`, pairwise distinct on `S` and none equal on `S` to a free value, costs
at least `c·|T|`. -/
def DistinctWritesMH100Floor : Prop := Pouw.Dimension.H100.DistinctWritesMH100Floor

/-- **Served sparse noise's words are distinct** (bit model, full coin grid; draft semantics, pending review): with `C`
coin words a row and `S ≥ 1` steps, for every data case `κ`, the 4-lane layout's committed words over `n ≤ 28,559`
columns (`S + 3 ≤ 32·C`) and the 5-lane layout's over `n ≤ 371,292` (`S + 4 ≤ 32·C`) are pairwise distinct, none is a
constant or a raw coin word, and there are `m·n·S` of them. -/
def SsnServedDistinct : Prop :=
  ∀ (m S C : ℕ) (κ : Fin m → ℕ → Fin S → ℤ) (n : ℕ), 0 < S →
    (∀ hC : S + 3 ≤ 32 * C, n ≤ 28559 →
      DistinctOn Set.univ (FreeB m C) (l4Words hC κ n) ∧ (l4Words hC κ n).card = m * (n * S)) ∧
    ∀ hC : S + 4 ≤ 32 * C, n ≤ 371292 →
      DistinctOn Set.univ (FreeB m C) (l5Words hC κ n) ∧ (l5Words hC κ n).card = m * (n * S)

/-- **Served sparse noise's words are charged** (`MH100w` at `h100Floor`, bit model, full coin grid; draft semantics,
pending review): for either layout in `SsnServedDistinct`'s range, a well-formed program on constants and raw coin words
whose registers match every committed word on the full coin grid costs at least `32·m·n·S`. -/
def SsnServedCharged : Prop :=
  ∀ (m S C : ℕ) (κ : Fin m → ℕ → Fin S → ℤ) (n : ℕ) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (Bits m C → ℚ))
    (P : List (Op100 (Bits m C))), 0 < S → (∀ f ∈ L, f ∈ FreeB m C) → (∀ o ∈ P, o.WFw h100Floor (FreeB m C)) →
    (∀ hC : S + 3 ≤ 32 * C, n ≤ 28559 → OutputsOn fpSem Set.univ P L (l4Words hC κ n) →
      32 * (m * (n * S)) ≤ cost100 h100Floor P) ∧
    ∀ hC : S + 4 ≤ 32 * C, n ≤ 371292 → OutputsOn fpSem Set.univ P L (l5Words hC κ n) →
      32 * (m * (n * S)) ≤ cost100 h100Floor P

/-- **Served sparse noise's words are shift-separated** (bit model, full coin grid; draft semantics, pending review):
for either layout in `SsnServedDistinct`'s range, no two committed words differ by a constant, and none is a constant
or a raw coin word plus a constant. -/
def SsnServedShift : Prop :=
  ∀ (m S C : ℕ) (κ : Fin m → ℕ → Fin S → ℤ) (n : ℕ), 0 < S →
    (∀ hC : S + 3 ≤ 32 * C, n ≤ 28559 → ShiftSeparatedOn Set.univ (FreeB m C) (l4Words hC κ n)) ∧
    ∀ hC : S + 4 ≤ 32 * C, n ≤ 371292 → ShiftSeparatedOn Set.univ (FreeB m C) (l5Words hC κ n)

open Classical in
/-- **A run writes at least its targets' new (position, value) pairs** (chunk model; draft model, pending review): for
every run from the memory `m₀`, whatever it branches on and wherever it copies, and every set `Y` of chunk values it
holds at some point, the (position, value) pairs of `Y` that no initial chunk holds at that position number at most
the run's word writes. -/
def RunPairs : Prop :=
  ∀ (A V : Type) [DecidableEq A] [DecidableEq V] [Inhabited V] (w : ℕ) (m₀ : A → Fin w → V)
    (P : List (Step A V w)) (Y : Finset (Fin w → V)), (∀ y ∈ Y, ∃ i a, run m₀ (P.take i) a = y) →
    ((Y.biUnion pairs).filter fun pv : Fin w × V => ∀ a, m₀ a pv.1 ≠ pv.2).card ≤ writes P

/-- **A run writes at least `Σ_y (r y + 1)`** (chunk model; draft model, pending review): for every run from the
memory `m₀` and every set `Y` of chunk values it holds at some point, when each `y` is further than `r y` in Hamming
distance from every initial chunk and any two are further apart than the sum of their radii. -/
def RunBall : Prop :=
  ∀ (A V : Type) [DecidableEq A] [DecidableEq V] [Inhabited V] (w : ℕ) (m₀ : A → Fin w → V)
    (P : List (Step A V w)) (Y : Finset (Fin w → V)) (r : (Fin w → V) → ℕ),
    (∀ y ∈ Y, ∃ i a, run m₀ (P.take i) a = y) → (∀ y ∈ Y, ∀ a, r y < hammingDist (m₀ a) y) →
    (∀ y ∈ Y, ∀ y' ∈ Y, y ≠ y' → r y + r y' < hammingDist y y') → ∑ y ∈ Y, (r y + 1) ≤ writes P

end Pouw.Guarantees.Writes
