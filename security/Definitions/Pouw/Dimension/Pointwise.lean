module

public import Mathlib.InformationTheory.Hamming
public import Mathlib.Algebra.BigOperators.Group.Finset.Basic

/-!
# The chunk model: what one run at fixed coins writes (trusted definitions)

**Draft model, pending review.** `MH100w` (`Pouw.Dimension.H100`) prices one straight-line program fixed before the
coins, with registers as functions of the coins. A prover that holds the coins while it computes runs a different
program at every coin value, so this model prices one run at a time.

* **Memory** is slots `A` of `w`-word chunks, starting at `m₀`: the salt-free tables and the coin words.
* **A run** is any list of steps (`Step`). `write a p v` sets word `p` of slot `a` to `v`. It is the only priced step,
  and stands for whatever produced `v`: an arithmetic op, a load or a per-word copy. `copy s d` copies slot `s` to slot
  `d` whole, free at any address. A run that branches or computes addresses is just another list.
* `run m₀ P` is the memory after `P`, `writes P` its word writes, and `pairs x` a chunk's (position, value) pairs.
* `Held m₀ P Y`: the run holds each chunk value of `Y` at some point. `Far m₀ Y r` and `Apart Y r`: each `y` is further
  than `r y` in Hamming distance from every initial chunk, and any two are further apart than the sum of their radii.
-/

@[expose] public section

namespace Pouw.Dimension.Pointwise

open Finset

variable {A V : Type} {w : ℕ}

/-- One step of a run: a priced word write, or a free whole-chunk copy. -/
inductive Step (A V : Type) (w : ℕ)
  | write (a : A) (p : Fin w) (v : V)
  | copy (s d : A)

variable [DecidableEq A]

/-- A step's effect on memory. -/
def Step.apply : Step A V w → (A → Fin w → V) → A → Fin w → V
  | .write a p v, m => Function.update m a (Function.update (m a) p v)
  | .copy s d, m => Function.update m d (m s)

/-- A step's price in word writes. -/
def Step.cost : Step A V w → ℕ
  | .write .. => 1
  | .copy .. => 0

/-- The memory after a run. -/
def run (m₀ : A → Fin w → V) (P : List (Step A V w)) : A → Fin w → V :=
  P.foldl (fun m st => st.apply m) m₀

/-- A run's word writes. -/
def writes (P : List (Step A V w)) : ℕ := (P.map Step.cost).sum

/-- A chunk's (position, value) pairs. -/
def pairs [DecidableEq V] (x : Fin w → V) : Finset (Fin w × V) := univ.image fun p => (p, x p)

/-- The run holds each chunk value of `Y` at some point: some slot holds it after some prefix of `P`. -/
def Held (m₀ : A → Fin w → V) (P : List (Step A V w)) (Y : Finset (Fin w → V)) : Prop :=
  ∀ y ∈ Y, ∃ i a, run m₀ (P.take i) a = y

/-- Each chunk value `y` of `Y` is further than `r y` in Hamming distance from every initial chunk. -/
def Far [DecidableEq V] (m₀ : A → Fin w → V) (Y : Finset (Fin w → V)) (r : (Fin w → V) → ℕ) : Prop :=
  ∀ y ∈ Y, ∀ a, r y < hammingDist (m₀ a) y

/-- Any two chunk values of `Y` are further apart in Hamming distance than the sum of their radii. -/
def Apart [DecidableEq V] (Y : Finset (Fin w → V)) (r : (Fin w → V) → ℕ) : Prop :=
  ∀ y ∈ Y, ∀ y' ∈ Y, y ≠ y' → r y + r y' < hammingDist y y'

end Pouw.Dimension.Pointwise
