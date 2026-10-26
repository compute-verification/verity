import Definitions.NetworkCertifier.AdviceV1
import Mathlib.Basic.Real.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

/-!
# Advice v1's charge with a separate count code

A window that sends counts sends its decision stream and its counts `κ : C` (the four per-window counts, in whatever
form a code takes). Here its charge is the decisions' charge (`window`, from the carry) plus a separate count code's
length for `κ`. The count code is any code whose lengths meet Kraft's inequality (`CountCode`), and every prefix-free
code's lengths do (`CountCode.ofPrefixFree`, in `Proofs.NetworkCertifier.AdviceV1`). It is the same in every window and
carries nothing from window to window: the carry is the decisions'. The decisions' part codes the book's `zeros` as
before; under a book whose `zeros` is empty, it is the decisions' charge without them.

This is not how advice v1 sends counts: `code_v1` codes them inside the arithmetic coder, in the decisions' segment
(`ChargeCount_v2`'s window), which a separate code does not model. So the bounds with this charge are retired lemmas
(`Proofs.NetworkCertifier.Lemmas`), and the whole-window charge (`Definitions.NetworkCertifier.WindowCharge`) is the
one the guarantees count.
-/

namespace NetTiming.AdviceV1

/-- **A code for a window's counts**: `len κ`, the bits it charges the counts `κ`, or `none` where it refuses them.
Its lengths meet Kraft's inequality: over any finite set of counts, the weights `2^(−len κ)` of those it doesn't
refuse sum to at most 1. -/
structure CountCode (C : Type) where
  len : C → Option ℕ
  kraft : ∀ A : Finset C, ∑ κ ∈ A, ((len κ).map fun l => (2⁻¹ : ℝ) ^ l).getD 0 ≤ 1

variable {C : Type}

/-- **One window's charge with its counts**: of a window of `N` slots whose slots that ask decide `x.1` and which
sends the counts `x.2`, the decisions' charge (`window`) plus the count code's length for `x.2`, and the next
window's carry, the decisions'. -/
def windowCounted (bk : Book) (N : ℕ) (cc : CountCode C) (cy : Carry) (x : List Bool × C) : Option (ℕ × Carry) :=
  (window bk N cy x.1).bind fun p => (cc.len x.2).map fun l => (p.1 + l, p.2)

/-- **A run's charges with counts**: each window's, the carry passed along. -/
def sessionCounted (bk : Book) (N : ℕ) (cc : CountCode C) : Carry → List (List Bool × C) → Option (List ℕ)
  | _, [] => some []
  | cy, x :: ws => (windowCounted bk N cc cy x).bind fun p => (sessionCounted bk N cc p.2 ws).map fun ls => p.1 :: ls

/-- **The run's charges per window with counts**, from the book's priors and run 0. -/
def chargesCounted (bk : Book) (N : ℕ) (cc : CountCode C) {W : ℕ} (a : Fin W → List Bool × C) : Option (List ℕ) :=
  sessionCounted bk N cc ⟨bk.priors, 0⟩ (List.ofFn a)

/-- **The advice with counts**: each of `W` windows' decision stream and counts, no window refused. -/
def StreamsCounted (bk : Book) (N : ℕ) (cc : CountCode C) (W : ℕ) : Type :=
  {a : Fin W → List Bool × C // (chargesCounted bk N cc a).isSome}

/-- **The charge with counts** (I1's `bits(advice, w)`): window `w`'s. -/
def bitsCounted (bk : Book) (N : ℕ) (cc : CountCode C) {W : ℕ} (a : StreamsCounted bk N cc W) (w : Fin W) : ℕ :=
  ((chargesCounted bk N cc a.1).getD []).getD w 0

end NetTiming.AdviceV1
