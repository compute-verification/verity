module

public import Definitions.Registration.Physical

/-! A device's record limit: `B4`'s count of the trusted devices' joint record, taken device by device. -/

@[expose] public section

namespace ResourceAccounting

open Verity Finset

/-- **Each device's record is within its limit `K`**: across the strategy `S`, what device `y` of `cert` places in its
record cells (`record y`, `Actual.Run.record`) during `[a, b)`, as seen, takes at most `2 ^ K y` values. It is a limit on
the record over every run of the strategy, whatever the developer's hardware sends; a record of a stated format has it
at its size in bits. -/
def RecordsWithin {Ω L : Type*} (S : Ω → Actual.Run L) (cert : Finset L) (record : L → Set ℕ) (a b : ℕ)
    (K : L → ℕ) : Prop :=
  ∀ y ∈ cert, (Set.range fun r =>
    (S r).seen '' ↑(((S r).record {y} record).filter fun g => a ≤ (S r).A.t g ∧ (S r).A.t g < b)).encard ≤
      ((2 ^ K y : ℕ) : ℕ∞)

/-- **The challengers' term**: what the memory challengers `chal` that are trusted devices add to a leak limit's
exponent, each its record limit `c y` and one jitter share. It is `0` when no challenger is a trusted device. -/
def challengerTerm {L : Type*} (chal : Finset L) (c : L → ℕ) (jitter buckets : ℕ) : ℕ :=
  (∑ y ∈ chal, c y) + jitter * buckets * chal.card

end ResourceAccounting
