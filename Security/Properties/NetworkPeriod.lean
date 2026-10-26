import NetworkCheck.Period
import Mathlib.Algebra.BigOperators.Fin

/-!
# The network period check's bound: what its rows form's `K` counts (`NetworkCheck.Period`)

On acceptance in the rows form, `NetworkCheck.Period.check` publishes `rowsTerms`'s `K`, the sum of a ceiling of `log2`
of each factor of `NetTiming.Guarantees.PeriodLeakAccepted`'s count at the number the check reads for it. A sum of
ceilings is at least the ceiling of the sum, so `K` is at least `⌈log2⌉` of the count, not equal to it;
`RowsKBoundsCount` says the count at the check's numbers is at most `2 ^ K`. `Proofs.Properties.NetworkPeriod` proves
it.

The count is the statement's, factor by factor, over the network certifier's `m` connections and the period's `n`
link-windows: `#Ss · ((∏_c (cap_c + 1)) · ∏_i (2^#(free inputs) · #faultySets · 2^β · Dj^k))`. A reader of `K` (the
communication audit's bridge to `PeriodLeakAccepted`) instantiates it at the run's caps and scans' facts, and supplies
each of the statement's quantities as at most the check's number: `#Ss ≤ ∏_c (cap_c + 1)` (a schedule is its lanes'
lengths), the scan's free inputs at most its Program's (`free`), each window's faulty sets at most the sets of at most
`b` of its `N` units, `β` at most `b · max(b_ing, b_eg)` (`faulty_bits`), and `Dj^k` at most `jitter ^ steps`.
-/

namespace Properties.NetworkPeriod

open NetworkCheck.Period

/-- **The rows form's `K` bounds `PeriodLeakAccepted`'s count at the check's numbers.** For the caps `caps` of `m`
connections and the facts `w` of `n` link-windows' scans, under one scan's terms `scan` and the jitter bound `jitter`:
`#Ss` and `∏_c (cap_c + 1)` each at `∏_j (caps j + 1)` (`schedules` and `cells`), and window `i`'s `2^#(free inputs)`
at `2 ^ scan.free`, `#faultySets` at `atMost N b`, `2^β` at `2 ^ (b · max(b_ing, b_eg))` and `Dj^k` at
`jitter ^ scan.steps`, their product is at most `2 ^ K`. -/
def RowsKBoundsCount : Prop :=
  ∀ (facts : Array Facts) (scan : Scan) (jitter windows : ℕ) {m n : ℕ} (caps : Fin m → ℕ) (w : Fin n → Facts),
    (∏ j, (caps j + 1)) * ((∏ j, (caps j + 1)) * ∏ i, (2 ^ scan.free * atMost (w i).units (w i).wrong *
        2 ^ ((w i).wrong * max scan.ingress scan.egress) * jitter ^ scan.steps)) ≤
      2 ^ (rowsTerms facts scan (Array.ofFn w) (Array.ofFn caps) jitter windows).K

end Properties.NetworkPeriod
