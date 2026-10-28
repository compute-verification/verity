import NetworkCheck.Period
import Mathlib.Algebra.BigOperators.Fin

/-!
# The network period check's bound: what its `K` counts in each form (`NetworkCheck.Period`)

On acceptance, `NetworkCheck.Period.check` publishes `K`, the sum of a ceiling of `log2` of each factor of
`NetTiming.Guarantees.PeriodLeakAccepted`'s count at the number the check reads for it: `terms`'s in the clock form,
`rowsTerms`'s in the rows form. A sum of ceilings is at least the ceiling of the sum, so `K` is at least `⌈log2⌉` of
the count, not equal to it; `ClockKBoundsCount` and `RowsKBoundsCount` say the count at the check's numbers is at most
`2 ^ K`. `Proofs.Properties.NetworkPeriod` proves both.

The count is the statement's, factor by factor, over the network certifier's `m` connections and the period's `n`
windows: `#Ss · ((∏_c (cap_c + 1)) · ∏_i (2^#(free inputs) · #faultySets · 2^β · Dj^k))`. A reader of `K` (the
communication audit's bridge to `PeriodLeakAccepted`) instantiates it at the run's caps and the windows' facts, and
supplies each of the statement's quantities as at most the check's number:

- in the clock form the windows are the compute audits: `#Ss ≤ #Σ_sync ^ (link-windows)` (each sampling block under one
  sync of Σ_sync), an audit's free inputs at most its layout's (`free`), its faulty sets at most the sets of at most `b`
  of its `N` units, `β` at most `b · bits` with `bits` the most one of its Program's units commits (`faulty_bits`), and
  `Dj^k` at most `jitter ^ steps`;
- in the rows form the windows are the link-windows' scans: `#Ss ≤ ∏_c (cap_c + 1)` (a schedule is its lanes'
  lengths), the scan's free inputs at most its Program's (`free`), each window's faulty sets at most the sets of at most
  `b` of its `N` units, `β` at most `b · max(b_ing, b_eg)` (`faulty_bits`), and `Dj^k` at most `jitter ^ steps`.
-/

namespace Properties.NetworkPeriod

open NetworkCheck.Period

/-- **The clock form's `K` bounds `PeriodLeakAccepted`'s count at the check's numbers.** For the caps `caps` of `m`
connections, `n` compute audits `a` and their facts `w`, the jitter bound `jitter`, `#Σ_sync` `syncs` and `windows`
sampling blocks: `#Ss` at `syncs ^ windows` (`schedules`), `∏_c (cap_c + 1)` at `∏_j (caps j + 1)` (`cells`), and audit
`i`'s `2^#(free inputs)` at `2 ^ (a i).free`, `#faultySets` at `atMost N b`, `2^β` at `2 ^ (b · (a i).bits)` (each of
its at most `b` wrong units commits at most its Program's widest unit's bits) and `Dj^k` at `jitter ^ (a i).steps`,
their product is at most `2 ^ K`. -/
def ClockKBoundsCount : Prop :=
  ∀ (jitter syncs windows : ℕ) {m n : ℕ} (caps : Fin m → ℕ) (a : Fin n → Audit) (w : Fin n → Facts),
    syncs ^ windows * ((∏ j, (caps j + 1)) * ∏ i, (2 ^ (a i).free * atMost (w i).units (w i).wrong *
        2 ^ ((w i).wrong * (a i).bits) * jitter ^ (a i).steps)) ≤
      2 ^ (terms (Array.ofFn a) (Array.ofFn w) (Array.ofFn caps) jitter syncs windows).K

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
