module

public import Definitions.Claims.Sampling
public import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
# The tolerance of a counts design at δ

δ, the guarantee's soundness error, is its only public parameter; the sampling design is the prover's own, registered
before any coin. The tolerance is computed from that design and δ in integer arithmetic: a public natural number `L`
with `ln(1/δ) ≤ L / 2^b` (rounded up), times the design's worst ratio of a block's declared use to its stratum's sampled
share `k_s / n_s`, plus the whole declared use of every stratum the design never draws. Declared use is per slot, in
one unit (gates or bits), read from the declared circuit.

* `tolerance`: that number.
* `toleranceCapped`: the tolerance with each drawn stratum's term capped at that stratum's whole declared use, which
  caps the total at the population's whole declared use.
* `ToleranceHolds`: every set of slots the design misses with probability at least δ holds declared use at most the
  tolerance.
* `CountsPaLeProd`, `ToleranceOfCounts`, `ToleranceCappedOfCounts`: the per-stratum miss bound and the two tolerances
  it gives.
-/

@[expose] public section

namespace Claims.Sampling

/-- The first slot of stratum `s`: the slots of the strata before it. -/
def stratumStart (ns : List ℕ) (s : ℕ) : ℕ := (ns.take s).sum

/-- The slots of stratum `s` that `B` holds. -/
def inStratum (ns : List ℕ) (s : ℕ) (B : Finset ℕ) : Finset ℕ :=
  B.filter fun v => stratumStart ns s ≤ v ∧ v < stratumStart ns s + ns[s]!

/-- The largest declared use of one block of stratum `s`, at declared use `w` per slot. -/
def stratumUseMax (ns : List ℕ) (w : ℕ → ℕ) (s : ℕ) : ℕ :=
  ((List.range ns[s]!).map fun j => w (stratumStart ns s + j)).foldr max 0

/-- The declared use of all of stratum `s`, at declared use `w` per slot. -/
def stratumUse (ns : List ℕ) (w : ℕ → ℕ) (s : ℕ) : ℕ :=
  ((List.range ns[s]!).map fun j => w (stratumStart ns s + j)).sum

/-- A drawn stratum's term: `⌈L · u_s · n_s / (k_s · 2^b)⌉`, `u_s` the stratum's largest declared use of one block. -/
def stratumTerm (ns ks : List ℕ) (w : ℕ → ℕ) (L b s : ℕ) : ℕ :=
  (L * stratumUseMax ns w s * ns[s]! + ks[s]! * 2 ^ b - 1) / (ks[s]! * 2 ^ b)

/-- **The tolerance** of the counts design `ks` over strata of sizes `ns`, at declared use `w` per slot and the public
constant `L` with `ln(1/δ) ≤ L / 2^b`: the whole declared use of every stratum it never draws (`k_s = 0`), plus the
largest, over the strata it draws, of `⌈L · u_s · n_s / (k_s · 2^b)⌉`, `u_s` the stratum's largest declared use of one
block. Every division rounds up. -/
def tolerance (ns ks : List ℕ) (w : ℕ → ℕ) (L b : ℕ) : ℕ :=
  let ss := List.range ns.length
  let whole := ((ss.filter fun s => ks[s]! = 0).map (stratumUse ns w)).sum
  let ratio := ((ss.filter fun s => ks[s]! ≠ 0).map (stratumTerm ns ks w L b)).foldr max 0
  whole + ratio

/-- **The capped tolerance**: `tolerance` with the drawn strata's part at most the sum, over the strata it draws, of each
stratum's term capped at that stratum's whole declared use. It is at most the population's whole declared use. The cap
is on the sum: capping each term inside the largest one would be false (two strata of 10 blocks of declared use 1 each,
`k_s = 1`, δ = 1/100: 9 blocks of each are missed with probability 1/100 and hold 18, above either stratum's 10). -/
def toleranceCapped (ns ks : List ℕ) (w : ℕ → ℕ) (L b : ℕ) : ℕ :=
  let ss := List.range ns.length
  let whole := ((ss.filter fun s => ks[s]! = 0).map (stratumUse ns w)).sum
  let ratio := ((ss.filter fun s => ks[s]! ≠ 0).map (stratumTerm ns ks w L b)).foldr max 0
  let capped := ((ss.filter fun s => ks[s]! ≠ 0).map fun s => min (stratumTerm ns ks w L b s) (stratumUse ns w s)).sum
  whole + min ratio capped

/-- **The tolerance holds**: every set `B` of slots that the counts design `ks` over strata of sizes `ns` misses with
probability at least δ holds declared use at most `tol`. -/
def ToleranceHolds (ns ks : List ℕ) (w : ℕ → ℕ) (δ : ℝ) (tol : ℕ) : Prop :=
  ∀ B : Finset ℕ, B ⊆ Finset.range ns.sum → δ ≤ (OneStage.counts ks).Pa ns B → ∑ v ∈ B, w v ≤ tol

/-- **A counts design misses `B` at most as each stratum's own share**: `P_a(B) ≤ ∏_s (1 − k_s/n_s)^{|B ∩ s|}`, at
counts `k_s ≤ n_s`, a stratum with `k_s = 0` a factor 1. -/
def CountsPaLeProd : Prop :=
  ∀ (ns ks : List ℕ) (B : Finset ℕ), ks.length = ns.length → (∀ p ∈ ns.zip ks, p.2 ≤ p.1) →
    (OneStage.counts ks).Pa ns B ≤
      ∏ s ∈ Finset.range ns.length, (1 - (ks[s]! : ℝ) / ns[s]!) ^ (inStratum ns s B).card

/-- **The tolerance from the design**: at counts `k_s ≤ n_s`, every `0 < δ` and a constant `L` with
`ln(1/δ) ≤ L / 2^b`, the counts design's `tolerance` holds. -/
def ToleranceOfCounts : Prop :=
  ∀ (ns ks : List ℕ) (w : ℕ → ℕ) (δ : ℝ) (L b : ℕ), 0 < δ → ks.length = ns.length →
    (∀ p ∈ ns.zip ks, p.2 ≤ p.1) → Real.log (1 / δ) ≤ (L : ℝ) / 2 ^ b →
      ToleranceHolds ns ks w δ (tolerance ns ks w L b)

/-- **The capped tolerance from the design**: under `ToleranceOfCounts`' hypotheses, `toleranceCapped` holds. -/
def ToleranceCappedOfCounts : Prop :=
  ∀ (ns ks : List ℕ) (w : ℕ → ℕ) (δ : ℝ) (L b : ℕ), 0 < δ → ks.length = ns.length →
    (∀ p ∈ ns.zip ks, p.2 ≤ p.1) → Real.log (1 / δ) ≤ (L : ℝ) / 2 ^ b →
      ToleranceHolds ns ks w δ (toleranceCapped ns ks w L b)

end Claims.Sampling
