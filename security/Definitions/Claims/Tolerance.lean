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
* `ToleranceHolds`: every set of slots the design misses with probability at least δ holds declared use at most the
  tolerance.
* `CountsPaLeProd`, `ToleranceOfCounts`: open statements, the per-stratum miss bound and the tolerance it gives.
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

/-- **The tolerance** of the counts design `ks` over strata of sizes `ns`, at declared use `w` per slot and the public
constant `L` with `ln(1/δ) ≤ L / 2^b`: the whole declared use of every stratum it never draws (`k_s = 0`), plus the
largest, over the strata it draws, of `⌈L · u_s · n_s / (k_s · 2^b)⌉`, `u_s` the stratum's largest declared use of one
block. Every division rounds up. -/
def tolerance (ns ks : List ℕ) (w : ℕ → ℕ) (L b : ℕ) : ℕ :=
  let ss := List.range ns.length
  let whole := ((ss.filter fun s => ks[s]! = 0).map (stratumUse ns w)).sum
  let ratio := ((ss.filter fun s => ks[s]! ≠ 0).map fun s =>
    (L * stratumUseMax ns w s * ns[s]! + ks[s]! * 2 ^ b - 1) / (ks[s]! * 2 ^ b)).foldr max 0
  whole + ratio

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

end Claims.Sampling
