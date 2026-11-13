module

public import Definitions.Core
public import Mathlib.Analysis.Real.Sqrt

/-!
# The inference-only policy

The developer declares `(D, θ)` (`Verity.RegisteredCircuit`). The input gates in `θ` hold the registered information,
committed beforehand; every other value is fresh input. A division of `D` into inference blocks is a `Verity.Partition`
of its computed gates. Input gates are in no inference block, so an inference block that uses an input reads it. The
gates `R` of an inference block read `width(in(R) \ θ)` fresh input bits (`novelReads`): registered information is free,
and so is anything `R` computes itself.

`Complies c k` is the inference-only policy: `D` has a `(c, k)`-partition (`Verity.Partition.IsCK`), every inference
block exactly `c` gates, padding included, reading at most `k` fresh input bits. Its rate is `α = k / c`, the most fresh
input an inference block reads per gate. `CompliesRate c α` is the policy as the verifier's check (`NciCheck`) reads it,
on a circuit of any widths: inference blocks of at most `c` gates, each reading at most `α` fresh input bits per gate. A
`(c, k)`-partition complies at rate `k / c`, and an inference block of at most `c` gates at rate `α` reads at most
`k = α c` fresh input bits. `CompliesAtMost c k n` is its count form on a Boolean declaration: `D` divides into at most
`n` proof blocks of a `(c, k)`-partition. A policy is a condition on declarations that the verifier checks from `D` and
`θ` alone (`Policy`), a protocol with no rounds of its own. `slowdown policy t` is the factor by which every declaration
that complies with `policy` and implements `t` is larger than `t`. `Reads c C` is the least total fresh input that the
inference blocks of a division of `C` into blocks of at most `c` gates read.
-/

@[expose] public section

namespace Nci

open Verity
open scoped ENNReal

/-- The fresh input that the gates `R` of `d`'s circuit read, in bits: the width of `in(R) \ θ`. -/
def novelReads (d : RegisteredCircuit) (R : Finset (Fin d.D.N)) : ℕ := d.D.width (d.D.into R \ d.θ)

/-- A policy: a condition on declarations. -/
abbrev Policy : Type := RegisteredCircuit → Prop

/-- **The inference-only policy** at `(c, k)`: `d` has a `(c, k)`-partition, its circuit Boolean and divided into
inference blocks of at most `c` gates, each with at most `k` fresh input bits. -/
def Complies (c k : ℕ) : Policy := fun d =>
  ∃ (B : ℕ) (P : Partition d.D B), P.IsCK d c k

/-- **The policy at a rate**, as the verifier's check reads it on a circuit of any widths: `d`'s circuit divides into
inference blocks of at most `c` gates, each reading at most `α` fresh input bits per gate (`novelReads R ≤ α · size R`,
multiplied out so that a block with no gates needs no division). -/
def CompliesRate (c : ℕ) (α : ℝ) : Policy := fun d =>
  ∃ (n : ℕ) (P : Partition d.D n),
    ∀ u, d.D.size (P.gates u) ≤ c ∧ (novelReads d (P.gates u) : ℝ) ≤ α * d.D.size (P.gates u)

/-- **At most `n` `(c, k)`-inference blocks**: `d`'s operation gates divide into at most `n` proof blocks of a
`(c, k)`-partition (`Verity.Partition.IsCK`). -/
def CompliesAtMost (c k n : ℕ) : Policy := fun d =>
  ∃ B ≤ n, ∃ P : Partition d.D B, P.IsCK d c k

/-- `Reads_c(C)`: the least total fresh input that the inference blocks of a division of `C` into blocks of at most `c`
gates read. -/
noncomputable def Reads (c : ℕ) (C : RegisteredCircuit) : ℕ :=
  sInf {r | ∃ (n : ℕ) (P : Partition C.D n), (∀ u, C.D.size (P.gates u) ≤ c) ∧ r = ∑ u, novelReads C (P.gates u)}

/-- **The slowdown factor** `ρ_policy(t)`: the least ratio of `size(d)` to `size(t)` over the declarations `d` that
implement `t` and comply with `policy`, and `⊤` when none does. -/
noncomputable def slowdown (policy : Policy) (t : RegisteredCircuit) : ℝ≥0∞ :=
  ⨅ (d : RegisteredCircuit) (_ : t.Implements d ∧ policy d), (d.D.size d.D.computed : ℝ≥0∞) / t.D.size t.D.computed

end Nci
