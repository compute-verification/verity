import Verity.Protocol

/-!
# Novel communication intensity: the policy

The developer declares `(D, θ)` (`Verity.Declaration`). The input gates in `θ` hold baseline information, committed
beforehand; every other value is novel information. A division of `D` into isolation units (IUs) is a
`Verity.Partition` of its computed gates. Input gates are in no IU, so an IU that uses an input reads it. The gates `R`
of an IU read `width(in(R) \ θ)` bits of novel information (`novelReads`): baseline information is free, and so is
anything `R` computes itself. `R`'s novel communication intensity is `NCI(R) = novelReads R / size R`, in bits per gate.

`Complies Γ α` is the NCI policy: `D` divides into IUs of at most `Γ` gates, each with NCI at most `α`. A policy is a
condition on declarations that the verifier checks from `D` and `θ` alone (`Policy`), a protocol with no rounds of its
own. `slowdown policy t` is the factor by which every declaration that complies with `policy` and implements `t` is
larger than `t`. `Reads Γ c` is the least total novel information that the IUs of a division of `c` into IUs of at most
`Γ` gates read.
-/

namespace Nci

open Verity
open scoped ENNReal

/-- The novel information that the gates `R` of `d`'s circuit read, in bits: the width of `in(R) \ θ`. -/
def novelReads (d : Declaration) (R : Finset (Fin d.D.N)) : ℕ := d.D.width (d.D.into R \ d.θ)

/-- A policy: a condition on declarations. -/
abbrev Policy : Type := Declaration → Prop

/-- **The NCI policy** at maximum IU size `Γ` and NCI limit `α`: `d`'s circuit divides into IUs of at most `Γ` gates,
each reading at most `α` bits of novel information per gate (`NCI(R) ≤ α`, multiplied out so that an IU with no gates
needs no division). -/
def Complies (Γ : ℕ) (α : ℝ) : Policy := fun d =>
  ∃ (n : ℕ) (P : Partition d.D n),
    ∀ u, d.D.size (P.gates u) ≤ Γ ∧ (novelReads d (P.gates u) : ℝ) ≤ α * d.D.size (P.gates u)

/-- `Reads_Γ(c)`: the least total novel information that the IUs of a division of `c` into IUs of at most `Γ` gates
read. -/
noncomputable def Reads (Γ : ℕ) (c : Declaration) : ℕ :=
  sInf {r | ∃ (n : ℕ) (P : Partition c.D n), (∀ u, c.D.size (P.gates u) ≤ Γ) ∧ r = ∑ u, novelReads c (P.gates u)}

/-- **The slowdown factor** `ρ_policy(t)`: the least ratio of `size(d)` to `size(t)` over the declarations `d` that
implement `t` and comply with `policy`, and `⊤` when none does. -/
noncomputable def slowdown (policy : Policy) (t : Declaration) : ℝ≥0∞ :=
  ⨅ (d : Declaration) (_ : t.Implements d ∧ policy d), (d.D.size d.D.computed : ℝ≥0∞) / t.D.size t.D.computed

end Nci
