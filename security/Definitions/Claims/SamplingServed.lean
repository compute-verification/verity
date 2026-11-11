module

public import Definitions.Claims.SamplingTwo

/-!
# Claims: what a set of incorrect proof blocks serves, request by request

Under a split `req` of the served outputs into requests, request `r` serves `|r|` bits (`servedBits`), and `T_r` is the
proof blocks of `T` with a committed value one of `r`'s served outputs reads (`blocksTo`, `Prog.Reads`). The patterns'
form counts the two-stage influence set (`Claims.Sampling.influenceTwo`) by the admissible sets' patterns
`r ↦ lightOf |r| T_r`, a request's class costing `lightBits` (`Claims.Guarantees.InfluenceTwoPatterns`).

Statements only: definitions, no proofs.
-/

@[expose] public section

namespace Claims.Sampling

/-- **Gate `g` reads gate `j`**: `j` is `g`, or a gate that one of the wires `g` reads reads. -/
inductive Prog.Reads (C : Prog) : ℕ → ℕ → Prop
  | refl (g : ℕ) : Prog.Reads C g g
  | step {g i j : ℕ} {x : BGate} : C.gates[g]? = some x → i ∈ x.reads → Prog.Reads C i j → Prog.Reads C g j

/-- **Request `r`'s served bits** `|r|` under a split `req` of the served outputs into requests: the served outputs it
holds. -/
def servedBits {R : Type} [DecidableEq R] {n : ℕ} (req : Fin n → R) (r : R) : ℕ :=
  (Finset.univ.filter fun i => req i = r).card

open Classical in
/-- **The proof blocks of `T` that request `r` reads** `T_r`: those with a committed value that one of `r`'s served
outputs reads. `T` touches `r` when `T_r` is nonempty. -/
noncomputable def blocksTo {R : Type} (C : Prog) (Pt : Partition) (req : Fin C.outs.length → R) (T : Finset ℕ)
    (r : R) : Finset ℕ :=
  T.filter fun v => ∃ g ∈ Pt.blocks.getD v [], C.Boundary (Pt.blocks.getD v []) g = true ∧
    ∃ i, req i = r ∧ C.Reads C.outs[i] g

/-- **A request's class of `U`**: `none` at `U = ∅`; `some none` when `U` commits at least `n` bits, so that its blocks
are forgotten; and `some (some U)` otherwise. The influence set is counted by the admissible sets' patterns
`r ↦ lightOf |r| T_r`. -/
def lightOf (s : AuditStatement) (Pt : Partition) (n : ℕ) (U : Finset ℕ) : Option (Option (Finset ℕ)) :=
  if U = ∅ then none else if n ≤ bitsOf s Pt U then some none else some (some U)

/-- A class's bits: `0` at `none`, `n` at `some none` and `b(U)` at `some (some U)`. -/
def lightBits (s : AuditStatement) (Pt : Partition) (n : ℕ) : Option (Option (Finset ℕ)) → ℕ
  | none => 0
  | some none => n
  | some (some U) => bitsOf s Pt U

end Claims.Sampling
