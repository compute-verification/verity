import Definitions.Nci.Ldl
import Definitions.Nci.Hardness
import Mathlib.Data.Fin.VecNotation
import Mathlib.Tactic.FinCases

/-!
# One product at block size one: the honest instance (trusted)

The instance the honest and nontrivial lemmas of the local-description-length and priced-read guarantees are stated at
(`Security/Definitions/Nci/Guarantees/Ldl/Witnesses.lean`, `Security/Definitions/Nci/Guarantees/Hardness/Witnesses.lean`):
one counted product whose activation is handed in, in a `(1, 1)`-partition of one block.

Gate 0 is an unregistered input, the activation. Gate 1 is the product, an operation gate that reads it (at weight
`1` it copies it). There are no registered inputs. The one block is `{1}`, of one gate, which reads one unregistered
input bit, gate 0's. The decoder prints its program, so a string's local description length is its length, and the
block's activation tuple, gate 0's bit, costs one bit: one bit per activation occurrence, at deficit `0`. That is
`ActivationRate` and `Priced` at `β = 1`, and at `β = 2` both fail, which is how each bound's nontrivial lemma drops
its rate. It rests on no run and no `art:`. It is not a deployed training workload, whose block sizes and rates are
the draft's §5 measurements.
-/

namespace Nci.Ldl.Honest

open Verity

/-- Gate 0 is an input, the activation; gate 1 is the product, which reads it. -/
abbrev C : Circuit where
  N := 2
  w _ := 1
  isInput := ![true, false]
  reads := ![∅, {0}]
  reads_lt := by decide
  input_reads := by decide
  op g v := if g = 1 then v 0 else 0
  op_congr g v v' hv := by
    fin_cases g
    · rfl
    · simpa using hv 0 (by decide)
  outputs := {1}

/-- No registered inputs: the activation is novel. -/
abbrev d : Declaration := ⟨C, ∅, fun _ h => absurd h (Finset.notMem_empty _)⟩

/-- One block, `{1}`. -/
abbrev P : Partition C 1 where
  unit := ![none, some 0]
  input_iff := by decide

/-- The registration: the partition, with no registered values to read. -/
abbrev R : Registration d 1 := ⟨fun _ => 0, P⟩

/-- The run's input: the activation bit `0`. -/
abbrev inp : d.D.Transcript := fun _ => 0

/-- **The decoder that prints its program**: `D p t = p`, at any time. -/
abbrev D : Decoder := fun p _ => some p

/-- Each block's activation tuple: gate 0's bit, the one activation the block reads. -/
abbrev a : Fin 1 → d.D.Transcript → List Bool := fun _ X => [(X 0).getLsbD 0]

/-- The one counted product: its representative is gate 1, and it reads the one activation occurrence. -/
abbrev M : Products d.D Unit Unit where
  J := {()}
  rep _ := 1
  rep_inj _ _ _ _ _ := Subsingleton.elim _ _
  rep_op _ _ := rfl
  act _ := ()

/-- The product's gates: gate 1. -/
abbrev gates : Unit → Finset (Fin d.D.N) := fun _ => {1}

/-- The product as a counted operation reading its activation occurrence. -/
abbrev O : Nci.Hardness.Operations d.D Unit Unit := Nci.Hardness.Operations.ofActs d.D M.J gates M.act

end Nci.Ldl.Honest
