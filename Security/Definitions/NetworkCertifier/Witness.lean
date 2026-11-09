module

public import Definitions.NetworkCertifier.SampledAudit

/-!
# The audit the leak bounds' nontrivial witnesses fail at

Definitions only. One input gate and its negation (`notProg`), sampled `all` (`notStmt`), in one class of one stratum
of one proof block (`notPart`), with each step's egress presence gate the negation (`notLay`). `atNot D` is a
deployment `D` at that statement and partition. A nontrivial witness's statement reads them, so they are in the
definitions tree, beside what the guarantees they fail read.
-/

@[expose] public section

namespace NetTiming.Witness

/-- One input gate and its negation, the negation returned. -/
def notProg : Claims.Sampling.Prog := ⟨[.input, .not 0], [1]⟩

/-- The program, sampled `all`. -/
def notStmt : Claims.Sampling.AuditStatement := ⟨[], notProg, ⟨"", 0, []⟩, ⟨[.all]⟩, []⟩

/-- One class of one stratum of one proof block, `{0, 1}`. -/
def notPart : Claims.Sampling.Partition := ⟨[[[[0, 1]]]]⟩

/-- Step `t`'s ingress is the input, and its egress slot's presence gate the negation. -/
def notLay : BoundaryGates 1 1 := ⟨fun _ _ _ => (1, []), fun _ => [0]⟩

/-- The deployment `D` at `notStmt`, partitioned by `notPart`. -/
def atNot {Coin Col T : Type} (D : Claims.Sampling.Deploy Coin Col T) : Claims.Sampling.Deploy Coin Col T :=
  { D with parse := fun _ => some notStmt, partition := fun _ => some notPart }

end NetTiming.Witness
