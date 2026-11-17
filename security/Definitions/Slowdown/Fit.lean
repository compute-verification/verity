module

public import Definitions.Slowdown.Threads

/-! The fit bound's vocabulary over windows, threads and parameter slots. -/

@[expose] public section

namespace Slowdown.Threading

open Verity Finset

section

variable {c : RegisteredCircuit} (T : Threading c)

/-- The weights thread `t` uses inside `R`: those of which `R` holds a counted multiplication of thread `t`. -/
def threadWeights (R : Finset (Fin c.D.N)) (t : Fin T.m) : Finset (Fin c.D.N) :=
  (T.threadMults R t).image T.weight

/-- The windows of `P` that do some of thread `t`'s counted multiplications. -/
def visits {k : ℕ} (P : Partition c.D k) (t : Fin T.m) : Finset (Fin k) :=
  univ.filter fun j => (T.threadMults (P.gates j) t).Nonempty

/-- The threads window `j` of `P` works on but didn't start. -/
def entered {k : ℕ} (P : Partition c.D k) (start : Fin T.m → Fin k) (j : Fin k) : Finset (Fin T.m) :=
  univ.filter fun t => j ∈ T.visits P t ∧ start t ≠ j

end

/-- **A slot map** for `T`: the parameter each weight gate is a version of. A thread's counted multiplications on the
versions of one slot number at most `reuse`, as `T.reuse_le` has it for one weight. A thread that uses one version of
each parameter (a token at one optimizer step, forward and backward) meets it, and so does one whose forward and
backward read different versions, each at most `reuse` times in all. -/
structure Slots {c : RegisteredCircuit} (T : Threading c) where
  p : ℕ
  /-- The slot of each weight gate: the parameter it is a version of. -/
  slot : Fin c.D.N → Fin p
  reuse_le : ∀ s t, (T.M.filter fun g => slot (T.weight g) = s ∧ T.thread (T.act g) = some t).card ≤ T.reuse

namespace Slots

variable {c : RegisteredCircuit} {T : Threading c} (σ : T.Slots)

/-- The slots `R` uses: those of the weights it uses. -/
def slotsUsed (R : Finset (Fin c.D.N)) : Finset (Fin σ.p) := (T.weightsUsed R).image σ.slot

/-- The slots thread `t` uses inside `R`: those of the weights it uses there. -/
def threadSlots (R : Finset (Fin c.D.N)) (t : Fin T.m) : Finset (Fin σ.p) := (T.threadWeights R t).image σ.slot

end Slots

end Slowdown.Threading
