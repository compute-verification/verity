import Definitions.Nci.Threads

/-!
# NCI over windows: the fit bound's vocabulary

The fit bound divides the reference circuit `c` into windows, a partition of its computed gates (`Verity.Partition`): in
the accounting model, one server's time from a wipe through whole wipe periods. When no window can hold a model that
reaches the target loss, each thread's work spans several windows, and every window that works on a thread other than
the one that starts it reads some of that thread's state. Over a `Threading` of `c`, a partition `P` into `k` windows
and the window `start t` that starts each thread `t`:

* `threadWeights R t`: the weights thread `t` uses inside `R`, those of which `R` holds a counted multiplication of `t`.
* `visits P t`: the windows that do some of thread `t`'s counted multiplications.
* `entered P start j`: the threads window `j` works on but didn't start.

A training circuit of several optimizer steps has a weight gate per step and parameter: each step's updated weights are
new gates. A slot map (`Slots`) sends each weight gate to its parameter, the slot it is a version of, so that a window
carrying a parameter across steps is counted once for it:

* `Slots.slotsUsed R`: the slots of the weights `R` uses.
* `Slots.threadSlots R t`: the slots of the weights thread `t` uses inside `R`.
-/

namespace Nci.Threading

open Verity Finset

section

variable {c : Declaration} (T : Threading c)

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
structure Slots {c : Declaration} (T : Threading c) where
  /-- The number of parameter slots. -/
  p : ℕ
  /-- The slot of each weight gate: the parameter it is a version of. -/
  slot : Fin c.D.N → Fin p
  reuse_le : ∀ s t, (T.M.filter fun g => slot (T.weight g) = s ∧ T.thread (T.act g) = some t).card ≤ T.reuse

namespace Slots

variable {c : Declaration} {T : Threading c} (σ : T.Slots)

/-- The slots `R` uses: those of the weights it uses. -/
def slotsUsed (R : Finset (Fin c.D.N)) : Finset (Fin σ.p) := (T.weightsUsed R).image σ.slot

/-- The slots thread `t` uses inside `R`: those of the weights it uses there. -/
def threadSlots (R : Finset (Fin c.D.N)) (t : Fin T.m) : Finset (Fin σ.p) := (T.threadWeights R t).image σ.slot

end Slots

end Nci.Threading
