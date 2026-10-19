import Definitions.Nci.Threads

/-!
# NCI over windows: the fit bound's vocabulary

The fit bound ("The fit bound", Notion) divides the reference circuit `c` into windows, a partition of its computed gates
(`Verity.Partition`): in the accounting model, one server's time from a wipe through whole wipe periods. When no window
can hold a model that reaches the target loss, each thread's work spans several windows, and every window that works on
a thread other than the one that starts it reads some of that thread's state. Over a `Threading` of `c`, a partition `P`
into `k` windows and the window `start t` that starts each thread `t`:

* `threadWeights R t`: the weights thread `t` uses inside `R`, those of which `R` holds a counted multiplication of `t`.
* `visits P t`: the windows that do some of thread `t`'s counted multiplications.
* `entered P start j`: the threads window `j` works on but didn't start.
-/

namespace Nci.Threading

open Verity Finset

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

end Nci.Threading
