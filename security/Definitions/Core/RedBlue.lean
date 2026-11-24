module

public import Definitions.Core.Structure

/-!
# A device's run of a circuit, as a red-blue pebble game in cycles (trusted)

A run of a circuit over `T` cycles (`Run`) gives every gate a cycle, at least `1` for a computed gate, and says which
gates' values are on the die (red) and which in device memory (blue) at the end of each cycle; a value can be both. A
computed gate's cycle is when it is computed and an input gate's is when it arrives, cycle `0` for one held at the
start. A cycle is one clock period at the device's highest clock, and a pebble is one bit: red and blue are counted in
`Circuit.width`.

* **Order** (`Run.Order`). A gate reads only gates of its own cycle and gates red at the end of the cycle before; a
  gate is red or blue at the end of a cycle only if it is of that cycle, or was red or blue at the end of the one
  before. Each gate is computed once: a value computed again is another gate.
* **Limits** (`Run.Within`, at a device's `Limits`). A cycle computes at most `rate` gates, on paths of at most `depth`
  gates; at most `chip` bits are red and at most `mem` blue at the end of a cycle; at most `bw` bits a cycle are
  loaded, red at its end after being blue and not red at the end of the one before; and at most `cap` bits of input
  arrive in each cycle from `1` on.

`feasible T H` is the circuits with such a run. It is a property of one run's circuit. What an audit's certificate
reads beyond it is how a device's runs vary with the verifier's coins, which a set of circuits cannot say.
-/

@[expose] public section

namespace Verity.RedBlue

open Finset

/-- **A run** of `C` over `T` cycles: each gate's cycle, and the gates on the die (`red`) and in device memory (`blue`)
at the end of each cycle. -/
structure Run (C : Circuit) (T : ℕ) where
  /-- the cycle a computed gate is computed in and an input gate arrives in, `0` for an input held at the start -/
  cycle : Fin C.N → ℕ
  cycle_le : ∀ g, cycle g ≤ T
  computed_pos : ∀ g, C.isInput g = false → 1 ≤ cycle g
  red : ℕ → Finset (Fin C.N)
  blue : ℕ → Finset (Fin C.N)

/-- **A device's limits**: gates a cycle, gates on a path within a cycle, bits on the die, bits in device memory, bits
loaded from device memory a cycle, and bits of input a cycle. -/
structure Limits where
  rate : ℕ
  depth : ℕ
  chip : ℕ
  mem : ℕ
  bw : ℕ
  cap : ℕ

namespace Run

variable {C : Circuit} {T : ℕ} (R : Run C T)

/-- The gates computed in cycle `c`. -/
def gatesAt (c : ℕ) : Finset (Fin C.N) := C.computed.filter fun g => R.cycle g = c

/-- The input gates that arrive in cycle `c`. -/
def inputsAt (c : ℕ) : Finset (Fin C.N) := univ.filter fun g => C.isInput g = true ∧ R.cycle g = c

/-- The gates loaded in cycle `c + 1`: red at its end, blue and not red at the end of cycle `c`. -/
def loaded (c : ℕ) : Finset (Fin C.N) := (R.red (c + 1)).filter fun a => a ∈ R.blue c ∧ a ∉ R.red c

/-- **A window**: the gates computed in the `Δ` cycles after cycle `a`. -/
def window (a Δ : ℕ) : Finset (Fin C.N) := C.computed.filter fun g => a < R.cycle g ∧ R.cycle g ≤ a + Δ

/-- **Order**: what a gate may read, and where a value may be. -/
structure Order : Prop where
  reads : ∀ g, ∀ a ∈ C.reads g, R.cycle a = R.cycle g ∨ a ∈ R.red (R.cycle g - 1)
  start : ∀ a ∈ R.red 0 ∪ R.blue 0, R.cycle a = 0
  kept : ∀ c, ∀ a ∈ R.red (c + 1) ∪ R.blue (c + 1), R.cycle a = c + 1 ∨ a ∈ R.red c ∪ R.blue c

/-- **Within the limits `H`**, at every cycle. -/
structure Within (H : Limits) : Prop where
  rate : ∀ c, C.size (R.gatesAt c) ≤ H.rate
  depth : ∀ c, Structure.depth C (R.gatesAt c) ≤ H.depth
  chip : ∀ c, C.width (R.red c) ≤ H.chip
  mem : ∀ c, C.width (R.blue c) ≤ H.mem
  bw : ∀ c, C.width (R.loaded c) ≤ H.bw
  cap : ∀ c, 1 ≤ c → C.width (R.inputsAt c) ≤ H.cap

end Run

/-- **The circuits a device with limits `H` can run in `T` cycles**: those with an ordered run within `H`. -/
def feasible (T : ℕ) (H : Limits) : Set Circuit := {C | ∃ R : Run C T, R.Order ∧ R.Within H}

end Verity.RedBlue
