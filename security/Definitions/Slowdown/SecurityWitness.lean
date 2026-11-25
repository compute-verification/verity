module

public import Definitions.Slowdown.Security
public import Definitions.Registration.Server

/-!
# The three sets: the instance their witnesses are stated at

The witnesses of `Slowdown.Guarantees.Secure` are stated at one run of four one-bit gates on one server of one leaf,
joined to the world outside by one link, over one tick and one window. Each leaf holds its inputs: gate `0` is an
input the server holds; `1` an input outside; `2` is computed outside from `1`; and `3` on the server from `0` and
`2`. So the server carries one bit into the window, reads one bit from outside and does one unit of work.

* `hw c m β`: the hardware, a leaf doing `c` units of work a tick and carrying `m` bits, the link `β` bits a tick;
* `frame c m β I`: the server over the one window, missed when it is in `I`;
* `served` and `trained`: the run with every gate the declared circuit's, and with none;
* `recipe`: one weight reaches the target loss, and a window reads two bits for each weight it uses.
-/

@[expose] public section

open InferenceOnly

namespace Slowdown.Security.Witness

open Verity ResourceAccounting Finset

/-- The run's circuit: the inputs `0` and `1`, gate `2` reading `1`, and gate `3` reading `0` and `2`. -/
abbrev circuit : Circuit where
  N := 4
  w _ := 1
  isInput g := decide (g.val < 2)
  reads g := if g.val = 2 then {1} else if g.val = 3 then {0, 2} else ∅
  reads_lt := by decide +kernel
  input_reads := by decide +kernel
  op _ _ := 0
  op_congr _ _ _ _ := rfl
  outputs := {3}

/-- The run: gates `0` and `3` on the server (`true`), gates `1` and `2` outside (`false`), all at tick 0. -/
abbrev run : Placed Bool where
  toCircuit := circuit
  x g := decide (g.val = 0 ∨ g.val = 3)
  t _ := 0
  cell g := g.val
  cell_inj := by decide +kernel

/-- Each gate uses one unit of its leaf's compute, and every wire runs on the one link. -/
abbrev usage : Usage run Unit := ⟨fun _ => 1, fun _ _ => ()⟩

/-- The run: every gate holds 0, and each leaf holds its inputs at every tick. -/
abbrev actual : Actual.Run Bool where
  A := run
  val := fun _ => 0
  faithful := fun _ _ => rfl
  held := fun y _ => univ.filter fun a => run.isInput a = true ∧ run.x a = y
  holding := ⟨fun _ _ _ ha => ⟨(mem_filter.1 ha).2.2, Or.inl (mem_filter.1 ha).2.1⟩,
    fun a ha => mem_filter.2 ⟨mem_univ a, ha, rfl⟩,
    fun g a _ hx s hs hsg => mem_filter.2 ⟨mem_univ a,
      hs.resolve_right (by have : s = 0 := Nat.le_zero.1 hsg; omega), hx⟩⟩

/-- The hardware: each leaf does `c` units of work a tick and carries `m` bits, and one link of `β` bits a tick joins
the server and the world outside. -/
abbrev hw (c m β : ℕ) : Hardware Bool Unit :=
  Hardware.ofLeaves (fun _ => c) (fun _ => m) (fun _ => (false, true)) (fun _ => β)

/-- One window of one tick. -/
abbrev cuts : Cuts 1 := ⟨1, fun j => j.val, fun _ _ h => h, rfl, rfl⟩

/-- The frame: the server `{true}` over one window, missed when it is in `I`. -/
abbrev frame (c m β : ℕ) (I : Finset (Fin (1 * cuts.J))) : Frame Bool Unit :=
  Frame.single (hw c m β) {true} (Set.mem_univ _) 1 cuts I

/-- The run with every gate the declared circuit's. -/
abbrev served : Computation Bool Unit := ⟨actual, usage, ∅, fun _ h => absurd h (notMem_empty _), univ⟩

/-- The run with no gate the declared circuit's. -/
abbrev trained : Computation Bool Unit := ⟨actual, usage, ∅, fun _ h => absurd h (notMem_empty _), ∅⟩

/-- The recipe: one weight reaches the target, a window reads two bits for each weight it uses and one for each
thread it enters, and a thread uses a weight once. -/
noncomputable abbrev recipe : Recipe := ⟨1, 2, 1, 1⟩

/-- The fit bound at the recipe. -/
noncomputable abbrev fitted : Training := Training.fit recipe

/-- The training bound with no premise and the count `0`, which does not hold. -/
abbrev free : Training := ⟨fun {_} _ {_} _ _ _ => True, fun _ => True, fun _ _ => 0, fun _ => rfl⟩

/-- Nothing is shared with the declared circuit. -/
abbrev apart : Shared := ⟨0, 0, 0⟩

end Slowdown.Security.Witness
