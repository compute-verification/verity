/-
What the rows of the verifier's builder (`Flock.HmNets`) force: a Hoare-style judgement `Sound m P` (whenever every row of
the final state holds, the result satisfies `P`) and the primitive gadgets' rules: a fresh bit, AND, majority, commit.
-/
import Mathlib.Tactic
import Flock.HmNets
import FlockSoundness.Protocol.Discharge.Hm.Gadget

namespace FlockSoundness.Discharge.Hm

open Flock.HmNets hiding and

namespace Sound

variable {α β : Type}

end Sound

theorem sound_input (n : ℕ) : Sound (input n) fun _ l => l.length = n := by
  intro s
  exact ⟨fun r hr => hr, fun z _ => by simp [input, StateT.run]⟩

end FlockSoundness.Discharge.Hm
