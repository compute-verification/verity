import Definitions.Flock.Game

/-!
# Two interactions run concurrently

`interleave g₁ g₂` runs two interactions side by side, with the prover choosing at every step
which one advances next, adaptively. This is how the two reps of a Flock table run on the live
coin server: each rep is its own stream, the server draws a stream's coins when that stream's
round arrives, and the prover may interleave the streams in any order (PROTOCOL.md §5.3, S15).
The verifier accepts only if both interactions accept.

The repetition theorem (`value_interleave_le`, the error squares) is in `Proofs.Flock.Soundness.Game.Interleave`.
-/

namespace FlockSoundness

open scoped ENNReal

namespace Game

variable {α β : Type}

/-- The concurrent run once the first interaction's next step is fixed: `adv₁ g₂` advances the
first interaction by one node while the second is at `g₂`, and `fin₁ b` finishes the first
interaction alone once the second has ended with `b`. Recursion is on the second interaction. -/
def interleaveWith (adv₁ : Game β → Game (α × β)) (fin₁ : β → Game (α × β)) :
    Game β → Game (α × β)
  | .ret b => fin₁ b
  | .send M k => .send Bool fun first =>
      if first then adv₁ (.send M k) else .send M fun m => interleaveWith adv₁ fin₁ (k m)
  | @Game.coin _ C iF iN k => .send Bool fun first =>
      if first then adv₁ (@Game.coin _ C iF iN k)
      else @Game.coin _ C iF iN fun c => interleaveWith adv₁ fin₁ (k c)

/-- Run two interactions concurrently; at every step the prover chooses which one advances. -/
def interleave : Game α → Game β → Game (α × β)
  | .ret a => fun g₂ => g₂.map fun b => (a, b)
  | .send M k => interleaveWith (fun g₂ => .send M fun m => interleave (k m) g₂)
      (fun b => (Game.send M k).map fun a => (a, b))
  | @Game.coin _ C iF iN k => interleaveWith
      (fun g₂ => @Game.coin _ C iF iN fun c => interleave (k c) g₂)
      (fun b => (@Game.coin _ C iF iN k).map fun a => (a, b))

end Game

end FlockSoundness
