module

public import Definitions.Flock.ZK.SHVZK

/-!
# A table's release behind the ZK server's check

The ZK server computes each rep's inner constraint from its own pads and the coins, and releases the proof file only
where it holds; elsewhere only the pre-final messages leave, with the stop. A worker's clear values need not pass the
constraint: they are any function of the witness, the mask slot and level 0's hiding (`Table.vals`), valid or not.

* `Table.Gated go w`: the check `go`, a function of the server's randomness alone, passes only where `w`'s values pass the
  inner constraint at the drawn pads, each rep's `Σ_i λ_i·h_i = θ`.
* `Table.release go w`: what leaves: the view where `go` passes, the pre-final view where it doesn't.

`Flock.Guarantees.TableReleaseAny` (`Definitions/Flock/Guarantees/ZkAny.lean`) states that the release is simulated at
every witness carrying the public region data, with no premise that its values pass the constraint.
-/

@[expose] public section

namespace FlockSoundness.ZK

namespace Table

variable {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] (T : Table F K W Salt Seed D Hi VL) [CharP F 2]

/-- **The check passes only where the constraint holds**: at every randomness `ω` where `go` passes, `w`'s values pass
each rep's inner constraint at the drawn pads. -/
def Gated (go : T.Rand → Bool) (w : W) : Prop :=
  ∀ ω, go ω = true → ∀ r : Fin 2,
    ∑ i, T.lam r (T.view w ω).pre i * T.hFull (ω.2.2.1 r) i = T.theta r (T.view w ω).pre

/-- **What the ZK server releases** behind the check `go`: the view where it passes, the pre-final view where it
doesn't. -/
noncomputable def release (go : T.Rand → Bool) (w : W) (ω : T.Rand) : T.View ⊕ T.PreView :=
  if go ω then .inl (T.view w ω) else .inr (T.preView w ω)

end Table

end FlockSoundness.ZK
