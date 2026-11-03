module

public import Definitions.Flock.ZK.SHVZK

/-!
# A table's release behind the ZK server's check

The ZK server computes each rep's inner constraint from its own pads and the coins, and releases the proof file only
where it holds; elsewhere only the pre-final messages leave, with the stop. A worker's clear values need not pass the
constraint: they are any function of the witness, the mask slot and level 0's hiding (`Table.vals`), valid or not.

* `Table.HoldsAt w ω`: `w`'s values pass the inner constraint at the pads `ω` draws, each rep's `Σ_i λ_i·h_i = θ`.
* `Table.Gated go w`: the check `go`, a function of the server's randomness alone, passes only where `HoldsAt` does.
* `Table.release go w`: what leaves: the view where `go` passes, the pre-final view where it doesn't.
* `Session.releaseAt`, `Session.simAt`: a session of `J` tables at an outcome (where it stopped): each table's view where
  its file left, its pre-final view elsewhere; and the simulator, `S_shvzk(x, e)` and the dummy's pre-final view there.

`Flock.Guarantees.TableReleaseAny` and `Flock.Guarantees.SessionReleaseAny` (`Definitions/Flock/Guarantees/ZkAny.lean`)
state that the release is simulated at every witness carrying the public region data, with no premise that its values
pass the constraint.
-/

@[expose] public section

namespace FlockSoundness.ZK

namespace Table

variable {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] (T : Table F K W Salt Seed D Hi VL) [CharP F 2]

/-- **The constraint holds at `ω`**: `w`'s values pass each rep's inner constraint at the pads `ω` draws. -/
def HoldsAt (w : W) (ω : T.Rand) : Prop :=
  ∀ r : Fin 2, ∑ i, T.lam r (T.view w ω).pre i * T.hFull (ω.2.2.1 r) i = T.theta r (T.view w ω).pre

/-- **The check passes only where the constraint holds**: at every randomness `ω` where `go` passes. -/
def Gated (go : T.Rand → Bool) (w : W) : Prop :=
  ∀ ω, go ω = true → T.HoldsAt w ω

/-- **What the ZK server releases** behind the check `go`: the view where it passes, the pre-final view where it
doesn't. -/
noncomputable def release (go : T.Rand → Bool) (w : W) (ω : T.Rand) : T.View ⊕ T.PreView :=
  if go ω then .inl (T.view w ω) else .inr (T.preView w ω)

end Table

namespace Session

variable {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] {J : ℕ} (T : Fin J → Table F K W Salt Seed D Hi VL)
  [CharP F 2] {St : Type}

/-- **What the ZK server releases from a session at the outcome `s`**: table `j`'s view where its file left
(`full s j`), its pre-final view elsewhere. -/
noncomputable def releaseAt (full : St → Fin J → Bool) (w : Fin J → W) (s : St) (ω : ∀ j, (T j).Rand) :
    ∀ j, (T j).View ⊕ (T j).PreView :=
  fun j => if full s j then .inl ((T j).view (w j) (ω j)) else .inr ((T j).preView (w j) (ω j))

/-- **The simulator at the outcome `s`**: `S_shvzk(x, e)` on the dummy where table `j`'s file left, the dummy's
pre-final view elsewhere, each on its own randomness. -/
noncomputable def simAt (full : St → Fin J → Bool) (w₀ : Fin J → W) (s : St) (σ : ∀ j, (T j).SimRand × (T j).Rand) :
    ∀ j, (T j).View ⊕ (T j).PreView :=
  fun j => if full s j then .inl ((T j).sim (w₀ j) (σ j).1) else .inr ((T j).preView (w₀ j) (σ j).2)

end Session

end FlockSoundness.ZK
