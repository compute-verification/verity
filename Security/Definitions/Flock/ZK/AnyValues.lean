module

public import Definitions.Core.Game
public import Definitions.Flock.ZK.SHVZK

/-!
# A table's release behind the ZK server's check

The ZK server computes each rep's inner constraint from its own pads and the coins, and releases the proof file only
where it holds; elsewhere only the pre-final messages leave, with the stop. A worker's clear values need not pass the
constraint: they are any function of the witness, the mask slot and level 0's hiding (`Table.vals`), valid or not.

* `Within X Y ε`, `Splits rel out B`, `AgreesWhere go view d`: a release simulated within `ε`, one-sided, for every
  event; a release that is, at each outcome, that outcome's branch; and a view that is the derived view wherever the
  check passes.
* `Table.Simulable`: the coins are non-degenerate for the table (`Rank`, `PadOnto`, `W^r` bijective, `PadsOnto`,
  `β^r ≠ 0`), so `S_shvzk`'s counting goes through; `Table.RegPublic w`: `w` carries the public region data.
* `Table.HoldsAt w ω`: `w`'s values pass the inner constraint at the pads `ω` draws, each rep's `Σ_i λ_i·h_i = θ`.
* `Table.Gated go w`: the check `go`, a function of the server's randomness alone, passes only where `HoldsAt` does.
* `Table.release go w`: what leaves: the view where `go` passes, the pre-final view where it doesn't.
* `Session.releaseAt`, `Session.simAt`: a session of `J` tables at an outcome (where it stopped): each table's view
  where its file left, its pre-final view elsewhere; and the simulator, `S_shvzk(x, e)` and the dummy's pre-final view
  there.
* `Session.Gated out full w`: a table's file leaves at an outcome only where its values pass its constraint.

`Flock.Guarantees.TableReleaseAny` and `Flock.Guarantees.SessionReleaseAny` (`Definitions/Flock/Guarantees/ZkAny.lean`)
state that the release is simulated at every witness carrying the public region data, with no premise that its values
pass the constraint.
-/

@[expose] public section

namespace FlockSoundness.ZK

open Verity.Game
open scoped ENNReal

/-- **`X` is simulated by `Y` within `ε`**, one-sided: no event is more than `ε` likelier under `X`'s randomness than
under `Y`'s. -/
def Within {Ω Ωs V : Type} [Fintype Ω] [Fintype Ωs] (X : Ω → V) (Y : Ωs → V) (ε : ℝ≥0∞) : Prop :=
  ∀ E : V → Prop, prCoin (fun ω => E (X ω)) ≤ prCoin (fun σ => E (Y σ)) + ε

/-- **The release is, at each outcome, that outcome's branch**: `rel ω = B (out ω) ω`. -/
def Splits {Ω St V : Type} (rel : Ω → V) (out : Ω → St) (B : St → Ω → V) : Prop :=
  ∀ ω, rel ω = B (out ω) ω

/-- **The view is the derived view wherever the check passes.** -/
def AgreesWhere {Ω V : Type} (go : Ω → Bool) (view d : Ω → V) : Prop :=
  ∀ ω, go ω = true → view ω = d ω

namespace Table

variable {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] (T : Table F K W Salt Seed D Hi VL) [CharP F 2]

/-- **The coins are non-degenerate for the table**: the prover's rank check passes, both paddings reach every vector of
values at their opened positions, each rep's `W^r` is a bijection and each `β^r` is nonzero, so `S_shvzk`'s counting
goes through. -/
def Simulable : Prop :=
  T.Rank ∧ T.PadOnto ∧ (∀ r, Function.Bijective fun p : F × F => T.Wr r p.1 p.2) ∧ T.PadsOnto ∧ ∀ r, T.β r ≠ 0

/-- **`w` carries the public region data** (`H_reg`): its region claims' values are the public ones. -/
def RegPublic (w : W) : Prop := T.sReg w = T.pubReg

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

/-- **Each table's file leaves only where its constraint holds**: at every randomness `ω`, wherever the outcome `out ω`
releases table `j`'s file, `w j`'s values pass its inner constraint at its drawn pads. -/
def Gated (out : (∀ j, (T j).Rand) → St) (full : St → Fin J → Bool) (w : Fin J → W) : Prop :=
  ∀ ω j, full (out ω) j = true → (T j).HoldsAt (w j) (ω j)

end Session

end FlockSoundness.ZK
