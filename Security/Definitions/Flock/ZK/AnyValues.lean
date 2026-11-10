module

public import Definitions.Core.Game
public import Definitions.Flock.ZK.SHVZK

/-!
# A table's release behind the ZK server's check

The ZK server computes each rep's inner constraint from its own pads and the coins, and releases the proof file only
where it holds; elsewhere only the pre-final messages leave, with the stop. A worker's clear values need not pass the
constraint: they are any function of the witness, the mask slot and level 0's hiding (`Table.vals`), valid or not. The
witness is any one carrying the statement's public region data (`Table.RegPublic`), which the ZK server's splice of the
public regions makes true of every worker.

* `PerOutcome out rel S ε`: the released bytes `rel`, with their public outcome `out` (where the release stopped),
  simulated per outcome: at every outcome `s` and every event, the bytes jointly with the outcome `s` are no likelier
  than the simulator `S s`'s output, plus `ε s`. It is the table model's form, at one statement and fixed coins, of
  the one zero-knowledge definition, `ZKOut` in the claims core, whose statements read it through the session loop.
* `Within X Y ε`, `Splits rel out B`, `AgreesWhere go view d`: released bytes simulated within `ε`, one-sided, for every
  event (`PerOutcome` at one outcome); bytes that are, at each outcome, that outcome's branch; and a view that is the
  derived view wherever the check passes.
* `gated go view cut`, `gatedSim simFull simCut`: the bytes behind a check, the view where it passes and the cut where
  it fails; and their simulator at each of the two outcomes.
* `Table.Simulable`: the coins are non-degenerate for the table (`Rank`, `PadOnto`, `W^r` bijective, `PadsOnto`,
  `β^r ≠ 0`), so `S_shvzk`'s counting goes through; `Table.RegPublic w`: `w` carries the public region data.
* `Table.HoldsAt w ω`: `w`'s values pass the inner constraint at the pads `ω` samples, each rep's `Σ_i λ_i·h_i = θ`.
* `Table.Gated go w`: the check `go`, a function of the server's randomness alone, passes only where `HoldsAt` does.
* `Table.release go w`: the released bytes, `gated` at the table's view and pre-final view.
* `Release.bytesAt`, `Release.simAt`: the released bytes of a release of `J` tables at an outcome (where it stopped):
  each table's view where its file left, its pre-final view elsewhere; and the simulator, `S_shvzk(x, e)` and the
  dummy's pre-final view there.
* `Release.Gated out full w`: a table's file leaves at an outcome only where its values pass its constraint.

At the table model the released bytes are simulated per outcome (`PerOutcome`) at every witness carrying the public
region data, with no premise that its values pass the constraint; the guarantee reads that at the session loop's release,
in `ZKOut`'s form.
-/

@[expose] public section

namespace FlockSoundness.ZK

open Verity.Game
open scoped ENNReal

/-- **Simulated per outcome**: at every outcome `s` and every event `E`, the released bytes `rel` jointly with the
outcome `out = s` are no likelier than the simulator's output at `s`, plus `ε s`. -/
def PerOutcome {Ω St Ωs V : Type} [Fintype Ω] [Fintype Ωs] (out : Ω → St) (rel : Ω → V) (S : St → Ωs → V)
    (ε : St → ℝ≥0∞) : Prop :=
  ∀ s (E : V → Prop), prCoin (fun ω => out ω = s ∧ E (rel ω)) ≤ prCoin (fun σ => E (S s σ)) + ε s

/-- **`X` is simulated by `Y` within `ε`**, one-sided: no event is more than `ε` likelier under `X`'s randomness than
under `Y`'s. -/
def Within {Ω Ωs V : Type} [Fintype Ω] [Fintype Ωs] (X : Ω → V) (Y : Ωs → V) (ε : ℝ≥0∞) : Prop :=
  ∀ E : V → Prop, prCoin (fun ω => E (X ω)) ≤ prCoin (fun σ => E (Y σ)) + ε

/-- **The released bytes are, at each outcome, that outcome's branch**: `rel ω = B (out ω) ω`. -/
def Splits {Ω St V : Type} (rel : Ω → V) (out : Ω → St) (B : St → Ω → V) : Prop :=
  ∀ ω, rel ω = B (out ω) ω

/-- **The view is the derived view wherever the check passes.** -/
def AgreesWhere {Ω V : Type} (go : Ω → Bool) (view d : Ω → V) : Prop :=
  ∀ ω, go ω = true → view ω = d ω

/-- **The released bytes behind a check**: the view where the check `go` passes, the cut where it fails. -/
def gated {Ω V P : Type} (go : Ω → Bool) (view : Ω → V) (cut : Ω → P) (ω : Ω) : V ⊕ P :=
  if go ω then .inl (view ω) else .inr (cut ω)

/-- **Their simulator at the outcome `b`**: the view's simulator where the check passed, the cut's where it failed,
each on its own coordinate of the simulator's randomness. -/
def gatedSim {Ωf Ωc V P : Type} (simFull : Ωf → V) (simCut : Ωc → P) (b : Bool) (σ : Ωf × Ωc) : V ⊕ P :=
  if b then .inl (simFull σ.1) else .inr (simCut σ.2)

namespace Table

variable {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] (T : Table F K W Salt Seed D Hi VL) [CharP F 2]

/-- **The coins are non-degenerate for the table**: the prover's rank check passes, both paddings reach every vector of
values at their opened positions, each rep's `W^r` is a bijection and each `β^r` is nonzero, so `S_shvzk`'s counting
goes through. -/
def Simulable : Prop :=
  T.Rank ∧ T.PadOnto ∧ (∀ r, Function.Bijective fun p : F × F => T.Wr r p.1 p.2) ∧ T.PadsOnto ∧ ∀ r, T.β r ≠ 0

/-- **`w` carries the public region data** (`H_reg`): its region claims' values are the public ones. -/
def RegPublic (w : W) : Prop := T.sReg w = T.pubReg

/-- **The constraint holds at `ω`**: `w`'s values pass each rep's inner constraint at the pads `ω` samples. -/
def HoldsAt (w : W) (ω : T.Rand) : Prop :=
  ∀ r : Fin 2, ∑ i, T.lam r (T.view w ω).pre i * T.hFull (ω.2.2.1 r) i = T.theta r (T.view w ω).pre

/-- **The check passes only where the constraint holds**: at every randomness `ω` where `go` passes. -/
def Gated (go : T.Rand → Bool) (w : W) : Prop :=
  ∀ ω, go ω = true → T.HoldsAt w ω

/-- **What the ZK server releases** behind the check `go`: the view where it passes, the pre-final view where it
doesn't. -/
noncomputable def release (go : T.Rand → Bool) (w : W) : T.Rand → T.View ⊕ T.PreView :=
  gated go (T.view w) (T.preView w)

end Table

namespace Release

variable {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] {J : ℕ} (T : Fin J → Table F K W Salt Seed D Hi VL)
  [CharP F 2] {St : Type}

/-- **The released bytes of `J` tables at the outcome `s`**: table `j`'s view where its file left (`full s j`), its
pre-final view elsewhere. -/
noncomputable def bytesAt (full : St → Fin J → Bool) (w : Fin J → W) (s : St) (ω : ∀ j, (T j).Rand) :
    ∀ j, (T j).View ⊕ (T j).PreView :=
  fun j => if full s j then .inl ((T j).view (w j) (ω j)) else .inr ((T j).preView (w j) (ω j))

/-- **The simulator at the outcome `s`**: `S_shvzk(x, e)` on the dummy where table `j`'s file left, the dummy's
pre-final view elsewhere, each on its own randomness. -/
noncomputable def simAt (full : St → Fin J → Bool) (w₀ : Fin J → W) (s : St) (σ : ∀ j, (T j).SimRand × (T j).Rand) :
    ∀ j, (T j).View ⊕ (T j).PreView :=
  fun j => if full s j then .inl ((T j).sim (w₀ j) (σ j).1) else .inr ((T j).preView (w₀ j) (σ j).2)

/-- **Each table's file leaves only where its constraint holds**: at every randomness `ω`, wherever the outcome `out ω`
releases table `j`'s file, `w j`'s values pass its inner constraint at its sampled pads. -/
def Gated (out : (∀ j, (T j).Rand) → St) (full : St → Fin J → Bool) (w : Fin J → W) : Prop :=
  ∀ ω j, full (out ω) j = true → (T j).HoldsAt (w j) (ω j)

end Release

end FlockSoundness.ZK
