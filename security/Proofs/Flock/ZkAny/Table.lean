module

public import Proofs.Flock.ZkAny.Generic
public import Proofs.Flock.Claims.ZK.Worker
public import Proofs.Flock.Soundness.ZK.SHVZK
public import Proofs.Flock.Soundness.ZK.Session

/-!
# One table's release behind the ZK server's check, at any values

`table_shvzk` is Lemma B at a witness whose values pass the inner constraint at every pad (`InnerHolds`), and
`worker_shvzk` reaches a worker's values through the switched prover, on the honest values' `InnerHolds`. Behind the ZK
server's check neither is needed:

* `derived_sim`: the derived view `psi (realCore w ω)` and `S_shvzk(x, e)` have the same distribution at any witness:
  `table_shvzk`'s counting reads no constraint, only its last step (`psi_realCore`) does.
* `release_any`: where the check passes, the constraint holds at the sampled pads, so the view is the derived view there
  (`psi_realCore_at`); where it fails, the pre-final view leaves, which Lemma A (`table_prefinal_indep`) bounds at any
  witness carrying the public region data. `gate_perOutcome` makes the released bytes simulated per outcome.
* `worker_release_any`: `release_any` at a worker's clear values (`onVals`), with no premise on them or on the honest
  values beyond the public region data.
* `sim_onVals`: there the simulator reads none of the dummy's clear values, so it is one simulator, fixed before the
  worker's values.
-/

@[expose] public section

namespace FlockSoundness.ZK.Table

open Verity.Game
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

variable {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2] [Fintype F] [DecidableEq F] [Fintype K]
  [DecidableEq K] [Fintype Salt] [DecidableEq Salt] [Fintype Seed] [DecidableEq Seed] [Fintype Hi] [DecidableEq Hi]
  [AddCommGroup Salt] [AddCommGroup Seed] [AddCommGroup Hi]

/-- **The derived view is distributed as `S_shvzk(x, e)` at any witness**: `table_shvzk` without its last step. -/
theorem derived_sim (T : Table F K W Salt Seed D Hi VL) (hrank : T.Rank) (hpad : T.PadOnto)
    (hW : ∀ r, Function.Bijective fun p : F × F => T.Wr r p.1 p.2) (hpads : T.PadsOnto) (hβ : ∀ r, T.β r ≠ 0)
    (w w₀ : W) :
    SameDist (fun ω => T.psi (T.realCore w ω)) (T.sim w₀) :=
  ((SameDist.of_const (T.card_realCore hrank hpad hW hpads hβ w) (T.card_simCore hpad hW w₀)).comp T.psi).of_eq
    (fun _ => rfl) (T.psi_simCore hW w₀)

/-- **One table's release behind the check, at any values**: the released bytes are simulated per outcome, exactly, by
`S_shvzk(x, e)` on the dummy where the check passed and the dummy's pre-final view where it failed. -/
theorem release_any (T : Table F K W Salt Seed D Hi VL) (hrank : T.Rank) (hpad : T.PadOnto)
    (hW : ∀ r, Function.Bijective fun p : F × F => T.Wr r p.1 p.2) (hpads : T.PadsOnto) (hβ : ∀ r, T.β r ≠ 0)
    (go : T.Rand → Bool) (w w₀ : W) (hreg : T.sReg w = T.pubReg) (hreg₀ : T.sReg w₀ = T.pubReg)
    (hgo : T.Gated go w) :
    PerOutcome go (T.release go w) (gatedSim (T.sim w₀) (T.preView w₀)) fun _ => 0 := by
  have : Nonempty T.Rand := ⟨0⟩
  have : Nonempty T.SimRand := ⟨0⟩
  exact gate_perOutcome (d := fun ω => T.psi (T.realCore w ω))
    (fun E => ((T.table_prefinal_indep hrank hW hβ w w₀ hreg hreg₀).prCoin_eq E).le.trans le_self_add)
    (fun E => ((T.derived_sim hrank hpad hW hpads hβ w w₀).prCoin_eq E).le.trans le_self_add)
    fun ω hg => (T.psi_realCore_at hW hβ w hreg ω (hgo ω hg)).symm

/-- **At a worker's clear values**: `release_any` at `onVals`, for any values `v` a worker supplies, valid or not,
behind a check that passes only where they pass the inner constraint; the simulators read the dummy at the honest
values. -/
theorem worker_release_any (T : Table F K W Salt Seed D Hi VL) (hrank : T.Rank) (hpad : T.PadOnto)
    (hW : ∀ r, Function.Bijective fun p : F × F => T.Wr r p.1 p.2) (hpads : T.PadsOnto) (hβ : ∀ r, T.β r ≠ 0)
    (go : T.Rand → Bool) (w w₀ : W) (hreg : T.sReg w = T.pubReg) (hreg₀ : T.sReg w₀ = T.pubReg) (v : T.Vals)
    (hgo : T.onVals.Gated go (w, v)) :
    PerOutcome go (T.onVals.release go (w, v))
      (gatedSim (T.onVals.sim (w₀, T.vals)) (T.onVals.preView (w₀, T.vals))) fun _ => 0 :=
  T.onVals.release_any hrank hpad hW hpads hβ go (w, v) (w₀, T.vals) hreg hreg₀ hgo

/-- **The simulator reads none of the dummy's clear values**: at the table whose witnesses carry a worker's clear
values, `S_shvzk(x, e)` on a dummy is the same at every value, so it is fixed before the worker's values. -/
theorem sim_onVals (T : Table F K W Salt Seed D Hi VL) (w₀ : W) (v v' : T.Vals) :
    T.onVals.sim (w₀, v) = T.onVals.sim (w₀, v') :=
  rfl

end FlockSoundness.ZK.Table
