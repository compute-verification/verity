module

public import Proofs.Flock.ZkAny.Generic
public import Proofs.Flock.Claims.ZK.Worker
public import Proofs.Flock.Soundness.ZK.SHVZK
public import Proofs.Flock.Soundness.ZK.Session

/-! One table's release behind the ZK server's check is simulated per outcome at any values (`release_any`, `worker_release_any`). -/

@[expose] public section

namespace FlockSoundness.ZK.Table

open Verity.Game
open scoped ENNReal

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

variable {F K W Salt Seed D Hi VL : Type} [Field F] [Field K] [CharP F 2] [Fintype F] [DecidableEq F] [Fintype K]
  [DecidableEq K] [Fintype Salt] [DecidableEq Salt] [Fintype Seed] [DecidableEq Seed] [Fintype Hi] [DecidableEq Hi]
  [AddCommGroup Salt] [AddCommGroup Seed] [AddCommGroup Hi]

theorem derived_sim (T : Table F K W Salt Seed D Hi VL) (hrank : T.Rank) (hpad : T.PadOnto)
    (hW : ∀ r, Function.Bijective fun p : F × F => T.Wr r p.1 p.2) (hpads : T.PadsOnto) (hβ : ∀ r, T.β r ≠ 0)
    (w w₀ : W) :
    SameDist (fun ω => T.psi (T.realCore w ω)) (T.sim w₀) :=
  ((SameDist.of_const (T.card_realCore hrank hpad hW hpads hβ w) (T.card_simCore hpad hW w₀)).comp T.psi).of_eq
    (fun _ => rfl) (T.psi_simCore hW w₀)

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

theorem worker_release_any (T : Table F K W Salt Seed D Hi VL) (hrank : T.Rank) (hpad : T.PadOnto)
    (hW : ∀ r, Function.Bijective fun p : F × F => T.Wr r p.1 p.2) (hpads : T.PadsOnto) (hβ : ∀ r, T.β r ≠ 0)
    (go : T.Rand → Bool) (w w₀ : W) (hreg : T.sReg w = T.pubReg) (hreg₀ : T.sReg w₀ = T.pubReg) (v : T.Vals)
    (hgo : T.onVals.Gated go (w, v)) :
    PerOutcome go (T.onVals.release go (w, v))
      (gatedSim (T.onVals.sim (w₀, T.vals)) (T.onVals.preView (w₀, T.vals))) fun _ => 0 :=
  T.onVals.release_any hrank hpad hW hpads hβ go (w, v) (w₀, T.vals) hreg hreg₀ hgo

theorem sim_onVals (T : Table F K W Salt Seed D Hi VL) (w₀ : W) (v v' : T.Vals) :
    T.onVals.sim (w₀, v) = T.onVals.sim (w₀, v') :=
  rfl

end FlockSoundness.ZK.Table
