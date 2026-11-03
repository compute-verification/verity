module

public import Proofs.Flock.ZkAny.Generic
public import Proofs.Flock.Claims.ZK.Worker

/-!
# One table's release behind the ZK server's check, at any values

`table_shvzk` is Lemma B at a witness whose values pass the inner constraint at every pad (`InnerHolds`), and
`worker_shvzk` reaches a worker's values through the switched prover, on the honest values' `InnerHolds`. Behind the ZK
server's check neither is needed:

* `derived_sim`: the derived view `psi (realCore w ω)` and `S_shvzk(x, e)` have the same distribution at any witness:
  `table_shvzk`'s counting reads no constraint, only its last step (`psi_realCore`) does.
* `release_any`: where the check passes, the constraint holds at the drawn pads, so the view is the derived view there
  (`psi_realCore_at`); where it fails, the pre-final view leaves, which Lemma A (`table_prefinal_indep`) bounds at any
  witness carrying the public region data. Each branch is bounded by its simulator, and the release by their sum.
* `Flock.SecurityProofs.TableReleaseAny`: `release_any`, the statement `Flock.Guarantees.TableReleaseAny`.
* `worker_release_any`: `release_any` at a worker's clear values (`onVals`), with no premise on them or on the honest
  values beyond the public region data.
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

/-- **One table's release behind the check, at any values**: each branch bounded by its simulator, `S_shvzk(x, e)` on
the dummy and the dummy's pre-final view, and the release by their sum. -/
theorem release_any (T : Table F K W Salt Seed D Hi VL) (hrank : T.Rank) (hpad : T.PadOnto)
    (hW : ∀ r, Function.Bijective fun p : F × F => T.Wr r p.1 p.2) (hpads : T.PadsOnto) (hβ : ∀ r, T.β r ≠ 0)
    (go : T.Rand → Bool) (w w₀ : W) (hreg : T.sReg w = T.pubReg) (hreg₀ : T.sReg w₀ = T.pubReg)
    (hgo : T.Gated go w) (E : T.View ⊕ T.PreView → Prop) :
    prCoin (fun ω => go ω = true ∧ E (.inl (T.view w ω))) ≤ prCoin (fun σ => E (.inl (T.sim w₀ σ))) ∧
    prCoin (fun ω => go ω = false ∧ E (.inr (T.preView w ω))) ≤ prCoin (fun ω => E (.inr (T.preView w₀ ω))) ∧
    prCoin (fun ω => E (T.release go w ω)) ≤
      prCoin (fun σ => E (.inl (T.sim w₀ σ))) + prCoin (fun ω => E (.inr (T.preView w₀ ω))) := by
  have : Nonempty T.Rand := ⟨0⟩
  have : Nonempty T.SimRand := ⟨0⟩
  have h1 : prCoin (fun ω => go ω = true ∧ E (.inl (T.view w ω))) ≤ prCoin (fun σ => E (.inl (T.sim w₀ σ))) := by
    refine le_trans (prCoin_mono fun ω h => ?_) ((T.derived_sim hrank hpad hW hpads hβ w w₀).prCoin_eq
      fun v => E (.inl v)).le
    obtain ⟨hg, hE⟩ := h
    rw [← T.psi_realCore_at hW hβ w hreg ω (hgo ω hg)] at hE
    exact hE
  have h2 : prCoin (fun ω => go ω = false ∧ E (.inr (T.preView w ω))) ≤
      prCoin (fun ω => E (.inr (T.preView w₀ ω))) :=
    le_trans (prCoin_mono fun ω h => h.2)
      ((T.table_prefinal_indep hrank hW hβ w w₀ hreg hreg₀).prCoin_eq fun v => E (.inr v)).le
  have hor : prCoin (fun ω => E (T.release go w ω)) ≤
      prCoin (fun ω => (go ω = true ∧ E (.inl (T.view w ω))) ∨ (go ω = false ∧ E (.inr (T.preView w ω)))) := by
    refine prCoin_mono fun ω (h : E (T.release go w ω)) => ?_
    show (go ω = true ∧ E (.inl (T.view w ω))) ∨ (go ω = false ∧ E (.inr (T.preView w ω)))
    by_cases hg : go ω = true
    · exact Or.inl ⟨hg, by rwa [release, ite_eq_left hg] at h⟩
    · exact Or.inr ⟨by simpa using hg, by rwa [release, ite_eq_right hg] at h⟩
  exact ⟨h1, h2, hor.trans ((prCoin_or_le _ _).trans (add_le_add h1 h2))⟩

/-- **At a worker's clear values**: `release_any` at `onVals`, for any values `v` a worker supplies, valid or not, behind a
check that passes only where they pass the inner constraint; the simulators read the dummy at the honest values. -/
theorem worker_release_any (T : Table F K W Salt Seed D Hi VL) (hrank : T.Rank) (hpad : T.PadOnto)
    (hW : ∀ r, Function.Bijective fun p : F × F => T.Wr r p.1 p.2) (hpads : T.PadsOnto) (hβ : ∀ r, T.β r ≠ 0)
    (go : T.Rand → Bool) (w w₀ : W) (hreg : T.sReg w = T.pubReg) (hreg₀ : T.sReg w₀ = T.pubReg) (v : T.Vals)
    (hgo : T.onVals.Gated go (w, v)) (E : T.onVals.View ⊕ T.onVals.PreView → Prop) :
    prCoin (fun ω => E (T.onVals.release go (w, v) ω)) ≤
      prCoin (fun σ => E (.inl (T.onVals.sim (w₀, T.vals) σ))) +
        prCoin (fun ω => E (.inr (T.onVals.preView (w₀, T.vals) ω))) :=
  (T.onVals.release_any hrank hpad hW hpads hβ go (w, v) (w₀, T.vals) hreg hreg₀ hgo E).2.2

end FlockSoundness.ZK.Table

namespace Flock.SecurityProofs

/-- **One table's release behind the ZK server's check, at any values** (`Flock.Guarantees.TableReleaseAny`). -/
theorem TableReleaseAny : Flock.Guarantees.TableReleaseAny := by
  intro F K W Salt Seed D Hi VL _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ T hrank hpad hW hpads hβ go w w₀ hreg hreg₀ hgo E
  exact T.release_any hrank hpad hW hpads hβ go w w₀ hreg hreg₀ hgo E

end Flock.SecurityProofs
