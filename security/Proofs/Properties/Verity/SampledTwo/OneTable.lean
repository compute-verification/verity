module

public import Proofs.Flock.SpotFlock.ScopeTable
public import Proofs.Properties.Verity.SampledTwo.SessionCoins

/-! A member session in scope sets up one table, so each of its two reps has `SessionCoins.repsOf` of its
round count. -/

@[expose] public section

namespace Verity.SampledTwo

open SpotFlock.RunsAs.Round2 (TwoHolds RoundsHeld ScopeHeld two_of_twoHolds)

theorem oneTable_of_scope {h : SpotFlock.Held} {m₂ : Claims.Msg} (hT : TwoHolds h) (hS : ScopeHeld h m₂) {t : ℕ}
    (ht : t < h.ms.length) :
    ∃ s spec st, FlockDeploy.setupOf true (SpotFlock.stmtAt h m₂ t) = some (s, spec, #[st]) := by
  obtain ⟨_, _, hd⟩ := two_of_twoHolds hT
  have hs := hS t ht
  have hz : SpotFlock.zkOk (SpotFlock.stmtAt h m₂ t) = true := by
    have hs' := hs
    simp only [SpotFlock.sessionOk, Bool.and_eq_true] at hs'
    exact hs'.1.1.1
  obtain ⟨s, spec, sts, hset, -⟩ := SpotFlock.zkOk_sound hz
  obtain ⟨st, rfl⟩ := Array.size_eq_one_iff.1 (oneTable_of_sessionOk hd ht hs hset).1
  exact ⟨s, spec, st, hset⟩

theorem rounds_of_oneTable {x : Claims.Msg} {s : FlockDeploy.Ins} {spec : Flock.SessionSpec} {st : Flock.Stmt}
    (hset : FlockDeploy.setupOf true x = some (s, spec, #[st])) :
    (FlockDeploy.V true).k x = 2 + 2 * FlockDeploy.repRounds true spec st ∧
      SessionCoins.repsOf ((FlockDeploy.V true).k x) = FlockDeploy.repRounds true spec st := by
  have hk : (FlockDeploy.V true).k x = 2 + FlockDeploy.rounds true spec #[st] := by simp only [FlockDeploy.V, hset]
  have hr : FlockDeploy.rounds true spec #[st] = 2 * FlockDeploy.repRounds true spec st := by
    simp [FlockDeploy.rounds]
  rw [hr] at hk
  refine ⟨hk, ?_⟩
  rw [hk]
  unfold SessionCoins.repsOf
  omega

theorem reps_of_scope {h : SpotFlock.Held} {m₂ : Claims.Msg} (hT : TwoHolds h) (hR : RoundsHeld h m₂)
    (hS : ScopeHeld h m₂) {t : ℕ} (ht : t < h.ms.length) :
    ∃ s spec st, FlockDeploy.setupOf true (SpotFlock.stmtAt h m₂ t) = some (s, spec, #[st]) ∧
      SpotFlock.k₀ h t = 2 + 2 * FlockDeploy.repRounds true spec st ∧
      SessionCoins.repsOf (SpotFlock.k₀ h t) = FlockDeploy.repRounds true spec st := by
  obtain ⟨s, spec, st, hset⟩ := oneTable_of_scope hT hS ht
  obtain ⟨hk, hreps⟩ := rounds_of_oneTable hset
  rw [hR t ht] at hk hreps
  exact ⟨s, spec, st, hset, hk, hreps⟩

end Verity.SampledTwo
