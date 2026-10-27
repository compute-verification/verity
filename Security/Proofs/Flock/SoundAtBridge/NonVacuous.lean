import Proofs.Flock.SoundAtBridge.Reduce

/-!
# `SoundAtBridge`'s layer premises hold together, and its event is C-Flock's

`soundAtBridge_nonvacuous` is the bridge's witness, as `ZkReg/NonVacuousHJ.lean` is `zk_session_soundHJR`'s: at any
family of C-Flock sessions, the layer `rowsLayer` that reads them (accepted where flock-verify accepts, opened at the
value layer, its relation that every drawn unit is right) meets `AcceptedLive`, `OpenedPlurAt` at `δo = 0` and
`RelOfRows` together; a proof of it is unsound exactly where its session's outcome is `RowsSession.BadU` (flock-verify
accepts and a drawn unit is wrong or its rows are off at its own reads), so `SoundAtBridge`'s conclusion there is
C-Flock's event and no weaker one; and a finite bound on every session's `n · bound` meets the `δ` hypothesis at its
real value.
-/

namespace Flock.SoundAtBridge

open FlockSoundness FlockSoundness.Discharge.ZkReg Accounting.TS
open Pouw.RelativeSecurity (Paths)
open scoped ENNReal

variable {Ω : Type} [Fintype Ω] {S nπ : ℕ} {V : Type}

/-- **`SoundAtBridge`'s layer premises hold together, and its event is C-Flock's.** At any family of sessions `sess`
with outcomes `out`, any reading `rows` of a session's value layer and any rightness `right`, the layer
`rowsLayer sess out rows right` meets `AcceptedLive`, `OpenedPlurAt` at `δo = 0` and `RelOfRows`; its proof `i` of
round `w` is unsound on the path `x` exactly where `(sess x w i).BadU (out x w i)`; and a bound `B ≠ ⊤` on every
session's `n · bound` meets `n · bound ≤ ofReal δ` at `δ = B.toReal ≥ 0`. Not covered: `WindowLaw`,
`RowsSession.CR` (`EndToEndRegisteredRows`' `cr/sha-512` and `ecr/sha-512` instances) and a
session's existence (`ZkReg.zk_session_soundHJR_nonvacuous` covers its draw hypotheses). -/
theorem soundAtBridge_nonvacuous (sess : Paths Ω S → Fin S → Fin nπ → RowsSession)
    (out : ∀ x w i, (sess x w i).Out) (rows : ∀ s : RowsSession, (Fin s.Circ.N → Bool) → V)
    (right : Paths Ω S → (Fin S → Fin nπ → V) → Prop) :
    AcceptedLive (rowsLayer sess out rows right) sess out ∧
      OpenedPlurAt (rowsLayer sess out rows right) sess rows 0 ∧
      RelOfRows (rowsLayer sess out rows right) sess out rows ∧
      (∀ x w i, (rowsLayer sess out rows right).Unsound x w i ↔ (sess x w i).BadU (out x w i)) ∧
      ∀ B : ℝ≥0∞, B ≠ ⊤ → (∀ x w i, ((sess x w i).n : ℝ≥0∞) * (sess x w i).bound ≤ B) →
        0 ≤ B.toReal ∧ ∀ x w i, ((sess x w i).n : ℝ≥0∞) * (sess x w i).bound ≤ ENNReal.ofReal B.toReal := by
  obtain ⟨-, hAcc, hOpen, hRel⟩ := rowsLayer_premises sess out rows right
  refine ⟨hAcc, hOpen, hRel, fun x w i => ⟨?_, ?_⟩, fun B hB hle => ⟨ENNReal.toReal_nonneg, fun x w i => ?_⟩⟩
  · rintro ⟨hA, hR⟩
    refine ⟨hA, ?_⟩
    by_contra hno
    exact hR fun u hu =>
      ⟨fun hw => hno ⟨u, hu, Or.inl hw⟩, Classical.byContradiction fun hr => hno ⟨u, hu, Or.inr hr⟩⟩
  · rintro ⟨hA, u, hu, h⟩
    refine ⟨hA, fun hR => ?_⟩
    obtain ⟨hw, hr⟩ := hR u hu
    rcases h with h | h
    · exact hw h
    · exact h hr
  · rw [ENNReal.ofReal_toReal hB]
    exact hle x w i

end Flock.SoundAtBridge
