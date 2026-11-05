module

public import Proofs.Flock.Bridge.SlotsKey
public import SampledProofs.OneStageZk

/-!
# v2's template check gives the population's slots

`templateOk_iff`: v2's template check (`SampledProofs.OneStageZk.templateOk`) passes on a circuit exactly when it has a
unit per VU and returns at most `UNIT_CLASS_MAX` bits. `slots_of_templateOk`: where it passes on `c`, every unit of
`keyProgP c pi hU n`'s partition at `outsOfP` has a lowered gate and commits at most `unitClassMax` values, the two
hypotheses of `Flock.Bridge.partitions_slots`; `partitions_of_templateOk` is `partitions_slots` there.
-/

@[expose] public section

open Flock FlockLevel3 Claims.Sampling

namespace FlockSoundness.Discharge.Layout

open Placed SampledProofs.OneStageZk

theorem templateOk_iff (c : Flock.Circuit) :
    templateOk c = .ok () ↔ c.upv ≠ 0 ∧ returnedBits c ≤ UNIT_CLASS_MAX := by
  unfold templateOk
  by_cases h1 : c.upv = 0 <;> by_cases h2 : returnedBits c > UNIT_CLASS_MAX <;>
    simp [h1, h2, bind, Except.bind, throw, throwThe, MonadExceptOf.throw, pure, Except.pure]
  omega

/-- **Where v2's template check passes, the key program's units fit the slots**: each has a lowered gate and commits
at most `unitClassMax` values. -/
theorem slots_of_templateOk {c : Flock.Circuit} (pi : HmIn.Ports) (hU : Built c.unit) (n : ℕ)
    (κ : Fin ((keyProgP c pi hU n).circuit (outsOfP c hU n pi)).N → Option Bool) (h : templateOk c = .ok ()) :
    (∀ u, Flock.Bridge.unitGates ((keyProgP c pi hU n).partition (outsOfP c hU n pi)) u ≠ []) ∧
      ∀ u, committed (Flock.Bridge.progOf ((keyProgP c pi hU n).circuit (outsOfP c hU n pi)) κ)
        (Flock.Bridge.unitGates ((keyProgP c pi hU n).partition (outsOfP c hU n pi)) u) ≤ unitClassMax :=
  ⟨unitGates_keyProgP_ne_nil c pi hU _ ((templateOk_iff c).1 h).1,
    fun u => (committed_keyProgP_le c pi hU n κ u).trans ((templateOk_iff c).1 h).2⟩

/-- **Where v2's template check passes, the population's slots, then the input class, partition the key program's
lowered program**, at any strata `ns` that hold its `n` units. -/
theorem partitions_of_templateOk {c : Flock.Circuit} (pi : HmIn.Ports) (hU : Built c.unit) (n : ℕ)
    (κ : Fin ((keyProgP c pi hU n).circuit (outsOfP c hU n pi)).N → Option Bool) {ns : List ℕ} (hns : n ≤ ns.sum)
    (h : templateOk c = .ok ()) :
    (InputSlot.withInputs ⟨[ns.splitLengths ((List.finRange n).map
        (Flock.Bridge.unitGates ((keyProgP c pi hU n).partition (outsOfP c hU n pi))))]⟩
      (Flock.Bridge.progOf ((keyProgP c pi hU n).circuit (outsOfP c hU n pi)) κ)).Partitions
      (Flock.Bridge.progOf ((keyProgP c pi hU n).circuit (outsOfP c hU n pi)) κ) :=
  Flock.Bridge.partitions_slots _ κ hns (slots_of_templateOk pi hU n κ h).1 (slots_of_templateOk pi hU n κ h).2

end FlockSoundness.Discharge.Layout
