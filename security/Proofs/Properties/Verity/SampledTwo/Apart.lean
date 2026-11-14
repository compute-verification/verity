module

public import Proofs.Properties.Verity.SampledTwo.Device

/-!
# `Apart`, from what each session's ledger lines open

The two-session history's `PrintsTwo` and `Binds` (`printsTwo_dev₂`, `binds_dev₂`) take `Apart`: the two setups'
ledger lines open neither session's run on the other's window. `Apart` reads both setups' lines at once. This file
splits it into one condition a session, which a two-session device can be held to a session at a time:

* `OpensOwn S w`: the setup's ledger lines open only its own run, and only on the window `w`;
* `apart_of_own`: two setups that each open only their own run on their own window, with different run ids, are
  `Apart` on two different windows;
* `binds_of_own`: so `Binds` at the two-session history, where at every play after `θ` each session's setup opens
  only its own run on its own one of the parameters' two windows and the two runs' ids differ.

## What this does not show

`OpensOwn` at any device. A device whose open line takes its window's root from the prover's request does not have
it: the request can name the other stage's window.
-/

@[expose] public section

namespace Verity.SampledTwo.Sessions

open Lean (Json)
open Properties.Challenger Spot.Challenger
open Verity.Sampled.Device (RunSetup)

/-- **A setup's ledger lines open only its own run, and only on the window `w`**: every window they open, withheld,
for a run (`OpensIn`) is `w`, and the run is the setup's. -/
def OpensOwn (S : RunSetup) (w : String) : Prop :=
  ∀ w' id, OpensIn S.ledger w' id → id = S.id ∧ w' = w

/-- Lines that open a window for a run, of two lists of ledger lines together, are in one of the two. -/
theorem opensIn_append {L₁ L₂ : List Json} {w id : String} (h : OpensIn (L₁ ++ L₂) w id) :
    OpensIn L₁ w id ∨ OpensIn L₂ w id := by
  unfold OpensIn OpensOn at *
  obtain ⟨line, hl, rest⟩ := h
  have hl' : line ∈ L₁ ++ L₂ := hl
  rcases List.mem_append.1 hl' with h1 | h2
  · exact .inl ⟨line, h1, rest⟩
  · exact .inr ⟨line, h2, rest⟩

/-- **`Apart`, from what each session's ledger lines open**: two setups that each open only their own run on their
own window (`OpensOwn`), with different run ids, open neither session's run on the other's window, where the two
windows differ. -/
theorem apart_of_own {S₁ S₂ : RunSetup} {w₁ w₂ : String} (h₁ : OpensOwn S₁ w₁) (h₂ : OpensOwn S₂ w₂)
    (hid : S₁.id ≠ S₂.id) (hw : w₁ ≠ w₂) : Apart S₁ S₂ w₁ w₂ := by
  refine ⟨fun h => ?_, fun h => ?_⟩
  · rcases opensIn_append h with h | h
    · exact hid (h₁ _ _ h).1.symm
    · exact hw (h₂ _ _ h).2
  · rcases opensIn_append h with h | h
    · exact hw (h₁ _ _ h).2.symm
    · exact hid (h₂ _ _ h).1

/-- **`Binds` at the two-session history, from what each session's ledger lines open** (`binds_dev₂`): at every play
after `θ`, the first session's setup opens only its own run on the parameters' first window, the second's only its
own on the second, and the two runs' ids differ. -/
theorem binds_of_own (S₁ S₂ : Exchange → RunSetup) (cut : List Release → ℕ) (θ : ByteArray)
    (h₁ : ∀ o : Exchange, o.params = θ → ∀ w₁ w₂, windowsOf θ = some (w₁, w₂) → OpensOwn (S₁ o) w₁)
    (h₂ : ∀ o : Exchange, o.params = θ → ∀ w₁ w₂, windowsOf θ = some (w₁, w₂) → OpensOwn (S₂ o) w₂)
    (hid : ∀ o : Exchange, o.params = θ → (S₁ o).id ≠ (S₂ o).id) :
    Binds (dev₂ S₁ S₂ cut) θ :=
  binds_dev₂ S₁ S₂ cut θ fun o hθ w₁ w₂ hw hne =>
    apart_of_own (h₁ o hθ w₁ w₂ hw) (h₂ o hθ w₁ w₂ hw) (hid o hθ) hne

end Verity.SampledTwo.Sessions
