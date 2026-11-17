module

public import Proofs.Flock.SpotFlock.Depths
public import Proofs.Flock.SpotFlock.SlotsAt

/-! `FrameIs` at SPOT's own frame and layout (`frameIs_built`), which every registration whose frame is set to SPOT's
has. -/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

variable (h : Held)

def Built : Prop := ∃ c₀ : Code, h.frame = frameAt (slotMap h) (revealCode h (addressMap h) c₀)

theorem frameIs_built (hB : Built h) (hS : Addresses h) : FrameIs h (layoutFor h) := by
  obtain ⟨c₀, hf⟩ := hB
  exact frameIs_of h hf (slotsAt_slotMap h hS c₀) (Nat.le_of_eq (depth_revealCode h (addressMap h) c₀).symm)

theorem built_spot (c₀ : Code) : Built { h with frame := frameAt (slotMap h) (revealCode h (addressMap h) c₀) } :=
  ⟨c₀, rfl⟩

theorem embeds_layoutFor : Embeds h (layoutFor h) := embeds_layoutAt_any h (rankAt h)

end SpotFlock
