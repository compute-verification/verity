module

public import Proofs.Flock.SpotFlock.Depths
public import Proofs.Flock.SpotFlock.SlotsAt

/-!
# `FrameIs` at SPOT's own frame

`frameIs_built`: where the registration's frame is SPOT's own, the base frame over the addresses (`Built`), and every
gate's value has one address (`Addresses`), `FrameIs` holds at SPOT's own layout `layoutFor h`. The slots hold
(`slotsAt_slotMap`), the base frame's own ports are 64 levels deep (`depth_revealCode`), and `frameIs_of` gives
`FrameIs` from those two.

`Built` is what SPOT's registration is: nothing in it reads the frame (`slotMap_frame`, `revealCode_frame`), so every
registration whose frame is set to SPOT's has it (`built_spot`).

`embeds_layoutFor`: SPOT's own layout embeds the statement at every registration, since it is `layoutAt` at a column
choice that places each unit's wires (`embeds_layoutAt_any`).
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

variable (h : Held)

/-- **The registration's frame is SPOT's own**: the base frame over the addresses, `frameAt` at the slot map and
`revealCode` at the address map, from some code `c₀` for the fields the base frame doesn't set. -/
def Built : Prop := ∃ c₀ : Code, h.frame = frameAt (slotMap h) (revealCode h (addressMap h) c₀)

/-- **`FrameIs` at SPOT's own frame and layout** (the module's docstring). -/
theorem frameIs_built (hB : Built h) (hS : Addresses h) : FrameIs h (layoutFor h) := by
  obtain ⟨c₀, hf⟩ := hB
  exact frameIs_of h hf (slotsAt_slotMap h hS c₀) (Nat.le_of_eq (depth_revealCode h (addressMap h) c₀).symm)

/-- **Every registration whose frame is set to SPOT's own is built**: the slot map and the address map don't read the
frame. -/
theorem built_spot (c₀ : Code) : Built { h with frame := frameAt (slotMap h) (revealCode h (addressMap h) c₀) } :=
  ⟨c₀, rfl⟩

/-- **SPOT's own layout embeds the statement** (the module's docstring). -/
theorem embeds_layoutFor : Embeds h (layoutFor h) := embeds_layoutAt_any h (rankAt h)

end SpotFlock
