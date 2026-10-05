import FlockSoundness.Discharge.Layout.OutSite
import FlockSoundness.Protocol.Discharge.Hidden.Facts

/-!
# What a hidden-output statement takes from the public file (`PublicExposes`)

`HmOut.loadPublic` and `HmOut.drawn` are `HmRow`'s on the circuit without output ports, so a public file's output words
are `n · 0` of them (`loadPublic_outputs_nil`, `drawn_outputs_nil`). `HmOut.regions` is one `mkRegion` per row, region
`p` row `p`'s `b ‖ c` at its `hm96` slot, as `HmRow.regions`' region "digest p" (`regions_get`).
-/

set_option Elab.async false

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Hidden

open Refine

/-- `HmOut.loadPublic` is `HmRow.loadPublic` on the circuit without output ports, then checks. -/
theorem loadPublic_nil {tags : Tags} {c : Circuit} {pf : ByteArray} {pop : Public}
    (h : HmOut.loadPublic tags c pf = .ok pop) : HmRow.loadPublic tags { c with outPorts := #[] } pf = .ok pop := by
  unfold HmOut.loadPublic at h
  dsimp only at h
  simp only [throw_bind'] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    assumption

/-- **`HmOut.regions`** is one region per row, region `p` as `HmRow.regions`' region "digest p": `mkRegion` of
`Layout.hmFree`, `Layout.hmFixed` and `Layout.hmBytes`. -/
theorem regions_get {c : Circuit} {pub : Public} {regs : Array Region} (h : HmOut.regions c pub = .ok regs) :
    regs.size = c.ports.size ∧ ∀ (p : ℕ) (hp : p < regs.size),
      mkRegion c.kLog s!"digest {p}" (Layout.hmFree c) (Layout.hmFixed c p) (Layout.hmBytes c pub p) = .ok regs[p] := by
  unfold HmOut.regions at h
  exact Exec.range_mapM_ok h

end FlockSoundness.Discharge.Hidden
