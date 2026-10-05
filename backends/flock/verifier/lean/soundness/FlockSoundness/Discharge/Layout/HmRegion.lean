import FlockSoundness.Discharge.Layout.Region
import FlockSoundness.ExecSetup
import FlockSoundness.Protocol.Discharge.Layout.HmRegion
import FlockSoundness.Binding.Layout
import FlockSoundness.Refine.Setup

/-!
# The `b ‖ c` that the statement pins at the `hm96` slots' outputs

`HmRow.regions` pins, for each input port `p`, every VU's `hm96` output `b ‖ c` (128 bytes) to the public digest of
the VU's instance (the pinned dummy past the population): region "digest p", whose free bits are a byte string's 10
bit-index bits and the VU's `log₂ g` bits, at the fixed address `hmFixed c p` (`regions_digest`). So in a satisfying
witness of the model's statement at the executable's regions, the 128 bytes at VU `g`'s `hm96` output of port `p` are
that digest (`bc_pinned`). The address is the `hm96` slot's (`bcAddr_eq_slotCol`), and setupH's statement's regions are
`HmRow.regions` (`setupH_regions`).
-/

set_option Elab.async false

namespace FlockSoundness.Discharge.Layout

open Flock Model Refine Binding

/-! ### Region "digest p" -/

/-! ### Where region "digest p" pins byte `j`, bit `t` of VU `g` -/

/-- **It is the `hm96` slot's output**: VU `g`'s port-`p` slot of the `hm96` net, at the net's first output column
plus `8j + t`, when the net has `2^lp` slots per VU. -/
theorem bcAddr_eq_slotCol (c : Circuit) (g p j t : ℕ)
    (hpv : c.perVu.getD c.hmNet 0 = 2 ^ (c.perVu.getD c.hmNet 1).log2) :
    bcAddr c g p j t =
      c.slotCol c.hmNet g p + ((portCols (c.nets.getD c.hmNet default).2.outGroups).getD 0 default).1 + 8 * j + t := by
  simp only [bcAddr, hmFixed, Circuit.slotCol, Range.slot, Range.start, hpv, pow_add]
  ring

/-! ### The bytes there -/

/-! ### What a satisfying witness holds there -/

/-! ### setupH's regions -/

end FlockSoundness.Discharge.Layout
