import FlockSoundness.Discharge.Layout.Scope
import FlockSoundness.Protocol.Discharge.Canonical.Format
import FlockSoundness.Discharge.Layout.KeyProg

set_option Elab.async false

/-!
# The canonical I/O format, as one check on the parsed circuit

`PROTOCOL.md` §16.16 specifies one row format for every commitment a session registers: every row, input or output,
is a `sha512/row/v2` bit row of any length `ℓ < 2^64`, hashed as `SHA-512(P(ℓ) ‖ packBits(bits))` with SHA-512's own
padding (so a last block may be partial, and is the `hm96` slot's); output rows hold exactly the bits of their run of
output ports; and every unit input is a row bit or the zero. `formatOk` is that format as code that runs on the circuit
`HmOut.parse` returns, in the style of `Layout.scopeOk`, and `formatOk_spec` reads it back as the conditions the binding
lemmas take. It replaces `Layout.scopeOk`: no condition on the leaf groups (any number of input rows and groups, leaf
cuts included), none on the out net (a tail's last stage may return the outputs), and none on row widths (`16 · words`
retires with `sha512/row/v1`).

`unitLog ≤ 32` and `portReadsB` stay: the first narrows nothing (`Layout.unitLog_le`), and the second is what makes
`readsRows`'s keys the keys a computed row reads (an input column outside every port is a self row, which Δ copies
nothing into). `vusPow2` is new and does narrow: it is what `outNet = unitNet` gave the region pin for free.

Every declaration here is proved.
-/

namespace FlockSoundness.Discharge.Canonical

open Flock FlockLevel3

/-- **A canonical circuit has a VU per block** (`g_pos` without `outNet = unitNet`). -/
theorem formatOk_g_pos {c : Circuit} (h : formatOk c = true) : 0 < c.g := by
  rw [← vusPow2_of_formatOk h]
  exact Nat.two_pow_pos _

end FlockSoundness.Discharge.Canonical
