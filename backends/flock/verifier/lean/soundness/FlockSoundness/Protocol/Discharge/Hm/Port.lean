/-
A row at its columns: port `p`'s row at VU `g` of a block whose bits are `w`. Its bytes are read at the `sha512x3` slots'
message inputs (`Typed.msgCol`), its salt at the `hm96` slot's input `y` (`saltCol`) and its `b ‖ c` at that slot's
output (`bcCol`, the address `HmRow.regions` pins). When the slots' nets hold on the block's bits there and the block
holds Δ's copies (the first `cv` the midstate, each next `cv` the previous output, the hm96 `cv` the last output, its `pad`
the padding block), the bytes at `bcCol` are the commit string of the row and salt (`port_commit`).
-/
import FlockSoundness.Protocol.Discharge.Hm.Row
import FlockSoundness.Protocol.ExecTyped
import FlockSoundness.Protocol.Binding.Layout

namespace FlockSoundness.Discharge.Hm

open Flock.HmNets hiding and

open Flock

/-! ## Bytes from bits -/

/-! ## The row's columns -/

/-- Compression `k` of port `p`'s row: its `sha512x3` slot at VU `g`, its first column. -/
def shaBase (c : Circuit) (g p k : ℕ) : ℕ := c.slotCol c.shaNet g ((HmRow.rowSlots c.ports).2[p]!)[k]!.1

/-- Compression `k` of port `p`'s row: its place in its slot (`sub < 3`). -/
def shaSub (c : Circuit) (p k : ℕ) : ℕ := ((HmRow.rowSlots c.ports).2[p]!)[k]!.2

/-- The `sha512x3` net's input port `i`'s first column in a slot (`cv_k`, `msg_k` at `2k`, `2k + 1`). -/
def shaIn (c : Circuit) (i : ℕ) : ℕ := (portCols (c.nets.getD c.shaNet default).2.inGroups)[i]!.1

/-- The `hm96` net's input port `i`'s first column in a slot (`cv`, `pad`, `y`). -/
def hmIn (c : Circuit) (i : ℕ) : ℕ := ((portCols (c.nets.getD c.hmNet default).2.inGroups).getD i default).1

/-- **Salt byte `j`, bit `t`** of port `p`'s row at VU `g`: its `hm96` slot's input `y`, bit `8 j + t`. -/
def saltCol (c : Circuit) (g p j t : ℕ) : ℕ := c.slotCol c.hmNet g p + hmIn c 2 + 8 * j + t

/-! ## The port -/

end FlockSoundness.Discharge.Hm
