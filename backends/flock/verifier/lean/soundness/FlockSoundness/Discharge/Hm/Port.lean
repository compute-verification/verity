/-
A row at its columns: port `p`'s row at VU `g` of a block whose bits are `w`. Its bytes are read at the `sha512x3` slots'
message inputs (`Typed.msgCol`), its salt at the `hm96` slot's input `y` (`saltCol`) and its `b ‖ c` at that slot's
output (`bcCol`, the address `HmRow.regions` pins). When the slots' nets hold on the block's bits there and the block
holds Δ's copies (the first `cv` the midstate, each next `cv` the previous output, the hm96 `cv` the last output, its `pad`
the padding block), the bytes at `bcCol` are the commit string of the row and salt (`port_commit`).
-/
import FlockSoundness.Discharge.Hm.Row
import FlockSoundness.ExecTyped
import FlockSoundness.Binding.Layout
import FlockSoundness.Protocol.Discharge.Hm.Port
import FlockSoundness.Discharge.Hm.Commit

namespace FlockSoundness.Discharge.Hm

open Flock.HmNets hiding and

open Flock

/-! ## Bytes from bits -/

/-- `Binding.bytesAt` at addresses `A (a j t)` reads `bytesOf` of the witness's bits. -/
theorem bytesAt_eq_bytesOf {m N : ℕ} (z : Model.Witness m) (A : ℕ → Fin (2 ^ m)) (a : ℕ → ℕ → ℕ) :
    Binding.bytesAt z (fun (j : Fin N) (t : Fin 8) => A (a j t)) =
      bytesOf (fun x => FlockSoundness.bit (z (A x))) N a := rfl

/-! ## The row's columns -/

/-! ## The port -/

end FlockSoundness.Discharge.Hm
