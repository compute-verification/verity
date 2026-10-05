import FlockSoundness.Discharge.Hidden.Flat
import FlockSoundness.Discharge.Layout.OutShape
import FlockSoundness.Protocol.Discharge.HiddenExec.Ret
import FlockSoundness.Discharge.Hidden.Delta
import FlockSoundness.Discharge.Hidden.Rows

set_option Elab.async false

/-!
# Where the circuit computes each output word

`HmOut.delta` copies an output row's message bits from `HmOut.outCol c g o`, the first column of output word `o` at VU
`g`. In an untyped circuit whose unit returns the outputs, that is the unit copy that returns `o`, at the column of its
returned-output port: `outCol` reads which copy and which port from `ret`, the last write of each output word over the
copies' leaf maps. That write is the only one, because `HmRow.check` throws on an output word some copy returns twice:

* `check_returned`: `HmRow.check`'s `seen` loop, so the copies' leaf maps, in order, are distinct (`returned`);
* `ret_leaf`: `ret` at copy `u'`'s output `j` is `(u', j)`;
* `outCol_leaf`: `HmOut.outCol` there is the unit slot `g · upv + u'`, column `16 j` of the returned-output group;
* `outPortOf_spec`: output word `o` is word `o - outAt jo` of output port `jo`;
* `checkRows_words`: each output row's words hold its port's (`HmOut.checkRows`).
-/

namespace FlockSoundness.Discharge.HiddenExec

open Flock FlockLevel3 FlockSoundness.Discharge.Layout

/-! ## Loop invariants -/

/-! ## `HmRow.check`'s `seen` loop -/

/-! ## `ret` -/

theorem retOf_size (c : Circuit) : ∀ (us : List ℕ) (r : Array (ℕ × ℕ)), (retOf c us r).size = r.size
  | [], _ => rfl
  | u :: us, r => by
    unfold retOf
    rw [List.foldl_cons]
    exact (retOf_size c us _).trans (by rw [← Array.foldl_toList, retStep_size])

/-! ## `HmOut.outCol` at a returned output -/

/-! ## The output port of an output word -/

/-! ## An output row's words -/

end FlockSoundness.Discharge.HiddenExec
