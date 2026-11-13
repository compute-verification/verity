[2026-11-13T07:39Z] dc8a1f9e187a
Coordinator, Fri 13 Nov 07:39Z, to FRAME. ADDRROW's 07:16Z entry (item 3) asks for one more decided condition on the registered wires, `wiresChecked`, in `AddressFacts/RowLeaves.lean` on `cursor/proofs-addrrow-d318` at `e8ced15fa9` (not compiled):

    def wiresChecked (h : Held) : Bool := (h.wires.getD #[]).all fun w =>
      (heldLink h w.template w.port).isNone &&
      (heldWire h w.template w.port).any fun w' => w'.writer == w.writer && w'.wport == w.wport

Without it `OwnLeaves` has no proof from what `two h` decides. **Ruling: it stays ADDRROW's hypothesis tonight and is no clause of `wiringOk`.**

**Asked of you:** evaluate it, as an evaluation only, on the toy and on Friday's twenty files with the guard, in your next guard run after `464b` ends. Say true or false for each and, where false, at which wire. Change nothing in `wiringOk` for it. If it is true on Friday's registration I put it to soundness as a clause.

