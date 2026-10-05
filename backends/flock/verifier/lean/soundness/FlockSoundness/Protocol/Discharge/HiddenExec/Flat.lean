import FlockSoundness.Protocol.Discharge.Hidden.Flat

/-!
# A hidden-output statement's flat twin

`Stmt.setupHidden`'s Δ is `HmRow.delta`'s, then `Hidden.outPairs` (`Hidden.HiddenDelta`): each output row's message
bits, copied from the output bits the circuit computes. Its flat twin `Hidden.flatTwin st` is the statement with
`HmRow.delta`'s Δ alone, the one `Stmt.setupH` would have built on the same circuit. It has `layout_of`'s layout
(`flat_layout`) and the unit facts the copy positions read (`flat_unitFacts`), as setupH's statement does.

Every first column of `outPairs` is a message bit of a row's `sha512x3` slot (`deltaMsgCol`), which no unit slot's row
is (`outPairs_ne_unit`, for `sha512/row/v1` rows). So at every unit slot's row the hidden statement's level-0 matrices
are its flat twin's (`unitRowsFlat`): the rows that place a VU and copy its inputs read the same in both.
-/

set_option Elab.async false

namespace FlockSoundness.Discharge.HiddenExec

open Flock FlockLevel3 Refine
open FlockSoundness.Discharge.Exec
open Lean (Json)

/-- **The hidden statement reads its flat twin's rows at every unit slot's row**, in `A` and in `B`. -/
def UnitRowsFlat (st : Stmt) : Prop :=
  ∀ s₀ < st.c.unitRange.count, ∀ b < 2 ^ st.c.unit.unitLog, ∀ r j : Fin (2 ^ st.c.kLog),
    r.val = st.c.unitRange.slot s₀ + b → placedA st r j = placedA (Hidden.flatTwin st) r j ∧
      placedB st r j = placedB (Hidden.flatTwin st) r j

/-! ## What `HmOut.checkRows` checks -/

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw partition : Option Json} {program : Option ByteArray} {st : Stmt}

/-- **The flat twin of an untyped hidden statement has the layout the placement reads** (`layout_of`). -/
theorem flat_layout (ht : tags.typed = false)
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) : Layout (Hidden.flatTwin st) :=
  have s := Hidden.setupHidden_spec' h
  layout_of (st := Hidden.flatTwin st) (parse_facts (Hidden.parse_untyped ht s.parse).1) s.pin rfl rfl s.m s.kLog

/-- **The flat twin's unit facts** (`Hidden.unitFacts_of_parse`). -/
theorem flat_unitFacts (ht : tags.typed = false)
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    Aliased.UnitFacts tags (Hidden.flatTwin st) :=
  Hidden.unitFacts_of_parse (st := Hidden.flatTwin st) (Hidden.parse_untyped ht (Hidden.setupHidden_spec' h).parse).1 rfl

/-! ## Δ's output pairs are at the rows' `sha512x3` slots -/

end FlockSoundness.Discharge.HiddenExec
