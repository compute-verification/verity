/-
**The bits a replay block returns**, counted from its registered Definition (`returnedBits`), which
`second_stage.returned_bits` mirrors and the difftest's `replay_returned` holds to it. A two-stage registration carries
the count (`service.WindowCommitmentRecord.returned`), and a served cap below it is refused (`plancheck.two`).

A replay block returns, to the boundary, the output fields of its templates' units that its Definition's return names:
the design's return links (`returnLinks`). The second stage's files lay each returned field on 16-bit words of its own,
the last word's high bits padding (`Bits.wordsOf`, `Bits.retWords`), and every bit of those words is a computed gate
(`Bits.check`). So a field of `b` bits returns `16 ⌈b / 16⌉` gates: an 18-bit value two words, eight of them in one
field nine. It is the count `OneStageZk.returnedBits` makes of one template instance from its circuit file, 16 bits an
output word, taken over the fields a replay block returns and read from the Definition.
-/
module

public import Spot.Bits
public import Spot.SecondStage

@[expose] public section

namespace Spot.SecondStage

/-- **The bits one replay block of `defs[d]` returns** to the boundary: each part of its return (`returnLinks`) is one
output field of a template's units, and gives its units times sixteen bits for each word of the field
(`Bits.wordsOf`). None when the design's links can't be derived (`linkRuns`). -/
def returnedBits (defs : Array Flock.Program.Defn) (d : Nat) : Option Nat := do
  let D ← defs[d]?
  let _ ← linkRuns defs d
  let outs ← returnLinks (callersOf defs D) D
  outs.foldlM (fun (n : Nat) (s : Share Link) => do
    let t ← defs[s.fn]?
    let fs := fields t.ret
    let f ← (List.range fs.size).find? fun f => "o" ++ toString f == s.edge.port
    let ty ← fs[f]?
    pure (n + s.run.count * 16 * Bits.wordsOf ty)) 0

/-- The bits one replay block returns (`returnedBits`), the replay block the one program `P` defines under descriptor
id `key`. -/
def replayReturned (P : Flock.Program.Prog) (key : String) : Option Nat := do
  returnedBits P.defs (← P.index[key]?)

end Spot.SecondStage
