module

public import Difftest
public import Spot.Pieces

/-!
# `sampled-proofs-wired-key-dev`: round 2 with the wired key's switch on, for development plays only

Round 2's statement check (`second_statements`, `Pieces.secondOwn`) with `Own.wiredKey` on: a read whose key one wire
fills whole is taken (`SecondStage.keyFailure`, `openedRow`). The build of record holds the switch off
(`SecondStage.WIRED_KEY`): `sampled-proofs-verifier` and every other executable read the field from no input. What this
one accepts is a development play, never an accept of record, and nothing that records an audit runs it.
-/

@[expose] public section

namespace Spot.WiredKeyDevMain

open Lean (Json)

/-- `Pieces.secondStatements` with the wired key's switch on. -/
def secondStatements (j : Json) : Except String Json := do
  let (allow, own, stmts) ← Pieces.secondOwn j
  return (SecondStage.checkAny allow { own with wiredKey := true } stmts).toJson

end Spot.WiredKeyDevMain

def main : IO Unit :=
  Difftest.serve (Difftest.table [("second_statements", Spot.WiredKeyDevMain.secondStatements)])
