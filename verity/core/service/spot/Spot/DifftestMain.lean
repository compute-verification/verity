module

public import Difftest
public import Spot.Pieces
public import Spot.OneStageZk

/-!
# The difftest's executable: the pieces, and the `slots` and `file_part` pieces over a circuit file

`Pieces.table` behind `Difftest.respond`'s line protocol, and two more requests, `slots` and `file_part`, whose input
names its circuit file by path (`circuit`; `tables`, lookup tables' files, each the `Flock.MUFU_TABLES` table of its
SHA-512): this process reads and parses it as the one-stage verifier does (`Slots.parseCircuit` under
`OneStageZk.STATEMENT`) and answers `Pieces.slotsWith` or `Pieces.filePartWith` on it, a file it can't read or parse a
reject.
-/

@[expose] public section

namespace Spot.DifftestMain

open Lean (Json)

/-- The bytes of each file of `paths`, or why one can't be read. -/
def readAll (paths : Array String) : IO (Except String (Array ByteArray)) := do
  try return .ok (← paths.mapM fun (p : String) => IO.FS.readBinFile ⟨p⟩)
  catch e => return .error s!"slots: {e}"

/-- The circuit file a `slots` or `file_part` input names, read and parsed, or why it can't be. -/
def circuitOf (input : Json) : IO (Flock.V Flock.Circuit) := do
  let paths : Except String (String × Array String) := do
    let .str c := (input.getObjVal? "circuit").toOption.getD .null | throw "slots: circuit is not a path"
    let tables ← match input.getObjVal? "tables" with
      | .error _ => pure #[]
      | .ok (.arr ts) => ts.mapM fun t => match t with
        | .str s => pure s
        | _ => throw "slots: a table is not a path"
      | .ok _ => throw "slots: tables is not a list"
    return (c, tables)
  match paths with
  | .error e => return .error e
  | .ok (c, tps) =>
    match ← readAll #[c], ← readAll tps with
    | .error e, _ => return .error e
    | _, .error e => return .error e
    | .ok bs, .ok ts =>
      let tables := Flock.MUFU_TABLES.filterMap fun t =>
        (ts.find? fun b => Flock.hexOf (Flock.Sha512.hash b) == t.sha512).map fun b => (t.name, b)
      return Slots.parseCircuit OneStageZk.STATEMENT bs[0]! tables

/-- The response line to one request line: `slots` or `file_part` on the circuit file it names, or a piece of
`Pieces.table`. -/
def respond (request : String) : IO String := do
  match Json.parse request with
  | .ok req =>
    match req.getObjValAs? String "fn", req.getObjVal? "input" with
    | .ok "slots", .ok input =>
      let c ← circuitOf input
      return Difftest.respond (Difftest.table [("slots", fun j => Pieces.slotsWith j c)]) request
    | .ok "file_part", .ok input =>
      let c ← circuitOf input
      return Difftest.respond (Difftest.table [("file_part", fun j => Pieces.filePartWith j c)]) request
    | _, _ => return Difftest.respond (Difftest.table Pieces.table) request
  | .error _ => return Difftest.respond (Difftest.table Pieces.table) request

end Spot.DifftestMain

/-- Answer requests from stdin until it ends, flushing each response. -/
def main : IO Unit := do
  let stdin ← IO.getStdin
  let stdout ← IO.getStdout
  repeat
    let request ← stdin.getLine
    if request.isEmpty then break
    stdout.putStrLn (← Spot.DifftestMain.respond request)
    stdout.flush
