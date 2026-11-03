import Difftest
import SampledProofs.Pieces
import SampledProofs.OneStageZk

/-!
# `sampled-proofs-difftest`: the pieces, and the `slots` piece over circuit files

`Pieces.table` behind `Difftest.respond`'s line protocol, and one more request, `slots`, whose input names its circuit
files by path (`files`, `[circuit, public]` pairs; `tables`, lookup tables' files, each the `Flock.MUFU_TABLES` table of
its SHA-512): this process reads and parses them as the one-stage verifier does (`Slots.parseFiles` under
`OneStageZk.STATEMENT`) and answers `Pieces.slotsWith` on them, a file it can't read or parse a reject.
-/

namespace SampledProofs.DifftestMain

open Lean (Json)

/-- The bytes of each file of `paths`, or why one can't be read. -/
def readAll (paths : Array String) : IO (Except String (Array ByteArray)) := do
  try return .ok (← paths.mapM IO.FS.readBinFile)
  catch e => return .error s!"slots: {e}"

/-- The circuit files a `slots` input names, read and parsed, or why they can't be. -/
def filesOf (input : Json) : IO (Flock.V (Array (Flock.Circuit × Flock.Public))) := do
  let paths : Except String (Array (String × String) × Array String) := do
    let .arr fs := (input.getObjVal? "files").toOption.getD .null | throw "slots: files is not a list"
    let pairs ← fs.mapM fun f => match f with
      | .arr #[.str c, .str p] => pure (c, p)
      | _ => throw "slots: a file is not [circuit, public]"
    let tables ← match input.getObjVal? "tables" with
      | .error _ => pure #[]
      | .ok (.arr ts) => ts.mapM fun t => match t with
        | .str s => pure s
        | _ => throw "slots: a table is not a path"
      | .ok _ => throw "slots: tables is not a list"
    return (pairs, tables)
  match paths with
  | .error e => return .error e
  | .ok (pairs, tps) =>
    match ← readAll (pairs.flatMap fun (c, p) => #[c, p]), ← readAll tps with
    | .error e, _ => return .error e
    | _, .error e => return .error e
    | .ok bs, .ok ts =>
      let tables := Flock.MUFU_TABLES.filterMap fun t =>
        (ts.find? fun b => Flock.hexOf (Flock.Sha512.hash b) == t.sha512).map fun b => (t.name, b)
      return Slots.parseFiles OneStageZk.STATEMENT
        ((Array.range pairs.size).map fun i => (bs[2 * i]!, bs[2 * i + 1]!)) tables

/-- The response line to one request line: `slots` on the files it names, or a piece of `Pieces.table`. -/
def respond (request : String) : IO String := do
  match Json.parse request with
  | .ok req =>
    match req.getObjValAs? String "fn", req.getObjVal? "input" with
    | .ok "slots", .ok input =>
      let files ← filesOf input
      return Difftest.respond (Difftest.table [("slots", fun j => Pieces.slotsWith j files)]) request
    | _, _ => return Difftest.respond (Difftest.table Pieces.table) request
  | .error _ => return Difftest.respond (Difftest.table Pieces.table) request

end SampledProofs.DifftestMain

/-- Answer requests from stdin until it ends, flushing each response. -/
def main : IO Unit := do
  let stdin ← IO.getStdin
  let stdout ← IO.getStdout
  repeat
    let request ← stdin.getLine
    if request.isEmpty then break
    stdout.putStrLn (← SampledProofs.DifftestMain.respond request)
    stdout.flush
