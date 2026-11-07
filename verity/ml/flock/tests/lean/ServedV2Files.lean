import Flock.Witness
import Flock.Desc
import Flock.KsScope
import Flock.PortRoots
import Flock.Schedule

/-!
`ServedFilesParseV2`'s facts of the circuit, public, partition and program files, each computed with the function the
premise states it with, under the tags it states them at (`Witness.zkTags`, one table, no lookup tables), so that
`test_lean_served_v2_files.py` asserts them on the files the premise pins:

    served-v2-files CIRCUIT PUBLIC PARTITION PROGRAM RECORD

prints one JSON line: whether `HmIn.parse` gives no ports, the public file's instances at none (`HmIn.loadPublic`),
whether every port is a `row/v2` row, `PortRoots.check`'s and `KsScope.check`'s verdicts, and at the session record's
draw (`Witness.recordDraw`) the units it proves, and `setupHidden`'s table's `Zk.shapeOk`, `fast100` and
`Zk.schedule`. A step that fails prints `{"error": …}`. `rowsV2` is `Flock.Bridge.RowsV2`'s body and `jsonOf`
`FlockDeploy.jsonOf`'s, which the premise reads from `Security/Proofs`, a package this one doesn't require.
-/

open Lean (Json toJson)
open Flock

namespace ServedV2Files

def verdict {α : Type} : Except String α → Json
  | .ok _ => Json.str "ok"
  | .error e => Json.str e

def rowsV2 (c : Circuit) : Bool := c.ports.all fun p => p.v2 && p.segs.isNone

def jsonOf (what : String) (b : ByteArray) : Except String Json :=
  match String.fromUTF8? b with
  | none => .error s!"{what}: not UTF-8"
  | some t => (JsonParse.parse t).mapError fun e => s!"{what}: not JSON: {e}"

def facts (cf pf pt pg : ByteArray) (record : String) : Except String Json := do
  let I : Witness.Inputs := ⟨Tags.circuit, cf, pf, 1, [], none, none⟩
  let tags := Witness.zkTags I
  let D ← (Desc.parse tags cf []).mapError (s!"Desc.parse: {·}")
  let c := D.unroll
  let pi ← (HmIn.parse c).mapError (s!"HmIn.parse: {·}")
  let pub ← (HmIn.loadPublic tags c {} pf).mapError (s!"HmIn.loadPublic: {·}")
  let ptJ ← jsonOf "partition" pt
  let some d₀ := Witness.recordDraw record | throw "the session record has no draw"
  let ud ← (Draw.UnitDraw.ofJson d₀).mapError (s!"UnitDraw.ofJson: {·}")
  let st ← (Stmt.setupHidden tags cf pf (Witness.zkCoins I) [] (some d₀) (some ptJ) (some pg) none).mapError
    (s!"setupHidden: {·}")
  pure <| Json.mkObj [
    ("hm_in_none", Json.bool (pi.json == Json.null && pi.ports.isEmpty && pi.leaves.isEmpty)),
    ("pub_n", toJson pub.n),
    ("rows_v2", Json.bool (rowsV2 c)),
    ("port_roots", verdict (PortRoots.check tags cf pf [])),
    ("ks_scope", verdict (KsScope.check tags cf pf [] 1 none)),
    ("proved", toJson ud.proved.size),
    ("shape_ok", verdict (Zk.shapeOk st)),
    ("fast100", Json.bool (fast100 st.m).isSome),
    ("schedule", verdict (Zk.schedule st.m))]

end ServedV2Files

def main (args : List String) : IO UInt32 := do
  let [circuit, pub, partition, program, record] := args
    | IO.eprintln "usage: served-v2-files CIRCUIT PUBLIC PARTITION PROGRAM RECORD"; return 2
  let out := ServedV2Files.facts (← IO.FS.readBinFile circuit) (← IO.FS.readBinFile pub)
    (← IO.FS.readBinFile partition) (← IO.FS.readBinFile program) (← IO.FS.readFile record)
  match out with
  | .ok j => IO.println j.compress; return 0
  | .error e => IO.println (Json.mkObj [("error", Json.str e)]).compress; return 1
