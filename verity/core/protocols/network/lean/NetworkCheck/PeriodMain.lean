import NetworkCheck.Period

/-! The `network-period` executable: `Core.BlobProtocol.main` of `NetworkCheck.Period.protocol`, audited under
`NetworkCheck`, and `opening PARAMS COMMITMENT EVIDENCE [BLOB ...]`: each clock-form link-window's opening alone
(`NetworkCheck.Period.openings`), one line `{"windows": [null | reason, ...]}` or `{"refused": why}`, exit 0 or 1. -/

open Lean (Json)

def main (args : List String) : IO UInt32 := do
  match args with
  | "opening" :: params :: commitment :: evidence :: blobs =>
    let bs ← blobs.toArray.mapM fun (f : String) => IO.FS.readBinFile f
    let r := NetworkCheck.Period.openings (← IO.FS.readBinFile params) (← IO.FS.readBinFile commitment)
      (← IO.FS.readBinFile evidence) (Core.Blobs.ofArray bs)
    let line := match r with
      | .ok os => Json.mkObj [("windows", Json.arr (os.map fun (o : Option String) => (o.map Json.str).getD Json.null))]
      | .error e => Json.mkObj [("refused", Json.str e)]
    IO.println line.compress
    return if r.toBool then 0 else 1
  | _ => NetworkCheck.Period.protocol.main args
