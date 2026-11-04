import NetworkCheck.Period
import NetworkCheck.CircuitFile

/-! The `network-period` executable: `Core.Protocol.main` of `NetworkCheck.Period.protocol`, audited under `NetworkCheck`,
and `check PARAMS COMMITMENT EVIDENCE CIRCUIT…`, the period's check with its listed circuit files
(`NetworkCheck.Period.protocolWith`), each read in chunks, its header alone parsed (`NetworkCheck.CircuitFile.read`), the
files read side by side, a dedicated thread a file, since the default task pool runs only a few tasks at once. A file
that can't be read rejects the period, naming the first such file by its place in the listing. -/

def readAll (files : List String) : IO (Except String (Array (ByteArray × Flock.Circuit))) := do
  let tasks ← files.mapM fun (f : String) => IO.asTask (NetworkCheck.CircuitFile.read f) Task.Priority.dedicated
  let mut fs := #[]
  for (t, i) in tasks.zipIdx do
    match ← IO.wait t with
    | .ok (.ok x) => fs := fs.push x
    | .ok (.error why) => return .error s!"circuit file {i}: {why}"
    | .error e => return .error s!"circuit file {i}: {e}"
  return .ok fs

def main (args : List String) : IO UInt32 := do
  match args with
  | "check" :: p :: c :: e :: files@(_ :: _) =>
    match ← readAll files with
    | .ok fs => (NetworkCheck.Period.protocolWith fs).main ["check", p, c, e]
    | .error why => (NetworkCheck.Period.refusing s!"circuit files: {why}").main ["check", p, c, e]
  | _ => NetworkCheck.Period.protocol.main args
