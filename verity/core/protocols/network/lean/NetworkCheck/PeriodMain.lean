import NetworkCheck.Period
import NetworkCheck.CircuitFile

/-! The `network-period` executable: `Core.Protocol.main` of `NetworkCheck.Period.protocol`, audited under `NetworkCheck`,
and `check PARAMS COMMITMENT EVIDENCE CIRCUIT…`, the period's check with its listed circuit files
(`NetworkCheck.Period.protocolWith`). The last files, as many as the parameters' `scan_circuits` lists (`scanFiles`), are
the scan's: each is read whole, as the scan's sessions' verifier parses it (`NetworkCheck.CircuitFile.readWhole`), one
after another so one file's text is held at a time. Every other file is read in chunks, its header alone parsed
(`NetworkCheck.CircuitFile.read`), the files side by side, a dedicated thread a file, since the default task pool runs
only a few tasks at once. A file that can't be read rejects the period, naming the first such file by its place in the
listing. -/

/-- How many circuit files the parameters at `p` list for the scan (`scan_circuits`): 0 where they are not canonical JSON
(`Core.canonicalJson`) or list none. A miscount reads a scan's file header-only, which the check refuses
(`NetworkCheck.Period.fileWhy`), or an audit's whole, which reads more of it than the check needs. -/
def scanFiles (p : String) : IO Nat := do
  let b ← tryCatch (IO.FS.readBinFile p) fun _ => pure ByteArray.empty
  return match (Core.canonicalJson b).toOption.map (·.getObjVal? "scan_circuits") with
    | some (.ok (.arr a)) => a.size
    | _ => 0

/-- The files `files` read whole (`NetworkCheck.CircuitFile.readWhole`), one after another; `k` the first one's place
among the files given. -/
def readWholes (files : List String) (k : Nat) : IO (Except String (Array (ByteArray × Flock.Circuit))) := do
  let mut out := #[]
  for (f, i) in files.zipIdx k do
    match ← tryCatch (NetworkCheck.CircuitFile.readWhole f) (fun e => pure (.error (toString e))) with
    | .ok x => out := out.push x
    | .error why => return .error s!"circuit file {i}: {why}"
  return .ok out

/-- The files `files`, the last `whole` of them read whole (`readWholes`, on one dedicated thread) and the others in
chunks (`NetworkCheck.CircuitFile.read`, a dedicated thread a file). -/
def readAll (files : List String) (whole : Nat) : IO (Except String (Array (ByteArray × Flock.Circuit))) := do
  let k := files.length - whole
  let tasks ← (files.take k).mapM fun (f : String) =>
    IO.asTask (NetworkCheck.CircuitFile.read f) Task.Priority.dedicated
  let wholes ← IO.asTask (readWholes (files.drop k) k) Task.Priority.dedicated
  let mut fs := #[]
  for (t, i) in tasks.zipIdx do
    match ← IO.wait t with
    | .ok (.ok x) => fs := fs.push x
    | .ok (.error why) => return .error s!"circuit file {i}: {why}"
    | .error e => return .error s!"circuit file {i}: {e}"
  match ← IO.wait wholes with
  | .ok (.ok xs) => return .ok (fs ++ xs)
  | .ok (.error why) => return .error why
  | .error e => return .error s!"circuit file {k}: {e}"

def main (args : List String) : IO UInt32 := do
  match args with
  | "check" :: p :: c :: e :: files@(_ :: _) =>
    match ← readAll files (← scanFiles p) with
    | .ok fs => (NetworkCheck.Period.protocolWith fs).main ["check", p, c, e]
    | .error why => (NetworkCheck.Period.refusing s!"circuit files: {why}").main ["check", p, c, e]
  | _ => NetworkCheck.Period.protocol.main args
