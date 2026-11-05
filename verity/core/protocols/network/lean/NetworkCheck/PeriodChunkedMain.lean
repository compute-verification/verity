import NetworkCheck.Period
import NetworkCheck.CircuitFile

/-! The test-only `network-period-chunked` executable: `network-period`'s commands (`NetworkCheck.PeriodMain`), audited
under `NetworkCheck`, with each audit's circuit file read in chunks, its header and its unit's input rows parsed
(`NetworkCheck.CircuitFile.read`), where `network-period` reads it whole and trims it. Only the network suite's tests
build it, and no deployment runs it: it reads a small file written for the check that `Flock.ProvedScope.circuitOf`
refuses, so it tests the check (`NetworkCheck.Period.checkWith`), not the deployed reader. -/

/-- The files `files`, the last `whole` of them read whole (`NetworkCheck.CircuitFile.readWholes`, on one dedicated
thread) and the others in chunks (`NetworkCheck.CircuitFile.read`, a dedicated thread a file). -/
def readAll (files : List String) (whole : Nat) : IO (Except String (Array (ByteArray × Flock.Circuit))) := do
  let k := files.length - whole
  let tasks ← (files.take k).mapM fun (f : String) =>
    IO.asTask (NetworkCheck.CircuitFile.read f) Task.Priority.dedicated
  let wholes ← IO.asTask (NetworkCheck.CircuitFile.readWholes (files.drop k) k) Task.Priority.dedicated
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
    match ← readAll files (NetworkCheck.Period.scanCount (← NetworkCheck.Period.paramsAt p)) with
    | .ok fs => (NetworkCheck.Period.protocolWith fs).main ["check", p, c, e]
    | .error why => (NetworkCheck.Period.refusing s!"circuit files: {why}").main ["check", p, c, e]
  | _ => NetworkCheck.Period.protocol.main args
