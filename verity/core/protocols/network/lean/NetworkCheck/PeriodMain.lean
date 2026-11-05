import NetworkCheck.Period
import NetworkCheck.CircuitFile

/-! The `network-period` executable: `Core.Protocol.main` of `NetworkCheck.Period.protocol`, audited under `NetworkCheck`,
and `check PARAMS COMMITMENT EVIDENCE CIRCUIT…`, the period's check with its listed circuit files
(`NetworkCheck.Period.protocolWith`). Every file is read whole, as a session's verifier parses it
(`NetworkCheck.CircuitFile.readWhole`). The last files, as many as the parameters' `scan_circuits` lists
(`NetworkCheck.Period.scanFiles`), are the scan's, held whole, one after another so one file's text is held at a time
(`NetworkCheck.CircuitFile.readWholes`). Every other file is an audit's, held trimmed to its header and its unit's input
rows (`NetworkCheck.Period.trim`), read at most `auditsAtOnce` and `auditBytesAtOnce` at a time, since a file's text and
parse are held until it is trimmed. A file that can't be read rejects the period, naming the first such file by its place
in the listing. The test-only `network-period-chunked` (`NetworkCheck.PeriodChunkedMain`) reads the audits' files in
chunks instead. -/

/-- How many audits' files are read at once, at most. -/
def auditsAtOnce : Nat := 3

/-- How many bytes of audits' files are read at once, at most, but for a larger file read alone: each one's text and
its whole parse, about four times the file, are held until it is trimmed (`NetworkCheck.Period.trim`). -/
def auditBytesAtOnce : Nat := 48 * 10 ^ 9

/-- The file `f`'s size in bytes, 0 where it has none (its read then says why). -/
def fileBytes (f : String) : IO Nat :=
  tryCatch (return (← System.FilePath.metadata f).byteSize.toNat) fun _ => return 0

/-- The file `f`, `i` its place and `n` its size, read whole and trimmed (`NetworkCheck.Period.trim`), or why not. -/
def readTrimmed (f : String) (i n : Nat) : IO (Nat × Nat × Except String (ByteArray × Flock.Circuit)) := do
  let r ← tryCatch (NetworkCheck.CircuitFile.readWhole f) (fun e => pure (.error (toString e)))
  return (i, n, r.map fun (sha, cf) => (sha, NetworkCheck.Period.trim cf))

/-- The files `files` read whole and trimmed (`readTrimmed`) in their order, each on a dedicated thread, as many at once
as `auditsAtOnce` and `auditBytesAtOnce` allow; or the first that can't be read, its place and why. A file is started
only while none has failed, so every file before a failed one has been read. -/
def readAudits (files : List String) : IO (Except String (Array (ByteArray × Flock.Circuit))) := do
  let mut todo := (← files.mapM fun f => return (f, ← fileBytes f)).zipIdx
  let mut running : List (Task (Except IO.Error (Nat × Nat × Except String (ByteArray × Flock.Circuit)))) := []
  let mut held := 0
  let mut got : Array (Nat × ByteArray × Flock.Circuit) := #[]
  let mut bad : Option (Nat × String) := none
  repeat
    if let ((f, n), i) :: rest := todo then
      if bad.isNone && running.length < auditsAtOnce && (running.isEmpty || held + n ≤ auditBytesAtOnce) then
        running := (← IO.asTask (readTrimmed f i n) Task.Priority.dedicated) :: running
        held := held + n
        todo := rest
        continue
    match running with
    | [] => break
    | t :: ts =>
      let (r, rest) ← IO.waitAny' (t :: ts)
      running := rest
      match r with
      | .ok (i, n, .ok x) => held := held - n; got := got.push (i, x)
      | .ok (i, n, .error why) => held := held - n; if bad.all (i < ·.1) then bad := some (i, why)
      | .error e => return .error s!"circuit files: {e}"
  if let some (i, why) := bad then return .error s!"circuit file {i}: {why}"
  return .ok ((got.qsort (·.1 < ·.1)).map (·.2))

/-- The files `files`, the last `whole` of them read whole (`NetworkCheck.CircuitFile.readWholes`, on one dedicated
thread) and the others read whole and trimmed (`readAudits`). -/
def readAll (files : List String) (whole : Nat) : IO (Except String (Array (ByteArray × Flock.Circuit))) := do
  let k := files.length - whole
  let wholes ← IO.asTask (NetworkCheck.CircuitFile.readWholes (files.drop k) k) Task.Priority.dedicated
  let audits ← readAudits (files.take k)
  match audits, ← IO.wait wholes with
  | .error why, _ => return .error why
  | .ok fs, .ok (.ok xs) => return .ok (fs ++ xs)
  | .ok _, .ok (.error why) => return .error why
  | .ok _, .error e => return .error s!"circuit file {k}: {e}"

def main (args : List String) : IO UInt32 := do
  match args with
  | "check" :: p :: c :: e :: files@(_ :: _) =>
    match ← readAll files (← NetworkCheck.Period.scanFiles p) with
    | .ok fs => (NetworkCheck.Period.protocolWith fs).main ["check", p, c, e]
    | .error why => (NetworkCheck.Period.refusing s!"circuit files: {why}").main ["check", p, c, e]
  | _ => NetworkCheck.Period.protocol.main args
