import NetworkCheck.Period
import NetworkCheck.CircuitFile

/-! # A period's listed circuit files, read within a budget

`read b j files` reads the circuit files that a period's parameters `j` list, in the order
`network-period check PARAMS COMMITMENT EVIDENCE CIRCUIT…` takes them, for the period's check with its files
(`NetworkCheck.Period.protocolWith`). `network-period` (`NetworkCheck.PeriodMain`) reads them at `Budget.period`, and a
protocol that composes the period's reads them at its own budget. Every file is read whole, as a session's verifier
parses it (`NetworkCheck.CircuitFile.readWhole`). The last files, as many as `j`'s `scan_circuits` lists
(`NetworkCheck.Period.scanCount`), are the scan's, held whole, one after another so one file's text is held at a time
(`NetworkCheck.CircuitFile.readWholes`). Every other file is an audit's, held trimmed to its header and its unit's input
rows (`NetworkCheck.Period.trim`), at most `b.files` files and `b.bytes` bytes of them read at a time, since a file's
text and parse are held until it is trimmed. A file that can't be read fails the read, which names the first such file
by its place in the listing. -/

namespace NetworkCheck.PeriodFiles

open Lean (Json)

/-- **How much of a period's audits' circuit files is read at once**: at most `files` files and `bytes` bytes of them,
but for a file larger than `bytes`, read alone, and one file at a time at `files` 0. Each file's text and its whole
parse, about four times the file, are held until it is trimmed (`NetworkCheck.Period.trim`). -/
structure Budget where
  files : Nat
  bytes : Nat

/-- `network-period`'s budget: 3 files and 48 GB at once. -/
def Budget.period : Budget := ⟨3, 48 * 10 ^ 9⟩

/-- The file `f`'s size in bytes, 0 where it has none (its read then says why). -/
def fileBytes (f : String) : IO Nat :=
  tryCatch (return (← System.FilePath.metadata f).byteSize.toNat) fun _ => return 0

/-- The file `f`, `i` its place and `n` its size, read whole and trimmed (`NetworkCheck.Period.trim`), or why not. -/
def readTrimmed (f : String) (i n : Nat) : IO (Nat × Nat × Except String (ByteArray × Flock.Circuit)) := do
  let r ← tryCatch (NetworkCheck.CircuitFile.readWhole f) (fun e => pure (.error (toString e)))
  return (i, n, r.map fun (sha, cf) => (sha, NetworkCheck.Period.trim cf))

/-- The files `files` read whole and trimmed (`readTrimmed`) in their order, each on a dedicated thread, as many at once
as `b` allows; or the first that can't be read, its place and why. A file is started only while none has failed, so
every file before a failed one has been read. -/
def readAudits (b : Budget) (files : List String) : IO (Except String (Array (ByteArray × Flock.Circuit))) := do
  let mut todo := (← files.mapM fun f => return (f, ← fileBytes f)).zipIdx
  let mut running : List (Task (Except IO.Error (Nat × Nat × Except String (ByteArray × Flock.Circuit)))) := []
  let mut held := 0
  let mut got : Array (Nat × ByteArray × Flock.Circuit) := #[]
  let mut bad : Option (Nat × String) := none
  repeat
    if let ((f, n), i) :: rest := todo then
      if bad.isNone && (running.isEmpty || running.length < b.files && held + n ≤ b.bytes) then
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

/-- **The circuit files `files` that the period's parameters `j` list**, each file's SHA-512 and circuit, in order:
the scan's, the last `NetworkCheck.Period.scanCount j`, read whole (`NetworkCheck.CircuitFile.readWholes`, on one
dedicated thread), and the audits' read whole and trimmed within `b` (`readAudits`); or why a file can't be read. -/
def read (b : Budget) (j : Json) (files : List String) : IO (Except String (Array (ByteArray × Flock.Circuit))) := do
  let k := files.length - NetworkCheck.Period.scanCount j
  let wholes ← IO.asTask (NetworkCheck.CircuitFile.readWholes (files.drop k) k) Task.Priority.dedicated
  let audits ← readAudits b (files.take k)
  match audits, ← IO.wait wholes with
  | .error why, _ => return .error why
  | .ok fs, .ok (.ok xs) => return .ok (fs ++ xs)
  | .ok _, .ok (.error why) => return .error why
  | .ok _, .error e => return .error s!"circuit file {k}: {e}"

end NetworkCheck.PeriodFiles
