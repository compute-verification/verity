import Difftest
import SampledProofs

/-!
`calls-time FILE`: on one `registration` request line as `one_stage.verifier` writes it, each step
`sampled-proofs-verifier` takes on it, timed apart: the line's `Json.parse`; the parameters' round trip through their
canonical text (`Flock.canon`, then `Core.canonicalJson`: parse and canonical check); the program's canonical text and
its two decodings (`Flock.TemplateQuery.Desc.parse`, `Flock.Program.decode`); `OneStage.paramsOfJson`, which decodes the
program again; and `Registration.check`. Then the whole request (`Difftest.respond` on the verifier's pieces), the
program loaded once (`Loaded.Store.load`), and the same request naming the loaded program by digest and annotations
(`Verifier.respondAt`), whose answer must be the whole request's (exit 1 otherwise). One `PHASE {json}` line each on
stdout. `calls.sh` builds it against the checkout's verifier and runs it; it is a measurement, not a verifier.
-/

open Lean SampledProofs

def statusKb (field : String) : IO Nat := do
  let s ← IO.FS.readFile "/proc/self/status"
  for l in s.splitOn "\n" do
    if l.startsWith field then
      return ((String.ofList ((l.toList.drop field.length).filter Char.isDigit)).toNat?).getD 0
  return 0

/-- `f`'s value and its milliseconds; `f` reads the clock's reading and `size` reads its value before the clock stops,
so the compiler can't move it out of the timing. -/
def timed {α : Type} (f : Nat → α) (size : α → Nat) : IO (α × Nat) := do
  let t0 ← IO.monoNanosNow
  let a := f t0
  if size a == 1000000007 then IO.eprintln "unreachable"
  let t1 ← IO.monoNanosNow
  return (a, (t1 - t0) / 1000000)

def row (fn : String) (ms : Nat) (extra : List (String × Json)) : IO Unit := do
  let hwm ← statusKb "VmHWM:"
  IO.println s!"PHASE {(Json.mkObj ([("fn", Json.str fn), ("ms", toJson ms), ("hwm_mb", toJson (hwm / 1024))] ++ extra)).compress}"
  (← IO.getStdout).flush

def ok {α : Type} : Except String α → Nat
  | .ok _ => 1
  | .error _ => 0

def main (args : List String) : IO UInt32 := do
  let [path] := args | do IO.eprintln "usage: calls-time FILE"; return 2
  let line ← IO.FS.readFile path
  row "read" 0 [("bytes", toJson line.utf8ByteSize)]
  let (req, ms) ← timed (fun t => if t == 0 then .error "" else Json.parse line) ok
  let req ← IO.ofExcept req
  row "parse_line" ms []
  let input := getD req "input"
  let (text, ms) ← timed (fun t => if t == 0 then .error "" else Flock.canon (getD input "params")) ok
  let text ← IO.ofExcept text
  row "canon_params" ms [("bytes", toJson text.utf8ByteSize)]
  let (pj, ms) ← timed (fun t => if t == 0 then .error "" else Core.canonicalJson text.toUTF8) ok
  let pj ← IO.ofExcept pj
  row "canonical_params" ms []
  let (prog, ms) ← timed (fun t => if t == 0 then .error "" else Flock.canon (getD pj "program")) ok
  let prog ← IO.ofExcept prog
  row "canon_program" ms [("bytes", toJson prog.utf8ByteSize)]
  let (desc, ms) ← timed (fun t => if t == 0 then .error "" else Flock.TemplateQuery.Desc.parse prog.toUTF8) ok
  let desc ← IO.ofExcept desc
  row "desc_parse" ms [("defs", toJson desc.defs.size)]
  let (dec, ms) ← timed (fun t => if t == 0 then .error "" else Flock.Program.decode prog.toUTF8) ok
  let _ ← IO.ofExcept dec
  row "program_decode" ms []
  let (params, ms) ← timed (fun t => if t == 0 then .error "" else OneStage.paramsOfJson pj) ok
  let params ← IO.ofExcept params
  row "params_of_json" ms []
  let (codes, ms) ← timed (fun t => if t == 0 then #[] else Registration.check (getD input "record") params.verifier)
    (·.size)
  row "registration_check" ms [("codes", toJson codes)]
  let table := Difftest.table Pieces.verifier
  let (out, ms) ← timed (fun t => if t == 0 then "" else Difftest.respond table line) (·.length)
  row "respond" ms [("answer", Json.str out)]
  let pIn := getD (getD input "params") "program"
  let (loaded, ms) ← timed (fun t => if t == 0 then .error "" else Loaded.Store.load #[] pIn) ok
  let (S, h) ← IO.ofExcept loaded
  row "load" ms [("sha512", Json.str h)]
  let named := Json.mkObj [("annotations", getD pIn "annotations"), ("loaded", Json.str h)]
  let refLine := (req.setObjVal! "input" (input.setObjVal! "params" ((getD input "params").setObjVal! "program" named))).compress
  let t0 ← IO.monoNanosNow
  let (refOut, _) ← Verifier.respondAt S refLine
  let t1 ← IO.monoNanosNow
  row "respond_ref" ((t1 - t0) / 1000000) [("bytes", toJson refLine.utf8ByteSize), ("answer", Json.str refOut),
    ("same", toJson (refOut == out))]
  return if refOut == out then 0 else 1
