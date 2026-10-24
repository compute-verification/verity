import SampledProofs

/-!
`registration-time FILE [--no-ref]`: `Registration.check` and `Registration.checkRef` (R3 as first written, quadratic in
the ports; `check_eq_checkRef` proves them equal) on one `{"params", "record"}` input, as the `registration` piece reads
it, each timed after the input is parsed, one `PHASE {json}` line each on stdout; exit 1 if their codes differ.
`registration.sh` builds it against the checkout's verifier and runs it; it is a measurement, not a verifier.
-/

open Lean SampledProofs

def statusKb (field : String) : IO Nat := do
  let s ← IO.FS.readFile "/proc/self/status"
  for l in s.splitOn "\n" do
    if l.startsWith field then
      return ((String.ofList ((l.toList.drop field.length).filter Char.isDigit)).toNat?).getD 0
  return 0

/-- `f`'s codes and its milliseconds; `f` reads the clock's reading so the compiler can't hoist it out of the timing. -/
def timed (f : Json → Registration.Verifier → Array String) (rec_ : Json) (v : Registration.Verifier) :
    IO (Array String × Nat) := do
  let t0 ← IO.monoNanosNow
  let v := if t0 == 0 then { v with ports := #[] } else v
  let cs := f rec_ v
  if cs.size == 1000000007 then IO.eprintln "unreachable"
  let t1 ← IO.monoNanosNow
  return (cs, (t1 - t0) / 1000000)

def main (args : List String) : IO UInt32 := do
  let (path, ref) ← match args with
    | [p] => pure (p, true)
    | [p, "--no-ref"] => pure (p, false)
    | _ => do IO.eprintln "usage: registration-time FILE [--no-ref]"; return 2
  let j ← IO.ofExcept (Json.parse (← IO.FS.readFile path))
  let p ← IO.ofExcept (OneStage.paramsOfJson (← IO.ofExcept (Core.canonicalJson (← IO.ofExcept (Flock.canon (getD j "params"))).toUTF8)))
  let rec_ := getD j "record"
  let v := p.verifier
  let roots := match getD rec_ "roots" with | .arr rs => rs.size | _ => 0
  let mut got : Array (Array String) := #[]
  for (name, f) in [("check", Registration.check), ("checkRef", Registration.checkRef)] do
    if name == "checkRef" && !ref then continue
    let (cs, ms) ← timed f rec_ v
    got := got.push cs
    let hwm ← statusKb "VmHWM:"
    let row := Json.mkObj [("fn", Json.str name), ("ports", toJson v.ports.size), ("roots", toJson roots),
      ("ms", toJson ms), ("hwm_mb", toJson (hwm / 1024)), ("codes", toJson cs)]
    IO.println s!"PHASE {row.compress}"
    (← IO.getStdout).flush
  if got.any (· != got[0]!) then
    IO.eprintln "check and checkRef disagree"
    return 1
  return 0
