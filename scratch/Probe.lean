import InferenceOnlyVerifier.Record
open InferenceOnlyVerifier

/-- Which of `Record.front`'s `program` checks a Program fails, each named, with the decoder's own error string. -/
def main (args : List String) : IO UInt32 := do
  let [path] := args | IO.eprintln "usage: Probe PROGRAM"; return 2
  let t0 ← IO.monoMsNow
  let say (s : String) : IO Unit := do
    IO.println s!"[{((← IO.monoMsNow) - t0) / 1000} s] {s}"; (← IO.getStdout).flush
  let bytes ← IO.FS.readBinFile path
  say s!"read {bytes.size} bytes"
  match Flock.Program.decode bytes with
  | .error e => say s!"decode: REFUSED: {e}"; return 1
  | .ok P =>
    say s!"decode: ok, {P.defs.size} definitions, root {P.root}"
    say s!"orderViolation: {(P.orderViolation.map (·.compress)).getD "none"}"
    say s!"rootLaidOut: {Desc.rootLaidOut P}"
    say s!"inputsAtRoot: {Desc.inputsAtRoot P}"
    say s!"calleesFirst: {Desc.calleesFirst P}"
    say s!"bodiesLaidOut: {Desc.bodiesLaidOut P}"
    say s!"resolvable: {Desc.resolvable P}"
    say s!"walkBound: {Desc.walkBound P} against FUEL/2 = {Flock.Steps.FUEL / 2}"
    say s!"walkable: {Desc.walkable P}"
    return 0
