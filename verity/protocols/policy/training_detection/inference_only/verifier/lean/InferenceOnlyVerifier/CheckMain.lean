module

public import InferenceOnlyVerifier.Theta
public import InferenceOnlyVerifier.Count

/-! `inference-only-verifier [--unrolled] INSTANCE TRUSTED`: the verdict on one instance, as JSON on stdout. By default the instance is
`InferenceOnlyVerifier.Desc`'s (θ and the owners per root node) with θ's `provenance`, checked at the size of the description and then
against the trusted input `TRUSTED` (the deployment's registration log, the run's record and the program function's
parameters: `InferenceOnlyVerifier.Theta`); `--unrolled` reads one of `InferenceOnlyVerifier.Check` (per gate) and runs the unrolled
check, the reference the description-sized one is held to. `inference-only-verifier --policy [--unrolled] INSTANCE` runs the policy
check alone, without θ's provenance: a measurement of `InferenceOnlyVerifier.Desc` (`benchmarks/verifier_cost/`), not the check of
record. `inference-only-verifier --count INSTANCE` runs the count of `(c, k)`-inference blocks on one of `InferenceOnlyVerifier.Count` and prints
what it publishes, the judgment against `n` (`InferenceOnlyVerifier.Count.check`); `--count --detail INSTANCE` prints the whole
verdict (`InferenceOnlyVerifier.Count.checkDetail`), for the private record only and never published. -/

@[expose] public section

def main (args : List String) : IO UInt32 := do
  let usage := "usage: inference-only-verifier [--unrolled] INSTANCE TRUSTED | inference-only-verifier --policy [--unrolled] INSTANCE | " ++
    "inference-only-verifier --count [--detail] INSTANCE"
  let read (path : String) : IO (Option Lean.Json) := do
    return (Lean.Json.parse (← IO.FS.readFile path)).toOption
  if let "--count" :: rest := args then
    let (detail, ps) := match rest with
      | "--detail" :: ps => (true, ps)
      | ps => (false, ps)
    let [p] := ps | IO.eprintln usage; return 2
    let some j ← read p | IO.eprintln s!"inference-only-verifier: {p} is not JSON"; return 2
    IO.println (if detail then InferenceOnlyVerifier.Count.checkDetail j else InferenceOnlyVerifier.Count.check j).compress
    return 0
  let (policy, unrolled, paths) := match args with
    | "--policy" :: "--unrolled" :: ps => (true, true, ps)
    | "--policy" :: ps => (true, false, ps)
    | "--unrolled" :: ps => (false, true, ps)
    | ps => (false, false, ps)
  match policy, paths with
  | true, [p] =>
    let some j ← read p | IO.eprintln s!"inference-only-verifier: {p} is not JSON"; return 2
    IO.println (if unrolled then InferenceOnlyVerifier.check j else InferenceOnlyVerifier.Desc.check j).compress
    return 0
  | false, [p, t] =>
    let some j ← read p | IO.eprintln s!"inference-only-verifier: {p} is not JSON"; return 2
    let some tr ← read t | IO.eprintln s!"inference-only-verifier: {t} is not JSON"; return 2
    IO.println (if unrolled then InferenceOnlyVerifier.Theta.checkUnrolled j tr else InferenceOnlyVerifier.Theta.check j tr).compress
    return 0
  | _, _ => IO.eprintln usage; return 2
