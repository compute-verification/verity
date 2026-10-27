import InferenceOnlyCheck.Baseline

/-! `inference-only-check [--unrolled] INSTANCE TRUSTED`: the verdict on one instance, as JSON on stdout. By default the instance is
`InferenceOnlyCheck.Desc`'s (θ and the owners per root node) with θ's `provenance`, checked at the size of the description and then
against the trusted input `TRUSTED` (the deployment's registration log, the run's record and the program function's
parameters: `InferenceOnlyCheck.Baseline`); `--unrolled` reads one of `InferenceOnlyCheck.Check` (per gate) and runs the unrolled
check, the reference the description-sized one is held to. `inference-only-check --policy [--unrolled] INSTANCE` runs the policy
check alone, without θ's provenance: a measurement of `InferenceOnlyCheck.Desc` (`benchmarks/verifier_cost/`), not the check of
record. -/

def main (args : List String) : IO UInt32 := do
  let usage := "usage: inference-only-check [--unrolled] INSTANCE TRUSTED | inference-only-check --policy [--unrolled] INSTANCE"
  let (policy, unrolled, paths) := match args with
    | "--policy" :: "--unrolled" :: ps => (true, true, ps)
    | "--policy" :: ps => (true, false, ps)
    | "--unrolled" :: ps => (false, true, ps)
    | ps => (false, false, ps)
  let read (path : String) : IO (Option Lean.Json) := do
    return (Lean.Json.parse (← IO.FS.readFile path)).toOption
  match policy, paths with
  | true, [p] =>
    let some j ← read p | IO.eprintln s!"inference-only-check: {p} is not JSON"; return 2
    IO.println (if unrolled then InferenceOnlyCheck.check j else InferenceOnlyCheck.Desc.check j).compress
    return 0
  | false, [p, t] =>
    let some j ← read p | IO.eprintln s!"inference-only-check: {p} is not JSON"; return 2
    let some tr ← read t | IO.eprintln s!"inference-only-check: {t} is not JSON"; return 2
    IO.println (if unrolled then InferenceOnlyCheck.Baseline.checkUnrolled j tr else InferenceOnlyCheck.Baseline.check j tr).compress
    return 0
  | _, _ => IO.eprintln usage; return 2
