import NciCheck.Desc

/-! `nci-check [--unrolled] INSTANCE`: the verdict on one instance, as JSON on stdout. By default the instance is
`NciCheck.Desc`'s (θ and the owners per root node), checked at the size of the description; `--unrolled` reads one of
`NciCheck.Check` (per gate) and runs the unrolled check, the reference the description-sized one is held to. -/

def main (args : List String) : IO UInt32 := do
  let (unrolled, path) ← match args with
    | ["--unrolled", p] => pure (true, p)
    | [p] => pure (false, p)
    | _ => do IO.eprintln "usage: nci-check [--unrolled] INSTANCE"; return 2
  let some j := (Lean.Json.parse (← IO.FS.readFile path)).toOption
    | IO.eprintln s!"nci-check: {path} is not JSON"; return 2
  IO.println (if unrolled then NciCheck.check j else NciCheck.Desc.check j).compress
  return 0
