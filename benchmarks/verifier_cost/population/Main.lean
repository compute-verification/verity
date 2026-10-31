import SampledProofs

/-!
`population-time FILE [--no-ref]`: on one `{"descriptor", "query", "population"}` input, `Partition.unitGates` (which runs
as `unitGatesFast`), `Slots.popOk` (the run's `population-order` check, which runs from the ranges as
`Slots.popOkRanges`) and, unless `--no-ref`, the per-unit reference (`TemplateQuery.gates` on every unit, as `unitGates`
first ran) and `popOk` as stated, off `Slots.layout`'s population, at the population and either side of it, each timed
after the program is parsed, one `PHASE {json}` line each on stdout; exit 1 if the reference's gates differ from
`unitGates`', the layout's `popOk` from the ranges', or `popOk` fails. `population.sh` builds it against the checkout's verifier and runs it; it is a measurement, not a verifier.
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

def row (fn : String) (units ms : Nat) (extra : List (String × Json)) : IO Unit := do
  let hwm ← statusKb "VmHWM:"
  IO.println s!"PHASE {(Json.mkObj ([("fn", Json.str fn), ("units", toJson units), ("ms", toJson ms),
    ("hwm_mb", toJson (hwm / 1024))] ++ extra)).compress}"
  (← IO.getStdout).flush

def main (args : List String) : IO UInt32 := do
  let (path, ref) ← match args with
    | [p] => pure (p, true)
    | [p, "--no-ref"] => pure (p, false)
    | _ => do IO.eprintln "usage: population-time FILE [--no-ref]"; return 2
  let j ← IO.ofExcept (Json.parse (← IO.FS.readFile path))
  let p ← IO.ofExcept (Program.parse (← IO.ofExcept (Flock.canon (getD j "descriptor"))).toUTF8)
  let q := getD j "query"
  let n := (getD j "population").getNat?.toOption.getD 0
  let rs ← IO.ofExcept (Partition.evaluate p q)
  let pop := Flock.TemplateQuery.population rs
  let (gs, ms) ← timed (fun t => if t == 0 then .error "unreachable" else Partition.unitGates p q)
    fun r => match r with | .ok a => a.size | .error _ => 0
  let gs ← IO.ofExcept gs
  row "unitGates" pop ms [("defs", toJson p.desc.defs.size), ("ranges", toJson rs.size)]
  let (ok, ms) ← timed (fun t => t != 0 && Slots.popOk p q n) fun b => if b then 1 else 0
  row "popOk" pop ms [("ok", toJson ok)]
  if !ok then
    IO.eprintln "popOk refused the population"
    return 1
  if ref then
    let (ref, ms) ← timed (fun t => if t == 0 then none else
      (Array.range pop).mapM fun u => Flock.TemplateQuery.gates p.desc rs u) fun r => (r.map (·.size)).getD 0
    let agree := ref == some gs
    row "reference" pop ms [("agree", toJson agree)]
    if !agree then
      IO.eprintln "the per-unit reference and unitGates disagree"
      return 1
    for m in [n, n - 1, n + 1] do
      let (lay, ms) ← timed (fun t => t != 0 && match Slots.layout p q with
        | .error _ => false
        | .ok l => decide ((Slots.popOf l).map (·.unit) = (List.range m).map some)) fun b => if b then 1 else 0
      let ranges := Slots.popOk p q m
      row "popOkLayout" m ms [("ok", toJson lay), ("agree", toJson (lay == ranges))]
      if lay != ranges then
        IO.eprintln s!"popOk from the ranges and from the layout disagree at {m} units"
        return 1
  return 0
