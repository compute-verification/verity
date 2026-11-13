import Spot

/-!
`whole-run-time run FILE [--no-whole] [--no-steps]`: the `run` piece (`Pieces.run`, the stream's whole run of one
registration, `Run.run`) on one input, timed after it is parsed, then `Run.run`'s steps one at a time in its order, with
`Slots.popOk`'s layout split into `Partition.evaluate`, `Partition.unitGates`, `Slots.inputGates` and `Slots.layout`.

`whole-run-time desc FILE [--unit-gates [--ref]] [--pop-ok]`: on a `{"program", "query"}` input, `Program.parse`,
`Partition.evaluate`, and per range one `Desc.gates` of its template and one `TemplateQuery.gates` of its last unit,
`inputGates`' count; with `--unit-gates`, `Partition.unitGates` too; with `--ref`, `Partition.unitGatesRef`, the walk unit
by unit `unitGates` replaced, whose gates must be `unitGates`' (exit 1 if not); with `--pop-ok`, last, `Slots.popOk`,
whose layout holds a proof block per input gate.

One `PHASE {json}` line per step on stdout. `whole_run.sh` builds it against the checkout's verifier and runs it; it is a
measurement, not a verifier.
-/

open Lean Spot

def statusKb (field : String) : IO Nat := do
  let s ← IO.FS.readFile "/proc/self/status"
  for l in s.splitOn "\n" do
    if l.startsWith field then
      return ((String.ofList ((l.toList.drop field.length).filter Char.isDigit)).toNat?).getD 0
  return 0

def isOk {α} : Flock.V α → Bool
  | .ok _ => true
  | .error _ => false

def emit (step : String) (ns : Nat) (extra : List (String × Json)) : IO Unit := do
  let hwm ← statusKb "VmHWM:"
  let row := Json.mkObj ([("step", Json.str step), ("ms", toJson (ns / 1000000)), ("us", toJson (ns / 1000)),
    ("hwm_mb", toJson (hwm / 1024))] ++ extra)
  IO.println s!"PHASE {row.compress}"
  (← IO.getStdout).flush

/-- `f x` and its time; `x` is read back from a reference after the clock so the compiler can't hoist `f x` out of the
timing, and `ok` reads the result before the second reading. -/
def timed {α β} (step : String) (x : α) (f : α → β) (ok : β → Bool)
    (extra : β → List (String × Json) := fun _ => []) : IO β := do
  let r ← IO.mkRef x
  let t0 ← IO.monoNanosNow
  let x ← r.get
  let y := f x
  if !ok y then IO.eprintln s!"{step}: not ok"
  let t1 ← IO.monoNanosNow
  emit step (t1 - t0) (extra y)
  return y

def strs (j : Json) : Array String := match j with
  | .arr xs => xs.filterMap (·.getStr?.toOption)
  | _ => #[]

def valuesOf (j : Json) : Array Service.Value := match j with
  | .arr vs => vs.filterMap fun v => match getD v "name", getD v "registered" with
    | .str name, .bool registered => some { name, registered }
    | _, _ => none
  | _ => #[]

def outcome : Flock.V Json → List (String × Json)
  | .ok o => [("symbol", getD o "symbol"), ("refused", getD o "step"), ("codes", getD o "codes")]
  | .error e => [("error", Json.str e)]

def runSteps (j : Json) (whole steps : Bool) : IO Unit := do
  if whole then
    let _ ← timed "run" j Pieces.run isOk outcome
  if !steps then return
  let p ← IO.ofExcept (← timed "params" (getD j "params")
    (fun pj => do OneStage.paramsParsed (← Core.canonicalJson (← Flock.canon pj).toUTF8)) isOk)
  let rule ← IO.ofExcept (Select.Rule.ofJson (getD j "rule"))
  let values := valuesOf (getD j "values")
  let declared := strs (getD j "public")
  let challenger := (getD j "challenger").getStr?.toOption.getD ""
  let received := (getD j "received").getStr?.toOption.getD ""
  let source ← IO.ofExcept (Flock.unhex ((getD j "source").getStr?.toOption.getD ""))
  let sessions := match getD j "sessions" with | .arr s => s | _ => #[]
  let verdicts := entries (getD j "verdicts")
  let outputs := strs (getD j "outputs")
  let rec_ := getD j "record"
  let v := p.verifier
  let qd ← IO.ofExcept (← timed "queryDigest" v.query Registration.queryDigest isOk)
  let _ ← timed "checkPublic" rec_ (fun r => Service.checkPublic r (Json.str v.scheme) qd values declared)
    (fun c => match c with | .ok c => c.codes.isEmpty | .error _ => false)
  let n := (nat? (getD rec_ "population")).getD 0
  let some rule' := rule.over n (Select.strataSizes v.program v.query)
    | throw (IO.userError "the rule's rate law refuses the population")
  let law ← IO.ofExcept (rule'.lawOver n)
  let v := { v with law }
  let _ ← timed "registration" rec_ (fun r => Registration.check r v) (·.isEmpty)
  let _ ← timed "partition.evaluate" v.program (fun pr => Partition.evaluate pr v.query) isOk
    (fun r => match r with | .ok rs => [("ranges", toJson rs.size)] | .error _ => [])
  let _ ← timed "partition.unitGates" v.program (fun pr => Partition.unitGates pr v.query) isOk
  let _ ← timed "slots.inputGates" v.program.desc Slots.inputGates (fun _ => true) (fun a => [("gates", toJson a.size)])
  let _ ← timed "slots.layout" v.program (fun pr => Slots.layout pr v.query) isOk
  let _ ← timed "popOk" v.program (fun pr => Slots.popOk pr v.query n) id
  let receipt ← IO.ofExcept (← timed "receipt" rec_ (fun r => Run.receipt r challenger received) isOk)
  let derived ← IO.ofExcept (← timed "derive" v.program (fun pr => Draw.derive pr v.query law v.workTable) isOk)
  let res ← IO.ofExcept (← timed "resolve" v.program (fun pr => Select.resolve rule pr v.query) isOk)
  let (draw, _, read) ← IO.ofExcept (← timed "select" v.program
    (fun pr => Select.select res pr v.query v.workTable none source) isOk)
  let _ ← timed "coinsHold" (getD j "coins") (fun c => Run.coinsHold c rec_ v.windowRoot (source.extract 0 read)) id
  let _ ← timed "checkDraw" draw (fun d => Draw.checkDraw d (getD rec_ "population") derived) (·.isEmpty)
  let proved := Draw.proved draw
  let _ ← timed "checkVerdicts" verdicts (fun vs => Service.checkVerdicts vs sessions outputs)
    (fun c => match c with | .ok c => c.codes.isEmpty | .error _ => false)
  let auditIn := Json.mkObj [("registration", rec_), ("receipt", receipt), ("draw", draw), ("law", derived),
    ("proved_units", Json.arr (proved.map toJson)), ("instances", toJson proved.size),
    ("statement", getD j "statement"), ("verdicts", Json.mkObj verdicts.toList), ("delta", getD j "delta"),
    ("sessions", Json.arr sessions), ("served", getD j "served"), ("session_roots", getD j "session_roots")]
  let ri ← IO.ofExcept (← timed "record.inputOf" auditIn Record.inputOf isOk)
  let recd ← IO.ofExcept (← timed "record.record" ri Record.record isOk)
  let _ ← timed "canonical" recd PyJson.canonical isOk

/-- `Slots.inputGates`' size, without the array. -/
def inputCount (d : Flock.TemplateQuery.Desc) : Nat := Id.run do
  let some root := d.def? d.root | return 0
  return root.nodes.foldl (fun acc x =>
    let tid := d.callees.getD x.fn ""
    if d.isInputNode x then acc + x.members * d.gates tid else acc) 0

def descSteps (j : Json) (unitGates ref popOk : Bool) : IO UInt32 := do
  let p ← IO.ofExcept (← timed "program.parse" (getD j "program")
    (fun pj => do Program.parse (← Flock.canon pj).toUTF8) isOk)
  let q := getD j "query"
  let d := p.desc
  let rs ← IO.ofExcept (← timed "partition.evaluate" p (fun pr => Partition.evaluate pr q) isOk
    (fun r => match r with | .ok rs => [("ranges", toJson rs.size)] | .error _ => []))
  let root := ((d.def? d.root).map (·.nodes.size)).getD 0
  let shape := Json.mkObj [("step", Json.str "shape"), ("defs", toJson d.defs.size), ("root_nodes", toJson root),
    ("units", toJson (Flock.TemplateQuery.population rs))]
  IO.println s!"PHASE {shape.compress}"
  for r in rs do
    let tag := [("template", Json.str r.template), ("node", toJson r.node), ("count", toJson r.count)]
    let _ ← timed "desc.gates" r.template (fun t => d.gates t) (fun _ => true) (fun g => tag ++ [("gates", toJson g)])
    let _ ← timed "templateQuery.gates" (r.first + r.count - 1) (fun u => Flock.TemplateQuery.gates d rs u) Option.isSome
      (fun _ => tag)
  let _ ← timed "inputGates.count" d inputCount (fun _ => true) (fun c => [("gates", toJson c)])
  if unitGates then
    let gs ← timed "partition.unitGates" p (fun pr => Partition.unitGates pr q) isOk
    if ref then
      let walk ← timed "partition.unitGatesRef" p (fun pr => Partition.unitGatesRef pr q) isOk
      let agree := match gs, walk with
        | .ok a, .ok b => a == b
        | .error a, .error b => a == b
        | _, _ => false
      IO.println s!"PHASE {(Json.mkObj [("step", Json.str "agree"), ("agree", toJson agree)]).compress}"
      if !agree then return 1
  if popOk then
    let n := Flock.TemplateQuery.population rs
    let _ ← timed "slots.popOk" p (fun pr => Slots.popOk pr q n) id (fun b => [("ok", toJson b)])
  return 0

def main (args : List String) : IO UInt32 := do
  match args with
  | "run" :: path :: flags =>
    let j ← IO.ofExcept (Json.parse (← IO.FS.readFile path))
    runSteps j (!flags.contains "--no-whole") (!flags.contains "--no-steps")
    return 0
  | "desc" :: path :: flags =>
    let j ← IO.ofExcept (Json.parse (← IO.FS.readFile path))
    descSteps j (flags.contains "--unit-gates") (flags.contains "--ref") (flags.contains "--pop-ok")
  | _ => do
    IO.eprintln "usage: whole-run-time run FILE [--no-whole] [--no-steps] | desc FILE [--unit-gates [--ref]] [--pop-ok]"
    return 2
