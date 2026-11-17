module

public import InferenceOnlyVerifier.Record

/-! Scratch: `inference-only-wires-stages CASE` times the record reader's wire stages on one case, one line each. -/

@[expose] public section

open InferenceOnlyVerifier.Record Spot.RecordText Spot.Wires in
def main (args : List String) : IO UInt32 := do
  let [path] := args | IO.eprintln "usage: CASE"; return 2
  let t0 ← IO.monoMsNow
  let mut last := t0
  let said (what : String) (last : Nat) : IO Nat := do
    let now ← IO.monoMsNow
    IO.println s!"[{now - t0} ms, +{now - last} ms] {what}"
    (← IO.getStdout).flush
    return now
  let .ok c := Lean.Json.parse (← IO.FS.readFile path) | IO.eprintln "not JSON"; return 2
  let j := c.getObjValD "instance"
  let tr := c.getObjValD "trusted"
  last ← said "Json.parse" last
  let .ok f := front j tr | IO.eprintln "front refused"; return 1
  last ← said s!"front: {f.ts.length} templates" last
  let .ok text := recode "wires" (wiring f.doc) | IO.eprintln "wiring refused"; return 1
  last ← said s!"wiring: {text.wires.length} wires" last
  let .ok W := wiringOf f text | IO.eprintln "wiringOf refused"; return 1
  last ← said "wiringOf" last
  let cuts := cutsOk W
  last ← said s!"cutsOk {cuts}" last
  let fw := (firstWire W (wireCode W)).isNone
  last ← said s!"firstWire wireCode {fw}" last
  let pd := pairwiseDisjoint W.wires
  last ← said s!"pairwiseDisjoint {pd}" last
  let o := match statedOrder W with | .ok o => o | .error _ => []
  last ← said s!"statedOrder: {o.length} ranges, stated {W.order.isSome}" last
  let ok := orderOk W o
  last ← said s!"orderOk {ok}" last
  let ow := (firstWire W fun w => if orderedWire o w then none else some .order).isNone
  last ← said s!"orderedWire over the wires {ow}" last
  let ck := (check W).toBool
  last ← said s!"check {ck}" last
  match inputsOf f.ts text f.doc with
  | .ok x => last ← said s!"inputsOf: {x.decls.length} decls" last
  | .error e => last ← said s!"inputsOf refused {e}" last
  let r := (read j tr).toBool
  last ← said s!"read (whole) {r}" last
  return 0
