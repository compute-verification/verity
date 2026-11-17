module

public import Properties.InferenceOnly.Record
public import InferenceOnlyCheck.Record

/-! The record check's own form. -/

@[expose] public section

namespace InferenceOnlyCheck.Record.SecurityProofs

open Spot.RecordText
open Lean (Json)

def form : Properties.Record.Form where
  Case := Record.Case
  read := Record.read
  inst c := ⟨c.f.i.program, c.f.i.record, c.f.i.gamma, c.f.i.num, c.f.i.den, c.f.i.pins.toList⟩
  wiring c := c.W
  slot := Record.Case.slot
  held := Record.Case.heldAt
  openedRow c r := match fillAt c.x.decls r with
    | some (.opening i) => c.x.openings[i]?.bind (·.row)
    | _ => none
  summarize c := ⟨(Record.summarize c).gates, (Record.summarize c).fresh⟩
  rate Γ num den s := (failure Γ num den #[(s.gates, s.fresh)]).map (·.2)
  layout d := (Record.layoutOf d).map fun l => ⟨l.key, l.root, l.domain, l.ports, l.verdict⟩

end InferenceOnlyCheck.Record.SecurityProofs
