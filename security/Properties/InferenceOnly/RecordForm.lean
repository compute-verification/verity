module

public import Properties.InferenceOnly.Record
public import InferenceOnlyCheck.Record

/-!
# The record check's own form

`form` is the record check (`Record`) as a `Properties.Record.Form`: the functions the record check exports, each the
check's own. Its instance is the one `read` parsed, its wiring, slots and held rows the ones `read` gives, an
opened row's output row the `row` its opening declares, its summary the two counts of `summarize` that the rate rule
reads, its rate rule `failure` on that one pair, and its layout `layoutOf`'s. What it carries and rates is
`Proofs.Properties.InferenceOnly.RecordForm`'s.
-/

@[expose] public section

namespace InferenceOnlyCheck.Record.SecurityProofs

open Spot.RecordText
open Lean (Json)

/-- **The record check's form**: what `Record` exports, as the statements about a record check read it. -/
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
