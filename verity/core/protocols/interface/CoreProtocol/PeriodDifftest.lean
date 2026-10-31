import Difftest
import CoreProtocol.Period

/-!
# The periods difftest: `CoreProtocol.Period`'s functions behind `Difftest.serve`

`core-periods-difftest` serves `table` to the core difftest harness (`verity/core/difftest`), whose Python half
(`verity/core/periods/tests/test_periods_difftest.py`) holds each function to `verity.core.periods`:

* `span`: a span read from its JSON, and its stop;
* `within`: whether span `inner` lies within span `outer`;
* `cut`: the parts of the cut of `span` at `boundaries`, refused unless they cut it;
* `monitoring`: a monitoring period read from its JSON (refused unless valid), its execution blocks' spans and each
  execution block's sampling blocks' spans.
-/

namespace Core.PeriodDifftest

open Lean (Json)
open Flock (V)

private def spans (ps : List TickSpan) : Json := Json.arr (ps.toArray.map TickSpan.toJson)

def span (i : Json) : V Json := do
  let p ← TickSpan.ofJson i
  return Json.mkObj [("span", p.toJson), ("stop", Lean.toJson p.stop)]

def within (i : Json) : V Json := do
  let inner ← TickSpan.ofJson (← Flock.J.get i "inner" "input")
  let outer ← TickSpan.ofJson (← Flock.J.get i "outer" "input")
  return Json.mkObj [("within", Lean.toJson (inner.within outer))]

def cut (i : Json) : V Json := do
  let p ← TickSpan.ofJson (← Flock.J.get i "span" "input")
  let bs ← (← Flock.J.arr i "boundaries" "input").mapM (Flock.J.asNat · "input.boundaries")
  if !p.isCut bs then throw "the boundaries don't cut the span"
  return Json.mkObj [("parts", spans (parts bs))]

def monitoring (i : Json) : V Json := do
  let m ← MonitoringPeriod.ofJson i
  return Json.mkObj [("execution", spans m.executionSpans),
    ("sampling", Json.arr (m.samplingSpans.toArray.map spans))]

def table := _root_.Difftest.table [("span", span), ("within", within), ("cut", cut), ("monitoring", monitoring)]

end Core.PeriodDifftest
