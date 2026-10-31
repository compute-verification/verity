/-
Ticks, spans of ticks and the monitoring period, as the code reads them: the code packages' copy of
`Security/Definitions/Core/Period.lean` (`Verity.Tick`, `Verity.TickSpan`, `Verity.MonitoringPeriod`), on Lean core and
C-Flock's canonical JSON, so a verifier reads them without requiring `security`. `Properties.Core.Period` states that the
copy is the spec's, and `verity.core.periods` is its Python mirror, which `core-periods-difftest` holds to it.

* A span of ticks (`TickSpan`): the ticks `[start, start + len)` on the one clock (`OneClock`), ending at `stop`;
  `within` decides that one span lies inside another.
* A cut of a span: its boundaries, from the span's start to its stop, each no later than the next (`TickSpan.isCut`).
  Its parts (`parts`) are the spans between consecutive boundaries.
* A monitoring period (`MonitoringPeriod`): a span, the boundaries of its cut into its execution blocks' spans, and for
  each execution block the boundaries of its span's cut into its sampling blocks' spans. `valid` checks every cut.

The JSON: a span is `{"len": n, "start": n}`; a monitoring period is `{"execution": [τ, …], "sampling": [[τ, …], …],
"span": SPAN}`. Each reader accepts exactly the keys its writer writes.
-/
import Lean.Data.Json
import Flock.Canon

namespace Core

open Lean (Json)
open Flock (V)

/-- A tick: one step of the clock, the device model's unit of time. -/
abbrev Tick : Type := Nat

/-- A span of ticks: the ticks `[start, start + len)`. -/
structure TickSpan where
  start : Tick
  len : Nat
  deriving BEq, DecidableEq, Repr

namespace TickSpan

/-- The tick after a span's last: `start + len`. -/
def stop (p : TickSpan) : Tick := p.start + p.len

/-- The ticks of a span, in order. -/
def ticks (p : TickSpan) : List Tick := List.range' p.start p.len

/-- `p` lies within `q`: it starts no earlier and stops no later. -/
def within (p q : TickSpan) : Bool := q.start ≤ p.start && p.stop ≤ q.stop

/-- Each boundary is no later than the next. -/
def ordered : List Tick → Bool
  | a :: b :: rest => a ≤ b && ordered (b :: rest)
  | _ => true

/-- `bs` cuts `p`: its first boundary is `p`'s start, its last `p`'s stop, and each is no later than the next. -/
def isCut (p : TickSpan) (bs : Array Tick) : Bool :=
  bs.toList.head? == some p.start && bs.toList.getLast? == some p.stop && ordered bs.toList

end TickSpan

/-- The parts of a cut with boundaries `bs`: the span from each boundary to the next. -/
def parts (bs : Array Tick) : List TickSpan :=
  List.zipWith (fun a b => ⟨a, b - a⟩) bs.toList bs.toList.tail

/-- A monitoring period: its span, the boundaries of its cut into its execution blocks' spans, and for each execution block
the boundaries of its span's cut into its sampling blocks' spans. -/
structure MonitoringPeriod where
  span : TickSpan
  execution : Array Tick
  sampling : Array (Array Tick)
  deriving BEq, DecidableEq, Repr

namespace MonitoringPeriod

/-- The execution blocks' spans. -/
def executionSpans (m : MonitoringPeriod) : List TickSpan := parts m.execution

/-- Each execution block's sampling blocks' spans. -/
def samplingSpans (m : MonitoringPeriod) : List (List TickSpan) := m.sampling.toList.map parts

/-- Every cut is one: the execution boundaries cut the span, there is one sampling cut per execution block, and each cuts
its execution block's span. -/
def valid (m : MonitoringPeriod) : Bool :=
  m.span.isCut m.execution && m.sampling.size + 1 == m.execution.size &&
    (List.zipWith TickSpan.isCut m.executionSpans m.sampling.toList).all id

end MonitoringPeriod

/-! ## JSON -/

private def keys (j : Json) (want : List String) (ctx : String) : V Unit :=
  match j with
  | .obj kvs => if kvs.keys == want then pure () else throw s!"{ctx}: the keys must be {want}"
  | _ => throw s!"{ctx}: not an object"

private def ticksOf (j : Json) (ctx : String) : V (Array Tick) := do
  (← Flock.J.asArr j ctx).mapM (Flock.J.asNat · ctx)

namespace TickSpan

def toJson (p : TickSpan) : Json := Json.mkObj [("len", Lean.toJson p.len), ("start", Lean.toJson p.start)]

/-- The span `toJson` writes, and nothing else. -/
def ofJson (j : Json) : V TickSpan := do
  keys j ["len", "start"] "span"
  return ⟨← Flock.J.nat j "start" "span", ← Flock.J.nat j "len" "span"⟩

end TickSpan

namespace MonitoringPeriod

def toJson (m : MonitoringPeriod) : Json :=
  let nats := fun (bs : Array Tick) => Json.arr (bs.map Lean.toJson)
  Json.mkObj [("execution", nats m.execution), ("sampling", Json.arr (m.sampling.map nats)),
    ("span", m.span.toJson)]

/-- The monitoring period `toJson` writes, refused unless it is `valid`. -/
def ofJson (j : Json) : V MonitoringPeriod := do
  keys j ["execution", "sampling", "span"] "monitoring period"
  let m : MonitoringPeriod := {
    span := ← TickSpan.ofJson (← Flock.J.get j "span" "monitoring period")
    execution := ← ticksOf (← Flock.J.get j "execution" "monitoring period") "monitoring period.execution"
    sampling := ← (← Flock.J.arr j "sampling" "monitoring period").mapM (ticksOf · "monitoring period.sampling") }
  if m.valid then return m else throw "monitoring period: a cut is not one"

end MonitoringPeriod

#guard (TickSpan.mk 3 4).stop == 7
#guard (TickSpan.mk 3 4).ticks == [3, 4, 5, 6]
#guard (TickSpan.mk 3 2).within ⟨0, 10⟩ && !(TickSpan.mk 9 2).within ⟨0, 10⟩ && (TickSpan.mk 0 10).within ⟨0, 10⟩
#guard TickSpan.isCut ⟨0, 10⟩ #[0, 4, 4, 10] && TickSpan.isCut ⟨5, 0⟩ #[5]
#guard !TickSpan.isCut ⟨0, 10⟩ #[0, 11, 10] && !TickSpan.isCut ⟨0, 10⟩ #[] && !TickSpan.isCut ⟨0, 10⟩ #[1, 10]
#guard parts #[0, 4, 10] == [⟨0, 4⟩, ⟨4, 6⟩] && parts #[7] == []
#guard (MonitoringPeriod.mk ⟨0, 10⟩ #[0, 4, 10] #[#[0, 4], #[4, 5, 10]]).valid
#guard !(MonitoringPeriod.mk ⟨0, 10⟩ #[0, 4, 10] #[#[0, 4]]).valid
#guard !(MonitoringPeriod.mk ⟨0, 10⟩ #[0, 4, 10] #[#[0, 4], #[5, 10]]).valid
#guard (TickSpan.ofJson (TickSpan.toJson ⟨3, 4⟩)).toOption == some ⟨3, 4⟩
#guard (TickSpan.ofJson (Json.mkObj [("len", 4), ("start", 3), ("clock", "a")])).toOption.isNone
#guard (TickSpan.ofJson (Json.mkObj [("len", Lean.toJson (-1 : Int)), ("start", 3)])).toOption.isNone
#guard let m := MonitoringPeriod.mk ⟨0, 10⟩ #[0, 4, 10] #[#[0, 4], #[4, 5, 10]]
  (MonitoringPeriod.ofJson m.toJson).toOption == some m
#guard (MonitoringPeriod.ofJson (MonitoringPeriod.mk ⟨0, 10⟩ #[0, 10] #[]).toJson).toOption.isNone

end Core
