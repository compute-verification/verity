import Mathlib.Order.Interval.Finset.Nat

/-!
# Ticks, spans of ticks and the monitoring period (trusted)

The **monitoring period** is the one span of time accounting is about: a result is per monitoring period, and it pools
the blocks and phases the monitoring period holds. Every other span of time is a block's or a phase's span: an execution
block's, an execution phase's, a server block's.

* **A tick** (`Tick`): one step of a clock, the device model's unit of time.
* **A span of ticks** (`TickSpan`): the ticks `[start, start + len)` on the one clock (`OneClock`), ending at `stop`.
  `Within` says one span lies inside another.
* **A cut** of a span into `n` consecutive spans (`TickSpan.Cut`): `n + 1` boundaries, in order, from the span's start
  to its stop. Its part `j` (`Cut.part`) is the span between boundaries `j` and `j + 1`.
* **A monitoring period** (`MonitoringPeriod`): a span of ticks cut into its execution blocks' spans, each cut into its
  execution phases' spans.
* **A play** over `n` execution phases (`Play`): phase `t`'s move is any function of the coins of the phases before it.
  `Play.at` reads every move along one whole draw of the coins, the non-adaptive reading.

A bound over a monitoring period has two readings: per execution phase, every execution phase's bound holds at once, at
their errors summed; pooled, the monitoring period's total holds at one error.
-/

namespace Verity

/-- **A tick**: one step of a clock, the device model's unit of time. -/
abbrev Tick : Type := ℕ

/-- **A span of ticks**: the ticks `[start, start + len)` on the one clock (`OneClock`). A monitoring period's span and a
block's or a phase's span are each a `TickSpan` read in that role. -/
structure TickSpan where
  /-- its first tick -/
  start : Tick
  /-- its length, in ticks -/
  len : ℕ

namespace TickSpan

/-- The tick after a span's last: `start + len`. -/
def stop (p : TickSpan) : Tick := p.start + p.len

/-- The ticks of a span. -/
def ticks (p : TickSpan) : Finset ℕ := Finset.Ico p.start p.stop

/-- **`p` lies within `q`**: it starts no earlier and stops no later. -/
def Within (p q : TickSpan) : Prop := q.start ≤ p.start ∧ p.stop ≤ q.stop

/-- **A cut of the span `p` into `n` consecutive spans**: the boundaries `boundary 0 ≤ … ≤ boundary n`, from `p`'s start
to its stop. -/
structure Cut (p : TickSpan) (n : ℕ) where
  /-- the boundaries, in order -/
  boundary : Fin (n + 1) → Tick
  mono : Monotone boundary
  first : boundary 0 = p.start
  last : boundary (Fin.last n) = p.stop

/-- **Part `j` of a cut**: the span from boundary `j` to boundary `j + 1`. -/
def Cut.part {p : TickSpan} {n : ℕ} (c : p.Cut n) (j : Fin n) : TickSpan :=
  ⟨c.boundary j.castSucc, c.boundary j.succ - c.boundary j.castSucc⟩

end TickSpan

/-- **A monitoring period**: its span of ticks, cut into its `executionBlocks` execution blocks' spans, and each
execution block's span cut into its execution phases' spans. -/
structure MonitoringPeriod where
  /-- its span of ticks -/
  span : TickSpan
  /-- the number of its execution blocks -/
  executionBlocks : ℕ
  /-- the cut of its span into its execution blocks' spans -/
  executionCut : span.Cut executionBlocks
  /-- the number of execution phases of each execution block -/
  executionPhases : Fin executionBlocks → ℕ
  /-- the cut of each execution block's span into its execution phases' spans -/
  phaseCut : (j : Fin executionBlocks) → (executionCut.part j).Cut (executionPhases j)

namespace MonitoringPeriod

/-- **Execution block `j`'s span**. -/
def executionSpan (m : MonitoringPeriod) (j : Fin m.executionBlocks) : TickSpan := m.executionCut.part j

/-- **Execution phase `t`'s span** in execution block `j`. -/
def phaseSpan (m : MonitoringPeriod) (j : Fin m.executionBlocks) (t : Fin (m.executionPhases j)) : TickSpan :=
  (m.phaseCut j).part t

end MonitoringPeriod

/-- **A play over `n` execution phases**, each drawing a coin from `Ω`: phase `t`'s move, an `X t`, is any function of
the coins of the phases before it. It is a `Game.Strategy` of the game that, phase by phase, receives the phase's move
and then draws its coin, indexed by phase where the strategy is a tree: the same data, uncurried. -/
abbrev Play (n : ℕ) (Ω : Type) (X : Fin n → Type) : Type :=
  (t : Fin n) → ({s : Fin n // s < t} → Ω) → X t

/-- **The moves along a whole draw `ω`**: phase `t`'s move on the coins `ω` gives the phases before it, the
non-adaptive reading every statement over one draw takes. -/
def Play.at {n : ℕ} {Ω : Type} {X : Fin n → Type} (P : Play n Ω X) (ω : Fin n → Ω) (t : Fin n) : X t :=
  P t fun s => ω s.1

end Verity
