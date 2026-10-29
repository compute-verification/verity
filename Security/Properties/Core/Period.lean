import CoreProtocol.Period
import Definitions.Core.Period

/-!
# The code's periods are the spec's

`CoreProtocol.Period` (package `core_protocol`) is the code's copy of `Definitions.Core.Period`, so that a verifier reads
ticks, spans and the monitoring period without requiring `security`. These say that the copy is the spec's:

* `TickSpanIsSpec`: a code span reads as the spec's span of the same start and length (`spec`), each spec span is one
  code span's, and its `stop`, its `ticks` and `within` are the spec's `stop`, `ticks` and `Within`;
* `MonitoringPeriodIsSpec`: a code monitoring period is `valid` exactly when it holds a spec monitoring period's
  boundaries (`code`), no two spec monitoring periods hold the same ones, and its execution and sampling blocks' spans
  are the spec's.

`Proofs.Properties.Core.Period` proves both.
-/

namespace Properties.Period

/-- The spec's span a code span reads as: the same start and length. -/
def spec (p : _root_.Core.TickSpan) : Verity.TickSpan := ⟨p.start, p.len⟩

/-- What the code holds of a spec cut: its boundaries, in order. -/
def boundaries {p : Verity.TickSpan} {n : ℕ} (c : p.Cut n) : Array ℕ := Array.ofFn c.boundary

/-- The code's monitoring period of a spec one: its span, its execution cut's boundaries, and each execution block's
sampling cut's. -/
def code (M : Verity.MonitoringPeriod) : _root_.Core.MonitoringPeriod :=
  ⟨⟨M.span.start, M.span.len⟩, boundaries M.executionCut, Array.ofFn fun j => boundaries (M.samplingCut j)⟩

/-- **The code's spans are the spec's.** Reading a code span as the spec's is a bijection, under which a span's stop,
its ticks and one span lying within another are the spec's. -/
def TickSpanIsSpec : Prop :=
  Function.Bijective spec ∧
  (∀ p : _root_.Core.TickSpan, (spec p).stop = p.stop) ∧
  (∀ p : _root_.Core.TickSpan, p.ticks.toFinset = (spec p).ticks) ∧
  (∀ p q : _root_.Core.TickSpan, p.within q = true ↔ (spec p).Within (spec q))

/-- **The code's monitoring periods are the spec's.** A code monitoring period is `valid` exactly when it is `code` of
a spec one, `code` is injective, and the execution and sampling blocks' spans of `code M` are `M`'s. -/
def MonitoringPeriodIsSpec : Prop :=
  (∀ m : _root_.Core.MonitoringPeriod, m.valid = true ↔ ∃ M, code M = m) ∧
  Function.Injective code ∧
  (∀ M : Verity.MonitoringPeriod, (code M).executionSpans.map spec = List.ofFn M.executionSpan) ∧
  (∀ M : Verity.MonitoringPeriod,
    (code M).samplingSpans.map (·.map spec) = List.ofFn fun j => List.ofFn (M.samplingSpan j))

end Properties.Period
