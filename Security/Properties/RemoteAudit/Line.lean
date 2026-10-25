import Definitions.Pouw.Basic.Prob
import Models.Accounting.Assumptions.RemoteHeld
import Models.Accounting.Assumptions.RemoteOutside
import Models.Accounting.Assumptions.RemoteDraw
import Properties.RemoteAudit.Line.Record
import Properties.RemoteAudit.Line.Premises

/-!
# The memory line of record, over what the Lean verifier accepts (`RemoteAudit.Line.lineOfRecord`)

The line a gate run publishes (`RemoteAudit.Line.lineOfRecord`, `line.py`'s `line_of_record`) reads each stretch of a
paced session's groups off the Lean verifier's record of it and credits it `span · (N − lost)` registered blocks held
over its span, and `max(0, w − lost)` weights bytes a tick, `w` the weights blocks its passes read on average.
`LineHeld` is the one statement that this is true of the trace that produced the record:

* **LineHeld** (link-charged, `forms[1]`): on any node, trace and opening draw of `min(s, U_t)` receipts a window at
  the session's `samples` s, whose trace of record (`ofRecord`, the trace with the Lean verifier's verdict on the
  record as its own) meets the memory model's premises (the links' ceiling `Links`, `SquaringCeiling`, `Disks`,
  `ChipCeiling`, `Routed`, `Paced` at the record's groups `recordPace`, `Covers`, `SketchBoundAt` at `δs` and
  `NoOutsideCompute`, statement C in full: no device outside the node computes on a registered block or a receipt
  during the session, and the draw's `Live`, `Refused`, `Counted`) and the record's code-to-model premises
  (`Registered`, `SizesOfRecord`, `DrawFits`, `Properties.RemoteAudit.Line.Premises`), the probability that the Lean
  verifier accepts the record's session and the line it reads off the Lean verifier's record (`lineOf`: its groups,
  windows and registered blocks the record's) credits a stretch more than the trace held over its span (`Credits`) is
  at most `δs + (δ + Σ_t s / 2^64 + Σ_t adv_t)`, `δ` the one the line reads its pooled bound at (`Node.delta`, a
  nonnegative rational; the line computes at none outside `(0, 1)`) and `adv_t` the advantage on `prf t` of window
  `t`'s distinguisher at that `δ` (`Session.distinguisher`). The rates are the node's in the model's units, a tick a
  nanosecond (`Node.squarings` … `Node.chipBlocks`), and the openings are the deployed draw's (`Accounting.Remote.Draw`:
  `Key.subset` on the first blocks `prf t` gives window `t`'s coin, at most `m < 2^448` receipts a window (`DrawFits`),
  the sampler's fallbacks `rest` free), at the error `Accounting.Guarantees.Remote.PooledDraw` states.
* **LineHeldIdentity** (time-held, `forms[0]`): the same under Identity, no block coming in over the links.

They follow from the model's argument for `Accounting.Guarantees.Remote.HeldOverTimeLinksDrawn`, the line's W covering
the session's pooled bound (`Properties.RemoteAudit.LineCoversSession`, at the record's windows, whose draws the line
holds to `min(s, U_t)`), the line's groups being one for each of the session's windows on an accepted record
(`Properties.RemoteAudit.AcceptedMinPasses`), and two properties of the line's code, proved of it:

* **LineChecked**: a line the code computes had a block of one or more bytes, a block's encode of one or more squarings,
  nonnegative rates and a nonnegative wrong receipts' bound (`check`).
* **StretchBound**, the arithmetic link: each stretch's credit is at most the guarantee's left-hand side at the stretch's
  own paced session (`stretchPace`) and span, term for term, the units and the clock included: the span `Σ`, its
  closing group's coin as `stop`, its start group 0's end (the first stretch) or the stretch's first coin (the rest),
  `L` at most the line's `L` (`lMax`, which also reads group `lo − 1`), `X` and `B` the form's charge, `S` the blocks
  the chips can hold any byte of, `chip / b` rounded up, and `fill(W)` at most the line's greedy fill at the `W` it
  read (the greedy over groups highest rate first attains the fractional knapsack's optimum, and the line's rates are
  the model's or above).

A stretch after the first starts at its first group's coin `s_lo`, earlier than its paced session's `start = e_lo`, so
the guarantee as stated (`[e_0, s_C)`) does not reach it: the model's argument bounds absence over `[s_0, s_C)` and
holds from any start in `[s_0, e_0]`, which the proof of `LineHeld` uses.
-/

namespace Properties.RemoteAudit.Line

open Accounting.Remote.Held Accounting.Remote.Held.Assumptions Accounting.Remote.Draw
  Accounting.Remote.Draw.Assumptions _root_.RemoteAudit.Line

/-- **LineChecked**: a line the code computes had a block of one or more bytes, a block's encode of one or more
squarings, nonnegative rates and bursts, and read a nonnegative wrong receipts' bound. -/
def LineChecked : Prop :=
  ∀ (gs : Array Group) (per blocks blockBytes chipBytes nodeBytes : ℕ) (windows : Array (ℕ × ℕ)) (δ : ℚ)
    (rates : Rates) (l : Line),
    line gs per blocks blockBytes chipBytes nodeBytes windows δ rates = .ok l →
    1 ≤ blockBytes ∧ 1 ≤ rates.encodeSquarings ∧ 0 ≤ rates.squaringsPerS ∧ 0 ≤ rates.diskBytesPerS ∧
      0 ≤ rates.linkBytesPerS ∧ 0 ≤ rates.rebuildBurstBytes ∧ 0 ≤ rates.diskBurstBytes ∧
      0 ≤ rates.linkBurstBytes ∧ 0 ≤ l.wrong

/-- **The arithmetic link.** Each stretch `x` of form `i` (time-held, link-charged) of a line the code computes: its
closing group is a group of the record, its span is `[spanStart, spanStop)` on the record's clock, nonempty, ending at
its paced session's `stop` and starting at its `ended 0` (the first stretch) or its `released 0` (the rest), its form's
charge is `charges`', its credit `span · (N − lost)` is at most the left-hand side of
`Accounting.Guarantees.Remote.HeldOverTimeLinks` at its paced session (`stretchPace`), that span, the chips' blocks
`chip / b` rounded up, the form's charge and the wrong receipts' bound the line read (`Line.wrong`), and its weights bytes
a tick are
`max(0, w − lost) · b`. -/
def StretchBound : Prop :=
  ∀ (gs : Array Group) (per blocks blockBytes chipBytes nodeBytes : ℕ) (windows : Array (ℕ × ℕ)) (δ : ℚ)
    (rates : Rates) (l : Line),
    line gs per blocks blockBytes chipBytes nodeBytes windows δ rates = .ok l →
    ∀ i < 2, ∀ f, l.forms[i]? = some f → ∀ x ∈ f.stretches,
      (x.stretch + 1) * per < gs.size ∧
      spanStart gs per x.stretch < spanStop gs per x.stretch ∧
      x.spanNs = spanStop gs per x.stretch - spanStart gs per x.stretch ∧
      spanStop gs per x.stretch = (stretchPace (fun _ : Unit => gs) per x.stretch).stop () ∧
      (spanStart gs per x.stretch = (stretchPace (fun _ : Unit => gs) per x.stretch).ended () 0 ∨
        spanStart gs per x.stretch = (stretchPace (fun _ : Unit => gs) per x.stretch).released () 0) ∧
      f.charge = (if i = 0 then (charges rates blockBytes).1 else (charges rates blockBytes).2) ∧
      (x.spanNs : ℝ) * ((blocks : ℝ) - x.time.lost) ≤
        (x.spanNs : ℝ) * ((blocks : ℝ) - ((chipBytes + blockBytes - 1) / blockBytes : ℕ))
          - 2 * (stretchPace (fun _ : Unit => gs) per x.stretch).longest () *
              ((f.charge.rate : ℝ) *
                  ((x.spanNs : ℝ) + 2 * (stretchPace (fun _ : Unit => gs) per x.stretch).longest ()) +
                f.charge.burst)
          - (stretchPace (fun _ : Unit => gs) per x.stretch).fill () blocks l.wrong ∧
      x.time.weights = max 0 (x.weights - x.time.lost) * (blockBytes : ℚ)

/-- What stretch `x` of a line over the record's groups `gs` says of the trace at `ω`, over its span
`[spanStart, spanStop)` on the record's clock: its `spanNs` is that span's; the registered blocks in node memory,
summed over its ticks, are at least `span · (N − lost)`; and of any set `R` of registered blocks at least as large as
the blocks of weights its passes read on average (the store's weights region), the bytes in node memory, summed over
its ticks, are at least `span` times its weights bytes a tick. -/
def Credits {Ω : Type} (T : Trace Ω) (ω : Ω) (gs : Array Group) (per b : ℕ) (x : Stretch) : Prop :=
  x.spanNs = spanStop gs per x.stretch - spanStart gs per x.stretch ∧
  (x.spanNs : ℝ) * ((T.blocks : ℝ) - x.time.lost) ≤
    T.heldOver ω (spanStart gs per x.stretch) (spanStop gs per x.stretch) ∧
  ∀ R : Finset (Fin T.blocks), (x.weights : ℝ) ≤ R.card →
    (x.spanNs : ℝ) * x.time.weights ≤
      b * ∑ t ∈ Finset.Ico (spanStart gs per x.stretch) (spanStop gs per x.stretch), (T.mem ω t ∩ R).card

/-- Every stretch of form `form` of the record's line, if the code computes one, credits what the trace held at `ω`. -/
def HeldAt {Ω : Type} (T : Trace Ω) (ω : Ω) (node : Node) (r : Record) (form : ℕ) : Prop :=
  ∀ l, lineOf node r = .ok l → ∀ f, l.forms[form]? = some f → ∀ x ∈ f.stretches,
    Credits T ω (r.groups node) node.per node.blockBytes x

/-- **The memory line of record, link-charged.** On a trace over a session's coins and the session's opening draw at
its `samples` s, whose trace of record (`ofRecord`: its verdict the Lean verifier's on the record) meets the memory
model's premises at the node's rates and the record's groups and the record's code-to-model premises, the probability
that the Lean verifier accepts the record's session and a stretch of the record's link-charged line credits more than
the trace held over its span is at most `δs + (δ + Σ_t s / 2^64 + Σ_t adv_t)`, `δ` the line's and `adv_t` the
advantage on `prf t` of window `t`'s distinguisher at it. -/
def LineHeld : Prop :=
  ∀ (node : Node) (n m κ : ℕ) (X : Type) [Fintype X] (prf : Fin n → Pous.Bits κ → Blocks m)
    (rest : Coins n κ X → Fin n → ℕ → ℕ) (D : Session n m (fun _ => node.params.samples) κ X)
    (T : Trace (Coins n κ X)) (rec : Coins n κ X → Record) (δs : ℝ) (O : Outside (ofRecord T node rec)),
    Links (ofRecord T node rec) node.link node.linkBurst →
    SquaringCeiling (ofRecord T node rec) node.squarings node.encode node.rebuildBurst →
    Disks (ofRecord T node rec) node.disk node.diskBurst → ChipCeiling (ofRecord T node rec) node.chipBlocks →
    Routed (ofRecord T node rec) → Paced (ofRecord T node rec) (recordPace node rec) → Covers (ofRecord T node rec) →
    SketchBoundAt (ofRecord T node rec) O δs → NoOutsideCompute (ofRecord T node rec) O →
    DrawFits m → Live D → Refused D prf rest (ofRecord T node rec) →
    Counted D (ofRecord T node rec) → Registered (ofRecord T node rec) node →
    SizesOfRecord D (ofRecord T node rec) node rec →
    Pouw.pr (fun ω => Verified node (rec ω) ∧ ¬ HeldAt (ofRecord T node rec) ω node (rec ω) 1) ≤
      δs + ((node.delta : ℝ) + ((∑ _t : Fin n, node.params.samples : ℕ) : ℝ) / 2 ^ 64 +
        (∑ t, Pous.StreamedWipe.prfAdvantage (prf t) (D.distinguisher prf (node.delta : ℝ) t)).toReal)

/-- **The memory line of record, time-held, under Identity.** `LineHeld` for the record's time-held line, with no
block coming in over the node's boundary links (`Identity`) in place of their ceiling. -/
def LineHeldIdentity : Prop :=
  ∀ (node : Node) (n m κ : ℕ) (X : Type) [Fintype X] (prf : Fin n → Pous.Bits κ → Blocks m)
    (rest : Coins n κ X → Fin n → ℕ → ℕ) (D : Session n m (fun _ => node.params.samples) κ X)
    (T : Trace (Coins n κ X)) (rec : Coins n κ X → Record) (δs : ℝ) (O : Outside (ofRecord T node rec)),
    Identity (ofRecord T node rec) →
    SquaringCeiling (ofRecord T node rec) node.squarings node.encode node.rebuildBurst →
    Disks (ofRecord T node rec) node.disk node.diskBurst → ChipCeiling (ofRecord T node rec) node.chipBlocks →
    Routed (ofRecord T node rec) → Paced (ofRecord T node rec) (recordPace node rec) → Covers (ofRecord T node rec) →
    SketchBoundAt (ofRecord T node rec) O δs → NoOutsideCompute (ofRecord T node rec) O →
    DrawFits m → Live D → Refused D prf rest (ofRecord T node rec) →
    Counted D (ofRecord T node rec) → Registered (ofRecord T node rec) node →
    SizesOfRecord D (ofRecord T node rec) node rec →
    Pouw.pr (fun ω => Verified node (rec ω) ∧ ¬ HeldAt (ofRecord T node rec) ω node (rec ω) 0) ≤
      δs + ((node.delta : ℝ) + ((∑ _t : Fin n, node.params.samples : ℕ) : ℝ) / 2 ^ 64 +
        (∑ t, Pous.StreamedWipe.prfAdvantage (prf t) (D.distinguisher prf (node.delta : ℝ) t)).toReal)

end Properties.RemoteAudit.Line
