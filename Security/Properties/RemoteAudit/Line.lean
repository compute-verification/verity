import Definitions.Pouw.Basic.Prob
import Definitions.Accounting.Assumptions.RemoteHeld
import Definitions.Accounting.Assumptions.RemoteOutside
import Definitions.Accounting.Assumptions.RemoteDraw
import Properties.RemoteAudit.Line.Record
import Properties.RemoteAudit.Line.Premises

/-!
# The memory line of record, over what the Lean verifier accepts (`RemoteAudit.Line.lineOfRecordAt`)

The line a gate run publishes (`RemoteAudit.Line.lineOfRecordAt`, `line.py`'s `line_of_record`) reads each stretch of a
paced session's groups off the Lean verifier's record of it and credits it `span · (N − lost)` registered blocks held
over its span, and `max(0, w − lost)` weights bytes a tick, `w` the weights blocks its passes read on average.
`LineHeld` is the one statement that this is true of the trace that produced the record, at any windows a draw
(`Node.groupsPerCoin` g, `groups_per_coin`: one openings coin over windows `c·g … (c + 1)·g − 1`, the session run by
`Shared.run g`, which at g = 1 is `Session.run`):

* **LineHeld** (link-charged, `forms[1]`): on any node, trace and correct receipts, with the prover's records read
  through the coin commitment (`recordCommitted`: holding commitment `c` its record at `ω` is `recs c ω`, and the
  session it plays that record's opening draw `recordDraw`, one draw for each of the session's `⌈W/g⌉` (`Node.draws`,
  a window each at g = 1), draw `c` with the receipts its windows have in the state the record's session reaches,
  drawing `min(s, U_c)` at the session's `samples` s) and
  the record of outcome `ω` the prover's holding the deployed commitment `com ω` (`played`), whose trace of record
  (`ofRecord`, the trace over the registration's blocks with the Lean verifier's verdict on the record as its own)
  meets the memory model's premises (the links' ceiling `Links`, `SquaringCeiling`, `Disks`, `ChipCeiling`, `Routed`,
  `Paced` at the record's groups `recordPace`, `Covers`, `SketchBoundAt` at `δs` and `NoOutsideCompute`, statement C in
  full: no device outside the node computes on a registered block or a receipt during the session, and the draw's
  `Live` at every fixed coin commitment, `Refused` at the receipts `R`, `Counted`, of the session the prover plays
  holding `com`, `play com`; no code-to-model premise, `Properties.RemoteAudit.Line.Premises`), the probability that
  the Lean verifier accepts the record's session and the line it reads off the Lean verifier's record (`lineOf`: its
  groups, draws and registered blocks the record's) credits a stretch more than the trace held over its span
  (`Credits`) is at most `δs + (δ + Σ_c s / 2^64 + Σ_c adv_c + Σ_c hide_c + bind)`, `δ` the one the line reads its
  pooled bound at (`Node.delta`, a nonnegative rational; the line computes at none outside `(0, 1)`), `adv_c` the
  advantage on `prf c` of draw `c`'s distinguisher at that `δ` (`Session.distinguisher`) with openings coin `c`'s
  commitment leaf committing zeros (`Committed.junk`), `hide_c` draw `c`'s hiding term (`Committed.hide`, hm96's, at
  most `n·2^-191` over the session's `n` draws in all), and `bind` the receipts' binding term over the session's draws
  (`Receipts.binding`: some opened receipt presented with an accepted row other than the one its root holds). The
  rates are the node's in the model's units, a tick a nanosecond (`Node.squarings` … `Node.chipBlocks`), and the
  openings are the deployed draw's (`Accounting.Remote.Draw`: `Key.subset` on the first blocks `prf c` gives draw `c`'s
  coin, at most `receiptMax = 2^64 − 1 < 2^448` receipts a draw, the verifier's `RECEIPT_LIMIT` less one, the
  sampler's fallbacks `rest` free), at the error `Accounting.Guarantees.Remote.PooledDraw` states.
* **LineHeldIdentity** (time-held, `forms[0]`): the same under Identity, no block coming in over the links.

They follow from the model's argument for `Accounting.Guarantees.Remote.HeldOverTimeLinksDrawn`, the line's W covering
the session's pooled bound (`Properties.RemoteAudit.LineCoversSession`, at the record's draws, whose draws the line
holds to `min(s, U_c)` and whose receipts on an accepted record are `recordDraw`'s sizes: the reached state is the
run's, a window's receipts are below `RECEIPT_LIMIT`, `Properties.RemoteAudit.ReceiptsBelowLimit`, and the line refuses
a draw of `g` > 1 windows with `RECEIPT_LIMIT` or more, `RemoteAudit.Line.ofSummaryAt`), the line's groups being one
for each of the session's windows on an accepted record (its parameters' `windows`, `Session.State.outcome`, which
`Shared.run` keeps at every g), and two properties of the line's code, proved of it:

* **LineChecked**: a line the code computes had a block of one or more bytes, a block's encode of one or more squarings,
  nonnegative rates and a nonnegative wrong receipts' bound (`check`).
* **StretchBound**, the arithmetic link: each stretch's credit is at most the guarantee's left-hand side at the stretch's
  own paced session (`stretchPace`) and span, term for term, the units and the clock included: the span `Σ`, its
  closing group's coin as `stop`, its start group 0's end (the first stretch) or the stretch's first coin (the rest),
  `L` at most the line's `L` (`lMax`, which also reads group `lo − 1`), `X` and `B` the form's charge, `S` the blocks
  the chips can hold any byte of, `chip / b` rounded up, and `fill(W)` at most the line's greedy fill at the `W` it
  read (the greedy over groups highest rate first attains the fractional knapsack's optimum, and the line's rates are
  the model's or above).

`LineHeld` reads the record the Lean verifier makes on the node's registration, whose registered blocks are the
registration's on an accepted run (`record_summary`). A record the verifier's `line_record` is handed bare it checks
against the registration given with it, a third property of the code, proved:

* **LineRegistered**: a line `line_record` reads off a record given bare (`lineOfRegistered`) is of a record of the
  registration given with it: the record names its digest, and its registered blocks, which the line credits, are
  the registration's.

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

/-- **The verifier refuses a line that isn't registered.** A line `line_record` computes off a record given bare,
drawn `g` windows a coin (`lineOfRegistered`), is of a record of the registration given with it: the record names that
registration's digest, its registered blocks, which the line credits, are the registration's, and the line is the
record's (`lineOfRecordOn`). -/
def LineRegistered : Prop :=
  ∀ (onRecord : Bool) (g : ℕ) (reg : RemoteAudit.Registration) (digest : ByteArray)
    (sm : RemoteAudit.Session.Summary) (samples : ℕ) (seen : Array Seen) (per blockBytes chipBytes nodeBytes : ℕ)
    (δ : ℚ) (rates : Rates) (l : Line),
    lineOfRegistered onRecord g reg digest sm samples seen per blockBytes chipBytes nodeBytes δ rates = .ok l →
    digest = reg.digest ∧ sm.blocks = reg.blocks ∧
      lineOfRecordOn onRecord g sm samples seen per blockBytes chipBytes nodeBytes δ rates = .ok l

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

/-- **The memory line of record, link-charged.** On a trace over a session's coins, one for each of its draws
(`Node.draws`, a window each at g = 1), the prover's records read through the coin commitment (`recordCommitted`,
each record's opening draw at its `samples` s) and the record of each outcome the one played holding the deployed
commitment (`played`), whose trace of record (`ofRecord`: over the registration's blocks, its verdict the Lean
verifier's on the record) meets the memory model's premises at the node's rates and the record's groups, the
probability that the Lean verifier accepts the record's session and a stretch of the record's link-charged line
credits more than the trace held over its span is at most `δs + (δ + Σ_c s / 2^64 + Σ_c adv_c + Σ_c hide_c + bind)`,
`δ` the line's, `adv_c` the advantage on `prf c` of draw `c`'s distinguisher at it with leaf `c` of the coin
commitment committing zeros, `hide_c` draw `c`'s hiding term and `bind` the receipts' binding term at `R`, over the
session's draws. -/
def LineHeld : Prop :=
  ∀ (node : Node) (κ : ℕ) (X Cm : Type) [Fintype X] (prf : Fin node.draws → Pous.Bits κ → Blocks receiptMax)
    (rest : Coins node.draws κ X → Fin node.draws → ℕ → ℕ) (recs : Cm → Coins node.draws κ X → Record)
    (correct : Cm → Fin node.draws → Coins node.draws κ X → Finset ℕ) (com : Coins node.draws κ X → Cm)
    (T : Trace (Coins node.draws κ X)) (R : Receipts node.draws κ X) (δs : ℝ),
    let r := played recs com
    let F := recordCommitted node recs correct
    ∀ O : Outside (ofRecord T node r),
    Links (ofRecord T node r) node.link node.linkBurst →
    SquaringCeiling (ofRecord T node r) node.squarings node.encode node.rebuildBurst →
    Disks (ofRecord T node r) node.disk node.diskBurst → ChipCeiling (ofRecord T node r) node.chipBlocks →
    Routed (ofRecord T node r) → Paced (ofRecord T node r) (recordPace node r) → Covers (ofRecord T node r) →
    SketchBoundAt (ofRecord T node r) O δs → NoOutsideCompute (ofRecord T node r) O → (∀ c, Live (F.view c)) →
    Refused (F.play com) prf rest (ofRecord T node r) R → Counted (F.play com) (ofRecord T node r) →
    Pouw.pr (fun ω => Verified node (r ω) ∧ ¬ HeldAt (ofRecord T node r) ω node (r ω) 1) ≤
      δs + ((node.delta : ℝ) + ((∑ _t : Fin node.draws, node.params.samples : ℕ) : ℝ) / 2 ^ 64 +
        (∑ t, Pous.StreamedWipe.prfAdvantage (prf t)
          ((F.play (Committed.junk com t)).distinguisher prf (node.delta : ℝ) t)).toReal +
        (∑ t, F.hide com prf (node.delta : ℝ) t).toReal + (R.binding (F.play com) prf rest).toReal)

/-- **The memory line of record, time-held, under Identity.** `LineHeld` for the record's time-held line, with no
block coming in over the node's boundary links (`Identity`) in place of their ceiling. -/
def LineHeldIdentity : Prop :=
  ∀ (node : Node) (κ : ℕ) (X Cm : Type) [Fintype X] (prf : Fin node.draws → Pous.Bits κ → Blocks receiptMax)
    (rest : Coins node.draws κ X → Fin node.draws → ℕ → ℕ) (recs : Cm → Coins node.draws κ X → Record)
    (correct : Cm → Fin node.draws → Coins node.draws κ X → Finset ℕ) (com : Coins node.draws κ X → Cm)
    (T : Trace (Coins node.draws κ X)) (R : Receipts node.draws κ X) (δs : ℝ),
    let r := played recs com
    let F := recordCommitted node recs correct
    ∀ O : Outside (ofRecord T node r),
    Identity (ofRecord T node r) →
    SquaringCeiling (ofRecord T node r) node.squarings node.encode node.rebuildBurst →
    Disks (ofRecord T node r) node.disk node.diskBurst → ChipCeiling (ofRecord T node r) node.chipBlocks →
    Routed (ofRecord T node r) → Paced (ofRecord T node r) (recordPace node r) → Covers (ofRecord T node r) →
    SketchBoundAt (ofRecord T node r) O δs → NoOutsideCompute (ofRecord T node r) O → (∀ c, Live (F.view c)) →
    Refused (F.play com) prf rest (ofRecord T node r) R → Counted (F.play com) (ofRecord T node r) →
    Pouw.pr (fun ω => Verified node (r ω) ∧ ¬ HeldAt (ofRecord T node r) ω node (r ω) 0) ≤
      δs + ((node.delta : ℝ) + ((∑ _t : Fin node.draws, node.params.samples : ℕ) : ℝ) / 2 ^ 64 +
        (∑ t, Pous.StreamedWipe.prfAdvantage (prf t)
          ((F.play (Committed.junk com t)).distinguisher prf (node.delta : ℝ) t)).toReal +
        (∑ t, F.hide com prf (node.delta : ℝ) t).toReal + (R.binding (F.play com) prf rest).toReal)

end Properties.RemoteAudit.Line
