module

public import Definitions.Core.Boolean

/-!
# The response tie: what its two Definitions compute, and what a release is held to

Definitions only. The response tie is the proof block each connection of a tied class of the network certifier's
content rows gets (`verity/protocols/accounting/certifier/prover/response_tie.py`), of one of two Definitions by the
kind of its class: `ResponseTieStop{O}` in a stop class, whose connections the stop rule ends after the stop word's n ids, and
`ResponseTieNoStop{O}` in a no-stop class, whose connections end where the network certifier's end row says. Each
reads the compute side's committed egress row of O u32 ids and the network certifier's end row (u16 class, u16 at),
and a stop class's also its stop word (n + finish·2^13, n of 13 bits); each returns o0, `accept` or 0, and o1, the
response row: a u32 count, `noCount` where o0 is 0, then O u32 ids.

`outputs` is what both compute, the Python reference's `outputs` and `refusals` read as one function; `argumentBits`
and `outputBits` are their signature's bits, each field little-endian, a row's bit 8b + i bit i of its byte b.
`Computes` is the statement that a Boolean circuit, read on its signature's wires, computes them on every input the
Definition takes. `ReleaseHeld` is what an accepted block holds a release to: where o1 opens to the signed
`response.i` row, the ids the network certifier released are the committed egress row's ids below the count, the
network certifier's own `at`, or the stop word's n where the stop rule ended the response.

Two statements are open, each a `def … : Prop` that nothing proves yet (the goal `network/egress-tied-to-trace`):
`OutputsHoldRelease`, that the outputs of an accepted block hold its release; and `ResponseTieComputesSpec`, that
each registered circuit of the two Definitions computes `outputs`, over the reading of the verifier's parsed circuit
file as a Boolean circuit with its signature's wires, which this tree does not name yet. With the one-stage audit's
soundness over the boundary record whose `b.egress` the tie's record registers, they give
`WeightSecurity.Assumptions.EgressTied` for every tied class.
-/

@[expose] public section

namespace NetTiming.ResponseTie

/-- o0's value on a row the block accepts (`response_tie.ACCEPT`). -/
def accept : ℕ := 1

/-- The response row's count where o0 is 0 (`content_rows.NO_COUNT`). -/
def noCount : ℕ := 0xFFFFFFFF

/-- The end row's class of a connection that ended in its `END` (`content_rows.ENDED`). -/
def ended : ℕ := 0

/-- The end row's class of a connection that failed (`content_rows.FAILED`). -/
def failed : ℕ := 1

/-- The network certifier's end row of a connection: its class and the cells it released before it ended. -/
structure EndRow where
  cls : ℕ
  cells : ℕ

/-- A stop class's stop word: the ids the stop rule ends the response after, and the reason, 0 where it doesn't end
it within the row. -/
structure StopWord where
  n : ℕ
  finish : ℕ

/-- Whether the block refuses the rows (`response_tie.refusals` is not empty): `stop` is the stop word in a stop
class and `none` in a no-stop class. -/
def refused (O : ℕ) (stop : Option StopWord) (e : EndRow) : Bool :=
  decide (failed < e.cls) ||
    match stop with
    | some w => decide (w.finish = 0) || decide (O < w.n) || (decide (e.cls = failed) && decide (w.n < e.cells))
    | none => decide (O < e.cells)

/-- The count the response row's ids are below: the stop word's n in a stop class whose end row's class is even,
and the end row's cells otherwise. -/
def count (stop : Option StopWord) (e : EndRow) : ℕ :=
  match stop with
  | some w => if e.cls % 2 = 0 then w.n else e.cells
  | none => e.cells

/-- An egress row of at most O ids, padded with 0 to O. -/
def pad (O : ℕ) (egress : List ℕ) : List ℕ := egress ++ List.replicate (O - egress.length) 0

/-- The response row's O ids: the padded egress row's first `k`, then 0. -/
def responseIds (O : ℕ) (egress : List ℕ) (k : ℕ) : List ℕ :=
  (pad O egress).take k ++ List.replicate (O - min k O) 0

/-- What both Definitions compute: o0, then the response row's count and its O ids. -/
def outputs (O : ℕ) (stop : Option StopWord) (egress : List ℕ) (e : EndRow) : ℕ × ℕ × List ℕ :=
  if refused O stop e then (0, noCount, responseIds O egress (count stop e))
  else (accept, count stop e, responseIds O egress (count stop e))

/-- The inputs a Definition takes: an egress row of at most O u32 ids, an end row of two u16, and a stop class's
stop word of a 13-bit n and a 3-bit finish. -/
def WellFormed (O : ℕ) (stop : Option StopWord) (egress : List ℕ) (e : EndRow) : Prop :=
  egress.length ≤ O ∧ (∀ t ∈ egress, t < 2 ^ 32) ∧ e.cls < 2 ^ 16 ∧ e.cells < 2 ^ 16 ∧
    ∀ w, stop = some w → w.n < 2 ^ 13 ∧ w.finish < 2 ^ 3

/-- The low `w` bits of `x`, least significant first. -/
def bitsLE (w x : ℕ) : List Bool := (List.range w).map x.testBit

/-- A Definition's arguments as bits, in its signature's order: the egress row's 32·O, the stop word's 16 in a stop
class, then the end row's 32. -/
def argumentBits (O : ℕ) (stop : Option StopWord) (egress : List ℕ) (e : EndRow) : List Bool :=
  (pad O egress).flatMap (bitsLE 32) ++
    (match stop with
     | some w => bitsLE 16 (w.n + w.finish * 2 ^ 13)
     | none => []) ++
    bitsLE 16 e.cls ++ bitsLE 16 e.cells

/-- A Definition's outputs as bits: o0's 16, then the response row's count's 32 and its ids' 32 each. -/
def outputBits (O : ℕ) (stop : Option StopWord) (egress : List ℕ) (e : EndRow) : List Bool :=
  match outputs O stop egress e with
  | (v, m, ids) => bitsLE 16 v ++ bitsLE 32 m ++ ids.flatMap (bitsLE 32)

/-- A Boolean circuit computes the Definition at O, of a stop class where `stop` holds: read on its input gates
`ins` and its output gates `outs`, each in its signature's order, it gives `outputBits` on every input the Definition
takes. -/
def Computes (C : Verity.Boolean.Circuit) (ins outs : List (Fin C.N)) (O : ℕ) (stop : Bool) : Prop :=
  ins.Nodup ∧ (∀ g ∈ ins, (C.op g).isInput = true) ∧
    ins.length = 32 * O + (if stop then 16 else 0) + 32 ∧ outs.length = 16 + 32 * (O + 1) ∧
    ∀ (egress : List ℕ) (e : EndRow) (w : Option StopWord), w.isSome = stop → WellFormed O w egress e →
      ∀ a : Fin C.N → Bool,
        (∀ i (h : i < ins.length), a (ins.get ⟨i, h⟩) = (argumentBits O w egress e).getD i false) →
          outs.map (C.eval a) = outputBits O w egress e

/-- What an accepted block holds a release to: `released`, the signed `response.i` row's ids its o1 opens to, are
the committed egress row's ids below the count, and the block refused nothing. -/
def ReleaseHeld (O : ℕ) (stop : Option StopWord) (egress : List ℕ) (e : EndRow) (released : List ℕ) : Prop :=
  refused O stop e = false ∧ released = responseIds O egress (count stop e)

/-- Open (goal `network/egress-tied-to-trace`): where o0 is `accept`, o1's ids hold the release. -/
def OutputsHoldRelease : Prop :=
  ∀ (O : ℕ) (stop : Option StopWord) (egress : List ℕ) (e : EndRow),
    (outputs O stop egress e).1 = accept → ReleaseHeld O stop egress e (outputs O stop egress e).2.2

/-- Open (goal `network/egress-tied-to-trace`): each registered circuit of the two Definitions computes `outputs`.
`read O stop` is the verifier's parsed circuit file of `ResponseTieStop{O}` (where `stop` holds) or
`ResponseTieNoStop{O}` read as a Boolean circuit with its signature's input and output wires, `none` where no such
circuit is registered; this tree does not name that reading yet. -/
def ResponseTieComputesSpec
    (read : ℕ → Bool → Option ((C : Verity.Boolean.Circuit) × List (Fin C.N) × List (Fin C.N))) : Prop :=
  ∀ O stop, ∀ r ∈ read O stop, Computes r.1 r.2.1 r.2.2 O stop

end NetTiming.ResponseTie
