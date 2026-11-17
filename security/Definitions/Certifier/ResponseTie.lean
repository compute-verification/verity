module

public import Definitions.Core.Boolean

/-! The response tie's two Definitions and what a release is held to; `OutputsHoldRelease` and `ResponseTieComputesSpec`
are open, and nothing proves them. -/

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

def count (stop : Option StopWord) (e : EndRow) : ℕ :=
  match stop with
  | some w => if e.cls % 2 = 0 then w.n else e.cells
  | none => e.cells

def pad (O : ℕ) (egress : List ℕ) : List ℕ := egress ++ List.replicate (O - egress.length) 0

def responseIds (O : ℕ) (egress : List ℕ) (k : ℕ) : List ℕ :=
  (pad O egress).take k ++ List.replicate (O - min k O) 0

def outputs (O : ℕ) (stop : Option StopWord) (egress : List ℕ) (e : EndRow) : ℕ × ℕ × List ℕ :=
  if refused O stop e then (0, noCount, responseIds O egress (count stop e))
  else (accept, count stop e, responseIds O egress (count stop e))

def WellFormed (O : ℕ) (stop : Option StopWord) (egress : List ℕ) (e : EndRow) : Prop :=
  egress.length ≤ O ∧ (∀ t ∈ egress, t < 2 ^ 32) ∧ e.cls < 2 ^ 16 ∧ e.cells < 2 ^ 16 ∧
    ∀ w, stop = some w → w.n < 2 ^ 13 ∧ w.finish < 2 ^ 3

/-- The low `w` bits of `x`, least significant first. -/
def bitsLE (w x : ℕ) : List Bool := (List.range w).map x.testBit

def argumentBits (O : ℕ) (stop : Option StopWord) (egress : List ℕ) (e : EndRow) : List Bool :=
  (pad O egress).flatMap (bitsLE 32) ++
    (match stop with
     | some w => bitsLE 16 (w.n + w.finish * 2 ^ 13)
     | none => []) ++
    bitsLE 16 e.cls ++ bitsLE 16 e.cells

def outputBits (O : ℕ) (stop : Option StopWord) (egress : List ℕ) (e : EndRow) : List Bool :=
  match outputs O stop egress e with
  | (v, m, ids) => bitsLE 16 v ++ bitsLE 32 m ++ ids.flatMap (bitsLE 32)

def Computes (C : Verity.Boolean.Circuit) (ins outs : List (Fin C.N)) (O : ℕ) (stop : Bool) : Prop :=
  ins.Nodup ∧ (∀ g ∈ ins, (C.op g).isInput = true) ∧
    ins.length = 32 * O + (if stop then 16 else 0) + 32 ∧ outs.length = 16 + 32 * (O + 1) ∧
    ∀ (egress : List ℕ) (e : EndRow) (w : Option StopWord), w.isSome = stop → WellFormed O w egress e →
      ∀ a : Fin C.N → Bool,
        (∀ i (h : i < ins.length), a (ins.get ⟨i, h⟩) = (argumentBits O w egress e).getD i false) →
          outs.map (C.eval a) = outputBits O w egress e

def ReleaseHeld (O : ℕ) (stop : Option StopWord) (egress : List ℕ) (e : EndRow) (released : List ℕ) : Prop :=
  refused O stop e = false ∧ released = responseIds O egress (count stop e)

/-- Unproved: where o0 is `accept`, o1's ids hold the release. -/
def OutputsHoldRelease : Prop :=
  ∀ (O : ℕ) (stop : Option StopWord) (egress : List ℕ) (e : EndRow),
    (outputs O stop egress e).1 = accept → ReleaseHeld O stop egress e (outputs O stop egress e).2.2

/-- Unproved. `read O stop` is the verifier's parsed circuit file of `ResponseTieStop{O}` or `ResponseTieNoStop{O}` read
as a Boolean circuit, `none` where none is registered; this tree does not name that reading. -/
def ResponseTieComputesSpec
    (read : ℕ → Bool → Option ((C : Verity.Boolean.Circuit) × List (Fin C.N) × List (Fin C.N))) : Prop :=
  ∀ O stop, ∀ r ∈ read O stop, Computes r.1 r.2.1 r.2.2 O stop

end NetTiming.ResponseTie
