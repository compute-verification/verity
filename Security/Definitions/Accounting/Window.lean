import Definitions.Pous.P2.Cost

/-!
# A location's window with the ring's oracle locations (trusted)

The vocabulary of B2's oracle-location form (`note:console/20261012T0645Z-draft-actual-circuit-model`, B2's row at
13:50Z and 14:05Z). The ring's multiplications, its additions and `one`, and its keys sit at oracle locations of their
own, each stateless and answering only its querier a fixed number of ticks after it is asked.

A `RingWindow` is what a location does with them during one window of `span` ticks, given its state and challenge
there: at tick `n < span` it asks `ask ds`, a function of the batches of answers delivered at the ticks before `n`
(`ds`, one batch per tick); a query `q` asked at tick `n` is answered at tick `n + lat q`; and what the window yields
is `out ds fl`, a function of every batch delivered in it and of the answers still in flight at its end (`fl`, which
reach the location in the next window). `trace f n` is the queries asked at the ticks before `n`, each with its tick,
and the batches delivered at those ticks, when the oracle locations answer `f`; `result f` is what the window yields,
and `keys f` and `ops f` the key and ring queries it asks.
-/

namespace Accounting

open Pous

/-- What a location does with the ring's oracle locations during one window: its span in ticks, the latency of each
query, the queries it asks at a tick from the batches delivered before it, and what it yields from every batch and the
answers still in flight at the end. -/
structure RingWindow (B w : ℕ) (α : Type) : Type where
  span : ℕ
  lat : RingQry B w → ℕ
  ask : List (List (Bits w)) → List (RingQry B w)
  out : List (List (Bits w)) → List (Bits w) → α

namespace RingWindow

variable {B w : ℕ} {α : Type} (W : RingWindow B w α)

/-- The answers by `g` to the queries of `as` (each with the tick it was asked at) that are due at tick `n`. -/
def dueAt (g : RingQry B w → Bits w) (as : List (ℕ × RingQry B w)) (n : ℕ) : List (Bits w) :=
  (as.filter fun e => decide (e.1 + W.lat e.2 = n)).map fun e => g e.2

/-- The answers by `g` to the queries of `as` still in flight at the window's end. -/
def inFlight (g : RingQry B w → Bits w) (as : List (ℕ × RingQry B w)) : List (Bits w) :=
  (as.filter fun e => decide (W.span ≤ e.1 + W.lat e.2)).map fun e => g e.2

/-- The queries asked at the ticks before `n`, each with its tick, and the batches delivered at those ticks, when `f`
answers. -/
def trace (f : RingQry B w → Bits w) : ℕ → List (ℕ × RingQry B w) × List (List (Bits w))
  | 0 => ([], [])
  | n + 1 =>
    let as := (trace f n).1 ++ (W.ask (trace f n).2).map fun q => (n, q)
    (as, (trace f n).2 ++ [W.dueAt f as n])

/-- What the window yields when `f` answers. -/
def result (f : RingQry B w → Bits w) : α :=
  W.out (W.trace f W.span).2 (W.inFlight f (W.trace f W.span).1)

/-- The key queries the window asks when `f` answers. -/
def keys (f : RingQry B w → Bits w) : ℕ :=
  (W.trace f W.span).1.countP fun e => e.2.isKey

/-- The ring queries (`add`, `mul`, `one`) the window asks when `f` answers. -/
def ops (f : RingQry B w → Bits w) : ℕ :=
  (W.trace f W.span).1.countP fun e => !e.2.isKey

end RingWindow

end Accounting
