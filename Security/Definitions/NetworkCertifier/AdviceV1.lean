import Mathlib.Data.Nat.Size
import Mathlib.Data.List.OfFn

/-!
# Advice v1's charge: the counted coded length of a decision stream under the registered book

Definitions only. They mirror `verity/core/protocols/certifier/arith_encode.py`'s references one for one, with each
Python name in its docstring: `ChargeCount_v1{N}`'s charge (`reference_charge`), the bits ENC-CODE's 32-bit
Witten–Neal–Cleary arithmetic coder (`benchmarks/network_e2e/encodings/code.py`'s `ArithEncoder`) would write for a
window that sends only decisions, counted rather than written (the counting variant, which the composition charges).

A window codes the number of decisions it sends (n_arrived) as the book's Exp-Golomb bin and its offset in the bin, then
each decision at the book's KT estimate for its context `min(run, 2)`, then the four counts the window sends as 0, and
the finish. The window's charge is the scalings the coder takes, one segment bit each, and 2 more at the finish of a
window that used the coder.

**Scope: decision-only windows.** The windows modelled send decisions only: their four counts are sent as 0 (`zeros`),
which is `ChargeCount_v1`'s scope. A window that sends a nonzero count is not covered.

**The decision stream.** A window's slots are the scan's asks, and `ChargeCount_v1`'s witness is which slots ask (`en`)
and their decisions (`bit`). A slot that doesn't ask changes nothing, so the charge is a function of the decisions of the
slots that ask, in order: the window's decision stream, a `List Bool`. The run's decision stream is one per window. The
asks themselves are not advice: the compile slice, its charge included, is proved on every advised window and never
sampled, so `en` is the scan that slice proves, pinned by it. The carry from window to window (the KT counts and the
run) is committed across windows and chained from the book's priors (`charge_chain.check_chain`), as `charges`
starts from `priors` and passes each window's carry on.

**What is left.** That `ChargeCount_v1`'s circuit computes `window` is not proved in Lean. circuit-check holds the circuit
to the Python reference, and these definitions are held to that reference by the charge difftest: the certifier
program's `NetworkCertifier.AdviceV1`, which `Proofs.NetworkCertifier.Code` proves equal to these (`CodeChargesV1`), gives
what `reference_charge` gives on 4,000 random instances, accepted and refused (`runs` in the certifier program's
`lean-audit.json`).

**Refusals.** Each reference refuses (ok 0) where the Python does, and then specifies nothing else, so each function here
returns `none` there. On the states a window reaches (`low ≤ high < 2^32`), the Python's masks `& (HALF − 1)` are
`% HALF`, as below.
-/

namespace NetTiming.AdviceV1

/-- `HALF`: the coder's ends are 32 bits (`PREC`). -/
def HALF : ℕ := 2 ^ 31

/-- `Q1`. -/
def Q1 : ℕ := 2 ^ 30

/-- `Q3`. -/
def Q3 : ℕ := 3 * 2 ^ 30

/-- `TOP`. -/
def TOP : ℕ := 2 ^ 32 - 1

/-- `MAX_TOTAL`: a symbol's total. -/
def MAX_TOTAL : ℕ := 2 ^ 16

/-- `K_MAX`: the scalings one symbol may take. -/
def K_MAX : ℕ := 18

/-- `CW`: the count's width, a window's charge below `2^CW`. -/
def CW : ℕ := 20

/-- `KW`: a KT count's width. -/
def KW : ℕ := 15

/-- **The counting encoder's state** (`Tally`): code.py's interval `[low, high]`, `used` (a symbol of frequency below
its total was coded) and `bits`, the segment bits counted. -/
structure Tally where
  low : ℕ
  high : ℕ
  used : Bool
  bits : ℕ

/-- A window's first state (`Tally()`). -/
def Tally.fresh : Tally := ⟨0, TOP, false, 0⟩

/-- code.py's loop scales the interval: E1 or E2 (`high < HALF or low >= HALF`), or E3. -/
def scales (lo hi : ℕ) : Bool := decide (hi < HALF ∨ HALF ≤ lo ∨ (Q1 ≤ lo ∧ hi < Q3))

/-- One scaling (`_scale`'s body): E1 and E2 drop the top bit, E3 subtracts `Q1`, then both ends shift up. -/
def shift (lo hi : ℕ) : ℕ × ℕ :=
  if hi < HALF ∨ HALF ≤ lo then (2 * (lo % HALF), 2 * (hi % HALF) + 1) else (2 * (lo - Q1), 2 * (hi - Q1) + 1)

/-- **The scalings** (`_scale`), at most `k` of them: the interval after them and their number, or `none` when a
symbol would take more than `k` (`reference_count_step`'s `n > K`). -/
def scale : ℕ → ℕ → ℕ → Option (ℕ × ℕ × ℕ)
  | 0, lo, hi => if scales lo hi then none else some (lo, hi, 0)
  | k + 1, lo, hi =>
    if scales lo hi then (scale k (shift lo hi).1 (shift lo hi).2).map fun p => (p.1, p.2.1, p.2.2 + 1)
    else some (lo, hi, 0)

/-- **One symbol, counted** (`reference_count_step` with `BIN` = 0 and `en` = 1): code.py's `encode(cum, freq,
total)`, then its scalings added to `bits`. It refuses a symbol code.py refuses (`_symbol`), more than `K_MAX`
scalings, and `bits` past `CW` bits. -/
def countStep (st : Tally) (cum freq total : ℕ) : Option Tally :=
  if 1 ≤ freq ∧ cum + freq ≤ total ∧ total ≤ MAX_TOTAL then
    (scale K_MAX (st.low + (st.high + 1 - st.low) * cum / total)
        (st.low + (st.high + 1 - st.low) * (cum + freq) / total - 1)).bind fun p =>
      if st.bits + p.2.2 < 2 ^ CW then some ⟨p.1, p.2.1, st.used || freq != total, st.bits + p.2.2⟩ else none
  else none

/-- **One decision, counted** (`reference_count_step` with `BIN` = 1): the bit `b` at frequencies `f0, f1`, coded as
`encode(f0 if b else 0, f1 if b else f0, f0 + f1)`. -/
def bitStep (st : Tally) (f0 f1 : ℕ) (b : Bool) : Option Tally :=
  if 1 ≤ f0 ∧ 1 ≤ f1 ∧ f0 + f1 ≤ MAX_TOTAL then countStep st (if b then f0 else 0) (if b then f1 else f0) (f0 + f1)
  else none

/-- **The book's KT estimate** (`reference_book_kt`'s `f0, f1`): `2n + 1` for each count of context `ctx`. It refuses a
context past the three, and counts the book can't hold. -/
def ktFreq (counts : List (ℕ × ℕ)) (ctx : ℕ) : Option (ℕ × ℕ) :=
  if ctx < 3 ∧ (counts.getD ctx (0, 0)).1 + (counts.getD ctx (0, 0)).2 < 2 ^ KW then
    some (2 * (counts.getD ctx (0, 0)).1 + 1, 2 * (counts.getD ctx (0, 0)).2 + 1)
  else none

/-- The bit `b` raises its count, and once the two reach `2^KW`, both are halved, rounding up. -/
def ktBump (c : ℕ × ℕ) (b : Bool) : ℕ × ℕ :=
  let m := if b then (c.1, c.2 + 1) else (c.1 + 1, c.2)
  if 2 ^ KW ≤ m.1 + m.2 then ((m.1 + 1) / 2, (m.2 + 1) / 2) else m

/-- **The book's KT counts after a decision** (`reference_book_kt`'s `counts'`). -/
def ktRaise (counts : List (ℕ × ℕ)) (ctx : ℕ) (b : Bool) : List (ℕ × ℕ) :=
  counts.set ctx (ktBump (counts.getD ctx (0, 0)) b)

/-- **What a window carries to the next for the decisions** (`Carry`): each context's KT counts `(n0, n1)`, and
`min(run, 2)`, the decisions since the last arrival. -/
structure Carry where
  counts : List (ℕ × ℕ)
  run : ℕ

/-- The run after a decision: an arrival (`b`) resets it, and it stops at 2. -/
def nextRun (run : ℕ) (b : Bool) : ℕ := if b then 0 else min (run + 1) 2

/-- **The decisions** (`reference_charge`'s loop over the slots that ask): each at the book's estimate for its context
`min(run, 2)`. -/
def decisions : Tally → Carry → List Bool → Option (Tally × Carry)
  | st, cy, [] => some (st, cy)
  | st, cy, b :: ds =>
    (ktFreq cy.counts cy.run).bind fun f =>
      (bitStep st f.1 f.2 b).bind fun st' => decisions st' ⟨ktRaise cy.counts cy.run b, nextRun cy.run b⟩ ds

/-- **The counts sent as 0** (`ZERO_COUNTS`), each at its table's bin 0 `(freq, total)`. -/
def zeros : Tally → List (ℕ × ℕ) → Option Tally
  | st, [] => some st
  | st, z :: zs => (countStep st 0 z.1 z.2).bind fun st' => zeros st' zs

/-- **The finish, counted** (`reference_count_finish`): the window's charge, `bits` and 2 more when `used`, which must
be below `2^CW`. -/
def finish (st : Tally) : Option ℕ :=
  if st.bits + 2 * st.used.toNat < 2 ^ CW then some (st.bits + 2 * st.used.toNat) else none

/-- **What the charge reads of the registered book** (`ChargeBook`, and `charge_book`'s first carry): `arrived`,
n_arrived's bin frequencies, and `total`, its table's total; `zeros`, each zero count's `(freq, total)` at bin 0; and
`priors`, the contexts' first KT counts. -/
structure Book where
  arrived : List ℕ
  total : ℕ
  zeros : List (ℕ × ℕ)
  priors : List (ℕ × ℕ)

/-- **n_arrived** (`reference_charge`'s first two steps): its Exp-Golomb bin `j = bitlen n` at the book's table, then,
once `j ≥ 2`, its offset in the bin, uniform. -/
def arrive (bk : Book) (n : ℕ) : Option Tally :=
  (countStep Tally.fresh (bk.arrived.take n.size).sum (bk.arrived.getD n.size 0) bk.total).bind fun st =>
    if 2 ≤ n.size then countStep st (n - 2 ^ (n.size - 1)) 1 (2 ^ (n.size - 1)) else some st

/-- **One window's charge** (`reference_charge`, `ChargeCount_v1{N}`): of a window of `N` slots whose slots that ask
decide `ds`, under the book `bk` and the carry `cy`, the charge and the next window's carry. -/
def window (bk : Book) (N : ℕ) (cy : Carry) (ds : List Bool) : Option (ℕ × Carry) :=
  if ds.length ≤ N ∧ cy.run < 3 then
    (arrive bk ds.length).bind fun st =>
      (decisions st cy ds).bind fun p => ((zeros p.1 bk.zeros).bind finish).map fun L => (L, p.2)
  else none

/-- **A run's charges**: each window's, the carry passed along. -/
def session (bk : Book) (N : ℕ) : Carry → List (List Bool) → Option (List ℕ)
  | _, [] => some []
  | cy, ds :: ws => (window bk N cy ds).bind fun p => (session bk N p.2 ws).map fun ls => p.1 :: ls

/-- **The run's charges per window**, from the book's priors and run 0, for the decision streams of `W` windows. -/
def charges (bk : Book) (N : ℕ) {W : ℕ} (a : Fin W → List Bool) : Option (List ℕ) :=
  session bk N ⟨bk.priors, 0⟩ (List.ofFn a)

/-- **Advice v1's decision streams**: those of `W` windows whose charge no window refuses. A refused window fails its
compile unit, so no accepted run has other advice. -/
def Streams (bk : Book) (N W : ℕ) : Type := {a : Fin W → List Bool // (charges bk N a).isSome}

/-- **Advice v1's charge** (I1's `bits(advice, w)`): window `w`'s. -/
def bits (bk : Book) (N : ℕ) {W : ℕ} (a : Streams bk N W) (w : Fin W) : ℕ :=
  ((charges bk N a.1).getD []).getD w 0

end NetTiming.AdviceV1
