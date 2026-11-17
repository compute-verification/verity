module

public import Mathlib.Data.Nat.Bits
public import Mathlib.Data.Nat.Size
public import Mathlib.Data.Subtype
public import Mathlib.Tactic.Attr.Core
public import Mathlib.Tactic.ToAdditive
import Mathlib.Tactic.Finiteness.Attr

/-! Advice v1's charge, mirroring `arith_encode.py`'s references: it covers decision-only windows, and that
`ChargeCount_v1`'s circuit computes `executionPhase` is not proved in Lean. -/

@[expose] public section

namespace NetTiming.AdviceV1

/-- `HALF`: the coder's ends are 32 bits (`PREC`). -/
def HALF : ℕ := 2 ^ 31

def Q1 : ℕ := 2 ^ 30

def Q3 : ℕ := 3 * 2 ^ 30

def TOP : ℕ := 2 ^ 32 - 1

/-- `MAX_TOTAL`: a symbol's total. -/
def MAX_TOTAL : ℕ := 2 ^ 16

/-- `K_MAX`: the scalings one symbol may take. -/
def K_MAX : ℕ := 18

/-- `CW`: the count's width, an execution phase's charge below `2^CW`. -/
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

def Tally.fresh : Tally := ⟨0, TOP, false, 0⟩

def scales (lo hi : ℕ) : Bool := decide (hi < HALF ∨ HALF ≤ lo ∨ (Q1 ≤ lo ∧ hi < Q3))

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

def bitStep (st : Tally) (f0 f1 : ℕ) (b : Bool) : Option Tally :=
  if 1 ≤ f0 ∧ 1 ≤ f1 ∧ f0 + f1 ≤ MAX_TOTAL then countStep st (if b then f0 else 0) (if b then f1 else f0) (f0 + f1)
  else none

/-- **The book's KT estimate** (`reference_book_kt`'s `f0, f1`): `2n + 1` for each count of context `ctx`. It refuses a
context past the three, and counts the book can't hold. -/
def ktFreq (counts : List (ℕ × ℕ)) (ctx : ℕ) : Option (ℕ × ℕ) :=
  if ctx < 3 ∧ (counts.getD ctx (0, 0)).1 + (counts.getD ctx (0, 0)).2 < 2 ^ KW then
    some (2 * (counts.getD ctx (0, 0)).1 + 1, 2 * (counts.getD ctx (0, 0)).2 + 1)
  else none

def ktBump (c : ℕ × ℕ) (b : Bool) : ℕ × ℕ :=
  let m := if b then (c.1, c.2 + 1) else (c.1 + 1, c.2)
  if 2 ^ KW ≤ m.1 + m.2 then ((m.1 + 1) / 2, (m.2 + 1) / 2) else m

def ktRaise (counts : List (ℕ × ℕ)) (ctx : ℕ) (b : Bool) : List (ℕ × ℕ) :=
  counts.set ctx (ktBump (counts.getD ctx (0, 0)) b)

/-- **What an execution phase carries to the next for the decisions** (`Carry`): each context's KT counts `(n0, n1)`, and
`min(run, 2)`, the decisions since the last arrival. -/
structure Carry where
  counts : List (ℕ × ℕ)
  run : ℕ

def nextRun (run : ℕ) (b : Bool) : ℕ := if b then 0 else min (run + 1) 2

def decisions : Tally → Carry → List Bool → Option (Tally × Carry)
  | st, cy, [] => some (st, cy)
  | st, cy, b :: ds =>
    (ktFreq cy.counts cy.run).bind fun f =>
      (bitStep st f.1 f.2 b).bind fun st' => decisions st' ⟨ktRaise cy.counts cy.run b, nextRun cy.run b⟩ ds

def zeros : Tally → List (ℕ × ℕ) → Option Tally
  | st, [] => some st
  | st, z :: zs => (countStep st 0 z.1 z.2).bind fun st' => zeros st' zs

/-- **The finish, counted** (`reference_count_finish`): the execution phase's charge, `bits` and 2 more when `used`, which must
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

def arrive (bk : Book) (n : ℕ) : Option Tally :=
  (countStep Tally.fresh (bk.arrived.take n.size).sum (bk.arrived.getD n.size 0) bk.total).bind fun st =>
    if 2 ≤ n.size then countStep st (n - 2 ^ (n.size - 1)) 1 (2 ^ (n.size - 1)) else some st

def executionPhase (bk : Book) (N : ℕ) (cy : Carry) (ds : List Bool) : Option (ℕ × Carry) :=
  if ds.length ≤ N ∧ cy.run < 3 then
    (arrive bk ds.length).bind fun st =>
      (decisions st cy ds).bind fun p => ((zeros p.1 bk.zeros).bind finish).map fun L => (L, p.2)
  else none

def chargesFrom (bk : Book) (N : ℕ) : Carry → List (List Bool) → Option (List ℕ)
  | _, [] => some []
  | cy, ds :: ws => (executionPhase bk N cy ds).bind fun p => (chargesFrom bk N p.2 ws).map fun ls => p.1 :: ls

def charges (bk : Book) (N : ℕ) {W : ℕ} (a : Fin W → List Bool) : Option (List ℕ) :=
  chargesFrom bk N ⟨bk.priors, 0⟩ (List.ofFn a)

/-- A refused execution phase fails its compile slice's proof block, so no accepted run has other advice. -/
def Streams (bk : Book) (N W : ℕ) : Type := {a : Fin W → List Bool // (charges bk N a).isSome}

def bits (bk : Book) (N : ℕ) {W : ℕ} (a : Streams bk N W) (w : Fin W) : ℕ :=
  ((charges bk N a.1).getD []).getD w 0

end NetTiming.AdviceV1
