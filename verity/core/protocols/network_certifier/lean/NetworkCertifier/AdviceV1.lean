/-!
# The certifier program: advice v1's charge

The executable forms of `Security/Definitions/NetworkCertifier/AdviceV1.lean` that the charge's difftest runs, under the
same names in `NetworkCertifier.AdviceV1`. Lean core only: `bitlen` is Lean core's `Nat.log2` plus one where the reference
reads Mathlib's `Nat.size`, and `charges` takes the windows' decision streams as a list where the reference takes them as
a function on `Fin W`. `Proofs.NetworkCertifier.Code` (`CodeChargesV1`) proves `charges` equal to the reference's.
-/

namespace NetworkCertifier.AdviceV1

/-- `HALF`: the coder's ends are 32 bits (`PREC`). -/
def HALF : Nat := 2 ^ 31

/-- `Q1`. -/
def Q1 : Nat := 2 ^ 30

/-- `Q3`. -/
def Q3 : Nat := 3 * 2 ^ 30

/-- `TOP`. -/
def TOP : Nat := 2 ^ 32 - 1

/-- `MAX_TOTAL`: a symbol's total. -/
def MAX_TOTAL : Nat := 2 ^ 16

/-- `K_MAX`: the scalings one symbol may take. -/
def K_MAX : Nat := 18

/-- `CW`: the count's width, a window's charge below `2^CW`. -/
def CW : Nat := 20

/-- `KW`: a KT count's width. -/
def KW : Nat := 15

/-- The counting encoder's state (`Tally`): the interval `[low, high]`, `used` and the segment bits counted. -/
structure Tally where
  low : Nat
  high : Nat
  used : Bool
  bits : Nat

/-- A window's first state (`Tally()`). -/
def Tally.fresh : Tally := ⟨0, TOP, false, 0⟩

/-- The interval scales: E1 or E2, or E3. -/
def scales (lo hi : Nat) : Bool := decide (hi < HALF ∨ HALF ≤ lo ∨ (Q1 ≤ lo ∧ hi < Q3))

/-- One scaling (`_scale`'s body). -/
def shift (lo hi : Nat) : Nat × Nat :=
  if hi < HALF ∨ HALF ≤ lo then (2 * (lo % HALF), 2 * (hi % HALF) + 1) else (2 * (lo - Q1), 2 * (hi - Q1) + 1)

/-- The scalings (`_scale`), at most `k` of them, or `none` when a symbol would take more. -/
def scale : Nat → Nat → Nat → Option (Nat × Nat × Nat)
  | 0, lo, hi => if scales lo hi then none else some (lo, hi, 0)
  | k + 1, lo, hi =>
    if scales lo hi then (scale k (shift lo hi).1 (shift lo hi).2).map fun p => (p.1, p.2.1, p.2.2 + 1)
    else some (lo, hi, 0)

/-- One symbol, counted (`reference_count_step` with `BIN` = 0 and `en` = 1). -/
def countStep (st : Tally) (cum freq total : Nat) : Option Tally :=
  if 1 ≤ freq ∧ cum + freq ≤ total ∧ total ≤ MAX_TOTAL then
    (scale K_MAX (st.low + (st.high + 1 - st.low) * cum / total)
        (st.low + (st.high + 1 - st.low) * (cum + freq) / total - 1)).bind fun p =>
      if st.bits + p.2.2 < 2 ^ CW then some ⟨p.1, p.2.1, st.used || freq != total, st.bits + p.2.2⟩ else none
  else none

/-- One decision, counted (`reference_count_step` with `BIN` = 1). -/
def bitStep (st : Tally) (f0 f1 : Nat) (b : Bool) : Option Tally :=
  if 1 ≤ f0 ∧ 1 ≤ f1 ∧ f0 + f1 ≤ MAX_TOTAL then countStep st (if b then f0 else 0) (if b then f1 else f0) (f0 + f1)
  else none

/-- The book's KT estimate (`reference_book_kt`'s `f0, f1`). -/
def ktFreq (counts : List (Nat × Nat)) (ctx : Nat) : Option (Nat × Nat) :=
  if ctx < 3 ∧ (counts.getD ctx (0, 0)).1 + (counts.getD ctx (0, 0)).2 < 2 ^ KW then
    some (2 * (counts.getD ctx (0, 0)).1 + 1, 2 * (counts.getD ctx (0, 0)).2 + 1)
  else none

/-- The bit `b` raises its count, and once the two reach `2^KW`, both are halved, rounding up. -/
def ktBump (c : Nat × Nat) (b : Bool) : Nat × Nat :=
  let m := if b then (c.1, c.2 + 1) else (c.1 + 1, c.2)
  if 2 ^ KW ≤ m.1 + m.2 then ((m.1 + 1) / 2, (m.2 + 1) / 2) else m

/-- The book's KT counts after a decision (`reference_book_kt`'s `counts'`). -/
def ktRaise (counts : List (Nat × Nat)) (ctx : Nat) (b : Bool) : List (Nat × Nat) :=
  counts.set ctx (ktBump (counts.getD ctx (0, 0)) b)

/-- What a window carries to the next for the decisions (`Carry`). -/
structure Carry where
  counts : List (Nat × Nat)
  run : Nat

/-- The run after a decision. -/
def nextRun (run : Nat) (b : Bool) : Nat := if b then 0 else min (run + 1) 2

/-- The decisions (`reference_charge`'s loop over the slots that ask). -/
def decisions : Tally → Carry → List Bool → Option (Tally × Carry)
  | st, cy, [] => some (st, cy)
  | st, cy, b :: ds =>
    (ktFreq cy.counts cy.run).bind fun f =>
      (bitStep st f.1 f.2 b).bind fun st' => decisions st' ⟨ktRaise cy.counts cy.run b, nextRun cy.run b⟩ ds

/-- The counts sent as 0 (`ZERO_COUNTS`). -/
def zeros : Tally → List (Nat × Nat) → Option Tally
  | st, [] => some st
  | st, z :: zs => (countStep st 0 z.1 z.2).bind fun st' => zeros st' zs

/-- The finish, counted (`reference_count_finish`). -/
def finish (st : Tally) : Option Nat :=
  if st.bits + 2 * st.used.toNat < 2 ^ CW then some (st.bits + 2 * st.used.toNat) else none

/-- What the charge reads of the registered book (`ChargeBook`, and `charge_book`'s first carry). -/
structure Book where
  arrived : List Nat
  total : Nat
  zeros : List (Nat × Nat)
  priors : List (Nat × Nat)

/-- `n`'s bit length (Python's `int.bit_length`, Mathlib's `Nat.size`). -/
def bitlen (n : Nat) : Nat := if n = 0 then 0 else n.log2 + 1

/-- n_arrived: its Exp-Golomb bin `j = bitlen n` at the book's table, then, once `j ≥ 2`, its offset in the bin. -/
def arrive (bk : Book) (n : Nat) : Option Tally :=
  (countStep Tally.fresh (bk.arrived.take (bitlen n)).sum (bk.arrived.getD (bitlen n) 0) bk.total).bind fun st =>
    if 2 ≤ bitlen n then countStep st (n - 2 ^ (bitlen n - 1)) 1 (2 ^ (bitlen n - 1)) else some st

/-- One window's charge (`reference_charge`, `ChargeCount_v1{N}`), and the next window's carry. -/
def window (bk : Book) (N : Nat) (cy : Carry) (ds : List Bool) : Option (Nat × Carry) :=
  if ds.length ≤ N ∧ cy.run < 3 then
    (arrive bk ds.length).bind fun st =>
      (decisions st cy ds).bind fun p => ((zeros p.1 bk.zeros).bind finish).map fun L => (L, p.2)
  else none

/-- A run's charges: each window's, the carry passed along. -/
def session (bk : Book) (N : Nat) : Carry → List (List Bool) → Option (List Nat)
  | _, [] => some []
  | cy, ds :: ws => (window bk N cy ds).bind fun p => (session bk N p.2 ws).map fun ls => p.1 :: ls

/-- The run's charges per window, from the book's priors and run 0. -/
def charges (bk : Book) (N : Nat) (ws : List (List Bool)) : Option (List Nat) :=
  session bk N ⟨bk.priors, 0⟩ ws

end NetworkCertifier.AdviceV1
