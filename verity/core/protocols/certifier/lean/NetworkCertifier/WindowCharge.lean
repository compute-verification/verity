import NetworkCertifier.AdviceV1

/-!
# The certifier program: advice v1's whole window charged (`ChargeCount_v2`)

The executable forms of `Security/Definitions/NetworkCertifier/WindowCharge.lean` that the window charge's difftest
runs, under the same names in `NetworkCertifier.WindowCharge`, on `NetworkCertifier.AdviceV1`'s counting encoder. Lean
core only: a bin is `AdviceV1.bitlen` where the reference reads Mathlib's `Nat.size`, and `charges` takes the windows as
a list where the reference takes them as a function on `Fin W`. `Proofs.NetworkCertifier.Code` (`CodeChargesWhole`)
proves `charges` equal to the reference's.
-/

namespace NetworkCertifier.WindowCharge

open AdviceV1 (Tally countStep finish ktFreq ktRaise nextRun MAX_TOTAL bitlen)

/-- `HIT_CAP`. -/
def HIT_CAP : Nat := 2 ^ 16

/-- `GAP_CAP`. -/
def GAP_CAP : Nat := 2 ^ 24

/-- `INDEX_CAP`. -/
def INDEX_CAP : Nat := 2 ^ 24

/-- `ZIG_CAP`. -/
def ZIG_CAP : Nat := 2 * INDEX_CAP + 1

/-- `_CHUNK`. -/
def CHUNK : Nat := 15

/-- A symbol `(cum, freq, total)`. -/
abbrev Sym : Type := Nat × Nat × Nat

/-- Symbols, counted (`_code`). -/
def codeAll : Tally → List Sym → Option Tally
  | st, [] => some st
  | st, s :: ss => (countStep st s.1 s.2.1 s.2.2).bind fun st' => codeAll st' ss

/-- A uniform value (`_uniform`). -/
def uniform (u n : Nat) : List Sym :=
  if n ≤ MAX_TOTAL then (if 1 < n then [(u, 1, n)] else [])
  else uniform (u / 2 ^ CHUNK) ((n - 1) / 2 ^ CHUNK + 1) ++ [(u % 2 ^ CHUNK, 1, 2 ^ CHUNK)]
termination_by n
decreasing_by
  simp only [CHUNK, MAX_TOTAL] at *
  omega

/-- A bin's first value (`_offset`). -/
def low (j : Nat) : Nat := if j = 0 then 0 else 2 ^ (j - 1)

/-- A table of the book. -/
structure Table where
  freqs : List Nat
  total : Nat

/-- An integer `v ∈ [0, hi]` (`binned_symbols`): its bin, then its offset. -/
def binned (t : Table) (hi v : Nat) : List Sym :=
  ((t.freqs.take (bitlen v)).sum, t.freqs.getD (bitlen v) 0, t.total) ::
    uniform (v - low (bitlen v)) (min (2 ^ bitlen v - 1) hi - low (bitlen v) + 1)

/-- The `hi` of a count over `S` slots. -/
def countHi (S : Nat) : Nat := 2 ^ bitlen S - 1

/-- `zigzag(r − rank)`. -/
def zigzag (r rank : Nat) : Nat := if rank ≤ r then 2 * (r - rank) else 2 * (rank - r) - 1

/-- What the whole window's charge reads of the registered book (`WindowBook`, and `first_carry`). -/
structure Book where
  arrived : Table
  nHits : Table
  hit : Table
  resumes : Table
  nOrder : Table
  gap : Table
  index : Table
  lag : Table
  priors : List (Nat × Nat)

/-- What a window carries to the next (`WindowCarry`). -/
structure Carry where
  dec : AdviceV1.Carry
  pos : Nat
  rank : Nat

/-- One window's advice: its decisions, its hits' values and its order entries `(p, r)`, each of the slots that hold
one, in order. -/
structure Advice where
  ds : List Bool
  hits : List Nat
  order : List (Nat × Nat)

/-- The decisions' symbols, and the carry after them. -/
def decSyms : AdviceV1.Carry → List Bool → Option (List Sym × AdviceV1.Carry)
  | cy, [] => some ([], cy)
  | cy, b :: ds =>
    (ktFreq cy.counts cy.run).bind fun f =>
      if 1 ≤ f.1 ∧ 1 ≤ f.2 ∧ f.1 + f.2 ≤ MAX_TOTAL then
        (decSyms ⟨ktRaise cy.counts cy.run b, nextRun cy.run b⟩ ds).map fun p =>
          ((if b then f.1 else 0, if b then f.2 else f.1, f.1 + f.2) :: p.1, p.2)
      else none

/-- The hits' symbols. -/
def hitSyms (bk : Book) : List Nat → Option (List Sym)
  | [] => some []
  | s :: ss => if s ≤ HIT_CAP then (hitSyms bk ss).map fun l => binned bk.hit HIT_CAP s ++ l else none

/-- The order entries' symbols, and the position and rank after them. -/
def entrySyms (bk : Book) : Nat → Nat → List (Nat × Nat) → Option (List Sym × Nat × Nat)
  | pos, rank, [] => some ([], pos, rank)
  | pos, rank, e :: es =>
    if pos ≤ e.1 ∧ e.1 < 2 ^ 32 - 1 ∧ e.1 - pos ≤ GAP_CAP ∧ e.2 ≤ INDEX_CAP then
      (entrySyms bk (e.1 + 1) (e.2 + 1) es).map fun q =>
        (binned bk.gap GAP_CAP (e.1 - pos) ++ (binned bk.index ZIG_CAP (zigzag e.2 rank) ++ q.1), q.2)
    else none

/-- The window's symbols in `code_v1._window`'s order, and the carry after them. -/
def syms (bk : Book) (N H O : Nat) (cy : Carry) (x : Advice) : Option (List Sym × Carry) :=
  (decSyms cy.dec x.ds).bind fun d =>
    (hitSyms bk x.hits).bind fun hs =>
      (entrySyms bk cy.pos cy.rank x.order).map fun e =>
        (binned bk.arrived (countHi N) x.ds.length ++ (d.1 ++ (binned bk.nHits (countHi H) x.hits.length ++
          (hs ++ (binned bk.resumes 0 0 ++ (binned bk.nOrder (countHi O) x.order.length ++
            (e.1 ++ binned bk.lag 0 0)))))), ⟨d.2, e.2.1, e.2.2⟩)

/-- One window's charge (`reference_window`, `ChargeCount_v2{N, H, O}`), and the next window's carry. -/
def window (bk : Book) (N H O : Nat) (cy : Carry) (x : Advice) : Option (Nat × Carry) :=
  if x.ds.length ≤ N ∧ x.hits.length ≤ H ∧ x.order.length ≤ O ∧ cy.dec.run < 3 ∧ cy.rank ≤ INDEX_CAP + 1 then
    (syms bk N H O cy x).bind fun q => ((codeAll Tally.fresh q.1).bind finish).map fun L => (L, q.2)
  else none

/-- A run's charges: each window's, the carry passed along. -/
def session (bk : Book) (N H O : Nat) : Carry → List Advice → Option (List Nat)
  | _, [] => some []
  | cy, x :: ws => (window bk N H O cy x).bind fun p => (session bk N H O p.2 ws).map fun ls => p.1 :: ls

/-- The session's first carry (`first_carry`). -/
def Book.first (bk : Book) : Carry := ⟨⟨bk.priors, 0⟩, 0, 0⟩

/-- The run's charges per window, from the session's first carry. -/
def charges (bk : Book) (N H O : Nat) (ws : List Advice) : Option (List Nat) :=
  session bk N H O bk.first ws

end NetworkCertifier.WindowCharge
