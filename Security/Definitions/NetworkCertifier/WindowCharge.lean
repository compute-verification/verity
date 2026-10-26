import Definitions.NetworkCertifier.AdviceV1

/-!
# Advice v1's whole window charged (`ChargeCount_v2`): decisions, hits and order entries, as `code_v1` codes them

Definitions only. They mirror `verity/core/protocols/certifier/window_charge.py`'s `reference_window` one for one, with
each Python name in its docstring: `ChargeCount_v2{N, H, O}`'s charge, the length of `code_v1._window`'s segment for
the whole window, counted by advice v1's counting encoder (`AdviceV1.countStep` and `AdviceV1.finish`, from
`arith_encode.py`'s references).

A window sends its decisions, its hits (v1's sent shortfalls, in hit units) and its order entries (each deviating
arrival's position and rank). Its symbols, in `code_v1._window`'s order, all in one segment with one finish:
1. n_arrived, then each decision at the book's KT estimate for its context `min(run, 2)`, as in v1;
2. n_hits, then each hit's value;
3. n_resumes, 0;
4. n_order, then each entry's gap `p − pos` and rank step `zigzag(r − rank)`, then `pos := p + 1` and `rank := r + 1`;
5. n_lag, 0.

Each count and each value is an integer under its table (`binned`): its Exp-Golomb bin, then its offset in the bin,
uniform (`uniform`). The window's charge is the scalings the coder takes for them, and 2 more at the finish of a window
that used the coder. The carry is the decisions' (`AdviceV1.Carry`) and the last order entry's position and rank, each
`+ 1` (0 before the session's first entry), chained from window to window under `window-charge`
(`window_charge.check_chain`).

**Scope.** Every advice v1 window: v1 sends hits and order entries, and never resumes or lag bits (`v1.py` refuses a
window that sends either), so the two are coded as 0. With no hits and no order entries the symbols are
`ChargeCount_v1`'s, with the n_hits, n_resumes, n_order and n_lag tables' bin 0 for its `zeros`.

**What is left.** That `ChargeCount_v2`'s circuit computes `window` is not proved in Lean. circuit-check holds the
circuit to the Python reference, and these definitions are held to that reference by the window charge's difftest
(the certifier program's `NetworkCertifier.WindowCharge`, which `Proofs.NetworkCertifier.Code` proves equal to these
(`CodeChargesWhole`), against `reference_window`).

**Refusals.** The reference refuses (ok 0) where the Python does, and then specifies nothing else, so each function here
returns `none` there.
-/

namespace NetTiming.WindowCharge

open AdviceV1 (Tally countStep finish ktFreq ktRaise nextRun MAX_TOTAL)

/-- `HIT_CAP`: v1's registered cap on a hit, in hit units (`v1.TABLE_CAPS`'s `hit_units`). -/
def HIT_CAP : ℕ := 2 ^ 16

/-- `GAP_CAP`: v1's cap on an order entry's gap (`gap`). -/
def GAP_CAP : ℕ := 2 ^ 24

/-- `INDEX_CAP`: v1's cap on an order entry's rank (`index`). -/
def INDEX_CAP : ℕ := 2 ^ 24

/-- `ZIG_CAP`: a rank step's cap, `2·index + 1`. -/
def ZIG_CAP : ℕ := 2 * INDEX_CAP + 1

/-- `_CHUNK`: a wide offset's last step is uniform over `2^CHUNK` values. -/
def CHUNK : ℕ := 15

/-- A symbol of code.py's encoder, `(cum, freq, total)`. -/
abbrev Sym : Type := ℕ × ℕ × ℕ

/-- **Symbols, counted** (`_code`): each through `AdviceV1.countStep`, in order. -/
def codeAll : Tally → List Sym → Option Tally
  | st, [] => some st
  | st, s :: ss => (countStep st s.1 s.2.1 s.2.2).bind fun st' => codeAll st' ss

/-- **A uniform value** (`_uniform`, code.py's `put_uniform(u, n)`): nothing when `n ≤ 1`, `(u, 1, n)` when
`n ≤ MAX_TOTAL`, and otherwise `⌊u / 2^CHUNK⌋` uniform over `⌊(n − 1) / 2^CHUNK⌋ + 1` values, then
`(u mod 2^CHUNK, 1, 2^CHUNK)`. -/
def uniform (u n : ℕ) : List Sym :=
  if n ≤ MAX_TOTAL then (if 1 < n then [(u, 1, n)] else [])
  else uniform (u / 2 ^ CHUNK) ((n - 1) / 2 ^ CHUNK + 1) ++ [(u % 2 ^ CHUNK, 1, 2 ^ CHUNK)]
termination_by n
decreasing_by
  simp only [CHUNK, MAX_TOTAL] at *
  omega

/-- A bin's first value `a_j` (`_offset`): 0 for bin 0, `2^(j − 1)` for bin `j ≥ 1`. -/
def low (j : ℕ) : ℕ := if j = 0 then 0 else 2 ^ (j - 1)

/-- **A table** of the book: its frequencies at the bins a window reaches (`freqs`) and its total (`total`). -/
structure Table where
  freqs : List ℕ
  total : ℕ

/-- **An integer** `v ∈ [0, hi]` (`binned_symbols`, code.py's `Binned.put(v, 0, hi)`): its Exp-Golomb bin
`j = bitlen v` at the table's `(Σ_{i<j} f_i, f_j, total)`, then its offset `v − a_j`, uniform over the bin's
`min(2^j − 1, hi) − a_j + 1` values. -/
def binned (t : Table) (hi v : ℕ) : List Sym :=
  ((t.freqs.take v.size).sum, t.freqs.getD v.size 0, t.total) ::
    uniform (v - low v.size) (min (2 ^ v.size - 1) hi - low v.size + 1)

/-- The `hi` of a count over `S` slots (`count`'s): `2^bitlen(S) − 1`. -/
def countHi (S : ℕ) : ℕ := 2 ^ S.size - 1

/-- `zigzag(r − rank)`: `2d` for `d = r − rank ≥ 0`, `−2d − 1` for `d < 0`. -/
def zigzag (r rank : ℕ) : ℕ := if rank ≤ r then 2 * (r - rank) else 2 * (rank - r) - 1

/-- **What the whole window's charge reads of the registered book** (`WindowBook`, and `first_carry`): each of
`code_v1._window`'s integer symbols' table (`SYMBOLS`: n_arrived, n_hits, hit, n_resumes, n_order, order_gap,
order_index, n_lag), and `priors`, the decisions' contexts' first KT counts. -/
structure Book where
  arrived : Table
  nHits : Table
  hit : Table
  resumes : Table
  nOrder : Table
  gap : Table
  index : Table
  lag : Table
  priors : List (ℕ × ℕ)

/-- **What a window carries to the next** (`WindowCarry`): the decisions' carry (each context's KT counts and
`min(run, 2)`), and `pos` and `rank`, the last order entry's position and rank, each `+ 1`. -/
structure Carry where
  dec : AdviceV1.Carry
  pos : ℕ
  rank : ℕ

/-- **One window's advice**: the decisions of the slots that hold one, in order (`ds`); the hits' values (`hits`); and
the order entries, each the arrival's position and rank `(p, r)` (`order`). A disabled slot changes nothing, so the
charge is a function of these. -/
structure Window where
  ds : List Bool
  hits : List ℕ
  order : List (ℕ × ℕ)

/-- **The decisions' symbols** (`reference_window`'s loop over the decision slots): each decision `b` at the book's KT
estimate `(f0, f1)` for its context, `(f0 if b else 0, f1 if b else f0, f0 + f1)` (`AdviceV1.bitStep`'s), and the
carry after them. -/
def decSyms : AdviceV1.Carry → List Bool → Option (List Sym × AdviceV1.Carry)
  | cy, [] => some ([], cy)
  | cy, b :: ds =>
    (ktFreq cy.counts cy.run).bind fun f =>
      if 1 ≤ f.1 ∧ 1 ≤ f.2 ∧ f.1 + f.2 ≤ MAX_TOTAL then
        (decSyms ⟨ktRaise cy.counts cy.run b, nextRun cy.run b⟩ ds).map fun p =>
          ((if b then f.1 else 0, if b then f.2 else f.1, f.1 + f.2) :: p.1, p.2)
      else none

/-- **The hits' symbols**: each hit's value under the hit table, `hi = HIT_CAP`. A hit past `HIT_CAP` is refused. -/
def hitSyms (bk : Book) : List ℕ → Option (List Sym)
  | [] => some []
  | s :: ss => if s ≤ HIT_CAP then (hitSyms bk ss).map fun l => binned bk.hit HIT_CAP s ++ l else none

/-- **The order entries' symbols**: each entry `(p, r)` from `(pos, rank)`, its gap `p − pos` (`hi = GAP_CAP`), then its
rank step `zigzag(r − rank)` (`hi = ZIG_CAP`), then `pos := p + 1` and `rank := r + 1`. An entry with `p < pos`,
`p = 2^32 − 1`, a gap past `GAP_CAP` or `r` past `INDEX_CAP` is refused. -/
def entrySyms (bk : Book) : ℕ → ℕ → List (ℕ × ℕ) → Option (List Sym × ℕ × ℕ)
  | pos, rank, [] => some ([], pos, rank)
  | pos, rank, e :: es =>
    if pos ≤ e.1 ∧ e.1 < 2 ^ 32 - 1 ∧ e.1 - pos ≤ GAP_CAP ∧ e.2 ≤ INDEX_CAP then
      (entrySyms bk (e.1 + 1) (e.2 + 1) es).map fun q =>
        (binned bk.gap GAP_CAP (e.1 - pos) ++ (binned bk.index ZIG_CAP (zigzag e.2 rank) ++ q.1), q.2)
    else none

/-- **The window's symbols** (`reference_window`), in `code_v1._window`'s order, and the carry after them: n_arrived,
the decisions, n_hits, the hits, n_resumes (0), n_order, the entries, n_lag (0). A count over `S` slots has
`hi = 2^bitlen(S) − 1`. -/
def syms (bk : Book) (N H O : ℕ) (cy : Carry) (x : Window) : Option (List Sym × Carry) :=
  (decSyms cy.dec x.ds).bind fun d =>
    (hitSyms bk x.hits).bind fun hs =>
      (entrySyms bk cy.pos cy.rank x.order).map fun e =>
        (binned bk.arrived (countHi N) x.ds.length ++ (d.1 ++ (binned bk.nHits (countHi H) x.hits.length ++
          (hs ++ (binned bk.resumes 0 0 ++ (binned bk.nOrder (countHi O) x.order.length ++
            (e.1 ++ binned bk.lag 0 0)))))), ⟨d.2, e.2.1, e.2.2⟩)

/-- **One window's charge** (`reference_window`, `ChargeCount_v2{N, H, O}`): of a window of `N` decision slots, `H` hit
slots and `O` order slots, under the book `bk` and the carry `cy`, the charge and the next window's carry. It refuses a
carry whose run is past 2 or whose rank is past `INDEX_CAP + 1`. -/
def window (bk : Book) (N H O : ℕ) (cy : Carry) (x : Window) : Option (ℕ × Carry) :=
  if x.ds.length ≤ N ∧ x.hits.length ≤ H ∧ x.order.length ≤ O ∧ cy.dec.run < 3 ∧ cy.rank ≤ INDEX_CAP + 1 then
    (syms bk N H O cy x).bind fun q => ((codeAll Tally.fresh q.1).bind finish).map fun L => (L, q.2)
  else none

/-- **A run's charges**: each window's, the carry passed along. -/
def session (bk : Book) (N H O : ℕ) : Carry → List Window → Option (List ℕ)
  | _, [] => some []
  | cy, x :: ws => (window bk N H O cy x).bind fun p => (session bk N H O p.2 ws).map fun ls => p.1 :: ls

/-- The session's first carry (`first_carry`): the book's priors, run 0, and no order entry yet. -/
def Book.first (bk : Book) : Carry := ⟨⟨bk.priors, 0⟩, 0, 0⟩

/-- **The run's charges per window**, from the session's first carry, for the advice of `W` windows. -/
def charges (bk : Book) (N H O : ℕ) {W : ℕ} (a : Fin W → Window) : Option (List ℕ) :=
  session bk N H O bk.first (List.ofFn a)

/-- **Advice v1's whole windows**: the advice of `W` windows whose charge no window refuses. A refused window fails its
unit, so no accepted run has other advice. -/
def Streams (bk : Book) (N H O W : ℕ) : Type := {a : Fin W → Window // (charges bk N H O a).isSome}

/-- **The whole window's charge** (I1's `bits(advice, w)`): window `w`'s. -/
def bits (bk : Book) (N H O : ℕ) {W : ℕ} (a : Streams bk N H O W) (w : Fin W) : ℕ :=
  ((charges bk N H O a.1).getD []).getD w 0

end NetTiming.WindowCharge
