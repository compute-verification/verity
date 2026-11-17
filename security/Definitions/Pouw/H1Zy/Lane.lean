module

public import Definitions.Pouw.H1Zy.Combine
public import Mathlib.Algebra.BigOperators.Fin
public import Mathlib.Algebra.BigOperators.Group.Finset.Defs
public import Mathlib.Data.Fintype.Basic
import Mathlib.Algebra.GroupWithZero.Nat

/-! `PouwH1LaneZy_v1{TILE}` on words at a 32 × 32 leaf over `PouwH1Tile_v1`, the dequant and the chain's words parameters; the `PouwH1Tile_v3` form is not stated here. -/

@[expose] public section

namespace Pouw.H1Zy

/-- `.satfinite`'s clamp to s32, `[-2^31, 2^31 - 1]`. -/
def satS32 (x : ℤ) : ℤ := max (-2 ^ 31) (min x (2 ^ 31 - 1))

/-- **`BlackwellS8ImmaDot32_v1`** on words (`integer.BLACKWELL_SM120_S8_M16N8K32.step`): the s32 accumulator plus the 32
exact s8 × s8 products, summed exactly and clamped once to s32. -/
def immaS8 (acc : BitVec 32) (a b : Fin 32 → BitVec 8) : BitVec 32 :=
  BitVec.ofInt 32 (satS32 (acc.toInt + ∑ k, (a k).toInt * (b k).toInt))

/-- Code `k` of a row of `K` codes, `0` past its end (`gemmAccS8` reads only `k < 32 ⌊K/32⌋ ≤ K`). -/
def codeAt {K : ℕ} (x : Fin K → BitVec 8) (k : ℕ) : BitVec 8 := if h : k < K then x ⟨k, h⟩ else 0

/-- `GemmAccumulatorS8`'s accumulator after its first `n` steps: step `s` adds codes `32 s .. 32 s + 31`. -/
def gemmAccSteps {K : ℕ} (x w : Fin K → BitVec 8) : ℕ → BitVec 32
  | 0 => 0
  | n + 1 => immaS8 (gemmAccSteps x w n) (fun j => codeAt x (32 * n + j.val)) (fun j => codeAt w (32 * n + j.val))

/-- **`GemmAccumulatorS8_v2{K, BlackwellS8ImmaDot32}`**'s word view (`gemm.GemmAccumulatorS8`): `K/32` steps in ascending
`k` from a zero accumulator. Its bind refuses a `K` that isn't a multiple of 32, or is at least `2^17`
(`S8_K_LIMIT`). -/
def gemmAccS8 {K : ℕ} (x w : Fin K → BitVec 8) : BitVec 32 := gemmAccSteps x w (K / 32)

/-- `PouwZyCell_v1`: `1` exactly when a record's cell `(zr, yr)` is `(z, y)`. -/
def zyCell (zr : BitVec 32) (yr : BitVec 16) (z : BitVec 32) (y : BitVec 16) : Bool :=
  decide (zr = z) && decide (yr = y)

/-- **`PouwZyCheck_v1{R, C}`** on words, at the dequant `dqW`: each cell `(i, j)`, row-major, has record z `z i j` and
record y `dqW (z i j) (sa i) (sw j)` (`PouwZyCell`), ANDed by a `PouwAndBit` tree. -/
def zyCheck {R C : ℕ} (dqW : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 16) (zr : Fin R → Fin C → BitVec 32)
    (yr : Fin R → Fin C → BitVec 16) (z : Fin R → Fin C → BitVec 32) (sa : Fin R → BitVec 32)
    (sw : Fin C → BitVec 32) : Bool :=
  tree andBit true (List.ofFn fun i => List.ofFn fun j => zyCell (zr i j) (yr i j) (z i j) (dqW (z i j) (sa i) (sw j))).flatten

/-- `PouwOrBit_v1`: `a or b`. -/
def orBit (a b : Bool) : Bool := a || b

def mux {α : Type} (s : Bool) (a b : α) : α := if s then b else a

/-- `PouwMask32_v1`: each bit of `w` AND `live`. -/
def mask32 (w : BitVec 32) (live : Bool) : BitVec 32 := if live then w else 0

def pick {α : Type} (d : α) (sel : List Bool) (cands : List α) : α :=
  (sel.foldl (fun cs s => pairs (mux s) cs) cands).headD d

/-- Lane `l`'s row at row slot `s` (`h1tile.Layout.rows` at `R = C = 32`): `⌊l/4⌋ + 8 s`. -/
def laneRow (l : Fin 32) (s : Fin 4) : Fin 32 :=
  ⟨l.val / 4 + 8 * s.val, by have := l.isLt; have := s.isLt; omega⟩

/-- Lane `l`'s column at column slot `t` (`h1tile.Layout.cols` at `R = C = 32`): `2 (l mod 4) + 8 ⌊t/2⌋ + (t mod 2)`. -/
def laneCol (l : Fin 32) (t : Fin 8) : Fin 32 :=
  ⟨2 * (l.val % 4) + 8 * (t.val / 2) + t.val % 2, by have := t.isLt; omega⟩

/-- **A lane's inputs on words** (`PouwH1LaneZy_v1`'s signature at a 32 × 32 leaf): its A rows' codes and scales by row
slot, its B rows' by column slot, the leaf's admit bits, the leaf's record (cell `(i, j)`'s z and y), its partials row,
the public seed inputs (salt, position and roots, of type `P`) and its index's 16 bits, least significant first. -/
structure LaneIn (K nw : ℕ) (P : Type) where
  a : Fin 4 → Fin K → BitVec 8
  sa : Fin 4 → BitVec 32
  b : Fin 8 → Fin K → BitVec 8
  sw : Fin 8 → BitVec 32
  admit : Fin 32 → Bool
  zr : Fin 32 → Fin 32 → BitVec 32
  yr : Fin 32 → Fin 32 → BitVec 16
  partials : Fin nw → BitVec 32
  pub : P
  lane : Fin 16 → Bool

/-- The number a list of bits spells, least significant first. -/
def bitsVal : List Bool → ℕ
  | [] => 0
  | s :: ss => (if s then 1 else 0) + 2 * bitsVal ss

theorem bitsVal_lt : ∀ l : List Bool, bitsVal l < 2 ^ l.length
  | [] => by simp [bitsVal]
  | s :: ss => by
      have := bitsVal_lt ss
      simp only [bitsVal, List.length_cons, pow_succ]
      split <;> omega

def LaneIn.sel {K nw : ℕ} {P : Type} (x : LaneIn K nw P) : List Bool := List.ofFn fun t : Fin 5 => x.lane (Fin.castLE (by omega) t)

def LaneIn.l {K nw : ℕ} {P : Type} (x : LaneIn K nw P) : Fin 32 :=
  ⟨bitsVal x.sel, by simpa [LaneIn.sel] using bitsVal_lt x.sel⟩

def laneOk {K nw : ℕ} {P : Type}
    (chain : P → Fin 32 → (Fin 4 → Fin K → BitVec 8) → (Fin 8 → Fin K → BitVec 8) → Fin nw → BitVec 32)
    (dqW : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 16) (x : LaneIn K nw P) : Bool :=
  let picked : Fin 4 → Fin 8 → BitVec 32 × BitVec 16 := fun s t =>
    pick (0, 0) x.sel (List.ofFn fun ln : Fin 32 => (x.zr (laneRow ln s) (laneCol ln t), x.yr (laneRow ln s) (laneCol ln t)))
  let ok := zyCheck dqW (fun s t => (picked s t).1) (fun s t => (picked s t).2) (fun s t => gemmAccS8 (x.a s) (x.b t))
    x.sa x.sw
  let live := tree orBit false (List.ofFn x.admit)
  let eq := List.ofFn fun w => eq32 (x.partials w) (mask32 (chain x.pub x.l x.a x.b w) live)
  let high := tree orBit false (List.ofFn fun t : Fin 11 => x.lane ⟨5 + t.val, by omega⟩)
  mux high (tree andBit true (ok :: eq)) false

end Pouw.H1Zy
