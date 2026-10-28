import Definitions.Pouw.H1Zy.Combine
import Mathlib.Algebra.BigOperators.Fin

/-!
# `PouwH1LaneZy_v1{TILE}` on words, at a 32 × 32 leaf over `PouwH1Tile_v1` (trusted)

Lane `l` of a drawn H1 leaf (`verity/core/protocols/pouw/circuit/h1zy.py`, `PouwH1LaneZy`, `_lane_body`) reads its 4 A
rows by row slot and 8 B rows by column slot (`K` s8 codes and an f32 scale each), the leaf's 32 admit bits, the leaf's
whole z/y record (cell `(i, j)` its s32 z and bf16 y), its committed partials row (`nw` u32 words), the leaf's salt,
position and roots, and its index, a 16-bit word, and returns `ok`. On its primitives' word views:

* **The index.** `sel`, the index's low 5 bits (`lane_bits(32, 32)`), least significant first, picks each of the
  lane's candidates by a `PouwMuxBit` tree (`pick`, `h1zy._pick`); the OR of its 11 high bits (`PouwOrBit` tree) sends
  `ok` to `0` (`PouwMuxBit(high, checked, 0)`).
* **The cells.** Cell `(s, t)` of the lane (row slot `s`, column slot `t`, row-major) is the leaf's cell
  `(laneRow l s, laneCol l t)` (`h1tile.TileShape.rows`, `cols` at `R = C = 32`; `h1zy.lane_cells`), picked from the 32
  lanes' by `sel`. Its z is the tile's EXACT on the A row's and B row's codes, `GemmAccumulatorS8_v2{K, sm120}`'s word
  view (`gemmAccS8`: `K/32` steps of `BlackwellS8ImmaDot32_v1`, `immaS8`, from a zero accumulator).
  `PouwZyCheck_v1{4, 8}` (`zyCheck`) is `1` exactly when every cell's record z is that z and its record y is the
  dequant of it at its row's and column's scales (`PouwZyCell`), ANDed by a `PouwAndBit` tree in row-major order.
* **The partials.** The lane's chain words (`h1tile._chain`: its fx1 chain's `PouwXinFinal` words per set, from its
  rows' and columns' noise, picked from the leaf's seed_j stream by `sel`, and its fold keys, block `l` of σ_R's) are the
  parameter `chain`; each is masked by the admit bits' OR (`PouwOrBit` tree, `PouwMask32`) and compared to the
  committed row's word (`PouwEq32`).
* `ok` is the `PouwAndBit` tree of the cell check and the `nw` comparisons, unless a high index bit is set.

The dequant (`PouwZyDequant_v2`'s word view, `PouwZyDequant_v1`, three float roundings; `Pouw.H1Zy.zyDequant`, which
`Accounting.Guarantees.TS.LeafLanesSound` reads) is the parameter `dqW`, and the
chain's words the parameter `chain` (`chainOf (h1Chain K)` in `LeafLanesSound`); everything else is stated here. The leaf's `PouwH1Tile_v3` form computes z from the
chain's words instead of EXACT, and is not stated here.
-/

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

/-- `PouwMuxBit_v1` on a candidate: `b` where `s` is `1`, else `a` (bitwise on a word, so on a whole candidate). -/
def mux {α : Type} (s : Bool) (a b : α) : α := if s then b else a

/-- `PouwMask32_v1`: each bit of `w` AND `live`. -/
def mask32 (w : BitVec 32) (live : Bool) : BitVec 32 := if live then w else 0

/-- **`h1zy._pick`**: `cands[Σ_b sel[b] 2^b]` as a `PouwMuxBit` tree over the selector's bits, least significant first,
each pass muxing adjacent candidates; `d` for an empty list. -/
def pick {α : Type} (d : α) (sel : List Bool) (cands : List α) : α :=
  (sel.foldl (fun cs s => pairs (mux s) cs) cands).headD d

/-- Lane `l`'s row at row slot `s` (`h1tile.TileShape.rows` at `R = C = 32`): `⌊l/4⌋ + 8 s`. -/
def laneRow (l : Fin 32) (s : Fin 4) : Fin 32 :=
  ⟨l.val / 4 + 8 * s.val, by have := l.isLt; have := s.isLt; omega⟩

/-- Lane `l`'s column at column slot `t` (`h1tile.TileShape.cols` at `R = C = 32`): `2 (l mod 4) + 8 ⌊t/2⌋ + (t mod 2)`. -/
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

/-- The index's selector bits, its low 5 (`lane_bits(32, 32)`). -/
def LaneIn.sel {K nw : ℕ} {P : Type} (x : LaneIn K nw P) : List Bool := List.ofFn fun t : Fin 5 => x.lane (Fin.castLE (by omega) t)

/-- The lane the selector names, `Σ_b sel[b] 2^b`. -/
def LaneIn.l {K nw : ℕ} {P : Type} (x : LaneIn K nw P) : Fin 32 :=
  ⟨bitsVal x.sel, by simpa [LaneIn.sel] using bitsVal_lt x.sel⟩

/-- **`PouwH1LaneZy_v1{PouwH1Tile_v1}`'s `ok`** at a 32 × 32 leaf (`h1zy._lane_body`), at the dequant `dqW` and the
chain's words `chain` (of the public inputs, the lane `sel` names and the rows' codes). -/
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
