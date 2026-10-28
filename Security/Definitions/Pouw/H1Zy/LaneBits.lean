import Definitions.Pouw.H1Zy.Lane
import Definitions.Pouw.H1Zy.CombineBits

/-!
# `PouwH1LaneZy_v1{PouwH1Tile_v1}` on bits (trusted)

The lane's body (`h1zy._lane_body`, stated on words in `Definitions.Pouw.H1Zy.Lane`) with its pieces at their
lowerings, each word a `Fin n → Bool`, bit `i` the word's bit `i` LSB first (`bv`):

* **The multiply.** `GemmAccumulatorS8_v2{K, BlackwellS8ImmaDot32_v2}` (`gemm._steps`, `gemmAccS8Bits`): `K/32` steps
  from the zero accumulator, each `integer.step_s8_bits` at sm_120's s8 model (`immaS8Bits`): 32 Baugh–Wooley arrays
  (`bwBit`: `x i AND y j`, negated where exactly one of `i`, `j` is 7), the accumulator sign-extended to 33 bits and
  `32 · _BW_CONSTANT mod 2^33` as constant bits, reduced to 33 bits by `compress_columns` (stated by its contract,
  `compressCols`), then `.satfinite`'s clamp on those bits.
* **The record check.** `PouwZyCell_v2` (`zyCellBits`: `and_reduce` of the 48 XNORs, 47 ANDs) per cell and a
  `PouwAndBit` tree (`zyCheckBits`), at the dequant on bits.
* **The picks.** `PouwMuxBit_v2` (`muxBit`: `a xor (s and (a xor b))`), batched over a record cell's 48 bits
  (`muxCand`), in `h1zy._pick`'s tree (`pickBy`).
* **The partials.** `PouwMask32_v2` (`mask32Bits`) and `PouwEq32_v2` (`eq32Bits`, `Definitions.Pouw.H1Zy.CombineBits`).
* `PouwOrBit` and `PouwAndBit` are one gate each, `orBit` and `andBit` themselves.

Two pieces stay at their word views, lifted to bits: the dequant (`liftDq`: `PouwZyDequant_v2`, its four float circuits
not restated) and the chain's words (`liftChain`). `Proofs.Pouw.H1Zy.LaneBits` proves each restated piece's lowering
computes its word view, and `laneBits` at those two lifts computes `laneOk`.
-/

namespace Pouw.H1Zy

/-- A word's bits, LSB first. -/
def bv {n : ℕ} (x : BitVec n) : Fin n → Bool := fun i => x.getLsbD i.val

/-- A bit's value. -/
def bval (b : Bool) : ℕ := if b then 1 else 0

/-- Baugh–Wooley's partial product `(i, j)` of `x` and `y`: `x i AND y j`, negated where exactly one of `i`, `j` is the
sign bit 7. -/
def bwBit (x y : Fin 8 → Bool) (i j : Fin 8) : Bool :=
  if (i.val = 7) ≠ (j.val = 7) then !(x i && y j) else (x i && y j)

/-- `_BW_CONSTANT` (`integer.py`): `-2^15 + 2^8`. -/
def bwConst : ℤ := -2 ^ 15 + 2 ^ 8

/-- The weighted sum of `step_s8_bits`' columns: 32 Baugh–Wooley arrays, the accumulator sign-extended to 33 bits, and
`32 · bwConst mod 2^33` as constant bits. -/
def stepColSum (acc : Fin 32 → Bool) (a b : Fin 32 → Fin 8 → Bool) : ℕ :=
  (∑ k, ∑ i : Fin 8, ∑ j : Fin 8, 2 ^ (i.val + j.val) * bval (bwBit (a k) (b k) i j)) +
    (∑ i : Fin 32, 2 ^ i.val * bval (acc i)) + 2 ^ 32 * bval (acc 31) + (32 * bwConst % 2 ^ 33).toNat

/-- `forms.compress_columns`' contract, not its gates: the low `W` bits of the columns' weighted sum. The lowering's
constant folding, Wallace passes and final ripple are not restated here; that they compute this is part of what
`Accounting.TS.Lanes.LaneLowers` takes. -/
def compressCols (W S : ℕ) : Fin W → Bool := fun i => S.testBit i.val

/-- **`BlackwellS8ImmaDot32_v2`'s lowering** (`integer.step_s8_bits` at `BLACKWELL_SM120_S8_M16N8K32`, `.satfinite`):
the columns compressed to 33 bits `s`, then clamped: `ovf = s 31 xor s 32`; bit `i < 31` is
`s i xor (ovf and (not s 32 xor s i))`, bit 31 is `s 32`. -/
def immaS8Bits (acc : Fin 32 → Bool) (a b : Fin 32 → Fin 8 → Bool) : Fin 32 → Bool :=
  let s := compressCols 33 (stepColSum acc a b)
  let ovf := xor (s 31) (s 32)
  fun i => if h : i.val < 31 then xor (s ⟨i.val, by omega⟩) (ovf && xor (!s 32) (s ⟨i.val, by omega⟩)) else s 32

/-- A word from its bits, LSB first. -/
def ofBits {n : ℕ} (f : Fin n → Bool) : BitVec n := (BitVec.ofBoolListLE (List.ofFn f)).cast (List.length_ofFn)

/-- `GemmAccumulatorS8_v2`'s lowering after its first `n` steps: step `s` is `immaS8Bits` on codes `32 s .. 32 s + 31`,
from the zero accumulator (`gemm._steps`). -/
def gemmStepsBits {K : ℕ} (x w : Fin K → Fin 8 → Bool) : ℕ → Fin 32 → Bool
  | 0 => fun _ => false
  | n + 1 => immaS8Bits (gemmStepsBits x w n)
      (fun j => if h : 32 * n + j.val < K then x ⟨32 * n + j.val, h⟩ else fun _ => false)
      (fun j => if h : 32 * n + j.val < K then w ⟨32 * n + j.val, h⟩ else fun _ => false)

/-- **`GemmAccumulatorS8_v2{K, BlackwellS8ImmaDot32_v2}`'s lowering**: `K/32` steps. -/
def gemmAccS8Bits {K : ℕ} (x w : Fin K → Fin 8 → Bool) : Fin 32 → Bool := gemmStepsBits x w (K / 32)

/-- `PouwMuxBit_v2`'s lowering: `a xor (s and (a xor b))`. -/
def muxBit (s a b : Bool) : Bool := xor a (s && xor a b)

/-- `PouwMask32_v2`'s lowering: each bit AND `live`. -/
def mask32Bits (w : Fin 32 → Bool) (live : Bool) : Fin 32 → Bool := fun i => w i && live

/-- `PouwZyCell_v2`'s lowering: `and_reduce` of the XNORs of `zr ++ yr` against `z ++ y`. -/
def zyCellBits (zr : Fin 32 → Bool) (yr : Fin 16 → Bool) (z : Fin 32 → Bool) (y : Fin 16 → Bool) : Bool :=
  tree andBit true ((List.ofFn fun i => !(xor (zr i) (z i))) ++ List.ofFn fun i => !(xor (yr i) (y i)))

/-- `PouwZyCheck_v1{R, C}` on bits, at a dequant on bits `dqB`. -/
def zyCheckBits {R C : ℕ} (dqB : (Fin 32 → Bool) → (Fin 32 → Bool) → (Fin 32 → Bool) → Fin 16 → Bool)
    (zr : Fin R → Fin C → Fin 32 → Bool) (yr : Fin R → Fin C → Fin 16 → Bool) (z : Fin R → Fin C → Fin 32 → Bool)
    (sa : Fin R → Fin 32 → Bool) (sw : Fin C → Fin 32 → Bool) : Bool :=
  tree andBit true
    (List.ofFn fun i => List.ofFn fun j => zyCellBits (zr i j) (yr i j) (z i j) (dqB (z i j) (sa i) (sw j))).flatten

/-- `h1zy._pick` at a candidate's mux `f`. -/
def pickBy {α : Type} (f : Bool → α → α → α) (d : α) (sel : List Bool) (cands : List α) : α :=
  (sel.foldl (fun cs s => pairs (f s) cs) cands).headD d

/-- A record candidate's mux on bits: `muxBit` at each of its 48 bits (z's 32, then y's 16). -/
def muxCand (s : Bool) (p q : (Fin 32 → Bool) × (Fin 16 → Bool)) : (Fin 32 → Bool) × (Fin 16 → Bool) :=
  (fun i => muxBit s (p.1 i) (q.1 i), fun i => muxBit s (p.2 i) (q.2 i))

/-- **A lane's inputs on bits**, each word's bits LSB first. -/
structure LaneInBits (K nw : ℕ) (P : Type) where
  a : Fin 4 → Fin K → Fin 8 → Bool
  sa : Fin 4 → Fin 32 → Bool
  b : Fin 8 → Fin K → Fin 8 → Bool
  sw : Fin 8 → Fin 32 → Bool
  admit : Fin 32 → Bool
  zr : Fin 32 → Fin 32 → Fin 32 → Bool
  yr : Fin 32 → Fin 32 → Fin 16 → Bool
  partials : Fin nw → Fin 32 → Bool
  pub : P
  lane : Fin 16 → Bool

/-- A lane's word inputs as bits. -/
def LaneIn.bits {K nw : ℕ} {P : Type} (x : LaneIn K nw P) : LaneInBits K nw P where
  a s k := bv (x.a s k)
  sa s := bv (x.sa s)
  b t k := bv (x.b t k)
  sw t := bv (x.sw t)
  admit := x.admit
  zr i j := bv (x.zr i j)
  yr i j := bv (x.yr i j)
  partials w := bv (x.partials w)
  pub := x.pub
  lane := x.lane

/-- **`PouwH1LaneZy_v1{PouwH1Tile_v1}`'s `ok` on bits** at a 32 × 32 leaf (`h1zy._lane_body`): `laneOk` with each
piece at its lowering, the dequant `dqB` and the chain's words `chainB` on bits. -/
def laneBits {K nw : ℕ} {P : Type}
    (chainB : P → Fin 32 → (Fin 4 → Fin K → Fin 8 → Bool) → (Fin 8 → Fin K → Fin 8 → Bool) → Fin nw → Fin 32 → Bool)
    (dqB : (Fin 32 → Bool) → (Fin 32 → Bool) → (Fin 32 → Bool) → Fin 16 → Bool) (x : LaneInBits K nw P) : Bool :=
  let sel := List.ofFn fun t : Fin 5 => x.lane (Fin.castLE (by omega) t)
  let l : Fin 32 := ⟨bitsVal sel, by simpa [sel] using bitsVal_lt sel⟩
  let picked : Fin 4 → Fin 8 → (Fin 32 → Bool) × (Fin 16 → Bool) := fun s t =>
    pickBy muxCand (fun _ => false, fun _ => false) sel
      (List.ofFn fun ln : Fin 32 => (x.zr (laneRow ln s) (laneCol ln t), x.yr (laneRow ln s) (laneCol ln t)))
  let ok := zyCheckBits dqB (fun s t => (picked s t).1) (fun s t => (picked s t).2)
    (fun s t => gemmAccS8Bits (x.a s) (x.b t)) x.sa x.sw
  let live := tree orBit false (List.ofFn x.admit)
  let eq := List.ofFn fun w => eq32Bits (x.partials w) (mask32Bits (chainB x.pub l x.a x.b w) live)
  let high := tree orBit false (List.ofFn fun t : Fin 11 => x.lane ⟨5 + t.val, by omega⟩)
  muxBit high (tree andBit true (ok :: eq)) false

/-- The dequant on bits at its word view `dqW` (`PouwZyDequant_v1`): its lowering (`S32ToF32Rn_v2`, two `F32Mul_v4`,
`F2fpBf16_v2`, 5,230 ANDs) is not restated here. -/
def liftDq (dqW : BitVec 32 → BitVec 32 → BitVec 32 → BitVec 16) :
    (Fin 32 → Bool) → (Fin 32 → Bool) → (Fin 32 → Bool) → Fin 16 → Bool :=
  fun z sa sw => bv (dqW (ofBits z) (ofBits sa) (ofBits sw))

/-- The chain's words on bits at their word view `chain`: their lowering (the seeds, the noise blocks, fx1's chain) is
not restated here. -/
def liftChain {K nw : ℕ} {P : Type}
    (chain : P → Fin 32 → (Fin 4 → Fin K → BitVec 8) → (Fin 8 → Fin K → BitVec 8) → Fin nw → BitVec 32) :
    P → Fin 32 → (Fin 4 → Fin K → Fin 8 → Bool) → (Fin 8 → Fin K → Fin 8 → Bool) → Fin nw → Fin 32 → Bool :=
  fun p l a b w => bv (chain p l (fun s k => ofBits (a s k)) (fun t k => ofBits (b t k)) w)

end Pouw.H1Zy
