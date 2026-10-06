import Definitions.Pouw.PearlC.Defs

/-!
# U's pinned order, bit-exact (trusted definitions)

Pearl-C's clean-up as `protocols/pouw/verity_pouw/schemes/pearl_c_u.py` pins it (#449's branch at commit `76134352`,
SHA-256 `aef10195…2485514a`), transcribed op for op. FP32 words are natural numbers below `2^32`, BF16 words below `2^16`.
A function returns `none` where the pin raises `NonFinite`, that is, where a step leaves the finite range.

* **The K16 step** (`bf16Product`, `bf16ProductTotal`, `hopperBf16K16`, `k16Step`): `HOPPER_BF16_WGMMA_K16`, the
  accumulator and 16 BF16 products in one group on a 26-bit truncating adder with exponent floor −133. It is
  `Fp8Atom`'s `groupStepTotal` (align to the largest exponent by truncating shifts, sum exactly, renormalise by
  truncation, denormalise below `2^−126`), with the BF16 product `(s_x·s_y) <<< 9` at exponent
  `max(E_x, 1) + max(E_y, 1) − 254` as its term, as `verity.ml.tc.total.tc_dot_total` has it.
* **The roundings** (`finite?`, `addWord?`, `fmaWord?`, `rnBf16?`): IEEE binary32 add and FMA (`verity.ml.tc.fp32`),
  each one round to nearest even of the exact value by `Defs`' `f32Word`, with IEEE's sign for an exact zero; and
  `RN_bf16` of an FP32 word by the pin's bit formula.
* **The chain** (`windowWord?`, `promoteWord?`, `chainWord?`): the atoms in windows of `Params.pi.G = 4`. Each window
  chains its `hopperStep`s from `+0`, and its word is promoted by one FP32 add into a total from `+0` (read `−0` as
  `+0`). It is `Defs.chain`'s order, on words.
* **The factor rows and U** (`paRow?`, `accChain?`, `pbRow?`, `uWord?`, `peelExact`):
  `P_A,i = [be_A,i ‖ RN_bf16(A′·F_B)]`, `P_B,j = [RN_bf16(RN_f32(acc_q − BF_jq)) ‖ −be_B,j]` with `acc_q` the FMA chain
  from `+0` over `p = 0 … 31` ascending, and `U_ij` from `C̃_ij` by four `k16Step`s over the rows' 16-word slices.
* **What a conformance vector asserts** (`K16Vec`, `ChainVec`, `PaVec`, `PbVec`, `CellVec`), for
  `PeelExactVectors`, generated from the pin's vectors `protocols/pouw/tests/vectors/pearl_c_u.json`.

The model follows the pin at `76134352`; a change to the pin needs the vectors regenerated and rechecked.
-/

namespace Pouw.PearlC.PeelExact

open Pouw.Fp8Atom Pouw.PearlC Matrix

/-! ## The K16 step -/

/-- A BF16 word's exponent field (bits 14..7). -/
def bf16Exp (x : Nat) : Nat := (x >>> 7) % 256

/-- A BF16 word's mantissa field (bits 6..0). -/
def bf16Man (x : Nat) : Nat := x % 128

/-- A BF16 word's sign bit (bit 15). -/
def bf16Sign (x : Nat) : Bool := (x >>> 15) % 2 == 1

/-- The significand, with the hidden bit when the exponent field is nonzero. -/
def bf16Sig (x : Nat) : Nat := if bf16Exp x = 0 then bf16Man x else 128 + bf16Man x

/-- `max(E, 1)`, the exponent field with subnormals read at `1`. -/
def bf16ExpOr1 (x : Nat) : Nat := max (bf16Exp x) 1

/-- `term.bf16_product`: the exact product of two finite BF16 words, `(s_x·s_y) <<< 9` at exponent
`max(E_x, 1) + max(E_y, 1) − 254`; a zero significand gives the unsigned zero. -/
def bf16Product (x y : Nat) : Term :=
  let mag := (bf16Sig x * bf16Sig y) <<< 9
  if mag = 0 then Term.zero else ⟨bf16Sign x ^^ bf16Sign y, (bf16ExpOr1 x : Int) + bf16ExpOr1 y - 254, mag⟩

/-- A BF16 NaN: exponent field 255, mantissa nonzero. -/
def bf16IsNaN (x : Nat) : Bool := bf16Exp x == 255 && bf16Man x != 0

/-- A BF16 infinity: exponent field 255, mantissa zero. -/
def bf16IsInf (x : Nat) : Bool := bf16Exp x == 255 && bf16Man x == 0

/-- A BF16 zero of either sign. -/
def bf16IsZero (x : Nat) : Bool := bf16Exp x == 0 && bf16Man x == 0

/-- `total.bf16_product_total`: NaN if either word is NaN or an infinity meets a zero, the infinity of the product's
sign if either word is infinite, and otherwise the exact product. -/
def bf16ProductTotal (x y : Nat) : Value :=
  if bf16IsNaN x || bf16IsNaN y then .nan
  else if bf16IsInf x || bf16IsInf y then
    if bf16IsZero x || bf16IsZero y then .nan else .inf (bf16Sign x ^^ bf16Sign y)
  else .fin (bf16Product x y)

/-- `HOPPER_BF16_WGMMA_K16`: one group of 16, a 26-bit adder, exponent floor −133. -/
def hopperBf16K16 : Pipeline := ⟨[16], 26, -133⟩

/-- **One `HOPPER_BF16_WGMMA_K16` step** (the pin's `k16`, `tc_dot_total` on `HOPPER_BF16_WGMMA_K16`): the accumulator
word and 16 BF16 word pairs to the FP32 word written. -/
def k16Step (acc : Nat) (x y : Fin 16 → Nat) : Nat :=
  packValue (groupStepTotal hopperBf16K16.width hopperBf16K16.zeroExp (fp32Value acc)
    (List.ofFn fun l => bf16ProductTotal (x l) (y l)))

/-! ## The roundings -/

/-- `_finite`: the word, unless its exponent field is 255 (an infinity or NaN). -/
def finite? (w : Nat) : Option Nat := if (w >>> 23) % 256 = 255 then none else some w

/-- An FP32 word's sign bit. -/
def signBit (w : Nat) : Bool := (w >>> 31) % 2 == 1

/-- `fp32.neg`: the FP32 sign bit flipped. -/
def negWord (w : Nat) : Nat := w ^^^ 0x80000000

/-- One round to nearest even of the exact value `v`, as an FP32 word; an exact zero is `−0` when `negZero`, else `+0`. -/
def rnWord (v : ℚ) (negZero : Bool) : Nat := if v = 0 then (if negZero then 0x80000000 else 0) else f32Word v

/-- `fp32.add` of two finite words, `none` for a non-finite operand or result. An exact zero sum is `−0` only for
`(−0) + (−0)`. -/
def addWord? (a b : Nat) : Option Nat :=
  match wordQ a, wordQ b with
  | some x, some y => finite? (rnWord (x + y) (decide (x = 0) && decide (y = 0) && signBit a && signBit b))
  | _, _ => none

/-- `fp32.fma a b c`, `RN(a·b + c)` with one rounding, of three finite words, `none` for a non-finite operand or
result. An exact zero is `−0` only when the product is a zero of negative sign and `c` is `−0`. -/
def fmaWord? (a b c : Nat) : Option Nat :=
  match wordQ a, wordQ b, wordQ c with
  | some x, some y, some z =>
    finite? (rnWord (x * y + z) (decide (x * y = 0) && decide (z = 0) && (signBit a != signBit b) && signBit c))
  | _, _, _ => none

/-- `rn_bf16`: `RN_bf16` of a finite FP32 word, one nearest-even rounding of its low 16 bits; `none` for a non-finite
word or a BF16 overflow. -/
def rnBf16? (w : Nat) : Option Nat :=
  (finite? w).bind fun w =>
    let out := ((w + 0x7FFF + (w >>> 16) % 2) >>> 16) % 65536
    if 0x7F80 ≤ out % 32768 then none else some out

/-! ## The chain -/

/-- One window's word: its atoms' `hopperStep`s from `+0`, each word finite. -/
def windowWord? (g : List AtomIn) : Option Nat := g.foldlM (fun w x => finite? (hopperStep w x.1 x.2)) 0

/-- One promotion: the window's word added to the total by one FP32 add, `−0` read as `+0`. -/
def promoteWord? (t w : Nat) : Option Nat := (addWord? t w).map fun s => if s = 0x80000000 then 0 else s

/-- **The chain** (the pin's `chain`, `Defs.chain`'s order on words): the atoms in windows of `Params.pi.G`, each
window's word promoted in order into a total from `+0`. -/
def chainWord? (atoms : List AtomIn) : Option Nat :=
  (atoms.toChunks Params.pi.G).foldlM (fun t g => (windowWord? g).bind (promoteWord? t)) 0

/-! ## The factor rows and U -/

/-- `pa_row`: `P_A,i = [be_A,i ‖ RN_bf16(A′·F_B)]`, from the 32 BF16 words `be` and the 32 FP32 words `af`. -/
def paRow? (be af : List Nat) : Option (List Nat) := (af.mapM rnBf16?).map (be ++ ·)

/-- `acc_q`: the FMA chain from `+0` over `p` ascending, `acc ← RN(be[p]·G[p][q] + acc)`, with the BF16 word `be[p]`
read as the FP32 word `be[p] <<< 16`; `gq` is column `q` of `G`. -/
def accChain? (be gq : List Nat) : Option Nat := (be.zip gq).foldlM (fun acc x => fmaWord? (x.1 <<< 16) x.2 acc) 0

/-- Column `q` of a matrix given by its rows. -/
def col (g : List (List Nat)) (q : Nat) : List Nat := g.map fun r => r.getD q 0

/-- `pb_row`: `P_B,j = [RN_bf16(RN_f32(acc_q − BF_jq)) for q < 32 ‖ −be_B,j]`, from the 32 BF16 words `be`, the 32 FP32
words `bf` and `G` by rows. -/
def pbRow? (be bf : List Nat) (g : List (List Nat)) : Option (List Nat) := do
  let mid ← (List.range Params.pi.r).mapM fun q => do
    let acc ← accChain? be (col g q)
    rnBf16? (← addWord? acc (negWord (bf.getD q 0)))
  pure (mid ++ be.map (· ^^^ 0x8000))

/-- Slice `s` of a factor row: positions `16s, …, 16s + 15`. -/
def slice16 (row : List Nat) (s : Nat) : Fin 16 → Nat := fun l => row.getD (16 * s + l) 0

/-- One step of U: `k16Step` over slice `s` of both rows, the word finite. -/
def uStep? (pa pb : List Nat) (acc s : Nat) : Option Nat := finite? (k16Step acc (slice16 pa s) (slice16 pb s))

/-- `u`: `U_ij` from `C̃_ij`, four `k16Step`s over the slices of `P_A,i` and `P_B,j` in order. -/
def uWord? (c : Nat) (pa pb : List Nat) : Option Nat := (List.range 4).foldlM (uStep? pa pb) c

/-- **U under the pinned order**, entry `(i, j)` as an FP32 word, or `none` where a step leaves the finite range.
The inputs are the formed operands `Aq` (`A′`, `m × k`) and `Bq` (`B̃`, `n × k`), the bases `FA` and `FB` (`k × 32`
E4M3 codes), and the peel's scaled noise rows `beA` and `beB` (32 BF16 words per row). `C̃`, `A′·F_B`, `B̃·F_A` and
`G = F_Bᵀ·F_A` are `chainWord?`s. -/
def peelExact {m n k : Nat} (Aq : Matrix (Fin m) (Fin k) Code) (Bq : Matrix (Fin n) (Fin k) Code)
    (FA FB : Matrix (Fin k) (Fin 32) Code) (beA : Matrix (Fin m) (Fin 32) Nat) (beB : Matrix (Fin n) (Fin 32) Nat) :
    Matrix (Fin m) (Fin n) (Option Nat) := fun i j => do
  let c ← chainWord? (rowAtoms (Aq i) (Bq j))
  let af ← (List.ofFn fun p => chainWord? (rowAtoms (Aq i) (FBᵀ p))).mapM id
  let bf ← (List.ofFn fun q => chainWord? (rowAtoms (Bq j) (FAᵀ q))).mapM id
  let g ← (List.ofFn fun p => (List.ofFn fun q => chainWord? (rowAtoms (FBᵀ p) (FAᵀ q))).mapM id).mapM id
  let pa ← paRow? (List.ofFn (beA i)) af
  let pb ← pbRow? (List.ofFn (beB j)) bf g
  uWord? c pa pb

/-- `peelExact`'s entries as values: `none` where a step leaves the finite range. -/
def peelExactVal {m n k : Nat} (Aq : Matrix (Fin m) (Fin k) Code) (Bq : Matrix (Fin n) (Fin k) Code)
    (FA FB : Matrix (Fin k) (Fin 32) Code) (beA : Matrix (Fin m) (Fin 32) Nat) (beB : Matrix (Fin n) (Fin 32) Nat) :
    Matrix (Fin m) (Fin n) (Option ℚ) := fun i j => (peelExact Aq Bq FA FB beA beB i j).bind wordQ

/-! ## What a conformance vector asserts -/

/-- 16 BF16 words packed little-endian in `X` (word `l` is bits `16l … 16l + 15`). -/
def words16 (X : Nat) : Fin 16 → Nat := fun l => (X >>> (16 * l.val)) % 65536

/-- A K16 step vector: the accumulator word, 16 packed BF16 words `x` and `y`, and the word the pin writes. -/
structure K16Vec where
  acc : Nat
  x : Nat
  y : Nat
  w : Nat

/-- A K16 step vector holds for `k16Step`. -/
def K16Vec.holds (v : K16Vec) : Bool := k16Step v.acc (words16 v.x) (words16 v.y) == v.w

/-- A traced chain: each atom's 32 code pairs packed as in `Fp8Atom.codes`, the word after each atom, the total after
each promotion, and the chain's word. -/
structure ChainVec where
  a : List Nat
  b : List Nat
  atoms : List Nat
  totals : List Nat
  c : Nat

/-- The chain's atoms. -/
def ChainVec.atomsIn (v : ChainVec) : List AtomIn := (v.a.zip v.b).map fun p => (codes p.1, codes p.2)

/-- A traced chain holds: every atom's word and every promotion's total are the recorded ones, `chainWord?` gives
the chain's word, and `Defs.chain` its value. -/
def ChainVec.holds (v : ChainVec) : Bool :=
  let gs := v.atomsIn.toChunks Params.pi.G
  gs.flatMap (fun g => (g.scanl (fun w x => hopperStep w x.1 x.2) 0).tail) == v.atoms &&
    (gs.scanl (fun t g => t.bind fun t => (windowWord? g).bind (promoteWord? t)) (some 0)).tail ==
      v.totals.map some &&
    chainWord? v.atomsIn == some v.c &&
    decide (chain Params.pi.G v.atomsIn = wordQ v.c)

/-- A `P_A` row: `be_A,i`, the words of `A′·F_B`, and `P_A,i`. -/
structure PaVec where
  be : List Nat
  af : List Nat
  pa : List Nat

/-- A `P_A` row holds for `paRow?`. -/
def PaVec.holds (v : PaVec) : Bool := paRow? v.be v.af == some v.pa

/-- A `P_B` row: `be_B,j`, the words of `B̃·F_A`, the FMA chains' words `acc_q`, the differences, and `P_B,j`. -/
structure PbVec where
  be : List Nat
  bf : List Nat
  acc : List Nat
  diff : List Nat
  pb : List Nat

/-- A `P_B` row holds for `pbRow?` on `G` (by rows), and each FMA chain's word and each difference is the recorded one. -/
def PbVec.holds (g : List (List Nat)) (v : PbVec) : Bool :=
  (List.range Params.pi.r).all (fun q => accChain? v.be (col g q) == some (v.acc.getD q 0)) &&
    (List.range Params.pi.r).all (fun q => addWord? (v.acc.getD q 0) (negWord (v.bf.getD q 0)) == some (v.diff.getD q 0)) &&
    pbRow? v.be v.bf g == some v.pb

/-- A cell: the indices of its `P_A` and `P_B` rows in the vectors' row lists, `C̃_ij`, the word after each K16 step,
and `U_ij`. -/
structure CellVec where
  ia : Nat
  jb : Nat
  c : Nat
  steps : List Nat
  u : Nat

/-- A cell holds on the rows `ra` and `rb`: from `C̃_ij`, the four steps over its rows' `P_A,i` and `P_B,j` give the
recorded words, and `uWord?` gives `U_ij`. -/
def CellVec.holds (ra : List PaVec) (rb : List PbVec) (v : CellVec) : Bool :=
  match ra[v.ia]?, rb[v.jb]? with
  | some a, some b =>
    ((List.range 4).scanl (fun t s => t.bind fun t => uStep? a.pa b.pb t s) (some v.c)).tail == v.steps.map some &&
      uWord? v.c a.pa b.pb == some v.u
  | _, _ => false

/-- `rnBf16?` agrees with `Defs`' `bf16Rn` on a finite word whose BF16 result is finite: the BF16 word, read as the
FP32 word `h <<< 16`, has the value `bf16Rn` gives. -/
def bf16Agrees (w : Nat) : Bool :=
  match rnBf16? w, wordQ w with
  | some h, some x => wordQ (h <<< 16) == some (bf16Rn x)
  | _, _ => false

end Pouw.PearlC.PeelExact
