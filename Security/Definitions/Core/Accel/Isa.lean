import Definitions.Core.Accel.Machine
import Definitions.Core.TC.Spec
import Definitions.Core.Fp32

/-!
# The instruction set (trusted)

The opcodes an accelerator's instances draw on, each with its exact semantics on 32-bit words. An instance
(`Accel.H200`, `Accel.Sm120`) says which of them the card has (`Opcode` restricted by its shapes), where each reads and
writes, and the units of each resource it uses; this module says what each computes.

* **MMA** (`Opcode.mma`): `D = C + A·B` on the tensor core, `m × n` accumulators from `m` rows of `A` and `n` columns of
  `B`, `k` products each. Output `(i, j)` is the format's dot (`Dot`) of accumulator `C i j` with row `i` and column
  `j` (`mmaEval`): the integer formats exactly (`intDot`, modulo `2^32` or clamped), the float formats as `TC.Spec`'s
  group-sum pipelines compute them (`floatDot`; BF16 is `TC.Spec.tcDotTotal` itself). A row's operands are its
  format's `bits` wide, packed into words, so an MMA can't multiply wider operands (P1), and each of its `m·n·k`
  products is formed inside one output's dot (P2).
* **Lane functions** (`Opcode.lane`): one lane's CUDA-core instruction on registers: integer add, multiply-add (low, wide,
  high), logic, shifts, byte permute, bit counts, compares, selects, `dp4a`, and FP32 `fma`, `add` and `mul`
  (`Verity.Fp32`).
* **Copies** (`Opcode.copy`): `n` words from one store to another, unchanged: loads, stores, register moves, TMA. A
  broadcast load (`Opcode.bcast`), the lanes of a warp instruction reading one address, reads its words once.
* **Atomics** (`Opcode.atom`): a read-modify-write of one word of a store: integer add, exchange.

An opcode reads and writes only its operands, so a product leaves its accumulator only through further instructions,
each of which pays its units (`Machine`'s G2).

**Over-approximated units.** A card's unit whose rounding no capture pins, and which is far slower than the INT8
tensor core, is in the set as an **exact integer version**, at the unit's rate and in the unit's register layout: on
the integers the unit computes exactly (its format's exact range), the version computes the same values, read and
written as two's-complement integers where the card holds float codes. `Card.decode` is that reading of the card's
registers, and `Assumptions.CardWithinAccel` is a refinement up to it. Out of range the version reads
an operand as zero and keeps its result's low bits where the unit rounds; such uses are outside the claim on the
contents where they occur (`CardWithinAccel`'s per-input scope limit). These are:
DMMA (`Opcode.dmma`, exact on operands within FP64's ±2^53, each element two words); the FP64 lanes (`Lane.dfma`,
likewise; `DADD` and `DMUL` are it with `b = 1` or `c = 0`); `HFMA2` (`Lane.hfma2`, each half exact on integers
within f16's ±2048, both halves in one word; `HADD2`, `HMUL2` and the `bf16` forms, exact within ±256, likewise); the
float reductions (`red`, `cp.reduce.async.bulk` on `f32`: exact integer adds, `iadd3`; on `f16x2` and `bf16x2`:
`Lane.add2`); and `redux.sync` (`Lane.redux`, exact; integer on the card).
-/

namespace Verity.Accel

open Verity.TC

/-! ## Tensor-core formats -/

/-- A tensor-core operand format: the integers `s8`, `u8` and `b1`; the floating formats, whose MMAs accumulate in
FP32; and the block-scaled FP4 formats `nvf4` (a UE4M3 scale per 16 E2M1 operands) and `mxf4` (a UE8M0 scale per 32). -/
inductive Fmt
  | s8 | u8 | b1 | e4m3 | e5m2 | e2m1 | f16 | bf16 | tf32 | nvf4 | mxf4
  /-- FP64, DMMA's format: no `Opcode.mma` takes it; its MMA is `Opcode.dmma`, and this names its tensor units a MAC -/
  | f64
  deriving DecidableEq

/-- An operand's bits. -/
def Fmt.bits : Fmt → ℕ
  | .b1 => 1
  | .e2m1 | .nvf4 | .mxf4 => 4
  | .s8 | .u8 | .e4m3 | .e5m2 => 8
  | .f16 | .bf16 => 16
  | .tf32 => 32
  | .f64 => 64

/-- The elements a row of `k` products carries: its `k` operands, then for a block-scaled format each block's 8-bit
scale as two 4-bit elements. -/
def Fmt.rowLen (f : Fmt) (k : ℕ) : ℕ :=
  match f with
  | .nvf4 => k + 2 * (k / 16)
  | .mxf4 => k + 2 * (k / 32)
  | _ => k

/-- An integer operand's value: `s8` two's complement, `u8` and `b1` unsigned. -/
def Fmt.intVal (f : Fmt) (x : ℕ) : ℤ :=
  match f with
  | .s8 => if x < 128 then x else (x : ℤ) - 256
  | _ => x

/-- Which words of a minifloat format are NaN or infinite (`term.Format.nan_rule`): IEEE (an all-ones exponent is
`±inf` or NaN), E4M3 (only all-ones exponent and fraction is NaN, no infinities), or none (E2M1). -/
inductive NanRule
  | ieee | e4m3 | none

/-- A minifloat encoding (`term.Format`): `1 + expBits + manBits` bits. -/
structure MiniFloat where
  expBits : ℕ
  manBits : ℕ
  nan : NanRule

namespace MiniFloat

variable (f : MiniFloat)

def bits : ℕ := 1 + f.expBits + f.manBits

def bias : ℤ := 2 ^ (f.expBits - 1) - 1

/-- `term.Format.decode`'s class of a word. -/
def cls (w : ℕ) : Spec.Cls :=
  let e := (w >>> f.manBits) &&& (2 ^ f.expBits - 1)
  let m := w &&& (2 ^ f.manBits - 1)
  let top := 2 ^ f.expBits - 1
  match f.nan with
  | .ieee => if e = top then (if m ≠ 0 then .nan else .inf) else if e = 0 ∧ m = 0 then .zero else .finite
  | .e4m3 => if e = top ∧ m = 2 ^ f.manBits - 1 then .nan else if e = 0 ∧ m = 0 then .zero else .finite
  | .none => if e = 0 ∧ m = 0 then .zero else .finite

/-- `term.Format.product`: the exact product of two finite words, its `2 (manBits + 1)`-bit significand product at
the top of 25 bits; subnormals keep their unnormalised significand. At BF16 it is `TC.Spec.bf16Product`. -/
def product (a b : ℕ) : Spec.Term :=
  let ea := (a >>> f.manBits) &&& (2 ^ f.expBits - 1)
  let eb := (b >>> f.manBits) &&& (2 ^ f.expBits - 1)
  let sa := (a &&& (2 ^ f.manBits - 1)) ||| (if ea ≠ 0 then 2 ^ f.manBits else 0)
  let sb := (b &&& (2 ^ f.manBits - 1)) ||| (if eb ≠ 0 then 2 ^ f.manBits else 0)
  let magnitude := (sa * sb) <<< (25 - 2 * (f.manBits + 1))
  if magnitude = 0 then Spec.ZERO
  else
    ⟨((a ^^^ b) >>> (f.bits - 1)) &&& 1 = 1,
     ((if ea ≠ 0 then ea else 1 : ℕ) : ℤ) + ((if eb ≠ 0 then eb else 1 : ℕ) : ℤ) - 2 * f.bias,
     magnitude⟩

/-- The product of two words on every encoding (`total.bf16_product_total`, `total_fp8.e4m3_product_total`,
`e5m2_product_total`): NaN if either is, `0 · inf` NaN, `inf` signed, else the exact product. -/
def productTotal (a b : ℕ) : Spec.Value :=
  let ca := f.cls a
  let cb := f.cls b
  if ca = .nan ∨ cb = .nan then .nan
  else if ca = .inf ∨ cb = .inf then
    if ca = .zero ∨ cb = .zero then .nan else .inf (((a ^^^ b) >>> (f.bits - 1)) &&& 1 = 1)
  else .fin (f.product a b)

end MiniFloat

/-- A float format's encoding: OCP E4M3, E5M2 and E2M1, IEEE binary16, BF16, and TF32 (an FP32 word's top 19 bits). -/
def Fmt.mini : Fmt → MiniFloat
  | .e4m3 => ⟨4, 3, .e4m3⟩
  | .e5m2 => ⟨5, 2, .ieee⟩
  | .e2m1 | .nvf4 | .mxf4 => ⟨2, 1, .none⟩
  | .f16 => ⟨5, 10, .ieee⟩
  | .tf32 => ⟨8, 10, .ieee⟩
  | _ => ⟨8, 7, .ieee⟩

/-- A float operand's code in its encoding: a TF32 operand is an FP32 word, read through its top 19 bits. -/
def Fmt.code (f : Fmt) (x : ℕ) : ℕ := if f = .tf32 then x >>> 13 else x

/-! ## Dots -/

/-- One accumulator's step: the accumulator word and a row's and a column's elements to the new accumulator word. -/
abbrev Dot : Type := Word → List ℕ → List ℕ → Word

/-- The integer MMA's dot (`silicon.integer.IntegerMma.step`): `C + Σ a·b` exactly, then clamped to the `s32` range
(`.satfinite`) or wrapped modulo `2^32`. At `b1` the products are `AND`s and the sum a popcount (`.and.popc`). -/
def intDot (fa fb : Fmt) (sat : Bool) : Dot := fun c a b =>
  let s : ℤ := c.toInt + (List.zipWith (fun x y => fa.intVal x * fb.intVal y) a b).sum
  BitVec.ofInt 32 (if sat then max (-2 ^ 31) (min (2 ^ 31 - 1) s) else s)

/-- `TC.Spec.runGroups` with the operand product a parameter: each group of the pipeline takes the next operand pairs. -/
def runGroupsWith (prod : ℕ → ℕ → Spec.Value) (width : ℕ) (zeroExponent : ℤ) :
    List ℕ → Spec.Value → List ℕ → List ℕ → Spec.Value
  | [], acc, _, _ => acc
  | size :: rest, acc, a, b =>
    runGroupsWith prod width zeroExponent rest
      (Spec.groupStepTotal acc (List.zipWith prod (a.take size) (b.take size)) width zeroExponent)
      (a.drop size) (b.drop size)

/-- A float MMA's dot on pipeline `p` (`total.tc_dot_total`, `total_fp8.tc_dot_total_e4m3`, `tc_dot_total_e5m2`): the
format's exact products, summed group by group through the pipeline's truncating adder, the accumulator joining the
first group. BF16 is `TC.Spec.tcDotTotal` itself. -/
def floatDot (f : Fmt) (p : Pipeline) : Dot := fun c a b =>
  if f = .bf16 then BitVec.ofNat 32 (Spec.tcDotTotal p c.toNat a b)
  else
    BitVec.ofNat 32 (Spec.packValue
      (runGroupsWith (fun x y => f.mini.productTotal (f.code x) (f.code y)) p.width p.zeroExponent p.groups
        (Spec.fp32Value c.toNat) a b))

/-! ## MMA geometry -/

/-- An MMA's shape: `m × n` accumulators, `k` products each. -/
structure Shape where
  m : ℕ
  n : ℕ
  k : ℕ
  deriving DecidableEq

/-- The words `r` rows of `len` elements of `b` bits occupy, packed row after row. -/
def packedWords (r len b : ℕ) : ℕ := (r * len * b + 31) / 32

/-- Element `e` of the `b`-bit elements packed into words `x` (`b` divides 32, so none straddles two words). -/
def elem (b : ℕ) (x : ℕ → Word) (e : ℕ) : ℕ := (x (e * b / 32)).toNat / 2 ^ (e * b % 32) % 2 ^ b

/-- The words of `A`: `m` rows of `fa.rowLen k` elements. -/
def Shape.wa (s : Shape) (fa : Fmt) : ℕ := packedWords s.m (fa.rowLen s.k) fa.bits

/-- The words of `B`: `n` columns of `fb.rowLen k` elements. -/
def Shape.wb (s : Shape) (fb : Fmt) : ℕ := packedWords s.n (fb.rowLen s.k) fb.bits

/-- Row `i` of `A`, from an MMA's inputs `v` (`A`'s words first). -/
def rowA (fa : Fmt) (s : Shape) (v : ℕ → Word) (i : ℕ) : List ℕ :=
  List.ofFn fun l : Fin (fa.rowLen s.k) => elem fa.bits v (i * fa.rowLen s.k + l)

/-- Column `j` of `B`, from an MMA's inputs `v` (`B`'s words after `A`'s). -/
def colB (fa fb : Fmt) (s : Shape) (v : ℕ → Word) (j : ℕ) : List ℕ :=
  List.ofFn fun l : Fin (fb.rowLen s.k) => elem fb.bits (fun w => v (s.wa fa + w)) (j * fb.rowLen s.k + l)

/-- Accumulator `o` of `C`, from an MMA's inputs `v` (`C`'s words after `B`'s). -/
def accC (fa fb : Fmt) (s : Shape) (v : ℕ → Word) (o : ℕ) : Word := v (s.wa fa + s.wb fb + o)

/-- **An MMA's output** `o = i·n + j`: the dot of accumulator `o` with row `i` of `A` and column `j` of `B`. -/
def mmaEval (dot : Dot) (fa fb : Fmt) (s : Shape) (v : ℕ → Word) (o : ℕ) : Word :=
  dot (accC fa fb s v o) (rowA fa s v (o / s.n)) (colB fa fb s v (o % s.n))

/-! A **2:4 sparse MMA** (`mma.sp`, `wgmma.mma_async.sp`) at logical shape `s`: of each group of 4 of a row of `A`, 2
elements are kept, and the metadata names each kept element's position in its group (2 bits). Its inputs are `A`'s
kept elements (`m` rows of `k/2`), the metadata (`m` rows of `k/2` selectors), `B` (`n` columns of `k`) and `C`. -/

/-- The words of a sparse MMA's kept `A`: `m` rows of `k/2` elements. -/
def Shape.waSp (s : Shape) (fa : Fmt) : ℕ := packedWords s.m (s.k / 2) fa.bits

/-- The words of a sparse MMA's metadata: `m` rows of `k/2` selectors of 2 bits. -/
def Shape.wsel (s : Shape) : ℕ := packedWords s.m (s.k / 2) 2

/-- Row `i` of a sparse MMA's kept `A`. -/
def rowASp (fa : Fmt) (s : Shape) (v : ℕ → Word) (i : ℕ) : List ℕ :=
  List.ofFn fun l : Fin (s.k / 2) => elem fa.bits v (i * (s.k / 2) + l)

/-- The selector of row `i`'s kept element `l`: its position in its group of 4. -/
def selSp (fa : Fmt) (s : Shape) (v : ℕ → Word) (i l : ℕ) : ℕ :=
  elem 2 (fun w => v (s.waSp fa + w)) (i * (s.k / 2) + l)

/-- The elements of column `j` of `B` that row `i`'s selectors name, one for each kept element: the kept element `l`
is in group `l / 2`, so it meets `B`'s element `4 · (l / 2) + selector`. -/
def colBSp (fa fb : Fmt) (s : Shape) (v : ℕ → Word) (i j : ℕ) : List ℕ :=
  List.ofFn fun l : Fin (s.k / 2) =>
    elem fb.bits (fun w => v (s.waSp fa + s.wsel + w)) (j * s.k + 4 * (l.val / 2) + selSp fa s v i l)

/-- Accumulator `o` of a sparse MMA's `C`. -/
def accCSp (fa fb : Fmt) (s : Shape) (v : ℕ → Word) (o : ℕ) : Word := v (s.waSp fa + s.wsel + s.wb fb + o)

/-- **A sparse MMA's output** `o = i·n + j`: the sparse dot of accumulator `o` with row `i`'s kept elements and the
elements of column `j` its selectors name. Only the kept pairs meet a multiplier, as on the card, so an unselected
element of `B` reaches no output. -/
def mmaSpEval (dot : Dot) (fa fb : Fmt) (s : Shape) (v : ℕ → Word) (o : ℕ) : Word :=
  dot (accCSp fa fb s v o) (rowASp fa s v (o / s.n)) (colBSp fa fb s v (o / s.n) (o % s.n))

/-! ## Lane functions -/

/-- An integer comparison. -/
inductive Cmp
  | eq | ne | ltu | leu | lts | les
  deriving DecidableEq

/-- An atomic's update of the word it reads. -/
inductive AtomFn
  | add | exch
  deriving DecidableEq

/-- The CUDA-core pipe a lane function issues to: the integer `alu`, the integer multiplier `imad`, or the FP32 `fma`
pipe. -/
inductive Pipe
  | alu | imad | fma | shfl | f64
  deriving DecidableEq

/-- A lane function: one lane's instruction on 32-bit registers. -/
inductive Lane
  /-- `IADD3`: `a + b + c` -/
  | iadd3
  /-- `IMAD`: `a·b + c`, low 32 bits -/
  | mad
  /-- `IMAD.WIDE.U32`: `a·b + (c₁:c₀)` as two words, low then high -/
  | madWide
  /-- `IMAD.HI`: the high 32 bits of `a·b`, plus `c` (PTX's `mad.hi`), each factor signed or not -/
  | madHi (signed : Bool)
  /-- `LOP3`: the bitwise function of `a, b, c` whose truth table is `lut` -/
  | lop3 (lut : BitVec 8)
  /-- `SHF`: the funnel shift of `b:a` by `c`, left or right, the amount clamped to 32 or taken modulo 32 -/
  | shf (left clamp : Bool)
  /-- `PRMT`: the bytes of `b:a` that the selector `c`'s nibbles pick, sign-replicated where a nibble's top bit is set -/
  | prmt
  /-- `POPC`: the bits set -/
  | popc
  /-- `FLO.U32`: the index of the highest bit set, all ones for zero -/
  | flo
  /-- `BREV`: the bits reversed -/
  | brev
  /-- `ISETP` to a register: 1 if the comparison holds, else 0 -/
  | setp (c : Cmp)
  /-- `SEL`: `a` if `c ≠ 0`, else `b` -/
  | sel
  /-- `IMNMX`: the minimum -/
  | min (signed : Bool)
  /-- `IMNMX`: the maximum -/
  | max (signed : Bool)
  /-- `LEA`: `(a << s) + b` -/
  | lea (s : Fin 32)
  /-- `LEA.HI`: `b` plus the high word of `(c:a) << s` -/
  | leaHi (s : Fin 32)
  /-- `IDP4A`: `c + Σ` of the four byte products of `a` and `b`, each side signed or not -/
  | dp4a (sa sb : Bool)
  /-- `FFMA`: `a·b + c` in FP32, rounded once to nearest even (`Fp32.f32FmaV2`) -/
  | ffma
  /-- `FADD` (`Fp32.f32AddV2`) -/
  | fadd
  /-- `FMUL` (`Fp32.f32MulV2`) -/
  | fmul
  /-- `SHFL`: a cross-lane move, the word `a` (the pooled register file names the source lane's register) -/
  | shfl
  /-- `DFMA` over-approximated: `a·b + c` exactly, each operand the two words' 64-bit two's-complement integer within
  ±2^53 (FP64's exact integers; zero outside), the result's 64-bit two's complement as two words, low first (FP64's
  layout; the result itself wherever the card's is exact) -/
  | dfma
  /-- `HFMA2` over-approximated: for each half of the words, `a·b + c` exactly on the halves' 16-bit two's-complement
  integers within ±2048 (f16's exact integers; zero outside), the result's 16-bit two's complement in that half -/
  | hfma2
  /-- a halfwise add modulo `2^16` (no carry between halves): the `f16x2` and `bf16x2` reductions over-approximated,
  exact on their exact integers -/
  | add2
  /-- `REDUX`: the reduction by `g` (a `reduces` function) of 32 words, one a lane, to one word; only on the uniform
  datapath (`Opcode.uni`), a warp instruction -/
  | redux (g : Lane)
  deriving DecidableEq

namespace Lane

def nIn : Lane → ℕ
  | .popc | .flo | .brev | .shfl => 1
  | .setp _ | .min _ | .max _ | .lea _ | .fadd | .fmul | .add2 => 2
  | .madWide => 4
  | .dfma => 6
  | .redux _ => 32
  | _ => 3

/-- The lane functions a reduction to memory applies (`red`, `cp.reduce.async.bulk`): 32-bit integer add, min and max,
and the bitwise ones (`LOP3` with its third input zero, any two-input function, a superset of `and`, `or`, `xor`). The
card's float reductions are over-approximated (the module's docstring): `f32` by `iadd3`, `f16x2` and `bf16x2` by
`add2`. Its 64-bit and `inc`/`dec` reductions are other functions, not here. -/
def reduces : Lane → Bool
  | .iadd3 | .min _ | .max _ | .lop3 _ | .add2 => true
  | _ => false

/-- A warp-wide function, available on the uniform datapath only. -/
def warp : Lane → Bool
  | .redux _ => true
  | _ => false

def nOut : Lane → ℕ
  | .madWide | .dfma => 2
  | _ => 1

def pipe : Lane → Pipe
  | .mad | .madWide | .madHi _ | .dp4a _ _ => .imad
  | .ffma | .fadd | .fmul | .hfma2 | .add2 => .fma
  | .shfl => .shfl
  | .dfma => .f64
  | _ => .alu

/-- Whether the uniform datapath runs `g`: an integer function, or `redux` of a reduction function. -/
def uniform : Lane → Bool
  | .redux g => g.reduces
  | g => g.pipe == .alu || g.pipe == .imad

/-- Byte `i` of `w`, signed or not. -/
def byte (signed : Bool) (w : Word) (i : ℕ) : ℤ :=
  let v := w.toNat / 2 ^ (8 * i) % 256
  if signed ∧ 128 ≤ v then (v : ℤ) - 256 else v

/-- The 64-bit two's-complement integer of words `lo`, `hi`, if within FP64's exact integers (±2^53), else 0. -/
def int53 (lo hi : Word) : ℤ :=
  let v := (BitVec.ofNat 64 (hi.toNat * 2 ^ 32 + lo.toNat)).toInt
  if v.natAbs ≤ 2 ^ 53 then v else 0

/-- Half `o` of `w` as a 16-bit two's-complement integer, if within f16's exact integers (±2048), else 0. -/
def half11 (w : Word) (o : ℕ) : ℤ :=
  let v := (BitVec.ofNat 16 (w.toNat / 2 ^ (16 * o))).toInt
  if v.natAbs ≤ 2048 then v else 0

def lop3Eval (lut : BitVec 8) (a b c : Word) : Word :=
  (List.range 8).foldl (fun acc t =>
    if lut.getLsbD t then
      acc ||| ((if t / 4 % 2 = 1 then a else ~~~a) &&& (if t / 2 % 2 = 1 then b else ~~~b) &&&
        (if t % 2 = 1 then c else ~~~c))
    else acc) 0

def prmtEval (a b c : Word) : Word :=
  let x := b.toNat * 2 ^ 32 + a.toNat
  BitVec.ofNat 32 ((List.range 4).foldl (fun acc i =>
    let s := c.toNat / 2 ^ (4 * i) % 16
    let v := x / 2 ^ (8 * (s % 8)) % 256
    let v' := if s / 8 = 1 then (if 128 ≤ v then 255 else 0) else v
    acc + v' * 2 ^ (8 * i)) 0)

def cmp : Cmp → Word → Word → Bool
  | .eq, a, b => a = b
  | .ne, a, b => a ≠ b
  | .ltu, a, b => a.ult b
  | .leu, a, b => a.ule b
  | .lts, a, b => a.slt b
  | .les, a, b => a.sle b

/-- A lane function's output `o` from its inputs `x 0, x 1, …`. -/
def eval : Lane → (ℕ → Word) → ℕ → Word
  | .iadd3, x, _ => x 0 + x 1 + x 2
  | .mad, x, _ => x 0 * x 1 + x 2
  | .madWide, x, o =>
    let s := (x 0).toNat * (x 1).toNat + ((x 3).toNat * 2 ^ 32 + (x 2).toNat)
    BitVec.ofNat 32 (if o = 0 then s else s / 2 ^ 32)
  | .madHi signed, x, _ =>
    let p : ℤ := if signed then (x 0).toInt * (x 1).toInt else (x 0).toNat * (x 1).toNat
    BitVec.ofInt 32 (p / 2 ^ 32) + x 2
  | .lop3 lut, x, _ => lop3Eval lut (x 0) (x 1) (x 2)
  | .shf left clamp, x, _ =>
    let n := if clamp then Nat.min (x 2).toNat 32 else (x 2).toNat % 32
    let w := (x 1).toNat * 2 ^ 32 + (x 0).toNat
    BitVec.ofNat 32 (if left then w * 2 ^ n / 2 ^ 32 else w / 2 ^ n)
  | .prmt, x, _ => prmtEval (x 0) (x 1) (x 2)
  | .popc, x, _ => BitVec.ofNat 32 ((List.range 32).countP fun i => (x 0).getLsbD i)
  | .flo, x, _ => if x 0 = 0 then BitVec.allOnes 32 else BitVec.ofNat 32 (x 0).toNat.log2
  | .brev, x, _ => BitVec.ofNat 32 ((List.range 32).foldl (fun acc i =>
      if (x 0).getLsbD i then acc + 2 ^ (31 - i) else acc) 0)
  | .setp c, x, _ => if cmp c (x 0) (x 1) then 1 else 0
  | .sel, x, _ => if x 2 ≠ 0 then x 0 else x 1
  | .min signed, x, _ => if (if signed then (x 0).slt (x 1) else (x 0).ult (x 1)) then x 0 else x 1
  | .max signed, x, _ => if (if signed then (x 0).slt (x 1) else (x 0).ult (x 1)) then x 1 else x 0
  | .lea s, x, _ => (x 0 <<< s.val) + x 1
  | .leaHi s, x, _ => x 1 + BitVec.ofNat 32 (((x 2).toNat * 2 ^ 32 + (x 0).toNat) * 2 ^ s.val / 2 ^ 32)
  | .dp4a sa sb, x, _ =>
    BitVec.ofInt 32 ((x 2).toInt + ((List.range 4).map fun i => byte sa (x 0) i * byte sb (x 1) i).sum)
  | .ffma, x, _ => BitVec.ofNat 32 (Fp32.f32FmaV2 (x 0).toNat (x 1).toNat (x 2).toNat)
  | .fadd, x, _ => BitVec.ofNat 32 (Fp32.f32AddV2 (x 0).toNat (x 1).toNat)
  | .fmul, x, _ => BitVec.ofNat 32 (Fp32.f32MulV2 (x 0).toNat (x 1).toNat)
  | .shfl, x, _ => x 0
  | .dfma, x, o => BitVec.ofInt 32 ((int53 (x 0) (x 1) * int53 (x 2) (x 3) + int53 (x 4) (x 5)) / 2 ^ (32 * o))
  | .hfma2, x, _ => BitVec.ofNat 32 ((BitVec.ofInt 16 (half11 (x 0) 0 * half11 (x 1) 0 + half11 (x 2) 0)).toNat +
      (BitVec.ofInt 16 (half11 (x 0) 1 * half11 (x 1) 1 + half11 (x 2) 1)).toNat * 2 ^ 16)
  | .add2, x, _ => BitVec.ofNat 32 (((x 0).toNat + (x 1).toNat) % 2 ^ 16 +
      ((x 0).toNat / 2 ^ 16 + (x 1).toNat / 2 ^ 16) % 2 ^ 16 * 2 ^ 16)
  | .redux g, x, _ =>
    (List.range 31).foldl (fun acc i => g.eval (fun j => if j = 0 then acc else if j = 1 then x (i + 1) else 0) 0) (x 0)

end Lane

/-! ## Opcodes -/

/-- An opcode over stores `S`. -/
inductive Opcode (S : Type) where
  /-- an MMA of `fa` rows by `fb` columns at shape `sh`, `A` read from store `sa` and `B` from `sb` (`C` and `D` are
  registers); `sat` clamps an integer MMA's result -/
  | mma (fa fb : Fmt) (sat : Bool) (sh : Shape) (sa sb : S)
  /-- a 2:4 sparse MMA at logical shape `sh`, its kept `A` read from store `sa`, its metadata from registers and `B`
  from `sb` (`C` and `D` are registers) -/
  | mmaSp (fa fb : Fmt) (sat : Bool) (sh : Shape) (sa sb : S)
  /-- a lane function on registers -/
  | lane (g : Lane)
  /-- a uniform-datapath instruction: one lane function a warp computes once (`UIADD3`, `ULOP3`, …), on registers -/
  | uni (g : Lane)
  /-- `n` words from store `s` to store `t` -/
  | copy (s t : S) (n : ℕ)
  /-- a broadcast load: `n` words of store `s`, read once, to the registers of `k` lanes (output `i` is word `i % n`),
  as when the lanes of a warp instruction read one address (`LDS`, `LDG`) -/
  | bcast (s : S) (n k : ℕ)
  /-- an atomic on a word of store `s`: input 0 and output 0 are that word, input 1 the register operand, and output
  1 the word's old value -/
  | atom (s : S) (g : AtomFn)
  /-- a reduction of `n` words of store `s` into `n` words of store `t` (`red.v4`, `red.v8`, TMA's
  `cp.reduce.async.bulk`): output `i` is lane function `g` of input `n + i`, the word of `t` it replaces, and input
  `i`, the word of `s` -/
  | red (s t : S) (n : ℕ) (g : Lane)
  /-- DMMA over-approximated: an exact integer MMA at shape `sh` on registers (`dmmaEval`) -/
  | dmma (sh : Shape)
  deriving DecidableEq

/-- **DMMA's output** `o`, over-approximated: word `o % 2` (low first) of the 64-bit two's complement of
`C q + Σ_l A i l · B l j` exactly, `q = o / 2 = i·n + j`, each element two words read by `Lane.int53` (FP64's exact
integers). The inputs `v` are `A` (`m` rows of `k`), `B` (`n` columns of `k`) and `C` (`m·n`), two words an element,
and so is the output, FP64's layout. -/
def dmmaEval (s : Shape) (v : ℕ → Word) (o : ℕ) : Word :=
  let e (w : ℕ) : ℤ := Lane.int53 (v w) (v (w + 1))
  let q := o / 2
  let i := q / s.n
  let j := q % s.n
  let d := e (2 * s.m * s.k + 2 * s.k * s.n + 2 * q) +
    ((List.range s.k).map fun l => e (2 * (i * s.k + l)) * e (2 * s.m * s.k + 2 * (j * s.k + l))).sum
  BitVec.ofInt 32 (d / 2 ^ (32 * (o % 2)))

/-- The `i`th of `n` words, zero past the end. -/
def getD {n : ℕ} (v : Fin n → Word) (i : ℕ) : Word := if h : i < n then v ⟨i, h⟩ else 0

namespace Opcode

variable {S : Type}

def nIn : Opcode S → ℕ
  | .mma fa fb _ sh _ _ => sh.wa fa + sh.wb fb + sh.m * sh.n
  | .mmaSp fa fb _ sh _ _ => sh.waSp fa + sh.wsel + sh.wb fb + sh.m * sh.n
  | .lane g | .uni g => g.nIn
  | .copy _ _ n | .bcast _ n _ => n
  | .atom _ _ => 2
  | .red _ _ n _ => 2 * n
  | .dmma sh => 2 * sh.m * sh.k + 2 * sh.k * sh.n + 2 * sh.m * sh.n

def nOut : Opcode S → ℕ
  | .mma _ _ _ sh _ _ | .mmaSp _ _ _ sh _ _ => sh.m * sh.n
  | .lane g | .uni g => g.nOut
  | .copy _ _ n | .red _ _ n _ => n
  | .bcast _ n k => k * n
  | .atom _ _ => 2
  | .dmma sh => 2 * sh.m * sh.n

/-- The store input `i` is read from (`reg` is the register store). -/
def inStore (reg : S) : (o : Opcode S) → Fin o.nIn → S
  | .mma fa fb _ sh sa sb, i => if i.val < sh.wa fa then sa else if i.val < sh.wa fa + sh.wb fb then sb else reg
  | .mmaSp fa fb _ sh sa sb, i =>
    if i.val < sh.waSp fa then sa
    else if sh.waSp fa + sh.wsel ≤ i.val ∧ i.val < sh.waSp fa + sh.wsel + sh.wb fb then sb else reg
  | .lane _, _ | .uni _, _ => reg
  | .copy s _ _, _ | .bcast s _ _, _ => s
  | .atom s _, i => if i.val = 0 then s else reg
  | .red s t n _, i => if i.val < n then s else t
  | .dmma _, _ => reg

/-- The store output `i` is written to. -/
def outStore (reg : S) : (o : Opcode S) → Fin o.nOut → S
  | .mma .., _ | .mmaSp .., _ => reg
  | .lane _, _ | .uni _, _ => reg
  | .copy _ t _, _ => t
  | .bcast _ _ _, _ => reg
  | .atom s _, i => if i.val = 0 then s else reg
  | .red _ t _ _, _ => t
  | .dmma _, _ => reg

/-- An opcode's semantics, the MMA's dot for formats `fa`, `fb`, saturation and `k` given by `dot`, and a sparse MMA's
at its logical `k` by `dotSp`. -/
def eval (dot dotSp : Fmt → Fmt → Bool → ℕ → Dot) : (o : Opcode S) → (Fin o.nIn → Word) → Fin o.nOut → Word
  | .mma fa fb sat sh _ _, v, o => mmaEval (dot fa fb sat sh.k) fa fb sh (getD v) o.val
  | .mmaSp fa fb sat sh _ _, v, o => mmaSpEval (dotSp fa fb sat sh.k) fa fb sh (getD v) o.val
  | .lane g, v, o | .uni g, v, o => g.eval (getD v) o.val
  | .copy _ _ _, v, o => getD v o.val
  | .bcast _ n _, v, o => getD v (o.val % n)
  | .atom _ .add, v, o => if o.val = 0 then getD v 0 + getD v 1 else getD v 0
  | .atom _ .exch, v, o => if o.val = 0 then getD v 1 else getD v 0
  | .red _ _ n g, v, o => g.eval (fun j => if j = 0 then getD v (n + o.val) else if j = 1 then getD v o.val else 0) 0
  | .dmma sh, v, o => dmmaEval sh (getD v) o.val

/-- The products an MMA forms: `m·n·k`, and a sparse MMA's `m·n·k/2`. -/
def macs : Opcode S → ℕ
  | .mma _ _ _ sh _ _ => sh.m * sh.n * sh.k
  | .mmaSp _ _ _ sh _ _ => sh.m * sh.n * (sh.k / 2)
  | .dmma sh => sh.m * sh.n * sh.k
  | _ => 0

end Opcode

end Verity.Accel
