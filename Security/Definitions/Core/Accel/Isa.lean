import Definitions.Core.TC.Spec
import Mathlib.Algebra.BigOperators.Group.Finset.Basic

/-!
# The ACCEL ops (trusted)

The machine 𝓜's instructions: a handful of coarse classes (`Op`), each at least as powerful a unit of its resource as
any instruction of the card it covers (`Assumptions.CardWithinAccel`). What a class computes is the card's own
(`Open`, the card's semantics, which no table states): the ops fix only what an instruction reads, what it writes and
its units (`Card.use`). Two things have semantics here, the ones an honest kernel needs: a move copies its
words unchanged, and an integer MMA is exact within the table's range (P1, `Card.Exact`). Every MMA's dot is a function
of its accumulator and its pairs' products (P2, `ProductForm`).

* **tensor** (`Op.mma`, `Op.mmaSp`): an MMA tile of any shape `m × n × k` on one of the card's MMA opcodes (operand
  formats, 32- or 64-bit accumulators): output `(i, j)` is the opcode's dot of accumulator `(i, j)` with row `i` of `A`
  and column `j` of `B` (`mmaEval`), so each of its `m·n·k` products meets one output; a 2:4 sparse tile keeps `k/2`
  pairs a row, chosen by metadata (`mmaSpEval`);
* **lane** (`Op.lane`): one of the card's lane functions, from at most `Pipe.nIn` register words to two;
* **move** (`Op.move`): `n` words of one store, read once, to `k` copies in another, unchanged;
* **read-modify-write** (`Op.atom`, `Op.red`): one of the card's functions of a memory word and two register words to
  the new word and a returned word (atomics), or of `n` words of one store into `n` words of another (reductions);
* **branch** (`Machine.Instr.bra`), on a predicate bit.

The hash oracle has no instruction: a program reaches it only through the accelerator's priced calls
(`Pouw.Dither.Accel.G3`, `Pouw.Dither.CallsAtPub`).
-/

namespace Verity.Accel

open Verity.TC

/-- A word: the stores hold 32-bit words. -/
abbrev Word : Type := BitVec 32

/-- The stores, each pooled over the card: registers (predicates are bits of register words), shared memory with L1,
L2 and device memory. -/
inductive Store
  | reg | smem | l2 | dram
  deriving DecidableEq

/-- The resources, each pooled over the card: the tensor core in `s8` MACs; the warp schedulers' issue slots in warp
instructions; the integer, multiply-add, shuffle, FP64 and special-function pipes in lanes; shared memory's banks, L2's
bandwidth (reads and writes) and device memory's in words. -/
inductive Res
  | tensor | issue | alu | imad | shfl | f64 | sfu | smem | l2 | dram
  deriving DecidableEq

/-- A lane function's pipe. The FP32 pipe (`fma`) is no resource: its rate is the issue rate. -/
inductive Pipe
  | alu | imad | fma | shfl | f64 | sfu
  deriving DecidableEq

/-- The resource a lane function on the pipe uses a lane of. -/
def Pipe.res : Pipe → Option Res
  | .alu => some .alu
  | .imad => some .imad
  | .shfl => some .shfl
  | .f64 => some .f64
  | .sfu => some .sfu
  | .fma => none

/-- The register words a lane function reads: six on the FP64 pipe (three FP64 operands), four on the others. -/
def Pipe.nIn : Pipe → ℕ
  | .f64 => 6
  | _ => 4

/-! ## Tensor-core formats -/

/-- A tensor-core operand format: the integers `s8`, `u8` and `b1`, the floats, the block-scaled FP4 formats `nvf4` (a
UE4M3 scale per 16 E2M1 operands) and `mxf4` (a UE8M0 scale per 32), and FP64. -/
inductive Fmt
  | s8 | u8 | b1 | e4m3 | e5m2 | e2m1 | f16 | bf16 | tf32 | nvf4 | mxf4 | f64
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

/-- An integer operand's value: `s8` two's complement, the others unsigned. -/
def Fmt.intVal (f : Fmt) (x : ℕ) : ℤ :=
  match f with
  | .s8 => if x < 128 then x else (x : ℤ) - 256
  | _ => x

/-- Which codes of a float format are NaN or infinite: IEEE (an all-ones exponent), E4M3 (all-ones exponent and
fraction is NaN, no infinities), or none (E2M1). -/
inductive NanRule
  | ieee | e4m3 | none

/-- A float encoding: a sign, `expBits` exponent bits and `manBits` fraction bits. -/
structure MiniFloat where
  expBits : ℕ
  manBits : ℕ
  nan : NanRule

/-- A code's class. -/
def MiniFloat.cls (f : MiniFloat) (w : ℕ) : Spec.Cls :=
  let e := (w >>> f.manBits) &&& (2 ^ f.expBits - 1)
  let m := w &&& (2 ^ f.manBits - 1)
  let top := 2 ^ f.expBits - 1
  match f.nan with
  | .ieee => if e = top then (if m ≠ 0 then .nan else .inf) else if e = 0 ∧ m = 0 then .zero else .finite
  | .e4m3 => if e = top ∧ m = 2 ^ f.manBits - 1 then .nan else if e = 0 ∧ m = 0 then .zero else .finite
  | .none => if e = 0 ∧ m = 0 then .zero else .finite

/-- A float format's encoding: OCP E4M3, E5M2 and E2M1, binary16, BF16, TF32 (an FP32 word's top 19 bits), FP64. -/
def Fmt.mini : Fmt → MiniFloat
  | .e4m3 => ⟨4, 3, .e4m3⟩
  | .e5m2 => ⟨5, 2, .ieee⟩
  | .e2m1 | .nvf4 | .mxf4 => ⟨2, 1, .none⟩
  | .f16 => ⟨5, 10, .ieee⟩
  | .tf32 => ⟨8, 10, .ieee⟩
  | .f64 => ⟨11, 52, .ieee⟩
  | _ => ⟨8, 7, .ieee⟩

/-- A float operand's code in its encoding: a TF32 operand is an FP32 word, read through its top 19 bits. -/
def Fmt.code (f : Fmt) (x : ℕ) : ℕ := if f = .tf32 then x >>> 13 else x

/-- The formats whose MMA is the integer path: `s8` and `u8`, either way. -/
def intPair : Fmt → Fmt → Bool
  | .s8, .s8 | .s8, .u8 | .u8, .s8 | .u8, .u8 => true
  | _, _ => false

/-- **What one multiplier forms** from operand `x` of format `fa` and `y` of `fb`: for an integer pair the exact product
(its sign and magnitude); for `b1` the two bits (`AND`, `XOR` and their popcounts are functions of them); for a float
pair the operands' classes, the product's sign, the significands' product and the exponent fields' sum (a subnormal's
as 1), which is the exact product with its exponent. -/
def mulOut (fa fb : Fmt) (x y : ℕ) : Spec.Cls × Spec.Cls × Bool × ℕ × ℕ :=
  if intPair fa fb then
    let p := fa.intVal x * fb.intVal y
    (.finite, .finite, decide (p < 0), p.natAbs, 0)
  else if fa = .b1 then (.finite, .finite, false, x, y)
  else
    let f := fa.mini
    let g := fb.mini
    let u := fa.code x
    let v := fb.code y
    let eu := (u >>> f.manBits) &&& (2 ^ f.expBits - 1)
    let ev := (v >>> g.manBits) &&& (2 ^ g.expBits - 1)
    let su := (u &&& (2 ^ f.manBits - 1)) ||| (if eu ≠ 0 then 2 ^ f.manBits else 0)
    let sv := (v &&& (2 ^ g.manBits - 1)) ||| (if ev ≠ 0 then 2 ^ g.manBits else 0)
    (f.cls u, g.cls v, ((u >>> (f.expBits + f.manBits)) ^^^ (v >>> (g.expBits + g.manBits))) &&& 1 = 1, su * sv,
      (if eu ≠ 0 then eu else 1) + (if ev ≠ 0 then ev else 1))

/-! ## MMA tiles -/

/-- One accumulator's step: the accumulator's bits and a row's and a column's elements to the new accumulator's bits. -/
abbrev Dot : Type := ℕ → List ℕ → List ℕ → ℕ

/-- **(P2) product form**: a dot of `k` products of formats `fa` and `fb` reads its operand pairs only through their
multiplier outputs: on rows of the format's length, it is a function of the accumulator, the `k` pairs' `mulOut` and
the rows' block scales (empty unless the format is block-scaled). So a product reaches no output but its own, and only
as its product. -/
def ProductForm (fa fb : Fmt) (k : ℕ) (d : Dot) : Prop :=
  ∃ F : ℕ → List (Spec.Cls × Spec.Cls × Bool × ℕ × ℕ) → List ℕ → List ℕ → ℕ,
    ∀ (c : ℕ) (a b : List ℕ), a.length = fa.rowLen k → b.length = fb.rowLen k →
      d c a b = F c (List.zipWith (mulOut fa fb) (a.take k) (b.take k)) (a.drop k) (b.drop k)

/-- An MMA tile's shape: `m × n` accumulators, `k` products each. -/
structure Shape where
  m : ℕ
  n : ℕ
  k : ℕ
  deriving DecidableEq

/-- The words `r` rows of `len` elements of `b` bits occupy, packed row after row. -/
def packedWords (r len b : ℕ) : ℕ := (r * len * b + 31) / 32

/-- Element `e` of the `b`-bit elements packed into words `x`: a 64-bit element is two words, low first; a narrower
one divides 32, so none straddles two words. -/
def elem (b : ℕ) (x : ℕ → Word) (e : ℕ) : ℕ :=
  if b = 64 then (x (2 * e)).toNat + (x (2 * e + 1)).toNat * 2 ^ 32
  else (x (e * b / 32)).toNat / 2 ^ (e * b % 32) % 2 ^ b

/-- The words of `A`: `m` rows of `fa.rowLen k` elements. -/
def Shape.wa (s : Shape) (fa : Fmt) : ℕ := packedWords s.m (fa.rowLen s.k) fa.bits

/-- The words of `B`: `n` columns of `fb.rowLen k` elements. -/
def Shape.wb (s : Shape) (fb : Fmt) : ℕ := packedWords s.n (fb.rowLen s.k) fb.bits

/-- Row `i` of `A`, from a tile's inputs `v` (`A`'s words first). -/
def rowA (fa : Fmt) (s : Shape) (v : ℕ → Word) (i : ℕ) : List ℕ :=
  List.ofFn fun l : Fin (fa.rowLen s.k) => elem fa.bits v (i * fa.rowLen s.k + l.val)

/-- Column `j` of `B`, from a tile's inputs `v` (`B`'s words `off` words in). -/
def colB (fb : Fmt) (s : Shape) (off : ℕ) (v : ℕ → Word) (j : ℕ) : List ℕ :=
  List.ofFn fun l : Fin (fb.rowLen s.k) => elem fb.bits (fun w => v (off + w)) (j * fb.rowLen s.k + l.val)

/-- The words an accumulator takes: two for a 64-bit one. -/
def accWords (wide : Bool) : ℕ := if wide then 2 else 1

/-- Accumulator `q`'s bits, from words `off` on of a tile's inputs `v`. -/
def accC (wide : Bool) (off : ℕ) (v : ℕ → Word) (q : ℕ) : ℕ :=
  if wide then (v (off + 2 * q)).toNat + (v (off + 2 * q + 1)).toNat * 2 ^ 32 else (v (off + q)).toNat

/-- **A tile's output word** `o`: word `o % accWords` (low first) of accumulator `q = o / accWords = i·n + j`, the dot
of the accumulator with row `i` and column `j`. -/
def mmaEval (dot : Dot) (fa fb : Fmt) (wide : Bool) (s : Shape) (v : ℕ → Word) (o : ℕ) : Word :=
  let q := o / accWords wide
  BitVec.ofNat 32 (dot (accC wide (s.wa fa + s.wb fb) v q) (rowA fa s v (q / s.n)) (colB fb s (s.wa fa) v (q % s.n)) /
    2 ^ (32 * (o % accWords wide)))

/-- The words of a sparse tile's kept `A`: `m` rows of `k/2` elements. -/
def Shape.waSp (s : Shape) (fa : Fmt) : ℕ := packedWords s.m (s.k / 2) fa.bits

/-- The words of a sparse tile's metadata: `m` rows of `k/2` selectors of 2 bits. -/
def Shape.wsel (s : Shape) : ℕ := packedWords s.m (s.k / 2) 2

/-- Row `i` of a sparse tile's kept `A`. -/
def rowASp (fa : Fmt) (s : Shape) (v : ℕ → Word) (i : ℕ) : List ℕ :=
  List.ofFn fun l : Fin (s.k / 2) => elem fa.bits v (i * (s.k / 2) + l.val)

/-- The selector of row `i`'s kept element `l`: its position in its group of 4. -/
def selSp (fa : Fmt) (s : Shape) (v : ℕ → Word) (i l : ℕ) : ℕ :=
  elem 2 (fun w => v (s.waSp fa + w)) (i * (s.k / 2) + l)

/-- The elements of column `j` of `B` that row `i`'s selectors name: kept element `l` is in group `l / 2`, so it meets
`B`'s element `4 · (l / 2) + selector`. -/
def colBSp (fa fb : Fmt) (s : Shape) (v : ℕ → Word) (i j : ℕ) : List ℕ :=
  List.ofFn fun l : Fin (s.k / 2) =>
    elem fb.bits (fun w => v (s.waSp fa + s.wsel + w)) (j * s.k + 4 * (l.val / 2) + selSp fa s v i l.val)

/-- **A sparse tile's output word** `o`: as `mmaEval`, over row `i`'s kept elements and the elements of column `j` its
selectors name, so an unselected element of `B` reaches no output. -/
def mmaSpEval (dot : Dot) (fa fb : Fmt) (wide : Bool) (s : Shape) (v : ℕ → Word) (o : ℕ) : Word :=
  let q := o / accWords wide
  BitVec.ofNat 32 (dot (accC wide (s.waSp fa + s.wsel + s.wb fb) v q) (rowASp fa s v (q / s.n))
    (colBSp fa fb s v (q / s.n) (q % s.n)) / 2 ^ (32 * (o % accWords wide)))

/-! ## The card's semantics and the ACCEL ops -/

/-- **The card's semantics**: the functions its instructions compute. No table states them; a statement over the model
holds at every `Open` meeting P1 (`Card.Exact`) and P2 (`Open.ProductForm`, as `Assumptions.AccelProductForm`), and
the card's own is one (`Assumptions.CardWithinAccel`). -/
structure Open where
  /-- the card's lane opcodes, each on its pipe, with what it computes from its input words (`pipe.nIn` of them) -/
  Lane : Type
  pipe : Lane → Pipe
  lane : Lane → (ℕ → Word) → Fin 2 → Word
  /-- the card's read-modify-write opcodes: from a memory word and two register words to the memory word's new value
  and a returned word -/
  Atom : Type
  atom : Atom → Word → Word → Word → Fin 2 → Word
  /-- the card's MMA opcodes: operand formats, 64-bit accumulators or not, and dot at each `k` (a 2:4 sparse tile's at
  its logical `k`, over its `k/2` kept pairs) -/
  Mma : Type
  fa : Mma → Fmt
  fb : Mma → Fmt
  wide : Mma → Bool
  dot : Mma → ℕ → Dot
  dotSp : Mma → ℕ → Dot

/-- **(P2) at the semantics `x`**: every MMA opcode's dot, dense and 2:4 sparse, has the product form
(`ProductForm`). -/
def Open.ProductForm (x : Open) : Prop :=
  ∀ (o : x.Mma) (k : ℕ), Accel.ProductForm (x.fa o) (x.fb o) k (x.dot o k) ∧
    Accel.ProductForm (x.fa o) (x.fb o) (k / 2) (x.dotSp o k)

/-- **The ACCEL ops** at the card's semantics `x`. -/
inductive Op (x : Open) where
  /-- an MMA tile of opcode `o` at shape `sh`, `A` read from store `sa` and `B` from `sb` (accumulators in registers) -/
  | mma (o : x.Mma) (sh : Shape) (sa sb : Store)
  /-- a 2:4 sparse tile at logical shape `sh`, its kept `A` from `sa`, its metadata from registers and `B` from `sb` -/
  | mmaSp (o : x.Mma) (sh : Shape) (sa sb : Store)
  /-- a lane function on registers -/
  | lane (o : x.Lane)
  /-- `n` words of store `s`, read once, to `k` copies in store `t` (output `i` is word `i % n`) -/
  | move (s t : Store) (n k : ℕ)
  /-- an atomic on a word of store `s`: input 0 and output 0 are that word, inputs 1 and 2 and output 1 registers -/
  | atom (s : Store) (o : x.Atom)
  /-- a reduction of `n` words of store `s` into `n` words of store `t`: output `i` replaces input `n + i` -/
  | red (s t : Store) (n : ℕ) (o : x.Atom)

/-- The `i`th word of `v`, zero past the end. -/
def getD {n : ℕ} (v : Fin n → Word) (i : ℕ) : Word := if h : i < n then v ⟨i, h⟩ else 0

namespace Op

variable {x : Open}

def nIn : Op x → ℕ
  | .mma o sh _ _ => sh.wa (x.fa o) + sh.wb (x.fb o) + sh.m * sh.n * accWords (x.wide o)
  | .mmaSp o sh _ _ => sh.waSp (x.fa o) + sh.wsel + sh.wb (x.fb o) + sh.m * sh.n * accWords (x.wide o)
  | .lane o => (x.pipe o).nIn
  | .move _ _ n _ => n
  | .atom _ _ => 3
  | .red _ _ n _ => 2 * n

def nOut : Op x → ℕ
  | .mma o sh _ _ | .mmaSp o sh _ _ => sh.m * sh.n * accWords (x.wide o)
  | .lane _ => 2
  | .move _ _ n k => k * n
  | .atom _ _ => 2
  | .red _ _ n _ => n

/-- The store input `i` is read from. -/
def inStore : (o : Op x) → Fin o.nIn → Store
  | .mma o sh sa sb, i =>
    if i.val < sh.wa (x.fa o) then sa else if i.val < sh.wa (x.fa o) + sh.wb (x.fb o) then sb else .reg
  | .mmaSp o sh sa sb, i =>
    if i.val < sh.waSp (x.fa o) then sa
    else if sh.waSp (x.fa o) + sh.wsel ≤ i.val ∧ i.val < sh.waSp (x.fa o) + sh.wsel + sh.wb (x.fb o) then sb
    else .reg
  | .lane _, _ => .reg
  | .move s _ _ _, _ => s
  | .atom s _, i => if i.val = 0 then s else .reg
  | .red s t n _, i => if i.val < n then s else t

/-- The store output `i` is written to. -/
def outStore : (o : Op x) → Fin o.nOut → Store
  | .mma .., _ | .mmaSp .., _ | .lane _, _ => .reg
  | .move _ t _ _, _ => t
  | .atom s _, i => if i.val = 0 then s else .reg
  | .red _ t _ _, _ => t

/-- What an opcode writes: its output words from its input words. -/
def eval : (o : Op x) → (Fin o.nIn → Word) → Fin o.nOut → Word
  | .mma o sh _ _, v, i => mmaEval (x.dot o sh.k) (x.fa o) (x.fb o) (x.wide o) sh (getD v) i.val
  | .mmaSp o sh _ _, v, i => mmaSpEval (x.dotSp o sh.k) (x.fa o) (x.fb o) (x.wide o) sh (getD v) i.val
  | .lane o, v, i => x.lane o (getD v) i
  | .move _ _ n _, v, i => getD v (i.val % n)
  | .atom _ o, v, i => x.atom o (getD v 0) (getD v 1) (getD v 2) i
  | .red _ _ n o, v, i => x.atom o (getD v (n + i.val)) (getD v i.val) 0 0

end Op

end Verity.Accel
