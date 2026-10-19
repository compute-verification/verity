import Definitions.Accounting.TS.Routes

/-!
# A device's instruction table (trusted)

A7 (`Accounting.Assumptions.TS.SheetSemantics`) and A6 (`SheetWords`) as data (note:20261019T0520Z-draft-mm-isa): one
row per instruction class of a device, each row the reading of its PTX ISA section (PTX ISA 9.4), and the checks a row
passes, computed from its fields.

* `Fmt`: the number formats an instruction reads or writes (PTX ISA §5.2.1, §5.2.3), with `Fmt.holdsS8` and
  `Fmt.holdsWord` computed from each float's significand bits and largest exponent (isa_escape.py's `exact_in_float`).
* `Row`: a class's family (which PTX section, and which named premise, covers it), where it executes, what it
  computes (`Arith`), whether it can take a data-dependent address, lane or selector, its operand and accumulator
  formats, the 32-bit words one instruction writes at most, its rate class (the data-sheet row it loads beside issue)
  and its route on the TS menu (`Route`).
* `Row.verdict`: isa_escape.py's classes, computed. `Row.ok`: the table's discipline, decided per row.
* `Heads`: a cheater's labelled run split by the class of each head (EPS1 §1.3's Definition 2), which the premises of
  `Models.Accounting.Assumptions.Isa` are stated over; `Fits`, `ReadsTable` and `RowsOf` relate a table, a model and
  that split.
-/

namespace Accounting.Isa

open Accounting.TS

/-- **A number format** an instruction reads or writes (PTX ISA §5.2.1 Fundamental Types, §5.2.3 Alternate
Floating-Point Data Formats); `none` where the row has no such operand. -/
inductive Fmt
  | none
  | b1 | u4 | s4 | u8 | s8 | u16 | s16 | u32 | s32 | u64 | s64
  | e2m1 | e3m2 | e2m3 | e4m3 | e5m2 | ue4m3 | ue8m0
  | f16 | bf16 | tf32 | f32 | f64
  deriving DecidableEq, Repr

/-- A signed float format's significand bits (the implicit one included) and its largest exponent (PTX ISA §5.2.3,
IEEE 754 for the rest); `none` for the integer formats, the unsigned scale formats and `none`. -/
def Fmt.float : Fmt → Option (ℕ × ℕ)
  | .e2m1 => some (2, 2)
  | .e3m2 => some (3, 4)
  | .e2m3 => some (4, 2)
  | .e4m3 => some (4, 8)
  | .e5m2 => some (3, 15)
  | .f16 => some (11, 15)
  | .bf16 => some (8, 127)
  | .tf32 => some (11, 127)
  | .f32 => some (24, 127)
  | .f64 => some (53, 1023)
  | _ => none

/-- An integer format's bits, its sign bit included (`0` for the others). -/
def Fmt.intBits : Fmt → ℕ
  | .b1 => 1
  | .u4 | .s4 => 4
  | .u8 | .s8 => 8
  | .u16 | .s16 => 16
  | .u32 | .s32 => 32
  | .u64 | .s64 => 64
  | _ => 0

/-- Whether a binary float with `sig` significand bits and largest exponent `emax` holds the integer magnitude `a`
exactly: `a = 0`, or `a < 2^(emax+1)` and `a = m · 2^e` with `m < 2^sig`. Exponents are searched below 25, which is
exact for `a < 2^25`. -/
def exactFloat (sig emax a : ℕ) : Bool :=
  a == 0 || (decide (a < 2 ^ (emax + 1)) &&
    (List.range (min (emax + 1) 25)).any fun e => a % 2 ^ e == 0 && decide (a / 2 ^ e < 2 ^ sig))

/-- **Every int8 operand is exact in the format**: every magnitude up to 128 for a float, at least 8 bits for an integer
format (an unsigned one up to the offset 128, which one correction term undoes). -/
def Fmt.holdsS8 (f : Fmt) : Bool :=
  match f.float with
  | some (sig, emax) => (List.range 129).all (exactFloat sig emax)
  | none => decide (8 ≤ f.intBits)

/-- **A committed word is exact in the format**: a 32-step sum of int8 products, below `2^19`, has at most 19
significant bits, so a float with 19 significand bits and exponents to 18 holds it, and an integer format of 20 bits. -/
def Fmt.holdsWord (f : Fmt) : Bool :=
  match f.float with
  | some (sig, emax) => decide (19 ≤ sig) && decide (18 ≤ emax)
  | none => decide (20 ≤ f.intBits)

/-- **What an instruction computes from its data operands**, as its PTX section defines it: `move` copies bits (a
register move, a load or store, a shuffle, a matrix load); `logic` is bitwise (and shifts and byte selects); `add` adds,
compares, takes a minimum or maximum or a sum of absolute differences, with no product of two data operands;
`convert` changes one operand's format; `unary` is a fixed function of one operand (MUFU's transcendentals, popc, flo,
brev); `mul` forms a product of two data operands (a multiply-add, a dot product, every MMA); `interp` weights its
operands by fixed or convex low-precision weights (texture filtering). -/
inductive Arith
  | move | logic | add | convert | unary | mul | interp
  deriving DecidableEq, Repr

/-- **Where an instruction executes**: on its warp's lanes, on the uniform datapath (one warp-uniform result), as one
result for the whole warp, on the tensor cores, or on a unit off the issuing lanes (TMA, the bulk engine, L2's
reductions, the texture unit). -/
inductive Exec
  | lane | uniform | warp | tensor | async
  deriving DecidableEq, Repr

/-- **The rate class**: the data-sheet facet a class loads beside issue (`Accounting.TS.Sheet`): the pipes Nsight
Compute names (`alu`, `imad` for the IMAD-class pipe, `xu`, `shfl`), shared memory's banks, the tensor pipe, and `issue`
for a class the sheet prices on issue alone. -/
inductive Pipe
  | issue | alu | imad | xu | shfl | smem | tensor
  deriving DecidableEq, Repr

/-- **A row family**: the classes one PTX section defines, each family with one named premise
(`Models.Accounting.Assumptions.Isa`). `mma` mma.sync (§9.7.16); `wgmma` (§9.7.17); `tcgen05` tcgen05.mma (§9.7.18.10);
`tmem` the tensor-memory loads, stores and copies (§9.7.18.8, §9.7.18.9); `lane` the lane arithmetic (§9.7.1–§9.7.9,
§9.7.20); `uniform` the uniform datapath, which no PTX instruction names (CUDA Binary Utilities, "Uniform Datapath
Instructions"); `memory` ld, st, ldmatrix, stmatrix, cp.async, atom and red (§9.7.10, §9.7.15.5–§9.7.15.6,
§9.7.16.5.15–§9.7.16.5.16); `bulk` the bulk and tensor copies and reductions (§9.7.10.28.4–§9.7.10.28.5); `texture`
tex, tld4 and the surface instructions (§9.7.12, §9.7.13); `warp` shfl.sync, vote, match and redux.sync (§9.7.10.6,
§9.7.15.10–§9.7.15.13). -/
inductive Family
  | mma | wgmma | tcgen05 | tmem | lane | uniform | memory | bulk | texture | warp
  deriving DecidableEq, Repr

/-- **A class's route on the TS menu** (`Accounting.TS.Head`): `traced`, a copy that §1.3's Definition 2 traces
through, on no route; `writer`, a writer kind `T_κ`; `lane`, a vector class, whose heads are `S_V` or table words;
`side`, its own side unit. -/
inductive Route
  | traced | writer | lane | side
  deriving DecidableEq, Repr

/-- **isa_escape.py's verdict classes**, computed from a row (`Row.verdict`): `copy` (MOVER), `select` (a move at a
data-dependent address, lane or selector: a gather), `noMultiply` (NO-MULTIPLY), `lowPrecision` (LOW-PRECISION),
`exact` (one instruction forms exact int8 products into an exact word) and `decompose` (DECOMPOSE: a product, in a
format an int8 product or a word needs several of). isa_escape.py's PRICED and WATCH are routes, not semantics. -/
inductive Verdict
  | copy | select | noMultiply | lowPrecision | exact | decompose
  deriving DecidableEq, Repr

/-- **An instruction class's row.** `fam`: the PTX section that defines it; `exec`: where it executes; `arith`: what it
computes; `dataAddr`: whether it can take a data-dependent address, lane or selector; `a`, `b`, `acc`: its operand and
accumulator formats (`none` where it has none); `words`: the 32-bit words one instruction writes at most, registers and
memory, every lane, at its class's widest form; `pipe`: its rate class; `route`: its route on the menu. -/
structure Row where
  fam : Family
  exec : Exec
  arith : Arith
  dataAddr : Bool
  a : Fmt
  b : Fmt
  acc : Fmt
  words : ℕ
  pipe : Pipe
  route : Route
  deriving DecidableEq, Repr

/-- An MMA form's row: operand formats `a`, `b` and accumulator `acc`, and the accumulator's 32-bit words an issuing
warp instruction writes (mma.sync m16n8 with a 32-bit accumulator: 128; m8n8: 64; m16n8 with an f16 accumulator, two
to a register: 64). -/
def Row.mma (fam : Family) (a b acc : Fmt) (words : ℕ) : Row :=
  { fam, exec := .tensor, arith := .mul, dataAddr := false, a, b, acc, words, pipe := .tensor, route := .writer }

/-- A copy's row: a register move from sources no data selects, traced through (on no route). -/
def Row.copy (fam : Family) (words : ℕ) : Row :=
  { fam, exec := .lane, arith := .move, dataAddr := false, a := .none, b := .none, acc := .none, words,
    pipe := .issue, route := .traced }

/-- A vector class's row: a lane instruction writing at most two words a lane (64 a warp instruction). -/
def Row.lane (fam : Family) (arith : Arith) (dataAddr : Bool) (pipe : Pipe) (a b acc : Fmt := .none) : Row :=
  { fam, exec := .lane, arith, dataAddr, a, b, acc, words := 64, pipe, route := .lane }

/-- A side unit's row: a class priced as its own route, at `words` a instruction. -/
def Row.side (fam : Family) (exec : Exec) (arith : Arith) (dataAddr : Bool) (words : ℕ) (pipe : Pipe := .issue)
    (a b acc : Fmt := .none) : Row :=
  { fam, exec, arith, dataAddr, a, b, acc, words, pipe, route := .side }

/-- **A mover**: its instances copy bits from sources the program can't choose by data, so Definition 2 traces
through them and they head nothing. -/
def Row.mover (r : Row) : Bool := r.arith == .move && !r.dataAddr

/-- isa_escape.py's verdict, computed from the row. -/
def Row.verdict (r : Row) : Verdict :=
  match r.arith with
  | .move => if r.dataAddr then .select else .copy
  | .logic | .add | .convert | .unary => .noMultiply
  | .interp => .lowPrecision
  | .mul => if r.a.holdsS8 && r.b.holdsS8 && r.acc.holdsWord then .exact else .decompose

/-- **The table's discipline**, decided per row: a traced class is a mover; a writer kind is an MMA form or a uniform
class that multiplies; a vector class executes on the lanes and writes at most 64 words a warp instruction (A6's two
a lane); a side unit writes at least one word (so its route isn't free). -/
def Row.ok (r : Row) : Bool :=
  match r.route with
  | .traced => r.mover
  | .writer => (r.exec == .tensor || r.exec == .uniform) && (r.verdict == .exact || r.verdict == .decompose)
  | .lane => r.exec == .lane && decide (r.words ≤ 64)
  | .side => decide (0 < r.words)

/-- **A labelled run's heads by class** (EPS1 §1.3, Definition 2): `on k h`, the owned right fixed committed words whose
head is an instance of class `k` that the labelling puts on route `h`, and `stray k`, those it puts on no route, per
unit of honest work. -/
structure Heads (K T L U : Type) where
  on : K → Head T L U → ℝ
  stray : K → ℝ

variable {K R T L U F X : Type}

/-- **Class `k`'s heads may sit on route `h`** of model `Z` (side units named by `σ`): a writer kind of `k`'s class
when the table routes `k` as a writer, `S_V` or a table route when it routes `k` as a vector class, and `k`'s own side
unit when it routes `k` as a side unit. -/
def Fits (Tb : K → Row) (σ : U → K) (Z : Model K R T L U F X) (k : K) : Head T L U → Prop
  | .writer t => (Tb k).route = .writer ∧ Z.cls t = k
  | .vector => (Tb k).route = .lane
  | .table _ => (Tb k).route = .lane
  | .side u => (Tb k).route = .side ∧ σ u = k

/-- **Model `Z` reads table `Tb`**, its side unit `u` being class `σ u`'s: a side unit's per-word vector is at most
`1/words` of an instruction of its class and nothing on another class, a writer kind's `ω` is at least its class's
words, and every vector class is in `V`. -/
def ReadsTable [DecidableEq K] (Tb : K → Row) (σ : U → K) (Z : Model K R T L U F X) : Prop :=
  (∀ u k, Z.side u k ≤ if k = σ u then 1 / ((Tb (σ u)).words : ℝ) else 0) ∧
    (∀ t, ((Tb (Z.cls t)).words : ℝ) ≤ Z.omega t) ∧ ∀ k, (Tb k).route = .lane → k ∈ Z.vec

/-- **Class `k`'s row holds for the labelled run**: an instance writes at most `words` words, so the words class `k`
heads on route `h` are at most `words` times the class-`k` instructions the labelling gives `h`; and a mover's
instances copy, so (Definition 2) they head nothing. -/
def RowAt (Tb : K → Row) (A : Anatomy K T L U) (H : Heads K T L U) (k : K) : Prop :=
  (∀ h, H.on k h ≤ ((Tb k).words : ℝ) * A.nword h k) ∧ ((Tb k).mover = true → H.stray k = 0 ∧ ∀ h, H.on k h = 0)

/-- **The rows of family `f` hold**: `RowAt` for every class of the family. -/
def RowsOf (f : Family) (Tb : K → Row) (A : Anatomy K T L U) (H : Heads K T L U) : Prop :=
  ∀ k, (Tb k).fam = f → RowAt Tb A H k

end Accounting.Isa
