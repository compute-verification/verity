import Definitions.Pouw.Assumptions.Dimension.H100Floor
import Definitions.Pouw.Basic.Mat
import Mathlib.LinearAlgebra.Matrix.Rank

/-!
# What each `sm_90` instruction class computes, and committing a word every `L` steps (trusted statements; sketch)

**Statement sketches, pending review.**

`MH100w` (`Pouw.Dimension.H100W`) lets every priced instruction write any function of the noise at its price, and its
`wgmma` reads no operands. So its only lower bound is one write per distinct word (`DistinctWritesMH100`), and a word
committed every `L` inner steps is credited 32, not `L`. Here each class's semantics is stated as a predicate on what
it writes, from what it may read:

* **What an instruction may read** (`Readable`, `LaneOf`): an earlier register of the program or a free value (an
  immediate, a constant, a pre-salt or operand word), at most three of them for a per-thread instruction, and for a
  tensor-core instruction one lane per multiplicand, row lanes shared along a tile row and column lanes along a column.
* **Tensor core** (`TcTile`, `TcAllowed`): `wgmma` and `mma.sync` write an `m × n` accumulator tile, element `(r, c)`
  a fixed function of its old value and the `k` products `a r t · b t c`, exact for the integer types, with `k` one of
  the type's documented `k`s (`TcKind.ks`, least `TcKind.minK`).
* **INT32** (`IntSem`): IMAD (low and high word), IDP4A (four int8 products), IADD3, LOP3, PRMT, SHF and POPC (so
  `b1`'s per-thread form, popc of AND, is LOP3 then POPC) of three readable registers.
* **FP32** is already semantic in `MH100` (`fpSem`); `FfmaOne` pins FFMA to one product.
* **FP64** (`Fp64Sem`): a 32-bit half of DFMA's rounded `x·y + z` of three readable register pairs.
* **Other** (`OtherSem`): HFMA2 (two `f16` products), a shuffle (a copy of a readable register), a data-dependent load
  (a word of a table of readable words, indexed by at most `w` bits of one readable register), a reduction or atomic
  (the wrapped sum of two readable words).

`Sm90Sem` asks every instruction of a program to meet its class's predicate after the registers before it. Each
predicate is a rank-1 premise (the ISA as the H200 executes it); its falsification is in the sweep's report.

**What they buy.** The tensor premise alone buys nothing: `.other` still writes any word at 32
(`anyFunctionCeiling`, `tcOnlyCeiling`; the proved facts are in `Proofs.Pouw.Dimension.H100SemProofs`). With every
class semantic, the bilinear accumulator model (`Bam`) is the
reduction's target (`AlgSuffL`, rank 3, the analogue of `A2_algebraicSufficiency`). In it:

* the counting lemma without purity (`BamFloor`, a sketch) gives `48` a word for `N` running words (`128/3` at
  `L = 64`), one and a half registers a word whatever `L`; its arithmetic core is `bamFloorArith`, and the rank fact it
  rests on is `innerProductRank`;
* on cell-pure programs (`CellPure`, rank 4, Theorem K's purity) the count is the full `L` a word, `L/32` registers of
  32 (`PureCount`, a sketch of a short proof from `innerProductRank`);
* on pure lanes (`PureLanes`: no tensor-core lane sums two noise variables) it is the full `L` a word whatever the
  program shares (`CoverCount`, a covering count, sketched);
* without purity the full `L` is `NoSharingGain`, a lower bound on matrix multiplication's cost with merges priced and
  operand ranges enforced: open at `L` 64 and 128, likely false from 256 (small data lets summed lanes fit s8).
-/

namespace Pouw.Dimension.H100

open Finset

variable {G : Type}

/-! ## Register readings -/

/-- A register's exact integer sum, wrapped to a signed 32-bit word. -/
def wrapQ (x : ℚ) : ℚ := (Pouw.wrap32 ⌊x⌋ : ℚ)

/-- A register's 32 bits, as an unsigned word. -/
def u32 (x : ℚ) : ℕ := (⌊x⌋ % 2 ^ 32).toNat

/-- The signed register whose 32 bits are `w`. -/
def ofU32 (w : ℕ) : ℚ := wrapQ w

/-- Byte `t` of a register. -/
def byteOf (x : ℚ) (t : ℕ) : ℕ := u32 x / 2 ^ (8 * t) % 256

/-- Byte `t` of a register, read as a signed int8. -/
def s8Lane (x : ℚ) (t : ℕ) : ℚ := if byteOf x t < 128 then (byteOf x t : ℚ) else (byteOf x t : ℚ) - 256

/-- Nibble `t` of a register, read as a signed int4. -/
def s4Lane (x : ℚ) (t : ℕ) : ℚ :=
  let v := u32 x / 2 ^ (4 * t) % 16
  if v < 8 then (v : ℚ) else (v : ℚ) - 16

/-- Bit `t` of a register, as `0` or `1`. -/
def bitLane (x : ℚ) (t : ℕ) : ℚ := if (u32 x).testBit t then 1 else 0

/-! ## What an instruction may read -/

/-- **What an instruction may read**: an earlier register of the program, or a free value. -/
def Readable (Rs : List (G → ℚ)) (F : Set (G → ℚ)) (h : G → ℚ) : Prop := h ∈ Rs ∨ h ∈ F

/-- A multiplicand lane: lane `j`, by the reading `lane`, of a readable register. -/
def LaneOf (lane : ℚ → ℕ → ℚ) (Rs : List (G → ℚ)) (F : Set (G → ℚ)) (h : G → ℚ) : Prop :=
  ∃ x, Readable Rs F x ∧ ∃ j, h = fun g => lane (x g) j

/-! ## Tensor-core semantics -/

/-- A tensor-core multiplicand type of `sm_90`. -/
inductive TcKind where
  | e4m3 | e5m2 | s8 | s4 | b1 | f16 | bf16 | tf32 | f64
  deriving DecidableEq

/-- The instruction `k`s `sm_90` documents for each type, over `wgmma` and `mma.sync` (PTX ISA): FP8 k32; int8 k16
(`m8n8k16`, `m16n8k16`) and k32; int4 k32 and k64 (`mma.sync` only); `b1` k128 and k256; `f16` and `bf16` k8 and k16;
`tf32` k4 and k8; `f64` k4, k8 and k16 (`mma.sync` only). `f16` `m8n8k4` is not a tensor-core instruction on `sm_90a`:
ptxas lowers it to scalar code, whose instructions are priced and semantic in their own classes. -/
def TcKind.ks : TcKind → List ℕ
  | .e4m3 | .e5m2 => [32]
  | .s8 => [16, 32]
  | .s4 => [32, 64]
  | .b1 => [128, 256]
  | .f16 => [8, 16]
  | .bf16 => [8, 16]
  | .tf32 => [4, 8]
  | .f64 => [4, 8, 16]

/-- The least documented `k` of each type. -/
def TcKind.minK : TcKind → ℕ
  | .e4m3 | .e5m2 => 32
  | .s8 => 16
  | .s4 => 32
  | .b1 => 128
  | .f16 => 8
  | .bf16 => 8
  | .tf32 => 4
  | .f64 => 4

/-- The integer types, whose accumulation is exact (wrapping at 32 bits). -/
def TcKind.exact : TcKind → Bool
  | .s8 | .s4 | .b1 => true
  | _ => false

/-- The type of a `wgmma` shape. -/
def Shape100.kind : Shape100 → TcKind
  | .e4m3F32 | .e4m3F16 => .e4m3
  | .e5m2F32 | .e5m2F16 => .e5m2
  | .s8 => .s8
  | .f16F32 | .f16F16 => .f16
  | .bf16F32 => .bf16
  | .tf32 => .tf32

/-- One tensor-core instruction's semantics at one `k`: how a multiplicand lane is read from a register, and how an
accumulator element is formed from its old value and its `k` products. -/
structure TcSem where
  /-- the instruction's `k` -/
  k : ℕ
  /-- the multiplicand lane reading -/
  lane : ℚ → ℕ → ℚ
  /-- the accumulator element from its old value and the `k` products -/
  acc : ℚ → (Fin k → ℚ) → ℚ

/-- Lane decodings that agree with the ISA on the integer types (the float decodings are the ISA's, a parameter). -/
def DecOK (dec : TcKind → ℚ → ℕ → ℚ) : Prop := dec .s8 = s8Lane ∧ dec .s4 = s4Lane ∧ dec .b1 = bitLane

/-- **The semantics `sm_90` documents for type `κ`** at the decodings `dec`: a documented `k`, the type's lane reading,
and for the integer types exact accumulation wrapped at 32 bits (`b1`'s product of bits is their AND, so the sum is
popc of AND). A float type accumulates by some fixed function of the old value and the `k` products (its rounding). -/
def TcAllowed (dec : TcKind → ℚ → ℕ → ℚ) (κ : TcKind) (σ : TcSem) : Prop :=
  σ.k ∈ κ.ks ∧ σ.lane = dec κ ∧ (κ.exact = true → σ.acc = fun c p => wrapQ (c + ∑ t, p t))

/-- **A tile's semantics** (`wgmma` or `mma.sync`): the instruction writes an `m × n` accumulator tile, row-major in
`outs`. Element `(r, c)` is `acc` of its old value `C r c` (zero or a readable register) and the `k` products
`a r t · b t c`. Row `r`'s multiplicands `a r ·` are shared by the whole row and column `c`'s `b · c` by the whole
column, each a lane of a readable register. -/
def TcTile (σ : TcSem) (Rs : List (G → ℚ)) (F : Set (G → ℚ)) (outs : List (G → ℚ)) : Prop :=
  ∃ (m n : ℕ) (a : Fin m → Fin σ.k → G → ℚ) (b : Fin σ.k → Fin n → G → ℚ) (C : Fin m → Fin n → G → ℚ),
    outs.length = m * n ∧ (∀ r t, LaneOf σ.lane Rs F (a r t)) ∧ (∀ t c, LaneOf σ.lane Rs F (b t c)) ∧
      (∀ r c, C r c = 0 ∨ Readable Rs F (C r c)) ∧
      ∀ (r : Fin m) (c : Fin n),
        outs.getD (r.val * n + c.val) 0 = fun g => σ.acc (C r c g) (fun t => a r t g * b t c g)

/-! ## Per-thread semantics -/

/-- LOP3 with truth table `τ` on 32-bit words (the table indexed by `4a + 2b + c`, bit by bit). -/
def lop3 (τ : Fin 256) (a b c : ℕ) : ℕ :=
  ∑ j ∈ range 32,
    if τ.val.testBit (4 * (if a.testBit j then 1 else 0) + 2 * (if b.testBit j then 1 else 0) +
      (if c.testBit j then 1 else 0)) then 2 ^ j else 0

/-- PRMT: byte `i` of the result is the byte of `{b, a}` that selector nibble `i` names, or its sign replicated. -/
def prmt (a b s : ℕ) : ℕ :=
  ∑ i ∈ range 4,
    let sel := s / 2 ^ (4 * i) % 16
    let src := sel % 8
    let byte := (if src < 4 then a / 2 ^ (8 * src) else b / 2 ^ (8 * (src - 4))) % 256
    (if 8 ≤ sel then (if 128 ≤ byte then 255 else 0) else byte) * 2 ^ (8 * i)

/-- POPC of a 32-bit word. -/
def popc (w : ℕ) : ℕ := ∑ j ∈ range 32, if w.testBit j then 1 else 0

/-- SHF.L: the high word of `{hi, lo} << s` (a funnel shift, `s < 32`). -/
def shfL (lo hi s : ℕ) : ℕ := (hi * 2 ^ 32 + lo) * 2 ^ (s % 32) / 2 ^ 32 % 2 ^ 32

/-- **An INT32-class instruction's semantics** (`.int f`): IMAD (low or high word), IDP4A (four int8 products,
signed or unsigned), IADD3, LOP3, PRMT, SHF or POPC of three readable registers. Integer registers are read by
their integer value. -/
def IntSem (Rs : List (G → ℚ)) (F : Set (G → ℚ)) (f : G → ℚ) : Prop :=
  ∃ x y z, Readable Rs F x ∧ Readable Rs F y ∧ Readable Rs F z ∧
    (f = (fun g => wrapQ (⌊x g⌋ * ⌊y g⌋ + ⌊z g⌋)) ∨
      f = (fun g => wrapQ (((⌊x g⌋ * ⌊y g⌋ + ⌊z g⌋) / 2 ^ 32 : ℤ))) ∨
      f = (fun g => wrapQ (∑ t ∈ range 4, s8Lane (x g) t * s8Lane (y g) t + ⌊z g⌋)) ∨
      f = (fun g => wrapQ (∑ t ∈ range 4, (byteOf (x g) t : ℚ) * byteOf (y g) t + ⌊z g⌋)) ∨
      f = (fun g => wrapQ (⌊x g⌋ + ⌊y g⌋ + ⌊z g⌋)) ∨
      (∃ τ : Fin 256, f = fun g => ofU32 (lop3 τ (u32 (x g)) (u32 (y g)) (u32 (z g)))) ∨
      f = (fun g => ofU32 (prmt (u32 (x g)) (u32 (y g)) (u32 (z g)))) ∨
      f = (fun g => ofU32 (shfL (u32 (x g)) (u32 (y g)) (u32 (z g)))) ∨
      f = (fun g => (popc (u32 (x g)) : ℚ)))

/-- **FFMA is one product** (premise on `MH100`'s FP32 semantics): opcode `κ` is `RN_f32(a·b + c)`, for the
rounding `rn32`. -/
def FfmaOne (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (κ : ℕ) (rn32 : ℚ → ℚ) : Prop :=
  ∀ a b c, fpSem κ a b c = rn32 (a * b + c)

/-- The ISA's FP64 reading: a register pair as a double, the rounding, and a double's 32-bit halves. -/
structure Fp64Isa where
  /-- the double a register pair holds -/
  d64 : ℚ → ℚ → ℚ
  /-- round to nearest double -/
  rn64 : ℚ → ℚ
  /-- the low (`false`) or high (`true`) 32-bit register of a double -/
  half : ℚ → Bool → ℚ

/-- **An FP64-class register's semantics** (`.fp64 f`): a 32-bit half of DFMA's `RN_f64(x·y + z)` (DADD and DMUL
are its special cases) of three readable register pairs. -/
def Fp64Sem (I : Fp64Isa) (Rs : List (G → ℚ)) (F : Set (G → ℚ)) (f : G → ℚ) : Prop :=
  ∃ x₀ x₁ y₀ y₁ z₀ z₁ : G → ℚ, (∀ h ∈ [x₀, x₁, y₀, y₁, z₀, z₁], Readable Rs F h) ∧ ∃ hi : Bool,
    f = fun g => I.half (I.rn64 (I.d64 (x₀ g) (x₁ g) * I.d64 (y₀ g) (y₁ g) + I.d64 (z₀ g) (z₁ g))) hi

/-- The ISA's `f16x2` reading: a register's two halves, the rounding, and the packing. -/
structure HalfIsa where
  /-- half `t < 2` of a register, as an `f16` value -/
  h16 : ℚ → ℕ → ℚ
  /-- round to nearest `f16` -/
  rn16 : ℚ → ℚ
  /-- the register holding two `f16` values -/
  pack : ℚ → ℚ → ℚ

/-- **An `other`-class word's semantics** (`.other f`): HFMA2 (two `f16` products) of three readable registers; a
shuffle, a copy of a readable register; a data-dependent load, a word of a table of readable words indexed by at most
`w` bits of one readable register; or a reduction or atomic, the wrapped sum of two readable words. -/
def OtherSem (H : HalfIsa) (w : ℕ) (Rs : List (G → ℚ)) (F : Set (G → ℚ)) (f : G → ℚ) : Prop :=
  (∃ x y z, Readable Rs F x ∧ Readable Rs F y ∧ Readable Rs F z ∧
      f = fun g => H.pack (H.rn16 (H.h16 (x g) 0 * H.h16 (y g) 0 + H.h16 (z g) 0))
        (H.rn16 (H.h16 (x g) 1 * H.h16 (y g) 1 + H.h16 (z g) 1))) ∨
    (∃ x, Readable Rs F x ∧ f = x) ∨
    (∃ (x : G → ℚ) (tbl : ℕ → G → ℚ), Readable Rs F x ∧ (∀ i < 2 ^ w, Readable Rs F (tbl i)) ∧
      f = fun g => tbl (u32 (x g) % 2 ^ w) g) ∨
    (∃ x y, Readable Rs F x ∧ Readable Rs F y ∧ f = fun g => wrapQ (⌊x g⌋ + ⌊y g⌋))

/-- The ISA's readings the premises take as parameters. -/
structure Sm90Isa where
  /-- each tensor-core type's lane decoding -/
  dec : TcKind → ℚ → ℕ → ℚ
  /-- the FP64 reading -/
  fp64 : Fp64Isa
  /-- the `f16x2` reading -/
  half : HalfIsa
  /-- the widest index a data-dependent load takes, in bits -/
  w : ℕ

/-- **An instruction meets its class's semantics** after the registers `Rs`. `fp32` is already semantic (`fpSem`). -/
def Op100.Sem (I : Sm90Isa) (F : Set (G → ℚ)) (Rs : List (G → ℚ)) : Op100 G → Prop
  | .wgmma s outs => ∃ σ, TcAllowed I.dec s.kind σ ∧ σ.k = s.k ∧ TcTile σ Rs F outs
  | .mma outs _ => ∃ κ σ, TcAllowed I.dec κ σ ∧ TcTile σ Rs F outs
  | .fp32 .. => True
  | .int f => IntSem Rs F f
  | .fp64 f => Fp64Sem I.fp64 Rs F f
  | .other f => OtherSem I.half I.w Rs F f
  | .move _ => True
  | .operand _ => True

/-- **The tensor-core premise alone**: tensor-core instructions meet their semantics, every other class is `MH100w`'s
any function. -/
def Op100.TcOnly (I : Sm90Isa) (F : Set (G → ℚ)) (Rs : List (G → ℚ)) : Op100 G → Prop
  | .wgmma s outs => ∃ σ, TcAllowed I.dec s.kind σ ∧ σ.k = s.k ∧ TcTile σ Rs F outs
  | .mma outs _ => ∃ κ σ, TcAllowed I.dec κ σ ∧ TcTile σ Rs F outs
  | _ => True

/-- **`Sm90Sem`**: every instruction of `P` meets its class's semantics after the registers before it. -/
def Sm90Sem (I : Sm90Isa) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (F : Set (G → ℚ)) (L : List (G → ℚ))
    (P : List (Op100 G)) : Prop :=
  ∀ i (hi : i < P.length), (P.get ⟨i, hi⟩).Sem I F (regs100 fpSem (P.take i) L)

/-- The tensor-core premise alone, for a program. -/
def Sm90TcOnly (I : Sm90Isa) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (F : Set (G → ℚ)) (L : List (G → ℚ))
    (P : List (Op100 G)) : Prop :=
  ∀ i (hi : i < P.length), (P.get ⟨i, hi⟩).TcOnly I F (regs100 fpSem (P.take i) L)

/-! ## The running words at commit interval `L`, on fresh noise -/

/-- The fresh-noise grid of an `m × K` by `K × n` product: one 6-bit noise value per multiplicand. -/
abbrev FreshGrid (m K n : ℕ) := (Fin m → Fin K → Fin 64) × (Fin K → Fin n → Fin 64)

/-- The left multiplicand: int7 data plus noise in `[−32, 31]`. -/
def xN {m K n : ℕ} (a : Fin m → Fin K → ℤ) (r : Fin m) (i : Fin K) : FreshGrid m K n → ℚ :=
  fun g => a r i + ((g.1 r i : ℕ) : ℚ) - 32

/-- The right multiplicand: int7 data plus noise in `[−32, 31]`. -/
def yN {m K n : ℕ} (b : Fin K → Fin n → ℤ) (i : Fin K) (c : Fin n) : FreshGrid m K n → ℚ :=
  fun g => b i c + ((g.2 i c : ℕ) : ℚ) - 32

/-- **The running word** `(r, c, j)` at commit interval `L`: the exact sum of the first `(j + 1)·L` products, wrapped
to 32 bits. -/
def runWordN {m K n : ℕ} (a : Fin m → Fin K → ℤ) (b : Fin K → Fin n → ℤ) (L : ℕ) (r : Fin m) (c : Fin n)
    (j : ℕ) : FreshGrid m K n → ℚ :=
  fun g => wrapQ (∑ i : Fin K with i.val < (j + 1) * L, xN a r i g * yN b i c g)

/-- The committed words: every running word of the tile at interval `L`. -/
def runWordsN {m K n : ℕ} (a : Fin m → Fin K → ℤ) (b : Fin K → Fin n → ℤ) (L : ℕ) :
    Finset (FreshGrid m K n → ℚ) :=
  (univ ×ˢ univ ×ˢ range (K / L)).image fun p : Fin m × Fin n × ℕ => runWordN a b L p.1 p.2.1 p.2.2

/-- Int7 data: every entry in `[−63, 63]`. -/
def Int7 {m K : ℕ} (a : Fin m → Fin K → ℤ) : Prop := ∀ r i, -63 ≤ a r i ∧ a r i ≤ 63

/-- The salt-independent functions: those constant on the grid. -/
def SaltFree (G : Type) : Set (G → ℚ) := {f | ∀ g g', f g = f g'}

/-! ## The bilinear accumulator model -/

/-- A register of the bilinear accumulator model: a bilinear form in the left coordinates `Option α` and the right
coordinates `Option β`, where `none` is the constant `1` and `some i` a noise variable. -/
abbrev BForm (α β : Type) := Matrix (Option α) (Option β) ℚ

/-- A tensor-core type of the model: its `k`, the exact range of a lane, and its price per accumulator register. -/
structure BamType where
  /-- the instruction's `k` -/
  k : ℕ
  /-- the least lane value held exactly -/
  lo : ℤ
  /-- the greatest lane value held exactly -/
  hi : ℤ
  /-- the price per accumulator register written -/
  price : ℕ

/-- A lane: an integer linear form in the constant and the noise variables. -/
abbrev Lane (α : Type) := Option α → ℤ

/-- The lane's value at noise `e`. -/
def Lane.eval {α : Type} [Fintype α] (u : Lane α) (e : α → ℤ) : ℤ := u none + ∑ i, u (some i) * e i

/-- The lane fits the type on noise in `[nlo, nhi]`: on every such noise value it lies in `[lo, hi]`. -/
def Lane.FitsOn {α : Type} [Fintype α] (nlo nhi : ℤ) (τ : BamType) (u : Lane α) : Prop :=
  ∀ e : α → ℤ, (∀ i, nlo ≤ e i ∧ e i ≤ nhi) → τ.lo ≤ u.eval e ∧ u.eval e ≤ τ.hi

/-- The lane fits the type: on every noise value in `[−32, 31]` it lies in `[lo, hi]`. -/
def Lane.Fits {α : Type} [Fintype α] (τ : BamType) (u : Lane α) : Prop := u.FitsOn (-32) 31 τ

/-- A lane as a coordinate vector. -/
def Lane.vec {α : Type} (u : Lane α) : Option α → ℚ := fun i => (u i : ℚ)

/-- An instruction of the bilinear accumulator model. -/
inductive BamOp (α β : Type) where
  /-- a tensor-core accumulator register of type `τ`: register `src` plus `τ.k` products of lanes -/
  | tc (τ : BamType) (src : ℕ) (u : Fin τ.k → Lane α) (v : Fin τ.k → Lane β)
  /-- a priced combination of at most three registers (IADD3, IMAD by a constant, FFMA, HFMA2) -/
  | merge (c : List (ℚ × ℕ))
  /-- a free copy of a register -/
  | copy (i : ℕ)
  /-- a free value: a salt-independent constant, the form `q` at `(none, none)` -/
  | const (q : ℚ)

/-- The register an instruction writes after the registers `Rs`. -/
def BamOp.out {α β : Type} [Fintype α] [Fintype β] [DecidableEq α] [DecidableEq β] (Rs : List (BForm α β)) :
    BamOp α β → BForm α β
  | .tc τ src u v => Rs.getD src 0 + ∑ t : Fin τ.k, Matrix.vecMulVec (u t).vec (v t).vec
  | .merge c => (c.map fun p => p.1 • Rs.getD p.2 0).sum
  | .copy i => Rs.getD i 0
  | .const q => Matrix.of fun i j => if i = none ∧ j = none then q else 0

/-- An instruction's price. -/
def BamOp.price {α β : Type} : BamOp α β → ℕ
  | .tc τ _ _ _ => τ.price
  | .merge _ => 32
  | .copy _ => 0
  | .const _ => 0

/-- Well-formed at the types `Ts`: a tensor-core register is of a listed type and its lanes fit it, and a merge reads
at most three registers. -/
def BamOp.WF {α β : Type} [Fintype α] [Fintype β] (Ts : List BamType) : BamOp α β → Prop
  | .tc τ _ u v => τ ∈ Ts ∧ (∀ t, (u t).Fits τ) ∧ ∀ t, (v t).Fits τ
  | .merge c => c.length ≤ 3
  | _ => True

/-- Well-formed at the types `Ts` on noise in `[nlo, nhi]`. -/
def BamOp.WFOn {α β : Type} [Fintype α] [Fintype β] (nlo nhi : ℤ) (Ts : List BamType) : BamOp α β → Prop
  | .tc τ _ u v => τ ∈ Ts ∧ (∀ t, (u t).FitsOn nlo nhi τ) ∧ ∀ t, (v t).FitsOn nlo nhi τ
  | .merge c => c.length ≤ 3
  | _ => True

/-- A program's registers. -/
def bamRegs {α β : Type} [Fintype α] [Fintype β] [DecidableEq α] [DecidableEq β] (P : List (BamOp α β)) :
    List (BForm α β) :=
  P.foldl (fun Rs o => Rs ++ [o.out Rs]) []

/-- A program's cost. -/
def bamCost {α β : Type} (P : List (BamOp α β)) : ℕ := (P.map BamOp.price).sum

/-- The H200's tensor-core types at the floor table (`h100Floor`): one accumulator register of every `wgmma` shape
and `mma.sync` form at 32, with each type's exact integer run (D4's caps): int8 k32 on `[−128, 127]`, `e4m3` k32 on
`[−16, 16]`, `e5m2` k32 on `[−8, 8]`, `f16` k16 and `tf32` k8 on `[−2048, 2048]`, `bf16` k16 on `[−256, 256]`, int4
k64 on `[−8, 7]`, `b1` k256 on `[0, 1]`. -/
def h200Types : List BamType :=
  [⟨32, -128, 127, 32⟩, ⟨32, -16, 16, 32⟩, ⟨32, -8, 8, 32⟩, ⟨16, -2048, 2048, 32⟩, ⟨8, -2048, 2048, 32⟩,
    ⟨16, -256, 256, 32⟩, ⟨64, -8, 7, 32⟩, ⟨256, 0, 1, 32⟩]

/-- The bilinear form of the running word `(r, c, j)` at interval `L`, over data `a`, `b` and the centered noise in
`[−32, 31]` (`xN` and `yN` less their data): the constant-constant entry is `Σ a·b`, the linear entries the data, and
noise pair `((r, i), (i, c))` has coefficient 1, for the first `(j + 1)·L` inner indices. -/
def runForm {m K n : ℕ} (a : Fin m → Fin K → ℤ) (b : Fin K → Fin n → ℤ) (L : ℕ) (r : Fin m) (c : Fin n)
    (j : ℕ) : BForm (Fin m × Fin K) (Fin K × Fin n) :=
  Matrix.of fun p q =>
    match p, q with
    | none, none => ∑ i : Fin K with i.val < (j + 1) * L, ((a r i * b i c : ℤ) : ℚ)
    | some (r', i), none => if r' = r ∧ i.val < (j + 1) * L then (b i c : ℚ) else 0
    | none, some (i, c') => if c' = c ∧ i.val < (j + 1) * L then (a r i : ℚ) else 0
    | some (r', i), some (i', c') => if r' = r ∧ c' = c ∧ i = i' ∧ i.val < (j + 1) * L then 1 else 0

/-- The running words' forms. -/
def runForms {m K n : ℕ} (a : Fin m → Fin K → ℤ) (b : Fin K → Fin n → ℤ) (L : ℕ) :
    Finset (BForm (Fin m × Fin K) (Fin K × Fin n)) :=
  (univ ×ˢ univ ×ˢ range (K / L)).image fun p : Fin m × Fin n × ℕ => runForm a b L p.1 p.2.1 p.2.2

/-! ## Pinned statements (sketch, pending review) -/

/-- `BamFloor`'s rate a word: `128/3` at `L = 64`, where two words may share a pre-image, and `48` above. -/
def bamFloorRate (L : ℕ) : ℚ := if L = 64 then 128 / 3 else 48

/-- **The counting lemma without purity** (sketch; rank 2, a combinatorial fact about the model, not proved): for
`32 ∣ L`, `L ∣ K` and `L ≥ 64`, every well-formed program at the H200's types whose registers include the forms of all
`N = m·n·(K/L)` running words of int7 data costs at least `N · bamFloorRate L`: `48` a word, `128/3` at `L = 64`.

The count is of distinct non-constant forms, each first written by a priced instruction (32). The `N` words are
linearly independent, and so are their noise parts. No lane of a noise variable holds more than 32 products (`b1`'s
and int4's lanes can't carry one, since a noise variable spans 64 values), and two distinct words' noise parts differ
by rank at least `L` (`innerProductRank`). So a word written by a tensor-core register has a pre-image (its `src`)
that is neither a word nor a constant; for `L > 64` no two words share one, and at `L = 64` at most two do. A word
written by a merge is `w = Σ (earlier words) + Σ (other registers read)`; ordered by write time these combinations
are triangular in the words, so the non-word registers the `N − p` merges read span at least `N − p` dimensions.
Pre-images and merge inputs may coincide, so with `p` words of the first kind there are at least `N + max(p, N − p)`
forms (`N + max(p/2, N − p)` at `L = 64`), least at `p = N/2` (`p = 2N/3`): `bamFloorArith`. The rank fact it rests on
is `innerProductRank`. -/
def BamFloor (L : ℕ) : Prop :=
  32 ∣ L → 64 ≤ L →
    ∀ (m K n : ℕ) (a : Fin m → Fin K → ℤ) (b : Fin K → Fin n → ℤ)
      (P : List (BamOp (Fin m × Fin K) (Fin K × Fin n))),
      L ∣ K → Int7 a → Int7 b → (∀ o ∈ P, o.WF h200Types) → (∀ f ∈ runForms a b L, f ∈ bamRegs P) →
        (m * n * (K / L) : ℚ) * bamFloorRate L ≤ bamCost P

/-- The registers an instruction reads. -/
def BamOp.reads {α β : Type} : BamOp α β → List ℕ
  | .tc _ src _ _ => [src]
  | .merge c => c.map Prod.snd
  | .copy i => [i]
  | .const _ => []

/-- A tensor-core instruction. -/
def BamOp.isTc {α β : Type} : BamOp α β → Bool
  | .tc .. => true
  | _ => false

/-- Register `s` is in the cone of register `t` of `P` (register `t` is instruction `t`'s): `t` itself, or read by a
register of the cone. -/
inductive InCone {α β : Type} (P : List (BamOp α β)) (t : ℕ) : ℕ → Prop
  | self : InCone P t t
  | read {u s : ℕ} (hu : InCone P t u) (h : u < P.length) (hs : s ∈ (P.get ⟨u, h⟩).reads) : InCone P t s

/-- **`CellPure P idx`** (rank 4, a restriction on the adversary's programs that no check observes; the analogue of
Theorem K's row and charging purity): no tensor-core register is in the cones of two cells' words, where word `p`
(cell `(p.1, p.2.1)`, chunk `p.2.2`) is register `idx p`. Prefix chains within a cell, merges and copies are free. -/
def CellPure {α β : Type} {m n : ℕ} (P : List (BamOp α β)) (J : ℕ) (idx : Fin m × Fin n × ℕ → ℕ) : Prop :=
  ∀ τ (hτ : τ < P.length), (P.get ⟨τ, hτ⟩).isTc = true →
    ∀ p q : Fin m × Fin n × ℕ, p.2.2 < J → q.2.2 < J → InCone P (idx p) τ → InCone P (idx q) τ →
      p.1 = q.1 ∧ p.2.1 = q.2.1

/-- **The count on cell-pure programs** (sketch of a theorem; rank 0 once proved): for `L ∣ K`, every well-formed
program at the H200's types whose register `idx p` is the form of running word `p`, and which is `CellPure`, costs at
least `N·L`, the honest `L/32` registers of 32 a word, for every data.

The proof: a cell's last word has noise part of rank `K` (`innerProductRank`); it is a combination of the products of
the tensor-core registers in its cone, at most 32 to a register; and cell purity makes the cones of distinct cells'
last words share no tensor-core register. So each cell has `K/32` registers of 32, `mnK = N·L` in all. -/
def PureCount (L : ℕ) : Prop :=
  ∀ (m K n : ℕ) (a : Fin m → Fin K → ℤ) (b : Fin K → Fin n → ℤ)
    (P : List (BamOp (Fin m × Fin K) (Fin K × Fin n))) (idx : Fin m × Fin n × ℕ → ℕ),
    L ∣ K → (∀ o ∈ P, o.WF h200Types) →
      (∀ p : Fin m × Fin n × ℕ, p.2.2 < K / L →
        idx p < P.length ∧ (bamRegs P).getD (idx p) 0 = runForm a b L p.1 p.2.1 p.2.2) →
      CellPure P (K / L) idx → (m * n * (K / L) : ℚ) * L ≤ bamCost P

/-- A lane that carries at most one noise variable. -/
def Lane.Pure {α : Type} (u : Lane α) : Prop := ∀ i j, u (some i) ≠ 0 → u (some j) ≠ 0 → i = j

/-- **`PureLanes P`**: every tensor-core lane of `P` carries at most one noise variable (no summed lanes). A lane that
fits a type on noise in `[nlo, nhi]` is pure once the type's range is narrower than `2·(nhi − nlo)`
(`Lane.pure_of_fitsOn`): an s8 lane on noise spanning 129 values (`[−64, 64]` beside int7 data), but no `f16`, `bf16` or
`tf32` lane on any noise an s8 operand can carry. At this file's 64-value noise it is a restriction on the adversary
(rank 4), since an s8 lane holds up to four noise variables there and an `f16` lane many. -/
def PureLanes {α β : Type} (P : List (BamOp α β)) : Prop :=
  ∀ o ∈ P, match o with
    | .tc _ _ u v => (∀ t, (u t).Pure) ∧ ∀ t, (v t).Pure
    | _ => True

/-- **The count on pure lanes** (sketch of a theorem; rank 0 once proved): every well-formed program at the H200's
types with pure lanes whose registers include every running word's form costs at least `N·L`, whatever it shares
across cells, for every data.

The proof is a covering count. On pure lanes a product's noise part is one monomial `e_α f_β`, a merge or copy adds
no monomial, and a constant has none, so every monomial of every register is a product of some tensor-core register.
The cells' last words hold all `m·n·K` monomials `e_(r,i) f_(i,c)`. A noise-carrying type has `k ≤ 32` (a lane with a
noise variable spans at least 64 values, so `e4m3`, `e5m2`, int4 and `b1` lanes carry none), so it takes `m·n·K/32`
registers of 32. It is KSW-K's one fresh monomial per honest MAC turned into a bound: summed lanes are what break it. -/
def CoverCount (L : ℕ) : Prop :=
  ∀ (m K n : ℕ) (a : Fin m → Fin K → ℤ) (b : Fin K → Fin n → ℤ)
    (P : List (BamOp (Fin m × Fin K) (Fin K × Fin n))),
    L ∣ K → (∀ o ∈ P, o.WF h200Types) → PureLanes P → (∀ f ∈ runForms a b L, f ∈ bamRegs P) →
      (m * n * (K / L) : ℚ) * L ≤ bamCost P

/-- **`NoSharingGain L`** (open at `L` 64 and 128; rank 4 to 5): the same programs cost at least `N·L`, the honest
`L/32` registers of 32 a word. It is a lower bound on matrix multiplication's cost with merges priced and operand
ranges enforced, over every int7 data. `PureCount` gives it on the programs `CellPure` admits and `CoverCount` on
those `PureLanes` admits; the open part is the programs that share products across cells through summed lanes.

The ranges block a summed lane only where the data are large: two entries' sum fits s8 when the two data values sum
to at most 64 in magnitude, and with zero data four entries' sum fits. So it is likely false from `L = 256` (an
estimate): one Strassen–Winograd level costs `7L/8` of the tensor work plus about one merge a word on running words,
`L` at `L = 256`; Hopcroft–Kerr's `⟨p, 2, n⟩` about `3L/4` plus `m` merges, below `L` once `L > 128·m`. D4's proved
`NaiveBeatsStrassen` covers full-range data only. -/
def NoSharingGain (L : ℕ) : Prop :=
  32 ∣ L → 64 ≤ L →
    ∀ (m K n : ℕ) (a : Fin m → Fin K → ℤ) (b : Fin K → Fin n → ℤ)
      (P : List (BamOp (Fin m × Fin K) (Fin K × Fin n))),
      L ∣ K → Int7 a → Int7 b → (∀ o ∈ P, o.WF h200Types) → (∀ f ∈ runForms a b L, f ∈ bamRegs P) →
        (m * n * (K / L) : ℚ) * L ≤ bamCost P

/-- **Algebraic sufficiency for the running words** (rank 3, the analogue of `A2_algebraicSufficiency`): every
`MH100w` program at `h100Floor` that meets `Sm90Sem` and outputs the running words on fresh noise costs at least the
least well-formed bilinear accumulator program that outputs their forms. It is what rules out nonlinear lanes (bit
planes, nibbles, lookups) beating bilinear ones. -/
def AlgSuffL (L : ℕ) : Prop :=
  ∀ (I : Sm90Isa), DecOK I.dec →
    ∀ (m K n : ℕ) (a : Fin m → Fin K → ℤ) (b : Fin K → Fin n → ℤ) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ)
      (L₀ : List (FreshGrid m K n → ℚ)) (P : List (Op100 (FreshGrid m K n))),
      L ∣ K → Int7 a → Int7 b → (∀ f ∈ L₀, f ∈ SaltFree _) → (∀ o ∈ P, o.WFw h100Floor (SaltFree _)) →
        Sm90Sem I fpSem (SaltFree _) L₀ P → (∀ t ∈ runWordsN a b L, t ∈ regs100 fpSem P L₀) →
          ∃ Q : List (BamOp (Fin m × Fin K) (Fin K × Fin n)),
            (∀ o ∈ Q, o.WF h200Types) ∧ (∀ f ∈ runForms a b L, f ∈ bamRegs Q) ∧ bamCost Q ≤ cost100 h100Floor P

/-- **The work bound at commit interval `L`** (sketch): under the semantic premises, every `MH100w` program at
`h100Floor` that outputs the running words on fresh noise costs at least `c` a word. `AlgSuffL L ∧ BamFloor L` gives
it at `c = bamFloorRate L`, and `AlgSuffL L ∧ NoSharingGain L` at `c = L` (on cell-pure programs `PureCount L`, and
on pure lanes `CoverCount L`, in place of `NoSharingGain L`). -/
def LCommitBound (L : ℕ) (c : ℚ) : Prop :=
  ∀ (I : Sm90Isa), DecOK I.dec →
    ∀ (m K n : ℕ) (a : Fin m → Fin K → ℤ) (b : Fin K → Fin n → ℤ) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ)
      (L₀ : List (FreshGrid m K n → ℚ)) (P : List (Op100 (FreshGrid m K n))),
      L ∣ K → Int7 a → Int7 b → (∀ f ∈ L₀, f ∈ SaltFree _) → (∀ o ∈ P, o.WFw h100Floor (SaltFree _)) →
        Sm90Sem I fpSem (SaltFree _) L₀ P → (∀ t ∈ runWordsN a b L, t ∈ regs100 fpSem P L₀) →
          (m * n * (K / L) : ℚ) * c ≤ cost100 h100Floor P

end Pouw.Dimension.H100
