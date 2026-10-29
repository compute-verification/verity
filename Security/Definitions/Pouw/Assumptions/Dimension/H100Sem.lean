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
(`anyFunctionCeiling`, `tcOnlyCeiling`). With every class semantic, the bilinear accumulator model (`Bam`) is the
reduction's target (`AlgSuffL`, rank 3, the analogue of `A2_algebraicSufficiency`). In it the counting lemma
(`BamFloor`) gives `N·(64 − 1024/L)` for `N` running words, at most two registers a word whatever `L`; its arithmetic
core is `lcommitFloorArith`, and the rank fact it rests on is `innerProductRank`. The full `L` a word, `L/32` registers
of 32, is `NoSharingGain`: a lower bound on matrix multiplication's cost with merges and operand ranges, open (rank 4
to 5). Without operand ranges it is false: one level of Strassen–Winograd costs `7L/8` of the tensor work plus at
most one merge a word.
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

/-- The lane fits the type: on every noise value in `[−32, 31]` it lies in `[lo, hi]`. -/
def Lane.Fits {α : Type} [Fintype α] (τ : BamType) (u : Lane α) : Prop :=
  ∀ e : α → ℤ, (∀ i, -32 ≤ e i ∧ e i ≤ 31) → τ.lo ≤ u.eval e ∧ u.eval e ≤ τ.hi

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

/-- The bilinear form of the running word `(r, c, j)` at interval `L`, over data `a`, `b`: the constant-constant entry
is `Σ a·b`, the linear entries the data, and noise pair `((r, i), (i, c))` has coefficient 1, for the first
`(j + 1)·L` inner indices. -/
def runForm {m K n : ℕ} (a : Fin m → Fin K → ℤ) (b : Fin K → Fin n → ℤ) (L : ℕ) (r : Fin m) (c : Fin n)
    (j : ℕ) : BForm (Fin m × Fin K) (Fin K × Fin n) :=
  Matrix.of fun p q =>
    match p, q with
    | none, none => ∑ i : Fin K with i.val < (j + 1) * L, ((a r i - 32) * (b i c - 32) : ℚ)
    | some (r', i), none => if r' = r ∧ i.val < (j + 1) * L then (b i c - 32 : ℚ) else 0
    | none, some (i, c') => if c' = c ∧ i.val < (j + 1) * L then (a r i - 32 : ℚ) else 0
    | some (r', i), some (i', c') => if r' = r ∧ c' = c ∧ i = i' ∧ i.val < (j + 1) * L then 1 else 0

/-- The running words' forms. -/
def runForms {m K n : ℕ} (a : Fin m → Fin K → ℤ) (b : Fin K → Fin n → ℤ) (L : ℕ) :
    Finset (BForm (Fin m × Fin K) (Fin K × Fin n)) :=
  (univ ×ˢ univ ×ˢ range (K / L)).image fun p : Fin m × Fin n × ℕ => runForm a b L p.1 p.2.1 p.2.2

/-! ## Pinned statements (sketch, pending review) -/

/-- **The counting lemma in the bilinear accumulator model** (sketch; rank 2, a combinatorial fact about the model):
for `32 ∣ L`, `L ∣ K` and `L ≥ 64`, every well-formed program at the H200's types whose registers include the forms of
all `N = m·n·(K/L)` running words of int7 data costs at least `N·(64 − 1024/L)`, that is
`32·N·(2 − 32/L)`.

The count: every form is written (32 each, distinct writes); a word whose cone has no merge is a chain of tensor-core
registers whose noise part differs from every other register's by rank at least `L` (`innerProductRank`), so its
chain has `L/32` registers of its own, since no lane of a noise variable holds more than 32 products (`b1`'s and int4's
lanes can't carry a noise variable, which spans 64 values); every other word pays a merge. With `p` words of the first
kind, the cost is at least `32·max(p·L/32, N) + 32·(N − p)`, least at `p = 32N/L` (`lcommitFloorArith`). -/
def BamFloor (L : ℕ) : Prop :=
  32 ∣ L → 64 ≤ L →
    ∀ (m K n : ℕ) (a : Fin m → Fin K → ℤ) (b : Fin K → Fin n → ℤ)
      (P : List (BamOp (Fin m × Fin K) (Fin K × Fin n))),
      L ∣ K → Int7 a → Int7 b → (∀ o ∈ P, o.WF h200Types) → (∀ f ∈ runForms a b L, f ∈ bamRegs P) →
        (m * n * (K / L) : ℚ) * (64 - 1024 / L) ≤ bamCost P

/-- **`NoSharingGain L`** (open; rank 4 to 5): the same programs cost at least `N·L`, the honest `L/32` registers of 32
a word. It is a lower bound on matrix multiplication's cost with merges priced and operand ranges enforced.
Without the ranges it is false: one Strassen–Winograd level costs `7L/8` of the tensor work plus `m` merges a word
(`m` from 1/2 to 1 by schedule), below `L` once `L > 256·m`. With them, D4's proved `NaiveBeatsStrassen` is the evidence for Strassen's
and Winograd's schemes at full int8 range, and its range argument extends to NCP's operands (190 values a block-1
entry). -/
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
it at `c = 64 − 1024/L`, and `AlgSuffL L ∧ NoSharingGain L` at `c = L`. -/
def LCommitBound (L : ℕ) (c : ℚ) : Prop :=
  ∀ (I : Sm90Isa), DecOK I.dec →
    ∀ (m K n : ℕ) (a : Fin m → Fin K → ℤ) (b : Fin K → Fin n → ℤ) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ)
      (L₀ : List (FreshGrid m K n → ℚ)) (P : List (Op100 (FreshGrid m K n))),
      L ∣ K → Int7 a → Int7 b → (∀ f ∈ L₀, f ∈ SaltFree _) → (∀ o ∈ P, o.WFw h100Floor (SaltFree _)) →
        Sm90Sem I fpSem (SaltFree _) L₀ P → (∀ t ∈ runWordsN a b L, t ∈ regs100 fpSem P L₀) →
          (m * n * (K / L) : ℚ) * c ≤ cost100 h100Floor P

/-! ## Proved facts -/

theorem TcKind.minK_mem (κ : TcKind) : κ.minK ∈ κ.ks := by cases κ <;> decide

theorem TcKind.minK_le (κ : TcKind) : ∀ k ∈ κ.ks, κ.minK ≤ k := by cases κ <;> decide

theorem regs100_map_other (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (l L : List (G → ℚ)) :
    regs100 fpSem (l.map Op100.other) L = L ++ l := by
  induction l generalizing L with
  | nil => simp [regs100]
  | cons f l ih =>
    have h := ih (L ++ [f])
    simp only [regs100, List.map_cons, List.foldl_cons, Op100.outs] at h ⊢
    rw [h, List.append_assoc, List.singleton_append]

theorem cost100_map_other (pr : Prices100) (l : List (G → ℚ)) :
    cost100 pr (l.map Op100.other) = pr.other * l.length := by
  induction l with
  | nil => simp [cost100]
  | cons f l ih =>
    simp only [cost100, List.map_map, List.map_cons, List.sum_cons, List.length_cons] at ih ⊢
    rw [ih]; simp [Op100.price]; ring

/-- **`MH100w` credits a word 32, whatever `L`**: for every finite set of targets, writing each with one `other`
instruction is a well-formed program at `h100Floor` that outputs them all at exactly `32·|T|`. -/
theorem anyFunctionCeiling (F : Set (G → ℚ)) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (T : Finset (G → ℚ))
    (L : List (G → ℚ)) :
    ∃ P : List (Op100 G), (∀ o ∈ P, o.WFw h100Floor F) ∧ (∀ t ∈ T, t ∈ regs100 fpSem P L) ∧
      cost100 h100Floor P = 32 * T.card := by
  refine ⟨T.toList.map Op100.other, ?_, ?_, ?_⟩
  · intro o ho
    obtain ⟨f, -, rfl⟩ := List.mem_map.1 ho
    trivial
  · intro t ht
    rw [regs100_map_other]
    exact List.mem_append_right _ (Finset.mem_toList.2 ht)
  · rw [cost100_map_other, Finset.length_toList]
    rfl

/-- **The tensor-core premise alone buys nothing**: the same program meets it (it has no tensor-core instruction). -/
theorem tcOnlyCeiling (I : Sm90Isa) (F : Set (G → ℚ)) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (T : Finset (G → ℚ))
    (L : List (G → ℚ)) :
    ∃ P : List (Op100 G), (∀ o ∈ P, o.WFw h100Floor F) ∧ Sm90TcOnly I fpSem F L P ∧
      (∀ t ∈ T, t ∈ regs100 fpSem P L) ∧ cost100 h100Floor P = 32 * T.card := by
  obtain ⟨P, hwf, hout, hcost⟩ := anyFunctionCeiling F fpSem T L
  refine ⟨T.toList.map Op100.other, ?_, ?_, ?_, ?_⟩
  · intro o ho
    obtain ⟨f, -, rfl⟩ := List.mem_map.1 ho
    trivial
  · intro i hi
    simp [Op100.TcOnly]
  · intro t ht
    rw [regs100_map_other]
    exact List.mem_append_right _ (Finset.mem_toList.2 ht)
  · rw [cost100_map_other, Finset.length_toList]
    rfl

/-- **The rank fact**: if `r` rank-one terms `α p ⊗ β p` sum to the identity on `L` coordinates, then `L ≤ r`. So an
`L`-term inner product of fresh variables needs `L` products, at most 32 to an accumulator register. -/
theorem innerProductRank (L r : ℕ) (α β : Fin r → Fin L → ℚ)
    (h : ∀ k l, ∑ p, α p k * β p l = if k = l then 1 else 0) : L ≤ r := by
  let A : Matrix (Fin L) (Fin r) ℚ := Matrix.of fun k p => α p k
  let B : Matrix (Fin r) (Fin L) ℚ := Matrix.of fun p l => β p l
  have hAB : A * B = 1 := by
    ext k l
    simp only [A, B, Matrix.mul_apply, Matrix.of_apply, Matrix.one_apply]
    exact h k l
  have h1 : (1 : Matrix (Fin L) (Fin L) ℚ).rank = L := by simp
  have h2 : (A * B).rank ≤ A.rank := Matrix.rank_mul_le_left A B
  have h3 : A.rank ≤ r := by simpa using Matrix.rank_le_width A
  rw [hAB, h1] at h2
  omega

/-- **The counting lemma's arithmetic**: with `p ≤ N` words of the chain kind at `L/32` registers each and every other
word paying a merge, `32·max(p·L/32, N) + 32·(N − p) ≥ N·(64 − 1024/L)`. -/
theorem lcommitFloorArith (L N p : ℚ) (hL : 32 ≤ L) (hp0 : 0 ≤ p) (hpN : p ≤ N) :
    N * (64 - 1024 / L) ≤ 32 * max (p * L / 32) N + 32 * (N - p) := by
  have hL0 : 0 < L := by linarith
  rcases le_total (p * L / 32) N with h | h
  · rw [max_eq_right h]
    have : p ≤ 32 * N / L := by
      rw [le_div_iff₀ hL0]; linarith
    have e : N * (64 - 1024 / L) = 64 * N - 32 * (32 * N / L) := by field_simp; ring
    rw [e]; linarith
  · rw [max_eq_left h]
    have hN : 0 ≤ N := le_trans hp0 hpN
    have e : N * (64 - 1024 / L) = 64 * N - 1024 * N / L := by field_simp
    have h' : 32 * N ≤ p * L := by linarith
    have hq : 1024 * N / L ≤ 32 * p := by
      rw [div_le_iff₀ hL0]; nlinarith
    have : 32 * N + p * L - 32 * p ≥ 64 * N - 1024 * N / L := by
      have : p * L - 32 * p ≥ 32 * N - 1024 * N / L := by
        have h4 : (p - 32 * N / L) * (L - 32) ≥ 0 := by
          apply mul_nonneg
          · rw [sub_nonneg, div_le_iff₀ hL0]; linarith
          · linarith
        have e2 : (p - 32 * N / L) * (L - 32) = p * L - 32 * p - 32 * N + 1024 * N / L := by
          field_simp; ring
        linarith
      linarith
    linarith

end Pouw.Dimension.H100
