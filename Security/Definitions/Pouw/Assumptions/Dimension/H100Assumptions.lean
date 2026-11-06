import Definitions.Pouw.Dimension.Defs

/-!
# `MH100`'s price table and the hypothesis `H32` (draft semantics)

**Draft semantics, pending review**.

`MH100` (`Pouw.Dimension.H100`) is the H100 (`sm_90`) analogue of `M4090`. Its prices are a parameter `Prices100`, in
units of one dense int8 MAC at the datasheet's 4,096 MACs per SM per clock, so that a measurement can replace any
entry.

* **`wgmma` shapes** (`Shape100`) at their least `k` (one K step): a register's price is
  `k · (accumulator elements per 32-bit register) · (price per MAC)`, the price per MAC being `Prices100.mac`.
* **Every other tensor-core instruction** (`mma.sync`, legacy shapes, sparse `wgmma`, DMMA, b1): its full measured
  price, well-formed at `Prices100.tc` or more per accumulator register written.
* **Per register**: `ffma` (FFMA, FADD, FMUL), `int32`, `fp64` (per 32-bit register of an FP64 result) and `other`
  (every other writer of a 32-bit word, data-dependent accesses and sub-word operations included). A word that a
  memory-side instruction computes counts as a register: reductions and atomics (`red`, `atom`,
  `cp.reduce.async.bulk`) are `other`, priced per word they write. A plain load or store of a whole 32-bit word at an
  oblivious address is a `move`, free: the stored word is a copy of its register, so the honest trace's stores cost
  nothing and add no target.

`h100Nominal` is the table of `genuine-fp8-interface.md` §0, the datasheet's, unmeasured: FP8 = int8 = 1 per MAC,
FP16 and BF16 2, TF32 4; FFMA 32, INT32 64, FP64 32 per register. The datasheet does not price `mma.sync` or the
`other` class; `h100Nominal` sets both to 32, the least value `H32` allows. At the `wgmma` per-MAC rates, `mma.sync`
m16n8k8 FP16 or BF16 with FP32 accumulate, m16n8k4 TF32, and m16n8k16 or m8n8k16 int8 would cost 16 per register, and
legacy m8n8k4 FP16 with FP32 accumulate 8. So `H32` asserts that `sm_90` runs those shapes at no more than half, or a
quarter, of the `wgmma` rate per MAC.

* **`Floor c pr`** (named hypothesis): every register writer costs at least `c` per register written. A measured table
  may be in any unit (every price scaled alike), with `c` its least price per register.
* **`H32 pr`** (named hypothesis, by analogy with the 4090's H_16): `Floor 32 pr`. It is to be measured on an H100; its
  satisfiability witness is `H32Sat` (`h100Nominal`). At the nominal prices it has no slack: the single-register
  writers and the k-minimal `wgmma` shapes cost exactly 32. Round 7's measured table meets it (`h32Measured`). So do
  Round 8's loop-free table (`h32LoopFree`) and the floor table (`h32Floor`), with no slack. It is a floor per
  instruction class: co-issued mixes write a register as often as every 21.45 units loop-free, and 21.33 at full
  dispatch (`H32`'s docstring), so the bounds are additive-W1 work, not time.
* **`FloorTc a c pr`** (named hypothesis): `a` per register, except `c` for the tensor-core instructions outside the
  `wgmma` shapes. Its witness is `FloorTcSat` (`h100Legacy`, a hypothetical part).
* **`FragLanes S p q W`** (named hypothesis): every multiplicand register in `W` depends on at most `q`
  lanes. It restricts the adversary's cheap tensor-core instructions for the k8 lemma, and it is not needed where
  `H32` holds. Its witness is `FragLanesSat`.
-/

namespace Pouw.Dimension.H100

/-- A `wgmma` shape of the H100 (`sm_90`) at its least `k`, by operand type and accumulator. The instruction's `N` (8
to 256) sets how many accumulator registers it writes. The model does not price `N`: every `N` costs `regPrice` per
register. On hardware that is the price at `N ≥ 64` only. Round 7 measured m64n8k32 E4M3 → F32 at 144 per register,
and sparse n8 at 160. The lower bounds are unaffected, since a real price is at least the model's. But
the pins that exhibit programs at the nominal price (`MH100Attains`, `HonestSplitK100`, `DistinctWritesAttained`) hold
on hardware only at `N ≥ 64`. -/
inductive Shape100 where
  /-- E4M3 m64nNk32, FP32 accumulate: the honest atom `HOPPER_E4M3_K32`. -/
  | e4m3F32
  /-- E5M2 m64nNk32, FP32 accumulate. -/
  | e5m2F32
  /-- E4M3 m64nNk32, FP16 accumulate. -/
  | e4m3F16
  /-- E5M2 m64nNk32, FP16 accumulate. -/
  | e5m2F16
  /-- s8 or u8 m64nNk32, s32 accumulate. -/
  | s8
  /-- FP16 m64nNk16, FP32 accumulate. -/
  | f16F32
  /-- FP16 m64nNk16, FP16 accumulate. -/
  | f16F16
  /-- BF16 m64nNk16, FP32 accumulate. -/
  | bf16F32
  /-- TF32 m64nNk8, FP32 accumulate. -/
  | tf32

/-- The shape's `k`: MACs per accumulator element. -/
def Shape100.k : Shape100 → ℕ
  | .e4m3F32 | .e5m2F32 | .e4m3F16 | .e5m2F16 | .s8 => 32
  | .f16F32 | .f16F16 | .bf16F32 => 16
  | .tf32 => 8

/-- Accumulator elements per 32-bit register: 2 for FP16 accumulate (`.f16x2`), else 1. -/
def Shape100.elems : Shape100 → ℕ
  | .e4m3F16 | .e5m2F16 | .f16F16 => 2
  | _ => 1

/-- `MH100`'s prices, in units of one dense int8 MAC. -/
structure Prices100 where
  /-- The price per MAC of each `wgmma` shape. -/
  mac : Shape100 → ℕ
  /-- The least price per accumulator register written of every other tensor-core instruction. -/
  tc : ℕ
  /-- FFMA, FADD or FMUL, one register. -/
  ffma : ℕ
  /-- An INT32 instruction, one register. -/
  int32 : ℕ
  /-- An FP64 instruction, per 32-bit register of its result. -/
  fp64 : ℕ
  /-- Every other writer of a 32-bit word, in a register or in memory (data-dependent accesses, sub-word operations,
  memory-side reductions and atomics included), per word written. -/
  other : ℕ

/-- The price of a `wgmma` shape per 32-bit accumulator register written. -/
def Shape100.regPrice (pr : Prices100) (s : Shape100) : ℕ := s.k * s.elems * pr.mac s

/-- **Hypothesis `Floor c pr`** (named; draft semantics): every register writer of `MH100` costs at least `c` per
32-bit register written. -/
def Floor (c : ℕ) (pr : Prices100) : Prop :=
  (∀ s : Shape100, c ≤ s.regPrice pr) ∧ c ≤ pr.tc ∧ c ≤ pr.ffma ∧ c ≤ pr.int32 ∧ c ≤ pr.fp64 ∧ c ≤ pr.other

/-- **Hypothesis `H32 pr`** (named; draft semantics): every register writer of `MH100` costs at least 32 per 32-bit
register written.

It is a floor per instruction class, not per register in time. Round 8 found
no single writer below 32 loop-free (`h32LoopFree`, `h32Floor`). But writers on separate pipes co-issue, and mixes go
below 32 per register: as low as 21.45 loop-free (1:1 DADD + FADD with an immediate in place of one source each), and
21.33 at full dispatch (the FP64 pipe's bound). So a bound on `cost100` under `H32` bounds additive-W1 work, not
time. A time bound needs a maximum over resources. -/
def H32 (pr : Prices100) : Prop := Floor 32 pr

/-- The datasheet's per-MAC prices (§0): FP8 and int8 1, FP16 and BF16 2, TF32 4. -/
def nominalMac : Shape100 → ℕ
  | .e4m3F32 | .e5m2F32 | .e4m3F16 | .e5m2F16 | .s8 => 1
  | .f16F32 | .f16F16 | .bf16F32 => 2
  | .tf32 => 4

/-- **The nominal H100 table** (§0, unmeasured): the datasheet's per-MAC prices; FFMA at 128 per SM per clock, 32;
INT32 at 64, 64; FP64 at 64 FMAs per SM per clock, 64 per result, 32 per register; `tc` and `other`, which the
datasheet does not give, at 32. -/
def h100Nominal : Prices100 where
  mac := nominalMac
  tc := 32
  ffma := 32
  int32 := 64
  fp64 := 32
  other := 32

/-- **Hypothesis `FloorTc a c pr`** (named; draft semantics): every register writer of `MH100` costs at least `a` per
32-bit register written, except the tensor-core instructions outside the `wgmma` shapes (legacy `mma.sync` among
them), which cost at least `c`. `Floor c pr` is `FloorTc c c pr`, so `H32 pr` is `FloorTc 32 32 pr`.

On the H100 SXM, Round 7 measured legacy m16n8k8 f16 → f32 `mma.sync` at 34.0 per register, and no tensor-core
instruction below 32 (`tc = 32`, from b1 `wgmma`). So the measured table is plain `H32` (`h32Measured`), not
`FloorTc 32 c` with `c` below 32. The 31.2–31.8 for legacy k8 was measured on the H800 PCIe,
power-limited at 350 W. `FloorTc 32 c` with `c` below 32 describes such a part, or the hypothetical `h100Legacy`
(`tc = 31`). -/
def FloorTc (a c : ℕ) (pr : Prices100) : Prop :=
  (∀ s : Shape100, a ≤ s.regPrice pr) ∧ c ≤ pr.tc ∧ a ≤ pr.ffma ∧ a ≤ pr.int32 ∧ a ≤ pr.fp64 ∧ a ≤ pr.other

/-- **Hypothesis `FragLanes S p q W`** (named; draft semantics): every multiplicand register in `W`
depends on at most `q` of the lanes `p` on `S`, that is, some `q` lanes determine it on `S`.

It restricts the adversary's cheap tensor-core instructions, those below `a` per register under `FloorTc a c`. The k8
lemma takes their multiplicands from `W` (`AccFragIn`, `Pouw.Dimension.H100Frag`). It is not a fact about the
hardware, and it can fail: a fragment shared across column tiles may depend on every lane, and then the lemma's bound
is false (`FragShareGain`). So the k8 lemma is cited only against adversaries whose cheap tensor-core multiplicands
meet `FragLanes` (`DenseWritesFragLanes`), not against every adversary.

It is not needed at Round 8's loop-free prices. The H100 SXM's legacy k8 `mma.sync` costs 32.00 per register loop-free,
which meets `H32` with no slack. So `DistinctWritesLoopFree` and `DistinctWritesFloor` (`Pouw.Dimension.H100Floor`)
charge 32 per distinct non-free target with no hypothesis on any instruction's multiplicands. Its
satisfiability witness is `FragLanesSat`: the lane words meet `FragLanes 1`. -/
def FragLanes {G Λ : Type} (S : Set G) (p : Λ → G → ℚ) (q : ℕ) (W : Set (G → ℚ)) : Prop :=
  ∀ w ∈ W, ∃ J : Finset Λ, J.card ≤ q ∧ ∀ g ∈ S, ∀ g' ∈ S, (∀ j ∈ J, p j g = p j g') → w g = w g'

end Pouw.Dimension.H100
