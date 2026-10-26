import Definitions.Core.Accel.Card

/-!
# The RTX PRO 6000 Blackwell (sm_120) as an accelerator (trusted)

The RTX PRO 6000 Blackwell Server Edition (GB202, sm_120a) at its data sheet's numbers, a tick one SM clock at
f_max = 2430 MHz (the hardware's maximum, which the prover may set: SPEC's P1, note:20261017T0324Z-draft-eps1-spec,
`benchmarks/pouw/machine_model/data/designs.toml`). Every per-SM figure is over 188 SMs (NVIDIA RTX PRO Blackwell GPU
Architecture whitepaper v1.1, Appendix A, Table 4, the same 188-SM GB202). These are data-sheet placeholders until
CALIB (CMP-115) measures them.

**Stores** (32-bit words): the register file, 64 Ki registers an SM (whitepaper: a 256 KB register file an SM);
shared memory with L1, 128 KB an SM (whitepaper: 128 KB of L1 and shared memory an SM); L2, 128 MB (whitepaper Table
4), taken as 128 MiB; GDDR7, 96 GB (product page, 'GPU Memory 96 GB GDDR7'), taken as 96 GiB.

**Rates** (units a tick):
* tensor: 1024 s8 MACs an SM clock (`IMMA.16832`, 4096 MACs in 16 tensor cycles of an SM sub-partition, the DS-120
  census, note:20261017T1818Z-draft-axioms; the whitepaper's 1,007.6 dense FP8 TFLOPS over 2 · 188 SMs at its
  2617 MHz boost is the same 1024), so 192,512; a MAC of `e4m3`, `e5m2` or `e2m1` (`kind::f8f6f4`, `QMMA.16832`) uses
  1 unit, of `nvf4` or `mxf4` (`OMMA.SF.16864`, 8192 MACs in 16 cycles; whitepaper FP4 2,015.2 dense) 1/2, of `f16` or
  `bf16` 2 (503.8), of `tf32` 4 (251.9); `b1` has no peak in any document (`Open.b1`);
* issue: 4 warp instructions an SM clock (four sub-partitions, one warp scheduler each), so 752;
* `alu` and `imad`: 64 lanes an SM clock each (the DS-120 census: `LOP3` and `IMAD` two cycles a warp instruction on a
  sub-partition; CUDA C++ Programming Guide §6.4.1 Table 7), so 12,032 each;
* shared memory: 32 banks of a 32-bit word a clock an SM (Programming Guide, 'Shared Memory'), so 6016;
* GDDR7: 1597 GB/s (product page, 'Memory Bandwidth 1597 GB/s'), 1597 · 10⁹ / (4 · 2.43 · 10⁹) = 39925/243 words.

**Opcodes** (PTX ISA 9.0 §9.7.14): `mma.sync` at `m16n8`, both operands in registers (sm_120 has no `wgmma`), `k` per
format (`ks`); `s8`/`u8` either way, `e4m3`, `e5m2` and `e2m1` in any pairing (`kind::f8f6f4`), the other formats with
themselves. Every lane function, and the integer ones on the uniform datapath too (`uni`, a whole issue slot each:
CUDA Binary Utilities §4.4 lists the uniform opcodes CC 10.0 to 12.0 share, and no document gives their issue cost;
one slot as measured on sm_100, RT-B200-EMP, run r20261019-055605-6cc2, and on sm_90, a placeholder until CALIB
measures sm_120; whether sm_120's uniform datapath has functions beyond the integer ones is a gap); copies of at most
4 words, and bulk copies with shared memory at one end (TMA);
atomics on shared memory, L2 and GDDR7.

**Captured dots** (`captured`): the integer path exactly (`models.BLACKWELL_SM120_S8_M16N8K32`, `_WRAP`,
`_U8S8_M16N8K32`); `bf16` at `k = 16`, `TC.HOPPER_BF16_M16N8K16` (its docstring: `mma.sync.m16n8k16.bf16` on Hopper and
on sm_120). `e4m3`, `e5m2` and `e2m1` with themselves at `k = 32` are `Open.dot`: their candidate is one group of 32 on
a 26-bit adder, floor −133 (`f8Pipeline`; `models.BLACKWELL_SM120_E4M3_M16N8K32`, `_E5M2_M16N8K32`, `_E2M1_M16N8K32`;
LIT: one group of 32 at width 26), but PEARLC-MODEL found the atom to be the exact sum's FP32 truncation in 99.96–99.997%
of atoms only, so no capture pins it. The block-scaled FP4 dots (`models.BLACKWELL_SM120_NVF4`, `_MXF4`, captured in Python, a block-scaled adder this module doesn't
transcribe) and every other dot are `Open.dot`.

**Coarser than the card**, and **not yet in the instance**: as for the H200 (`Accel.H200`), with FP6 (`e3m2`, `e2m3`)
and the block-scaled FP8 and FP6 forms (`kind::mxf8f6f4`) among the gaps, and no DMMA at all (sm_120 has none at rate).
-/

open scoped NNRat

namespace Verity.Accel.Sm120

/-- The words of each store. -/
def size : Store → ℕ
  | .reg => 188 * 65536
  | .smem => 188 * 32768
  | .l2 => 128 * 2 ^ 20 / 4
  | .dram => 96 * 2 ^ 30 / 4

/-- Each resource's units a tick. -/
def ρ : Res → ℚ≥0
  | .tensor => 188 * 1024
  | .issue => 188 * 4
  | .alu => 188 * 64
  | .imad => 188 * 64
  | .smem => 188 * 32
  | .dram => 39925 / 243

/-- The tensor units a MAC of each format uses. -/
def weight : Fmt → ℚ≥0
  | .s8 | .u8 | .e4m3 | .e5m2 | .e2m1 => 1
  | .nvf4 | .mxf4 => 1 / 2
  | .f16 | .bf16 => 2
  | .tf32 => 4
  | .b1 => 0

/-- PTX's `k` for each format at `m16n8`. -/
def ks : Fmt → List ℕ
  | .s8 | .u8 => [16, 32]
  | .e4m3 | .e5m2 | .e2m1 => [32]
  | .f16 | .bf16 => [8, 16]
  | .tf32 => [4, 8]
  | .nvf4 | .mxf4 => [64]
  | .b1 => [128, 256]

/-- Whether a format is one of `kind::f8f6f4`'s. -/
def f8f6f4 (f : Fmt) : Bool := f = .e4m3 || f = .e5m2 || f = .e2m1

/-- The formats an MMA pairs. -/
def pair (fa fb : Fmt) : Bool := intPair fa fb || (f8f6f4 fa && f8f6f4 fb) || fa = fb

/-- The opcodes the RTX PRO 6000 has. -/
def Avail : Op Store → Prop
  | .mma fa fb _ sh sa sb => pair fa fb ∧ sh.k ∈ ks fa ∧ sh.m = 16 ∧ sh.n = 8 ∧ sa = .reg ∧ sb = .reg
  | .lane _ => True
  | .uni g => g.pipe ≠ .fma
  | .copy s t n => n ≤ 4 ∨ (s = .smem ∧ t ≠ .reg) ∨ (t = .smem ∧ s ≠ .reg)
  | .atom s _ => s ≠ .reg

/-- The candidate for sm_120's FP8 and `kind::f8f6f4` E2M1 atom (`models.BLACKWELL_SM120_E4M3_M16N8K32`, `_E5M2_`,
`_E2M1_`), not pinned (`captured`). -/
def f8Pipeline : TC.Pipeline := { groups := [32], width := 26, zeroExponent := -133 }

/-- The dots the RTX PRO 6000's captures pin. -/
def captured (fa fb : Fmt) (sat : Bool) (k : ℕ) : Option Dot :=
  match intCaptured fa fb sat with
  | some d => some d
  | none =>
    if fa = .bf16 ∧ fb = .bf16 ∧ k = 16 then some (floatDot .bf16 TC.HOPPER_BF16_M16N8K16) else none

/-- **The RTX PRO 6000's table.** A program's code is at most GDDR7's bytes over 16-byte instructions; the coarse
device's rates are the tensor core's s8 MACs, the 188 · 128 FP32 lanes, and GDDR7's bits a tick rounded up
(1597 · 10⁹ · 8 / 2.43 · 10⁹ = 5,257.6). -/
def card : Card where
  size := size
  size_pos := by intro s; cases s <;> decide
  ρ := ρ
  weight := weight
  Avail := Avail
  scale _ _ := 1
  captured := captured
  zeroOnly _ _ _ _ := false
  text := 96 * 2 ^ 30 / 16
  tc := 188 * 1024
  cuda := 188 * 128
  bw := 5258

/-- **The RTX PRO 6000 (sm_120)** as an accelerator, its open parameters `x`. -/
def machine {ι : Type} (x : Open) (cap : ι → ℕ) (oracle : ℕ) : Machine ι := card.machine x cap oracle

end Verity.Accel.Sm120
