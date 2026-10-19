import Definitions.Pouw.Fp8Atom.Fp32
import Definitions.Pouw.PearlC.Fp4

/-!
# The FP MMA premise, per device (trusted; statement only, red team before proof)

What an FP-pipe floor reads of the FP8 and FP4 MMAs it credits, beside the data-sheet axioms A1–A7
(`Models.Accounting.Assumptions.Sheet`), on the RTX PRO 6000 (sm_120), H200 (sm_90a) and B200 (sm_100a).
note:20261019T0457Z-draft-fpx-machine grades every fact's source.

**Every `Prop` here is measured only: no vendor document states it.** PTX ISA 9.4 says an MMA's products are formed
"with specified precision", that accumulation is "at least single precision" (mma) or "higher than half precision but
lower than single precision" (wgmma with E4M3/E5M2), and that "the accumulation order, rounding, and handling of
subnormal inputs are unspecified"; tcgen05.mma states none of it. So decision 96 bars a floor's soundness from reading
them. None does: completeness and γ read them, through a design's circuit, which checks the committed word against its
Definition, so a device that breaks one fails the honest proof visibly and no cheater gains. Soundness against
elementwise routes reads only the code widths, A3 and A7 (`benchmarks/pouw/theory/isa_escape_fp.py`).

- (a) Exact-integer designs read `Fp8GridExact` and `Fp4GridExact`, the weakest form: exactness inside one grid
  `2^g`, nothing about the window, rounding or order outside it. `p = 24` on sm_120 and B200, `p = 14` on H200. Every
  measured exact route (k3, k4, k3z, k4s, k3h, e4r, the ±16 grid, the FP4 grids) lives inside it.
- (b) FP-native designs read `Fp8Model` and `Fp4Model`, the whole bit-exact step, and `FpZeroPositive`. Lean has the
  models for sm_120 (`Pouw.PearlC.sm120E4m3`, `Pouw.PearlC.Fp4.sm120Nvf4`, `sm120Mxf4`) and for Hopper's H100 captures
  (`Pouw.Fp8Atom.hopperE4m3K32`); B200 has no FP8 model of record.

Each is a `Prop` on a step function, the device's behaviour, which Lean does not model; the claim for a device is the
`Prop` at the step that device's instruction computes. The model instances (`Fp8GridExact 24 Pouw.PearlC.sm120Step`,
`Fp8GridExact 14 Pouw.Fp8Atom.hopperStep`, and the FP4 pair at `p = 24`) are theorem targets, replayed by
`isa_escape_fp.py`. A guarantee that reads one of these lists it in `lean-audit.json`.
-/

namespace Accounting.Assumptions.TS

open Pouw.Fp8Atom

/-- An FP8 MMA step on one `k = 32` group: the FP32 accumulator word and two rows of E4M3 codes, to the FP32 word it
writes. It is sm_120's `mma.sync … .e4m3` (QMMA), one k32 slice of H200's `wgmma … .e4m3`, or one k32 slice of B200's
`tcgen05.mma .kind::f8f6f4` with E4M3 operands. -/
abbrev Fp8Step : Type := ℕ → (Fin 32 → Code) → (Fin 32 → Code) → ℕ

/-- A block-scaled FP4 MMA step on one `k = 64` slice: the FP32 accumulator word, two rows of E2M1 nibbles, and each
side's scale bytes by block index (element `l` reads block `l / 16` for UE4M3, `l / 32` for UE8M0), to the FP32 word it
writes. It is sm_120's `mma.sync … .kind::mxf4nvf4` (OMMA) or one k64 slice of B200's `tcgen05.mma .kind::mxf4nvf4`. -/
abbrev Fp4Step : Type :=
  ℕ → (Fin 64 → Pouw.PearlC.Fp4.Nib) → (Fin 64 → Pouw.PearlC.Fp4.Nib) → (ℕ → Fin 256) → (ℕ → Fin 256) → ℕ

/-- **The small-grid regime** at precision `p`: the accumulator's value `c` and every term are integer multiples of one
`2^g` with `−126 ≤ g` and `g + p ≤ 128`, and their magnitudes sum below `2^(g+p)`. Every partial sum then has at most
`p` significant bits on the grid `2^g` and is an FP32 value. -/
def FpGridSmall (p : ℕ) (c : ℚ) (ts : List ℚ) : Prop :=
  ∃ g : ℤ, -126 ≤ g ∧ g + p ≤ 128 ∧ (∀ t ∈ c :: ts, ∃ n : ℤ, t = n * (2 : ℚ) ^ g) ∧
    |c| + (ts.map fun t => |t|).sum < (2 : ℚ) ^ (g + p)

/-- The products of an FP8 step whose operand codes decode to `x` and `y`. -/
def fp8Terms (x y : Fin 32 → ℚ) : List ℚ := List.ofFn fun l => x l * y l

/-- The block-scaled products of an FP4 step: element `l`'s two nibbles' values times its blocks' scale values `u l` and
`v l`. -/
def fp4Terms (a b : Fin 64 → Pouw.PearlC.Fp4.Nib) (u v : Fin 64 → ℚ) : List ℚ :=
  List.ofFn fun l => Pouw.PearlC.Fp4.decode (a l) * u l * (Pouw.PearlC.Fp4.decode (b l) * v l)

/-- `accounting-fp-grid-exact` (measured only: FP-EXACT r20261016-200609-3d39 on sm_120; LOWBIT-DC's sweeps
r20261017-070305-1fe2 on H200 and r20261017-070314-2de1 on B200, whose window tests are exact at every gap ≤ 13 (H200)
and ≤ 23 (B200) and inexact at 14 and 24, the accumulator included, and whose one failure outside the controls, a
27-bit gap on B200, lies outside the regime; both exit 4, recorded failed, because the broader "exact whenever
FP32-representable" fails; no vendor document). **FP-GRID, FP8**: an E4M3 MMA
step whose finite accumulator and products lie in the small-grid regime at `p` writes the exact sum. Claimed at
`p = 24` for sm_120's QMMA and B200's `kind::f8f6f4`, and at `p = 14` for H200's `wgmma` (its adder is 14 bits wide,
the accumulator included). Re-testable by auditor-run conformance vectors on a same-SKU device. -/
def Fp8GridExact (p : ℕ) (step : Fp8Step) : Prop :=
  ∀ (acc : ℕ) (a b : Fin 32 → Code) (c : ℚ) (x y : Fin 32 → ℚ),
    wordQ acc = some c → (∀ l, Pouw.Fp8Atom.decode (a l) = some (x l)) →
      (∀ l, Pouw.Fp8Atom.decode (b l) = some (y l)) → FpGridSmall p c (fp8Terms x y) →
        wordQ (step acc a b) = some (c + (fp8Terms x y).sum)

/-- `accounting-fp-grid-exact` (measured only: FP-EXACT-SCALES r20261016-214357-f15d on sm_120; LOWBIT-DC
r20261017-070314-2de1 on B200, MXF4 and NVF4 windows exact at every gap ≤ 23 and inexact at 24; no vendor document).
**FP-GRID, FP4**: a block-scaled E2M1 MMA step with scale format
`f`, whose finite accumulator and scaled products lie in the small-grid regime at `p`, writes the exact sum. Claimed at
`p = 24` for sm_120's OMMA and B200's `kind::mxf4nvf4`, for UE4M3 (NVFP4) and UE8M0 (MXFP4). H200 has no FP4 kind. -/
def Fp4GridExact (f : Pouw.PearlC.Fp4.ScaleFmt) (p : ℕ) (step : Fp4Step) : Prop :=
  ∀ (acc : ℕ) (a b : Fin 64 → Pouw.PearlC.Fp4.Nib) (sa sb : ℕ → Fin 256) (c : ℚ) (u v : Fin 64 → ℚ),
    wordQ acc = some c →
      (∀ l : Fin 64, Pouw.PearlC.Fp4.scaleValue f (sa (l.val / f.block)) = some (u l)) →
      (∀ l : Fin 64, Pouw.PearlC.Fp4.scaleValue f (sb (l.val / f.block)) = some (v l)) →
      FpGridSmall p c (fp4Terms a b u v) →
        wordQ (step acc a b sa sb) = some (c + (fp4Terms a b u v).sum)

/-- `accounting-fp-zero` (measured only: the registry's FP4 models on sm_120 and B200, `underflow_zero = "positive"`;
no −0 word in any chain of LOWBIT-DC's sweeps on B200 and H200, which don't count their zero results; no vendor
document). **FP-ZERO**: a step's zero result
is the `+0` word. A design that reads the word's bits at zero, rather than its value, needs it. -/
def FpZeroPositive {α : Type} (word : α → ℕ) : Prop := ∀ x, wordQ (word x) = some 0 → word x = 0

/-- `accounting-fp-model` (measured only: registry `sm120.mma.m16n8k32.e4m3` pinned,
art:b3074aba4c9ddeef7f87454bfc23e73a5a3c5f73f1676260545115fe8589cc89, run r20260930-063457-5410, 25,041,024 elements,
0 mismatches; the H100 captures r20260922-183435-fd5d; no vendor document).
**FP-MODEL, FP8**: the step is the pipeline `P`'s bit-exact step on every input. Claimed for sm_120 at
`Pouw.PearlC.sm120E4m3`. Hopper's `Pouw.Fp8Atom.hopperE4m3K32` is fitted on H100, and H200 is measured for its window
only. B200 has no model of record, so an FP-native FP8 design can't be served there until one is fitted. -/
def Fp8Model (P : Pipeline) (step : Fp8Step) : Prop :=
  ∀ (acc : ℕ) (a b : Fin 32 → Code), step acc a b = P.step acc (List.ofFn a) (List.ofFn b)

/-- `accounting-fp-model` (measured only: the FP4 registry pins,
art:3e02931df28cccd398de4219f9d977371a58cdc6e8addfb7ff0a6269e5481b0e and
art:d8d4d1b602c9fde61df218c315430535fe5181882ab6b26c2b95dd81dc1b5d92; no vendor document).
**FP-MODEL, FP4**: wherever the atom `q`'s bit-exact step is defined, the step writes its word. Claimed for sm_120 at
`Pouw.PearlC.Fp4.sm120Nvf4` and `sm120Mxf4` (Td is open for NVF4). B200's FP4 models, pinned in the Python registry
only (`BLACKWELL_SM100_NVF4`, `BLACKWELL_SM100_MXF4`: tcgen05, 0 mismatches over 16,461,824 and 12,898,304 elements),
carry sm_120's parameters, so B200's claim is the same two instances at its k64 slice. -/
def Fp4Model (q : Pouw.PearlC.Fp4.Fp4Atom) (step : Fp4Step) : Prop :=
  ∀ (acc : ℕ) (a b : Fin 64 → Pouw.PearlC.Fp4.Nib) (sa sb : ℕ → Fin 256) (w : ℕ),
    q.step acc (List.ofFn a) (List.ofFn b) (List.ofFn fun j : Fin q.scales => sa j.val)
        (List.ofFn fun j : Fin q.scales => sb j.val) = some w →
      step acc a b sa sb = w

end Accounting.Assumptions.TS
