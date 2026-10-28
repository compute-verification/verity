import Definitions.Pouw.PearlC.W1
import Definitions.Pouw.PearlC.NoiseSha512

/-!
# Checks of W1's cost model and SHA-512's noise

- `sha512_abc`, `sha512_two_blocks`: `Sha512.digest` against `hashlib.sha512` (FIPS 180-2's `"abc"`, and 112 bytes,
  which pad to two blocks), evaluated by the kernel.
- `rowDigest_vec`, `rowSeed_vec`, `seedB_vec`, `lineEA_vec`, `lineFB_vec`: `NoiseSha512` against
  `verity/core/protocols/pouw/schemes/pearl_c.py` (`row_digest`, `row_seed`, `seed_b`, `sample_line`) on one 15-word row
  of unit 3, row 5, with salt `0, 1, …, 31` and root `(7i + 3) mod 256`, evaluated by the kernel.
- `toPrices_h100`, `toPrices_sm120`, `toPrices_sm120Loop`: W1's tables give the coarse records the devices use;
  `toFp4Prices_fp4Sm120` (and `toFp4Prices_fp4Sm120At` at any scale path): the FP4 record's, `Fp4Prices.sm120`.
- `price_e4m3_sm120`, `price_bf16_sm120`, `price_e2m1_sm120`: each sm_120 fragment costs 4,096 units, however few of its
  outputs are used; `price_e4m3_sm120_taint`: under a branch on data, 2,048 more (16 per output word).
- `cost_oneMma`: a whole program of one E4M3 fragment on scratch costs exactly 4,096 under `w1`, for every protocol,
  workload, activations, state and salt.
- `w1_queries`: `w1` meets `CostModel.Queries` (it makes no oracle call).
-/

namespace Pouw.PearlC.W1Checks

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.W1 Pouw.PearlC.Sha512 Pouw.PearlC.NoiseSha512

/-! ## SHA-512 -/

/-- SHA-512 of `"abc"` (FIPS 180-2, appendix C.1). -/
theorem sha512_abc : digest (ascii "abc") =
    [0xdd, 0xaf, 0x35, 0xa1, 0x93, 0x61, 0x7a, 0xba, 0xcc, 0x41, 0x73, 0x49, 0xae, 0x20, 0x41, 0x31,
     0x12, 0xe6, 0xfa, 0x4e, 0x89, 0xa9, 0x7e, 0xa2, 0x0a, 0x9e, 0xee, 0xe6, 0x4b, 0x55, 0xd3, 0x9a,
     0x21, 0x92, 0x99, 0x2a, 0x27, 0x4f, 0xc1, 0xa8, 0x36, 0xba, 0x3c, 0x23, 0xa3, 0xfe, 0xeb, 0xbd,
     0x45, 0x4d, 0x44, 0x23, 0x64, 0x3c, 0xe8, 0x0e, 0x2a, 0x9a, 0xc9, 0x4f, 0xa5, 0x4c, 0xa4, 0x9f] := by
  decide +kernel

/-- SHA-512 of 112 bytes `7`, whose padding takes a second block (`hashlib.sha512(bytes([7] * 112))`). -/
theorem sha512_two_blocks : (digest (List.replicate 112 7)).take 4 = [154, 87, 92, 91] := by decide +kernel

/-! ## The noise against `pearl_c.py` -/

/-- The vectors' salt. -/
def vSalt : List ℕ := List.range 32

/-- The vectors' weight root. -/
def vRoot : List ℕ := (List.range 64).map fun i => (7 * i + 3) % 256

/-- The vectors' activation row: 15 FP32 words, little-endian. -/
def vRow : List ℕ :=
  (List.flatten (List.replicate 3 [0x3f800000, 0xbf000000, 0x40490fdb, 0x00000000, 0x3dcccccd])).flatMap (leBytes 4)

theorem rowDigest_vec : (rowDigest vRow).take 8 = [225, 54, 37, 2, 134, 5, 77, 26] := by decide +kernel

/-- A row of 2,500 bytes: three segments, the last of 452 bytes. -/
theorem rowDigest_segments_vec : (rowDigest ((List.range 2500).map (· % 253))).take 8 = [120, 135, 221, 184, 244, 28, 39, 7] := by
  decide +kernel

theorem rowSeed_vec : (rowSeed vSalt vRow vRoot 3 5).take 8 = [90, 99, 57, 103, 5, 175, 11, 41] := by decide +kernel

theorem seedB_vec : (seedB vSalt vRoot).take 8 = [247, 238, 5, 151, 197, 237, 99, 121] := by decide +kernel

/-- Row 5's `E_A` line (norm 256). -/
theorem lineEA_vec : (lineList (rowSeed vSalt vRow vRoot 3 5) 0 0 0 256).map (·.val) =
    [103, 233, 104, 227, 104, 95, 83, 223, 231, 95, 100, 105, 177, 230, 93, 229, 228, 228, 218, 208, 96, 229, 94, 65,
     98, 232, 207, 102, 91, 91, 97, 232] := by
  decide +kernel

/-- `F_B`'s line 7 (norm 16). -/
theorem lineFB_vec : (lineList (seedB vSalt vRoot) 1 1 7 16).map (·.val) =
    [71, 184, 67, 191, 192, 146, 201, 65, 181, 187, 72, 188, 201, 190, 194, 72, 200, 200, 73, 62, 72, 57, 186, 71,
     154, 66, 199, 194, 183, 66, 154, 196] := by
  decide +kernel

/-! ## Prices -/

theorem toPrices_h100 (sha fetch : ℚ) : (W1Prices.h100 sha fetch).toPrices = Prices.h100 := by
  simp only [W1Prices.toPrices, W1Prices.h100, Prices.h100, Costs.adopted]
  norm_num

theorem toPrices_sm120 (sha fetch : ℚ) : (W1Prices.sm120 sha fetch).toPrices = Prices.sm120 := by
  norm_num [W1Prices.toPrices, W1Prices.sm120, Prices.sm120]

theorem toPrices_sm120Loop (sha fetch : ℚ) : (W1Prices.sm120Loop sha fetch).toPrices = Prices.sm120Loop := by
  simp only [W1Prices.toPrices, W1Prices.sm120Loop, W1Prices.sm120, Prices.sm120Loop]
  norm_num [Nat.ceil_eq_iff]

/-- The FP4 record's BF16 MAC and FP32 add, in FP4 units, are `Fp4Prices.sm120`'s 4 and twice the loop's FADD. -/
theorem fp4Sm120_prices (sha fetch : ℚ) :
    (W1Prices.fp4Sm120 sha fetch).mac .bf16 = Fp4Prices.sm120.bf16 ∧ (W1Prices.fp4Sm120 sha fetch).mac .e2m1 = 1 ∧
      (W1Prices.fp4Sm120 sha fetch).fp32 = 2 * (1047 / 125) := by
  simp only [W1Prices.fp4Sm120, W1Prices.scale, W1Prices.sm120Loop, W1Prices.sm120, Fp4Prices.sm120]
  norm_num

/-- W1's FP4 table gives the FP4 record's coarse prices at any block-scale path: `Fp4Prices.sm120At` at the loop's
FADD, `1047/125`. -/
theorem toFp4Prices_fp4Sm120At (sha fetch : ℚ) (p : Fp4ScalePath) :
    (W1Prices.fp4Sm120 sha fetch).toFp4Prices p = Fp4Prices.sm120At (1047 / 125) p := by
  have hb : (W1Prices.fp4Sm120 sha fetch).mac .bf16 = 4 := by
    norm_num [W1Prices.fp4Sm120, W1Prices.scale, W1Prices.sm120Loop, W1Prices.sm120]
  have he : (W1Prices.fp4Sm120 sha fetch).mac .e2m1 = 1 := by
    norm_num [W1Prices.fp4Sm120, W1Prices.scale, W1Prices.sm120Loop, W1Prices.sm120]
  have hf : (W1Prices.fp4Sm120 sha fetch).fp32 = 2 * (1047 / 125) := by
    norm_num [W1Prices.fp4Sm120, W1Prices.scale, W1Prices.sm120Loop, W1Prices.sm120]
  have hc : (W1Prices.fp4Sm120 sha fetch).cvt = 32 := by
    norm_num [W1Prices.fp4Sm120, W1Prices.scale, W1Prices.sm120Loop, W1Prices.sm120]
  rw [W1Prices.toFp4Prices, Fp4Prices.sm120At, hb, he, hf, hc]
  congr 1 <;> ring

/-- **The FP4 record's pairing**: `W1Device.fp4Sm120`'s table gives `devFp4Sm120`'s prices `Fp4Prices.sm120` (the block
scale on `lut256`, FADD at `1047/125`). -/
theorem toFp4Prices_fp4Sm120 (sha fetch : ℚ) :
    (W1Prices.fp4Sm120 sha fetch).toFp4Prices Fp4ScalePath.lut256 = Fp4Prices.sm120 := by
  rw [toFp4Prices_fp4Sm120At]
  norm_num [Fp4Prices.sm120At, Fp4ScalePath.lut256, Fp4Prices.sm120]

/-! ## Fragments are priced whole -/

variable (sha fetch : ℚ) (d a b c sa sb : Addr)

theorem price_e4m3_sm120 : (Instr.mma ⟨.e4m3, 16, 8⟩ d a b c sa sb).price (W1Device.sm120 sha fetch) false = 4096 := by
  norm_num [Instr.price, Instr.base, Instr.dynamic, W1Device.sm120, W1Device.sm120At, W1Device.has, sm120Frags,
    W1Device.depth, sm120E4m3, sm120E4m3K32, W1Prices.sm120]

theorem price_bf16_sm120 : (Instr.mma ⟨.bf16, 16, 8⟩ d a b c sa sb).price (W1Device.sm120 sha fetch) false = 4096 := by
  norm_num [Instr.price, Instr.base, Instr.dynamic, W1Device.sm120, W1Device.sm120At, W1Device.has, sm120Frags,
    W1Device.depth, PeelExact.hopperBf16K16, W1Prices.sm120]

theorem price_e2m1_sm120 : (Instr.mma ⟨.e2m1, 16, 8⟩ d a b c sa sb).price (W1Device.sm120 sha fetch) false = 4096 := by
  norm_num [Instr.price, Instr.base, Instr.dynamic, W1Device.sm120, W1Device.sm120At, W1Device.has, sm120Frags,
    W1Device.depth, Fp4.sm120Nvf4, W1Prices.sm120]

/-- Under a branch on data, the fragment also pays the taint rule's INT32 per output word: `4096 + 16·128`. -/
theorem price_e4m3_sm120_taint :
    (Instr.mma ⟨.e4m3, 16, 8⟩ d a b c sa sb).price (W1Device.sm120 sha fetch) true = 6144 := by
  norm_num [Instr.price, Instr.base, Instr.dynamic, Instr.words, W1Device.sm120, W1Device.sm120At, W1Device.has, sm120Frags,
    W1Device.depth, sm120E4m3, sm120E4m3K32, W1Prices.sm120]

/-! ## A whole program -/

/-- One E4M3 `m16n8k32` on scratch words: `D` at 0, `A` at 100, `B` at 200, `C` at 300. Its text is two nodes and no
constant, within W1's bound. -/
def oneMma : Listing := ⟨fun _ => [], fun _ => .op (.mma ⟨.e4m3, 16, 8⟩ (0, 0) (0, 100) (0, 200) (0, 300) (0, 0) (0, 0)) .done,
  fun _ => by norm_num [W1.Code.size]⟩

variable {Q R S : Type}

/-- **W1 charges the fragment 4,096 units**, and nothing else: its operands are scratch, so no fetch is priced. -/
theorem cost_oneMma (P : Protocol Q R S) (salt : S → List ℕ) (H : Q → R) (U : Workload) (act : ℕ → Codes) (σ : State)
    (s : S) : (w1 (W1Device.sm120 sha fetch) P salt).cost oneMma H U act σ s = 4096 := by
  have hf : (final (W1Device.sm120 sha fetch) (salt s) oneMma U act σ).fetched = [] := by
    simp only [final, oneMma, exec, List.nil_append]
    rw [List.filter_eq_nil_iff]
    intro x hx
    simp only [Instr.reads, W1Device.has, W1Device.sm120, W1Device.sm120At, sm120Frags] at hx
    simp only [List.mem_cons, List.not_mem_nil, or_false, true_or, decide_true, ite_true] at hx
    simp only [Addr.span, List.mem_append, List.mem_map, Addr.off] at hx
    simp only [preSalt]
    rcases hx with ((⟨o, -, rfl⟩ | ⟨o, -, rfl⟩) | ⟨o, -, rfl⟩) | h <;> simp_all
  have hc : (final (W1Device.sm120 sha fetch) (salt s) oneMma U act σ).cost = 4096 := by
    simp only [final, oneMma, exec, zero_add]
    exact price_e4m3_sm120 sha fetch _ _ _ _ _ _
  simp only [w1, hf, hc, List.dedup_nil, List.filter_nil, List.length_nil, Nat.cast_zero, mul_zero, add_zero]
  norm_num

/-- W1 makes no oracle call, so its runs are oracle trees with no query. -/
theorem w1_queries (dv : W1Device) (P : Protocol Q R S) (salt : S → List ℕ) : (w1 dv P salt).Queries := by
  intro pr U act σ s
  exact ⟨.done (readout U act (final dv (salt s) pr U act σ).mem), fun _ => ⟨rfl, rfl⟩⟩

end Pouw.PearlC.W1Checks
