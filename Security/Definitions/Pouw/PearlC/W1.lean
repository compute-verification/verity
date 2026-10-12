import Definitions.Pouw.PearlC.DevicePrices
import Definitions.Pouw.PearlC.DevicePricesLoop
import Definitions.Pouw.PearlC.DeviceFp4
import Definitions.Pouw.PearlC.Sha512

/-!
# The accounting W1 as a cost model (trusted; definitions only)

`CostModel Q R S` (`Game.lean`) is abstract, and every TT_OUT form names "W1 at the device's measured prices" only in
its docstring. This module is W1 as a definition: a machine, its instructions, their prices per device, and
`w1 dv P salt : CostModel Q R S`, the cost of a program as W1 charges it. It states no assumption and changes no
statement; a restatement of TT_OUT over it is the proofs lane's.

**What W1 is** (the problem statement §3.1, `note:20261004T2202Z-report-relay-docs-pouw-problem-statement`; decided
28 Sep, amended by H5):
- an instruction is priced by its class at the measured rate, per 32-bit lane result, and a tensor-core instruction by
  its whole fragment, however few of its outputs a program uses (`w1-complete/sm120`);
- oblivious whole-word moves are free, at any memory level;
- an instruction whose address, guard or enclosing branch depends on data costs one INT32 instruction per 32-bit word it
  moves or produces (the taint discipline, H5);
- the first fetch of a pre-salt byte (the preprocessed state, the declared weights, the program's constants) or of a
  supplied one (the activations) costs the DRAM rate, except a unit's operands when the unit comes out correct (W1-1,
  W1-3);
- oracle calls are free, and counted in `q` (decision 1).

**The machine** (`Instr`, `Code`, `exec`). Memory is addressed by `(region, index)` (`Addr`) and holds 32-bit words.
The regions are the scratch (0), the salt's bytes (1), the preprocessed state σ (2: its length, then its entries as
32-bit words), the program's constant bank (3), the activations (4) and the weights (5), both FP32 words at
`Nat.pair`-coded positions, and the outputs: the ticket words (6) and the useful words (7). A program (`Prog`) is a
constant bank and a code tree, both functions of the public layout, so the layout's control flow, unrolling and
addressing are compiled in and free, as public control flow is on a GPU. What depends on data is an `ite` (a branch on a
word) or an `ld`/`st` with a register address, and W1's taint rule prices both: an `ite` costs one INT32 instruction,
and every instruction under it pays one INT32 instruction per word it writes (`Instr.price`). An instruction is one lane:
its price is W1's per-lane price, except a tensor-core instruction, which is the warp's whole fragment.

**The run.** `run` executes the program from the initial memory (`init`) and reads the transcript off the final memory
(`readout`): every declared unit commits its activations (as the program left them; positions outside the unit's shape
read back as supplied), its ticket words and its useful words. The commitment is ideal (`Commit`), so the readout is
free. The model makes no oracle call (`calls = 0`), so it meets `CostModel.Queries` for any `Q`, `R`
(`Proofs.Pouw.PearlC.W1Checks.w1_queries`).

**The cost** is the instructions' prices plus `fetch` per distinct pre-salt or supplied word read, except the words of
a correct unit's operands (its activation rows and its weight rows, within its shape), which is why the model takes the
protocol `P`: whether a unit is correct is `P`'s `Correct` on the transcript the run commits.

**The instructions and their classes** (`Instr`):
- free moves: `mov`; with a register address, `ld` and `st`, which pay the taint rule's INT32 per word;
- INT32 (`int32`): `iop` (add, subtract, multiply low and high, and, or, xor, shifts, comparisons, min, max), `iadd3`,
  `lop3` (any three-input bitwise function), `prmt` (byte permute, the sub-word rule's PRMT) and `sel`;
- FP32 (`fp32`): FADD, FMUL, FFMA, FMNMX on FP32 words, round to nearest even (`PeelExact.addWord?`, `fmaWord?`);
- a cast (`cvt`): two FP32 words to two E4M3, BF16 or E2M1 codes packed into a field of the destination
  (`F2FP…PACK_AB_MERGE_C`), one instruction;
- a tensor-core fragment (`mma`): `m × n` outputs of depth `k` (the device's for the format), at `m·n·k` times the
  format's price per MAC, on the device's captured step (`W1Device`);
- a SHA-512 compression (`sha`), at `sha512`.

**Prices per device** (`W1Prices`, in the record's unit: one dense E4M3 MAC with FP32 accumulation for the FP8
records, one NVFP4 MAC for the FP4 record):
- `W1Prices.sm120`: node 2's RTX PRO 6000 at locked 2,100 MHz (`r20260930-063213-c94a`, `r20260930-063827-e4c6`, every
  SASS, identity and numeric gate passed): QMMA E4M3 m16n8k32 1 per MAC, OMMA NVFP4 m16n8k64 0.5, HMMA BF16 m16n8k16
  2; FADD and FFMA 8.00 issue-bound (8.376 in the measured loop: `W1Prices.sm120Loop`); a cast 16 per F2FP instruction
  (7.7–8.0 per code alone, the shared half-rate integer pipe's 64 lanes per SM per clock); INT32 16 on that pipe.
- `W1Prices.h100`: the datasheet table `Dimension.H100.h100Nominal` (FP8 1, BF16 2 per MAC; FFMA 32; INT32 64) and the
  cast at 32 per code; the `wgmma` shapes `m64nNk32`/`m64nNk16` at `N ≥ 64` only, where Round 7's measured price per
  register meets the nominal one (m64n8k32 measured 144 per register).
- `W1Prices.fp4Sm120`: `sm120Loop` in FP4 units (an FP8 unit is 2), the record `devFp4Sm120` uses.

`sha512` (a compression) and `fetch` (a first-fetched word) are unmeasured on both devices, so the records take them as
parameters. The game's preprocessing is a `QTree` over `H`, which computes anything, SHA-512 included, for free before
the salt; online, `sha512` is the one place W1 charges a compression.

`W1Prices.toPrices` is the coarse record `Device.Prices` that `creditDev` and `wrefDev` read: the FP32 add, the BF16 MAC,
and the forming per element, `fs = 32·mac(E4M3) + cvt/2` (the k32 noise atom and half a cast) and `qa = ⌈fp32/8⌉` (one
FFMA per 8 elements). At `h100`, `sm120` and `sm120Loop` it is `Prices.h100`, `Prices.sm120` and `Prices.sm120Loop`
(`Proofs.Pouw.PearlC.W1Checks`).

**Where the model is narrower than W1**, each a named gap for the restatement:
- the opcode set is the list above: an opcode outside it (MUFU, IMAD.WIDE, HADD2, warp shuffles and votes, atomics)
  has no instruction here, so a program that needs it pays its expansion into these;
- a non-finite FP32 result is the one NaN word `0x7FFFFFFF`, and a BF16 cast's overflow is `0x7FFF`;
- a register address always pays the taint rule, and an `ite` always does: a public address or branch is compiled into
  the program, which is free;
- the BF16 fragment's step is the one-group pipeline `bf16Step` at the device's BF16 `Pipeline`; on sm_120 that is
  `PeelExact.hopperBf16K16`, the stand-in `PeelExactP` uses, not a capture of sm_120's HMMA;
- program text other than the constant bank is free: the code tree is the program's control flow, and unrolled code
  carries no data (every operand is an address).
-/

namespace Pouw.PearlC.W1

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.PeelExact Pouw.PearlC.Fp4 Pouw.PearlC.Fp4Dev

/-! ## Formats, fragments, prices, devices -/

/-- A tensor-core input format. -/
inductive Fmt where
  | e4m3
  | bf16
  | e2m1
  deriving DecidableEq, Repr

/-- A code's width in bits. -/
def Fmt.bits : Fmt → ℕ
  | .e4m3 => 8
  | .bf16 => 16
  | .e2m1 => 4

/-- Codes per 32-bit word. -/
def Fmt.perWord (f : Fmt) : ℕ := 32 / f.bits

/-- A tensor-core fragment: its input format and its `m × n` outputs; its depth is the device's for the format. -/
structure Frag where
  fmt : Fmt
  m : ℕ
  n : ℕ
  deriving DecidableEq, Repr

/-- **W1's prices** at a device, in the record's unit: per MAC of each tensor-core format, per FP32 instruction, per cast
instruction (two codes), per INT32 instruction (and per word under the taint rule), per SHA-512 compression, and per
32-bit word of pre-salt or supplied data on first fetch. -/
structure W1Prices where
  mac : Fmt → ℚ
  fp32 : ℚ
  cvt : ℚ
  int32 : ℚ
  sha512 : ℚ
  fetch : ℚ

/-- **sm_120's W1 prices**, issue-bound (node 2, locked 2,100 MHz; `r20260930-063213-c94a`, `r20260930-063827-e4c6`):
E4M3 1, BF16 2 and NVFP4 0.5 per MAC; FP32 8; a cast 16 per instruction (8 per code); INT32 16. -/
def W1Prices.sm120 (sha fetch : ℚ) : W1Prices where
  mac := fun | .e4m3 => 1 | .bf16 => 2 | .e2m1 => 1 / 2
  fp32 := 8
  cvt := 16
  int32 := 16
  sha512 := sha
  fetch := fetch

/-- sm_120's W1 prices with FP32 at the measured loop's `1047/125 = 8.376`. -/
def W1Prices.sm120Loop (sha fetch : ℚ) : W1Prices := { W1Prices.sm120 sha fetch with fp32 := 1047 / 125 }

/-- **The H100's W1 prices** (`Dimension.H100.h100Nominal`, the datasheet): E4M3 1 and BF16 2 per MAC (the H100 has no
E2M1 fragment, so its price is never read); FP32 32; a cast 64 per instruction (32 per code); INT32 64. -/
def W1Prices.h100 (sha fetch : ℚ) : W1Prices where
  mac := fun | .e4m3 => 1 | .bf16 => 2 | .e2m1 => 1 / 2
  fp32 := 32
  cvt := 64
  int32 := 64
  sha512 := sha
  fetch := fetch

/-- Every price times `c` (a change of unit). -/
def W1Prices.scale (c : ℚ) (pr : W1Prices) : W1Prices :=
  ⟨fun f => c * pr.mac f, c * pr.fp32, c * pr.cvt, c * pr.int32, c * pr.sha512, c * pr.fetch⟩

/-- **sm_120's W1 prices in FP4 units** (an FP8 unit is 2, `Fp4Prices.sm120`): `sm120Loop` doubled, with `sha` and
`fetch` given in FP4 units. -/
def W1Prices.fp4Sm120 (sha fetch : ℚ) : W1Prices :=
  { (W1Prices.sm120Loop 0 0).scale 2 with sha512 := sha, fetch := fetch }

/-- **The coarse record** `creditDev` and `wrefDev` read: the FP32 add, the BF16 MAC, and the forming per element,
`fs = ⌈32·mac(E4M3) + cvt/2⌉` (the noise atom's 32 MACs and half a cast) and `qa = ⌈fp32/8⌉` (ρ's FFMA on every 8th). -/
def W1Prices.toPrices (pr : W1Prices) : Prices :=
  ⟨pr.fp32, pr.mac .bf16, ⟨⌈32 * pr.mac .e4m3 + pr.cvt / 2⌉₊, ⌈pr.fp32 / 8⌉₊⟩⟩

/-- **A W1 device**: its prices, its captured tensor-core steps (E4M3 and BF16 as a `Pipeline`, E2M1 as an `Fp4Atom`),
and the fragments it has. A fragment it lacks is a no-op that costs nothing. -/
structure W1Device where
  prices : W1Prices
  e4m3 : Pipeline
  bf16 : Pipeline
  fp4 : Fp4Atom
  frags : List Frag

/-- A format's depth `k` on the device. -/
def W1Device.depth (dv : W1Device) : Fmt → ℕ
  | .e4m3 => dv.e4m3.groups.sum
  | .bf16 => dv.bf16.groups.sum
  | .e2m1 => dv.fp4.k

/-- sm_120's `mma.sync` fragments: E4M3 m16n8k32 (QMMA.16832), BF16 m16n8k16 (HMMA.16816), NVFP4 m16n8k64
(OMMA.SF.16864, UE4M3 per 16). -/
def sm120Frags : List Frag := [⟨.e4m3, 16, 8⟩, ⟨.bf16, 16, 8⟩, ⟨.e2m1, 16, 8⟩]

/-- The H100's `wgmma` fragments at its nominal per-MAC prices: E4M3 m64nNk32 and BF16 m64nNk16 for `N = 64, 72, …, 256`. -/
def h100Frags : List Frag := (List.range 25).flatMap fun i => [⟨.e4m3, 64, 64 + 8 * i⟩, ⟨.bf16, 64, 64 + 8 * i⟩]

/-- The RTX PRO 6000 (sm_120) at prices `pr`: `sm120E4m3`, the BF16 stand-in `hopperBf16K16`, and `sm120Nvf4`. -/
def W1Device.sm120At (pr : W1Prices) : W1Device := ⟨pr, sm120E4m3, hopperBf16K16, sm120Nvf4, sm120Frags⟩

/-- The H100 at prices `pr`: `hopperE4m3K32` and `hopperBf16K16`; no E2M1 fragment. -/
def W1Device.h100At (pr : W1Prices) : W1Device := ⟨pr, hopperE4m3K32, hopperBf16K16, sm120Nvf4, h100Frags⟩

/-- `devH100`'s setting (`Prices.h100`). -/
def W1Device.h100 (sha fetch : ℚ) : W1Device := W1Device.h100At (W1Prices.h100 sha fetch)

/-- `devSm120v1 Prices.sm120` and `devSm120v2 Prices.sm120`'s setting: both run the same E4M3 step (`G` is the
program's choice, not the machine's). -/
def W1Device.sm120 (sha fetch : ℚ) : W1Device := W1Device.sm120At (W1Prices.sm120 sha fetch)

/-- `devSm120v1 Prices.sm120Loop` and `devSm120v2 Prices.sm120Loop`'s setting. -/
def W1Device.sm120Loop (sha fetch : ℚ) : W1Device := W1Device.sm120At (W1Prices.sm120Loop sha fetch)

/-- `devFp4Sm120`'s setting, in FP4 units. -/
def W1Device.fp4Sm120 (sha fetch : ℚ) : W1Device := W1Device.sm120At (W1Prices.fp4Sm120 sha fetch)

/-! ## Memory -/

/-- An address: a region and an index in it. -/
abbrev Addr : Type := ℕ × ℕ

/-- Memory: a word at every address. -/
abbrev Mem : Type := Addr → ℕ

/-- The address `o` words past `a`, in `a`'s region. -/
def Addr.off (a : Addr) (o : ℕ) : Addr := (a.1, a.2 + o)

/-- The `n` addresses from `a`. -/
def Addr.span (a : Addr) (n : ℕ) : List Addr := (List.range n).map a.off

/-- Byte `i` of the salt. -/
def saltAt (i : ℕ) : Addr := (1, i)

/-- The preprocessed state: index 0 its length, index `i + 1` its entry `i`. -/
def stateAt (i : ℕ) : Addr := (2, i)

/-- Word `i` of the program's constant bank. -/
def constAt (i : ℕ) : Addr := (3, i)

/-- Activation `(i, l)` of unit `u`. -/
def actAt (u i l : ℕ) : Addr := (4, Nat.pair u (Nat.pair i l))

/-- Weight `(j, l)` of weight `w`. -/
def weightAt (w j l : ℕ) : Addr := (5, Nat.pair w (Nat.pair j l))

/-- Ticket word `(i, j)` of unit `u`. -/
def ticketAt (u i j : ℕ) : Addr := (6, Nat.pair u (Nat.pair i j))

/-- Useful word `(i, j)` of unit `u`. -/
def usefulAt (u i j : ℕ) : Addr := (7, Nat.pair u (Nat.pair i j))

/-- Pre-salt or supplied data, priced on first fetch: the state, the constant bank, the activations and the weights. -/
def preSalt (a : Addr) : Bool := a.1 = 2 || a.1 = 3 || a.1 = 4 || a.1 = 5

/-- A word: the low 32 bits. -/
def w32 (x : ℕ) : ℕ := x % 2 ^ 32

/-- The NaN word every non-finite FP32 result writes. -/
def nan32 : ℕ := 0x7FFFFFFF

/-- Field `p` (of width `w` bits) of the word `x`. -/
def field (x w p : ℕ) : ℕ := (x >>> (w * p)) % 2 ^ w

/-- The word `x` with field `p` (of width `w` bits) replaced by `v`. -/
def setField (x w p v : ℕ) : ℕ := x - field x w p * 2 ^ (w * p) + v % 2 ^ w * 2 ^ (w * p)

/-- A state entry as a word: zigzag (`0, −1, 1, −2, …` to `0, 1, 2, 3, …`), low 32 bits. -/
def zig (z : ℤ) : ℕ := w32 (if 0 ≤ z then 2 * z.toNat else 2 * (-z).toNat - 1)

/-- `M` with `n` words from `d` replaced by `f 0, …, f (n − 1)`. -/
def Mem.fill (M : Mem) (d : Addr) (n : ℕ) (f : ℕ → ℕ) : Mem :=
  fun x => if x.1 = d.1 ∧ d.2 ≤ x.2 ∧ x.2 < d.2 + n then w32 (f (x.2 - d.2)) else M x

/-! ## Instruction semantics -/

/-- The INT32 binary operations. -/
inductive Iop where
  | add | sub | mul | mulhi | and | or | xor | shl | shr | lt | eq | min | max

/-- An INT32 operation on two words. -/
def Iop.eval : Iop → ℕ → ℕ → ℕ
  | .add, a, b => w32 (a + b)
  | .sub, a, b => w32 (a + 2 ^ 32 - b)
  | .mul, a, b => w32 (a * b)
  | .mulhi, a, b => w32 (a * b / 2 ^ 32)
  | .and, a, b => a &&& b
  | .or, a, b => a ||| b
  | .xor, a, b => a ^^^ b
  | .shl, a, b => w32 (a <<< (b % 32))
  | .shr, a, b => a >>> (b % 32)
  | .lt, a, b => if a < b then 1 else 0
  | .eq, a, b => if a = b then 1 else 0
  | .min, a, b => Nat.min a b
  | .max, a, b => Nat.max a b

/-- `LOP3`: bit `t` of the result is bit `4·a_t + 2·b_t + c_t` of `lut`. -/
def lop3Word (a b c lut : ℕ) : ℕ :=
  (List.range 32).foldl (fun acc t =>
    acc + (lut >>> (4 * ((a >>> t) % 2) + 2 * ((b >>> t) % 2) + (c >>> t) % 2)) % 2 * 2 ^ t) 0

/-- `PRMT` (default mode): byte `i` of the result is byte `s_i mod 8` of `b:a`, or its sign replicated when bit 3 of the
selector nibble `s_i` is set. -/
def prmtWord (a b s : ℕ) : ℕ :=
  (List.range 4).foldl (fun acc i =>
    let sel := (s >>> (4 * i)) % 16
    let x := ((b * 2 ^ 32 + a) >>> (8 * (sel % 8))) % 256
    acc + (if 8 ≤ sel then (if 128 ≤ x then 255 else 0) else x) * 2 ^ (8 * i)) 0

/-- The FP32 operations. -/
inductive Fop where
  | add | mul | fma | min | max

/-- An FP32 operation on words, round to nearest even; a non-finite operand or result writes `nan32`. -/
def Fop.eval : Fop → ℕ → ℕ → ℕ → ℕ
  | .add, a, b, _ => (addWord? a b).getD nan32
  | .mul, a, b, _ => (fmaWord? a b 0x80000000).getD nan32
  | .fma, a, b, c => (fmaWord? a b c).getD nan32
  | .min, a, b, _ => match wordQ a, wordQ b with
    | some x, some y => if x ≤ y then a else b
    | _, _ => nan32
  | .max, a, b, _ => match wordQ a, wordQ b with
    | some x, some y => if y ≤ x then a else b
    | _, _ => nan32

/-- A cast's code of an FP32 word: `e4m3Word` (saturating), `rnBf16?` (`0x7FFF` past BF16's range), `e2m1Word`. -/
def cvtCode : Fmt → ℕ → ℕ
  | .e4m3, w => (e4m3Word w).val
  | .bf16, w => (rnBf16? w).getD 0x7FFF
  | .e2m1, w => (e2m1Word w).val

/-- A BF16 fragment's step on one group: `acc + Σ x_l·y_l` through the pipeline's adder, as `PeelExact.k16Step`. -/
def bf16Step (p : Pipeline) (acc : ℕ) (x y : List ℕ) : ℕ :=
  packValue (groupStepTotal p.width p.zeroExp (fp32Value acc) (List.zipWith bf16ProductTotal x y))

/-- Words per row of a fragment operand of format `f` at depth `k`. -/
def rowWords (f : Fmt) (k : ℕ) : ℕ := k / f.perWord

/-- Code `l` of row `i` of the operand at `a` (format `f`, depth `k`). -/
def opCode (M : Mem) (f : Fmt) (k : ℕ) (a : Addr) (i l : ℕ) : ℕ :=
  field (M (a.off (i * rowWords f k + l / f.perWord))) f.bits (l % f.perWord)

/-- Scale-byte words per row of an E2M1 operand (four bytes a word). -/
def scaleWords (p : Fp4Atom) : ℕ := (p.scales + 3) / 4

/-- Scale byte `q` of row `i` of the scales at `a`. -/
def opScale (M : Mem) (p : Fp4Atom) (a : Addr) (i q : ℕ) : Fin 256 :=
  ⟨field (M (a.off (i * scaleWords p + q / 4))) 8 (q % 4), Nat.mod_lt _ (by positivity)⟩

/-- Output `(i, j)` of a fragment: `C`'s word through the device's step on row `i` of `A` and row `j` of `B`. -/
def mmaWord (dv : W1Device) (M : Mem) (f : Frag) (a b c sa sb : Addr) (i j : ℕ) : ℕ :=
  let k := dv.depth f.fmt
  let acc := w32 (M (c.off (i * f.n + j)))
  let xs := (List.range k).map (opCode M f.fmt k a i)
  let ys := (List.range k).map (opCode M f.fmt k b j)
  match f.fmt with
  | .e4m3 => dv.e4m3.step acc (xs.map fun x => ⟨x % 256, Nat.mod_lt _ (by decide)⟩)
      (ys.map fun y => ⟨y % 256, Nat.mod_lt _ (by decide)⟩)
  | .bf16 => bf16Step dv.bf16 acc xs ys
  | .e2m1 => (dv.fp4.step acc (xs.map fun x => ⟨x % 16, Nat.mod_lt _ (by decide)⟩)
      (ys.map fun y => ⟨y % 16, Nat.mod_lt _ (by decide)⟩)
      ((List.range dv.fp4.scales).map (opScale M dv.fp4 sa i)) ((List.range dv.fp4.scales).map (opScale M dv.fp4 sb j))).getD nan32

/-- A 64-bit word from two 32-bit words at `a` (high first). -/
def word64 (M : Mem) (a : Addr) (t : ℕ) : ℕ := w32 (M (a.off (2 * t))) * 2 ^ 32 + w32 (M (a.off (2 * t + 1)))

/-- **An instruction.** Operands are addresses; `ld` and `st` address `(rg, M r + off)`. -/
inductive Instr where
  | mov (d a : Addr)
  | ld (d : Addr) (rg : ℕ) (r : Addr) (off : ℕ)
  | st (rg : ℕ) (r : Addr) (off : ℕ) (s : Addr)
  | iop (o : Iop) (d a b : Addr)
  | iadd3 (d a b c : Addr)
  | lop3 (d a b c : Addr) (lut : ℕ)
  | prmt (d a b s : Addr)
  | sel (d p a b : Addr)
  | fop (o : Fop) (d a b c : Addr)
  /-- Two casts, `a`'s code low, packed into field `pos` of `d`'s word (the rest of `d` kept). -/
  | cvt (f : Fmt) (d a b : Addr) (pos : ℕ)
  /-- `D = A·Bᵀ + C` on `f`'s `m × n` words (row-major at `d` and `c`); `A`'s `m` rows and `B`'s `n` rows of packed
  codes at `a` and `b`, and for E2M1 their scale bytes at `sa` and `sb`. -/
  | mma (f : Frag) (d a b c sa sb : Addr)
  /-- One SHA-512 compression: the state's 8 words at `h` and the block's 16 at `b` (each 64-bit word two 32-bit
  words, high first), the new state at `d`. -/
  | sha (d h b : Addr)

/-- The device has the fragment. -/
def W1Device.has (dv : W1Device) (f : Frag) : Bool := f ∈ dv.frags

/-- The memory after the instruction. -/
def Instr.exec (dv : W1Device) (M : Mem) : Instr → Mem
  | .mov d a => Function.update M d (w32 (M a))
  | .ld d rg r off => Function.update M d (w32 (M (rg, w32 (M r) + off)))
  | .st rg r off s => Function.update M (rg, w32 (M r) + off) (w32 (M s))
  | .iop o d a b => Function.update M d (o.eval (w32 (M a)) (w32 (M b)))
  | .iadd3 d a b c => Function.update M d (w32 (M a + M b + M c))
  | .lop3 d a b c lut => Function.update M d (lop3Word (w32 (M a)) (w32 (M b)) (w32 (M c)) lut)
  | .prmt d a b s => Function.update M d (prmtWord (w32 (M a)) (w32 (M b)) (w32 (M s)))
  | .sel d p a b => Function.update M d (if w32 (M p) = 0 then w32 (M b) else w32 (M a))
  | .fop o d a b c => Function.update M d (w32 (o.eval (w32 (M a)) (w32 (M b)) (w32 (M c))))
  | .cvt f d a b pos => Function.update M d
      (w32 (setField (w32 (M d)) (2 * f.bits) (pos % (32 / (2 * f.bits)))
        (cvtCode f (w32 (M a)) + cvtCode f (w32 (M b)) * 2 ^ f.bits)))
  | .mma f d a b c sa sb => if dv.has f then M.fill d (f.m * f.n) (fun o => mmaWord dv M f a b c sa sb (o / f.n) (o % f.n))
      else M
  | .sha d h b =>
      let out := Sha512.compress ((List.range 8).map (word64 M h)) ((List.range 16).map (word64 M b))
      M.fill d 16 fun o => if o % 2 = 0 then out.getD (o / 2) 0 / 2 ^ 32 else out.getD (o / 2) 0

/-- The addresses the instruction reads. -/
def Instr.reads (dv : W1Device) (M : Mem) : Instr → List Addr
  | .mov _ a => [a]
  | .ld _ rg r off => [r, (rg, w32 (M r) + off)]
  | .st _ r _ s => [r, s]
  | .iop _ _ a b => [a, b]
  | .iadd3 _ a b c => [a, b, c]
  | .lop3 _ a b c _ => [a, b, c]
  | .prmt _ a b s => [a, b, s]
  | .sel _ p a b => [p, a, b]
  | .fop _ _ a b c => [a, b, c]
  | .cvt _ d a b _ => [a, b, d]
  | .mma f _ a b c sa sb =>
      if dv.has f then
        let w := rowWords f.fmt (dv.depth f.fmt)
        a.span (f.m * w) ++ b.span (f.n * w) ++ c.span (f.m * f.n) ++
          (if f.fmt = .e2m1 then sa.span (f.m * scaleWords dv.fp4) ++ sb.span (f.n * scaleWords dv.fp4) else [])
      else []
  | .sha _ h b => h.span 16 ++ b.span 32

/-- The words the instruction writes. -/
def Instr.words (dv : W1Device) : Instr → ℕ
  | .mma f _ _ _ _ _ _ => if dv.has f then f.m * f.n else 0
  | .sha _ _ _ => 16
  | _ => 1

/-- The instruction's address is a register's value (data, under W1's taint rule). -/
def Instr.dynamic : Instr → Bool
  | .ld _ _ _ _ => true
  | .st _ _ _ _ => true
  | _ => false

/-- The instruction's class price: free moves; INT32; FP32; a cast; a fragment's `m·n·k` MACs; a compression. -/
def Instr.base (dv : W1Device) : Instr → ℚ
  | .mov _ _ => 0
  | .ld _ _ _ _ => 0
  | .st _ _ _ _ => 0
  | .iop _ _ _ _ => dv.prices.int32
  | .iadd3 _ _ _ _ => dv.prices.int32
  | .lop3 _ _ _ _ _ => dv.prices.int32
  | .prmt _ _ _ _ => dv.prices.int32
  | .sel _ _ _ _ => dv.prices.int32
  | .fop _ _ _ _ _ => dv.prices.fp32
  | .cvt _ _ _ _ _ => dv.prices.cvt
  | .mma f _ _ _ _ _ _ => if dv.has f then (f.m * f.n * dv.depth f.fmt : ℕ) * dv.prices.mac f.fmt else 0
  | .sha _ _ _ => dv.prices.sha512

/-- **W1's price of an instruction**: its class price, plus one INT32 instruction per word it writes when its address
or an enclosing branch depends on data (`taint`). -/
def Instr.price (dv : W1Device) (taint : Bool) (i : Instr) : ℚ :=
  i.base dv + if taint || i.dynamic then dv.prices.int32 * i.words dv else 0

/-! ## Programs and runs -/

/-- A code tree: stop, an instruction then the rest, or a branch on a word (nonzero: `t`, else `e`) then the rest. -/
inductive Code where
  | done
  | op (i : Instr) (k : Code)
  | ite (p : Addr) (t e k : Code)

/-- A program: a constant bank and a code tree, each a function of the public layout. -/
structure Prog where
  consts : Layout → List ℕ
  code : Layout → Code

/-- A run's state: the memory, the pre-salt and supplied addresses read so far, and the instructions' cost. -/
structure St where
  mem : Mem
  fetched : List Addr
  cost : ℚ

/-- **Execution** under the taint flag (set inside a branch on data). -/
def exec (dv : W1Device) : Bool → Code → St → St
  | _, .done, st => st
  | t, .op i k, st =>
      exec dv t k ⟨i.exec dv st.mem, st.fetched ++ (i.reads dv st.mem).filter preSalt, st.cost + i.price dv t⟩
  | t, .ite p a b k, st =>
      let st1 : St := ⟨st.mem, st.fetched ++ [p].filter preSalt, st.cost + dv.prices.int32⟩
      exec dv t k (if w32 (st.mem p) = 0 then exec dv true b st1 else exec dv true a st1)

/-- The initial memory: the salt's bytes, the state, the constant bank, the activations and the weights, as words. -/
def init (salt : List ℕ) (pr : Prog) (U : Workload) (act : ℕ → Codes) (σ : State) : Mem := fun a =>
  let x := Nat.unpair a.2
  let y := Nat.unpair x.2
  if a.1 = 1 then salt.getD a.2 0 % 256
  else if a.1 = 2 then (if a.2 = 0 then w32 σ.length else zig (σ.getD (a.2 - 1) 0))
  else if a.1 = 3 then w32 ((pr.consts U.layout).getD a.2 0)
  else if a.1 = 4 then w32 (act x.1 y.1 y.2)
  else if a.1 = 5 then w32 (U.weight x.1 y.1 y.2)
  else 0

/-- The committed transcript of a final memory: every declared unit's activations (within its shape as the program
left them, outside it as supplied), ticket words and useful words. -/
def readout (U : Workload) (act : ℕ → Codes) (M : Mem) : Transcript := fun u =>
  if u < U.layout.N then
    some ⟨fun i l => if i < (U.layout.shape u).m ∧ l < (U.layout.shape u).k then M (actAt u i l) else act u i l,
      fun i j => M (ticketAt u i j), fun i j => M (usefulAt u i j)⟩
  else none

/-- The final state of a run. -/
def final (dv : W1Device) (salt : List ℕ) (pr : Prog) (U : Workload) (act : ℕ → Codes) (σ : State) : St :=
  exec dv false (pr.code U.layout) ⟨init salt pr U act σ, [], 0⟩

variable {Q R S : Type}

/-- A first fetch W1 does not charge: an activation within a correct unit's shape, or a weight row and column a correct
unit of that weight reads. -/
def FreeFetch (P : Protocol Q R S) (U : Workload) (H : Q → R) (s : S) (τ : Transcript) (a : Addr) : Prop :=
  let x := Nat.unpair a.2
  let y := Nat.unpair x.2
  (a.1 = 4 ∧ x.1 ∈ correctSet P U H s τ ∧ y.1 < (U.layout.shape x.1).m ∧ y.2 < (U.layout.shape x.1).k) ∨
    (a.1 = 5 ∧ ∃ u ∈ correctSet P U H s τ, U.layout.wid u = x.1 ∧ y.1 < (U.layout.shape u).n ∧ y.2 < (U.layout.shape u).k)

open Classical in
/-- **The cost model W1** on the device `dv`, for the protocol `P` (whose `Correct` decides which operand fetches are
free) and the salt's bytes `salt`: the instructions' prices plus `fetch` per distinct pre-salt or supplied word read and
not free. No oracle call. -/
noncomputable def w1 (dv : W1Device) (P : Protocol Q R S) (salt : S → List ℕ) : CostModel Q R S where
  Prog := Prog
  run := fun pr _ U act σ s => readout U act (final dv (salt s) pr U act σ).mem
  cost := fun pr H U act σ s =>
    let st := final dv (salt s) pr U act σ
    ((st.cost + dv.prices.fetch *
      (st.fetched.dedup.filter fun a => ¬ FreeFetch P U H s (readout U act st.mem) a).length : ℚ) : ℝ)
  calls := fun _ _ _ _ _ _ => 0

/-- The words a run fetches first: its distinct pre-salt or supplied words, every one charged, none refunded. This is
the run's device-memory read as the bandwidth account counts it, in 32-bit words. -/
def w1Read (dv : W1Device) (salt : List ℕ) (pr : Prog) (U : Workload) (act : ℕ → Codes) (σ : State) : ℕ :=
  (final dv salt pr U act σ).fetched.dedup.length

/-- **W1 priced, with no refund** (Daniel, item 9): a run's compute, the instructions' prices (`final`'s `cost`), plus
`fetch` per word it reads (`w1Read`). Compute goes to the compute account and the read to the bandwidth account, and the
γ-gap property is stated over this priced sum. It reads neither the oracle nor the protocol. -/
noncomputable def w1Priced (dv : W1Device) (salt : S → List ℕ) : CostModel Q R S where
  Prog := Prog
  run := fun pr _ U act σ s => readout U act (final dv (salt s) pr U act σ).mem
  cost := fun pr _ U act σ s =>
    (((final dv (salt s) pr U act σ).cost + dv.prices.fetch * w1Read dv (salt s) pr U act σ : ℚ) : ℝ)
  calls := fun _ _ _ _ _ _ => 0

end Pouw.PearlC.W1
