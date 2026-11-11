module

public import Evaluate.Fp32
public import Evaluate.Casts
public import Evaluate.TC.Dots
public import Flock.Hash
public import Flock.HmRow
public import PouwPearlPlus.Records

/-!
# A tile's words under Pearl-C v1 -h3 on the H100 (`pearl_c.PearlC.encode`, `checked`, `leaf`)

The words a drawn tile's openings determine, each step in the Python's order and exact semantics (`pearl_c`, `pearl_kw`;
FP32 steps `verity.definitions.silicon.fp32`, here `Evaluate.Fp32`'s; the FP8 atom and BF16 step the H100's,
`HOPPER_E4M3_K32` and `HOPPER_BF16_WGMMA_K16`, here `Evaluate.TC`'s):

* **Noise.** -h3's `s512(ℓ, m)` is the first 32 bytes of SHA-512 over `pearl-c/h3s/ℓ\0` zero-padded to 128 bytes, then `m`.
  The fixed operand's seed is `seed_B = s512("seed-B", salt ‖ root_B)`; row i of A seeds its own E_A line from its row digest
  (`row_seed`, `s512("seed-A", salt ‖ digest ‖ root_B ‖ (index·2³² + i) u64 LE)`). A noise line (`sample_line`) is 32 signed
  integers from `s512("line", s512("noise-line", seed) ‖ side ‖ factor ‖ line u32 LE)`, scaled in BF16 to its norm and cast
  to E4M3. E lines are at norm 256; F_A and F_B are k basis lines each at the line norm 16.
* **Forming v1 (§5)**: per row `s5_stats` (s, ρ, α, β), E′ = e4m3(β·E), and per element one atom with α·x as its
  accumulator over E′ and F's line, then the cast.
* **The words**: C̃ = the chain of A′ and B̃ (windows of 4 atoms from +0, each promoted by one FP32 add, −0 to +0), and U
  from C̃ by four BF16 k16 steps over the peel factors P_A = [bf16(E′_A) ‖ RNE_bf16(A′·F_B)] and
  P_B = [RNE_bf16(bf16(E′_B)·G − B̃·F_A) ‖ −bf16(E′_B)], G = F_Bᵀ·F_A.
* **The leaf**: `sha512/row-seg/v1` over the (row, 32-column chunk) messages, le32 C̃ ‖ le32 U.

`Reject` is Pearl's verifier's rejection (a non-finite value): the tile's inputs are inadmissible.
-/

@[expose] public section

namespace PouwCheck.PearlPlus.Words

open Evaluate.Fp32 Evaluate.Casts Evaluate.TC Flock

abbrev Reject := Except String

def finite (w : Nat) : Reject Nat :=
  if (w >>> 23) &&& 0xFF == 0xFF then throw "non-finite FP32 accumulator" else pure w

/-- `pearl_kw.f32_to_bf16` of an FP32 word: nearest-even, refusing a non-finite input or a BF16 overflow. -/
def toBf16 (w : Nat) : Reject Nat := do
  let w ← finite w
  let out := ((w + 0x7FFF + ((w >>> 16) &&& 1)) >>> 16) &&& 0xFFFF
  if out &&& 0x7FFF ≥ 0x7F80 then throw "bf16 overflow" else pure out

/-- `pearl_kw.bf16_to_f32` as an FP32 word. -/
def ofBf16 (c : Nat) : Reject Nat :=
  if (c >>> 7) &&& 0xFF == 0xFF then throw "non-finite bf16" else pure (c <<< 16)

/-- `pearl_kw.f32_to_fp8` of an FP32 word: nearest-even, saturating to ±448. -/
def toFp8 (w : Nat) : Reject Nat := do pure (f32ToE4m3Sat (← finite w))

/-- `pearl_kw.fp8_to_f32` as an FP32 word (exact). -/
def ofFp8 (c : Nat) : Reject Nat :=
  if c &&& 0x7F == 0x7F then throw "the E4M3 NaN code" else pure (e4m3ToF32 c)

/-- `pearl_kw.bf16_mul`: one FP32 product of the BF16 values, rounded to BF16. -/
def bf16Mul (a b : Nat) : Reject Nat := do toBf16 (← finite (f32MulV2 (← ofBf16 a) (← ofBf16 b)))

/-- `pearl_kw.bf16_div`: the FP32 quotient (f64's then f32's rounding is the FP32 one), rounded to BF16. -/
def bf16Div (a b : Nat) : Reject Nat := do
  let x ← ofBf16 a
  let y ← ofBf16 b
  if y &&& 0x7FFFFFFF == 0 then throw "division by zero"
  toBf16 (← finite (f32DivV2 x y))

/-- The FP32 word of an integer of magnitude below 2^24 (exact). -/
def ofInt (x : Int) : Nat :=
  let a := x.natAbs
  if a = 0 then 0
  else
    let e := a.log2
    (if x < 0 then 0x80000000 else 0) ||| ((e + 127) <<< 23) ||| ((a - 2 ^ e) <<< (23 - e))

def bytesOf (xs : List Nat) : ByteArray := ⟨xs.toArray.map (·.toUInt8)⟩

/-- -h3's labelled hash: `SHA-512(pearl-c/h3s/<role>\0 ‖ zeros to 128 bytes ‖ msg)[:32]`. -/
def s512 (role : String) (msg : ByteArray) : ByteArray :=
  let label := ("pearl-c/h3s/" ++ role).toUTF8.push 0
  (Sha512.hash (label ++ ⟨Array.replicate (128 - label.size) 0⟩ ++ msg)).extract 0 32

def le (w x : Nat) : ByteArray := ⟨(Array.range w).map fun i => (x / 256 ^ i % 256).toUInt8⟩

/-- `seed_b(salt, root_b, "h3")`. -/
def seedB (salt rootB : ByteArray) : ByteArray := s512 "seed-B" (salt ++ rootB)

/-- `row_seed(salt, row_digest(row), root_b, unit, row)`: the row's 64-byte `sha512/row-seg/v1` digest keys its seed. -/
def rowSeed (salt rootB : ByteArray) (unit i : Nat) (rowBytes : ByteArray) : ByteArray :=
  s512 "seed-A" (salt ++ HmRow.segDigest (8 * rowBytes.size) rowBytes ++ rootB ++ le 8 (unit * 2 ^ 32 + i))

/-- The BF16 words of `norm · 32` at the two norms: E's 256 and F's line norm 16. -/
def NORM_E : Nat := 0x4600
def NORM_F : Nat := 0x4400

/-- `sample_line(seed, side, factor, line, norm)` for an -h3 seed, `norm32` the BF16 word of `norm · 32`. -/
def sampleLine (lineKey : ByteArray) (side factor line norm32 : Nat) : Reject (Array Nat) := do
  let raw := s512 "line" (lineKey ++ bytesOf [side, factor] ++ le 4 line)
  let xs : Array Int := raw.toList.toArray.map fun b =>
    (if b.toNat >>> 7 = 1 then -1 else 1) * ((b.toNat &&& 0x7F : Nat) + 1 : Int)
  let sq := xs.foldl (fun s x => s + x.natAbs * x.natAbs) 0
  let scale ← bf16Div norm32 (← toBf16 (ofInt (isqrt (sq * 32 * 32))))
  xs.mapM fun x => do toFp8 (← ofBf16 (← bf16Mul (← toBf16 (ofInt x)) scale))

def lineKey (seed : ByteArray) : ByteArray := s512 "noise-line" seed

/-- F_A (side 0) or F_B (side 1): k basis lines of 32 codes at the line norm. -/
def basis (seedB : ByteArray) (side k : Nat) : Reject (Array (Array Nat)) :=
  let key := lineKey seedB
  (Array.range k).mapM fun t => sampleLine key side 1 t NORM_F

def column (b : Array (Array Nat)) (p : Nat) : Array Nat := b.map (·[p]!)

def slice (a : Array Nat) (s w : Nat) : List Nat := (a.extract s (s + w)).toList

/-- The H100's FP8 atom: one k32 step of E4M3 codes into an FP32 accumulator. -/
def atom (acc : Nat) (a b : List Nat) : Nat := hopperE4m3Dot32 acc a b

/-- `pearl_c.chain(a, b)` on the H100: per window of 128 products a fresh +0 accumulator chains four atoms, promoted into
the total by one nearest-even FP32 add; a zero total is +0. -/
def chain (a b : Array Nat) : Reject Nat := do
  if a.size != b.size || a.size % 32 != 0 then throw "the chain multiplies in atoms of k = 32"
  let mut total := 0
  for ws in [0:a.size:WINDOW] do
    let mut win := 0
    for s in [ws:min (ws + WINDOW) a.size:32] do
      win ← finite (atom win (slice a s 32) (slice b s 32))
    total ← finite (f32AddV2 total win)
    if total == 0x80000000 then total := 0
  return total

/-- `pearl_c.peel`: U from C̃, four BF16 k16 steps over the 64 peel operands. -/
def peel (c : Nat) (pa pb : Array Nat) : Reject Nat := do
  let mut acc := c
  for s in [0:PEEL_K:16] do
    acc ← finite (hopperBf16Dot16 acc (slice pa s 16) (slice pb s 16))
  return acc

def S5_FLOOR : Nat := 0x2F800000
def S5_FOUR : Nat := 0x40800000
def S5_QMAX : Nat := 0x43E00000
/-- `S5_LINE = f32(√32 / (256 · 16))`. -/
def S5_LINE : Nat := 0x3AB504F3

/-- `s5_stats(x)`: (s, ρ, α, β_row) of one row of FP32 words, in the pinned order: s = max |x|, ρ² over every 8th position
in 32 partial FMA chains folded by a butterfly, α = 448/max(s + 4ρ, 2⁻³²), β = ρ·α·`S5_LINE`. -/
def s5Stats (x : Array Nat) : Reject (Nat × Nat × Nat × Nat) := do
  let s := x.foldl (fun m w => max m (w &&& 0x7FFFFFFF)) 0
  let mut part := Array.replicate 32 0
  let n := (x.size + 7) / 8
  for i in [0:n] do
    let w := x[8 * i]!
    part := part.set! (i % 32) (← finite (f32FmaV2 w w part[i % 32]!))
  for o in [16, 8, 4, 2, 1] do
    let prev := part
    part ← (Array.range 32).mapM fun j => finite (f32AddV2 prev[j]! prev[j ^^^ o]!)
  let rho := sqrtRn (f32DivV2 part[0]! (ofInt n))
  let t ← finite (f32AddV2 s (f32MulV2 S5_FOUR rho))
  let alpha := f32DivV2 S5_QMAX (max t S5_FLOOR)
  return (s, rho, alpha, ← finite (f32MulV2 (f32MulV2 rho alpha) S5_LINE))

/-- One side formed by v1: each row's A′ (or B̃) codes, α, ρ, β, E′ and its FP32 words. -/
structure Formed where
  codes : Array (Array Nat)
  alpha : Array Nat
  rho : Array Nat
  beta : Array Nat
  ep : Array (Array Nat)
  x : Array (Array Nat)
  deriving Inhabited

/-- `form_v1(rows, e_rows, f)`: §5's forming, per element one atom with α·x as its accumulator over E′ and F's line. -/
def formV1 (rows : Array (Array Nat)) (eRows : Array (Array Nat)) (f : Array (Array Nat)) : Reject Formed := do
  let mut out : Formed := ⟨#[], #[], #[], #[], #[], #[]⟩
  for (x, e) in rows.zip eRows do
    let (_, rho, alpha, beta) ← s5Stats x
    let ep ← e.mapM fun c => do toFp8 (f32MulV2 beta (← ofFp8 c))
    let epl := ep.toList
    let codes ← (Array.range x.size).mapM fun t => do toFp8 (← finite (atom (f32MulV2 alpha x[t]!) epl f[t]!.toList))
    out := { codes := out.codes.push codes, alpha := out.alpha.push alpha, rho := out.rho.push rho,
             beta := out.beta.push beta, ep := out.ep.push ep, x := out.x.push x }
  return out

/-- `parse_f32_row`: a row's k FP32 words, each finite. -/
def parseRow (row : ByteArray) (k : Nat) : Reject (Array Nat) := do
  if row.size != 4 * k then throw "a v1 row has k FP32 words"
  (Array.range k).mapM fun i =>
    finite (row[4 * i]!.toNat ||| (row[4 * i + 1]!.toNat <<< 8) ||| (row[4 * i + 2]!.toNat <<< 16) |||
      (row[4 * i + 3]!.toNat <<< 24))

/-- What one fixed operand's noise reads, once per fixed operand and epoch: seed_B, F_A and F_B as columns, and G. -/
structure FixedNoise where
  seedB : ByteArray
  fa : Array (Array Nat)
  fb : Array (Array Nat)
  faCols : Array (Array Nat)
  fbCols : Array (Array Nat)
  gram : Array (Array Nat)
  deriving Inhabited

def fixedNoise (salt rootB : ByteArray) (k : Nat) : Reject FixedNoise := do
  let sb := seedB salt rootB
  let fa ← basis sb 0 k
  let fb ← basis sb 1 k
  let faCols := (Array.range R).map (column fa)
  let fbCols := (Array.range R).map (column fb)
  let gram ← fbCols.mapM fun fbp => faCols.mapM fun faq => chain fbp faq
  return ⟨sb, fa, fb, faCols, fbCols, gram⟩

def bf16OfCode (c : Nat) : Reject Nat := do toBf16 (← ofFp8 c)

/-- `peel_factors_a`: P_A rows, [bf16(E′) ‖ RNE_bf16(A′·F_B)]. -/
def peelA (wn : FixedNoise) (side : Formed) : Reject (Array (Array Nat)) :=
  (side.codes.zip side.ep).mapM fun (row, ep) => do
    let be ← ep.mapM bf16OfCode
    let fused ← wn.fbCols.mapM fun c => do toBf16 (← chain row c)
    pure (be ++ fused)

/-- `peel_factors_b`: P_B rows, [RNE_bf16(be·G − B̃·F_A) ‖ −be], be·G the FP32 FMA chain from +0 in p order. -/
def peelB (wn : FixedNoise) (side : Formed) : Reject (Array (Array Nat)) :=
  (side.codes.zip side.ep).mapM fun (row, ep) => do
    let be ← ep.mapM bf16OfCode
    let mid ← (Array.range R).mapM fun q => do
      let mut acc := 0
      for p in [0:R] do
        acc ← finite (f32FmaV2 (be[p]! <<< 16) wn.gram[p]![q]! acc)
      toBf16 (← finite (f32AddV2 acc ((← chain row wn.faCols[q]!) ^^^ 0x80000000)))
    pure (mid ++ be.map (· ^^^ 0x8000))

/-- A tile's formed operands and peel factors (`pearl_c.Ops`). -/
structure Ops where
  a : Formed
  b : Formed
  pa : Array (Array Nat)
  pb : Array (Array Nat)
  deriving Inhabited

/-- `PearlC.encode(ctx, noise, a_rows, b_rows, tile)` on the tile's opened rows (values less their salts): `none` where
Pearl rejects. -/
def encode (salt rootB : ByteArray) (wn : FixedNoise) (index k : Nat) (rows : List Nat) (aRows : List ByteArray)
    (cols : List Nat) (bRows : List ByteArray) : Reject Ops := do
  let ea ← (rows.zip aRows).toArray.mapM fun (i, r) => sampleLine (lineKey (rowSeed salt rootB index i r)) 0 0 0 NORM_E
  let a ← formV1 (← aRows.toArray.mapM (parseRow · k)) ea wn.fa
  let keyB := lineKey wn.seedB
  let eb ← cols.toArray.mapM fun j => sampleLine keyB 1 0 j NORM_E
  let b ← formV1 (← bRows.toArray.mapM (parseRow · k)) eb wn.fb
  return ⟨a, b, ← peelA wn a, ← peelB wn b⟩

/-- The tile's checked words: C̃ and U, row by row. -/
structure Checked where
  c : Array (Array Nat)
  u : Array (Array Nat)
  deriving Inhabited

def checked (ops : Ops) : Reject Checked := do
  let c ← ops.a.codes.mapM fun ra => ops.b.codes.mapM fun rb => chain ra rb
  let u ← (Array.range c.size).mapM fun i => (Array.range ops.pb.size).mapM fun j => peel c[i]![j]! ops.pa[i]! ops.pb[j]!
  return ⟨c, u⟩

def CHUNK : Nat := 32

def le32s (xs : Array Nat) : ByteArray := xs.foldl (fun acc x => acc ++ le 4 x) ByteArray.empty

/-- `leaf(checked, "h3")`: `sha512/row-seg/v1` over the (row, 32-column chunk) messages, le32 C̃ ‖ le32 U. -/
def leaf (ch : Checked) : ByteArray :=
  let joined := (ch.c.zip ch.u).foldl (fun acc (crow, urow) =>
    (List.range ((crow.size + CHUNK - 1) / CHUNK)).foldl (fun acc h =>
      acc ++ le32s (crow.extract (CHUNK * h) (CHUNK * h + CHUNK)) ++ le32s (urow.extract (CHUNK * h) (CHUNK * h + CHUNK)))
      acc) ByteArray.empty
  HmRow.segDigest (8 * joined.size) joined

end PouwCheck.PearlPlus.Words
