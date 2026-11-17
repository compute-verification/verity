module

public import Mathlib.Data.Matrix.ColumnRowPartitioned
public import Definitions.Pouw.Fp8Atom.Fp32

/-! Pearl-C's Π, the parts other files state things about, with `σ = 1`, `G = 4` and `r = 32`. -/

@[expose] public section

namespace Pouw.PearlC

open Pouw.Fp8Atom Matrix

/-- Pearl-C's parameters: the noise scale `σ` relative to the row RMS (Pearl's spec calls it `δ`), the promotion depth
`G` in atoms, the noise rank `r`. -/
structure Params where
  σ : ℚ
  G : Nat
  r : Nat

/-- The values Π pins: `σ = 1`, promotion every `G = 4` atoms, rank `r = 32`. -/
def Params.pi : Params := ⟨1, 4, 32⟩

/-- A GEMM shape: `A` is `m × k`, `B` is `n × k`, and `C̃ = A′·B̃ᵀ` is `m × n`. -/
structure Shape where
  m : Nat
  n : Nat
  k : Nat

def Shape.Tiled (p : Params) (s : Shape) : Prop := 32 * p.G ∣ s.k

/-- `x` has a `p`-bit significand at some exponent (BF16 is `p = 8`, E4M3 `p = 4`, and `Representable` is `p = 24`). -/
def HasSig (p : Nat) (x : ℚ) : Prop := ∃ m e : ℤ, |m| < 2 ^ p ∧ x = m * 2 ^ e

/-- The magnitude bits of E4M3 RNE-with-saturation for `t ≥ 0`: `0` at `0`, `126` (448) from `464` up, else `8·(E + 6) + roundEven (t / 2^(E − 3))` with `E = max (binade t) (−6)`. -/
def e4m3Mag (t : ℚ) : Nat :=
  if t = 0 then 0
  else if 464 ≤ t then 126
  else
    let E := max (binade t) (-6)
    (8 * (E + 6) + roundEven (t / 2 ^ (E - 3))).toNat

/-- The E4M3 code of a rational: the sign bit for `x < 0`, then `e4m3Mag |x|`; it saturates to `±448` and is never NaN. -/
def e4m3 (x : ℚ) : Code := ⟨((if x < 0 then 128 else 0) + e4m3Mag |x|) % 256, Nat.mod_lt _ (by decide)⟩

/-- `verity.ml.tc.cast.f32_to_e4m3_sat_word`: NaN to `0x7F`, `±inf` to `±448`, a finite word to its sign bit and `e4m3Mag |value|`. -/
def e4m3Word (u : Nat) : Code :=
  let s := if (u >>> 31) % 2 = 1 then 128 else 0
  match wordQ u with
  | some x => ⟨(s + e4m3Mag |x|) % 256, Nat.mod_lt _ (by decide)⟩
  | none => if u % 2 ^ 23 = 0 then ⟨(s + 126) % 256, Nat.mod_lt _ (by decide)⟩ else ⟨127, by decide⟩

/-- A code's exact value; the NaN codes, which `e4m3` never produces, read as `0`. -/
def codeVal (c : Code) : ℚ := (decode c).getD 0

def vals {ι κ : Type} (M : Matrix ι κ Code) : Matrix ι κ ℚ := M.map codeVal

def ulpBf16 (x : ℚ) : ℚ := 2 ^ (binade |x| - 7)

/-- `cvt.rn.bf16.f32` on values with no exponent range, so exact where the result is finite and normal. -/
def bf16Rn (x : ℚ) : ℚ := (if x < 0 then -1 else 1) * (roundEven (|x| / ulpBf16 x) : ℚ) * ulpBf16 x

structure EncVec where
  u : Nat
  c : Nat

def EncVec.holds (v : EncVec) : Bool :=
  (e4m3Word v.u).val == v.c &&
    match wordQ v.u with
    | some x => x == 0 || (e4m3 x).val == v.c
    | none => true

structure Bf16Vec where
  u : Nat
  h : Nat

def Bf16Vec.holds (v : Bf16Vec) : Bool :=
  match wordQ v.u, wordQ (v.h <<< 16) with
  | some x, some y => decide (bf16Rn x = y)
  | _, _ => false

/-- The FP32 word of `x` rounded to nearest even with IEEE binary32's range: `+0` at `0`, subnormals below `2^−126`
(a negative value that underflows is `−0`), and `±inf` once the rounded magnitude reaches `2^128`. On the exact product
of two FP32 values it is the word an FP32 FMUL writes. -/
def f32Word (x : ℚ) : Nat :=
  let s := if x < 0 then 2 ^ 31 else 0
  let a := |x|
  if a = 0 then 0
  else
    let E₀ := max (binade a) (-126)
    let N₀ := (roundEven (a / 2 ^ (E₀ - 23))).toNat
    let E := if N₀ = 2 ^ 24 then E₀ + 1 else E₀
    let N := if N₀ = 2 ^ 24 then 2 ^ 23 else N₀
    if 127 < E then s + 0x7F800000
    else if N < 2 ^ 23 then s + N
    else s + (E + 127).toNat * 2 ^ 23 + (N - 2 ^ 23)

/-- The number the cast reads from an FP32 word: its value when finite; `±480` for `±inf` (and for NaN, which no chain
here writes), which `e4m3` saturates to `±448` as `e4m3Word` does `±inf`. -/
def wordVal (w : Nat) : ℚ :=
  match wordQ w with
  | some v => v
  | none => if (w >>> 31) % 2 = 1 then -480 else 480

structure F32Vec where
  a : Nat
  b : Nat
  p : Nat

/-- `a` and `b` are nonzero because an exact zero has no sign: `f32Word 0` is `+0` where a signed zero operand gives `−0`. -/
def F32Vec.holds (v : F32Vec) : Bool :=
  match wordQ v.a, wordQ v.b with
  | some x, some y => x != 0 && y != 0 && f32Word (x * y) == v.p
  | _, _ => false

/-- One unit's noise factors as E4M3 codes, as Pearl's lines are (`sample_line`): `E_A` (`m × r`), `F_A` (`k × r`),
`E_B` (`n × r`) and `F_B` (`k × r`). -/
structure Noise (m n k r : Nat) where
  EA : Matrix (Fin m) (Fin r) Code
  FA : Matrix (Fin k) (Fin r) Code
  EB : Matrix (Fin n) (Fin r) Code
  FB : Matrix (Fin k) (Fin r) Code

variable {m n k r : Nat}

def Noise.NA (z : Noise m n k r) : Matrix (Fin m) (Fin k) ℚ := vals z.EA * (vals z.FA)ᵀ

def Noise.NB (z : Noise m n k r) : Matrix (Fin n) (Fin k) ℚ := vals z.EB * (vals z.FB)ᵀ

def Noise.gram (z : Noise m n k r) : Matrix (Fin r) (Fin r) ℚ := (vals z.FB)ᵀ * vals z.FA

def rowScale {p q : Nat} (β : Fin p → ℚ) (M : Matrix (Fin p) (Fin q) ℚ) : Matrix (Fin p) (Fin q) ℚ := diagonal β * M

/-- A row's mean square, `rms²`. -/
def meanSq (x : Fin k → ℚ) : ℚ := (∑ l, x l ^ 2) / k

/-- The row statistic in use for the noise scale, as a square: `st k x` is `ρ²` for a row `x` of length `k`. It is
stated squared so that it stays rational; `betaOf` takes the square root as a parameter. -/
abbrev Stat : Type := (k : Nat) → (Fin k → ℚ) → ℚ

/-- The RMS over every position: `ρ² = meanSq x`. -/
abbrev Stat.rms : Stat := fun _ x => meanSq x

/-- §5's statistic: the RMS over the salt-independent subsample of every 8th position (`0, 8, 16, …`),
`ρ² = Σ_{l ≡ 0 (mod 8)} x_l² / ⌈k/8⌉`. -/
def Stat.every8 : Stat := fun k x =>
  (∑ l ∈ Finset.univ.filter (fun l : Fin k => l.val % 8 = 0), x l ^ 2) / (((k + 7) / 8 : Nat) : ℚ)

/-- Domain D's liveness rule for a row under the statistic `st`: at most `k/64` entries, over every position, have
`|x| ≥ 8σ·ρ`, stated on squares (`x² ≥ 64σ²·ρ²`). The statistic defaults to `Stat.rms`; §5 checks the rule with
`Stat.every8`'s `ρ`. -/
def LiveRow (σ : ℚ) (x : Fin k → ℚ) (st : Stat := Stat.rms) : Prop :=
  64 * (Finset.univ.filter fun l => 64 * σ ^ 2 * st k x ≤ x l ^ 2).card ≤ k

/-- For noise of unit RMS; `root` is the square root in use, which is not pinned. -/
def betaOf (st : Stat) (root : ℚ → ℚ) (σ α : ℚ) (x : Fin k → ℚ) : ℚ := α * σ * root (st k x)

/-- The forming's chain before the cast: `value α x β e f` is the number the cast reads for one element, from the row's
scale `α`, the element `x`, the row's noise scale `β` and the element's noise rows `e` (of `E`) and `f` (of `F`), as
E4M3 codes. -/
structure FormChain (r : Nat) where
  value : (α x β : ℚ) → (e f : Fin r → Code) → ℚ

def FormChain.exact (r : Nat) : FormChain r := ⟨fun α x β e f => α * x + β * ∑ t, codeVal (e t) * codeVal (f t)⟩

/-- The noise wgmma's accumulation order is not pinned, so its FP32 value `noise` is a parameter; §1 doesn't fix where `β`
enters, and here it scales the noise lines. -/
def FormChain.bf16 (r : Nat) (noise : (Fin r → ℚ) → (Fin r → ℚ) → ℚ) : FormChain r :=
  ⟨fun α x β e f => bf16Rn (α * x + bf16Rn (noise (fun t => β * codeVal (e t)) (fun t => codeVal (f t))))⟩

/-- The FP32 word an FMUL writes for `β` times an E4M3 code's value (`fp32.mul(β, E4m3ToF32(c))`): `f32Word` of the
product, and where the product is zero, the zero whose sign is the XOR of `β`'s and the code's, so `β > 0` times `−0`
(`0x80`) is `−0` and `+0` times a negative code is `−0`. -/
def mulCode (β : ℚ) (c : Code) : Nat :=
  if β * codeVal c = 0 then (if decide (β < 0) != c.sign then 0x80000000 else 0) else f32Word (β * codeVal c)

/-- §5's noise line of one row: `E′ = e4m3(β·e)` on the noise codes `e`, the FP32 product `β·e_t` (an FMUL, `mulCode`)
cast by `cvt.rn.satfinite.e4m3`, as `pearl_c.eprime`. A `−0` code keeps its sign. -/
def lineCodes (β : ℚ) (e : Fin r → Code) : Fin r → Code := fun t => e4m3Word (mulCode β (e t))

/-- **§5's atom-mix chain** (`r = 32`): one `HOPPER_E4M3_K32` step `D = step(C, E′, F)`, whose accumulator input is
`C = f32(α·x)`, the glue value plain FP8 would cast, and whose operands are the noise line `E′ = lineCodes β e` and the
`F` row's codes. The cast reads `D`. The chain is specified for finite `C`, which domain D gives (`|α·x| ≤ 448`). -/
def FormChain.atomMix : FormChain 32 :=
  ⟨fun α x β e f => wordVal (hopperStep (f32Word (α * x)) (lineCodes β e) f)⟩

/-- The noisy quantize of one element: `e4m3` of the chain's value for `α`, `x`, `β` and the element's noise rows `e`, `f`. -/
def FormChain.elem (fc : FormChain r) (α x β : ℚ) (e f : Fin r → Code) : Code := e4m3 (fc.value α x β e f)

/-- The noisy quantize of a `p × k` operand with per-row `α`, `β` and noise factors `E`, `F`: element `(i, l)` reads `X i l`, `E i` and `F l`. -/
def FormChain.quant {p : Nat} (fc : FormChain r) (α β : Fin p → ℚ) (X : Matrix (Fin p) (Fin k) ℚ)
    (E : Matrix (Fin p) (Fin r) Code) (F : Matrix (Fin k) (Fin r) Code) : Matrix (Fin p) (Fin k) Code :=
  fun i l => fc.elem (α i) (X i l) (β i) (E i) (F l)

/-- `A′`: A's noisy quantize, which reads `x`, `E_A` and `F_A` only. -/
def FormChain.quantA (fc : FormChain r) (α β : Fin m → ℚ) (X : Matrix (Fin m) (Fin k) ℚ) (z : Noise m n k r) :
    Matrix (Fin m) (Fin k) Code := fc.quant α β X z.EA z.FA

/-- `B̃`: B's noisy quantize, which reads `B`, `E_B` and `F_B` only. -/
def FormChain.quantB (fc : FormChain r) (α β : Fin n → ℚ) (B : Matrix (Fin n) (Fin k) ℚ) (z : Noise m n k r) :
    Matrix (Fin n) (Fin k) Code := fc.quant α β B z.EB z.FB

def castLines {p : Nat} (β : Fin p → ℚ) (E : Matrix (Fin p) (Fin r) Code) : Matrix (Fin p) (Fin r) Code :=
  fun i => lineCodes (β i) (E i)

def Noise.atomLines (βA : Fin m → ℚ) (βB : Fin n → ℚ) (z : Noise m n k r) : Noise m n k r :=
  ⟨castLines βA z.EA, z.FA, castLines βB z.EB, z.FB⟩

structure MixVec where
  α : Nat
  x : Nat
  β : Nat
  e : Nat
  f : Nat
  c : Nat

def MixVec.holds (v : MixVec) : Bool :=
  match wordQ v.α, wordQ v.x, wordQ v.β with
  | some α, some x, some β =>
    (e4m3 (wordVal (hopperStep (f32Word (α * x))
      (fun t => e4m3Word (f32Word (β * wordVal ((v.e >>> (32 * t.val)) % 2 ^ 32)))) (codes v.f)))).val == v.c
  | _, _, _ => false

/-- One atom's operands: the 32 code pairs of one `HOPPER_E4M3_K32` step. -/
abbrev AtomIn := (Fin 32 → Code) × (Fin 32 → Code)

/-- A promotion group's word: `hopperStep` folded over the group's atoms, starting from `+0`. -/
def groupWord (g : List AtomIn) : Nat := g.foldl (fun w x => hopperStep w x.1 x.2) 0

/-- One promotion: the group's word `rnAdd`ed into the FP32 total; `none` once a total or a word is not finite. -/
def promote (t : Option ℚ) (w : Nat) : Option ℚ :=
  match t, wordQ w with
  | some t, some x => some (rnAdd t x)
  | _, _ => none

/-- The promoted total of a chain given by its groups: each group's word promoted, in order, into a total from `0`. -/
def total (gs : List (List AtomIn)) : Option ℚ := (gs.map groupWord).foldl promote (some 0)

/-- The chain over a list of atoms, cut into groups of `G` (Π's `G` is 4). -/
def chain (G : Nat) (atoms : List AtomIn) : Option ℚ := total (atoms.toChunks G)

/-- One output's `k / 32` atoms: atom `j` reads positions `32j, …, 32j + 31` of the two rows. -/
def rowAtoms (a b : Fin k → Code) : List AtomIn :=
  List.ofFn fun j : Fin (k / 32) =>
    (fun i => a ⟨32 * j + i, by have := j.isLt; have := i.isLt; omega⟩,
     fun i => b ⟨32 * j + i, by have := j.isLt; have := i.isLt; omega⟩)

/-- `C̃ = A′·B̃ᵀ` under the promoted chain: entry `(i, j)` is the chain over row `i` of `A′` and row `j` of `B̃`. -/
def ctChain (G : Nat) (Aq : Matrix (Fin m) (Fin k) Code) (Bq : Matrix (Fin n) (Fin k) Code) :
    Matrix (Fin m) (Fin n) (Option ℚ) := fun i j => chain G (rowAtoms (Aq i) (Bq j))

def ctExact (Aq : Matrix (Fin m) (Fin k) Code) (Bq : Matrix (Fin n) (Fin k) Code) : Matrix (Fin m) (Fin n) ℚ :=
  vals Aq * (vals Bq)ᵀ

def peelLeft (βA : Fin m → ℚ) (A' : Matrix (Fin m) (Fin k) ℚ) (z : Noise m n k r) : Matrix (Fin m) (Fin r ⊕ Fin r) ℚ :=
  fromCols (rowScale βA (vals z.EA)) (A' * vals z.FB)

def peelRight (βB : Fin n → ℚ) (B' : Matrix (Fin n) (Fin k) ℚ) (z : Noise m n k r) : Matrix (Fin n) (Fin r ⊕ Fin r) ℚ :=
  fromCols ((rowScale βB z.NB - B') * vals z.FA) (-rowScale βB (vals z.EB))

def peelU (Ct : Matrix (Fin m) (Fin n) ℚ) (βA : Fin m → ℚ) (βB : Fin n → ℚ) (A' : Matrix (Fin m) (Fin k) ℚ)
    (B' : Matrix (Fin n) (Fin k) ℚ) (z : Noise m n k r) : Matrix (Fin m) (Fin n) ℚ :=
  Ct + peelLeft βA A' z * (peelRight βB B' z)ᵀ

def rawU (Ct : Matrix (Fin m) (Fin n) ℚ) (A : Matrix (Fin m) (Fin k) ℚ) (B : Matrix (Fin n) (Fin k) ℚ)
    (NA : Matrix (Fin m) (Fin k) ℚ) (NB : Matrix (Fin n) (Fin k) ℚ) : Matrix (Fin m) (Fin n) ℚ :=
  Ct - A * NBᵀ - NA * Bᵀ - NA * NBᵀ

end Pouw.PearlC
