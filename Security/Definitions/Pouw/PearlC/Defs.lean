import Mathlib.Data.Matrix.ColumnRowPartitioned
import Batteries.Data.List.Basic
import Definitions.Pouw.Fp8Atom.Fp32

/-!
# Pearl-C's Π (trusted definitions)

Pearl-C is Pearl's product and clean-up (the "peel"), repaired after review, as the cheap-binding doc's §1
specifies it. This file holds the parts of Π that other files state things about:

* **Parameters** (`Params`, `Params.pi`, `Shape`): the noise scale `δ`, the promotion depth `G` and the noise rank `r`;
  Π pins `δ = 1`, `G = 4`, `r = 32`.
* **The E4M3 encoder** (`e4m3Mag`, `e4m3`, `e4m3Word`): `cvt.rn.satfinite.e4m3x2.f32`, round to nearest even with
  saturation to `±448`, as `verity.ml.tc.cast.f32_to_e4m3_sat_word`; pinned by the vectors of `EncVectors`, checked in
  `EncChecks`. The BF16 conversion `bf16Rn` (`cvt.rn.bf16.f32` on values) and the FP32 encoder `f32Word` (IEEE binary32
  round to nearest even, the word an FMUL writes) are pinned the same way.
* **The row statistic** (`Stat`, `Stat.rms`, `Stat.every8`): the statistic in use for the noise scale, as `ρ²`. The
  liveness rule `LiveRow` and the noise scale `betaOf` take it; `LiveRow` defaults to `Stat.rms`.
* **The noise and the forming** (`Noise`, `rowScale`, `FormChain`, `FormChain.quant`): the noise factors are E4M3
  codes, as Pearl's lines are. `A′ = e4m3(v)` per element, where `v = FormChain.value α x β e f` is the number the cast
  reads. Three chains: `FormChain.exact` (`α·x + β·n`, with `n = E_A·F_Aᵀ` exact), `FormChain.bf16` (Pearl's BF16 chain)
  and `FormChain.atomMix` (§5's forming: the noise and `α·x` in one `HOPPER_E4M3_K32` step, then one cast).
  `lineCodes` is §5's `E′ = e4m3(β·e)` on the codes (`mulCode`, the FMUL with a zero product's sign), and `castLines`
  and `Noise.atomLines` are `E′` as the peel reuses it.
* **The promoted product chain** (`groupWord`, `promote`, `total`, `chain`, `ctChain`): `hopperStep` groups of `G`
  atoms, each from `+0`, each group's word promoted into an FP32 total by `rnAdd`.
* **U's two forms** (`peelLeft`, `peelRight`, `peelU`, `rawU`) as exact rational matrices, beside `ctExact`, the
  exact `C̃ = A′·B̃ᵀ`.

Shapes: `A` and `A′` are `m × k`, `B` and `B̃` are `n × k`, `C̃` and `U` are `m × n`; the noise factors are `E_A`
(`m × r`), `F_A` (`k × r`), `E_B` (`n × r`) and `F_B` (`k × r`). Per-row scales are functions of the row index, so the
scalar `β` of the doc's formulas is a constant function.
-/

namespace Pouw.PearlC

open Pouw.Fp8Atom Matrix

/-! ## Parameters -/

/-- Pearl-C's parameters: the noise scale `δ` relative to the row RMS, the promotion depth `G` in atoms, the noise rank `r`. -/
structure Params where
  δ : ℚ
  G : Nat
  r : Nat

/-- The values Π pins: `δ = 1`, promotion every `G = 4` atoms, rank `r = 32`. -/
def Params.pi : Params := ⟨1, 4, 32⟩

/-- A GEMM shape: `A` is `m × k`, `B` is `n × k`, and `C̃ = A′·B̃ᵀ` is `m × n`. -/
structure Shape where
  m : Nat
  n : Nat
  k : Nat

/-- The shape is a whole number of promotion groups: `32·G` divides `k`. -/
def Shape.Tiled (p : Params) (s : Shape) : Prop := 32 * p.G ∣ s.k

/-- `x` has a `p`-bit significand at some exponent (BF16 is `p = 8`, E4M3 `p = 4`, and `Representable` is `p = 24`). -/
def HasSig (p : Nat) (x : ℚ) : Prop := ∃ m e : ℤ, |m| < 2 ^ p ∧ x = m * 2 ^ e

/-! ## The E4M3 encoder and the BF16 conversion -/

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

/-- A matrix of codes as exact values. -/
def vals {ι κ : Type} (M : Matrix ι κ Code) : Matrix ι κ ℚ := M.map codeVal

/-- The spacing of BF16's 8-bit significands in the binade of `x ≠ 0`. -/
def ulpBf16 (x : ℚ) : ℚ := 2 ^ (binade |x| - 7)

/-- `cvt.rn.bf16.f32` on values: round to an 8-bit significand, ties to even, with no exponent range (exact where finite and normal). -/
def bf16Rn (x : ℚ) : ℚ := (if x < 0 then -1 else 1) * (roundEven (|x| / ulpBf16 x) : ℚ) * ulpBf16 x

/-- An encoder conformance vector: an FP32 word and the code `f32_to_e4m3_sat_word` gives it. -/
structure EncVec where
  u : Nat
  c : Nat

/-- An encoder vector holds for `e4m3Word`, and for `e4m3` on the word's value when that value is finite and nonzero. -/
def EncVec.holds (v : EncVec) : Bool :=
  (e4m3Word v.u).val == v.c &&
    match wordQ v.u with
    | some x => x == 0 || (e4m3 x).val == v.c
    | none => true

/-- A BF16 conversion vector: a finite normal FP32 word and the finite normal BF16 word `f32_to_bf16_rn_word` gives it. -/
structure Bf16Vec where
  u : Nat
  h : Nat

/-- A BF16 vector holds: `bf16Rn` maps the FP32 word's value to the value of the BF16 word (read as the FP32 word `h <<< 16`). -/
def Bf16Vec.holds (v : Bf16Vec) : Bool :=
  match wordQ v.u, wordQ (v.h <<< 16) with
  | some x, some y => decide (bf16Rn x = y)
  | _, _ => false

/-! ## FP32 words from values -/

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

/-- An FP32 multiply vector: the words `a`, `b` and the word `p` IEEE binary32 multiplication writes for them. -/
structure F32Vec where
  a : Nat
  b : Nat
  p : Nat

/-- An FP32 multiply vector holds: `a` and `b` are finite and nonzero, and `f32Word` maps the exact product of their
values to `p`. (An exact zero has no sign, so `f32Word 0` is `+0` where a signed zero operand gives `−0`.) -/
def F32Vec.holds (v : F32Vec) : Bool :=
  match wordQ v.a, wordQ v.b with
  | some x, some y => x != 0 && y != 0 && f32Word (x * y) == v.p
  | _, _ => false

/-! ## The noise and the forming -/

/-- One unit's noise factors as E4M3 codes, as Pearl's lines are (`sample_line`): `E_A` (`m × r`), `F_A` (`k × r`),
`E_B` (`n × r`) and `F_B` (`k × r`). -/
structure Noise (m n k r : Nat) where
  EA : Matrix (Fin m) (Fin r) Code
  FA : Matrix (Fin k) (Fin r) Code
  EB : Matrix (Fin n) (Fin r) Code
  FB : Matrix (Fin k) (Fin r) Code

variable {m n k r : Nat}

/-- A's noise `N_A = E_A·F_Aᵀ` (`m × k`), on the codes' values. -/
def Noise.NA (z : Noise m n k r) : Matrix (Fin m) (Fin k) ℚ := vals z.EA * (vals z.FA)ᵀ

/-- B's noise `N_B = E_B·F_Bᵀ` (`n × k`), on the codes' values. -/
def Noise.NB (z : Noise m n k r) : Matrix (Fin n) (Fin k) ℚ := vals z.EB * (vals z.FB)ᵀ

/-- The `r × r` Gram factor `F_Bᵀ·F_A`, through which the Gram order forms `N_B·F_A = E_B·(F_Bᵀ·F_A)`. -/
def Noise.gram (z : Noise m n k r) : Matrix (Fin r) (Fin r) ℚ := (vals z.FB)ᵀ * vals z.FA

/-- `β ⊙ M`: row `i` of `M` scaled by `β i`. -/
def rowScale {p q : Nat} (β : Fin p → ℚ) (M : Matrix (Fin p) (Fin q) ℚ) : Matrix (Fin p) (Fin q) ℚ := diagonal β * M

/-- A row's mean square, `rms²`. -/
def meanSq (x : Fin k → ℚ) : ℚ := (∑ l, x l ^ 2) / k

/-! ## The row statistic -/

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
`|x| ≥ 8δ·ρ`, stated on squares (`x² ≥ 64δ²·ρ²`). The statistic defaults to `Stat.rms`; §5 checks the rule with
`Stat.every8`'s `ρ`. -/
def LiveRow (δ : ℚ) (x : Fin k → ℚ) (st : Stat := Stat.rms) : Prop :=
  64 * (Finset.univ.filter fun l => 64 * δ ^ 2 * st k x ≤ x l ^ 2).card ≤ k

/-- The noise scale of a row, `β = α·δ·ρ`, for noise of unit RMS, where `ρ = root (st k x)` and `root` is the square
root in use (the FP32 square root, once pinned). §5 folds the noise lines' norm into `β` as well. -/
def betaOf (st : Stat) (root : ℚ → ℚ) (δ α : ℚ) (x : Fin k → ℚ) : ℚ := α * δ * root (st k x)

/-! ## The forming chains -/

/-- The forming's chain before the cast: `value α x β e f` is the number the cast reads for one element, from the row's
scale `α`, the element `x`, the row's noise scale `β` and the element's noise rows `e` (of `E`) and `f` (of `F`), as
E4M3 codes. -/
structure FormChain (r : Nat) where
  value : (α x β : ℚ) → (e f : Fin r → Code) → ℚ

/-- **The exact chain**: `α·x + β·(e·f)` on the codes' values, so the forming is `e4m3(α·x + β·n)` with `n = E·Fᵀ`
exact. -/
def FormChain.exact (r : Nat) : FormChain r := ⟨fun α x β e f => α * x + β * ∑ t, codeVal (e t) * codeVal (f t)⟩

/-- **Pearl's BF16 chain**: the noise wgmma over the noise lines `β·e` and `f` (on the codes' values), whose FP32 value
is `noise` (its accumulation order is not pinned, so it is a parameter), one f32→bf16 conversion of it, and one HFMA2
`fma(α, x, bf16(n))` rounded to BF16, which the cast reads. `α` and `x` are the HFMA2's BF16 operands. §1's step list
fixes the conversion and the HFMA2 but not where `β` enters; here it scales the noise lines, as in §5. -/
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

/-- §5's noise lines of a `p`-row operand: row `i` is `E′_i = lineCodes (β i) (E i)`, the codes the atom-mix reads. -/
def castLines {p : Nat} (β : Fin p → ℚ) (E : Matrix (Fin p) (Fin r) Code) : Matrix (Fin p) (Fin r) Code :=
  fun i => lineCodes (β i) (E i)

/-- The noise as §5's peel reuses it: `E_A` and `E_B` replaced by their noise lines `E′_A` and `E′_B`, `F_A` and `F_B`
unchanged, so at unit `β` the peel's `N_A` is `E′_A·F_Aᵀ`, the noise the atom-mix added up to its accumulator's rounding. -/
def Noise.atomLines (βA : Fin m → ℚ) (βB : Fin n → ℚ) (z : Noise m n k r) : Noise m n k r :=
  ⟨castLines βA z.EA, z.FA, castLines βB z.EB, z.FB⟩

/-- An atom-mix conformance vector: the FP32 words of `α`, `x` and `β`; the `E` row's 32 FP32 words, packed 32 bits
each; the `F` row's 32 E4M3 codes, packed 8 bits each (`codes`); and the code `verity.ml.tc` gives `A′`. -/
structure MixVec where
  α : Nat
  x : Nat
  β : Nat
  e : Nat
  f : Nat
  c : Nat

/-- An atom-mix vector holds: `α`, `x` and `β` are finite, and the atom-mix's step and cast give the code `c` on the `E`
row's FP32 values, `E′_t = e4m3Word (f32Word (β·e_t))` (`lineCodes` where `e_t` is a nonzero code's value), and the
`F` row's codes. -/
def MixVec.holds (v : MixVec) : Bool :=
  match wordQ v.α, wordQ v.x, wordQ v.β with
  | some α, some x, some β =>
    (e4m3 (wordVal (hopperStep (f32Word (α * x))
      (fun t => e4m3Word (f32Word (β * wordVal ((v.e >>> (32 * t.val)) % 2 ^ 32)))) (codes v.f)))).val == v.c
  | _, _, _ => false

/-! ## The promoted product chain -/

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

/-- `C̃` in exact arithmetic: `A′·B̃ᵀ` on the codes' values. -/
def ctExact (Aq : Matrix (Fin m) (Fin k) Code) (Bq : Matrix (Fin n) (Fin k) Code) : Matrix (Fin m) (Fin n) ℚ :=
  vals Aq * (vals Bq)ᵀ

/-! ## U's two forms, exact -/

/-- Pearl's left factor `[β·E_A ‖ A′·F_B]` (`m × 2r`); `A'` is `A′`'s values. -/
def peelLeft (βA : Fin m → ℚ) (A' : Matrix (Fin m) (Fin k) ℚ) (z : Noise m n k r) : Matrix (Fin m) (Fin r ⊕ Fin r) ℚ :=
  fromCols (rowScale βA (vals z.EA)) (A' * vals z.FB)

/-- Pearl's right factor `[(β·N_B − B̃)·F_A ‖ −β·E_B]` (`n × 2r`); `B'` is `B̃`'s values. -/
def peelRight (βB : Fin n → ℚ) (B' : Matrix (Fin n) (Fin k) ℚ) (z : Noise m n k r) : Matrix (Fin n) (Fin r ⊕ Fin r) ℚ :=
  fromCols ((rowScale βB z.NB - B') * vals z.FA) (-rowScale βB (vals z.EB))

/-- Pearl's form of U: `C̃ + [β·E_A ‖ A′·F_B]·[(β·N_B − B̃)·F_A ‖ −β·E_B]ᵀ`. -/
def peelU (Ct : Matrix (Fin m) (Fin n) ℚ) (βA : Fin m → ℚ) (βB : Fin n → ℚ) (A' : Matrix (Fin m) (Fin k) ℚ)
    (B' : Matrix (Fin n) (Fin k) ℚ) (z : Noise m n k r) : Matrix (Fin m) (Fin n) ℚ :=
  Ct + peelLeft βA A' z * (peelRight βB B' z)ᵀ

/-- The whitepaper's raw-A form of U: `C̃ − A·N_Bᵀ − N_A·Bᵀ − N_A·N_Bᵀ`, on the raw `A`, `B` and the noise as added. -/
def rawU (Ct : Matrix (Fin m) (Fin n) ℚ) (A : Matrix (Fin m) (Fin k) ℚ) (B : Matrix (Fin n) (Fin k) ℚ)
    (NA : Matrix (Fin m) (Fin k) ℚ) (NB : Matrix (Fin n) (Fin k) ℚ) : Matrix (Fin m) (Fin n) ℚ :=
  Ct - A * NBᵀ - NA * Bᵀ - NA * NBᵀ

end Pouw.PearlC
