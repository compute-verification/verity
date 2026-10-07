import Definitions.Pouw.PearlC.Row4Circuit
import Definitions.Pouw.PearlC.DeviceFp4

/-!
# A row's statistics, scales and forming as the Programs compute them (spec)

`pc8Sem ops p noise` and `pc4Sem ops lines` are Pearl-C's and Pearl-C4's semantics (`PearlCSem`, `Fp4Sem`) at the
Programs' own row arithmetic: `α`, `β`, the noise floor and (for `pc8`) the forming are word formulas on core's
primitives (`RowOps`, each primitive reduced to its width as a gate computes it, `Prim.eval`), in the order
`verity.protocols.accounting.work.pouw.circuit.pc8`'s and `pc4`'s `_stats`, `_alpha` and `_form` compute them and `PROTOCOL.md` fixes. They are
not read off the gate tables: `SecurityProofs.PearlC.RowCircuit` proves that the tables compute them.

On a row of `K` FP32 words `x_0, …, x_{K−1}` (`pc4`: BF16 words, widened):

* `s = rowMax`: the running max of `|x_t|`, `s ← SelectI32(I32Le(s, |x_t|), |x_t|, s)` from `|x_0|`;
* `ρ = rowRho`: with `n = K/8`, the FMA partials `fma(x_{8i}, x_{8i}, ·)` into accumulator `i mod 32` from `+0`
  (`fmaAcc`), the butterfly of `F32Add_v2` sums over `16, 8, 4, 2, 1` (`bfly`), then
  `ρ = PouwF32SqrtRn(F32Div(Σ, f32(n)))`;
* `pc8`: `α = F32Div(448, F32Fmaxf(F32Add(s, F32Mul(4, ρ)), 2⁻³²))`, `β = F32Mul(F32Mul(ρ, α), 2^(−19/2))`, the floor
  `F32Fma(ρ, α, −1)` (admitted at `≥ 0`), each noise code `ẽ_l = F32ToE4m3Sat(F32Mul(β, E4m3ToF32(e_l)))`, and position
  `t`'s formed word `D_t = atom_p(F32Mul(α, x_t); ẽ; F_t)`, its code `F32ToE4m3Sat(D_t)`;
* `pc4`: `δρ = F32Mul(ρ, 1/4)`, `α = F32Div(2688, F32Fmaxf(F32Add(s, F32Mul(4, δρ)), 2⁻³²))` and
  `β = F32Mul(F32Mul(δρ, α), k_β)`, `k_β = 0x3EA920A7`.

A `PearlCSem` reads a row as values (`wordsVal`), so these formulas run on the row's words re-encoded (`f32Word`; `pc4`'s
`bf16WordOf`), and read their results back with `wordVal`. `pc8Sem`'s forming value is `D_t`'s, so `FormChain.elem`'s
`e4m3` of it is `F32ToE4m3Sat(D_t)` wherever the cast agrees with `e4m3Word` and `D_t` is not NaN. The noise
(`noise`, `lines`) stays a parameter, of codes in both: `pc8`'s noise codes are the words the Program reads.
-/

namespace Pouw.PearlC.RowCircuit

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.TileCircuit

variable (ops : RowOps)

/-! ## The statistics both families compute -/

/-- `s_t`: the running max of `|x_0|, …, |x_t|`, `|x|` read through `ab`. -/
def rowMax (ab : Wire) (x : ℕ → ℕ) : ℕ → ℕ
  | 0 => ab.eval ops (x 0)
  | t + 1 =>
    Prim.eval ops .selI32 [Prim.eval ops .le [rowMax ab x t, ab.eval ops (x (t + 1))], ab.eval ops (x (t + 1)),
      rowMax ab x t]

/-- FMA partial `i`: `fma(x_{8i}, x_{8i}, ·)` (`x` read through `w`) into accumulator `i mod 32`, from `+0`. -/
def fmaAcc (w : Wire) (x : ℕ → ℕ) (i : ℕ) : ℕ :=
  Prim.eval ops .fma [w.eval ops (x (8 * i)), w.eval ops (x (8 * i)), if i < 32 then 0 else fmaAcc w x (i - 32)]
termination_by i
decreasing_by omega

/-- The last partial into accumulator `j` of `n`. -/
def accLast (n j : ℕ) : ℕ := j + 32 * ((n - 1 - j) / 32)

/-- A butterfly round over `n` live sums: sum `j` plus sum `j + h` (`+0` past the live ones). -/
def bfly (h n : ℕ) (v : ℕ → ℕ) (j : ℕ) : ℕ := Prim.eval ops .add [v j, if j + h < n then v (j + h) else 0]

/-- The butterfly's sum of the `n` partials' accumulators. -/
def sumSq (w : Wire) (x : ℕ → ℕ) (n : ℕ) : ℕ :=
  bfly ops 1 n (bfly ops 2 n (bfly ops 4 n (bfly ops 8 n (bfly ops 16 n fun j => fmaAcc ops w x (accLast n j))))) 0

/-- `ρ`: the root of the mean of `x²` over every 8th position. -/
def rowRho (w : Wire) (x : ℕ → ℕ) (K : ℕ) : ℕ :=
  Prim.eval ops .sqrt [Prim.eval ops .div [sumSq ops w x (K / 8), f32Word ((K / 8 : ℕ) : ℚ)]]

/-! ## `pc8` -/

/-- `pc8`'s `α`: `448 / max(s + 4·ρ, 2⁻³²)`. -/
def alpha8 (x : ℕ → ℕ) (K : ℕ) : ℕ :=
  Prim.eval ops .div [0x43E00000, Prim.eval ops .fmaxf
    [Prim.eval ops .add [rowMax ops .fabs x (K - 1), Prim.eval ops .mul [0x40800000, rowRho ops .id x K]], 0x2F800000]]

/-- `pc8`'s `β`: `ρ·α·2^(−19/2)`. -/
def beta8 (x : ℕ → ℕ) (K : ℕ) : ℕ :=
  Prim.eval ops .mul [Prim.eval ops .mul [rowRho ops .id x K, alpha8 ops x K], 0x3AB504F3]

/-- `pc8`'s floor: `ρ·α − 1` by one FMA; the row is above the floor where it is `≥ 0`. -/
def floor8 (x : ℕ → ℕ) (K : ℕ) : ℕ := Prim.eval ops .fma [rowRho ops .id x K, alpha8 ops x K, 0xBF800000]

/-- A noise code rescaled: `e4m3(β·e)` on words, `e` an E4M3 code. -/
def eTilde8 (β e : ℕ) : ℕ := Prim.eval ops .toE4m3 [Prim.eval ops .mul [β, Prim.eval ops .e4m3ToF32 [e]]]

/-- `pc8`'s formed word: one atom on the step `p` from `α·x`, over the rescaled noise codes `et` against a form basis
row's codes `f`. -/
def form8 (p : Pipeline) (α x : ℕ) (et f : ℕ → ℕ) : ℕ :=
  Prim.eval ops (.atom8 p) (Prim.eval ops .mul [α, x] :: (List.range 32).map et ++ (List.range 32).map f)

/-- `pc8`'s code: the formed word's E4M3 cast. -/
def code8W (p : Pipeline) (α x : ℕ) (et f : ℕ → ℕ) : ℕ := Prim.eval ops .toE4m3 [form8 ops p α x et f]

/-! ## `pc4` -/

/-- `pc4`'s `δρ`, `δ = 1/4`. -/
def drho4 (x : ℕ → ℕ) (K : ℕ) : ℕ := Prim.eval ops .mul [rowRho ops .bf16 x K, 0x3E800000]

/-- `pc4`'s `α`: `2688 / max(s + 4·δρ, 2⁻³²)`. -/
def alpha4 (x : ℕ → ℕ) (K : ℕ) : ℕ :=
  Prim.eval ops .div [0x45280000, Prim.eval ops .fmaxf
    [Prim.eval ops .add [rowMax ops .bf16Fabs x (K - 1), Prim.eval ops .mul [0x40800000, drho4 ops x K]], 0x2F800000]]

/-- `pc4`'s `β`: `δρ·α·k_β`. -/
def beta4 (x : ℕ → ℕ) (K : ℕ) : ℕ := Prim.eval ops .mul [Prim.eval ops .mul [drho4 ops x K, alpha4 ops x K], 0x3EA920A7]

/-! ## The semantics -/

/-- A row of values as FP32 words, `0` past its end. -/
def encRow (k : ℕ) (X : Fin k → ℚ) (l : ℕ) : ℕ := if h : l < k then f32Word (X ⟨l, h⟩) else 0

/-- A row of values as BF16 words, `0` past its end. -/
def encRow16 (k : ℕ) (X : Fin k → ℚ) (l : ℕ) : ℕ := if h : l < k then bf16WordOf (X ⟨l, h⟩) else 0

/-- A noise line's codes as words, `0` past its end. -/
def encCodes {r : ℕ} (e : Fin r → Code) (l : ℕ) : ℕ := if h : l < r then (e ⟨l, h⟩).val else 0

/-- **`pc8`'s forming chain**: the formed word `form8` at `α`, `x` and `β`'s words and the noise' and basis' codes,
read as a value. -/
def pc8Form (p : Pipeline) : FormChain Params.pi.r :=
  ⟨fun α x β e f => wordVal (form8 ops p (f32Word α) (f32Word x) (fun l => eTilde8 ops (f32Word β) (encCodes e l))
    (encCodes f))⟩

/-- **Pearl-C's semantics at `pc8`'s row arithmetic**, on the step `p`: `α`, `β` and the floor from the row's words
(`alpha8`, `beta8`, `floor8 + 1`, so the floor rule `1 ≤ floor` is `ρ·α − 1 ≥ 0`), the forming `pc8Form`, and the noise
as given. The floor is `0`, so the row is refused, where `ρ`'s word is not finite (NaN or ±inf): Python's `s5_stats`
refuses such a row (`_check_finite` on the partials, which are non-negative sums) and the circuit's flag asks
`ρ ≤ F32_MAX`. -/
def pc8Sem {Q R S : Type} (p : Pipeline)
    (noise : (Q → R) → S → (U : Workload) → (u : ℕ) → Codes →
      Noise (U.layout.shape u).m (U.layout.shape u).n (U.layout.shape u).k Params.pi.r) : PearlCSem Q R S where
  noise := noise
  fc := pc8Form ops p
  alpha := fun k X => wordVal (alpha8 ops (encRow k X) k)
  beta := fun k X => wordVal (beta8 ops (encRow k X) k)
  floor := fun k X =>
    if (wordQ (rowRho ops .id (encRow k X) k)).isSome then wordVal (floor8 ops (encRow k X) k) + 1 else 0

/-- **Pearl-C4's semantics at `pc4`'s row arithmetic**: `α` and `β` from the row's BF16 words (`alpha4`, `beta4`), and
the noise lines as given. `β` is `0`, so `RowAdmit4` refuses the row, where `β`'s word is not finite (NaN or ±inf), as
Python refuses it (`s5_stats`' `_check_finite`, `pearl_c4.stats`' `_finite`) and the circuit's flag asks
`β ≤ F32_MAX`. -/
def pc4Sem {Q R S : Type}
    (lines : (Q → R) → S → (U : Workload) → (u : ℕ) → Codes →
      Lines4 (U.layout.shape u).m (U.layout.shape u).n (U.layout.shape u).k) : Fp4Sem Q R S where
  lines := lines
  alpha := fun k X => wordVal (alpha4 ops (encRow16 k X) k)
  beta := fun k X => if (wordQ (beta4 ops (encRow16 k X) k)).isSome then wordVal (beta4 ops (encRow16 k X) k) else 0

end Pouw.PearlC.RowCircuit
