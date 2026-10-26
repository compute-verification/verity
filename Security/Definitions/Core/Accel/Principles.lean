import Definitions.Core.Accel.H200
import Definitions.Core.Accel.Sm120

/-!
# P1 and P2 from the instruction set (trusted)

The two data-flow principles that restrict an adversary, as properties of the instruction set's semantics. All but one
are statements to prove (in `Proofs/Core/Accel`), not assumptions: they follow from `Isa`'s definitions.

* **(P1) Exact products only through fixed operand formats.** An MMA reads each operand through its format's `bits`
  (`elem`), and its dot reads the operand pairs only through what one multiplier forms from a pair (`mulOut`): the
  exact product with its exponent, nothing else of the pair (`ProductForm`). The integer path is exact on its whole
  range (`P1Int`); a float pipeline of adder width `w` is exact on integers whose magnitudes sum below
  `2^min(w, 24)` (`P1Float`), which covers the captured float atoms of each card (`H200.P1Captured`,
  `Sm120.P1Captured`). Where no capture pins a float dot (the H200's FP8 at a nonzero accumulator, sm_120's FP8 and
  E2M1), P1 is a property of the open parameter (`H200.P1Open`, `Sm120.P1Open`), measurable on the card. A wider
  operand therefore takes several MMAs: that much is semantics, not an assumption.
* **(P2) An MMA's products sum into exactly one destination; C is a register read.** An MMA's output `o` reads row
  `o / n` of `A`, column `o % n` of `B` and accumulator `o` only (`P2Local`), and distinct outputs read distinct row
  and column pairs (`P2Distinct`), so each of its `m·n·k` products is formed in one output's dot. The accumulator `C`
  is an input like `A` and `B`, read from registers its operands name, and `D` is written where its own operands name
  (`mma.sync` on both cards takes `D ≠ C`): one accumulator value may seed several MMAs, each at its own units and no
  more. That fork is the card's, and the model keeps it. An instruction writes only the locations its outputs name
  (`P2Frame`), so a product reaches another word only through further instructions, and each of those uses at least
  `1/32` of an issue slot (`P2Priced`), which G2 bounds.

**The one named assumption** is `Assumptions.AccelProductForm`: the MMAs whose semantics no capture pins (`Open.dot`)
have the captured ones' data flow, each output a function of its accumulator, its pairs' multiplier outputs and its
block scales (`ProductForm`, in `Card`). Every captured dot has the form (`CapturedProductForm`), so for them it is a
statement to prove.
-/

open scoped NNRat

namespace Verity.Accel

open Verity.TC

/-- The value of a term: `(-1)^negative · magnitude · 2^(exponent - 23)`. -/
def termVal (t : Spec.Term) : ℚ := (if t.negative then -1 else 1) * t.magnitude * 2 ^ (t.exponent - 23)

/-- Every dot a card's captures pin has the product form (to prove, for `H200.card` and `Sm120.card`). -/
def CapturedProductForm (c : Card) : Prop :=
  ∀ fa fb sat k d, c.captured fa fb sat k = some d → ProductForm fa fb k d

/-- **P1, the integer path**: an integer MMA's dot is the exact sum whenever it lies in the `s32` range, with or
without `.satfinite`. -/
def P1Int : Prop :=
  ∀ (fa fb : Fmt) (sat : Bool) (c : Word) (a b : List ℕ), intPair fa fb = true →
    let s := c.toInt + (List.zipWith (fun x y => fa.intVal x * fb.intVal y) a b).sum
    (-2 ^ 31 ≤ s) → s < 2 ^ 31 → (intDot fa fb sat c a b).toInt = s

/-- **P1 at a dot**: dot `d` of format `f` over `k` pairs, on finite operands and a finite accumulator whose values are
integers, with the accumulator's and the products' magnitudes summing below `2^min(w, 24)`, is the exact sum. With
`zero`, only at a zero accumulator. -/
def P1Dot (f : Fmt) (k w : ℕ) (zero : Bool) (d : Dot) : Prop :=
  ∀ (c : Word) (a b : List ℕ), a.length = k → b.length = k → (zero → c = 0) →
    (∀ x ∈ a ++ b, f.code x < 2 ^ f.mini.bits ∧ f.mini.cls (f.code x) ≠ .nan ∧ f.mini.cls (f.code x) ≠ .inf) →
    Spec.fp32Class c.toNat ≠ .nan → Spec.fp32Class c.toNat ≠ .inf →
    let ps := List.zipWith (fun x y => termVal (f.mini.product (f.code x) (f.code y))) a b
    let c₀ := termVal (Spec.fp32Term c.toNat)
    (∃ z : ℤ, c₀ = z) → (∀ p ∈ ps, ∃ z : ℤ, p = z) → |c₀| + (ps.map fun p => |p|).sum < 2 ^ min w 24 →
    termVal (Spec.fp32Term (d c a b).toNat) = c₀ + ps.sum

/-- **P1, the float path**: format `f`'s dot on pipeline `p` is exact below `2^min(width, 24)`. The adder aligns every
term to `2^(E + 1 - width)`, `E` the largest exponent, and truncates; FP32 keeps 24 bits. -/
def P1Float (f : Fmt) (p : Pipeline) : Prop := P1Dot f p.k p.width false (floatDot f p)

/-- **P1 on the H200's captured float atoms**: BF16 on `TC.HOPPER_BF16_M16N8K16` (`mma.sync` and `wgmma`), exact below
`2^24`; the E4M3 and E5M2 K32 atom (LIT: one group of 32 at width 14, the alignment truncation of `pearl_kw`'s
`_group`), exact below `2^14`. The H100's FP8 capture fixed a zero accumulator, so the card's FP8 dot is this one only
there (`H200.card.zeroOnly`); elsewhere it is `Open.dot`, and P1 there is `H200.P1Open`. -/
def H200.P1Captured : Prop :=
  P1Float .bf16 HOPPER_BF16_M16N8K16 ∧ P1Float .e4m3 H200.fp8Pipeline ∧ P1Float .e5m2 H200.fp8Pipeline

/-- **P1 at the H200's uncaptured FP8 accumulators**, a property of the open parameter `x` (measurable: the FP8
`wgmma` K32 at a nonzero accumulator). -/
def H200.P1Open (x : Open) : Prop :=
  ∀ f ∈ [Fmt.e4m3, .e5m2], ∀ sat, P1Dot f 32 14 false (x.dot f f sat 32)

/-- **P1 on the RTX PRO 6000's captured float atom**: BF16 on `TC.HOPPER_BF16_M16N8K16`, exact below `2^24`. -/
def Sm120.P1Captured : Prop := P1Float .bf16 HOPPER_BF16_M16N8K16

/-- **P1 at the RTX PRO 6000's FP8 and E2M1 K32 atom**, a property of the open parameter `x`. LIT reports one group
of 32 at width 26 (`Sm120.f8Pipeline`, exact below `2^24` if it were the dot), but PEARLC-MODEL found the atom the
exact sum's FP32 truncation in 99.96–99.997% of atoms only, so the dot is not captured. -/
def Sm120.P1Open (x : Open) : Prop :=
  ∀ f ∈ [Fmt.e4m3, .e5m2, .e2m1], ∀ sat, P1Dot f 32 26 false (x.dot f f sat 32)

/-- The candidate pipelines' P1, a statement about the semantics alone (`Sm120.f8Pipeline` is no card's dot). -/
def P1Pipelines : Prop := P1Float .e4m3 Sm120.f8Pipeline ∧ P1Float .e5m2 Sm120.f8Pipeline ∧
  P1Float .e2m1 Sm120.f8Pipeline

/-- **P2, locality**: an MMA's output `o` changes only with row `o / n` of `A`, column `o % n` of `B` and accumulator
`o`. -/
def P2Local : Prop :=
  ∀ (dot : Dot) (fa fb : Fmt) (s : Shape) (v v' : ℕ → Word) (o : ℕ),
    rowA fa s v (o / s.n) = rowA fa s v' (o / s.n) → colB fa fb s v (o % s.n) = colB fa fb s v' (o % s.n) →
    accC fa fb s v o = accC fa fb s v' o → mmaEval dot fa fb s v o = mmaEval dot fa fb s v' o

/-- **P2, one accumulator a pair**: distinct outputs of an MMA read distinct row and column pairs. -/
def P2Distinct : Prop :=
  ∀ (s : Shape) (o o' : ℕ), o < s.m * s.n → o' < s.m * s.n → o / s.n = o' / s.n → o % s.n = o' % s.n → o = o'

/-- **P2, the frame**: an instruction leaves every location none of its outputs names. -/
def P2Frame : Prop :=
  ∀ {ι : Type} (m : Machine ι) (o : m.Op) (src : Fin (m.nIn o) → Machine.Operand)
    (dst : Fin (m.nOut o) → Machine.Operand) (pred : Option ℕ) (M : m.Mem) (l : m.Loc),
    (∀ i, (dst i).locIn (m.outStore o i) M ≠ some l) → Machine.execOp o src dst pred M l = M l

/-- **P2, priced moves**: every opcode uses at least `1/32` of an issue slot, at any tensor weights, so no instruction
is free on every resource: not a `b1` MMA (a slot, whatever `Open.b1` is), a uniform instruction (a slot) or a move. -/
def P2Priced : Prop :=
  ∀ (weight : Fmt → ℚ≥0) (scale : Shape → ℚ≥0) (hi : ℚ≥0) (o : Op Store), 1 / 32 ≤ opUse weight scale hi o .issue

end Verity.Accel
