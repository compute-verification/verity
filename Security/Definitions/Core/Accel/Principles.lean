import Definitions.Core.Accel.Card

/-!
# P1 and P2 from the instruction set (trusted)

The two data-flow principles that restrict an adversary, as properties of the instruction set's semantics. All but one
are statements to prove (in `Proofs/Core/Accel`), not assumptions: they follow from `Isa`'s definitions.

* **(P1) Exact products only through fixed operand formats.** An MMA reads each operand through its format's `bits`
  (`elem`), and its dot reads the operand pairs only through what one multiplier forms from a pair (`mulOut`): the
  exact product with its exponent, nothing else of the pair (`ProductForm`). The integer path is exact on its whole
  range (`P1Int`); BF16 on Hopper's and sm_120's pipeline is exact on integers whose sum stays below `2^24`
  (`P1Bf16`). A wider operand therefore takes several MMAs: that much is semantics, not an assumption.
* **(P2) A product lives in one accumulator.** An MMA's output `o` reads row `o / n` of `A`, column `o % n` of `B` and
  accumulator `o` only (`P2Local`), and distinct outputs read distinct row and column pairs (`P2Distinct`), so each of
  its `m·n·k` products is formed in one output's dot. An instruction writes only the locations its outputs name
  (`P2Frame`), so a product reaches another word only through further instructions, and each of those uses at least
  `1/32` of an issue slot (`P2Priced`), which G2 bounds.

**The one named assumption** is `Open.ProductForm`: the MMAs whose semantics no capture pins (`Open.dot`) have the
captured ones' data flow, each output a function of its accumulator, its pairs' multiplier outputs and its block
scales. It is a fact about the tensor core's wiring (a multiplier sees one operand pair; the adder tree sees the
multipliers and the accumulator), checkable by a capture that swaps operand pairs with equal multiplier outputs;
every captured dot has the form (`CapturedProductForm`).
-/

open scoped NNRat

namespace Verity.Accel

open Verity.TC

/-- The value of a term: `(-1)^negative · magnitude · 2^(exponent - 23)`. -/
def termVal (t : Spec.Term) : ℚ := (if t.negative then -1 else 1) * t.magnitude * 2 ^ (t.exponent - 23)

/-- What one multiplier forms from operand `x` of format `fa` and `y` of `fb`: the classes of the two operands, the
product's sign, the product of their significands and the sum of their exponent fields (a subnormal's as 1), which
is the exact product with its exponent; for an integer pair, the exact product, its sign and magnitude. -/
def mulOut (fa fb : Fmt) (x y : ℕ) : Spec.Cls × Spec.Cls × Bool × ℕ × ℕ :=
  if intPair fa fb then
    let p := fa.intVal x * fb.intVal y
    (.finite, .finite, decide (p < 0), p.natAbs, 0)
  else
    let f := fa.mini
    let g := fb.mini
    let u := fa.code x
    let v := fb.code y
    let eu := (u >>> f.manBits) &&& (2 ^ f.expBits - 1)
    let ev := (v >>> g.manBits) &&& (2 ^ g.expBits - 1)
    let su := (u &&& (2 ^ f.manBits - 1)) ||| (if eu ≠ 0 then 2 ^ f.manBits else 0)
    let sv := (v &&& (2 ^ g.manBits - 1)) ||| (if ev ≠ 0 then 2 ^ g.manBits else 0)
    (f.cls u, g.cls v, ((u >>> (f.bits - 1)) ^^^ (v >>> (g.bits - 1))) &&& 1 = 1, su * sv,
      (if eu ≠ 0 then eu else 1) + (if ev ≠ 0 then ev else 1))

/-- A dot of rows of format `fa` and `fb` and `k` products reads its operand pairs only through their multiplier
outputs: on rows of the format's length, it is a function of the accumulator, the `k` pairs' `mulOut` and the rows'
block scales (empty unless the format is block-scaled). -/
def ProductForm (fa fb : Fmt) (k : ℕ) (d : Dot) : Prop :=
  ∃ F : Word → List (Spec.Cls × Spec.Cls × Bool × ℕ × ℕ) → List ℕ → List ℕ → Word,
    ∀ (c : Word) (a b : List ℕ), a.length = fa.rowLen k → b.length = fb.rowLen k →
      d c a b = F c (List.zipWith (mulOut fa fb) (a.take k) (b.take k)) (a.drop k) (b.drop k)

/-- **The named assumption**: every dot no capture pins has the product form. -/
def Open.ProductForm (x : Open) : Prop := ∀ fa fb sat k, Accel.ProductForm fa fb k (x.dot fa fb sat k)

/-- Every dot a card's captures pin has the product form (to prove, for `H200.card` and `Sm120.card`). -/
def CapturedProductForm (c : Card) : Prop :=
  ∀ fa fb sat k d, c.captured fa fb sat k = some d → ProductForm fa fb k d

/-- **P1, the integer path**: an integer MMA's dot is the exact sum whenever it lies in the `s32` range, with or
without `.satfinite`. -/
def P1Int : Prop :=
  ∀ (fa fb : Fmt) (sat : Bool) (c : Word) (a b : List ℕ), intPair fa fb = true →
    let s := c.toInt + (List.zipWith (fun x y => fa.intVal x * fb.intVal y) a b).sum
    (-2 ^ 31 ≤ s) → s < 2 ^ 31 → (intDot fa fb sat c a b).toInt = s

/-- **P1, BF16** on `TC.HOPPER_BF16_M16N8K16` (Hopper's `mma.sync` and `wgmma`, and sm_120's): on finite operands and a
finite accumulator whose values are integers, with the accumulator's and the products' magnitudes summing below
`2^24`, the dot is the exact sum. -/
def P1Bf16 : Prop :=
  ∀ (c : Word) (a b : List ℕ), a.length = 16 → b.length = 16 →
    (∀ x ∈ a ++ b, x < 2 ^ 16 ∧ Spec.bf16Class x ≠ .nan ∧ Spec.bf16Class x ≠ .inf) →
    Spec.fp32Class c.toNat ≠ .nan → Spec.fp32Class c.toNat ≠ .inf →
    let ps := List.zipWith (fun x y => termVal (Spec.bf16Product x y)) a b
    let c₀ := termVal (Spec.fp32Term c.toNat)
    (∃ z : ℤ, c₀ = z) → (∀ p ∈ ps, ∃ z : ℤ, p = z) → |c₀| + (ps.map fun p => |p|).sum < 2 ^ 24 →
    termVal (Spec.fp32Term (floatDot .bf16 HOPPER_BF16_M16N8K16 c a b).toNat) = c₀ + ps.sum

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

/-- **P2, priced moves**: every opcode uses at least `1/32` of an issue slot, at any tensor weights. -/
def P2Priced : Prop := ∀ (weight : Fmt → ℚ≥0) (o : Op Store), 1 / 32 ≤ opUse weight o .issue

end Verity.Accel
