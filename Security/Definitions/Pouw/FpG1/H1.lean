import Definitions.Pouw.FpG1.Encoding

/-!
# FP8-g1 and W4A8-g1 on sm_120's record H1: the grids and a round's operand leaves (trusted)

What a g1 window on H1 (co32c-u16-cohalf, decision 82) commits, for `Definitions.Accounting.Guarantees.FpG1`'s statements.
H1 commits activations `V` with `|V| ≤ 126`, served `X = V + b` in `[−126, 127]`, and weights `W` with `|W| ≤ 110`,
served `W′ = W + c + U` with H1's noise `U ∈ [−16, 16]` in `[−126, 127]`; the coins `b` and `c` are drawn after the
round's leaves are committed.

* `Grids.h1`: g1's grids on H1: E4M3's integers to 120 for activations and to 104 for weights (`e4m3i104`), and
  W4A8-g1's block scales to 9 (`nvi9`), so that a weight moved by its coin and H1's noise stays in s8
  (`104 + 1 + 16 = 121`, `9·12 + 1 + 16 = 125`).
* `Level`: an operand's committed level, `Entry` without its coin, since a round's leaves hold the levels and the coin
  comes after them; `Level.withCoin` is the entry the int8 MMA reads once the coin is drawn.
* `Leaves`: a round's operand leaves, the committed levels of its activations `A(i, l)` and weights `W(l, j)`.
* `Leaves.Fp8 g`, `Leaves.W4a8 g`: every leaf an FP8-g1 level on `g`'s E4M3 grids, or the record's int8 activations
  beside W4A8-g1's weights on `g` (`Grids.On` at any coin: `Proofs.Pouw.FpG1.H1`'s `Leaves.fp8_on_act` and its
  siblings).
* `Leaves.InH1`: H1's committed ranges, `|V| ≤ 126` and `|W| ≤ 110`.
* `encodeFp8`, `encodeW4a8`: the honest encoders' leaves, RN onto the grids (`rn`), the record's floor for W4A8-g1's
  activations. No check reads them: like `rn`, they are one honest prover's choice.

No Program checks `Leaves.Fp8` or `Leaves.W4a8`: they describe what an honest g1 prover commits.
-/

namespace Pouw.FpG1

namespace Grids

/-- **g1 on sm_120's record H1** (co32c-u16-cohalf): activations on E4M3's integers to 120, weights to 104 and
W4A8-g1's block scales to 9 (`e4m3i104`, `nvi9`), so that a weight plus its coin plus H1's noise `U ∈ [−16, 16]` stays
in s8. -/
def h1 : Grids := ⟨120, 104, 9⟩

end Grids

/-- **An operand's committed level**: `Entry` without its coin. A round's leaves are committed before its coin is
drawn, so they hold the level. -/
inductive Level
  | fp8 (level : ℤ)
  | fp4 (scale : ℕ) (level : ℤ)
  | int8 (level : ℤ)

namespace Level

/-- The level with its coin: the entry the int8 MMA reads once the coin is drawn. -/
def withCoin : Level → Coin → Entry
  | fp8 R, b => .fp8 R b
  | fp4 m q, b => .fp4 m q b
  | int8 R, b => .int8 R b

/-- The committed integer: the level, times the block scale for W4A8-g1's weights. -/
def int : Level → ℤ
  | fp8 R => R
  | fp4 m q => (m : ℤ) * q
  | int8 R => R

end Level

/-- **A round's operand leaves**: the committed levels of its activations `A(i, l)` and its weights `W(l, j)`. -/
structure Leaves where
  act : ℕ → ℕ → Level
  wt : ℕ → ℕ → Level

namespace Leaves

/-- **FP8-g1's leaves on `g`**: every activation an E4M3 level on `g`'s activation grid and every weight one on its
weight grid. -/
def Fp8 (g : Grids) (v : Leaves) : Prop :=
  (∀ i l, ∃ R, v.act i l = .fp8 R ∧ R ∈ e4m3Grid g.act) ∧ ∀ l j, ∃ R, v.wt l j = .fp8 R ∧ R ∈ e4m3Grid g.wt

/-- **W4A8-g1's leaves on `g`**: every activation the record's int8 level in `[−126, 126]`, and every weight a doubled
E2M1 level under an integer block scale in `[1, g.scale]`. -/
def W4a8 (g : Grids) (v : Leaves) : Prop :=
  (∀ i l, ∃ R, v.act i l = .int8 R ∧ -126 ≤ R ∧ R ≤ 126) ∧
    ∀ l j, ∃ m q, v.wt l j = .fp4 m q ∧ 1 ≤ m ∧ m ≤ g.scale ∧ q ∈ e2m1x2Grid

/-- **H1's committed ranges**: every activation's level in `[−126, 126]` and every weight's in `[−110, 110]`. -/
def InH1 (v : Leaves) : Prop :=
  (∀ i l, -126 ≤ (v.act i l).int ∧ (v.act i l).int ≤ 126) ∧ ∀ l j, -110 ≤ (v.wt l j).int ∧ (v.wt l j).int ≤ 110

end Leaves

/-- **The honest FP8-g1 encoder's leaves**: the model's activations `x` and weights `y`, over their scales, rounded to
the nearest level of `g`'s E4M3 grids (`rn`). -/
def encodeFp8 (g : Grids) (sa sw : ℕ → ℚ) (x y : ℕ → ℕ → ℚ) : Leaves where
  act i l := .fp8 (rn (e4m3Grid g.act) (x i l / sa i))
  wt l j := .fp8 (rn (e4m3Grid g.wt) (y l j / sw j))

/-- **The honest W4A8-g1 encoder's leaves**: activations the record's floor `⌊x/s_a⌋`, clamped to `[−126, 126]`;
weights `y/(s_w·m)` rounded to the nearest doubled E2M1 level (`rn`) under the block scale `m`, clamped to
`[1, g.scale]`. -/
def encodeW4a8 (g : Grids) (sa sw : ℕ → ℚ) (m : ℕ → ℕ → ℕ) (x y : ℕ → ℕ → ℚ) : Leaves where
  act i l := .int8 (max (-126) (min 126 ⌊x i l / sa i⌋))
  wt l j := .fp4 (max 1 (min g.scale (m l j))) (rn e2m1x2Grid (y l j / (sw j * (max 1 (min g.scale (m l j)) : ℕ))))

end Pouw.FpG1
