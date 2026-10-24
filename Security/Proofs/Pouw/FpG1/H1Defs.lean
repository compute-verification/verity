import Definitions.Pouw.FpG1.H1
import Proofs.Pouw.FpG1.Encoding

/-! The proof-only part of `Definitions.Pouw.FpG1.H1`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Pouw.FpG1

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

namespace Leaves

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
