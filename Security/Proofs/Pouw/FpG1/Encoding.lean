import Definitions.Pouw.FpG1.Encoding

/-! The proof-only part of `Definitions.Pouw.FpG1.Encoding`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Pouw.FpG1

/-- The levels of `G` nearest to `v`. -/
def nearest (G : Finset ℤ) (v : ℚ) : Finset ℤ := G.filter fun g => ∀ h ∈ G, |v - g| ≤ |v - h|

/-- **Round to nearest onto `G`, ties to the lower level**: the least of the levels nearest to `v` (`0` for an empty
grid). -/
def rn (G : Finset ℤ) (v : ℚ) : ℤ := if h : (nearest G v).Nonempty then (nearest G v).min' h else 0

/-- **FP8-g1's committed integer** for a value `x` at scale `s`: the E4M3 level nearest `x / s` on the grid to `B`,
plus the coin. -/
def fp8 (B : ℕ) (s x : ℚ) (b : Coin) : ℤ := rn (e4m3Grid B) (x / s) + (b : ℕ)

namespace Entry

/-- What the int8 MMA reads: the level (times the block scale) plus the coin. -/
def int : Entry → ℤ
  | fp8 R b => R + (b : ℕ)
  | fp4 m q b => (m : ℤ) * q + (b : ℕ)
  | int8 R b => R + (b : ℕ)

/-- The centring: `½` for a g1 entry, `0` for Dither's. -/
def offset : Entry → ℚ
  | fp8 _ _ => 1 / 2
  | fp4 _ _ _ => 1 / 2
  | int8 _ _ => 0

/-- The value the served model multiplies: `int − offset`. -/
def served (e : Entry) : ℚ := (e.int : ℚ) - e.offset

end Entry

namespace Grids

/-- **sm_120 (RTX PRO 6000)**: activations on E4M3's integers to 120, weights to 60 and W4A8-g1's block scales to 5
(`steps.py`'s `e4m3i60` and `nvi5`), so that a weight moved by its coin stays in co32c's 7-bit weights. FPX-DESIGN
derived these for Dither-C's co32c; on H1 (co32c-u16, weights to ±110) they are conservative (RT2-FP-2). -/
def sm120 : Grids := ⟨120, 60, 5⟩

/-- **B200 and H200 on Dither-2S**: both sides on E4M3's integers to 120 and W4A8-g1's block scales to 10, so that an
operand moved by its coin stays in int8 (`e4m3i120` and `nvi10`). -/
def dc : Grids := ⟨120, 120, 10⟩

/-- **An entry on the device's grids at its side**: an activation is FP8-g1's on E4M3's integers to `act` or the
record's Dither int8 (`⌊v⌋` in `[−126, 126]`, top 127); a weight is FP8-g1's on E4M3's integers to `wt` or W4A8-g1's,
a doubled E2M1 level under a block scale in `[1, scale]`. -/
def On (g : Grids) : Side → Entry → Prop
  | .act, .fp8 R _ => R ∈ e4m3Grid g.act
  | .act, .fp4 _ _ _ => False
  | .act, .int8 R _ => -126 ≤ R ∧ R ≤ 126
  | .wt, .fp8 R _ => R ∈ e4m3Grid g.wt
  | .wt, .fp4 m q _ => 1 ≤ m ∧ m ≤ g.scale ∧ q ∈ e2m1x2Grid
  | .wt, .int8 _ _ => False

end Grids

end Pouw.FpG1
