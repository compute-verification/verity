import Mathlib.Data.Rat.Defs
import Definitions.Pous.Game.Oracle

/-!
# Encoding schemes

A scheme in the ideal model `M` fixes the sizes (`W` has `n` bits, `pp` has `t`, the codeword `C` has `B` blocks of
`ℓ` bits), a trusted randomized encoder, and a public decoder. The encoder sees the whole primitive (it may invert
it, as a trapdoor would let it), and its coins are erased after encoding. The decoder sees the primitive only
through queries and never sees the coins. There is no verifier key: responses are checked against the true blocks
(`Model/Audit.lean`), which idealizes the per-block tags.
-/

namespace Pous

/-- An encoding scheme in the ideal model `M`. -/
structure Scheme (M : IdealModel) : Type 1 where
  /-- `W` has `n` bits and `pp` has `t` bits. -/
  n : ℕ
  t : ℕ
  /-- The codeword has `B` blocks of `ℓ` bits. -/
  B : ℕ
  ℓ : ℕ
  B_pos : 0 < B
  /-- The encoder's coins, drawn uniformly and erased after encoding. -/
  Coins : Type
  [fintypeCoins : Fintype Coins]
  [nonemptyCoins : Nonempty Coins]
  /-- The trusted encoder `Enc(W) → (pp, C)`. -/
  enc : Bits n → M.Ω → Coins → Bits t × (Fin B → Bits ℓ)
  /-- The public decoder `Dec(pp, C)`: it reaches the primitive only through its query interface. -/
  dec : Bits t → (Fin B → Bits ℓ) → (M.Qry → M.Ans) → Bits n

attribute [instance] Scheme.fintypeCoins Scheme.nonemptyCoins

namespace Scheme

variable {M : IdealModel} (sch : Scheme M)

instance instNeZeroNatB : NeZero sch.B := ⟨sch.B_pos.ne'⟩

/-- A codeword: `B` blocks of `ℓ` bits. -/
abbrev Code : Type := Fin sch.B → Bits sch.ℓ

/-- Bit-exact public decoding: `Dec(pp, C) = W` for every `W`, primitive and coins. -/
def Correct : Prop :=
  ∀ (W : Bits sch.n) (ω : M.Ω) (r : sch.Coins),
    sch.dec (sch.enc W ω r).1 (sch.enc W ω r).2 (M.answer ω) = W

/-- The space bound `|C| + |pp| ≤ X·|W|` (the problem statement's `X = 1.05`). -/
def SpaceBound (X : ℚ) : Prop :=
  ((sch.B * sch.ℓ + sch.t : ℕ) : ℚ) ≤ X * sch.n

/-- `|C| ≥ |W|`. With the space bound this caps `|pp|` at `(X − 1)·|W|`, so `W` cannot sit in `pp`, which the
audit never checks. **Provisional:** a clarification of the problem statement, waiting on Daniel. -/
def CodeHoldsW : Prop := sch.n ≤ sch.B * sch.ℓ

end Scheme

end Pous
