import Definitions.Pouw.Assumptions.Fp8Atom.H1TTile

/-!
# H-1T's per-tile count statement (a named assumption)

* **`TTH1T r c γ η`**: take legal units with independent
  salts, a tiling of their outputs whose every tile spans at least `r` rows and `c` columns (`Tiling.AtLeast`), and a
  well-formed `MH100w` program (draft semantics) on free inputs with a fixed output slot per credited word, under
  `H32`. Over the joint salt, the probability that `(1 − γ)·32·rightWords > cost` is at most `η` of the number of
  tiles. For tiles of `w` credited words each: except with probability `η`, at most `cost/((1 − γ)·32·w)` tiles are
  right. This is the count form that Theorem 3.1 and sampled proofs' layout A need, as `TTNCP_U` has: a
  union over the salt-dependent sets of right tiles would be vacuous.

**The domain** is layout A's: its tiles are `16 × 16` blocks, `AtLeast 16 16`. A free constant is right on a
tile at the salts where the tile's words take that value, and rows have independent salts, so such a lucky guess gets
a tile of `r` rows right with a probability that falls geometrically in `r`. Over all tilings, a program with no
instructions gets one-row tiles right often enough that `η` must tend to 1 (`η N ≥ 1 − (63/64)^N`, on the
zero-coded unit). Columns share their row's salt, so a count of credited words would not bound the guess, and the
domain states what layout A fixes before the salt, as `TTNCP_U`'s `D` does for NCP.

It counts tiles by their credited words alone, so Z (`R_(2T−1)`, not credited) is outside it. Its input is
`HalfApartH1T`, with independent row salts giving the tail over rows. The lifting from pointwise-right tiles to a
cost bound is research (the FP8 analogue of `LiftMinRank`), so `TTH1T` is Assumed, with `HalfApartH1T` and the
simulation evidence. `MH100w` is straight-line, so branching is outside it. Two pins check its form: at
`γ = 0` a program right on every tile at every salt is never in the event (`tth1tAllRight`, from `H1TGamma`, over
every tiling), and it holds at `γ = 1`, `η = 0` for every domain (`tth1tSat`).
-/

namespace Pouw.Fp8Atom.H1T

open Pouw.Dimension.H100

/-- **`TTH1T r c γ η`** (Assumed): under `H32`, for legal units `Us`, every tiling `tl` of their outputs by the tiles
`C` whose every tile spans at least `r` rows and `c` columns, and every well-formed `MH100w` program on free inputs
with output slots `slot`, the joint salts `G` at which `(1 − γ)·32·rightWords` exceeds the program's cost are at most
an `η (Nat.card C)` share of all joint salts. -/
def TTH1T (r c : ℕ) (γ : ℚ) (η : ℕ → ℚ) : Prop :=
  ∀ pr : Prices100, H32 pr → ∀ (N : ℕ) (Us : Fin N → Unit), (∀ u, (Us u).Legal) →
    ∀ (C : Type) (tl : Tiling Us C), tl.AtLeast r c → ∀ (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (MSalt Us → ℚ))
      (P : List (Op100 (MSalt Us))) (slot : Idx Us → ℕ),
      (∀ f ∈ L, f ∈ mfree Us) → (∀ o ∈ P, o.WFw pr (mfree Us)) →
      ({G : MSalt Us | (cost100 pr P : ℚ) < (1 - γ) * 32 * rightWords (regs100 fpSem P L) slot tl G}.ncard : ℚ) ≤
        η (Nat.card C) * Nat.card (MSalt Us)

end Pouw.Fp8Atom.H1T
