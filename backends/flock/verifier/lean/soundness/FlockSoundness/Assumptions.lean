import ArkLib.Data.CodingTheory.ProximityGap.CapacityBounds

/-!
# Named assumptions

The results the soundness proof takes without proof. Each is a `Prop`, and each theorem that
uses one takes it as a hypothesis: the development declares no `axiom`, so `#print axioms`
lists only Lean's three standard axioms, and a mis-stated assumption can make a theorem
vacuous but can never make Lean prove something false.

Merkle binding is not an assumption: it is proved (VCVio's binding theorem, `Merkle`), and
SHA-256's collision resistance enters the compiled theorem as an explicit probability term (the
probability that an explicit, efficient reduction finds a collision), not as a hypothesis.
-/

namespace FlockSoundness.Assumptions

open scoped NNReal
open CodingTheory CoreDefinitions

/-- **Assumption 1: Reed–Solomon mutual correlated agreement up to the Johnson bound**
(BCHKS25 Theorem 4.6, ePrint 2025/2055, in the printed form that ArkLib pins as
`CodingTheory.rs_mcaError_le_in_johnson_range` and admits).

For a Reed–Solomon code of `k` coefficients on `n` points, reduced rate `ρ = (k − 1)/n`, and
radius `0 < δ < 1 − √ρ`: whatever two words `u₀, u₁` the prover fixes, the probability over a
uniform `γ ∈ F` that `u₀ + γ·u₁` agrees with a codeword on a set `T` of at least `(1 − δ)·n`
positions on which `u₀` and `u₁` are not both codewords is at most `a / |F|`, with
`m = max(⌈√ρ / (1 − √ρ − δ)⌉, 3)` and
`a = (2(m + ½)^5 + 3(m + ½)·δ·ρ) / (3ρ^{3/2})·n + (m + ½)/√ρ`.

Flock applies it with `F = GF(2^256)` (the folding challenges' field, PROTOCOL.md §13) to the
code of each Ligerito level embedded from GF(2^128). The Flock paper (Theorem 8) cites the
theorem with the smaller multiplicity `⌈√ρ / (2η)⌉`, which BCHKS25's proof supports but does not
state; the bound here uses the printed one, which costs nothing visible (the term is below
2^-170 against 2^-97.8 for the queries). -/
def BCHKS25Thm46 : Prop :=
  ∀ {ι : Type} [Fintype ι] [Nonempty ι] {F : Type} [Field F] [Fintype F] [SampleableType F]
    (domain : ι ↪ F) (k : ℕ) (δ : ℝ≥0), 1 < k → 0 < δ →
    (δ : ℝ) < 1 - ((((k - 1 : ℕ) : ℝ) / Fintype.card ι) ^ ((1 : ℝ) / 2)) →
    mcaError (AffineLineGenerator F) (ReedSolomon.code domain k) (δ : ℝ) ≤
      ENNReal.ofReal
        (let n : ℝ := Fintype.card ι
         let ρ : ℝ := (k - 1 : ℕ) / n
         let m : ℝ := max ⌈(ρ ^ ((1 : ℝ) / 2)) / (1 - ρ ^ ((1 : ℝ) / 2) - δ)⌉ 3
         ((2 * (m + 1/2) ^ 5 + 3 * (m + 1/2) * δ * ρ)
            / (3 * ρ ^ ((3 : ℝ) / 2)) * n
          + (m + 1/2) / ρ ^ ((1 : ℝ) / 2))
           / (Fintype.card F : ℝ))

end FlockSoundness.Assumptions
