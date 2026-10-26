import Definitions.Core.Formalism

/-!
# The working draft's Lemma 1 and Theorem 3

Stated on core's circuit and partition (`Definitions.Core.Formalism`). Appendix A's translation of a `(c, k)`-partition
into NCI's coverings is `Nci.Guarantees.PartitionCovers`.
-/

namespace Verity.Guarantees

open Finset

/-- **The size of a `(c, k)`-partitioned circuit** (§3): `|C| = c · B`. -/
def CKSize : Prop :=
  ∀ (d : Declaration) (B c k : ℕ) (P : Partition d.D B), P.IsCK d c k → d.D.size d.D.computed = c * B

/-- **Lemma 1 (proof-block description).** Let `R` register a `(c, k)`-partition into `B` proof blocks, and let a fixed
evaluator `e` of at most `a_blk` bits perform an extraction rule `y` within time `t` for every proof block. Then in every
run of the registered circuit, each proof block's extracted string has local description length at most
`k + ⌈log₂ B⌉ + a_blk`. The registered information `r` is any the decoder is given, side information included. -/
def BlockDescription : Prop :=
  ∀ (R : Registration) (c k : ℕ), R.P.IsCK R.d c k →
    ∀ (ρ : Type) (M : Decoder ρ) (r : ρ) (t a_blk : ℕ) (e : List Bool)
      (y : Fin R.B → R.d.D.Transcript → List Bool),
      e.length ≤ a_blk → Evaluates M r t R e y →
      ∀ a, R.Honest a → ∀ u, LDL M t r (y u (R.d.D.eval a)) ≤ (k + Nat.clog 2 R.B + a_blk : ℕ)

/-- **Theorem 3, summed over any partition.** If each proof block `u` holds counted weight at most `q u` of the operation
gates `V`, the total counted weight is at most `Σ_u q u`. Proof blocks need not be of one size. -/
def CountingSum : Prop :=
  ∀ (C : Circuit) (B : ℕ) (P : Partition C B) (V : Finset (Fin C.N)) (ω : Fin C.N → ℝ) (q : Fin B → ℝ),
    V ⊆ C.computed → (∀ u, ∑ v ∈ V ∩ P.gates u, ω v ≤ q u) → ∑ v ∈ V, ω v ≤ ∑ u, q u

/-- **Theorem 3 (counting).** For a `(c, k)`-partition, operation gates `V` with weights `ω` and an allowance `q > 0` that
bounds every proof block's counted weight: `|C| ≥ (c / q) Σ_{v ∈ V} ω_v`, and relative to any reference of size `T₀ > 0`,
`ρ ≥ (c / (T₀ q)) Σ_{v ∈ V} ω_v`. -/
def Counting : Prop :=
  ∀ (d : Declaration) (B c k : ℕ) (P : Partition d.D B), P.IsCK d c k →
    ∀ (V : Finset (Fin d.D.N)) (ω : Fin d.D.N → ℝ) (q : ℝ), V ⊆ d.D.computed → 0 < q →
      (∀ u, ∑ v ∈ V ∩ P.gates u, ω v ≤ q) →
      (c : ℝ) / q * ∑ v ∈ V, ω v ≤ d.D.size d.D.computed ∧
        ∀ T₀ : ℝ, 0 < T₀ → (c : ℝ) / (T₀ * q) * ∑ v ∈ V, ω v ≤ slowdown d.D T₀

end Verity.Guarantees
