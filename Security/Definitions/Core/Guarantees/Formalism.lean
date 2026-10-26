import Definitions.Core.Formalism

/-!
# The working draft's Lemma 1 and Theorem 3

Stated on core's circuit and partition (`Definitions.Core.Formalism`). Appendix A's translation of a `(c, k)`-partition
into NCI's coverings is `Nci.Guarantees.PartitionCovers`, and NCI's policy as a padded `(c, k)`-partition is
`Nci.Guarantees.CompliesCK`. The serving blocks are a `Padding` at `(c_S, k_S)`, summed by `ServingBlockSum`, with each
serving block's inference cap from `InferenceCap`.
-/

namespace Verity.Guarantees

open Finset

/-- **The size of a `(c, k)`-partitioned circuit** (§3): `|C| = c · B`. -/
def CKSize : Prop :=
  ∀ (d : Declaration) (B c k : ℕ) (P : Partition d.D B), P.IsCK d c k → d.D.size d.D.computed = c * B

/-- **Lemma 1 (block description).** Let `R` register a `(c, k)`-partition into `B` blocks, and let a fixed evaluator
`e` of at most `a_blk` bits perform an extraction rule `y` within time `t` for every block. Then in every run of the
registered circuit, each block's extracted string has local description length at most `k + ⌈log₂ B⌉ + a_blk`. The
registered information `r` is any the decoder is given, side information included. -/
def BlockDescription : Prop :=
  ∀ (R : Registration) (c k : ℕ), R.P.IsCK R.d c k →
    ∀ (ρ : Type) (M : Decoder ρ) (r : ρ) (t a_blk : ℕ) (e : List Bool)
      (y : Fin R.B → R.d.D.Transcript → List Bool),
      e.length ≤ a_blk → Evaluates M r t R e y →
      ∀ a, R.Honest a → ∀ u, LDL M t r (y u (R.d.D.eval a)) ≤ (k + Nat.clog 2 R.B + a_blk : ℕ)

/-- **Theorem 3, summed over any partition.** If each block `u` holds counted weight at most `q u` of the operation gates
`V`, the total counted weight is at most `Σ_u q u`. Blocks need not be of one size. -/
def CountingSum : Prop :=
  ∀ (C : Circuit) (B : ℕ) (P : Partition C B) (V : Finset (Fin C.N)) (ω : Fin C.N → ℝ) (q : Fin B → ℝ),
    V ⊆ C.computed → (∀ u, ∑ v ∈ V ∩ P.gates u, ω v ≤ q u) → ∑ v ∈ V, ω v ≤ ∑ u, q u

/-- **Theorem 3 (counting).** For a `(c, k)`-partition, operation gates `V` with weights `ω` and an allowance `q > 0` that
bounds every block's counted weight: `|C| ≥ (c / q) Σ_{v ∈ V} ω_v`, and relative to any reference of size `T₀ > 0`,
`ρ ≥ (c / (T₀ q)) Σ_{v ∈ V} ω_v`. -/
def Counting : Prop :=
  ∀ (d : Declaration) (B c k : ℕ) (P : Partition d.D B), P.IsCK d c k →
    ∀ (V : Finset (Fin d.D.N)) (ω : Fin d.D.N → ℝ) (q : ℝ), V ⊆ d.D.computed → 0 < q →
      (∀ u, ∑ v ∈ V ∩ P.gates u, ω v ≤ q) →
      (c : ℝ) / q * ∑ v ∈ V, ω v ≤ d.D.size d.D.computed ∧
        ∀ T₀ : ℝ, 0 < T₀ → (c : ℝ) / (T₀ * q) * ∑ v ∈ V, ω v ≤ slowdown d.D T₀

/-- **Padding to a `(c, k)`-partition.** On a Boolean declaration, blocks of at most `c` gates that each read at most `k`
unregistered input bits pad to a `(c, k)`-partition of a declaration that contains it (`Pad`). The serving blocks are
this at `(c_S, k_S)`: each server's region between two consecutive wipes, padded to exactly `c_S` gates. -/
def Padding : Prop :=
  ∀ (d : Declaration) (B c k : ℕ) (P : Partition d.D B), d.D.Boolean →
    (∀ u, (P.gates u).card ≤ c) → (∀ u, (d.unregistered (P.gates u)).card ≤ k) →
    ∃ (d' : Declaration) (P' : Partition d'.D B), Nonempty (Pad d P d' P') ∧ P'.IsCK d' c k

/-- **Theorem 3 at the serving blocks** (the serving-block bound's shape). Let the blocks `S` of a partition be the
serving blocks and the others the dark remainder. If each serving block's counted weight is at most both its serving cap
`qS u` (a per-block cap at `(c_S, k_S)`) and its inference cap `qI u` (`n_S q_𝒯(c, k) + q_U(S)`), and the dark
remainder's at most `qDark`, the total counted weight is at most `Σ_{u ∈ S} min(qS u, qI u) + qDark`. -/
def ServingBlockSum : Prop :=
  ∀ (C : Circuit) (B : ℕ) (P : Partition C B) (S : Finset (Fin B)) (V : Finset (Fin C.N)) (ω : Fin C.N → ℝ)
    (qS qI : Fin B → ℝ) (qDark : ℝ), V ⊆ C.computed →
    (∀ u ∈ S, ∑ v ∈ V ∩ P.gates u, ω v ≤ qS u) → (∀ u ∈ S, ∑ v ∈ V ∩ P.gates u, ω v ≤ qI u) →
    ∑ u ∈ univ \ S, ∑ v ∈ V ∩ P.gates u, ω v ≤ qDark →
    ∑ v ∈ V, ω v ≤ ∑ u ∈ S, min (qS u) (qI u) + qDark

/-- **The inference cap.** When the inference blocks `Pf` lie inside the serving blocks `Pc` (`Refines`: the alignment
requirement) and each inference block's counted weight outside `U` is at most `q`, a serving block's counted weight is
at most `q` times its number of inference blocks plus its counted weight in `U`: `U` holds the counted operations not
computed wholly inside one inference block, which `q_U(S)` caps. -/
def InferenceCap : Prop :=
  ∀ (C : Circuit) (nf nc : ℕ) (Pf : Partition C nf) (Pc : Partition C nc) (r : Pf.Refines Pc)
    (V U : Finset (Fin C.N)) (ω : Fin C.N → ℝ) (q : ℝ), V ⊆ C.computed →
    (∀ i, ∑ v ∈ (V \ U) ∩ Pf.gates i, ω v ≤ q) →
    ∀ s, ∑ v ∈ V ∩ Pc.gates s, ω v ≤
      ((univ.filter fun i => r.parent i = s).card : ℝ) * q + ∑ v ∈ (V ∩ U) ∩ Pc.gates s, ω v

end Verity.Guarantees
