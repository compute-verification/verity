module

public import Definitions.Core.Formalism
public import Definitions.Core.Boolean

/-!
# The working draft's Lemma 1 and Theorem 3

Stated on core's circuit and partition (`Definitions.Core.Formalism`). Appendix A's translation of a `(c, k)`-partition
into NCI's coverings is `Nci.Guarantees.PartitionCovers`, and NCI's policy as a `(c, k)`-partition is
`Nci.Guarantees.CompliesCK`. The serving blocks are inference blocks of at most `c_S` gates reading at most `k_S`, summed by
`ServingBlockSum`, with each serving block's inference cap from `InferenceCap`. The AND count `andSize`, in
which credit and reporting are stated, is at most `size` (`AndSizeLe`), and on Flock's circuits counts its AND gates
and rows (`ToCircuitAnd`). Credit under the pinned lowering is at most the lowered circuit's size (`CreditSizeLe`).
-/

@[expose] public section

namespace Verity.Guarantees

open Finset

/-- **The size of a `(c, k)`-partitioned circuit** (§3): `|C| ≤ c · B`. -/
def CKSize : Prop :=
  ∀ (d : RegisteredCircuit) (B c k : ℕ) (P : Partition d.D B), P.IsCK d c k → d.D.size d.D.computed ≤ c * B

/-- **ANDs are operation gates**: among any gates `S`, the AND gates number at most the operation gates, and so at most
`size S`. -/
def AndSizeLe : Prop :=
  ∀ (C : Circuit) (S : Finset (Fin C.N)), C.andSize S ≤ C.size (S ∩ C.computed) ∧ C.andSize S ≤ C.size S

/-- **Credit is at most the lowered circuit's size**: under any pinned lowering, a declaration's credit is at most the
number of operation gates of its lowered circuit. -/
def CreditSizeLe : Prop :=
  ∀ (lower : RegisteredCircuit → Circuit) (d : RegisteredCircuit), creditSize lower d ≤ (lower d).size (lower d).computed

/-- **Flock's AND gates are `IsAnd`.** On a Flock circuit, an AND gate of `toCircuit` is an `and` or a row (its XOR,
NOT and input gates are not AND gates); `and a b` with `a ≠ b` is one; and so is a row `(⊕ a) ∧ (⊕ b)` with a gate `x`
in `a` an odd number of times and in `b` an even number, and a gate `y` the other way round (the two XORs differ). -/
def ToCircuitAnd : Prop :=
  ∀ (B : Verity.Boolean.Circuit) (g : Fin B.N),
    (B.toCircuit.IsAnd g → (∃ a b, B.op g = .and a b) ∨ ∃ a b, B.op g = .row a b) ∧
      (∀ a b, B.op g = .and a b → a ≠ b → B.toCircuit.IsAnd g) ∧
      ∀ a b x y, B.op g = .row a b → Odd (a.count x) → Even (b.count x) → Even (a.count y) → Odd (b.count y) →
        B.toCircuit.IsAnd g

/-- **Lemma 1 (inference-block description).** Let `R` commit a `(c, k)`-partition into `B` inference blocks, and let a fixed
evaluator `e` of at most `a_blk` bits perform an extraction rule `y` within time `t` for every inference block. Then in every
run of the registered circuit, each inference block's extracted string has local description length at most
`k + ⌈log₂ B⌉ + a_blk`. The registered information `r` is any the decoder is given, side information included. -/
def BlockDescription : Prop :=
  ∀ (R : Committed) (c k : ℕ), R.d.𝓘.IsCK R.d c k →
    ∀ (ρ : Type) (M : Decoder ρ) (r : ρ) (t a_blk : ℕ) (e : List Bool)
      (y : Fin R.d.n → R.d.D.Transcript → List Bool),
      e.length ≤ a_blk → Evaluates M r t R e y →
      ∀ a, R.Honest a → ∀ u, LDL M t r (y u (R.d.D.eval a)) ≤ (k + Nat.clog 2 R.d.n + a_blk : ℕ)

/-- **Theorem 3, summed over any partition.** If each inference block `u` holds counted weight at most `q u` of the operation
gates `V`, the total counted weight is at most `Σ_u q u`. Inference blocks need not be of one size. -/
def CountingSum : Prop :=
  ∀ (C : Circuit) (B : ℕ) (P : Partition C B) (V : Finset (Fin C.N)) (ω : Fin C.N → ℝ) (q : Fin B → ℝ),
    V ⊆ C.computed → (∀ u, ∑ v ∈ V ∩ P.gates u, ω v ≤ q u) → ∑ v ∈ V, ω v ≤ ∑ u, q u

/-- **Theorem 3 (counting).** For any partition into `B` inference blocks, each priced at `c` gates, operation gates `V`
with weights `ω` and an allowance `q > 0` that bounds every inference block's counted weight:
`c · B ≥ (c / q) Σ_{v ∈ V} ω_v`, and relative to any reference of size `T₀ > 0`, the slowdown
`c · B / T₀ ≥ (c / (T₀ q)) Σ_{v ∈ V} ω_v`. The draft states it of a `(c, k)`-partition, whose `c · B` is at least `|C|`
(`CKSize`); the bound needs no more than the partition. -/
def Counting : Prop :=
  ∀ (d : RegisteredCircuit) (B c : ℕ) (P : Partition d.D B) (V : Finset (Fin d.D.N)) (ω : Fin d.D.N → ℝ) (q : ℝ),
    V ⊆ d.D.computed → 0 < q → (∀ u, ∑ v ∈ V ∩ P.gates u, ω v ≤ q) →
      (c : ℝ) / q * ∑ v ∈ V, ω v ≤ (c * B : ℕ) ∧
        ∀ T₀ : ℝ, 0 < T₀ → (c : ℝ) / (T₀ * q) * ∑ v ∈ V, ω v ≤ slowdown c B T₀

/-- **Theorem 3 at the serving blocks** (the serving-block bound's shape). Let the inference blocks `S` of a partition be
the serving blocks and the others the dark remainder. If each serving block's counted weight is at most both its serving
cap `qS u` (a inference-block cap at `(c_S, k_S)`) and its inference cap `qI u` (`n_S q_𝒯(c, k) + q_U(S)`), and the dark
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
