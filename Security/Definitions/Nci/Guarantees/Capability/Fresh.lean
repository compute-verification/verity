import Definitions.Nci.Capability.Fresh

/-!
# Local bounds: the effective fresh input

The budgets (`Budgets`) read a block's `k` unregistered input bits and its evaluator's `aBlk` bits. Two statements put an
inference block's other fresh input (`Security/README.md`, part 3: `k_{a,I}` and `k_{F,I}`) into those parameters, so
every bound of the repertoire takes it with no restatement:

* `BlockDescriptionAdvice`, the advice's share: Lemma 1 (`Verity.Guarantees.BlockDescription`) for an evaluator that
  reads the advice, `k + ⌈log₂ B⌉ + aBlk + ℓa`. For each advice `a`, `e ++ a` is an evaluator of `build a` of at most
  `aBlk + ℓa` bits, so `HeldBudget` and `RecoveredBudget` apply to `build a` at `aBlk + ℓa`.
* `FaultyInjection`, what faulty proof blocks inject: a trace in which at most `m` faulty proof blocks of at most `w`
  output bits each have outputs inside any one inference block is, on every committed wire, an honest run of the cut
  registration, a `(c, k + m·w)`-partition; every bound stated for honest runs of a `(c, k)`-partition applies to it at
  `k + m·w`.
-/

namespace Nci.Guarantees

open Verity Finset Nci.Capability

/-- **Lemma 1 with advice** (the advice's share, `k_a`). The builder makes a registration `build a` of each advice `a` of
`ℓa` bits, so every block's structure may depend on all of `a`. If one evaluator `e` of at most `aBlk` bits, followed by
the advice, evaluates `y a` on every block of `build a` within `t`, then in every honest run of a `(c, k)`-partition
`build a`, each block's output has a description of at most `k + ⌈log₂ B⌉ + aBlk + ℓa` bits. -/
def BlockDescriptionAdvice : Prop :=
  ∀ (build : List Bool → Registration) (ℓa c k : ℕ) (ρ : Type) (M : Decoder ρ) (r : ρ) (t aBlk : ℕ)
    (e : List Bool) (y : (a : List Bool) → Fin (build a).B → (build a).d.D.Transcript → List Bool),
    e.length ≤ aBlk → EvaluatesAdvice M r t build ℓa e y →
    ∀ a, a.length = ℓa → (build a).P.IsCK (build a).d c k →
      ∀ x, (build a).Honest x → ∀ u,
        LDL M t r (y a u ((build a).d.D.eval x)) ≤ (k + Nat.clog 2 (build a).B + aBlk + ℓa : ℕ)

/-- **Faulty proof blocks inject at most `m · w` bits** into an inference block (`k_F`). Let `R` be a `(c, k)`-partition
into inference blocks, `Q` a partition of the same circuit into proof blocks, and `X` a trace whose registered inputs
hold their registered values, in which each faulty proof block has at most `w` output bits and at most `m` faulty proof
blocks have outputs inside any one inference block. Then the cut registration at what they inject is a
`(c, k + m · w)`-partition, `X` on both of its copies is an honest run of it, and that run's evaluation is `X` on every
wire `Q` commits. A faulty block's outputs that an inference block reads from outside it are among its unregistered
inputs already, counted in `k`; only those inside it, which it would otherwise compute, add to its fresh input. -/
def FaultyInjection : Prop :=
  ∀ (R : Registration) (c k : ℕ), R.P.IsCK R.d c k →
    ∀ (n : ℕ) (Q : Partition R.d.D n) (X : R.d.D.Transcript) (m w : ℕ), R.Honest X →
      (∀ v ∈ Q.wrong X, R.d.D.width (R.d.D.exits (Q.gates v)) ≤ w) →
      (∀ u, ((Q.wrong X).filter fun v => ¬ Disjoint (R.d.D.exits (Q.gates v)) (R.P.gates u)).card ≤ m) →
      (cutReg R (injected Q X)).P.IsCK (cutReg R (injected Q X)).d c (k + m * w) ∧
        (cutReg R (injected Q X)).Honest (cutInputs (injected Q X) X) ∧
        ∀ g ∈ Q.committed, (cutReg R (injected Q X)).d.D.eval (cutInputs (injected Q X) X)
          (Fin.natAdd R.d.D.N g) = (X g).setWidth 1

end Nci.Guarantees
