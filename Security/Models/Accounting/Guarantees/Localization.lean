import Definitions.Accounting.Physical
import Definitions.Accounting.Localization

/-!
# Localization (trusted)

Section 6 of `note:console/20261012T0645Z-draft-actual-circuit-model` (v2.1), at `e = 0`, as a statement about one actual
circuit: when a challenge `C` reaches the prover through `X` during the window `W = [t₀, t₁)`, starting no later than
`C`'s ticks, what the rest `Y` of the prover reads during `W` from downstream of `C` is `X`'s, sent during `W`; and when
`X` sends the rest nothing of `C` (`e = 0`), the rest during `W` is not downstream of `C` at all, reads nothing that is,
and sends `X` nothing that is. So `X` during `W`, with the rest's sends to it fixed before `C` (at most `in_X(W)` bits,
the model's `h`), is all of the prover that `C` reaches: the lemma memory accounting's B2 and B3 reduce to. It takes
causality (`Causal`), the one physical constraint it uses. The proof is in
`Security/Proofs/Accounting/Actual/Localization.lean`.
-/

namespace Accounting.Guarantees

open Finset

/-- **Localization** (`e = 0`). For every actual circuit `A` under causality, challenge `C`, location `X` it reaches the
prover through during `W = [t₀, t₁)` (`t₀` at most each tick of `C`), and rest `Y` of the prover:

* every gate downstream of `C` that `A[Y, W]` reads from outside itself is in `A[X, W]` (`X`'s egress during `W`);
* when no gate of `C` is placed in `Y` and `X` sends `Y` nothing of `C` during `W`: no gate of `A[Y, W]` is downstream of
  `C`, nor is any gate it reads from outside itself, nor any gate placed in `Y` that `A[X, W]` reads. -/
def Localization : Prop :=
  ∀ {L : Type*} (A : Accounting.Placed L) (ℓ : L → L → ℕ) (C : Finset (Fin A.N)) (X Y : Finset L) (t₀ t₁ : ℕ),
    A.Causal ℓ → (∀ c ∈ C, t₀ ≤ A.t c) → A.Reaches C X Y (Set.Ico t₀ t₁) →
    (∀ a ∈ A.into (A.region Y (Set.Ico t₀ t₁)), a ∈ A.downstream C → a ∈ A.region X (Set.Ico t₀ t₁)) ∧
    ((∀ c ∈ C, A.x c ∉ Y) → A.SendsIndependent C X Y (Set.Ico t₀ t₁) →
      Disjoint (A.region Y (Set.Ico t₀ t₁)) (A.downstream C) ∧
      Disjoint (A.into (A.region Y (Set.Ico t₀ t₁))) (A.downstream C) ∧
      ∀ g ∈ A.region X (Set.Ico t₀ t₁), ∀ a ∈ A.reads g, A.x a ∈ Y → a ∉ A.downstream C)

end Accounting.Guarantees
