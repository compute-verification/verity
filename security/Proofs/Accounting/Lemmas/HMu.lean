module

public import Definitions.Accounting.Guarantees.HMu

/-!
# (H-μ⁺) from the coins: Lemma B, the point mass of fresh noise, and Lemma B′ for persistent patterns: lemmas

The statements of `Definitions.Accounting.Guarantees.HMu` that are steps of its guarantees' proofs rather than
guarantees. Each is a `def G : Prop` under the guarantees' namespace, proved by the `SecurityProofs` theorem of the same
name, and audited with the proofs: no record pins it.

* `LemmaB`: under (H-μ⁺) at `(μ, Z)` (`SkipRound.HMuPlus`), a round is lucky with probability at most `Z·μ` times its
  probability of being deficient. This is (L)'s per-round form (`Accounting.TS.RoundGame.LuckAt`) at `p = Z·μ`, for a
  check that passes only when the committed receipt is the honest one.
* `PersistentOneWord`: Lemma B′ for a persistent pattern, one word a unit. For a fold whose step is injective in the
  state and in the word, if the pattern's first word is its only word before a committed checkpoint, the checkpoint's
  state is injective in the skipped value: (Inj) with `Z = 1`, and no differential term.
-/

@[expose] public section

namespace Accounting.Guarantees.HMu

open Accounting.HMu Pouw.RelativeSecurity

/-- **Lemma B (blind luck).** Under (H-μ⁺) at `(μ, Z)`, a round is lucky (deficient, with the honest receipt
committed) with probability at most `Z·μ` times its probability of being deficient. -/
def LemmaB : Prop :=
  ∀ (Ω O α β : Type) [Fintype Ω] (R : SkipRound Ω O α β) (μ : ℝ) (Z : ℕ), R.HMuPlus μ Z →
    prob R.lucky ≤ Z * μ * prob (fun ω => 0 < R.d (R.view ω))

/-- **Lemma B′ for a persistent pattern, one word a unit.** For a fold whose step is injective in the state for every
word and in the word for every state, a pattern `P` whose first position `i` is its only position before the checkpoint
`c > i`: the state at `c` of the words with `X` added at every position of `P` determines `X`. -/
def PersistentOneWord : Prop :=
  ∀ (S W : Type) [AddCommGroup W] (step : ℕ → S → W → S) (s₀ : S) (w : ℕ → W) (P : Finset ℕ) (i c : ℕ),
    (∀ n x, Function.Injective (fun s => step n s x)) → (∀ n s, Function.Injective (step n s)) →
    i ∈ P → i < c → (∀ n ∈ P, n ≠ i → c ≤ n) →
    Function.Injective (fun X : W => chain step s₀ (persist w P X) c)

end Accounting.Guarantees.HMu
