module

public import Definitions.Core.Prob
public import Definitions.Core.Hash
public import Mathlib.Data.Fintype.Card

/-!
# A check moved inside proofs

The memory rows that move a check of the auditor's verifier inside zero-knowledge proofs (PoUB's read session, PoUS's
setup, PoUS's session) state one thing each in this shape: a run of the proofs that the verifier accepts gives values
that today's check accepts, except with a stated probability.

A run of such a check is `n` proof blocks over the run's outcomes `Ω` (`Run`): whether the verifier accepts it, a value
of type `V u` at each proof block `u`'s ports, and the SHA-512 inputs the run carries. Its proof blocks may be of several
kinds, so each has a type of values of its own. Every field is free here: a guarantee that instantiates a row's lemma
builds `ports` from the counted extractor `IntegrityPeriod` reads, and `hashInputs` from the run, so that neither is a
value the prover picks.

`Gives μ r ok K₀ ε`: the verifier accepts while at least `K₀` proof blocks' values fail `ok`, with probability at most
`ε`. A row's lemma takes it at `ok` a relation `right` of its own, which the instantiating guarantee sets to each proof
block's circuit (correct on the values, with its registered output `1`), at `K₀ = 1` when every proof block is proved and
at the sample's threshold when it is sampled; and it concludes the same bound at `ok` today's check.
-/

@[expose] public section

namespace Verity.InProofs

open Verity.Game
open scoped ENNReal

/-- **A check run inside proofs, as one run leaves it**: over the run's outcomes `Ω`, whether the verifier accepts
(`accepts`), a value at proof block `u`'s ports (`ports u`), and the SHA-512 inputs the run carries (`hashInputs`). -/
structure Run (Ω : Type) (n : ℕ) (V : Fin n → Type) where
  /-- The verifier accepts the run. -/
  accepts : Ω → Prop
  /-- A value at proof block `u`'s ports. -/
  ports : (u : Fin n) → Ω → V u
  /-- The SHA-512 inputs the run carries. -/
  hashInputs : Ω → List Hash.Msg

open Classical in
/-- The proof blocks of `r` whose values fail `ok` at `ω`. -/
noncomputable def Run.failing {Ω : Type} {n : ℕ} {V : Fin n → Type} (r : Run Ω n V) (ok : (u : Fin n) → V u → Prop)
    (ω : Ω) : Finset (Fin n) :=
  Finset.univ.filter fun u => ¬ ok u (r.ports u ω)

/-- **An accepted run leaves fewer than `K₀` proof blocks failing `ok`, except with probability `ε`.** -/
def Gives {Ω : Type} {n : ℕ} {V : Fin n → Type} (μ : PMF Ω) (r : Run Ω n V) (ok : (u : Fin n) → V u → Prop)
    (K₀ : ℕ) (ε : ℝ≥0∞) : Prop :=
  prPMF μ (fun ω => r.accepts ω ∧ K₀ ≤ (r.failing ok ω).card) ≤ ε

end Verity.InProofs
