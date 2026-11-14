module

public import Definitions.Core.Prob
public import Mathlib.Data.Fintype.Card

/-!
# A check moved inside proofs

The memory rows that move a check of the auditor's verifier inside zero-knowledge proofs (PoUB's read session, PoUS's
setup, PoUS's session) state one thing each: an accepted run of the proofs gives values that today's check accepts, except
with a stated probability. This is the shape they state it in.

A run of such a check is `n` proof blocks over the run's outcomes `Ω` (`Run`): whether the verifier accepts it, and for each
proof block the values the extraction opens at its ports, from the committed trace fixed at registration. Its proof blocks may
be of several kinds, so each has a type of values of its own.

`Gives μ r ok K₀ ε`: the verifier accepts while at least `K₀` proof blocks' opened values fail `ok`, with probability at most
`ε`. Read at `ok` the relation the proof blocks' circuits define (a proof block's circuit is correct on the values with its
registered output `1`), it is what the proof system gives: at `K₀ = 1` when every proof block is proved, at the sample's
threshold when it is sampled. Read at `ok` today's check of one proof block's values, it is what a row states.
-/

@[expose] public section

namespace Verity.InProofs

open Verity.Game
open scoped ENNReal

/-- **A check run inside proofs, as one run leaves it**: over the run's outcomes `Ω`, whether the verifier accepts
(`accepts`), and the values the extraction opens at proof block `u`'s ports (`ports u`). -/
structure Run (Ω : Type) (n : ℕ) (V : Fin n → Type) where
  /-- The verifier accepts the run. -/
  accepts : Ω → Prop
  /-- The values the extraction opens at proof block `u`'s ports. -/
  ports : (u : Fin n) → Ω → V u

open Classical in
/-- The proof blocks of `r` whose opened values fail `ok` at `ω`. -/
noncomputable def Run.failing {Ω : Type} {n : ℕ} {V : Fin n → Type} (r : Run Ω n V) (ok : (u : Fin n) → V u → Prop)
    (ω : Ω) : Finset (Fin n) :=
  Finset.univ.filter fun u => ¬ ok u (r.ports u ω)

/-- **An accepted run leaves fewer than `K₀` proof blocks failing `ok`, except with probability `ε`.** -/
def Gives {Ω : Type} {n : ℕ} {V : Fin n → Type} (μ : PMF Ω) (r : Run Ω n V) (ok : (u : Fin n) → V u → Prop)
    (K₀ : ℕ) (ε : ℝ≥0∞) : Prop :=
  prPMF μ (fun ω => r.accepts ω ∧ K₀ ≤ (r.failing ok ω).card) ≤ ε

/-- **Values one relation accepts, another does**: `ok'` holds at every proof block's values `ok` holds at. -/
def Refines {n : ℕ} {V : Fin n → Type} (ok ok' : (u : Fin n) → V u → Prop) : Prop :=
  ∀ u v, ok u v → ok' u v

end Verity.InProofs
