module

public import Definitions.Core.Coins
public import Definitions.Spot.Dataflow
public import Mathlib.Basic.Real.Basic
public import Mathlib.Algebra.BigOperators.Group.Finset.Defs

/-!
# The IOP's sampling certificate, and what each policy needs of it

The IOP: the prover sends a transcript `σ : T` (an oracle) before any coin; the verifier samples `k` units independently
from a distribution `π` fixed before the transcript, checks each sampled proof block, and checks every unit of an always-proved
set `A`. There are no hashes: binding is free, and the protocol fixes the oracle's shape.

* `chk u σ`: unit `u`'s check on the transcript. `bad chk σ` is the incorrect set `B(σ)`, the units whose check fails.
* `π : PMF U`: the sample distribution. `mass π B` is its weight on `B`, the probability that a sample lands in `B`.
* `Accepts chk A d σ`: at the sample `d : Fin k → U`, every always-proved check and every sampled check passes.
* `prob π k E`: the probability of an event of the sample over `k` independent samples from `π`, core's
  `Verity.Game.prPMFReal` under `iid π k`. A transcript committed before the coins is a `σ` that the event reads but
  that does not depend on `d`.
* `Certified chk π A ε σ`, **the certificate's conclusion**: no always-proved unit is incorrect, and `π(B) ≤ ε`.
* `Local reads chk`: unit `u`'s check reads only the oracle positions `reads u`. A population's checks are local by
  type (`Dataflow.Fns`).
* `AmpLe chk A c dmg K`, **amplification at most `K`**: off the always-proved set, a policy's damage `dmg σ` is at most
  `K` times the incorrect proof blocks' cost `∑_{u ∈ B} c u`. The policy is inferable from the certificate exactly when such a
  `K` is known (`Proofs.Spot.NecessitySkeleton`, `inference_iff`).

For a population with fixed unit functions (`Dataflow`):
* `DeterminateAt P f a`: exactly one assignment holds the inputs `a` and makes every unit correct, for these `f`.
* `RobustAt P f`: an assignment is determined by its inputs and the writes of any set containing its incorrect proof blocks,
  for these `f`. `Dataflow.eq_of_writes` proves it from single assignment and acyclicity, for every `f`.
* `damaged O σ τ`: the egress slots where `σ` and the honest `τ` differ, integrity's damage.
* `egressWidth f a X O M`: the egress values of the runs whose incorrect proof blocks write at most `M` slots, control measured
  at the public write widths.

For a program (`DynPopulation`):
* `rangeFlow D R`: the static dataflow in which unit `u` reads the public bound `R u`, the range a computed address is
  confined to.
-/

@[expose] public section

namespace Population.Iop

open Finset

variable {U T : Type}

open Classical in
/-- The incorrect set: the units whose check fails on the transcript. -/
noncomputable def bad [Fintype U] (chk : U → T → Prop) (σ : T) : Finset U := univ.filter fun u => ¬ chk u σ

/-- The sample's weight on a set of units: the probability that a sample from `π` lands in it. -/
noncomputable abbrev mass (π : PMF U) (B : Finset U) : ℝ := Verity.Game.prPMFReal π (· ∈ B)

/-- The verifier accepts the transcript `σ` at the sample `d`: every always-proved unit's check passes, and every sampled
proof block's. -/
def Accepts (chk : U → T → Prop) (A : Finset U) {k : ℕ} (d : Fin k → U) (σ : T) : Prop :=
  (∀ u ∈ A, chk u σ) ∧ ∀ i, chk (d i) σ

/-- The probability of an event of the sample, over `k` independent samples from `π`: core's `Verity.Game.prPMFReal`
under their law `iid π k`. -/
noncomputable abbrev prob [Fintype U] (π : PMF U) (k : ℕ) (E : (Fin k → U) → Prop) : ℝ :=
  Verity.Game.prPMFReal (Verity.Game.iid π k) E

/-- **The certificate's conclusion** about the transcript `σ`: no always-proved unit is incorrect, and the sample's weight on
the incorrect proof blocks is at most `ε`. -/
def Certified [Fintype U] (chk : U → T → Prop) (π : PMF U) (A : Finset U) (ε : ℝ) (σ : T) : Prop :=
  Disjoint (bad chk σ) A ∧ mass π (bad chk σ) ≤ ε

/-- **Local checks**: unit `u`'s check reads only the oracle positions `reads u`. -/
def Local {Q V : Type} (reads : U → Finset Q) (chk : U → (Q → V) → Prop) : Prop :=
  ∀ u σ τ, (∀ q ∈ reads u, σ q = τ q) → (chk u σ ↔ chk u τ)

/-- **Amplification at most `K`**: on every transcript whose incorrect set misses the always-proved set, the damage is at
most `K` times the incorrect proof blocks' cost. -/
def AmpLe [Fintype U] (chk : U → T → Prop) (A : Finset U) (c : U → ℝ) (dmg : T → ℝ) (K : ℝ) : Prop :=
  ∀ σ, Disjoint (bad chk σ) A → dmg σ ≤ K * ∑ u ∈ bad chk σ, c u

section Dataflow

variable {S : Type} (P : Dataflow U S)

/-- A population's checks: unit `u` is correct in the assignment. -/
def corr {V : Type} (f : P.Fns V) (u : U) (σ : S → V) : Prop := P.Correct f σ u

/-- **Determinacy at the given unit functions**: exactly one assignment holds the inputs and makes every unit
correct. -/
def DeterminateAt {V : Type} (f : P.Fns V) (a : S → V) : Prop :=
  ∃! σ : S → V, (∀ s ∈ P.input, σ s = a s) ∧ ∀ u, P.Correct f σ u

/-- **Robust determinacy at the given unit functions**: two assignments with the same inputs, whose incorrect proof blocks lie in
`B`, and that agree on what `B` writes, are equal. -/
def RobustAt [Fintype U] {V : Type} (f : P.Fns V) : Prop :=
  ∀ (B : Finset U) (σ τ : S → V), (∀ s ∈ P.input, σ s = τ s) → P.incorrect f σ ⊆ B → P.incorrect f τ ⊆ B →
    (∀ u ∈ B, ∀ s ∈ P.writes u, σ s = τ s) → σ = τ

open Classical in
/-- **Integrity's damage**: the egress slots where the run `σ` differs from the honest run `τ`. -/
noncomputable def damaged {V : Type} (O : Finset S) (σ τ : S → V) : Finset S := O.filter fun o => σ o ≠ τ o

open Classical in
/-- **The egress values within a write budget `M`**: what the assignments that hold the inputs `a` off the secret slots
`X`, and whose incorrect proof blocks write at most `M` slots in all, put on the egress `O`. Control at the public widths. -/
noncomputable def egressWidth [Fintype U] [Fintype S] {V : Type} [Fintype V] (f : P.Fns V) (a : S → V)
    (X O : Finset S) (M : ℕ) : Finset (O → V) :=
  (univ.filter fun σ : S → V => P.Holds a X σ ∧ ∑ u ∈ P.incorrect f σ, (P.writes u).card ≤ M).image fun σ o => σ o

end Dataflow

/-- **The static dataflow of a public read bound** `R`: a program's inputs and writes, with unit `u` reading `R u`.
When `R u` holds every slot `u` reads in every admitted run, `R` is a range restriction: addresses may be computed and
corrupted, but stay inside `R u`. -/
def rangeFlow {V S : Type} (D : DynPopulation U S V) (R : U → Finset S) : Dataflow U S where
  input := D.input
  reads := R
  writes := D.writes

end Population.Iop
