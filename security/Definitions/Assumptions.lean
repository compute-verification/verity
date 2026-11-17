module

public import Definitions.Assumptions.Flock
public import Definitions.Hash
public import Definitions.Core
public import Definitions.ResourceAccounting
public import Definitions.Assumptions.ResourceAccounting
public import Definitions.Assumptions.Hash
public import Definitions.Assumptions.Pous
public import Definitions.Slowdown.Cut
public import Definitions.Slowdown.Formalism
public import Definitions.Assumptions.Pouw
public import Definitions.Assumptions.WeightSecurity

/-! The named assumptions: the `Prop`s a guarantee takes as hypotheses without proof, never as an `axiom`. -/

@[expose] public section

/-! ## The min-cut bound's cut conjectures (H) and (E): no check of the circuit decides either, and one set of gates that beats it refutes it. -/

namespace Slowdown.Cut.Assumptions

open Verity Finset

/-- `training-cut-hardness` (conjectured; refuted by a set `S` of words with `κ(S) ≤ k` whose values cost more than
`κ(S)`). **(H), hardness** at arrays `A`, costs `β` a value and read cap `k`: every set `S` of words of `d` with
`κ(S) ≤ k` has `∑_r β_r |val(S ∩ A r)| ≤ κ(S)`. -/
def Hardness {ρ V : Type} [Fintype ρ] [DecidableEq V] (d : RegisteredCircuit) (A : ρ → Finset (Finset (Fin d.D.N)))
    (val : Finset (Fin d.D.N) → V) (β : ρ → ℝ) (k : ℕ) : Prop :=
  ∀ S, kappaWords d S ≤ k → ∑ r, β r * ((S ∩ A r).image val).card ≤ kappaWords d S

/-- `training-cut-exports` (conjectured; refuted by a set `B` of computed gates with `κ(B → E(B)) < b |E(B)|`).
**(E), exports** at `b`: every set `B` of computed gates has `κ(B → E(B)) ≥ b |E(B)|`. -/
def Exports {C : Circuit} {ι : Type} [Fintype ι] [DecidableEq ι] (W : Work C ι) (b : ℝ) : Prop :=
  ∀ B ⊆ C.computed, b * (W.clear B).card ≤ kappa C B ((W.clear B).biUnion W.y)

end Slowdown.Cut.Assumptions

/-! ## The training formalism's assumptions A1, the token-work bound and A2: the token-work bound does not follow from A1,
and A2 is empirical, stated over a class of runs and not over every circuit. -/

namespace Slowdown.Formalism.Assumptions

open Finset
open Verity (RegisteredCircuit)
open Slowdown.Formalism
open scoped NNReal ENNReal

/-- `training-cut-cost` (conjectured; its values are the reads of the witness blocks `𝒲`, and it is refuted by an
inference block that reads less). **A1, cut cost**: every inference block `I` of `d` reads beyond `θ` at least, for
every token `T` it works on, the state bits `a_T(e_T(I))` where its work on `T` begins, the least of `T`'s own gates
that a witness block of `𝒲` entering there reads, and its shared bits `u_I`, what those witness blocks read beyond the
tokens' own state, counted once and net of what `I` computes: `|in(I) \ θ| ≥ u_I + Σ_T a_T(e_T(I))`. -/
def CutCost {d : RegisteredCircuit} {L : Type} [Fintype L] (A : Annotation d L) (c k : ℕ) {E : Type}
    (e : Finset (Fin d.D.N) → L → E) (𝒲 : Finset (Finset (Fin d.D.N))) : Prop :=
  ∀ I, InferenceBlock d c k I →
    A.blockShared e 𝒲 I + ∑ T ∈ A.worksOn I, A.stateAt e 𝒲 T (e I T) ≤ (d.unregistered I).card

/-- `training-token-work` (conjectured; its values are A1's at the witness blocks `𝒲`, and it is refuted by inference
blocks with shared bits at most `ν` that, entering tokens `S` at entries `f` with `ν + Σ_{T ∈ S} a_T(f_T) ≤ k`, hold
more than `cap` gates of `X` in those tokens). **The token-work bound**: `X` is a set of token gates, and at every set
of tokens `S`, entry profile `f` and shared bits `ν` within A1's budget, `ν + Σ_{T ∈ S} a_T(f_T) ≤ k`, the token work
from those entries is at most `cap`, `Σ_{T ∈ S} W_T(f_T, ν) ≤ cap`. -/
def TokenWork {d : RegisteredCircuit} {L : Type} [Fintype L] (A : Annotation d L) (c k : ℕ) {E : Type}
    (e : Finset (Fin d.D.N) → L → E) (𝒲 : Finset (Finset (Fin d.D.N))) (X : Finset (Fin d.D.N)) (cap : ℝ≥0) : Prop :=
  X ⊆ A.tokenGates ∧
    ∀ (S : Finset L) (f : L → E) (ν : ℕ), ν + ∑ T ∈ S, A.stateAt e 𝒲 T (f T) ≤ k →
      ∑ T ∈ S, (A.entryWork c k e 𝒲 X T (f T) ν : ℝ≥0) ≤ cap

/-- `training-efficiency-floor` (empirical; falsified by a measured run). **A2, efficiency floor**: no run of `methods`
with a valid annotation whose update stays within `D` bits (its `Δ`, decoded by `M` within `c` steps) reaches `ℓ*` for
less than `p · H(ℓ*)`, `H` the honest cost over `methods`. -/
def EfficiencyFloor (methods : Set Run) (M : Verity.Decoder (List Bool)) (c : ℕ) (ℓ : List Bool → ℝ) (ℓstar : ℝ)
    (D : ℕ∞) (p : ℝ) : Prop :=
  ∀ r ∈ methods, r.A.Valid → r.Within M c D → r.Reaches ℓ ℓstar →
    ENNReal.ofReal p * honestCost methods ℓ ℓstar ≤ (r.d.D.computed.card : ℝ≥0∞)

end Slowdown.Formalism.Assumptions
