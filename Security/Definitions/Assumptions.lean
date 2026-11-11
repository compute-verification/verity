module

public import Definitions.Assumptions.Flock
public import Definitions.Hash
public import Definitions.Core
public import Definitions.Accounting
public import Definitions.Nci.Cut
public import Definitions.Nci.Formalism
public import Definitions.Assumptions.Pouw
public import Definitions.Assumptions.WeightSecurity

/-!
# The named assumptions

The module a named assumption is declared in, with the modules under it (`assumptions_module` in `lean-audit.json`): the
few `Prop`s the guarantees take without proof because they are real assumptions (a hash's collision resistance, the
auditor's devices), each taken as a hypothesis, never an `axiom`. Every other premise a guarantee takes is a domain
hypothesis, part of what the guarantee says, which the lock lists under `domain` with a one-line `why`, or open: the
lock lists it under `open` with the issue that discharges it and its due date. An area moves an assumption of its own
here once it is one of the real ones, into a module under this one that reads nothing private when the area's readers
mustn't import the rest, since this one reads Nci: Flock's are `Definitions.Assumptions.Flock`, PoUW's hashless cost
models `Definitions.Assumptions.Pouw` and weight security's model claim under faults
`Definitions.Assumptions.WeightSecurity`, all of which this module re-exports.
-/

@[expose] public section

/-!
## The min-cut bound's cut conjectures, (H) and (E)

`Nci.Cut.Guarantees.Bound` holds for every circuit; `HE` and `HEShared` bound its `Π(k)` and `α` under two premises
on cuts, each a `Prop` about the circuit, its charged work and the parameters. Neither is decided by a check of the
circuit: each is a cut conjecture, refuted by one set of gates that beats it.

* **(H), hardness** (`Hardness`): the values a set of words carries cost cut gates. Every set `S` of words with
  `κ(S) ≤ k` has `∑_r β_r |val(S ∩ A_r)| ≤ κ(S)`: each value of array `A_r` that `S`'s words carry takes `β_r` of the
  gates separating them from the unregistered inputs. It fails for values a few gates determine: a weight barely moved
  since `θ`, a copy of another value under a name `val` doesn't identify, an activation recomputed from a narrower cut.
* **(E), exports** (`Exports`): the outputs a set of computed gates finishes cost cut gates to send on. Every set `B` of
  computed gates has `κ(B → E(B)) ≥ b |E(B)|`, `E(B)` the outputs fed by a MAC inside `B` with no wire in `B`. At
  `b = b_y` (`Nci.Cut.Work.bY`) it holds by definition; at a given `b` it says `b_y ≥ b`, which no check computes, and
  it fails for outputs a few gates determine, such as a low-rank product's.

Both count gates of the circuit and its values, never a value's width.
-/

namespace Nci.Cut.Assumptions

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

end Nci.Cut.Assumptions

/-!
## The training formalism v2, §4: the assumptions A1, the token-work bound and A2

* **A1, cut cost** (`CutCost`): no inference block reads fewer bits than a finite family `𝒲` of witness blocks of the
  run's circuit gives it. Every inference block `I` reads beyond `θ` at least, for every token it works on, the token's
  own state where its work on it begins, and, once, the rest of what that work reads and `I` doesn't compute:
  `|in(I) \ θ| ≥ u_I + Σ_T a_T(e_T(I))`. `e_T(I)` is where `I`'s work on token `T` begins, `a_T(x)` the least bits of
  `T`'s own state, the gates of its block it reads and doesn't compute, any witness block entering `T` at `x` reads
  (`Nci.Formalism.Annotation.stateAt`), and `u_I` the shared bits: what the witness blocks entering `I`'s tokens where
  `I`'s work on them begins all read beyond the tokens' own state, counted once and net of what `I` computes
  (`Nci.Formalism.Annotation.readsAt`, `Nci.Formalism.Annotation.blockShared`). The shared bits are the update's reads,
  `θ'` or the encoding it is decoded from (LoRA factors, a seed log, a code), and the shared context and the earlier
  tokens' keys and values, so two tokens of a block that read the same context pay for it once, and a block that
  computes a token's keys and values pays nothing for a later token's reads of them. A witness's reads are `in(W) \ θ`,
  counted as the policy counts a block's. So the values are tied to blocks of the circuit, not free: a checker produces
  `𝒲`, and A1 says no block beats the witnesses, refuted by an inference block that reads less. A better witness lowers
  a value: one reading a shorter encoding of `θ'` lowers `u`, one reading a code of a token's activations, gates of its
  block, lowers `a`. A block that computes a token from its inputs enters it there with no state of its own, and A1
  prices it as any other block: its shared bits, the token's data, the context and the earlier tokens' keys and values
  it reads among them.
* **The token-work bound** (`TokenWork`): a cap `cap` bounds the token work within A1's budget. At every set of tokens
  `S`, entry profile `f` and shared bits `ν` with `ν + Σ_{T ∈ S} a_T(f_T) ≤ k`, the token work from those entries,
  `Σ_{T ∈ S} W_T(f_T, ν)` (`Nci.Formalism.Annotation.entryWork`: the most gates of a set `X` of token gates in `T` that
  an inference block holds whose work on `T` begins at `f_T` and whose shared bits are at most `ν`), is at most `cap`.
  It is not a consequence of A1, which prices a block's reads at the witnesses' values and says nothing of how much
  token work those reads reach: it is assumed, refuted by inference blocks within those shared bits that, entering their
  tokens at those entries, hold more of `X`. The token kind's cap (`Nci.Formalism.Guarantees.TokenCap`) reads it, and
  its clean form gives it at `max_ν (k - ν) r(ν)` from a bound `r(ν)` on the work a bit of state
  (`Nci.Formalism.Guarantees.TokenCapClean`).
* **A2, efficiency floor** (`EfficiencyFloor`): no run of the methods considered whose update stays within `D` bits
  reaches `ℓ*` for less than `p · H(ℓ*)`. It is empirical. It is stated over a class of runs, not over every circuit: a
  circuit that outputs a run's `θ'` as constants reaches the same loss at about `|θ'|` gates, so over every circuit the
  floor fails at any `p · H(ℓ*)` above that (`Definitions.Nci.Formalism.Runs`).
-/

namespace Nci.Formalism.Assumptions

open Finset
open Verity (RegisteredCircuit)
open Nci.Formalism
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

end Nci.Formalism.Assumptions
