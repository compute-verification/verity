module

public import Definitions.Nci.Formalism

/-!
# The training formalism v2, §4: the assumptions A1, the token-work bound and A2

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

@[expose] public section

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
