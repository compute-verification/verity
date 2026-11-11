module

public import Definitions.Assumptions

/-!
# The training formalism v2: its theorems

The theorems of the training formalism v2 (`Definitions.Nci.Formalism`), §5, proved in
`Nci.Formalism.SecurityProofs`. Its two lemmas, reconstruction (§1) and the three kinds of a block's inputs (§2), are
`Nci.Formalism.reconstruction` and `Nci.Formalism.reads_split` in `Security/Proofs/Nci/Formalism/Reconstruction.lean`.

A lower bound on `ρ` is stated, as each is proved, for every partition `𝓘` of `C` into inference blocks:
`X ≤ |𝓘| c / |C|` for each is `X ≤ ρ` (`Nci.Formalism.le_slowdown_iff`). The slowdown is one counting bound over kinds
of work, its factors explicit, and it reads the upper bounds A1 gives, never best-known values:

* **The counting lemma over kinds** (`Counting`, no assumption): for kinds `K` of work `B_K ⊆ C`, each with a cap `cap_K`
  on the gates of `B_K` an inference block holds (`Caps`), and kinds `J` whose work is amortized over the partition,
  each with a bound `Λ_J` for the whole partition (`PartitionBound`),
  `ρ ≥ max(max_K (|B_K| / |C|) / (cap_K / c), max_J Λ_J)`. A cap a block is the special case: it gives the kind's bound
  for the whole partition (`Nci.Formalism.partitionBound_kindBound`). The lemma reads nothing of the circuit beyond the
  caps and the partition-level bounds. Training's slowdown code's kinds are the unreachable token work, gradient
  reductions, update application and decoding; the token kind's cap is a theorem here from a named assumption, and the
  others' caps are premises. The bound is a maximum: kinds whose work shares blocks are not added, which would need a
  joint cap, `Σ_K |I ∩ B_K| / cap_K ≤ 1` for every block;
* **the token kind's cap** (`TokenCap`): under A1 at a family `𝒲` of witness blocks and the token-work bound at
  `cap`, the named assumption `training-token-work` (`TokenWork`: `cap` bounds the token work within A1's budget), an
  inference block holds at most `cap` gates of a set `X` of token gates: a block with shared bits `u_I` whose work on
  its tokens begins at the entries `e_T` holds at most the token work reachable from them, and A1 gives
  `u_I + Σ_T a_T(e_T) ≤ k`, at the values `𝒲` sets. The cap is the assumption's, not A1's: A1 prices a block's reads
  and says nothing of how much token work they reach. Its clean form (`TokenCapClean`): with at most `r(ν)` gates of
  token work a bit of state, the token-work bound holds at `max_ν (k - ν) r(ν)`, the bound `(k - u) max (W / a)`;
* **Slowdown** (`Slowdown`), the one bound: under A1 and the token-work bound,
  `ρ ≥ max((|X| / |C|) / (cap_X / c), max_K (|B_K| / |C|) / (cap_K / c), max_J Λ_J)`, the counting lemma over the kinds
  with the token kind's cap from the token-work bound;
* **Fit**: a partition whose every block holds at least `c - d` gates and reads within `k` gives `ρ ≤ c / (c - d)`.
  Running `θ'` (`FitServing`), session blocks with `Δ + s τ ≤ k`; honest serving, `Δ = 0` and no decoder, has exactly
  `ρ = 1`, one block per session (`FitHonestServing`). Training (`FitTraining`), compute blocks with `Δ + s τ ≤ k`,
  reduction blocks with `2 Δ_grad ≤ k` and update application blocks with `Δ_t + Δ_grad ≤ k`. Fit counts the blocks'
  gates in `C`: `c / (c - d)` is `ρ` against a circuit whose blocks each leave `d` of their `c` gates to a decoder;
* **Security** (`Security`), under A1, the token-work bound and A2: a run of the methods considered that reaches `ℓ*`
  through a partition into inference blocks that fit in `N` periods of `n` blocks has `N n c ≥ ρ̲ p H(ℓ*)`, `ρ̲`
  Slowdown's bound for it and `p` A2's floor: reaching `ℓ*` covertly costs at least `ρ̲ p H(ℓ*)`, with at most `n c`
  available per period.

The counting lemma, the token kind's cap, Slowdown, Fit's bounds and Security each have an honest and a nontrivial
lemma (`Definitions.Nci.Guarantees.Formalism.Witnesses`).
-/

@[expose] public section

namespace Nci.Formalism.Guarantees

open Finset
open Verity (RegisteredCircuit Partition)
open Nci.Formalism Nci.Formalism.Assumptions
open scoped NNReal ENNReal

/-- **The counting lemma over kinds**: for kinds `K` of work `B_K ⊆ C`, each with a cap `cap_K` on the gates of `B_K`
an inference block holds, and kinds `J` that each give a bound `Λ_J` for the whole partition, every partition `𝓘` into
inference blocks has `|𝓘| c / |C| ≥ max(max_K (|B_K| / |C|) / (cap_K / c), max_J Λ_J)`. -/
def Counting : Prop :=
  ∀ (d : RegisteredCircuit) (c k : ℕ) (κ : Type) (B : κ → Finset (Fin d.D.N)) (cap : κ → ℝ≥0) (ι : Type) (Λ : ι → ℝ≥0∞)
    (m : ℕ) (P : Partition d.D m),
    Caps d c k B cap → (∀ J, PartitionBound d c k (Λ J)) → P.IsCK d c k →
    max (⨆ K, kindBound d c (B K) (cap K)) (⨆ J, Λ J) ≤ partitionSlowdown d m c

/-- **The token kind's cap**: under A1 at the witness blocks `𝒲` and the token-work bound at `cap` (`TokenWork`, the
named assumption `training-token-work`, which A1 doesn't give), every inference block holds at most `cap` gates of
`X`. -/
def TokenCap : Prop :=
  ∀ (d : RegisteredCircuit) (L : Type) [Fintype L] (A : Annotation d L) (c k : ℕ) (E : Type)
    (e : Finset (Fin d.D.N) → L → E) (𝒲 : Finset (Finset (Fin d.D.N))) (X : Finset (Fin d.D.N)) (cap : ℝ≥0),
    CutCost A c k e 𝒲 → TokenWork A c k e 𝒲 X cap → ∀ I, InferenceBlock d c k I → ((I ∩ X).card : ℝ≥0) ≤ cap

/-- **The token cap's clean form**: with at most `r(ν)` gates of token work a bit of state, the token-work bound holds
at `max_ν (k - ν) r(ν)`, the bound `(k - u) max (W / a)`. -/
def TokenCapClean : Prop :=
  ∀ (d : RegisteredCircuit) (L : Type) [Fintype L] (A : Annotation d L) (c k : ℕ) (E : Type)
    (e : Finset (Fin d.D.N) → L → E) (𝒲 : Finset (Finset (Fin d.D.N))) (X : Finset (Fin d.D.N)) (r : ℕ → ℝ≥0)
    (M : ℝ≥0),
    A.WorkPerBit c k e 𝒲 X r → PeakAtMost k r M → TokenWork A c k e 𝒲 X M

/-- **Slowdown**, the one bound: under A1 and the token-work bound at `cap_X` (`TokenWork`, the named assumption
`training-token-work`), for kinds `K` of work with caps `cap_K`, kinds `J` with partition-level bounds `Λ_J` and a set
`X` of token gates, every partition `𝓘` into inference blocks has
`|𝓘| c / |C| ≥ max((|X| / |C|) / (cap_X / c), max_K (|B_K| / |C|) / (cap_K / c), max_J Λ_J)`. -/
def Slowdown : Prop :=
  ∀ (d : RegisteredCircuit) (L : Type) [Fintype L] (A : Annotation d L) (c k : ℕ) (E : Type)
    (e : Finset (Fin d.D.N) → L → E) (𝒲 : Finset (Finset (Fin d.D.N))) (X : Finset (Fin d.D.N)) (capX : ℝ≥0)
    (κ : Type) (B : κ → Finset (Fin d.D.N)) (cap : κ → ℝ≥0) (ι : Type) (Λ : ι → ℝ≥0∞) (m : ℕ) (P : Partition d.D m),
    A.Valid → CutCost A c k e 𝒲 → TokenWork A c k e 𝒲 X capX → Caps d c k B cap →
    (∀ J, PartitionBound d c k (Λ J)) → P.IsCK d c k →
    max (kindBound d c X capX) (max (⨆ K, kindBound d c (B K) (cap K)) (⨆ J, Λ J)) ≤ partitionSlowdown d m c

/-- **Fit, running `θ'`**: session blocks of at least `c - d` gates that read within `k`, `Δ + s τ ≤ k`, give
`ρ ≤ c / (c - d)`. -/
def FitServing : Prop :=
  ∀ (d : RegisteredCircuit) (B : ℕ) (P : Partition d.D B) (c k dg Δ s τ : ℕ), d.D.Boolean → SessionBlocks d P c dg Δ s τ →
    ServingFits Δ s τ k → slowdown d c k ≤ (c : ℝ≥0∞) / ((c - dg : ℕ) : ℝ≥0∞)

/-- **Fit, honest serving**: with `Δ = 0` and no decoder, session blocks that read `s τ ≤ k` give exactly `ρ = 1`, one
block per session. -/
def FitHonestServing : Prop :=
  ∀ (d : RegisteredCircuit) (B : ℕ) (P : Partition d.D B) (c k s τ : ℕ), d.D.Boolean → SessionBlocks d P c 0 0 s τ →
    ServingFits 0 s τ k → 0 < B → 0 < c → slowdown d c k = 1

/-- **Fit, training**: compute, reduction and update application blocks of at least `c - d` gates that each read
within `k` give `ρ ≤ c / (c - d)`. -/
def FitTraining : Prop :=
  ∀ (d : RegisteredCircuit) (B : ℕ) (P : Partition d.D B) (c k dg Δ Δt Δgrad s τ : ℕ), d.D.Boolean →
    TrainingBlocks d P c dg Δ Δt Δgrad s τ → TrainingFits Δ Δt Δgrad s τ k →
    slowdown d c k ≤ (c : ℝ≥0∞) / ((c - dg : ℕ) : ℝ≥0∞)

/-- **Security**: under A1, the token-work bound (`training-token-work`) and A2, a run of `methods` whose update stays
within `D` bits and that reaches `ℓ*` through a partition into inference blocks that fit in `N` periods of `n` blocks
has `N n c ≥ ρ̲ p H(ℓ*)`, `ρ̲` Slowdown's bound. -/
def Security : Prop :=
  ∀ (methods : Set Run) (dec : Verity.Decoder (List Bool)) (ℓ : List Bool → ℝ) (ℓstar : ℝ) (D : ℕ∞) (p : ℝ) (r : Run)
    (c k n N : ℕ) (E : Type) (e : Finset (Fin r.d.D.N) → Fin r.numTokens → E) (𝒲 : Finset (Finset (Fin r.d.D.N)))
    (X : Finset (Fin r.d.D.N)) (capX : ℝ≥0) (κ : Type) (B : κ → Finset (Fin r.d.D.N)) (cap : κ → ℝ≥0) (ι : Type)
    (Λ : ι → ℝ≥0∞) (m : ℕ) (P : Partition r.d.D m),
    EfficiencyFloor methods dec c ℓ ℓstar D p → r.Among methods → r.A.Valid → r.Within dec c D → r.Reaches ℓ ℓstar →
    CutCost r.A c k e 𝒲 → TokenWork r.A c k e 𝒲 X capX → Caps r.d c k B cap →
    (∀ J, PartitionBound r.d c k (Λ J)) → P.IsCK r.d c k → InPeriods m N n →
    max (kindBound r.d c X capX) (max (⨆ K, kindBound r.d c (B K) (cap K)) (⨆ J, Λ J)) * ENNReal.ofReal p *
      honestCost methods ℓ ℓstar ≤ ((N * n * c : ℕ) : ℝ≥0∞)

end Nci.Formalism.Guarantees
