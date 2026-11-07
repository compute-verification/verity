module

public import Definitions.Claims.SamplingServed
public import Mathlib.Analysis.SpecialFunctions.Pow.Real

/-!
# Claims: Rankin's bound on the two-stage count

`CompiledTwo.Exfil` bounds the influence set by `Σ_{T ∈ admissibleTwo} 2^{b(T)}`, its patterns' form by the sum over the
admissible sets' patterns, and its served-output form by `Σ_T 2^{min(b(T), Σ_r min(b(T_r), |r|))}`. All three are exact
sums over the admissible sets, which no one evaluates directly.

Take a first stage that misses each set `T` of replay blocks with probability at most `∏_{ρ∈T} (1 − q_ρ)`, as
independent catches at rates `q_ρ ≤ 1` would: a Bernoulli first stage (`bernoulli:NUM/DEN`) at `q_ρ = NUM/DEN`, or the
deployed uniform `Q`-subset of `N` replay blocks with every input replay block, at `Q/N` and `1`. Then the acceptance
probability at a set `T`'s counts is `P̄_a ≤ ∏_ρ (1 − q_ρ(1 − e_ρ(T)))` (`Design.Two.passes`), each factor reading only
`T`'s proof blocks in `ρ` (`blocksIn`). Rankin's bound `1[δ < P̄_a] ≤ (P̄_a/δ)^θ`, at any `θ ≥ 0`, turns the sum over
the admissible sets into a sum over every set, which factors over the replay blocks:

* `rankinFactor`: `G_ρ(θ) = Σ_{U ⊆ ρ} 2^{b(U)} (1 − q_ρ(1 − e_ρ(U)))^θ`.
* `AdmissibleTwoRankin`: `Σ_{T admissible} 2^{b(T)} ≤ δ^{−θ} ∏_ρ G_ρ(θ)`.
* `rankinFactorPatterns`: the patterns' factor, a sum over replay block `ρ`'s patterns
  `c = (r ↦ lightOf |r| U_r)` of `2^{bits(c)}` times the largest `(1 − q_ρ(1 − e_ρ(U)))^θ` over the sets `U ⊆ ρ` of
  pattern `c`.
* `AdmissibleTwoRankinPatterns`: the patterns' sum is at most `δ^{−θ} ∏_ρ` of the patterns' factors, when each request
  reads the proof blocks of one replay block, so that a set's pattern is its replay blocks' patterns.
* `rankinFactorCapped`: `G_ρ(θ)` with each set's bits capped at `S_ρ`, `Σ_{U ⊆ ρ} 2^{min(b(U), S_ρ)} (…)^θ`.
* `AdmissibleTwoRankinServed`: the served-output cap's sum, `Σ_{T admissible} 2^{min(b(T), Σ_r min(b(T_r), |r|))}`, is
  at most `δ^{−θ} ∏_ρ` of the capped factors, when distinct requests read disjoint proof blocks and `S_ρ` bounds the
  served bits of the requests that read in `ρ`.

Statements only: definitions, no proofs.
-/

@[expose] public section

namespace Claims.Sampling

/-- **Replay block `ρ`'s proof blocks**: the union of its cells. -/
def blocksIn (Pt Pρ : Partition) (ρ : ℕ) : Finset ℕ :=
  (Finset.range Pt.strata.length).biUnion (cell Pt Pρ ρ)

/-- **Replay block `ρ` passes**, caught by the first stage at rate `q`, at a footprint `fp`: uncaught, or caught and
missed by the second stage, `1 − q(1 − e_ρ(fp))`. -/
noncomputable def Design.Two.passes (d : Design.Two) (Pt : Partition) (q : ℝ) (fp : ℕ → ℕ → ℕ) (ρ : ℕ) : ℝ :=
  1 - q * (1 - d.esc Pt fp ρ)

/-- **Rankin's factor of replay block `ρ`** at rate `q` and `θ`: `G_ρ(θ) = Σ_{U ⊆ ρ} 2^{b(U)} (1 − q(1 − e_ρ(U)))^θ`,
`U` over the sets of `ρ`'s proof blocks and `e_ρ(U)` the escape at `U`'s counts. -/
noncomputable def rankinFactor (s : AuditStatement) (Pt : Partition) (d : Design.Two) (q θ : ℝ) (ρ : ℕ) : ℝ :=
  ∑ U ∈ (blocksIn Pt d.part ρ).powerset, (2 : ℝ) ^ bitsOf s Pt U * d.passes Pt q (counts Pt d.part U) ρ ^ θ

/-- **Rankin's patterns' factor of replay block `ρ`** at rate `q` and `θ`, under a split `req` of the served outputs
into requests: over the patterns `c = (r ↦ lightOf |r| U_r)` of the sets `U` of `ρ`'s proof blocks,
`Σ_c 2^{Σ_r bits(c r)} · max_{U ⊆ ρ of pattern c} (1 − q(1 − e_ρ(U)))^θ`. -/
noncomputable def rankinFactorPatterns (s : AuditStatement) (Pt : Partition) (d : Design.Two) {R : Type} [Fintype R]
    [DecidableEq R] (req : Fin s.prog.outs.length → R) (q θ : ℝ) (ρ : ℕ) : ℝ :=
  ∑ c ∈ (blocksIn Pt d.part ρ).powerset.image
      (fun U r => lightOf s Pt (servedBits req r) (blocksTo s.prog Pt req U r)),
    (2 : ℝ) ^ (∑ r, lightBits s Pt (servedBits req r) (c r)) *
      ⨆ U : ((blocksIn Pt d.part ρ).powerset.filter
          fun U => (fun r => lightOf s Pt (servedBits req r) (blocksTo s.prog Pt req U r)) = c),
        d.passes Pt q (counts Pt d.part U.1) ρ ^ θ

/-- **Rankin's bound on the two-stage count**: on a statement in scope whose first stage misses each set `T` of replay
blocks with probability at most `∏_{ρ∈T} (1 − q_ρ)`, at rates `q_ρ ≤ 1`, at every `δ > 0` and `θ ≥ 0`,
`Σ_{T admissible} 2^{b(T)} ≤ δ^{−θ} ∏_ρ G_ρ(θ)` over the replay blocks `ρ`. -/
def AdmissibleTwoRankin : Prop :=
  ∀ (s : AuditStatement) (Pt : Partition) (d : Design.Two) (q : ℕ → ℝ) (δ θ : ℝ), InScopeTwo Pt d s →
    (∀ T ⊆ Finset.range d.part.blocks.length, d.replay.Pa d.part T ≤ ∏ ρ ∈ T, (1 - q ρ)) → (∀ ρ, q ρ ≤ 1) →
    0 < δ → 0 ≤ θ →
      ((∑ T ∈ admissibleTwo Pt d δ, 2 ^ bitsOf s Pt T : ℕ) : ℝ) ≤
        δ ^ (-θ) * ∏ ρ ∈ Finset.range d.part.blocks.length, rankinFactor s Pt d (q ρ) θ ρ

/-- **Rankin's bound on the patterns' count**: on a statement in scope whose first stage misses each set `T` of replay
blocks with probability at most `∏_{ρ∈T} (1 − q_ρ)`, at rates `q_ρ ≤ 1`, at every `δ > 0`, `θ ≥ 0` and split `req` of
the served outputs into requests each reading the proof blocks of one replay block, the admissible sets' patterns' sum
`Σ_p 2^{Σ_r bits(p r)}` is at most `δ^{−θ}` times the product of the replay blocks' patterns' factors. -/
def AdmissibleTwoRankinPatterns : Prop :=
  ∀ (s : AuditStatement) (Pt : Partition) (d : Design.Two) (q : ℕ → ℝ) (δ θ : ℝ) (R : Type) [Fintype R]
    [DecidableEq R] (req : Fin s.prog.outs.length → R), InScopeTwo Pt d s →
    (∀ T ⊆ Finset.range d.part.blocks.length, d.replay.Pa d.part T ≤ ∏ ρ ∈ T, (1 - q ρ)) → (∀ ρ, q ρ ≤ 1) →
    0 < δ → 0 ≤ θ →
    (∀ r, ∃ ρ, ∀ v ∈ blocksTo s.prog Pt req (Finset.range Pt.blocks.length) r, replayOf Pt d.part v = some ρ) →
      ((∑ p ∈ (admissibleTwo Pt d δ).image
            (fun T r => lightOf s Pt (servedBits req r) (blocksTo s.prog Pt req T r)),
          2 ^ ∑ r, lightBits s Pt (servedBits req r) (p r) : ℕ) : ℝ) ≤
        δ ^ (-θ) * ∏ ρ ∈ Finset.range d.part.blocks.length, rankinFactorPatterns s Pt d req (q ρ) θ ρ

/-- **Rankin's capped factor of replay block `ρ`** at cap `S`, rate `q` and `θ`:
`Σ_{U ⊆ ρ} 2^{min(b(U), S)} (1 − q(1 − e_ρ(U)))^θ`, each set's bits capped at `S`. -/
noncomputable def rankinFactorCapped (s : AuditStatement) (Pt : Partition) (d : Design.Two) (S : ℕ) (q θ : ℝ)
    (ρ : ℕ) : ℝ :=
  ∑ U ∈ (blocksIn Pt d.part ρ).powerset, (2 : ℝ) ^ min (bitsOf s Pt U) S * d.passes Pt q (counts Pt d.part U) ρ ^ θ

/-- **Rankin's bound on the served-output cap's sum**: on a statement in scope whose first stage misses each set `T` of
replay blocks with probability at most `∏_{ρ∈T} (1 − q_ρ)`, at rates `q_ρ ≤ 1`, at every `δ > 0`, `θ ≥ 0`, split `req`
of the served outputs into requests whose proof blocks are pairwise disjoint, and caps `S_ρ` at least the served bits
`Σ |r|` of the requests that read in replay block `ρ`,
`Σ_{T admissible} 2^{min(b(T), Σ_{r : T_r ≠ ∅} min(b(T_r), |r|))}` is at most `δ^{−θ}` times the product of the replay
blocks' capped factors. -/
def AdmissibleTwoRankinServed : Prop :=
  ∀ (s : AuditStatement) (Pt : Partition) (d : Design.Two) (q : ℕ → ℝ) (δ θ : ℝ) (R : Type) [Fintype R]
    [DecidableEq R] (req : Fin s.prog.outs.length → R) (S : ℕ → ℕ), InScopeTwo Pt d s →
    (∀ T ⊆ Finset.range d.part.blocks.length, d.replay.Pa d.part T ≤ ∏ ρ ∈ T, (1 - q ρ)) → (∀ ρ, q ρ ≤ 1) →
    0 < δ → 0 ≤ θ →
    (∀ r r', r ≠ r' → Disjoint (blocksTo s.prog Pt req (Finset.range Pt.blocks.length) r)
      (blocksTo s.prog Pt req (Finset.range Pt.blocks.length) r')) →
    (∀ ρ, ∑ r ∈ Finset.univ.filter (fun r => (blocksTo s.prog Pt req (blocksIn Pt d.part ρ) r).Nonempty),
      servedBits req r ≤ S ρ) →
      ((∑ T ∈ admissibleTwo Pt d δ, 2 ^ min (bitsOf s Pt T)
          (∑ r ∈ Finset.univ.filter (fun r => (blocksTo s.prog Pt req T r).Nonempty),
            min (bitsOf s Pt (blocksTo s.prog Pt req T r)) (servedBits req r)) : ℕ) : ℝ) ≤
        δ ^ (-θ) * ∏ ρ ∈ Finset.range d.part.blocks.length, rankinFactorCapped s Pt d (S ρ) (q ρ) θ ρ

end Claims.Sampling
