module

public import Definitions.Core.Game
public import Definitions.Spot.Sampling.SamplingDesign

/-! The sampling layer's game and its two analyses, with the per-unit hypotheses their theorems take. -/

@[expose] public section

namespace Spot.Sampling

open Verity Verity.Game
open scoped ENNReal

variable {U : Type}

/-- **The one-stage audit.** The prover registers `R : Reg`; the verifier samples `S` by `L`; the
session on the sampled proof blocks runs against `R`. Output: the verdict and the sample. -/
def audit (L : Verity.SamplingDesign U) (Reg : Type) (session : Finset U → Reg → Game Bool) :
    Game (Bool × Finset U) :=
  .send Reg fun R => .coin L.Ω fun ω => (session (L.draw ω) R).map fun ok => (ok, L.draw ω)

variable {L : Verity.SamplingDesign U} {Reg : Type} {session : Finset U → Reg → Game Bool}

/-- A prover's state after registering `R`: its session strategy for every sample. -/
abbrev Cont (L : Verity.SamplingDesign U) (Reg : Type) (session : Finset U → Reg → Game Bool) (R : Reg) :=
  ∀ ω : L.Ω, Strategy (session (L.draw ω) R)

/-- The registration of an audit strategy. -/
def reg (σ : Strategy (audit L Reg session)) : Reg := σ.1

/-- The prover's state after registration. -/
def cont (σ : Strategy (audit L Reg session)) : Cont L Reg session (reg σ) :=
  fun ω => Strategy.ofMap _ _ (σ.2 ω)

/-- What the analysis of an audit fixes.
* `Tr`, `incorrect`: transcripts, and the units incorrect in each;
* `committed`: the committed transcript, a function of the prover's state at registration;
* `ksFail S R τ u`: extraction of sampled proof block `u` from the session state `τ` fails;
* `linkFail S R τ u X`: it succeeds, but disagrees with `X` on `u`'s wires;
* `cover`: a unit incorrect in `X` fails one way or the other, from every state. -/
structure Analysis (L : Verity.SamplingDesign U) (Reg : Type) (session : Finset U → Reg → Game Bool) where
  Tr : Type
  incorrect : Tr → Finset U
  committed : (R : Reg) → Cont L Reg session R → Tr
  ksFail : (S : Finset U) → (R : Reg) → Strategy (session S R) → U → Prop
  linkFail : (S : Finset U) → (R : Reg) → Strategy (session S R) → U → Tr → Prop
  cover : ∀ S R (τ : Strategy (session S R)) u X, u ∈ incorrect X → ksFail S R τ u ∨ linkFail S R τ u X

namespace Analysis

variable (A : Analysis L Reg session)

/-- The committed transcript of an audit strategy. -/
def committedOf (σ : Strategy (audit L Reg session)) : A.Tr := A.committed (reg σ) (cont σ)

/-- **Knowledge soundness, per unit** (named hypothesis `ε_ks`): for every prover and every rule
`tgt` that picks a sampled proof block before the session's first coin, the probability that the session
accepts while extraction of the picked unit fails is at most `ε_ks`. -/
def KnowledgeSound (εks : ℝ≥0∞) : Prop :=
  ∀ (R : Reg) (τ : Cont L Reg session R) (tgt : L.Ω → Option U),
    avg (fun ω => prob (fun ok => ok = true ∧
      ∃ u, tgt ω = some u ∧ u ∈ L.draw ω ∧ A.ksFail (L.draw ω) R (τ ω) u)
      (session (L.draw ω) R) (τ ω)) ≤ εks

/-- **The binding error** (named hypothesis `ε_bind`): for every prover and every rule `tgt` that
picks a sampled proof block before the session's first coin, the probability that the session accepts while
the picked unit's extraction disagrees with the committed transcript is at most `ε_bind`. -/
def BindSound (εbind : ℝ≥0∞) : Prop :=
  ∀ (R : Reg) (τ : Cont L Reg session R) (tgt : L.Ω → Option U),
    avg (fun ω => prob (fun ok => ok = true ∧
      ∃ u, tgt ω = some u ∧ u ∈ L.draw ω ∧ A.linkFail (L.draw ω) R (τ ω) u (A.committed R τ))
      (session (L.draw ω) R) (τ ω)) ≤ εbind

end Analysis

variable [DecidableEq U]

/-- A per-unit quantity at the unit a rule picks, and zero when it picks none or an unsampled one. -/
noncomputable def atTgt (S : Finset U) (t : Option U) (f : U → ℝ≥0∞) : ℝ≥0∞ :=
  t.elim 0 fun u => if u ∈ S then f u else 0

/-- What the analysis of an audit fixes when extraction is randomized: the committed transcript,
and for each sampled proof block the joint probabilities of the two ways its table can accept while it is
incorrect. -/
structure ExtractionAnalysis (L : Verity.SamplingDesign U) (Reg : Type) (session : Finset U → Reg → Game Bool)
    where
  Tr : Type
  incorrect : Tr → Finset U
  committed : (R : Reg) → Cont L Reg session R → Tr
  /-- `Pr[the session accepts ∧ extraction of u fails]` -/
  ks : (S : Finset U) → (R : Reg) → Strategy (session S R) → U → ℝ≥0∞
  /-- `Pr[it accepts ∧ extraction of u succeeds but differs from X on u's wires]` -/
  link : (S : Finset U) → (R : Reg) → Strategy (session S R) → U → Tr → ℝ≥0∞
  cover : ∀ S R (τ : Strategy (session S R)) u X, u ∈ S → u ∈ incorrect X →
    prob (· = true) (session S R) τ ≤ ks S R τ u + link S R τ u X

namespace ExtractionAnalysis

variable (A : ExtractionAnalysis L Reg session)

/-- The committed transcript of an audit strategy. -/
def committedOf (σ : Strategy (audit L Reg session)) : A.Tr := A.committed (reg σ) (cont σ)

/-- **Knowledge soundness, per unit** (named hypothesis `ε_ks`, which may depend on the prover's
state): for every prover and every rule that picks a sampled proof block before the session's first coin,
the average over the sample of `Pr[accept ∧ extraction of the picked unit fails]`. -/
def KnowledgeSound (εks : (R : Reg) → Cont L Reg session R → ℝ≥0∞) : Prop :=
  ∀ (R : Reg) (τ : Cont L Reg session R) (tgt : L.Ω → Option U),
    avg (fun ω => atTgt (L.draw ω) (tgt ω) (A.ks (L.draw ω) R (τ ω))) ≤ εks R τ

/-- **The binding error** (named hypothesis `ε_bind`, which may depend on the prover's state):
likewise, with `Pr[accept ∧ extraction succeeds but disagrees with the committed transcript]`. -/
def BindSound (εbind : (R : Reg) → Cont L Reg session R → ℝ≥0∞) : Prop :=
  ∀ (R : Reg) (τ : Cont L Reg session R) (tgt : L.Ω → Option U),
    avg (fun ω => atTgt (L.draw ω) (tgt ω) fun u => A.link (L.draw ω) R (τ ω) u (A.committed R τ)) ≤
      εbind R τ

end ExtractionAnalysis

end Spot.Sampling
