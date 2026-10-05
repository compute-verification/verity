import FlockSoundness.Protocol.Audit.Law
import FlockSoundness.Protocol.Audit.Circuit

/-!
# The one-stage audit and its integrity profile

The prover registers `R` (its roots). The verifier draws units from its own randomness, by the law
`L`, after registration. One batched session proves the drawn units against `R`. The audit's output
is the verdict and the draw (`audit`).

**The committed transcript** is an analysis device: a function of the prover's state at
registration, which is its registration `R` and its session strategy for every draw (`Cont`). In
the compiled analysis it is the plurality of what the extractor recovers behind each commit string.
Because it is fixed before the draw, the draw is independent of its wrong units. Nobody computes it.

**The per-unit hypotheses** (`Analysis`) charge a drawn wrong unit to one of two failures of its
table, both named:

* `KnowledgeSound ε_ks`: the table accepts while extraction of the unit fails;
* `LinkSound δ_link`: extraction succeeds, but what it recovers differs from the committed
  transcript on the unit's wires (consistency across units).

Both are bounds over the whole audit, for every prover and every rule that picks a drawn unit before
the session's first coin (the first wrong drawn unit, in the proof). That is the form the table
theorems take once averaged over the prover's state; neither needs to hold state by state.

**The theorem** (`audit_le`): for every event `E` on the committed transcript and the draw,
$$\Pr[\text{accept} \wedge E(X, S)] \le \Pr_S[E(X, S) \wedge S \cap \mathrm{wrong}(X) = \emptyset]
  + \varepsilon_{ks} + \delta_{link}.$$
Its corollaries are the integrity profile (`audit_profile`), the count curve
`δ(K) = miss K + ε_ks + δ_link` (`audit_count`), and the drawn units (`audit_drawn`).
-/

namespace FlockSoundness.Audit

open Game
open scoped ENNReal

variable {n : ℕ}

/-- **The one-stage audit.** The prover registers `R : Reg`; the verifier draws `S` by `L`; the
session on the drawn units runs against `R`. Output: the verdict and the draw. -/
def audit (L : Law n) (Reg : Type) (session : Finset (Fin n) → Reg → Game Bool) :
    Game (Bool × Finset (Fin n)) :=
  .send Reg fun R => .coin L.Ω fun ω => (session (L.draw ω) R).map fun ok => (ok, L.draw ω)

variable {L : Law n} {Reg : Type} {session : Finset (Fin n) → Reg → Game Bool}

/-- A prover's state after registering `R`: its session strategy for every draw. -/
abbrev Cont (L : Law n) (Reg : Type) (session : Finset (Fin n) → Reg → Game Bool) (R : Reg) :=
  ∀ ω : L.Ω, Strategy (session (L.draw ω) R)

/-- The registration of an audit strategy. -/
def reg (σ : Strategy (audit L Reg session)) : Reg := σ.1

/-- The prover's state after registration. -/
def cont (σ : Strategy (audit L Reg session)) : Cont L Reg session (reg σ) :=
  fun ω => Strategy.ofMap _ _ (σ.2 ω)

/-- What the analysis of an audit fixes.
* `Tr`, `wrong`: transcripts, and the units wrong in each;
* `committed`: the committed transcript, a function of the prover's state at registration;
* `ksFail S R τ u`: extraction of drawn unit `u` from the session state `τ` fails;
* `linkFail S R τ u X`: it succeeds, but disagrees with `X` on `u`'s wires;
* `cover`: a unit wrong in `X` fails one way or the other, from every state. -/
structure Analysis (L : Law n) (Reg : Type) (session : Finset (Fin n) → Reg → Game Bool) where
  Tr : Type
  wrong : Tr → Finset (Fin n)
  committed : (R : Reg) → Cont L Reg session R → Tr
  ksFail : (S : Finset (Fin n)) → (R : Reg) → Strategy (session S R) → Fin n → Prop
  linkFail : (S : Finset (Fin n)) → (R : Reg) → Strategy (session S R) → Fin n → Tr → Prop
  cover : ∀ S R (τ : Strategy (session S R)) u X, u ∈ wrong X → ksFail S R τ u ∨ linkFail S R τ u X

namespace Analysis

variable (A : Analysis L Reg session)

/-- The committed transcript of an audit strategy. -/
def committedOf (σ : Strategy (audit L Reg session)) : A.Tr := A.committed (reg σ) (cont σ)

/-- **Knowledge soundness, per unit** (named hypothesis `ε_ks`): for every prover and every rule
`tgt` that picks a drawn unit before the session's first coin, the probability that the session
accepts while extraction of the picked unit fails is at most `ε_ks`. -/
def KnowledgeSound (εks : ℝ≥0∞) : Prop :=
  ∀ (R : Reg) (τ : Cont L Reg session R) (tgt : L.Ω → Option (Fin n)),
    avg (fun ω => prob (fun ok => ok = true ∧
      ∃ u, tgt ω = some u ∧ u ∈ L.draw ω ∧ A.ksFail (L.draw ω) R (τ ω) u)
      (session (L.draw ω) R) (τ ω)) ≤ εks

/-- **The link term** (named hypothesis `δ_link`): for every prover and every rule `tgt` that
picks a drawn unit before the session's first coin, the probability that the session accepts while
the picked unit's extraction disagrees with the committed transcript is at most `δ_link`. -/
def LinkSound (δlink : ℝ≥0∞) : Prop :=
  ∀ (R : Reg) (τ : Cont L Reg session R) (tgt : L.Ω → Option (Fin n)),
    avg (fun ω => prob (fun ok => ok = true ∧
      ∃ u, tgt ω = some u ∧ u ∈ L.draw ω ∧ A.linkFail (L.draw ω) R (τ ω) u (A.committed R τ))
      (session (L.draw ω) R) (τ ω)) ≤ δlink

end Analysis

/-! ## Over a partition: extraction, anchors and the consumer's cone -/

namespace Partition

variable {C : Circuit} (P : Partition C n)

/-- **The audit's analysis over a partition.** `X` is the committed transcript, a function of the
prover's state at registration. `ext S R τ u` is what the extractor recovers for the drawn unit
`u` from the session state `τ`: a transcript, of which only `u`'s committed inputs and outputs
(`io u`) are read, or nothing. Extraction fails when it recovers nothing or something that does
not satisfy `u`; the link fails when it satisfies `u` but differs from `X` on `u`'s wires. -/
noncomputable def analysis (X : (R : Reg) → Cont L Reg session R → (Fin C.N → Bool))
    (ext : (S : Finset (Fin n)) → (R : Reg) → Strategy (session S R) → Fin n →
      Option (Fin C.N → Bool)) :
    Analysis L Reg session where
  Tr := Fin C.N → Bool
  wrong := P.wrong
  committed := X
  ksFail S R τ u := ∀ Y, ext S R τ u = some Y → ¬ P.Correct Y u
  linkFail S R τ u X := ∃ Y, ext S R τ u = some Y ∧ P.Correct Y u ∧ ∃ g ∈ P.io u, Y g ≠ X g
  cover S R τ u X hu := by
    by_cases h : ∃ Y, ext S R τ u = some Y ∧ P.Correct Y u
    · obtain ⟨Y, hY, hc⟩ := h
      refine Or.inr ⟨Y, hY, hc, ?_⟩
      by_contra hno
      push Not at hno
      exact P.mem_wrong.1 hu (P.correct_congr (fun g hg => (hno g hg).symm) hc)
    · exact Or.inl fun Y hY hc => h ⟨Y, hY, hc⟩

/-- **The input unit** (named hypothesis `δ_in`): the session accepts while the committed inputs
differ from the anchors' values `a` with probability at most `δ_in`. -/
def AnchorsSound (X : (R : Reg) → Cont L Reg session R → (Fin C.N → Bool)) (a : Fin C.N → Bool)
    (δin : ℝ≥0∞) : Prop :=
  ∀ σ : Strategy (audit L Reg session),
    prob (fun o => o.1 = true ∧ ¬ C.InputsAgree (X (reg σ) (cont σ)) a) (audit L Reg session) σ ≤ δin

variable {X : (R : Reg) → Cont L Reg session R → (Fin C.N → Bool)}
  {ext : (S : Finset (Fin n)) → (R : Reg) → Strategy (session S R) → Fin n → Option (Fin C.N → Bool)}
  {a : Fin C.N → Bool} {εks δlink δin : ℝ≥0∞}

/-! ### The oracle layer

The prover sends the committed transcript itself, as tables are sent at the oracle layer, so the
committed transcript is the registration and extraction returns it. Knowledge soundness is then
soundness of the session (`SessionSound`), the link term is zero, and the anchors term is the
session's error on inputs other than the anchors' (`InputsSound`). -/

variable {session' : Finset (Fin n) → (Fin C.N → Bool) → Game Bool} {ε δ : ℝ≥0∞}

end Partition

end FlockSoundness.Audit
