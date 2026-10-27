import Definitions.Accounting.Guarantees.ServingRoute

/-!
# The serving route: honest and nontrivial lemmas

Each guarantee of `Accounting.Guarantees.ServingRoute` has two statements here. `<Guarantee>_honest`: its premises hold
together at an instance the proof constructs, one at which something is counted. `<Guarantee>_nontrivial`: with one of
its premises dropped, the guarantee is false.

* `ServingPerBlockCap_honest`: one constant gate computing one operation, a product of one activation, with the decoder
  that outputs its program: each block's activations are one bit, described by one bit (`β = 1`, `Λ = 0`).
* `ServingPerBlockCap_nontrivial`: without `ActivationRate` (a decoder that prints the activations on the empty
  program), or without `EvaluatesBlocks` (a decoder that outputs nothing, under which every activation rate holds), the
  cap `e (k_S + ℓ + a_blk + Λ)/β` fails at that gate.
* `ServingEndToEnd_honest`: one server, one window of one gate, one operation of weight 1, on a verifier that accepts
  every run, with every error `0`; the serving class holds and the threshold `windowsMin N` is positive.
* `ServingEndToEnd_nontrivial`: at a verifier that accepts everything, dropping the joint guarantee, the serving bridge
  or coverage makes the statement false.

The proofs are in `Security/Proofs/Accounting/ServingRoute/Witnesses.lean`.
-/

namespace Accounting.Guarantees.ServingRoute

open Verity Accounting.ServingBlocks Accounting.ServingRoute
open scoped ENNReal

/-- **The per-block cap's premises hold together**, at a declaration with one counted operation, a product occurrence
of fan-out `e`, and decoders that evaluate every registration's blocks and meet the activation rate `β > 0`. -/
def ServingPerBlockCap_honest : Prop :=
  ∃ (d : Declaration) (J : Counted d.D) (M : Nci.Ldl.Products d.D Unit Unit) (e : ℕ)
    (D : (n : ℕ) → Nci.Ldl.Registration d n → Nci.Ldl.Decoder) (t aBlk ℓ : ℕ)
    (y : Finset (Fin d.D.N) → d.D.Transcript → List Bool) (β Λ : ℝ) (inp : d.D.Transcript) (cS kS : ℕ)
    (q : ℕ → ℕ → ℝ),
    J.total = 1 ∧
    (∀ v ∈ J.V, v ∈ J.body v) ∧ (∀ v ∈ J.V, J.ω v ≤ 1) ∧ J.V ⊆ M.J.image M.rep ∧ M.FanOut e ∧
    1 ≤ cS ∧ (∀ g ∈ d.D.computed, Nci.novelReads d {g} ≤ kS) ∧ d.D.N < 2 ^ ℓ ∧ 0 < β ∧
    (∀ n (R : Nci.Ldl.Registration d n), R.Consistent (d.D.eval inp) → (∀ u, d.D.size (R.P.gates u) ≤ cS) →
      Nci.Ldl.ReadCap d R.P kS → Nci.Ldl.Assumptions.EvaluatesBlocks (D n R) t R (fun u => y (R.P.gates u)) aBlk) ∧
    (∀ n (R : Nci.Ldl.Registration d n), R.Consistent (d.D.eval inp) → (∀ u, d.D.size (R.P.gates u) ≤ cS) →
      Nci.Ldl.ReadCap d R.P kS →
      Nci.Ldl.Assumptions.ActivationRate β Λ t (D n R) R M (fun u => y (R.P.gates u)) (d.D.eval inp)) ∧
    e * (kS + ℓ + aBlk + Λ) / β ≤ q cS kS

/-- **The per-block cap needs the activation premise and Lemma 1's**: without `ActivationRate`, and without
`EvaluatesBlocks`, the statement is false. -/
def ServingPerBlockCap_nontrivial : Prop :=
  (¬ ∀ (d : Declaration) (J : Counted d.D) {ι α : Type} [DecidableEq α] (M : Nci.Ldl.Products d.D ι α) (e : ℕ)
      (D : (n : ℕ) → Nci.Ldl.Registration d n → Nci.Ldl.Decoder) (t aBlk ℓ : ℕ)
      (y : Finset (Fin d.D.N) → d.D.Transcript → List Bool) (β Λ : ℝ) (inp : d.D.Transcript) (cS kS : ℕ)
      (q : ℕ → ℕ → ℝ),
      (∀ v ∈ J.V, v ∈ J.body v) → (∀ v ∈ J.V, J.ω v ≤ 1) → J.V ⊆ M.J.image M.rep → M.FanOut e →
      1 ≤ cS → (∀ g ∈ d.D.computed, Nci.novelReads d {g} ≤ kS) → d.D.N < 2 ^ ℓ → 0 < β →
      (∀ n (R : Nci.Ldl.Registration d n), R.Consistent (d.D.eval inp) → (∀ u, d.D.size (R.P.gates u) ≤ cS) →
        Nci.Ldl.ReadCap d R.P kS → Nci.Ldl.Assumptions.EvaluatesBlocks (D n R) t R (fun u => y (R.P.gates u)) aBlk) →
      e * (kS + ℓ + aBlk + Λ) / β ≤ q cS kS →
        PerBlockCap d.D d.θ J q cS kS) ∧
  (¬ ∀ (d : Declaration) (J : Counted d.D) {ι α : Type} [DecidableEq α] (M : Nci.Ldl.Products d.D ι α) (e : ℕ)
      (D : (n : ℕ) → Nci.Ldl.Registration d n → Nci.Ldl.Decoder) (t aBlk ℓ : ℕ)
      (y : Finset (Fin d.D.N) → d.D.Transcript → List Bool) (β Λ : ℝ) (inp : d.D.Transcript) (cS kS : ℕ)
      (q : ℕ → ℕ → ℝ),
      (∀ v ∈ J.V, v ∈ J.body v) → (∀ v ∈ J.V, J.ω v ≤ 1) → J.V ⊆ M.J.image M.rep → M.FanOut e →
      1 ≤ cS → (∀ g ∈ d.D.computed, Nci.novelReads d {g} ≤ kS) → d.D.N < 2 ^ ℓ → 0 < β →
      (∀ n (R : Nci.Ldl.Registration d n), R.Consistent (d.D.eval inp) → (∀ u, d.D.size (R.P.gates u) ≤ cS) →
        Nci.Ldl.ReadCap d R.P kS →
        Nci.Ldl.Assumptions.ActivationRate β Λ t (D n R) R M (fun u => y (R.P.gates u)) (d.D.eval inp)) →
      e * (kS + ℓ + aBlk + Λ) / β ≤ q cS kS →
        PerBlockCap d.D d.θ J q cS kS)

/-- **The serving route's premises hold together**, at errors `0`, in an experiment whose verifier accepts and whose
computation succeeds on every run, at a positive threshold `windowsMin N`. -/
def ServingEndToEnd_honest : Prop :=
  ∃ (G : Game Unit) (P : Params Unit Unit Unit) (play : Game.Strategy G → Unit → Play Unit Unit)
    (Accept Claims Success : Game.Strategy G → Unit → Prop) (N : ℝ),
    (∀ s, P.hw.compute (P.X s) * P.ω ≤ P.cS) ∧ 0 < P.qS ∧ 0 < P.windowsMin N ∧
    (∀ s, Game.prob (fun o => Accept s o ∧ Success s o) G s = 1) ∧
    (∀ s, Game.prob (fun o => Accept s o ∧ ¬ Claims s o) G s = 0) ∧
    Assumptions.ServingBridge G P play Claims (fun _ => 0) ∧ Assumptions.ServingCoverage G P play Success N (fun _ => 0)

/-- **The serving route needs each of its premises, at a verifier that accepts everything**: with the joint guarantee's
bound, the serving bridge or coverage dropped, the statement is false even for experiments whose verifier accepts every
run. -/
def ServingEndToEnd_nontrivial : Prop :=
  (¬ ∀ (O L σ E : Type) [Fintype σ] [Fintype E] (G : Game O) (P : Params L σ E)
      (play : Game.Strategy G → O → Play L σ) (Accept Claims Success : Game.Strategy G → O → Prop) (N : ℝ)
      (δ δb η : Game.Strategy G → ℝ≥0∞),
      (∀ s o, Accept s o) → (∀ s, P.hw.compute (P.X s) * P.ω ≤ P.cS) → 0 < P.qS →
      Assumptions.ServingBridge G P play Claims δb → Assumptions.ServingCoverage G P play Success N η →
      ∀ s, Game.prob (fun o => Accept s o ∧ Success s o ∧ ((play s o).windows : ℝ) < P.windowsMin N) G s ≤
        δ s + δb s + η s) ∧
  (¬ ∀ (O L σ E : Type) [Fintype σ] [Fintype E] (G : Game O) (P : Params L σ E)
      (play : Game.Strategy G → O → Play L σ) (Accept Claims Success : Game.Strategy G → O → Prop) (N : ℝ)
      (δ δb η : Game.Strategy G → ℝ≥0∞),
      (∀ s o, Accept s o) → (∀ s, P.hw.compute (P.X s) * P.ω ≤ P.cS) → 0 < P.qS →
      (∀ s, Game.prob (fun o => Accept s o ∧ ¬ Claims s o) G s ≤ δ s) →
      Assumptions.ServingCoverage G P play Success N η →
      ∀ s, Game.prob (fun o => Accept s o ∧ Success s o ∧ ((play s o).windows : ℝ) < P.windowsMin N) G s ≤
        δ s + δb s + η s) ∧
  (¬ ∀ (O L σ E : Type) [Fintype σ] [Fintype E] (G : Game O) (P : Params L σ E)
      (play : Game.Strategy G → O → Play L σ) (Accept Claims Success : Game.Strategy G → O → Prop) (N : ℝ)
      (δ δb η : Game.Strategy G → ℝ≥0∞),
      (∀ s o, Accept s o) → (∀ s, P.hw.compute (P.X s) * P.ω ≤ P.cS) → 0 < P.qS →
      (∀ s, Game.prob (fun o => Accept s o ∧ ¬ Claims s o) G s ≤ δ s) →
      Assumptions.ServingBridge G P play Claims δb →
      ∀ s, Game.prob (fun o => Accept s o ∧ Success s o ∧ ((play s o).windows : ℝ) < P.windowsMin N) G s ≤
        δ s + δb s + η s)

end Accounting.Guarantees.ServingRoute
