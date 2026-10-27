import Definitions.Accounting.Overhead

/-!
# The end-to-end overhead guarantee's generic steps (trusted)

Steps 1 and 8 of `note:training-20261027T0134Z-draft-e2e-overhead-proof`, over its objects (`Accounting.Overhead`):

* `EndToEnd` (Step 1): in one experiment, a causal game (`Verity.Game`) at every strategy, the probability that the verifier
  accepts, the computation succeeds and its cost is below `s_min` is at most the joint guarantee's error, the bridges'
  and the coverage exception's, when the cost is at least `s_min` on every run in the class whose count reaches `N`. No
  independence is used. The onsite and remote end-to-end guarantees instantiate it, the joint guarantee
  (`Accounting.Guarantees.JointPeriodLive`) giving its first error;
* `ThresholdOfPointwise`: a size bound on every allowed circuit with a property `P`, and a bound `η` on every circuit's
  probability of succeeding without it (the coverage exception), bound the threshold `T_𝒢(η; 𝒜)` below;
* `ReferenceBound`: a circuit that succeeds with probability above `η` bounds the threshold above by its size;
* `OverheadOfReference` (Step 8): the two give the overhead `T_𝒢(η; 𝒜) / T_𝒢(η; 𝒜_phys) ≥ ρ` against a reference
  circuit of size `T₀`.

Each has an honest statement (`_honest`: its hypotheses hold at a constructed instance where its conclusion says
something) and a nontrivial one (`_nontrivial`: without one of its premises it is false, at a constructed instance; for
`EndToEnd`, at a verifier that accepts everything), the top's rule of 26 Oct (LEAN-66).

The honest instances of the threshold's statements are the NOT task: the input `x` a uniform bit, its one acceptable
output `¬ x`, read off a circuit's evaluation with every input gate holding `x` as whether some output gate's low bit is
set. A circuit with no operation gate outputs only what its input gates hold, so it succeeds with probability at most
`1/2`, and one NOT gate succeeds always.
-/

namespace Accounting.Guarantees.Overhead

open Verity Accounting.Overhead
open scoped ENNReal

/-- **The end-to-end event decomposition.** In one experiment, the causal game `G` played at a strategy `s`, read on its
run `o`: the verifier accepts (`Accept`), every subprotocol's claim holds (`Claims`), the actual computation is in the
class (`InClass`) and it succeeds (`Success`); `M` is a counted quantity and `S` a cost (a size or a time). Suppose that
at every strategy `s`:

* `Pr[Accept ∧ ¬ Claims] ≤ δ s` (the joint guarantee);
* `Pr[Claims ∧ ¬ InClass] ≤ δ_b s` (the bridges from the claims to the class);
* on every run in the class whose count reaches `N`, the cost is at least `s_min`;
* `Pr[InClass ∧ Success ∧ M < N] ≤ η s` (the coverage exception).

Then at every strategy `Pr[Accept ∧ Success ∧ S < s_min] ≤ δ s + δ_b s + η s`. The events are of one run of one
experiment, and no independence between them is used. The errors may depend on the strategy, as a collision term at
the strategy's finder does. -/
def EndToEnd : Prop :=
  ∀ (O : Type) (G : Game O) (Accept Claims InClass Success : Game.Strategy G → O → Prop)
    (M S : Game.Strategy G → O → ℝ) (N sMin : ℝ) (δ δb η : Game.Strategy G → ℝ≥0∞),
    (∀ s, Game.prob (fun o => Accept s o ∧ ¬ Claims s o) G s ≤ δ s) →
    (∀ s, Game.prob (fun o => Claims s o ∧ ¬ InClass s o) G s ≤ δb s) →
    (∀ s o, InClass s o → N ≤ M s o → sMin ≤ S s o) →
    (∀ s, Game.prob (fun o => InClass s o ∧ Success s o ∧ M s o < N) G s ≤ η s) →
    ∀ s, Game.prob (fun o => Accept s o ∧ Success s o ∧ S s o < sMin) G s ≤ δ s + δb s + η s

/-- **`EndToEnd`'s hypotheses hold where its conclusion says something**: at an experiment where the verifier accepts
and the computation succeeds with probability `1` at every strategy, every hypothesis holds at errors `0`. -/
def EndToEnd_honest : Prop :=
  ∃ (O : Type) (G : Game O) (Accept Claims InClass Success : Game.Strategy G → O → Prop)
    (M S : Game.Strategy G → O → ℝ) (N sMin : ℝ),
    (∀ s, Game.prob (fun o => Accept s o ∧ Success s o) G s = 1) ∧
    (∀ s, Game.prob (fun o => Accept s o ∧ ¬ Claims s o) G s = 0) ∧
    (∀ s, Game.prob (fun o => Claims s o ∧ ¬ InClass s o) G s = 0) ∧
    (∀ s o, InClass s o → N ≤ M s o → sMin ≤ S s o) ∧
    (∀ s, Game.prob (fun o => InClass s o ∧ Success s o ∧ M s o < N) G s = 0)

/-- **`EndToEnd` needs each of its premises, at a verifier that accepts everything**: with the joint guarantee's, the
bridges', the pointwise cost bound or the coverage exception dropped, the statement is false even for experiments whose
verifier accepts every run. -/
def EndToEnd_nontrivial : Prop :=
  (¬ ∀ (O : Type) (G : Game O) (Accept Claims InClass Success : Game.Strategy G → O → Prop)
      (M S : Game.Strategy G → O → ℝ) (N sMin : ℝ) (δ δb η : Game.Strategy G → ℝ≥0∞),
      (∀ s o, Accept s o) →
      (∀ s, Game.prob (fun o => Claims s o ∧ ¬ InClass s o) G s ≤ δb s) →
      (∀ s o, InClass s o → N ≤ M s o → sMin ≤ S s o) →
      (∀ s, Game.prob (fun o => InClass s o ∧ Success s o ∧ M s o < N) G s ≤ η s) →
      ∀ s, Game.prob (fun o => Accept s o ∧ Success s o ∧ S s o < sMin) G s ≤ δ s + δb s + η s) ∧
  (¬ ∀ (O : Type) (G : Game O) (Accept Claims InClass Success : Game.Strategy G → O → Prop)
      (M S : Game.Strategy G → O → ℝ) (N sMin : ℝ) (δ δb η : Game.Strategy G → ℝ≥0∞),
      (∀ s o, Accept s o) →
      (∀ s, Game.prob (fun o => Accept s o ∧ ¬ Claims s o) G s ≤ δ s) →
      (∀ s o, InClass s o → N ≤ M s o → sMin ≤ S s o) →
      (∀ s, Game.prob (fun o => InClass s o ∧ Success s o ∧ M s o < N) G s ≤ η s) →
      ∀ s, Game.prob (fun o => Accept s o ∧ Success s o ∧ S s o < sMin) G s ≤ δ s + δb s + η s) ∧
  (¬ ∀ (O : Type) (G : Game O) (Accept Claims InClass Success : Game.Strategy G → O → Prop)
      (M S : Game.Strategy G → O → ℝ) (N sMin : ℝ) (δ δb η : Game.Strategy G → ℝ≥0∞),
      (∀ s o, Accept s o) →
      (∀ s, Game.prob (fun o => Accept s o ∧ ¬ Claims s o) G s ≤ δ s) →
      (∀ s, Game.prob (fun o => Claims s o ∧ ¬ InClass s o) G s ≤ δb s) →
      (∀ s, Game.prob (fun o => InClass s o ∧ Success s o ∧ M s o < N) G s ≤ η s) →
      ∀ s, Game.prob (fun o => Accept s o ∧ Success s o ∧ S s o < sMin) G s ≤ δ s + δb s + η s) ∧
  (¬ ∀ (O : Type) (G : Game O) (Accept Claims InClass Success : Game.Strategy G → O → Prop)
      (M S : Game.Strategy G → O → ℝ) (N sMin : ℝ) (δ δb η : Game.Strategy G → ℝ≥0∞),
      (∀ s o, Accept s o) →
      (∀ s, Game.prob (fun o => Accept s o ∧ ¬ Claims s o) G s ≤ δ s) →
      (∀ s, Game.prob (fun o => Claims s o ∧ ¬ InClass s o) G s ≤ δb s) →
      (∀ s o, InClass s o → N ≤ M s o → sMin ≤ S s o) →
      ∀ s, Game.prob (fun o => Accept s o ∧ Success s o ∧ S s o < sMin) G s ≤ δ s + δb s + η s)

/-- **The threshold from a pointwise size bound and the coverage exception.** On a task `𝒢` with outputs `out`, if every
allowed computation `(C, x) ∈ 𝒜` with the property `P` has `|C| ≥ ρ T₀`, and every circuit succeeds in `𝒜` without `P`
with probability at most `η`, then `T_𝒢(η; 𝒜) ≥ ρ T₀`. -/
def ThresholdOfPointwise : Prop :=
  ∀ (X Y : Type) (G : Task X Y) (A : Class X) (out : Circuit → X → Y) (P : Circuit → X → Prop) (ρ T₀ η : ℝ≥0∞),
    (∀ C x, (C, x) ∈ A → P C x → ρ * T₀ ≤ ((C.size C.computed : ℕ) : ℝ≥0∞)) →
    (∀ C, G.prob (fun x 𝒴 => (C, x) ∈ A ∧ out C x ∈ 𝒴 ∧ ¬ P C x) ≤ η) →
    ρ * T₀ ≤ threshold G A out η

/-- **`ThresholdOfPointwise`'s hypotheses hold at the NOT task**, over every circuit, with `P` "at least one operation
gate", `ρ T₀ = 1` and coverage exception `η = 1/2`, where some circuit succeeds with probability above `1/2`: a circuit
with no operation gate succeeds with probability at most `1/2`. -/
def ThresholdOfPointwise_honest : Prop :=
  let G : Task Bool Bool := { Ω := Bool, draw := fun ω => (ω, {!ω}) }
  let out : Circuit → Bool → Bool := fun C x =>
    decide (∃ o ∈ C.outputs, (C.eval (fun g => BitVec.ofNat (C.w g) x.toNat) o).getLsbD 0 = true)
  (∃ C, 2⁻¹ < success G Set.univ out C) ∧
  (∀ C x, (C, x) ∈ (Set.univ : Class Bool) → 1 ≤ C.size C.computed →
    (1 : ℝ≥0∞) * 1 ≤ ((C.size C.computed : ℕ) : ℝ≥0∞)) ∧
  (∀ C, G.prob (fun x 𝒴 => (C, x) ∈ (Set.univ : Class Bool) ∧ out C x ∈ 𝒴 ∧ ¬ 1 ≤ C.size C.computed) ≤ 2⁻¹)

/-- **`ThresholdOfPointwise` needs both premises**: without the coverage exception, or without the pointwise size bound,
it is false. -/
def ThresholdOfPointwise_nontrivial : Prop :=
  (¬ ∀ (X Y : Type) (G : Task X Y) (A : Class X) (out : Circuit → X → Y) (P : Circuit → X → Prop) (ρ T₀ η : ℝ≥0∞),
      (∀ C x, (C, x) ∈ A → P C x → ρ * T₀ ≤ ((C.size C.computed : ℕ) : ℝ≥0∞)) →
      ρ * T₀ ≤ threshold G A out η) ∧
  (¬ ∀ (X Y : Type) (G : Task X Y) (A : Class X) (out : Circuit → X → Y) (P : Circuit → X → Prop) (ρ T₀ η : ℝ≥0∞),
      (∀ C, G.prob (fun x 𝒴 => (C, x) ∈ A ∧ out C x ∈ 𝒴 ∧ ¬ P C x) ≤ η) →
      ρ * T₀ ≤ threshold G A out η)

/-- **The threshold from a reference circuit.** A circuit `C₀` of size at most `T₀` that succeeds in `𝒜_phys` with
probability above `η` gives `T_𝒢(η; 𝒜_phys) ≤ T₀`. -/
def ReferenceBound : Prop :=
  ∀ (X Y : Type) (G : Task X Y) (Aphys : Class X) (out : Circuit → X → Y) (C₀ : Circuit) (T₀ η : ℝ≥0∞),
    ((C₀.size C₀.computed : ℕ) : ℝ≥0∞) ≤ T₀ → η < success G Aphys out C₀ → threshold G Aphys out η ≤ T₀

/-- **`ReferenceBound`'s hypotheses hold at the NOT task**: a circuit of one operation gate succeeds over every circuit
with probability above `1/2`. -/
def ReferenceBound_honest : Prop :=
  let G : Task Bool Bool := { Ω := Bool, draw := fun ω => (ω, {!ω}) }
  let out : Circuit → Bool → Bool := fun C x =>
    decide (∃ o ∈ C.outputs, (C.eval (fun g => BitVec.ofNat (C.w g) x.toNat) o).getLsbD 0 = true)
  ∃ C₀ : Circuit, ((C₀.size C₀.computed : ℕ) : ℝ≥0∞) ≤ 1 ∧ 2⁻¹ < success G Set.univ out C₀

/-- **`ReferenceBound` needs both premises**: without the reference's success, or without its size, it is false. -/
def ReferenceBound_nontrivial : Prop :=
  (¬ ∀ (X Y : Type) (G : Task X Y) (Aphys : Class X) (out : Circuit → X → Y) (C₀ : Circuit) (T₀ η : ℝ≥0∞),
      ((C₀.size C₀.computed : ℕ) : ℝ≥0∞) ≤ T₀ → threshold G Aphys out η ≤ T₀) ∧
  (¬ ∀ (X Y : Type) (G : Task X Y) (Aphys : Class X) (out : Circuit → X → Y) (C₀ : Circuit) (T₀ η : ℝ≥0∞),
      η < success G Aphys out C₀ → threshold G Aphys out η ≤ T₀)

/-- **The overhead against a reference circuit.** On a task `𝒢` with outputs `out`, if every allowed computation
`(C, x) ∈ 𝒜` with the property `P` has `|C| ≥ ρ T₀`, every circuit succeeds in `𝒜` without `P` with probability at most
`η`, and a reference circuit `C₀` of size at most `T₀`, a positive finite size, succeeds in `𝒜_phys` with probability
above `η`, then the overhead's denominator `T_𝒢(η; 𝒜_phys)` is at most `T₀`, so finite, and the overhead
`T_𝒢(η; 𝒜) / T_𝒢(η; 𝒜_phys)` is at least `ρ`. -/
def OverheadOfReference : Prop :=
  ∀ (X Y : Type) (G : Task X Y) (A Aphys : Class X) (out : Circuit → X → Y) (P : Circuit → X → Prop)
    (C₀ : Circuit) (ρ T₀ η : ℝ≥0∞),
    T₀ ≠ 0 → T₀ ≠ ⊤ →
    (∀ C x, (C, x) ∈ A → P C x → ρ * T₀ ≤ ((C.size C.computed : ℕ) : ℝ≥0∞)) →
    (∀ C, G.prob (fun x 𝒴 => (C, x) ∈ A ∧ out C x ∈ 𝒴 ∧ ¬ P C x) ≤ η) →
    ((C₀.size C₀.computed : ℕ) : ℝ≥0∞) ≤ T₀ → η < success G Aphys out C₀ →
    threshold G Aphys out η ≤ T₀ ∧ ρ ≤ overhead G A Aphys out η

/-- **`OverheadOfReference`'s hypotheses hold at the NOT task with `ρ = 2`**: against every circuit as `𝒜_phys`, the class
`𝒜` of circuits with at least two operation gates, `P` true, `T₀ = 1` and `η = 1/2`, with a reference circuit of one
operation gate. -/
def OverheadOfReference_honest : Prop :=
  let G : Task Bool Bool := { Ω := Bool, draw := fun ω => (ω, {!ω}) }
  let out : Circuit → Bool → Bool := fun C x =>
    decide (∃ o ∈ C.outputs, (C.eval (fun g => BitVec.ofNat (C.w g) x.toNat) o).getLsbD 0 = true)
  let A : Class Bool := {p | 2 ≤ p.1.size p.1.computed}
  (∀ C x, (C, x) ∈ A → True → (2 : ℝ≥0∞) * 1 ≤ ((C.size C.computed : ℕ) : ℝ≥0∞)) ∧
  (∀ C, G.prob (fun x 𝒴 => (C, x) ∈ A ∧ out C x ∈ 𝒴 ∧ ¬ True) ≤ 2⁻¹) ∧
  ∃ C₀ : Circuit, ((C₀.size C₀.computed : ℕ) : ℝ≥0∞) ≤ 1 ∧ 2⁻¹ < success G Set.univ out C₀

/-- **`OverheadOfReference` needs its premises**: without the coverage exception, without the pointwise size bound, or
without the reference's success, it is false. -/
def OverheadOfReference_nontrivial : Prop :=
  (¬ ∀ (X Y : Type) (G : Task X Y) (A Aphys : Class X) (out : Circuit → X → Y) (P : Circuit → X → Prop)
      (C₀ : Circuit) (ρ T₀ η : ℝ≥0∞),
      T₀ ≠ 0 → T₀ ≠ ⊤ →
      (∀ C x, (C, x) ∈ A → P C x → ρ * T₀ ≤ ((C.size C.computed : ℕ) : ℝ≥0∞)) →
      ((C₀.size C₀.computed : ℕ) : ℝ≥0∞) ≤ T₀ → η < success G Aphys out C₀ →
      threshold G Aphys out η ≤ T₀ ∧ ρ ≤ overhead G A Aphys out η) ∧
  (¬ ∀ (X Y : Type) (G : Task X Y) (A Aphys : Class X) (out : Circuit → X → Y) (P : Circuit → X → Prop)
      (C₀ : Circuit) (ρ T₀ η : ℝ≥0∞),
      T₀ ≠ 0 → T₀ ≠ ⊤ →
      (∀ C, G.prob (fun x 𝒴 => (C, x) ∈ A ∧ out C x ∈ 𝒴 ∧ ¬ P C x) ≤ η) →
      ((C₀.size C₀.computed : ℕ) : ℝ≥0∞) ≤ T₀ → η < success G Aphys out C₀ →
      threshold G Aphys out η ≤ T₀ ∧ ρ ≤ overhead G A Aphys out η) ∧
  (¬ ∀ (X Y : Type) (G : Task X Y) (A Aphys : Class X) (out : Circuit → X → Y) (P : Circuit → X → Prop)
      (C₀ : Circuit) (ρ T₀ η : ℝ≥0∞),
      T₀ ≠ 0 → T₀ ≠ ⊤ →
      (∀ C x, (C, x) ∈ A → P C x → ρ * T₀ ≤ ((C.size C.computed : ℕ) : ℝ≥0∞)) →
      (∀ C, G.prob (fun x 𝒴 => (C, x) ∈ A ∧ out C x ∈ 𝒴 ∧ ¬ P C x) ≤ η) →
      ((C₀.size C₀.computed : ℕ) : ℝ≥0∞) ≤ T₀ →
      threshold G Aphys out η ≤ T₀ ∧ ρ ≤ overhead G A Aphys out η)

end Accounting.Guarantees.Overhead
