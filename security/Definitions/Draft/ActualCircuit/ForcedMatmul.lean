module

public import Definitions.Pouw.Assumptions.PouwWork
public import Definitions.Pouw.Assumptions.PouwFloor
public import Definitions.Draft.ActualCircuit.Epoch
public import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
# Forced matrix multiplication's statement (draft)

Today's useful work (PoUW, Pearl+ among its schemes) on the actual circuit, over `ResourceAccounting.PouwWork`'s
vocabulary: the verifier at `P.V` sends the epoch's challenges, reads back each block's word by the deadline and
accepts by its rule (`Verdict`); the node is the location `X`, the epoch `[P.ts, P.td)`. The work is priced at `π` and a
marking `m` of gates is left out, each a function of the actual circuit, so the set a statement names is a set of actual
circuits. Pearl+ is one choice of `Verdict`'s words and targets; nothing here is specific to it.

These statements are about the actual circuit; useful work's deployed guarantee
(`Properties.Pouw.MonitoringPeriod.SecurityProofs.h1CheckedCreditMonitoringPeriod`) is about the declared circuit, with
its tolerance counted once a monitoring period. They rest on the premises of
`ResourceAccounting.Guarantees.PouwLocalOff` (A2, A2b and A3, the last an open conjecture), or on their pooled versions
below, and they reach the deployed check only through `AcceptIsVerdict`, which ties the verifier's transcript to PoUW's
rule.

* `forcedSet`: the actual circuits in which, if the node took in no help during the epoch, its own work outside the
  marking is at least `W`.
* `ForcedMatmulEpoch`: (a), one epoch, one set, one error: `forcedSet` at `W = Wb · Nmin`, error `δdet + δluck + δskip`.
* `forcedDeficit`, `BudgetSound`, `TailSound`, `ForcedMatmulEpochPooled`: (b), one epoch with the pooling kept: the
  node's shortfall below `Wb · N` meets an exponential-moment bound off a skip event, which gives a bound at every
  tolerance at once, and whose epochs combine by multiplying (`Draft.ActualCircuit.PooledComposition`).

A5 (`ResourceAccounting.Assumptions.PouwHelpBound` at `0`, the node isolated) is a premise on one strategy in
`PouwLocalOff`. Taken of every strategy it would say no strategy takes help, so it is inside the set instead: each set
here says nothing of an actual circuit whose node took in help during the epoch.
-/

@[expose] public section

namespace Draft.ActualCircuit

open ResourceAccounting ResourceAccounting.PouwWork

variable {L : Type}

open Classical in
/-- **The forced set**: the actual circuits in which, if the node `X` took in no help during the ticks `I` (no gate
placed outside `X` and `V` is read by `X` or `V`, `Placed.ingress`), its own computed gates during `I`, the marking
`m` left out, have work at least `W` at the pricing `π`. -/
def forcedSet (V X : Finset L) (I : Set ℕ) (π : (A : Placed L) → Fin A.N → ℕ) (m : (A : Placed L) → Finset (Fin A.N))
    (W : ℕ) : Set (Placed L) :=
  {A | A.ingress (X ∪ V) I = 0 → W ≤ A.work (π A) (A.region X I \ m A)}

/-- **The verifier's transcript decides PoUW's rule**: in every feasible run, at every challenge where the verifier at
`P.V` accepts on its transcript by `acc`, PoUW's verdict accepts (`PouwWork.Accepts`, on the challenges and what
crossed into `P.V`). New: `Sound` and `Composition` read acceptance on the transcript, `PouwWork` on the challenges and
the messages in; this says that in a feasible run the verifier's leaves run its program and that program computes
`P.accept`, the honest verifier of the trust base and a fact of the verifier's code. -/
def AcceptIsVerdict {Ω : Type} (𝓕 : Set (Actual.Run L)) (P : Verdict Ω L) (acc : Transcript L → Prop) : Prop :=
  ∀ (S : Ω → Actual.Run L) (r : Ω), S r ∈ 𝓕 → acc ((S r).view P.V) → Accepts S P r

/-- **(a) Forced matrix multiplication in one epoch.** If the verifier accepts in the epoch `[P.ts, P.td)` then, except
with probability `δdet + δluck + δskip` over its challenges, the actual circuit lies in `forcedSet` at `W = Wb · Nmin`:
had the node taken in no help during the epoch, its own work outside the marking is at least `Wb · Nmin`. Premises:
the transcript decides PoUW's rule (`AcceptIsVerdict`), and for every strategy A2 (`PouwDetection`), A2b (`PouwLuck`)
and A3 (`PouwFloorOff`, an open conjecture), as in `ResourceAccounting.Guarantees.PouwLocalOff`, of which this is the
statement over every strategy.

(a) fixes one tolerance, `Wb · (N − Nmin)`, before the epoch and pays its error in every epoch, so over a monitoring
period of `n` epochs either the errors add `n` times or each epoch's tolerance is taken separately. It drops the shape
of the shortfall's distribution, which (b) keeps so that the period's tolerance is paid once. -/
def ForcedMatmulEpoch : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (L : Type) [LinearOrder L] (P : Verdict Ω L) (𝓕 : Set (Actual.Run L))
    (acc : Transcript L → Prop) (X : Finset L) (π : (A : Placed L) → Fin A.N → ℕ)
    (m : (A : Placed L) → Finset (Fin A.N)) (Wb Nmin ρ : ℕ) (δdet δluck δskip : ℝ),
    AcceptIsVerdict 𝓕 P acc →
    ForStrategies Ω 𝓕 P.V (fun S => Assumptions.PouwDetection S P Nmin δdet ∧ Assumptions.PouwLuck S P Nmin δluck ∧
      Assumptions.PouwFloorOff S P X (fun r => π (S r).A) (fun r => m (S r).A) Wb Nmin ρ δskip) →
    Sound Ω 𝓕 P.V acc (Lies P.V fun _ => forcedSet P.V X (Set.Ico P.ts P.td) π m (Wb * Nmin)) (δdet + δluck + δskip)

open Classical in
/-- **The node's shortfall** below `W`: `W` less its own work during `I` outside the marking, at `π`, when it took in
no help during `I`, and `0` when it took in help. -/
noncomputable def forcedDeficit (V X : Finset L) (I : Set ℕ) (π : (A : Placed L) → Fin A.N → ℕ)
    (m : (A : Placed L) → Finset (Fin A.N)) (W : ℕ) (A : Placed L) : ℝ :=
  if A.ingress (X ∪ V) I = 0 then ((W - A.work (π A) (A.region X I \ m A) : ℕ) : ℝ) else 0

open Classical in
/-- **A2 and A2b, pooled**: the blocks not right at every challenge, `N − goodAll`, meet an exponential-moment bound at
`Λ` blocks: `E[exp((N − goodAll)/Λ) · 1{accept}] ≤ 1` over the challenges. `goodAll` doesn't depend on the challenge,
so this reads `Pr[accept] ≤ exp(−(N − goodAll)/Λ)`. New, open: the sample's detection bound at every number of wrong
blocks at once (`ResourceAccounting.TS.RoundGame.BudgetAt`'s shape on PoUW's verdict); nothing proves it on the actual
circuit today. -/
noncomputable def PouwDetectionBudget {Ω : Type} [Fintype Ω] (S : Ω → Actual.Run L) (P : Verdict Ω L) (Λ : ℝ) : Prop :=
  (∑ r, if Accepts S P r then Real.exp (((P.N - goodAll S P : ℕ) : ℝ) / Λ) else 0) ≤ Fintype.card Ω

/-- **A3 at every count of right blocks, off one skip event**: some event `Z` of probability at most `δ` is such that
at every challenge outside it the node's credits outside the marking, plus `ρ` a bit of help, are at least `Wb` per block
right at every challenge. New, an open conjecture: `PouwFloorOff` at every `Nmin` with one `Z` for all of them. -/
def PouwFloorLinear {Ω : Type} [Fintype Ω] (S : Ω → Actual.Run L) (P : Verdict Ω L) (X : Finset L)
    (π : ∀ r, Fin (S r).A.N → ℕ) (m : ∀ r, Finset (Fin (S r).A.N)) (Wb ρ : ℕ) (δ : ℝ) : Prop :=
  ∃ Z : Ω → Prop, Pouw.pr Z ≤ δ ∧ ∀ r, ¬ Z r → Wb * goodAll S P ≤ credits S P X π m r + ρ * help S P X r

open Classical in
/-- **An exponential-moment bound, off a skip event**, for the verifier at `V` accepting by `acc`: for every strategy
`Sound` reads, some event `Z` of probability at most `δ` is such that `E[exp(d/Λ) · 1{accept ∧ ¬Z}] ≤ 1` over the
challenges, `d` the shortfall of the actual circuit at the transcript. `d ≥ 0` is the caller's. -/
noncomputable def BudgetSound [LinearOrder L] (Ω : Type) [Fintype Ω] (𝓕 : Set (Actual.Run L)) (V : Finset L)
    (acc : Transcript L → Prop) (d : Transcript L → Placed L → ℝ) (Λ δ : ℝ) : Prop :=
  ForStrategies Ω 𝓕 V fun S => ∃ Z : Ω → Prop, Pouw.pr Z ≤ δ ∧
    (∑ r, if acc ((S r).view V) ∧ ¬ Z r then Real.exp (d ((S r).view V) (S r).A / Λ) else 0) ≤ Fintype.card Ω

/-- **A bound at every tolerance at once**: for every `t`, if the verifier accepts then, except with probability
`δ + exp(−t/Λ)`, the actual circuit's shortfall is below `t`. `BudgetSound` gives it at every `t` (Markov's inequality on
`exp(d/Λ)`); it alone does not give a bound on a sum of epochs' shortfalls at one `Λ`, which `BudgetSound` does. -/
noncomputable def TailSound [LinearOrder L] (Ω : Type) [Fintype Ω] (𝓕 : Set (Actual.Run L)) (V : Finset L)
    (acc : Transcript L → Prop) (d : Transcript L → Placed L → ℝ) (Λ δ : ℝ) : Prop :=
  ∀ t : ℝ, Sound Ω 𝓕 V acc (Lies V fun τ => {A | d τ A < t}) (δ + Real.exp (-t / Λ))

/-- **(b) Forced matrix multiplication in one epoch, the pooling kept.** If the verifier accepts in the epoch
`[P.ts, P.td)` then, off a skip event of probability at most `δskip`, the node's shortfall below `Wb · N`
(`forcedDeficit`, `0` when it took in help) has `E[exp(shortfall/(Wb · Λ)) · 1{accept ∧ ¬skip}] ≤ 1`
(`BudgetSound`): so at every tolerance `t` at once, except with probability `δskip + exp(−t/(Wb · Λ))`, the shortfall
is below `t` (`TailSound`), and over a monitoring period the epochs' bounds multiply, the skip events add, and the
period's tolerance `Wb · Λ · ln(1/δ)` is counted once (`PooledComposition`). Premises: `Λ > 0`, the transcript decides
PoUW's rule, and for every strategy A2 and A2b pooled (`PouwDetectionBudget`) and A3 at every count with one skip event
(`PouwFloorLinear`). -/
def ForcedMatmulEpochPooled : Prop :=
  ∀ (Ω : Type) [Fintype Ω] (L : Type) [LinearOrder L] (P : Verdict Ω L) (𝓕 : Set (Actual.Run L))
    (acc : Transcript L → Prop) (X : Finset L) (π : (A : Placed L) → Fin A.N → ℕ)
    (m : (A : Placed L) → Finset (Fin A.N)) (Wb ρ : ℕ) (Λ δskip : ℝ),
    0 < Λ → AcceptIsVerdict 𝓕 P acc →
    ForStrategies Ω 𝓕 P.V (fun S => PouwDetectionBudget S P Λ ∧
      PouwFloorLinear S P X (fun r => π (S r).A) (fun r => m (S r).A) Wb ρ δskip) →
    BudgetSound Ω 𝓕 P.V acc (fun _ => forcedDeficit P.V X (Set.Ico P.ts P.td) π m (Wb * P.N)) (Wb * Λ) δskip

end Draft.ActualCircuit
