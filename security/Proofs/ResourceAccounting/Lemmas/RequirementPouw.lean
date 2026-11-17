module

public import Definitions.Pouw.PouwWork
public import Definitions.ResourceAccounting.Requirement.Placed

/-! Steps of proofs on PoUW's credits and help when a node runs a registered circuit as placed, pinned by no record. -/

@[expose] public section

namespace ResourceAccounting.Guarantees

open ResourceAccounting.Requirement

/-- It says what an honest node is credited, not what a node that passes PoUW ran. -/
def RequirementCredits : Prop :=
  ∀ (Ω L : Type) (n : ℕ) (W : Wiring (Fin n)) (p : Ω → Placement L (Fin n)) (P : PouwWork.Verdict Ω L)
    (X : Finset L) (π : Ω → Fin n → ℕ) (m : Ω → Finset (Fin n)) (r : Ω),
    PouwWork.credits (fun r => (p r).actual W) P X π m r = work (π r) (region W (p r) X (Set.Ico P.ts P.td) \ m r)

open Classical in
def RequirementHelp : Prop :=
  ∀ (Ω L : Type) (n : ℕ) (W : Wiring (Fin n)) (p : Ω → Placement L (Fin n)) (P : PouwWork.Verdict Ω L)
    (X : Finset L) (r : Ω), W.Topological → (p r).Follows W →
    PouwWork.help (fun r => (p r).actual W) P X r = ingress W (p r) (X ∪ P.V) (Set.Ico P.ts P.td)

open Classical in
def RequirementHelpBound : Prop :=
  ∀ (Ω L : Type) (n : ℕ) (W : Wiring (Fin n)) (p : Ω → Placement L (Fin n)) (P : PouwWork.Verdict Ω L)
    (X : Finset L) (r : Ω),
    PouwWork.help (fun r => (p r).actual W) P X r ≤ ingress W (p r) (X ∪ P.V) (Set.Ico P.ts P.td)

def RequirementStepsInner : Prop :=
  ∀ {G : Type} [DecidableEq G] {L : Type} (W : Wiring G) (p : Placement L G) (X : Finset L) (w : Window) (ts td : ℕ),
    p.Paced X → (∀ x ∈ X, ts ≤ p.ticks x w.start ∧ p.ticks x (w.start + w.len) ≤ td) →
    stepRegion W p X w ⊆ region W p X (Set.Ico ts td)

def RequirementStepsOuter : Prop :=
  ∀ {G : Type} [DecidableEq G] {L : Type} (W : Wiring G) (p : Placement L G) (X : Finset L) (w : Window) (ts td : ℕ),
    p.Paced X → (∀ x ∈ X, p.ticks x w.start ≤ ts ∧ td ≤ p.ticks x (w.start + w.len)) →
    region W p X (Set.Ico ts td) ⊆ stepRegion W p X w

def RequirementCreditsSteps : Prop :=
  ∀ (Ω L : Type) (n : ℕ) (W : Wiring (Fin n)) (p : Ω → Placement L (Fin n)) (P : PouwWork.Verdict Ω L)
    (X : Finset L) (π : Ω → Fin n → ℕ) (m : Ω → Finset (Fin n)) (r : Ω) (w : Window),
    (p r).Paced X → (∀ x ∈ X, P.ts ≤ (p r).ticks x w.start ∧ (p r).ticks x (w.start + w.len) ≤ P.td) →
    work (π r) (stepRegion W (p r) X w \ m r) ≤ PouwWork.credits (fun r => (p r).actual W) P X π m r

end ResourceAccounting.Guarantees
