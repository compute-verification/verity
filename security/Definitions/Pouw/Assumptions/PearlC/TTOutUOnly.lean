module

public import Definitions.Pouw.PearlC.W1
public import Definitions.Pouw.PearlC.DeviceRev1

/-! TT_OUT under U-only binding, `TTOutU` and its Pearl-C instance, a new conjecture that asserts more than the (C̃, U) row, not a weakening. -/

@[expose] public section

namespace Pouw.PearlC

open Finset Pouw.Fp8Atom

variable {Q R S : Type}

def CorrectU (P : Protocol Q R S) (U : Workload) (H : Q → R) (s : S) (u : ℕ) (c : Commit) : Prop :=
  P.actOK (U.layout.shape u) c.act ∧ P.inertBoundOK H s U u c.act ∧
    (∀ j < (U.layout.shape u).n, P.fixedOK (U.layout.shape u) (U.fixed (U.layout.fixedId u)) j) ∧
    ∀ i < (U.layout.shape u).m, ∀ j < (U.layout.shape u).n, wordQ (c.useful i j) = P.useful H s U u c.act i j

open Classical in
noncomputable def correctSetU (P : Protocol Q R S) (U : Workload) (H : Q → R) (s : S) (τ : Trace) : Finset ℕ :=
  (range U.layout.N).filter fun u => ∃ c, τ u = some c ∧ CorrectU P U H s u c

def WinsU (P : Protocol Q R S) (U : Workload) (g T : ℝ) (τ : Trace) (H : Q → R) (s : S) : Prop :=
  T / (1 - g) < ∑ u ∈ correctSetU P U H s τ, P.Wref U.layout u

def GGapU [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (P : Protocol Q R S)
    (D : Layout → Prop) (g : ℝ) (δ : ℕ → ℕ → ℝ) : Prop :=
  ∀ U : Workload, U.InDomain P D → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 → ∀ A : Adversary CM, A.Bounded U T q →
    pr (fun x : (Q → R) × S => WinsU P U g T (A.trace U x.1 x.2) x.1 x.2) ≤ δ q U.layout.N

end Pouw.PearlC

namespace Pouw.PearlC.Assumptions

variable {Q R S : Type}

/-- **TT_OUT(g₀) over U alone**: `TTOut` with `K′ = correctSetU`, the units whose useful words are correct and whose
debit is within the cap, whatever their ticket words. -/
def TTOutU [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (P : Protocol Q R S)
    (D : Layout → Prop) (g₀ : ℝ) (δ : ℕ → ℕ → ℝ) : Prop :=
  CM.Queries ∧
  ∀ U : Workload, U.InDomain P D → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 → ∀ A : Adversary CM, A.Bounded U T q →
    pr (fun x : (Q → R) × S =>
      T / (1 - g₀) < ∑ u ∈ correctSetU P U x.1 x.2 (A.trace U x.1 x.2),
        P.credit x.1 x.2 U u (actOf (A.trace U x.1 x.2) u))
      ≤ δ q U.layout.N

/-- **TT_OUT(1/400) over U alone for Pearl-C at a device record `d`**: `TTOutPearlCDevRev1` with the ticket words
unchecked. It also carries cross-atom sets whose effect on C̃ lands inside U's slack; its cost model charges a
tensor-core instruction its whole fragment shape. -/
def TTOutPearlCDevUOnly [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (CM : CostModel Q R S) (d : PearlCDevice)
    (sem : PearlCSem Q R S) (ρ : ℚ) : Prop :=
  TTOutU CM (pearlCProtocolDevRev1 d sem ρ) (pearlCDomainDev d) (1 / 400) δPearlC

def TTOutUW1 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (dv : W1.W1Device) (salt : S → List ℕ)
    (P : Protocol Q R S) (D : Layout → Prop) (g₀ : ℝ) (δ : ℕ → ℕ → ℝ) : Prop :=
  ∀ U : Workload, U.InDomain P D → ∀ T : ℝ, 0 ≤ T → ∀ q : ℕ, q ≤ 2 ^ 64 →
    ∀ A : Adversary (W1.w1Priced dv salt : CostModel Q R S), A.Bounded U T q →
    pr (fun x : (Q → R) × S =>
      T / (1 - g₀) < ∑ u ∈ correctSetU P U x.1 x.2 (A.trace U x.1 x.2),
        P.credit x.1 x.2 U u (actOf (A.trace U x.1 x.2) u))
      ≤ δ q U.layout.N

def TTOutPearlCDevUOnlyW1 [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (dv : W1.W1Device) (salt : S → List ℕ)
    (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) : Prop :=
  TTOutUW1 dv salt (pearlCProtocolDevRev1 d sem ρ) (pearlCDomainDev d) (1 / 400) δPearlC

end Pouw.PearlC.Assumptions
