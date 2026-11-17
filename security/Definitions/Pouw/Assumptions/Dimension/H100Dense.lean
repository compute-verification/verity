module

public import Definitions.Pouw.Assumptions.Dimension.H100W

/-! Dense words and the legacy k8 class in `MH100w`: the accumulator form `AccIn`, dense and apart targets, and their certificates (draft semantics, pending review). -/

@[expose] public section

namespace Pouw.Dimension.H100

variable {G : Type}

def DependsOn {Λ : Type} (S : Set G) (p : Λ → G → ℚ) (J : Finset Λ) (d : G → ℚ) : Prop :=
  ∀ g ∈ S, ∀ g' ∈ S, (∀ j ∈ J, p j g = p j g') → d g = d g'

/-- The increments of a cheap tensor-core instruction with `m` lanes; `β` is the rounding of the add. -/
def LaneFns {Λ : Type} (m : ℕ) (β : ℚ) (S : Set G) (p : Λ → G → ℚ) : Set (G → ℚ) :=
  {d | ∃ J : Finset Λ, J.card ≤ m ∧ ∃ d₀ : G → ℚ, DependsOn S p J d₀ ∧ ∀ g ∈ S, |d g - d₀ g| ≤ β}

def Contributes {Λ : Type} (S : Set G) (p : Λ → G → ℚ) (θ : ℚ) (t : G → ℚ) (j : Λ) : Prop :=
  ∃ g ∈ S, ∃ g' ∈ S, (∀ i, i ≠ j → p i g = p i g') ∧ θ < |t g - t g'|

def Op100.AccOK (a : ℕ) (S : Set G) (F D : Set (G → ℚ)) (Rs : List (G → ℚ)) : Op100 G → Prop
  | .mma outs p => p < a * outs.length →
      ∀ f ∈ outs, ∃ C : G → ℚ, (C ∈ Rs ∨ C ∈ F) ∧ ∃ d ∈ D, ∀ g ∈ S, f g = C g + d g
  | _ => True

def AccIn (a : ℕ) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (S : Set G) (F D : Set (G → ℚ)) :
    List (Op100 G) → List (G → ℚ) → Prop
  | [], _ => True
  | o :: P, Rs => o.AccOK a S F D Rs ∧ AccIn a fpSem S F D P (Rs ++ o.outs fpSem Rs)

def DenseOn (S : Set G) (F D : Set (G → ℚ)) (T : Finset (G → ℚ)) : Prop :=
  ∀ t ∈ T, ∀ f ∈ F, ∀ d ∈ D, ¬ ∀ g ∈ S, t g = f g + d g

def ApartOn (S : Set G) (D : Set (G → ℚ)) (t t' : G → ℚ) : Prop :=
  ∀ d ∈ D, ∀ d' ∈ D, ¬ ∀ g ∈ S, t g - d g = t' g - d' g

def SeparatedOn (S : Set G) (D : Set (G → ℚ)) (T : Finset (G → ℚ)) : Prop :=
  ∀ t ∈ T, ∀ t' ∈ T, t ≠ t' → ApartOn S D t t'

/-- A word to which every lane contributes. -/
def sumCoords (m : ℕ) : (Fin (m + 1) → ℚ) → ℚ := fun g => ∑ j, g j

/-! ## Guarantees (draft semantics, pending review) -/

/-- **Dense words cost `a` against the legacy class** (draft semantics, pending review). -/
def DenseWritesLegacy : Prop :=
  ∀ (a c : ℕ) (pr : Prices100), FloorTc a c pr → c ≤ a → a ≤ 2 * c →
    ∀ (G : Type) (F D : Set (G → ℚ)), (∀ q : ℚ, (fun _ : G => q) ∈ F) → (0 : G → ℚ) ∈ D →
    ∀ (S : Set G) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (G → ℚ)) (P : List (Op100 G))
      (T₁ T₂ : Finset (G → ℚ)),
      (∀ f ∈ L, f ∈ F) → (∀ o ∈ P, o.WFw pr F) → AccIn a fpSem S F D P L →
      DenseOn S F D T₁ → SeparatedOn S D T₁ → DistinctOn S F T₂ → (∀ t ∈ T₁, ∀ t' ∈ T₂, ApartOn S D t t') →
      OutputsOn fpSem S P L T₁ → OutputsOn fpSem S P L T₂ →
      a * T₁.card + c * T₂.card ≤ cost100 pr P

/-- **The dense certificate** (draft semantics, pending review). -/
def DenseCert : Prop :=
  ∀ (G Λ : Type) (S : Set G) (p : Λ → G → ℚ) (F : Set (G → ℚ)) (m r : ℕ) (β : ℚ) (T : Finset (G → ℚ)),
    (∀ f ∈ F, ∃ J : Finset Λ, J.card ≤ r ∧ DependsOn S p J f) →
    (∀ t ∈ T, ∃ W : Finset Λ, m + r < W.card ∧ ∀ j ∈ W, Contributes S p (2 * β) t j) →
    DenseOn S F (LaneFns m β S p) T

/-- **The apart certificate** (draft semantics, pending review). -/
def ApartCert : Prop :=
  ∀ (G Λ : Type) (S : Set G) (p : Λ → G → ℚ) (m : ℕ) (β : ℚ) (t t' : G → ℚ),
    (∃ W : Finset Λ, 2 * m < W.card ∧ ∀ j ∈ W, Contributes S p (4 * β) (t - t') j) →
    ApartOn S (LaneFns m β S p) t t'

/-- **The γ bound with a legacy class** (draft semantics, pending review). -/
def LegacyGap : Prop :=
  ∀ (a c : ℕ) (pr : Prices100), FloorTc a c pr → c ≤ a → a ≤ 2 * c → 0 < a →
    ∀ (G : Type) (F D : Set (G → ℚ)), (∀ q : ℚ, (fun _ : G => q) ∈ F) → (0 : G → ℚ) ∈ D →
    ∀ (S : Set G) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (G → ℚ)) (P : List (Op100 G)) (ts : List (G → ℚ))
      (T₁ T₂ : Finset (G → ℚ)) (ε μ : ℚ),
      0 < ts.length → (∀ t ∈ T₁, t ∈ ts) → (∀ t ∈ T₂, t ∈ ts) →
      (∀ f ∈ L, f ∈ F) → (∀ o ∈ P, o.WFw pr F) → AccIn a fpSem S F D P L →
      DenseOn S F D T₁ → SeparatedOn S D T₁ → DistinctOn S F T₂ → (∀ t ∈ T₁, ∀ t' ∈ T₂, ApartOn S D t t') →
      OutputsOn fpSem S P L T₁ → OutputsOn fpSem S P L T₂ →
      (1 - ε) * ts.length ≤ (T₁.card + T₂.card : ℚ) → (T₂.card : ℚ) ≤ μ * ts.length →
      1 - (cost100 pr P : ℚ) / (a * ts.length) ≤ ε + μ * (a - c) / a

/-- **Dense words exist for every lane count** (draft semantics, pending review). It is at `β = 0` on a synthetic
grid; no witness on the pinned atom, with `β` carrying its truncation, is stated. -/
def DenseLaneWitness : Prop :=
  ∀ m : ℕ,
    DenseOn Set.univ {f | ∃ q : ℚ, f = fun _ => q} (LaneFns m 0 Set.univ fun j (g : Fin (m + 1) → ℚ) => g j)
        {sumCoords m} ∧
      SeparatedOn Set.univ (LaneFns m 0 Set.univ fun j (g : Fin (m + 1) → ℚ) => g j) {sumCoords m}

end Pouw.Dimension.H100
