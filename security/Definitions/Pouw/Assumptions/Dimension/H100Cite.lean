module

public import Definitions.Pouw.Assumptions.Dimension.H100Frag
public import Definitions.Pouw.Assumptions.Dimension.H100Measured

/-! What a citation of the H100 pins may claim (draft semantics, pending review). -/

@[expose] public section

namespace Pouw.Dimension.H100

/-! ## Guarantees (draft semantics, pending review) -/

/-- **Credit per register, codes per register** (draft semantics, pending review): each code earns `c/κ`. -/
def DistinctWritesPacked : Prop :=
  ∀ (c : ℕ) (pr : Prices100), Floor c pr →
    ∀ (G : Type) (F : Set (G → ℚ)), (∀ q : ℚ, (fun _ : G => q) ∈ F) →
    ∀ (S : Set G) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (G → ℚ)) (P : List (Op100 G)) (T : Finset (G → ℚ))
      (ι : Type) (codes : Finset ι) (slot : ι → G → ℚ) (κ : ℕ),
      (∀ f ∈ L, f ∈ F) → (∀ o ∈ P, o.WFw pr F) → DistinctOn S F T → OutputsOn fpSem S P L T →
      (∀ i ∈ codes, slot i ∈ T) → (∀ t ∈ T, ∀ s : Finset ι, s ⊆ codes → (∀ i ∈ s, slot i = t) → s.card ≤ κ) →
      c * codes.card ≤ κ * cost100 pr P

/-- **The k8 lemma under the named hypothesis `FragLanes`** (draft semantics, pending review): the form in which the
k8 lemma is cited, only against adversaries whose cheap tensor-core multiplicands meet `FragLanes`. -/
def DenseWritesFragLanes : Prop :=
  ∀ (a c : ℕ) (pr : Prices100), FloorTc a c pr → c ≤ a → a ≤ 2 * c →
    ∀ (G Λ : Type) (S : Set G) (p : Λ → G → ℚ) (F W : Set (G → ℚ)) (K q : ℕ) (β : ℚ),
    (∀ x : ℚ, (fun _ : G => x) ∈ F) → 0 ≤ β → FragLanes S p q W →
    ∀ (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (G → ℚ)) (P : List (Op100 G)) (T₁ T₂ : Finset (G → ℚ)),
      (∀ f ∈ L, f ∈ F) → (∀ o ∈ P, o.WFw pr F) → AccFragIn a K β fpSem S F W P L →
      DenseOn S F (LaneFns (K * q) β S p) T₁ → SeparatedOn S (LaneFns (K * q) β S p) T₁ → DistinctOn S F T₂ →
      (∀ t ∈ T₁, ∀ t' ∈ T₂, ApartOn S (LaneFns (K * q) β S p) t t') →
      OutputsOn fpSem S P L T₁ → OutputsOn fpSem S P L T₂ →
      a * T₁.card + c * T₂.card ≤ cost100 pr P

/-- **`FragLanes` is satisfiable, and has content**: the lane words meet `FragLanes 1`, and on the grid
`Fin (m + 1) → ℚ` with the coordinates as lanes, the sum of the coordinates fails `FragLanes m`. -/
def FragLanesSat : Prop :=
  (∀ (G Λ : Type) (S : Set G) (p : Λ → G → ℚ), FragLanes S p 1 (Set.range p)) ∧
    ∀ m : ℕ, ¬ FragLanes Set.univ (fun j (g : Fin (m + 1) → ℚ) => g j) m {sumCoords m}

/-- **At `h100Measured` the distinct-writes bound needs no hypothesis on the tensor-core instructions** (draft
semantics, pending review). It is not a hardware citation: `h100Measured` does not bound the hardware from below. -/
def DistinctWritesMeasured : Prop :=
  ∀ (G : Type) (F : Set (G → ℚ)), (∀ q : ℚ, (fun _ : G => q) ∈ F) →
    ∀ (S : Set G) (fpSem : ℕ → ℚ → ℚ → ℚ → ℚ) (L : List (G → ℚ)) (P : List (Op100 G)) (T : Finset (G → ℚ)),
      (∀ f ∈ L, f ∈ F) → (∀ o ∈ P, o.WFw h100Measured F) → DistinctOn S F T → OutputsOn fpSem S P L T →
      32 * T.card ≤ cost100 h100Measured P

end Pouw.Dimension.H100
