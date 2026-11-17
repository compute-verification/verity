module

public import Definitions.Pouw.Guarantees.TS

/-! Steps of `Definitions.Pouw.Guarantees.TS`'s proofs, each a `def G : Prop` proved by the `SecurityProofs` theorem
of the same name and pinned by no record. -/

@[expose] public section

namespace ResourceAccounting.Guarantees.TS

open ResourceAccounting.TS ResourceAccounting.Assumptions.TS Pouw.RelativeSecurity

def DecompositionD : Prop :=
  ∀ {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U] [Fintype F] [Fintype X]
    [Nonempty R] [Nonempty F] [Nonempty X]
    (Z : Model K R T L U F X) (H : Honest K) (Ω : Type) [Fintype Ω] (accepts : Ω → Prop) (M : Ω → Run K)
    (fixed : Ω → ℝ) (lam : R → ℝ) (dbar abar δ : ℝ),
    IsCert lam → Z.Wf H.prog → 0 < H.CH → 0 < H.fH →
    (∀ ω, FacetCeilings Z.G (M ω)) →
    (∀ ω, accepts ω → Z.RouteDominated (M ω).n (fixed ω)) →
    prob (fun ω => accepts ω ∧ fixed ω < (1 - dbar - abar) * Z.Nw) ≤ δ →
    prob (fun ω => accepts ω ∧ (M ω).time / H.time <
      (1 - Z.epsSkip H.prog lam - Z.epsMand H.prog lam - Z.epsMachine H.prog lam - Z.epsLuck H.prog lam dbar -
        Z.epsAdapt H.prog lam abar) * (1 - Z.epsSched H) * (1 - Z.epsClock H)) ≤ δ

/-- It holds at every ♭, `π` and fold menu, so the fold's instructions a word `f` are no hypothesis. -/
def FloorCreditD3 : Prop :=
  ∀ {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U] [Fintype F] [Fintype X]
    [Nonempty R] [Nonempty F] [Nonempty X]
    (Z : Model K R T L U F X) (H : Honest K) (Fl : Labelled K) (Ω : Type) [Fintype Ω] (accepts : Ω → Prop)
    (M : Ω → Run K) (fixed : Ω → ℝ) (lam : R → ℝ) (dbar abar δ : ℝ),
    IsCert lam → Z.Wf Fl → 0 < H.CH → 0 < H.fH →
    (∀ ω, FacetCeilings Z.G (M ω)) →
    (∀ ω, accepts ω → Z.RouteDominated (M ω).n (fixed ω)) →
    prob (fun ω => accepts ω ∧ fixed ω < (1 - dbar - abar) * Z.Nw) ≤ δ →
    prob (fun ω => accepts ω ∧ (M ω).time / Z.credit Fl <
      1 - Z.epsSkip Fl lam - Z.epsCert Fl lam - Z.epsRoute Fl lam - Z.epsMand Fl lam - Z.epsLuck Fl lam dbar -
        Z.epsAdapt Fl lam abar) ≤ δ ∧
    (Z.FloorKnown Fl → (∀ k, 0 ≤ Fl.ns k) → 0 ≤ dbar → 0 ≤ abar →
      0 ≤ Z.epsSkip Fl lam ∧ 0 ≤ Z.epsCert Fl lam ∧ 0 ≤ Z.epsRoute Fl lam ∧ 0 ≤ Z.epsMand Fl lam ∧
        0 ≤ Z.epsLuck Fl lam dbar ∧ 0 ≤ Z.epsAdapt Fl lam abar) ∧
    H.time = Z.gammaFloor H Fl * Z.credit Fl

def BindingFacetD1 : Prop :=
  ∀ {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U] [Fintype F] [Fintype X]
    [DecidableEq R] (Z : Model K R T L U F X) (H : Honest K) (r : R),
    Z.G.load H.prog.n r = Z.bnd H.prog → 0 < Z.bnd H.prog →
    Z.wStar (onFacet r) = Z.G.price (onFacet r) H.prog.nw / Z.Nw →
    Z.phiStar (onFacet r) = Z.G.price (onFacet r) H.prog.nf / Z.Nw →
    Z.chiStar (onFacet r) = Z.G.price (onFacet r) H.prog.nx →
    Z.epsMachine H.prog (onFacet r) = 0

def VectorThresholdD1 : Prop :=
  ∀ {K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U] [Fintype F] [Fintype X]
    (Z : Model K R T L U F X) (t : T) (phiDel : ℝ) (r : R), 0 < Z.vecMin r → 0 < Z.omega t →
    (Z.tensorDemandDel t phiDel r ≤ Z.demand .vector r ↔
      32 * (Z.G.a r (Z.cls t) + Z.vecMin r * phiDel) / (Z.vecMin r * Z.omega t) ≤ Z.q)

def RoutesOffD2 : Prop :=
  ∀ {K R ι : Type} [Fintype K] [Fintype R] [Fintype ι] (G : Machine K R) (n : K → ℝ) (rs : R) (cap : ι → R)
    (N s κ : ι → ℝ) (B₀ : ℝ),
    (∀ u, 0 ≤ N u) → (∀ u, 0 ≤ s u) → (∀ u, 0 ≤ κ u) →
    B₀ - ∑ u, N u * s u ≤ G.load n rs → (∀ u, N u ≤ κ u * G.load n (cap u)) →
    B₀ / (1 + ∑ u, κ u * s u) ≤ G.bound n

def TensorSlotTight : Prop :=
  ∀ ρH ρT dH L : ℝ, 0 < ρH → ρH ≤ ρT → 0 < dH → dH ≤ L →
    (1 - ρH / ρT * (dH / L) = 0 ↔ L = dH ∧ ρH = ρT)

def ProofsSoundOfPinned : Prop :=
  ∀ {K T L U V : Type} [Fintype K] [Fintype T] [Fintype L] [Fintype U] (Ω : Type) [Fintype Ω] (S nπ : ℕ)
    (P : Play Ω S K T L U) (pf : ProofLayer Ω S nπ V) (Nw δbind δzk : ℝ),
    pf.Of P → pf.BindingAt δbind → pf.SoundAt δzk → pf.Pins → RightFixed P pf Nw →
    ProofsSound P Nw δbind δzk nπ

def BindingOfFork : Prop :=
  ∀ {V : Type} (Ω : Type) [Fintype Ω] (S nπ : ℕ) (pf : ProofLayer Ω S nπ V) (w : Fin S),
    prob (fun x => pf.Moved x w) ^ 2 ≤ prob (fun p : Paths Ω S × Paths Ω S => pf.Forked w p.1 p.2)

def SampledRightOff : Prop :=
  ∀ {J Q Pos V : Type} [Nonempty V] (C B : Type) [Fintype C] [Fintype B] (S m s : ℕ)
    (pf : SampledLayer C B S m s J Q Pos V) (δbind δks : ℝ),
    pf.BindingAt δbind → pf.JointSound δks → pf.Pins →
    ∃ E : Paths (Paths C m × B) S → Prop, prob E ≤ S * (δbind + δks) ∧
      ∀ x, ¬ E x → ∀ w i, pf.accepted x w i → pf.right x w (pf.drawnAt x w i) (pf.view x w (pf.drawnAt x w i))

def ProofsSoundOfSampled : Prop :=
  ∀ {K T L U J Q Pos V : Type} [Fintype K] [Fintype T] [Fintype L] [Fintype U] [Fintype J] [Nonempty V]
    (C B : Type) [Fintype C] [Fintype B] (S m s : ℕ) (P : Play (Paths C m × B) S K T L U)
    (pf : SampledLayer C B S m s J Q Pos V) (mode : Paths (Paths C m × B) S → Fin S × J → Mode)
    (wt : Fin S → J → ℝ) (folded : Paths (Paths C m × B) S → ℝ) (Nw δbind δks : ℝ),
    0 < S → ∑ w, ∑ j, wt w j = S * Nw → pf.WriterReads → pf.Of P → pf.BindingAt δbind →
    pf.JointSound δks → pf.Pins → pf.Covers P mode wt Nw → UnitFixed P pf mode wt folded →
      ProofsSound P Nw δbind δks 1

def SampledBindingOfFork : Prop :=
  ∀ {J Q Pos V : Type} [Nonempty V] (C B : Type) [Fintype C] [Fintype B] (S m s : ℕ)
    (pf : SampledLayer C B S m s J Q Pos V) (w : Fin S) (L : Finset Pos), (∀ j k, pf.pos w j k ∈ L) →
    prob (fun x => pf.MovedIn x w) ^ 2 ≤
      (L.card : ℝ) ^ 2 *
        prob (fun p : Paths (Paths C m × B) S × Paths (Paths C m × B) S => ∃ ℓ ∈ L, pf.Forked ℓ p.1 p.2)

def SeesOfInjective : Prop :=
  ∀ {ι W : Type} (D : Dataflow ι W) (O : Finset (ι ⊕ Fin D.n)) (f : (ι → W) → O → W),
    (∀ x v, D.Runs x v → ∀ o : O, v o = f x o) →
    (∀ j x x', (∀ j', j' ≠ j → x j' = x' j') → x j ≠ x' j → f x ≠ f x') → D.Sees O

def ThinFoldInjective : Prop :=
  ∀ (xor : Bool) (s : BitVec 32) (pre post : List (BitVec 32 × BitVec 32)) (w₁ w₁' w₂ w₂' : BitVec 32),
    (w₁ ≠ w₁' → thinFold xor s (pre ++ (w₁, w₂) :: post) ≠ thinFold xor s (pre ++ (w₁', w₂) :: post)) ∧
    (w₂ ≠ w₂' → thinFold xor s (pre ++ (w₁, w₂) :: post) ≠ thinFold xor s (pre ++ (w₁, w₂') :: post)) ∧
    ∀ s', s ≠ s' → thinFold xor s (pre ++ post) ≠ thinFold xor s' (pre ++ post)

end ResourceAccounting.Guarantees.TS
