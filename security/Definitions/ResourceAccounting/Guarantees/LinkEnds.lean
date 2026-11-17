module

public import Definitions.Pouw.Guarantees.Bridges
public import Definitions.ResourceAccounting.Guarantees.RegisteredSite

/-! The links premise at the places a link may end at, in the actual-circuit model; `B4Via` is not proved here. -/

@[expose] public section

universe u v w

namespace ResourceAccounting

/-- A wire within one link end is on no link. -/
def Actual.Run.signalVia {L E : Type*} (R : Actual.Run L) (f : L → L) (link : Fin R.A.N → Fin R.A.N → E) (e : E)
    (y : L) (u : ℕ) : Set (L × (ℕ × L × ℕ × ℕ × ℕ)) :=
  {p | ∃ g a, a ∈ R.A.reads g ∧ f (R.A.x a) ≠ f (R.A.x g) ∧ link g a = e ∧ f (R.A.x g) = y ∧ R.A.t g = u ∧
    p = (R.A.x g, R.seen a)}

/-- Counted once at the link end, over every leaf `f` sends there: what crosses toward a host is charged whichever leaf
inside reads it. -/
def AlphabetVia {Ω L E : Type*} (S : Ω → Actual.Run L) (f : L → L) (β : E → ℕ)
    (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → E) : Prop :=
  ∀ e y u, (Set.range fun r => (S r).signalVia f (link r) e y u).encard ≤ ((2 ^ β e : ℕ) : ℕ∞)

/-- One physical premise. A wire between two leaves `f` sends to one place crosses no link and is charged nothing. -/
def DeclaredLinksVia {Ω L E : Type*} (S : Ω → Actual.Run L) (f : L → L) (ends : E → L × L) (β : E → ℕ)
    (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → E) : Prop :=
  AlphabetVia S f β link ∧ ∀ r, ((S r).A.onto f).Links ends β (link r)

end ResourceAccounting

namespace ResourceAccounting.Guarantees

/-- Only the links premise reads `f`, and no run chooses it. -/
def B4Under {L : Type u} {E : Type v} [LinearOrder L] [Fintype L] [Fintype E] (f : L → L) (ends : E → L × L)
    (β : E → ℕ) : Prop :=
  ∀ {Ω : Type w} (S : Ω → Actual.Run L) (ℓ : L → L → ℕ) (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → E)
    (𝒱 cert X : Finset L) (spec : ∀ r, (S r).A.Spec) (record : L → Set ℕ) (bucket : ℕ → ℕ) (jitter a b : ℕ),
    (∀ y, y ∈ cert ↔ f y ∈ cert) → (∀ y, y ∈ X ↔ f y ∈ X) →
    CausalStrategy S ℓ → DeclaredLinksVia S f ends β link →
    TrustedUnits S 𝒱 cert spec record bucket jitter → Secrets S cert X a →
    (Set.range fun r => (S r).acts (X ∪ cert)ᶜ (Set.Ico a b)).encard ≤
      (Set.range fun r =>
        (S r).seen '' ↑(((S r).record cert record).filter fun g => a ≤ (S r).A.t g ∧ (S r).A.t g < b)).encard *
        ((2 ^ (jitter * ((Finset.Ico a b).image bucket).card * cert.card +
          uncertifiedLinkCapacity ends β cert X * (b - a)) : ℕ) : ℕ∞)

/-- A signal between two leaves of one host is charged nothing, which is right only because the host is inside the
prover's boundary. Of a Site with any unrated link this says nothing, and the registration's check does not
discharge the rated hypothesis. -/
def B4Via : Prop :=
  ∀ (T : Verity.Site) (β : Fin T.links.length → ℕ), T.RatedOn β Set.univ →
    B4Under.{0, 0, w} T.linkPlace T.placeEnds β

def B4UnderId : Prop :=
  B4.{u, v, w} ↔ ∀ {L : Type u} {E : Type v} [LinearOrder L] [Fintype L] [Fintype E] (ends : E → L × L) (β : E → ℕ),
    B4Under.{u, v, w} (id : L → L) ends β

end ResourceAccounting.Guarantees
