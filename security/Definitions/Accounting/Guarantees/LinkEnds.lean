module

public import Definitions.Accounting.Guarantees.Bridges
public import Definitions.Accounting.Guarantees.RegisteredSite

/-!
# The links premise at the places a link may end at (trusted)

Stated in the actual-circuit model; not proved here.

A strategy's gates sit at any leaf, a device or a memory leaf as much as a host, while a link ends only at some of them.
`f` sends each leaf to the leaf a link may end at for it (`Site.linkPlace` at a Site): a link delivers to that leaf,
which routes inside, and the route is part of what the link delivers.

* `Actual.Run.signalVia`: what link `e` delivers at the link end `y` at a tick, each source with the leaf that reads it.
* `AlphabetVia`, `DeclaredLinksVia`: `Alphabet` and `DeclaredLinks` with each wire taken between its two ends' link
  ends. At `f = id` the reader's leaf is `y` itself, and they are `Alphabet` and `DeclaredLinks` read again.
* `B4Via`: `B4` under `DeclaredLinksVia`, for a location and certifiers that hold each leaf with its link end. At
  `f = id` its premises are `B4`'s and its conclusion is `B4`'s [argued, not proved].

`LeakPartsRegistered` takes `AtLinkEnds` and `DeclaredLinks` today, so a run with a gate at a device is outside it; at
`DeclaredLinksVia S.linkPlace` it would need no `AtLinkEnds`, once `B4Via` is proved.
-/

@[expose] public section

namespace Accounting

/-- **The signal at a link end**: the sources, as seen, of the wires on `e` whose reader's link end is `y`, read at
tick `u`, each with the leaf that reads it. A wire within one link end is on no link. -/
def Actual.Run.signalVia {L E : Type*} (R : Actual.Run L) (f : L → L) (link : Fin R.A.N → Fin R.A.N → E) (e : E)
    (y : L) (u : ℕ) : Set (L × (ℕ × L × ℕ × ℕ × ℕ)) :=
  {p | ∃ g a, a ∈ R.A.reads g ∧ f (R.A.x a) ≠ f (R.A.x g) ∧ link g a = e ∧ f (R.A.x g) = y ∧ R.A.t g = u ∧
    p = (R.A.x g, R.seen a)}

/-- **A link is an alphabet at its ends**, across the runs of `S`: each way, at each tick, link `e` delivers one of at
most `2^{β e}` signals at a link end (`signalVia`), the leaf each source is read at included. -/
def AlphabetVia {Ω L E : Type*} (S : Ω → Actual.Run L) (f : L → L) (β : E → ℕ)
    (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → E) : Prop :=
  ∀ e y u, (Set.range fun r => (S r).signalVia f (link r) e y u).encard ≤ ((2 ^ β e : ℕ) : ℕ∞)

/-- **The links as declared, at the places a link may end at**: across the strategy each link delivers one of at most
`2^{β e}` signals a tick each way at its ends (`AlphabetVia`), and in every run each wire between two link ends has a
computed source and runs on a declared link joining them, within its capacity (`Placed.Links` of the run's image under
`f`). One physical premise, that nothing crosses between the places a link may end at but the declared links, at their
capacities, a device's route inside its host part of what the link carries. -/
def DeclaredLinksVia {Ω L E : Type*} (S : Ω → Actual.Run L) (f : L → L) (ends : E → L × L) (β : E → ℕ)
    (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → E) : Prop :=
  AlphabetVia S f β link ∧ ∀ r, ((S r).A.onto f).Links ends β (link r)

/-- **B4 at the places a link may end at.** `B4`'s premises and conclusion, with `DeclaredLinksVia` at `f` for
`DeclaredLinks`, for certifiers `cert` and a location `X` that each hold a leaf exactly when they hold its link end:
what the outside `(X ∪ cert)ᶜ` does during `I = [a, b)` takes at most (the number of records placed in `I`) ×
`2^{res_out}` values, `res_out = jitter · (buckets of I) · #cert + β°_X (b − a)`. The strategy's gates may sit at
any leaf; only the links premise reads `f`. -/
def Guarantees.B4Via : Prop :=
  ∀ {L E Ω : Type*} [LinearOrder L] [Fintype L] [Fintype E] (S : Ω → Actual.Run L) (ℓ : L → L → ℕ) (f : L → L)
    (ends : E → L × L) (β : E → ℕ) (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → E) (𝒱 cert X : Finset L)
    (spec : ∀ r, (S r).A.Spec) (record : L → Set ℕ) (bucket : ℕ → ℕ) (jitter a b : ℕ),
    (∀ y, y ∈ cert ↔ f y ∈ cert) → (∀ y, y ∈ X ↔ f y ∈ X) →
    CausalStrategy S ℓ → DeclaredLinksVia S f ends β link →
    TrustedUnits S 𝒱 cert spec record bucket jitter → Secrets S cert X a →
    (Set.range fun r => (S r).acts (X ∪ cert)ᶜ (Set.Ico a b)).encard ≤
      (Set.range fun r =>
        (S r).seen '' ↑(((S r).record cert record).filter fun g => a ≤ (S r).A.t g ∧ (S r).A.t g < b)).encard *
        ((2 ^ (jitter * ((Finset.Ico a b).image bucket).card * cert.card +
          uncertifiedCapacity ends β cert X * (b - a)) : ℕ) : ℕ∞)

end Accounting
