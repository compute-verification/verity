module

public import Definitions.Pouw.Guarantees.Bridges
public import Definitions.ResourceAccounting.Guarantees.RegisteredSite

/-!
# The links premise at the places a link may end at (trusted)

Stated in the actual-circuit model; `B4Via` is not proved here.

A strategy's gates sit at any leaf, a device or a memory leaf as much as a host, while a registered link ends only at
a host, a certifier, a challenger or the outside. A map of leaves `f` sends each leaf to the leaf a link may end at for
it. At a Site that map is the Site's own, `Site.linkPlace` (each device and memory leaf to its host, read through the
Site's `host?`), and the registration fixes it, with the links' ends and capacities, before any coin: no run, and so
no prover, chooses where its signals count.

**A link's alphabet is counted once at the host, over every leaf `f` sends there.** What crosses a link toward a host
is charged whichever device inside reads it, and a signal between two leaves of one host crosses no link and is
charged nothing, which is right only because the host is inside the prover's boundary. The host's routing is the
prover's, and nothing here trusts it: a location and the certifiers are asked to hold each leaf with its host.

* `Actual.Run.signalVia`: what link `e` delivers at the link end `y` at a tick, each source with the leaf that reads it.
* `AlphabetVia`, `DeclaredLinksVia`: `Alphabet` and `DeclaredLinks` with each wire taken between its two ends' link
  ends under `f`.
* `Guarantees.B4Under`: `B4` on a fixed table of leaves, links and link ends `f`, for every strategy on it.
* `Guarantees.B4Via`: `B4Under` at every Site's own places, links and `Site.linkPlace`. It quantifies over no map of
  leaves: `f` is the Site's.
* `Guarantees.B4UnderId`: `B4Under` at `f = id` over every table is `B4`.

`LeakPartsRegistered` takes `AtLinkEnds` and `DeclaredLinks` today, so a run with a gate at a device is outside it; it
would take `DeclaredLinksVia` at `Site.linkPlace` and no `AtLinkEnds` once `B4Via` is proved.
-/

@[expose] public section

universe u v w

namespace ResourceAccounting

/-- **The signal at a link end**: the sources, as seen, of the wires on `e` whose reader's link end under `f` is `y`,
read at tick `u`, each with the leaf that reads it. A wire within one link end is on no link. -/
def Actual.Run.signalVia {L E : Type*} (R : Actual.Run L) (f : L → L) (link : Fin R.A.N → Fin R.A.N → E) (e : E)
    (y : L) (u : ℕ) : Set (L × (ℕ × L × ℕ × ℕ × ℕ)) :=
  {p | ∃ g a, a ∈ R.A.reads g ∧ f (R.A.x a) ≠ f (R.A.x g) ∧ link g a = e ∧ f (R.A.x g) = y ∧ R.A.t g = u ∧
    p = (R.A.x g, R.seen a)}

/-- **A link is an alphabet at its ends**, across the runs of `S`: each way, at each tick, link `e` delivers one of at
most `2^{β e}` signals at a link end (`signalVia`). It is counted once at the link end, over every leaf `f` sends
there, the leaf each source is read at included: what crosses toward a host is charged whichever leaf inside reads
it. -/
def AlphabetVia {Ω L E : Type*} (S : Ω → Actual.Run L) (f : L → L) (β : E → ℕ)
    (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → E) : Prop :=
  ∀ e y u, (Set.range fun r => (S r).signalVia f (link r) e y u).encard ≤ ((2 ^ β e : ℕ) : ℕ∞)

/-- **The links as declared, at the places a link may end at**: across the strategy each link delivers one of at most
`2^{β e}` signals a tick each way at its ends (`AlphabetVia`), and in every run each wire between two link ends has a
computed source and runs on a declared link joining them, within its capacity (`Placed.Links` of the run's image under
`f`). One physical premise, that nothing crosses between the places a link may end at but the declared links, at their
capacities. A wire between two leaves `f` sends to one place crosses no link and is charged nothing. -/
def DeclaredLinksVia {Ω L E : Type*} (S : Ω → Actual.Run L) (f : L → L) (ends : E → L × L) (β : E → ℕ)
    (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → E) : Prop :=
  AlphabetVia S f β link ∧ ∀ r, ((S r).A.onto f).Links ends β (link r)

end ResourceAccounting

namespace ResourceAccounting.Guarantees

/-- **B4 on a fixed table**: the leaves `L`, the links' ends `ends` and capacities `β`, and the link ends `f`, all fixed
before any strategy. `B4`'s premises and conclusion, with `DeclaredLinksVia` at `f` for `DeclaredLinks`, for every
strategy `S` on the table and certifiers `cert` and a location `X` that each hold a leaf exactly when they hold its link
end: what the outside `(X ∪ cert)ᶜ` does during `I = [a, b)` takes at most (the number of records placed in `I`) ×
`2^{res_out}` values, `res_out = jitter · (buckets of I) · #cert + β°_X (b − a)`. The strategy's gates may sit at any
leaf; only the links premise reads `f`, and no run chooses it. -/
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

/-- **B4 at the places a link may end at, on a registered Site.** For every Site `T`, `B4Under` with the Site's places
as the leaves, its registered links as the table (`Site.placeEnds`, and the rates `β`, which are the registered ones
and exist at every link: `Site.RatedOn`, so a Site with a link that has no rate has no statement here) and its own link ends
(`Site.linkPlace`, each device and memory leaf to its host through the Site's `host?`). It quantifies over no map of
leaves: the link ends are a function of the Site, which the registration fixes before any coin, so a strategy, which
is quantified after the Site, cannot choose where its signals count.

A link's alphabet is counted once at the host, over every leaf `Site.linkPlace` sends there. What crosses a link toward
a host is charged whichever device inside reads it, and a signal between two leaves of one host crosses no link and is
charged nothing, which is right only because the host is inside the prover's boundary: the certifiers and the location
are asked to hold each leaf with its host, and the host's routing is never trusted. On a Site whose every place is its
own link end it is `B4` at the Site's places and links (`B4UnderId`).

**What the rated hypothesis costs.** `B4Under` chooses the location and the certifiers inside, so `β` is asked to be
the registered rate at every link, and of a registered Site with any unrated link this says nothing. The registration's
check does not discharge it: it asks a rate only of a link with no `unmediated` (stated beside `Properties.Core.Site`,
not proved), the network check's `registered` holds only the one certifier link it is named to a rate
(`Properties.Certifier.Registered.RegisteredLinkRated`), and an unmediated link may be unrated, and then no limit is
stated across it. -/
def B4Via : Prop :=
  ∀ (T : Verity.Site) (β : Fin T.links.length → ℕ), T.RatedOn β Set.univ →
    B4Under.{0, 0, w} T.linkPlace T.placeEnds β

/-- **`B4Under` at `f = id` is `B4`**: on every table, with each leaf its own link end, `B4Under`'s premises and
conclusion are `B4`'s. -/
def B4UnderId : Prop :=
  B4.{u, v, w} ↔ ∀ {L : Type u} {E : Type v} [LinearOrder L] [Fintype L] [Fintype E] (ends : E → L × L) (β : E → ℕ),
    B4Under.{u, v, w} (id : L → L) ends β

end ResourceAccounting.Guarantees
