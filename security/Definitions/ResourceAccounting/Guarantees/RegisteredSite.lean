module

public import Definitions.Registration.Physical
public import Definitions.ResourceAccounting.Records
public import Definitions.Core.SiteSubtree
public import Mathlib.Data.String.Basic

/-! A subtree's limits read off the registration, in the actual-circuit model; none is proved here. -/

@[expose] public section

namespace ResourceAccounting

/-- **An actual circuit's image under a map of leaves**: the same circuit at the same ticks, each gate at the leaf
`f` sends its own to. Each gate takes a cell of its own: two leaves that `f` sends to one may each hold a gate in the
same cell at the same tick, and a cell takes one gate a tick. Wires, widths, ticks, and so `egress` and `Links`, read
no cell. -/
def Placed.onto {L L' : Type*} (A : Placed L) (f : L → L') : Placed L' :=
  { A.toCircuit with
    x := fun g => f (A.x g)
    t := A.t
    cell := fun g => g.val
    cell_inj := fun g h e => by
      simp only [Prod.mk.injEq] at e
      exact Fin.ext e.2.1 }

/-- A host with everything in it is one leaf of the model, so a run with a gate at a device or at a memory leaf is not
placed so. -/
def AtLinkEnds (S : Verity.Site) {Ω : Type*} (R : Ω → Actual.Run S.Place) : Prop :=
  ∀ r g, S.linkPlace ((R r).A.x g) = (R r).A.x g

end ResourceAccounting

namespace ResourceAccounting.Guarantees

/-- **A subtree's egress limit, read off the registered links.** For a Site `S` and the subtree at `i`
(`Site.under`), with the Site's places as the model's leaves and its links as the model's table, and an actual
circuit `A` placed on the Site's places, a device or a memory leaf as much as a host:

* **The links it sums have registered rates** (`Site.RatedOn`): each link with exactly one end in the subtree is
  registered with a rate, `β e`. A subtree with a link across its boundary that has no rate has no limit stated here:
  the Site registers no capacity for that link. The registration's check does not discharge this: it asks a rate only
  of a link with no `unmediated` (stated beside `Properties.Core.Site`, not proved), the network check's `registered`
  holds only the one certifier link it is named to a rate (`Properties.Certifier.Registered.RegisteredLinkRated`), and
  an unmediated link may be unrated, onsite a boundary link among them (it may be unmediated instead of certified),
  and then no limit is stated across it.
* **The wires run on the registered links** (`Placed.Links` at the Site's links and `β`, the one physical premise). It is
  taken of `A`'s image under `Site.linkPlace`, which sends each device and each memory leaf to its host (the Site's
  own `host?`) and every other place to itself, since a registered link ends at a host, a certifier, a challenger or
  the outside: a wire between two places of one host needs no link, and every other wire between two places runs on
  a registered link between the places a link may end at for them, within the link's capacity.
* **The subtree holds each host whole** (`Site.HostsWhole`), so that what leaves it crosses a link. A subtree at a
  device or a memory leaf does not, and the statement says nothing of it.
* **The bound.** The subtree's egress during `[a, b)` (`Placed.egress`, of `A` as placed: the width of its gates that
  a gate outside it reads in those ticks) is at most `β (b − a)`, with `β` the summed capacity of the registered
  links with exactly one end in the subtree (`linkCapacity`). A link between two places of the subtree is not
  counted. On a Site with no registered link the bound is stated of the empty circuit only.

The limit is read off the Site: it has no parameter but the Site and the place. Where a network attester stands in
front of a host without splitting its interface, the registration holds the host's own links to the outside and,
beside them, the host's link to the certifier and the certifier's to the outside, each at the interfaces' rates
summed. The limit of that host's subtree, and of its site's, then counts the interfaces' rates twice, and the bound
is loose there by up to a factor of two. -/
def EgressRegistered : Prop :=
  ∀ (S : Verity.Site) (i : String) (A : Placed S.Place) (link : Fin A.N → Fin A.N → Fin S.links.length)
    (β : Fin S.links.length → ℕ),
    S.RatedOn β ↑(boundary S.placeEnds (S.placesIn (S.under i))) →
    (A.onto S.linkPlace).Links S.placeEnds β link → S.HostsWhole i → ∀ a b : ℕ,
      A.egress (S.placesIn (S.under i)) (Set.Ico a b) ≤
        linkCapacity S.placeEnds β (S.placesIn (S.under i)) * (b - a)

/-- **The leak limit as a sum over the trusted devices.** In the model: the certifiers `cert`; the memory challengers
`chal` that are trusted devices, apart from the certifiers; the location `X`; and the sinks `sink`, apart from `X`,
sealed: the links with exactly one end among the sinks and no end at a certifier, at a challenger of `chal` or in `X`
have no capacity (`Covered` at the sinks). Under `B4`'s premises at the trusted devices `cert ∪ chal` and the location
`X ∪ sink`, with each certifier's record over `[a, b)` within `K` and each challenger's of `chal` within `c`
(`RecordsWithin`), what the outside of `X`, the sinks and the trusted devices does during `[a, b)` takes at most
`2 ^ (∑ K + jitter · buckets · #cert + β° (b − a) + challengerTerm)` values, with `β°` the capacity of the links across
`X`'s boundary with no certifier, challenger or sink at an end (`uncertifiedLinkCapacity`).

A sink adds nothing, and nothing is asked of what it does, only of its links. What the sinks do is not on the left
side: this says nothing of what a sink learns. A challenger of `chal` is asked what a certifier is (`TrustedUnits`),
and its whole cost is the last summand (`challengerTerm`), which is `0` when `chal` is empty. `K` and `c` are
parameters: this gives no device's record limit a value. -/
def LeakParts : Prop :=
  ∀ {L E Ω : Type*} [LinearOrder L] [Fintype L] [Fintype E] (S : Ω → Actual.Run L) (ℓ : L → L → ℕ)
    (ends : E → L × L) (β : E → ℕ) (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → E)
    (𝒱 cert chal X sink : Finset L) (spec : ∀ r, (S r).A.Spec) (record : L → Set ℕ) (bucket : ℕ → ℕ)
    (jitter a b : ℕ) (K c : L → ℕ),
    CausalStrategy S ℓ → DeclaredLinks S ends β link → Disjoint cert chal →
    TrustedUnits S 𝒱 (cert ∪ chal) spec record bucket jitter → Disjoint X sink →
    Secrets S (cert ∪ chal) (X ∪ sink) a → Covered ends β (cert ∪ chal ∪ X) sink →
    RecordsWithin S cert record a b K → RecordsWithin S chal record a b c →
    (Set.range fun r => (S r).acts (X ∪ sink ∪ (cert ∪ chal))ᶜ (Set.Ico a b)).encard ≤
      ((2 ^ ((∑ y ∈ cert, K y) + jitter * ((Finset.Ico a b).image bucket).card * cert.card +
        uncertifiedLinkCapacity ends β (cert ∪ chal ∪ sink) X * (b - a) +
        challengerTerm chal c jitter ((Finset.Ico a b).image bucket).card) : ℕ) : ℕ∞)

/-- A subtree with an unrated link that `β°` sums has no leak limit stated here, and the registration's check does not
discharge the rates. A run with a gate at a device or at a memory leaf is outside this statement, and it gives no
device's record limit a value. -/
def LeakPartsRegistered : Prop :=
  ∀ (D : Verity.Registration.Digests) (A : Verity.Registration.Auditor) (c : Verity.Registration.Composite)
    (l : Verity.Registration),
    let S := Verity.Site.ofLog l
    ∀ {Ω : Type*} (pinned units : Finset String) (i : String) (R : Ω → Actual.Run S.Place)
      (ℓ : S.Place → S.Place → ℕ) (link : ∀ r, Fin (R r).A.N → Fin (R r).A.N → Fin S.links.length)
      (β : Fin S.links.length → ℕ) (𝒱 : Finset S.Place) (spec : ∀ r, (R r).A.Spec) (record : S.Place → Set ℕ) (bucket : ℕ → ℕ)
      (jitter a b : ℕ) (K : S.Place → ℕ),
      Verity.Registration.Valid D A c l → AtLinkEnds S R →
      units ⊆ S.challengersOf pinned i → S.sharedOf pinned i ⊆ units →
      S.RatedOn β ↑((boundary S.placeEnds (S.placesIn (S.proverUnder i))).filter fun e =>
        (S.placeEnds e).1 ∉ S.placesIn (S.auditorsOf pinned i) ∧
          (S.placeEnds e).2 ∉ S.placesIn (S.auditorsOf pinned i)) →
      CausalStrategy R ℓ → DeclaredLinks R S.placeEnds β link →
      TrustedUnits R 𝒱 (S.placesIn (S.certifiersOf i ∪ units)) spec record bucket jitter →
      Secrets R (S.placesIn (S.certifiersOf i ∪ units))
        (S.placesIn (S.proverUnder i ∪ (S.challengersOf pinned i \ units))) a →
      RecordsWithin R (S.placesIn (S.certifiersOf i ∪ units)) record a b K →
      (Set.range fun r => (R r).acts (S.placesIn (S.proverUnder i ∪ (S.challengersOf pinned i \ units)) ∪
          S.placesIn (S.certifiersOf i ∪ units))ᶜ (Set.Ico a b)).encard ≤
        ((2 ^ ((∑ y ∈ S.placesIn (S.certifiersOf i ∪ units), K y) +
          jitter * ((Finset.Ico a b).image bucket).card * (S.placesIn (S.certifiersOf i ∪ units)).card +
          uncertifiedLinkCapacity S.placeEnds β (S.placesIn (S.auditorsOf pinned i))
            (S.placesIn (S.proverUnder i)) * (b - a)) : ℕ) : ℕ∞)

end ResourceAccounting.Guarantees
