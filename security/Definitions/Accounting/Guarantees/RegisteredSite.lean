module

public import Definitions.Accounting.Physical
public import Definitions.Accounting.Records
public import Definitions.Core.SiteSubtree
public import Mathlib.Data.String.Basic

/-!
# A subtree's limits, read off the registration (trusted)

Stated in the actual-circuit model, the first and third at the Site's own places and links; not proved here.

* `EgressRegistered`: what leaves a subtree of the Site over any ticks is at most the capacity of the registered links
  with exactly one end in the subtree, a tick; and that limit is at most the limit of the subtree's root alone plus
  its children's subtrees' limits. The limit is the model's `β_X` (`linkCapacity`) at the Site's links
  (`Site.placeEnds`, `Site.linkβ`) and the subtree's places (`Site.under`), so the boundary has one definition.
* `LeakParts`: in the model, what the outside of a location can be seen to do over a window is at most 2 to the power
  of the sum of its trusted devices' record limits, a jitter share for each, and the window's length times the capacity
  of the links no such device is on. A memory challenger is a sealed sink, which adds nothing, or a trusted device,
  which adds its record limit and a jitter share.
* `LeakPartsRegistered`: the same at a subtree of a valid registration's Site, where the sealing of the sinks is
  concluded from the registration.
-/

@[expose] public section

namespace Accounting.Guarantees

/-- **A subtree's egress limit, read off the registered links.** For a Site `S` and the subtree at `i`
(`Site.under`), with the Site's places as the model's leaves and its links as the model's table:

* **The bound.** For an actual circuit placed on the Site's places whose every wire between two places runs on a
  registered link within its capacity (`Placed.Links` at the Site's links, the one physical premise), the subtree's
  egress during `[a, b)` (`Placed.egress`: the width of its gates that a gate outside it reads in those ticks) is at
  most `β (b − a)`, with `β` the summed capacity of the registered links with exactly one end in the subtree
  (`linkCapacity`). A link between two places of the subtree is not counted. On a Site with no registered link the
  premise holds only of a circuit with no wire between places.
* **Limits add over parts, at most.** For a place `i`, the subtree's limit is at most the limit of `i` alone plus the
  limits of the subtrees of the locations registered directly under `i` (`Site.Child`). It is an inequality: a link
  between two of those parts is counted on the right and not on the left.

The limit is read off the Site: it has no parameter but the Site and the place. -/
def EgressRegistered : Prop :=
  ∀ (S : Verity.Site) (i : String),
    (∀ (A : Placed S.Place) (link : Fin A.N → Fin A.N → Fin S.links.length),
      A.Links S.placeEnds S.linkβ link → ∀ a b : ℕ,
        A.egress (S.placesIn (S.under i)) (Set.Ico a b) ≤
          linkCapacity S.placeEnds S.linkβ (S.placesIn (S.under i)) * (b - a)) ∧
    (i ∈ S.places →
      linkCapacity S.placeEnds S.linkβ (S.placesIn (S.under i)) ≤
        linkCapacity S.placeEnds S.linkβ (S.placesIn {i}) +
          ∑ c ∈ S.places.filter (S.Child i), linkCapacity S.placeEnds S.linkβ (S.placesIn (S.under c)))

/-- **The leak limit as a sum over the trusted devices.** In the model: the certifiers `cert`; the memory challengers
`chal` that are trusted devices, apart from the certifiers; the location `X`; and the sinks `sink`, apart from `X`,
sealed: the links with exactly one end among the sinks and no end at a certifier, at a challenger of `chal` or in `X`
have no capacity (`Covered` at the sinks). Under `B4`'s premises at the trusted devices `cert ∪ chal` and the location
`X ∪ sink`, with each certifier's record over `[a, b)` within `K` and each challenger's of `chal` within `c`
(`RecordsWithin`), what the outside of `X`, the sinks and the trusted devices does during `[a, b)` takes at most
`2 ^ (∑ K + jitter · buckets · #cert + β° (b − a) + challengerTerm)` values, with `β°` the capacity of the links across
`X`'s boundary with no certifier, challenger or sink at an end (`uncertifiedCapacity`).

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
        uncertifiedCapacity ends β (cert ∪ chal ∪ sink) X * (b - a) +
        challengerTerm chal c jitter ((Finset.Ico a b).image bucket).card) : ℕ) : ℕ∞)

/-- **The leak limit at a subtree of a valid registration's Site.** For a valid registration log `l`
(`Registration.Valid`) with Site `S`, the subtree at `i`, the pins `pinned` and a set `units` of the subtree's memory
challengers (`Site.challengersOf`) that holds every shared one (`Site.sharedOf`): the trusted devices are the
subtree's certifiers with `units`; the location is the subtree's prover places with the other challengers, each a
sink. Under `B4`'s premises there, with each trusted device's record over `[a, b)` within `K` (`RecordsWithin`), what
the outside does during `[a, b)` takes at most `2 ^ (∑ K + jitter · buckets · # + β° (b − a))` values, the sum and the
count over the certifiers and `units`, with `β°` the capacity of the registered links with exactly one end among the
subtree's prover places and none of the subtree's auditor's devices at either end (`uncertifiedCapacity` at the
Site's links).

That the sinks are sealed is no premise: a valid log's links join kinds a link may join and end at registered
places, so a challenger outside `units`, linked only into the subtree, is linked only to the subtree's hosts. A
challenger outside `units` adds nothing and nothing is asked of what it does; one of `units` adds its record limit and
a jitter share. `pinned`, `units` above the shared challengers, and `K` are parameters: this gives no device's record
limit a value, and reads no protocol's check or output. -/
def LeakPartsRegistered : Prop :=
  ∀ (D : Verity.Registration.Digests) (A : Verity.Registration.Auditor) (c : Verity.Registration.Composite)
    (l : Verity.Registration),
    let S := Verity.Site.ofLog l
    ∀ {Ω : Type*} (pinned units : Finset String) (i : String) (R : Ω → Actual.Run S.Place)
      (ℓ : S.Place → S.Place → ℕ) (link : ∀ r, Fin (R r).A.N → Fin (R r).A.N → Fin S.links.length)
      (𝒱 : Finset S.Place) (spec : ∀ r, (R r).A.Spec) (record : S.Place → Set ℕ) (bucket : ℕ → ℕ)
      (jitter a b : ℕ) (K : S.Place → ℕ),
      Verity.Registration.Valid D A c l →
      units ⊆ S.challengersOf pinned i → S.sharedOf pinned i ⊆ units →
      CausalStrategy R ℓ → DeclaredLinks R S.placeEnds S.linkβ link →
      TrustedUnits R 𝒱 (S.placesIn (S.certifiersOf i ∪ units)) spec record bucket jitter →
      Secrets R (S.placesIn (S.certifiersOf i ∪ units))
        (S.placesIn (S.proverUnder i ∪ (S.challengersOf pinned i \ units))) a →
      RecordsWithin R (S.placesIn (S.certifiersOf i ∪ units)) record a b K →
      (Set.range fun r => (R r).acts (S.placesIn (S.proverUnder i ∪ (S.challengersOf pinned i \ units)) ∪
          S.placesIn (S.certifiersOf i ∪ units))ᶜ (Set.Ico a b)).encard ≤
        ((2 ^ ((∑ y ∈ S.placesIn (S.certifiersOf i ∪ units), K y) +
          jitter * ((Finset.Ico a b).image bucket).card * (S.placesIn (S.certifiersOf i ∪ units)).card +
          uncertifiedCapacity S.placeEnds S.linkβ (S.placesIn (S.auditorsOf pinned i))
            (S.placesIn (S.proverUnder i)) * (b - a)) : ℕ) : ℕ∞)

end Accounting.Guarantees
