module

public import Definitions.Accounting.Physical
public import Definitions.Accounting.Records
public import Definitions.Core.SiteSubtree
public import Mathlib.Data.String.Basic

/-!
# A subtree's limits, read off the registration (trusted)

Stated in the actual-circuit model, the first and third at the Site's own places and links; not proved here.

* `Placed.onto`: an actual circuit's image under a map of leaves, as `Annotation.map` is an annotation's.
* `EgressRegistered`: what leaves a subtree of the Site over any ticks is at most the capacity of the registered links
  with exactly one end in the subtree, a tick. The limit is the model's `β_X` (`linkCapacity`) at the Site's links
  (`Site.placeEnds`, `Site.linkβ`) and the subtree's places (`Site.under`), so the boundary has one definition.
* `LeakParts`: in the model, what the outside of a location can be seen to do over a window is at most 2 to the power
  of the sum of its trusted devices' record limits, a jitter share for each, and the window's length times the capacity
  of the links no such device is on. A memory challenger is a sealed sink, which adds nothing, or a trusted device,
  which adds its record limit and a jitter share.
* `LeakPartsRegistered`: the same at a subtree of a valid registration's Site, where the sealing of the sinks is
  concluded from the registration.
-/

@[expose] public section

namespace Accounting

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

end Accounting

namespace Accounting.Guarantees

/-- **A subtree's egress limit, read off the registered links.** For a Site `S` and the subtree at `i`
(`Site.under`), with the Site's places as the model's leaves and its links as the model's table, and an actual
circuit `A` placed on the Site's places, a device or a memory leaf as much as a host:

* **The wires run on the registered links** (`Placed.Links` at the Site's links, the one physical premise). It is
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

The limit is read off the Site: it has no parameter but the Site and the place. Where a network certifier stands in
front of a host without splitting its interface, the registration holds the host's own links to the outside and,
beside them, the host's link to the certifier and the certifier's to the outside, each at the interfaces' rates
summed. The limit of that host's subtree, and of its site's, then counts the interfaces' rates twice, and the bound
is loose there by up to a factor of two. -/
def EgressRegistered : Prop :=
  ∀ (S : Verity.Site) (i : String) (A : Placed S.Place) (link : Fin A.N → Fin A.N → Fin S.links.length),
    (A.onto S.linkPlace).Links S.placeEnds S.linkβ link → S.HostsWhole i → ∀ a b : ℕ,
      A.egress (S.placesIn (S.under i)) (Set.Ico a b) ≤
        linkCapacity S.placeEnds S.linkβ (S.placesIn (S.under i)) * (b - a)

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
