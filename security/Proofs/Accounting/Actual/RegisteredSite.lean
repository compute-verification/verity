module

public import Definitions.Accounting.Guarantees.RegisteredSite
public import Proofs.Accounting.Actual.Parts
public import Proofs.Accounting.Actual.Physical
public import Proofs.Core.Site
public import Proofs.Core.SiteSubtree
import Mathlib.Algebra.BigOperators.Fin
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Algebra.Order.BigOperators.Group.Finset

/-!
# A subtree's limits, read off the registration

* `Placed.egress_onto`: a set of leaves that a map of leaves respects has the same egress in the image.
* `Site.egress_le_registered`: `egress_le` at the image under `Site.linkPlace`, the Site's links and a subtree that
  holds each host whole, carried back to the circuit as placed.
* `EgressRegistered`: the guarantee.
* `linkCapacity_union_le`, `linkCapacity_biUnion_le`: in the model, the capacity across the boundary of a union of
  locations is at most the locations' capacities summed, since a link across the union's boundary crosses one's.
* `egressLimit_le_children`: a subtree's limit is at most its root's alone plus its root's children's subtrees'
  (`Site.placesIn_under`).
* `EgressRegistered_nonvacuous`: at the example's remote Site, a circuit with a gate at a device, read at the outside
  across the host's uplink, meets the guarantee's two premises at the host's subtree, and its egress there is above
  0.
* `Site.uncertified_registered_eq`: where every registered link ends at places, the model's `β°` at the Site's links is
  the sum of `β` over the uncovered links (`Site.uncovered`).
* `Site.leak_le_registered`: `leak_le_sum_K` at a subtree, its certifiers and a set of its memory challengers the
  trusted devices and its other challengers sinks, which the registration seals (`Site.challengers_sealed`,
  `Site.uncovered_sealed`).
* `LeakPartsRegistered`: that at a valid log's Site (`Registration.Valid.linkKinds`, `Registration.Valid.ends_known`),
  the guarantee.
-/

@[expose] public section

namespace Accounting

open Verity Finset

section Onto

variable {L L' : Type*}

/-- **A set of leaves that a map of leaves respects has the same egress in the image**: where a leaf is in `X`
exactly when `f` sends it into `X'`, what leaves `X'` in the image of `A` under `f` is what leaves `X` in `A`. -/
theorem Placed.egress_onto (A : Placed L) (f : L → L') (X : Finset L) (X' : Finset L')
    (hX : ∀ p, p ∈ X ↔ f p ∈ X') (I : Set ℕ) : (A.onto f).egress X' I = A.egress X I := by
  unfold Placed.egress
  refine congrArg A.width (Finset.filter_congr fun g _ => ?_)
  exact and_congr (hX (A.x g)).symm (exists_congr fun h => and_congr (not_congr (hX (A.x h)).symm) Iff.rfl)

end Onto

section Union

variable {L E ι : Type*} [Fintype E] [DecidableEq L]

/-- A link across the boundary of two locations' union crosses the boundary of one of them. -/
theorem linkCapacity_union_le (ends : E → L × L) (β : E → ℕ) (X Y : Finset L) :
    linkCapacity ends β (X ∪ Y) ≤ linkCapacity ends β X + linkCapacity ends β Y := by
  classical
  have hsub : boundary ends (X ∪ Y) ⊆ boundary ends X ∪ boundary ends Y := by
    intro e he
    simp only [boundary, mem_filter, mem_univ, true_and, mem_union] at he ⊢
    by_cases h1 : (ends e).1 ∈ X <;> by_cases h2 : (ends e).2 ∈ X <;>
      by_cases h3 : (ends e).1 ∈ Y <;> by_cases h4 : (ends e).2 ∈ Y <;> simp_all
  unfold linkCapacity
  exact (Finset.sum_le_sum_of_subset hsub).trans (Nat.le.intro Finset.sum_union_inter)

/-- **Limits add over a union, at most**: the capacity across the boundary of a union of locations is at most the
locations' capacities summed. -/
theorem linkCapacity_biUnion_le (ends : E → L × L) (β : E → ℕ) (P : Finset ι) (X : ι → Finset L) :
    linkCapacity ends β (P.biUnion X) ≤ ∑ p ∈ P, linkCapacity ends β (X p) := by
  classical
  induction P using Finset.induction_on with
  | empty => simp [linkCapacity, boundary]
  | insert p P hp ih =>
    rw [Finset.biUnion_insert, Finset.sum_insert hp]
    exact (linkCapacity_union_le ends β (X p) (P.biUnion X)).trans (Nat.add_le_add_left ih _)

end Union

end Accounting

namespace Verity.Site

open Finset

variable (S : Site)

/-- **A subtree's egress is at most its registered limit**: for an actual circuit on the Site's places whose wires,
each taken between the places a link may end at, run on the registered links within their capacity, what leaves a
subtree that holds each host whole over the ticks `[a, b)` is at most the capacity of the registered links with
exactly one end in it, a tick. -/
theorem egress_le_registered (i : String) (A : Accounting.Placed S.Place)
    (link : Fin A.N → Fin A.N → Fin S.links.length) (h : (A.onto S.linkPlace).Links S.placeEnds S.linkβ link)
    (hi : S.HostsWhole i) (a b : ℕ) :
    A.egress (S.placesIn (S.under i)) (Set.Ico a b) ≤
      Accounting.linkCapacity S.placeEnds S.linkβ (S.placesIn (S.under i)) * (b - a) := by
  have hX : ∀ p : S.Place, p ∈ S.placesIn (S.under i) ↔ S.linkPlace p ∈ S.placesIn (S.under i) :=
    fun p => S.mem_placesIn.trans ((hi p).trans S.mem_placesIn.symm)
  rw [← Accounting.Placed.egress_onto A S.linkPlace _ _ hX]
  exact Accounting.SecurityProofs.Actual.egress_le (A.onto S.linkPlace) h _ a b

/-- **The uncovered term is read off the registered links**: where every registered link ends at places of the Site,
the model's `β°` at the registered links, a covering set `cov` and a location `X` is the sum of `β` over the
registered links with exactly one end in `X` and neither end in `cov` (`uncovered`). -/
theorem uncertified_registered_eq (cov X : Finset String)
    (hends : ∀ k ∈ S.links, k.ends.1 ∈ S.places ∧ k.ends.2 ∈ S.places) :
    Accounting.uncertifiedCapacity S.placeEnds S.linkβ (S.placesIn cov) (S.placesIn X) =
      ((S.uncovered cov X).map (·.β)).sum := by
  classical
  have hl : ∀ (l : List Link) (p : Link → Bool),
      ((l.filter p).map (·.β)).sum = (l.map fun k => if p k = true then k.β else 0).sum := by
    intro l p
    induction l with
    | nil => rfl
    | cons k l ih =>
      rw [List.filter_cons]
      by_cases hk : p k = true
      · rw [ite_eq_left hk, List.map_cons, List.sum_cons, List.map_cons, List.sum_cons, ite_eq_left hk, ih]
      · rw [ite_eq_right hk, List.map_cons, List.sum_cons, ite_eq_right hk, ih, Nat.zero_add]
  have hp : ∀ (e : Fin S.links.length) (Y : Finset String),
      ((S.placeEnds e).1 ∈ S.placesIn Y ↔ (S.links[e.1]).ends.1 ∈ Y) ∧
        ((S.placeEnds e).2 ∈ S.placesIn Y ↔ (S.links[e.1]).ends.2 ∈ Y) := by
    intro e Y
    have h := hends (S.links[e.1]) (List.getElem_mem e.2)
    simp [placeEnds, placesIn, place, h.1, h.2]
  unfold Accounting.uncertifiedCapacity Accounting.boundary uncovered crossing
  rw [List.filter_filter, hl, Finset.sum_filter, Finset.sum_filter, ← Fin.sum_univ_fun_getElem]
  refine Finset.sum_congr rfl fun e _ => ?_
  have hX := hp e X
  have hC := hp e cov
  by_cases h : ((S.links[e.1]).ends.1 ∈ X ↔ (S.links[e.1]).ends.2 ∉ X) <;>
    by_cases h1 : (S.links[e.1]).ends.1 ∈ cov <;> by_cases h2 : (S.links[e.1]).ends.2 ∈ cov <;>
    simp [linkβ, hX, hC, h, h1, h2]

/-- **The leak limit at a subtree, each memory challenger a sealed sink or a trusted device.** At a Site whose links
join kinds a link may join and end at places, for the subtree at `i` and a set `units` of its memory challengers that
holds every shared one: the trusted devices are the subtree's certifiers with `units`, the location the subtree's
prover places with the other challengers. With `B4`'s premises there and each trusted device's record within its `K`,
what the outside does during `[a, b)` takes at most `2 ^ (∑ K + jitter · buckets · # + β° (b − a))` values, `β°` the
model's at the Site's links, the subtree's auditor's devices covering its prover places. That a challenger outside
`units` is sealed is concluded: it is not shared, so it is linked only into the subtree, and its links end at hosts. -/
theorem leak_le_registered {Ω : Type*} (hkinds : S.LinkKinds)
    (hends : ∀ k ∈ S.links, k.ends.1 ∈ S.places ∧ k.ends.2 ∈ S.places) (pinned units : Finset String) (i : String)
    (R : Ω → Accounting.Actual.Run S.Place) (ℓ : S.Place → S.Place → ℕ)
    (link : ∀ r, Fin (R r).A.N → Fin (R r).A.N → Fin S.links.length) (𝒱 : Finset S.Place) (spec : ∀ r, (R r).A.Spec)
    (record : S.Place → Set ℕ) (bucket : ℕ → ℕ) (jitter a b : ℕ) (K : S.Place → ℕ)
    (hu : units ⊆ S.challengersOf pinned i) (hsh : S.sharedOf pinned i ⊆ units)
    (hcs : Accounting.CausalStrategy R ℓ) (hdl : Accounting.DeclaredLinks R S.placeEnds S.linkβ link)
    (hT : Accounting.TrustedUnits R 𝒱 (S.placesIn (S.certifiersOf i ∪ units)) spec record bucket jitter)
    (hsec : Accounting.Secrets R (S.placesIn (S.certifiersOf i ∪ units))
      (S.placesIn (S.proverUnder i ∪ (S.challengersOf pinned i \ units))) a)
    (hK : Accounting.RecordsWithin R (S.placesIn (S.certifiersOf i ∪ units)) record a b K) :
    (Set.range fun r =>
        (R r).acts (S.placesIn (S.proverUnder i ∪ (S.challengersOf pinned i \ units)) ∪
          S.placesIn (S.certifiersOf i ∪ units))ᶜ (Set.Ico a b)).encard ≤
      ((2 ^ ((∑ y ∈ S.placesIn (S.certifiersOf i ∪ units), K y) +
        jitter * ((Finset.Ico a b).image bucket).card * (S.placesIn (S.certifiersOf i ∪ units)).card +
        Accounting.uncertifiedCapacity S.placeEnds S.linkβ (S.placesIn (S.auditorsOf pinned i))
          (S.placesIn (S.proverUnder i)) * (b - a)) : ℕ) : ℕ∞) := by
  have hin : ∀ y ∈ S.challengersOf pinned i \ units, S.Within i y := fun y hy =>
    S.within_of_not_shared (Finset.mem_sdiff.2
      ⟨(Finset.mem_sdiff.1 hy).1, fun h => (Finset.mem_sdiff.1 hy).2 (hsh h)⟩)
  have h := Accounting.leak_le_sum_K R ℓ S.placeEnds S.linkβ link 𝒱 (S.placesIn (S.certifiersOf i ∪ units))
    (S.placesIn (S.proverUnder i ∪ (S.challengersOf pinned i \ units))) spec record bucket jitter a b K hcs hdl hT
    hsec hK
  rw [S.uncertified_registered_eq _ _ hends,
    S.uncovered_sealed (S.certifiersOf i ∪ units) (S.proverUnder i) (S.challengersOf pinned i \ units)
      (S.challengers_sealed hkinds i (S.challengersOf pinned i \ units)
        fun y hy => ⟨(S.mem_challengersOf (Finset.mem_sdiff.1 hy).1).1, hin y hy⟩),
    S.units_union_sinks hu, ← S.uncertified_registered_eq _ _ hends] at h
  exact h

end Verity.Site

namespace Accounting.SecurityProofs

open Verity Finset

theorem EgressRegistered : Guarantees.EgressRegistered := by
  intro S i A link h hi a b
  exact S.egress_le_registered i A link h hi a b

/-- **Limits add over a subtree's root and its children's subtrees, at most**: a subtree's limit is at most the limit
of its root alone plus the limits of the subtrees of the locations registered directly under the root. It is an
inequality: a link between two of those is counted on the right and not on the left. -/
theorem egressLimit_le_children (S : Site) (i : String) :
    linkCapacity S.placeEnds S.linkβ (S.placesIn (S.under i)) ≤
      linkCapacity S.placeEnds S.linkβ (S.placesIn {i}) +
        ∑ c ∈ S.places.filter (S.Child i), linkCapacity S.placeEnds S.linkβ (S.placesIn (S.under c)) := by
  rw [S.placesIn_under i]
  exact (linkCapacity_union_le _ _ _ _).trans (Nat.add_le_add_left (linkCapacity_biUnion_le _ _ _ _) _)

namespace RegisteredSite

/-- The example's remote Site: one host with two devices, each with its memory, the host's own memory, and the
host's uplink to the outside. -/
abbrev site : Site := Site.ofLog [Registration.Example.remote]

/-- The host's uplink, the Site's one link. -/
def uplink : Fin site.links.length := ⟨0, by decide +kernel⟩

/-- Two gates of one bit at tick 0: a computed gate at the device `gpu0`, and a gate at the outside that reads it. -/
def wire : Placed site.Place where
  N := 2
  w := fun _ => 1
  isInput := fun _ => false
  reads := fun g => if g = 1 then {0} else ∅
  reads_lt := by decide +kernel
  input_reads := fun _ h => absurd h Bool.false_ne_true
  op := fun _ _ => 0
  op_congr := fun _ _ _ _ => rfl
  outputs := ∅
  x := fun g => if g = 0 then site.place "gpu0" else site.place Site.outside
  t := fun _ => 0
  cell := fun g => g.val
  cell_inj := fun g h e => by
    simp only [Prod.mk.injEq] at e
    exact Fin.ext e.2.1

/-- The places of the host's subtree: the host, its two devices with their memory, and its own memory. -/
def node0 : Finset String := {"node0", "gpu0", "hbm0", "gpu1", "hbm1", "dram0", "nvme0"}

theorem under_node0 : site.under "node0" = node0 := by
  refine site.under_eq_of_closed (by decide +kernel) (by decide +kernel) (by decide +kernel) ?_
  have g0 : site.Child "node0" "gpu0" := by decide +kernel
  have m0 : site.Child "gpu0" "hbm0" := by decide +kernel
  have g1 : site.Child "node0" "gpu1" := by decide +kernel
  have m1 : site.Child "gpu1" "hbm1" := by decide +kernel
  have d0 : site.Child "node0" "dram0" := by decide +kernel
  have n0 : site.Child "node0" "nvme0" := by decide +kernel
  intro j hj
  simp only [node0, mem_insert, mem_singleton] at hj
  rcases hj with rfl | rfl | rfl | rfl | rfl | rfl | rfl
  · exact Relation.ReflTransGen.refl
  · exact Relation.ReflTransGen.single g0
  · exact Relation.ReflTransGen.tail (Relation.ReflTransGen.single g0) m0
  · exact Relation.ReflTransGen.single g1
  · exact Relation.ReflTransGen.tail (Relation.ReflTransGen.single g1) m1
  · exact Relation.ReflTransGen.single d0
  · exact Relation.ReflTransGen.single n0

/-- The host's subtree holds the host whole: each of its places is sent to the host, and the site and the outside
to themselves. -/
theorem hostsWhole_node0 : site.HostsWhole "node0" := by
  unfold Site.HostsWhole
  rw [under_node0]
  decide +kernel

/-- The one wire runs on the uplink between the places a link may end at for its two gates: the device's host and
the outside. -/
theorem wire_ends : ∀ g : Fin 2, ∀ a ∈ wire.reads g,
    site.placeEnds uplink = (site.linkPlace (wire.x a), site.linkPlace (wire.x g)) := by decide +kernel

theorem linkβ_eq : ∀ e : Fin site.links.length, site.linkβ e = 100000 := by decide +kernel

theorem width_wire : (wire.onto site.linkPlace).width Finset.univ = 2 := by decide +kernel

/-- The wire's image runs on the registered links within their capacity: its one wire on the uplink, two bits at
most in any ticks against the uplink's capacity a tick. -/
theorem wire_links : (wire.onto site.linkPlace).Links site.placeEnds site.linkβ (fun _ _ => uplink) := by
  refine ⟨fun g a ha _ => ⟨rfl, Or.inl (wire_ends g a ha)⟩, fun e y a b => ?_⟩
  rw [linkβ_eq e]
  rcases Nat.lt_or_ge a b with hab | hab
  · exact ((Actual.width_mono (wire.onto site.linkPlace) (Finset.subset_univ _)).trans width_wire.le).trans
      (by omega)
  · have h0 : (wire.onto site.linkPlace).width
        ((wire.onto site.linkPlace).readOn (fun _ _ => uplink) e y (Set.Ico a b)) = 0 :=
      Finset.sum_eq_zero fun c hc => by
        simp only [Placed.readOn, mem_filter, mem_univ, true_and] at hc
        obtain ⟨g, -, -, -, -, hg⟩ := hc
        exact absurd (lt_of_le_of_lt hg.1 hg.2) (not_lt.2 hab)
    exact h0.le.trans (Nat.zero_le _)

/-- The device's gate leaves the host's subtree: the gate at the outside reads it. -/
theorem egress_node0 : 0 < wire.egress (site.placesIn (site.under "node0")) (Set.Ico 0 1) := by
  rw [under_node0]
  have hin : wire.x (0 : Fin 2) ∈ site.placesIn node0 := by decide +kernel
  have hout : wire.x (1 : Fin 2) ∉ site.placesIn node0 := by decide +kernel
  unfold Placed.egress Circuit.width
  refine lt_of_lt_of_le (by decide : 0 < wire.w (0 : Fin 2))
    (Finset.single_le_sum (f := wire.w) (fun _ _ => Nat.zero_le _) ?_)
  simp only [mem_filter, mem_univ, true_and]
  exact ⟨hin, (1 : Fin 2), hout, ⟨Nat.zero_le _, Nat.zero_lt_one⟩, by decide⟩

end RegisteredSite

/-- **The guarantee says something of a circuit placed at a device.** At the example's remote log, which is valid,
and the subtree at its host `node0`: a circuit with a gate at the device `gpu0`, read by a gate at the outside,
has its wire on the host's uplink within the uplink's capacity (`Placed.Links` of its image under `Site.linkPlace`),
the subtree holds its host whole (`Site.HostsWhole`), and the circuit's egress from the subtree is above 0. So
`EgressRegistered`'s two premises hold together of a circuit placed at a device, at a registered Site, where the
bound is not `0 ≤ 0`. -/
theorem EgressRegistered_nonvacuous :
    Registration.Valid Registration.Example.digests Registration.Example.auditor .remote
        [Registration.Example.remote] ∧
      (RegisteredSite.wire.onto RegisteredSite.site.linkPlace).Links RegisteredSite.site.placeEnds
        RegisteredSite.site.linkβ (fun _ _ => RegisteredSite.uplink) ∧
      RegisteredSite.site.HostsWhole "node0" ∧
      0 < RegisteredSite.wire.egress (RegisteredSite.site.placesIn (RegisteredSite.site.under "node0"))
        (Set.Ico 0 1) :=
  ⟨Registration.Example.remote_valid, RegisteredSite.wire_links, RegisteredSite.hostsWhole_node0,
    RegisteredSite.egress_node0⟩

theorem LeakPartsRegistered : Guarantees.LeakPartsRegistered := by
  intro D A c l S Ω pinned units i R ℓ link 𝒱 spec record bucket jitter a b K hv _ hu hsh hcs hdl hT hsec hK
  exact S.leak_le_registered hv.linkKinds
    (fun k hk => ⟨S.mem_places.2 (hv.ends_known k hk).1, S.mem_places.2 (hv.ends_known k hk).2⟩) pinned units i R
    ℓ link 𝒱 spec record bucket jitter a b K hu hsh hcs hdl hT hsec hK

end Accounting.SecurityProofs
