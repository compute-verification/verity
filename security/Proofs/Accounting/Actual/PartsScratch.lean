module

public import Definitions.Core.Site
public import Definitions.Accounting.Guarantees.Bridges
public import Proofs.Accounting.Actual.Physical
public import Proofs.Accounting.Actual.Bridges
public import Proofs.Core.Site
public import Mathlib.Data.String.Basic
public import Mathlib.Data.Set.Card
public import Mathlib.Algebra.BigOperators.Fin
public import Mathlib.Algebra.BigOperators.Group.Finset.Basic
public import Mathlib.Algebra.Order.BigOperators.Group.Finset
public import Mathlib.Algebra.Order.BigOperators.Ring.Finset

/-!
# Scratch: a node's egress and leak limits read off the registered tree

A scratch module, never landing: the statements of the two "limits add over parts" theorems for links, with the
definitions each needs that the tree does not hold. No body is `sorry`.

* The node: `Site.Child`, `Site.places`, `Site.under`, `Site.proverUnder`, `Site.certifiersOf`.
* The covered form of (2), where a memory challenger covers its link, two ways. Form (a), `leak_le_sum_K_joined`:
  the challenger is one of the covering set, and adds its record's `K` and a jitter share. Form (b),
  `leak_le_sum_K_sinks` and `leak_le_sum_K_sealed`: the challenger is a sink inside the location, and adds
  nothing while no declared link joins it to the outside (`Sealed`). At a node: `Site.challengersOf`,
  `Site.auditorsOf`, `Site.uncovered`, `Site.uncertified_registered_eq`, `Site.leak_le_sum_K_auditors`,
  `Site.leak_le_sum_K_sinks_registered`.
* The runs: `onsite_node0_covered` (the example site), `oneNode_charged` and `twoNodes_charged` (two hosts with a
  link between them, under one site and under two).
* The Site as the hardware model's links: `Site.linkEnds`, `Site.linkβ` over ids; `Site.Place`, `Site.place`,
  `Site.placeEnds`, `Site.loc` over the finite type of places, which `B4` needs.
* (1) `Site.egress_le_registered`, with `Site.egressLimit_eq` (the limit as the sum over the registered links with
  exactly one end in the node) and the parts lemmas `linkCapacity_biUnion_le`, `linkCapacity_biUnion`.
* (2) `leak_le_sum_K` in the model, `leak_le_sum_K_covered` under coverage, `Site.leak_le_sum_K_registered` at a node.
* `chal0_link_uncertified`: on the example onsite site, the challenger's link crosses the node's boundary with no
  certifier at an end.
-/

@[expose] public section

namespace Verity.Site

open Finset

variable (S : Site)

/-- `j` is registered directly under `i`. -/
def Child (i j : String) : Prop := ∃ x ∈ S.locs, x.id = j ∧ x.parent = some i

/-- **The places** of the Site: every registered location and the outside. A link ends at a place, and a gate of the
actual circuit is placed at one: a host is a place though locations are registered under it. -/
def places : Finset String := insert outside (S.locs.map (·.id)).toFinset

open Classical in
/-- **The node `i`**: the places at or under `i` in the registered tree. -/
noncomputable def under (i : String) : Finset String :=
  S.places.filter fun j => Relation.ReflTransGen S.Child i j

open Classical in
/-- **The prover's places at or under `i`**: the node less the auditor's devices. -/
noncomputable def proverUnder (i : String) : Finset String :=
  (S.under i).filter fun j => ∀ k ∈ S.kind? j, k.auditor = false

open Classical in
/-- **The certifiers of node `i`**: the registered certifiers a link joins to one of `i`'s prover places. -/
noncomputable def certifiersOf (i : String) : Finset String :=
  S.places.filter fun y => S.kind? y = some .certifier ∧
    ∃ k ∈ S.links, (k.ends.1 = y ∧ k.ends.2 ∈ S.proverUnder i) ∨ (k.ends.2 = y ∧ k.ends.1 ∈ S.proverUnder i)

open Classical in
/-- **The memory challengers that cover node `i`**, under the pins `pinned`: the registered memory challengers among
`pinned` that a registered link joins to one of `i`'s prover places. Three facts, each read here and not in prose:
the challenger is pinned (`pinned` holds the ids of the challengers whose key and session the auditor's pins hold; a
parameter until the pins are a definition this can read), it is registered as a challenger (`kind?`), and it is
registered at that link's end (`links`). An unregistered device at a link's end, or an unpinned challenger, is not
among them, so it covers nothing. -/
noncomputable def challengersOf (pinned : Finset String) (i : String) : Finset String :=
  S.places.filter fun y => S.kind? y = some .challenger ∧ y ∈ pinned ∧
    ∃ k ∈ S.links, (k.ends.1 = y ∧ k.ends.2 ∈ S.proverUnder i) ∨ (k.ends.2 = y ∧ k.ends.1 ∈ S.proverUnder i)

/-- **The auditor's devices at node `i`**: its certifiers and the memory challengers that cover it. -/
noncomputable def auditorsOf (pinned : Finset String) (i : String) : Finset String :=
  S.certifiersOf i ∪ S.challengersOf pinned i

/-- The registered links as the model's, by their position in the Site: link `e`'s two ends. -/
def linkEnds (e : Fin S.links.length) : String × String := (S.links.get e).ends

/-- The registered links as the model's: link `e`'s capacity each way. -/
def linkβ (e : Fin S.links.length) : ℕ := (S.links.get e).β

/-- **The registered egress limit of node `i`**, a tick: the model's `β_X` at the registered links and the node. -/
noncomputable def egressLimit (i : String) : ℕ := Accounting.linkCapacity S.linkEnds S.linkβ (S.under i)

/-- The registered links with exactly one end in `X`. -/
def crossing (X : Finset String) : List Link :=
  S.links.filter fun k => decide (k.ends.1 ∈ X ↔ k.ends.2 ∉ X)

/-- **The uncovered links of `X`** under the covering set `cov`: the registered links with exactly one end in `X` and
neither end in `cov`. The covered form of (2) charges each at its `β`. -/
def uncovered (cov X : Finset String) : List Link :=
  (S.crossing X).filter fun k => decide (k.ends.1 ∉ cov ∧ k.ends.2 ∉ cov)

/-- **(1) A node's egress is at most its registered limit**: over any ticks `[a, b)`, what leaves node `i` of the
registered tree is at most the capacity of the registered links with exactly one end in it, a tick. -/
theorem egress_le_registered (i : String) (A : Accounting.Placed String)
    (link : Fin A.N → Fin A.N → Fin S.links.length) (h : A.Links S.linkEnds S.linkβ link) (a b : ℕ) :
    A.egress (S.under i) (Set.Ico a b) ≤ S.egressLimit i * (b - a) :=
  Accounting.SecurityProofs.Actual.egress_le A h (S.under i) a b

/-- **The limit is read off the registered links**: the sum of `β` over the registered links with exactly one end in
the node, a link between two places of the node uncounted. -/
theorem egressLimit_eq (i : String) : S.egressLimit i = ((S.crossing (S.under i)).map (·.β)).sum := by
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
  unfold egressLimit Accounting.linkCapacity Accounting.boundary crossing
  rw [hl, Finset.sum_filter, ← Fin.sum_univ_fun_getElem]
  refine Finset.sum_congr rfl fun e _ => ?_
  by_cases h : ((S.links[e.1]).ends.1 ∈ S.under i ↔ (S.links[e.1]).ends.2 ∉ S.under i)
  · simp [linkEnds, linkβ, h]
  · simp [linkEnds, h]

/-- A place of the Site as a leaf of the model: the model's bound on what the outside does (`B4`) takes a finite,
ordered type of leaves. -/
abbrev Place : Type := {j : String // j ∈ S.places}

/-- The place an id names; the outside where it names none. -/
def place (i : String) : S.Place :=
  if h : i ∈ S.places then ⟨i, h⟩ else ⟨outside, Finset.mem_insert_self outside _⟩

/-- The registered links as the model's, over places: link `e`'s two ends. -/
def placeEnds (e : Fin S.links.length) : S.Place × S.Place :=
  (S.place (S.links.get e).ends.1, S.place (S.links.get e).ends.2)

/-- A set of ids as a location of the model. -/
def loc (X : Finset String) : Finset S.Place := Finset.univ.filter fun p => p.1 ∈ X

end Verity.Site

namespace Accounting

open Verity Finset

section Parts

variable {L E ι : Type*} [Fintype E] [DecidableEq L]

/-- A link across the boundary of two parts' union crosses the boundary of one of them. -/
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

/-- **Limits add over parts, at most**: the capacity across the boundary of a union of parts is at most the parts'
capacities summed. -/
theorem linkCapacity_biUnion_le (ends : E → L × L) (β : E → ℕ) (P : Finset ι) (X : ι → Finset L) :
    linkCapacity ends β (P.biUnion X) ≤ ∑ p ∈ P, linkCapacity ends β (X p) := by
  classical
  induction P using Finset.induction_on with
  | empty => simp [linkCapacity, boundary]
  | insert p P hp ih =>
    rw [Finset.biUnion_insert, Finset.sum_insert hp]
    exact (linkCapacity_union_le ends β (X p) (P.biUnion X)).trans (Nat.add_le_add_left ih _)

open Classical in
/-- The capacity of the links between two locations, either way round. -/
noncomputable def betweenCapacity (ends : E → L × L) (β : E → ℕ) (X Y : Finset L) : ℕ :=
  ∑ e ∈ univ.filter (fun e => ((ends e).1 ∈ X ∧ (ends e).2 ∈ Y) ∨ ((ends e).1 ∈ Y ∧ (ends e).2 ∈ X)), β e

/-- **Limits add over two parts, exactly**: for two disjoint parts, the two capacities summed are the union's and each
link between the two twice, once at each end. -/
theorem linkCapacity_union (ends : E → L × L) (β : E → ℕ) (X Y : Finset L) (hd : Disjoint X Y) :
    linkCapacity ends β (X ∪ Y) + 2 * betweenCapacity ends β X Y =
      linkCapacity ends β X + linkCapacity ends β Y := by
  classical
  unfold linkCapacity betweenCapacity boundary
  rw [Finset.sum_filter, Finset.sum_filter, Finset.sum_filter, Finset.sum_filter, Finset.mul_sum,
    ← Finset.sum_add_distrib, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun e _ => ?_
  have h1 : (ends e).1 ∈ X → (ends e).1 ∉ Y := fun h => Finset.disjoint_left.1 hd h
  have h2 : (ends e).2 ∈ X → (ends e).2 ∉ Y := fun h => Finset.disjoint_left.1 hd h
  simp only [mem_union]
  by_cases a : (ends e).1 ∈ X <;> by_cases b : (ends e).2 ∈ X <;>
    by_cases c : (ends e).1 ∈ Y <;> by_cases d : (ends e).2 ∈ Y <;> simp_all <;> omega

open Classical in
/-- The capacity of the links between two different parts. -/
noncomputable def innerCapacity (ends : E → L × L) (β : E → ℕ) (P : Finset ι) (X : ι → Finset L) : ℕ :=
  ∑ e ∈ univ.filter (fun e => ∃ p ∈ P, ∃ q ∈ P, p ≠ q ∧ (ends e).1 ∈ X p ∧ (ends e).2 ∈ X q), β e

/-- The links between two different parts of `insert p P` are those between two parts of `P` and those between part
`p` and the rest. -/
theorem innerCapacity_insert [DecidableEq ι] (ends : E → L × L) (β : E → ℕ) {P : Finset ι} {p : ι}
    (X : ι → Finset L) (hp : p ∉ P)
    (hd : ((insert p P : Finset ι) : Set ι).PairwiseDisjoint X) :
    innerCapacity ends β (insert p P) X =
      innerCapacity ends β P X + betweenCapacity ends β (X p) (P.biUnion X) := by
  classical
  unfold innerCapacity betweenCapacity
  rw [Finset.sum_filter, Finset.sum_filter, Finset.sum_filter, ← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun e _ => ?_
  have hne : ∀ q ∈ P, p ≠ q := fun q hq h => hp (h ▸ hq)
  have hpq : ∀ q ∈ P, ∀ x, x ∈ X p → x ∉ X q := fun q hq x hx hx' =>
    Finset.disjoint_left.1 (hd (Finset.mem_coe.2 (Finset.mem_insert_self p P))
      (Finset.mem_coe.2 (Finset.mem_insert_of_mem hq)) (hne q hq)) hx hx'
  by_cases hI : ∃ p' ∈ P, ∃ q' ∈ P, p' ≠ q' ∧ (ends e).1 ∈ X p' ∧ (ends e).2 ∈ X q'
  · have hB : ¬(((ends e).1 ∈ X p ∧ (ends e).2 ∈ P.biUnion X) ∨
        ((ends e).1 ∈ P.biUnion X ∧ (ends e).2 ∈ X p)) := by
      obtain ⟨p', hp', q', hq', -, h1, h2⟩ := hI
      rintro (⟨h, -⟩ | ⟨-, h⟩)
      · exact hpq p' hp' _ h h1
      · exact hpq q' hq' _ h h2
    have hL : ∃ p' ∈ insert p P, ∃ q' ∈ insert p P, p' ≠ q' ∧ (ends e).1 ∈ X p' ∧ (ends e).2 ∈ X q' := by
      obtain ⟨p', hp', q', hq', hne', h1, h2⟩ := hI
      exact ⟨p', Finset.mem_insert_of_mem hp', q', Finset.mem_insert_of_mem hq', hne', h1, h2⟩
    rw [ite_eq_left hL, ite_eq_left hI, ite_eq_right hB, Nat.add_zero]
  · by_cases hB : ((ends e).1 ∈ X p ∧ (ends e).2 ∈ P.biUnion X) ∨ ((ends e).1 ∈ P.biUnion X ∧ (ends e).2 ∈ X p)
    · have hL : ∃ p' ∈ insert p P, ∃ q' ∈ insert p P, p' ≠ q' ∧ (ends e).1 ∈ X p' ∧ (ends e).2 ∈ X q' := by
        rcases hB with ⟨h1, h2⟩ | ⟨h1, h2⟩
        · obtain ⟨q, hq, h2q⟩ := Finset.mem_biUnion.1 h2
          exact ⟨p, Finset.mem_insert_self p P, q, Finset.mem_insert_of_mem hq, hne q hq, h1, h2q⟩
        · obtain ⟨q, hq, h1q⟩ := Finset.mem_biUnion.1 h1
          exact ⟨q, Finset.mem_insert_of_mem hq, p, Finset.mem_insert_self p P, (hne q hq).symm, h1q, h2⟩
      rw [ite_eq_left hL, ite_eq_right hI, ite_eq_left hB, Nat.zero_add]
    · have hL : ¬∃ p' ∈ insert p P, ∃ q' ∈ insert p P, p' ≠ q' ∧ (ends e).1 ∈ X p' ∧ (ends e).2 ∈ X q' := by
        rintro ⟨p', hp', q', hq', hne', h1, h2⟩
        rcases Finset.mem_insert.1 hp' with hpe | hp'
        · rcases Finset.mem_insert.1 hq' with hqe | hq'
          · exact hne' (hpe.trans hqe.symm)
          · exact hB (Or.inl ⟨hpe ▸ h1, Finset.mem_biUnion.2 ⟨q', hq', h2⟩⟩)
        · rcases Finset.mem_insert.1 hq' with hqe | hq'
          · exact hB (Or.inr ⟨Finset.mem_biUnion.2 ⟨p', hp', h1⟩, hqe ▸ h2⟩)
          · exact hI ⟨p', hp', q', hq', hne', h1, h2⟩
      rw [ite_eq_right hL, ite_eq_right hI, ite_eq_right hB, Nat.add_zero]

/-- **Limits add over parts, exactly**: for disjoint parts, the parts' capacities summed are the whole's and each
link between two parts twice, once at each end. -/
theorem linkCapacity_biUnion (ends : E → L × L) (β : E → ℕ) (P : Finset ι) (X : ι → Finset L)
    (hd : (P : Set ι).PairwiseDisjoint X) :
    linkCapacity ends β (P.biUnion X) + 2 * innerCapacity ends β P X = ∑ p ∈ P, linkCapacity ends β (X p) := by
  classical
  induction P using Finset.induction_on with
  | empty => simp [linkCapacity, boundary, innerCapacity]
  | insert p P hp ih =>
    have hdP : (P : Set ι).PairwiseDisjoint X := hd.subset (Finset.coe_subset.2 (Finset.subset_insert p P))
    have hdis : Disjoint (X p) (P.biUnion X) := by
      rw [Finset.disjoint_biUnion_right]
      intro q hq
      exact hd (Finset.mem_coe.2 (Finset.mem_insert_self p P)) (Finset.mem_coe.2 (Finset.mem_insert_of_mem hq))
        fun h => hp (h ▸ hq)
    have h := linkCapacity_union ends β (X p) (P.biUnion X) hdis
    have h' := ih hdP
    rw [Finset.biUnion_insert, Finset.sum_insert hp, innerCapacity_insert ends β X hp hd]
    omega

end Parts

/-- **Each certifier's record is within its `K`**: across the strategy, what certifier `y` of `cert` places in its
record during `[a, b)` takes at most `2 ^ K y` values. The parts' limits, which `leak_le_sum_K` adds. -/
def RecordsWithin {Ω L : Type*} (S : Ω → Actual.Run L) (cert : Finset L) (record : L → Set ℕ) (a b : ℕ)
    (K : L → ℕ) : Prop :=
  ∀ y ∈ cert, (Set.range fun r =>
    (S r).seen '' ↑(((S r).record {y} record).filter fun g => a ≤ (S r).A.t g ∧ (S r).A.t g < b)).encard ≤
      ((2 ^ K y : ℕ) : ℕ∞)

/-- One certifier's record placed during `[a, b)`, as seen, is the certifiers' at its leaf. -/
theorem seen_record_single_Ico {L : Type*} (R : Actual.Run L) {cert : Finset L} (cells : L → Set ℕ) {y : L}
    (hy : y ∈ cert) (a b : ℕ) :
    R.seen '' ↑((R.record {y} cells).filter fun g => a ≤ R.A.t g ∧ R.A.t g < b) =
      R.seen '' ↑((R.record cert cells).filter fun g => a ≤ R.A.t g ∧ R.A.t g < b) ∩ {p | p.2.1 = y} := by
  ext p
  simp only [Set.mem_image, Finset.coe_filter, Actual.Run.record, Finset.mem_filter, Finset.mem_univ, true_and,
    Set.mem_ofPred_eq, Set.mem_inter_iff, Finset.mem_singleton]
  constructor
  · rintro ⟨g, ⟨⟨hx, hc⟩, ht⟩, rfl⟩
    exact ⟨⟨g, ⟨⟨hx ▸ hy, hc⟩, ht⟩, rfl⟩, hx⟩
  · rintro ⟨⟨g, ⟨⟨-, hc⟩, ht⟩, rfl⟩, hgy⟩
    exact ⟨g, ⟨⟨hgy, hc⟩, ht⟩, rfl⟩

/-- **A value read back from its parts takes at most the product of the parts' counts**: where every `z` of `Zs` is
`dec` of its views and view `j` takes at most `2 ^ K j` values over `Zs`, `Zs` holds at most `2 ^ ∑ K` values. -/
theorem encard_le_two_pow_sum {Z ι : Type*} [Fintype ι] {V : ι → Type*} (Zs : Set Z) (view : ∀ j, Z → V j)
    (dec : (∀ j, V j) → Z) (hdec : ∀ z ∈ Zs, dec (fun j => view j z) = z) (K : ι → ℕ)
    (hK : ∀ j, (view j '' Zs).encard ≤ ((2 ^ K j : ℕ) : ℕ∞)) :
    Zs.encard ≤ ((2 ^ ∑ j, K j : ℕ) : ℕ∞) := by
  have hsub : Zs ⊆ dec '' (Set.pi Set.univ fun j => view j '' Zs) := by
    intro z hz
    exact ⟨fun j => view j z, fun j _ => ⟨z, hz, rfl⟩, hdec z hz⟩
  calc Zs.encard ≤ (dec '' (Set.pi Set.univ fun j => view j '' Zs)).encard := Set.encard_le_encard hsub
    _ ≤ (Set.pi Set.univ fun j => view j '' Zs).encard := Set.encard_image_le _ _
    _ = ∏ j, (view j '' Zs).encard := Set.encard_pi_eq_prod_encard
    _ ≤ ∏ j, ((2 ^ K j : ℕ) : ℕ∞) := Finset.prod_le_prod fun j _ => hK j
    _ = ((∏ j, 2 ^ K j : ℕ) : ℕ∞) := (Nat.cast_prod _ _).symm
    _ = ((2 ^ ∑ j, K j : ℕ) : ℕ∞) := by rw [Finset.prod_pow_eq_pow_sum]

/-- **The certifiers' joint record is within the sum of their `K`**: the joint record during `[a, b)` is the union of
each certifier's, its part at that leaf, so it takes at most `2 ^ ∑ K` values. -/
theorem records_encard_le {Ω L : Type*} (S : Ω → Actual.Run L) (cert : Finset L) (record : L → Set ℕ) (a b : ℕ)
    (K : L → ℕ) (hK : RecordsWithin S cert record a b K) :
    (Set.range fun r =>
      (S r).seen '' ↑(((S r).record cert record).filter fun g => a ≤ (S r).A.t g ∧ (S r).A.t g < b)).encard ≤
        ((2 ^ ∑ y ∈ cert, K y : ℕ) : ℕ∞) := by
  refine (encard_le_two_pow_sum (ι := ↥cert) (V := fun _ => Set (ℕ × L × ℕ × ℕ × ℕ))
    (Set.range fun r =>
      (S r).seen '' ↑(((S r).record cert record).filter fun g => a ≤ (S r).A.t g ∧ (S r).A.t g < b))
    (fun j z => z ∩ {p | p.2.1 = j.1}) (fun f => ⋃ j, f j) ?_ (fun j => K j.1) ?_).trans_eq ?_
  · rintro _ ⟨r, rfl⟩
    ext p
    constructor
    · intro hp
      obtain ⟨j, hpj⟩ := Set.mem_iUnion.1 hp
      exact ((Set.mem_inter_iff _ _ _).1 hpj).1
    · intro hp
      obtain ⟨g, hg, rfl⟩ := (Set.mem_image _ _ _).1 hp
      have hx : (S r).A.x g ∈ cert := by
        have hrec := (Finset.mem_filter.1 (Finset.mem_coe.1 hg)).1
        simp only [Actual.Run.record, Finset.mem_filter, Finset.mem_univ, true_and] at hrec
        exact hrec.1
      exact Set.mem_iUnion.2 ⟨⟨(S r).A.x g, hx⟩, (Set.mem_inter_iff _ _ _).2 ⟨Set.mem_image_of_mem _ hg, rfl⟩⟩
  · intro j
    refine le_of_eq_of_le (congrArg Set.encard ?_) (hK j.1 j.2)
    ext z
    constructor
    · rintro ⟨_, ⟨r, rfl⟩, rfl⟩
      exact ⟨r, seen_record_single_Ico (S r) record j.2 a b⟩
    · rintro ⟨r, rfl⟩
      exact ⟨_, ⟨r, rfl⟩, (seen_record_single_Ico (S r) record j.2 a b).symm⟩
  · rw [Finset.sum_coe_sort cert K]

/-- **(2) A location's leak is at most the sum of its certifiers' `K`**, and what its uncertified links carry: under
`B4`'s premises, with each certifier's record within its `K` (`RecordsWithin`), what the outside does during `[a, b)`
takes at most `2 ^ (∑ K + jitter · buckets · #cert + β°_X (b − a))` values. -/
theorem leak_le_sum_K {L E Ω : Type*} [LinearOrder L] [Fintype L] [Fintype E] (S : Ω → Actual.Run L)
    (ℓ : L → L → ℕ) (ends : E → L × L) (β : E → ℕ) (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → E)
    (𝒱 cert X : Finset L) (spec : ∀ r, (S r).A.Spec) (record : L → Set ℕ) (bucket : ℕ → ℕ) (jitter a b : ℕ)
    (K : L → ℕ) (hcs : CausalStrategy S ℓ) (hdl : DeclaredLinks S ends β link)
    (hT : TrustedUnits S 𝒱 cert spec record bucket jitter) (hsec : Secrets S cert X a)
    (hK : RecordsWithin S cert record a b K) :
    (Set.range fun r => (S r).acts (X ∪ cert)ᶜ (Set.Ico a b)).encard ≤
      ((2 ^ ((∑ y ∈ cert, K y) + jitter * ((Finset.Ico a b).image bucket).card * cert.card +
        uncertifiedCapacity ends β cert X * (b - a)) : ℕ) : ℕ∞) := by
  refine (SecurityProofs.B4 S ℓ ends β link 𝒱 cert X spec record bucket jitter a b hcs hdl hT hsec).trans ?_
  refine (mul_le_mul' (records_encard_le S cert record a b K hK) le_rfl).trans_eq ?_
  rw [← Nat.cast_mul, ← pow_add, Nat.add_assoc]

/-- **(2) at a covered location**: under coverage the uncertified term drops. -/
theorem leak_le_sum_K_covered {L E Ω : Type*} [LinearOrder L] [Fintype L] [Fintype E] (S : Ω → Actual.Run L)
    (ℓ : L → L → ℕ) (ends : E → L × L) (β : E → ℕ) (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → E)
    (𝒱 cert X : Finset L) (spec : ∀ r, (S r).A.Spec) (record : L → Set ℕ) (bucket : ℕ → ℕ) (jitter a b : ℕ)
    (K : L → ℕ) (hcs : CausalStrategy S ℓ) (hdl : DeclaredLinks S ends β link)
    (hT : TrustedUnits S 𝒱 cert spec record bucket jitter) (hsec : Secrets S cert X a)
    (hcov : Covered ends β cert X) (hK : RecordsWithin S cert record a b K) :
    (Set.range fun r => (S r).acts (X ∪ cert)ᶜ (Set.Ico a b)).encard ≤
      ((2 ^ ((∑ y ∈ cert, K y) + jitter * ((Finset.Ico a b).image bucket).card * cert.card) : ℕ) : ℕ∞) := by
  have h := leak_le_sum_K S ℓ ends β link 𝒱 cert X spec record bucket jitter a b K hcs hdl hT hsec hK
  rw [show uncertifiedCapacity ends β cert X = 0 from hcov, Nat.zero_mul, Nat.add_zero] at h
  exact h

section Covering

variable {L E : Type*} [Fintype E]

/-- **An uncovered link across the boundary is charged**: a link with exactly one end in `X` and neither end in the
covering set `cov` is in `β°_X`, whatever the tree above its two ends. -/
theorem le_uncertifiedCapacity (ends : E → L × L) (β : E → ℕ) (cov X : Finset L) (e : E)
    (hx : (ends e).1 ∈ X ↔ (ends e).2 ∉ X) (h1 : (ends e).1 ∉ cov) (h2 : (ends e).2 ∉ cov) :
    β e ≤ uncertifiedCapacity ends β cov X := by
  classical
  unfold uncertifiedCapacity
  refine Finset.single_le_sum (f := β) (fun _ _ => Nat.zero_le _) ?_
  simp only [boundary, mem_filter, mem_univ, true_and]
  exact ⟨hx, h1, h2⟩

/-- **The uncovered links of a location with sinks**: an uncovered link across the boundary of `X ∪ Y`, for `X` and
`Y` apart, is an uncovered link of `X` that ends at no place of `Y`, or one of `Y` that ends at no place of `X`. -/
theorem uncertifiedCapacity_union_le [DecidableEq L] (ends : E → L × L) (β : E → ℕ) (cert X Y : Finset L)
    (hXY : Disjoint X Y) :
    uncertifiedCapacity ends β cert (X ∪ Y) ≤
      uncertifiedCapacity ends β (cert ∪ Y) X + uncertifiedCapacity ends β (cert ∪ X) Y := by
  classical
  have hd : ∀ z, z ∈ X → z ∉ Y := fun z h => Finset.disjoint_left.1 hXY h
  unfold uncertifiedCapacity
  refine le_trans ?_ (Nat.le.intro Finset.sum_union_inter)
  refine Finset.sum_le_sum_of_subset ?_
  intro e he
  have ha := hd (ends e).1
  have hb := hd (ends e).2
  simp only [boundary, mem_filter, mem_univ, true_and, mem_union] at he ⊢
  by_cases a1 : (ends e).1 ∈ X <;> by_cases a2 : (ends e).2 ∈ X <;>
    by_cases b1 : (ends e).1 ∈ Y <;> by_cases b2 : (ends e).2 ∈ Y <;> simp_all

/-- **The sinks are sealed**: every link across the boundary of the sinks ends at a certifier or at a place of the
location `X`, so with the links as declared (`DeclaredLinks`) nothing a sink is sent leaves it but back into `X` or
through a certifier. It is `Covered` at the sinks, with the certifiers and `X` as what covers them: a fact of the
declared links, which names no behaviour of a sink. -/
def Sealed [DecidableEq L] (ends : E → L × L) (β : E → ℕ) (cert X sink : Finset L) : Prop :=
  Covered ends β (cert ∪ X) sink

end Covering

/-- **(2), the covered form, (a): a memory challenger that covers a link is one of the covering set.** With the
certifiers `cert` and the challengers `chal` apart, `B4`'s premises at `cert ∪ chal` and each of their records within
its `K`: what the outside of `X`, the certifiers and the challengers does during `[a, b)` takes at most
`2 ^ (∑ K over cert + ∑ K over chal + jitter · buckets · (#cert + #chal) + β° (b − a))` values, `β°` the capacity of
the links across `X`'s boundary with neither a certifier nor a challenger at an end.

A challenger adds its own record's `K` and one jitter share. `TrustedUnits` at `cert ∪ chal` asks of a challenger what
it asks of a certifier: its gates are the protocol's, and what it sends outside itself is a function of its own record
and jitter word. -/
theorem leak_le_sum_K_joined {L E Ω : Type*} [LinearOrder L] [Fintype L] [Fintype E] (S : Ω → Actual.Run L)
    (ℓ : L → L → ℕ) (ends : E → L × L) (β : E → ℕ) (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → E)
    (𝒱 cert chal X : Finset L) (spec : ∀ r, (S r).A.Spec) (record : L → Set ℕ) (bucket : ℕ → ℕ) (jitter a b : ℕ)
    (K : L → ℕ) (hcs : CausalStrategy S ℓ) (hdl : DeclaredLinks S ends β link) (hd : Disjoint cert chal)
    (hT : TrustedUnits S 𝒱 (cert ∪ chal) spec record bucket jitter) (hsec : Secrets S (cert ∪ chal) X a)
    (hK : RecordsWithin S (cert ∪ chal) record a b K) :
    (Set.range fun r => (S r).acts (X ∪ (cert ∪ chal))ᶜ (Set.Ico a b)).encard ≤
      ((2 ^ ((∑ y ∈ cert, K y) + (∑ c ∈ chal, K c) +
        jitter * ((Finset.Ico a b).image bucket).card * (cert.card + chal.card) +
        uncertifiedCapacity ends β (cert ∪ chal) X * (b - a)) : ℕ) : ℕ∞) := by
  have h := leak_le_sum_K S ℓ ends β link 𝒱 (cert ∪ chal) X spec record bucket jitter a b K hcs hdl hT hsec hK
  rw [Finset.sum_union hd, Finset.card_union_eq_card_add_card.2 hd] at h
  exact h

/-- **(2), the covered form, (b): a memory challenger that covers a link is a sink inside the location.** With the
sinks `sink` apart from `X`, `B4`'s premises at the location `X ∪ sink` and each certifier's record within its `K`:
what the outside of `X`, the sinks and the certifiers does during `[a, b)` takes at most
`2 ^ (∑ K over cert + jitter · buckets · #cert + (β° + β°ₛ) (b − a))` values, `β°` the capacity of the links across
`X`'s boundary with neither a certifier nor a sink at an end and `β°ₛ` that of the links from a sink to a place outside
`X`, the sinks and the certifiers.

A sink adds no `K` and no jitter share, and nothing is asked of what it does: `TrustedUnits` is at the certifiers
alone. What the sinks do is not on the left side: this says nothing of what a sink, or a place among the sinks, learns.
-/
theorem leak_le_sum_K_sinks {L E Ω : Type*} [LinearOrder L] [Fintype L] [Fintype E] (S : Ω → Actual.Run L)
    (ℓ : L → L → ℕ) (ends : E → L × L) (β : E → ℕ) (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → E)
    (𝒱 cert X sink : Finset L) (spec : ∀ r, (S r).A.Spec) (record : L → Set ℕ) (bucket : ℕ → ℕ) (jitter a b : ℕ)
    (K : L → ℕ) (hcs : CausalStrategy S ℓ) (hdl : DeclaredLinks S ends β link)
    (hT : TrustedUnits S 𝒱 cert spec record bucket jitter) (hXs : Disjoint X sink)
    (hsec : Secrets S cert (X ∪ sink) a) (hK : RecordsWithin S cert record a b K) :
    (Set.range fun r => (S r).acts (X ∪ sink ∪ cert)ᶜ (Set.Ico a b)).encard ≤
      ((2 ^ ((∑ y ∈ cert, K y) + jitter * ((Finset.Ico a b).image bucket).card * cert.card +
        (uncertifiedCapacity ends β (cert ∪ sink) X + uncertifiedCapacity ends β (cert ∪ X) sink) * (b - a)) : ℕ) :
          ℕ∞) := by
  refine (leak_le_sum_K S ℓ ends β link 𝒱 cert (X ∪ sink) spec record bucket jitter a b K hcs hdl hT hsec
    hK).trans ?_
  exact Nat.mono_cast (Nat.pow_le_pow_right Nat.two_pos (Nat.add_le_add_left
    (Nat.mul_le_mul_right _ (uncertifiedCapacity_union_le ends β cert X sink hXs)) _))

/-- **(2), form (b), with the sinks sealed** (`Sealed`): the sinks' own term drops, and what is left of `β°` is the
capacity of the links across `X`'s boundary with neither a certifier nor a sink at an end. -/
theorem leak_le_sum_K_sealed {L E Ω : Type*} [LinearOrder L] [Fintype L] [Fintype E] (S : Ω → Actual.Run L)
    (ℓ : L → L → ℕ) (ends : E → L × L) (β : E → ℕ) (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → E)
    (𝒱 cert X sink : Finset L) (spec : ∀ r, (S r).A.Spec) (record : L → Set ℕ) (bucket : ℕ → ℕ) (jitter a b : ℕ)
    (K : L → ℕ) (hcs : CausalStrategy S ℓ) (hdl : DeclaredLinks S ends β link)
    (hT : TrustedUnits S 𝒱 cert spec record bucket jitter) (hXs : Disjoint X sink)
    (hsec : Secrets S cert (X ∪ sink) a) (hseal : Sealed ends β cert X sink)
    (hK : RecordsWithin S cert record a b K) :
    (Set.range fun r => (S r).acts (X ∪ sink ∪ cert)ᶜ (Set.Ico a b)).encard ≤
      ((2 ^ ((∑ y ∈ cert, K y) + jitter * ((Finset.Ico a b).image bucket).card * cert.card +
        uncertifiedCapacity ends β (cert ∪ sink) X * (b - a)) : ℕ) : ℕ∞) := by
  have h := leak_le_sum_K_sinks S ℓ ends β link 𝒱 cert X sink spec record bucket jitter a b K hcs hdl hT hXs hsec hK
  rw [show uncertifiedCapacity ends β (cert ∪ X) sink = 0 from hseal, Nat.add_zero] at h
  exact h

end Accounting

namespace Verity.Site

open Finset

variable (S : Site)

open Classical in
/-- A node is its own place and its children's nodes. -/
theorem under_eq (i : String) (hi : i ∈ S.places) :
    S.under i = insert i ((S.places.filter (S.Child i)).biUnion S.under) := by
  ext j
  simp only [under, mem_filter, mem_insert, mem_biUnion]
  constructor
  · rintro ⟨hj, h⟩
    rcases Relation.ReflTransGen.cases_head h with rfl | ⟨c, hc, hcj⟩
    · exact Or.inl rfl
    · have hcp : c ∈ S.places := by
        obtain ⟨x, hx, hxc, -⟩ := hc
        exact Finset.mem_insert_of_mem (List.mem_toFinset.2 (List.mem_map.2 ⟨x, hx, hxc⟩))
      exact Or.inr ⟨c, ⟨hcp, hc⟩, hj, hcj⟩
  · rintro (rfl | ⟨c, ⟨-, hc⟩, hj, hcj⟩)
    · exact ⟨hi, Relation.ReflTransGen.refl⟩
    · exact ⟨hj, Relation.ReflTransGen.head hc hcj⟩

open Classical in
/-- **A node's limit is at most its own place's and its children's, summed**: limits add over the parts of a node of
the registered tree, at most. -/
theorem egressLimit_le_parts (i : String) (hi : i ∈ S.places) :
    S.egressLimit i ≤ Accounting.linkCapacity S.linkEnds S.linkβ {i} +
      ∑ c ∈ S.places.filter (S.Child i), S.egressLimit c := by
  unfold egressLimit
  rw [S.under_eq i hi, Finset.insert_eq]
  exact (Accounting.linkCapacity_union_le _ _ _ _).trans
    (Nat.add_le_add_left (Accounting.linkCapacity_biUnion_le _ _ _ _) _)

/-- **(2) at a node of the registered tree**: for node `i`, its prover places as the location and its certifiers
(`certifiersOf`) as the certifiers, at the registered links. -/
theorem leak_le_sum_K_registered {Ω : Type*} (i : String) (R : Ω → Accounting.Actual.Run S.Place)
    (ℓ : S.Place → S.Place → ℕ) (link : ∀ r, Fin (R r).A.N → Fin (R r).A.N → Fin S.links.length)
    (𝒱 : Finset S.Place) (spec : ∀ r, (R r).A.Spec) (record : S.Place → Set ℕ) (bucket : ℕ → ℕ)
    (jitter a b : ℕ) (K : S.Place → ℕ) (hcs : Accounting.CausalStrategy R ℓ)
    (hdl : Accounting.DeclaredLinks R S.placeEnds S.linkβ link)
    (hT : Accounting.TrustedUnits R 𝒱 (S.loc (S.certifiersOf i)) spec record bucket jitter)
    (hsec : Accounting.Secrets R (S.loc (S.certifiersOf i)) (S.loc (S.proverUnder i)) a)
    (hK : Accounting.RecordsWithin R (S.loc (S.certifiersOf i)) record a b K) :
    (Set.range fun r =>
        (R r).acts (S.loc (S.proverUnder i) ∪ S.loc (S.certifiersOf i))ᶜ (Set.Ico a b)).encard ≤
      ((2 ^ ((∑ y ∈ S.loc (S.certifiersOf i), K y) +
        jitter * ((Finset.Ico a b).image bucket).card * (S.loc (S.certifiersOf i)).card +
        Accounting.uncertifiedCapacity S.placeEnds S.linkβ (S.loc (S.certifiersOf i)) (S.loc (S.proverUnder i)) *
          (b - a)) : ℕ) : ℕ∞) := by
  exact Accounting.leak_le_sum_K R ℓ S.placeEnds S.linkβ link 𝒱 (S.loc (S.certifiersOf i))
    (S.loc (S.proverUnder i)) spec record bucket jitter a b K hcs hdl hT hsec hK

/-- **The uncovered term is read off the registered links**: where every registered link ends at places of the Site
(a valid registration's `unknownRef` check), the model's `β°` at the registered links, a covering set `cov` and a
location `X` is the sum of `β` over the registered links with exactly one end in `X` and neither end in `cov`
(`uncovered`). -/
theorem uncertified_registered_eq (cov X : Finset String)
    (hends : ∀ k ∈ S.links, k.ends.1 ∈ S.places ∧ k.ends.2 ∈ S.places) :
    Accounting.uncertifiedCapacity S.placeEnds S.linkβ (S.loc cov) (S.loc X) =
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
      ((S.placeEnds e).1 ∈ S.loc Y ↔ (S.links[e.1]).ends.1 ∈ Y) ∧
        ((S.placeEnds e).2 ∈ S.loc Y ↔ (S.links[e.1]).ends.2 ∈ Y) := by
    intro e Y
    have h := hends (S.links[e.1]) (List.getElem_mem e.2)
    simp [placeEnds, loc, place, h.1, h.2]
  unfold Accounting.uncertifiedCapacity Accounting.boundary uncovered crossing
  rw [List.filter_filter, hl, Finset.sum_filter, Finset.sum_filter, ← Fin.sum_univ_fun_getElem]
  refine Finset.sum_congr rfl fun e _ => ?_
  have hX := hp e X
  have hC := hp e cov
  by_cases h : ((S.links[e.1]).ends.1 ∈ X ↔ (S.links[e.1]).ends.2 ∉ X) <;>
    by_cases h1 : (S.links[e.1]).ends.1 ∈ cov <;> by_cases h2 : (S.links[e.1]).ends.2 ∈ cov <;>
    simp [linkβ, hX, hC, h, h1, h2]

/-- **(2), the covered form, (a), at a node of the registered tree**: for node `i`, its prover places as the location
and the auditor's devices at it (`auditorsOf`: its certifiers and the pinned, registered memory challengers a link
joins to it) as the covering set, at the registered links. The last term is over the registered links with exactly
one end among the node's prover places and no auditor's device at either end (`uncertified_registered_eq`). -/
theorem leak_le_sum_K_auditors {Ω : Type*} (pinned : Finset String) (i : String)
    (R : Ω → Accounting.Actual.Run S.Place) (ℓ : S.Place → S.Place → ℕ)
    (link : ∀ r, Fin (R r).A.N → Fin (R r).A.N → Fin S.links.length) (𝒱 : Finset S.Place) (spec : ∀ r, (R r).A.Spec)
    (record : S.Place → Set ℕ) (bucket : ℕ → ℕ) (jitter a b : ℕ) (K : S.Place → ℕ)
    (hcs : Accounting.CausalStrategy R ℓ) (hdl : Accounting.DeclaredLinks R S.placeEnds S.linkβ link)
    (hT : Accounting.TrustedUnits R 𝒱 (S.loc (S.auditorsOf pinned i)) spec record bucket jitter)
    (hsec : Accounting.Secrets R (S.loc (S.auditorsOf pinned i)) (S.loc (S.proverUnder i)) a)
    (hK : Accounting.RecordsWithin R (S.loc (S.auditorsOf pinned i)) record a b K) :
    (Set.range fun r =>
        (R r).acts (S.loc (S.proverUnder i) ∪ S.loc (S.auditorsOf pinned i))ᶜ (Set.Ico a b)).encard ≤
      ((2 ^ ((∑ y ∈ S.loc (S.auditorsOf pinned i), K y) +
        jitter * ((Finset.Ico a b).image bucket).card * (S.loc (S.auditorsOf pinned i)).card +
        Accounting.uncertifiedCapacity S.placeEnds S.linkβ (S.loc (S.auditorsOf pinned i))
          (S.loc (S.proverUnder i)) * (b - a)) : ℕ) : ℕ∞) := by
  exact Accounting.leak_le_sum_K R ℓ S.placeEnds S.linkβ link 𝒱 (S.loc (S.auditorsOf pinned i))
    (S.loc (S.proverUnder i)) spec record bucket jitter a b K hcs hdl hT hsec hK

/-- **(2), the covered form, (b), at a node of the registered tree**: for node `i`, its prover places with the memory
challengers that cover it (`challengersOf`) as the location, and its certifiers as the certifiers. The last term is
over the registered links with exactly one end among those places and no certifier at either end: a link from the
node to a challenger that covers it is inside the location, and a link from such a challenger to any other place is
across its boundary. -/
theorem leak_le_sum_K_sinks_registered {Ω : Type*} (pinned : Finset String) (i : String)
    (R : Ω → Accounting.Actual.Run S.Place) (ℓ : S.Place → S.Place → ℕ)
    (link : ∀ r, Fin (R r).A.N → Fin (R r).A.N → Fin S.links.length) (𝒱 : Finset S.Place) (spec : ∀ r, (R r).A.Spec)
    (record : S.Place → Set ℕ) (bucket : ℕ → ℕ) (jitter a b : ℕ) (K : S.Place → ℕ)
    (hcs : Accounting.CausalStrategy R ℓ) (hdl : Accounting.DeclaredLinks R S.placeEnds S.linkβ link)
    (hT : Accounting.TrustedUnits R 𝒱 (S.loc (S.certifiersOf i)) spec record bucket jitter)
    (hsec : Accounting.Secrets R (S.loc (S.certifiersOf i))
      (S.loc (S.proverUnder i ∪ S.challengersOf pinned i)) a)
    (hK : Accounting.RecordsWithin R (S.loc (S.certifiersOf i)) record a b K) :
    (Set.range fun r =>
        (R r).acts (S.loc (S.proverUnder i ∪ S.challengersOf pinned i) ∪ S.loc (S.certifiersOf i))ᶜ
          (Set.Ico a b)).encard ≤
      ((2 ^ ((∑ y ∈ S.loc (S.certifiersOf i), K y) +
        jitter * ((Finset.Ico a b).image bucket).card * (S.loc (S.certifiersOf i)).card +
        Accounting.uncertifiedCapacity S.placeEnds S.linkβ (S.loc (S.certifiersOf i))
          (S.loc (S.proverUnder i ∪ S.challengersOf pinned i)) * (b - a)) : ℕ) : ℕ∞) := by
  exact Accounting.leak_le_sum_K R ℓ S.placeEnds S.linkβ link 𝒱 (S.loc (S.certifiersOf i))
    (S.loc (S.proverUnder i ∪ S.challengersOf pinned i)) spec record bucket jitter a b K hcs hdl hT hsec hK

end Verity.Site

namespace Verity.Registration.Example

/-- **Registration's coverage is not the model's at a node**: on the example onsite site, whose one entry is valid
(`onsite_valid`), the registered links with exactly one end in the node `node0` and no certifier at an end are the
memory challenger's link alone, at a positive capacity. -/
theorem chal0_link_uncertified :
    (((Site.ofLog [onsite]).links.filter fun k =>
        decide (k.ends.1 ∈ ["node0", "gpu0", "hbm0", "gpu1", "hbm1", "dram0", "nvme0"] ↔
          k.ends.2 ∉ ["node0", "gpu0", "hbm0", "gpu1", "hbm1", "dram0", "nvme0"]) &&
        !decide ((Site.ofLog [onsite]).Certified k)).map fun k => (k.id, k.β)) = [("chal0-link", 100000)] := by
  decide +kernel

/-- The places at or under `node0` on the example onsite site, two levels down, are the seven the two statements
beside this one list by hand. -/
theorem node0_places :
    ((((Site.ofLog [onsite]).locs.filter fun x =>
          x.id == "node0" || x.parent == some "node0" ||
            (x.parent.bind fun p => ((Site.ofLog [onsite]).loc? p).bind (·.parent)) == some "node0").map
          (·.id)).all (fun j => decide (j ∈ ["node0", "gpu0", "hbm0", "gpu1", "hbm1", "dram0", "nvme0"])) &&
      ["node0", "gpu0", "hbm0", "gpu1", "hbm1", "dram0", "nvme0"].all fun j =>
        decide (j ∈ ((Site.ofLog [onsite]).locs.filter fun x =>
          x.id == "node0" || x.parent == some "node0" ||
            (x.parent.bind fun p => ((Site.ofLog [onsite]).loc? p).bind (·.parent)) == some "node0").map
          (·.id))) = true := by
  decide +kernel

/-- With the challenger read as one of the auditor's devices beside the certifier, no registered link with exactly
one end in the node `node0` of the example onsite site is without one at an end. -/
theorem node0_links_at_auditor :
    (((Site.ofLog [onsite]).links.filter fun k =>
        decide (k.ends.1 ∈ ["node0", "gpu0", "hbm0", "gpu1", "hbm1", "dram0", "nvme0"] ↔
          k.ends.2 ∉ ["node0", "gpu0", "hbm0", "gpu1", "hbm1", "dram0", "nvme0"]) &&
        !(((Site.ofLog [onsite]).kind? k.ends.1).any Site.Kind.auditor ||
          ((Site.ofLog [onsite]).kind? k.ends.2).any Site.Kind.auditor)).map fun k => k.id) = [] := by
  decide +kernel

/-- The prover's places at or under `i`, read two levels down the registered tree: on the sites below, the whole of a
host, and of a site whose hosts hold only memory. -/
def proverPlaces (S : Site) (i : String) : Finset String :=
  (((S.locs.filter fun x => x.id == i || x.parent == some i ||
      (x.parent.bind fun p => (S.loc? p).bind (·.parent)) == some i).map (·.id)).filter fun j =>
    (S.kind? j).all fun k => !k.auditor).toFinset

/-- **On the example onsite site the challenger's link leaves the sum.** At `node0`: with the certifier and the
pinned challenger as the covering set, no registered link is uncovered; with the challenger unpinned, so out of the
covering set, its link is uncovered, at its capacity; and with the challenger a sink inside the location and the
certifier alone covering, none is. -/
theorem onsite_node0_covered :
    ((Site.ofLog [onsite]).uncovered {"cert0", "chal0"} (proverPlaces (Site.ofLog [onsite]) "node0")).map
        (fun k => (k.id, k.β)) = [] ∧
      ((Site.ofLog [onsite]).uncovered {"cert0"} (proverPlaces (Site.ofLog [onsite]) "node0")).map
        (fun k => (k.id, k.β)) = [("chal0-link", 100000)] ∧
      ((Site.ofLog [onsite]).uncovered {"cert0"} (proverPlaces (Site.ofLog [onsite]) "node0" ∪ {"chal0"})).map
        (fun k => (k.id, k.β)) = [] := by
  decide +kernel

/-- The same on the model's term: at `node0` of the example onsite site, `β°` under the certifier and the pinned
challenger is 0. -/
theorem onsite_node0_capacity :
    Accounting.uncertifiedCapacity (Site.ofLog [onsite]).placeEnds (Site.ofLog [onsite]).linkβ
        ((Site.ofLog [onsite]).loc {"cert0", "chal0"})
        ((Site.ofLog [onsite]).loc (proverPlaces (Site.ofLog [onsite]) "node0")) = 0 := by
  have hends : ∀ k ∈ (Site.ofLog [onsite]).links,
      k.ends.1 ∈ (Site.ofLog [onsite]).places ∧ k.ends.2 ∈ (Site.ofLog [onsite]).places := by
    decide +kernel
  rw [Site.uncertified_registered_eq (Site.ofLog [onsite]) _ _ hends]
  decide +kernel

/-- Two hosts with a link `n01` between them, each with its memory, the second under the site `under1`; the first
host's uplink split by a certifier; one memory challenger linked to both hosts. -/
def hosts (under1 : String) (sites : List Site.Loc) : Entry := .site "genesis"
  { locs := sites ++
      [{ id := "node0", kind := .host, parent := some "site0" },
       { id := "node1", kind := .host, parent := some under1 },
       memory "dram0" "node0" .dram (2 ^ 30), memory "dram1" "node1" .dram (2 ^ 30),
       { id := "cert0", kind := .certifier, parent := some "site0", owner := .auditor, key := "key0" },
       { id := "chal0", kind := .challenger, parent := some "site0", owner := .auditor }]
    links := [⟨"uplink0", ("node0", "cert0"), 100000, 100000⟩, ⟨"uplink0-out", ("cert0", Site.outside), 100000, 0⟩,
      ⟨"chal0-link", ("chal0", "node0"), 100000, 1000000⟩, ⟨"chal0-link1", ("chal0", "node1"), 100000, 1000000⟩,
      ⟨"n01", ("node0", "node1"), 7, 1⟩] }

/-- Both hosts under one site. -/
def oneNode : Entry := hosts "site0" [{ id := "site0", kind := .site }]

/-- The hosts under two sites. -/
def twoNodes : Entry := hosts "site1" [{ id := "site0", kind := .site }, { id := "site1", kind := .site }]

/-- Registration accepts both for the onsite composite: a link between two hosts is no boundary link of its check, so
no certifier is asked on it. -/
theorem hosts_valid :
    Valid digests auditor .onsite [oneNode] ∧ Valid digests auditor .onsite [twoNodes] := by
  decide +kernel

/-- **A link between two hosts with no auditor's device at either end is charged, inside one node.** Both hosts under
one site. At the host `node0`, the link `n01` is the one uncovered link, at its capacity, with the challenger in the
covering set. With the challenger a sink instead, its link to the other host is uncovered too: a challenger two hosts
share is sealed at neither. At the site, which holds both hosts, `n01` crosses no boundary and nothing is uncovered.
-/
theorem oneNode_charged :
    ((Site.ofLog [oneNode]).uncovered {"cert0", "chal0"} (proverPlaces (Site.ofLog [oneNode]) "node0")).map
        (fun k => (k.id, k.β)) = [("n01", 7)] ∧
      ((Site.ofLog [oneNode]).uncovered {"cert0"} (proverPlaces (Site.ofLog [oneNode]) "node0" ∪ {"chal0"})).map
        (fun k => (k.id, k.β)) = [("chal0-link1", 100000), ("n01", 7)] ∧
      ((Site.ofLog [oneNode]).uncovered {"cert0", "chal0"} (proverPlaces (Site.ofLog [oneNode]) "site0")).map
        (fun k => (k.id, k.β)) = [] := by
  decide +kernel

/-- **And across two nodes.** The hosts under two sites: at the host `node0` and at its site alike, `n01` is the one
uncovered link, at its capacity. -/
theorem twoNodes_charged :
    ((Site.ofLog [twoNodes]).uncovered {"cert0", "chal0"} (proverPlaces (Site.ofLog [twoNodes]) "node0")).map
        (fun k => (k.id, k.β)) = [("n01", 7)] ∧
      ((Site.ofLog [twoNodes]).uncovered {"cert0", "chal0"} (proverPlaces (Site.ofLog [twoNodes]) "site0")).map
        (fun k => (k.id, k.β)) = [("n01", 7)] := by
  decide +kernel

/-- The same on the model's term: at the host `node0` with both hosts under one site, `β°` under the certifier and the
pinned challenger is the link `n01`'s capacity. -/
theorem oneNode_capacity :
    Accounting.uncertifiedCapacity (Site.ofLog [oneNode]).placeEnds (Site.ofLog [oneNode]).linkβ
        ((Site.ofLog [oneNode]).loc {"cert0", "chal0"})
        ((Site.ofLog [oneNode]).loc (proverPlaces (Site.ofLog [oneNode]) "node0")) = 7 := by
  have hends : ∀ k ∈ (Site.ofLog [oneNode]).links,
      k.ends.1 ∈ (Site.ofLog [oneNode]).places ∧ k.ends.2 ∈ (Site.ofLog [oneNode]).places := by
    decide +kernel
  rw [Site.uncertified_registered_eq (Site.ofLog [oneNode]) _ _ hends]
  decide +kernel

end Verity.Registration.Example
