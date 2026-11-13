module

public import Definitions.Core.Site
public import Definitions.Accounting.Guarantees.Bridges
public import Proofs.Accounting.Actual.Physical
public import Proofs.Accounting.Actual.Bridges
public import Proofs.Core.Site
public import Mathlib.Data.String.Basic

/-!
# Scratch: a node's egress and leak limits read off the registered tree

A scratch module, never landing: the statements of the two "limits add over parts" theorems for links, with the
definitions each needs that the tree does not hold. Bodies marked `sorry` are statements only.

* The node: `Site.Child`, `Site.places`, `Site.under`, `Site.proverUnder`, `Site.certifiersOf`.
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

/-- The registered links as the model's, by their position in the Site: link `e`'s two ends. -/
def linkEnds (e : Fin S.links.length) : String × String := (S.links.get e).ends

/-- The registered links as the model's: link `e`'s capacity each way. -/
def linkβ (e : Fin S.links.length) : ℕ := (S.links.get e).β

/-- **The registered egress limit of node `i`**, a tick: the model's `β_X` at the registered links and the node. -/
noncomputable def egressLimit (i : String) : ℕ := Accounting.linkCapacity S.linkEnds S.linkβ (S.under i)

/-- The registered links with exactly one end in `X`. -/
def crossing (X : Finset String) : List Link :=
  S.links.filter fun k => decide (k.ends.1 ∈ X ↔ k.ends.2 ∉ X)

/-- **(1) A node's egress is at most its registered limit**: over any ticks `[a, b)`, what leaves node `i` of the
registered tree is at most the capacity of the registered links with exactly one end in it, a tick. -/
theorem egress_le_registered (i : String) (A : Accounting.Placed String)
    (link : Fin A.N → Fin A.N → Fin S.links.length) (h : A.Links S.linkEnds S.linkβ link) (a b : ℕ) :
    A.egress (S.under i) (Set.Ico a b) ≤ S.egressLimit i * (b - a) :=
  Accounting.SecurityProofs.Actual.egress_le A h (S.under i) a b

/-- **The limit is read off the registered links**: the sum of `β` over the registered links with exactly one end in
the node, a link between two places of the node uncounted. -/
theorem egressLimit_eq (i : String) : S.egressLimit i = ((S.crossing (S.under i)).map (·.β)).sum := by
  sorry

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

/-- **Limits add over parts, at most**: the capacity across the boundary of a union of parts is at most the parts'
capacities summed. -/
theorem linkCapacity_biUnion_le (ends : E → L × L) (β : E → ℕ) (P : Finset ι) (X : ι → Finset L) :
    linkCapacity ends β (P.biUnion X) ≤ ∑ p ∈ P, linkCapacity ends β (X p) := by
  sorry

open Classical in
/-- The capacity of the links between two different parts. -/
noncomputable def innerCapacity (ends : E → L × L) (β : E → ℕ) (P : Finset ι) (X : ι → Finset L) : ℕ :=
  ∑ e ∈ univ.filter (fun e => ∃ p ∈ P, ∃ q ∈ P, p ≠ q ∧ (ends e).1 ∈ X p ∧ (ends e).2 ∈ X q), β e

/-- **Limits add over parts, exactly**: for disjoint parts, the parts' capacities summed are the whole's and each
link between two parts twice, once at each end. -/
theorem linkCapacity_biUnion (ends : E → L × L) (β : E → ℕ) (P : Finset ι) (X : ι → Finset L)
    (hd : (P : Set ι).PairwiseDisjoint X) :
    linkCapacity ends β (P.biUnion X) + 2 * innerCapacity ends β P X = ∑ p ∈ P, linkCapacity ends β (X p) := by
  sorry

end Parts

/-- **Each certifier's record is within its `K`**: across the strategy, what certifier `y` of `cert` places in its
record during `[a, b)` takes at most `2 ^ K y` values. The parts' limits, which `leak_le_sum_K` adds. -/
def RecordsWithin {Ω L : Type*} (S : Ω → Actual.Run L) (cert : Finset L) (record : L → Set ℕ) (a b : ℕ)
    (K : L → ℕ) : Prop :=
  ∀ y ∈ cert, (Set.range fun r =>
    (S r).seen '' ↑(((S r).record {y} record).filter fun g => a ≤ (S r).A.t g ∧ (S r).A.t g < b)).encard ≤
      ((2 ^ K y : ℕ) : ℕ∞)

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
  sorry

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

end Accounting

namespace Verity.Site

open Finset

variable (S : Site)

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
  sorry

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

end Verity.Registration.Example
