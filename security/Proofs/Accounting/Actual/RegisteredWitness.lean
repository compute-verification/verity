module

public import Definitions.Accounting.Guarantees.RegisteredSite
public import Definitions.Accounting.RecordsToy
public import Proofs.Core.Site
public import Proofs.Core.SiteSubtree
import Mathlib.Data.Fintype.Card
import Mathlib.Data.Set.Card

/-!
# `LeakPartsRegistered`'s premises hold somewhere, and its limit is not true of every strategy

* `Records.Toy.recordsWithin`: over the toy's four runs a device's record takes at most two values, the size of its
  one-bit format (`encard_range_bool_le`).
* `LeakPartsRegistered_honest`: at the example onsite log, which is valid, and the subtree at its host `node0` with no
  memory challenger a trusted device, the toy's strategy, its one gate at the certifier's place, is placed at the
  places a link may end at and has each trusted device's record within one bit, and the subtree has a trusted device.
  So the guarantee's three scope premises, `Registration.Valid`, `AtLinkEnds` and `RecordsWithin`, hold together at
  an instance. It does not state the limit there: the instance is not shown to meet the guarantee's four physical
  premises.
* `Records.Toy.acts_injective`, `uncertifiedCapacity_le_sum`, `Records.Toy.outside_exceeds`: a strategy whose one gate
  is at the outside and one bit wider than the links' whole capacity a tick does more there, over one tick, than
  `2 ^ β°` values.
* `LeakPartsRegistered_nontrivial`: that at the example onsite log's Site and the same subtree, every record limit
  `0`: the limit's inequality fails. The strategy varies the outside's own input, so it does not meet `Secrets`.
-/

@[expose] public section

namespace Accounting

/-- The capacity of the uncertified links across a boundary is at most the links' whole capacity. -/
theorem uncertifiedCapacity_le_sum {L E : Type*} [Fintype E] (ends : E → L × L) (β : E → ℕ) (cert X : Finset L) :
    uncertifiedCapacity ends β cert X ≤ ∑ e, β e := by
  unfold uncertifiedCapacity
  exact Finset.sum_le_sum_of_subset (Finset.subset_univ _)

end Accounting

namespace Accounting.Records.Toy

open Verity

/-- A function of one bit takes at most two values. -/
theorem encard_range_bool_le {α : Type*} (F : Bool → α) : (Set.range F).encard ≤ ((2 ^ 1 : ℕ) : ℕ∞) := by
  have h : (Set.univ : Set Bool).encard = ((2 ^ 1 : ℕ) : ℕ∞) := by
    rw [Set.encard_univ, ENat.card_eq_coe_fintype_card, Fintype.card_bool, pow_one]
  rw [← Set.image_univ, ← h]
  exact Set.encard_image_le F Set.univ

/-- **The toy's records are within one bit**: over the four runs, what any leaf places in its record cells during
`[0, 1)` is a function of the first secret bit, so it takes at most `2 ^ 1` values. -/
theorem recordsWithin {L : Type*} [DecidableEq L] (y : L) (cert : Finset L) :
    RecordsWithin (strategy y) cert cells 0 1 (fun _ => 1) := by
  intro z _
  refine le_trans (Set.encard_le_encard ?_) (encard_range_bool_le fun b : Bool =>
    (run y (BitVec.ofBool b)).seen '' ↑(((run y (BitVec.ofBool b)).record {z} cells).filter fun g =>
      0 ≤ (run y (BitVec.ofBool b)).A.t g ∧ (run y (BitVec.ofBool b)).A.t g < 1))
  rintro _ ⟨r, rfl⟩
  exact ⟨r.1, rfl⟩

/-- **Runs whose gates hold different values do different things at the gate's leaf**: over the `2 ^ w` values of a
`w`-bit gate at `y`, what a location that holds `y` does during `[0, 1)` tells the value. -/
theorem acts_injective {L : Type*} [DecidableEq L] (y : L) (X : Finset L) (hy : y ∈ X) (w : ℕ) :
    Function.Injective fun r : Fin (2 ^ w) => (run y (BitVec.ofFin r)).acts X (Set.Ico 0 1) := by
  intro r r' h
  have h1 : ((run y (BitVec.ofFin r)).acts X (Set.Ico 0 1)).1 = ((run y (BitVec.ofFin r')).acts X (Set.Ico 0 1)).1 :=
    congrArg Prod.fst h
  have hm : (run y (BitVec.ofFin r)).seen (0 : Fin 1) ∈ ((run y (BitVec.ofFin r)).acts X (Set.Ico 0 1)).1 :=
    ⟨(0 : Fin 1), hy, ⟨Nat.le_refl 0, Nat.zero_lt_one⟩, rfl⟩
  rw [h1] at hm
  obtain ⟨g, -, -, hg⟩ := hm
  exact Fin.ext (congrArg (fun p : ℕ × L × ℕ × ℕ × ℕ => p.2.2.2.2) hg)

/-- **A strategy at the outside does more than the limit**: at a Site none of whose locations has the outside's id,
and a subtree at a place other than the outside, with no pinned challenger, every record limit `0` and no jitter, the
strategy whose one gate is at the outside and holds one bit more than the registered links' whole capacity a tick
has the outside do more over `[0, 1)` than the limit's `2 ^ β°` values. -/
theorem outside_exceeds (S : Site) (hno : ∀ x ∈ S.locs, x.id ≠ Site.outside)
    (hk : S.kind? Site.outside ≠ some .certifier) {i : String} (hi : i ≠ Site.outside) :
    ¬ ((Set.range fun r : Fin (2 ^ ((∑ e, S.linkβ e) + 1)) =>
        (run (S.place Site.outside) (BitVec.ofFin r)).acts
          (S.placesIn (S.proverUnder i ∪ (S.challengersOf ∅ i \ ∅)) ∪ S.placesIn (S.certifiersOf i ∪ ∅))ᶜ
          (Set.Ico 0 1)).encard ≤
      ((2 ^ ((∑ _y ∈ S.placesIn (S.certifiersOf i ∪ ∅), 0) +
        0 * ((Finset.Ico (0 : ℕ) 1).image id).card * (S.placesIn (S.certifiersOf i ∪ ∅)).card +
        uncertifiedCapacity S.placeEnds S.linkβ (S.placesIn (S.auditorsOf ∅ i)) (S.placesIn (S.proverUnder i)) *
          (1 - 0)) : ℕ) : ℕ∞)) := by
  intro h
  rw [← Set.image_univ,
    (acts_injective (S.place Site.outside) _ (S.outside_mem_compl hno hk hi) _).encard_image,
    Set.encard_univ, ENat.card_eq_coe_fintype_card, Fintype.card_fin] at h
  have h2 := (Nat.pow_le_pow_iff_right (by decide : 1 < 2)).1 (ENat.natCast_le_natCast.1 h)
  simp only [Finset.sum_const_zero, Nat.zero_mul, Nat.zero_add, Nat.sub_zero, Nat.mul_one] at h2
  exact Nat.not_add_one_le_self _ (h2.trans (uncertifiedCapacity_le_sum _ _ _ _))

end Accounting.Records.Toy

namespace Accounting.SecurityProofs

open Verity Accounting.Records

/-- **`LeakPartsRegistered`'s honest lemma**: the example onsite log is valid; at its Site and the subtree at `node0`,
with no memory challenger a trusted device, the toy's strategy is placed at the places a link may end at, its one
gate at the certifier's place and none at a device or a memory leaf; each trusted device's record over `[0, 1)` is
within one bit; and the subtree has a trusted device, its certifier. -/
theorem LeakPartsRegistered_honest :
    Registration.Valid Registration.Example.digests Registration.Example.auditor .onsite
        [Registration.Example.onsite] ∧
      AtLinkEnds (Site.ofLog [Registration.Example.onsite])
        (Toy.strategy ((Site.ofLog [Registration.Example.onsite]).place "cert0")) ∧
      RecordsWithin (Toy.strategy ((Site.ofLog [Registration.Example.onsite]).place "cert0"))
        ((Site.ofLog [Registration.Example.onsite]).placesIn
          ((Site.ofLog [Registration.Example.onsite]).certifiersOf "node0" ∪ ∅)) Toy.cells 0 1 (fun _ => 1) ∧
      ((Site.ofLog [Registration.Example.onsite]).placesIn
        ((Site.ofLog [Registration.Example.onsite]).certifiersOf "node0" ∪ ∅)).Nonempty := by
  have hend : (Site.ofLog [Registration.Example.onsite]).linkPlace
      ((Site.ofLog [Registration.Example.onsite]).place "cert0") =
        (Site.ofLog [Registration.Example.onsite]).place "cert0" := by decide +kernel
  refine ⟨Registration.Example.onsite_valid, fun _ _ => hend, Toy.recordsWithin _ _, ?_⟩
  have hplace : "cert0" ∈ (Site.ofLog [Registration.Example.onsite]).places := by decide +kernel
  have hkind : (Site.ofLog [Registration.Example.onsite]).kind? "cert0" = some .certifier := by decide +kernel
  have hfar : "cert0" ∈ (Site.ofLog [Registration.Example.onsite]).far "node0" := by decide +kernel
  have hroot : "node0" ∈ (Site.ofLog [Registration.Example.onsite]).places := by decide +kernel
  have hhost : (Site.ofLog [Registration.Example.onsite]).kind? "node0" = some .host := by decide +kernel
  have hnode : "node0" ∈ (Site.ofLog [Registration.Example.onsite]).proverUnder "node0" := by
    classical
    unfold Site.proverUnder Site.under
    refine Finset.mem_filter.2 ⟨Finset.mem_filter.2 ⟨hroot, Relation.ReflTransGen.refl⟩, ?_⟩
    intro k hk
    rw [hhost, Option.mem_def, Option.some.injEq] at hk
    subst hk
    rfl
  have hcert : "cert0" ∈ (Site.ofLog [Registration.Example.onsite]).certifiersOf "node0" := by
    classical
    unfold Site.certifiersOf
    exact Finset.mem_filter.2 ⟨hplace, hkind, "node0", hnode, hfar⟩
  exact ⟨⟨"cert0", hplace⟩, Finset.mem_filter.2 ⟨Finset.mem_univ _, Finset.mem_union_left _ hcert⟩⟩

/-- **`LeakPartsRegistered`'s nontrivial lemma**: the limit is not true of every strategy. At the example onsite
log's Site and the subtree at `node0`, with no pinned challenger, every record limit `0` and no jitter, the strategy
whose one gate is at the outside and holds one bit more than the registered links' whole capacity a tick has the
outside do more over `[0, 1)` than the limit's `2 ^ β°` values. The strategy varies the outside's own input, so the
coins do not reach the window only through the subtree's secrets (`Secrets`). -/
theorem LeakPartsRegistered_nontrivial :
    ¬ ((Set.range fun r : Fin (2 ^ ((∑ e, (Site.ofLog [Registration.Example.onsite]).linkβ e) + 1)) =>
        (Toy.run ((Site.ofLog [Registration.Example.onsite]).place Site.outside) (BitVec.ofFin r)).acts
          ((Site.ofLog [Registration.Example.onsite]).placesIn
              ((Site.ofLog [Registration.Example.onsite]).proverUnder "node0" ∪
                ((Site.ofLog [Registration.Example.onsite]).challengersOf ∅ "node0" \ ∅)) ∪
            (Site.ofLog [Registration.Example.onsite]).placesIn
              ((Site.ofLog [Registration.Example.onsite]).certifiersOf "node0" ∪ ∅))ᶜ
          (Set.Ico 0 1)).encard ≤
      ((2 ^ ((∑ _y ∈ (Site.ofLog [Registration.Example.onsite]).placesIn
            ((Site.ofLog [Registration.Example.onsite]).certifiersOf "node0" ∪ ∅), 0) +
        0 * ((Finset.Ico (0 : ℕ) 1).image id).card * ((Site.ofLog [Registration.Example.onsite]).placesIn
          ((Site.ofLog [Registration.Example.onsite]).certifiersOf "node0" ∪ ∅)).card +
        uncertifiedCapacity (Site.ofLog [Registration.Example.onsite]).placeEnds
          (Site.ofLog [Registration.Example.onsite]).linkβ
          ((Site.ofLog [Registration.Example.onsite]).placesIn
            ((Site.ofLog [Registration.Example.onsite]).auditorsOf ∅ "node0"))
          ((Site.ofLog [Registration.Example.onsite]).placesIn
            ((Site.ofLog [Registration.Example.onsite]).proverUnder "node0")) * (1 - 0)) : ℕ) : ℕ∞)) :=
  Toy.outside_exceeds (Site.ofLog [Registration.Example.onsite]) (by decide +kernel) (by decide +kernel)
    (by decide)

end Accounting.SecurityProofs
