module

public import Definitions.Accounting.Guarantees.RegisteredSite
public import Definitions.Accounting.RecordsToy
public import Proofs.Core.Site
import Mathlib.Data.Fintype.Card
import Mathlib.Data.Set.Card

/-!
# `LeakPartsRegistered`'s premises hold somewhere

* `Records.Toy.recordsWithin`: over the toy's four runs a device's record takes at most two values, the size of its
  one-bit format (`encard_range_bool_le`).
* `LeakPartsRegistered_honest`: at the example onsite log, which is valid, and the subtree at its host `node0` with no
  memory challenger a trusted device, the toy's strategy at the certifier's place has each trusted device's record
  within one bit, and the subtree has a trusted device. So the guarantee's two scope premises, `Registration.Valid` and
  `RecordsWithin`, hold together at an instance. It does not state the limit there: the instance is not shown to
  meet the guarantee's four physical premises.
-/

@[expose] public section

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
    (run y b).seen '' ↑(((run y b).record {z} cells).filter fun g => 0 ≤ (run y b).A.t g ∧ (run y b).A.t g < 1))
  rintro _ ⟨r, rfl⟩
  exact ⟨r.1, rfl⟩

end Accounting.Records.Toy

namespace Accounting.SecurityProofs

open Verity Accounting.Records

/-- **`LeakPartsRegistered`'s honest lemma**: the example onsite log is valid; at its Site and the subtree at `node0`,
with no memory challenger a trusted device, the toy's strategy at the certifier's place has each trusted device's
record over `[0, 1)` within one bit; and the subtree has a trusted device, its certifier. -/
theorem LeakPartsRegistered_honest :
    Registration.Valid Registration.Example.digests Registration.Example.auditor .onsite
        [Registration.Example.onsite] ∧
      RecordsWithin (Toy.strategy ((Site.ofLog [Registration.Example.onsite]).place "cert0"))
        ((Site.ofLog [Registration.Example.onsite]).placesIn
          ((Site.ofLog [Registration.Example.onsite]).certifiersOf "node0" ∪ ∅)) Toy.cells 0 1 (fun _ => 1) ∧
      ((Site.ofLog [Registration.Example.onsite]).placesIn
        ((Site.ofLog [Registration.Example.onsite]).certifiersOf "node0" ∪ ∅)).Nonempty := by
  refine ⟨Registration.Example.onsite_valid, Toy.recordsWithin _ _, ?_⟩
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

end Accounting.SecurityProofs
