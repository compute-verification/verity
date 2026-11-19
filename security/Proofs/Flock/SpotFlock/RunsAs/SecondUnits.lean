module

public import Proofs.Properties.Spot
public import VStarRecursive.SampledTwo

/-!
# What the two-stage form's last step accepts: the units, at the reads it hands the record's checks

`Spot.OneStage.check` hands the record's checks (`Spot.Record.checks`) the reads the verdicts report for the
registration's own values less the reads of its own ports at their own roots, with those of the window values its
params name less the reads at their window entries' roots (`openedRoots`), and `UnitsProved` says what those checks
hold. The two-stage form's last step is the same check at a sampling design it is handed (`Spot.OneStage.checkIn`), so
it hands the record's checks the same reads.

* `UnitsAt c e opened` is `UnitsProved`'s six clauses with the reads the registered check is handed a parameter;
  `UnitsProved named c e` is it at `openedRoots named c.record e.verdicts` (`unitsProved_at`).
* `checksAt_nil`: none of the record's checks fails, the sample's aside, exactly where `UnitsAt` holds at the reads
  they are handed.
* `checkIn_units`: what `checkIn` accepts at a sampling design it is handed keeps the registration's rules R1–R8, has
  its sample a sample of that sampling design, has `UnitsAt` at the reads `openedRoots` gives at its params' window
  values, and has verdicts that accept.
-/

@[expose] public section

open Lean (Json)
open Spot

namespace Properties.Spot

open OneStage (Params Commitment Evidence Public)

/-- **The units, as the record holds them, at the reads `opened` the registered check is handed**: `UnitsProved`'s
clauses, the last at `opened`: each read in `opened` is of a value the record registers, opened against the record's
root for it (`Record.registeredRoots`). -/
def UnitsAt (c : Commitment) (e : Evidence) (opened : Option (Array (String × Json))) : Prop :=
  e.instances = (Draw.proved e.draw).size ∧ e.provedUnits.map (fun n : Nat => (n : Int)) = Draw.proved e.draw ∧
  Receipt c e ∧
  (∀ served, e.served = some served →
    (served.map fun n : Nat => (n : Int)).qsort (· < ·) = (sampledInts e.draw).qsort (· < ·)) ∧
  (∀ roots, e.sessionRoots = some roots → entries roots ≠ #[] ∧
    ∀ pr ∈ entries roots, ∃ reg, (Record.byPort c.record).find? (·.1 == pr.1) = some reg ∧ jeq (getD reg.2 "root") pr.2) ∧
  OpenedRegistered (opened.getD #[]) (Record.registeredRoots c.record)

/-- `UnitsProved` is `UnitsAt` at the reads the record's checks are handed (`openedRoots`). -/
theorem unitsProved_at (named : Array String) (c : Commitment) (e : Evidence) :
    UnitsProved named c e ↔ UnitsAt c e (openedRoots named c.record e.verdicts) := Iff.rfl

/-- **None of the record's checks fails, the sample's aside, exactly where the units hold at the reads the checks
are handed**: `recordChecks_nil` at any reads `opened`. -/
theorem checksAt_nil (c : Commitment) (e : Evidence) (opened : Option (Array (String × Json))) :
    ((Record.checks c.record e.receipt e.draw Json.null
        (match getD e.draw "units" with | .arr us => us.filterMap int? | _ => #[])
        (e.provedUnits.map Lean.toJson) (Lean.toJson e.instances) (Flock.hexOf c.digest)
        (e.served.map (·.map fun (n : Nat) => (n : Int))) (e.sessionRoots.map entries) opened).filter
      fun (n, c) => n != "draw" && !c.codes.isEmpty).map (·.1) = #[] ↔ UnitsAt c e opened := by
  simp only [Array.map_eq_empty_iff, Array.filter_eq_empty_iff]
  have hf : ∀ cs : Array (String × Check),
      (∀ x ∈ cs, ¬(match x with | (n, c) => n != "draw" && !c.codes.isEmpty) = true) ↔ ∀ x ∈ cs, Fails x = false := by
    intro cs; simp [Fails]
  rw [hf, checks_ok, checkProved_nil]
  unfold UnitsAt Receipt
  simp only [Option.map_eq_some_iff, forall_exists_index, and_imp, forall_apply_eq_imp_iff₂, and_assoc]
  exact Iff.rfl

private theorem accept_of_ite {α β : Type} {b : Bool} {r : String} {x : Core.Outcome α β} {pub : α} {prof : β}
    (h : (if (!b) = true then Core.Outcome.reject r else x) = .accept pub prof) :
    b = true ∧ x = .accept pub prof := by
  cases b
  · simp at h
  · exact ⟨rfl, by simpa using h⟩

/-- **What the check at a sampling design it is handed accepts** (`OneStage.checkIn`, the two-stage form's last step):
the registration keeps R1–R8 against the params' verifier; the sample is a sample of the sampling design
`samplingDesign`, over the registered population; the units hold at the reads the check hands the record's checks
(`UnitsAt` at `openedRoots` of the params' window values); and the verdicts accept. -/
theorem checkIn_units {p : Params} {c : Commitment} {e : Evidence} {samplingDesign : Json}
    {pub : Public} {prof : Core.Integrity}
    (h : OneStage.checkIn p c e (.ok samplingDesign) = .accept pub prof) :
    (∃ samplingDesign', Registered p.verifier c.record samplingDesign') ∧
      Drawn e.draw (getD c.record "population") samplingDesign ∧
      UnitsAt c e (openedRoots p.windowValues c.record e.verdicts) ∧ VerdictsAccept e.verdicts := by
  unfold OneStage.checkIn at h
  obtain ⟨h1, h⟩ := accept_of_ite h
  obtain ⟨h2, h⟩ := accept_of_ite h
  obtain ⟨h3, h⟩ := accept_of_ite h
  refine ⟨(docket_nil _ _).1 (Array.isEmpty_iff.1 h1), (checkSample_nil _ _ _).1 (Array.isEmpty_iff.1 h2),
    ((recordChecks_nil p c e).1 (Array.isEmpty_iff.1 h3)).1, ?_⟩
  rcases hv : OneStage.verdictOf e.verdicts with ⟨⟨⟩, ⟨⟩⟩ | r | r
  · exact (verdictOf_accept _).1 hv
  · rw [hv] at h; cases h
  · rw [hv] at h; cases h

end Properties.Spot
