module

public import Proofs.Flock.SpotFlock.RunsAs.OwnReadsForm
public import Proofs.Flock.SpotFlock.RunsAs.StampedReads
public import Proofs.Properties.Verity.SampledTwo.RecordNames
public import Proofs.Flock.SpotFlock.AddressFacts.RowKind
public import Proofs.Flock.SpotFlock.OwnReads
public import Proofs.Flock.SpotFlock.Sessions
public import Proofs.Properties.Spot

/-!
# The reads an interior record's checks leave out, with no port whose root prints as null

`accepted_leftOut` leaves one case beside `HeldRead`: a port of the interior record whose root prints as null, which
is how a value the verdicts report under two roots is reported. A record that passed the registration check has no
such port where the verifier lists each port once: R3 holds each of the verifier's ports to a hex root at the record's
last root naming it (`Properties.Spot.R3`), `Record.byPort` keeps for each name the last root naming it
(`rootFor_of_byPort`), and a record that passed names exactly the verifier's ports (`RecordNames.recordOk_of_check`),
so each of the record's ports has a root that is a string (`byPort_root_str`), which `jeq` never holds to null.

The registration check does not ask that the verifier list each port once: a verifier listing one port twice passes a
record with a second port R3 reads no root of. So the fact is stated under that hypothesis on the interior
registration's parameters (`accepted_leftOut_held`, `leftOut_of_check_held`, `ownReads_form_held`).
-/

@[expose] public section

namespace SpotFlock.RunsAs

open Lean (Json)

theorem jeq_str_self (s : String) : Spot.jeq (.str s) (.str s) = true := by
  have hs : Flock.canon (.str s) = .ok (Flock.canonStr s) := by rw [Flock.canon]; rfl
  unfold Spot.jeq
  rw [hs]
  simp

/-- `jeq` holds nothing but the string to a string. -/
theorem jeq_str_ne {a : Json} {p : String} (h : a ≠ .str p) : Spot.jeq a (.str p) = false := by
  cases hj : Spot.jeq a (.str p) with
  | false => rfl
  | true => exact absurd (str_of_jeq (jeq_symm hj)) h

/-- The fold `Record.byPort` runs keeps, for each name, the last root naming it. -/
theorem foldl_last :
    ∀ (l pre : List Json) (acc : Array (String × Json)),
      (∀ p x, (p, x) ∈ acc → pre.reverse.find? (fun r => Spot.jeq (Spot.getD r "port") (.str p)) = some x) →
      ∀ p x, (p, x) ∈ l.foldl (fun (acc : Array (String × Json)) r => match Spot.getD r "port" with
          | .str p => (acc.filter fun (q : String × Json) => q.1 != p).push (p, r)
          | _ => acc) acc →
        (pre ++ l).reverse.find? (fun r => Spot.jeq (Spot.getD r "port") (.str p)) = some x
  | [], pre, acc, hacc, p, x, h => by
    rw [List.append_nil]
    exact hacc p x h
  | a :: l, pre, acc, hacc, p, x, h => by
    rw [List.foldl_cons] at h
    have hstep := foldl_last l (pre ++ [a]) _ ?_ p x h
    · rwa [List.append_assoc, List.singleton_append] at hstep
    · intro p' x' hx'
      rw [List.reverse_append, List.reverse_singleton, List.singleton_append, List.find?_cons]
      cases hpa : Spot.getD a "port" with
      | str q =>
        rw [hpa] at hx'
        rcases Array.mem_push.1 hx' with hx' | hx'
        · obtain ⟨hin, hne⟩ := Array.mem_filter.1 hx'
          have hne' : q ≠ p' := fun e => by simp [e] at hne
          rw [jeq_str_ne (fun e => hne' (Json.str.inj e))]
          exact hacc p' x' hin
        · obtain ⟨rfl, rfl⟩ := Prod.mk.inj hx'
          rw [jeq_str_self]
      | null | bool _ | num _ | arr _ | obj _ =>
        rw [hpa] at hx'
        rw [jeq_str_ne (by simp)]
        exact hacc p' x' hx'

/-- **A port of `Record.byPort` is the record's root for its name**, the last root naming it. -/
theorem rootFor_of_byPort {rec : Json} {p : String} {x : Json} (hx : (p, x) ∈ Spot.Record.byPort rec) :
    Properties.Spot.rootFor rec p = some x := by
  unfold Spot.Record.byPort at hx
  unfold Properties.Spot.rootFor Properties.Spot.rootsOf
  cases hr : Spot.getD rec "roots" with
  | arr rs =>
    rw [hr] at hx
    try dsimp only at hx ⊢
    rw [← Array.foldl_toList] at hx
    rw [← Array.find?_toList, Array.toList_reverse]
    have := foldl_last rs.toList [] #[] (fun p x h => by simp at h) p x hx
    rwa [List.nil_append] at this
  | null | bool _ | num _ | str _ | obj _ =>
    rw [hr] at hx
    simp at hx

/-- **Each port of a record that passed the registration check has a root that is a string**, where the verifier
lists each port once: the record names exactly the verifier's ports, and R3 holds each to a hex root. -/
theorem byPort_root_str {rec : Json} {v : Spot.Registration.Verifier} (h : Spot.Registration.check rec v = #[])
    (hn : (v.ports.map (·.1)).toList.Nodup) {p : String} {x : Json} (hx : (p, x) ∈ Spot.Record.byPort rec) :
    ∃ s, Spot.getD x "root" = .str s := by
  obtain ⟨law, -, -, h3, -⟩ := (Properties.Spot.registration_nil rec v).1 h
  obtain ⟨-, -, -, -, -, -, hroot⟩ := h3
  obtain ⟨⟨ps, hps, -, hmem⟩, -⟩ := recordOk_spec (RecordNames.recordOk_of_check h hn)
  have hp : p ∈ ps := (AddressRow.byPort_isSome_iff hps p).1 (find?_isSome_of ⟨(p, x), hx, rfl⟩)
  have hp' : p ∈ (v.ports.map (·.1)).toList := (hmem p).1 hp
  rw [Array.toList_map, List.mem_map] at hp'
  obtain ⟨pw, hpw, hpp⟩ := hp'
  have hpp' : pw.1 = p := hpp
  subst hpp'
  obtain ⟨⟨s, hs, -⟩, -, -⟩ := hroot pw (Array.mem_toList_iff.1 hpw) x (rootFor_of_byPort hx)
  exact ⟨s, hs⟩

/-- **Each read left out is one the second stage's statement check held**, where the interior registration's
parameters list each port once: `accepted_leftOut` without the port whose root prints as null, which a record that
passed the registration check under such parameters doesn't have (`byPort_root_str`). -/
theorem accepted_leftOut_held {i : Spot.Run.Input} {it : Spot.Run.Interior} {j : Json} (ha : AcceptedTwo i it j)
    (hports : (it.params.verifier.ports.map (·.1)).toList.Nodup) {opened : Array (String × Json)}
    (ho : Spot.Record.ownRegistered it.record (Spot.Record.sessionRegistered i.verdicts) = some opened)
    {v : String} {r : Json} (hv : (v, r) ∈ opened)
    (hno : (v, r) ∉ (Spot.Record.lessOwnReads it.record (some opened)).getD #[]) :
    ∃ x, (v, x) ∈ Spot.Record.byPort it.record ∧ Spot.jeq (Spot.getD x "root") r = true ∧ HeldRead i it v r := by
  obtain ⟨x, hx, hxr, hn | hh⟩ := accepted_leftOut ha ho hv hno
  · obtain ⟨n, rule, law, workTable, resolved, draw, rest, read, ns, -, -, -, -, -, -, -, -, -, hchk⟩ :=
      accepted_registered ha
    obtain ⟨s, hs⟩ := byPort_root_str (Array.isEmpty_iff.1 hchk) hports hx
    rw [hs] at hn
    cases str_of_jeq hn
  · exact ⟨x, hx, hxr, hh⟩

/-- The run of record's interior has the form's interior parameters. -/
theorem inputOf_interior_params (p : VStar.Recursive.SampledTwo.Params) (c : Spot.OneStage.Commitment)
    (ξ ξ₂ : ByteArray) (e : VStar.Recursive.SampledTwo.Evidence) (received₂ : String) (vs : Array (String × Json))
    {it : Spot.Run.Interior} (h : (VStar.Recursive.SampledTwo.inputOf p c ξ ξ₂ e received₂ vs).interior = some it) :
    it.params = p.interior := by
  unfold VStar.Recursive.SampledTwo.inputOf at h
  cases h
  rfl

/-- **At the form's accept, each read the interior record's checks leave out is one the second-stage statement check
held**, where the form's interior parameters list each port once: `leftOut_of_check` without the port whose root
prints as null. -/
theorem leftOut_of_check_held {form : Bool} {p : VStar.Recursive.SampledTwo.Params} {c : Spot.OneStage.Commitment}
    {ξ : ByteArray} {e : VStar.Recursive.SampledTwo.Evidence} {bs pub prof}
    (hp : ∃ grain replay k, p.rule = .twoStage grain replay k)
    (hports : (p.interior.verifier.ports.map (·.1)).toList.Nodup)
    (h : VStar.Recursive.SampledTwo.checkBlobsAtWith form p c ξ e bs = .accept pub prof) :
    ∃ c₂ ξ₂ r i it, VStar.Recursive.SampledTwo.inputAt p c ξ e bs = .ok (c₂, ξ₂, r, i) ∧ i.interior = some it ∧
      ∀ opened v root,
        Spot.Record.ownRegistered it.record (Spot.Record.sessionRegistered i.verdicts) = some opened →
        (v, root) ∈ opened → (v, root) ∉ (Spot.Record.lessOwnReads it.record (some opened)).getD #[] →
        ∃ x, (v, x) ∈ Spot.Record.byPort it.record ∧ Spot.jeq (Spot.getD x "root") root = true ∧
          HeldRead i it v root := by
  obtain ⟨-, c₂, ξ₂, r, i, it, run, hin, ha, -⟩ := acceptedTwo_of_check hp h
  have hit : it.params = p.interior := by
    obtain ⟨-, -, -, -, vs, received₂, j₁, j₂, -, rfl, -, -, -⟩ := VStar.Recursive.SampledTwo.inputAt_ok hin
    exact inputOf_interior_params _ _ _ _ _ _ _ ha.interior
  have hports' : (it.params.verifier.ports.map (·.1)).toList.Nodup := by
    rw [hit]
    exact hports
  exact ⟨c₂, ξ₂, r, i, it, hin, ha.interior, fun _ _ _ ho hv hno => accepted_leftOut_held ha hports' ho hv hno⟩

/-- **At the form's accept, each read its last step's record checks leave out is one the second-stage statement
check held**, at the last step's own reading (`c₂.record` and the sessions' own verdicts `r.out`), where the form's
interior parameters list each port once and the interior record the run of record reads is the record the form parsed
(`RecordReadSame`). -/
theorem ownReads_form_held (hrec : RecordReadSame) {form : Bool} {p : VStar.Recursive.SampledTwo.Params}
    {c : Spot.OneStage.Commitment} {ξ : ByteArray} {e : VStar.Recursive.SampledTwo.Evidence} {bs pub prof}
    (hp : ∃ grain replay k, p.rule = .twoStage grain replay k)
    (hports : (p.interior.verifier.ports.map (·.1)).toList.Nodup)
    (h : VStar.Recursive.SampledTwo.checkBlobsAtWith form p c ξ e bs = .accept pub prof) :
    ∃ c₂ ξ₂ r i it, VStar.Recursive.SampledTwo.inputAt p c ξ e bs = .ok (c₂, ξ₂, r, i) ∧ i.interior = some it ∧
      ∀ opened v root,
        Spot.Record.ownRegistered c₂.record (Spot.Record.sessionRegistered r.out) = some opened →
        (v, root) ∈ opened → (v, root) ∉ (Spot.Record.lessOwnReads c₂.record (some opened)).getD #[] →
        ∃ x, (v, x) ∈ Spot.Record.byPort c₂.record ∧ Spot.jeq (Spot.getD x "root") root = true ∧
          HeldRead i it v root := by
  obtain ⟨c₂, ξ₂, r, i, it, hin, hint, H⟩ := leftOut_of_check_held hp hports h
  have hr : it.record = c₂.record := hrec p c ξ e bs c₂ ξ₂ r i it hin hint
  have hs : Spot.Record.sessionRegistered i.verdicts = Spot.Record.sessionRegistered r.out := inputAt_reported hin
  refine ⟨c₂, ξ₂, r, i, it, hin, hint, fun opened v root ho hv hno => ?_⟩
  rw [← hr, ← hs] at ho
  rw [← hr] at hno ⊢
  exact H opened v root ho hv hno

end SpotFlock.RunsAs
