module

public import Proofs.Flock.SpotFlock.RunsAs.Statement
import all Lean.Data.Json.Basic
import Std.Data.TreeMap.Raw.Lemmas
import Std.Data.TreeMap.Raw.WF

/-!
# The outcome: an interior audit record that reads accepted had round 2's check pass

`Run.run`'s interior audit record is the audit record (`Record.record`, an object built by `Json.mkObj`) with round 2's
check attached (`SecondStage.attach`), and it accepts only where that record's `accepted` reads `true`. Here:

* `failures_allow`, `check_true_of_codes`: round 2's check at the form's switch `b` reports no code only where the
  check with the form allowed reports none, the switch adding the one refusal `two-stage-form`;
* `codes_of_attach`: a well-formed object with a check attached reads accepted only where the check reports no code.
-/

@[expose] public section

namespace SpotFlock.RunsAs

open Lean (Json)

/-- **The form's switch adds one refusal**: round 2's failures at the switch `b` are the failures with the form
allowed, or the form's refusal alone. -/
theorem failures_allow (b : Bool) (own : Spot.SecondStage.Own) (ps : Array Spot.SecondStage.Part) (ss : List Json) :
    Spot.SecondStage.failures b own ps ss = Spot.SecondStage.failures true own ps ss ∨
      Spot.SecondStage.failures b own ps ss = [("two-stage-form", Json.str Spot.SecondStage.FORM_WHY)] := by
  cases b
  · unfold Spot.SecondStage.failures
    simp only [Bool.not_false, Bool.and_true, Bool.not_true, Bool.and_false, Bool.false_eq_true, if_false]
    split
    · exact Or.inr rfl
    · exact Or.inl rfl
  · exact Or.inl rfl

/-- The form's refusal alone is a code. -/
theorem codes_form :
    ([("two-stage-form", Json.str Spot.SecondStage.FORM_WHY)].foldl (fun c x => c.fail x.1 x.2)
      ({} : Spot.Check)).codes ≠ #[] := by
  simp [Spot.Check.fail]

/-- **Round 2's check passes with the form allowed where it passes at any switch**: its failures at the switch `b`
are those with the form allowed, or the form's refusal, which is a code. -/
theorem check_true_of_codes {b : Bool} {own : Spot.SecondStage.Own} {stmts : Option (Array Json)}
    (h : (Spot.SecondStage.check b own stmts).codes = #[]) :
    (Spot.SecondStage.check true own stmts).codes = #[] := by
  unfold Spot.SecondStage.check at h ⊢
  cases hp : own.parts with
  | none => simpa only [hp] using h
  | some ps =>
    cases stmts with
    | none => simpa only [hp] using h
    | some ss =>
      simp only [hp] at h ⊢
      cases hd : Spot.SecondStage.designOk ps own
      · simpa only [hd, Bool.false_eq_true, if_false] using h
      · simp only [hd, if_true] at h ⊢
        rcases failures_allow b own ps ss.toList with e | e
        · rwa [e] at h
        · rw [e] at h
          exact absurd h codes_form

/-- `getD` of an object at `k` is its map's entry at `k`, `null` where it has none. -/
theorem getD_obj (m : Std.TreeMap.Raw String Json compare) (k : String) :
    Spot.getD (.obj m) k = (m.get? k).getD .null := by
  unfold Spot.getD Spot.field? Json.getObjVal?
  cases m.get? k <;> first | rfl | simp [Except.toOption]

/-- `setObjVal!` on an object is its map's `insert`. -/
theorem setObjVal_obj (m : Std.TreeMap.Raw String Json compare) (k : String) (v : Json) :
    (Json.obj m).setObjVal! k v = .obj (m.insert k v) := rfl

theorem cmp_wrong_units : compare "wrong_units" "accepted" ≠ Ordering.eq := by
  first | decide | simp [compare, compareOfLessAndEq]

theorem cmp_integrity_profile : compare "integrity_profile" "accepted" ≠ Ordering.eq := by
  first | decide | simp [compare, compareOfLessAndEq]

theorem cmp_profile : compare "profile" "accepted" ≠ Ordering.eq := by
  first | decide | simp [compare, compareOfLessAndEq]

theorem cmp_accepted : compare "accepted" "accepted" = Ordering.eq := by
  first | rfl | decide | simp [compare, compareOfLessAndEq]

/-- **A refused record reads not accepted**: on a well-formed object, `accepted` set to `false` and then `profile`,
`integrity_profile` and `wrong_units` set, as `attach` does where its check has a code, reads `false`. -/
theorem accepted_false {m : Std.TreeMap.Raw String Json compare} (hm : m.WF) :
    Spot.jeq (Spot.getD (((((Json.obj m).setObjVal! "accepted" (.bool false)).setObjVal! "profile" .null).setObjVal!
      "integrity_profile" .null).setObjVal! "wrong_units" .null) "accepted") (.bool true) = false := by
  simp only [setObjVal_obj, getD_obj, Std.TreeMap.Raw.get?_eq_getElem?]
  rw [Std.TreeMap.Raw.getElem?_insert hm.insert.insert.insert, if_neg cmp_wrong_units,
    Std.TreeMap.Raw.getElem?_insert hm.insert.insert, if_neg cmp_integrity_profile,
    Std.TreeMap.Raw.getElem?_insert hm.insert, if_neg cmp_profile,
    Std.TreeMap.Raw.getElem?_insert hm, if_pos cmp_accepted]
  simp [Spot.jeq, Flock.canon, pure, Except.pure]

/-- **An attached record reads accepted only where its check reports no code**: on a well-formed object, where the
check `c` has a code, `attach` sets `accepted` to `false` and then sets other keys only (`accepted_false`). -/
theorem codes_of_attach {m : Std.TreeMap.Raw String Json compare} (hm : m.WF) {c : Spot.Check}
    {reads : Array Json}
    (h : Spot.jeq (Spot.getD (Spot.SecondStage.attach (.obj m) c reads) "accepted") (.bool true) = true) :
    c.codes = #[] := by
  by_contra hc
  have hne : c.codes.isEmpty = false := by
    cases e : c.codes.isEmpty
    · rfl
    · exact absurd (Array.isEmpty_iff.1 e) hc
  unfold Spot.SecondStage.attach at h
  simp only [hne, Bool.false_eq_true, if_false] at h
  split at h
  · rw [setObjVal_obj, accepted_false hm.insert] at h
    exact Bool.false_ne_true h
  · rw [setObjVal_obj, setObjVal_obj, accepted_false hm.insert.insert] at h
    exact Bool.false_ne_true h

end SpotFlock.RunsAs
