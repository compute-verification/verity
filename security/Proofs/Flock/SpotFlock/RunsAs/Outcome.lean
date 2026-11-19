module

public import Proofs.Flock.SpotFlock.RunsAs.Statement
import all Lean.Data.Json.Basic
import Std.Data.TreeMap.Raw.Lemmas
import Std.Data.TreeMap.Raw.WF

/-!
# The outcome: an interior audit record that reads accepted had round 2's check pass

`Run.run`'s interior audit record is the audit record (`Record.record`, an object built by `Json.mkObj`) with round 2's
check attached (`SecondStage.attach`), and it accepts only where that record's `accepted` reads `true`. Here:

* `codes_of_attach`: a well-formed object with a check attached reads accepted only where the check reports no code.
-/

@[expose] public section

namespace SpotFlock.RunsAs

open Lean (Json)

/-- `getD` of an object at `k` is its map's entry at `k`, `null` where it has none. -/
theorem getD_obj (m : Std.TreeMap.Raw String Json compare) (k : String) :
    Spot.getD (.obj m) k = (m[k]?).getD .null := by
  cases hk : m[k]? <;>
    simp [Spot.getD, Spot.field?, Json.getObjVal?, Std.TreeMap.Raw.get?_eq_getElem?, hk, Except.toOption, pure,
      Except.pure, throw, throwThe, MonadExceptOf.throw]

/-- `setObjVal!` on an object is its map's `insert`. -/
theorem setObjVal_obj (m : Std.TreeMap.Raw String Json compare) (k : String) (v : Json) :
    (Json.obj m).setObjVal! k v = .obj (m.insert k v) := by
  simp only [Json.setObjVal!]

theorem cmp_ne {a b : String} (h : a ≠ b) : compare a b ≠ Ordering.eq :=
  fun e => h (Std.LawfulEqCmp.compare_eq_iff_eq.1 e)

theorem cmp_incorrect_blocks : compare "wrong_units" "accepted" ≠ Ordering.eq := cmp_ne (by decide)

theorem cmp_integrity_profile : compare "integrity_profile" "accepted" ≠ Ordering.eq := cmp_ne (by decide)

theorem cmp_profile : compare "profile" "accepted" ≠ Ordering.eq := cmp_ne (by decide)

theorem cmp_accepted : compare "accepted" "accepted" = Ordering.eq := Std.LawfulEqCmp.compare_eq_iff_eq.2 rfl

/-- **A refused record reads not accepted**: on a well-formed object, `accepted` set to `false` and then `profile`,
`integrity_profile` and `wrong_units` set, as `attach` does where its check has a code, reads `false`. -/
theorem accepted_false {m : Std.TreeMap.Raw String Json compare} (hm : m.WF) :
    Spot.jeq (Spot.getD (((((Json.obj m).setObjVal! "accepted" (.bool false)).setObjVal! "profile" .null).setObjVal!
      "integrity_profile" .null).setObjVal! "wrong_units" .null) "accepted") (.bool true) = false := by
  simp only [setObjVal_obj, getD_obj]
  rw [Std.TreeMap.Raw.getElem?_insert hm.insert.insert.insert, if_neg cmp_incorrect_blocks,
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
