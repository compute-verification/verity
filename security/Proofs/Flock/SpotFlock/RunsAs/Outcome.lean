module

public import Proofs.Flock.SpotFlock.RunsAs.Statement
import all Lean.Data.Json.Basic
import Std.Data.TreeMap.Raw.Lemmas
import Std.Data.TreeMap.Raw.WF

/-! An interior audit record that reads accepted had round 2's check pass (`codes_of_attach`, `check_true_of_codes`). -/

@[expose] public section

namespace SpotFlock.RunsAs

open Lean (Json)

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

theorem codes_form :
    ([("two-stage-form", Json.str Spot.SecondStage.FORM_WHY)].foldl (fun c x => c.fail x.1 x.2)
      ({} : Spot.Check)).codes ≠ #[] := by
  simp [Spot.Check.fail]

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

theorem getD_obj (m : Std.TreeMap.Raw String Json compare) (k : String) :
    Spot.getD (.obj m) k = (m[k]?).getD .null := by
  cases hk : m[k]? <;>
    simp [Spot.getD, Spot.field?, Json.getObjVal?, Std.TreeMap.Raw.get?_eq_getElem?, hk, Except.toOption, pure,
      Except.pure, throw, throwThe, MonadExceptOf.throw]

theorem setObjVal_obj (m : Std.TreeMap.Raw String Json compare) (k : String) (v : Json) :
    (Json.obj m).setObjVal! k v = .obj (m.insert k v) := by
  simp only [Json.setObjVal!]

theorem cmp_ne {a b : String} (h : a ≠ b) : compare a b ≠ Ordering.eq :=
  fun e => h (Std.LawfulEqCmp.compare_eq_iff_eq.1 e)

theorem cmp_incorrect_blocks : compare "wrong_units" "accepted" ≠ Ordering.eq := cmp_ne (by decide)

theorem cmp_integrity_profile : compare "integrity_profile" "accepted" ≠ Ordering.eq := cmp_ne (by decide)

theorem cmp_profile : compare "profile" "accepted" ≠ Ordering.eq := cmp_ne (by decide)

theorem cmp_accepted : compare "accepted" "accepted" = Ordering.eq := Std.LawfulEqCmp.compare_eq_iff_eq.2 rfl

theorem accepted_false {m : Std.TreeMap.Raw String Json compare} (hm : m.WF) :
    Spot.jeq (Spot.getD (((((Json.obj m).setObjVal! "accepted" (.bool false)).setObjVal! "profile" .null).setObjVal!
      "integrity_profile" .null).setObjVal! "wrong_units" .null) "accepted") (.bool true) = false := by
  simp only [setObjVal_obj, getD_obj]
  rw [Std.TreeMap.Raw.getElem?_insert hm.insert.insert.insert, if_neg cmp_incorrect_blocks,
    Std.TreeMap.Raw.getElem?_insert hm.insert.insert, if_neg cmp_integrity_profile,
    Std.TreeMap.Raw.getElem?_insert hm.insert, if_neg cmp_profile,
    Std.TreeMap.Raw.getElem?_insert hm, if_pos cmp_accepted]
  simp [Spot.jeq, Flock.canon, pure, Except.pure]

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
