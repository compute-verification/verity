module

public import Proofs.Core.QSort
public import Proofs.Properties.Pouw.LiveSalt.Canon
public import Proofs.Properties.NetworkCertifier.Records.Bytes
import all Lean.Data.Json.Basic
import all Init.Data.Ord.String

/-!
# A coin commitment's identity bytes name it

`LiveCommitment.toJson`'s identity bytes (`Flock.canonUtf8`) are `{"epoch":E,"families":F,"root":"HEX","scheme":S}`
(`canon_commitment`): its four keys are already in order, which `qsort` leaves as they are (`commitment_sort`). That
string parses back uniquely (`commStr_inj`), by `Canon`'s splits and the root's hex (`hexOf_inj`), so two commitments
with one form are one (`canon_inj`).
-/

@[expose] public section

namespace Properties.Challenger.Commitment

open Lean (Json)
open Spot.Challenger (LiveCommitment familiesJson)
open Properties.Pouw.LiveSalt.Canon (canon_obj canon_nat canon_families canon_str famStr canonStr_split nat_split
  fam_split)

/-- A commitment's identity bytes. -/
def commStr (c : LiveCommitment) : String :=
  "{" ++ ",".intercalate [Flock.canonStrUtf8 "epoch" ++ ":" ++ toString c.epoch,
    Flock.canonStrUtf8 "families" ++ ":" ++ famStr c.families,
    Flock.canonStrUtf8 "root" ++ ":" ++ Flock.canonStrUtf8 (Flock.hexOf c.root),
    Flock.canonStrUtf8 "scheme" ++ ":" ++ Flock.canonStrUtf8 c.scheme] ++ "}"

theorem commitment_sort {β : Type} (a b c d : β) :
    #[("epoch", a), ("families", b), ("root", c), ("scheme", d)].qsort (fun x y => decide (x.1 < y.1)) =
      #[("epoch", a), ("families", b), ("root", c), ("scheme", d)] := by
  refine Verity.QSort.qsort_sorted _ _ fun i hi j hj => ?_
  simp only [List.size_toArray, List.length_cons, List.length_nil] at hi hj
  obtain _ | _ | _ | _ | i := i <;> obtain _ | _ | _ | _ | j := j <;>
    first | omega | simp (config := {decide := true})

theorem canon_commitment (c : LiveCommitment) : Flock.canonUtf8 c.toJson = .ok (commStr c) := by
  show Flock.canonUtf8 (Json.obj _) = _
  rw [canon_obj]
  have ht : (Std.TreeMap.Raw.ofList [("epoch", Lean.toJson c.epoch), ("families", familiesJson c.families),
      ("root", Json.str (Flock.hexOf c.root)), ("scheme", Json.str c.scheme)] : Std.TreeMap.Raw String Json).toArray =
      #[("epoch", Lean.toJson c.epoch), ("families", familiesJson c.families), ("root", Json.str (Flock.hexOf c.root)),
        ("scheme", Json.str c.scheme)] := rfl
  rw [ht]
  simp only [List.mapM_toArray, List.mapM_cons, List.mapM_nil, canon_nat, canon_families, canon_str, pure_bind,
    map_pure]
  rw [commitment_sort]
  rfl

/-- A commitment's identity bytes name its epoch, its families, its root and its scheme. -/
theorem commStr_inj {c c' : LiveCommitment} (h : commStr c = commStr c') : c = c' := by
  obtain ⟨s, e, fs, r⟩ := c
  obtain ⟨s', e', fs', r'⟩ := c'
  have h := congrArg String.toList h
  simp only [commStr, String.toList_append, String.toList_intercalate, List.map_cons, List.map_nil,
    List.append_assoc] at h
  simp only [show "{".toList = ['{'] from rfl, show "}".toList = ['}'] from rfl, show ":".toList = [':'] from rfl,
    show ",".toList = [','] from rfl, List.intercalate_cons_cons, List.intercalate_singleton, List.cons_append,
    List.nil_append, List.append_assoc, List.cons.injEq, true_and, List.append_cancel_left_eq] at h
  obtain ⟨rfl, -, h⟩ := nat_split (by decide) (by decide) h
  simp only [List.append_cancel_left_eq, List.cons.injEq, true_and] at h
  obtain ⟨rfl, h⟩ := fam_split h
  simp only [List.cons.injEq, true_and, List.append_cancel_left_eq] at h
  obtain ⟨hr, h⟩ := canonStr_split h
  simp only [List.cons.injEq, true_and, List.append_cancel_left_eq] at h
  obtain ⟨rfl, -⟩ := canonStr_split h
  obtain rfl := Properties.NetworkRecords.Bytes.hexOf_inj hr
  rfl

/-- Two commitments with one identity form are one. -/
theorem canon_inj (c c' : LiveCommitment) (s : String) (h : Flock.canonUtf8 c.toJson = .ok s)
    (h' : Flock.canonUtf8 c'.toJson = .ok s) : c = c' :=
  commStr_inj (Except.ok.inj ((canon_commitment c).symm.trans (h.trans (h'.symm.trans (canon_commitment c')))))

end Properties.Challenger.Commitment
