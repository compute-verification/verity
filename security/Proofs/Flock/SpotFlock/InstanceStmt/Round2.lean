module

public import Proofs.Flock.SpotFlock.InstanceStmt.Header
public import Proofs.Flock.SpotFlock.EntriesAt
import Proofs.Pouw.MonitoringPeriod.Sort
import Std.Data.TreeMap.Raw.Lemmas
import Flock.Comp
import Flock.JsonParse
import Flock.ZkServer
import Flock.Norm
import Flock.Canon
import all Lean.Data.Json.Basic
import all Lean.Data.Json.FromToJson.Basic
import all Std.Data.TreeMap.Raw.Basic
import all Std.Data.DTreeMap.Raw.Basic
import all Std.Data.DTreeMap.Internal.Operations
import all Std.Data.DTreeMap.Internal.Queries
import all Std.Data.DTreeMap.Internal.Balancing
import all Std.Data.DTreeMap.Internal.Def
import all Init.Data.Ord.String
import all Init.Data.Ord.Basic
import all Init.Data.String.Defs
import all Init.Data.String.Basic

/-! Round 2's statement of the two-instance reference run, and the digests it is named by (`stmt_ofJson`,
`checked_stmt`). -/

@[expose] public section

namespace SpotFlock.InstanceStmt

open Flock
open FlockDeploy.Concrete (objT IntJ canon_eq)
open FlockSoundness.Refine
open Lean (Json)
open Std.DTreeMap.Internal (Impl)

theorem keys_toArray (t : Impl String (fun _ => Json)) :
    (⟨⟨t⟩⟩ : Std.TreeMap.Raw String Json compare).toArray.map (·.1) = (t.toListModel.map (·.1)).toArray := by
  apply Array.ext'
  rw [Array.toList_map, Std.TreeMap.Raw.toList_toArray, List.toList_toArray]
  show (Impl.Const.toList t).map (·.1) = _
  rw [Impl.Const.toList_eq_toListModel_map, List.map_map]
  all_goals rfl

theorem keys?_model {t : Impl String (fun _ => Json)} {l : List String} (hs : l.Pairwise (· < ·))
    (h : t.toListModel.map (·.1) = l) : Spot.keys? (.obj ⟨⟨t⟩⟩) = some l.toArray := by
  show some (Spot.sortStrings ((⟨⟨t⟩⟩ : Std.TreeMap.Raw String Json compare).toArray.map (·.1))) = _
  rw [keys_toArray, h, Spot.sortStrings]
  exact congrArg some (Array.ext' (Properties.Pouw.MonitoringPeriod.SecurityProofs.Sort.qsort_of_sorted (as := l.toArray) hs))

theorem stmt_obj (r : Run) (circ cls t : String) :
    Json.mkObj [("draw", sampleJ), ("header", hdrJ r circ cls), ("registered", regJ), ("template", .str t)] =
      .obj ⟨⟨objT (Json.mkObj [("draw", sampleJ), ("header", hdrJ r circ cls), ("registered", regJ),
        ("template", .str t)])⟩⟩ := by kernel_rfl
theorem stmt_model (r : Run) (circ cls t : String) :
    (objT (Json.mkObj [("draw", sampleJ), ("header", hdrJ r circ cls), ("registered", regJ),
      ("template", .str t)])).toListModel =
      [⟨"draw", sampleJ⟩, ⟨"header", hdrJ r circ cls⟩, ⟨"registered", regJ⟩, ⟨"template", .str t⟩] := by kernel_rfl

theorem keys_stmt (r : Run) (circ cls t : String) :
    Spot.keys? (Json.mkObj [("draw", sampleJ), ("header", hdrJ r circ cls), ("registered", regJ),
      ("template", .str t)]) = some Spot.SecondStage.STMT_KEYS := by
  rw [stmt_obj r circ cls t]
  exact keys?_model (l := ["draw", "header", "registered", "template"]) (by decide)
    (by simp only [stmt_model, List.map_cons, List.map_nil])

theorem stmt_ofJson (r : Run) (circ cls t : String) :
    Spot.SecondStage.Stmt.ofJson? (Json.mkObj [("draw", sampleJ), ("header", hdrJ r circ cls), ("registered", regJ),
      ("template", .str t)]) = some ⟨t, hdrJ r circ cls, sampleJ, regJ⟩ := by
  have hd : FlockComplete.JsonRT.Distinct
      [("draw", sampleJ), ("header", hdrJ r circ cls), ("registered", regJ), ("template", Json.str t)] := by simp
  unfold Spot.SecondStage.Stmt.ofJson?
  rw [keys_stmt r circ cls t, bne_self_eq_false, ite_eq_right Bool.false_ne_true,
    SpotFlock.getD_mkObj hd (k := "template") (v := .str t) (by simp),
    SpotFlock.getD_mkObj hd (k := "header") (v := hdrJ r circ cls) (by simp),
    SpotFlock.getD_mkObj hd (k := "draw") (v := sampleJ) (by simp),
    SpotFlock.getD_mkObj hd (k := "registered") (v := regJ) (by simp)]
  all_goals rfl

def canonText (j : Json) : String := (canon j).toOption.getD ""

def shaOf (j : Json) : String := Spot.sha512Hex (canonText j).toUTF8

theorem canonSha512_eq {j : Json} (h : IntJ j = true) : Spot.canonSha512 j = .ok (shaOf j) := by
  unfold Spot.canonSha512 shaOf canonText
  rw [canon_eq h]
  rfl

theorem intJ_hdr (r : Run) (circ cls : String) : IntJ (hdrJ r circ cls) = true := by kernel_rfl
theorem intJ_reg : IntJ regJ = true := by kernel_rfl

theorem checked_stmt (r : Run) (circ cls t : String) :
    (⟨t, hdrJ r circ cls, sampleJ, regJ⟩ : Spot.SecondStage.Stmt).checked =
      some ⟨shaOf (hdrJ r circ cls), .str (shaOf regJ), sampleJ, Spot.SecondStage.ONE_TABLE⟩ := by
  unfold Spot.SecondStage.Stmt.checked
  simp only [canonSha512_eq (intJ_hdr r circ cls)]
  split
  · next h => exact absurd h (by rw [reg_obj]; exact fun h => Json.noConfusion h)
  · simp only [canonSha512_eq intJ_reg]
    rfl

theorem good_sampleJ : Claims.Json.Good sampleJ := by
  unfold sampleJ
  refine FlockComplete.JsonRT.good_mkObj (by simp) ?_
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq]
  refine ⟨.num _, .str _, .num _, .arr fun x hx => ?_⟩
  simp at hx
  subst hx
  exact .num _

end SpotFlock.InstanceStmt
