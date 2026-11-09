module

public import Proofs.Flock.SpotFlock.InstanceCodes
public import Proofs.Flock.Recursive.FirewallJson
public import Proofs.Flock.Claims.Complete.Encode
public import Proofs.Flock.SessionLoop.Concrete.Public
import Proofs.Flock.Soundness.Refine.Walk
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
import all Init.Data.String.TakeDrop
import all Init.Data.String.Slice
import all Init.Data.String.Pattern.Basic
import all Init.Data.String.Pattern.String
import all Init.Data.String.Pattern.Char
import all Init.Data.String.Search
import all Init.Data.String.Iterate
import all Init.Data.String.Decode
import all Init.Data.String.Termination
import all Init.Data.String.Defs
import all Init.Data.String.Basic
import all Init.Data.String.Extra
import all Init.Data.Array.QSort.Basic
import all Init.Data.Array.Basic
import all Init.Data.Array.DecidableEq
import all Init.Data.List.Control
import all Init.Data.Range.Basic
import all Init.Data.Int.Repr
import all Init.Data.Repr
import all Init.Data.Nat.Fold
import all Init.Data.Array.Attach
import all Init.WF

/-!
# The run's interior record at the two-stage instance

`SpotFlock.Instance.interior0` is the interior record the instance's run sends in round 1, and `m1` its bytes as an
honest worker prints them (`Flock.Firewall.printJson`). The kernel decides what round 2 reads of it:
* `jsonMsg_m1`: `m1` reads back as `interior0`, through `printJson`'s round trip (`parse_printJson`), since every
  string and key in it is plain and every object's keys are written ascending (`norm_interior0`);
* `interior0_sample`: its window lays out replay block `0`, so the interior order holds at it
  (`interiorOrderOk_interior0`);
* `runOf_interior0`: it names the run `RUN0`.

Its ports are the template's, `i0` with one leaf and `o0` with two, named without their template. Round 2 reads of an
interior port only its name and leaves, in its binding (`SecondStage.portBinding`), at a port the statement reads
registered; its root and domain are placeholders.

## What this does not show

The boundary record `x`, whose roots the statement's registered reads open, and the statement's bindings, which hash
`interior0`'s program, partition and leaves with the run's name, are not built here. Nothing here states that
`interior0` is any served run's record: it is the record of the synthetic run whose acceptance the instance needs.
-/

@[expose] public section

namespace SpotFlock.Instance

open Claims Claims.Sampling SamplingDeploy Properties.Spot
open Lean (Json)

/-- A JSON natural. -/
def natJ (n : ℕ) : Json := .num ⟨n, 0⟩

/-- The run's name in the instance's records. -/
def RUN0 : String := "run-0"

/-- The interior record's entry for port `i0`: one leaf. -/
def rootI0 : Json :=
  Json.mkObj [("domain", .str "00"), ("leaves", natJ 1), ("port", .str "i0"), ("root", .str "00")]

/-- The interior record's entry for port `o0`: two leaves, one per instance. -/
def rootO0 : Json :=
  Json.mkObj [("domain", .str "00"), ("leaves", natJ 2), ("port", .str "o0"), ("root", .str "00")]

/-- The interior record's window: a served run's interior, laying out replay block `0`. -/
def window0 : Json :=
  Json.mkObj [("interior", .bool true), ("kind", .str "served"), ("replay_sample", .arr #[natJ 0]),
    ("run", .str RUN0)]

/-- **The run's interior record**, its keys ascending at every level. -/
def interior0 : Json :=
  Json.mkObj [("partition", .str "partition-0"), ("population", natJ 2), ("program", .str "program-0"),
    ("roots", .arr #[rootI0, rootO0]), ("scheme", .str "frame-v3-sha512/hm96-sha512"), ("window", window0)]

/-- `interior0`'s bytes, as an honest worker prints it. -/
def m1 : Msg := (Flock.Firewall.printJson interior0).toUTF8.data.toList

theorem rootI0_obj : rootI0 = .obj ⟨⟨FlockDeploy.Concrete.objT rootI0⟩⟩ := by kernel_rfl

theorem rootI0_model : (FlockDeploy.Concrete.objT rootI0).toListModel =
    [⟨"domain", .str "00"⟩, ⟨"leaves", natJ 1⟩, ⟨"port", .str "i0"⟩, ⟨"root", .str "00"⟩] := by kernel_rfl

theorem rootI0_asc : Flock.Firewall.ascending (FlockDeploy.Concrete.objT rootI0) = true := by kernel_rfl

theorem plain_rootI0 : Flock.Firewall.plain rootI0 = true := by
  rw [rootI0_obj]
  refine FlockDeploy.Concrete.plain_obj_of rootI0_model rootI0_asc ?_
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq]
  simp [Flock.Firewall.plain, Flock.Firewall.plainStr, String.all_bool_eq, natJ]

theorem rootO0_obj : rootO0 = .obj ⟨⟨FlockDeploy.Concrete.objT rootO0⟩⟩ := by kernel_rfl

theorem rootO0_model : (FlockDeploy.Concrete.objT rootO0).toListModel =
    [⟨"domain", .str "00"⟩, ⟨"leaves", natJ 2⟩, ⟨"port", .str "o0"⟩, ⟨"root", .str "00"⟩] := by kernel_rfl

theorem rootO0_asc : Flock.Firewall.ascending (FlockDeploy.Concrete.objT rootO0) = true := by kernel_rfl

theorem plain_rootO0 : Flock.Firewall.plain rootO0 = true := by
  rw [rootO0_obj]
  refine FlockDeploy.Concrete.plain_obj_of rootO0_model rootO0_asc ?_
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq]
  simp [Flock.Firewall.plain, Flock.Firewall.plainStr, String.all_bool_eq, natJ]

theorem window0_obj : window0 = .obj ⟨⟨FlockDeploy.Concrete.objT window0⟩⟩ := by kernel_rfl

theorem window0_model : (FlockDeploy.Concrete.objT window0).toListModel =
    [⟨"interior", .bool true⟩, ⟨"kind", .str "served"⟩, ⟨"replay_sample", .arr #[natJ 0]⟩,
      ⟨"run", .str RUN0⟩] := by kernel_rfl

theorem window0_asc : Flock.Firewall.ascending (FlockDeploy.Concrete.objT window0) = true := by kernel_rfl

theorem plain_window0 : Flock.Firewall.plain window0 = true := by
  rw [window0_obj]
  refine FlockDeploy.Concrete.plain_obj_of window0_model window0_asc ?_
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq]
  simp [Flock.Firewall.plain, Flock.Firewall.plainList, Flock.Firewall.plainStr, String.all_bool_eq, natJ, RUN0]

theorem interior0_obj : interior0 = .obj ⟨⟨FlockDeploy.Concrete.objT interior0⟩⟩ := by kernel_rfl

theorem interior0_model : (FlockDeploy.Concrete.objT interior0).toListModel =
    [⟨"partition", .str "partition-0"⟩, ⟨"population", natJ 2⟩, ⟨"program", .str "program-0"⟩,
      ⟨"roots", .arr #[rootI0, rootO0]⟩, ⟨"scheme", .str "frame-v3-sha512/hm96-sha512"⟩,
      ⟨"window", window0⟩] := by kernel_rfl

theorem interior0_asc : Flock.Firewall.ascending (FlockDeploy.Concrete.objT interior0) = true := by kernel_rfl

theorem plain_interior0 : Flock.Firewall.plain interior0 = true := by
  rw [interior0_obj]
  refine FlockDeploy.Concrete.plain_obj_of interior0_model interior0_asc ?_
  simp only [List.mem_cons, List.not_mem_nil, or_false, forall_eq_or_imp, forall_eq]
  simp [Flock.Firewall.plain, Flock.Firewall.plainList, Flock.Firewall.plainStr, String.all_bool_eq, natJ,
    plain_rootI0, plain_rootO0, plain_window0]

/-- The parser rebuilds each object by inserting its members in ascending order, which is the order `interior0` and
its objects are built in, so it rebuilds the same trees. -/
theorem norm_interior0 : FlockSoundness.Discharge.Recursion.FirewallJson.norm interior0 = interior0 := by kernel_rfl

/-- **`m1` reads back as the interior record.** -/
theorem jsonMsg_m1 : jsonMsg m1 = some interior0 := by
  have hb : (⟨m1.toArray⟩ : ByteArray) = (Flock.Firewall.printJson interior0).toUTF8 := by
    unfold m1
    rw [Array.toArray_toList]
  unfold jsonMsg FlockDeploy.jsonOf
  rw [hb, FlockComplete.fromUTF8?_toUTF8]
  simp only [FlockSoundness.Discharge.Recursion.FirewallJson.parse_printJson plain_interior0, norm_interior0]
  rfl

theorem interior0_sample :
    Spot.getD (Spot.getD interior0 "window") "replay_sample" = .arr #[Lean.toJson (0 : ℕ)] := by kernel_rfl

theorem runOf_interior0 : Spot.SecondStage.runOf interior0 = some RUN0 := by kernel_rfl

variable (R : Ref) (desc : Msg) (q : Query) (c₀ : Code)

/-- Round 2's interior order holds at the instance's run, whatever its boundary record and draw. -/
theorem interiorOrderOk_interior0 (st : Roots) (draw : Json) :
    Spot.SecondStage.interiorOrderOk (ownOf (held R bnd0 desc q c₀) st interior0 draw coin0) = true :=
  interiorOrderOk_held R desc q c₀ interior0_sample

end SpotFlock.Instance
