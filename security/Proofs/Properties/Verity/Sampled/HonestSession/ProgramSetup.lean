module

public import Proofs.Properties.Verity.Sampled.HonestSession.ProgramPop
public import Proofs.Properties.Verity.Sampled.HonestSession.SetupReads
import Proofs.Flock.Verifier.Hash
import Proofs.Properties.CanonRoundTrip
import Spot.Partition
import Flock.HmRow
import Flock.TemplateQuery
import Flock.LinkUnits
import Commitments.Draw
import Circuits.Program
import Circuits.Steps
import all Init.Data.Array.QSort.Basic
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
import all Init.Data.String.Basic
import all Init.Data.String.Defs
import all Init.Data.String.Legacy
import all Init.Data.String.Slice
import all Init.Data.String.Search
import all Init.Data.String.TakeDrop
import all Init.Data.String.Pattern.String
import all Init.Data.String.Lemmas.Pattern.String.ForwardPattern
import all Init.Data.ToString.Basic
import all Init.Data.Repr
import all Init.Data.Nat.ToString
import all Init.Data.Array.Basic
import all Init.Data.Array.DecidableEq
import all Init.Data.List.Control
import all Init.Data.Stream
import all Init.Data.Slice.Basic
import all Init.Data.Slice.Array.Basic
import all Init.Data.Array.Subarray
import all Init.Data.Range.Basic

/-!
# The setup's program-side results: `checkPartition`, `deriveTemplateUnits`, `HmLinks.check`

`SetupReads.setupFacts_of` takes three results about the verifier's own copy of the program; this file proves them.
Each is a walk generic in what it reads of a hash, then instantiated at the honest session:

* `checkPartition_of`: `checkPartition` passes on any circuit and object whose META `partition` rule is
  `verity/partition/v1`, whose digest is the object's (`partitionDigest`) and whose object names META's `program_sha512`.
  At `cH`, META's reads are the kernel's (`cH_eq`), the object's checks are `cpo_V` and `partitionDigest_ok`.
* `derive_U`: `deriveTemplateUnits` passes on any program bytes that decode to `programJ` (`String.fromUTF8?`, then
  `JsonParse.parse`) and whose bare descriptor's canonical text is `s`, an object whose `query` is `query` and whose
  `program` is `hexOf (Sha512.hash s.toUTF8)`, and a circuit whose template is `templateId`. The digest is of a variable, so
  the program check closes by `bne_ok_some` and no step evaluates a hash. At the session: `programFile_eq`,
  `parse_printJson`, `canon_bare`.
* `links_W`: `HmLinks.check` passes, with no cut wire, on any circuit without META `links`, program bytes that decode to
  `progB b ix`, and a public file whose header names unit 0 alone. `nodes` filters the root's nodes by `isInputId`
  under a binder, which no rule reaches, so its value is read by `nodes_progB`. At the session: `cH_links`,
  `decode_file` with `decV_progB`, and the readers `#[0]` (`readers_draw`); the public file's unit indices are a
  hypothesis (`hidxI`), the public-file equation the result reads.

`setupFacts_prog` passes the three to `setupFacts_of`.
-/

@[expose] public section

set_option maxHeartbeats 4000000
set_option maxRecDepth 20000

namespace Verity.Sampled.Instance.HonestSession.ProgramSetup

open Flock
open Lean (Json)
open ProgramProof ProgramParse ProgramPop CircuitFileProof SetupReads

/-! ## `checkPartition` -/

theorem checkPartition_of (c : Flock.Circuit) (o : Json) (D : String) (nv : String × Nat) (B : ByteArray)
    (hrule : ((c.metaJ.getObjVal? "partition").toOption.getD Json.null).getObjValAs? String "rule" = .ok HmRow.PARTITION_V1)
    (hdig : ((c.metaJ.getObjVal? "partition").toOption.getD Json.null).getObjValAs? String "digest" = .ok D)
    (ho : HmRow.checkPartitionObject o = .ok nv)
    (hB : HmRow.partitionDigest o = .ok B) (hD : hexOf B = D)
    (x : String) (hx1 : o.getObjValAs? String "program" = .ok x)
    (hx2 : c.metaJ.getObjValAs? String "program_sha512" = .ok x) :
    ∃ f, HmRow.checkPartition c (some o) = .ok f := by
  subst hD
  unfold HmRow.checkPartition
  simp only [hrule, hdig, hx2]
  simp only [Except.toOption, Option.getD, HmRow.PARTITION_V1]
  simp only [ho, hB, hx1, bind, Except.bind, pure, Except.pure, bne_self_eq_false, Bool.false_eq_true, ite_false]
  exact ⟨_, rfl⟩

theorem meta_rule :
    ((cH.metaJ.getObjVal? "partition").toOption.getD Json.null).getObjValAs? String "rule" = .ok HmRow.PARTITION_V1 := by
  rw [cH_eq]; kernel_rfl'

theorem meta_digest :
    ((cH.metaJ.getObjVal? "partition").toOption.getD Json.null).getObjValAs? String "digest" =
      .ok (hexOf partitionDigestB) := by
  rw [cH_eq]; kernel_rfl'

theorem meta_program : cH.metaJ.getObjValAs? String "program_sha512" = .ok (hexOf programSha) := by
  rw [cH_eq]; kernel_rfl'

theorem obj_program : partitionObj.getObjValAs? String "program" = .ok (hexOf programSha) := by
  unfold partitionObj
  rw [build_query]
  kernel_rfl'

theorem programSha_hex : HmRow.isSha512Hex (hexOf programSha) = true :=
  hexOf_sha512Hex _ (Flock.Sha512.hash_size _)

theorem cpo_obj : HmRow.checkPartitionObject partitionObj = .ok (cpoV (hexOf programSha)) := by
  unfold partitionObj
  rw [cpo_K, build_query]
  exact cpo_V _ programSha_hex

/-- **`hpart`** of `setupFacts_of`. -/
theorem partition_ok : ∃ f, HmRow.checkPartition cH (some partitionObj) = .ok f :=
  checkPartition_of cH partitionObj _ _ partitionDigestB meta_rule meta_digest cpo_obj partitionDigest_ok rfl _
    obj_program meta_program

/-! ## `deriveTemplateUnits` -/

theorem bne_ok_some (x : String) : ((Except.ok x : Except String String).toOption != some x) = false := by
  simp [Except.toOption]

theorem isInput_not16 (s : String) (h : s = "Not16_v1") : Flock.TemplateQuery.isInput s = false :=
  isInputId_not16 s h

fwd_def dU derive_U (c : Flock.Circuit) (obj : Json) (bytes : ByteArray) (text s : String)
    (hq : obj.getObjVal? "query" = .ok query)
    (hprog : obj.getObjValAs? String "program" = .ok (hexOf (Sha512.hash s.toUTF8)))
    (hqt : HmRow.queryTemplate c = some (templateId, false))
    (hu : String.fromUTF8? bytes = some text)
    (hj : JsonParse.parse text = .ok programJ) (hc : ∀ j, j = programBare → canon j = .ok s) :
    HmRow.deriveTemplateUnits c obj bytes #[0]
  by [bne_ok_some, hq, hprog, hqt, hu, hj, hc, mapM_except, templatesOf_q, isInputNode_eq, isInput_in16, isInput_unit,
    isInput_not16, forIn_range_ex, forIn_arr_ex, forIn_list_ex, forInEx.eq_def]
  block [String.hash, Flock.canon, JsonParse.parse] shallow []

theorem programSha_eq : programSha = Sha512.hash (Firewall.printJson programBare).toUTF8 := by
  unfold programSha
  rw [canon_programBare]
  rfl

theorem queryTemplate_cH : HmRow.queryTemplate cH = some (templateId, false) := by
  rw [cH_eq]; kernel_rfl'

theorem utf8_file : String.fromUTF8? programFile = some (Firewall.printJson programJ) := by
  rw [programFile_eq, Properties.CanonRoundTrip.fromUTF8?_toUTF8]

theorem parse_programJ : JsonParse.parse (Firewall.printJson programJ) = .ok programJ := by
  rw [FlockSoundness.Discharge.Recursion.FirewallJson.parse_printJson plain_programJ, norm_programJ]

/-- **`hderive`** of `setupFacts_of`. -/
theorem derive_ok : ∃ note, HmRow.deriveTemplateUnits cH partitionObj programFile #[0] = .ok note :=
  ⟨_, derive_U cH partitionObj programFile (Firewall.printJson programJ) (Firewall.printJson programBare)
    partitionObj_query (by rw [obj_program, programSha_eq]) queryTemplate_cH utf8_file parse_programJ canon_bare⟩

/-! ## `HmLinks.check` -/

theorem isInputId_in16 (s : String) (h : s = "Input16_v1") : Flock.Program.isInputId s = true := by
  subst h
  simp [Flock.Program.isInputId, inputId_in16 _ rfl]

theorem nodes_progB (b : ByteArray) (ix : Std.HashMap String Nat) :
    HmLinks.nodes (progB b ix) = [(1, 1, 1, 1)] := by
  have h := isInputId_in16 _ rfl
  unfold HmLinks.nodes
  simp [progB, defRoot, rootBody, defIn16, defUnit, h, List.range_succ]
  rfl

fwd_def lkW links_W (c : Flock.Circuit) (pop : Flock.Public) (bytes : ByteArray) (b : ByteArray)
    (ix : Std.HashMap String Nat) (U : Json) (e : String)
    (hl : c.metaJ.getObjVal? "links" = .error e)
    (hdec : Flock.Program.decode bytes = .ok (progB b ix))
    (hU : J.get pop.header "units" "public file" = .ok U)
    (hI : J.arr U "indices" "public file" = .ok #[Lean.toJson (0 : Nat)]) :
    HmLinks.check c pop (some #[0]) bytes
  by [nodes_progB, hl, hdec, hU, hI, isInputId_in16, isInputId_not16, leaves_eq, widths_eq, mapM_except,
    forIn_range_ex, forIn_arr_ex, forIn_list_ex, forInEx.eq_def]
  block [String.hash, Sha512.hash] shallow []

theorem readers_draw : HmLinks.readersOf (some draw₀) = some #[0] := by
  show (Flock.Draw.UnitDraw.ofJson draw₀).toOption.map (·.proved) = _
  rw [draw_ofJson]
  rfl

theorem links_none : ∃ e, cH.metaJ.getObjVal? "links" = .error e := by
  rcases hx : cH.metaJ.getObjVal? "links" with e | v
  · exact ⟨e, rfl⟩
  · have h := cH_links
    simp [J.has, hx] at h

/-- **`hlinks`** of `setupFacts_of`, given the public file's unit indices (`hidxI`: its header's `units` names unit 0
alone), the one public-file equation it reads. -/
theorem links_ok (hidxI : ∃ U, J.get pubI.header "units" "public file" = .ok U ∧
      J.arr U "indices" "public file" = .ok #[Lean.toJson (0 : Nat)]) :
    ∃ w, HmLinks.check cH pubI (HmLinks.readersOf (some draw₀)) programFile = .ok w := by
  obtain ⟨U, hU, hI⟩ := hidxI
  obtain ⟨e, he⟩ := links_none
  have hdec : Flock.Program.decode programFile = .ok (progB decV.sha512 decV.index) := by
    rw [← decV_progB]; exact decode_file
  rw [readers_draw]
  exact ⟨_, links_W cH pubI programFile decV.sha512 decV.index U e he hdec hU hI⟩

/-- **`SetupFacts`** from `setupFacts_of`, its three program-side results proved here: what remains is the public
file's (`PublicFileFacts` and its reads before and after the draw). -/
theorem setupFacts_prog (hP : PublicFileFacts) (hn : pubD.n = 1)
    (hidx : ∃ U, J.get pubD.header "units" "public file" = .ok U ∧
      J.arr U "indices" "public file" = .ok #[Lean.toJson (0 : Nat)])
    (hdig : (pubD.digest 0 0 1).size = 128)
    (hidxI : ∃ U, J.get pubI.header "units" "public file" = .ok U ∧
      J.arr U "indices" "public file" = .ok #[Lean.toJson (0 : Nat)]) : SetupFacts :=
  setupFacts_of hP hn hidx hdig partition_ok derive_ok (links_ok hidxI)

end Verity.Sampled.Instance.HonestSession.ProgramSetup
