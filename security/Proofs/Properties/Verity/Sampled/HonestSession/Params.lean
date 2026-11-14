module

public import Proofs.Properties.Verity.Sampled.HonestSession.Data.Inputs
public import Proofs.Properties.Verity.Sampled.HonestSession.JsonCanon
public import Proofs.Properties.Verity.Sampled.HonestSession.Program
public import Proofs.Properties.Verity.Sampled.HonestSession.Hex
public import Proofs.Properties.Verity.Sampled.Accepts
public import Proofs.Flock.Recursive.FirewallObj
public import Proofs.Flock.Claims.Complete.Json.Canon
public import Proofs.Flock.SessionLoop.Concrete.Fwd
import Base.Bytes
import Flock.Scan
import Flock.Canon
import Flock.Firewall
import Flock.Norm
import Flock.HmRow
import Flock.Registered
import Flock.JsonParse
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
# v2's parameters in `θ₀`

`θ₀` is canonical and decodes to one component whose parameters `paramsOfJson` reads at the one-stage form (`params_spec`):
the verifier's partition object and program are `partitionObj` and `programFile`, its program, query, law and work table
the session's, and its challenger the deployed key at `label₀`. The digests it names are the circuit and public files'
(`digests`). Every step rewrites by `Flock.canon`'s equation lemmas and the JSON facts (`JsonCanon.JsonFacts`), taken as
a hypothesis here; nothing is decided over a hash.
-/

@[expose] public section

namespace Verity.Sampled.Instance.HonestSession.Params

open Flock
open Lean (Json)
open FlockSoundness.Discharge.Recursion.FirewallObj (plain_mkObj)
open FlockSoundness.Discharge.Recursion.FirewallJson (norm)

set_option maxHeartbeats 0
set_option maxRecDepth 200000

/-! ## Plain values and their ASCII texts -/

/-- A digest's hex name is plain: no `"`, `\` or control character. -/
private theorem plainStr_hexOf (b : ByteArray) : Firewall.plainStr (hexOf b) = true := by
  rw [Firewall.plainStr, String.all_bool_eq, List.all_eq_true, Hex.hexOf_toList]
  have key : ∀ d < 16, ((Hex.hd d != '"' && Hex.hd d != '\\' && 0x20 ≤ (Hex.hd d).val) = true) := by
    decide +kernel
  intro c hc
  rw [List.mem_flatMap] at hc
  obtain ⟨x, _, hx⟩ := hc
  have hlt : x.toNat < 256 := x.toNat_lt
  simp only [List.mem_cons, List.not_mem_nil, or_false] at hx
  rcases hx with rfl | rfl
  · exact key _ (by omega)
  · exact key _ (Nat.mod_lt _ (by decide))

/-- An ASCII text's UTF-8 bytes are each below `0x7f`. -/
private theorem ascii_bytes {s : String} (h : Claims.Json.Ascii s) : s.toUTF8.data.all (· < 0x7f) = true := by
  have hu : ∀ c ∈ s.toList, c.utf8Size = 1 := by
    intro c hc
    have hv := h c hc
    have : c.val.toNat < 0x80 := by simpa [Char.toNat] using Nat.lt_of_lt_of_le hv (by decide)
    unfold Char.utf8Size
    simp only [this, decide_true, ↓reduceIte]
  have hb : s.toUTF8.data.toList = s.toList.map (·.val.toUInt8) := by
    rw [String.toUTF8_eq_toByteArray, ← String.utf8Encode_toList, List.utf8Encode, Flock.Scan.flatMap_ascii _ hu]
  rw [Array.all_eq_true]
  intro x hx
  rw [← Array.mem_toList_iff, hb, List.mem_map] at hx
  obtain ⟨c, hc, rfl⟩ := hx
  have hv := h c hc
  have : c.val.toNat < 0x7f := by simpa [Char.toNat] using hv
  have : (c.val.toUInt8).toNat < 0x7f := by
    rw [UInt32.toNat_toUInt8]
    omega
  exact decide_eq_true (by simpa [UInt8.lt_iff_toNat_lt] using this)

/-- `Flock.canon` of a plain value is ASCII at its UTF-8 bytes, so `canonicalJson` admits its text. -/
private theorem canon_ascii_bytes {j : Json} {s : String} (hc : Flock.canon j = .ok s) :
    s.toUTF8.data.all (· < 0x7f) = true :=
  ascii_bytes (FlockComplete.JsonRT.canon_ascii j hc)

/-- The program descriptor is plain (no hash is spliced into it). -/
private theorem plain_programJ : Firewall.plain programJ = true := by decide +kernel

/-- `partitionObj`, as `Spot.Partition.build` writes it, with `query` (whose keys are not empty) at `query`. -/
private theorem partitionObj_eq :
    partitionObj = Json.mkObj [("format", (Flock.HmRow.PARTITION_V1 : Json)),
      ("program", ((hexOf programSha) : Json)), ("query", query)] := by
  rw [partitionObj, Spot.Partition.build, if_neg (by decide)]

/-- The composite's component parameters are plain. -/
private theorem plain_paramsJ : Firewall.plain paramsJ = true := by
  rw [paramsJ]
  refine plain_mkObj (by decide) ?_
  intro e he
  simp only [List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false] at he
  rcases he with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  · exact ⟨by decide, by decide⟩                                  -- anchors
  · -- challenger
    refine ⟨by decide, ?_⟩
    refine plain_mkObj (by decide) ?_
    intro e2 he2
    simp only [List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false] at he2
    rcases he2 with rfl | rfl
    · -- key
      refine ⟨by decide, ?_⟩
      refine plain_mkObj (by decide) ?_
      intro e3 he3
      simp only [List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false] at he3
      rcases he3 with rfl | rfl | rfl
      · exact ⟨by decide, by decide⟩                              -- H
      · exact ⟨by decide, by decide⟩                              -- W
      · exact ⟨by decide, plainStr_hexOf _⟩                       -- public
    · exact ⟨by decide, by decide⟩                                -- label
  · exact ⟨by decide, by decide⟩                                  -- delta
  · exact ⟨by decide, by decide +kernel⟩                          -- law
  · exact ⟨by decide, by decide⟩                                  -- leaf_layer
  · exact ⟨by decide, by decide⟩                                  -- log
  · -- partition
    refine ⟨by decide, ?_⟩
    rw [partitionObj_eq]
    refine plain_mkObj (by decide) ?_
    intro e2 he2
    simp only [List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false] at he2
    rcases he2 with rfl | rfl | rfl
    · exact ⟨by decide, by decide +kernel⟩                        -- format
    · exact ⟨by decide, plainStr_hexOf _⟩                         -- program
    · exact ⟨by decide, by decide +kernel⟩                        -- query
  · exact ⟨by decide, by decide⟩                                  -- ports
  · exact ⟨by decide, plain_programJ⟩                             -- program
  · exact ⟨by decide, by decide +kernel⟩                          -- query
  · exact ⟨by decide, by decide⟩                                  -- registered
  · exact ⟨by decide, by decide⟩                                  -- scheme
  · exact ⟨by decide, plainStr_hexOf _⟩                           -- window_root
  · exact ⟨by decide, by decide⟩                                  -- work_table

/-- The composite's parameters are plain. -/
private theorem plain_θJ : Firewall.plain θJ = true := by
  rw [θJ]
  refine plain_mkObj (by decide) ?_
  intro e he
  simp only [List.mem_cons, List.not_mem_nil, or_false] at he
  rcases he with rfl
  refine ⟨by decide, ?_⟩
  show (Firewall.plain (Json.mkObj [("id", .str VStar.Recursive.Sampled.PROTOCOL), ("params", paramsJ)])
      && Firewall.plainList []) = true
  rw [Bool.and_eq_true]
  refine ⟨?_, by decide⟩
  refine plain_mkObj (by decide) ?_
  intro e2 he2
  simp only [List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false] at he2
  rcases he2 with rfl | rfl
  · exact ⟨by decide, by decide⟩                                  -- id
  · exact ⟨by decide, plain_paramsJ⟩                              -- params

/-! ## The canonical-JSON equations the walk rewrites by -/

/-- `canonicalJson θ₀` reads back the composite's parameters. -/
private theorem canon_θ₀ (hJ : JsonCanon.JsonFacts) : Core.canonicalJson θ₀ = .ok (norm θJ) :=
  hJ.canonicalJson_canon plain_θJ (canon_ascii_bytes (hJ.canon_plain plain_θJ)) (hJ.canon_plain plain_θJ)

/-- `canon` of the parsed component parameters is their text. -/
private theorem canon_normParams (hJ : JsonCanon.JsonFacts) : Flock.canon (norm paramsJ) = .ok (Firewall.printJson paramsJ) :=
  (hJ.canon_norm plain_paramsJ).trans (hJ.canon_plain plain_paramsJ)

/-- `canonicalJson` of the component parameters' text reads them back. -/
private theorem canon_params (hJ : JsonCanon.JsonFacts) :
    Core.canonicalJson (Firewall.printJson paramsJ).toUTF8 = .ok (norm paramsJ) :=
  hJ.canonicalJson_canon plain_paramsJ (canon_ascii_bytes (hJ.canon_plain plain_paramsJ)) (hJ.canon_plain plain_paramsJ)

/-- `canon` of the parsed program descriptor is its text. -/
private theorem canon_normProgram (hJ : JsonCanon.JsonFacts) :
    Flock.canon (norm programJ) = .ok (Firewall.printJson programJ) :=
  (hJ.canon_norm plain_programJ).trans (hJ.canon_plain plain_programJ)

/-- The program file is the descriptor's text. -/
private theorem programFile_eq (hJ : JsonCanon.JsonFacts) : programFile = (Firewall.printJson programJ).toUTF8 := by
  show ((Flock.canon programJ).toOption.getD "").toUTF8 = _
  rw [hJ.canon_plain plain_programJ]
  rfl

/-- The descriptor bytes `Program.parse` reads are the program file. -/
private theorem desc_eq (hJ : JsonCanon.JsonFacts) : (⟨desc.toArray⟩ : ByteArray) = (Firewall.printJson programJ).toUTF8 := by
  have h : (⟨desc.toArray⟩ : ByteArray) = programFile := by
    show ByteArray.mk (programFile.data.toList).toArray = programFile
    rw [Array.toArray_toList]
  rw [h, programFile_eq hJ]

/-- `Program.parse` of the descriptor's text is the program. -/
private theorem parse_program (hJ : JsonCanon.JsonFacts) :
    Spot.Program.parse (Firewall.printJson programJ).toUTF8 = .ok program := by
  rw [← desc_eq hJ]; exact Program.program_parse

/-! ## v2's parameters, with every field named -/

/-- The parameters `θ₀` decodes to, with each field `paramsOfJson` reads. -/
private theorem params_full (hJ : JsonCanon.JsonFacts) : ∃ p : Spot.OneStageZk.Params,
    ParamsAt θ₀ p ∧ p.own.partition = partitionObj ∧ p.own.program = programFile ∧
    p.own.circuit = hexOf circuitSha ∧ p.own.pub = hexOf (Sha512.hash publicFile) ∧
    p.base.verifier.program = program ∧ p.base.verifier.query = query ∧ p.base.verifier.law = law ∧
    p.base.verifier.workTable = none ∧ p.base.challenger = some (key₀, label₀) := by
  have hA := canon_θ₀ hJ
  have hB := canon_normParams hJ
  have hC := canon_params hJ
  have hD := canon_normProgram hJ
  have hE := parse_program hJ
  have hU := Flock.Hex.unhex_hexOf keyPub
  have hpf := programFile_eq hJ
  refine ⟨?p, ⟨_, ?decode, _, ?find, ?pchain⟩, ?f1, ?f2, ?f3, ?f4, ?f5, ?f6, ?f7, ?f8, ?f9⟩
  case decode =>
    fwd_walk [hA, hB] block [Core.canonicalJson, Flock.canon] shallow []
  case find => kernel_rfl'
  case pchain =>
    fwd_walk [hC, hD, hE, hU] block [Core.canonicalJson, Flock.canon, Spot.Program.parse, Flock.unhex] shallow []
  case f1 => kernel_rfl'            -- own.partition = partitionObj
  case f2 => rw [hpf]               -- own.program = programFile
  case f3 => kernel_rfl'            -- own.circuit
  case f4 => kernel_rfl'            -- own.pub
  case f5 => rfl                    -- verifier.program = program
  case f6 => kernel_rfl'            -- verifier.query = query
  case f7 => kernel_rfl'            -- verifier.law = law
  case f8 => rfl                    -- verifier.workTable = none
  case f9 => kernel_rfl'            -- challenger = some (key₀, label₀)

theorem params_spec (hJ : JsonCanon.JsonFacts) : ∃ p, ParamsAt θ₀ p ∧ p.own.partition = partitionObj ∧
    p.own.program = programFile ∧ p.base.verifier.program = program ∧ p.base.verifier.query = query ∧
    p.base.verifier.law = law ∧ p.base.verifier.workTable = none ∧ p.base.challenger = some (key₀, label₀) := by
  obtain ⟨p, hpat, hpart, hprog, _, _, hvp, hvq, hvl, hvt, hch⟩ := params_full hJ
  exact ⟨p, hpat, hpart, hprog, hvp, hvq, hvl, hvt, hch⟩

theorem digests (hJ : JsonCanon.JsonFacts) :
    programDigestOf θ₀ = some (Verity.sha512 circuitFile.data.toList) ∧
      publicDigestOf θ₀ = some (Verity.sha512 publicFile.data.toList) := by
  obtain ⟨p, hpat, _, _, hc, hpub, _⟩ := params_full hJ
  refine ⟨?_, ?_⟩
  · rw [programDigestOf, paramsOf_eq hpat, Option.some_bind, hc]
    simp only [digestOf, Flock.Hex.unhex_hexOf, circuitSha]
    rw [← sha512_data circuitFile]
  · rw [publicDigestOf, paramsOf_eq hpat, Option.some_bind, hpub]
    simp only [digestOf, Flock.Hex.unhex_hexOf]
    rw [← sha512_data publicFile]

end Verity.Sampled.Instance.HonestSession.Params
