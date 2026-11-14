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

`programJ` nests the served unit's definitions many levels deep. That it is plain (`ProgramPlain`) is a concrete-JSON
fact of this instance that no tactic here shows robustly, so this piece takes it as a hypothesis; its full `plain_mkObj`
decomposition over the descriptor's definitions is deferred.
-/

@[expose] public section

namespace Verity.Sampled.Instance.HonestSession.Params

open Flock
open Lean (Json)
open FlockSoundness.Discharge.Recursion.FirewallObj (plain_mkObj)
open FlockSoundness.Discharge.Recursion.FirewallJson renaming norm → jnorm

set_option maxHeartbeats 0
set_option maxRecDepth 200000

/-- `plstr`: a plain literal string, number, null or empty array is plain, by the tree's canonical recipe (the string's
text is reduced to its characters and each checked). The def-named strings are unfolded so their literals reduce too. -/
local macro "plstr" : tactic =>
  `(tactic| simp [Firewall.plain, Firewall.plainList, Firewall.plainStr, String.all_bool_eq, jn,
      label₀, templateId, Flock.HmRow.PARTITION_V1, VStar.Recursive.Sampled.PROTOCOL])

/-- An object's members have distinct keys: their keys ascend (`compare … = .lt`, which the kernel reduces), so no two
are equal (`Asc.ne`). The decidable `Pairwise` is ascribed to `of_decide_eq_true` so Lean uses its `List.Pairwise`
instance rather than looking for one on `Asc` (there is none); the result coerces to `Asc Prod.fst L` by defeq. -/
private theorem keys_ne {L : List (String × Json)}
    (h : decide (L.Pairwise fun a b => compare a.1 b.1 = .lt) = true) :
    L.Pairwise fun a b => a.1 ≠ b.1 :=
  FlockSoundness.Discharge.Recursion.FirewallObj.Asc.ne (key := Prod.fst)
    (of_decide_eq_true h : L.Pairwise fun a b => compare a.1 b.1 = .lt)

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

/-- An ASCII character's low byte stays below `0x7f`. -/
private theorem byte_lt (c : Char) (h : c.toNat < 127) : c.val.toUInt8 < 127 := by
  rw [UInt8.lt_iff_toNat_lt, UInt32.toNat_toUInt8]
  have h127 : (127 : UInt8).toNat = 127 := by decide
  have hv : c.val.toNat < 127 := h
  have h8 : c.val.toNat < 2 ^ 8 := Nat.lt_trans hv (by decide)
  rw [Nat.mod_eq_of_lt h8, h127]
  exact hv

/-- An ASCII text's UTF-8 bytes are each below `0x7f`. -/
private theorem ascii_bytes {s : String} (h : Claims.Json.Ascii s) : s.toUTF8.data.all (· < 0x7f) = true := by
  have hu : ∀ c ∈ s.toList, c.utf8Size = 1 := by
    intro c hc
    rw [Char.utf8Size_eq_one_iff, UInt32.le_iff_toNat_le]
    have hv : c.val.toNat < 127 := h c hc
    have h127 : (127 : UInt32).toNat = 127 := by decide
    omega
  have hb : s.toUTF8.data = (s.toList.map (·.val.toUInt8)).toArray := by
    rw [String.toUTF8_eq_toByteArray, ← String.utf8Encode_toList, List.utf8Encode,
      Flock.Scan.flatMap_ascii _ hu, List.data_toByteArray]
  rw [hb, List.all_toArray, List.all_eq_true]
  intro x hx
  obtain ⟨c, hc, rfl⟩ := List.mem_map.1 hx
  exact decide_eq_true (byte_lt c (h c hc))

/-- `Flock.canon` of a plain value is ASCII at its UTF-8 bytes, so `canonicalJson` admits its text. -/
private theorem canon_ascii_bytes {j : Json} {s : String} (hc : Flock.canon j = .ok s) :
    s.toUTF8.data.all (· < 0x7f) = true :=
  ascii_bytes (FlockComplete.JsonRT.canon_ascii j hc)

/-- The program descriptor is plain (no hash is spliced into it): a concrete-JSON fact of this instance, taken as a
hypothesis `hpp : ProgramPlain` by every lemma below that reads `programJ`. -/
def ProgramPlain : Prop := Firewall.plain programJ = true

/-- The parser's rebuild (`norm`) of these two plain, ascending-key JSON literals is the literal itself. `canonicalJson`
returns `norm` of the parsed value, and the decoders must then read that value's fields as a concrete object; but `norm`
is well-founded-recursive, so the kernel does not reduce it (`canon_norm` only gives `canon (norm j) = canon j`, not
`norm j = j`). A concrete-JSON fact of this instance, taken as a hypothesis `hN : NormFacts` by `params_full` and the
exports that depend on it. -/
def NormFacts : Prop := jnorm θJ = θJ ∧ jnorm paramsJ = paramsJ

/-- The session's query is plain. -/
private theorem plain_query : Firewall.plain query = true := by
  rw [query]
  refine plain_mkObj (keys_ne (by kernel_rfl')) ?_
  intro e he
  simp only [List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false] at he
  rcases he with rfl | rfl | rfl
  · exact ⟨by plstr, by plstr⟩                                     -- name
  · refine ⟨by plstr, ?_⟩                                          -- params
    refine plain_mkObj (keys_ne (by kernel_rfl')) ?_
    intro e2 he2
    simp only [List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false] at he2
    rcases he2 with rfl
    exact ⟨by plstr, by plstr⟩                                     -- template
  · exact ⟨by plstr, by decide⟩                                    -- version

/-- The session's subset law is plain. -/
private theorem plain_law : Firewall.plain law = true := by
  rw [law]
  refine plain_mkObj (keys_ne (by kernel_rfl')) ?_
  intro e he
  simp only [List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false] at he
  rcases he with rfl | rfl
  · exact ⟨by plstr, by decide⟩                                    -- k
  · exact ⟨by plstr, by plstr⟩                                     -- law

/-- The query `build` falls back to for an empty one is plain. -/
private theorem plain_defaultQuery : Firewall.plain Spot.Partition.defaultQuery = true := by
  unfold Spot.Partition.defaultQuery
  -- `defaultQuery`'s keys are name, version, params — distinct but not ascending, so prove distinctness directly.
  refine plain_mkObj (by decide) ?_
  intro e he
  simp only [List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false] at he
  rcases he with rfl | rfl | rfl
  · refine ⟨by plstr, ?_⟩                                          -- name
    show Firewall.plain (Json.str "Q_word") = true
    plstr
  · exact ⟨by plstr, by decide⟩                                    -- version
  · refine ⟨by plstr, ?_⟩                                          -- params
    refine plain_mkObj (by decide) ?_                              -- keys X, W: distinct but not ascending
    intro e2 he2
    simp only [List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false] at he2
    rcases he2 with rfl | rfl
    · exact ⟨by plstr, by decide⟩                                  -- X
    · exact ⟨by plstr, by decide⟩                                  -- W

/-- The partition object is plain. The query member is `build`'s `if`, so split on it: either branch (`defaultQuery` or
`query`) is plain, and nothing decides the stuck `keys? query`. -/
private theorem plain_partition : Firewall.plain partitionObj = true := by
  unfold partitionObj Spot.Partition.build
  refine plain_mkObj (keys_ne (by kernel_rfl')) ?_
  intro e he
  simp only [List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false] at he
  rcases he with rfl | rfl | rfl
  · refine ⟨by plstr, ?_⟩                                          -- format
    show Firewall.plain (Json.str Flock.HmRow.PARTITION_V1) = true
    plstr
  · refine ⟨by plstr, ?_⟩                                          -- program
    show Firewall.plain (Json.str (hexOf programSha)) = true
    exact plainStr_hexOf _
  · refine ⟨by plstr, ?_⟩                                          -- query (build's if)
    dsimp only
    split <;> first | exact plain_defaultQuery | exact plain_query

/-- The composite's component parameters are plain. -/
private theorem plain_paramsJ (hpp : ProgramPlain) : Firewall.plain paramsJ = true := by
  rw [paramsJ]
  refine plain_mkObj (keys_ne (by kernel_rfl')) ?_
  intro e he
  simp only [List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false] at he
  rcases he with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  · exact ⟨by plstr, by decide⟩                                    -- anchors
  · -- challenger
    refine ⟨by plstr, ?_⟩
    refine plain_mkObj (keys_ne (by kernel_rfl')) ?_
    intro e2 he2
    simp only [List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false] at he2
    rcases he2 with rfl | rfl
    · -- key
      refine ⟨by plstr, ?_⟩
      refine plain_mkObj (keys_ne (by kernel_rfl')) ?_
      intro e3 he3
      simp only [List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false] at he3
      rcases he3 with rfl | rfl | rfl
      · exact ⟨by plstr, by decide⟩                                -- H
      · exact ⟨by plstr, by decide⟩                                -- W
      · exact ⟨by plstr, plainStr_hexOf _⟩                         -- public
    · exact ⟨by plstr, by plstr⟩                                   -- label
  · exact ⟨by plstr, by plstr⟩                                     -- delta
  · exact ⟨by plstr, by plstr⟩                                     -- form
  · exact ⟨by plstr, plain_law⟩                                    -- law
  · exact ⟨by plstr, by decide⟩                                    -- leaf_layer
  · exact ⟨by plstr, by decide⟩                                    -- log
  · exact ⟨by plstr, plain_partition⟩                              -- partition
  · exact ⟨by plstr, by decide⟩                                    -- ports
  · exact ⟨by plstr, hpp⟩                                          -- program
  · -- proofs
    refine ⟨by plstr, ?_⟩
    refine plain_mkObj (keys_ne (by kernel_rfl')) ?_
    intro e2 he2
    simp only [List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false] at he2
    rcases he2 with rfl | rfl | rfl
    · exact ⟨by plstr, plainStr_hexOf _⟩                           -- circuit
    · exact ⟨by plstr, plainStr_hexOf _⟩                           -- public
    · exact ⟨by plstr, by decide⟩                                  -- registered
  · exact ⟨by plstr, plain_query⟩                                  -- query
  · exact ⟨by plstr, by decide⟩                                    -- registered
  · exact ⟨by plstr, by plstr⟩                                     -- scheme
  · exact ⟨by plstr, plainStr_hexOf _⟩                             -- window_root
  · exact ⟨by plstr, by decide⟩                                    -- work_table

/-- The composite's parameters are plain. -/
private theorem plain_θJ (hpp : ProgramPlain) : Firewall.plain θJ = true := by
  rw [θJ]
  refine plain_mkObj (keys_ne (by kernel_rfl')) ?_
  intro e he
  simp only [List.mem_cons, List.not_mem_nil, or_false] at he
  rcases he with rfl
  refine ⟨by plstr, ?_⟩
  show (Firewall.plain (Json.mkObj [("id", .str VStar.Recursive.Sampled.PROTOCOL), ("params", paramsJ)])
      && Firewall.plainList []) = true
  rw [Bool.and_eq_true]
  refine ⟨?_, by decide⟩
  refine plain_mkObj (keys_ne (by kernel_rfl')) ?_
  intro e2 he2
  simp only [List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false] at he2
  rcases he2 with rfl | rfl
  · exact ⟨by plstr, by plstr⟩                                     -- id
  · exact ⟨by plstr, plain_paramsJ hpp⟩                            -- params

/-! ## The canonical-JSON equations the walk rewrites by -/

/-- `canonicalJson θ₀` reads back the composite's parameters. -/
private theorem canon_θ₀ (hJ : JsonCanon.JsonFacts) (hpp : ProgramPlain) :
    Core.canonicalJson θ₀ = .ok (jnorm θJ) :=
  hJ.canonicalJson_canon (plain_θJ hpp) (canon_ascii_bytes (hJ.canon_plain (plain_θJ hpp)))
    (hJ.canon_plain (plain_θJ hpp))

/-- `canonicalJson` of the component parameters' text reads them back. -/
private theorem canon_params (hJ : JsonCanon.JsonFacts) (hpp : ProgramPlain) :
    Core.canonicalJson (Firewall.printJson paramsJ).toUTF8 = .ok (jnorm paramsJ) :=
  hJ.canonicalJson_canon (plain_paramsJ hpp) (canon_ascii_bytes (hJ.canon_plain (plain_paramsJ hpp)))
    (hJ.canon_plain (plain_paramsJ hpp))

/-- The program file is the descriptor's text. -/
private theorem programFile_eq (hJ : JsonCanon.JsonFacts) (hpp : ProgramPlain) :
    programFile = (Firewall.printJson programJ).toUTF8 := by
  have hp : Firewall.plain programJ = true := hpp
  show ((Flock.canon programJ).toOption.getD "").toUTF8 = _
  rw [hJ.canon_plain hp]
  rfl

/-- The descriptor bytes `Program.parse` reads are the program file. -/
private theorem desc_eq (hJ : JsonCanon.JsonFacts) (hpp : ProgramPlain) :
    (⟨desc.toArray⟩ : ByteArray) = (Firewall.printJson programJ).toUTF8 := by
  have h : (⟨desc.toArray⟩ : ByteArray) = programFile := by
    show ByteArray.mk (programFile.data.toList).toArray = programFile
    rw [Array.toArray_toList]
  rw [h, programFile_eq hJ hpp]

/-- `Program.parse` of the descriptor's text is the program. -/
private theorem parse_program (hJ : JsonCanon.JsonFacts) (hpp : ProgramPlain) :
    Spot.Program.parse (Firewall.printJson programJ).toUTF8 = .ok program := by
  rw [← desc_eq hJ hpp]; exact Program.program_parse

/-! ## v2's parameters, with every field named -/

/-- The parameters `θ₀` decodes to, with each field `paramsOfJson` reads. -/
private theorem params_full (hJ : JsonCanon.JsonFacts) (hpp : ProgramPlain) (hN : NormFacts) :
    ∃ p : Spot.OneStageZk.Params,
    ParamsAt θ₀ p ∧ p.own.partition = partitionObj ∧ p.own.program = programFile ∧
    p.own.circuit = hexOf circuitSha ∧ p.own.pub = hexOf (Sha512.hash publicFile) ∧
    p.base.verifier.program = program ∧ p.base.verifier.query = query ∧ p.base.verifier.law = law ∧
    p.base.verifier.workTable = none ∧ p.base.challenger = some (key₀, label₀) := by
  have hA := canon_θ₀ hJ hpp
  have hC := canon_params hJ hpp
  have hE := parse_program hJ hpp
  have hU := Flock.Hex.unhex_hexOf keyPub
  have hpf := programFile_eq hJ hpp
  -- `canon` of each plain literal directly (the decoders read the literal's fields, not `norm`'s).
  have hBp : Flock.canon paramsJ = .ok (Firewall.printJson paramsJ) := hJ.canon_plain (plain_paramsJ hpp)
  have hDp : Flock.canon programJ = .ok (Firewall.printJson programJ) :=
    hJ.canon_plain (show Firewall.plain programJ = true from hpp)
  obtain ⟨hNθ, hNp⟩ := hN
  refine ⟨?p, ⟨[{ id := VStar.Recursive.Sampled.PROTOCOL, params := Firewall.printJson paramsJ }], ?decode,
      { id := VStar.Recursive.Sampled.PROTOCOL, params := Firewall.printJson paramsJ }, ?find, ?pchain⟩,
      ?f1, ?f2, ?f3, ?f4, ?f5, ?f6, ?f7, ?f8, ?f9⟩
  case decode =>
    -- Rewrite the outer `canonicalJson` by `hA`, then `norm θJ` to the literal `θJ` so field lookups are on a concrete
    -- object (`norm` is well-founded-recursive and does not reduce in the kernel).
    show (Core.canonicalJson θ₀ >>= Core.Composition.decodeParams) = .ok
      [({ id := VStar.Recursive.Sampled.PROTOCOL, params := Firewall.printJson paramsJ } : Core.Composition.Named)]
    rw [hA, hNθ]
    fwd_walk [hBp] block [Flock.canon] shallow []
  case find => kernel_rfl'
  case pchain =>
    -- Likewise rewrite the outer `canonicalJson` by `hC`, then `norm paramsJ` to the literal `paramsJ`.
    show (Core.canonicalJson (Firewall.printJson paramsJ).toUTF8 >>= VStar.Recursive.Sampled.paramsOfJson) = _
    rw [hC, hNp]
    fwd_walk [hDp, hE, hU] block [Flock.canon, Spot.Program.parse, Flock.unhex] shallow []
  case f1 => kernel_rfl'            -- own.partition = partitionObj
  case f2 => rw [hpf]               -- own.program = programFile
  case f3 => kernel_rfl'            -- own.circuit
  case f4 => kernel_rfl'            -- own.pub
  case f5 => rfl                    -- verifier.program = program
  case f6 => kernel_rfl'            -- verifier.query = query
  case f7 => kernel_rfl'            -- verifier.law = law
  case f8 => rfl                    -- verifier.workTable = none
  case f9 => kernel_rfl'            -- challenger = some (key₀, label₀)

theorem params_spec (hJ : JsonCanon.JsonFacts) (hpp : ProgramPlain) (hN : NormFacts) : ∃ p, ParamsAt θ₀ p ∧
    p.own.partition = partitionObj ∧ p.own.program = programFile ∧ p.base.verifier.program = program ∧
    p.base.verifier.query = query ∧ p.base.verifier.law = law ∧ p.base.verifier.workTable = none ∧
    p.base.challenger = some (key₀, label₀) := by
  obtain ⟨p, hpat, hpart, hprog, _, _, hvp, hvq, hvl, hvt, hch⟩ := params_full hJ hpp hN
  exact ⟨p, hpat, hpart, hprog, hvp, hvq, hvl, hvt, hch⟩

theorem digests (hJ : JsonCanon.JsonFacts) (hpp : ProgramPlain) (hN : NormFacts) :
    programDigestOf θ₀ = some (Verity.sha512 circuitFile.data.toList) ∧
      publicDigestOf θ₀ = some (Verity.sha512 publicFile.data.toList) := by
  obtain ⟨p, hpat, _, _, hc, hpub, _⟩ := params_full hJ hpp hN
  refine ⟨?_, ?_⟩
  · rw [programDigestOf, paramsOf_eq hpat, Option.bind_some, hc]
    simp only [digestOf, Flock.Hex.unhex_hexOf, circuitSha]
    rw [← sha512_data circuitFile]
  · rw [publicDigestOf, paramsOf_eq hpat, Option.bind_some, hpub]
    simp only [digestOf, Flock.Hex.unhex_hexOf]
    rw [← sha512_data publicFile]

end Verity.Sampled.Instance.HonestSession.Params
