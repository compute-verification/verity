module

public import Proofs.Properties.Verity.Sampled.HonestSession.Data.Inputs
public import Proofs.Properties.Verity.Sampled.HonestSession.JsonCanon
public import Proofs.Properties.Verity.Sampled.HonestSession.Program
public import Proofs.Properties.Verity.Sampled.HonestSession.ProgramParse
public import Proofs.Properties.Verity.Sampled.HonestSession.ProgramProof
public import Proofs.Properties.Verity.Sampled.HonestSession.Hex
public import Proofs.Properties.Verity.Sampled.Accepts
public import Proofs.Flock.Recursive.FirewallObj
public import Proofs.Flock.Claims.Complete.Json.Canon
public import Proofs.Flock.SessionLoop.Concrete.Fwd
import Proofs.Properties.Verity.Sampled.HonestSession.Delta
import Proofs.Properties.Verity.Sampled.HonestSession.PortsFacts
import Proofs.Pouw.Period.Sort
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
(`digests`). Every step rewrites by `Flock.canon`'s equation lemmas and the JSON facts (`JsonCanon.JsonFacts`), which
`params_full` discharges by `JsonCanon.jsonFacts` (the canonical-JSON reader's deliverable); nothing is decided over a
hash.

Two facts about `programJ` remain this piece's hypotheses, as `ProgramPlain` and `NormFacts`: that it is plain, and that
the parser rebuilds its exact tree (`jnorm programJ = programJ`). Both are concrete-JSON facts of this instance that no
tactic here shows robustly; their `plain_mkObj`/`norm_mkObj_fix` decomposition over the descriptor's deeply nested,
partly non-ascending `definitions` is deferred. The composite's and the component's own norm and plainness facts are
discharged below from these two (`normFacts_derive`, `plain_θJ`, `plain_paramsJ`).
-/

@[expose] public section

namespace Verity.Sampled.Instance.HonestSession.Params

open Flock
open Lean (Json)
open FlockSoundness.Discharge.Recursion.FirewallObj
open FlockSoundness.Discharge.Recursion.FirewallJson renaming norm → jnorm
open FlockSoundness.Discharge.Recursion.FirewallJson (normObj normObj_eq normList normList_eq jsonCharsList_eq
  jsonCharsObj_eq memberChars)
open Std.DTreeMap.Internal (Impl)

set_option maxHeartbeats 400000
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

/-- The parser's rebuild (`norm`) of these two JSON literals is the literal itself. `canonicalJson` returns `norm` of the
parsed value, and the decoders must then read that value's fields as a concrete object; but `norm` is
well-founded-recursive, so the kernel does not reduce it (`canon_norm` only gives `canon (norm j) = canon j`, not
`norm j = j`, and no `plain j → norm j = j` lemma exists).

`jnorm θJ = θJ` and `jnorm paramsJ = paramsJ` are *discharged* below (`norm_θJ`, `norm_paramsJ`): `norm` rebuilds each
object as the tree of its members re-inserted in ascending key order (`normObj`), which reproduces a literal exactly when
that literal is `mkObj` of an ascending, duplicate-free key list at every level (`norm_mkObj_fix`), as `θJ`, `paramsJ`
and all but one of `programJ`'s objects are. The one object this piece does not yet descend is `programJ`, for two
reasons: its literal is nested ~15 objects deep, so discharging its fixpoint the same way needs a per-level tree equality
for each (the concrete `kernel_rfl'` over the whole literal exhausts memory, so each level's tree equality is proved with
its values abstract, as the `*_tree` lemmas do); and its `definitions` member is the one non-ascending literal (source
keys `Input16_v1, Unit_v1, Root_v1`; sorted order `Input16_v1 < Root_v1 < Unit_v1`), so `norm_mkObj_fix` does not apply
to it directly — it needs a value-preserving reorder to the ascending key list first (`mkObj` of either list is the same
tree, since `ofList` sorts), which the scratch `NormProbe` validated through `norm_mkObj_fix_ns`. Until then
`norm_θJ`/`norm_paramsJ` take `jnorm programJ = programJ` as a hypothesis, and `NormFacts` is exactly that one
structural fact. -/
def NormFacts : Prop := jnorm programJ = programJ

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

/-! ### `Printable` (every printed character is ASCII, `< 0x7f`), the witness `canon_plain` now takes

A whole-literal `decide +kernel` (as `ProgramParse.printable_programJ` uses for the descriptor) cannot prove this for
`paramsJ` or `θJ`: both splice hash-derived hex strings (`hexOf programSha`, the digests, the key and window root), which
the kernel does not reduce. Instead we compose printability over the literal's structure — `printable_mkObj`,
`printable_arr`, `printable_hexOf` and the kernel for each concrete leaf — mirroring the `plain_*` proofs above. -/

/-- `Printable j` reads back as a condition on `jsonChars j`, since `printJson` is their `String.ofList`. -/
private theorem printable_iff_chars {j : Json} :
    JsonCanon.Printable j ↔ ∀ c ∈ Firewall.jsonChars j, c.toNat < 0x7f := by
  simp only [JsonCanon.Printable, Firewall.printJson, String.toList_ofList]

/-- A character of an intercalation is the separator's, or one of a joined piece. -/
private theorem mem_intercalate_rev {α : Type} {sep : List α} {c : α} :
    ∀ (L : List (List α)), c ∈ sep.intercalate L → c ∈ sep ∨ ∃ x ∈ L, c ∈ x
  | [], h => by simp [List.intercalate] at h
  | [x], h => by rw [List.intercalate_singleton] at h; exact Or.inr ⟨x, List.mem_singleton_self x, h⟩
  | x :: y :: L, h => by
    rw [List.intercalate_cons_cons] at h
    rcases List.mem_append.1 h with h1 | h2
    · rcases List.mem_append.1 h1 with hx | hs
      · exact Or.inr ⟨x, List.mem_cons_self, hx⟩
      · exact Or.inl hs
    · rcases mem_intercalate_rev (y :: L) h2 with hs | ⟨z, hz, hcz⟩
      · exact Or.inl hs
      · exact Or.inr ⟨z, List.mem_cons_of_mem _ hz, hcz⟩

/-- An array of printable values prints to ASCII: the brackets and the separating commas are ASCII, and each element's
text is ASCII by hypothesis. -/
private theorem printable_arr {arr : Array Json} (h : ∀ e ∈ arr, JsonCanon.Printable e) :
    JsonCanon.Printable (Json.arr arr) := by
  obtain ⟨a⟩ := arr
  rw [printable_iff_chars]
  intro c hc
  rw [Firewall.jsonChars] at hc
  rcases List.mem_cons.1 hc with rfl | hc1
  · decide
  rcases List.mem_append.1 hc1 with hmid | hbr
  · rcases mem_intercalate_rev _ hmid with hsep | ⟨x, hx, hcx⟩
    · rw [List.mem_singleton] at hsep; subst hsep; decide
    · rw [jsonCharsList_eq] at hx
      obtain ⟨e, he, rfl⟩ := List.mem_map.1 hx
      exact (printable_iff_chars.1 (h e (Array.mem_toList_iff.1 he))) c hcx
  · rw [List.mem_singleton] at hbr; subst hbr; decide

/-- An object whose keys and values print to ASCII prints to ASCII: the braces, quotes, colons and commas are ASCII, and
each member's key and value text is ASCII by hypothesis. The tree the keys sort into is a permutation of the member list,
so the per-member hypothesis covers every member of it. -/
private theorem printable_mkObj {L : List (String × Json)} (hd : L.Pairwise fun a b => a.1 ≠ b.1)
    (h : ∀ e ∈ L, (∀ c ∈ e.1.toList, c.toNat < 0x7f) ∧ JsonCanon.Printable e.2) :
    JsonCanon.Printable (Json.mkObj L) := by
  rw [printable_iff_chars, mkObj_eq, Firewall.jsonChars]
  intro c hc
  rcases List.mem_cons.1 hc with rfl | hc1
  · decide
  rcases List.mem_append.1 hc1 with hmid | hbr
  · rcases mem_intercalate_rev _ hmid with hsep | ⟨x, hx, hcx⟩
    · rw [List.mem_singleton] at hsep; subst hsep; decide
    · rw [jsonCharsObj_eq] at hx
      obtain ⟨e, he, rfl⟩ := List.mem_map.1 hx
      obtain ⟨e', he', rfl⟩ := List.mem_map.1 ((perm_ofList hd).mem_iff.1 he)
      rw [memberChars] at hcx
      rcases List.mem_append.1 hcx with hkey | hval
      · rcases List.mem_cons.1 hkey with rfl | hk2
        · decide
        rcases List.mem_append.1 hk2 with hkk | hqc
        · exact (h e' he').1 c hkk
        · fin_cases hqc <;> decide
      · exact (printable_iff_chars.1 (h e' he').2) c hval

/-- A digest's hex name prints to ASCII: every hex digit is below `0x7f`, as are the enclosing quotes. -/
private theorem printable_hexOf (b : ByteArray) : JsonCanon.Printable (.str (hexOf b)) := by
  rw [printable_iff_chars, Firewall.jsonChars]
  intro c hc
  rcases List.mem_cons.1 hc with rfl | hc1
  · decide
  rcases List.mem_append.1 hc1 with hmain | hq
  · rw [Hex.hexOf_toList, List.mem_flatMap] at hmain
    obtain ⟨x, _, hx⟩ := hmain
    have hlt : x.toNat < 256 := x.toNat_lt
    have key : ∀ d < 16, (Hex.hd d).toNat < 0x7f := by decide +kernel
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hx
    rcases hx with rfl | rfl
    · exact key _ (by omega)
    · exact key _ (Nat.mod_lt _ (by decide))
  · rw [List.mem_singleton] at hq; subst hq; decide

set_option maxHeartbeats 4000000 in
/-- The partition object prints to ASCII: its format tag and query are concrete literals, its one spliced leaf is the
program digest (`printable_hexOf`). -/
private theorem printable_partition : JsonCanon.Printable partitionObj := by
  unfold partitionObj Spot.Partition.build
  refine printable_mkObj (keys_ne (by kernel_rfl')) ?_
  intro e he
  simp only [List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false] at he
  rcases he with rfl | rfl | rfl
  · refine ⟨by decide, ?_⟩                                          -- format
    show JsonCanon.Printable (Json.str Flock.HmRow.PARTITION_V1)
    exact ProgramParse.printable_of _ (by decide +kernel)
  · refine ⟨by decide, ?_⟩                                          -- program
    show JsonCanon.Printable (Json.str (hexOf programSha))
    exact printable_hexOf _
  · refine ⟨by decide, ?_⟩                                          -- query (build's if)
    dsimp only
    split <;> exact ProgramParse.printable_of _ (by decide +kernel)

set_option maxHeartbeats 4000000 in
/-- The component parameters print to ASCII: every character of `printJson paramsJ` is below `0x7f`. The concrete members
are discharged by the kernel; the hash-spliced ones (the challenger key, the proofs' digests, the program and the window
root) by `printable_hexOf`, `printable_partition` and `ProgramParse.printable_programJ`. -/
theorem printable_paramsJ : JsonCanon.Printable paramsJ := by
  rw [paramsJ]
  refine printable_mkObj (keys_ne (by kernel_rfl')) ?_
  intro e he
  simp only [List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false] at he
  rcases he with rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl | rfl
  · exact ⟨by decide, ProgramParse.printable_of _ (by decide)⟩     -- anchors
  · -- challenger
    refine ⟨by decide, ?_⟩
    refine printable_mkObj (keys_ne (by kernel_rfl')) ?_
    intro e2 he2
    simp only [List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false] at he2
    rcases he2 with rfl | rfl
    · -- key
      refine ⟨by decide, ?_⟩
      refine printable_mkObj (keys_ne (by kernel_rfl')) ?_
      intro e3 he3
      simp only [List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false] at he3
      rcases he3 with rfl | rfl | rfl
      · exact ⟨by decide, ProgramParse.printable_of _ (by decide +kernel)⟩  -- H
      · exact ⟨by decide, ProgramParse.printable_of _ (by decide +kernel)⟩  -- W
      · exact ⟨by decide, printable_hexOf _⟩                       -- public
    · exact ⟨by decide, ProgramParse.printable_of _ (by decide +kernel)⟩    -- label
  · exact ⟨by decide, ProgramParse.printable_of _ (by decide +kernel)⟩      -- delta
  · exact ⟨by decide, ProgramParse.printable_of _ (by decide +kernel)⟩      -- form
  · exact ⟨by decide, ProgramParse.printable_of _ (by decide +kernel)⟩      -- law
  · exact ⟨by decide, ProgramParse.printable_of _ (by decide +kernel)⟩      -- leaf_layer
  · exact ⟨by decide, ProgramParse.printable_of _ (by decide +kernel)⟩      -- log
  · exact ⟨by decide, printable_partition⟩                         -- partition
  · exact ⟨by decide, ProgramParse.printable_of _ (by decide +kernel)⟩      -- ports
  · exact ⟨by decide, ProgramParse.printable_programJ⟩             -- program
  · -- proofs
    refine ⟨by decide, ?_⟩
    refine printable_mkObj (keys_ne (by kernel_rfl')) ?_
    intro e2 he2
    simp only [List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false] at he2
    rcases he2 with rfl | rfl | rfl
    · exact ⟨by decide, printable_hexOf _⟩                         -- circuit
    · exact ⟨by decide, printable_hexOf _⟩                         -- public
    · exact ⟨by decide, ProgramParse.printable_of _ (by decide +kernel)⟩    -- registered
  · exact ⟨by decide, ProgramParse.printable_of _ (by decide +kernel)⟩      -- query
  · exact ⟨by decide, ProgramParse.printable_of _ (by decide +kernel)⟩      -- registered
  · exact ⟨by decide, ProgramParse.printable_of _ (by decide +kernel)⟩      -- scheme
  · exact ⟨by decide, printable_hexOf _⟩                           -- window_root
  · exact ⟨by decide, ProgramParse.printable_of _ (by decide +kernel)⟩      -- work_table

set_option maxHeartbeats 4000000 in
/-- The composite's parameters print to ASCII; the `Printable` witness `canon_plain` takes for `θJ`. The one component's
`id` is a concrete literal and its `params` is `paramsJ`. -/
theorem printable_θJ : JsonCanon.Printable θJ := by
  rw [θJ]
  refine printable_mkObj (keys_ne (by kernel_rfl')) ?_
  intro e he
  simp only [List.mem_cons, List.not_mem_nil, or_false] at he
  rcases he with rfl
  refine ⟨by decide, ?_⟩                                          -- components
  refine printable_arr ?_
  intro e2 he2
  have hmem : e2 = Json.mkObj [("id", .str VStar.Recursive.Sampled.PROTOCOL), ("params", paramsJ)] := by
    simpa using he2
  subst hmem
  refine printable_mkObj (keys_ne (by kernel_rfl')) ?_
  intro e3 he3
  simp only [List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false] at he3
  rcases he3 with rfl | rfl
  · exact ⟨by decide, ProgramParse.printable_of _ (by decide +kernel)⟩      -- id
  · exact ⟨by decide, printable_paramsJ⟩                           -- params

/-- `canonicalJson θ₀` reads back the composite's parameters. -/
private theorem canon_θ₀ (hJ : JsonCanon.JsonFacts) (hpp : ProgramPlain) :
    Core.canonicalJson θ₀ = .ok (jnorm θJ) :=
  hJ.canonicalJson_canon (plain_θJ hpp) (canon_ascii_bytes (hJ.canon_plain (plain_θJ hpp) printable_θJ))
    (hJ.canon_plain (plain_θJ hpp) printable_θJ)

/-- `canonicalJson` of the component parameters' text reads them back. -/
private theorem canon_params (hJ : JsonCanon.JsonFacts) (hpp : ProgramPlain) :
    Core.canonicalJson (Firewall.printJson paramsJ).toUTF8 = .ok (jnorm paramsJ) :=
  hJ.canonicalJson_canon (plain_paramsJ hpp) (canon_ascii_bytes (hJ.canon_plain (plain_paramsJ hpp) printable_paramsJ))
    (hJ.canon_plain (plain_paramsJ hpp) printable_paramsJ)

/-- The program file is the descriptor's text. -/
private theorem programFile_eq (hJ : JsonCanon.JsonFacts) (hpp : ProgramPlain) :
    programFile = (Firewall.printJson programJ).toUTF8 := by
  have hp : Firewall.plain programJ = true := hpp
  show ((Flock.canon programJ).toOption.getD "").toUTF8 = _
  rw [hJ.canon_plain hp ProgramParse.printable_programJ]
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
  rw [← desc_eq hJ hpp]; exact ProgramParse.program_parse

/-- A spliced 64-byte-hash digest passes `paramsOfJson`'s form check (`digest?`): its hex is 128 characters, non-empty,
all `[0-9a-f]`. Proved from the hex facts (piece P6), so the walk need not evaluate the hash. -/
private theorem digest?_hexOf {b : ByteArray} (h : b.size = 64) : Spot.OneStageZk.digest? (hexOf b) = true := by
  have hlen : (hexOf b).length = 128 := by rw [Hex.length_hexOf, h]
  have hne : hexOf b ≠ "" := fun he => by simp [he] at hlen
  have hemp : (hexOf b).isEmpty = false := String.isEmpty_eq_false_iff.mpr hne
  have hsha := Hex.isSha512Hex_hexOf h
  unfold Flock.HmRow.isSha512Hex at hsha
  rw [Bool.and_eq_true] at hsha
  obtain ⟨_, hall⟩ := hsha
  unfold Spot.OneStageZk.digest? Spot.isHex
  rw [hlen, hemp, hall]
  decide

/-- Insert into a list, comparing with `String`'s `<`. Written over `<` (not `≤`) on purpose: the kernel evaluates
`String`'s `<` (its `Decidable` instance is computable), while `≤`'s is not, so `List.insertionSort (· ≤ ·)` would not
reduce. Structural, so the kernel runs it. -/
private def insLt (a : String) : List String → List String
  | [] => [a]
  | b :: l => if a < b then a :: b :: l else b :: insLt a l

/-- Insertion sort of a `String` list by `<`; kernel-reducible (see `insLt`). -/
private def sortLt (l : List String) : List String := l.foldr insLt []

private theorem insLt_perm (a : String) : ∀ l : List String, (insLt a l).Perm (a :: l)
  | [] => .refl _
  | b :: l => by
    simp only [insLt]
    split
    · exact .refl _
    · exact ((insLt_perm a l).cons b).trans (List.Perm.swap a b l)

private theorem sortLt_perm : ∀ l : List String, (sortLt l).Perm l
  | [] => .refl _
  | a :: l => by
    simp only [sortLt, List.foldr]
    exact (insLt_perm a _).trans ((sortLt_perm l).cons a)

private theorem insLt_sorted {a : String} : ∀ {l : List String}, l.Pairwise (· ≤ ·) →
    (insLt a l).Pairwise (· ≤ ·)
  | [], _ => by simp [insLt]
  | b :: l, h => by
    simp only [insLt]
    split
    · rename_i hab
      refine List.Pairwise.cons (fun x hx => ?_) h
      rcases List.mem_cons.1 hx with rfl | hx
      · exact le_of_lt hab
      · exact le_of_lt (lt_of_lt_of_le hab ((List.pairwise_cons.1 h).1 x hx))
    · rename_i hab
      have hba : b ≤ a := not_lt.1 hab
      refine List.Pairwise.cons (fun x hx => ?_) (insLt_sorted (List.pairwise_cons.1 h).2)
      rcases List.mem_cons.1 ((insLt_perm a l).mem_iff.1 hx) with rfl | hx'
      · exact hba
      · exact (List.pairwise_cons.1 h).1 x hx'

private theorem sortLt_sorted : ∀ l : List String, (sortLt l).Pairwise (· ≤ ·)
  | [] => List.Pairwise.nil
  | a :: l => by simp only [sortLt, List.foldr]; exact insLt_sorted (sortLt_sorted l)

/-- `sortStrings` leaves an already-ascending array unchanged: its `qsort` is the sorted permutation, which a
duplicate-free ascending array already is. The walk tries this first on every `sortStrings`; its side condition
`Pairwise (· < ·)` costs only `n - 1` string `<` comparisons (cheap), so each already-sorted key set — every subobject,
and the top-level object's own keys read back from the tree — reduces at that price. Only the push-chain key set (built in
insertion order) falls through to `sortStrings_eq_sortLt`. -/
private theorem sortStrings_of_sorted {as : Array String} (h : as.toList.Pairwise (· < ·)) :
    Spot.sortStrings as = as := by
  show as.qsort (· < ·) = as
  calc as.qsort (· < ·) = (as.qsort (· < ·)).toList.toArray := rfl
    _ = as.toList.toArray := by rw [Properties.Pouw.Period.SecurityProofs.Sort.qsort_of_sorted h]
    _ = as := rfl

/-- `sortStrings` equals `sortLt` of its list. `sortStrings = Array.qsort (· < ·)`, which the kernel cannot run
(well-founded recursion); `sortLt` is structural over `String`'s computable `<`, so the kernel evaluates it. Both are
`Pairwise (· ≤ ·)` permutations of the input, hence equal (`Perm.eq_of_pairwise`, `≤` on `String` antisymmetric). This is
the fallback for the one push-chain key set `sortStrings_of_sorted` cannot take (it is not ascending); sorting it in the
kernel costs `O(n²)` string `<` comparisons, so the walk is given `sortStrings_of_sorted` first to keep this to one set. -/
private theorem sortStrings_eq_sortLt (as : Array String) :
    Spot.sortStrings as = (sortLt as.toList).toArray := by
  show as.qsort (· < ·) = (sortLt as.toList).toArray
  have hl : (as.qsort (· < ·)).toList = sortLt as.toList :=
    List.Perm.eq_of_pairwise (fun _ _ _ _ h1 h2 => String.le_antisymm h1 h2)
      (Properties.Pouw.Period.SecurityProofs.Sort.qsort_sorted as) (sortLt_sorted _)
      ((Properties.RemoteAudit.qsort_toList_perm as _).trans (sortLt_perm _).symm)
  calc as.qsort (· < ·) = (as.qsort (· < ·)).toList.toArray := rfl
    _ = (sortLt as.toList).toArray := by rw [hl]

/-- `keys?` of an `mkObj`, as its definitional equation. Keyed on `mkObj` (not `.obj`) so the walk's reducible match
finds it in the literal objects of `Data/`; it exposes the inner `sortStrings` (over the tree's structurally-reducible key
array) that `sortStrings_of_sorted` then rewrites away. -/
private theorem keys?_mkObj (o : List (String × Lean.Json)) :
    Spot.keys? (Lean.Json.mkObj o) = some (Spot.sortStrings ((Std.TreeMap.Raw.ofList o).toArray.map (·.1))) := rfl

/-- `Spot.OneStage.poolValues` on the empty pool: the supplied, output and row-format values pass through unchanged,
independent of the program digest. Fed to the walk so it rewrites past the call (which is otherwise tainted by the
blocked `program`, whose digest the `none` case never reads) instead of evaluating it. -/
private theorem poolValues_none (program : String) (log : Array Lean.Json) (supplied outputs schemas : Option Lean.Json) :
    Spot.OneStage.poolValues none program log supplied outputs schemas = (supplied, outputs, schemas) := rfl

/-- `keys?` of a `.obj`, as its definitional equation. Keyed on the `.obj` constructor (not `mkObj`), for the objects a
field read extracts from a literal: `getD (mkObj …) k` reduces (in the kernel, which the walk reaches through `normStep`
once the sort taint stops it) to the stored `.obj <tree>`, never back to a syntactic `mkObj`, so `keys?_mkObj` cannot fire
on it; this exposes the inner `sortStrings` over the tree's structurally-reducible key array for `sortStrings_of_sorted`. -/
private theorem keys?_obj (kvs : Std.TreeMap.Raw String Lean.Json) :
    Spot.keys? (Lean.Json.obj kvs) = some (Spot.sortStrings (kvs.toArray.map (·.1))) := rfl

/-! ## v2's parameters, with every field named -/

/-- A verifier that names its window's open root has no unwindowed registered value: `unwindowed` returns `#[]` at once
(the `windowRoot.isSome` branch), never reaching its well-founded body, which the walk's kernel reduction cannot step.
The `windowRoot := some r` field is structural in the statement (not a side condition on `v.windowRoot`), so the walk's
rewrite matches the final guard's concrete record by `kabstract` alone — a side condition the walk cannot discharge, the
record being tainted by the derived `query` and `partition`. -/
private theorem unwindowed_some
    (program : Spot.Program) (query law : Json) (workTable : Option Json) (scheme : String)
    (ports : Array (String × Json)) (leafLayer : Option String) (anchors log : Array Json) (partition : Option Json)
    (registered : Array (String × Nat)) (params : Array Spot.Registered.Param)
    (supplied : Array (String × Spot.Supplied.Rows)) (pinned : Array (String × String))
    (outputs : Array (String × String × Spot.Registration.Earlier)) (binding : Option (String × String))
    (schemas : Array (String × String)) (r : String) (held : Array String)
    (declared : Option (Nat × Array (Nat × String))) (seedSalt : Option String) (pool : Option Spot.Registration.Pool)
    (tied : Array String) (members : Array (Array (Nat × Nat) × Nat)) :
    Spot.Registration.unwindowed
      { program, query, law, workTable, scheme, ports, leafLayer, anchors, log, partition, registered, params,
        supplied, pinned, outputs, binding, schemas, windowRoot := some r, held, declared, seedSalt, pool, tied,
        members } = #[] := by
  unfold Spot.Registration.unwindowed; rfl

/-! ## The parser's rebuild (`jnorm`) of `θJ` and `paramsJ` is the literal

`norm (mkObj L)` rebuilds the object's tree by folding `insert` over `L`; this equals `mkObj L`'s own tree
(`ofList L`) when `L`'s keys ascend without duplicates and every value is itself a `norm`-fixpoint. `norm_mkObj_fix`
packages that once, taking the tree equality `hofl` as a hypothesis — discharged per call by `kernel_rfl'` over the
abstract key set (a reusable `*_tree` lemma whose values are universally quantified, so the kernel never carries the
giant `programJ` value). `nfix` applies it (and `norm_arr_fix`) recursively for the shallow literals. -/

private theorem norm_obj_eq {t : Impl String (fun _ => Json)} :
    jnorm (.obj ⟨⟨t⟩⟩) = .obj (normObj t ∅) := rfl

private theorem foldl_norm_eq {L : List (String × Json)} (hv : ∀ e ∈ L, jnorm e.2 = e.2)
    (acc : Std.TreeMap.Raw String Json) :
    L.foldl (fun a e => a.insert e.1 (jnorm e.2)) acc = L.foldl (fun a e => a.insert e.1 e.2) acc := by
  induction L generalizing acc with
  | nil => rfl
  | cons h t ih =>
    simp only [List.foldl_cons]
    rw [hv h List.mem_cons_self, ih (fun e he => hv e (List.mem_cons_of_mem _ he))]

private theorem asc_of {L : List (String × Json)}
    (h : decide (L.Pairwise fun a b => compare a.1 b.1 = .lt) = true) : Asc Prod.fst L :=
  (of_decide_eq_true h : L.Pairwise fun a b => compare a.1 b.1 = .lt)

/-- `norm (mkObj L) = mkObj L` given keys ascending (`hasc`), each value a fixpoint (`hv`), and the concrete tree
equality (`hofl`). `hofl` is general in `L`; callers discharge it by a `*_tree` lemma or `kernel_rfl'`. -/
private theorem norm_mkObj_fix {L : List (String × Json)} (hasc : Asc Prod.fst L)
    (hv : ∀ e ∈ L, jnorm e.2 = e.2)
    (hofl : L.foldl (fun a e => a.insert e.1 e.2) (∅ : Std.TreeMap.Raw String Json)
      = ⟨⟨Impl.Const.ofList L⟩⟩) :
    jnorm (Json.mkObj L) = Json.mkObj L := by
  rw [mkObj_eq, norm_obj_eq]
  congr 1
  rw [normObj_eq, toListModel_ofList hasc, List.foldl_map, foldl_norm_eq hv, hofl]

private theorem norm_arr_fix {a : Array Json} (hv : ∀ x ∈ a.toList, jnorm x = x) :
    jnorm (.arr a) = .arr a := by
  obtain ⟨l⟩ := a
  show Json.arr ⟨FlockSoundness.Discharge.Recursion.FirewallJson.normList l⟩ = Json.arr ⟨l⟩
  rw [normList_eq]
  refine congrArg (fun z => Json.arr ⟨z⟩) ?_
  exact (List.map_congr_left (fun x hx => hv x hx)).trans (List.map_id l)

/-- Discharge `jnorm j = j` for a shallow literal built from `mkObj`, array literals and leaves: `List.forall_mem_cons`
splits each object's/array's membership into a conjunction, `repeat' apply And.intro` breaks it into one goal per member,
and each is recursed (the empty tail closes by `List.forall_mem_nil`). -/
syntax "nfix" : tactic
macro_rules
  | `(tactic| nfix) =>
    `(tactic|
      first
      | rfl
      | (refine norm_arr_fix ?_;
         simp only [List.toList_toArray, List.forall_mem_cons];
         repeat' apply And.intro) <;> (first | nfix | exact List.forall_mem_nil _)
      | (refine norm_mkObj_fix (asc_of (by kernel_rfl')) ?_ (by kernel_rfl');
         simp only [List.forall_mem_cons];
         repeat' apply And.intro) <;> (first | nfix | exact List.forall_mem_nil _))

private theorem norm_query : jnorm query = query := by unfold query; nfix
private theorem norm_law : jnorm law = law := by unfold law; nfix

/-- The partition object's `query` member is `build`'s `if` on `keys? query`; `query` is non-empty, so it is `query`,
whose rebuild is `norm_query`; the other two members are string leaves. -/
private theorem norm_partition : jnorm partitionObj = partitionObj := by
  have hq : Spot.keys? query = some #["name", "params", "version"] := by
    unfold query; rw [keys?_mkObj, sortStrings_eq_sortLt]; kernel_rfl'
  have hif : (if Spot.keys? query == some #[] then Spot.Partition.defaultQuery else query) = query := by
    rw [hq]; kernel_rfl'
  unfold partitionObj Spot.Partition.build
  rw [hif]
  refine norm_mkObj_fix (asc_of (by kernel_rfl')) ?_ (by kernel_rfl')
  intro e he
  simp only [List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false] at he
  rcases he with rfl | rfl | rfl
  · nfix             -- format
  · nfix             -- program
  · exact norm_query -- query

/-- The tree equality for `paramsJ`'s 16-key object, values universally quantified so `kernel_rfl'` never carries the
giant `programJ` value (which otherwise exhausts memory). Instantiated at `paramsJ`'s values it discharges `hofl`. -/
private theorem paramsJ_tree (v0 v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 : Json) :
    ([("anchors", v0), ("challenger", v1), ("delta", v2), ("form", v3), ("law", v4), ("leaf_layer", v5),
      ("log", v6), ("partition", v7), ("ports", v8), ("program", v9), ("proofs", v10), ("query", v11),
      ("registered", v12), ("scheme", v13), ("window_root", v14), ("work_table", v15)]
        : List (String × Json)).foldl (fun a e => a.insert e.1 e.2) (∅ : Std.TreeMap.Raw String Json)
      = ⟨⟨Impl.Const.ofList
          [("anchors", v0), ("challenger", v1), ("delta", v2), ("form", v3), ("law", v4), ("leaf_layer", v5),
           ("log", v6), ("partition", v7), ("ports", v8), ("program", v9), ("proofs", v10), ("query", v11),
           ("registered", v12), ("scheme", v13), ("window_root", v14), ("work_table", v15)]⟩⟩ := by
  kernel_rfl'

/-- The tree equality for `θJ`'s inner component object `{id, params}`, values abstract. -/
private theorem θinner_tree (v0 v1 : Json) :
    ([("id", v0), ("params", v1)] : List (String × Json)).foldl (fun a e => a.insert e.1 e.2)
        (∅ : Std.TreeMap.Raw String Json)
      = ⟨⟨Impl.Const.ofList [("id", v0), ("params", v1)]⟩⟩ := by kernel_rfl'

/-- The tree equality for `θJ`'s top object `{components}`. -/
private theorem θtop_tree (v0 : Json) :
    ([("components", v0)] : List (String × Json)).foldl (fun a e => a.insert e.1 e.2)
        (∅ : Std.TreeMap.Raw String Json)
      = ⟨⟨Impl.Const.ofList [("components", v0)]⟩⟩ := by kernel_rfl'

/-- `paramsJ`'s rebuild is itself, given the program descriptor's own fixpoint (`hprog`). Every other member is a shallow
literal (`nfix`), the subset law (`norm_law`), the partition object (`norm_partition`) or the query (`norm_query`). -/
private theorem norm_paramsJ (hprog : jnorm programJ = programJ) : jnorm paramsJ = paramsJ := by
  rw [paramsJ]
  refine norm_mkObj_fix (asc_of (by kernel_rfl')) ?_ (paramsJ_tree _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _)
  intro e he
  simp only [List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false] at he
  rcases he with rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl|rfl
  · nfix                 -- anchors
  · nfix                 -- challenger
  · nfix                 -- delta
  · nfix                 -- form
  · exact norm_law       -- law
  · nfix                 -- leaf_layer
  · nfix                 -- log
  · exact norm_partition -- partition
  · nfix                 -- ports
  · exact hprog          -- program
  · nfix                 -- proofs
  · exact norm_query     -- query
  · nfix                 -- registered
  · nfix                 -- scheme
  · nfix                 -- window_root
  · nfix                 -- work_table

/-- `θJ`'s rebuild is itself, given `paramsJ`'s (`hp`): one component, whose `params` is `paramsJ` and whose `id` is a
string leaf. -/
private theorem norm_θJ (hp : jnorm paramsJ = paramsJ) : jnorm θJ = θJ := by
  rw [θJ]
  refine norm_mkObj_fix (asc_of (by kernel_rfl')) ?_ (θtop_tree _)
  intro e he
  simp only [List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false] at he
  rcases he with rfl
  refine norm_arr_fix ?_
  intro x hx
  simp only [List.toList_toArray, List.mem_cons, List.not_mem_nil, or_false] at hx
  rcases hx with rfl
  refine norm_mkObj_fix (asc_of (by kernel_rfl')) ?_ (θinner_tree _ _)
  intro e2 he2
  simp only [List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false] at he2
  rcases he2 with rfl | rfl
  · nfix          -- id
  · exact hp      -- params

/-- From the one structural fact `NormFacts` (the program descriptor's rebuild), both the composite's and the
component's rebuilds. -/
private theorem normFacts_derive (hN : NormFacts) : jnorm θJ = θJ ∧ jnorm paramsJ = paramsJ :=
  ⟨norm_θJ (norm_paramsJ hN), norm_paramsJ hN⟩

set_option maxHeartbeats 4000000 in
/-- The parameters `θ₀` decodes to, with each field `paramsOfJson` reads. -/
private theorem params_full :
    ∃ p : Spot.OneStageZk.Params,
    ParamsAt θ₀ p ∧ p.own.partition = partitionObj ∧ p.own.program = programFile ∧
    p.own.circuit = hexOf circuitSha ∧ p.own.pub = hexOf (Sha512.hash publicFile) ∧
    p.base.verifier.program = program ∧ p.base.verifier.query = query ∧ p.base.verifier.law = law ∧
    p.base.verifier.workTable = none ∧ p.base.challenger = some (key₀, label₀) := by
  -- The two instance facts are now proved (not hypotheses): the descriptor is plain (`ProgramProof.plain_programJ`) and
  -- `norm` leaves it fixed (`ProgramParse.norm_programJ`), so `ProgramPlain` and `NormFacts` are discharged here.
  have hpp : ProgramPlain := ProgramProof.plain_programJ
  have hN : NormFacts := ProgramParse.norm_programJ
  have hJ := JsonCanon.jsonFacts
  have hA := canon_θ₀ hJ hpp
  have hC := canon_params hJ hpp
  -- Annotate `hE`'s type so `fwd_walk`'s `ruleAt` sees a concrete `Eq` (its inferred type otherwise carries the `Except`
  -- error metavariable, which `ruleAt` only instantiates away since this change; the annotation makes the rule explicit).
  have hE : Spot.Program.parse (Firewall.printJson programJ).toUTF8 = Except.ok program := parse_program hJ hpp
  have hU := Flock.Hex.unhex_hexOf keyPub
  have hpf := programFile_eq hJ hpp
  -- `canon` of each plain literal directly (the decoders read the literal's fields, not `norm`'s).
  have hBp : Flock.canon paramsJ = .ok (Firewall.printJson paramsJ) :=
    hJ.canon_plain (plain_paramsJ hpp) printable_paramsJ
  have hDp : Flock.canon programJ = .ok (Firewall.printJson programJ) :=
    hJ.canon_plain (show Firewall.plain programJ = true from hpp) ProgramParse.printable_programJ
  obtain ⟨hNθ, hNp⟩ := normFacts_derive hN
  -- The decode: `θJ` is canon-tainted only through the `paramsJ` constant, which the walk hands whole to `canon`; so
  -- generalize `paramsJ` to a fresh value, detainting the literal, and give the walk `canon`'s equation over it.
  have hdec : (Core.canonicalJson θ₀ >>= Core.Composition.decodeParams) = .ok
      [({ id := VStar.Recursive.Sampled.PROTOCOL, params := Firewall.printJson paramsJ } : Core.Composition.Named)] := by
    rw [hA, hNθ]
    unfold θJ
    generalize hgpj : paramsJ = gpj
    have rC : Flock.canon gpj = .ok (Firewall.printJson gpj) := by rw [← hgpj]; exact hBp
    -- Rebuild the `Eq` head so the walk's `eq?` is not defeated by `mdata` left on the goal by `unfold`/`generalize`.
    refine Eq.trans ?g rfl
    fwd_walk [rC] block [Flock.canon] shallow []
  -- The parameters and every field `paramsOfJson` reads. `paramsJ` is canon-tainted only through `partitionObj`, whose
  -- one tainted leaf is the program digest `hexOf programSha`; expose it (unfold the partition builder) and generalize it
  -- to a fresh string, so the whole literal the walk descends into mentions no `canon`.
  have hpar : ∃ P : Spot.OneStageZk.Params,
      (Core.canonicalJson (Firewall.printJson paramsJ).toUTF8 >>= VStar.Recursive.Sampled.paramsOfJson) =
        .ok (.inl P) ∧
      P.own.partition = partitionObj ∧ P.own.program = programFile ∧ P.own.circuit = hexOf circuitSha ∧
      P.own.pub = hexOf (Sha512.hash publicFile) ∧ P.base.verifier.program = program ∧
      P.base.verifier.query = query ∧ P.base.verifier.law = law ∧ P.base.verifier.workTable = none ∧
      P.base.challenger = some (key₀, label₀) := by
    rw [hC, hNp]
    -- The two spliced digests pass the form check; proved off the hash so the walk rewrites by them.
    have hdc : Spot.OneStageZk.digest? (hexOf circuitSha) = true := by
      apply digest?_hexOf; unfold circuitSha; exact Hex.size_hash circuitFile
    have hdp : Spot.OneStageZk.digest? (hexOf (Sha512.hash publicFile)) = true := by
      apply digest?_hexOf; exact Hex.size_hash publicFile
    -- The partition's structural check reads `keys? query` (inside `Spot.Partition.build`, to pick the default query only
    -- for an empty one); `query` is a non-empty literal, so its keys are a fixed set and the check keeps `query`. Proved off
    -- the sort (the walk's condition is otherwise stuck on `keys? query`'s `qsort`, buried in the partition object where the
    -- taint does not reach), so the walk rewrites by it.
    have hq : Spot.keys? query = some #["name", "params", "version"] := by
      unfold query; rw [keys?_mkObj, sortStrings_eq_sortLt]; kernel_rfl'
    unfold paramsJ partitionObj Spot.Partition.build
    -- Detaint every canon-/hash-derived leaf so the field lookups reduce: the program digest in the partition, and the
    -- circuit and public digests in `proofs`. The digest facts are restated over the fresh names.
    generalize hgh : hexOf programSha = gh
    generalize hgc : hexOf circuitSha = gc
    generalize hgp : hexOf (Sha512.hash publicFile) = gp
    generalize hgk : hexOf keyPub = gk
    generalize hgw : hexOf (ByteArray.mk (Array.replicate 64 0)) = gw
    rw [hgc] at hdc
    rw [hgp] at hdp
    rw [hgk] at hU
    -- The genesis/window-root and program digests each face a `length == 128 && isHex` form check (`digest?`); their
    -- bytes are 64, so the check passes off the size fact (`digest?_hexOf`), which the walk reads from the context.
    have hgw_dg : (gw.length == 128 && Spot.isHex gw) = true := by
      rw [← hgw]
      have hsz : (ByteArray.mk (Array.replicate 64 0)).size = 64 := by decide
      exact digest?_hexOf hsz
    have hgh_dg : (gh.length == 128 && Spot.isHex gh) = true := by
      rw [← hgh]
      have hsz : programSha.size = 64 := by unfold programSha; exact Hex.size_hash _
      exact digest?_hexOf hsz
    -- The challenger key's public part decodes to `keyPub`: `hexOf?` of its hex (`gk = hexOf keyPub`) is `some keyPub`
    -- off the hex round trip, so the walk takes the `some pub` branch without evaluating the blocked `unhex`.
    have hgk_pub : Spot.Challenger.hexOf? (.str gk) = some keyPub := by
      show (Flock.unhex gk).toOption = some keyPub
      rw [← hgk, Flock.Hex.unhex_hexOf]; rfl
    refine ⟨?P, ?walk, ?f1, ?f2, ?f3, ?f4, ?f5, ?f6, ?f7, ?f8, ?f9⟩
    -- `Spot.keys?` and `Spot.sortStrings` are blocked so the walk never hands the parser's key checks (`exactly`, whose
    -- body reads them) to the kernel's `whnf`, which cannot reduce `sortStrings`'s well-founded `Array.qsort`; the rules
    -- below (`keys?_mkObj`, `keys?_obj`, `sortStrings_of_sorted`, `sortStrings_eq_sortLt`) rewrite each occurrence instead.
    case walk => fwd_walk [hDp, hE, hU, hdc, hdp, hq, hgk_pub, hgw_dg, hgh_dg, ratio_half, split_half, poolValues_none,
      keys?_mkObj,       keys?_obj, sortStrings_of_sorted, sortStrings_eq_sortLt, PortsFacts.ports_entries, PortsFacts.rest_eq,
      unwindowed_some, Array.mapM_empty]
      block [Flock.canon, Spot.Program.parse, Flock.unhex, Spot.keys?, Spot.sortStrings] shallow []
    case f1 => rw [hq]; kernel_rfl'   -- own.partition = partitionObj (resolve the RHS's `keys? query` keys check by `hq`)
    case f2 => rw [hpf]               -- own.program = programFile
    case f3 => kernel_rfl'            -- own.circuit
    case f4 => kernel_rfl'            -- own.pub
    case f5 => rfl                    -- verifier.program = program
    case f6 => kernel_rfl'            -- verifier.query = query
    case f7 => kernel_rfl'            -- verifier.law = law
    case f8 => rfl                    -- verifier.workTable = none
    case f9 => kernel_rfl'            -- challenger = some (key₀, label₀)
  obtain ⟨P, hwalk, hpart, hprog, hc, hpub, hvp, hvq, hvl, hvt, hch⟩ := hpar
  exact ⟨P, ⟨[{ id := VStar.Recursive.Sampled.PROTOCOL, params := Firewall.printJson paramsJ }], hdec,
      { id := VStar.Recursive.Sampled.PROTOCOL, params := Firewall.printJson paramsJ }, by rfl, hwalk⟩,
      hpart, hprog, hc, hpub, hvp, hvq, hvl, hvt, hch⟩

theorem params_spec : ∃ p, ParamsAt θ₀ p ∧
    p.own.partition = partitionObj ∧ p.own.program = programFile ∧ p.base.verifier.program = program ∧
    p.base.verifier.query = query ∧ p.base.verifier.law = law ∧ p.base.verifier.workTable = none ∧
    p.base.challenger = some (key₀, label₀) := by
  obtain ⟨p, hpat, hpart, hprog, _, _, hvp, hvq, hvl, hvt, hch⟩ := params_full
  exact ⟨p, hpat, hpart, hprog, hvp, hvq, hvl, hvt, hch⟩

theorem digests :
    programDigestOf θ₀ = some (Verity.sha512 circuitFile.data.toList) ∧
      publicDigestOf θ₀ = some (Verity.sha512 publicFile.data.toList) := by
  obtain ⟨p, hpat, _, _, hc, hpub, _⟩ := params_full
  refine ⟨?_, ?_⟩
  · rw [programDigestOf, paramsOf_eq hpat, Option.bind_some, hc]
    simp only [digestOf, Flock.Hex.unhex_hexOf, circuitSha]
    rw [← sha512_data circuitFile]
  · rw [publicDigestOf, paramsOf_eq hpat, Option.bind_some, hpub]
    simp only [digestOf, Flock.Hex.unhex_hexOf]
    rw [← sha512_data publicFile]

end Verity.Sampled.Instance.HonestSession.Params
