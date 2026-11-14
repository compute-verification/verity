module

public import Proofs.Flock.SpotFlock.InstanceStmt.Meta
public import Proofs.Flock.SessionLoop.Concrete
public import Proofs.Flock.SessionLoop.Concrete.Stmt
import all Init.Data.String.TakeDrop
import all Init.Data.String.Slice
import all Init.Data.String.Pattern.Basic
import all Init.Data.String.Pattern.String
import all Init.Data.String.Pattern.Char
import all Init.Data.String.Search
import all Init.Data.String.Iterate
import all Init.Data.String.Decode
import all Init.Data.String.Termination
import all Init.Data.Array.Attach
import all Init.WF
import Flock.HmRow
import Flock.HmNets
import Flock.Canon
import Flock.JsonParse
import Flock.Net
import Flock.Firewall
import Flock.Bytes
import Flock.FV3
import Flock.Hm96
import Flock.Hash
import Flock.Tags
import Flock.Zk
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
import all Init.Data.String.Extra
import all Init.Data.Array.QSort.Basic
import all Init.Data.Array.Basic
import all Init.Data.Array.DecidableEq
import all Init.Data.List.Control
import all Init.Data.Range.Basic
import all Init.Data.Int.Repr
import all Init.Data.Repr
import all Init.Data.Nat.Fold

/-! # The two-instance statement fits its length words, and decodes

At the open guarantee that the two fixed nets build (`FixedNetsBuild`) and a plain run (`Run.Plain`), every length
`x r` writes fits its eight bytes (`raw_fits`), so `decode` reads back `raw r` (`decode_x`), and `x r` is under 2^64
bytes (`x_small`). The circuit file is bounded line by line, as the concrete statement's (`circuit_le`); the public file
by its header's text, at most the closed parts' length plus each string the run and the two digests put in it
(`hdr_len`), and the rows after it.
-/

@[expose] public section
open Flock Flock.Firewall RecDeploy.VStarNets FlockVBridge.Names
open Properties.Flock.Nets (LinesFit FixedNetsBuild)
open FlockDeploy.Concrete (hash_size toUTF8_size utf8_ofList_le joinNl_size secLs_fit unit_fit hexOf_len unitLaid)
open FlockSoundness.Discharge.Recursion.FirewallJson (jsonCharsObj_eq jsonCharsList_eq memberChars)
open Std.DTreeMap.Internal (Impl)
open Lean (Json)

set_option maxHeartbeats 500000000 -- the `decide +kernel` steps below evaluate META's and the header's closed parts
set_option maxRecDepth 200000

namespace SpotFlock.InstanceStmt

/-! ## The circuit file -/

theorem circuit_le (ls lh : HmNets.Laid) (hs : LinesFit VStar.SHA_NET ls) (hh : LinesFit VStar.HM_NET lh)
    (hls : VStar.netLines ls = 256002) (hlh : VStar.netLines lh = 246786) :
    (bytesOf ls lh).size ≤ 503051 * (2 ^ 20 + 1) := by
  have hfit : ∀ x ∈ lines ls lh, x.utf8ByteSize < 2 ^ 20 := by
    intro x hx
    simp only [lines, fileLs, List.mem_append, List.cons_append, List.nil_append, List.mem_cons, or_assoc] at hx
    rcases hx with rfl | rfl | hx | hx | hx
    · decide +kernel
    · rw [String.utf8ByteSize_append, metaText, printJson]
      have := utf8_ofList_le (jsonChars metaJ)
      have hm : (jsonChars metaJ).length < 2 ^ 17 := by decide +kernel
      have h5 : "META ".utf8ByteSize = 5 := by decide +kernel
      omega
    · exact unit_fit x hx
    · exact secLs_fit hs (by rw [hls]; decide +kernel) x hx
    · exact secLs_fit hh (by rw [hlh]; decide +kernel) x hx
  have hlen : (lines ls lh).length = 2 + (1 + VStar.netLines unitLaid) + (1 + 256002) + (1 + 246786) := by
    simp only [lines, fileLs, List.length_append, secLs_length, hls, hlh]
    rfl
  have hu : VStar.netLines unitLaid = 258 := by decide +kernel
  have := joinNl_size _ hfit
  unfold bytesOf
  rw [toUTF8_size]
  rw [hlen, hu] at this
  omega

/-! ## The header's length -/

theorem icl_le : ∀ L : List (List Char), ([','].intercalate L).length ≤ (L.map (·.length + 1)).sum
  | [] => by simp
  | [x] => by simp
  | x :: y :: L => by
    have ih := icl_le (y :: L)
    rw [List.intercalate_cons_cons]
    simp only [List.length_append, List.length_singleton, List.map_cons, List.sum_cons] at ih ⊢
    omega

theorem mem_len (e : (_ : String) × Json) :
    (memberChars e).length = e.1.toList.length + 3 + (jsonChars e.2).length := by
  simp only [memberChars, List.length_append, List.length_cons, List.length_nil]

theorem str_len (s : String) : (jsonChars (.str s)).length = s.toList.length + 2 := by
  show ('"' :: s.toList ++ ['"']).length = _
  simp only [List.length_cons, List.length_append, List.length_nil]

theorem obj_le {t : Impl String (fun _ => Json)} {L : List ((_ : String) × Json)} (hm : t.toListModel = L) :
    (jsonChars (.obj ⟨⟨t⟩⟩)).length ≤ (L.map fun e => (memberChars e).length + 1).sum + 2 := by
  have h := icl_le (L.map memberChars)
  simp only [List.map_map, Function.comp_def] at h
  show ('{' :: [','].intercalate (jsonCharsObj t) ++ ['}']).length ≤ _
  rw [jsonCharsObj_eq, hm]
  simp only [List.length_cons, List.length_append, List.length_nil]
  omega

theorem arr_le (l : List Json) :
    (jsonChars (.arr l.toArray)).length ≤ (l.map fun j => (jsonChars j).length + 1).sum + 2 := by
  have h := icl_le (l.map jsonChars)
  simp only [List.map_map, Function.comp_def] at h
  show ('[' :: [','].intercalate (jsonCharsList l) ++ [']']).length ≤ _
  rw [jsonCharsList_eq]
  simp only [List.length_cons, List.length_append, List.length_nil]
  omega

theorem io_len (a b : Json) :
    (jsonChars (Json.mkObj [("i0", a), ("o0", b)])).length ≤ (jsonChars a).length + (jsonChars b).length + 14 := by
  rw [io_obj]
  refine Nat.le_trans (obj_le (io_model a b)) ?_
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, mem_len]
  have hk : "i0".toList.length + "o0".toList.length = 4 := by decide +kernel
  omega

theorem fv3_len (r : Run) : (jsonChars (fv3J r)).length ≤ r.bindI.toList.length + r.bindO.toList.length + 2 ^ 11 := by
  rw [fv3_obj]
  refine Nat.le_trans (obj_le (fv3_model r)) ?_
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, mem_len]
  have hk : "bindings".toList.length + "domain_ids".toList.length + "frame".toList.length + "roots".toList.length +
      "row_key_sha512".toList.length + "schemas".toList.length < 2 ^ 6 := by decide +kernel
  have h1 := io_len (.str r.bindI) (.str r.bindO)
  rw [str_len, str_len] at h1
  have h2 : (jsonChars (Json.mkObj [("i0", .str (hexOf fdomI)), ("o0", .str (hexOf fdomO))])).length < 2 ^ 6 := by
    decide +kernel
  have h3 : (jsonChars (Json.mkObj [("i0", .str (hexOf rootI)), ("o0", .str (hexOf rootO))])).length < 2 ^ 9 := by
    decide +kernel
  have h4 : (jsonChars (.str keyHex)).length < 2 ^ 8 := by decide +kernel
  have h5 : (jsonChars (Json.mkObj [("i0", .str v2), ("o0", .str v2)])).length < 2 ^ 7 := by decide +kernel
  have h6 : (jsonChars (.str "frame-v3-sha512")).length < 2 ^ 5 := by decide +kernel
  omega

theorem units_len (cls : String) : (jsonChars (unitsJ cls)).length ≤ 2 * cls.toList.length + 2 ^ 8 := by
  rw [units_obj]
  refine Nat.le_trans (obj_le (units_model cls)) ?_
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, mem_len]
  have hk : "blocks".toList.length + "classes".toList.length + "indices".toList.length + "units_per_vu".toList.length +
      "vus_per_block".toList.length < 2 ^ 6 := by decide +kernel
  have h1 := arr_le [.str cls, .str cls]
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, str_len] at h1
  have h2 : (jsonChars (.arr #[.arr #[jn 0], .arr #[jn 1]])).length < 2 ^ 5 := by decide +kernel
  have h3 : (jsonChars (.arr #[jn 0, jn 1])).length < 2 ^ 4 := by decide +kernel
  have h4 : (jsonChars (jn 1)).length < 2 := by decide +kernel
  omega

/-- **The header's text** is at most its closed parts' length plus each string the run and the digests put in it. -/
theorem hdr_len (r : Run) (circ cls : String) : (jsonChars (hdrJ r circ cls)).length ≤
    circ.toList.length + r.set.toList.length + r.bindI.toList.length + r.bindO.toList.length + 2 * cls.toList.length +
      2 ^ 13 := by
  rw [hdr_obj]
  refine Nat.le_trans (obj_le (hdr_model r circ cls)) ?_
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, mem_len, str_len]
  have hk : "circuit_sha512".toList.length + "format".toList.length + "flock-circuit-inputs".toList.length +
      "frame_v3".toList.length + "instances".toList.length + "registered".toList.length + "rows".toList.length +
      "scheme".toList.length + "set".toList.length + "shared_rows".toList.length + "statement".toList.length +
      "units".toList.length < 2 ^ 7 := by decide +kernel
  have h1 := fv3_len r
  have h2 := units_len cls
  have h3 : (jsonChars regHdrJ).length < 2 ^ 9 := by decide +kernel
  have h4 : (jsonChars (Json.mkObj [("i0", jn 1), ("o0", jn 2)])).length < 2 ^ 6 := by decide +kernel
  have h5 : (jsonChars (jn 2)).length < 2 := by decide +kernel
  have h6 : (jsonChars (.bool false)).length = 5 := by decide +kernel
  have h7 : HmRow.SCHEME.toList.length < 2 ^ 6 := by decide +kernel
  have h8 : NAME.toList.length < 2 ^ 6 := by decide +kernel
  omega

theorem tail_size : tailB.size < 2 ^ 12 := by decide +kernel

theorem public_le (r : Run) (circ cls : String) : (pubOf r circ cls).size ≤
    4 * (circ.length + r.set.length + r.bindI.length + r.bindO.length + 2 * cls.length + 2 ^ 13) + 2 ^ 12 := by
  unfold pubOf printJson
  rw [ByteArray.size_append, toUTF8_size]
  have h1 := utf8_ofList_le (jsonChars (hdrJ r circ cls))
  have h2 := hdr_len r circ cls
  have h3 := tail_size
  simp only [String.length_toList] at h2
  omega

/-! ## The statement -/

theorem classHash_size : classHash.size = 64 := by
  obtain ⟨m, hm⟩ := classOfB_hash (HmNets.shaLaid ()) (HmNets.hmLaid ())
  rw [classHash, hm, hash_size]

theorem public_lt {r : Run} (hr : r.Plain) : (pubOf r (hexOf circuitHash) (hexOf classHash)).size < 2 ^ 24 := by
  have h := public_le r (hexOf circuitHash) (hexOf classHash)
  have hC : circuitHash.size = 64 := hash_size _
  rw [hexOf_len, hexOf_len, hC, classHash_size] at h
  have := hr.size
  omega

theorem sample_size : (printJson sampleJ).toUTF8.size < 2 ^ 12 := by decide +kernel
theorem partition_size : (printJson partObj).toUTF8.size < 2 ^ 12 := by decide +kernel
theorem program_size : progBytes.size < 2 ^ 12 := by decide +kernel
theorem registered_size : (printJson regJ).toUTF8.size < 2 ^ 12 := by decide +kernel
theorem name_size : NAME.toUTF8.size < 2 ^ 12 := by decide +kernel

theorem raw_fits (hn : FixedNetsBuild) {r : Run} (hr : r.Plain) : (raw r).Fits where
  statement := Nat.lt_trans name_size (by decide)
  circuit := Nat.lt_of_le_of_lt
    (circuit_le _ _ hn.1.2.2.2 hn.2.2.2.2 (FlockDeploy.sha_lines hn) (FlockDeploy.hm_lines hn)) (by decide)
  publicFile := Nat.lt_trans (public_lt hr) (by decide)
  tables := by simp [raw, rawOf]
  tableItems := by simp [raw, rawOf]
  count := by simp [raw, rawOf]
  draw := by
    intro b h
    simp only [raw, rawOf, Option.some.injEq] at h
    subst h
    exact Nat.lt_trans sample_size (by decide)
  partition := by
    intro b h
    simp only [raw, rawOf, Option.some.injEq] at h
    subst h
    exact Nat.lt_trans partition_size (by decide)
  program := by
    intro b h
    simp only [raw, rawOf, Option.some.injEq] at h
    subst h
    exact Nat.lt_trans program_size (by decide)
  work := by intro _ h; cases h
  workTable := by intro _ h; cases h
  stratified := by intro _ h; cases h
  closure := by intro _ h; cases h
  registered := by
    intro b h
    simp only [raw, rawOf, Option.some.injEq] at h
    subst h
    exact Nat.lt_trans registered_size (by decide)
  publicInputs := by intro _ h; cases h

/-- **The statement decodes** to the `Raw` it encodes. -/
theorem decode_x (hn : FixedNetsBuild) {r : Run} (hr : r.Plain) : FlockDeploy.decode (x r) = some (raw r) :=
  FlockDeploy.decode_encode_of_fits (raw_fits hn hr)

theorem encNat_length (n : Nat) : (FlockDeploy.encNat n).length = 8 := by simp [FlockDeploy.encNat]

theorem encBytes_length (b : ByteArray) : (FlockDeploy.encBytes b).length = 8 + b.size := by
  show (FlockDeploy.encNat b.size ++ b.data.toList).length = 8 + b.data.size
  rw [List.length_append, encNat_length, Array.length_toList]

/-- **The statement is under 2^64 bytes.** -/
theorem x_small (hn : FixedNetsBuild) {r : Run} (hr : r.Plain) : (x r).length < 2 ^ 64 := by
  have hc := circuit_le _ _ hn.1.2.2.2 hn.2.2.2.2 (FlockDeploy.sha_lines hn) (FlockDeploy.hm_lines hn)
  have hp := public_lt hr
  have hd := sample_size
  have hq := partition_size
  have hg := program_size
  have hg' := registered_size
  have hs := name_size
  have hf : FlockDeploy.FORMAT.toUTF8.size < 2 ^ 12 := by decide +kernel
  simp only [x, raw, rawOf, FlockDeploy.Raw.encode, FlockDeploy.encStr, FlockDeploy.encList, FlockDeploy.encBool,
    FlockDeploy.encOpt, List.length_append, List.length_cons, List.length_nil, List.flatMap_nil, encBytes_length,
    encNat_length]
  omega

end SpotFlock.InstanceStmt
