module

public import Proofs.Flock.SessionLoop.Concrete.Public
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
import Flock.Norm
import Flock.Public
import Flock.Statement
import Flock.Comp
import Flock.Bytes
import Flock.FV3
import Flock.Hm96
import Flock.Hash
import Flock.RowLeaf
import Flock.Circuit
import Flock.Tags
import Flock.Zk
import Flock.CoinTree
import Flock.Merkle
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

/-! # The concrete statement's files fit their length words

`Raw.Fits` asks every length the statement's encoding writes to fit 8 bytes. The circuit file's (`circuit_size`) is
bounded line by line: every line of the unit's and the two fixed nets' sections is under 2^20 bytes, the nets' at the
open guarantee's `LinesFit`, and there are 503,051 lines. The public file's (`public_size`) is the header's text, whose
length is the closed header's plus the digest's 128 hex characters (`hdr_len`), and the row after it.
-/

@[expose] public section
open Flock Flock.Firewall RecDeploy.VStarNets FlockVBridge.Names
open Properties.Flock.Nets (LinesFit)
open Lean (Json)

set_option maxHeartbeats 0 -- the `decide +kernel` steps below evaluate the unit's and the header's lines
set_option maxRecDepth 200000

namespace FlockDeploy.Concrete
open FlockSoundness.Discharge.Recursion.FirewallJson

theorem hash_size (m : ByteArray) : (Sha512.hash m).size = 64 := by
  unfold Sha512.hash
  exact Array.size_ofFn

theorem toUTF8_size (s : String) : s.toUTF8.size = s.utf8ByteSize := String.size_toByteArray

theorem utf8_ofList_le : ∀ l : List Char, (String.ofList l).utf8ByteSize ≤ 4 * l.length
  | [] => by simp
  | c :: l => by
    rw [String.ofList_cons, String.utf8ByteSize_append, String.utf8ByteSize_singleton, List.length_cons]
    have := Char.utf8Size_le_four c
    have := utf8_ofList_le l
    omega

theorem joinNl_size : ∀ (L : List String), (∀ x ∈ L, x.utf8ByteSize < 2 ^ 20) →
    (joinNl L).utf8ByteSize ≤ L.length * (2 ^ 20 + 1)
  | [], _ => by simp [joinNl]
  | x :: L, h => by
    have h1 := h x List.mem_cons_self
    have ih := joinNl_size L (fun y hy => h y (List.mem_cons_of_mem _ hy))
    simp only [joinNl, List.map_cons, String.join_cons, String.utf8ByteSize_append, List.length_cons] at ih ⊢
    have hn : "\n".utf8ByteSize = 1 := rfl
    rw [hn, Nat.succ_mul]
    omega

theorem rowsLs_fit {name : String} {l : HmNets.Laid} (h : LinesFit name l) :
    ∀ x ∈ rowsLs l, x.utf8ByteSize < 2 ^ 20 := by
  intro x hx
  simp only [rowsLs, List.mem_map, List.mem_range'] at hx
  obtain ⟨i, ⟨j, hj, rfl⟩, rfl⟩ := hx
  exact h.2 _ (by omega)

theorem secLs_fit {nm name : String} {l : HmNets.Laid} (h : LinesFit name l)
    (hc : ("CIRCUIT " ++ nm ++ " " ++ toString (VStar.netLines l)).utf8ByteSize < 2 ^ 20) :
    ∀ x ∈ secLs nm name l, x.utf8ByteSize < 2 ^ 20 := by
  intro x hx
  simp only [secLs, netLs, List.mem_cons] at hx
  rcases hx with rfl | rfl | hx
  · exact hc
  · exact h.1
  · exact rowsLs_fit h x hx

theorem unit_fit : ∀ x ∈ secLs "unit" "copy16" unitLaid, x.utf8ByteSize < 2 ^ 20 := by decide +kernel

theorem circuit_size (ls lh : HmNets.Laid) (hs : LinesFit VStar.SHA_NET ls) (hh : LinesFit VStar.HM_NET lh)
    (hls : VStar.netLines ls = 256002) (hlh : VStar.netLines lh = 246786) : (bytesOf ls lh).size < 2 ^ 64 := by
  have hfit : ∀ x ∈ lines ls lh, x.utf8ByteSize < 2 ^ 20 := by
    intro x hx
    simp only [lines, fileLs, List.mem_append, List.cons_append, List.nil_append, List.mem_cons, or_assoc] at hx
    rcases hx with rfl | rfl | hx | hx | hx
    · decide +kernel
    · rw [String.utf8ByteSize_append, metaText]
      have := utf8_ofList_le metaChars
      have hm : metaChars.length < 2 ^ 17 := by decide +kernel
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

theorem hexOf_len (b : ByteArray) : (hexOf b).length = 2 * b.size := by
  have h2 : ∀ x : UInt8, (hexOf ⟨#[x]⟩).toList.length = 2 := fun x => by
    unfold hexOf
    rw [foldl_eq, foldl_push]
    simp
  rw [← String.length_toList, hexOf_toList, List.length_flatMap]
  simp only [h2, List.map_const', List.sum_replicate_nat, Array.length_toList, ByteArray.size]
  omega

def RM : List (List Char) := [⟨"format", .str "flock-circuit-inputs"⟩, ⟨"frame_v3", fv3J⟩, ⟨"instances", jn 1⟩,
  ⟨"rows", .bool false⟩, ⟨"statement", .str "verity/flock-circuit"⟩, ⟨"units", unitsJ⟩].map memberChars

theorem hdr_chars (s : String) :
    jsonChars (hdrJ s) = '{' :: [','].intercalate (memberChars ⟨"circuit_sha512", .str s⟩ :: RM) ++ ['}'] := by
  rw [hdr_obj]
  show '{' :: [','].intercalate (jsonCharsObj (hdrT s)) ++ ['}'] = _
  rw [jsonCharsObj_eq, hdr_model]
  rfl

theorem icl_len (x y : List Char) (R : List (List Char)) :
    ([','].intercalate (x :: R)).length + y.length = ([','].intercalate (y :: R)).length + x.length := by
  cases R with
  | nil => simp only [List.intercalate_singleton]; omega
  | cons r R => simp only [List.intercalate_cons_cons, List.length_append]; omega

theorem len_aux {a b x y : List Char} {R : List (List Char)} {n : Nat}
    (ha : a = '{' :: [','].intercalate (x :: R) ++ ['}']) (hb : b = '{' :: [','].intercalate (y :: R) ++ ['}'])
    (hxy : x.length = y.length + n) : a.length = b.length + n := by
  subst ha hb
  have := icl_len x y R
  simp only [List.length_cons, List.length_append]
  omega

theorem member_len (s : String) : (memberChars ⟨"circuit_sha512", .str s⟩).length =
    (memberChars ⟨"circuit_sha512", .str ""⟩).length + s.length := by
  simp only [memberChars, jsonChars, List.length_append, List.length_cons, String.length_toList,
    String.toList_empty, List.length_nil]
  omega

theorem hdr_len (s : String) : (jsonChars (hdrJ s)).length = (jsonChars (hdrJ "")).length + s.length :=
  len_aux (hdr_chars s) (hdr_chars "") (member_len s)

theorem hdr0_len : (jsonChars (hdrJ "")).length < 2 ^ 20 := by
  unfold hdrJ restL
  rw [fv3_eq]
  decide +kernel

theorem tail_size : tailB.size < 2 ^ 20 := by
  unfold tailB bodyB
  rw [bc_eq]
  decide +kernel

theorem public_size (H : ByteArray) (hH : H.size = 64) : (pubOf (hexOf H)).size < 2 ^ 64 := by
  unfold pubOf printJson
  rw [ByteArray.size_append, toUTF8_size]
  have h1 := utf8_ofList_le (jsonChars (hdrJ (hexOf H)))
  rw [hdr_len, hexOf_len, hH] at h1
  have h2 := hdr0_len
  have h3 := tail_size
  omega

end FlockDeploy.Concrete
