module

public import Proofs.Flock.SpotFlock.InstanceStmt.Meta
public import Proofs.Flock.SessionLoop.Concrete.File
public import Proofs.Flock.SessionLoop.Concrete.Inputs
import Flock.HmRow
import Flock.Net
import Flock.HmNets
import Flock.Canon
import Flock.JsonParse
import Flock.RecOpen
import Flock.RecResiduals
import Flock.RecSparse
import Flock.RecOpening
import all Init.Data.Array.QSort.Basic
import all Init.Data.String.Defs
import all Init.Data.String.Basic
import all Lean.Data.Json.Basic
import all Init.Data.Ord.String
import all Init.Data.Array.Basic
import all Init.Data.Array.DecidableEq
import all Init.Data.List.Control
import all Init.Data.Range.Basic
import all Init.Data.Int.Repr
import all Init.Data.Repr
import all Lean.Data.Json.FromToJson.Basic
import all Std.Data.TreeMap.Raw.Basic
import all Std.Data.DTreeMap.Raw.Basic
import all Std.Data.DTreeMap.Internal.Operations
import all Std.Data.DTreeMap.Internal.Queries
import all Std.Data.DTreeMap.Internal.Balancing
import all Std.Data.DTreeMap.Internal.Def
import all Init.Data.Ord.Basic
import all Init.Data.String.TakeDrop
import all Init.Data.String.Slice
import all Init.Data.String.Pattern.Basic
import all Init.Data.String.Pattern.String
import all Init.Data.String.Pattern.Char
import all Init.Data.String.Pattern.Pred
import all Init.Data.String.Subslice
import all Init.Data.String.Search
import all Init.Data.String.Iterate
import all Init.Data.String.Decode
import all Init.Data.String.PosRaw
import all Init.Data.String.FindPos
import all Init.Data.String.Termination
import all Init.Data.String.ToSlice
import all Init.Data.String.Bootstrap
import all Init.Data.String.Iter
import all Init.Data.String.Iterator
import all Init.Data.String.Substring
import all Init.Data.String.Modify
import all Init.Data.String.Extra
import all Init.Data.String.Legacy

/-! # The two-instance statement's circuit file, and what `HmOut.parse` makes of it

`parse_cX` walks `HmRow.parse` on the circuit file's bytes (`bytesOf ls lh`) to the circuit `cX`, as
`FlockDeploy.Concrete.parse_cW` walks the concrete statement's: at the same hypotheses on the two fixed nets, and at the
circuit's class (`hcl`), which `classOf_eq` shows is `classOfB ls lh`. `parseOut_cX` adds `HmOut.checkRows`: the
statement's circuit as `HmOut.parse` reads one with hidden outputs.

The file's lines are `FlockDeploy.Concrete`'s at this META (`metaText`, `lines_eq`): the head, META, then the unit's,
`sha512x3`'s and `hm96`'s sections, at the same line numbers.
-/

@[expose] public section
open Flock Flock.Firewall RecDeploy.VStarNets
open FlockVBridge.Names (NoNl)
open FlockDeploy.Concrete (unitLaid startsWith_append drop_append trim_of getD_toArray getElem!_toArray forIn_yield
  forIn_done getD3 bne_self toNat?_toString open_none res_none sparse_none opening_none unit_eq lines_if noNl_copy16
  unit_lines netLs_length getD_mid extract_mid circuit_unit circuit_sha circuit_hm parse_unit tokens3 shaB hmB cU cS cH)
open FlockSoundness.Refine
open Lean (Json)

set_option maxHeartbeats 0 -- the walks below step through `HmRow.parse` and leave closed checks to the kernel
set_option maxRecDepth 200000

namespace SpotFlock.InstanceStmt

/-! ## The file's lines -/

theorem lines1 (ls lh : HmNets.Laid) : (lines ls lh).getD 1 "" = "META " ++ metaText := rfl

section
variable (ls lh : HmNets.Laid) (hls : VStar.netLines ls = 256002) (hlh : VStar.netLines lh = 246786)
include hls hlh

theorem lines_eq : lines ls lh =
    ["flock-circuit copy16", "META " ++ metaText] ++
      (("CIRCUIT unit " ++ toString 258) :: netLs "copy16" unitLaid) ++
      (("CIRCUIT sha512x3 " ++ toString 256002) :: netLs "sha512x3" ls) ++
      (("CIRCUIT hm96 " ++ toString 246786) :: netLs "hm96" lh) := by
  simp only [lines, fileLs, secLs, VStar.SHA_NET, VStar.HM_NET, unit_lines, hls, hlh]
  simp

theorem size_lines : (lines ls lh).toArray.size = 503051 := by
  rw [lines_eq ls lh hls hlh]
  simp only [List.size_toArray, List.length_append, List.length_cons, netLs_length, unit_lines, hls, hlh]
  rfl

end

abbrev PL : List String := ["flock-circuit copy16", "META " ++ metaText]

section
variable (ls lh : HmNets.Laid) (hls : VStar.netLines ls = 256002) (hlh : VStar.netLines lh = 246786)
include hls hlh

theorem line_unit (i : Nat) (hi : i = 2) : (lines ls lh).getD i "" = "CIRCUIT" ++ " " ++ "unit" ++ " " ++ toString 258 := by
  have e : lines ls lh = PL ++ (cU :: netLs "copy16" unitLaid) ++ ((cS :: netLs "sha512x3" ls) ++ (cH :: netLs "hm96" lh)) := by
    rw [lines_eq ls lh hls hlh]; simp
  rw [e]; exact (getD_mid _ _ _ _ _ i (by simp [hi])).trans circuit_unit

theorem line_sha (i : Nat) (hi : i = 261) :
    (lines ls lh).getD i "" = "CIRCUIT" ++ " " ++ "sha512x3" ++ " " ++ toString 256002 := by
  have e : lines ls lh = (PL ++ (cU :: netLs "copy16" unitLaid)) ++ (cS :: netLs "sha512x3" ls) ++ (cH :: netLs "hm96" lh) := by
    rw [lines_eq ls lh hls hlh]
  rw [e]; exact (getD_mid _ _ _ _ _ i (by simp [hi, netLs_length, unit_lines])).trans circuit_sha

theorem line_hm (i : Nat) (hi : i = 256264) :
    (lines ls lh).getD i "" = "CIRCUIT" ++ " " ++ "hm96" ++ " " ++ toString 246786 := by
  have e : lines ls lh = (PL ++ (cU :: netLs "copy16" unitLaid) ++ (cS :: netLs "sha512x3" ls)) ++ (cH :: netLs "hm96" lh) ++ [] := by
    rw [lines_eq ls lh hls hlh, List.append_nil]
  rw [e]; exact (getD_mid _ _ _ _ _ i (by simp [hi, netLs_length, unit_lines, hls])).trans circuit_hm

theorem ext_unit (a b : Nat) (ha : a = 3) (hb : b = 261) :
    (lines ls lh).toArray.extract a b = (netLs "copy16" unitLaid).toArray := by
  have e : lines ls lh = (PL ++ [cU]) ++ netLs "copy16" unitLaid ++ ((cS :: netLs "sha512x3" ls) ++ (cH :: netLs "hm96" lh)) := by
    rw [lines_eq ls lh hls hlh]; simp
  rw [e]; exact extract_mid _ _ _ a b (by simp [ha]) (by simp [hb, netLs_length, unit_lines])

theorem ext_sha (a b : Nat) (ha : a = 262) (hb : b = 256264) :
    (lines ls lh).toArray.extract a b = (netLs "sha512x3" ls).toArray := by
  have e : lines ls lh = (PL ++ (cU :: netLs "copy16" unitLaid) ++ [cS]) ++ netLs "sha512x3" ls ++ (cH :: netLs "hm96" lh) := by
    rw [lines_eq ls lh hls hlh]; simp
  rw [e]; exact extract_mid _ _ _ a b (by simp [ha, netLs_length, unit_lines]) (by simp [hb, netLs_length, unit_lines, hls])

theorem ext_hm (a b : Nat) (ha : a = 256265) (hb : b = 503051) :
    (lines ls lh).toArray.extract a b = (netLs "hm96" lh).toArray := by
  have e : lines ls lh = (PL ++ (cU :: netLs "copy16" unitLaid) ++ (cS :: netLs "sha512x3" ls) ++ [cH]) ++ netLs "hm96" lh ++ [] := by
    rw [lines_eq ls lh hls hlh]; simp
  rw [e]; exact extract_mid _ _ _ a b (by simp [ha, netLs_length, unit_lines, hls]) (by simp [hb, netLs_length, unit_lines, hls, hlh])

end

theorem hutf (ls lh : HmNets.Laid) : String.fromUTF8? (bytesOf ls lh) = some (joinNl (lines ls lh)) :=
  fromUTF8?_toUTF8 _
theorem hsplit (ls lh : HmNets.Laid) : (joinNl (lines ls lh)).splitOn "\n" = lines ls lh ++ [""] :=
  splitOn_join _ (fileLs_noNl _ _ _ _ _ noNl_copy16 noNl_meta)

/-! ## `HmRow.parse` and `HmOut.parse` -/

def lsJ : Lean.Json := match J.get metaJ "leaf_scheme" "" with | .ok j => j | .error _ => .null
def pinJ : Lean.Json := tz.pinnedLeafScheme.getD .null

fwd_def cX parse_cX (ls lh : HmNets.Laid) (H C : ByteArray) (A B A' B' : Sparse) (s1 : String)
    (hH : Sha512.hash (bytesOf ls lh) = H)
    (hcl : HmRow.classOf (bytesOf ls lh) "flock-circuit copy16" metaJ = .ok C)
    (hls : VStar.netLines ls = 256002) (hlh : VStar.netLines lh = 246786)
    (hps : Net.parse (netLs "sha512x3" ls).toArray = .ok (shaB A B))
    (hph : Net.parse (netLs "hm96" lh).toArray = .ok (hmB A' B'))
    (hc1 : canon lsJ = .ok s1) (hc2 : canon pinJ = .ok s1)
    (hck : ∀ c : Circuit, (c.nets.getD c.shaNet default).2 = shaB A B → (c.nets.getD c.hmNet default).2 = hmB A' B' →
      HmNets.check c = .ok ()) :
    HmRow.parse tz (bytesOf ls lh) [] by
  [hH, hcl, hutf ls lh, hsplit ls lh, lines_if _, getD_toArray, getElem!_toArray, lines1, trim_meta, starts_meta,
    drop_meta, parse_meta, trim_of, size_lines ls lh hls hlh, line_unit ls lh hls hlh, line_sha ls lh hls hlh,
    line_hm ls lh hls hlh, tokens3, getD3, toNat?_toString, ext_unit ls lh hls hlh, ext_sha ls lh hls hlh,
    ext_hm ls lh hls hlh, parse_unit, hps, hph, forIn_yield, forIn_done, hc1, hc2, bne_self, hck, unit_eq, open_none,
    res_none, sparse_none, opening_none]
  block [HmNets.check, HmNets.shaLaid, HmNets.hmLaid, HmNets.sha512x3Net, HmNets.hm96Net, canon, canonUtf8, Net.parse,
    HmRow.classOf]
  shallow [metaText]

fwd_def cO parseOut_cX (ls lh : HmNets.Laid) (H C : ByteArray) (A B A' B' : Sparse) (s1 : String)
    (hH : Sha512.hash (bytesOf ls lh) = H)
    (hcl : HmRow.classOf (bytesOf ls lh) "flock-circuit copy16" metaJ = .ok C)
    (hls : VStar.netLines ls = 256002) (hlh : VStar.netLines lh = 246786)
    (hps : Net.parse (netLs "sha512x3" ls).toArray = .ok (shaB A B))
    (hph : Net.parse (netLs "hm96" lh).toArray = .ok (hmB A' B'))
    (hc1 : canon lsJ = .ok s1) (hc2 : canon pinJ = .ok s1)
    (hck : ∀ c : Circuit, (c.nets.getD c.shaNet default).2 = shaB A B → (c.nets.getD c.hmNet default).2 = hmB A' B' →
      HmNets.check c = .ok ()) :
    HmOut.parse tz (bytesOf ls lh) [] by
  [parse_cX ls lh H C A B A' B' s1 hH hcl hls hlh hps hph hc1 hc2 hck]
  block [HmNets.check, HmNets.shaLaid, HmNets.hmLaid, HmNets.sha512x3Net, HmNets.hm96Net, canon, canonUtf8, Net.parse,
    HmRow.classOf, HmRow.parse]
  shallow [metaText]

end SpotFlock.InstanceStmt
