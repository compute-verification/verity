module

public import Proofs.Flock.SpotFlock.InstanceStmt.Partition
public import Proofs.Flock.SpotFlock.InstanceStmt.File
public import Proofs.Flock.SpotFlock.InstanceStmt.Decode
import Proofs.Flock.Claims.Sampling.Select.SecondWitness
import Flock.Program
import Flock.HmRow
import Flock.HmNets
import Flock.Canon
import Flock.JsonParse
import Flock.Net
import Flock.ZkServer
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
import Flock.Draw
import Flock.TemplateQuery
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

/-! # The two-instance statement's setup: `Stmt.setupHidden` and `Stmt.setupTables`

`load_pubX` walks `HmIn.loadPublic` on the public file `pubOf` to the population `popX`: its header reads back
(`findByte_pub`, `parse_hdr`), both instances are of the circuit's class `C` (`any_bne_two`), and the header's units
and the header itself are as `canon` and `canonUtf8` write them, which it takes as hypotheses (`hu1`, `hu2`, `hd`).
`setup_stX` walks `Stmt.setupHidden` on the circuit and public files, the sample `sampleJ`, the partition `partObj` and
the program `progBytes` to the statement `stX`: the partition checks (`cpo_part`, `canon_part`), the sample's header is
`hdrD`, whose text it takes as a hypothesis (`hd2`), and Δ is kept as `oA` and `oB`, as `FlockDeploy.Concrete.setup_stW`
keeps `HmRow.delta`'s. Its partition finding is `checkPartition_X`'s, `HmRow.checkPartition` at the statement's circuit
and partition object, derived from the program: the sampled proof block is an instance of the circuit's template
(`deriveTemplateUnits_X`, `HmRow.deriveTemplateUnits`), and no cut wire runs between the units (`linksCheck_X`,
`HmLinks.check`). `setupTables_X` is `Stmt.setupTables` at one table, `tabX`.
-/

@[expose] public section
open Flock Flock.ZkServer RecDeploy.VStarNets
open FlockDeploy.Concrete (shaB hmB bne_self unit_eq plainStr_hexOf)
open Lean (Json)

set_option maxHeartbeats 200000000 -- the walks below step through `Stmt.setupHidden` and leave closed checks to the kernel
set_option maxRecDepth 200000

namespace SpotFlock.InstanceStmt

/-! ## The public file -/

theorem extract_hdr (r : Run) (circ cls : String) :
    (pubOf r circ cls).extract 0 (printJson (hdrJ r circ cls)).toUTF8.size = (printJson (hdrJ r circ cls)).toUTF8 :=
  ByteArray.extract_append_eq_left rfl

theorem extract_body (r : Run) (circ cls : String) :
    (pubOf r circ cls).extract ((printJson (hdrJ r circ cls)).toUTF8.size + 1) (pubOf r circ cls).size = bodyB := by
  rw [pubOf, tailB, ← ByteArray.append_assoc]
  exact ByteArray.extract_append_eq_right (by rw [ByteArray.size_append]; rfl) ByteArray.size_append

/-- Both instances are of class `s`: `classes.any (· != s)` is under a lambda, which a rewrite of `s != s` does not
reach. -/
theorem any_bne_two {l : Array String} {s : String} (hl : l = #[s, s]) : (l.any fun x => x != s) = false := by
  subst hl
  simp

/-- `some <$>` a computation that returns, as the setup reads the partition: no rule matches a `Functor.map`. Its side
condition is walked. -/
theorem map_some_ok {ε α : Type} {m : Except ε α} {a : α} (h : m = .ok a) : (some <$> m) = .ok (some a) := by
  rw [h]; rfl

theorem cO_sha (ls lh : HmNets.Laid) (H C : ByteArray) (A B A' B' : Sparse) (s1 : String) :
    (cO ls lh H C A B A' B' s1).sha = H := rfl
theorem cO_classSha (ls lh : HmNets.Laid) (H C : ByteArray) (A B A' B' : Sparse) (s1 : String) :
    (cO ls lh H C A B A' B' s1).classSha = C := rfl
theorem cO_meta (ls lh : HmNets.Laid) (H C : ByteArray) (A B A' B' : Sparse) (s1 : String) :
    (cO ls lh H C A B A' B' s1).metaJ = metaJ := rfl
theorem cO_g (ls lh : HmNets.Laid) (H C : ByteArray) (A B A' B' : Sparse) (s1 : String) :
    (cO ls lh H C A B A' B' s1).g = 1 := by kernel_rfl'
theorem cO_upv (ls lh : HmNets.Laid) (H C : ByteArray) (A B A' B' : Sparse) (s1 : String) :
    (cO ls lh H C A B A' B' s1).upv = 1 := by kernel_rfl'

/-- The partition's digest in hex, as `HmRow.checkPartition` compares it with META's. -/
theorem partDigest_hex : hexOf (Sha512.hash (HmRow.PARTITION_V1.toUTF8.push 0 ++ partText.toUTF8)) = partHex := by
  have h := partHex_eq
  unfold HmRow.partitionDigest at h
  rw [canon_part] at h
  exact Option.some.inj h

fwd_def popX load_pubX (r : Run) (ls lh : HmNets.Laid) (H C : ByteArray) (A B A' B' : Sparse) (s1 s3 s4 : String)
    (hr : r.Plain) (hCe : C.isEmpty = false)
    (hu1 : canon (unitsJ (hexOf C)) = .ok s3)
    (hu2 : canon (HmRow.unitsEntry { cO ls lh H C A B A' B' s1 with outPorts := #[] } #[0, 1]) = .ok s3)
    (hd : canonUtf8 (hdrJ r (hexOf H) (hexOf C)) = .ok s4) :
    HmIn.loadPublic tz (cO ls lh H C A B A' B' s1) {} (pubOf r (hexOf H) (hexOf C)) by
  [findByte_pub hr (plainStr_hexOf H) (plainStr_hexOf C), extract_hdr r (hexOf H) (hexOf C), fromUTF8?_toUTF8,
    parse_hdr hr (plainStr_hexOf H) (plainStr_hexOf C), norm_hdr r (hexOf H) (hexOf C),
    extract_body r (hexOf H) (hexOf C), cO_sha, cO_classSha, bne_self, hCe, any_bne_two, hu1, hu2, hd]
  block [HmNets.check, HmNets.shaLaid, HmNets.hmLaid, HmNets.sha512x3Net, HmNets.hm96Net, canon, canonUtf8, Net.parse,
    HmRow.classOf, HmRow.parse, HmOut.parse, Flock.norm]
  shallow [metaText]

/-! ## The statement -/

def oA (c : Circuit) (p : Nat) : Array (Nat × Nat) := (HmOut.delta c p).1
def oB (c : Circuit) (p : Nat) : Array (Nat × Nat) := (HmOut.delta c p).2

/-- Δ of a statement without public input ports, `HmOut.delta`'s arrays kept as they are. -/
theorem delta_in (c : Circuit) (p : Nat) : HmIn.delta c {} p = (oA c p, oB c p) := rfl

fwd_def cpX checkPartition_X (ls lh : HmNets.Laid) (H C : ByteArray) (A B A' B' : Sparse) (s1 : String) :
    HmRow.checkPartition (cO ls lh H C A B A' B' s1) (some partObj) by
  [cpo_part, canon_part, cO_meta, partDigest_hex, bne_self]
  block [canon, canonUtf8, HmRow.checkPartitionObject]
  shallow [metaText]

open Properties.Spot.SecurityProofs (qsort_fuel) in
fwd_def dtX deriveTemplateUnits_X (ls lh : HmNets.Laid) (H C : ByteArray) (A B A' B' : Sparse) (s1 : String) :
    HmRow.deriveTemplateUnits (cO ls lh H C A B A' B' s1) partObj progBytes #[0] by
  [Decode.descParse_X, Decode.descX_hex, cO_meta, qsort_fuel, Decode.ex_forIn_one, Decode.ex_forIn_range1]
  block [canon, canonUtf8, TemplateQuery.Desc.parse]
  shallow [metaText]

fwd_def lcX linksCheck_X (r : Run) (ls lh : HmNets.Laid) (H C : ByteArray) (A B A' B' : Sparse) (s1 s3 s4 : String) :
    HmLinks.check (cO ls lh H C A B A' B' s1) (popX r ls lh H C A B A' B' s1 s3 s4) (HmLinks.readersOf (some sampleJ))
      progBytes by
  [Decode.decode_prog, cO_meta, Decode.ex_forIn_one, Decode.ex_forIn_empty, Decode.ex_forIn_range_empty]
  block [canon, canonUtf8, Program.decode]
  shallow [metaText]

fwd_def stX setup_stX (r : Run) (ls lh : HmNets.Laid) (H C : ByteArray) (A B A' B' : Sparse)
    (s1 s3 s4 s5 s6 : String) (hello : ByteArray → Nat → String)
    (hr : r.Plain)
    (hH : Sha512.hash (bytesOf ls lh) = H)
    (hcl : HmRow.classOf (bytesOf ls lh) "flock-circuit copy16" metaJ = .ok C)
    (hls : VStar.netLines ls = 256002) (hlh : VStar.netLines lh = 246786)
    (hps : Net.parse (netLs "sha512x3" ls).toArray = .ok (shaB A B))
    (hph : Net.parse (netLs "hm96" lh).toArray = .ok (hmB A' B'))
    (hc1 : canon lsJ = .ok s1) (hc2 : canon pinJ = .ok s1)
    (hck : ∀ c : Circuit, (c.nets.getD c.shaNet default).2 = shaB A B → (c.nets.getD c.hmNet default).2 = hmB A' B' →
      HmNets.check c = .ok ())
    (hCe : C.isEmpty = false)
    (hu1 : canon (unitsJ (hexOf C)) = .ok s3)
    (hu2 : canon (HmRow.unitsEntry { cO ls lh H C A B A' B' s1 with outPorts := #[] } #[0, 1]) = .ok s3)
    (hd : canonUtf8 (hdrJ r (hexOf H) (hexOf C)) = .ok s4)
    (hd2 : canonUtf8 (hdrD r (cO ls lh H C A B A' B' s1) (hexOf H) (hexOf C)) = .ok s5)
    (hid : canonUtf8 (Tags.rowV2Identity tz.identity) = .ok s6)
    (hhello : ∀ σ m, helloOf tz σ m coinsX = .ok (hello σ m)) :
    Stmt.setupHidden tz (bytesOf ls lh) (pubOf r (hexOf H) (hexOf C)) coinsX [] (some sampleJ) (some partObj)
      (some progBytes) none by
  [parseOut_cX ls lh H C A B A' B' s1 hH hcl hls hlh hps hph hc1 hc2 hck,
    load_pubX r ls lh H C A B A' B' s1 s3 s4 hr hCe hu1 hu2 hd, checkPartition_X ls lh H C A B A' B' s1, map_some_ok,
    deriveTemplateUnits_X ls lh H C A B A' B' s1, linksCheck_X r ls lh H C A B A' B' s1 s3 s4,
    cpo_part, canon_part, cO_sha, cO_classSha, cO_meta, cO_g, cO_upv, hCe, bne_self, hd2, hid,
    hhello, unit_eq, delta_in, hH, hcl, hls, hlh, hps, hph, hc1, hc2, hck, hu1, hu2, hd]
  block [HmNets.check, HmNets.shaLaid, HmNets.hmLaid, HmNets.sha512x3Net, HmNets.hm96Net, canon, canonUtf8, Net.parse,
    HmRow.classOf, HmRow.parse, HmOut.parse, HmIn.loadPublic, HmRow.loadPublic, HmRow.checkPartitionObject,
    HmRow.deriveTemplateUnits, HmLinks.check, HmRow.delta, HmOut.delta, oA, oB]
  shallow [metaText]

fwd_def tabX setupTables_X (r : Run) (ls lh : HmNets.Laid) (H C : ByteArray) (A B A' B' : Sparse)
    (s1 s3 s4 s5 s6 : String) (hello : ByteArray → Nat → String)
    (hr : r.Plain)
    (hH : Sha512.hash (bytesOf ls lh) = H)
    (hcl : HmRow.classOf (bytesOf ls lh) "flock-circuit copy16" metaJ = .ok C)
    (hls : VStar.netLines ls = 256002) (hlh : VStar.netLines lh = 246786)
    (hps : Net.parse (netLs "sha512x3" ls).toArray = .ok (shaB A B))
    (hph : Net.parse (netLs "hm96" lh).toArray = .ok (hmB A' B'))
    (hc1 : canon lsJ = .ok s1) (hc2 : canon pinJ = .ok s1)
    (hck : ∀ c : Circuit, (c.nets.getD c.shaNet default).2 = shaB A B → (c.nets.getD c.hmNet default).2 = hmB A' B' →
      HmNets.check c = .ok ())
    (hCe : C.isEmpty = false)
    (hu1 : canon (unitsJ (hexOf C)) = .ok s3)
    (hu2 : canon (HmRow.unitsEntry { cO ls lh H C A B A' B' s1 with outPorts := #[] } #[0, 1]) = .ok s3)
    (hd : canonUtf8 (hdrJ r (hexOf H) (hexOf C)) = .ok s4)
    (hd2 : canonUtf8 (hdrD r (cO ls lh H C A B A' B' s1) (hexOf H) (hexOf C)) = .ok s5)
    (hid : canonUtf8 (Tags.rowV2Identity tz.identity) = .ok s6)
    (hhello : ∀ σ m, helloOf tz σ m coinsX = .ok (hello σ m)) :
    Stmt.setupTables tz (bytesOf ls lh) (pubOf r (hexOf H) (hexOf C)) coinsX 1 [] (some sampleJ) (some partObj)
      (some progBytes) none none none none by
  [setup_stX r ls lh H C A B A' B' s1 s3 s4 s5 s6 hello hr hH hcl hls hlh hps hph hc1 hc2 hck hCe hu1 hu2 hd hd2 hid
    hhello, cO_meta, cO_classSha, cO_g, cO_upv, hCe, hH, hcl, hls, hlh, hps, hph, hc1, hc2, hck, hu1, hu2, hd, hd2, hid,
    hhello]
  block [HmNets.check, HmNets.shaLaid, HmNets.hmLaid, HmNets.sha512x3Net, HmNets.hm96Net, canon, canonUtf8, Net.parse,
    HmRow.classOf, HmRow.parse, HmOut.parse, HmIn.loadPublic, HmRow.loadPublic, Stmt.setupHidden]
  shallow [metaText]

end SpotFlock.InstanceStmt
