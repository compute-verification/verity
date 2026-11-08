module

public import Proofs.Flock.SessionLoop.Concrete.Public
public import Proofs.Flock.SessionLoop.Concrete.Delta
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

/-! # The concrete statement's setup: `Stmt.setupH` and `Stmt.setupTables`

`setup_stW` walks `Stmt.setupH` on the circuit and public files to the statement `stW`, through `parse_cW`,
`load_pubV` and `checkPublic_cW`, with `HmRow.delta`'s arrays kept as `dA` and `dB` (`Concrete/Delta.lean`), which
`shapeOk_stW` then checks through `deltaL`. `setupTables_W` is `Stmt.setupTables` at one table, `tabW`.
-/

@[expose] public section
open Flock Flock.Firewall RecDeploy.VStarNets FlockVBridge.Names
open Lean (Json)

set_option maxHeartbeats 0 -- the walks below step through `Stmt.setupH` and leave closed checks to the kernel
set_option maxRecDepth 200000

namespace FlockDeploy.Concrete

fwd_def stW setup_stW (ls lh : HmNets.Laid) (H : ByteArray) (A B A' B' : Sparse) (s1 s3 s4 s6 : String)
    (hello : ByteArray → Nat → String)
    (hH : Sha512.hash (bytesOf ls lh) = H)
    (hls : VStar.netLines ls = 256002) (hlh : VStar.netLines lh = 246786)
    (hps : Net.parse (netLs "sha512x3" ls).toArray = .ok (shaB A B))
    (hph : Net.parse (netLs "hm96" lh).toArray = .ok (hmB A' B'))
    (hc1 : canon lsJ = .ok s1) (hc2 : canon pinJ = .ok s1)
    (hck : ∀ c : Circuit, (c.nets.getD c.shaNet default).2 = shaB A B → (c.nets.getD c.hmNet default).2 = hmB A' B' →
      HmNets.check c = .ok ())
    (hu1 : canon unitsJ = .ok s3) (hu2 : canon (HmRow.unitsEntry (cW ls lh H A B A' B' s1) #[0]) = .ok s3)
    (hd : canonUtf8 (hdrJ (hexOf H)) = .ok s4)
    (hid : canonUtf8 tz.identity = .ok s6) (hhello : ∀ σ m, helloOf tz σ m coins0 = .ok (hello σ m)) :
    Stmt.setupH tz (bytesOf ls lh) (pubOf (hexOf H)) coins0 [] none none none by
  [parse_cW ls lh H A B A' B' s1 hH hls hlh hps hph hc1 hc2 hck, load_pubV ls lh H A B A' B' s1 s3 s4 hu1 hu2 hd,
    checkPublic_cW ls lh H A B A' B' s1 s3 s4, bne_self, hid, hhello, unit_eq, delta_ab, crw_nil]
  block [HmNets.check, HmNets.shaLaid, HmNets.hmLaid, HmNets.sha512x3Net, HmNets.hm96Net, canon, canonUtf8, Net.parse,
    HmRow.parse, HmRow.loadPublic, HmRow.checkPublic, HmRow.delta, dA, dB]
  shallow [metaText]

theorem shapeOk_stW (ls lh : HmNets.Laid) (H : ByteArray) (A B A' B' : Sparse) (s1 s3 s4 s6 : String)
    (hello : ByteArray → Nat → String) : Zk.shapeOk (stW ls lh H A B A' B' s1 s3 s4 s6 hello) = .ok () := by
  have hda : (stW ls lh H A B A' B' s1 s3 s4 s6 hello).da =
      dA (cW ls lh H A B A' B' s1) (stW ls lh H A B A' B' s1 s3 s4 s6 hello).pin := rfl
  have hdb : (stW ls lh H A B A' B' s1 s3 s4 s6 hello).db =
      dB (cW ls lh H A B A' B' s1) (stW ls lh H A B A' B' s1 s3 s4 s6 hello).pin := rfl
  unfold Zk.shapeOk Zk.maskFree
  simp only [hda, hdb, all_dA, all_dB]
  kernel_rfl'

fwd_def tabW setupTables_W (ls lh : HmNets.Laid) (H : ByteArray) (A B A' B' : Sparse) (s1 s3 s4 s6 : String)
    (hello : ByteArray → Nat → String)
    (hH : Sha512.hash (bytesOf ls lh) = H)
    (hls : VStar.netLines ls = 256002) (hlh : VStar.netLines lh = 246786)
    (hps : Net.parse (netLs "sha512x3" ls).toArray = .ok (shaB A B))
    (hph : Net.parse (netLs "hm96" lh).toArray = .ok (hmB A' B'))
    (hc1 : canon lsJ = .ok s1) (hc2 : canon pinJ = .ok s1)
    (hck : ∀ c : Circuit, (c.nets.getD c.shaNet default).2 = shaB A B → (c.nets.getD c.hmNet default).2 = hmB A' B' →
      HmNets.check c = .ok ())
    (hu1 : canon unitsJ = .ok s3) (hu2 : canon (HmRow.unitsEntry (cW ls lh H A B A' B' s1) #[0]) = .ok s3)
    (hd : canonUtf8 (hdrJ (hexOf H)) = .ok s4)
    (hid : canonUtf8 tz.identity = .ok s6) (hhello : ∀ σ m, helloOf tz σ m coins0 = .ok (hello σ m)) :
    Stmt.setupTables tz (bytesOf ls lh) (pubOf (hexOf H)) coins0 1 [] none none none none none none none by
  [setup_stW ls lh H A B A' B' s1 s3 s4 s6 hello hH hls hlh hps hph hc1 hc2 hck hu1 hu2 hd hid hhello]
  block [HmNets.check, HmNets.shaLaid, HmNets.hmLaid, HmNets.sha512x3Net, HmNets.hm96Net, canon, canonUtf8, Net.parse,
    HmRow.parse, HmRow.loadPublic, HmRow.checkPublic, Stmt.setupH]
  shallow [metaText]

end FlockDeploy.Concrete
