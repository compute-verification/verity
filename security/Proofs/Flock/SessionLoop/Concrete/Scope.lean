module

public import Proofs.Flock.SessionLoop.Concrete.Stmt
public import Proofs.Flock.Claims.DeployCode
public import Proofs.Flock.Claims.Scope.Matches
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
import Flock.ProvedScope
import Flock.Verify
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
import all Init.Data.Array.Attach
import all Init.WF

/-! # The concrete statement is in the deployed `--zk` scope

`concreteStmt_scope`: `Claims.Flock.Scope (deployZk relation) concreteStmt`, at the open guarantee that the verifier's two
fixed nets build. By `Readable.scope_iff` that is the verifier's parse of the statement (`Read.parse`: its one table,
`stW`, passes `Zk.shapeOk`) and its scope refusal (`Parsed.scope`) passing:

* `check_rC`: `ProvedScope.check` at the session's tags `tz`, which have public outputs, so the circuit is `HmRow.parse`'s
  (`parse_cW`) and the public file `HmRow.loadPublic`'s (`load_pubV`). The circuit has no template and no shared-row public
  file, its wires feed neither the unit nor the outputs, and it passes `scopeOk`: the unit `copy16` reads its one leaf
  group, returns the outputs, has `2^9` rows and reads only port bits, and its one row is a `sha512/row/v1` row;
* `sizes_stW`: the table's `m` is `k_log + 3 = 25`, the least `--zk` takes.
-/

@[expose] public section
open Flock RecDeploy.VStarNets FlockVBridge.Names
open Flock.Readable (Read)

set_option maxHeartbeats 0 -- the closed checks below leave the unit's and the META's reads to the kernel
set_option maxRecDepth 200000

namespace FlockDeploy.Concrete

theorem hidden_tz : tz.hiddenOutputs = false := by decide +kernel

theorem rules_tz : ProvedScope.firstFailure (ProvedScope.statementRules tz true 1 false) = none := by kernel_rfl

theorem circuitOf_cW (ls lh : HmNets.Laid) (H : ByteArray) (A B A' B' : Sparse) (s1 : String)
    (hH : Sha512.hash (bytesOf ls lh) = H)
    (hls : VStar.netLines ls = 256002) (hlh : VStar.netLines lh = 246786)
    (hps : Net.parse (netLs "sha512x3" ls).toArray = .ok (shaB A B))
    (hph : Net.parse (netLs "hm96" lh).toArray = .ok (hmB A' B'))
    (hc1 : canon lsJ = .ok s1) (hc2 : canon pinJ = .ok s1)
    (hck : ∀ c : Circuit, (c.nets.getD c.shaNet default).2 = shaB A B → (c.nets.getD c.hmNet default).2 = hmB A' B' →
      HmNets.check c = .ok ()) :
    ProvedScope.circuitOf tz (bytesOf ls lh) [] = .ok (cW ls lh H A B A' B' s1) := by
  rewrite [ProvedScope.circuitOf, ite_eq_right (show ¬ (tz.hiddenOutputs = true) by rw [hidden_tz]; decide)]
  exact parse_cW ls lh H A B A' B' s1 hH hls hlh hps hph hc1 hc2 hck

theorem publicOf_cW (ls lh : HmNets.Laid) (H : ByteArray) (A B A' B' : Sparse) (s1 s3 s4 : String)
    (hu1 : canon unitsJ = .ok s3) (hu2 : canon (HmRow.unitsEntry (cW ls lh H A B A' B' s1) #[0]) = .ok s3)
    (hd : canonUtf8 (hdrJ (hexOf H)) = .ok s4) :
    ProvedScope.publicOf tz (cW ls lh H A B A' B' s1) (pubOf (hexOf H)) = .ok (pubV ls lh H A B A' B' s1 s3 s4) := by
  rewrite [ProvedScope.publicOf, ite_eq_right (show ¬ (tz.hiddenOutputs = true) by rw [hidden_tz]; decide)]
  exact load_pubV ls lh H A B A' B' s1 s3 s4 hu1 hu2 hd

/-- **The circuit and public file pass `check`'s circuit rules**: no shared-row public file, no template, wires that feed
neither the unit nor the outputs, and `scopeOk`. -/
theorem circuitRules_cW (ls lh : HmNets.Laid) (H : ByteArray) (A B A' B' : Sparse) (s1 s3 s4 : String) :
    ProvedScope.circuitRules tz true 1 false (cW ls lh H A B A' B' s1) (pubV ls lh H A B A' B' s1 s3 s4) = .ok () := by
  kernel_rfl

/-- **`verify`'s scope check passes the concrete statement** (`ProvedScope.check`, as `Read.check` calls it). -/
theorem check_rC (ls lh : HmNets.Laid) (H : ByteArray) (A B A' B' : Sparse) (s1 s3 s4 : String)
    (hH : Sha512.hash (bytesOf ls lh) = H)
    (hls : VStar.netLines ls = 256002) (hlh : VStar.netLines lh = 246786)
    (hps : Net.parse (netLs "sha512x3" ls).toArray = .ok (shaB A B))
    (hph : Net.parse (netLs "hm96" lh).toArray = .ok (hmB A' B'))
    (hc1 : canon lsJ = .ok s1) (hc2 : canon pinJ = .ok s1)
    (hck : ∀ c : Circuit, (c.nets.getD c.shaNet default).2 = shaB A B → (c.nets.getD c.hmNet default).2 = hmB A' B' →
      HmNets.check c = .ok ())
    (hu1 : canon unitsJ = .ok s3) (hu2 : canon (HmRow.unitsEntry (cW ls lh H A B A' B' s1) #[0]) = .ok s3)
    (hd : canonUtf8 (hdrJ (hexOf H)) = .ok s4) :
    Read.check (rC ls lh H) = .ok () := by
  unfold Read.check
  rewrite [tags_rC]
  show ProvedScope.check tz true (bytesOf ls lh) (pubOf (hexOf H)) [] 1 false false = .ok ()
  rewrite [ProvedScope.check_eq, circuitOf_cW ls lh H A B A' B' s1 hH hls hlh hps hph hc1 hc2 hck]
  unfold ProvedScope.checkOf
  rewrite [rules_tz]
  dsimp only
  rewrite [publicOf_cW ls lh H A B A' B' s1 s3 s4 hu1 hu2 hd]
  dsimp only
  exact circuitRules_cW ls lh H A B A' B' s1 s3 s4

/-- **The table's size is one `--zk` takes**: `m = 25`. -/
theorem sizes_stW (ls lh : HmNets.Laid) (H : ByteArray) (A B A' B' : Sparse) (s1 s3 s4 s6 : String)
    (hello : ByteArray → Nat → String) :
    ProvedScope.sizesOk true
      ((#[stW ls lh H A B A' B' s1 s3 s4 s6 hello].map _root_.Flock.Setup.ofCircuit).map _root_.Flock.Setup.m) = .ok () := by
  kernel_rfl

/-- A `--zk` read whose setup makes one table passing `Zk.shapeOk` has that table. -/
theorem tablesOf_of {r : Read} {spec : _root_.Flock.SessionSpec} {st : _root_.Flock.Stmt} (hz : r.zk = true)
    (hs : r.setup = .ok (spec, #[st])) (hk : Zk.shapeOk st = .ok ()) : r.tablesOf = some #[st] := by
  unfold Read.tablesOf
  rewrite [hs]
  dsimp only
  rewrite [ite_eq_left hz, forM_one, hk]
  rfl

/-- A parsed statement whose scope check passes and whose tables' sizes `sizesOk` takes passes `Parsed.scope`. -/
theorem scope_of {p : _root_.Flock.Readable.Parsed} (hc : p.check = .ok ())
    (hs : ProvedScope.sizesOk p.zk ((p.sts.map _root_.Flock.Setup.ofCircuit).map _root_.Flock.Setup.m) = .ok ()) :
    p.scope = .ok () := by
  unfold _root_.Flock.Readable.Parsed.scope
  rewrite [hc]
  exact hs

/-- **The verifier parses the concrete statement** to its one table, `stW`. -/
theorem parse_rC (ls lh : HmNets.Laid) (H : ByteArray) (A B A' B' : Sparse) (s1 s3 s4 s6 : String)
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
    Read.parse (rC ls lh H) = some { rC ls lh H with sts := #[stW ls lh H A B A' B' s1 s3 s4 s6 hello] } := by
  have hs : Read.setup (rC ls lh H) =
      .ok ((tabW ls lh H A B A' B' s1 s3 s4 s6 hello).1, #[stW ls lh H A B A' B' s1 s3 s4 s6 hello]) :=
    (setup_rC ls lh H A B A' B' s1 s3 s4 s6 hello hH hls hlh hps hph hc1 hc2 hck hu1 hu2 hd hid hhello).trans rfl
  unfold Read.parse
  rewrite [tablesOf_of (r := rC ls lh H) rfl hs (shapeOk_stW ls lh H A B A' B' s1 s3 s4 s6 hello)]
  rfl

/-- **The parsed concrete statement passes `verify`'s scope refusal.** -/
theorem scope_rC (ls lh : HmNets.Laid) (H : ByteArray) (A B A' B' : Sparse) (s1 s3 s4 s6 : String)
    (hello : ByteArray → Nat → String)
    (hH : Sha512.hash (bytesOf ls lh) = H)
    (hls : VStar.netLines ls = 256002) (hlh : VStar.netLines lh = 246786)
    (hps : Net.parse (netLs "sha512x3" ls).toArray = .ok (shaB A B))
    (hph : Net.parse (netLs "hm96" lh).toArray = .ok (hmB A' B'))
    (hc1 : canon lsJ = .ok s1) (hc2 : canon pinJ = .ok s1)
    (hck : ∀ c : Circuit, (c.nets.getD c.shaNet default).2 = shaB A B → (c.nets.getD c.hmNet default).2 = hmB A' B' →
      HmNets.check c = .ok ())
    (hu1 : canon unitsJ = .ok s3) (hu2 : canon (HmRow.unitsEntry (cW ls lh H A B A' B' s1) #[0]) = .ok s3)
    (hd : canonUtf8 (hdrJ (hexOf H)) = .ok s4) :
    ({ rC ls lh H with sts := #[stW ls lh H A B A' B' s1 s3 s4 s6 hello] } : _root_.Flock.Readable.Parsed).scope =
      .ok () :=
  scope_of (check_rC ls lh H A B A' B' s1 s3 s4 hH hls hlh hps hph hc1 hc2 hck hu1 hu2 hd)
    (sizes_stW ls lh H A B A' B' s1 s3 s4 s6 hello)

end FlockDeploy.Concrete

namespace FlockDeploy

open Properties.Flock.Nets (sparseOf FixedNetsBuild)

/-- **The concrete statement is in the deployed `--zk` scope**, at the open guarantee that the verifier's two fixed nets
build: the verifier parses it and its scope refusal passes (`Readable.scope_iff`). -/
theorem concreteStmt_scope (hn : FixedNetsBuild) :
    Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) concreteStmt := by
  have hp := fixedNets_parse hn
  have hpar := Concrete.parse_rC (HmNets.shaLaid ()) (HmNets.hmLaid ()) concreteHash (sparseOf (HmNets.shaLaid ()).ra)
    (sparseOf (HmNets.shaLaid ()).rb) (sparseOf (HmNets.hmLaid ()).ra) (sparseOf (HmNets.hmLaid ()).rb) Concrete.s1C
    Concrete.s3C (Concrete.s4Of concreteHash) Concrete.s6C Concrete.helloC rfl (sha_lines hn) (hm_lines hn) hp.1.2
    hp.2.2 Concrete.hc1_of Concrete.hc2_of (fun c h1 h2 => Concrete.check_of hp.1.1 hp.2.1 c h1 h2) Concrete.hu1_of
    (Concrete.hu2_of _ _ _ _ _ _ _ _) (Concrete.hd_of _) Concrete.hid_of Concrete.hhello_of
  have hsc := Concrete.scope_rC (HmNets.shaLaid ()) (HmNets.hmLaid ()) concreteHash (sparseOf (HmNets.shaLaid ()).ra)
    (sparseOf (HmNets.shaLaid ()).rb) (sparseOf (HmNets.hmLaid ()).ra) (sparseOf (HmNets.hmLaid ()).rb) Concrete.s1C
    Concrete.s3C (Concrete.s4Of concreteHash) Concrete.s6C Concrete.helloC rfl (sha_lines hn) (hm_lines hn) hp.1.2
    hp.2.2 Concrete.hc1_of Concrete.hc2_of (fun c h1 h2 => Concrete.check_of hp.1.1 hp.2.1 c h1 h2) Concrete.hu1_of
    (Concrete.hu2_of _ _ _ _ _ _ _ _) (Concrete.hd_of _)
  refine (_root_.Flock.Readable.scope_iff (FlockDeploy.deployZk _root_.Flock.Readable.relation)
    _root_.Flock.Readable.relation (FlockDeploy.readOf true) rfl (fun _ _ h => FlockDeploy.readOf_zk h) concreteStmt).2
    ⟨_, ?_, hsc⟩
  unfold _root_.Flock.Readable.parseBytes FlockDeploy.readOf
  rewrite [concreteStmt_ins hn]
  exact hpar

end FlockDeploy
