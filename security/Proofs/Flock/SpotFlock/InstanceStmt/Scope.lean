module

public import Proofs.Flock.SpotFlock.InstanceStmt.InsSetup
public import Proofs.Flock.SessionLoop.Concrete.Scope
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
import Flock.Draw
import Flock.Desc
import Flock.ProvedScope
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

/-! # The two-instance statement in Flock's scope, and as a template's reference statement

`scopeCheck_x` is `FlockDeploy.scopeCheck true` at the statement `x r`, as `FlockDeploy.concreteStmt_scope` is the
concrete statement's: the verifier parses it to its one table `stX` (`parse_rX`: `setup_rX`, then `shapeOk_stX`), and
its scope refusal passes (`scope_rX`). That refusal is `ProvedScope.check` (`check_rX`) at the reading's masked tags
`tz`, which hide the outputs: the circuit is `HmOut.parse`'s (`circuitOf_cO`), the public file `HmOut.loadPublic`'s
(`load_pubO`), and the two pass `check`'s circuit rules (`circuitRules_cO`); then `sizesOk` at the table's `m`
(`sizes_stX`).

`zkOk_x` is `zkOk` at the statement, from its setup (`setupOf_x`) and its table's sizes (`zkBody_tabX`).
`templateOk_x` is `templateOk` at a registration whose reference statement is `x r`: its reading's tags hide the outputs
and are untyped, its circuit file reads as the flat circuit `cF` (`parseStmt_x`), whose unit has at most `2^32` rows and
whose ports are `sha512/row/v2` rows (`unitLog_cO`, `v2_cO`), and it sets up in one table (`shape_x`). `stmtOf_x` is the
members' reading of its circuit, `cF`. `sessionOk_x` is `sessionOk` at a session statement that is the reference
statement: `zkOk`, the same tables, the scope and hidden outputs (`hiddenOf_x`).
-/

@[expose] public section
open Flock Flock.Firewall RecDeploy.VStarNets
open Flock.Readable (Read)
open Properties.Flock.Nets (sparseOf FixedNetsBuild)
open FlockDeploy.Concrete (shaB hmB bne_self plainStr_hexOf)
open Lean (Json)

set_option maxHeartbeats 200000000 -- the closed checks below leave the circuit's, its rows' and the table's reads to the kernel
set_option maxRecDepth 200000

namespace SpotFlock.InstanceStmt

theorem hidden_tzX : tz.hiddenOutputs = true := by decide +kernel
theorem typed_tzX : tz.typed = false := by decide +kernel
theorem tmpl_tzX : tz.isTmpl = false := by decide +kernel
theorem rules_tzX : ProvedScope.firstFailure (ProvedScope.statementRules tz true 1 false) = none := by kernel_rfl'

/-! ## The circuit and the public file, as the scope check reads them -/

theorem circuitOf_cO (ls lh : HmNets.Laid) (H C : ByteArray) (A B A' B' : Sparse) (s1 : String)
    (hH : Sha512.hash (bytesOf ls lh) = H)
    (hcl : HmRow.classOf (bytesOf ls lh) "flock-circuit copy16" metaJ = .ok C)
    (hls : VStar.netLines ls = 256002) (hlh : VStar.netLines lh = 246786)
    (hps : Net.parse (netLs "sha512x3" ls).toArray = .ok (shaB A B))
    (hph : Net.parse (netLs "hm96" lh).toArray = .ok (hmB A' B'))
    (hc1 : canon lsJ = .ok s1) (hc2 : canon pinJ = .ok s1)
    (hck : ∀ c : Circuit, (c.nets.getD c.shaNet default).2 = shaB A B → (c.nets.getD c.hmNet default).2 = hmB A' B' →
      HmNets.check c = .ok ()) :
    ProvedScope.circuitOf tz (bytesOf ls lh) [] = .ok (cO ls lh H C A B A' B' s1) :=
  ProvedScope.circuitOf_of_parse hidden_tzX (parseOut_cX ls lh H C A B A' B' s1 hH hcl hls hlh hps hph hc1 hc2 hck)

fwd_def pubO load_pubO (r : Run) (ls lh : HmNets.Laid) (H C : ByteArray) (A B A' B' : Sparse) (s1 s3 s4 : String)
    (hr : r.Plain) (hCe : C.isEmpty = false)
    (hu1 : canon (unitsJ (hexOf C)) = .ok s3)
    (hu2 : canon (HmRow.unitsEntry { cO ls lh H C A B A' B' s1 with outPorts := #[] } #[0, 1]) = .ok s3)
    (hd : canonUtf8 (hdrJ r (hexOf H) (hexOf C)) = .ok s4) :
    HmOut.loadPublic tz (cO ls lh H C A B A' B' s1) (pubOf r (hexOf H) (hexOf C)) by
  [findByte_pub hr (plainStr_hexOf H) (plainStr_hexOf C), extract_hdr r (hexOf H) (hexOf C), fromUTF8?_toUTF8,
    parse_hdr hr (plainStr_hexOf H) (plainStr_hexOf C), norm_hdr r (hexOf H) (hexOf C),
    extract_body r (hexOf H) (hexOf C), cO_sha, cO_classSha, bne_self, hCe, any_bne_two, hu1, hu2, hd]
  block [HmNets.check, HmNets.shaLaid, HmNets.hmLaid, HmNets.sha512x3Net, HmNets.hm96Net, canon, canonUtf8, Net.parse,
    HmRow.classOf, HmRow.parse, HmOut.parse, Flock.norm]
  shallow [metaText]

theorem publicOf_cO (r : Run) (ls lh : HmNets.Laid) (H C : ByteArray) (A B A' B' : Sparse) (s1 s3 s4 : String)
    (hr : r.Plain) (hCe : C.isEmpty = false)
    (hu1 : canon (unitsJ (hexOf C)) = .ok s3)
    (hu2 : canon (HmRow.unitsEntry { cO ls lh H C A B A' B' s1 with outPorts := #[] } #[0, 1]) = .ok s3)
    (hd : canonUtf8 (hdrJ r (hexOf H) (hexOf C)) = .ok s4) :
    ProvedScope.publicOf tz (cO ls lh H C A B A' B' s1) (pubOf r (hexOf H) (hexOf C)) =
      .ok (pubO r ls lh H C A B A' B' s1 s3 s4) := by
  rewrite [ProvedScope.publicOf, ite_eq_left hidden_tzX]
  exact load_pubO r ls lh H C A B A' B' s1 s3 s4 hr hCe hu1 hu2 hd

/-- **The circuit and public file pass `check`'s circuit rules**: a shared-row public file under the verifier's own
registered reads, no template, wires that feed neither the unit nor the outputs, and `scopeOkI`. -/
theorem circuitRules_cO (r : Run) (ls lh : HmNets.Laid) (H C : ByteArray) (A B A' B' : Sparse) (s1 s3 s4 : String) :
    ProvedScope.circuitRules tz true 1 true (cO ls lh H C A B A' B' s1) (pubO r ls lh H C A B A' B' s1 s3 s4) =
      .ok () := by
  kernel_rfl'

/-! ## The scope refusal -/

/-- **`verify`'s scope check passes the statement** (`ProvedScope.check`, as `Read.check` calls it). -/
theorem check_rX (r : Run) (ls lh : HmNets.Laid) (H C : ByteArray) (A B A' B' : Sparse) (s1 s3 s4 : String)
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
    (hd : canonUtf8 (hdrJ r (hexOf H) (hexOf C)) = .ok s4) :
    Read.check (rX r ls lh H C) = .ok () := by
  unfold Read.check
  rewrite [tags_rX]
  show ProvedScope.check tz true (bytesOf ls lh) (pubOf r (hexOf H) (hexOf C)) [] 1 false true = .ok ()
  rewrite [ProvedScope.check_eq, circuitOf_cO ls lh H C A B A' B' s1 hH hcl hls hlh hps hph hc1 hc2 hck]
  unfold ProvedScope.checkOf
  rewrite [rules_tzX]
  dsimp only
  rewrite [publicOf_cO r ls lh H C A B A' B' s1 s3 s4 hr hCe hu1 hu2 hd]
  dsimp only
  exact circuitRules_cO r ls lh H C A B A' B' s1 s3 s4

/-- **The table's size is one `--zk` takes.** -/
theorem sizes_stX (r : Run) (ls lh : HmNets.Laid) (H C : ByteArray) (A B A' B' : Sparse) (s1 s3 s4 s5 s6 : String)
    (hello : ByteArray → Nat → String) :
    ProvedScope.sizesOk true
      ((#[stX r ls lh H C A B A' B' s1 s3 s4 s5 s6 hello].map _root_.Flock.Setup.ofCircuit).map _root_.Flock.Setup.m) =
      .ok () := by
  kernel_rfl'

/-- **The verifier parses the statement** to its one table, `stX`. -/
theorem parse_rX (r : Run) (ls lh : HmNets.Laid) (H C : ByteArray) (A B A' B' : Sparse) (s1 s3 s4 s5 s6 : String)
    (hello : ByteArray → Nat → String)
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
    Read.parse (rX r ls lh H C) =
      some { rX r ls lh H C with sts := #[stX r ls lh H C A B A' B' s1 s3 s4 s5 s6 hello] } := by
  have hs : Read.setup (rX r ls lh H C) =
      .ok ((tabX r ls lh H C A B A' B' s1 s3 s4 s5 s6 hello).1, #[stX r ls lh H C A B A' B' s1 s3 s4 s5 s6 hello]) :=
    (setup_rX r ls lh H C A B A' B' s1 s3 s4 s5 s6 hello hr hH hcl hls hlh hps hph hc1 hc2 hck hCe hu1 hu2 hd hd2 hid
      hhello).trans rfl
  unfold Read.parse
  rewrite [FlockDeploy.Concrete.tablesOf_of (r := rX r ls lh H C) rfl hs
    (shapeOk_stX r ls lh H C A B A' B' s1 s3 s4 s5 s6 hello)]
  rfl

/-- **The parsed statement passes `verify`'s scope refusal.** -/
theorem scope_rX (r : Run) (ls lh : HmNets.Laid) (H C : ByteArray) (A B A' B' : Sparse) (s1 s3 s4 s5 s6 : String)
    (hello : ByteArray → Nat → String)
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
    (hd : canonUtf8 (hdrJ r (hexOf H) (hexOf C)) = .ok s4) :
    ({ rX r ls lh H C with sts := #[stX r ls lh H C A B A' B' s1 s3 s4 s5 s6 hello] } :
      _root_.Flock.Readable.Parsed).scope = .ok () :=
  FlockDeploy.Concrete.scope_of
    (check_rX r ls lh H C A B A' B' s1 s3 s4 hr hH hcl hls hlh hps hph hc1 hc2 hck hCe hu1 hu2 hd)
    (sizes_stX r ls lh H C A B A' B' s1 s3 s4 s5 s6 hello)

/-- **The statement is in the scope Flock's soundness proof covers**, where the verifier's two fixed nets build: the
verifier parses it and its scope refusal passes. -/
theorem scopeCheck_x (hn : FixedNetsBuild) {r : Run} (hr : r.Plain) : FlockDeploy.scopeCheck true (x r) = true := by
  have hp := FlockDeploy.fixedNets_parse hn
  have hCe : classHash.isEmpty = false := classOfB_isEmpty (HmNets.shaLaid ()) (HmNets.hmLaid ())
  have hpar := parse_rX r (HmNets.shaLaid ()) (HmNets.hmLaid ()) circuitHash classHash AF BF AF' BF' s1X
    (s3X (hexOf classHash)) (s4X r (hexOf circuitHash) (hexOf classHash))
    (s5X r cF (hexOf circuitHash) (hexOf classHash)) s6X helloX hr rfl (classOf_eq _ _)
    (FlockDeploy.sha_lines hn) (FlockDeploy.hm_lines hn) hp.1.2 hp.2.2 hc1_of hc2_of
    (fun c h1 h2 => FlockDeploy.Concrete.check_of hp.1.1 hp.2.1 c h1 h2) hCe (hu1_of _)
    (hu2_of _ _ _ _ _ _ _ _ _ hCe) (hd_of _ _ _) (hd2_of _ _ _ _ _ _ _ _ _ _ _ _) hid_of hhello_of
  have hsc := scope_rX r (HmNets.shaLaid ()) (HmNets.hmLaid ()) circuitHash classHash AF BF AF' BF' s1X
    (s3X (hexOf classHash)) (s4X r (hexOf circuitHash) (hexOf classHash))
    (s5X r cF (hexOf circuitHash) (hexOf classHash)) s6X helloX hr rfl (classOf_eq _ _)
    (FlockDeploy.sha_lines hn) (FlockDeploy.hm_lines hn) hp.1.2 hp.2.2 hc1_of hc2_of
    (fun c h1 h2 => FlockDeploy.Concrete.check_of hp.1.1 hp.2.1 c h1 h2) hCe (hu1_of _)
    (hu2_of _ _ _ _ _ _ _ _ _ hCe) (hd_of _ _ _)
  unfold FlockDeploy.scopeCheck _root_.Flock.Readable.scopeCheckOf _root_.Flock.Readable.parseBytes
  rewrite [readOf_x hn hr, Option.bind_some, show Read.parse (ins r).read = _ from hpar]
  exact (congrArg Except.isOk hsc).trans rfl

/-! ## The masked scope -/

/-- What `zkOk` asks of a setup's session and tables. -/
def zkBody (spec : _root_.Flock.SessionSpec) (sts : Array _root_.Flock.Stmt) : Bool :=
  0 < sts.size && sts.size ≤ 200 && spec.points ≤ 2 &&
    sts.toList.all fun st => 25 ≤ st.m && st.m ≤ 35 && st.c.kLog ≤ 27 && st.regions.size ≤ 1024 && spec.mPts ≤ 64

theorem zkOk_of {xS : Claims.Msg} {s : FlockDeploy.Ins} {p : _root_.Flock.SessionSpec × Array _root_.Flock.Stmt}
    (h : FlockDeploy.setupOf true xS = some (s, p)) (hb : zkBody p.1 p.2 = true) : SpotFlock.zkOk xS = true := by
  unfold SpotFlock.zkOk
  rewrite [h]
  exact hb

theorem zkBody_tabX (r : Run) (ls lh : HmNets.Laid) (H C : ByteArray) (A B A' B' : Sparse) (s1 s3 s4 s5 s6 : String)
    (hello : ByteArray → Nat → String) :
    zkBody (tabX r ls lh H C A B A' B' s1 s3 s4 s5 s6 hello).1 (tabX r ls lh H C A B A' B' s1 s3 s4 s5 s6 hello).2 =
      true := by
  kernel_rfl'

/-- **The statement is in Flock's masked scope**, where the verifier's two fixed nets build. -/
theorem zkOk_x (hn : FixedNetsBuild) {r : Run} (hr : r.Plain) : SpotFlock.zkOk (x r) = true :=
  zkOk_of (setupOf_x hn hr) (zkBody_tabX r _ _ _ _ _ _ _ _ _ _ _ _ _ _)

/-! ## The statement as a template's reference statement -/

theorem parseOut_x (hn : FixedNetsBuild) :
    HmOut.parse tz (bytesOf (HmNets.shaLaid ()) (HmNets.hmLaid ())) [] = .ok cF := by
  have hp := FlockDeploy.fixedNets_parse hn
  exact parseOut_cX (HmNets.shaLaid ()) (HmNets.hmLaid ()) circuitHash classHash AF BF AF' BF' s1X rfl
    (classOf_eq _ _) (FlockDeploy.sha_lines hn) (FlockDeploy.hm_lines hn) hp.1.2 hp.2.2 hc1_of hc2_of
    (fun c h1 h2 => FlockDeploy.Concrete.check_of hp.1.1 hp.2.1 c h1 h2)

theorem tags_ins (r : Run) : Read.tags (ins r).read = tz := tags_rX _ _ _ _ _

theorem circ_x (hn : FixedNetsBuild) (r : Run) : Read.circ (ins r).read = cF := by
  have hc : ProvedScope.circuitOf (Read.tags (ins r).read) (ins r).read.I.circuitFile (ins r).read.I.tables =
      .ok cF := by
    rewrite [tags_ins]
    exact ProvedScope.circuitOf_of_parse hidden_tzX (parseOut_x hn)
  unfold Read.circ
  rewrite [hc]
  rfl

theorem unrolled_cO (ls lh : HmNets.Laid) (H C : ByteArray) (A B A' B' : Sparse) (s1 : String) :
    HmOut.unrolled (cO ls lh H C A B A' B' s1) = cO ls lh H C A B A' B' s1 := by
  kernel_rfl'

theorem stmt_ins (hn : FixedNetsBuild) (r : Run) : Read.stmt (ins r).read = cF := by
  unfold Read.stmt
  rewrite [circ_x hn r]
  exact unrolled_cO _ _ _ _ _ _ _ _ _

/-- **The members' reading of the statement's circuit** (`stmtOf`) is the flat circuit `cF`. -/
theorem stmtOf_x (hn : FixedNetsBuild) {r : Run} (hr : r.Plain) : SpotFlock.stmtOf (x r) = cF := by
  unfold SpotFlock.stmtOf
  rewrite [readOf_x hn hr]
  exact stmt_ins hn r

theorem parseStmt_x (hn : FixedNetsBuild) (r : Run) :
    _root_.Flock.Desc.parseStmt tz (ins r).read.I.circuitFile (ins r).read.I.tables = .ok (.single cF) := by
  unfold _root_.Flock.Desc.parseStmt
  rewrite [ite_eq_right (by rw [tmpl_tzX]; decide)]
  exact _root_.Flock.Desc.parse_single (parseOut_x hn)

theorem unitLog_cO (ls lh : HmNets.Laid) (H C : ByteArray) (A B A' B' : Sparse) (s1 : String) :
    decide ((cO ls lh H C A B A' B' s1).unit.unitLog ≤ 32) = true := by
  kernel_rfl'

theorem v2_cO (ls lh : HmNets.Laid) (H C : ByteArray) (A B A' B' : Sparse) (s1 : String) :
    (cO ls lh H C A B A' B' s1).ports.all ProvedScope.v2Row = true := by
  kernel_rfl'

theorem shape_x (hn : FixedNetsBuild) {r : Run} (hr : r.Plain) : (SpotFlock.shapeOf (x r)).length = 1 := by
  unfold SpotFlock.shapeOf
  rewrite [setupOf_x hn hr]
  show ((tabAt r).2.toList.map (·.m)).length = 1
  rewrite [List.length_map]
  rfl

/-- **The statement reads as a member's** (`templateOk`) at a registration whose reference statement it is. -/
theorem templateOk_x (hn : FixedNetsBuild) {r : Run} (hr : r.Plain) (h : SpotFlock.Held) (h0 : h.xS₀ 0 = x r) :
    SpotFlock.templateOk h 0 = true := by
  unfold SpotFlock.templateOk
  rewrite [h0, readOf_x hn hr, shape_x hn hr]
  simp only [tags_ins, stmt_ins hn r, parseStmt_x hn r, hidden_tzX, typed_tzX, unitLog_cO, v2_cO, beq_self_eq_true,
    Bool.not_false, Bool.and_self]

/-- **A session statement that is the reference statement is in scope** (`sessionOk`), at a registration whose reference
statement is `x r`. -/
theorem sessionOk_x (hn : FixedNetsBuild) {r : Run} (hr : r.Plain) (h : SpotFlock.Held) (h0 : h.xS₀ 0 = x r) :
    SpotFlock.sessionOk h 0 (x r) = true := by
  unfold SpotFlock.sessionOk
  simp only [h0, zkOk_x hn hr, scopeCheck_x hn hr, hiddenOf_x hn hr, beq_self_eq_true, Bool.and_self]

end SpotFlock.InstanceStmt
