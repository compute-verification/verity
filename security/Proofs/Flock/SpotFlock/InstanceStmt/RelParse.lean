module

public import Proofs.Flock.SpotFlock.InstanceStmt.TraceRegions
public import Proofs.Flock.SpotFlock.InstanceStmt.TraceDa
public import Proofs.Flock.SpotFlock.InstanceStmt.TraceFacts
public import Proofs.Flock.SpotFlock.InstanceStmt.ScopeZk
public import Proofs.Flock.SessionLoop.Concrete.Live
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



/-!
# The statement as the verifier parses it

`pF`, the parsed statement, `wX` the trace as a witness, `zOf_wX` the share of its hidden values, and `parse_x`.
-/
@[expose] public section

open Flock Flock.ZkServer RecDeploy.VStarNets FlockSoundness FlockSoundness.Discharge
open Properties.Flock.Nets (sparseOf FixedNetsBuild)
open FlockDeploy.Concrete (shaB hmB)
open SpotFlock.InstanceStmt.TraceHolds (Facts)
open SpotFlock.InstanceStmt.TraceDa (stF da_stF)
open SpotFlock.InstanceStmt.TraceRegions (RegionFacts bcP commitString_zero_I commitString_zero_O)

set_option maxHeartbeats 200000000
set_option maxRecDepth 200000

namespace SpotFlock.InstanceStmt.Rel
/-! ## The statement at the fixed nets -/

/-- The statement, parsed: its inputs' readable fields and its one table. -/
noncomputable def pF (r : Run) : _root_.Flock.Readable.Parsed :=
  { rX r (HmNets.shaLaid ()) (HmNets.hmLaid ()) circuitHash classHash with sts := #[stF r] }

/-- **The witness**: the trace `TraceVal.zIX`, one hidden value per bit. -/
noncomputable def wX (r : Run) : _root_.Claims.Flock.Wit :=
  ⟨List.ofFn fun a : Fin (2 ^ (stF r).m) => TraceVal.zIX (a.val % 2 ^ 22)⟩

theorem zOf_wX (r : Run) :
    _root_.Flock.Readable.zOf 1 (wX r) 0 (stF r).m = TraceRows.zOf TraceVal.zIX (stF r).m := by
  funext a
  simp only [_root_.Flock.Readable.zOf, wX, Nat.mul_one, Nat.add_zero, List.getD_eq_getElem?_getD,
    List.getElem?_ofFn, a.isLt, dite_true, Option.getD_some, _root_.Flock.Readable.toZ, TraceRows.zOf]

theorem parse_x (hn : FixedNetsBuild) {r : Run} (hr : r.Plain) :
    _root_.Flock.Readable.parseBytes (FlockDeploy.readOf true) (x r) = some (pF r) := by
  have hp := FlockDeploy.fixedNets_parse hn
  have hCe : classHash.isEmpty = false := classOfB_isEmpty (HmNets.shaLaid ()) (HmNets.hmLaid ())
  have hpar := parse_rX r (HmNets.shaLaid ()) (HmNets.hmLaid ()) circuitHash classHash AF BF AF' BF' s1X
    (s3X (hexOf classHash)) (s4X r (hexOf circuitHash) (hexOf classHash))
    (s5X r cF (hexOf circuitHash) (hexOf classHash)) s6X helloX hr rfl (classOf_eq _ _)
    (FlockDeploy.sha_lines hn) (FlockDeploy.hm_lines hn) hp.1.2 hp.2.2 hc1_of hc2_of
    (fun c h1 h2 => FlockDeploy.Concrete.check_of hp.1.1 hp.2.1 c h1 h2) hCe (hu1_of _)
    (hu2_of _ _ _ _ _ _ _ _ _ hCe) (hd_of _ _ _) (hd2_of _ _ _ _ _ _ _ _ _ _ _ _) hid_of hhello_of
  unfold _root_.Flock.Readable.parseBytes
  rewrite [readOf_x hn hr, Option.bind_some, show _root_.Flock.Readable.Read.parse (ins r).read = _ from hpar]
  rfl

/-- **The deployed parser reads a message the verifier's parser reads** (stated at variables, so the message is not unfolded). -/
theorem parse_of {m : _root_.Claims.Msg} {p : _root_.Flock.Readable.Parsed}
    (h : _root_.Flock.Readable.parseBytes (FlockDeploy.readOf true) m = some p) :
    (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse m =
      some (_root_.Flock.Readable.toReadable _root_.Flock.Readable.relation p) := by
  change (_root_.Flock.Readable.parseBytes (FlockDeploy.readOf true) m).map
    (_root_.Flock.Readable.toReadable _root_.Flock.Readable.relation) = _
  rw [h]
  rfl

/-- **A one-table statement whose model the trace satisfies is in the deployed relation.** Stated at any statement `st`,
so the table's numbers (`Tab.Fits`) are read off `st` as a variable: at the concrete statement, the unifier would unfold
it (`Stmt.m` reads the public file) to compare them. -/
theorem relS_of_stmt {p : _root_.Flock.Readable.Parsed} {st : Stmt} {w : _root_.Claims.Flock.Wit}
    (hp : p.sts = #[st]) (L : FlockSoundness.Layout st)
    (hz : _root_.Flock.Readable.zOf 1 w 0 st.m = TraceRows.zOf TraceVal.zIX st.m)
    (hS : (L.model (Refine.regionsOf st)).Satisfies
      (TraceRows.zOf TraceVal.zIX (L.model (Refine.regionsOf st)).m)) :
    _root_.Claims.Flock.RelS (_root_.Flock.Readable.toReadable _root_.Flock.Readable.relation p) w := by
  refine (_root_.Flock.Readable.relS_relation p w).2 fun c hc => ?_
  have h1 : p.sts.size = 1 := by rw [hp]; rfl
  obtain rfl : c = 0 := by omega
  have e : p.sts[0]! = st := by rw [hp]; rfl
  have hm : (_root_.Flock.Readable.toReadable _root_.Flock.Readable.relation p).m = st.m := by
    show (p.sts[0]?.map (·.m)).getD 0 = st.m
    rw [hp]
    rfl
  have hk : (_root_.Flock.Readable.toReadable _root_.Flock.Readable.relation p).kLog = st.c.kLog := by
    show (p.sts[0]?.map (·.c.kLog)).getD 0 = st.c.kLog
    rw [hp]
    rfl
  rw [e]
  refine ⟨⟨hm.symm, hk.symm, (L.model []).hk6, (L.model []).hk, (L.model []).pin.isLt⟩, ?_⟩
  rw [h1, hz]
  exact hS

end SpotFlock.InstanceStmt.Rel
