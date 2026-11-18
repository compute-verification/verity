module

public import Proofs.Flock.SpotFlock.InstanceStmt.TraceRegions
public import Proofs.Flock.SpotFlock.InstanceStmt.RelFacts
public import Proofs.Flock.SpotFlock.InstanceStmt.RelParse
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
# The trace is a witness of the deployed relation at the two-instance statement

`rel_x`: the deployed masked relation holds of `x r` at the trace `wX r`: `parse_x`, then `relS_relation` reads it as the table's block R1CS being satisfied by the witness's share, which is `TraceHolds.satisfies_rows` at regions `[]` and `TraceRegions.satisfies_regions_x` at the table's regions, at the layout and row ranges the hidden-output setup gives.
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
/-- **The statement is in the deployed masked relation** with the trace as its witness. -/
theorem rel_x (hn : FixedNetsBuild) {r : Run} (hr : r.Plain) :
    _root_.Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse (x r) (wX r) := by
  have hp := FlockDeploy.fixedNets_parse hn
  have hCe : classHash.isEmpty = false := classOfB_isEmpty (HmNets.shaLaid ()) (HmNets.hmLaid ())
  have hd := setup_stX r (HmNets.shaLaid ()) (HmNets.hmLaid ()) circuitHash classHash AF BF AF' BF' s1X
    (s3X (hexOf classHash)) (s4X r (hexOf circuitHash) (hexOf classHash))
    (s5X r cF (hexOf circuitHash) (hexOf classHash)) s6X helloX hr rfl (classOf_eq _ _)
    (FlockDeploy.sha_lines hn) (FlockDeploy.hm_lines hn) hp.1.2 hp.2.2 hc1_of hc2_of
    (fun c h1 h2 => FlockDeploy.Concrete.check_of hp.1.1 hp.2.1 c h1 h2) hCe (hu1_of _)
    (hu2_of _ _ _ _ _ _ _ _ _ hCe) (hd_of _ _ _) (hd2_of _ _ _ _ _ _ _ _ _ _ _ _) hid_of hhello_of
  have hc := TraceFacts.c_stX r (HmNets.shaLaid ()) (HmNets.hmLaid ()) circuitHash classHash AF BF AF' BF' s1X
    (s3X (hexOf classHash)) (s4X r (hexOf circuitHash) (hexOf classHash))
    (s5X r cF (hexOf circuitHash) (hexOf classHash)) s6X helloX
  have ht : tz.typed = false := typed_tzX
  have hpin0 : HmRow.pinnedCount (stF r).c.metaJ = 0 := by
    rw [hc]
    exact TraceFacts.pinned_cO (HmNets.shaLaid ()) (HmNets.hmLaid ()) circuitHash classHash AF BF AF' BF' s1X
  have hC := (Hidden.setupHidden_specCC hd).committed
  have hrows : (Discharge.Layout.v1Ports (stF r).c || Discharge.Layout.bitRows (stF r).c) = true := by
    rw [hc]
    kernel_rfl'
  have hrf : Hidden.Any.RowsFitS (stF r).c := HiddenExec.rowsFit_of_committed ht hC hrows
  have L : FlockSoundness.Layout (stF r) := (Hidden.setupHidden_specCC hd).layout ht hpin0
  have R : Hm.RowRanges (stF r).c :=
    Hm.RowRanges.ofL (st := Hidden.flatTwin (stF r)) ((Hidden.setupHidden_specCC hd).flatLayout ht)
      ((Hidden.setupHidden_specCC hd).unitFacts ht).parse ((Hidden.setupHidden_specCC hd).unitFacts ht).check
  have hS := TraceRegions.satisfies_regions_x hd ht hrf hpin0 hn L R
    (facts_stX r _ _ _ _ _ _ _ _ _ _ _ _ _ _ (da_stF r)) (regionFacts_stF r)
  exact ⟨_root_.Flock.Readable.toReadable _root_.Flock.Readable.relation (pF r), parse_of (parse_x hn hr),
    relS_of_stmt (p := pF r) (st := stF r) rfl L (zOf_wX r) hS⟩

end SpotFlock.InstanceStmt.Rel
