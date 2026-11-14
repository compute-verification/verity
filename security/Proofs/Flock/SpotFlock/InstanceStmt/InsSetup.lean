module

public import Proofs.Flock.SpotFlock.InstanceStmt.SetupAt
public import Proofs.Flock.SessionLoop.Concrete.Inputs
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
import Flock.Registered
import Flock.RecSetup
import Flock.RecCoins
import Flock.RecVStar
import Flock.ProvedScope
import Flock.Program
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

/-! # The two-instance statement's checks: `Ins.setup`

`shapeOk_stX` is `Zk.shapeOk` at the statement `stX`. Its Δ is `HmOut.delta`'s: `HmRow.delta`'s arrays, which
`FlockDeploy.Concrete.all_dA` and `all_dB` read through `HmRow.deltaL`, then the output rows' pairs (`outPairs`) and the
pinned outputs' (`HmOut.pinPairs`), whose loops the kernel reads over their lists
(`Std.Legacy.Range.forIn_eq_forIn_range'`). `setup_X` walks `Ins.setup` to `setX` at the reading's tags and setup
(`tags_rX`, `setup_rX`): the registered reads (`Registered.check`, on the units the setup derived from the program), the
full-sample check, the statement checks without an inner shape, the zero-knowledge shape (`shapeOk_stX`), the
recursion's coins and public rows (none at unit `copy16`) and the sampling design. `setupOf_x` is
`FlockDeploy.setupOf true` at the statement `x r`: its inputs `ins r` and its setup `tabAt r`, where the verifier's two
fixed nets build.
-/

@[expose] public section
open Flock RecDeploy.VStarNets
open Flock.Readable (Read)
open FlockDeploy (Ins)
open Properties.Flock.Nets (sparseOf FixedNetsBuild)
open FlockDeploy.Concrete (shaB hmB)
open Lean (Json)

set_option maxHeartbeats 0 -- the walk below steps through `Ins.setup` and leaves closed checks to the kernel
set_option maxRecDepth 200000

namespace SpotFlock.InstanceStmt

/-! ## The zero-knowledge shape -/

/-- `HmOut.delta`'s pairs on the output rows' message bits, its loop as it pushes them. -/
def outPairs (c : Circuit) : Array (Nat × Nat) := Id.run do
  let (_, where_) := HmRow.rowSlots c.ports
  let sin := portCols (c.nets.getD c.shaNet default).2.inGroups
  let hin := portCols (c.nets.getD c.hmNet default).2.inGroups
  let msgCol := fun (g p byte t : Nat) =>
    let (k, o) := HmRow.byteAt (c.ports.getD p default) byte
    if k < where_[p]!.size then
      let (slot, sub) := (where_[p]!)[k]!
      c.slotCol c.shaNet g slot + sin[2 * sub + 1]!.1 + 64 * (o / 8) + 56 - 8 * (o % 8) + t
    else c.slotCol c.hmNet g p + hin[1]!.1 + 64 * (o / 8) + 56 - 8 * (o % 8) + t
  let forcedZero := c.unitRange.slot 0 + c.unit.useful
  let col := HmOut.outCol c
  let k0 := HmOut.nIn c
  let mut d : Array (Nat × Nat) := #[]
  for g in [0:c.g] do
    for j in [0:c.outRowPorts.size] do
      let (_, _, at_, w) := HmOut.outGroup c j
      for k in [0:(c.ports.getD (k0 + j) default).words] do
        for t in [0:16] do
          let i := msgCol g (k0 + j) (2 * k + t / 8) (t % 8)
          d := d.push (i, i) |>.push (i, if k < w then col g (at_ + k) + t else forcedZero)
  return d

theorem oA_eq (c : Circuit) (p : Nat) :
    oA c p = FlockDeploy.Concrete.dA c p ++ outPairs c ++ HmOut.pinPairs c := by kernel_rfl'

theorem oB_eq (c : Circuit) (p : Nat) :
    oB c p = FlockDeploy.Concrete.dB c p ++ outPairs c ++ HmOut.pinPairs c := by kernel_rfl'

theorem shapeOk_stX (r : Run) (ls lh : HmNets.Laid) (H C : ByteArray) (A B A' B' : Sparse) (s1 s3 s4 s5 s6 : String)
    (hello : ByteArray → Nat → String) : Zk.shapeOk (stX r ls lh H C A B A' B' s1 s3 s4 s5 s6 hello) = .ok () := by
  have hda : (stX r ls lh H C A B A' B' s1 s3 s4 s5 s6 hello).da =
      oA (cO ls lh H C A B A' B' s1) (stX r ls lh H C A B A' B' s1 s3 s4 s5 s6 hello).pin := rfl
  have hdb : (stX r ls lh H C A B A' B' s1 s3 s4 s5 s6 hello).db =
      oB (cO ls lh H C A B A' B' s1) (stX r ls lh H C A B A' B' s1 s3 s4 s5 s6 hello).pin := rfl
  unfold Zk.shapeOk Zk.maskFree
  simp only [hda, hdb, oA_eq, oB_eq, Array.all_append, FlockDeploy.Concrete.all_dA, FlockDeploy.Concrete.all_dB,
    outPairs, HmOut.pinPairs, Std.Legacy.Range.forIn_eq_forIn_range']
  kernel_rfl'

/-! ## The checks -/

theorem cO_name (ls lh : HmNets.Laid) (H C : ByteArray) (A B A' B' : Sparse) (s1 : String) :
    (cO ls lh H C A B A' B' s1).unit.name = "copy16" := by kernel_rfl'

-- Blocking a constant here stops the walk unfolding any value whose body reaches it; `stX`'s body reads `oA` and `oB`.
fwd_def setX setup_X (r : Run) (ls lh : HmNets.Laid) (H C : ByteArray) (A B A' B' : Sparse)
    (s1 s3 s4 s5 s6 : String) (hello : ByteArray → Nat → String) (s : Ins)
    (hr : s.read = rX r ls lh H C) (hreg : s.registered = some regPorts) (hst : s.stmtTags = t0)
    (hT : Read.tags (rX r ls lh H C) = tz) (hCe : C.isEmpty = false)
    (hS : Read.setup (rX r ls lh H C) = .ok ((tabX r ls lh H C A B A' B' s1 s3 s4 s5 s6 hello).1,
      (tabX r ls lh H C A B A' B' s1 s3 s4 s5 s6 hello).2)) :
    Ins.setup s by
  [hr, hreg, hst, hT, hCe, hS, FlockDeploy.Concrete.forIn_one, FlockDeploy.Concrete.forM_one,
    shapeOk_stX r ls lh H C A B A' B' s1 s3 s4 s5 s6 hello, cO_name ls lh H C A B A' B' s1,
    FlockDeploy.Concrete.open_none, FlockDeploy.Concrete.res_none, FlockDeploy.Concrete.sparse_none,
    FlockDeploy.Concrete.opening_none]
  block [HmNets.check, HmNets.shaLaid, HmNets.hmLaid, canon, canonUtf8, Net.parse, HmRow.parse, HmIn.loadPublic,
    HmRow.loadPublic, Stmt.setupHidden, Stmt.setupTables, Zk.maskLogOf, Zk.shapeOk, HmIn.delta]
  shallow [metaText]

/-! ## At the fixed nets -/

/-- **The statement's inputs and setup**, where the verifier's two fixed nets build. -/
theorem setupOf_x (hn : FixedNetsBuild) {r : Run} (hr : r.Plain) :
    FlockDeploy.setupOf true (x r) = some (ins r, tabAt r) := by
  have hs := setup_X r (HmNets.shaLaid ()) (HmNets.hmLaid ()) circuitHash classHash AF BF AF' BF' s1X
    (s3X (hexOf classHash)) (s4X r (hexOf circuitHash) (hexOf classHash))
    (s5X r cF (hexOf circuitHash) (hexOf classHash)) s6X helloX (ins r) rfl rfl rfl (tags_rX _ _ _ _ _)
    (classOfB_isEmpty (HmNets.shaLaid ()) (HmNets.hmLaid ())) (setup_x hn hr)
  exact FlockDeploy.setupOf_of (insOf_x hn hr) hs

end SpotFlock.InstanceStmt
