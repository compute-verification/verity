module

public import Proofs.Flock.SpotFlock.InstanceStmt.InsSetup
public import Proofs.Flock.SessionLoop.Concrete.Trace
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
# The two-instance table's layout

Kernel facts of `x r`'s circuit (`cO`) and statement (`stX`), at any nets and hashes: the pin is the first `hm96`
slot's constant; two `hm96` slots (net 2) at `0` and `2^18`, one `sha512x3` slot (net 1) at `2^19`, the unit (net 0,
`UN`) at `3 · 2^18`, and the mask's `2^14` rows at `2^20`; no template, no typed rows, no pinned leaves.
-/

@[expose] public section

open Flock FlockDeploy.Concrete
open FlockSoundness (constPairs)

set_option maxRecDepth 200000
set_option maxHeartbeats 2000000

namespace SpotFlock.InstanceStmt.TraceFacts

variable (r : Run) (ls lh : HmNets.Laid) (H C : ByteArray) (A B A' B' : Sparse) (s1 s3 s4 s5 s6 : String)
  (hello : ByteArray → Nat → String)

theorem pin_stX : (stX r ls lh H C A B A' B' s1 s3 s4 s5 s6 hello).pin = 246784 := by kernel_rfl'

theorem c_stX : (stX r ls lh H C A B A' B' s1 s3 s4 s5 s6 hello).c = cO ls lh H C A B A' B' s1 := by kernel_rfl'

theorem kLog_cO : (cO ls lh H C A B A' B' s1).kLog = 22 := by kernel_rfl'

theorem nets_cO : (cO ls lh H C A B A' B' s1).unitNet = 0 ∧ (cO ls lh H C A B A' B' s1).shaNet = 1 ∧
    (cO ls lh H C A B A' B' s1).hmNet = 2 :=
  ⟨by kernel_rfl', by kernel_rfl', by kernel_rfl'⟩

theorem ranges_cO : (cO ls lh H C A B A' B' s1).ranges.toList.map (fun r => (r.kind, r.slotLog, r.pos0, r.count)) =
    [(Kind.net 2, 18, 0, 2), (Kind.net 1, 18, 2, 1), (Kind.net 0, 9, 1536, 1), (Kind.mask, 14, 64, 1)] := by
  kernel_rfl'

theorem unit_cO : (cO ls lh H C A B A' B' s1).unit = UN := by kernel_rfl'

theorem tmpl_cO : (cO ls lh H C A B A' B' s1).tmpl = none := by kernel_rfl'

theorem typed_cO : (cO ls lh H C A B A' B' s1).typed = none := by kernel_rfl'

theorem g_cO : (cO ls lh H C A B A' B' s1).g = 1 ∧ (cO ls lh H C A B A' B' s1).upv = 1 :=
  ⟨by kernel_rfl', by kernel_rfl'⟩

theorem pinned_cO : HmRow.pinnedCount (cO ls lh H C A B A' B' s1).metaJ = 0 := by kernel_rfl'

theorem pinPairs_cO : HmOut.pinPairs (cO ls lh H C A B A' B' s1) = #[] := by kernel_rfl'

theorem unitRange_cO : (cO ls lh H C A B A' B' s1).unitRange.count = 1 ∧
    (cO ls lh H C A B A' B' s1).unitRange.slot 0 = 786432 :=
  ⟨by kernel_rfl', by kernel_rfl'⟩

theorem shaRange_cO : ((cO ls lh H C A B A' B' s1).rangeOf (cO ls lh H C A B A' B' s1).shaNet).count = 1 ∧
    ((cO ls lh H C A B A' B' s1).rangeOf (cO ls lh H C A B A' B' s1).shaNet).slotLog = 18 ∧
    ((cO ls lh H C A B A' B' s1).rangeOf (cO ls lh H C A B A' B' s1).shaNet).slot 0 = 524288 :=
  ⟨by kernel_rfl', by kernel_rfl', by kernel_rfl'⟩

theorem hmRange_cO : ((cO ls lh H C A B A' B' s1).rangeOf (cO ls lh H C A B A' B' s1).hmNet).count = 2 ∧
    ((cO ls lh H C A B A' B' s1).rangeOf (cO ls lh H C A B A' B' s1).hmNet).slotLog = 18 ∧
    ((cO ls lh H C A B A' B' s1).rangeOf (cO ls lh H C A B A' B' s1).hmNet).slot 0 = 0 ∧
    ((cO ls lh H C A B A' B' s1).rangeOf (cO ls lh H C A B A' B' s1).hmNet).slot 1 = 262144 :=
  ⟨by kernel_rfl', by kernel_rfl', by kernel_rfl', by kernel_rfl'⟩

theorem constPos_cO : ((cO ls lh H C A B A' B' s1).nets.getD 0 default).2.constPos = 256 ∧
    ((cO ls lh H C A B A' B' s1).nets.getD 1 default).2.constPos = 256000 ∧
    ((cO ls lh H C A B A' B' s1).nets.getD 2 default).2.constPos = 246784 :=
  ⟨by kernel_rfl', by kernel_rfl', by kernel_rfl'⟩

theorem inWords_cO : ((cO ls lh H C A B A' B' s1).nets.getD 0 default).2.inWords = 1 ∧
    ((cO ls lh H C A B A' B' s1).nets.getD 1 default).2.inWords = 64 ∧
    ((cO ls lh H C A B A' B' s1).nets.getD 2 default).2.inWords = 32 :=
  ⟨by kernel_rfl', by kernel_rfl', by kernel_rfl'⟩

/-- The slot constants' pairs: the second `hm96` slot's, the `sha512x3` slot's and the unit's constants read the pin,
the first `hm96` slot's constant. -/
theorem constPairs_cO : constPairs (cO ls lh H C A B A' B' s1) 246784 =
    [(508928, 508928), (508928, 246784), (780288, 780288), (780288, 246784), (786688, 786688), (786688, 246784)] := by
  kernel_rfl'

end SpotFlock.InstanceStmt.TraceFacts
