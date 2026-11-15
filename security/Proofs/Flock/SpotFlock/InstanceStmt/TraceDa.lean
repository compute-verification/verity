module

public import Proofs.Flock.SpotFlock.InstanceStmt.TraceFacts
public import Proofs.Flock.SpotFlock.InstanceStmt.TraceDelta
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
# Δ of `x r`'s table, read by the kernel

`HmRow.deltaL` at the statement's circuit is the list `EX` reversed (`deltaL_cO`), so `HmRow.delta`'s array is `EX`
(`dA_cO`, as `Concrete.Trace.dA_cW`); the output rows' pairs are `OX` (`outPairs_cO`) and no output is pinned
(`TraceFacts.pinPairs_cO`). So the statement's Δ is `EX ++ OX` (`da_stX`). Each is a file of its own so a kernel
failure blocks only what reads it.
-/

@[expose] public section

open Flock FlockSoundness
open FlockDeploy.Concrete (dA)
open SpotFlock.InstanceStmt.TraceDelta (EX OX)

set_option maxRecDepth 200000
set_option maxHeartbeats 0 -- the kernel evaluates the parsed circuit's Δ

namespace SpotFlock.InstanceStmt.TraceDa

variable (r : Run) (ls lh : HmNets.Laid) (H C : ByteArray) (A B A' B' : Sparse) (s1 s3 s4 s5 s6 : String)
  (hello : ByteArray → Nat → String)

theorem deltaL_cO :
    (HmRow.deltaL (cO ls lh H C A B A' B' s1) 246784).1 = EX.reverse := by kernel_rfl'

theorem dA_cO : (dA (cO ls lh H C A B A' B' s1) 246784).toList = EX := by
  have h := congrArg Prod.fst (FlockDeploy.Concrete.deltaL_eq (cO ls lh H C A B A' B' s1) 246784)
  rw [deltaL_cO] at h
  simp only [FlockDeploy.Concrete.Φ, FlockDeploy.Concrete.φ, List.reverse_inj] at h
  exact h

theorem outPairs_cO : (outPairs (cO ls lh H C A B A' B' s1)).toList = OX := by kernel_rfl'

/-- **The statement's Δ is `EX ++ OX`.** -/
theorem da_stX : (stX r ls lh H C A B A' B' s1 s3 s4 s5 s6 hello).da.toList = EX ++ OX := by
  have hda : (stX r ls lh H C A B A' B' s1 s3 s4 s5 s6 hello).da =
      oA (cO ls lh H C A B A' B' s1) (stX r ls lh H C A B A' B' s1 s3 s4 s5 s6 hello).pin := rfl
  rw [hda, TraceFacts.pin_stX, oA_eq, Array.toList_append, Array.toList_append, dA_cO,
    outPairs_cO, TraceFacts.pinPairs_cO]
  simp

end SpotFlock.InstanceStmt.TraceDa
