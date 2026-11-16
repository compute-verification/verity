module

public import Proofs.Flock.SpotFlock.InstanceStmt.TraceFacts
public import Proofs.Flock.SpotFlock.InstanceStmt.TraceDelta
public import Proofs.Flock.SpotFlock.InstanceStmt.SetupAt
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
# Δ at the fixed nets' circuit, read by the kernel

`HmRow.deltaL` at the statement's circuit `cF` is the list `EX` reversed (`deltaL_cF`), so `HmRow.delta`'s array is `EX`
(`dA_cF`, as `Concrete.Trace.dA_cW`). Stated at the closed circuit, so the kernel shares what the loops recompute: at the
circuit's layouts and hashes as variables it recomputes each column once per pair.
-/

@[expose] public section

open Flock FlockSoundness
open FlockDeploy.Concrete (dA)
open SpotFlock.InstanceStmt.TraceDelta (EX OX)

set_option maxRecDepth 200000
set_option maxHeartbeats 0 -- the kernel evaluates the parsed circuit's Δ

namespace SpotFlock.InstanceStmt.TraceDaA

theorem deltaL_cF : (HmRow.deltaL cF 246784).1 = EX.reverse := by kernel_rfl'

theorem dA_cF : (dA cF 246784).toList = EX := by
  have h := congrArg Prod.fst (FlockDeploy.Concrete.deltaL_eq cF 246784)
  rw [deltaL_cF] at h
  simp only [FlockDeploy.Concrete.Φ, FlockDeploy.Concrete.φ, List.reverse_inj] at h
  exact h

end SpotFlock.InstanceStmt.TraceDaA
