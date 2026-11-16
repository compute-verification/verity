module

public import Proofs.Flock.SpotFlock.InstanceStmt.TraceFacts
public import Proofs.Flock.SpotFlock.InstanceStmt.SetupAt
public import Proofs.Flock.SpotFlock.InstanceStmt.TraceDelta
public import Proofs.Flock.SpotFlock.InstanceStmt.TraceDaA
public import Proofs.Flock.SpotFlock.InstanceStmt.TraceDaB
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
# The table's Δ is `EX ++ OX`

At the fixed nets the statement's Δ is `HmOut.delta`'s: `HmRow.delta`'s (`EX`, `TraceDaA.dA_cF`), then the output rows'
pairs (`OX`, `TraceDaB.outPairs_cF`) and no pinned output's (`TraceFacts.pinPairs_cO`): `da_stF`.
-/

@[expose] public section

open Flock FlockSoundness
open SpotFlock.InstanceStmt.TraceDelta (EX OX)

set_option maxRecDepth 200000

namespace SpotFlock.InstanceStmt.TraceDa

/-- The statement's one table at the fixed nets. -/
noncomputable abbrev stF (r : Run) : Stmt :=
  stX r (HmNets.shaLaid ()) (HmNets.hmLaid ()) circuitHash classHash AF BF AF' BF' s1X (s3X (hexOf classHash))
    (s4X r (hexOf circuitHash) (hexOf classHash)) (s5X r cF (hexOf circuitHash) (hexOf classHash)) s6X helloX

/-- **The statement's Δ is `EX ++ OX`.** -/
theorem da_stF (r : Run) : (stF r).da.toList = EX ++ OX := by
  have hda : (stF r).da = oA cF (stF r).pin := rfl
  have hpin := TraceFacts.pin_stX r (HmNets.shaLaid ()) (HmNets.hmLaid ()) circuitHash classHash AF BF AF' BF' s1X
    (s3X (hexOf classHash)) (s4X r (hexOf circuitHash) (hexOf classHash))
    (s5X r cF (hexOf circuitHash) (hexOf classHash)) s6X helloX
  rw [hda, hpin, oA_eq, Array.toList_append, Array.toList_append, TraceDaA.dA_cF, TraceDaB.outPairs_cF,
    TraceFacts.pinPairs_cO]
  simp

end SpotFlock.InstanceStmt.TraceDa
