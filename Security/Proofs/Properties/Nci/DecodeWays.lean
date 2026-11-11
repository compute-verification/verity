module

public import Proofs.Properties.Nci.RecordServed
import Batteries.Tactic.OpenPrivate
import Std.Data.HashMap.Lemmas
import Std.Data.HashSet.Lemmas
import all Flock.Program
import all Flock.Canon
import all Flock.JsonParse
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
import all Init.Data.String.TakeDrop
import all Init.Data.String.Slice
import all Init.Data.String.Pattern.Basic
import all Init.Data.String.Pattern.String
import all Init.Data.String.Pattern.Char
import all Init.Data.String.Search
import all Init.Data.String.Iterate
import all Init.Data.String.Decode
import all Init.Data.String.Termination
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

/-!
# The other ways to run the decoder on a closed text

A scratch file, never merged: `decide` by the elaborator, `rfl`, `rfl` unfolding everything, `unfold` then `rfl`, and
`simp` with the decoder's equation, each on whether `Flock.Program.decode` accepts the text of the Program of no
node; `way_parse` is the same `decide` on the parse alone.
-/

namespace Flock.Program.Closed

open NciCheck.Properties.Record.Attack (programEmpty)

theorem way_parse : (JsonParse.parse programEmpty).toOption.isSome = true := by decide

theorem way_decide : (decode programEmpty.toUTF8).toOption.isSome = true := by decide

theorem way_rfl : (decode programEmpty.toUTF8).toOption.isSome = true := rfl

theorem way_unfold : (decode programEmpty.toUTF8).toOption.isSome = true := by
  unfold decode
  rfl

theorem way_all : (decode programEmpty.toUTF8).toOption.isSome = true := by
  with_unfolding_all rfl

theorem way_simp : (decode programEmpty.toUTF8).toOption.isSome = true := by
  simp [decode]

end Flock.Program.Closed
