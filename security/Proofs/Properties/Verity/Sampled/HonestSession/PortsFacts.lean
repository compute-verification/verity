module

public import Lean
public import Spot.OneStageZk
import all Lean.Data.Json.Basic
import all Std.Data.TreeMap.Raw.Basic
import all Std.Data.DTreeMap.Raw.Basic
import all Std.Data.DTreeMap.Internal.Operations
import all Std.Data.DTreeMap.Internal.Queries
import all Std.Data.DTreeMap.Internal.Balancing
import all Std.Data.DTreeMap.Internal.Def
import all Init.Data.Ord.String
import all Init.Data.Ord.Basic
import all Init.Data.String.Basic
import all Init.Data.Array.Basic
import all Init.Data.List.Control
import all Init.WF

/-!
# The derived object's `ports` field, as a rewrite rule for the forward walk

`paramsOfJson` reads the component's parameters off `rest`, the object `split` and `OneStageZk.paramsOfJson` derive from
`paramsJ` by dropping `"form"` and `"proofs"`. One read, `(entries (getD rest "ports")).mapM …`, stalls the forward walk:
`Array.mapM` over a collection the walk never reduces unfolds to a kernel `WellFounded.fix` it cannot step.

This proves the one field fact that unblocks it: over `rest` built from any 16 entry values, its `"ports"` field is the
empty object, so `entries (getD rest "ports") = #[]`. Fed to the walk as a rewrite rule, it fires before the `mapM`
unfolds, and `Array.mapM_empty` then rewrites `Array.mapM f #[]` to `pure #[]`. The entry values are universally
quantified so the rule matches whatever the walk carries in the other fields (the partition object, the challenger, the
generalized digests); only the keys and the object algebra are concrete, which is all the fact needs.

Reducing the `Std.TreeMap.Raw` lookup needs the internal modules opened (`import all …`), the same set the main file
carries; without them the lookup is opaque to `rfl`, `decide` and the kernel. The free entry values rule out `rfl` (its
defeq stalls comparing the two objects) and `decide` (which refuses free variables), so the proof is `kernel_rfl'`: it
closes the goal with `Eq.refl` of the left-hand side and lets the kernel check the two sides equal, the values carried as
opaque locals. The lookup is small (16 short keys), so the one reduction is cheap.
-/

@[expose] public section

namespace Verity.Sampled.Instance.HonestSession.PortsFacts

open Lean (Json)

open Lean Elab Tactic Meta in
/-- Close `a = b` with `Eq.refl a`, leaving the kernel to check `a ≡ b` (a local copy of the forward walk's tactic of
the same name). Unlike `rfl` it does not run the elaborator's defeq, and unlike `decide` it accepts free variables. -/
elab "kernel_rfl'" : tactic => do
  let g ← getMainGoal
  let some (_, a, _) := (← instantiateMVars (← g.getType)).consumeMData.eq? | throwError "kernel_rfl': not an equation"
  g.assign (← mkEqRefl a)
  replaceMainGoal []

/-- `rest`'s `"ports"` field is the empty object, so its entries are `#[]`. The term mirrors what `split` and
`OneStageZk.paramsOfJson` build: `paramsJ`'s members as a tree, drop `"form"`, wrap, drop `"proofs"`, wrap. The `"ports"`
member is the empty object regardless of the other values, so the fact holds for any of them. -/
theorem ports_entries
    (va vc vd vf vla vle vlo vpa vpg vpr vq vr vs vw vwt : Json) :
    Spot.entries
      (Spot.getD
        (Json.mkObj
          (Spot.OneStageZk.without
            (Json.mkObj
              (Array.filter (fun x => x.1 != "form")
                (Std.TreeMap.Raw.ofList
                  [("anchors", va), ("challenger", vc), ("delta", vd), ("form", vf), ("law", vla),
                   ("leaf_layer", vle), ("log", vlo), ("partition", vpa), ("ports", Json.mkObj []),
                   ("program", vpg), ("proofs", vpr), ("query", vq), ("registered", vr),
                   ("scheme", vs), ("window_root", vw), ("work_table", vwt)]).toArray).toList)
            "proofs"))
        "ports") = #[] := by kernel_rfl'

/-- `rest` collapses to a plain object of its surviving keys: `split` and `OneStageZk.paramsOfJson` drop `"form"` and
`"proofs"` from `paramsJ`'s 16 members, leaving 14. Fed to the walk as a rewrite rule, this replaces the derived
`without (mkObj (filter …)) "proofs"` term (whose `ofList`/`filter`/`without` the walk reduces afresh for every field it
reads, each reduction exhausting the heartbeat budget) with one `mkObj` literal, so each later field read is a single
lookup the walk reduces cheaply. The values are universally quantified, so `kabstract` restores whatever the walk carries
in each (the concrete `null`s that drive control flow, the generalized digests). -/
theorem rest_eq
    (va vc vd vf vla vle vlo vpa vpo vpg vpr vq vr vs vw vwt : Json) :
    Json.mkObj
      (Spot.OneStageZk.without
        (Json.mkObj
          (Array.filter (fun x => x.1 != "form")
            (Std.TreeMap.Raw.ofList
              [("anchors", va), ("challenger", vc), ("delta", vd), ("form", vf), ("law", vla),
               ("leaf_layer", vle), ("log", vlo), ("partition", vpa), ("ports", vpo),
               ("program", vpg), ("proofs", vpr), ("query", vq), ("registered", vr),
               ("scheme", vs), ("window_root", vw), ("work_table", vwt)]).toArray).toList)
        "proofs")
      = Json.mkObj
          [("anchors", va), ("challenger", vc), ("delta", vd), ("law", vla), ("leaf_layer", vle), ("log", vlo),
           ("partition", vpa), ("ports", vpo), ("program", vpg), ("query", vq), ("registered", vr), ("scheme", vs),
           ("window_root", vw), ("work_table", vwt)] := by kernel_rfl'

end Verity.Sampled.Instance.HonestSession.PortsFacts
