module

public import Spot.OneStageZkStatement

@[expose] public section

namespace Spot.HeaderHeldAt

open Lean (Json)

/-- `OneStageZk.headerHeldAt` on `{reads, slice, header}`, `reads` a list of `[port, value]`: the empty string where it
holds, else its refusal. -/
def piece (j : Json) : Except String Json := do
  let .arr rs := getD j "reads" | throw "reads is not a list"
  let reads ← rs.mapM fun r => match r with
    | .arr #[.str p, .str v] => pure (p, v)
    | _ => throw "a read is not [port, value]"
  match OneStageZk.headerHeldAt reads (getD j "slice") (getD j "header") with
  | .ok () => pure (Json.str "")
  | .error e => pure (Json.str e)

end Spot.HeaderHeldAt
