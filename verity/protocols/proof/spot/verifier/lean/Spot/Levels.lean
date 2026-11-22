module

@[expose] public section

namespace Spot.Levels

/-- The refusal of two levels by name (`levels.TWO_LEVELS_REFUSED`): the two-level audit is built, and no soundness
theorem covers its accept, so SPOT's parameters at two levels (`VStar.Recursive.Sampled.paramsOfJson`) and the run
piece under a two-stage rule (`Pieces.run`) are refused before any check. -/
def TWO_LEVELS_REFUSED : String :=
  "levels-not-proved: the parameters name 2 levels: the two-level audit is built and refused, since no soundness " ++
    "theorem covers its accept; an audit is accepted at one level"

end Spot.Levels
