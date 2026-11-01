import Proofs.Flock.SessionLoop.Fill.Sched
import Proofs.Flock.SessionLoop.Fill.Vals

/-!
# `FillRoundEq` at the session loop's own frames

`LeanRelease.fillRoundEq_ofLoopLink`: `FillRoundEq` at `SessionLoopZKDeployed`'s release from its equation at the rounds
the loop schedules (`FillSched`) and at the inner proof's (`FillInner`, where `Loop.InnerAt` holds). `Loop.FillVals`:
what those rounds ask of the switched session's values `vs`, the stand-in's values at each stopped round.
-/
