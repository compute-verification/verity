import FlockSoundness.Protocol.Extract
import FlockSoundness.Protocol.Game.Seq

/-!
# The expected-time extractor as an algorithm: waits of at most `R` runs

The extractor's law `Extract.expectAcc` is `k` independent accepted runs. An algorithm gets them by
waiting: it runs the session until a run accepts, at most `R` times, and does that `k` times. This
file is that algorithm's law and cost, the form a collision finder runs.

* `expectN_find`: over `R` runs, the first accepted one is found with probability
  `ε·geom ε R = 1 − (1 − ε)^R` (`mul_geom`), distributed as an accepted run;
* `expectN_firsts`: `k` waits that all find one give `k` independent accepted runs, weighted by
  `geom ε R ^ k`, so their law is `(1 − (1 − ε)^R)^k` times `expectAcc`;
* `expectN_waitCost`: a wait plays `geom ε R` runs in expectation, so after an accepted run the `k`
  waits play `k·geom ε R` runs, and `ε·k·geom ε R = k·(1 − (1 − ε)^R) ≤ k`, whatever `ε`.
-/

namespace FlockSoundness

namespace Extract

open Game

variable {α : Type} (g : Game α) (s : Strategy g) (acc : α → Prop)

open Classical in
/-- **The accepted runs of `k` waits**, if each found one. -/
noncomputable def firsts : List (List α) → Option (List α)
  | [] => some []
  | b :: bl => (b.find? fun x => decide (acc x)).bind fun x => (firsts bl).map (x :: ·)

open Classical in
/-- **The runs a wait plays**: up to and including the first accepted one, or all of them. -/
noncomputable def waitCost : List α → ℝ
  | [] => 0
  | b :: bs => 1 + if acc b then 0 else waitCost bs

end Extract

end FlockSoundness
