import Mathlib.Logic.Equiv.Fin.Basic
import Mathlib.Tactic.FinCases
import FlockSoundness.Protocol.Audit.OneStage

/-!
# Tightness and insecurity: NOT chains

`K` chains of `nv` NOT gates. Chain `c` is one coarse unit and each of its gates one fine unit, so a
prover can move a lie to any fine unit of a wrong chain. The second stage draws a uniform
`k₂`-subset of each drawn chain's gates. The session is ideal: it accepts exactly when every drawn
fine unit is correct.

* `two_stage_b_tight`: for **every** first-stage law, a prover whose chains are all wrong is accepted
  with probability exactly `effEscape univ`, so `twoStage_count`'s bound is attained;
* `naive_two_level_unsound`: that is more than `L₁.miss K`, what one-stage over the coarse
  partition with `L₁` (and the `m`-dependent reading at `m = nv`) would promise. Nested sampling
  dilutes: after the first draw the prover concentrates each lie into one fine unit;
* `late_interior_insecure`: if the interior is committed after the fine draw, a wrong chain is
  accepted with certainty.
-/

namespace FlockSoundness.Audit

open Game
open scoped ENNReal

namespace Law

/-- A law relabelled along an embedding of its units. -/
def map {m n : ℕ} (L : Law m) (e : Fin m ↪ Fin n) : Law n where
  Ω := L.Ω
  draw ω := (L.draw ω).map e

end Law

namespace NotChains

variable (K nv : ℕ)

variable {K nv}

variable (K nv)

variable {K nv}

variable (K nv)

variable {K nv}

/-! ## The two-stage audit on NOT chains -/

variable (K nv)

variable {K nv}

variable (K nv)

variable {K nv}

/-! ## (b2) with a Bernoulli first stage is one-stage over the coarse partition, thinned -/

/-! ## Fine values chosen after the fine draw -/

end NotChains

end FlockSoundness.Audit
