import FlockSoundness.Protocol.SessionBatch

/-!
# Several tables in one batched session, from accepted reruns (DESIGN.md §3)

`SessionBatch.lean` for the expected-time extractor. The extractor of table `j` fills it from `k`
reruns of everything after `Commit` in which **the session** accepts (every table accepts), so its
relative term is relative to the probability that the audit accepts. Conditioning on table `j`'s
own acceptance would make it relative to that, which can be far larger.

* `extractProbAccB`, `failProbAccB`: the probabilities over those reruns (`Extract.expectAcc`);
* `joint_le_accB`: `Pr[the session accepts] · failProbAccB ≤ ksBoundAccB + rateB · Pr[accepts]`,
  with `ksBoundAccB` table `j`'s `ε_c⁻ + √(k·adv₀)` and `rateB` its `N₀/(ek)`, from
  `session_knowledge_sound_acc`;
* `one_le_fail_add_accB`: the audit's `cover`, when the session accepts with positive probability.
-/

namespace FlockSoundness

open Game Model Accounting

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
variable {D : Type} {mPts : ℕ}

/-- **The rate of table `j`'s relative term**, `N₀/(ek)`. -/
noncomputable def rateB (ts : List (TabSpec mPts)) (j : Fin ts.length) (k : ℕ) : ℝ :=
  2 ^ (ts.get j).l₀.logLen / (Real.exp 1 * k)

end FlockSoundness
