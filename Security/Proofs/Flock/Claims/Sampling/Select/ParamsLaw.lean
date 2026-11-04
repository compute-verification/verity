import Proofs.Flock.Soundness.Refine.Walk
import SampledProofs.OneStage

/-!
# The verifier's law is one `OneStage.rule` accepts as it stands

`OneStage.paramsParsed` keeps the law `OneStage.rule` returns, and `rule` returns its input, so the verifier's law is one
`rule` accepts as it stands (`paramsOfJson_rule`), which is `Properties.SampledProofs.SameLawProfile`'s premise.

* `SameLaw.rule_eq`, `SameLaw.rule_self`: `rule` returns its input, which it then accepts.
* `paramsParsed_rule`, `paramsOfJson_rule`: the verifier's law, as each keeps it.
-/

open SampledProofs

namespace Properties.SampledProofs.SecurityProofs

open Lean (Json)

namespace SameLaw

theorem throw_bind_eq {α β : Type} (e : String) (f : α → Flock.V β) : ((throw e : Flock.V α) >>= f) = throw e := rfl

theorem rule_eq {x y : Json} (h : OneStage.rule x = .ok y) : y = x := by
  unfold OneStage.rule at h
  try dsimp only at h
  try simp only [throw_bind_eq] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    exact h.symm

theorem rule_self {x y : Json} (h : OneStage.rule x = .ok y) : OneStage.rule y = .ok y := by
  obtain rfl := rule_eq h
  exact h

end SameLaw

open SameLaw

/-- **The verifier's law, as `OneStage.paramsParsed` keeps it, is one `OneStage.rule` accepts as it stands.** The body's
`let`s are inlined first (`dsimp only`), so the record it returns names the law `rule` returned. -/
theorem paramsParsed_rule {j : Json} {p : OneStage.Params} (h : OneStage.paramsParsed j = .ok p) :
    OneStage.rule p.verifier.law = .ok p.verifier.law := by
  revert p
  show FlockSoundness.Refine.Ret (OneStage.paramsParsed j) fun p => OneStage.rule p.verifier.law = .ok p.verifier.law
  rw [OneStage.paramsParsed.eq_def]
  dsimp only
  ret_walk [OneStage.rule]
  all_goals exact rule_self (by assumption)

/-- **The verifier's law, as `OneStage.paramsOfJson` keeps it, is one `OneStage.rule` accepts as it stands**:
`SameLawProfile`'s premise holds of every verifier `OneStage.paramsOfJson` accepts. -/
theorem paramsOfJson_rule {j : Json} {p : OneStage.Params} (h : OneStage.paramsOfJson j = .ok p) :
    OneStage.rule p.verifier.law = .ok p.verifier.law := by
  unfold OneStage.paramsOfJson at h
  try dsimp only at h
  try simp only [throw_bind_eq] at h
  repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact paramsParsed_rule (by assumption)

end Properties.SampledProofs.SecurityProofs
