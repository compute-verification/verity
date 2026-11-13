module

public import VStarRecursive.SampledTwo
public import Proofs.Properties.Challenger.Limits

/-!
# The two-stage form's challenger key is at the deployed parameters

The two-stage form's parameters hand the interior's parameters to one-stage v2's per-member parse
(`Spot.OneStageZk.membersParamsOfJson`), which parses them as v2's (`Spot.OneStageZk.paramsOfJson`) and refuses a
challenger key at other parameters than `Lms.deployed` (`KEY_PARAMS`). So every parameters the form accepts name a
challenger key at the deployed parameters (`paramsOfJson_deployed`), the key both stages' coins are read under.
-/

@[expose] public section

namespace VStar.Recursive.SampledTwo

open Lean (Json)
open Spot

private theorem bind_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : x >>= f = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error => cases h
  | ok a => exact ⟨a, rfl, h⟩

/-- **The per-member parse keeps v2's challenger key**, which is at `Lms.deployed`. -/
theorem membersParamsOfJson_deployed {j : Json} {p : OneStageZk.Params} (h : OneStageZk.membersParamsOfJson j = .ok p) :
    ∃ key label, p.base.challenger = some (key, label) ∧ key.params = Lms.deployed := by
  unfold OneStageZk.membersParamsOfJson at h
  split at h
  · cases h
  · obtain ⟨_, -, h⟩ := bind_ok h
    obtain ⟨q, hq, h⟩ := bind_ok h
    try dsimp only at h
    obtain ⟨_, -, h⟩ := bind_ok h
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact Properties.Challenger.Limits.paramsOfJson_deployed hq

/-- **The two-stage form's parameters name a challenger key at the deployed parameters** (module docstring). -/
theorem paramsOfJson_deployed {j : Json} {p : Params} (h : paramsOfJson j = .ok p) :
    ∃ key label, p.audit.base.challenger = some (key, label) ∧ key.params = Lms.deployed := by
  unfold paramsOfJson at h
  simp only [bind, Except.bind] at h
  repeat' split at h
  all_goals try contradiction
  all_goals try (simp [throw, throwThe, MonadExceptOf.throw] at h; done)
  all_goals try (simp only [pure, Except.pure, Except.ok.injEq] at h; subst h;
    exact membersParamsOfJson_deployed (by assumption))

end VStar.Recursive.SampledTwo
