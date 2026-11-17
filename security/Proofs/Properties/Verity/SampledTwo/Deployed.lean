module

public import VStarRecursive.SampledTwo
public import Proofs.Properties.Challenger.Limits

/-! Every parameters the two-stage form accepts name a challenger key at the deployed LMS parameters. -/

@[expose] public section

namespace VStar.Recursive.SampledTwo

open Lean (Json)
open Spot

private theorem bind_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : x >>= f = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error => cases h
  | ok a => exact ⟨a, rfl, h⟩

theorem membersParamsOfJson_deployed {j : Json} {p : OneStageZk.Params} (h : OneStageZk.membersParamsOfJson j = .ok p) :
    ∃ key label, p.base.challenger = some (key, label) ∧ key.params = Lms.deployed := by
  unfold OneStageZk.membersParamsOfJson at h
  split at h
  all_goals first
    | (cases h
       done)
    | (obtain ⟨_, -, h⟩ := bind_ok h
       obtain ⟨q, hq, h⟩ := bind_ok h
       try dsimp only at h
       obtain ⟨_, -, h⟩ := bind_ok h
       obtain ⟨_, -, h⟩ := bind_ok h
       simp only [pure, Except.pure, Except.ok.injEq] at h
       subst h
       have hd := Properties.Challenger.Limits.paramsParsed_deployed hq
       exact hd)

theorem paramsOfJson_audit {j : Json} {p : Params} (h : paramsOfJson j = .ok p) :
    (∃ j', OneStageZk.membersParamsOfJson j' = .ok p.audit) ∧
      (!familiesApart (p.boundary.challenger.map (·.2)) (p.audit.base.challenger.map (·.2))) ≠ true := by
  replace h := (paramsOfJson_ok h).1
  unfold paramsParsed at h
  simp only [bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at h
  repeat' split at h
  all_goals try contradiction
  all_goals try (cases h; done)
  all_goals
    (simp only [pure, Except.pure, Except.ok.injEq] at h
     subst h
     exact ⟨⟨_, by assumption⟩, by assumption⟩)

theorem paramsOfJson_deployed {j : Json} {p : Params} (h : paramsOfJson j = .ok p) :
    ∃ key label, p.audit.base.challenger = some (key, label) ∧ key.params = Lms.deployed := by
  obtain ⟨⟨_, hm⟩, -⟩ := paramsOfJson_audit h
  exact membersParamsOfJson_deployed hm

end VStar.Recursive.SampledTwo
