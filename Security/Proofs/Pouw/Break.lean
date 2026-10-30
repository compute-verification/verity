import Definitions.Pouw.Game.Defs
import Proofs.Pouw.Prob

/-!
# The game's break

A collision among some inputs is one among any inputs that contain them (`collisionIn_mono`), and a model that hashes
nothing never breaks (`not_breaks_of_hashless`), so there an event and no break has the event's probability
(`pr_and_not_breaks`). Every model built without `hashed` hashes nothing (`hashless_of_default`).
-/

namespace Pouw.Proofs

variable {Q R S : Type}

theorem collisionIn_mono {H : Hash.Msg → Hash.Digest} {xs ys : List Hash.Msg} (h : ∀ m ∈ xs, m ∈ ys) :
    Hash.CollisionIn H xs → Hash.CollisionIn H ys := by
  rintro ⟨x, hx, y, hy, hc⟩
  exact ⟨x, h x hx, y, h y hy, hc⟩

theorem not_collisionIn_nil (H : Hash.Msg → Hash.Digest) : ¬ Hash.CollisionIn H [] := by
  rintro ⟨x, hx, -⟩
  exact List.not_mem_nil hx

theorem not_breaks_of_hashless {CM : CostModel Q R S} (hCM : CM.Hashless) (A : Adversary CM) (U : Workload)
    (H : Q → R) (s : S) : ¬ A.Breaks U H s := by
  unfold Adversary.Breaks Adversary.hashed
  rw [hCM]
  exact not_collisionIn_nil _

/-- A model whose `hashed` is the default hashes nothing. -/
theorem hashless_of_default {CM : CostModel Q R S} (h : CM.hashed = fun _ _ _ _ _ _ => []) : CM.Hashless :=
  fun prog H U act σ s => by rw [h]

theorem pr_and_not_breaks {α : Type} [Fintype α] {CM : CostModel Q R S} (hCM : CM.Hashless) (A : Adversary CM)
    (U : Workload) (h : α → Q → R) (s : α → S) (E : α → Prop) :
    pr (fun x => E x ∧ ¬ A.Breaks U (h x) (s x)) = pr E :=
  congrArg pr (funext fun x => propext (and_iff_left (not_breaks_of_hashless hCM A U (h x) (s x))))

/-- A bound on an event bounds it with no break. -/
theorem pr_and_not_breaks_le {α : Type} [Fintype α] {CM : CostModel Q R S} (A : Adversary CM) (U : Workload)
    (h : α → Q → R) (s : α → S) (E : α → Prop) {e : ℝ} (hE : pr E ≤ e) :
    pr (fun x => E x ∧ ¬ A.Breaks U (h x) (s x)) ≤ e :=
  (pr_mono fun _ hx => hx.1).trans hE

end Pouw.Proofs
