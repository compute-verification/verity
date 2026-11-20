module

public import Properties.Spot.PortsHeld
import Proofs.Properties.Verity.Sampled.MemberForm.Passes

/-! # Proof that the per-member form holds every port of every member's statement (`Properties.Spot.MembersPortsHeld`) -/

@[expose] public section

open Lean (Json)
open Spot

namespace Properties.Spot.PortsHeldProof

/-- The slice and header after dropping the ports `D` in order (`OneStageZk.dropPort`). -/
def dropAll (s : Json × Json) (D : List String) : Json × Json :=
  D.foldl (fun c q => OneStageZk.dropPort c.1 c.2 q) s

theorem dropAll_cons (s : Json × Json) (q : String) (D : List String) :
    dropAll s (q :: D) = dropAll (OneStageZk.dropPort s.1 s.2 q) D := rfl

/-- **A loop that only keeps its state or drops a port that holds `P` ends at its start with ports holding `P`
dropped.** -/
theorem forIn_drops {α : Type} (P : String → Prop) (f : α → Json × Json → Except String (ForInStep (Json × Json)))
    : ∀ (xs : List α), (∀ a ∈ xs, ∀ s s', f a s = .ok s' →
        s' = .yield s ∨ ∃ q, P q ∧ s' = .yield (OneStageZk.dropPort s.1 s.2 q)) →
      ∀ {s0 s : Json × Json}, forIn xs s0 f = .ok s → ∃ D, (∀ q ∈ D, P q) ∧ s = dropAll s0 D
  | [], _, s0, s, h => ⟨[], by simp, by simp [forIn, List.forIn, List.forIn', pure, Except.pure] at h; simpa [dropAll] using h.symm⟩
  | a :: as, hf, s0, s, h => by
    rw [List.forIn_cons] at h
    cases hfa : f a s0 with
    | error e => simp [hfa, bind, Except.bind] at h
    | ok s' =>
      simp only [hfa, bind, Except.bind] at h
      rcases hf a (List.mem_cons_self) s0 s' hfa with rfl | ⟨q, hq, rfl⟩
      · obtain ⟨D, hD, rfl⟩ := forIn_drops P f as (fun b hb => hf b (List.mem_cons_of_mem _ hb)) h
        exact ⟨D, hD, rfl⟩
      · obtain ⟨D, hD, rfl⟩ := forIn_drops P f as (fun b hb => hf b (List.mem_cons_of_mem _ hb)) h
        exact ⟨q :: D, by simp [hq, hD], rfl⟩

set_option pp.proofs false in
#print Spot.OneStageZk.headerHeldAt

end Properties.Spot.PortsHeldProof
