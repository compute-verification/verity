import Mathlib
import Definitions.Accounting.Guarantees.Actual.WipeBridge
import Proofs.Accounting.Actual.Transducer

/-!
# The lag between reads

`Accounting.Guarantees.Actual.WipeLag`, from the transducer (`Proofs/Accounting/Actual/Transducer.lean`): in a strategy
causal from every tick, what `X` does during `[s, t)` is a function of its content at `s` and of what crosses into it
meanwhile, and its state at `t` is part of what it does then (`content_of_acts`, the state at the end of tick `t − 1`).
With every wire into `X` on a link across its boundary (`DeclaredLinks`), what crosses in at each tick is fixed by those
links' signals, at most `2^{β_X}` values a tick (`portSignals_encard_le`), so the contents at `t` are at most the contents
at `s` times `2^{β_X (t − s)}`.
-/

namespace Accounting.SecurityProofs.Actual

open Finset

variable {L : Type*} [LinearOrder L]

/-- **The state at `u` is part of what `X` does before `u`**: runs alike in what `X` does during `[s, u)`, `s < u`, are
alike in its content at `u`, its state at the end of tick `u − 1`. -/
theorem content_of_acts {R R' : Actual.Run L} {X : Finset L} {s u : ℕ} (hs : s < u)
    (h : R.acts X (Set.Ico s u) = R'.acts X (Set.Ico s u)) : R.content X u = R'.content X u := by
  obtain ⟨v, rfl⟩ : ∃ v, u = v + 1 := ⟨u - 1, by omega⟩
  have h3 : {p | ∃ w a, w ∈ Set.Ico s (v + 1) ∧ a ∈ R.state X (w + 1) ∧ p = (w, R.seen a)} =
      {p | ∃ w a, w ∈ Set.Ico s (v + 1) ∧ a ∈ R'.state X (w + 1) ∧ p = (w, R'.seen a)} :=
    congrArg (fun q => q.2.2) h
  refine content_congr (Set.ext fun q => ?_)
  have key : ∀ Q : Actual.Run L, q ∈ Q.seen '' ↑(Q.state X (v + 1)) ↔
      (v, q) ∈ {p | ∃ w a, w ∈ Set.Ico s (v + 1) ∧ a ∈ Q.state X (w + 1) ∧ p = (w, Q.seen a)} := by
    intro Q
    constructor
    · rintro ⟨a, ha, rfl⟩
      exact ⟨v, a, Set.mem_Ico.2 ⟨by omega, by omega⟩, ha, rfl⟩
    · rintro ⟨w, a, -, ha, he⟩
      obtain ⟨rfl, rfl⟩ := Prod.mk.inj he
      exact ⟨a, ha, rfl⟩
  rw [key R, key R', h3]

theorem WipeLag : Guarantees.Actual.WipeLag := by
  intro Ω L E _ _ S ends β link X s t K hc hd hst
  classical
  obtain ⟨k, rfl⟩ := Nat.exists_eq_add_of_le hst
  rw [Nat.add_sub_cancel_left]
  rcases Nat.eq_zero_or_pos k with rfl | hk
  · simp
  obtain ⟨ports, hports, hsum⟩ : ∃ ports : List (E × L),
      (∀ r g, ∀ a ∈ (S r).A.reads g, (S r).A.x a ∉ X → (S r).A.x g ∈ X → (link r g a, (S r).A.x g) ∈ ports) ∧
      (ports.map fun p => β p.1).sum = linkCapacity ends β X := by
    refine ⟨(boundary ends X).toList.map fun e => (e, if (ends e).1 ∈ X then (ends e).1 else (ends e).2),
      fun r g a hr ha hg => ?_, ?_⟩
    · obtain ⟨-, hends⟩ := (hd.2 r).1 g a hr fun h => ha (h ▸ hg)
      refine List.mem_map.2 ⟨link r g a, Finset.mem_toList.2 ?_, ?_⟩
      · rcases hends with h | h <;> simp [boundary, h, ha, hg]
      · rcases hends with h | h <;> simp [h, ha, hg]
    · rw [List.map_map]
      exact Finset.sum_map_toList _ _
  have hf : Function.FactorsThrough (fun r => (S r).content X (s + k))
      (fun r => ((S r).content X s, (List.range k).map fun i => portSignals S link ports r (s + i))) := by
    intro r r' h
    simp only [Prod.mk.injEq] at h
    exact content_of_acts (by omega) (hc X s (s + k) r r' h.1 (crossing_Ico_congr fun i hi =>
      crossing_of_ports S link X ports hports _ r r' (List.map_inj_left.1 h.2 i (List.mem_range.2 hi))))
  have hsub : Actual.contents S X (s + k) K ⊆
      Function.extend
          (fun r => ((S r).content X s, (List.range k).map fun i => portSignals S link ports r (s + i)))
          (fun r => (S r).content X (s + k)) (fun _ => []) ''
        (Actual.contents S X s K ×ˢ
          Set.range fun r => (List.range k).map fun i => portSignals S link ports r (s + i)) := by
    rintro _ ⟨r, hr, rfl⟩
    exact ⟨(_, _), ⟨⟨r, hr, rfl⟩, ⟨r, rfl⟩⟩, hf.extend_apply _ r⟩
  refine (Set.encard_le_encard hsub).trans ((Set.encard_image_le _ _).trans ?_)
  rw [Set.encard_prod, ← hsum]
  exact mul_le_mul' le_rfl
    (obs_encard_le (portSignals S link ports) s _ (fun u => portSignals_encard_le S β link hd.1 u ports) k)

end Accounting.SecurityProofs.Actual
