import Proofs.Flock.Claims.ZK.Interact

/-!
# `ZKmalR` at two provers with the same messages

* `runStar_msgs_congr`, `runStar_pLog_congr`: two provers whose messages agree at every history within `V*`'s rounds
  give the same transcript, and where one's calls include the other's, so do its run's.
* `fork_pr_mono`: `Fork` finds at least as much at a prover with the same messages and more calls.
* `zkmalR_congr`, `zkmalR_congr_eq`: so `ZKmalR` carries over to such a prover, at `γ ≥ 0`, and to a prover with the
  same messages and calls, at every `γ`.
-/

namespace Claims.ZKMal

open Claims

/-! ## Two provers with the same messages within the rounds -/

/-- Two provers whose messages agree at every history shorter than `K` give the same transcript of at most `K` rounds. -/
theorem interact_congr_lt (ask : List Msg → Out) (pm pm' : List (Msg × Msg) → Msg) (K : ℕ)
    (hv : ∀ hist : List (Msg × Msg), hist.length < K → pm hist = pm' hist) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)), hist.length + n ≤ K → interact ask pm n hist = interact ask pm' n hist
  | 0, _, _ => rfl
  | n + 1, hist, hn => by
      have e := hv hist (by omega)
      simp only [interact, e]
      cases ask (hist.map Prod.fst ++ [pm' hist]) with
      | msg m =>
          dsimp only
          rw [interact_congr_lt ask pm pm' K hv n _ (by simp only [List.length_append, List.length_singleton]; omega)]
      | acc => rfl
      | rej => rfl

theorem runStar_msgs_congr {Ω Θ : Type} (H : Msg → Msg) (P Q : Prover Ω) (Vs : VStar Θ) (x : Msg) (ω : Ω) (θ : Θ)
    (hv : ∀ hist : List (Msg × Msg), hist.length < Vs.k x →
      ((P.next x hist ω).eval H).1 = ((Q.next x hist ω).eval H).1) :
    (runStar H P Vs x ω θ).msgs = (runStar H Q Vs x ω θ).msgs := by
  simp only [runStar_msgs]
  exact interact_congr_lt _ _ _ (Vs.k x) hv _ [] (by simp)

theorem stepWith_pLog_congr {Ω : Type} (H : Msg → Msg) (P Q : Prover Ω) (x : Msg) (ω : Ω)
    (vnext : List Msg → OC Out) (K : ℕ)
    (hv : ∀ hist : List (Msg × Msg), hist.length < K → ((P.next x hist ω).eval H).1 = ((Q.next x hist ω).eval H).1)
    (hl : ∀ hist : List (Msg × Msg), hist.length < K → ((P.next x hist ω).eval H).2 ⊆ ((Q.next x hist ω).eval H).2) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t t' : Trace), hist.length + n ≤ K → t.pLog ⊆ t'.pLog →
      (stepWith H P x ω vnext n hist as t).pLog ⊆ (stepWith H Q x ω vnext n hist as t').pLog
  | 0, _, _, _, _, _, h => by simpa [stepWith] using h
  | n + 1, hist, as, t, t', hn, h => by
      have hs : t.pLog ++ ((P.next x hist ω).eval H).2 ⊆ t'.pLog ++ ((Q.next x hist ω).eval H).2 :=
        List.append_subset.mpr
          ⟨List.Subset.trans h (List.subset_append_left _ _),
            List.Subset.trans (hl hist (by omega)) (List.subset_append_right _ _)⟩
      simp only [stepWith, hv hist (by omega)]
      cases ((vnext (as ++ [((Q.next x hist ω).eval H).1])).eval H).1 with
      | msg m =>
          exact stepWith_pLog_congr H P Q x ω vnext K hv hl n _ _ _ _
            (by simp only [List.length_append, List.length_singleton]; omega) hs
      | acc => exact hs
      | rej => exact hs

theorem runStar_pLog_congr {Ω Θ : Type} (H : Msg → Msg) (P Q : Prover Ω) (Vs : VStar Θ) (x : Msg) (ω : Ω) (θ : Θ)
    (hv : ∀ hist : List (Msg × Msg), hist.length < Vs.k x →
      ((P.next x hist ω).eval H).1 = ((Q.next x hist ω).eval H).1)
    (hl : ∀ hist : List (Msg × Msg), hist.length < Vs.k x →
      ((P.next x hist ω).eval H).2 ⊆ ((Q.next x hist ω).eval H).2) :
    (runStar H P Vs x ω θ).pLog ⊆ (runStar H Q Vs x ω θ).pLog :=
  stepWith_pLog_congr H P Q x ω _ (Vs.k x) hv hl (Vs.k x) [] [] _ _ (by simp) (List.Subset.refl _)

/-- **`Fork` finds at least as much at a prover with the same messages and more calls** within `V*`'s rounds. -/
theorem fork_pr_mono {N Ω Θ : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] (H : Msg → Msg)
    (P Q : Prover (N × Ω)) (Vs : VStar Θ) (θ : Θ) (x : Msg)
    (hv : ∀ (hist : List (Msg × Msg)) ω, hist.length < Vs.k x →
      ((P.next x hist ω).eval H).1 = ((Q.next x hist ω).eval H).1)
    (hl : ∀ (hist : List (Msg × Msg)) ω, hist.length < Vs.k x →
      ((P.next x hist ω).eval H).2 ⊆ ((Q.next x hist ω).eval H).2) :
    (Fork H P Vs θ x).Pr H ≤ (Fork H Q Vs θ x).Pr H := by
  refine le_trans (Pr_mono fun s hs => ?_) (Fork_pr_ge H Q Vs θ x)
  obtain ⟨u, v, hs, -⟩ := hs
  have hne : (Fork H P Vs θ x).out s ≠ none := by
    rw [hs]
    exact Option.some_ne_none _
  obtain ⟨p, hp, q, hq, h₁, h₂⟩ := (findColl_ne_none_iff _).mp hne
  have hsub : ∀ e ∈ (runStar H P Vs x (s.1, s.2.1) θ).pLog ++ (runStar H P Vs x (s.1, s.2.2) θ).pLog,
      e ∈ (runStar H Q Vs x (s.1, s.2.1) θ).pLog ++ (runStar H Q Vs x (s.1, s.2.2) θ).pLog := fun e he => by
    rcases List.mem_append.mp he with he | he
    · exact List.mem_append_left _
        (runStar_pLog_congr H P Q Vs x _ θ (fun h hh => hv h _ hh) (fun h hh => hl h _ hh) he)
    · exact List.mem_append_right _
        (runStar_pLog_congr H P Q Vs x _ θ (fun h hh => hv h _ hh) (fun h hh => hl h _ hh) he)
  exact (findColl_ne_none_iff _).mpr ⟨p, hsub p hp, q, hsub q hq, h₁, h₂⟩

/-- **`ZKmalR` carries over to a prover with the same messages and more calls** within the rounds, at `γ ≥ 0`: the real
transcripts are the same, and its `Fork` finds at least as much. -/
theorem zkmalR_congr {W N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] {H : Msg → Msg} {k : Msg → ℕ}
    {P Q : W → Prover (N × Ω)} {R : Rel W} {S : Simulator} {εzk γ : Msg → ℝ} (hγ : ∀ x, 0 ≤ γ x)
    (hv : ∀ w x (hist : List (Msg × Msg)) ω, hist.length < k x →
      (((P w).next x hist ω).eval H).1 = (((Q w).next x hist ω).eval H).1)
    (hl : ∀ w x (hist : List (Msg × Msg)) ω, hist.length < k x →
      (((P w).next x hist ω).eval H).2 ⊆ (((Q w).next x hist ω).eval H).2)
    (h : ZKmalR H k P R S εzk γ) : ZKmalR H k Q R S εzk γ := by
  intro x w hR Θ Vs θ hk Ev
  have e : ∀ ω, (runStar H (Q w) Vs x ω θ).msgs = (runStar H (P w) Vs x ω θ).msgs := fun ω =>
    (runStar_msgs_congr H (P w) (Q w) Vs x ω θ fun hist hh => hv w x hist ω (hk ▸ hh)).symm
  simp only [e]
  exact (h x w hR Θ Vs θ hk Ev).trans (add_le_add le_rfl (mul_le_mul_of_nonneg_left
    (fork_pr_mono H (P w) (Q w) Vs θ x (fun hist ω hh => hv w x hist ω (hk ▸ hh))
      (fun hist ω hh => hl w x hist ω (hk ▸ hh))) (hγ x)))

/-- **`ZKmalR` carries over to a prover with the same messages and the same calls** within the rounds, at every `γ`. -/
theorem zkmalR_congr_eq {W N Ω : Type} [Fintype N] [Nonempty N] [Fintype Ω] [Nonempty Ω] {H : Msg → Msg}
    {k : Msg → ℕ} {P Q : W → Prover (N × Ω)} {R : Rel W} {S : Simulator} {εzk γ : Msg → ℝ}
    (hv : ∀ w x (hist : List (Msg × Msg)) ω, hist.length < k x →
      ((P w).next x hist ω).eval H = ((Q w).next x hist ω).eval H)
    (h : ZKmalR H k P R S εzk γ) : ZKmalR H k Q R S εzk γ := by
  intro x w hR Θ Vs θ hk Ev
  have e : ∀ ω, (runStar H (Q w) Vs x ω θ).msgs = (runStar H (P w) Vs x ω θ).msgs := fun ω =>
    (runStar_msgs_congr H (P w) (Q w) Vs x ω θ fun hist hh => by rw [hv w x hist ω (hk ▸ hh)]).symm
  have f : (Fork H (Q w) Vs θ x).Pr H = (Fork H (P w) Vs θ x).Pr H :=
    le_antisymm
      (fork_pr_mono H _ _ Vs θ x (fun hist ω hh => by rw [hv w x hist ω (hk ▸ hh)])
        (fun hist ω hh => by rw [hv w x hist ω (hk ▸ hh)]))
      (fork_pr_mono H _ _ Vs θ x (fun hist ω hh => by rw [hv w x hist ω (hk ▸ hh)])
        (fun hist ω hh => by rw [hv w x hist ω (hk ▸ hh)]))
  simp only [e, f]
  exact h x w hR Θ Vs θ hk Ev

end Claims.ZKMal
