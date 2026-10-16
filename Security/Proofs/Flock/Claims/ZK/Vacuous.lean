import Proofs.Flock.Claims.Prob

/-!
# `ZKmal` as frozen holds only at `εzk ≥ 1`

The simulator reads the verifier only through its next message (`Simulator.out` asks `Vs.next`), never through its
rounds `Vs.k`, while the real transcript (`runStar`) lasts `Vs.k x` rounds. The verifiers `chatty n` answer every
message alike and stop after `n` rounds: one simulated distribution, and a real transcript of length exactly `n` for
each `n`. No distribution has every length with probability near `1`, so `ZKmal` holds exactly when `εzk x ≥ 1` at
every statement with a witness (`ZKmal_iff`), where `|Pr − Pr| ≤ 1` says nothing; so does `Flock.ZK` (`flockZK_iff`).
-/

namespace Claims.ZKMal

open Claims

/-- The verifier that answers every message with the empty message, for `n` rounds. -/
def chatty (n : ℕ) : VStar Unit := ⟨fun _ => n, fun _ _ _ => .ret (.msg [])⟩

theorem stepWith_chatty_length {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace),
      (stepWith H P x ω (fun _ => .ret (.msg [])) n hist as t).msgs.length = t.msgs.length + n
  | 0, _, _, _ => rfl
  | n + 1, hist, as, t => by
      simp only [stepWith, OC.eval]
      rw [stepWith_chatty_length H P x ω n]
      simp only [List.length_append, List.length_singleton]
      omega

/-- Against `chatty n`, every prover's transcript has exactly `n` rounds. -/
theorem runStar_chatty_length {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω) (n : ℕ) :
    (runStar H P (chatty n) x ω ()).msgs.length = n := by
  have h := stepWith_chatty_length H P x ω n [] [] ⟨[], [], [], .rej⟩
  unfold runStar
  simpa [chatty] using h

/-- The simulator's transcript against `chatty n` does not depend on `n`. -/
theorem simOut_chatty (H : Msg → Msg) (S : Simulator) (n : ℕ) (x : Msg) (c : S.C x) :
    S.out H (chatty n) () x c = S.out H (chatty 0) () x c := rfl

theorem Pr_le_one' {α : Type} [Fintype α] (p : α → Prop) : Pr p ≤ 1 := by
  classical
  unfold Pr
  rcases Nat.eq_zero_or_pos (Fintype.card α) with h | h
  · simp [h]
  · rw [div_le_one (by exact_mod_cast h)]
    exact_mod_cast Finset.card_filter_le _ _

theorem Pr_empty {α : Type} [Fintype α] [IsEmpty α] (p : α → Prop) : Pr p = 0 := by
  simp [Pr]

theorem Pr_true {α : Type} [Fintype α] [Nonempty α] : Pr (fun _ : α => True) = 1 := by
  classical
  have : (0 : ℝ) < Fintype.card α := by exact_mod_cast Fintype.card_pos
  rw [Pr_eq_sum (fun _ : α => True)]
  simp only [ite_true, Finset.sum_const, Finset.card_univ, nsmul_eq_mul, mul_one]
  field_simp

/-- Disjoint events `f c = n`, `n < M`, have total probability at most `1`. -/
theorem sum_Pr_eq_le_one {α : Type} [Fintype α] (f : α → ℕ) (M : ℕ) :
    ∑ n ∈ Finset.range M, Pr (fun c => f c = n) ≤ 1 := by
  classical
  simp only [Pr_eq_sum (fun c => f c = _)]
  rw [← Finset.sum_div, Finset.sum_comm]
  have h1 : ∀ c, (∑ n ∈ Finset.range M, if f c = n then (1 : ℝ) else 0) ≤ 1 := by
    intro c
    rw [Finset.sum_ite_eq]
    split_ifs <;> norm_num
  rcases Nat.eq_zero_or_pos (Fintype.card α) with h | h
  · simp [h]
  · rw [div_le_one (by exact_mod_cast h)]
    calc (∑ c, ∑ n ∈ Finset.range M, if f c = n then (1 : ℝ) else 0) ≤ ∑ _c : α, (1 : ℝ) :=
          Finset.sum_le_sum fun c _ => h1 c
      _ = Fintype.card α := by simp

/-- **`ZKmal` holds exactly at `εzk ≥ 1`** on the statements with a witness, for every prover, relation and
simulator: the simulator does not see the verifier's rounds. -/
theorem ZKmal_iff {W Ω : Type} [Fintype Ω] (H : Msg → Msg) (P : W → Prover Ω) (R : Rel W) (S : Simulator)
    (εzk : Msg → ℝ) : ZKmal H P R S εzk ↔ ∀ x w, R x w → 1 ≤ εzk x := by
  constructor
  · intro h x w hR
    rcases isEmpty_or_nonempty Ω with hΩ | hΩ
    · have e := h x w hR Unit (chatty 0) () (fun _ => True)
      rw [Pr_empty, Pr_true] at e
      simpa using e
    · by_contra hlt
      replace hlt := not_le.mp hlt
      have hp : ∀ n, 1 - εzk x ≤ Pr (fun c => (S.out H (chatty 0) () x c).length = n) := by
        intro n
        have e := h x w hR Unit (chatty n) () (fun t => t.length = n)
        simp only [runStar_chatty_length, simOut_chatty] at e
        rw [show (fun _ : Ω => True) = (fun _ : Ω => True) from rfl, Pr_true] at e
        have := (abs_sub_le_iff.mp e).1
        linarith
      obtain ⟨M, hM⟩ := exists_nat_gt (1 / (1 - εzk x))
      have hpos : 0 < 1 - εzk x := by linarith
      have hsum := sum_Pr_eq_le_one (fun c => (S.out H (chatty 0) () x c).length) M
      have hlow : (M : ℝ) * (1 - εzk x) ≤ ∑ n ∈ Finset.range M,
          Pr (fun c => (S.out H (chatty 0) () x c).length = n) := by
        calc (M : ℝ) * (1 - εzk x) = ∑ _n ∈ Finset.range M, (1 - εzk x) := by
              rw [Finset.sum_const, Finset.card_range, nsmul_eq_mul]
          _ ≤ _ := Finset.sum_le_sum fun n _ => hp n
      have : 1 < (M : ℝ) * (1 - εzk x) := by
        rw [div_lt_iff₀ hpos] at hM
        linarith
      linarith
  · intro h x w hR Θ Vs θ Ev
    refine le_trans ?_ (h x w hR)
    have a0 := Pr_nonneg (fun ω : Ω => Ev (runStar H (P w) Vs x ω θ).msgs)
    have a1 := Pr_le_one' (fun ω : Ω => Ev (runStar H (P w) Vs x ω θ).msgs)
    have b0 := Pr_nonneg (fun c : S.C x => Ev (S.out H Vs θ x c))
    have b1 := Pr_le_one' (fun c : S.C x => Ev (S.out H Vs θ x c))
    exact abs_sub_le_iff.mpr ⟨by linarith, by linarith⟩

/-- **`Flock.ZK` says nothing beyond `D.zk`**: it holds exactly when the deployed verifier is the masked one and
`εzk ≥ 1` on every statement with a witness. -/
theorem flockZK_iff {Coin Col T Ω : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
    [Fintype Ω] (H : Msg → Msg) (D : Flock.Deploy Coin Col T) (P : Flock.Wit → Prover Ω) (S : Simulator)
    (εzk : Msg → ℝ) :
    Flock.ZK H D P S εzk ↔ D.zk = true ∧ ∀ x w, Flock.R H D.parse x w → 1 ≤ εzk x := by
  unfold Flock.ZK
  rw [ZKmal_iff]

end Claims.ZKMal
