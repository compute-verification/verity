module

public import Proofs.Flock.Claims.KS.Fork
public import Proofs.Flock.Claims.Recursion.Abort

/-! The finders' escape events at runs that may abort, covered by `BAbort` and bounded by one square root
(`EscapesD.le`). -/

@[expose] public section

namespace Claims.Recursion

open Claims

def forkSample {Coin A : Type} {n : ℕ} (j : Fin n) (d d' : (Fin n → Coin) × A) : (Fin n → Coin) × A :=
  (forkCoins j d.1 d'.1, d'.2)

structure EscapesD {Ω Coin A : Type} (H : Msg → Msg) (n : ℕ) (tr : Ω → (Fin n → Coin) × A → Transcript) where
  I : Type
  [finI : Fintype I]
  c : I → ℝ
  j : I → Fin n
  ev : I → Ω → (Fin n → Coin) × A → (Fin n → Coin) × A → Prop
  L : Type
  [finL : Fintype L]
  β : L → ℝ
  ev₁ : L → Ω → (Fin n → Coin) × A → Prop
  c_nonneg : ∀ i, 0 ≤ c i
  β_nonneg : ∀ l, 0 ≤ β l
  tied : ∀ i ω d d', ev i ω d d' → TiedColl H ((tr ω d).vLog ++ (tr ω (forkSample (j i) d d')).vLog)
  tied₁ : ∀ l ω d, ev₁ l ω d → TiedColl H (tr ω d).vLog

attribute [instance] EscapesD.finI EscapesD.finL

section EscapesD

variable {Ω Coin A : Type} {H : Msg → Msg} {n : ℕ} {tr : Ω → (Fin n → Coin) × A → Transcript}

noncomputable def EscapesD.term [Fintype Coin] [Fintype A] (e : EscapesD H n tr) (ω : Ω) : ℝ :=
  (∑ i, Real.sqrt (e.c i * Pr fun s : ((Fin n → Coin) × A) × ((Fin n → Coin) × A) => e.ev i ω s.1 s.2)) +
    ∑ l, e.β l * Pr fun d : (Fin n → Coin) × A => e.ev₁ l ω d

def EscapesD.Fits (e : EscapesD H n tr) (C : ℝ) : Prop :=
  ∃ mult : ℕ, (∀ r, (Finset.univ.filter fun i => e.j i = r).card ≤ mult) ∧ mult * ∑ i, e.c i ≤ 2 * C ∧
    ∑ l, e.β l ≤ n

theorem EscapesD.term_nonneg [Fintype Coin] [Fintype A] (e : EscapesD H n tr) (ω : Ω) : 0 ≤ e.term ω :=
  add_nonneg (Finset.sum_nonneg fun _ _ => Real.sqrt_nonneg _)
    (Finset.sum_nonneg fun l _ => mul_nonneg (e.β_nonneg l) (Pr_nonneg _))

def EscapesD.Kept (e : EscapesD H n tr) (keep : Ω → (Fin n → Coin) → A → Bool) : Prop :=
  (∀ i ω d d', e.ev i ω d d' → keep ω d.1 d.2 = true ∧
    keep ω (forkSample (e.j i) d d').1 (forkSample (e.j i) d d').2 = true) ∧
  ∀ l ω d, e.ev₁ l ω d → keep ω d.1 d.2 = true

end EscapesD

section Tie

variable {Ω Coin A : Type} [Inhabited Coin] {H : Msg → Msg} {V : Verifier Coin} {P : Prover Ω} {x : Msg}
  {keep : Ω → (Fin (V.k x) → Coin) → A → Bool}

theorem abortRun_of_keep {ω : Ω} {t : (Fin (V.k x) → Coin) × A} (h : keep ω t.1 t.2 = true) :
    abortRun H P V x keep ω t = run H P V x ω t.1 :=
  ite_eq_left h

def EscapesD.toAbort (e : EscapesD H (V.k x) fun ω d => run H P V x ω d.1) (hk : e.Kept keep) :
    EscapesD H (V.k x) (abortRun H P V x keep) where
  I := e.I
  c := e.c
  j := e.j
  ev := e.ev
  L := e.L
  β := e.β
  ev₁ := e.ev₁
  c_nonneg := e.c_nonneg
  β_nonneg := e.β_nonneg
  tied i ω d d' h := by
    rw [abortRun_of_keep (hk.1 i ω d d' h).1, abortRun_of_keep (hk.1 i ω d d' h).2]
    exact e.tied i ω d d' h
  tied₁ l ω d h := by
    rw [abortRun_of_keep (hk.2 l ω d h)]
    exact e.tied₁ l ω d h

theorem EscapesD.term_toAbort [Fintype Coin] [Fintype A] (e : EscapesD H (V.k x) fun ω d => run H P V x ω d.1)
    (hk : e.Kept keep) : (e.toAbort hk).term = e.term := rfl

end Tie

section BAbort

variable {Ω Coin A : Type} [Fintype Ω] [Nonempty Ω] [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [Fintype A]
  [Nonempty A]

omit [Fintype Ω] [Nonempty Ω] [Fintype Coin] [Nonempty Coin] [Fintype A] [Nonempty A] in
theorem abortRun_vLog (H : Msg → Msg) (P : Prover Ω) (V : Verifier Coin) (x : Msg)
    (keep : Ω → (Fin (V.k x) → Coin) → A → Bool) (ω : Ω) (t : (Fin (V.k x) → Coin) × A) :
    ∀ p ∈ (abortRun H P V x keep ω t).vLog, H p.1 = p.2 := by
  unfold abortRun
  split
  · exact (run_logs H P V x ω t.1).2
  · simp [abortTranscript]

omit [Fintype Ω] [Nonempty Ω] [Fintype Coin] [Nonempty Coin] [Fintype A] [Nonempty A] in
theorem abortRun_vLog_of_k_zero (H : Msg → Msg) (P : Prover Ω) (V : Verifier Coin) (x : Msg)
    (keep : Ω → (Fin (V.k x) → Coin) → A → Bool) (h : V.k x = 0) (ω : Ω) (t : (Fin (V.k x) → Coin) × A) :
    (abortRun H P V x keep ω t).vLog = [] := by
  unfold abortRun
  split
  · exact run_vLog_of_k_zero H P V x h ω t.1
  · rfl

theorem PrBAbort_nonneg (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω) (x : Msg)
    (keep : Ω → (Fin (V.k x) → Coin) → A → Bool) : 0 ≤ PrBAbort H V P x keep := by
  unfold PrBAbort
  split_ifs
  · exact Pr_nonneg _
  · exact le_rfl

theorem PrBAbort_le_one (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω) (x : Msg)
    (keep : Ω → (Fin (V.k x) → Coin) → A → Bool) : PrBAbort H V P x keep ≤ 1 := by
  unfold PrBAbort
  split_ifs
  · exact Pr_le_one _
  · exact zero_le_one

noncomputable def forkPrA (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω) (x : Msg)
    (keep : Ω → (Fin (V.k x) → Coin) → A → Bool) (hk : 0 < V.k x) (j : Fin (V.k x)) : ℝ :=
  Pr fun s : Ω × ((Fin (V.k x) → Coin) × A) × ((Fin (V.k x) → Coin) × A) => ∃ u v,
    (BAbort H V P x keep hk).out ((j, s.1, s.2.1.1, s.2.2.1), s.2.1.2, s.2.2.2) = some (u, v) ∧ Collision H u v

theorem PrBAbort_eq_avg (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω) (x : Msg)
    (keep : Ω → (Fin (V.k x) → Coin) → A → Bool) (hk : 0 < V.k x) :
    PrBAbort H V P x keep = (∑ j, forkPrA H V P x keep hk j) / V.k x := by
  simp only [PrBAbort, hk, ↓reduceDIte]
  unfold Finder.Pr
  let e : Fin (V.k x) × (Ω × ((Fin (V.k x) → Coin) × A) × ((Fin (V.k x) → Coin) × A)) ≃
      (BAbort H V P x keep hk).S :=
    { toFun := fun t => ((t.1, t.2.1, t.2.2.1.1, t.2.2.2.1), t.2.2.1.2, t.2.2.2.2)
      invFun := fun s => (s.1.1, s.1.2.1, (s.1.2.2.1, s.2.1), (s.1.2.2.2, s.2.2))
      left_inv := fun _ => rfl
      right_inv := fun _ => rfl }
  rw [← Pr_equiv e]
  exact Pr_guess fun t => ∃ u v, (BAbort H V P x keep hk).out (e t) = some (u, v) ∧ Collision H u v

theorem Pr_fork_le_abort (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω) (x : Msg)
    (keep : Ω → (Fin (V.k x) → Coin) → A → Bool) (hk : 0 < V.k x) (j : Fin (V.k x))
    (ev : Ω → (Fin (V.k x) → Coin) × A → (Fin (V.k x) → Coin) × A → Prop)
    (hev : ∀ ω d d', ev ω d d' →
      TiedColl H ((abortRun H P V x keep ω d).vLog ++ (abortRun H P V x keep ω (forkSample j d d')).vLog)) :
    Pr (fun s : Ω × ((Fin (V.k x) → Coin) × A) × ((Fin (V.k x) → Coin) × A) => ev s.1 s.2.1 s.2.2) ≤
      forkPrA H V P x keep hk j := by
  unfold forkPrA
  refine Pr_mono fun s hs => ?_
  have hl : ∀ p ∈ (abortRun H P V x keep s.1 s.2.1).vLog ++
      (abortRun H P V x keep s.1 (forkSample j s.2.1 s.2.2)).vLog, H p.1 = p.2 := by
    intro p hp
    rcases List.mem_append.1 hp with hp | hp
    · exact abortRun_vLog H P V x keep _ _ p hp
    · exact abortRun_vLog H P V x keep _ _ p hp
  obtain ⟨⟨u, v⟩, huv⟩ := Option.isSome_iff_exists.1 (findColl_isSome H hl (hev _ _ _ hs))
  exact ⟨u, v, huv, findColl_spec H _ hl u v huv⟩

theorem Pr_one_le_PrBAbort (H : Msg → Msg) (V : Verifier Coin) (P : Prover Ω) (x : Msg)
    (keep : Ω → (Fin (V.k x) → Coin) → A → Bool) (ev₁ : Ω → (Fin (V.k x) → Coin) × A → Prop)
    (hev : ∀ ω d, ev₁ ω d → TiedColl H (abortRun H P V x keep ω d).vLog) :
    Pr (fun s : Ω × ((Fin (V.k x) → Coin) × A) => ev₁ s.1 s.2) ≤ PrBAbort H V P x keep := by
  by_cases hk : 0 < V.k x
  · have hk' : (0 : ℝ) < V.k x := by exact_mod_cast hk
    rw [PrBAbort_eq_avg H V P x keep hk, le_div_iff₀ hk']
    have hj : ∀ j, Pr (fun s : Ω × ((Fin (V.k x) → Coin) × A) => ev₁ s.1 s.2) ≤ forkPrA H V P x keep hk j := by
      intro j
      rw [← Pr_drop (γ := (Fin (V.k x) → Coin) × A) fun ω d => ev₁ ω d]
      exact Pr_fork_le_abort H V P x keep hk j (fun ω d _ => ev₁ ω d) fun ω d d' h => (hev ω d h).append_right _
    calc Pr (fun s : Ω × ((Fin (V.k x) → Coin) × A) => ev₁ s.1 s.2) * V.k x
        = ∑ _j : Fin (V.k x), Pr (fun s : Ω × ((Fin (V.k x) → Coin) × A) => ev₁ s.1 s.2) := by
          rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul, mul_comm]
      _ ≤ ∑ j, forkPrA H V P x keep hk j := Finset.sum_le_sum fun j _ => hj j
  · have h0 : V.k x = 0 := by omega
    rw [Pr_congr (q := fun _ => False) fun s => ⟨fun h => ?_, False.elim⟩, Pr_false]
    · exact PrBAbort_nonneg H V P x keep
    obtain ⟨u, v, _, hu, _⟩ := hev s.1 s.2 h
    rw [abortRun_vLog_of_k_zero H P V x keep h0] at hu
    simp at hu

theorem EscapesD.le {H : Msg → Msg} {V : Verifier Coin} {P : Prover Ω} {x : Msg}
    {keep : Ω → (Fin (V.k x) → Coin) → A → Bool} (e : EscapesD H (V.k x) (abortRun H P V x keep)) (C : ℝ)
    (hfit : e.Fits C) :
    4 * Ex e.term ≤ Real.sqrt (16 * (Real.sqrt (2 * V.k x * C) + V.k x) ^ 2 * PrBAbort H V P x keep) := by
  obtain ⟨mult, hm, hmc, hβ⟩ := hfit
  have hc0 : 0 ≤ ∑ i, e.c i := Finset.sum_nonneg fun i _ => e.c_nonneg i
  have hC : 0 ≤ 2 * C := (mul_nonneg (Nat.cast_nonneg mult) hc0).trans hmc
  have hp0 := PrBAbort_nonneg H V P x keep
  have hp1 := PrBAbort_le_one H V P x keep
  have hkx : (0 : ℝ) ≤ V.k x := Nat.cast_nonneg _
  set a₂ : e.I → ℝ := fun i =>
    Pr fun s : Ω × ((Fin (V.k x) → Coin) × A) × ((Fin (V.k x) → Coin) × A) => e.ev i s.1 s.2.1 s.2.2 with ha₂
  set a₁ : e.L → ℝ := fun l => Pr fun s : Ω × ((Fin (V.k x) → Coin) × A) => e.ev₁ l s.1 s.2 with ha₁
  have hEx : Ex e.term ≤ (∑ i, Real.sqrt (e.c i * a₂ i)) + ∑ l, e.β l * a₁ l := by
    unfold EscapesD.term
    rw [Ex_add, Ex_sum, Ex_sum]
    refine add_le_add (Finset.sum_le_sum fun i _ => ?_) (le_of_eq (Finset.sum_congr rfl fun l _ => ?_))
    · refine (Ex_sqrt_le _ fun ω => mul_nonneg (e.c_nonneg i) (Pr_nonneg _)).trans (le_of_eq ?_)
      rw [Ex_mul_left, Ex_Pr]
    · rw [Ex_mul_left, Ex_Pr]
  have hfork : (∑ i, Real.sqrt (e.c i * a₂ i)) ≤ Real.sqrt (2 * V.k x * C * PrBAbort H V P x keep) := by
    by_cases hk : 0 < V.k x
    · have hcol := collapseS Finset.univ hk e.c a₂ (fun i _ => e.c_nonneg i) (fun i _ => Pr_nonneg _) e.j
        (forkPrA H V P x keep hk) (fun _ => Pr_nonneg _)
        (fun i _ => Pr_fork_le_abort H V P x keep hk (e.j i) (e.ev i) (e.tied i)) mult fun r => hm r
      rw [← PrBAbort_eq_avg H V P x keep hk] at hcol
      refine hcol.trans (Real.sqrt_le_sqrt ?_)
      have := mul_le_mul_of_nonneg_left hmc (mul_nonneg hkx hp0)
      nlinarith [this]
    · have : IsEmpty e.I := ⟨fun i => absurd (e.j i).pos hk⟩
      simp only [Finset.univ_eq_empty, Finset.sum_empty]
      exact Real.sqrt_nonneg _
  have hlin : ∑ l, e.β l * a₁ l ≤ V.k x * PrBAbort H V P x keep := by
    refine (Finset.sum_le_sum fun l _ =>
      mul_le_mul_of_nonneg_left (Pr_one_le_PrBAbort H V P x keep (e.ev₁ l) (e.tied₁ l)) (e.β_nonneg l)).trans ?_
    rw [← Finset.sum_mul]
    exact mul_le_mul_of_nonneg_right hβ hp0
  have hroot := sqrt_add_lin (C := 2 * V.k x * C) (β := V.k x) (by nlinarith) hkx hp0 hp1
  have h16 : Real.sqrt (16 * (Real.sqrt (2 * V.k x * C) + V.k x) ^ 2 * PrBAbort H V P x keep) =
      4 * Real.sqrt ((Real.sqrt (2 * V.k x * C) + V.k x) ^ 2 * PrBAbort H V P x keep) := by
    rw [mul_assoc, Real.sqrt_mul (by norm_num : (0 : ℝ) ≤ 16), show (16 : ℝ) = 4 ^ 2 by norm_num,
      Real.sqrt_sq (by norm_num)]
  rw [h16]
  linarith

end BAbort

end Claims.Recursion
