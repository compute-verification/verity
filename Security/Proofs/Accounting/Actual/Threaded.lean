import Models.Accounting.Guarantees.Actual.Threaded
import Proofs.Accounting.Actual.Windows
import Proofs.Nci.Threads

/-!
# The memory cap over threads

`ThreadedThrottled`'s proof with the declared division replaced by the windows. `Windows` divides the developer's
region over `[0, t)` into `n ⌈t/ω⌉` windows of at most `Φ_X ω ≤ Γ` gates (its memory constraint holds at any circuit's
total width, so the bound on each window's starting state takes its place); a window reads at most its starting state
plus `β_X ω`, so at most `K = R + r ω` (`windows_small`). Pulled back through the containment
(`Nci.SecurityProofs.pull`), the windows divide the reference circuit into pieces no larger that read no more, each
small at `(Γ, K)`, and Theorem 1 (`Nci.SecurityProofs.ThreadedReads`) bounds each piece's counted multiplications by
`K Λ(K) / b_a`; every counted multiplication lies in one piece (`card_M_eq_sum`). Through a partial containment the
windows divide only the gates the declaration computes (`Nci.SecurityProofs.piece`), and every other counted
multiplication is a dark gate (`ThreadedRuntimeDark`).

The runtime follows from `⌈t/ω⌉ ω ≤ t + ω` (`runtime_of_count`); the rate at `ω = R/r` from `(2R)² / ω = 4 R r`.
-/

namespace Accounting.SecurityProofs.Actual

open Verity Nci Finset

/-- Every counted multiplication lies in exactly one unit of a division of the reference circuit. -/
theorem card_M_eq_sum {c : Declaration} (T : Threading c) {m : ℕ} (Q : Partition c.D m) :
    T.M.card = ∑ u, (T.mults (Q.gates u)).card := by
  classical
  rw [← card_biUnion fun u _ v _ huv => disjoint_left.2 fun g hu hv => huv (by
    rw [Threading.mults, mem_inter, Partition.mem_gates] at hu hv
    exact Option.some_injective _ (hu.2.symm.trans hv.2))]
  congr 1
  ext g
  simp only [mem_biUnion, mem_univ, true_and, Threading.mults, mem_inter]
  constructor
  · intro hg
    have hgc : c.D.isInput g = false := (mem_filter.1 (T.M_computed hg)).2
    cases hu : Q.unit g with
    | none =>
      rw [Q.input_iff, hgc] at hu
      exact absurd hu Bool.false_ne_true
    | some u => exact ⟨u, hg, Q.mem_gates.2 hu⟩
  · rintro ⟨_, hg, _⟩
    exact hg

/-- A window below `⌈t/ω⌉` starts before `t`. -/
theorem mul_lt_of_lt_ceil {k t ω : ℕ} (hω : 0 < ω) (hk : k < (t + ω - 1) / ω) : k * ω < t := by
  have h := (Nat.le_div_iff_mul_le hω).1 (Nat.lt_iff_add_one_le.1 hk)
  rw [add_one_mul] at h
  omega

/-- **The windows are small**: `Windows`' division of the developer's declaration over `[0, t)` into `n ⌈t/ω⌉`
windows, each of at most `Γ` gates and reading at most `R + r ω`, its starting state and what its links carry. -/
theorem windows_small {L E : Type*} [Fintype E] [DecidableEq L] (A : Placed L) (ℓ : L → L → ℕ) (Φ : Finset L → ℕ)
    (ends : E → L × L) (β : E → ℕ) (link : Fin A.N → Fin A.N → E) (n : ℕ) (X : Fin n → Finset L)
    (θ : Finset (Fin A.N)) (hθ : ∀ a ∈ θ, A.isInput a = true) (t ω Γ : ℕ) (R r : ℝ) (hω : 0 < ω)
    (hX : ∀ i j, i ≠ j → Disjoint (X i) (X j)) (hC : A.Causal ℓ) (hΦ : A.ComputeRate (Set.range X) (fun _ => 1) Φ)
    (hL : A.Links ends β link) (hΓ : ∀ i, Φ (X i) * ω ≤ Γ) (hr : ∀ i, (linkCapacity ends β (X i) : ℝ) ≤ r)
    (hR : ∀ i (k : ℕ), k * ω < t → (A.state (X i) (k * ω) : ℝ) ≤ R) :
    ∃ P : Partition (A.declaration (univ.biUnion X) t θ hθ).D (n * ((t + ω - 1) / ω)), ∀ u,
      (A.declaration (univ.biUnion X) t θ hθ).D.size (P.gates u) ≤ Γ ∧
        (novelReads (A.declaration (univ.biUnion X) t θ hθ) (P.gates u) : ℝ) ≤ R + r * ω := by
  have hMem : A.Memory (Set.range X) (fun _ => A.width univ) := fun _ _ _ => width_mono A (subset_univ _)
  obtain ⟨P, hP⟩ := Windows A ℓ Φ (fun _ => A.width univ) ends β link n X θ hθ t ω hω hX hC hΦ hMem hL
  refine ⟨P, fun u => ?_⟩
  obtain ⟨⟨i, k⟩, rfl⟩ := finProdFinEquiv.surjective u
  refine ⟨(hP i k).2.1.trans (hΓ i), ?_⟩
  have hreads : novelReads (A.declaration (univ.biUnion X) t θ hθ) (P.gates (finProdFinEquiv (i, k))) ≤
      A.state (X i) (k * ω) + linkCapacity ends β (X i) * ω := by
    rw [(hP i k).1, Set.Ico_inter_Iio]
    have hRS : A.region (X i) (Set.Ico (k * ω) (min ((k + 1) * ω) t)) ⊆
        A.region (univ.biUnion X) (Set.Iio t) :=
      region_subset A (subset_biUnion_of_mem X (mem_univ i)) fun s hs => by
        simp only [Set.mem_Ico, lt_min_iff] at hs
        exact Set.mem_Iio.2 hs.2.2
    unfold novelReads
    change A.width ((A.cut (A.region (univ.biUnion X) (Set.Iio t))).into
      (A.region (X i) (Set.Ico (k * ω) (min ((k + 1) * ω) t))) \ θ) ≤ _
    rw [cut_into A hRS]
    exact (width_mono A sdiff_subset).trans ((width_into_region_le A _ _ _ (forward_of_causal A hC _)).trans
      (Nat.add_le_add le_rfl ((ingress_le A hL _ _ _).trans (Nat.mul_le_mul_left _ window_le))))
  refine (Nat.cast_le.2 hreads).trans ?_
  push_cast
  exact add_le_add (hR i k (mul_lt_of_lt_ceil hω k.2)) (mul_le_mul_of_nonneg_right (hr i) (Nat.cast_nonneg _))

/-- The runtime from the count: `b_a x ≤ n ⌈t/ω⌉ · KΛ` gives `t ≥ ω (b_a x / (n KΛ) − 1)`, since `⌈t/ω⌉ ω ≤ t + ω`. -/
theorem runtime_of_count {n t ω : ℕ} {b_a x KΛ : ℝ} (hx : 0 ≤ b_a * x)
    (h : b_a * x ≤ ((n * ((t + ω - 1) / ω) : ℕ) : ℝ) * KΛ) : (ω : ℝ) * (b_a * x / (n * KΛ) - 1) ≤ t := by
  have hm : (((t + ω - 1) / ω : ℕ) : ℝ) * ω ≤ t + ω := by
    exact_mod_cast (Nat.div_mul_le_self (t + ω - 1) ω).trans (Nat.sub_le (t + ω) 1)
  have hdiv : b_a * x / (n * KΛ) ≤ (((t + ω - 1) / ω : ℕ) : ℝ) := by
    rcases le_or_gt ((n : ℝ) * KΛ) 0 with h' | h'
    · exact (div_nonpos_of_nonneg_of_nonpos hx h').trans (Nat.cast_nonneg _)
    · rw [div_le_iff₀ h']
      calc b_a * x ≤ ((n * ((t + ω - 1) / ω) : ℕ) : ℝ) * KΛ := h
        _ = (((t + ω - 1) / ω : ℕ) : ℝ) * (n * KΛ) := by push_cast; ring
  calc (ω : ℝ) * (b_a * x / (n * KΛ) - 1) ≤ ω * ((((t + ω - 1) / ω : ℕ) : ℝ) - 1) :=
        mul_le_mul_of_nonneg_left (by linarith) (Nat.cast_nonneg _)
    _ ≤ t := by linarith

theorem ThreadedRuntime : Guarantees.Actual.ThreadedRuntime := by
  intro L E _ _ A ℓ Φ ends β link n X θ hθ t ω Γ R r b_w b_a c T d e hω hX hC hΦ hL hbw hba hd hΓ hr hR hcont hT hW
  obtain ⟨M, hθM⟩ := hcont
  obtain ⟨P, hP⟩ := windows_small A ℓ Φ ends β link n X θ hθ t ω Γ R r hω hX hC hΦ hL hΓ hr hR
  have hΛ : 0 ≤ T.bound (R + r * ω) b_w d e := le_max_of_le_left (Nat.cast_nonneg _)
  have hpiece : ∀ u, Small c Γ (R + r * ω) ((Nci.SecurityProofs.pull M P).gates u) := fun u =>
    ⟨Partition.gates_subset_computed _ u, (Nci.SecurityProofs.card_pull_gates_le (M := M) (P := P) u).trans (hP u).1,
      (Nat.cast_le.2 (Nci.SecurityProofs.novelReads_pull M hθM P u)).trans (hP u).2⟩
  have hcount : b_a * T.M.card ≤
      ((n * ((t + ω - 1) / ω) : ℕ) : ℝ) * ((R + r * ω) * T.bound (R + r * ω) b_w d e) :=
    calc b_a * (T.M.card : ℝ) = ∑ u, b_a * ((T.mults ((Nci.SecurityProofs.pull M P).gates u)).card : ℝ) := by
          rw [card_M_eq_sum T (Nci.SecurityProofs.pull M P), Nat.cast_sum, mul_sum]
      _ ≤ ∑ u, (novelReads c ((Nci.SecurityProofs.pull M P).gates u) : ℝ) * T.bound (R + r * ω) b_w d e :=
          sum_le_sum fun u _ =>
            Nci.SecurityProofs.ThreadedReads c T Γ (R + r * ω) b_w b_a d e hbw hba hd hT hW _ (hpiece u)
      _ ≤ ∑ _u : Fin (n * ((t + ω - 1) / ω)), (R + r * ω) * T.bound (R + r * ω) b_w d e :=
          sum_le_sum fun u _ => mul_le_mul_of_nonneg_right (hpiece u).2.2 hΛ
      _ = _ := by simp
  exact ⟨hcount, runtime_of_count (mul_nonneg hba.le (Nat.cast_nonneg _)) hcount⟩

theorem ThreadedRate : Guarantees.Actual.ThreadedRate := by
  intro L E _ _ A ℓ Φ ends β link n X θ hθ t ω p Γ R r b_w b_a c T d e hω hX hC hΦ hL hbw hba hd hΓ hr hwipe hpω hωt
    hopt hquad hcont hT hW
  have hstart : ∀ i (k : ℕ), k * ω < t → (A.state (X i) (k * ω) : ℝ) ≤ R := by
    intro i k hk
    obtain ⟨q, hq⟩ := hpω
    have hkq : k * ω = k * q * p := by rw [hq]; ring
    rw [hkq] at hk ⊢
    exact hwipe i (k * q) hk
  have h2R : R + r * ω = 2 * R := by linarith
  have hrun := ThreadedRuntime A ℓ Φ ends β link n X θ hθ t ω Γ R r b_w b_a c T d e hω hX hC hΦ hL hbw hba hd hΓ hr
    hstart hcont (by rw [h2R]; exact hT) (by rw [h2R]; exact hW)
  have hΛ : T.bound (2 * R) b_w d e = T.reuse * (2 * R) / (4 * d * b_w) := by
    rw [Threading.bound]
    exact max_eq_right hquad
  have hmq : (t + ω - 1) / ω = t / ω := by
    obtain ⟨q, hq⟩ := hωt
    rw [hq, Nat.mul_div_cancel_left q hω, div_eq_iff_mem hω, Set.mem_Ico, add_one_mul, Nat.mul_comm q ω]
    omega
  have ht : (t : ℝ) = (((t + ω - 1) / ω : ℕ) : ℝ) * ω := by
    rw [hmq]
    exact_mod_cast (Nat.div_mul_cancel hωt).symm
  have hdb : (0 : ℝ) < 4 * d * b_w := by
    have : (0 : ℝ) < d := Nat.cast_pos.2 hd
    positivity
  obtain ⟨h1, -⟩ := hrun
  rw [h2R, hΛ] at h1
  have h4 : b_a * T.M.card * (4 * d * b_w) ≤
      ((n * ((t + ω - 1) / ω) : ℕ) : ℝ) * (2 * R) * T.reuse * (2 * R) := by
    rw [← le_div_iff₀ hdb]
    calc b_a * (T.M.card : ℝ)
        ≤ ((n * ((t + ω - 1) / ω) : ℕ) : ℝ) * (2 * R * (T.reuse * (2 * R) / (4 * d * b_w))) := h1
      _ = ((n * ((t + ω - 1) / ω) : ℕ) : ℝ) * (2 * R) * T.reuse * (2 * R) / (4 * d * b_w) := by ring
  have hrate : b_a * b_w * d * T.M.card ≤ T.reuse * n * R * r * t := by
    rw [ht]
    subst hopt
    push_cast at h4 ⊢
    linarith
  refine ⟨hrate, fun κ hκ => ?_⟩
  have hbbd : 0 ≤ b_a * b_w * d := mul_nonneg (mul_nonneg hba.le hbw.le) (Nat.cast_nonneg _)
  calc κ * b_a * b_w * d * c.D.size c.D.computed = (b_a * b_w * d) * (κ * c.D.size c.D.computed) := by ring
    _ ≤ (b_a * b_w * d) * (2 * T.M.card) := mul_le_mul_of_nonneg_left hκ hbbd
    _ = 2 * (b_a * b_w * d * T.M.card) := by ring
    _ ≤ 2 * (T.reuse * n * R * r * t) := by linarith
    _ = 2 * T.reuse * n * R * r * t := by ring

theorem ThreadedRuntimeDark : Guarantees.Actual.ThreadedRuntimeDark := by
  intro L E _ _ A ℓ Φ ends β link n X θ hθ t ω Γ R r b_w b_a c T d e M hω hX hC hΦ hL hbw hba hd hΓ hr hR hT hW
  obtain ⟨P, hP⟩ := windows_small A ℓ Φ ends β link n X θ hθ t ω Γ R r hω hX hC hΦ hL hΓ hr hR
  have hΛ : 0 ≤ T.bound (R + r * ω) b_w d e := le_max_of_le_left (Nat.cast_nonneg _)
  have hpiece : ∀ u, Small c Γ (R + r * ω) (Nci.SecurityProofs.piece M P u) := fun u =>
    ⟨Nci.SecurityProofs.piece_subset_computed u, (Nci.SecurityProofs.card_piece_le u).trans (hP u).1,
      (Nat.cast_le.2 (Nci.SecurityProofs.novelReads_piece u)).trans (hP u).2⟩
  have hM : (T.M ∩ M.S).card = ∑ u, (T.mults (Nci.SecurityProofs.piece M P u)).card := by
    have h := Nci.SecurityProofs.sum_card_filter_piece (M := M) (P := P) T.M fun g => g
    simp only [filter_mem_eq_inter] at h
    exact h.symm
  have hcount : b_a * (T.M ∩ M.S).card ≤
      ((n * ((t + ω - 1) / ω) : ℕ) : ℝ) * ((R + r * ω) * T.bound (R + r * ω) b_w d e) :=
    calc b_a * ((T.M ∩ M.S).card : ℝ) = ∑ u, b_a * ((T.mults (Nci.SecurityProofs.piece M P u)).card : ℝ) := by
          rw [hM, Nat.cast_sum, mul_sum]
      _ ≤ ∑ u, (novelReads c (Nci.SecurityProofs.piece M P u) : ℝ) * T.bound (R + r * ω) b_w d e :=
          sum_le_sum fun u _ =>
            Nci.SecurityProofs.ThreadedReads c T Γ (R + r * ω) b_w b_a d e hbw hba hd hT hW _ (hpiece u)
      _ ≤ ∑ _u : Fin (n * ((t + ω - 1) / ω)), (R + r * ω) * T.bound (R + r * ω) b_w d e :=
          sum_le_sum fun u _ => mul_le_mul_of_nonneg_right (hpiece u).2.2 hΛ
      _ = _ := by simp
  refine ⟨hcount, runtime_of_count (mul_nonneg hba.le (Nat.cast_nonneg _)) hcount, ?_⟩
  have hS : ((T.M ∩ M.S).card : ℝ) ≤
      ((n * ((t + ω - 1) / ω) : ℕ) : ℝ) * ((R + r * ω) * T.bound (R + r * ω) b_w d e) / b_a := by
    rw [le_div_iff₀ hba]
    linarith
  have hs : (T.M.card : ℝ) ≤ M.dark.card + (T.M ∩ M.S).card := by
    exact_mod_cast Nci.SecurityProofs.card_le_dark_add_inter M T.M_computed
  linarith

end Accounting.SecurityProofs.Actual
