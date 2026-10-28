import Definitions.Accounting.Guarantees.Actual.Threaded
import Proofs.Accounting.Actual.Windows
import Proofs.Nci.Threads

/-!
# The memory cap over threads

`ThreadedThrottled`'s proof with the declared division replaced by the windows. `WindowsAt`, at the cuts `min(kω, t)`,
divides the developer's region over `[0, t)` into `n ⌈t/ω⌉` windows of at most `Φ_X ω ≤ Γ` gates, each reading in the
containment's image at most the state at its start plus `β_X ω`, so at most `K = R + r ω` (`windows_small`). Pulled back
through the containment (`Nci.SecurityProofs.pull`), the windows divide the reference circuit into pieces no larger whose
novel reads are at most their windows' reads in the image (`Nci.SecurityProofs.novelReads_pull_le_readsIn`, which needs
no condition on the baselines), each small at `(Γ, K)`, and Theorem 1 (`Nci.SecurityProofs.ThreadedReads`) bounds each
piece's counted multiplications by `K Λ(K) / b_a`; every counted multiplication lies in one piece (`card_M_eq_sum`).
Through a partial containment the windows divide only the gates the declaration computes (`Nci.SecurityProofs.piece`),
each reading novelly at most its window's reads in the image of those gates and the gates they read
(`Nci.SecurityProofs.novelReads_piece_le_readsIn`), and every other counted multiplication is a dark gate
(`ThreadedRuntimeDark`).

The runtime follows from `⌈t/ω⌉ ω ≤ t + ω` (`runtime_of_count`); the rate at `ω = R/r` from `(2R)² / ω = 4 R r`. Each is
proved per parameter slot, through Theorem 1 per slot (`Nci.SecurityProofs.ThreadedReadsSlots`), and the form over weight
gates is its instance at one slot per weight gate (`Nci.SecurityProofs.Fit.idSlots`).
-/

namespace Accounting.SecurityProofs.Actual

open Verity Nci Finset

/-- Every counted multiplication lies in exactly one unit of a division of the reference circuit. -/
theorem card_M_eq_sum {c : Registered} (T : Threading c) {m : ℕ} (Q : Partition c.D m) :
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

/-- **The windows are small**: `WindowsAt`'s division of the developer's declaration over `[0, t)` at the cuts
`min(kω, t)` into `n ⌈t/ω⌉` windows, each of at most `Γ` gates and reading in `G` at most `R + r ω`, its starting state
and what its links carry. -/
theorem windows_small {L E : Type*} [Fintype E] [DecidableEq L] (A : Placed L) (ℓ : L → L → ℕ) (Φ : Finset L → ℕ)
    (ends : E → L × L) (β : E → ℕ) (link : Fin A.N → Fin A.N → E) (n : ℕ) (X : Fin n → Finset L)
    (θ : Finset (Fin A.N)) (hθ : ∀ a ∈ θ, A.isInput a = true) (t ω Γ : ℕ) (R r : ℝ) (G : Finset (Fin A.N))
    (hω : 0 < ω) (hX : ∀ i j, i ≠ j → Disjoint (X i) (X j)) (hC : A.Causal ℓ)
    (hΦ : A.ComputeRate (Set.range X) (fun _ => 1) Φ) (hL : A.Links ends β link) (hΓ : ∀ i, Φ (X i) * ω ≤ Γ)
    (hr : ∀ i, (linkCapacity ends β (X i) : ℝ) ≤ r) (hR : ∀ i (k : ℕ), k * ω < t → (A.state (X i) (k * ω) : ℝ) ≤ R) :
    ∃ P : Partition (A.declaration (univ.biUnion X) t θ hθ).D (n * ((t + ω - 1) / ω)), ∀ u,
      (A.declaration (univ.biUnion X) t θ hθ).D.size (P.gates u) ≤ Γ ∧
        (readsIn (A.declaration (univ.biUnion X) t θ hθ) (P.gates u) G : ℝ) ≤ R + r * ω := by
  obtain ⟨P, hP⟩ := WindowsAt A ℓ Φ ends β link n X θ hθ t (uniformCuts t ω hω) G hX hC hΦ hL
  refine ⟨P, fun u => ?_⟩
  obtain ⟨⟨i, k⟩, rfl⟩ := finProdFinEquiv.surjective u
  obtain ⟨-, hsize, hreads⟩ := hP i k
  have hk : (k : ℕ) * ω < t := mul_lt_of_lt_ceil hω k.2
  have e : min ((k : ℕ) * ω) t = k * ω := min_eq_left hk.le
  have h1 : (A.stateIn (X i) (k * ω) G : ℝ) ≤ R :=
    (Nat.cast_le.2 (stateIn_le_state A _ _ G)).trans (hR i k hk)
  have h2 : ((linkCapacity ends β (X i) * (min (((k : ℕ) + 1) * ω) t - k * ω) : ℕ) : ℝ) ≤ r * ω := by
    rw [Nat.cast_mul]
    exact mul_le_mul (hr i) (Nat.cast_le.2 window_le) (Nat.cast_nonneg _) ((Nat.cast_nonneg _).trans (hr i))
  refine ⟨hsize.trans ?_, (Nat.cast_le.2 hreads).trans ?_⟩
  · show Φ (X i) * (min (((k : ℕ) + 1) * ω) t - min ((k : ℕ) * ω) t) ≤ Γ
    rw [e]
    exact (Nat.mul_le_mul_left _ window_le).trans (hΓ i)
  · show ((A.stateIn (X i) (min ((k : ℕ) * ω) t) G +
        linkCapacity ends β (X i) * (min (((k : ℕ) + 1) * ω) t - min ((k : ℕ) * ω) t) : ℕ) : ℝ) ≤ R + r * ω
    rw [e, Nat.cast_add]
    exact add_le_add h1 h2

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

theorem ThreadedRuntimeSlots : Guarantees.Actual.ThreadedRuntimeSlots := by
  intro L E _ _ A ℓ Φ ends β link n X θ hθ t ω Γ R r b_w b_a c T σ d e hω hX hC hΦ hL hbw hba hd hΓ hr hR hcont hT hW
  obtain ⟨M⟩ := hcont
  obtain ⟨P, hP⟩ := windows_small A ℓ Φ ends β link n X θ hθ t ω Γ R r (univ.map ⟨M.emb, M.emb_inj⟩) hω hX hC hΦ
    hL hΓ hr hR
  have hΛ : 0 ≤ T.bound (R + r * ω) b_w d e := le_max_of_le_left (Nat.cast_nonneg _)
  have hpiece : ∀ u, Small c Γ (R + r * ω) ((Nci.SecurityProofs.pull M P).gates u) := fun u =>
    ⟨Partition.gates_subset_computed _ u, (Nci.SecurityProofs.card_pull_gates_le (M := M) (P := P) u).trans (hP u).1,
      (Nat.cast_le.2 (Nci.SecurityProofs.novelReads_pull_le_readsIn M P u)).trans (hP u).2⟩
  have hcount : b_a * T.M.card ≤
      ((n * ((t + ω - 1) / ω) : ℕ) : ℝ) * ((R + r * ω) * T.bound (R + r * ω) b_w d e) :=
    calc b_a * (T.M.card : ℝ) = ∑ u, b_a * ((T.mults ((Nci.SecurityProofs.pull M P).gates u)).card : ℝ) := by
          rw [card_M_eq_sum T (Nci.SecurityProofs.pull M P), Nat.cast_sum, mul_sum]
      _ ≤ ∑ u, (novelReads c ((Nci.SecurityProofs.pull M P).gates u) : ℝ) * T.bound (R + r * ω) b_w d e :=
          sum_le_sum fun u _ =>
            Nci.SecurityProofs.ThreadedReadsSlots c T σ Γ (R + r * ω) b_w b_a d e hbw hba hd hT hW _ (hpiece u)
      _ ≤ ∑ _u : Fin (n * ((t + ω - 1) / ω)), (R + r * ω) * T.bound (R + r * ω) b_w d e :=
          sum_le_sum fun u _ => mul_le_mul_of_nonneg_right (hpiece u).2.2 hΛ
      _ = _ := by simp
  exact ⟨hcount, runtime_of_count (mul_nonneg hba.le (Nat.cast_nonneg _)) hcount⟩

/-- `ThreadedRuntimeSlots` at one slot per weight gate. -/
theorem ThreadedRuntime : Guarantees.Actual.ThreadedRuntime := by
  intro L E _ _ A ℓ Φ ends β link n X θ hθ t ω Γ R r b_w b_a c T d e hω hX hC hΦ hL hbw hba hd hΓ hr hR hcont hT hW
  exact ThreadedRuntimeSlots A ℓ Φ ends β link n X θ hθ t ω Γ R r b_w b_a c T (Nci.SecurityProofs.Fit.idSlots T) d e
    hω hX hC hΦ hL hbw hba hd hΓ hr hR hcont hT
    (Nci.SecurityProofs.weightsIncompressibleSlots_of (Nci.SecurityProofs.Fit.idSlots T) hbw.le hW)

theorem ThreadedRateSlots : Guarantees.Actual.ThreadedRateSlots := by
  intro L E _ _ A ℓ Φ ends β link n X θ hθ t ω p Γ R r b_w b_a c T σ d e hω hX hC hΦ hL hbw hba hd hΓ hr hwipe hpω
    hωt hopt hquad hcont hT hW
  have hstart : ∀ i (k : ℕ), k * ω < t → (A.state (X i) (k * ω) : ℝ) ≤ R := by
    intro i k hk
    obtain ⟨q, hq⟩ := hpω
    have hkq : k * ω = k * q * p := by rw [hq]; ring
    rw [hkq] at hk ⊢
    exact hwipe i (k * q) hk
  have h2R : R + r * ω = 2 * R := by linarith
  have hrun := ThreadedRuntimeSlots A ℓ Φ ends β link n X θ hθ t ω Γ R r b_w b_a c T σ d e hω hX hC hΦ hL hbw hba hd hΓ
    hr hstart hcont (by rw [h2R]; exact hT) (by rw [h2R]; exact hW)
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

/-- `ThreadedRateSlots` at one slot per weight gate. -/
theorem ThreadedRate : Guarantees.Actual.ThreadedRate := by
  intro L E _ _ A ℓ Φ ends β link n X θ hθ t ω p Γ R r b_w b_a c T d e hω hX hC hΦ hL hbw hba hd hΓ hr hwipe hpω hωt
    hopt hquad hcont hT hW
  exact ThreadedRateSlots A ℓ Φ ends β link n X θ hθ t ω p Γ R r b_w b_a c T (Nci.SecurityProofs.Fit.idSlots T) d e
    hω hX hC hΦ hL hbw hba hd hΓ hr hwipe hpω hωt hopt hquad hcont hT
    (Nci.SecurityProofs.weightsIncompressibleSlots_of (Nci.SecurityProofs.Fit.idSlots T) hbw.le hW)

theorem ThreadedRuntimeDarkSlots : Guarantees.Actual.ThreadedRuntimeDarkSlots := by
  intro L E _ _ A ℓ Φ ends β link n X θ hθ t ω Γ R r b_w b_a c T σ d e M hω hX hC hΦ hL hbw hba hd hΓ hr hR hT hW
  obtain ⟨P, hP⟩ := windows_small A ℓ Φ ends β link n X θ hθ t ω Γ R r ((M.S ∪ c.D.into M.S).image M.emb) hω hX hC
    hΦ hL hΓ hr hR
  have hΛ : 0 ≤ T.bound (R + r * ω) b_w d e := le_max_of_le_left (Nat.cast_nonneg _)
  have hpiece : ∀ u, Small c Γ (R + r * ω) (Nci.SecurityProofs.piece M P u) := fun u =>
    ⟨Nci.SecurityProofs.piece_subset_computed u, (Nci.SecurityProofs.card_piece_le u).trans (hP u).1,
      (Nat.cast_le.2 (Nci.SecurityProofs.novelReads_piece_le_readsIn M P u)).trans (hP u).2⟩
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
            Nci.SecurityProofs.ThreadedReadsSlots c T σ Γ (R + r * ω) b_w b_a d e hbw hba hd hT hW _ (hpiece u)
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

/-- `ThreadedRuntimeDarkSlots` at one slot per weight gate. -/
theorem ThreadedRuntimeDark : Guarantees.Actual.ThreadedRuntimeDark := by
  intro L E _ _ A ℓ Φ ends β link n X θ hθ t ω Γ R r b_w b_a c T d e M hω hX hC hΦ hL hbw hba hd hΓ hr hR hT hW
  exact ThreadedRuntimeDarkSlots A ℓ Φ ends β link n X θ hθ t ω Γ R r b_w b_a c T (Nci.SecurityProofs.Fit.idSlots T) d
    e M hω hX hC hΦ hL hbw hba hd hΓ hr hR hT
    (Nci.SecurityProofs.weightsIncompressibleSlots_of (Nci.SecurityProofs.Fit.idSlots T) hbw.le hW)

end Accounting.SecurityProofs.Actual
