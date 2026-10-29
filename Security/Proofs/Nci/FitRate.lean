import Definitions.Nci.Guarantees.FitRate
import Proofs.Accounting.Actual.Windows
import Proofs.Nci.Fit

/-!
# Proof of the fit bound on a run

`WindowsAt` at the run's cuts divides the developer's declaration into the windows `A[X i, [τ j, τ (j + 1))]`, each
reading in `G` at most the part in `G` of its server's state at `τ j` plus `β_{X i} (τ (j + 1) − τ j)`. At `G` the image
of `c`'s novel gates, the windows pulled into `c` (`pull`) read at most that (`novelReads_pull_le_readsIn_compl`), so at
most `R + r ω`, and `FitBound` gives the count over the `n J` windows. Its compute-rate hypothesis holds at `Φ = A.N`
(`computeRate_N`), and the count does not read `Φ`. Windows of at least `ω₀` ticks number at most `t / ω₀`
(`cuts_mul_le`), which gives the rate.
-/

namespace Nci.SecurityProofs

open Verity Accounting Finset

/-- Through a containment `M : c ≼ d`, the piece of `c` that a unit of `d` induces reads at most what the unit reads in
the image of `c`'s novel gates. -/
theorem novelReads_pull_le_readsIn_compl {c d : Declaration} (M : Containment c.D d.D) {n : ℕ} (P : Partition d.D n)
    (u : Fin n) : novelReads c ((pull M P).gates u) ≤ readsIn d (P.gates u) ((c.θᶜ).map ⟨M.emb, M.emb_inj⟩) := by
  unfold novelReads readsIn
  rw [width_map (M := M)]
  refine sum_le_sum_of_subset fun b hb => ?_
  obtain ⟨a, ha, rfl⟩ := mem_map.1 hb
  exact mem_inter.2 ⟨into_pull_map u (mem_map_of_mem _ (mem_sdiff.1 ha).1),
    mem_map_of_mem _ (Finset.mem_compl.2 (mem_sdiff.1 ha).2)⟩

/-- Every actual circuit does at most `A.N` gates a tick at any location, so the compute rate holds at `Φ = A.N`. -/
theorem computeRate_N {L : Type*} (A : Placed L) (𝒳 : Set (Finset L)) :
    A.ComputeRate 𝒳 (fun _ => 1) (fun _ => A.N) := by
  intro X _ a b
  show A.work (fun _ => 1) (A.region X (Set.Ico a b)) ≤ A.N * (b - a)
  simp only [Placed.work, sum_const, smul_eq_mul, mul_one]
  rcases Nat.lt_or_ge a b with hab | hab
  · exact ((card_le_univ _).trans (Fintype.card_fin A.N).le).trans
      (Nat.le_mul_of_pos_right _ (Nat.sub_pos_of_lt hab))
  · have hempty : A.region X (Set.Ico a b) = ∅ :=
      eq_empty_of_forall_notMem fun g hg => by
        obtain ⟨-, -, ht⟩ := (Accounting.SecurityProofs.Actual.mem_region A).1 hg
        have := Set.mem_Ico.1 ht
        omega
    rw [hempty, card_empty]
    exact Nat.zero_le _

/-- Cuts of `[0, t)` into windows of at least `ω₀` ticks number at most `t / ω₀`. -/
theorem cuts_mul_le {t : ℕ} (κ : Cuts t) {ω₀ : ℕ} (h : ∀ j : Fin κ.J, ω₀ ≤ κ.τ j.succ - κ.τ j.castSucc) :
    κ.J * ω₀ ≤ t := by
  have key : ∀ k (hk : k ≤ κ.J), k * ω₀ ≤ κ.τ ⟨k, Nat.lt_succ_of_le hk⟩ := by
    intro k
    induction k with
    | zero => intro _; simp
    | succ k ih =>
      intro hk
      have h1 := ih (Nat.le_of_succ_le hk)
      have h2 : ω₀ ≤ κ.τ ⟨k + 1, Nat.lt_succ_of_le hk⟩ - κ.τ ⟨k, Nat.lt_succ_of_le (Nat.le_of_succ_le hk)⟩ :=
        h ⟨k, hk⟩
      have h3 : κ.τ ⟨k, Nat.lt_succ_of_le (Nat.le_of_succ_le hk)⟩ ≤ κ.τ ⟨k + 1, Nat.lt_succ_of_le hk⟩ :=
        κ.mono (Fin.le_def.2 (Nat.le_succ k))
      rw [add_one_mul]
      omega
  have h5 : κ.τ ⟨κ.J, Nat.lt_succ_of_le le_rfl⟩ = t := κ.last
  exact (key κ.J le_rfl).trans h5.le

theorem FitRate : Guarantees.FitRate := by
  intro L E _ _ A ℓ ends β link n X θ hθ t κ ω R r c M T start Nstar b_w w_e hX hC hL hω hr hR hgap hReach hFits
  obtain ⟨P, hP⟩ := Accounting.SecurityProofs.Actual.WindowsAt A ℓ (fun _ => A.N) ends β link n X θ hθ t κ
    ((c.θᶜ).map ⟨M.emb, M.emb_inj⟩) hX hC (computeRate_N A _) hL
  have hQ : ∀ (i : Fin n) (j : Fin κ.J) (g : Fin c.D.N), g ∈ (pull M P).gates (finProdFinEquiv (i, j)) ↔
      M.emb g ∈ A.region (X i) (Set.Ico (κ.τ j.castSucc) (κ.τ j.succ)) := fun i j g =>
    mem_pull_gates.trans (Iff.of_eq (congrArg (fun s => M.emb g ∈ s) (hP i j).1))
  have hread : ∀ u, (novelReads c ((pull M P).gates u) : ℝ) ≤ R + r * ω := by
    intro u
    obtain ⟨⟨i, j⟩, rfl⟩ := finProdFinEquiv.surjective u
    have h1 : novelReads c ((pull M P).gates (finProdFinEquiv (i, j))) ≤
        A.stateIn (X i) (κ.τ j.castSucc) ((c.θᶜ).map ⟨M.emb, M.emb_inj⟩) +
          linkCapacity ends β (X i) * (κ.τ j.succ - κ.τ j.castSucc) :=
      (novelReads_pull_le_readsIn_compl M P _).trans (hP i j).2.2
    have h2 : ((linkCapacity ends β (X i) : ℕ) : ℝ) * ((κ.τ j.succ - κ.τ j.castSucc : ℕ) : ℝ) ≤ r * ω :=
      mul_le_mul (hr i) (Nat.cast_le.2 (hω j)) (Nat.cast_nonneg _) ((Nat.cast_nonneg _).trans (hr i))
    have h3 : (novelReads c ((pull M P).gates (finProdFinEquiv (i, j))) : ℝ) ≤
        (A.stateIn (X i) (κ.τ j.castSucc) ((c.θᶜ).map ⟨M.emb, M.emb_inj⟩) : ℝ) +
          ((linkCapacity ends β (X i) : ℕ) : ℝ) * ((κ.τ j.succ - κ.τ j.castSucc : ℕ) : ℝ) := by
      exact_mod_cast h1
    linarith [hR i j]
  have hbound := FitBound c T (n * κ.J) (pull M P) start Nstar (R + r * ω) b_w w_e hgap hread hReach (hFits _ hQ)
  refine ⟨hbound, fun ω₀ hω₀ => ?_⟩
  have hJ : ((κ.J * ω₀ : ℕ) : ℝ) ≤ t := Nat.cast_le.2 (cuts_mul_le κ hω₀)
  have hY : (0 : ℝ) ≤ n * T.reuse * Nstar * (R + r * ω) ^ 2 := by positivity
  calc (ω₀ : ℝ) * (4 * w_e * (Nstar * b_w - (R + r * ω)) * T.M.card)
      ≤ ω₀ * ((n * κ.J : ℕ) * T.reuse * Nstar * (R + r * ω) ^ 2) :=
        mul_le_mul_of_nonneg_left hbound (Nat.cast_nonneg _)
    _ = ((κ.J * ω₀ : ℕ) : ℝ) * (n * T.reuse * Nstar * (R + r * ω) ^ 2) := by push_cast; ring
    _ ≤ t * (n * T.reuse * Nstar * (R + r * ω) ^ 2) := mul_le_mul_of_nonneg_right hJ hY
    _ = n * t * T.reuse * Nstar * (R + r * ω) ^ 2 := by ring

end Nci.SecurityProofs
