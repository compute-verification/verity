module

public import Definitions.Slowdown.Guarantees.FitRate
public import Proofs.ResourceAccounting.Actual.ServerBlocks
public import Proofs.Slowdown.FitKeep

/-!
# Proof of the fit bound on a run

`ServerBlocksAt` at the run's cuts divides the developer's declaration into the windows `A[X i, [τ j, τ (j + 1))]`, each
reading in `G` at most the part in `G` of its server's state at `τ j` plus `β_{X i} (τ (j + 1) − τ j)`. At `G` the image
of `c`'s novel gates, the windows pulled into `c` (`pull`) read at most that (`freshInput_pull_le_readsIn_compl`), so at
most `R + r ω`, and `FitBound` gives the count over the `n J` windows. Its compute-rate hypothesis holds at `Φ = A.N`
(`computeRate_N`), and the count does not read `Φ`. Windows of at least `ω₀` ticks number at most `t / ω₀`
(`cuts_mul_le`), which gives the rate.

`fitRate_of_reads_but` is the count at any bound on the reads of the windows outside a set `I`, with nothing asked of
the windows in `I`: their counted multiplications are added to the bound (`Fit.fitBoundSlots_but`).
-/

@[expose] public section

open InferenceOnly InferenceOnly.SecurityProofs

namespace Slowdown.SecurityProofs

open Verity ResourceAccounting Finset

/-- Through a containment `M : c ≼ d`, the piece of `c` that a unit of `d` induces reads at most what the unit reads in
the image of `c`'s novel gates. -/
theorem freshInput_pull_le_readsIn_compl {c d : RegisteredCircuit} (M : Containment c.D d.D) {n : ℕ} (P : Partition d.D n)
    (u : Fin n) : freshInput c ((pull M P).gates u) ≤ readsIn d (P.gates u) ((c.θᶜ).map ⟨M.emb, M.emb_inj⟩) := by
  unfold freshInput readsIn
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
        obtain ⟨-, -, ht⟩ := (ResourceAccounting.SecurityProofs.Actual.mem_region A).1 hg
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

open Classical in
/-- **The fit bound from the windows' reads, with windows left out**: `FitRate`'s count when a bound `S ≥ 0` on a
window's reads in the image of `c`'s novel gates, whatever bounds them, and the window half of `Fits`, hold at the
windows outside `I` only, and each thread started outside `I` that works in its start window uses at least `Nstar`
weights outside `I`'s windows. The counted multiplications placed in `I`'s windows are added to the bound
(`Fit.fitBoundSlots_but`). -/
theorem fitRate_of_reads_but {L E : Type*} [Fintype E] [DecidableEq L] (A : Placed L) (ℓ : L → L → ℕ)
    (ends : E → L × L) (β : E → ℕ) (link : Fin A.N → Fin A.N → E) (n : ℕ) (X : Fin n → Finset L)
    (θ : Finset (Fin A.N)) (hθ : ∀ a ∈ θ, A.isInput a = true) (t : ℕ) (κ : Cuts t) (S : ℝ)
    (c : RegisteredCircuit) (M : Containment c.D (A.declaration (univ.biUnion X) t θ hθ).D) (T : Threading c)
    (start : Fin T.m → Fin (n * κ.J)) (I : Finset (Fin (n * κ.J))) (Nstar : ℕ) (b_w w_e : ℝ)
    (hX : ∀ i j, i ≠ j → Disjoint (X i) (X j)) (hC : A.Causal ℓ) (hL : A.Links ends β link)
    (hbw : 0 ≤ b_w) (hwe : 0 ≤ w_e) (hS0 : 0 ≤ S)
    (hS : ∀ (i : Fin n) (j : Fin κ.J), finProdFinEquiv (i, j) ∉ I →
      (readsIn (A.declaration (univ.biUnion X) t θ hθ)
        (A.region (X i) (Set.Ico (κ.τ j.castSucc) (κ.τ j.succ))) ((c.θᶜ).map ⟨M.emb, M.emb_inj⟩) : ℝ) ≤ S)
    (hgap : S < Nstar * b_w)
    (hRF : ∀ Q : Partition c.D (n * κ.J), (∀ (i : Fin n) (j : Fin κ.J) (g : Fin c.D.N),
        g ∈ Q.gates (finProdFinEquiv (i, j)) ↔ M.emb g ∈ A.region (X i) (Set.Ico (κ.τ j.castSucc) (κ.τ j.succ))) →
      (∀ u, start u ∉ I → (T.threadMults (Q.gates (start u)) u).Nonempty →
        Nstar ≤ (T.threadWeights (Fit.outside Q I) u).card) ∧
      ∀ j ∉ I, b_w * (T.weightsUsed (Q.gates j)).card + w_e * (T.entered Q start j).card ≤
        freshInput c (Q.gates j)) :
    4 * w_e * (Nstar * b_w - S) * T.M.card ≤ (n * κ.J : ℕ) * T.reuse * Nstar * S ^ 2 +
      4 * w_e * (Nstar * b_w - S) * (T.M.filter fun g => ∃ i j, finProdFinEquiv (i, j) ∈ I ∧
        M.emb g ∈ A.region (X i) (Set.Ico (κ.τ j.castSucc) (κ.τ j.succ))).card := by
  obtain ⟨P, hP⟩ := ResourceAccounting.SecurityProofs.Actual.ServerBlocksAt A ℓ (fun _ => A.N) ends β link n X θ hθ t κ
    ((c.θᶜ).map ⟨M.emb, M.emb_inj⟩) hX hC (computeRate_N A _) hL
  have hQ : ∀ (i : Fin n) (j : Fin κ.J) (g : Fin c.D.N), g ∈ (pull M P).gates (finProdFinEquiv (i, j)) ↔
      M.emb g ∈ A.region (X i) (Set.Ico (κ.τ j.castSucc) (κ.τ j.succ)) := fun i j g =>
    mem_pull_gates.trans (Iff.of_eq (congrArg (fun s => M.emb g ∈ s) (hP i j).1))
  have hread : ∀ u ∉ I, (freshInput c ((pull M P).gates u) : ℝ) ≤ S := by
    intro u hu
    obtain ⟨⟨i, j⟩, rfl⟩ := finProdFinEquiv.surjective u
    have h1 := freshInput_pull_le_readsIn_compl M P (finProdFinEquiv (i, j))
    rw [(hP i j).1] at h1
    exact (Nat.cast_le.2 h1).trans (hS i j hu)
  obtain ⟨hR, hF⟩ := hRF _ hQ
  have hk := Fit.fitBoundSlots_but T (Fit.idSlots T) (pull M P) I start Nstar S b_w w_e hbw hwe hgap hS0 hread
    (fun u hu hne => by
      have h1 := hR u hu (hne.mono fun g hg => ((Fit.mem_keep_threadMults T _ _ u g).1 hg).1)
      have h2 := Fit.card_threadSlots (Fit.keepSlots (Fit.idSlots T) (Fit.outside (pull M P) I))
        (Fit.idSlots_injOn T) univ u
      rw [Fit.keep_threadWeights_univ] at h2
      exact h1.trans h2.ge)
    (fun j hj => by
      have h2 := Fit.card_slotsUsed (Fit.idSlots T) (Fit.idSlots_injOn T) ((pull M P).gates j)
      rw [h2]
      exact hF j hj)
  have hout : (T.M.filter fun g => ∃ j ∈ I, g ∈ (pull M P).gates j) = T.M.filter fun g => ∃ i j,
      finProdFinEquiv (i, j) ∈ I ∧ M.emb g ∈ A.region (X i) (Set.Ico (κ.τ j.castSucc) (κ.τ j.succ)) := by
    refine filter_congr fun g _ => ⟨fun ⟨u, hu, hg⟩ => ?_, fun ⟨i, j, hu, hg⟩ => ⟨_, hu, (hQ i j g).2 hg⟩⟩
    obtain ⟨⟨i, j⟩, rfl⟩ := finProdFinEquiv.surjective u
    exact ⟨i, j, hu, (hQ i j g).1 hg⟩
  rw [hout] at hk
  exact_mod_cast hk

theorem FitRate : Guarantees.FitRate := by
  intro L E _ _ A ℓ ends β link n X θ hθ t κ ω R r c M T start Nstar b_w w_e hX hC hL hω hr hR hgap hReach hFits
  obtain ⟨P, hP⟩ := ResourceAccounting.SecurityProofs.Actual.ServerBlocksAt A ℓ (fun _ => A.N) ends β link n X θ hθ t κ
    ((c.θᶜ).map ⟨M.emb, M.emb_inj⟩) hX hC (computeRate_N A _) hL
  have hQ : ∀ (i : Fin n) (j : Fin κ.J) (g : Fin c.D.N), g ∈ (pull M P).gates (finProdFinEquiv (i, j)) ↔
      M.emb g ∈ A.region (X i) (Set.Ico (κ.τ j.castSucc) (κ.τ j.succ)) := fun i j g =>
    mem_pull_gates.trans (Iff.of_eq (congrArg (fun s => M.emb g ∈ s) (hP i j).1))
  have hread : ∀ u, (freshInput c ((pull M P).gates u) : ℝ) ≤ R + r * ω := by
    intro u
    obtain ⟨⟨i, j⟩, rfl⟩ := finProdFinEquiv.surjective u
    have h1 : freshInput c ((pull M P).gates (finProdFinEquiv (i, j))) ≤
        A.stateIn (X i) (κ.τ j.castSucc) ((c.θᶜ).map ⟨M.emb, M.emb_inj⟩) +
          linkCapacity ends β (X i) * (κ.τ j.succ - κ.τ j.castSucc) :=
      (freshInput_pull_le_readsIn_compl M P _).trans (hP i j).2.2
    have h2 : ((linkCapacity ends β (X i) : ℕ) : ℝ) * ((κ.τ j.succ - κ.τ j.castSucc : ℕ) : ℝ) ≤ r * ω :=
      mul_le_mul (hr i) (Nat.cast_le.2 (hω j)) (Nat.cast_nonneg _) ((Nat.cast_nonneg _).trans (hr i))
    have h3 : (freshInput c ((pull M P).gates (finProdFinEquiv (i, j))) : ℝ) ≤
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

end Slowdown.SecurityProofs
