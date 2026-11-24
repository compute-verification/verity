module

public import Proofs.Slowdown.FitRate
public import Proofs.ResourceAccounting.Actual.Outside

/-!
# The fit bound on what the memory and bandwidth audits leave

`FitRate` bounds a reference training algorithm `c` contained in the developer's declaration by what each window reads
of `c`'s novel gates: its server's state in them at the window's start, plus what crosses the server's boundary. Here
the two are bounded by what audits leave. `S` is a set of gates, the audited computation's. Outside `S` a server
carries at most `r_s` bits across a cut and a window reads at most `r_b` bits from outside its server: the residuals
`ResourceAccounting.SecurityProofs.Actual.state_outside`, `state_outside_of_held` and `ingress_outside` give from a
limit and what `S` uses of it. Of `S` itself, at most `o_s` bits carried and `o_b` bits read are of `c`'s novel gates.
The window then reads at most `(r_s + o_s) + (r_b + o_b)` bits of `c`'s novel gates (`window_reads_le`).

* `fit_on_residual`: that at every window, and `fitRate_of_reads` gives the count.
* `fit_on_residual_but`: that at the windows outside a set `I`, and nothing at those in `I`. The counted
  multiplications placed in `I`'s windows are at most the work those windows can do, `Φ ω` each under `ComputeRate`,
  and `fitRate_of_reads_but` adds them to the count.
-/

@[expose] public section

open InferenceOnly InferenceOnly.SecurityProofs

namespace Slowdown.SecurityProofs

open Verity ResourceAccounting ResourceAccounting.SecurityProofs.Actual Finset

/-- **What a window reads in `G`, beside what is accounted for.** Outside `S`, `X` carries at most `r_s` bits across
`a` and `A[X, [a, b))` reads at most `r_b` bits from outside `X`; of `S`, at most `o_s` bits carried and `o_b` bits read
are in `G`. Then in the developer's declaration the window reads at most `(r_s + o_s) + (r_b + o_b)` bits of `G`. -/
theorem window_reads_le {L : Type*} [DecidableEq L] (A : Placed L) {ℓ : L → L → ℕ} (hC : A.Causal ℓ)
    {X Y : Finset L} (hXY : X ⊆ Y) {θ : Finset (Fin A.N)} (hθ : ∀ a ∈ θ, A.isInput a = true)
    {t a b : ℕ} (hb : b ≤ t) (S G : Finset (Fin A.N)) {r_s r_b o_s o_b : ℕ}
    (hrs : A.stateIn X a Sᶜ ≤ r_s)
    (hrb : A.width (((A.into (A.region X (Set.Ico a b))).filter fun g => A.x g ∉ X) ∩ Sᶜ) ≤ r_b)
    (hos : A.stateIn X a (S ∩ G) ≤ o_s)
    (hob : A.width (((A.into (A.region X (Set.Ico a b))).filter fun g => A.x g ∉ X) ∩ (S ∩ G)) ≤ o_b) :
    readsIn (A.declaration Y t θ hθ) (A.region X (Set.Ico a b)) G ≤ (r_s + o_s) + (r_b + o_b) := by
  have hR : A.region X (Set.Ico a b) ⊆ A.region Y (Set.Iio t) :=
    region_subset A hXY fun s hs => Set.mem_Iio.2 (hs.2.trans_le hb)
  have h1 := width_into_region_inter_le_boundary A X a b G (forward_of_causal A hC _)
  have h2 : A.stateIn X a G ≤ r_s + o_s := (stateIn_le_split A X a S G).trans (Nat.add_le_add hrs hos)
  have h3 : A.width (((A.into (A.region X (Set.Ico a b))).filter fun g => A.x g ∉ X) ∩ G) ≤ r_b + o_b :=
    (width_inter_le_split A _ S G).trans (Nat.add_le_add hrb hob)
  exact (readsIn_declaration A hθ hR G).le.trans (h1.trans (Nat.add_le_add h2 h3))

/-- **The fit bound on what the memory and bandwidth audits leave.** `FitRate` with `(r_s + o_s) + (r_b + o_b)` for
`R + r ω`: outside the gates `S`, each server carries at most `r_s` bits across each cut and each window reads at most
`r_b` bits from outside its server; of `S`, at most `o_s` bits carried and `o_b` bits read are of the image of `c`'s
novel gates. -/
theorem fit_on_residual {L E : Type*} [Fintype E] [DecidableEq L] (A : Placed L) (ℓ : L → L → ℕ)
    (ends : E → L × L) (β : E → ℕ) (link : Fin A.N → Fin A.N → E) (n : ℕ) (X : Fin n → Finset L)
    (θ : Finset (Fin A.N)) (hθ : ∀ a ∈ θ, A.isInput a = true) (t : ℕ) (κ : Cuts t)
    (r_s r_b o_s o_b : ℕ) (S : Finset (Fin A.N))
    (c : RegisteredCircuit) (M : Containment c.D (A.declaration (univ.biUnion X) t θ hθ).D) (T : Threading c)
    (start : Fin T.m → Fin (n * κ.J)) (Nstar : ℕ) (b_w w_e : ℝ)
    (hX : ∀ i j, i ≠ j → Disjoint (X i) (X j)) (hC : A.Causal ℓ) (hL : A.Links ends β link)
    (hrs : ∀ (i : Fin n) (j : Fin κ.J), A.stateIn (X i) (κ.τ j.castSucc) Sᶜ ≤ r_s)
    (hrb : ∀ (i : Fin n) (j : Fin κ.J), A.width
      (((A.into (A.region (X i) (Set.Ico (κ.τ j.castSucc) (κ.τ j.succ)))).filter fun g => A.x g ∉ X i) ∩ Sᶜ) ≤ r_b)
    (hos : ∀ (i : Fin n) (j : Fin κ.J),
      A.stateIn (X i) (κ.τ j.castSucc) (S ∩ (c.θᶜ).map ⟨M.emb, M.emb_inj⟩) ≤ o_s)
    (hob : ∀ (i : Fin n) (j : Fin κ.J), A.width
      (((A.into (A.region (X i) (Set.Ico (κ.τ j.castSucc) (κ.τ j.succ)))).filter fun g => A.x g ∉ X i) ∩
        (S ∩ (c.θᶜ).map ⟨M.emb, M.emb_inj⟩)) ≤ o_b)
    (hgap : (((r_s + o_s) + (r_b + o_b) : ℕ) : ℝ) < Nstar * b_w) (hReach : Assumptions.Reach c T Nstar)
    (hFits : ∀ Q : Partition c.D (n * κ.J), (∀ (i : Fin n) (j : Fin κ.J) (g : Fin c.D.N),
        g ∈ Q.gates (finProdFinEquiv (i, j)) ↔ M.emb g ∈ A.region (X i) (Set.Ico (κ.τ j.castSucc) (κ.τ j.succ))) →
      Assumptions.Fits c T Q start b_w w_e) :
    4 * w_e * (Nstar * b_w - (((r_s + o_s) + (r_b + o_b) : ℕ) : ℝ)) * T.M.card ≤
        (n * κ.J : ℕ) * T.reuse * Nstar * (((r_s + o_s) + (r_b + o_b) : ℕ) : ℝ) ^ 2 ∧
      ∀ ω₀ : ℕ, (∀ j : Fin κ.J, ω₀ ≤ κ.τ j.succ - κ.τ j.castSucc) →
        ω₀ * (4 * w_e * (Nstar * b_w - (((r_s + o_s) + (r_b + o_b) : ℕ) : ℝ)) * T.M.card) ≤
          n * t * T.reuse * Nstar * (((r_s + o_s) + (r_b + o_b) : ℕ) : ℝ) ^ 2 := by
  refine fitRate_of_reads A ℓ ends β link n X θ hθ t κ _ c M T start Nstar b_w w_e hX hC hL (fun i j => ?_) hgap
    hReach hFits
  have hτ : κ.τ j.succ ≤ t := (κ.mono (Fin.le_last _)).trans κ.last.le
  exact Nat.cast_le.2 (window_reads_le A hC (subset_biUnion_of_mem X (mem_univ i)) hθ hτ S _ (hrs i j) (hrb i j)
    (hos i j) (hob i j))

open Classical in
/-- The counted multiplications placed in the windows of `I` number at most the work those windows can do: `Φ ω` each,
at a price of at least 1 a counted multiplication. -/
theorem card_left_out_le {L : Type*} [DecidableEq L] (A : Placed L) (n : ℕ) (X : Fin n → Finset L)
    (θ : Finset (Fin A.N)) (hθ : ∀ a ∈ θ, A.isInput a = true) (t : ℕ) (κ : Cuts t) (ω : ℕ)
    (compute : Finset L → ℕ) (π : Fin A.N → ℕ) (Φ : ℕ)
    (c : RegisteredCircuit) (M : Containment c.D (A.declaration (univ.biUnion X) t θ hθ).D) (T : Threading c)
    (I : Finset (Fin (n * κ.J)))
    (hW : A.ComputeRate (Set.range X) π compute) (hω : ∀ j : Fin κ.J, κ.τ j.succ - κ.τ j.castSucc ≤ ω)
    (hΦ : ∀ i, compute (X i) ≤ Φ) (hπ : ∀ g ∈ T.M, 1 ≤ π (M.emb g)) :
    (T.M.filter fun g => ∃ i j, finProdFinEquiv (i, j) ∈ I ∧
      M.emb g ∈ A.region (X i) (Set.Ico (κ.τ j.castSucc) (κ.τ j.succ))).card ≤ I.card * (Φ * ω) := by
  have hsub : (T.M.filter fun g => ∃ i j, finProdFinEquiv (i, j) ∈ I ∧
      M.emb g ∈ A.region (X i) (Set.Ico (κ.τ j.castSucc) (κ.τ j.succ))) ⊆
      I.biUnion fun u => T.M.filter fun g => M.emb g ∈ A.region (X (finProdFinEquiv.symm u).1)
        (Set.Ico (κ.τ (finProdFinEquiv.symm u).2.castSucc) (κ.τ (finProdFinEquiv.symm u).2.succ)) := by
    intro g hg
    obtain ⟨hgs, i, j, hu, hgr⟩ := mem_filter.1 hg
    refine mem_biUnion.2 ⟨_, hu, mem_filter.2 ⟨hgs, ?_⟩⟩
    rw [Equiv.symm_apply_apply]
    exact hgr
  have hwin : ∀ R : Finset (Fin A.N), (T.M.filter fun g => M.emb g ∈ R).card ≤ A.work π R := fun R => by
    convert card_filter_le_work A π R T.M ⟨M.emb, M.emb_inj⟩ hπ
  refine (card_le_card hsub).trans (card_biUnion_le.trans ?_)
  calc _ ≤ ∑ _u ∈ I, Φ * ω := sum_le_sum fun u _ =>
        (hwin _).trans ((hW _ ⟨_, rfl⟩ _ _).trans (Nat.mul_le_mul (hΦ _) (hω _)))
    _ = I.card * (Φ * ω) := by rw [sum_const, smul_eq_mul]

/-- **The fit bound on what the audits leave, with windows left out.** `fit_on_residual`'s count when its premises on
`S` hold at the windows outside `I` only: the window half of `Fits` is asked there, and each thread started outside `I`
that works in its start window uses at least `Nstar` weights outside `I`'s windows. Nothing is asked of a window in
`I` but the compute rate: it does at most `Φ ω` work, at a price of at least 1 a counted multiplication, and that many
counted multiplications a window in `I` are added to the bound. -/
theorem fit_on_residual_but {L E : Type*} [Fintype E] [DecidableEq L] (A : Placed L) (ℓ : L → L → ℕ)
    (ends : E → L × L) (β : E → ℕ) (link : Fin A.N → Fin A.N → E) (n : ℕ) (X : Fin n → Finset L)
    (θ : Finset (Fin A.N)) (hθ : ∀ a ∈ θ, A.isInput a = true) (t : ℕ) (κ : Cuts t) (ω : ℕ)
    (compute : Finset L → ℕ) (π : Fin A.N → ℕ) (Φ r_s r_b o_s o_b : ℕ) (S : Finset (Fin A.N))
    (c : RegisteredCircuit) (M : Containment c.D (A.declaration (univ.biUnion X) t θ hθ).D) (T : Threading c)
    (start : Fin T.m → Fin (n * κ.J)) (I : Finset (Fin (n * κ.J))) (Nstar : ℕ) (b_w w_e : ℝ)
    (hX : ∀ i j, i ≠ j → Disjoint (X i) (X j)) (hC : A.Causal ℓ) (hL : A.Links ends β link)
    (hW : A.ComputeRate (Set.range X) π compute)
    (hω : ∀ j : Fin κ.J, κ.τ j.succ - κ.τ j.castSucc ≤ ω) (hΦ : ∀ i, compute (X i) ≤ Φ)
    (hπ : ∀ g ∈ T.M, 1 ≤ π (M.emb g)) (hbw : 0 ≤ b_w) (hwe : 0 ≤ w_e)
    (hrs : ∀ (i : Fin n) (j : Fin κ.J), finProdFinEquiv (i, j) ∉ I → A.stateIn (X i) (κ.τ j.castSucc) Sᶜ ≤ r_s)
    (hrb : ∀ (i : Fin n) (j : Fin κ.J), finProdFinEquiv (i, j) ∉ I → A.width
      (((A.into (A.region (X i) (Set.Ico (κ.τ j.castSucc) (κ.τ j.succ)))).filter fun g => A.x g ∉ X i) ∩ Sᶜ) ≤ r_b)
    (hos : ∀ (i : Fin n) (j : Fin κ.J), finProdFinEquiv (i, j) ∉ I →
      A.stateIn (X i) (κ.τ j.castSucc) (S ∩ (c.θᶜ).map ⟨M.emb, M.emb_inj⟩) ≤ o_s)
    (hob : ∀ (i : Fin n) (j : Fin κ.J), finProdFinEquiv (i, j) ∉ I → A.width
      (((A.into (A.region (X i) (Set.Ico (κ.τ j.castSucc) (κ.τ j.succ)))).filter fun g => A.x g ∉ X i) ∩
        (S ∩ (c.θᶜ).map ⟨M.emb, M.emb_inj⟩)) ≤ o_b)
    (hgap : (((r_s + o_s) + (r_b + o_b) : ℕ) : ℝ) < Nstar * b_w)
    (hRF : ∀ Q : Partition c.D (n * κ.J), (∀ (i : Fin n) (j : Fin κ.J) (g : Fin c.D.N),
        g ∈ Q.gates (finProdFinEquiv (i, j)) ↔ M.emb g ∈ A.region (X i) (Set.Ico (κ.τ j.castSucc) (κ.τ j.succ))) →
      (∀ u, start u ∉ I → (T.threadMults (Q.gates (start u)) u).Nonempty →
        Nstar ≤ (T.threadWeights (Fit.outside Q I) u).card) ∧
      ∀ j ∉ I, b_w * (T.weightsUsed (Q.gates j)).card + w_e * (T.entered Q start j).card ≤
        freshInput c (Q.gates j)) :
    4 * w_e * (Nstar * b_w - (((r_s + o_s) + (r_b + o_b) : ℕ) : ℝ)) * T.M.card ≤
      (n * κ.J : ℕ) * T.reuse * Nstar * (((r_s + o_s) + (r_b + o_b) : ℕ) : ℝ) ^ 2 +
        4 * w_e * (Nstar * b_w - (((r_s + o_s) + (r_b + o_b) : ℕ) : ℝ)) * (I.card * (Φ * ω) : ℕ) := by
  have hc : 0 ≤ 4 * w_e * (Nstar * b_w - (((r_s + o_s) + (r_b + o_b) : ℕ) : ℝ)) :=
    mul_nonneg (mul_nonneg (by norm_num) hwe) (sub_nonneg.2 hgap.le)
  have hk := fitRate_of_reads_but A ℓ ends β link n X θ hθ t κ (((r_s + o_s) + (r_b + o_b) : ℕ) : ℝ) c M T
    start I Nstar b_w w_e hX hC hL hbw hwe (Nat.cast_nonneg _) (fun i j hu => by
      have hτ : κ.τ j.succ ≤ t := (κ.mono (Fin.le_last _)).trans κ.last.le
      exact Nat.cast_le.2 (window_reads_le A hC (subset_biUnion_of_mem X (mem_univ i)) hθ hτ S _ (hrs i j hu)
        (hrb i j hu) (hos i j hu) (hob i j hu))) hgap hRF
  have hcount := card_left_out_le A n X θ hθ t κ ω compute π Φ c M T I hW hω hΦ hπ
  exact hk.trans (add_le_add le_rfl (mul_le_mul_of_nonneg_left (Nat.cast_le.2 hcount) hc))

end Slowdown.SecurityProofs
