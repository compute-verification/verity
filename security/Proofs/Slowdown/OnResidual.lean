module

public import Proofs.Slowdown.FitRate
public import Proofs.ResourceAccounting.Actual.Outside

/-!
# The fit bound on what the memory and bandwidth audits leave

`FitRate` bounds a reference training algorithm `c` contained in the developer's declaration by what each window reads
of `c`'s novel gates: its server's state in them at the window's start, plus what crosses the server's boundary. Here
the two are bounded by what audits leave. A set `S` of gates, the audited computation's, shares no gate with the image
of `c`'s novel gates; at every cut each server carries at least `x_s` bits of `S`, and in every window it reads at
least `x_b` bits of `S` from outside itself. Memory `m` at the cuts and links of capacity `b` then leave each window at
most `(m − x_s) + (b ω − x_b)` bits of `c`'s novel gates to read
(`ResourceAccounting.SecurityProofs.Actual.state_outside`, `ingress_outside`), and `fitRate_of_reads` gives the count
(`fit_on_residual`).
-/

@[expose] public section

open InferenceOnly InferenceOnly.SecurityProofs

namespace Slowdown.SecurityProofs

open Verity ResourceAccounting ResourceAccounting.SecurityProofs.Actual Finset

/-- **The fit bound on what the memory and bandwidth audits leave.** `FitRate` with `(m − x_s) + (b ω − x_b)` for
`R + r ω`: the servers' memory at the cuts is at most `m` and their link capacity at most `b`, the gates `S` share none
with the image of `c`'s novel gates, each server carries at least `x_s` bits of `S` across each cut, and each window
reads at least `x_b` bits of `S` from outside its server. -/
theorem fit_on_residual {L E : Type*} [Fintype E] [DecidableEq L] (A : Placed L) (ℓ : L → L → ℕ)
    (ends : E → L × L) (β : E → ℕ) (link : Fin A.N → Fin A.N → E) (n : ℕ) (X : Fin n → Finset L)
    (θ : Finset (Fin A.N)) (hθ : ∀ a ∈ θ, A.isInput a = true) (t : ℕ) (κ : Cuts t) (ω : ℕ)
    (mem : Finset L → ℕ) (m b x_s x_b : ℕ) (S : Finset (Fin A.N))
    (c : RegisteredCircuit) (M : Containment c.D (A.declaration (univ.biUnion X) t θ hθ).D) (T : Threading c)
    (start : Fin T.m → Fin (n * κ.J)) (Nstar : ℕ) (b_w w_e : ℝ)
    (hX : ∀ i j, i ≠ j → Disjoint (X i) (X j)) (hC : A.Causal ℓ) (hL : A.Links ends β link)
    (hM : A.MemoryAt (Set.range X) mem (Set.range fun j : Fin κ.J => κ.τ j.castSucc))
    (hω : ∀ j : Fin κ.J, κ.τ j.succ - κ.τ j.castSucc ≤ ω)
    (hm : ∀ i, mem (X i) ≤ m) (hb : ∀ i, linkCapacity ends β (X i) ≤ b)
    (hS : Disjoint S ((c.θᶜ).map ⟨M.emb, M.emb_inj⟩))
    (hxs : ∀ (i : Fin n) (j : Fin κ.J), x_s ≤ A.stateIn (X i) (κ.τ j.castSucc) S)
    (hxb : ∀ (i : Fin n) (j : Fin κ.J), x_b ≤ A.width
      (((A.into (A.region (X i) (Set.Ico (κ.τ j.castSucc) (κ.τ j.succ)))).filter fun g => A.x g ∉ X i) ∩ S))
    (hgap : (((m - x_s) + (b * ω - x_b) : ℕ) : ℝ) < Nstar * b_w) (hReach : Assumptions.Reach c T Nstar)
    (hFits : ∀ Q : Partition c.D (n * κ.J), (∀ (i : Fin n) (j : Fin κ.J) (g : Fin c.D.N),
        g ∈ Q.gates (finProdFinEquiv (i, j)) ↔ M.emb g ∈ A.region (X i) (Set.Ico (κ.τ j.castSucc) (κ.τ j.succ))) →
      Assumptions.Fits c T Q start b_w w_e) :
    4 * w_e * (Nstar * b_w - (((m - x_s) + (b * ω - x_b) : ℕ) : ℝ)) * T.M.card ≤
        (n * κ.J : ℕ) * T.reuse * Nstar * (((m - x_s) + (b * ω - x_b) : ℕ) : ℝ) ^ 2 ∧
      ∀ ω₀ : ℕ, (∀ j : Fin κ.J, ω₀ ≤ κ.τ j.succ - κ.τ j.castSucc) →
        ω₀ * (4 * w_e * (Nstar * b_w - (((m - x_s) + (b * ω - x_b) : ℕ) : ℝ)) * T.M.card) ≤
          n * t * T.reuse * Nstar * (((m - x_s) + (b * ω - x_b) : ℕ) : ℝ) ^ 2 := by
  refine fitRate_of_reads A ℓ ends β link n X θ hθ t κ _ c M T start Nstar b_w w_e hX hC hL (fun i j => ?_) hgap
    hReach hFits
  have hτ : κ.τ j.succ ≤ t := (κ.mono (Fin.le_last _)).trans κ.last.le
  have hR : A.region (X i) (Set.Ico (κ.τ j.castSucc) (κ.τ j.succ)) ⊆ A.region (univ.biUnion X) (Set.Iio t) :=
    region_subset A (subset_biUnion_of_mem X (mem_univ i)) fun s hs => Set.mem_Iio.2 (hs.2.trans_le hτ)
  have hG : (c.θᶜ).map ⟨M.emb, M.emb_inj⟩ ⊆ Sᶜ := fun a ha => mem_compl.2 fun hs => disjoint_left.1 hS hs ha
  have h1 := width_into_region_inter_le_boundary A (X i) (κ.τ j.castSucc) (κ.τ j.succ)
    ((c.θᶜ).map ⟨M.emb, M.emb_inj⟩) (forward_of_causal A hC _)
  have h2 : A.stateIn (X i) (κ.τ j.castSucc) ((c.θᶜ).map ⟨M.emb, M.emb_inj⟩) ≤ m - x_s :=
    ((stateIn_mono A _ _ hG).trans
      (state_outside A hM (X := X i) ⟨i, rfl⟩ (t := κ.τ j.castSucc) ⟨j, rfl⟩ S (hxs i j))).trans
      (Nat.sub_le_sub_right (hm i) x_s)
  have h3 : A.width (((A.into (A.region (X i) (Set.Ico (κ.τ j.castSucc) (κ.τ j.succ)))).filter
      fun g => A.x g ∉ X i) ∩ (c.θᶜ).map ⟨M.emb, M.emb_inj⟩) ≤ b * ω - x_b :=
    ((width_mono A (inter_subset_inter (Finset.Subset.refl _) hG)).trans
      (ingress_outside A hL (X i) _ _ S (hxb i j))).trans
      (Nat.sub_le_sub_right (Nat.mul_le_mul (hb i) (hω j)) x_b)
  exact Nat.cast_le.2 ((readsIn_declaration A hθ hR _).le.trans (h1.trans (Nat.add_le_add h2 h3)))

end Slowdown.SecurityProofs
