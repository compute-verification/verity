import Proofs.Accounting.RateAudit.Deployed
import Proofs.Accounting.Remote

/-!
# The deployed session as a run of the remote audit: its routes, and conservation at the same `γ`

`Accounting.Remote.Run.ofDeployed` reads a deployed session as a run of the remote audit. Its routes:

* `Sampled d̄`: an accepting run whose correct receipts cover less than `(1 − d̄)·n·P·N·b` bits holds at most
  `⌊(1 − d̄)·n·P·N⌋` correct receipts, the horizon's draw (`DeployedSession.pr_draws_horizon_le`).
* `Paid`: a run whose correct receipts' words exceed its payments and slacks has a window whose do
  (`DeployedSession.pr_unpaid_le`), which each window's bound (`DeployedWindow.pr_unpaid_draw_le`) covers.

`Accounting.SecurityProofs.Remote.Conservation` composes them; the run's payments other than its tier reads are the
`n·P` passes' `S + (L − ρ) + X` bits, so its `(1 − d̄ − λ)·C` of payments are `(1 − (d̄ + γ))·C` of tier reads, with
`γ` the deployed game's `Deployed.gammaPass`. The remote model's probability is real (`Pouw.pr`), the game's in `ℝ≥0∞`
(`Pous.pr`); `pouw_pr_le` carries a bound across.
-/

namespace Accounting.Remote

open Accounting.Rate
open scoped ENNReal

variable {n P N b h ρ S L q r s d G τ : ℕ}

/-- The remote model's probability is the game's, as a real. -/
theorem pouw_pr_eq {α : Type} [Fintype α] (Q : α → Prop) : Pouw.pr Q = (Pous.pr Q).toReal := by
  unfold Pouw.pr Pous.pr
  rw [ENNReal.toReal_div]
  simp

/-- A bound on the game's probability bounds the remote model's. -/
theorem pouw_pr_le {α : Type} [Fintype α] {Q : α → Prop} {B : ℝ≥0∞} (h : Pous.pr Q ≤ B) (hB : B ≠ ⊤) :
    Pouw.pr Q ≤ B.toReal := by
  rw [pouw_pr_eq]
  exact ENNReal.toReal_mono hB h

/-- The run's tier reads are the session's windows' reads. -/
theorem ofDeployed_reads (A : DeployedSession n P N b h ρ S L q r s d G) (tier : ℕ → Fin τ) (X : ℕ)
    (Nd : Node (Fin τ)) (a : Audit) (x : Store N b × (Fin n → WindowCoins P N b h ρ r s d)) :
    (Run.ofDeployed A tier X Nd a).reads x = ∑ w, (A.win w x.2).reads tier x.1 (x.2 w).1 := by
  simp only [Run.reads, Run.ofDeployed, DeployedWindow.reads]
  rw [Finset.sum_comm]
  exact Finset.sum_congr rfl fun w _ => Finset.sum_comm

/-- The run's payments at every location: its tier reads and the `n·P` passes' `L − ρ`, `S` and `X` bits. -/
theorem ofDeployed_paid (A : DeployedSession n P N b h ρ S L q r s d G) (tier : ℕ → Fin τ) (X : ℕ)
    (Nd : Node (Fin τ)) (a : Audit) (x : Store N b × (Fin n → WindowCoins P N b h ρ r s d)) :
    (Run.ofDeployed A tier X Nd a).paid x =
      (Run.ofDeployed A tier X Nd a).reads x + n * P * (L - ρ) + n * P * S + n * P * X := by
  simp only [Run.paid]
  simp [Run.ofDeployed]

/-- The deployed error terms, finite. -/
theorem roErr_ne_top (N h q lam : ℕ) :
    (N : ℝ≥0∞) / 2 ^ h + (q.choose 2 : ℝ≥0∞) / 2 ^ h + (2 ^ N : ℝ≥0∞) / 2 ^ lam ≠ ⊤ := by
  refine ENNReal.add_ne_top.2 ⟨ENNReal.add_ne_top.2 ⟨?_, ?_⟩, ?_⟩ <;>
    exact ENNReal.div_ne_top (by simp) (by simp)

theorem roErr_toReal (n P G N h q lam : ℕ) :
    ((n : ℝ≥0∞) * (P * (G * ((N : ℝ≥0∞) / 2 ^ h + (q.choose 2 : ℝ≥0∞) / 2 ^ h + (2 ^ N : ℝ≥0∞) / 2 ^ lam)))).toReal =
      (n * P * G : ℝ) * ((N : ℝ) / 2 ^ h + (q.choose 2 : ℝ) / 2 ^ h + (2 ^ N : ℝ) / 2 ^ lam) := by
  have h1 : (N : ℝ≥0∞) / 2 ^ h ≠ ⊤ := ENNReal.div_ne_top (by simp) (by simp)
  have h2 : (q.choose 2 : ℝ≥0∞) / 2 ^ h ≠ ⊤ := ENNReal.div_ne_top (by simp) (by simp)
  have h3 : (2 ^ N : ℝ≥0∞) / 2 ^ lam ≠ ⊤ := ENNReal.div_ne_top (by simp) (by simp)
  rw [ENNReal.toReal_mul, ENNReal.toReal_mul, ENNReal.toReal_mul, ENNReal.toReal_add (ENNReal.add_ne_top.2 ⟨h1, h2⟩) h3,
    ENNReal.toReal_add h1 h2, ENNReal.toReal_div, ENNReal.toReal_div, ENNReal.toReal_div]
  simp only [ENNReal.toReal_natCast, ENNReal.toReal_pow, ENNReal.toReal_ofNat]
  ring

/-- **Sampled** for the deployed session's run, at the horizon's draw. -/
theorem ofDeployed_sampled (A : DeployedSession n P N b h ρ S L q r s d G) (tier : ℕ → Fin τ) (X : ℕ)
    (Nd : Node (Fin τ)) (a : Audit) (dbar : ℝ) (hn : 0 < n) (hP : 0 < P) (hN : 0 < N) (hb : 0 < b)
    (hdbar : dbar ≤ 1) (hab : a.b = b) (hC : Nd.declared a.T = n * P * N * b) :
    (Run.ofDeployed A tier X Nd a).Sampled dbar ((1 - dbar) ^ (d * n)) := by
  classical
  have hb' : (0 : ℝ) < b := by exact_mod_cast hb
  have hK := DeployedSession.pr_draws_horizon_le A ⌊(1 - dbar) * ((n * P * N : ℕ) : ℝ)⌋₊
  have h2 := Deployed.horizon_le dbar (n * P * N) (d * n) (by positivity)
  push_cast at hK h2
  have hB := hK.trans h2
  have hT : ENNReal.ofReal (1 - dbar) ^ (d * n) ≠ ⊤ := ENNReal.pow_ne_top ENNReal.ofReal_ne_top
  have e : (ENNReal.ofReal (1 - dbar) ^ (d * n)).toReal = (1 - dbar) ^ (d * n) := by
    rw [ENNReal.toReal_pow, ENNReal.toReal_ofReal (by linarith)]
  unfold Run.Sampled Run.Holds
  rw [← e]
  refine pouw_pr_le ((Bandwidth.pr_mono fun x hx => ?_).trans hB) hT
  obtain ⟨hacc, hlt⟩ := hx
  refine ⟨hacc, Nat.le_floor ?_⟩
  simp only [Run.ofDeployed, hab, hC, not_le] at hlt
  push_cast at hlt ⊢
  have : (∑ w, (((A.win w x.2).correctAll x.1 (x.2 w).1).card : ℝ)) * b < (1 - dbar) * (n * P * N) * b := by
    linarith
  exact (lt_of_mul_lt_mul_right this hb'.le).le

/-- **Paid** for the deployed session's run: some window's words exceed its payments whenever the session's do. -/
theorem ofDeployed_paid_route (A : DeployedSession n P N b h ρ S L q r s d G) (tier : ℕ → Fin τ)
    (Nd : Node (Fin τ)) (a : Audit) {lam rounds cRoot F : ℕ} {cost : ℕ → ℕ} (hrc : 0 < rounds * cRoot)
    (ha : a.P = n * P) (hab : a.b = b) (hal : a.lam = lam)
    (hW : Accounting.Assumptions.WorkHardness b rounds cRoot cost)
    (hF : ∀ w η j ι c, Accounting.Assumptions.SquaringCeiling cost F ((A.window w η).device j ι c))
    (hA : ∀ w η j ι c E R O O', ((A.window w η).device j ι c).side E R O = ((A.window w η).device j ι c).side E R O') :
    (Run.ofDeployed A tier (Deployed.rebuildBits b rounds cRoot F) Nd a).Paid
      ((n * P * G : ℝ) * ((N : ℝ) / 2 ^ h + (q.choose 2 : ℝ) / 2 ^ h + (2 ^ N : ℝ) / 2 ^ lam)) := by
  classical
  set X := Deployed.rebuildBits b rounds cRoot F
  have hX : ∀ w η j ι c E R O, ((A.window w η).device j ι c).completed E R O ≤ X :=
    fun w η j ι c E R O => PIDevice.completed_le _ hrc hW (hF w η j ι c) E R O
  have hε := fun w η j ι c => PIDevice.pr_unpaid_le ((A.window w η).device j ι c) tier X lam (hA w η j ι c)
  have h₂ := DeployedSession.pr_unpaid_le A tier (S + (L - ρ) + X + lam) _
    (fun w η => (A.window w η).pr_unpaid_draw_le tier X _ d _ (hX w η) (hε w η))
  have hT : (n : ℝ≥0∞) * (P * (G * ((N : ℝ≥0∞) / 2 ^ h + (q.choose 2 : ℝ≥0∞) / 2 ^ h + (2 ^ N : ℝ≥0∞) / 2 ^ lam)))
      ≠ ⊤ :=
    ENNReal.mul_ne_top (ENNReal.natCast_ne_top n) (ENNReal.mul_ne_top (ENNReal.natCast_ne_top P)
      (ENNReal.mul_ne_top (ENNReal.natCast_ne_top G) (roErr_ne_top N h q lam)))
  unfold Run.Paid Run.Holds
  rw [← roErr_toReal]
  refine pouw_pr_le ((Bandwidth.pr_mono fun x hx => ?_).trans h₂) hT
  obtain ⟨-, hlt⟩ := hx
  have hc : (Run.ofDeployed A tier X Nd a).correct x = ∑ w, ((A.win w x.2).correctAll x.1 (x.2 w).1).card := rfl
  rw [not_le, ofDeployed_paid, ofDeployed_reads, hc, ha, hab, hal] at hlt
  by_contra hno
  push Not at hno
  have hle := Finset.sum_le_sum (fun w (_ : w ∈ Finset.univ) => hno w)
  rw [Finset.sum_add_distrib, Finset.sum_const, Finset.card_univ, Fintype.card_fin, smul_eq_mul,
    ← Finset.sum_mul] at hle
  have e : n * (P * (S + (L - ρ) + X + lam)) = n * P * (L - ρ) + n * P * S + n * P * X + n * P * lam := by ring
  omega

theorem roErr_toReal_window (P G N h q lam : ℕ) :
    ((P : ℝ≥0∞) * (G * ((N : ℝ≥0∞) / 2 ^ h + (q.choose 2 : ℝ≥0∞) / 2 ^ h + (2 ^ N : ℝ≥0∞) / 2 ^ lam))).toReal =
      (P * G : ℝ) * ((N : ℝ) / 2 ^ h + (q.choose 2 : ℝ) / 2 ^ h + (2 ^ N : ℝ) / 2 ^ lam) := by
  have := roErr_toReal 1 P G N h q lam
  simpa using this

/-- Window `w`'s run's tier reads are the window's reads. -/
theorem ofDeployedWindow_reads (A : DeployedSession n P N b h ρ S L q r s d G) (w : Fin n) (tier : ℕ → Fin τ)
    (X : ℕ) (Nd : Node (Fin τ)) (a : Audit) (x : Store N b × (Fin n → WindowCoins P N b h ρ r s d)) :
    (Run.ofDeployedWindow A w tier X Nd a).reads x = (A.win w x.2).reads tier x.1 (x.2 w).1 := by
  simp only [Run.reads, Run.ofDeployedWindow, DeployedWindow.reads]
  exact Finset.sum_comm

/-- Window `w`'s run's payments at every location: its tier reads and the `P` passes' `L − ρ`, `S` and `X` bits. -/
theorem ofDeployedWindow_paid (A : DeployedSession n P N b h ρ S L q r s d G) (w : Fin n) (tier : ℕ → Fin τ)
    (X : ℕ) (Nd : Node (Fin τ)) (a : Audit) (x : Store N b × (Fin n → WindowCoins P N b h ρ r s d)) :
    (Run.ofDeployedWindow A w tier X Nd a).paid x =
      (Run.ofDeployedWindow A w tier X Nd a).reads x + P * (L - ρ) + P * S + P * X := by
  simp only [Run.paid]
  simp [Run.ofDeployedWindow]

/-- **Sampled** for window `w`'s run, at the window's draw. -/
theorem ofDeployedWindow_sampled (A : DeployedSession n P N b h ρ S L q r s d G) (w : Fin n) (tier : ℕ → Fin τ)
    (X : ℕ) (Nd : Node (Fin τ)) (a : Audit) (dw : ℝ) (hP : 0 < P) (hN : 0 < N) (hb : 0 < b)
    (hdw : dw ≤ 1) (hab : a.b = b) (hC : Nd.declared a.T = P * N * b) :
    (Run.ofDeployedWindow A w tier X Nd a).Sampled dw ((1 - dw / n) ^ (d * n)) := by
  classical
  have hn : 0 < n := w.pos
  have hb' : (0 : ℝ) < b := by exact_mod_cast hb
  have hn' : (1 : ℝ) ≤ n := by exact_mod_cast hn
  have hu : dw / n ≤ 1 := div_le_one_of_le₀ (by linarith) (by linarith)
  have hK := DeployedSession.pr_draws_window_le A ⌊(1 - dw) * ((P * N : ℕ) : ℝ)⌋₊
  have h2 := Deployed.window_le dw n (P * N) (d * n) hn (by positivity) hdw
  rw [show ((n : ℝ≥0∞) * ((P * N : ℕ) : ℝ≥0∞)) = n * P * N by push_cast; ring] at h2
  have hB := hK.trans h2
  have hT : ENNReal.ofReal (1 - dw / n) ^ (d * n) ≠ ⊤ := ENNReal.pow_ne_top ENNReal.ofReal_ne_top
  have e : (ENNReal.ofReal (1 - dw / n) ^ (d * n)).toReal = (1 - dw / n) ^ (d * n) := by
    rw [ENNReal.toReal_pow, ENNReal.toReal_ofReal (by linarith)]
  unfold Run.Sampled Run.Holds
  rw [← e]
  refine pouw_pr_le ((Bandwidth.pr_mono fun x hx => ?_).trans hB) hT
  obtain ⟨hacc, hlt⟩ := hx
  refine ⟨hacc, w, Nat.le_floor ?_⟩
  simp only [Run.ofDeployedWindow, hab, hC, not_le] at hlt
  push_cast at hlt ⊢
  have : (((A.win w x.2).correctAll x.1 (x.2 w).1).card : ℝ) * b < (1 - dw) * (P * N) * b := by linarith
  exact (lt_of_mul_lt_mul_right this hb'.le).le

/-- **Paid** for window `w`'s run: `RatePaidNamedRO` at each of its passes and candidates. -/
theorem ofDeployedWindow_paid_route (A : DeployedSession n P N b h ρ S L q r s d G) (w : Fin n) (tier : ℕ → Fin τ)
    (Nd : Node (Fin τ)) (a : Audit) {lam rounds cRoot F : ℕ} {cost : ℕ → ℕ} (hrc : 0 < rounds * cRoot)
    (ha : a.P = P) (hab : a.b = b) (hal : a.lam = lam)
    (hW : Accounting.Assumptions.WorkHardness b rounds cRoot cost)
    (hF : ∀ w η j ι c, Accounting.Assumptions.SquaringCeiling cost F ((A.window w η).device j ι c))
    (hA : ∀ w η j ι c E R O O', ((A.window w η).device j ι c).side E R O = ((A.window w η).device j ι c).side E R O') :
    (Run.ofDeployedWindow A w tier (Deployed.rebuildBits b rounds cRoot F) Nd a).Paid
      ((P * G : ℝ) * ((N : ℝ) / 2 ^ h + (q.choose 2 : ℝ) / 2 ^ h + (2 ^ N : ℝ) / 2 ^ lam)) := by
  classical
  set X := Deployed.rebuildBits b rounds cRoot F
  have hX : ∀ w η j ι c E R O, ((A.window w η).device j ι c).completed E R O ≤ X :=
    fun w η j ι c E R O => PIDevice.completed_le _ hrc hW (hF w η j ι c) E R O
  have hε := fun w η j ι c => PIDevice.pr_unpaid_le ((A.window w η).device j ι c) tier X lam (hA w η j ι c)
  have h₂ := DeployedSession.pr_window_le A w
    (fun C y => C.reads tier y.1 y.2.1 + P * (S + (L - ρ) + X + lam) < (C.correctAll y.1 y.2.1).card * b) _
    (fun η => (A.window w η).pr_unpaid_draw_le tier X _ d _ (hX w η) (hε w η))
  have hT : (P : ℝ≥0∞) * (G * ((N : ℝ≥0∞) / 2 ^ h + (q.choose 2 : ℝ≥0∞) / 2 ^ h + (2 ^ N : ℝ≥0∞) / 2 ^ lam)) ≠ ⊤ :=
    ENNReal.mul_ne_top (ENNReal.natCast_ne_top P)
      (ENNReal.mul_ne_top (ENNReal.natCast_ne_top G) (roErr_ne_top N h q lam))
  unfold Run.Paid Run.Holds
  rw [← roErr_toReal_window]
  refine pouw_pr_le ((Bandwidth.pr_mono fun x hx => ?_).trans h₂) hT
  obtain ⟨-, hlt⟩ := hx
  have hc : (Run.ofDeployedWindow A w tier X Nd a).correct x = ((A.win w x.2).correctAll x.1 (x.2 w).1).card := rfl
  rw [not_le, ofDeployedWindow_paid, ofDeployedWindow_reads, hc, ha, hab, hal] at hlt
  show (A.win w x.2).reads tier x.1 (x.2 w).1 + P * (S + (L - ρ) + X + lam) <
    ((A.win w x.2).correctAll x.1 (x.2 w).1).card * b
  have e : P * (S + (L - ρ) + X + lam) = P * (L - ρ) + P * S + P * X + P * lam := by ring
  omega

end Accounting.Remote

namespace Accounting.SecurityProofs

open Accounting.Rate Accounting.Remote

/-- **The deployed session meets the remote audit's routes** (`Accounting.Guarantees.RemoteSessionRoutes`). -/
theorem RemoteSessionRoutes : Accounting.Guarantees.RemoteSessionRoutes := by
  intro n P N b h ρ S L q r s d G lam τ rounds cRoot F cost tier A Nd a dbar hn hP hN hb hrc hdbar ha hab hal hC hW hF hA
  exact ⟨ofDeployed_sampled A tier _ Nd a dbar hn hP hN hb hdbar hab hC,
    ofDeployed_paid_route A tier Nd a hrc ha hab hal hW hF hA⟩

/-- **The deployed session conserves its words as a run of the remote audit, at the same `γ`**
(`Accounting.Guarantees.RemoteSessionRunConservation`), from `Accounting.SecurityProofs.Remote.Conservation`. -/
theorem RemoteSessionRunConservation : Accounting.Guarantees.RemoteSessionRunConservation := by
  intro n P N b h ρ S L q r s d G lam τ rounds cRoot F cost tier A Nd a dbar hn hP hN hb hrc hdbar ha hab hal hC hW hF hA
  obtain ⟨hs, hp⟩ := RemoteSessionRoutes n P N b h ρ S L q r s d G lam τ rounds cRoot F cost tier A Nd a dbar hn hP hN
    hb hrc hdbar ha hab hal hC hW hF hA
  have hC0 : 0 < Nd.declared a.T := by rw [hC]; positivity
  refine Remote.holds_mono ?_ (Remote.Conservation (Fin τ) _ Nd a _ dbar _ _ hC0 hs hp)
  rintro x - ⟨h1, h2⟩
  refine ⟨h1, ?_⟩
  rw [ofDeployed_paid] at h2
  have hCr : (Nd.declared a.T : ℝ) = n * P * N * b := by rw [hC]; push_cast; ring
  have eΛ : muLambda Nd a * Nd.declared a.T = a.P * a.lam := by
    have : (0 : ℝ) < Nd.declared a.T := by exact_mod_cast hC0
    unfold muLambda; field_simp
  rw [ha, hal] at eΛ
  have hγ := Deployed.gammaPass_mul N b ρ S L lam rounds cRoot F hN hb
  have k : Deployed.gammaPass N b ρ S L lam rounds cRoot F * Nd.declared a.T =
      n * P * ((S + (L - ρ) + Deployed.rebuildBits b rounds cRoot F + lam : ℕ) : ℝ) := by
    rw [hCr, ← hγ]; ring
  push_cast at h2 eΛ k
  rw [hCr] at h2 eΛ k ⊢
  nlinarith [h2, eΛ, k]

/-- **Each window conserves its words as a run of the remote audit, at the same `γ_w`**
(`Accounting.Guarantees.RemoteWindowRunConservation`), from `Accounting.SecurityProofs.Remote.Conservation`. -/
theorem RemoteWindowRunConservation : Accounting.Guarantees.RemoteWindowRunConservation := by
  intro n P N b h ρ S L q r s d G lam τ rounds cRoot F cost tier A w Nd a dw hP hN hb hrc hdw ha hab hal hC hW hF hA
  have hs := ofDeployedWindow_sampled A w tier (Deployed.rebuildBits b rounds cRoot F) Nd a dw hP hN hb hdw hab hC
  have hp := ofDeployedWindow_paid_route A w tier Nd a hrc ha hab hal hW hF hA
  have hC0 : 0 < Nd.declared a.T := by rw [hC]; positivity
  refine Remote.holds_mono ?_ (Remote.Conservation (Fin τ) _ Nd a _ dw _ _ hC0 hs hp)
  rintro x - ⟨h1, h2⟩
  refine ⟨h1, ?_⟩
  rw [ofDeployedWindow_paid] at h2
  have hCr : (Nd.declared a.T : ℝ) = P * N * b := by rw [hC]; push_cast; ring
  have eΛ : muLambda Nd a * Nd.declared a.T = a.P * a.lam := by
    have : (0 : ℝ) < Nd.declared a.T := by exact_mod_cast hC0
    unfold muLambda; field_simp
  rw [ha, hal] at eΛ
  have hγ := Deployed.gammaPass_mul N b ρ S L lam rounds cRoot F hN hb
  have k : Deployed.gammaPass N b ρ S L lam rounds cRoot F * Nd.declared a.T =
      P * ((S + (L - ρ) + Deployed.rebuildBits b rounds cRoot F + lam : ℕ) : ℝ) := by
    rw [hCr, ← hγ]; ring
  push_cast at h2 eΛ k
  rw [hCr] at h2 eΛ k ⊢
  nlinarith [h2, eΛ, k]

end Accounting.SecurityProofs
