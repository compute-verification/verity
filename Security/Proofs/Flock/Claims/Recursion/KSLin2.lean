import Proofs.Flock.Claims.Recursion.VCalls

/-!
# Route 2C: `KSLin2` from `KSFlockLin` at `DE`

`Bound.innerLin_of_plainKSAbort_binds` reads (P1) at `DE`, `Inner.KSAbort`, and pays its root: `√(γ_BI·Pr[B_I])`.
Here the inner stage reads `KSFlockLin` at `DE` instead, against `P_O`'s aborting runs, and each of its terms is carried
to the composite with no root:

* `B` at `P_O`'s aborting runs is at most `B_I` over the draws' evaluations (`Bound.PrBAbort_le_PrBinnerV`), so
  `4k_I·Pr[B]` is at most `4k_I·Pr[B_I]`, and a rejecting judged draw one more `Pr[B_I]`
  (`Pr_innerRejects_le_PrBinnerV_at`).
* The caps' link `FC` at `P_O`'s aborting runs is the composite's `FCI` over `E_O`'s draws, but where the loop meets a
  draw `P_O`'s runs don't show (`BadO`) (`Bound.PrFC_le_FCI`). A run that opens is kept, so its draw isn't bad; off the
  bad draws a run opens exactly where its draw's inner view accepts and its inner coins query the position
  (`Bound.opensAt_iff`), the loop's picks are the same (`firstPick_map_of_loopCost_zero`), and the runs' calls are
  among the draws' witnesses' or those collide (`Bound.findColl_cover`). The loop meets a bad draw at most as often as
  one draw is bad (`Pr_gate_loop_fork_le`, Wald's bound on the loop with the forked coins uniform).
* Level 0's conflict `F0` is `F0I` the same way, its `k_I` waits each meeting a bad draw at most as often as one draw
  is bad (`Bound.PrF0_le_F0I`).

So the bad draws cost `N_I + 4·W_I + 4·k_I` draws' worth, each at most `k_I·(Pr[FdevV] + 1/(M+1))`
(`Bound.Pr_badO_le_PrFdevV`), within `β₂·(Pr[FdevV] + 1/(M+1))` (`Bound.innerLin2_of_flockLin_binds`). A V\* part's
check escapes to a collision with the compiled verifier's own evaluations at its V\* statement, so the facts read of
V*'s parts are `InnerAcceptsCollAtV` and `ForksBindAtV`, and `B_I` and `Fdev` search the draws' evaluations
(`BinnerV`, `FdevV`).
`Bound.ksLin2_of_flockLin_binds` is the recursion's bound from it, as `KSLin_of` (`PrBad_le_of_inner`).
-/

namespace Claims.Recursion

open Claims

section Loops

variable {U : Type}

/-- Where `c` is `0` on every draw the loop makes, and `g` agrees with `g'` through `f` wherever `c` is `0`, the loops
pick the same draw. -/
theorem firstPick_map_of_loopCost_zero {U' : Type} (f : U → U') (g : U → Prop) (g' : U' → Prop) (c : U → ℕ)
    (h : ∀ u, c u = 0 → (g u ↔ g' (f u))) :
    ∀ (M : ℕ) (us : Fin M → U), loopCost g c M us = 0 →
      (firstPick g M us).map f = firstPick g' M (fun i => f (us i))
  | 0, _, _ => rfl
  | M + 1, us, h0 => by
    simp only [loopCost] at h0
    obtain ⟨hc, hr⟩ := Nat.add_eq_zero_iff.1 h0
    have hgg := h (us 0) hc
    by_cases hg : g (us 0)
    · have hg' : g' (f (us 0)) := hgg.1 hg
      simp [firstPick, hg, hg']
    · have hg' : ¬ g' (f (us 0)) := fun h' => hg (hgg.2 h')
      rw [if_neg hg] at hr
      simp only [firstPick, hg, hg', ↓reduceIte]
      exact firstPick_map_of_loopCost_zero f g g' c h M (Fin.tail us) hr

theorem optionGetD_map {α β : Type} (f : α → β) (o : Option α) (a : α) : (o.map f).getD (f a) = f (o.getD a) := by
  cases o <;> rfl

open Classical in
/-- **Wald's bound on a gated loop**: the loop's expected `c`, where the first draw passes `g`, is at most one
draw's. -/
theorem Ex_gate_loopCost_le [Fintype U] [Nonempty U] (g : U → Prop) (c : U → ℕ) (M : ℕ) :
    Ex (fun p : U × (Fin M → U) => if g p.1 then (loopCost g c M p.2 : ℝ) else 0) ≤ Ex (fun u => (c u : ℝ)) := by
  have h1 : Ex (fun p : U × (Fin M → U) => if g p.1 then (loopCost g c M p.2 : ℝ) else 0) =
      Ex (fun u => if g u then Ex (fun vs : Fin M → U => (loopCost g c M vs : ℝ)) else 0) := by
    rw [Ex_prod]
    congr 1
    funext u
    by_cases hu : g u
    · simp only [hu, ↓reduceIte]
    · simp only [hu, ↓reduceIte, Ex_const]
  rw [h1, Ex_if_zero, Ex_loopCost]
  have hc : 0 ≤ Ex (fun u => (c u : ℝ)) := Ex_nonneg fun u => Nat.cast_nonneg _
  have hq0 : 0 ≤ Pr g := Pr_nonneg _
  have hq1 : Pr g ≤ 1 := Pr_le_one _
  have hgm := geomS_mul (Pr g) M
  have ht : 0 ≤ (1 - Pr g) ^ M := pow_nonneg (by linarith) M
  have he : Ex (fun u => (c u : ℝ)) * geomS (1 - Pr g) M * Pr g =
      Ex (fun u => (c u : ℝ)) * (1 - (1 - Pr g) ^ M) := by
    rw [← hgm]; ring
  nlinarith [mul_nonneg hc ht, he]

open Classical in
/-- A gated loop makes a draw of positive `c` at most as often as one draw's expected `c`. -/
theorem Pr_gate_loop_le [Fintype U] [Nonempty U] (g : U → Prop) (c : U → ℕ) (M : ℕ) :
    Pr (fun p : U × (Fin M → U) => g p.1 ∧ loopCost g c M p.2 ≠ 0) ≤ Ex (fun u => (c u : ℝ)) := by
  rw [← Ex_ind]
  refine (Ex_mono fun p => ?_).trans (Ex_gate_loopCost_le g c M)
  by_cases hp : g p.1 ∧ loopCost g c M p.2 ≠ 0
  · rw [if_pos hp, if_pos hp.1]
    exact_mod_cast Nat.one_le_iff_ne_zero.2 hp.2
  · rw [if_neg hp]
    split_ifs
    · exact Nat.cast_nonneg _
    · exact le_rfl

/-- A draw forked at `j` from a uniform prefix is uniform. -/
theorem unifMap_forkPair {Coin A : Type} [Fintype Coin] [Nonempty Coin] [Fintype A] {n : ℕ} (j : ℕ) :
    UnifMap (fun p : (Fin n → Coin) × ((Fin n → Coin) × A) => ((fork j p.1 p.2.1, p.2.2) : (Fin n → Coin) × A)) := by
  let e : (Fin n → Coin) × ((Fin n → Coin) × A) ≃ ((Fin n → Coin) × (Fin n → Coin)) × A :=
    { toFun := fun p => ((p.1, p.2.1), p.2.2)
      invFun := fun q => (q.1.1, q.1.2, q.2)
      left_inv := fun _ => rfl
      right_inv := fun _ => rfl }
  have hf : UnifMap (fun q : (Fin n → Coin) × (Fin n → Coin) => fork j q.1 q.2) :=
    fun G => Ex_fork (off := fun _ => j) (fun _ _ _ => rfl) G
  exact (UnifMap.equiv e).comp (hf.prod UnifMap.self)

/-- **Wald's bound on a gated loop over forked draws**: the loop's draws, forked at `j` from the first, make a draw of
positive `c` at most as often as one draw's expected `c`. -/
theorem Pr_gate_loop_fork_le {Coin A : Type} [Fintype Coin] [Nonempty Coin] [Fintype A] [Nonempty A] {n : ℕ}
    (j : ℕ) (g : (Fin n → Coin) × A → Prop) (c : (Fin n → Coin) × A → ℕ) (M : ℕ) :
    Pr (fun p : ((Fin n → Coin) × A) × (Fin M → (Fin n → Coin) × A) =>
        g p.1 ∧ loopCost (fun v => g (fork j p.1.1 v.1, v.2)) (fun v => c (fork j p.1.1 v.1, v.2)) M p.2 ≠ 0) ≤
      Ex (fun u => (c u : ℝ)) := by
  let e : (Fin n → Coin) × (((Fin n → Coin) × A) × (Fin M → (Fin n → Coin) × A)) ≃
      ((Fin n → Coin) × ((Fin n → Coin) × A)) × (Fin M → (Fin n → Coin) × A) :=
    { toFun := fun a => ((a.1, a.2.1), a.2.2)
      invFun := fun b => (b.1.1, b.1.2, b.2)
      left_inv := fun _ => rfl
      right_inv := fun _ => rfl }
  have hΨ : UnifMap (fun a : (Fin n → Coin) × (((Fin n → Coin) × A) × (Fin M → (Fin n → Coin) × A)) =>
      (((fork j a.1 a.2.1.1, a.2.1.2), a.2.2) : ((Fin n → Coin) × A) × (Fin M → (Fin n → Coin) × A))) :=
    (UnifMap.equiv e).comp ((unifMap_forkPair j).prod UnifMap.self)
  have h1 : ∀ a : (Fin n → Coin) × (((Fin n → Coin) × A) × (Fin M → (Fin n → Coin) × A)),
      (g (fork j a.1 a.2.1.1, a.2.1.2) ∧ loopCost (fun v => g (fork j (fork j a.1 a.2.1.1) v.1, v.2))
        (fun v => c (fork j (fork j a.1 a.2.1.1) v.1, v.2)) M a.2.2 ≠ 0) ↔
      (g (fork j a.1 a.2.1.1, a.2.1.2) ∧ loopCost (fun v => g (fork j a.1 v.1, v.2))
        (fun v => c (fork j a.1 v.1, v.2)) M a.2.2 ≠ 0) := fun a => by
    simp only [fork_fork]
  calc Pr (fun p : ((Fin n → Coin) × A) × (Fin M → (Fin n → Coin) × A) =>
        g p.1 ∧ loopCost (fun v => g (fork j p.1.1 v.1, v.2)) (fun v => c (fork j p.1.1 v.1, v.2)) M p.2 ≠ 0)
      = Pr (fun a : (Fin n → Coin) × (((Fin n → Coin) × A) × (Fin M → (Fin n → Coin) × A)) =>
        g (fork j a.1 a.2.1.1, a.2.1.2) ∧ loopCost (fun v => g (fork j a.1 v.1, v.2))
          (fun v => c (fork j a.1 v.1, v.2)) M a.2.2 ≠ 0) := (hΨ.Pr _).symm.trans (Pr_congr h1)
    _ = Ex fun ρ : Fin n → Coin => Pr fun b : ((Fin n → Coin) × A) × (Fin M → (Fin n → Coin) × A) =>
        g (fork j ρ b.1.1, b.1.2) ∧ loopCost (fun v => g (fork j ρ v.1, v.2))
          (fun v => c (fork j ρ v.1, v.2)) M b.2 ≠ 0 := Pr_prod _
    _ ≤ Ex fun ρ : Fin n → Coin => Ex fun u : (Fin n → Coin) × A => (c (fork j ρ u.1, u.2) : ℝ) := by
        refine Ex_mono fun ρ => ?_
        exact Pr_gate_loop_le (U := (Fin n → Coin) × A) (fun v => g (fork j ρ v.1, v.2))
          (fun v => c (fork j ρ v.1, v.2)) M
    _ = Ex (fun u => (c u : ℝ)) :=
        (Ex_prod (fun q : (Fin n → Coin) × ((Fin n → Coin) × A) => (c (fork j q.1 q.2.1, q.2.2) : ℝ))).symm.trans
          (unifMap_forkPair j (fun u => (c u : ℝ)))

/-- An inner run's coins forked at `r`, extended, are the extended coins forked at `r` capped at the inner rounds. -/
theorem extend_fork_min {Coin : Type} [Inhabited Coin] {n N : ℕ} (r : ℕ) (ρ ρ' : Fin n → Coin)
    (a a' : Fin N → Coin) : extend (fork r ρ ρ') a' = fork (min r n) (extend ρ a) (extend ρ' a') := by
  funext i
  simp only [extend, fork, coinSeq, lt_min_iff]
  split_ifs <;> first | rfl | omega

end Loops

/-! ## The recursion's bound from an inner stage's -/

section Outer

variable {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

/-- **The recursion's bound from any bound on its inner stage**: `KSLin_of`'s split, at one statement and prover, the
outer sessions' `κ + √(γ·Pr[B])` added to whatever the inner stage pays past `κ_I`. -/
theorem PrBad_le_of_inner [Zero Col] (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W)
    (DO : Flock.Deploy Coin Col T) (hRef : Refuses H D) (hO : OuterKS H D I DO) (decO : List (ℕ × Col) → List T)
    (hdec : decOK DO decO) (x : Msg) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (Q : ℝ)
    (hin : PrBadGood H D.V (innerR D I) outR (EO D I DO decO) (dec H D) (fun x => I.waits (xI D x))
        (fun x => I.k (xI D x)) (fun x => I.cap (xI D x)) I.opens (fun x => I.assemble (xI D x))
        (fun x => I.check (xI D x)) x P ≤ I.κ (xI D x) + Q) :
    PrBad H D.V (R D I) (E H D I DO decO) x P ≤ κ D I x + Real.sqrt (γ D I x * PrB H D.V P x) + Q := by
  set A : ℝ := (NI D I x : ℝ)
  set p := PrB H D.V P x
  set G := ∑ s : Fin (D.nS x), Real.sqrt ((D.nums x s).γ * D.V.k x)
  have hA : (1 + (I.waits (xI D x) : ℝ) * (I.k (xI D x) : ℝ)) = A := by simp only [A, NI]; push_cast; ring
  have hA0 : 0 ≤ A := Nat.cast_nonneg _
  have hp0 : 0 ≤ p := PrB_nonneg H D.V P x
  have hG0 : 0 ≤ G := Finset.sum_nonneg fun s _ => Real.sqrt_nonneg _
  have hsplit := servedERep_bad_le_good H D.V (R D I) outR (EO D I DO decO) (dec H D) (fun x => I.waits (xI D x))
    (fun x => I.k (xI D x)) (fun x => I.cap (xI D x)) I.opens (fun x => I.assemble (xI D x))
    (fun x => I.check (xI D x)) x P
  rw [hA] at hsplit
  have hgood : PrBadGood H D.V (R D I) outR (EO D I DO decO) (dec H D) (fun x => I.waits (xI D x))
      (fun x => I.k (xI D x)) (fun x => I.cap (xI D x)) I.opens (fun x => I.assemble (xI D x))
      (fun x => I.check (xI D x)) x P ≤
      PrBadGood H D.V (innerR D I) outR (EO D I DO decO) (dec H D) (fun x => I.waits (xI D x))
      (fun x => I.k (xI D x)) (fun x => I.cap (xI D x)) I.opens (fun x => I.assemble (xI D x))
      (fun x => I.check (xI D x)) x P := by
    refine Pr_mono fun t h => ⟨h.1, h.2.1, fun hi => h.2.2 ?_⟩
    obtain ⟨s, hs, hc⟩ := hRef x Ω P t.1 t.2.1 h.1
    exact R_of_innerR D I hs hc _ hi
  have hEO : PrBad H D.V outR (EO D I DO decO) x P ≤
      ∑ s : Fin (D.nS x), (D.nums x s).κ + G * Real.sqrt p := by
    refine (PrBad_EO_le H D I DO decO x P).trans ?_
    rw [Finset.sum_mul, ← Finset.sum_add_distrib]
    refine Finset.sum_le_sum fun s _ => (hO decO hdec x Ω P s).trans (le_of_eq ?_)
    rw [Real.sqrt_mul (mul_nonneg (Nums.γ_nonneg _) (Nat.cast_nonneg _))]
  have hB : A * (G * Real.sqrt p) = Real.sqrt (γ D I x * p) := by
    have hγ : γ D I x = (A * G) ^ 2 := rfl
    rw [hγ, Real.sqrt_mul' _ hp0, Real.sqrt_sq (mul_nonneg hA0 hG0), mul_assoc]
  have hκ : κ D I x = I.κ (xI D x) + A * ∑ s : Fin (D.nS x), (D.nums x s).κ := rfl
  show PrBad H D.V (R D I) (served D.V (EO D I DO decO) (dec H D) (fun x => I.waits (xI D x))
    (fun x => I.k (xI D x)) (fun x => I.cap (xI D x)) I.opens (fun x => I.assemble (xI D x))
    (fun x => I.check (xI D x))) x P ≤ _
  rw [served_eq, hκ]
  have hmul := mul_le_mul_of_nonneg_left hEO hA0
  nlinarith [hsplit, hgood, hin, hmul, hB]

end Outer

/-! ## `P_O`'s aborting runs, as the composite's draws -/

section Coupling

variable {Coin Col T Col' T' : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
  [DecidableEq Col'] {Bd : Bound Coin Col T} {decI : List (ℕ × Col) → List T} {H : Msg → Msg} {D : Deploy Coin}
  {DO : Flock.Deploy Coin Col' T'} {decO : List (ℕ × Col') → List T'} {Ω : Type} {P : Prover Ω} {x : Msg}

omit [DecidableEq Col'] in
/-- `P_O`'s run at a draw accepts exactly where it is kept. -/
theorem Bound.innerFO_acc_iff (decC : Msg → Msg → Coin) (hdec : ∀ x c, decC x (Bd.ans c) = c)
    (hV : InnerViewRun H D (Inner.ofFlock Bd.deploy decI) DO) (ω : Ω)
    (d : (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) :
    (Bd.innerFO decI H D DO decO P x decC ω d).out = .acc ↔ Bd.keepDO decI H D DO decO P x ω d = true := by
  constructor
  · intro h
    by_contra hk
    simp [Bound.innerFO, hk, abortTrace] at h
  · intro hk
    have hk' : (eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x ω d).isSome = true ∧
        (dec H D x d.1 (eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x ω d)).out = .acc ∧
        Bd.AgreesO decI H D DO decO P x ω d := by
      simpa only [Bound.keepDO, decide_eq_true_eq] using hk
    obtain ⟨ws, hws⟩ := Option.isSome_iff_exists.1 hk'.1
    have hacc : (D.innerView H x d.1 ws).out = .acc := by
      have h := hk'.2.1
      rw [hws] at h
      exact h
    unfold Bound.innerFO
    rw [if_pos hk]
    exact (Bound.runO decC hdec hV ω d hws hacc hk'.2.2).2.1

omit [DecidableEq Col] [DecidableEq Col'] in
/-- A kept draw is not bad. -/
theorem Bound.not_badO_of_keepDO {ω : Ω}
    {d : (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x}
    (h : Bd.keepDO decI H D DO decO P x ω d = true) : ¬ Bd.BadO decI H D DO decO P x ω d := fun hb => by
  simp only [Bound.keepDO, decide_eq_true_eq] at h
  exact hb.2 h.2.2

omit [DecidableEq Col] [DecidableEq Col'] in
/-- Off the bad draws, a draw is kept exactly where its inner view accepts. -/
theorem Bound.keepDO_iff_accAtI {ω : Ω}
    {d : (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x}
    (hb : ¬ Bd.BadO decI H D DO decO P x ω d) :
    Bd.keepDO decI H D DO decO P x ω d = true ↔ accAtI H D (Inner.ofFlock Bd.deploy decI) DO decO P x ω d := by
  simp only [Bound.keepDO, decide_eq_true_eq]
  change _ ↔ (dec H D x d.1 (eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x ω d)).out = .acc
  constructor
  · exact fun h => h.2.1
  · intro h
    have hs : (eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x ω d).isSome = true := by
      rcases ho : eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x ω d with _ | ws
      · rw [ho] at h
        simp [dec] at h
      · rfl
    exact ⟨hs, h, Classical.byContradiction fun hn => hb ⟨hs, hn⟩⟩

omit [DecidableEq Col'] in
/-- **Off the bad draws, `P_O`'s aborting run accepts exactly where its draw's inner view does.** -/
theorem Bound.accAt_iff (decC : Msg → Msg → Coin) (hdec : ∀ x c, decC x (Bd.ans c) = c)
    (hV : InnerViewRun H D (Inner.ofFlock Bd.deploy decI) DO) (hn : Bd.DE.V.k (xI D x) ≤ D.V.k x) (ω : Ω)
    (t : (Fin (Bd.DE.V.k (xI D x)) → Coin) × (Fin (D.V.k x) → Coin) ×
      (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x)
    (hb : ¬ Bd.BadO decI H D DO decO P x ω (extendDraw t)) :
    accAt H Bd.DE.V (Bd.innerPO decI H D DO decO P x decC) (xI D x) (Bd.keepAO decI H D DO decO P x) ω t ↔
      accAtI H D (Inner.ofFlock Bd.deploy decI) DO decO P x ω (extendDraw t) := by
  unfold accAt
  rw [Bound.abortRun_eqO decC hn, Bound.innerFO_acc_iff decC hdec hV, Bound.keepDO_iff_accAtI hb]

omit [DecidableEq Col'] in
/-- **Off the bad draws, `P_O`'s aborting run opens a cap position exactly where its draw does**: the run's coins are
the draw's inner coins (`innerCoins_extend`). -/
theorem Bound.opensAt_iff (decC : Msg → Msg → Coin) (hdec : ∀ x c, decC x (Bd.ans c) = c)
    (hV : InnerViewRun H D (Inner.ofFlock Bd.deploy decI) DO) (hn : Bd.DE.V.k (xI D x) ≤ D.V.k x)
    (Cp : Caps Coin) (q : Fin (Cp.width (xI D x))) (ω : Ω)
    (t : (Fin (Bd.DE.V.k (xI D x)) → Coin) × (Fin (D.V.k x) → Coin) ×
      (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x)
    (hb : ¬ Bd.BadO decI H D DO decO P x ω (extendDraw t)) :
    opensAt H Bd.DE.V Cp (Bd.innerPO decI H D DO decO P x decC) (xI D x) (Bd.keepAO decI H D DO decO P x) q ω t ↔
      opensAtI H D (Inner.ofFlock Bd.deploy decI) DO decO Cp P x q ω (extendDraw t) := by
  have hc : (fun i : Fin ((Inner.ofFlock Bd.deploy decI).V.k (xI D x)) => coinSeq (extendDraw t).1 i.val) = t.1 :=
    innerCoins_extend D (Inner.ofFlock Bd.deploy decI) x hn t.1 t.2.1
  unfold opensAt opensAtI
  rw [hc]
  exact and_congr (Bound.accAt_iff decC hdec hV hn ω t hb) Iff.rfl

omit [DecidableEq Col'] in
/-- **A collision among `P_O`'s aborting runs' calls is one among the calls searched at the draws**, where every run's
call is among some draw's searched calls (`Deploy.callsAt`) that the list holds: each run's calls are among its draw's
or those collide (`innerFO_vLog_subBy`, `CallsLogCollAt`). -/
theorem Bound.findColl_cover (decC : Msg → Msg → Coin) (hdec : ∀ x c, decC x (Bd.ans c) = c)
    (hV : InnerViewRun H D (Inner.ofFlock Bd.deploy decI) DO) (hL : CallsLogCollAt H D (Inner.ofFlock Bd.deploy decI) DO)
    (ω : Ω) {L L' : List (Msg × Msg)} (ht : ∀ p ∈ L', H p.1 = p.2)
    (hc : ∀ e ∈ L, ∃ d, e ∈ (Bd.innerFO decI H D DO decO P x decC ω d).vLog ∧
      ∀ p ∈ D.callsAt H x d.1 ((eoOut H D (Inner.ofFlock Bd.deploy decI) DO decO P x ω d).getD []), p ∈ L')
    (h : (findColl L).isSome) : ∃ u v, findColl L' = some (u, v) ∧ Collision H u v := by
  have hs : (findColl L').isSome := by
    by_contra hn
    refine hn (findColl_isSome_of_sub H ht (fun e he => ?_) h)
    obtain ⟨d, hed, hsub⟩ := hc e he
    rcases Bound.innerFO_vLog_subBy decC hdec hV hL ω d with h1 | ⟨u, v, huv, -⟩
    · exact hsub e (h1 e hed)
    · exact absurd (findColl_isSome_of_sub H ht hsub (Option.isSome_iff_exists.2 ⟨_, huv⟩)) hn
  obtain ⟨⟨u, v⟩, huv⟩ := Option.isSome_iff_exists.1 hs
  exact ⟨u, v, huv, findColl_spec H _ ht u v huv⟩

omit [DecidableEq Col] [DecidableEq Col'] in
/-- A run's coins forked at a cap's round, extended, are the draws forked at that round (capped at the inner
rounds). -/
theorem Bound.extendDraw_capFork (Cp : Caps Coin) (q : Fin (Cp.width (xI D x)))
    (d₀ v : (Fin (Bd.DE.V.k (xI D x)) → Coin) × (Fin (D.V.k x) → Coin) ×
      (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) :
    extendDraw (capFork Bd.DE.V Cp (xI D x) q d₀ v) =
      capForkI D (Inner.ofFlock Bd.deploy decI) DO decO Cp x q (extendDraw d₀) (extendDraw v) :=
  Prod.ext (extend_fork_min (Cp.round (xI D x) q) d₀.1 v.1 d₀.2.1 v.2.1) rfl

omit [DecidableEq Col] [DecidableEq Col'] in
open Classical in
/-- One extended draw is bad as often as one draw. -/
theorem Bound.Ex_badO (hR : InnerRounds D (Inner.ofFlock Bd.deploy decI)) [Fintype Ω] (ω : Ω) :
    Ex (fun u : (Fin (Bd.DE.V.k (xI D x)) → Coin) × (Fin (D.V.k x) → Coin) ×
        (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x =>
      ((if Bd.BadO decI H D DO decO P x ω (extendDraw u) then 1 else 0 : ℕ) : ℝ)) =
      Pr (fun d : (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x =>
        Bd.BadO decI H D DO decO P x ω d) := by
  have hn : Bd.DE.V.k (xI D x) ≤ D.V.k x := hR x
  have h1 := (unifMap_extendDraw (C := (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) hn).Pr
    (fun d => Bd.BadO decI H D DO decO P x ω d)
  rw [← h1, ← Ex_ind]
  refine congrArg Ex (funext fun u => ?_)
  by_cases hb : Bd.BadO decI H D DO decO P x ω (extendDraw u) <;> simp [hb]

omit [DecidableEq Col'] in
/-- **The caps' link at `P_O`'s aborting runs is `FCI`, but where its loop meets a bad draw**: the first run, opening,
is kept, so its draw isn't bad; off the bad draws the loop opens where the draws do and picks the same draw
(`opensAt_iff`, `firstPick_map_of_loopCost_zero`), and the runs' collision is the draws' (`findColl_cover`). The loop
meets a bad draw at most as often as one draw is bad (`Pr_gate_loop_fork_le`). -/
theorem Bound.PrFC_le_FCI (decC : Msg → Msg → Coin) (hdec : ∀ x c, decC x (Bd.ans c) = c)
    (hV : InnerViewRun H D (Inner.ofFlock Bd.deploy decI) DO) (hL : CallsLogCollAt H D (Inner.ofFlock Bd.deploy decI) DO)
    (hT : CallsTrue H D (Inner.ofFlock Bd.deploy decI) DO) (hN0 : CallsNil H D)
    (hR : InnerRounds D (Inner.ofFlock Bd.deploy decI)) [Fintype Ω] [Nonempty Ω] (Cp : Caps Coin) (M : ℕ) :
    PrFC H Bd.DE.V Cp (Bd.innerPO decI H D DO decO P x decC) (xI D x) (Bd.keepAO decI H D DO decO P x) M ≤
      PrFCI H D (Inner.ofFlock Bd.deploy decI) DO decO Cp P x M +
        Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x =>
          Bd.BadO decI H D DO decO P x t.1 t.2) := by
  classical
  have hn : Bd.DE.V.k (xI D x) ≤ D.V.k x := hR x
  unfold PrFC PrFCI
  by_cases hW : 0 < Cp.width (xI D x)
  swap
  · simp only [hW, ↓reduceDIte, zero_add]
    exact Pr_nonneg _
  simp only [hW, ↓reduceDIte]
  unfold Finder.Pr
  have key : ∀ s : CapCoins Bd.DE.V Cp Ω ((Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x)
      (xI D x) M,
      (∃ u v, (FC H Bd.DE.V Cp (Bd.innerPO decI H D DO decO P x decC) (xI D x) (Bd.keepAO decI H D DO decO P x) M
        hW).out s = some (u, v) ∧ Collision H u v) →
      (∃ u v, (FCI H D (Inner.ofFlock Bd.deploy decI) DO decO Cp P x M hW).out
          (s.1, s.2.1, extendDraw s.2.2.1, fun i => extendDraw (s.2.2.2 i)) = some (u, v) ∧ Collision H u v) ∨
        (opensAt H Bd.DE.V Cp (Bd.innerPO decI H D DO decO P x decC) (xI D x) (Bd.keepAO decI H D DO decO P x)
            s.1 s.2.1 s.2.2.1 ∧
          loopCost (fun v => opensAt H Bd.DE.V Cp (Bd.innerPO decI H D DO decO P x decC) (xI D x)
              (Bd.keepAO decI H D DO decO P x) s.1 s.2.1 (capFork Bd.DE.V Cp (xI D x) s.1 s.2.2.1 v))
            (fun v => if Bd.BadO decI H D DO decO P x s.2.1
              (extendDraw (capFork Bd.DE.V Cp (xI D x) s.1 s.2.2.1 v)) then 1 else 0) M s.2.2.2 ≠ 0) := by
    rintro s ⟨a, b, hout, -⟩
    have hop : opensAt H Bd.DE.V Cp (Bd.innerPO decI H D DO decO P x decC) (xI D x)
        (Bd.keepAO decI H D DO decO P x) s.1 s.2.1 s.2.2.1 := by
      by_contra hop
      simp [FC, hop] at hout
    by_cases hz : loopCost (fun v => opensAt H Bd.DE.V Cp (Bd.innerPO decI H D DO decO P x decC) (xI D x)
        (Bd.keepAO decI H D DO decO P x) s.1 s.2.1 (capFork Bd.DE.V Cp (xI D x) s.1 s.2.2.1 v))
      (fun v => if Bd.BadO decI H D DO decO P x s.2.1
        (extendDraw (capFork Bd.DE.V Cp (xI D x) s.1 s.2.2.1 v)) then 1 else 0) M s.2.2.2 = 0
    swap
    · exact Or.inr ⟨hop, hz⟩
    left
    have hk : Bd.keepDO decI H D DO decO P x s.2.1 (extendDraw s.2.2.1) = true := by
      have h : (abortRun H (Bd.innerPO decI H D DO decO P x decC) Bd.DE.V (xI D x)
          (Bd.keepAO decI H D DO decO P x) s.2.1 s.2.2.1).out = .acc := hop.1
      rw [Bound.abortRun_eqO decC hn] at h
      exact (Bound.innerFO_acc_iff decC hdec hV s.2.1 _).1 h
    have hopI := (Bound.opensAt_iff decC hdec hV hn Cp s.1 s.2.1 s.2.2.1 (Bound.not_badO_of_keepDO hk)).1 hop
    have hpick := firstPick_map_of_loopCost_zero extendDraw
      (fun v => opensAt H Bd.DE.V Cp (Bd.innerPO decI H D DO decO P x decC) (xI D x)
        (Bd.keepAO decI H D DO decO P x) s.1 s.2.1 (capFork Bd.DE.V Cp (xI D x) s.1 s.2.2.1 v))
      (fun v => opensAtI H D (Inner.ofFlock Bd.deploy decI) DO decO Cp P x s.1 s.2.1
        (capForkI D (Inner.ofFlock Bd.deploy decI) DO decO Cp x s.1 (extendDraw s.2.2.1) v))
      (fun v => if Bd.BadO decI H D DO decO P x s.2.1
        (extendDraw (capFork Bd.DE.V Cp (xI D x) s.1 s.2.2.1 v)) then 1 else 0)
      (fun v hv => by
        have hbv : ¬ Bd.BadO decI H D DO decO P x s.2.1 (extendDraw (capFork Bd.DE.V Cp (xI D x) s.1 s.2.2.1 v)) := by
          intro hb
          simp [hb] at hv
        have h := Bound.opensAt_iff decC hdec hV hn Cp s.1 s.2.1 (capFork Bd.DE.V Cp (xI D x) s.1 s.2.2.1 v) hbv
        rw [Bound.extendDraw_capFork] at h
        exact h)
      M s.2.2.2 hz
    have hpick' : (firstPick (fun v => opensAtI H D (Inner.ofFlock Bd.deploy decI) DO decO Cp P x s.1 s.2.1
          (capForkI D (Inner.ofFlock Bd.deploy decI) DO decO Cp x s.1 (extendDraw s.2.2.1) v)) M
          (fun i => extendDraw (s.2.2.2 i))).getD (extendDraw s.2.2.1) =
        extendDraw ((firstPick (fun v => opensAt H Bd.DE.V Cp (Bd.innerPO decI H D DO decO P x decC) (xI D x)
          (Bd.keepAO decI H D DO decO P x) s.1 s.2.1 (capFork Bd.DE.V Cp (xI D x) s.1 s.2.2.1 v)) M
          s.2.2.2).getD s.2.2.1) := by
      rw [← hpick, optionGetD_map]
    simp only [FC, hop, ↓reduceIte] at hout
    rw [Bound.abortRun_eqO decC hn, Bound.abortRun_eqO decC hn] at hout
    simp only [FCI, hopI, ↓reduceIte, hpick']
    rw [← Bound.extendDraw_capFork]
    have ht : ∀ d₁ d₂ : (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x,
        ∀ p ∈ D.callsAt H x d₁.1 (drawWits H D (Inner.ofFlock Bd.deploy decI) DO decO P x s.2.1 d₁) ++
          D.callsAt H x d₂.1 (drawWits H D (Inner.ofFlock Bd.deploy decI) DO decO P x s.2.1 d₂), H p.1 = p.2 :=
      fun d₁ d₂ p hp => (List.mem_append.1 hp).elim (callsAt_true hT hN0 s.2.1 d₁ p) (callsAt_true hT hN0 s.2.1 d₂ p)
    refine Bound.findColl_cover (x := x) (P := P) (decO := decO) decC hdec hV hL s.2.1 (ht _ _) (fun e he => ?_)
      (Option.isSome_iff_exists.2 ⟨_, hout⟩)
    rcases List.mem_append.1 he with he | he
    · exact ⟨_, he, fun p hp => List.mem_append_left _ hp⟩
    · exact ⟨_, he, fun p hp => List.mem_append_right _ hp⟩
  have hbad : Pr (fun s : CapCoins Bd.DE.V Cp Ω
      ((Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) (xI D x) M =>
      opensAt H Bd.DE.V Cp (Bd.innerPO decI H D DO decO P x decC) (xI D x) (Bd.keepAO decI H D DO decO P x)
          s.1 s.2.1 s.2.2.1 ∧
        loopCost (fun v => opensAt H Bd.DE.V Cp (Bd.innerPO decI H D DO decO P x decC) (xI D x)
            (Bd.keepAO decI H D DO decO P x) s.1 s.2.1 (capFork Bd.DE.V Cp (xI D x) s.1 s.2.2.1 v))
          (fun v => if Bd.BadO decI H D DO decO P x s.2.1
            (extendDraw (capFork Bd.DE.V Cp (xI D x) s.1 s.2.2.1 v)) then 1 else 0) M s.2.2.2 ≠ 0) ≤
      Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x =>
        Bd.BadO decI H D DO decO P x t.1 t.2) := by
    refine (le_of_eq (Pr_prod _)).trans ?_
    have : Nonempty (Fin (Cp.width (xI D x))) := ⟨⟨0, hW⟩⟩
    refine (Ex_mono fun q => ?_).trans (Ex_const _).le
    refine (le_of_eq (Pr_prod _)).trans ((Ex_mono fun ω => ?_).trans (le_of_eq (Pr_prod
      (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x =>
        Bd.BadO decI H D DO decO P x t.1 t.2)).symm))
    exact (Pr_gate_loop_fork_le (Cp.round (xI D x) q)
      (opensAt H Bd.DE.V Cp (Bd.innerPO decI H D DO decO P x decC) (xI D x) (Bd.keepAO decI H D DO decO P x) q ω)
      (fun t => if Bd.BadO decI H D DO decO P x ω (extendDraw t) then 1 else 0) M).trans
      (le_of_eq (Bound.Ex_badO hR ω))
  have hΦ : UnifMap (fun s : CapCoins Bd.DE.V Cp Ω
      ((Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) (xI D x) M =>
      ((s.1, s.2.1, extendDraw s.2.2.1, fun i => extendDraw (s.2.2.2 i)) :
        CapCoinsI D (Inner.ofFlock Bd.deploy decI) DO decO Cp Ω x M)) :=
    UnifMap.self.prod (UnifMap.self.prod ((unifMap_extendDraw hn).prod (UnifMap.pi (unifMap_extendDraw hn) M)))
  exact (Pr_mono key).trans ((Pr_or_le _ _).trans (add_le_add (le_of_eq (hΦ.Pr (fun s' =>
    ∃ u v, (FCI H D (Inner.ofFlock Bd.deploy decI) DO decO Cp P x M hW).out s' = some (u, v) ∧ Collision H u v)))
    hbad))

omit [DecidableEq Col'] in
/-- **Level 0's conflict at `P_O`'s aborting runs is `F0I`, but where a wait meets a bad draw**: the first run,
accepting, is kept; off the bad draws each wait accepts where the draws do and picks the same draw
(`accAt_iff`, `firstPick_map_of_loopCost_zero`), and the runs' collision is the draws' (`findColl_cover`). Each of the
`k_I` waits meets a bad draw at most as often as one draw is bad (`Pr_gate_loop_le`). -/
theorem Bound.PrF0_le_F0I (decC : Msg → Msg → Coin) (hdec : ∀ x c, decC x (Bd.ans c) = c)
    (hV : InnerViewRun H D (Inner.ofFlock Bd.deploy decI) DO) (hL : CallsLogCollAt H D (Inner.ofFlock Bd.deploy decI) DO)
    (hT : CallsTrue H D (Inner.ofFlock Bd.deploy decI) DO) (hN0 : CallsNil H D)
    (hR : InnerRounds D (Inner.ofFlock Bd.deploy decI)) [Fintype Ω] [Nonempty Ω] (R : ℕ) :
    PrF0 H Bd.DE.V (Bd.innerPO decI H D DO decO P x decC) (xI D x) (Bd.keepAO decI H D DO decO P x)
        ((Inner.ofFlock Bd.deploy decI).k (xI D x)) R ≤
      PrF0I H D (Inner.ofFlock Bd.deploy decI) DO decO P x R +
        ((Inner.ofFlock Bd.deploy decI).k (xI D x) : ℝ) *
          Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x =>
            Bd.BadO decI H D DO decO P x t.1 t.2) := by
  classical
  have hn : Bd.DE.V.k (xI D x) ≤ D.V.k x := hR x
  unfold PrF0 PrF0I Finder.Pr
  have key : ∀ s : ConflictCoins Bd.DE.V Ω
      ((Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) (xI D x)
      ((Inner.ofFlock Bd.deploy decI).k (xI D x)) R,
      (∃ u v, (F0 H Bd.DE.V (Bd.innerPO decI H D DO decO P x decC) (xI D x) (Bd.keepAO decI H D DO decO P x)
        ((Inner.ofFlock Bd.deploy decI).k (xI D x)) R).out s = some (u, v) ∧ Collision H u v) →
      (∃ u v, (F0I H D (Inner.ofFlock Bd.deploy decI) DO decO P x R).out
          (s.1, extendDraw s.2.1, fun i j => extendDraw (s.2.2 i j)) = some (u, v) ∧ Collision H u v) ∨
        ∃ i, accAt H Bd.DE.V (Bd.innerPO decI H D DO decO P x decC) (xI D x) (Bd.keepAO decI H D DO decO P x)
            s.1 s.2.1 ∧
          loopCost (accAt H Bd.DE.V (Bd.innerPO decI H D DO decO P x decC) (xI D x)
              (Bd.keepAO decI H D DO decO P x) s.1)
            (fun t => if Bd.BadO decI H D DO decO P x s.1 (extendDraw t) then 1 else 0) R (s.2.2 i) ≠ 0 := by
    rintro s ⟨a, b, hout, -⟩
    have hacc : accAt H Bd.DE.V (Bd.innerPO decI H D DO decO P x decC) (xI D x) (Bd.keepAO decI H D DO decO P x)
        s.1 s.2.1 := by
      by_contra hacc
      simp [F0, hacc] at hout
    by_cases hz : ∀ i, loopCost (accAt H Bd.DE.V (Bd.innerPO decI H D DO decO P x decC) (xI D x)
        (Bd.keepAO decI H D DO decO P x) s.1)
      (fun t => if Bd.BadO decI H D DO decO P x s.1 (extendDraw t) then 1 else 0) R (s.2.2 i) = 0
    swap
    · push_neg at hz
      obtain ⟨i, hi⟩ := hz
      exact Or.inr ⟨i, hacc, hi⟩
    left
    have hk : Bd.keepDO decI H D DO decO P x s.1 (extendDraw s.2.1) = true := by
      have h : (abortRun H (Bd.innerPO decI H D DO decO P x decC) Bd.DE.V (xI D x)
          (Bd.keepAO decI H D DO decO P x) s.1 s.2.1).out = .acc := hacc
      rw [Bound.abortRun_eqO decC hn] at h
      exact (Bound.innerFO_acc_iff decC hdec hV s.1 _).1 h
    have haccI := (Bound.accAt_iff decC hdec hV hn s.1 s.2.1 (Bound.not_badO_of_keepDO hk)).1 hacc
    have hpick : ∀ i, (firstPick (accAt H Bd.DE.V (Bd.innerPO decI H D DO decO P x decC) (xI D x)
        (Bd.keepAO decI H D DO decO P x) s.1) R (s.2.2 i)).map extendDraw =
        firstPick (accAtI H D (Inner.ofFlock Bd.deploy decI) DO decO P x s.1) R
          (fun j => extendDraw (s.2.2 i j)) := fun i =>
      firstPick_map_of_loopCost_zero extendDraw
        (accAt H Bd.DE.V (Bd.innerPO decI H D DO decO P x decC) (xI D x) (Bd.keepAO decI H D DO decO P x) s.1)
        (accAtI H D (Inner.ofFlock Bd.deploy decI) DO decO P x s.1)
        (fun t => if Bd.BadO decI H D DO decO P x s.1 (extendDraw t) then 1 else 0)
        (fun t ht => Bound.accAt_iff decC hdec hV hn s.1 t (by intro hb; simp [hb] at ht)) R (s.2.2 i) (hz i)
    simp only [F0, hacc, ↓reduceIte] at hout
    rw [Bound.abortRun_eqO decC hn] at hout
    simp only [F0I, haccI, ↓reduceIte]
    refine Bound.findColl_cover (x := x) (P := P) (decO := decO) decC hdec hV hL s.1 (fun p hp => ?_)
      (fun e he => ?_)
      (Option.isSome_iff_exists.2 ⟨_, hout⟩)
    · rcases List.mem_append.1 hp with hp | hp
      · exact callsAt_true hT hN0 s.1 (extendDraw s.2.1) p hp
      · obtain ⟨i, -, hpi⟩ := List.mem_flatMap.1 hp
        rcases hfp : firstPick (accAtI H D (Inner.ofFlock Bd.deploy decI) DO decO P x s.1) R
            (fun j => extendDraw (s.2.2 i j)) with _ | d
        · simp [hfp] at hpi
        · have hpi' : p ∈ D.callsAt H x d.1 (drawWits H D (Inner.ofFlock Bd.deploy decI) DO decO P x s.1 d) := by
            simpa [hfp] using hpi
          exact callsAt_true hT hN0 s.1 d p hpi'
    · rcases List.mem_append.1 he with he | he
      · exact ⟨_, he, fun p hp => List.mem_append_left _ hp⟩
      · obtain ⟨i, -, hei⟩ := List.mem_flatMap.1 he
        rcases hfp : firstPick (accAt H Bd.DE.V (Bd.innerPO decI H D DO decO P x decC) (xI D x)
            (Bd.keepAO decI H D DO decO P x) s.1) R (s.2.2 i) with _ | t
        · simp [hfp] at hei
        · have hei' : e ∈ (Bd.innerFO decI H D DO decO P x decC s.1 (extendDraw t)).vLog := by
            rw [← Bound.abortRun_eqO decC hn]
            simpa [hfp] using hei
          have hm : firstPick (accAtI H D (Inner.ofFlock Bd.deploy decI) DO decO P x s.1) R
              (fun j => extendDraw (s.2.2 i j)) = some (extendDraw t) := by
            rw [← hpick i, hfp]
            try rfl
          refine ⟨_, hei', fun p hp => List.mem_append_right _ (List.mem_flatMap.2 ⟨i, List.mem_finRange i, ?_⟩)⟩
          have hp' : p ∈ D.callsAt H x (extendDraw t).1
              (drawWits H D (Inner.ofFlock Bd.deploy decI) DO decO P x s.1 (extendDraw t)) := hp
          simpa [hm] using hp'
  have hbad : Pr (fun s : ConflictCoins Bd.DE.V Ω
      ((Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) (xI D x)
      ((Inner.ofFlock Bd.deploy decI).k (xI D x)) R =>
      ∃ i, accAt H Bd.DE.V (Bd.innerPO decI H D DO decO P x decC) (xI D x) (Bd.keepAO decI H D DO decO P x)
          s.1 s.2.1 ∧
        loopCost (accAt H Bd.DE.V (Bd.innerPO decI H D DO decO P x decC) (xI D x)
            (Bd.keepAO decI H D DO decO P x) s.1)
          (fun t => if Bd.BadO decI H D DO decO P x s.1 (extendDraw t) then 1 else 0) R (s.2.2 i) ≠ 0) ≤
      ((Inner.ofFlock Bd.deploy decI).k (xI D x) : ℝ) *
        Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x =>
          Bd.BadO decI H D DO decO P x t.1 t.2) := by
    refine (Pr_exists_le _).trans ?_
    have hi : ∀ i : Fin ((Inner.ofFlock Bd.deploy decI).k (xI D x)),
        Pr (fun s : ConflictCoins Bd.DE.V Ω
          ((Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) (xI D x)
          ((Inner.ofFlock Bd.deploy decI).k (xI D x)) R =>
          accAt H Bd.DE.V (Bd.innerPO decI H D DO decO P x decC) (xI D x) (Bd.keepAO decI H D DO decO P x)
              s.1 s.2.1 ∧
            loopCost (accAt H Bd.DE.V (Bd.innerPO decI H D DO decO P x decC) (xI D x)
                (Bd.keepAO decI H D DO decO P x) s.1)
              (fun t => if Bd.BadO decI H D DO decO P x s.1 (extendDraw t) then 1 else 0) R (s.2.2 i) ≠ 0) ≤
        Pr (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x =>
          Bd.BadO decI H D DO decO P x t.1 t.2) := by
      intro i
      have hΨ : UnifMap (fun s : ConflictCoins Bd.DE.V Ω
          ((Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) (xI D x)
          ((Inner.ofFlock Bd.deploy decI).k (xI D x)) R =>
          ((s.1, s.2.1, s.2.2 i) : Ω × ((Fin (Bd.DE.V.k (xI D x)) → Coin) × (Fin (D.V.k x) → Coin) ×
            (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) ×
            (Fin R → (Fin (Bd.DE.V.k (xI D x)) → Coin) × (Fin (D.V.k x) → Coin) ×
              (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x))) :=
        UnifMap.self.prod (UnifMap.self.prod (UnifMap.coord i))
      refine (le_of_eq (hΨ.Pr (fun b => accAt H Bd.DE.V (Bd.innerPO decI H D DO decO P x decC) (xI D x)
          (Bd.keepAO decI H D DO decO P x) b.1 b.2.1 ∧
        loopCost (accAt H Bd.DE.V (Bd.innerPO decI H D DO decO P x decC) (xI D x)
            (Bd.keepAO decI H D DO decO P x) b.1)
          (fun t => if Bd.BadO decI H D DO decO P x b.1 (extendDraw t) then 1 else 0) R b.2.2 ≠ 0))).trans ?_
      refine (le_of_eq (Pr_prod _)).trans ((Ex_mono fun ω => ?_).trans (le_of_eq (Pr_prod
        (fun t : Ω × (Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x =>
          Bd.BadO decI H D DO decO P x t.1 t.2)).symm))
      exact (Pr_gate_loop_le
        (accAt H Bd.DE.V (Bd.innerPO decI H D DO decO P x decC) (xI D x) (Bd.keepAO decI H D DO decO P x) ω)
        (fun t => if Bd.BadO decI H D DO decO P x ω (extendDraw t) then 1 else 0) R).trans
        (le_of_eq (Bound.Ex_badO hR ω))
    refine (Finset.sum_le_sum fun i _ => hi i).trans (le_of_eq ?_)
    rw [Finset.sum_const, Finset.card_univ, Fintype.card_fin, nsmul_eq_mul]
  have hΦ : UnifMap (fun s : ConflictCoins Bd.DE.V Ω
      ((Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) (xI D x)
      ((Inner.ofFlock Bd.deploy decI).k (xI D x)) R =>
      ((s.1, extendDraw s.2.1, fun i j => extendDraw (s.2.2 i j)) :
        ConflictCoinsI D (Inner.ofFlock Bd.deploy decI) DO decO Ω x ((Inner.ofFlock Bd.deploy decI).k (xI D x)) R)) :=
    UnifMap.self.prod ((unifMap_extendDraw hn).prod (UnifMap.pi (UnifMap.pi (unifMap_extendDraw hn) R) _))
  exact (Pr_mono key).trans ((Pr_or_le _ _).trans (add_le_add (le_of_eq (hΦ.Pr (fun s' =>
    ∃ u v, (F0I H D (Inner.ofFlock Bd.deploy decI) DO decO P x R).out s' = some (u, v) ∧ Collision H u v)))
    hbad))

/-- **The inner stage at the bound record from `KSFlockLin` at `DE`, with no root**: at most
`κ_I + (4k_I + 1)·Pr[B_I] + 4·W_I·(Pr[FCI] + 1/(M+1)) + 4·(Pr[F0I] + (k_I + 1)/(R+1)) + β₂·(Pr[Fdev] + 1/(M+1))`.
`E^rep` at `DE` against `P_O`'s aborting runs is `KSFlockLin`'s bound (`PrBadGood_le_merged`), its `B` at most `B_I`
(`PrBAbort_le_PrBinnerV`), its `FC` and `F0` at most `FCI` and `F0I` but for bad draws (`PrFC_le_FCI`, `PrF0_le_F0I`); a
rejecting judged draw is at most `Pr[B_I]` over the draws' evaluations (`Pr_innerRejects_le_PrBinnerV_at`); and the bad draws, `N_I + 4·W_I + 4·k_I` draws' worth, are each at most
`k_I·(Pr[FdevV] + 1/(M+1))` (`Pr_badO_le_PrFdevV`), within `β₂`. -/
theorem Bound.innerLin2_of_flockLin_binds [Zero Col] [Zero Col'] (decC : Msg → Msg → Coin) (Cp : Caps Coin)
    (hKS : KSFlockLin H Bd.DE Cp) (hdI : decOK Bd.deploy decI) (hdec : ∀ x c, decC x (Bd.ans c) = c)
    (hB : Bd.Binds) (hR : InnerRounds D (Inner.ofFlock Bd.deploy decI))
    (hV : InnerViewRun H D (Inner.ofFlock Bd.deploy decI) DO) (hT : CallsTrue H D (Inner.ofFlock Bd.deploy decI) DO)
    (hL : CallsLogCollAt H D (Inner.ofFlock Bd.deploy decI) DO)
    (hFk : ForksBindAtV H D (Inner.ofFlock Bd.deploy decI) DO x)
    (hA : InnerAcceptsCollAtV H D (Inner.ofFlock Bd.deploy decI) DO x)
    (hM : RoundsMargin H D (Inner.ofFlock Bd.deploy decI)) (hN0 : CallsNil H D) [Fintype Ω] [Nonempty Ω] (M R : ℕ) :
    PrBadGood H D.V (innerR D (Inner.ofFlock Bd.deploy decI)) outR (EO D (Inner.ofFlock Bd.deploy decI) DO decO)
        (dec H D) (fun x => (Inner.ofFlock Bd.deploy decI).waits (xI D x))
        (fun x => (Inner.ofFlock Bd.deploy decI).k (xI D x)) (fun x => (Inner.ofFlock Bd.deploy decI).cap (xI D x))
        (Inner.ofFlock Bd.deploy decI).opens (fun x => (Inner.ofFlock Bd.deploy decI).assemble (xI D x))
        (fun x => (Inner.ofFlock Bd.deploy decI).check (xI D x)) x P ≤
      (Inner.ofFlock Bd.deploy decI).κ (xI D x) +
        ((4 * (Inner.ofFlock Bd.deploy decI).V.k (xI D x) + 1) *
            PrBinnerV H D (Inner.ofFlock Bd.deploy decI) DO decO P x +
          4 * Cp.width (xI D x) * (PrFCI H D (Inner.ofFlock Bd.deploy decI) DO decO Cp P x M + 1 / ((M : ℝ) + 1)) +
          4 * (PrF0I H D (Inner.ofFlock Bd.deploy decI) DO decO P x R +
            (((Inner.ofFlock Bd.deploy decI).k (xI D x) : ℝ) + 1) / ((R : ℝ) + 1)) +
          βdev2 D (Inner.ofFlock Bd.deploy decI) Cp x *
            (PrFdevV H D (Inner.ofFlock Bd.deploy decI) DO decO P x M + 1 / ((M : ℝ) + 1))) := by
  have h1 := Bound.PrBadGood_le_merged (decO := decO) (P := P) (x := x) decC hdec hV hR
  have h2 : (Inner.ofFlock Bd.DE decI).PrBadAbort H (xI D x) (Bd.innerPO decI H D DO decO P x decC)
      (Bd.keepAO decI H D DO decO P x) ≤
      (Inner.ofFlock Bd.deploy decI).κ (xI D x) +
        4 * ((Inner.ofFlock Bd.deploy decI).V.k (xI D x) : ℝ) *
          PrBAbort H Bd.DE.V (Bd.innerPO decI H D DO decO P x decC) (xI D x) (Bd.keepAO decI H D DO decO P x) +
        4 * (Cp.width (xI D x) : ℝ) *
          (PrFC H Bd.DE.V Cp (Bd.innerPO decI H D DO decO P x decC) (xI D x) (Bd.keepAO decI H D DO decO P x) M +
            1 / ((M : ℝ) + 1)) +
        4 * (PrF0 H Bd.DE.V (Bd.innerPO decI H D DO decO P x decC) (xI D x) (Bd.keepAO decI H D DO decO P x)
            ((Inner.ofFlock Bd.deploy decI).k (xI D x)) R +
          (((Inner.ofFlock Bd.deploy decI).k (xI D x) : ℝ) + 1) / ((R : ℝ) + 1)) :=
    hKS decI hdI (xI D x) Ω (Bd.innerPO decI H D DO decO P x decC)
      ((Fin (D.V.k x) → Coin) × (EO D (Inner.ofFlock Bd.deploy decI) DO decO).C x) (Bd.keepAO decI H D DO decO P x)
      M R
  have h3 := Bound.PrBAbort_le_PrBinnerV (decO := decO) (P := P) (x := x) decC hdec hV hL hT hN0 hR
  have h4 := Bound.PrFC_le_FCI (decO := decO) (P := P) (x := x) decC hdec hV hL hT hN0 hR Cp M
  have h5 := Bound.PrF0_le_F0I (decO := decO) (P := P) (x := x) decC hdec hV hL hT hN0 hR R
  have h6 := Pr_innerRejects_le_PrBinnerV_at hA hT hN0 hM decO P
  have h7 := Bound.Pr_badO_le_PrFdevV (decO := decO) (P := P) (x := x) hB hV hT hL hN0 hFk hA M
  have hK0 : (0 : ℝ) ≤ (Inner.ofFlock Bd.deploy decI).V.k (xI D x) := Nat.cast_nonneg _
  have hW0 : (0 : ℝ) ≤ Cp.width (xI D x) := Nat.cast_nonneg _
  have hk0 : (0 : ℝ) ≤ (Inner.ofFlock Bd.deploy decI).k (xI D x) := Nat.cast_nonneg _
  have hN0' : (0 : ℝ) ≤ NI D (Inner.ofFlock Bd.deploy decI) x := Nat.cast_nonneg _
  have ht0 : (0 : ℝ) ≤ 1 / ((M : ℝ) + 1) := by positivity
  have hF0 := PrFdevV_nonneg H D (Inner.ofFlock Bd.deploy decI) DO decO P x M
  have hKF0 : (0 : ℝ) ≤ (Inner.ofFlock Bd.deploy decI).V.k (xI D x) *
      (PrFdevV H D (Inner.ofFlock Bd.deploy decI) DO decO P x M + 1 / ((M : ℝ) + 1)) :=
    mul_nonneg hK0 (by linarith)
  have hWKF0 := mul_nonneg hW0 hKF0
  have hKB := mul_le_mul_of_nonneg_left h3 hK0
  have hWF := mul_le_mul_of_nonneg_left h4 hW0
  have hNB := mul_le_mul_of_nonneg_left h7 hN0'
  have hWB := mul_le_mul_of_nonneg_left h7 hW0
  have hkB := mul_le_mul_of_nonneg_left h7 hk0
  have hβ : βdev2 D (Inner.ofFlock Bd.deploy decI) Cp x *
      (PrFdevV H D (Inner.ofFlock Bd.deploy decI) DO decO P x M + 1 / ((M : ℝ) + 1)) =
      NI D (Inner.ofFlock Bd.deploy decI) x * ((Inner.ofFlock Bd.deploy decI).V.k (xI D x) *
          (PrFdevV H D (Inner.ofFlock Bd.deploy decI) DO decO P x M + 1 / ((M : ℝ) + 1))) +
        8 * (Cp.width (xI D x) * ((Inner.ofFlock Bd.deploy decI).V.k (xI D x) *
          (PrFdevV H D (Inner.ofFlock Bd.deploy decI) DO decO P x M + 1 / ((M : ℝ) + 1)))) +
        4 * ((Inner.ofFlock Bd.deploy decI).k (xI D x) * ((Inner.ofFlock Bd.deploy decI).V.k (xI D x) *
          (PrFdevV H D (Inner.ofFlock Bd.deploy decI) DO decO P x M + 1 / ((M : ℝ) + 1)))) +
        4 * ((Inner.ofFlock Bd.deploy decI).V.k (xI D x) *
          (PrFdevV H D (Inner.ofFlock Bd.deploy decI) DO decO P x M + 1 / ((M : ℝ) + 1))) := by
    unfold βdev2; ring
  linarith

/-- **The recursion's KS at the bound record from Flock's KS at the outer deployment and `KSFlockLin` at `DE`**, with
no root past the outer sessions': `PrBad_le_of_inner` at `innerLin2_of_flockLin_binds`, on
`KS_of_plainKSAbort_binds`' premises less (P1) and the weight's. -/
theorem Bound.ksLin2_of_flockLin_binds [Zero Col] [Zero Col'] (decC : Msg → Msg → Coin) (Cp : Caps Coin)
    (hF : Flock.KS H DO) (hKS : KSFlockLin H Bd.DE Cp) (hdI : decOK Bd.deploy decI) (hdO : decOK DO decO)
    (hdec : ∀ x c, decC x (Bd.ans c) = c) (hB : Bd.Binds) (hRef : Refuses H D)
    (hRounds : InnerRounds D (Inner.ofFlock Bd.deploy decI)) (hN : SessionNums D DO) (hε : TableErrNonneg D)
    (hRd : SessionRounds D (Inner.ofFlock Bd.deploy decI) DO) (hS : SessionAccepts D)
    (hRs : SessionRuns H D (Inner.ofFlock Bd.deploy decI) DO) (hV : InnerViewRun H D (Inner.ofFlock Bd.deploy decI) DO)
    (hT : CallsTrue H D (Inner.ofFlock Bd.deploy decI) DO) (hL : CallsLogCollAt H D (Inner.ofFlock Bd.deploy decI) DO)
    (hFk : ForksBindAtV H D (Inner.ofFlock Bd.deploy decI) DO x)
    (hA : InnerAcceptsCollAtV H D (Inner.ofFlock Bd.deploy decI) DO x)
    (hM : RoundsMargin H D (Inner.ofFlock Bd.deploy decI)) (hN0 : CallsNil H D) [Fintype Ω] [Nonempty Ω] (M R : ℕ) :
    PrBad H D.V (Recursion.R D (Inner.ofFlock Bd.deploy decI)) (E H D (Inner.ofFlock Bd.deploy decI) DO decO) x P ≤
      κ D (Inner.ofFlock Bd.deploy decI) x +
        Real.sqrt (γ D (Inner.ofFlock Bd.deploy decI) x * PrB H D.V P x) +
        (4 * (Inner.ofFlock Bd.deploy decI).V.k (xI D x) + 1) *
          PrBinnerV H D (Inner.ofFlock Bd.deploy decI) DO decO P x +
        4 * Cp.width (xI D x) * (PrFCI H D (Inner.ofFlock Bd.deploy decI) DO decO Cp P x M + 1 / ((M : ℝ) + 1)) +
        4 * (PrF0I H D (Inner.ofFlock Bd.deploy decI) DO decO P x R +
          (((Inner.ofFlock Bd.deploy decI).k (xI D x) : ℝ) + 1) / ((R : ℝ) + 1)) +
        βdev2 D (Inner.ofFlock Bd.deploy decI) Cp x *
          (PrFdevV H D (Inner.ofFlock Bd.deploy decI) DO decO P x M + 1 / ((M : ℝ) + 1)) := by
  have h := PrBad_le_of_inner H D (Inner.ofFlock Bd.deploy decI) DO hRef (outerKS_of_flockKS hF hN hε hRd hS hRs)
    decO hdO x P _ (Bound.innerLin2_of_flockLin_binds (decO := decO) (P := P) (x := x) decC Cp hKS hdI hdec hB hRounds
      hV hT hL hFk hA hM hN0 M R)
  linarith

end Coupling

section Scoped

variable {Coin Col T Col' T' : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]
  [DecidableEq Col']

/-- **`KSLin2` at the statements `G` admits**: its bound at every statement `x` with `G x`. The bound at one statement
reads the code facts about V*'s circuits only at that statement's inner form, so a deployment whose circuits are shown
at one form gets `KSLin2` at that form's statements. -/
def KSLin2On [Zero Col] [Zero Col'] (H : Msg → Msg) (D : RecAt Coin) (Bd : Bound Coin Col T)
    (DO : Flock.Deploy Coin Col' T') (Cp : Caps Coin) (G : Msg → Prop) : Prop :=
  ∀ decI, decOK Bd.deploy decI → ∀ decO, decOK DO decO → ∀ (x : Msg), G x → ∀ (Ω : Type) [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (M R : ℕ),
    PrBad H (D.record Bd.IVStar).V (Recursion.R (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI))
        (E H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO) x P ≤
      κ (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) x +
        Real.sqrt (γ (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) x * PrB H (D.record Bd.IVStar).V P x) +
        (4 * (Inner.ofFlock Bd.deploy decI).V.k (xI (D.record Bd.IVStar) x) + 1) *
          PrBinnerV H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO P x +
        4 * Cp.width (xI (D.record Bd.IVStar) x) *
          (PrFCI H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO Cp P x M +
            1 / ((M : ℝ) + 1)) +
        4 * (PrF0I H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO P x R +
          (((Inner.ofFlock Bd.deploy decI).k (xI (D.record Bd.IVStar) x) : ℝ) + 1) / ((R : ℝ) + 1)) +
        βdev2 (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) Cp x *
          (PrFdevV H (D.record Bd.IVStar) (Inner.ofFlock Bd.deploy decI) DO decO P x M +
            1 / ((M : ℝ) + 1))

end Scoped

end Claims.Recursion
