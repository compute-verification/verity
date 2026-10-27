import Proofs.Flock.Claims.Witnesses.RecursionLin

/-!
# The witnesses of the recursion's claims with the inner Flock's roots linear

`KSFlockLin` fails at `Flock.ksD` with no cap position (`noCaps`): the silent prover is always accepted on a statement
with no witness, every run records no verifier call, so `B` and `F0` find nothing, and `F0`'s tail `4·(k + 1)/(R+1)` is
below `1/4` at `R = 16·(k + 1)` (`f0_tail_lt`), while `κ < 1/2`. `KSLin2` and `BitsLin2` fail at `KSBoundLin`'s toy
records (`ksRAt` with `ksB`), where the outer witnesses record no call, so `B_I`, `Fdev` and `F0I` find nothing, the
inner statement has no cap position, and `Fdev`'s tail `β₂/(M+1)` is below `1/4` at `M = ⌈4β₂⌉` (`tail_lt_quarter`);
there `κ < 1/6` (`ksD_κ_lt_sixth`), so `λ₂ > 0` (`ksB_lamLin2_pos`). Their `meets` witnesses are `Inner.KSAbort`'s and
`KSBound`'s: the claims read the same decoders. The toy records stage no V\* statement, so `B_I` and `Fdev` over the
draws' evaluations (`BinnerV`, `FdevV`) find nothing either (`drawCalls_noCalls`).
-/

namespace Claims.Witness.Recursion

open Claims Claims.Recursion

/-- No cap position at any statement. -/
def noCaps {Coin : Type} : Caps Coin := ⟨fun _ => 0, fun _ _ => 0, fun _ _ _ => false⟩

section Numbers

variable {Coin W Col T Col' T' : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem getD_map_nil {α β : Type} (o : Option α) : (o.map fun _ => ([] : List β)).getD [] = [] := by
  cases o <;> rfl

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem flatMap_nil_fun {α β : Type} (l : List α) : l.flatMap (fun _ => ([] : List β)) = [] :=
  List.flatMap_eq_nil_iff.2 fun _ _ => rfl

omit [Fintype Coin] [Nonempty Coin] in
/-- A run that may abort records no verifier call where the run records none. -/
theorem abortRun_vLog_nil {Ω A : Type} (H : Msg → Msg) (P : Prover Ω) (V : Verifier Coin) (x : Msg)
    (keep : Ω → (Fin (V.k x) → Coin) → A → Bool) (hv : ∀ ω r, (run H P V x ω r).vLog = []) (ω : Ω)
    (d : (Fin (V.k x) → Coin) × A) : (abortRun H P V x keep ω d).vLog = [] := by
  unfold abortRun
  split_ifs
  exacts [hv _ _, rfl]

/-- `B` at runs that may abort finds nothing where no run records a verifier call. -/
theorem prBAbort_vLog_nil {Ω A : Type} [Fintype Ω] [Nonempty Ω] [Fintype A] [Nonempty A] (H : Msg → Msg)
    (P : Prover Ω) (V : Verifier Coin) (x : Msg) (keep : Ω → (Fin (V.k x) → Coin) → A → Bool)
    (hv : ∀ ω r, (run H P V x ω r).vLog = []) : PrBAbort H V P x keep = 0 := by
  unfold PrBAbort
  split_ifs with hk
  · unfold Finder.Pr
    rw [Pr_congr fun s => iff_false_intro ?_]
    · exact Pr_false
    · rintro ⟨u, v, hs, -⟩
      simp only [BAbort, abortRun_vLog_nil H P V x keep hv, List.nil_append, findColl, reduceCtorEq] at hs
  · rfl

/-- `F0` finds nothing where no run records a verifier call. -/
theorem prF0_vLog_nil {Ω A : Type} [Fintype Ω] [Nonempty Ω] [Fintype A] [Nonempty A] (H : Msg → Msg)
    (P : Prover Ω) (V : Verifier Coin) (x : Msg) (keep : Ω → (Fin (V.k x) → Coin) → A → Bool) (k R : ℕ)
    (hv : ∀ ω r, (run H P V x ω r).vLog = []) : PrF0 H V P x keep k R = 0 := by
  unfold PrF0 Finder.Pr
  rw [Pr_congr fun s => iff_false_intro ?_]
  · exact Pr_false
  · rintro ⟨u, v, h, -⟩
    simp only [F0, abortRun_vLog_nil H P V x keep hv, getD_map_nil, flatMap_nil_fun, List.nil_append, findColl,
      ite_self, reduceCtorEq] at h

/-- `F0I` finds nothing where the outer witnesses record no call. -/
theorem f0I_pr_noCalls [DecidableEq Col'] (H : Msg → Msg) (D : Recursion.Deploy Coin)
    (hc : ∀ x (r : Fin (D.V.k x) → Coin) ws, D.callsAt H x r ws = []) (I : Recursion.Inner Coin W) (DO : Flock.Deploy Coin Col' T')
    (decO : List (ℕ × Col') → List T') {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (x : Msg) (R : ℕ) : (F0I H D I DO decO P x R).Pr H = 0 := by
  unfold Finder.Pr
  rw [Pr_congr fun s => iff_false_intro ?_]
  · exact Pr_false
  · rintro ⟨u, v, h, -⟩
    simp only [F0I, hc, getD_map_nil, flatMap_nil_fun, List.nil_append, findColl, ite_self, reduceCtorEq] at h

theorem prF0I_noCalls [DecidableEq Col'] (H : Msg → Msg) (D : Recursion.Deploy Coin)
    (hc : ∀ x (r : Fin (D.V.k x) → Coin) ws, D.callsAt H x r ws = []) (I : Recursion.Inner Coin W) (DO : Flock.Deploy Coin Col' T')
    (decO : List (ℕ × Col') → List T') {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (x : Msg) (R : ℕ) : PrF0I H D I DO decO P x R = 0 :=
  f0I_pr_noCalls H D hc I DO decO P x R

theorem f0I_expectedCR_noCalls [DecidableEq Col'] (H : Msg → Msg) (D : Recursion.Deploy Coin)
    (hc : ∀ x (r : Fin (D.V.k x) → Coin) ws, D.callsAt H x r ws = []) (I : Recursion.Inner Coin W) (DO : Flock.Deploy Coin Col' T')
    (decO : List (ℕ × Col') → List T') {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (x : Msg) (R : ℕ) : ExpectedCR H (F0I H D I DO decO P x R) := by
  unfold ExpectedCR
  rw [f0I_pr_noCalls H D hc I DO decO P x R]
  exact div_nonneg (Ex_nonneg fun _ => Nat.cast_nonneg _) (by positivity)

/-- A draw's evaluations are none where the outer witnesses record no call and the compiled verifier stages no V\*
statement. -/
theorem drawCalls_noCalls [DecidableEq Col'] (H : Msg → Msg) (D : Recursion.Deploy Coin)
    (hc : ∀ x (r : Fin (D.V.k x) → Coin) ws, D.callsAt H x r ws = []) (hs : ∀ x ps, D.vstar x ps = []) (I : Recursion.Inner Coin W)
    (DO : Flock.Deploy Coin Col' T') (decO : List (ℕ × Col') → List T') {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (d : (Fin (D.V.k x) → Coin) × (EO D I DO decO).C x) : drawCalls H D I DO decO P x ω d = [] := by
  unfold drawCalls runStmts
  rw [hc, hs]
  rfl

/-- `B_I` over the draws' evaluations finds nothing where there are none. -/
theorem binnerV_pr_noCalls [DecidableEq Col'] (H : Msg → Msg) (D : Recursion.Deploy Coin)
    (hc : ∀ x (r : Fin (D.V.k x) → Coin) ws, D.callsAt H x r ws = []) (hs : ∀ x ps, D.vstar x ps = []) (I : Recursion.Inner Coin W)
    (DO : Flock.Deploy Coin Col' T') (decO : List (ℕ × Col') → List T') {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (x : Msg) (hk : 0 < I.V.k (xI D x)) : (BinnerV H D I DO decO P x hk).Pr H = 0 := by
  unfold Finder.Pr
  rw [Pr_congr fun s => iff_false_intro ?_]
  · exact Pr_false
  · rintro ⟨u, v, h, -⟩
    simp only [BinnerV, drawCalls_noCalls H D hc hs, List.append_nil, findColl, reduceCtorEq] at h

/-- `Fdev` over the draws' evaluations finds nothing where there are none. -/
theorem fdevV_pr_noCalls [DecidableEq Col'] (H : Msg → Msg) (D : Recursion.Deploy Coin)
    (hc : ∀ x (r : Fin (D.V.k x) → Coin) ws, D.callsAt H x r ws = []) (hs : ∀ x ps, D.vstar x ps = []) (I : Recursion.Inner Coin W)
    (DO : Flock.Deploy Coin Col' T') (decO : List (ℕ × Col') → List T') {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (x : Msg) (M : ℕ) (hk : 0 < I.V.k (xI D x)) : (FdevV H D I DO decO P x M hk).Pr H = 0 := by
  unfold Finder.Pr
  rw [Pr_congr fun s => iff_false_intro ?_]
  · exact Pr_false
  · rintro ⟨u, v, h, -⟩
    simp only [FdevV, linkOutV, drawCalls_noCalls H D hc hs, List.append_nil, findColl, reduceCtorEq] at h

theorem prBinnerV_noCalls [DecidableEq Col'] (H : Msg → Msg) (D : Recursion.Deploy Coin)
    (hc : ∀ x (r : Fin (D.V.k x) → Coin) ws, D.callsAt H x r ws = []) (hs : ∀ x ps, D.vstar x ps = []) (I : Recursion.Inner Coin W)
    (DO : Flock.Deploy Coin Col' T') (decO : List (ℕ × Col') → List T') {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (x : Msg) : PrBinnerV H D I DO decO P x = 0 := by
  unfold PrBinnerV
  split_ifs with hk
  · exact binnerV_pr_noCalls H D hc hs I DO decO P x hk
  · rfl

theorem prFdevV_noCalls [DecidableEq Col'] (H : Msg → Msg) (D : Recursion.Deploy Coin)
    (hc : ∀ x (r : Fin (D.V.k x) → Coin) ws, D.callsAt H x r ws = []) (hs : ∀ x ps, D.vstar x ps = []) (I : Recursion.Inner Coin W)
    (DO : Flock.Deploy Coin Col' T') (decO : List (ℕ × Col') → List T') {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (x : Msg) (M : ℕ) : PrFdevV H D I DO decO P x M = 0 := by
  unfold PrFdevV
  split_ifs with hk
  · exact fdevV_pr_noCalls H D hc hs I DO decO P x M hk
  · rfl

theorem binnerV_expectedCR_noCalls [DecidableEq Col'] (H : Msg → Msg) (D : Recursion.Deploy Coin)
    (hc : ∀ x (r : Fin (D.V.k x) → Coin) ws, D.callsAt H x r ws = []) (hs : ∀ x ps, D.vstar x ps = []) (I : Recursion.Inner Coin W)
    (DO : Flock.Deploy Coin Col' T') (decO : List (ℕ × Col') → List T') {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (x : Msg) (hk : 0 < I.V.k (xI D x)) : ExpectedCR H (BinnerV H D I DO decO P x hk) := by
  unfold ExpectedCR
  rw [binnerV_pr_noCalls H D hc hs I DO decO P x hk]
  exact div_nonneg (Ex_nonneg fun _ => Nat.cast_nonneg _) (by positivity)

theorem fdevV_expectedCR_noCalls [DecidableEq Col'] (H : Msg → Msg) (D : Recursion.Deploy Coin)
    (hc : ∀ x (r : Fin (D.V.k x) → Coin) ws, D.callsAt H x r ws = []) (hs : ∀ x ps, D.vstar x ps = []) (I : Recursion.Inner Coin W)
    (DO : Flock.Deploy Coin Col' T') (decO : List (ℕ × Col') → List T') {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (x : Msg) (M : ℕ) (hk : 0 < I.V.k (xI D x)) : ExpectedCR H (FdevV H D I DO decO P x M hk) := by
  unfold ExpectedCR
  rw [fdevV_pr_noCalls H D hc hs I DO decO P x M hk]
  exact div_nonneg (Ex_nonneg fun _ => Nat.cast_nonneg _) (by positivity)

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- `Fdev`'s tail at `M = ⌈4β⌉` is below `1/4`. -/
theorem tail_lt_quarter (β : ℝ) : β * (1 / (((⌈4 * β⌉₊ : ℕ) : ℝ) + 1)) < 1 / 4 := by
  have hM : 4 * β ≤ ((⌈4 * β⌉₊ : ℕ) : ℝ) := Nat.le_ceil _
  have hp : (0 : ℝ) < ((⌈4 * β⌉₊ : ℕ) : ℝ) + 1 := by positivity
  rw [mul_one_div, div_lt_iff₀ hp]
  linarith

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- `F0`'s tail at `R = 16·(k + 1)` is below `1/4`. -/
theorem f0_tail_lt (k : ℕ) : 4 * (((k : ℝ) + 1) / (((16 * (k + 1) : ℕ) : ℝ) + 1)) < 1 / 4 := by
  have hp : (0 : ℝ) < ((16 * (k + 1) : ℕ) : ℝ) + 1 := by positivity
  have hc : ((16 * (k + 1) : ℕ) : ℝ) = 16 * ((k : ℝ) + 1) := by push_cast; ring
  rw [← mul_div_assoc, div_lt_iff₀ hp, hc]
  linarith

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem exp_neg_four_le_48th : Real.exp (-((32 : ℕ) : ℝ) / 8) ≤ 1 / 48 := by
  have h2 : (2.7 : ℝ) ≤ Real.exp 1 := by linarith [Real.exp_one_gt_d9]
  have h4 : (48 : ℝ) ≤ Real.exp 4 := by
    rw [show (4 : ℝ) = (4 : ℕ) * 1 by norm_num, Real.exp_nat_mul]
    calc (48 : ℝ) ≤ 2.7 ^ 4 := by norm_num
      _ ≤ Real.exp 1 ^ 4 := by gcongr
  rw [show -((32 : ℕ) : ℝ) / 8 = -4 by norm_num, Real.exp_neg, one_div]
  exact inv_anti₀ (by norm_num) h4

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem ksD_κ_lt_sixth (x : Msg) : Claims.Flock.κ (Flock.ksD Coin) x < 1 / 6 := by
  have he := exp_neg_four_le_48th
  have hw : (2 : ℝ) ^ (-((4 : ℕ) : ℝ)) = 1 / 16 := by
    rw [Real.rpow_neg (by norm_num), Real.rpow_natCast]; norm_num
  unfold Claims.Flock.κ
  rw [Flock.ksD_k]
  simp only [Flock.ksD] at hw ⊢
  rw [hw]
  push_cast at he ⊢
  linarith

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem logb_lt_hundred {y : ℝ} (h0 : 0 ≤ y) (h : y < 2 ^ 100) : Real.logb 2 y < 100 := by
  rcases h0.lt_or_eq with h0 | h0
  · rw [Real.logb_lt_iff_lt_rpow (by norm_num) h0, show (100 : ℝ) = ((100 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
    exact h
  · rw [← h0, Real.logb_zero]; norm_num

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem lt_sub_logb (c : ℝ) {y : ℝ} (hc : 103 ≤ c) (h0 : 0 ≤ y) (h : y < 2 ^ 100) (h6 : Real.logb 2 6 < 3) :
    Real.logb 2 6 < c - Real.logb 2 y := by
  have := logb_lt_hundred h0 h
  linarith

omit [Fintype Coin] [Nonempty Coin] in
/-- `λ₂ > 0` at `ksRV` over `ksB` with no cap position: `κ < 1/6` binds, against `log₂ 6 < 3`; the rest are above
`155` (`γ = 0`, `k_I = 1`, `c_Be = 4`, `k = 32`, `N_I = 129`, `β₂ = 261`). -/
theorem ksB_lamLin2_pos (decI : List (ℕ × Unit) → List Unit) :
    0 < lamLin2 (ksRV Coin (ksB Coin).IVStar) (Inner.ofFlock (ksB Coin).deploy decI) noCaps [] 0 0 := by
  have hx : xI (ksRV Coin (ksB Coin).IVStar) [] = [] := rfl
  have hkV : (Inner.ofFlock (ksB Coin).deploy decI).V.k [] = 1 := rfl
  have hk : (Inner.ofFlock (ksB Coin).deploy decI).k [] = 32 := Flock.ksD_k (Coin := Coin) []
  have hw : (Inner.ofFlock (ksB Coin).deploy decI).waits [] = 4 := rfl
  have hW : (noCaps (Coin := Coin)).width [] = 0 := rfl
  have hc : (cBe (ksRV Coin (ksB Coin).IVStar) [] : ℝ) = 4 := by
    have : IsEmpty (Fin ((ksRV Coin (ksB Coin).IVStar).nS [])) := Fin.isEmpty'
    unfold cBe
    rw [Finset.univ_eq_empty, Finset.sum_empty]
    norm_num
  have hγ : Recursion.γ (ksRV Coin (ksB Coin).IVStar) (Inner.ofFlock (ksB Coin).deploy decI) [] = 0 := by
    have : IsEmpty (Fin ((ksRV Coin (ksB Coin).IVStar).nS [])) := Fin.isEmpty'
    unfold Recursion.γ
    rw [Finset.univ_eq_empty, Finset.sum_empty, mul_zero]
    norm_num
  have hκ : Recursion.κ (ksRV Coin (ksB Coin).IVStar) (Inner.ofFlock (ksB Coin).deploy decI) [] =
      Claims.Flock.κ (Flock.ksD Coin) [] := ksRV_κ _ _ _
  have hκ6 := ksD_κ_lt_sixth (Coin := Coin) []
  have hκ0 := Flock.ksD_κ_pos (Coin := Coin) []
  have h1 : Real.logb 2 6 < Real.logb 2 (1 / Claims.Flock.κ (Flock.ksD Coin) []) :=
    Real.logb_lt_logb (by norm_num) (by norm_num) ((lt_one_div (by norm_num) hκ0).2 hκ6)
  have h6 : Real.logb 2 6 < 3 := by
    rw [Real.logb_lt_iff_lt_rpow (by norm_num) (by norm_num), show (3 : ℝ) = ((3 : ℕ) : ℝ) by norm_num,
      Real.rpow_natCast]
    norm_num
  have hs : (cBe (ksRV Coin (ksB Coin).IVStar) [] : ℝ) + setupLen (ksRV Coin (ksB Coin).IVStar) [] +
      2 * ((0 + 0 : ℕ) : ℝ) = 4 := by
    rw [hc, show setupLen (ksRV Coin (ksB Coin).IVStar) [] = 0 from rfl]
    norm_num
  unfold lamLin2 βdev2 NI
  rw [hx, hkV, hk, hw, hW, hs, hγ, hκ, Real.logb_zero, sub_pos]
  refine lt_min (lt_min (lt_min (lt_min (lt_min h1 ?_) ?_) ?_) ?_) ?_
  · linarith
  all_goals exact lt_sub_logb _ (by norm_num) (by norm_num) (by norm_num) h6

end Numbers

/-! ## The claims -/

section Claims

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- At `Flock.ksD` as the inner protocol, the silent prover never aborting is accepted on `[]`, where no witness is in
the relation. -/
theorem ksD_prBadAbort (H : Msg → Msg) (decode : List (ℕ × Unit) → List Unit) :
    (Inner.ofFlock (Flock.ksD Coin) decode).PrBadAbort (A := Unit) H [] silentP (fun _ _ _ => true) = 1 := by
  unfold Inner.PrBadAbort
  refine (Pr_congr fun s => iff_true_intro ⟨?_, ?_⟩).trans ZKMal.Pr_true
  · simp only [abortRun, ite_true]
    exact run_accV_out H silentP [] s.1 s.2.1.1
  · generalize ((erep _ _ _ _ _ _ _ _).eval _ _).1 = o
    cases o with
    | none => exact id
    | some w => exact Flock.ksD_noR [] w

/-- **Flock's KS with no root fails** at `Flock.ksD` with no cap position. -/
theorem KSFlockLin_fails (H : Msg → Msg) : ¬ KSFlockLin H (Flock.ksD Coin) noCaps := by
  intro h
  have h1 := h (fun _ => [()]) (Flock.listWithin_unit _ _ _) [] Unit silentP Unit (fun _ _ _ => true) 0
    (16 * ((Flock.ksD Coin).k [] + 1))
  have hB : PrBAbort (A := Unit) H (Flock.ksD Coin).V silentP [] (fun _ _ _ => true) = 0 :=
    prBAbort_vLog_nil H silentP _ [] _ fun ω r => run_accV_vLog H silentP [] ω r
  have hF : PrF0 (A := Unit) H (Flock.ksD Coin).V silentP [] (fun _ _ _ => true) ((Flock.ksD Coin).k [])
      (16 * ((Flock.ksD Coin).k [] + 1)) = 0 :=
    prF0_vLog_nil H silentP _ [] _ _ _ fun ω r => run_accV_vLog H silentP [] ω r
  have hW : (noCaps (Coin := Coin)).width [] = 0 := rfl
  have h2 := (ksD_prBadAbort (Coin := Coin) H fun _ => [()]).symm.le.trans h1
  rw [hB, hF, hW] at h2
  have := f0_tail_lt ((Flock.ksD Coin).k [])
  have := Flock.ksD_κ (Coin := Coin) []
  simp only [Nat.cast_zero, mul_zero, zero_mul, add_zero, zero_add] at h2
  linarith

/-- **The recursion's `KSLin2` fails** at `ksRAt`, `ksB` inside, `Flock.ksD` outside and no cap position: the bound
record's `κ` is `Flock.ksD`'s. -/
theorem KSLin2_fails (H : Msg → Msg) : ¬ KSLin2 H (ksRAt Coin) (ksB Coin) (Flock.ksD Coin) noCaps := by
  intro h
  have h1 := h (fun _ => [()]) (Flock.listWithin_unit _ _ _) (fun _ => [()]) (Flock.listWithin_unit _ _ _) []
    Unit silentP
    ⌈4 * βdev2 (ksRV Coin (ksB Coin).IVStar) (Inner.ofFlock (ksB Coin).deploy fun _ => [()]) noCaps []⌉₊
    (16 * ((Inner.ofFlock (ksB Coin).deploy fun _ => [()]).k (xI (ksRV Coin (ksB Coin).IVStar) []) + 1))
  try beta_reduce at h1
  have hW : (noCaps (Coin := Coin)).width (xI (ksRV Coin (ksB Coin).IVStar) []) = 0 := rfl
  rw [ksRAt_record, ksRV_prBad H _ _ _ [] (ksRV_noR _ _ []), ksRV_prB, ksRV_κ,
    prBinnerV_noCalls H (ksRV Coin (ksB Coin).IVStar) (fun _ _ _ => rfl) (fun _ _ => rfl),
    prFdevV_noCalls H (ksRV Coin (ksB Coin).IVStar) (fun _ _ _ => rfl) (fun _ _ => rfl),
    prF0I_noCalls H (ksRV Coin (ksB Coin).IVStar) (fun _ _ _ => rfl), hW] at h1
  have := tail_lt_quarter
    (βdev2 (ksRV Coin (ksB Coin).IVStar) (Inner.ofFlock (ksB Coin).deploy fun _ => [()]) noCaps [])
  have := f0_tail_lt ((Inner.ofFlock (ksB Coin).deploy fun _ => [()]).k (xI (ksRV Coin (ksB Coin).IVStar) []))
  have hκ : (Inner.ofFlock (ksB Coin).deploy fun _ => [()]).κ [] < 1 / 4 := ksD_κ_lt_quarter (Coin := Coin) []
  simp only [Nat.cast_zero, mul_zero, zero_mul, Real.sqrt_zero, add_zero, zero_add] at h1
  linarith

theorem expectedCR_charge_zero {H : Msg → Msg} {F : Finder} {n : ℕ} (h : ExpectedCR H F) (hn : n = 0) :
    ExpectedCR H (F.charge n) := by
  subst hn
  have hF : F.charge 0 = F := by
    cases F
    rfl
  rw [hF]
  exact h

/-- **The recursion's `BitsLin2` fails** at `ksRAt`, `ksB` inside, `Flock.ksD` outside and no cap position, where
`λ₂ > 0`. -/
theorem BitsLin2_fails (H : Msg → Msg) : ¬ BitsLin2 H (ksRAt Coin) (ksB Coin) (Flock.ksD Coin) noCaps := fun h =>
  bits_bound_fails H _ _ _ (ksRV_noR _ _ []) (ksB_lamLin2_pos _) <|
    h (fun _ => [()]) (Flock.listWithin_unit _ _ _) (fun _ => [()]) (Flock.listWithin_unit _ _ _) [] Unit silentP 1
      le_rfl (calls_silentP_accV H []) 0 (fun _ _ => le_rfl) 0 (fun _ => le_rfl) (strictBirthday_B_accV H silentP [])
      (fun hk => expectedCR_charge_zero (binnerV_expectedCR_noCalls H (ksRV Coin (ksB Coin).IVStar)
        (fun _ _ _ => rfl) (fun _ _ => rfl) _ _ _ silentP [] hk) rfl)
      (fun M hk => expectedCR_charge_zero (fdevV_expectedCR_noCalls H (ksRV Coin (ksB Coin).IVStar)
        (fun _ _ _ => rfl) (fun _ _ => rfl) _ _ _ silentP [] M hk) rfl)
      (fun _ hW => absurd hW (Nat.lt_irrefl 0))
      (fun R => expectedCR_charge_zero (f0I_expectedCR_noCalls H (ksRV Coin (ksB Coin).IVStar)
        (fun _ _ _ => rfl) _ _ _ silentP [] R) rfl)

end Claims

end Claims.Witness.Recursion
