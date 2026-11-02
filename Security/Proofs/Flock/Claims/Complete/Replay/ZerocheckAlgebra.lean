import Proofs.Flock.Claims.Complete.Replay.Zerocheck
import Proofs.Flock.Soundness.Discharge.Composed.Inner.Zerocheck
import Proofs.Flock.Soundness.Refine.Regions

/-!
# The honest prover's zerocheck passes its final check

`fold_final`: when the trace satisfies the block R1CS bitwise (`az ∧ bz = z`) and no `r_outer` coin is `1`, the residual
claim the verifier folds over the prover's zerocheck (`zcFold`, from `zcStart`) ends at `final_a · final_b`, the
hypothesis `zerocheckWith_of` asks for.

The prover's messages (`Zc.proof`, in the executable's `sumF`, `eqTable`, `skipHat` and `foldLow`) are the soundness
analysis's honest messages (`Zerocheck.honest`, in the field's sums, `eqR`, `skipAt` and `foldN`) on the trace's bits
(`bitF`), the zerocheck vector and the residual coins; the fold is that analysis's replayed claim (`zcFold_fst`), and
`Zerocheck.final_ab` closes it.
-/

namespace FlockComplete.Replay.Zc

open Flock FlockLevel3 FlockSoundness.Discharge.Composed.Zerocheck

local notation "𝔽" => FlockLevel3.GF128

/-- A bit table as field values. -/
noncomputable def bitF (bits : Array Bool) (i : ℕ) : 𝔽 := if bits[i]! then 1 else 0

/-! ## The executable's sums as the field's -/

theorem of_sumF (n : ℕ) (f : ℕ → F128) : GF128.of (sumF n f) = ∑ x ∈ Finset.range n, GF128.of (f x) := by
  unfold sumF
  suffices h : ∀ acc : F128, GF128.of ((List.range n).foldl (fun acc i => acc + f i) acc) =
      GF128.of acc + ∑ x ∈ Finset.range n, GF128.of (f x) by
    rw [h, ← GF128.zero_eq, zero_add]
  induction n with
  | zero => intro acc; simp
  | succ n ih =>
    intro acc
    rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil, of_add, ih, Finset.sum_range_succ,
      add_assoc]

theorem of_skipHat (bits : Array Bool) (ζ : F128) (x : ℕ) :
    GF128.of (skipHat bits (skipWeights ζ) x) = skipAt (bitF bits) (GF128.of ζ) x := by
  unfold skipHat skipAt
  rw [of_sumF, Finset.sum_range]
  refine Finset.sum_congr rfl fun s _ => ?_
  unfold bitF lagS
  by_cases hb : bits[s.val + 64 * x]! = true
  · simp only [hb, ↓reduceIte, mul_one]
    rfl
  · simp only [hb, Bool.false_eq_true, ↓reduceIte, mul_zero]
    exact GF128.zero_eq.symm

theorem of_eqTable (r : Array F128) (b : ℕ) (hb : b < 2 ^ r.size) :
    GF128.of (eqTable r)[b]! = eqR (coin r) 0 r.size b := by
  rw [eqTable_eqAt r b hb]
  unfold eqAt eqR
  rw [Finset.prod_range]
  refine Finset.prod_congr rfl fun j _ => ?_
  simp only [zero_add]
  rfl

theorem of_eqTable_extract (r : Array F128) (a hi : ℕ) (hhi : hi = r.size) (ha : a ≤ hi) (b : ℕ)
    (hb : b < 2 ^ (hi - a)) : GF128.of (eqTable (r.extract a hi))[b]! = eqR (coin r) a (hi - a) b := by
  have h := FlockSoundness.Refine.eqTable_extract r a (hi - a) (by omega) b hb
  rw [show a + (hi - a) = hi by omega] at h
  rw [h]
  rfl

/-! ## The prover's tables and messages -/

theorem tab_size (n : ℕ) (z : F128) (ρ : ℕ → F128) (bits : Array Bool) :
    ∀ i ≤ n, (tab n z ρ bits i).size = 2 ^ (n - i)
  | 0, _ => by simp [tab]
  | i + 1, hi => by
    simp only [tab, foldLow, Array.size_map, Array.size_range]
    rw [tab_size n z ρ bits i (by omega), show n - i = n - (i + 1) + 1 by omega, pow_succ,
      Nat.mul_div_cancel _ (by norm_num)]

/-- **The prover's folded tables are the honest ones**: its table `i` at row `x` is the skip row at `z`, folded at the
residual coins `rho` (which read `ρ`). -/
theorem of_tab (n : ℕ) (z : F128) (ρ : ℕ → F128) (bits : Array Bool) (rho : Array F128)
    (hρ : ∀ k < n, rho[k]! = ρ k) :
    ∀ i ≤ n, ∀ x < 2 ^ (n - i),
      GF128.of (tab n z ρ bits i)[x]! = foldN (coin rho) i (skipAt (bitF bits) (GF128.of z)) x
  | 0, _, x, hx => by
    simp only [tab, foldN]
    rw [get_map_range _ (by simpa using hx)]
    exact of_skipHat bits z x
  | i + 1, hi, x, hx => by
    have hs := tab_size n z ρ bits i (by omega)
    have hx' : x < (tab n z ρ bits i).size / 2 := by
      rw [hs, show n - i = n - (i + 1) + 1 by omega, pow_succ, Nat.mul_div_cancel _ (by norm_num)]
      exact hx
    have h2 : 2 * x + 1 < 2 ^ (n - i) := by
      rw [show n - i = n - (i + 1) + 1 by omega, pow_succ]
      omega
    simp only [tab, foldN, foldLo, foldLow]
    rw [get_map_range _ hx', of_add, of_mul, of_add, of_tab n z ρ bits rho hρ i (by omega) _ (by omega),
      of_tab n z ρ bits rho hρ i (by omega) _ h2]
    simp only [coin, hρ i (by omega)]

/-- **The prover's residual round `j` is the honest message.** -/
theorem msg_eq (tr : Trace) (n : ℕ) (r : Array F128) (hr : r.size = n) (z : F128) (ρ : ℕ → F128)
    (rho : Array F128) (hρ : ∀ k < n, rho[k]! = ρ k) (j : ℕ) (hj : j < n) :
    msg tr n r z ρ j =
      ((roundMsg n (coin r) (coin rho) (skipAt (bitF tr.az) (GF128.of z)) (skipAt (bitF tr.bz) (GF128.of z)) j).1.val,
       (roundMsg n (coin r) (coin rho) (skipAt (bitF tr.az) (GF128.of z)) (skipAt (bitF tr.bz) (GF128.of z)) j).2.val) := by
  have hs : (tab n z ρ tr.az j).size / 2 = 2 ^ (n - j - 1) := by
    rw [tab_size n z ρ tr.az j hj.le, show n - j = n - j - 1 + 1 by omega, pow_succ,
      Nat.mul_div_cancel _ (by norm_num)]
    rfl
  have hk : ∀ k < 2 ^ (n - j - 1), 2 * k + 1 < 2 ^ (n - j) := fun k hk => by
    rw [show n - j = n - j - 1 + 1 by omega, pow_succ]
    omega
  have he : ∀ k < 2 ^ (n - j - 1), GF128.of (eqTable (r.extract (j + 1) n))[k]! = eqR (coin r) (j + 1) (n - j - 1) k :=
    fun k hk' => by
      rw [of_eqTable_extract r (j + 1) n hr.symm (by omega) k (by rw [show n - (j + 1) = n - j - 1 by omega]; exact hk'),
        show n - (j + 1) = n - j - 1 by omega]
  have ha := of_tab n z ρ tr.az rho hρ j hj.le
  have hb := of_tab n z ρ tr.bz rho hρ j hj.le
  unfold msg roundMsg
  dsimp only
  rw [hs]
  refine Prod.ext (eq_of_of ?_) (eq_of_of ?_)
  · rw [of_sumF, of_val]
    refine Finset.sum_congr rfl fun k hk' => ?_
    have hk'' := Finset.mem_range.mp hk'
    rw [of_mul, of_mul, he k hk'', ha _ (hk k hk''), hb _ (hk k hk''), mul_assoc]
  · rw [of_sumF, of_val]
    refine Finset.sum_congr rfl fun k hk' => ?_
    have hk'' := Finset.mem_range.mp hk'
    rw [of_mul, of_mul, of_add, of_add, he k hk'', ha _ (hk k hk''), hb _ (hk k hk''), ha _ (by have := hk k hk''; omega),
      hb _ (by have := hk k hk''; omega)]
    ring

/-! ## The fold is the replayed claim -/

/-- The verifier's residual fold (`zcFold`) is the soundness analysis's replayed claim on the same coins. -/
theorem zcFold_fst (r : Array F128) (p : ZerocheckProof) (C : ℕ → Array F128) (rho : Array F128) (N : ℕ)
    (hC : ∀ k < N, (C k).getD 0 0 = rho[k]!) (z : F128) :
    ∀ n ≤ N, (Flock.RecAlgebra.zcFold r p C n (Flock.RecAlgebra.zcStart p z)).1 = claim p r rho z n
  | 0, _ => rfl
  | n + 1, hn => by
    have ih := zcFold_fst r p C rho N hC z n (by omega)
    unfold Flock.RecAlgebra.zcFold at ih ⊢
    rw [List.range'_concat, List.foldl_append, List.foldl_cons, List.foldl_nil, claim, ← ih]
    simp only [FlockSoundness.Refine.zStep, step, Nat.zero_add, Nat.one_mul]
    rw [hC n (by omega)]
    apply eq_of_of
    simp only [of_add, of_mul, of_inv, of_one']
    ring

theorem claim_congr {p q : ZerocheckProof} (r rho : Array F128) (z : F128) (n : ℕ)
    (h1 : p.round1ab = q.round1ab) (h2 : p.round1c = q.round1c) (h3 : ∀ j < n, p.rounds[j]! = q.rounds[j]!) :
    ∀ j ≤ n, claim p r rho z j = claim q r rho z j
  | 0, _ => by simp only [claim, claim0, h1, h2]
  | j + 1, hj => by rw [claim, claim, claim_congr r rho z n h1 h2 h3 j (by omega), h3 j (by omega)]

/-! ## The final check -/

theorem r_size (m : ℕ) (hm : 13 ≤ m) (cs : Challenges) : (cs.r m).size = m - 6 := by
  simp only [Challenges.r, Array.size_append, pinned_size, Array.size_map, Array.size_range]
  omega

theorem r1c_eq (tr : Trace) (n : ℕ) (r : Array F128) (hrs : r.size = n) (a b : ℕ → 𝔽) (rho : Array F128)
    (z : F128) : r1c tr n r = (honest n a b (bitF tr.z) r rho z).round1c := by
  unfold r1c
  simp only [honest]
  congr 1
  funext i
  apply eq_of_of
  rw [of_sumF, of_val]
  refine Finset.sum_congr rfl fun x hx => ?_
  rw [of_mul, of_eqTable _ x (by rw [hrs]; exact Finset.mem_range.mp hx), hrs, of_skipHat]
  rfl

/-- **The prover's zerocheck passes the final check** (Piop.lean:98): for a trace satisfying the block R1CS bitwise and
`r_outer` coins other than `1`, the residual fold ends at `final_a · final_b`. -/
theorem fold_final {m : ℕ} (hm : 14 ≤ m) (cs : Challenges) (tr : Trace)
    (hR1CS : ∀ i < 2 ^ m, (tr.az[i]! && tr.bz[i]!) = tr.z[i]!) (hc1 : ∀ k < m - 13, cs.c1 k ≠ 1) :
    (Flock.RecAlgebra.zcFold (cs.r m) (proof m cs tr) (fun i => #[cs.ρ i]) (m - 6)
      (Flock.RecAlgebra.zcStart (proof m cs tr) cs.z)).1 = (proof m cs tr).finalA * (proof m cs tr).finalB := by
  set n := m - 6 with hn
  set rho := pushes cs.ρ n
  have hrs : (cs.r m).size = n := r_size m (by omega) cs
  have hρ : ∀ k < n, rho[k]! = cs.ρ k := pushes_get cs.ρ n
  let H := honest n (bitF tr.az) (bitF tr.bz) (bitF tr.z) (cs.r m) rho cs.z
  have hvalid : ∀ i < 2 ^ (n + 6), bitF tr.az i * bitF tr.bz i = bitF tr.z i := by
    intro i hi
    rw [show n + 6 = m by omega] at hi
    unfold bitF
    rw [← hR1CS i hi]
    cases tr.az[i]! <;> cases tr.bz[i]! <;> simp
  have hr : ∀ k < n, (cs.r m)[k]! ≠ 1 := fun k _ =>
    pinned_append_ne_one _ (fun k hk => by
      rw [Array.size_map, Array.size_range] at hk
      rw [get_map_range _ hk]
      exact hc1 k hk) k
  have h1 : (proof m cs tr).round1ab = H.round1ab := by
    show r1ab tr n (cs.r m) = H.round1ab
    unfold r1ab
    simp only [H, honest]
    congr 1
    funext i
    apply eq_of_of
    rw [of_sumF, of_val]
    refine Finset.sum_congr rfl fun x hx => ?_
    rw [of_mul, of_mul, of_eqTable _ x (by rw [hrs]; exact Finset.mem_range.mp hx), hrs, of_skipHat, of_skipHat,
      mul_assoc]
    rfl
  have h2 : (proof m cs tr).round1c = H.round1c := r1c_eq tr n (cs.r m) hrs _ _ rho cs.z
  have h3 : ∀ j < n, (proof m cs tr).rounds[j]! = H.rounds[j]! := by
    intro j hj
    show (pushes (msg tr n (cs.r m) cs.z cs.ρ) n)[j]! = H.rounds[j]!
    rw [pushes_get _ n j hj]
    simp only [H, honest]
    rw [get_map_range _ hj]
    exact msg_eq tr n (cs.r m) hrs cs.z cs.ρ rho hρ j hj
  have hA : (proof m cs tr).finalA = H.finalA := by
    apply eq_of_of
    show GF128.of (tab n cs.z cs.ρ tr.az n)[0]! = _
    rw [of_tab n cs.z cs.ρ tr.az rho hρ n le_rfl 0 (by simp)]
    rfl
  have hB : (proof m cs tr).finalB = H.finalB := by
    apply eq_of_of
    show GF128.of (tab n cs.z cs.ρ tr.bz n)[0]! = _
    rw [of_tab n cs.z cs.ρ tr.bz rho hρ n le_rfl 0 (by simp)]
    rfl
  rw [zcFold_fst (cs.r m) (proof m cs tr) (fun i => #[cs.ρ i]) rho n (fun k hk => (hρ k hk).symm) cs.z n le_rfl,
    claim_congr (cs.r m) rho cs.z n h1 h2 h3 n le_rfl, final_ab n _ _ _ (cs.r m) rho cs.z hvalid hr, hA, hB]

end FlockComplete.Replay.Zc
