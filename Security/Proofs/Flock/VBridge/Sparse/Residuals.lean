import Proofs.Flock.VBridge.Sparse.Defs
import Proofs.Flock.VBridge.Algebra.Build
import Proofs.Flock.Soundness.Refine.Arith
import Proofs.Flock.Soundness.Refine.Regions

/-!
# `RecSparse_v1`'s residuals and verifier row over the model's field

On ports of the sizes `sparseS` wants, `residuals` succeeds (`residuals_sparseS`) and residual `nm` is its `spec`
evaluated (`rget_eq`): its added terms plus its products, read through `fOf`; `specVal_*` spell each one out. The
verifier row's parts are the model's values at the coins (`v_*`): the equality tables are `eqAt` at the coins they
slice, the rounds' slots are `r` and `r^k + r`, and the two dots are `ŵ(ζ_r)` and `Ũ(r_s)`.
-/

namespace FlockVBridge.Sparse

open Flock Flock.RecAlgebra FlockSoundness.Refine FlockSoundness.Model

local notation "𝔽" => FlockLevel3.GF128

noncomputable section

variable (sh : SparseShape)

/-! ## Sums -/

theorem fOf_foldl_sum {β : Type} (f : β → F128) : ∀ (l : List β) (x : F128),
    fOf (l.foldl (fun x b => x + f b) x) = fOf x + (l.map fun b => fOf (f b)).sum
  | [], x => by simp
  | b :: l, x => by rw [List.foldl_cons, fOf_foldl_sum f l, fOf_add, List.map_cons, List.sum_cons, add_assoc]

theorem sum_range_list {M : Type} [AddCommMonoid M] (f : ℕ → M) :
    ∀ n, ((List.range n).map f).sum = ∑ i ∈ Finset.range n, f i
  | 0 => by simp
  | n + 1 => by
      rw [List.range_succ, List.map_append, List.sum_append, sum_range_list f n, Finset.sum_range_succ]
      simp

/-! ## The residuals -/

/-- A residual's value from its spec: its added terms plus its products. -/
def specVal (w v : Array F128) (s : List (Term × Term) × List Term) : 𝔽 :=
  (s.2.map fun t => fOf (termValue w v t)).sum +
    (s.1.map fun ab => fOf (termValue w v ab.1) * fOf (termValue w v ab.2)).sum

theorem fOf_resValue_lists (S : GfS) (w v : Array F128) (ps : List (ℕ × ℕ)) (ts : List Term) :
    fOf (resValue S w v ⟨ps.toArray, ts.toArray⟩) = (ts.map fun t => fOf (termValue w v t)).sum +
      (ps.map fun ab => fOf (opValue S w v ab.1) * fOf (opValue S w v ab.2)).sum := by
  simp only [resValue, List.foldl_toArray]
  rw [fOf_foldl_sum, fOf_foldl_sum, fOf_zero, zero_add]
  simp only [fOf_mul]

theorem sparseS_nw : (sparseS sh).nw = nw sh := rfl
theorem sparseS_nv : (sparseS sh).nv = nv sh := rfl
theorem sparseS_ops : (sparseS sh).ops = ((opList sh).map fun t => #[t]).toArray := rfl
theorem sparseS_res : (sparseS sh).res = ((names sh).map (resOf sh)).toArray := rfl

theorem mem_opTerms {nm : RName} (hnm : nm ∈ names sh) {ab : Term × Term} (hab : ab ∈ (spec sh nm).1) :
    ab.1 ∈ opTerms sh ∧ ab.2 ∈ opTerms sh := by
  unfold opTerms
  exact ⟨List.mem_flatMap.mpr ⟨nm, hnm, List.mem_flatMap.mpr ⟨ab, hab, by simp⟩⟩,
    List.mem_flatMap.mpr ⟨nm, hnm, List.mem_flatMap.mpr ⟨ab, hab, by simp⟩⟩⟩

theorem opValue_opIx (w v : Array F128) {t : Term} (ht : t ∈ opTerms sh) :
    opValue (sparseS sh) w v (opIx sh t) = termValue w v t := by
  have hl : (opList sh).idxOf t < (opList sh).length := List.idxOf_lt_length_of_mem (List.mem_eraseDups.mpr ht)
  refine opValue_single w v ?_
  rw [sparseS_ops, opIx, List.getElem?_toArray, List.getElem?_map, List.getElem?_eq_getElem hl,
    List.getElem_idxOf hl]
  rfl

theorem fOf_resOf (w v : Array F128) {nm : RName} (hnm : nm ∈ names sh) :
    fOf (resValue (sparseS sh) w v (resOf sh nm)) = specVal w v (spec sh nm) := by
  rw [resOf, fOf_resValue_lists, List.map_map, specVal]
  refine congrArg (_ + ·) (congrArg List.sum (List.map_congr_left fun ab hab => ?_))
  obtain ⟨h1, h2⟩ := mem_opTerms sh hnm hab
  simp only [Function.comp_apply, opValue_opIx sh w v h1, opValue_opIx sh w v h2]

/-- The residuals `residuals_sparseS` gives. -/
abbrev resArr (w v : Array F128) : Array F128 := ((names sh).map (resOf sh)).toArray.map (resValue (sparseS sh) w v)

theorem residuals_sparseS (w v : Array F128) (hw : w.size = nw sh) (hv : v.size = nv sh) :
    residuals (sparseS sh) w v = .ok (resArr sh w v) := by
  unfold residuals
  rw [ite_eq_right (by simp [sparseS_nw, sparseS_nv, hw, hv]), sparseS_res]

theorem rget_eq (w v : Array F128) {nm : RName} (hnm : nm ∈ names sh) :
    fOf (rget sh (resArr sh w v) nm) = specVal w v (spec sh nm) := by
  have hl : (names sh).idxOf nm < (names sh).length := List.idxOf_lt_length_of_mem hnm
  unfold rget
  rw [getElem!_pos _ _ (by simpa [resArr] using hl)]
  simp only [resArr, Array.getElem_map, List.getElem_toArray, List.getElem_map, List.getElem_idxOf hl]
  exact fOf_resOf sh w v hnm

/-! ## Each residual -/

theorem mem_h1 : RName.h1 ∈ names sh := by simp [names]
theorem mem_pre {j : ℕ} (hj : j < 6) : RName.pre j ∈ names sh := by simp [names, hj]
theorem mem_wv : RName.wv ∈ names sh := by simp [names]
theorem mem_sc {j : ℕ} (hj : j < nR sh) : RName.sc j ∈ names sh := by simp [names, hj]
theorem mem_f0 : RName.f0 ∈ names sh := by simp [names]
theorem mem_fx {j : ℕ} (hj : j < sh.ell) : RName.fx j ∈ names sh := by simp [names, hj]
theorem mem_fy {j : ℕ} (hj : j < sh.ell) : RName.fy j ∈ names sh := by simp [names, hj]
theorem mem_bp : RName.bp ∈ names sh := by simp [names]
theorem mem_claim (ht : sh.tLog ≠ 0) : RName.claim ∈ names sh := by simp [names, ht]

variable (w v : Array F128)

theorem specVal_h1 : specVal w v (spec sh .h1) = ∑ s ∈ Finset.range 64, fOf v[s]! * fOf w[s]! := by
  simp only [specVal, spec, List.map_nil, List.sum_nil, zero_add, List.map_map, Function.comp_def, termValue]
  rw [sum_range_list]

theorem specVal_pre (j : ℕ) :
    specVal w v (spec sh (.pre j)) = fOf w[wPre j]! + fOf w[wPre j + 1]! * fOf v[vPre j + 1]! := by
  simp only [specVal, spec, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero, termValue]

theorem specVal_wv : specVal w v (spec sh .wv) = fOf v[vWhat]! * fOf w[wV2]! := by
  simp only [specVal, spec, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero, zero_add, termValue]

theorem specVal_sc (j : ℕ) : specVal w v (spec sh (.sc j)) = fOf w[wSc sh j]! +
    ∑ i ∈ Finset.range (deg sh j - 1), fOf w[wSc sh j + (i + 1)]! * fOf v[vSc sh j + (i + 1)]! := by
  simp only [specVal, spec, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero, List.map_map,
    Function.comp_def, termValue]
  rw [sum_range_list]

theorem specVal_f0 : specVal w v (spec sh .f0) =
    fOf w[wU sh + (2 * sh.ell + 1)]! + fOf v[vAlpha sh]! * fOf w[wU sh + 2 * sh.ell]! := by
  simp only [specVal, spec, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero, termValue]

theorem specVal_fx (j : ℕ) : specVal w v (spec sh (.fx j)) = fOf v[vAx sh + j]! + fOf w[wU sh + j]! := by
  simp only [specVal, spec, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero, termValue]

theorem specVal_fy (j : ℕ) :
    specVal w v (spec sh (.fy j)) = fOf v[vAy sh + j]! + fOf w[wU sh + (sh.ell + j)]! := by
  simp only [specVal, spec, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero, termValue]

theorem specVal_bp : specVal w v (spec sh .bp) = (if sh.tLog = 0 then fOf w[wClaim sh]! else 0) +
    ∑ p ∈ Finset.range (2 ^ sh.pLog), fOf v[vEqp sh + p]! * fOf w[wU sh + p]! := by
  simp only [specVal, spec, List.map_map, Function.comp_def, termValue]
  rw [sum_range_list]
  split_ifs <;> simp

theorem specVal_claim : specVal w v (spec sh .claim) = fOf w[wClaim sh]! +
    ∑ t ∈ Finset.range (2 ^ sh.tLog), fOf v[vEqt sh + t]! * fOf w[wVt sh + t]! := by
  simp only [specVal, spec, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero, List.map_map,
    Function.comp_def, termValue]
  rw [sum_range_list]

/-! ## The rounds' slots -/

theorem degSum_succ (j : ℕ) : degSum sh (j + 1) = degSum sh j + deg sh j := by
  unfold degSum deg
  have e0 : (j + 1) * (2 * sh.ell + 1) = j * (2 * sh.ell + 1) + (2 * sh.ell + 1) := by ring
  split_ifs <;> omega

theorem degSum_mono {j j' : ℕ} (h : j ≤ j') : degSum sh j ≤ degSum sh j' := by
  unfold degSum
  have := Nat.mul_le_mul_right (2 * sh.ell + 1) h
  omega

theorem degSum_add_lt {j k : ℕ} (hj : j < nR sh) (hk : k < deg sh j) : degSum sh j + k < degSum sh (nR sh) := by
  have := degSum_mono sh (show j + 1 ≤ nR sh by omega)
  rw [degSum_succ] at this
  omega

theorem roundOf_degSum {j k : ℕ} (hk : k < deg sh j) : roundOf sh (degSum sh j + k) = j := by
  unfold deg at hk
  unfold roundOf degSum
  by_cases hj : j < sh.nu
  · rw [ite_eq_left hj] at hk
    have h1 := Nat.mul_le_mul_right (2 * sh.ell + 1) (show j + 1 ≤ sh.nu by omega)
    have e0 : (j + 1) * (2 * sh.ell + 1) = j * (2 * sh.ell + 1) + (2 * sh.ell + 1) := by ring
    rw [show j * (2 * sh.ell + 1) + (j - sh.nu) + k = j * (2 * sh.ell + 1) + k by omega, ite_eq_left (by omega)]
    exact Nat.div_eq_of_lt_le (by omega) (by omega)
  · rw [ite_eq_right hj] at hk
    obtain ⟨e, rfl⟩ : ∃ e, j = sh.nu + e := ⟨j - sh.nu, by omega⟩
    have e1 : (sh.nu + e) * (2 * sh.ell + 1) = sh.nu * (2 * sh.ell + 1) + e * (2 * sh.ell + 1) := by ring
    have e2 : (e + 1) * (2 * sh.ell + 2) = e * (2 * sh.ell + 2) + (2 * sh.ell + 2) := by ring
    have e3 : e * (2 * sh.ell + 2) = e * (2 * sh.ell + 1) + e := by ring
    rw [ite_eq_right (by omega), show (sh.nu + e) * (2 * sh.ell + 1) + (sh.nu + e - sh.nu) + k -
      sh.nu * (2 * sh.ell + 1) = e * (2 * sh.ell + 2) + k by omega]
    congr 1
    exact Nat.div_eq_of_lt_le (by omega) (by omega)

theorem fOf_powN (r : F128) : ∀ n, fOf (powN r n) = fOf r ^ n
  | 0 => fOf_one
  | n + 1 => by rw [powN, fOf_mul, fOf_powN r n, pow_succ]

/-! ## The verifier's row -/

variable (c : SCoins)

theorem offs : vAlpha sh = 77 + degSum sh (nR sh) ∧ vAx sh = vAlpha sh + 1 ∧ vAy sh = vAx sh + sh.ell ∧
    vUr sh = vAy sh + sh.ell ∧ vEqp sh = vUr sh + 1 ∧ vEqs sh = vEqp sh + 2 ^ sh.pLog ∧
    vEqt sh = vEqs sh + 2 ^ sh.sLog := ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl⟩

/-- `omega` takes `2 ^ k` as an atom it doesn't know is positive. -/
theorem pows : 0 < 2 ^ sh.pLog ∧ 0 < 2 ^ sh.sLog ∧ 0 < 2 ^ sh.tLog :=
  ⟨Nat.two_pow_pos _, Nat.two_pow_pos _, Nat.two_pow_pos _⟩

theorem vEqp_lt_nv : vEqp sh + 2 ^ sh.pLog ≤ nv sh := by
  unfold nv
  exact Nat.le_add_right _ _

theorem nv_of_tLog (ht : sh.tLog ≠ 0) : nv sh = vEqt sh + 2 ^ sh.tLog := by
  unfold nv vEqt vEqs
  rw [ite_eq_right ht]
  omega

theorem vrow_get {i : ℕ} (hi : i < nv sh) : (verifierRow sh c)[i]! = vAt sh c i := by
  unfold verifierRow
  rw [getElem!_pos _ _ (by simpa using hi), Array.getElem_ofFn]

theorem slice_size {a n : ℕ} (h : a + n ≤ c.cs.size) : (slice c a n).size = n := by
  simp only [slice, Array.size_extract]
  omega

theorem eqTable_slice (a n : ℕ) (hr : a + n ≤ c.cs.size) (b : ℕ) (hb : b < 2 ^ n) :
    fOf (eqTable (slice c a n))[b]! = FlockSoundness.Model.eqAt (fun j : Fin n => fOf c.cs[a + j.val]!) b := by
  unfold slice FlockSoundness.Model.eqAt
  exact (eqTable_extract c.cs a n hr b hb).trans (Finset.prod_range _)

theorem v_eqc (hc : 6 ≤ c.cs.size) {s : ℕ} (hs : s < 64) :
    fOf (verifierRow sh c)[s]! = FlockSoundness.Model.eqAt (fun j : Fin 6 => fOf c.cs[0 + j.val]!) s := by
  obtain ⟨o1, o2, o3, o4, o5, o6, o7⟩ := offs sh
  obtain ⟨p1, p2, p3⟩ := pows sh
  have := vEqp_lt_nv sh
  rw [vrow_get sh c (by omega), vAt, ite_eq_left hs]
  exact eqTable_slice c 0 6 (by omega) s (lt_of_lt_of_le hs (by norm_num))

theorem v_pre0 {j : ℕ} (hj : j < 6) : fOf (verifierRow sh c)[vPre j]! = fOf c.cs[6 + j]! := by
  obtain ⟨o1, o2, o3, o4, o5, o6, o7⟩ := offs sh
  obtain ⟨p1, p2, p3⟩ := pows sh
  have := vEqp_lt_nv sh
  unfold vPre
  rw [vrow_get sh c (by omega), vAt, ite_eq_right (by omega), ite_eq_left (by omega), ite_eq_left (by omega),
    show (64 + 2 * j - 64) / 2 = j by omega]

theorem v_pre1 {j : ℕ} (hj : j < 6) : fOf (verifierRow sh c)[vPre j + 1]! =
    fOf c.cs[6 + j]! * fOf c.cs[6 + j]! + fOf c.cs[6 + j]! := by
  obtain ⟨o1, o2, o3, o4, o5, o6, o7⟩ := offs sh
  obtain ⟨p1, p2, p3⟩ := pows sh
  have := vEqp_lt_nv sh
  unfold vPre
  rw [vrow_get sh c (by omega), vAt, ite_eq_right (by omega), ite_eq_left (by omega), ite_eq_right (by omega),
    show (64 + 2 * j + 1 - 64) / 2 = j by omega, fOf_add, fOf_mul]

theorem v_what (hc : 12 ≤ c.cs.size) : fOf (verifierRow sh c)[vWhat]! =
    ∑ s : Fin 64, FlockSoundness.Model.eqAt (fun j : Fin 6 => fOf c.cs[6 + j.val]!) s.val *
      fOf (skipWeights c.z)[s.val]! := by
  obtain ⟨o1, o2, o3, o4, o5, o6, o7⟩ := offs sh
  obtain ⟨p1, p2, p3⟩ := pows sh
  have := vEqp_lt_nv sh
  unfold vWhat
  rw [vrow_get sh c (by omega), vAt, ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_left rfl,
    fOf_dot _ _ 64 (by rw [FlockLevel3.eqTable_size, slice_size c (show 6 + 6 ≤ c.cs.size by omega)]; norm_num)]
  refine Finset.sum_congr rfl fun s _ => ?_
  rw [eqTable_slice c 6 6 (by omega) s.val (lt_of_lt_of_le s.isLt (by norm_num))]

theorem v_sc {j k : ℕ} (hj : j < nR sh) (hk : k < deg sh j) : fOf (verifierRow sh c)[vSc sh j + k]! =
    if k = 0 then fOf c.cs[12 + j]! else fOf c.cs[12 + j]! ^ (k + 1) + fOf c.cs[12 + j]! := by
  obtain ⟨o1, o2, o3, o4, o5, o6, o7⟩ := offs sh
  obtain ⟨p1, p2, p3⟩ := pows sh
  have := vEqp_lt_nv sh
  have hlt := degSum_add_lt sh hj hk
  have hr := roundOf_degSum sh hk
  unfold vSc
  rw [vrow_get sh c (by omega), vAt, ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega),
    ite_eq_left (by omega), show 77 + degSum sh j + k - 77 = degSum sh j + k by omega, hr,
    show degSum sh j + k - degSum sh j = k by omega]
  split_ifs
  · rfl
  · rw [fOf_add, fOf_powN]

theorem v_alpha : fOf (verifierRow sh c)[vAlpha sh]! = fOf c.alpha := by
  obtain ⟨o1, o2, o3, o4, o5, o6, o7⟩ := offs sh
  obtain ⟨p1, p2, p3⟩ := pows sh
  have := vEqp_lt_nv sh
  rw [vrow_get sh c (by omega), vAt, ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega),
    ite_eq_right (by omega), ite_eq_left rfl]

theorem v_ax {j : ℕ} (hj : j < sh.ell) : fOf (verifierRow sh c)[vAx sh + j]! = fOf (xLo c j) + 1 := by
  obtain ⟨o1, o2, o3, o4, o5, o6, o7⟩ := offs sh
  obtain ⟨p1, p2, p3⟩ := pows sh
  have := vEqp_lt_nv sh
  rw [vrow_get sh c (by omega), vAt, ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega),
    ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_left (by omega), Nat.add_sub_cancel_left, fOf_add,
    fOf_one]

theorem v_ay {j : ℕ} (hj : j < sh.ell) : fOf (verifierRow sh c)[vAy sh + j]! = fOf (yLo c j) + 1 := by
  obtain ⟨o1, o2, o3, o4, o5, o6, o7⟩ := offs sh
  obtain ⟨p1, p2, p3⟩ := pows sh
  have := vEqp_lt_nv sh
  rw [vrow_get sh c (by omega), vAt, ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega),
    ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_left (by omega),
    Nat.add_sub_cancel_left, fOf_add, fOf_one]

theorem v_ur (hc : 12 + nR sh ≤ c.cs.size) : fOf (verifierRow sh c)[vUr sh]! =
    ∑ σ : Fin (2 ^ sh.sLog), FlockSoundness.Model.eqAt (fun j : Fin sh.sLog => fOf c.cs[12 + sh.nu + j.val]!) σ.val *
      fOf c.U[σ.val]! := by
  obtain ⟨o1, o2, o3, o4, o5, o6, o7⟩ := offs sh
  obtain ⟨p1, p2, p3⟩ := pows sh
  have := vEqp_lt_nv sh
  have hn : 12 + sh.nu + sh.sLog ≤ c.cs.size := by unfold nR at hc; omega
  rw [vrow_get sh c (by omega), vAt, ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega),
    ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega),
    ite_eq_left rfl, fOf_dot _ _ (2 ^ sh.sLog) (by rw [FlockLevel3.eqTable_size, slice_size c hn])]
  refine Finset.sum_congr rfl fun σ _ => ?_
  rw [eqTable_slice c (12 + sh.nu) sh.sLog hn σ.val σ.isLt]

/-- Past `vEqp`, `vAt` reads the last three tables. -/
theorem vAt_tail {i : ℕ} (hi : vEqp sh ≤ i) : vAt sh c i =
    if i < vEqs sh then (eqTable (slice c (12 + nR sh) sh.pLog))[i - vEqp sh]!
    else if i < vEqt sh then (eqTable (slice c (12 + sh.nu) sh.sLog))[i - vEqs sh]!
    else (eqTable (slice c (12 + nR sh + sh.pLog) sh.tLog))[i - vEqt sh]! := by
  obtain ⟨o1, o2, o3, o4, o5, o6, o7⟩ := offs sh
  rw [vAt, ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega),
    ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega)]

theorem v_eqp (hc : 12 + nR sh + sh.pLog ≤ c.cs.size) {p : ℕ} (hp : p < 2 ^ sh.pLog) :
    fOf (verifierRow sh c)[vEqp sh + p]! =
      FlockSoundness.Model.eqAt (fun j : Fin sh.pLog => fOf c.cs[12 + nR sh + j.val]!) p := by
  obtain ⟨o1, o2, o3, o4, o5, o6, o7⟩ := offs sh
  have := vEqp_lt_nv sh
  rw [vrow_get sh c (by omega), vAt_tail sh c (by omega), ite_eq_left (by omega), Nat.add_sub_cancel_left]
  exact eqTable_slice c _ _ hc p hp

theorem v_eqs (ht : sh.tLog ≠ 0) (hc : 12 + nR sh ≤ c.cs.size) {s : ℕ} (hs : s < 2 ^ sh.sLog) :
    fOf (verifierRow sh c)[vEqs sh + s]! =
      FlockSoundness.Model.eqAt (fun j : Fin sh.sLog => fOf c.cs[12 + sh.nu + j.val]!) s := by
  obtain ⟨o1, o2, o3, o4, o5, o6, o7⟩ := offs sh
  obtain ⟨p1, p2, p3⟩ := pows sh
  have := nv_of_tLog sh ht
  rw [vrow_get sh c (by omega), vAt_tail sh c (by omega), ite_eq_right (by omega), ite_eq_left (by omega),
    Nat.add_sub_cancel_left]
  exact eqTable_slice c _ _ (by unfold nR at hc; omega) s hs

theorem v_eqt (ht : sh.tLog ≠ 0) (hc : nc sh ≤ c.cs.size) {t : ℕ} (htt : t < 2 ^ sh.tLog) :
    fOf (verifierRow sh c)[vEqt sh + t]! =
      FlockSoundness.Model.eqAt (fun j : Fin sh.tLog => fOf c.cs[12 + nR sh + sh.pLog + j.val]!) t := by
  obtain ⟨o1, o2, o3, o4, o5, o6, o7⟩ := offs sh
  obtain ⟨p1, p2, p3⟩ := pows sh
  have := nv_of_tLog sh ht
  rw [vrow_get sh c (by omega), vAt_tail sh c (by omega), ite_eq_right (by omega), ite_eq_right (by omega),
    Nat.add_sub_cancel_left]
  exact eqTable_slice c _ _ (by unfold nc at hc; omega) t htt

end

end FlockVBridge.Sparse
