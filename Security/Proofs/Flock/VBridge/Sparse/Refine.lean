import Proofs.Flock.VBridge.Sparse.Plays

/-!
# `RecSparse_v1` accepts exactly when the rep's sparse phase does

`sparse_refines`: on w of the message ports' size and the phase's coins, `RecSparse_v1`'s output over F at the
verifier's row (`out`) is zero, residuals and tail, exactly when every slot's type is below `T` (`TypesIn`) and the run
of the model's `sparsePhase` on w's messages and the coins (`sparsePhase_plays`) ends at the registered claim with the
`claim` port's value (`sparseClaim`). Each output residual is one of the model's checks written as a sum that is zero
in characteristic 2: H2's `c₂ + ŵ·v₂`, H3's `c₄ + Ũ·Φ`, then with `T > 1` H4's batch against the type-selected sum and
the claim against `Σ_t eq(ρ_t, t)·vt_t`, else the claim against the batch. The type-selected weights read the types'
low `T_LOG` bits, which are the types themselves when the tail is zero.
-/

namespace FlockVBridge.Sparse

open Flock Flock.RecAlgebra FlockSoundness FlockSoundness.Refine FlockSoundness.Model FlockSoundness.Game

local notation "𝔽" => FlockLevel3.GF128

noncomputable section

/-! ## Folds, sums and characteristic 2 -/

theorem fOf_foldl {β : Type} (g : F128 → β → F128) (g' : 𝔽 → β → 𝔽) :
    ∀ (l : List β), (∀ x, ∀ b ∈ l, fOf (g x b) = g' (fOf x) b) → ∀ x, fOf (l.foldl g x) = l.foldl g' (fOf x)
  | [], _, _ => rfl
  | b :: l, hg, x => by
    simp only [List.foldl_cons]
    rw [← hg x b List.mem_cons_self]
    exact fOf_foldl g g' l (fun y a ha => hg y a (List.mem_cons_of_mem _ ha)) _

theorem foldl_mul_prod {M : Type*} [CommMonoid M] (f : ℕ → M) (x : M) :
    ∀ n, (List.range n).foldl (fun y j => y * f j) x = x * ∏ j ∈ Finset.range n, f j
  | 0 => by simp
  | n + 1 => by
    rw [List.range_succ, List.foldl_append, foldl_mul_prod f x n, Finset.prod_range_succ]
    simp only [List.foldl_cons, List.foldl_nil, mul_assoc]

theorem fOf_eq_zero {a : F128} : fOf a = 0 ↔ a = 0 :=
  ⟨fun h => fOf_injective (h.trans fOf_zero.symm), fun h => by rw [h, fOf_zero]⟩

theorem char2_add_eq_zero {a b : 𝔽} : a + b = 0 ↔ a = b := by
  rw [add_eq_zero_iff_eq_neg, CharTwo.neg_eq]

theorem mod_tail_zero {a T : ℕ} (hT : 0 < T) : a - a % T = 0 ↔ a < T := by
  constructor
  · intro h
    have h1 : a % T = a := by have := Nat.mod_le a T; omega
    calc a = a % T := h1.symm
      _ < T := Nat.mod_lt a hT
  · intro h
    rw [Nat.mod_eq_of_lt h, Nat.sub_self]

/-- A round's claim, with the round's coefficients against `r^{i+2} + r` (the `sc<j>` and `pre<j>` residuals' form). -/
theorem degNext_eq {k : ℕ} (claim c0 r : 𝔽) (cs : Fin k → 𝔽) :
    degNext claim (c0, cs) r = c0 + ∑ i : Fin k, cs i * (r ^ (i.val + 2) + r) + claim * r := by
  have e1 : ∑ i : Fin k, cs i * (r ^ (i.val + 2) + r) = ∑ i : Fin k, (cs i * r ^ (i.val + 2) + cs i * r) :=
    Finset.sum_congr rfl fun i _ => mul_add _ _ _
  have e2 : ∑ i : Fin k, (cs i * r ^ (i.val + 2) + cs i * r) =
      ∑ i : Fin k, cs i * r ^ (i.val + 2) + ∑ i : Fin k, cs i * r := Finset.sum_add_distrib
  have e3 : ∑ i : Fin k, cs i * r = (∑ i : Fin k, cs i) * r := (Finset.sum_mul _ _ _).symm
  simp only [degNext]
  rw [e1, e2, e3]
  ring

theorem degNext_one (claim c0 c2 r : 𝔽) :
    degNext claim (c0, fun _ : Fin 1 => c2) r = c0 + c2 * (r * r + r) + claim * r := by
  simp only [degNext, Fin.sum_univ_one, Fin.val_eq_zero]
  ring

/-! ## The output, part by part -/

variable (sh : SparseShape)

/-- H2's claim after its six rounds, on the residuals `rs`. -/
def c2C (rs v : Array F128) : F128 := (List.range 6).foldl (fun c j => rget sh rs (.pre j) + c * v[vPre j]!) (rget sh rs .h1)

/-- H3's claim after its rounds. -/
def c4C (rs w v : Array F128) : F128 :=
  (List.range (nR sh)).foldl (fun c j => rget sh rs (.sc j) + c * v[vSc sh j]!) w[wV2]!

/-- Φ's product. -/
def phiC (rs : Array F128) : F128 :=
  (List.range sh.ell).foldl (fun x j => x * (rget sh rs (.fx j) * rget sh rs (.fy j))) (rget sh rs .f0)

/-- The type-selected weight `w_t`, on the types' low `T_LOG` bits. -/
def wtC (v : Array F128) (τ : Array ℕ) (t : ℕ) : F128 :=
  (List.range (2 ^ sh.sLog)).foldl (fun x s => x + if τ[s]! % 2 ^ sh.tLog = t then v[vEqs sh + s]! else 0) 0

/-- H4's residual with `T > 1`. -/
def h4C (rs w v : Array F128) (τ : Array ℕ) : F128 :=
  (List.range (2 ^ sh.tLog)).foldl (fun x t => x + wtC sh v τ t * w[wVt sh + t]!) (rget sh rs .bp)

/-- The tail: each slot's type with its low `T_LOG` bits cleared. -/
def tailC (τ : Array ℕ) : Array ℕ := ((List.range (2 ^ sh.sLog)).map fun s => τ[s]! - τ[s]! % 2 ^ sh.tLog).toArray

theorem outOf_eq (rs w v : Array F128) (τ : Array ℕ) : outOf sh rs w v τ =
    if sh.tLog = 0 then (#[c2C sh rs v + rget sh rs .wv, c4C sh rs w v + phiC sh rs * v[vUr sh]!, rget sh rs .bp], #[])
    else (#[c2C sh rs v + rget sh rs .wv, c4C sh rs w v + phiC sh rs * v[vUr sh]!, h4C sh rs w v τ, rget sh rs .claim],
      tailC sh τ) := rfl

theorem vrow_size (c : SCoins) : (verifierRow sh c).size = nv sh := Array.size_ofFn

theorem out_eq (w : Array F128) (τ : Array ℕ) (c : SCoins) (hw : w.size = nw sh) :
    out sh w (verifierRow sh c) τ = .ok (outOf sh (resArr sh w (verifierRow sh c)) w (verifierRow sh c) τ) := by
  unfold out
  rw [residuals_sparseS sh w _ hw (vrow_size sh c)]
  rfl

theorem ell_lt : 2 * sh.ell + 1 < 2 ^ sh.pLog := by
  unfold SparseShape.pLog
  exact Nat.lt_log2_self

theorem nc_le {c : SCoins} (hc : nc sh ≤ c.cs.size) : 12 + nR sh + sh.pLog + sh.tLog ≤ c.cs.size := hc

variable (w : Array F128) (c : SCoins)

theorem xLo_cat6 {j : ℕ} (hj : j < sh.ell) : fOf (xLo c j) =
    cat6 (fun i : Fin 6 => fOf c.cs[6 + i.val]!) (fun i : Fin (sh.ell - 6) => fOf c.xRest[i.val]!) j := by
  unfold xLo cat6 vget
  split_ifs <;> first | rfl | omega

theorem yLo_cat6 {j : ℕ} (hj : j < sh.ell) : fOf (yLo c j) =
    cat6 (fun i : Fin 6 => fOf c.cs[0 + i.val]!) (fun i : Fin (sh.ell - 6) => fOf c.yRest[i.val]!) j := by
  unfold yLo cat6 vget
  split_ifs <;> first | rfl | simp only [Nat.zero_add] | omega

theorem vget_uOf {i : ℕ} (hi : i < 2 ^ sh.pLog) : vget (uOf sh w) i = fOf w[wU sh + i]! := by
  unfold vget uOf
  rw [dite_eq_left hi]

/-! ### H1 and H2 -/

theorem h1_val (hc : 6 ≤ c.cs.size) : fOf (rget sh (resArr sh w (verifierRow sh c)) .h1) = h1Claim w c.cs := by
  rw [rget_eq sh _ _ (mem_h1 sh), specVal_h1, Finset.sum_range, h1Claim]
  exact Finset.sum_congr rfl fun s _ => by rw [v_eqc sh c hc s.isLt]

theorem pre_step {j : ℕ} (hj : j < 6) (x : F128) :
    fOf (rget sh (resArr sh w (verifierRow sh c)) (.pre j) + x * (verifierRow sh c)[vPre j]!) =
      degNext (fOf x) (msg2 w j) (fOf c.cs[6 + j]!) := by
  rw [fOf_add, fOf_mul, rget_eq sh _ _ (mem_pre sh hj), specVal_pre, v_pre1 sh c hj, v_pre0 sh c hj, msg2,
    degNext_one]

theorem h2_chain (hc : 6 ≤ c.cs.size) :
    fOf (c2C sh (resArr sh w (verifierRow sh c)) (verifierRow sh c)) =
      degFold (msg2 w) (fun j => fOf c.cs[6 + j]!) 0 6 (h1Claim w c.cs) := by
  have hs : ∀ x, ∀ j ∈ List.range 6, fOf (rget sh (resArr sh w (verifierRow sh c)) (.pre j) +
      x * (verifierRow sh c)[vPre j]!) = degNext (fOf x) (msg2 w j) (fOf c.cs[6 + j]!) :=
    fun x j hj => pre_step sh w c (List.mem_range.1 hj) x
  unfold c2C
  rw [fOf_foldl _ (fun y j => degNext y (msg2 w j) (fOf c.cs[6 + j]!)) _ hs, h1_val sh w c hc, degFold,
    List.range_eq_range']

/-- **H2's residual**: `c₂ + ŵ(ζ_r)·v₂`. -/
theorem h2_val (hc : nc sh ≤ c.cs.size) :
    fOf (c2C sh (resArr sh w (verifierRow sh c)) (verifierRow sh c) + rget sh (resArr sh w (verifierRow sh c)) .wv) =
      degFold (msg2 w) (fun j => fOf c.cs[6 + j]!) 0 6 (h1Claim w c.cs) +
        (∑ s : Fin 64, eqAt (fun i : Fin 6 => fOf c.cs[6 + i.val]!) s.val * execArith.lagS (fOf c.z) s) *
          fOf w[wV2]! := by
  have h' := nc_le sh hc
  rw [fOf_add, h2_chain sh w c (by omega), rget_eq sh _ _ (mem_wv sh), specVal_wv, v_what sh c (by omega)]
  rfl

/-! ### H3 -/

theorem sc_step {j : ℕ} (hj : j < nR sh) {k : ℕ} (hk : k = deg sh j - 1) (x : F128) :
    fOf (rget sh (resArr sh w (verifierRow sh c)) (.sc j) + x * (verifierRow sh c)[vSc sh j]!) =
      degNext (fOf x) (msg3 sh w k j) (fOf c.cs[12 + j]!) := by
  have hd : 0 < deg sh j := by unfold deg; split_ifs <;> omega
  have h0 : fOf (verifierRow sh c)[vSc sh j]! = fOf c.cs[12 + j]! := by
    have h := v_sc sh c hj hd
    rwa [Nat.add_zero, ite_eq_left rfl] at h
  have hs : ∑ i ∈ Finset.range (deg sh j - 1),
      fOf w[wSc sh j + (i + 1)]! * fOf (verifierRow sh c)[vSc sh j + (i + 1)]! =
        ∑ i : Fin k, fOf w[wSc sh j + (i.val + 1)]! * (fOf c.cs[12 + j]! ^ (i.val + 2) + fOf c.cs[12 + j]!) := by
    rw [← hk, Finset.sum_range]
    refine Finset.sum_congr rfl fun i _ => ?_
    rw [v_sc sh c hj (by have := i.isLt; omega), ite_eq_right (by omega), show i.val + 1 + 1 = i.val + 2 from rfl]
  rw [fOf_add, fOf_mul, rget_eq sh _ _ (mem_sc sh hj), specVal_sc, h0, hs]
  exact (degNext_eq (fOf x) (fOf w[wSc sh j]!) (fOf c.cs[12 + j]!) fun i : Fin k => fOf w[wSc sh j + (i.val + 1)]!).symm

theorem range_nR : List.range (nR sh) = List.range' 0 sh.nu ++ List.range' sh.nu sh.sLog := by
  rw [List.range_eq_range', nR, ← List.range'_append_1, Nat.zero_add]

theorem h3_chain : fOf (c4C sh (resArr sh w (verifierRow sh c)) w (verifierRow sh c)) =
    degFold (msg3 sh w (2 * sh.ell + 1)) (fun j => fOf c.cs[12 + j]!) sh.nu sh.sLog (c3Of sh w c.cs) := by
  have s1 : ∀ x, ∀ j ∈ List.range' 0 sh.nu, fOf (rget sh (resArr sh w (verifierRow sh c)) (.sc j) +
      x * (verifierRow sh c)[vSc sh j]!) = degNext (fOf x) (msg3 sh w (2 * sh.ell) j) (fOf c.cs[12 + j]!) := by
    intro x j hj
    rw [List.mem_range'_1] at hj
    exact sc_step sh w c (by unfold nR; omega) (by unfold deg; split_ifs <;> omega) x
  have s2 : ∀ x, ∀ j ∈ List.range' sh.nu sh.sLog, fOf (rget sh (resArr sh w (verifierRow sh c)) (.sc j) +
      x * (verifierRow sh c)[vSc sh j]!) = degNext (fOf x) (msg3 sh w (2 * sh.ell + 1) j) (fOf c.cs[12 + j]!) := by
    intro x j hj
    rw [List.mem_range'_1] at hj
    exact sc_step sh w c (by unfold nR; omega) (by unfold deg; split_ifs <;> omega) x
  unfold c4C
  rw [range_nR, List.foldl_append,
    fOf_foldl _ (fun y j => degNext y (msg3 sh w (2 * sh.ell + 1) j) (fOf c.cs[12 + j]!)) _ s2,
    fOf_foldl _ (fun y j => degNext y (msg3 sh w (2 * sh.ell) j) (fOf c.cs[12 + j]!)) _ s1, degFold, c3Of, degFold]

theorem phi_val : fOf (phiC sh (resArr sh w (verifierRow sh c))) =
    sparsePhi sh.ell (fOf c.alpha)
      (cat6 (fun i : Fin 6 => fOf c.cs[6 + i.val]!) (fun i : Fin (sh.ell - 6) => fOf c.xRest[i.val]!))
      (cat6 (fun i : Fin 6 => fOf c.cs[0 + i.val]!) (fun i : Fin (sh.ell - 6) => fOf c.yRest[i.val]!))
      (vget (uOf sh w)) := by
  have hP := ell_lt sh
  have s : ∀ x, ∀ j ∈ List.range sh.ell, fOf (x * (rget sh (resArr sh w (verifierRow sh c)) (.fx j) *
      rget sh (resArr sh w (verifierRow sh c)) (.fy j))) = fOf x *
      ((1 + cat6 (fun i : Fin 6 => fOf c.cs[6 + i.val]!) (fun i : Fin (sh.ell - 6) => fOf c.xRest[i.val]!) j +
          vget (uOf sh w) j) *
        (1 + cat6 (fun i : Fin 6 => fOf c.cs[0 + i.val]!) (fun i : Fin (sh.ell - 6) => fOf c.yRest[i.val]!) j +
          vget (uOf sh w) (sh.ell + j))) := by
    intro x j hj
    rw [List.mem_range] at hj
    rw [fOf_mul, fOf_mul, rget_eq sh _ _ (mem_fx sh hj), rget_eq sh _ _ (mem_fy sh hj), specVal_fx, specVal_fy,
      v_ax sh c hj, v_ay sh c hj, xLo_cat6 sh c hj, yLo_cat6 sh c hj, vget_uOf sh w (by omega),
      vget_uOf sh w (by omega)]
    ring
  unfold phiC
  rw [fOf_foldl _ (fun y j => y *
      ((1 + cat6 (fun i : Fin 6 => fOf c.cs[6 + i.val]!) (fun i : Fin (sh.ell - 6) => fOf c.xRest[i.val]!) j +
          vget (uOf sh w) j) *
        (1 + cat6 (fun i : Fin 6 => fOf c.cs[0 + i.val]!) (fun i : Fin (sh.ell - 6) => fOf c.yRest[i.val]!) j +
          vget (uOf sh w) (sh.ell + j)))) _ s, foldl_mul_prod, rget_eq sh _ _ (mem_f0 sh), specVal_f0, v_alpha, sparsePhi,
    vget_uOf sh w (show 2 * sh.ell < 2 ^ sh.pLog by omega), vget_uOf sh w (show 2 * sh.ell + 1 < 2 ^ sh.pLog by omega)]
  ring

/-- **H3's residual**: `c₄ + Ũ(r_s)·Φ(u)`. -/
theorem h3_val (hc : nc sh ≤ c.cs.size) :
    fOf (c4C sh (resArr sh w (verifierRow sh c)) w (verifierRow sh c) +
        phiC sh (resArr sh w (verifierRow sh c)) * (verifierRow sh c)[vUr sh]!) =
      degFold (msg3 sh w (2 * sh.ell + 1)) (fun j => fOf c.cs[12 + j]!) sh.nu sh.sLog (c3Of sh w c.cs) +
        (∑ σ : Fin (2 ^ sh.sLog), eqAt (fun i : Fin sh.sLog => fOf c.cs[12 + sh.nu + i.val]!) σ.val *
          fOf c.U[σ.val]!) *
        sparsePhi sh.ell (fOf c.alpha)
          (cat6 (fun i : Fin 6 => fOf c.cs[6 + i.val]!) (fun i : Fin (sh.ell - 6) => fOf c.xRest[i.val]!))
          (cat6 (fun i : Fin 6 => fOf c.cs[0 + i.val]!) (fun i : Fin (sh.ell - 6) => fOf c.yRest[i.val]!))
          (vget (uOf sh w)) := by
  have h' := nc_le sh hc
  rw [fOf_add, fOf_mul, h3_chain sh w c, phi_val sh w c, v_ur sh c (by omega)]
  ring

/-! ### H4 and the claim -/

theorem batch_val (hc : nc sh ≤ c.cs.size) :
    ∑ p ∈ Finset.range (2 ^ sh.pLog), fOf (verifierRow sh c)[vEqp sh + p]! * fOf w[wU sh + p]! =
      ∑ q : Fin (2 ^ sh.pLog), eqAt (fun i : Fin sh.pLog => fOf c.cs[12 + nR sh + i.val]!) q.val * uOf sh w q := by
  have h' := nc_le sh hc
  rw [Finset.sum_range]
  exact Finset.sum_congr rfl fun q _ => by rw [v_eqp sh c (by omega) q.isLt]; rfl

/-- **The plane batch's residual with `T = 1`**: the claim against `Σ_p eq(ζ_p, p)·u_p`. -/
theorem bp_val (hc : nc sh ≤ c.cs.size) (ht : sh.tLog = 0) :
    fOf (rget sh (resArr sh w (verifierRow sh c)) .bp) = fOf w[wClaim sh]! +
      ∑ q : Fin (2 ^ sh.pLog), eqAt (fun i : Fin sh.pLog => fOf c.cs[12 + nR sh + i.val]!) q.val * uOf sh w q := by
  rw [rget_eq sh _ _ (mem_bp sh), specVal_bp, ite_eq_left ht, batch_val sh w c hc]

theorem wt_val (τ : Array ℕ) (hc : nc sh ≤ c.cs.size) (ht : sh.tLog ≠ 0)
    (hτ : ∀ s < 2 ^ sh.sLog, τ[s]! < 2 ^ sh.tLog) (t : ℕ) :
    fOf (wtC sh (verifierRow sh c) τ t) = ∑ σ : Fin (2 ^ sh.sLog),
      if τ[σ.val]! = t then eqAt (fun i : Fin sh.sLog => fOf c.cs[12 + sh.nu + i.val]!) σ.val else 0 := by
  have h' := nc_le sh hc
  unfold wtC
  rw [fOf_foldl_sum, fOf_zero, zero_add, sum_range_list, Finset.sum_range]
  refine Finset.sum_congr rfl fun σ _ => ?_
  rw [Nat.mod_eq_of_lt (hτ σ σ.isLt)]
  split_ifs
  · exact v_eqs sh c ht (by omega) σ.isLt
  · exact fOf_zero

/-- **H4's residual with `T > 1`**: the batch against the type-selected sum `Σ_t w_t·vt_t`, when every type is
below `T`. -/
theorem h4_val (τ : Array ℕ) (hc : nc sh ≤ c.cs.size) (ht : sh.tLog ≠ 0)
    (hτ : ∀ s < 2 ^ sh.sLog, τ[s]! < 2 ^ sh.tLog) :
    fOf (h4C sh (resArr sh w (verifierRow sh c)) w (verifierRow sh c) τ) =
      ∑ q : Fin (2 ^ sh.pLog), eqAt (fun i : Fin sh.pLog => fOf c.cs[12 + nR sh + i.val]!) q.val * uOf sh w q +
        ∑ t : Fin (2 ^ sh.tLog), (∑ σ : Fin (2 ^ sh.sLog),
          if τ[σ.val]! = t.val then eqAt (fun i : Fin sh.sLog => fOf c.cs[12 + sh.nu + i.val]!) σ.val else 0) *
          vtOf sh w t := by
  unfold h4C
  rw [fOf_foldl_sum, sum_range_list, rget_eq sh _ _ (mem_bp sh), specVal_bp, ite_eq_right ht, zero_add,
    batch_val sh w c hc, Finset.sum_range]
  congr 1
  refine Finset.sum_congr rfl fun t _ => ?_
  rw [fOf_mul, wt_val sh c τ hc ht hτ, vtOf, ite_eq_right ht]

/-- **The claim's residual with `T > 1`**: the claim against `Σ_t eq(ρ_t, t)·vt_t`. -/
theorem claim_val (hc : nc sh ≤ c.cs.size) (ht : sh.tLog ≠ 0) :
    fOf (rget sh (resArr sh w (verifierRow sh c)) .claim) = fOf w[wClaim sh]! +
      ∑ t : Fin (2 ^ sh.tLog), eqAt (fun i : Fin sh.tLog => fOf c.cs[12 + nR sh + sh.pLog + i.val]!) t.val *
        vtOf sh w t := by
  rw [rget_eq sh _ _ (mem_claim sh ht), specVal_claim, Finset.sum_range]
  refine congrArg (fOf w[wClaim sh]! + ·) (Finset.sum_congr rfl fun t _ => ?_)
  rw [v_eqt sh c ht hc t.isLt, vtOf, ite_eq_right ht]

/-! ## The model's verdict -/

/-- `sparseVerdict` ends at the registered claim `⟨val, r_k, ζ_p, ρ_t⟩` exactly when its checks pass and `val` is
the claim it computes. -/
theorem sparseVerdict_some_iff {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] (A : Arith F K)
    (pub : SparsePub F sh) (τ : Fin (2 ^ sh.sLog) → ℕ) (ζc ζr : Fin 6 → F) (c₂ v₂ : F) (rk : Fin sh.nu → F)
    (rs : Fin sh.sLog → F) (c₄ : F) (u : Fin (2 ^ sh.pLog) → F) (ζp : Fin sh.pLog → F)
    (vt : Fin (2 ^ sh.tLog) → F) (ρ : Fin sh.tLog → F) (val : F) :
    sparseVerdict A sh pub τ ζc ζr c₂ v₂ rk rs c₄ u ζp vt ρ = some ⟨val, rk, ζp, ρ⟩ ↔
      c₂ = (∑ s : Fin 64, eqAt ζr s.val * A.lagS pub.z s) * v₂ ∧
      c₄ = (∑ σ : Fin (2 ^ sh.sLog), eqAt rs σ.val * pub.U σ) *
        sparsePhi sh.ell pub.alpha (cat6 ζr pub.xRest) (cat6 ζc pub.yRest) (vget u) ∧
      (if sh.tLog = 0 then val = ∑ q : Fin (2 ^ sh.pLog), eqAt ζp q.val * u q
      else ((∑ q : Fin (2 ^ sh.pLog), eqAt ζp q.val * u q) =
          ∑ t : Fin (2 ^ sh.tLog), (∑ σ : Fin (2 ^ sh.sLog), if τ σ = t.val then eqAt rs σ.val else 0) * vt t) ∧
        val = ∑ t : Fin (2 ^ sh.tLog), eqAt ρ t.val * vt t) := by
  simp only [sparseVerdict]
  by_cases ht : sh.tLog = 0
  · rw [ite_eq_left ht, ite_eq_left ht]
    split
    · rename_i h
      simp only [Option.some.injEq, RegClaim.mk.injEq, and_true]
      exact ⟨fun e => ⟨h.1, h.2, e.symm⟩, fun e => e.2.2.symm⟩
    · rename_i h
      exact ⟨(fun e => nomatch e), fun e => absurd ⟨e.1, e.2.1⟩ h⟩
  · rw [ite_eq_right ht, ite_eq_right ht]
    split
    · rename_i h
      simp only [Option.some.injEq, RegClaim.mk.injEq, and_true]
      exact ⟨fun e => ⟨h.1, h.2.1, h.2.2, e.symm⟩, fun e => e.2.2.2.symm⟩
    · rename_i h
      exact ⟨(fun e => nomatch e), fun e => absurd ⟨e.1, e.2.1, e.2.2.1⟩ h⟩

/-! ## The refinement -/

/-- `RecSparse_v1` accepts: every residual and every tail entry is zero. -/
def SparseAccepts (o : Array F128 × Array ℕ) : Prop := (∀ x ∈ o.1, x = 0) ∧ ∀ y ∈ o.2, y = 0

/-- The slots' types are below `T = 2^T_LOG` (with `T = 1` the circuit has no tail, and nothing is asked). -/
def TypesIn (τ : Array ℕ) : Prop := sh.tLog ≠ 0 → ∀ σ < 2 ^ sh.sLog, τ[σ]! < 2 ^ sh.tLog

/-- The registered claim at the `claim` port's value, at `(r_k, ζ_p, ρ_t)`. -/
def sparseClaim : RegClaim 𝔽 sh :=
  ⟨fOf w[wClaim sh]!, fun i => fOf c.cs[12 + i.val]!, fun i => fOf c.cs[12 + nR sh + i.val]!,
    fun i => fOf c.cs[12 + nR sh + sh.pLog + i.val]!⟩

theorem accepts3 (a b d : F128) : SparseAccepts (#[a, b, d], #[]) ↔ fOf a = 0 ∧ fOf b = 0 ∧ fOf d = 0 := by
  simp [SparseAccepts, fOf_eq_zero]

theorem accepts4 (a b d e : F128) (tl : Array ℕ) :
    SparseAccepts (#[a, b, d, e], tl) ↔ (fOf a = 0 ∧ fOf b = 0 ∧ fOf d = 0 ∧ fOf e = 0) ∧ ∀ y ∈ tl, y = 0 := by
  simp [SparseAccepts, fOf_eq_zero]

theorem tail_iff (τ : Array ℕ) : (∀ y ∈ tailC sh τ, y = 0) ↔ ∀ s < 2 ^ sh.sLog, τ[s]! < 2 ^ sh.tLog := by
  have hT : 0 < 2 ^ sh.tLog := Nat.two_pow_pos _
  unfold tailC
  constructor
  · intro h s hs
    exact (mod_tail_zero hT).1 (h _ (List.mem_toArray.2 (List.mem_map.2 ⟨s, List.mem_range.2 hs, rfl⟩)))
  · intro h y hy
    obtain ⟨s, hs, rfl⟩ := List.mem_map.1 (List.mem_toArray.1 hy)
    exact (mod_tail_zero hT).2 (h s (List.mem_range.1 hs))

theorem typesIn_iff (τ : Array ℕ) (ht : sh.tLog ≠ 0) :
    TypesIn sh τ ↔ ∀ σ < 2 ^ sh.sLog, τ[σ]! < 2 ^ sh.tLog := by
  unfold TypesIn
  exact ⟨fun h => h ht, fun h _ => h⟩

theorem typesIn_of_zero (τ : Array ℕ) (ht : sh.tLog = 0) : TypesIn sh τ := by
  unfold TypesIn
  exact fun h => absurd ht h

theorem accepts_iff (τ : Array ℕ) (hc : nc sh ≤ c.cs.size) :
    SparseAccepts (outOf sh (resArr sh w (verifierRow sh c)) w (verifierRow sh c) τ) ↔
      TypesIn sh τ ∧ verdictOf sh w τ c = some (sparseClaim sh w c) := by
  rw [verdictOf, sparseClaim, sparseVerdict_some_iff, outOf_eq]
  by_cases ht : sh.tLog = 0
  · rw [ite_eq_left ht, ite_eq_left ht, accepts3, h2_val sh w c hc, h3_val sh w c hc, bp_val sh w c hc ht,
      char2_add_eq_zero, char2_add_eq_zero, char2_add_eq_zero]
    exact ⟨fun h => ⟨typesIn_of_zero sh τ ht, h⟩, fun h => h.2⟩
  · rw [ite_eq_right ht, ite_eq_right ht, accepts4, tail_iff, typesIn_iff sh τ ht]
    constructor
    · rintro ⟨⟨h2, h3, h4, hcl⟩, hτ⟩
      rw [h2_val sh w c hc, char2_add_eq_zero] at h2
      rw [h3_val sh w c hc, char2_add_eq_zero] at h3
      rw [h4_val sh w c τ hc ht hτ, char2_add_eq_zero] at h4
      rw [claim_val sh w c hc ht, char2_add_eq_zero] at hcl
      exact ⟨hτ, h2, h3, h4, hcl⟩
    · rintro ⟨hτ, h2, h3, h4, hcl⟩
      refine ⟨⟨?_, ?_, ?_, ?_⟩, hτ⟩
      · rw [h2_val sh w c hc, char2_add_eq_zero]; exact h2
      · rw [h3_val sh w c hc, char2_add_eq_zero]; exact h3
      · rw [h4_val sh w c τ hc ht hτ, char2_add_eq_zero]; exact h4
      · rw [claim_val sh w c hc ht, char2_add_eq_zero]; exact hcl

/-- **`RecSparse_v1` refines the sparse phase.** On w of the message ports' size and at least the phase's coins,
`RecSparse_v1`'s output over F at the verifier's row is defined; the model's `sparsePhase`, fed w's messages and the
coins one word each, ends at `verdictOf`; and the output is all zero exactly when every slot's type is below `T` and
that run ends at the registered claim with the `claim` port's value. -/
theorem sparse_refines (τ : Array ℕ) (hw : w.size = nw sh) (hc : nc sh ≤ c.cs.size) :
    ∃ o, out sh w (verifierRow sh c) τ = .ok o ∧
      Plays fOf (sparsePhase execArith sh (pubOf sh c) (fun σ => τ[σ.val]!) (fun s => fOf w[s.val]!))
        (phaseMsgs sh w) (coinTape c.cs 0 (nc sh)) (verdictOf sh w τ c) ∧
      (SparseAccepts o ↔ TypesIn sh τ ∧ verdictOf sh w τ c = some (sparseClaim sh w c)) := by
  refine ⟨_, out_eq sh w τ c hw, ?_, accepts_iff sh w c τ hc⟩
  rw [← phaseCoins_eq]
  exact sparsePhase_plays sh w τ c

end

end FlockVBridge.Sparse
