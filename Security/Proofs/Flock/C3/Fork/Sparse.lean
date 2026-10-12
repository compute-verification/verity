import Proofs.Flock.Recursive.HoloExec
import Proofs.Flock.C3.Fork.SparseRounds

/-!
# Piece (a), the sparse count: one rep's sparse phase is sound

`Model.sparsePhase` (H1–H4) checks a rep's hidden fold `h` against a registered table `z` at the rep's public values
`pub` and the slots' types `τ`, and hands on a registered claim on `z`. `sparse_sound`: where `h` is not `z`'s fold
(`sparseFold`), the phase hands on a claim that holds on `z` with probability at most `sparseError`, each coin its degree
over `qF` (Schwartz–Zippel): H1's 6, H2's 6 rounds of degree 2, H3's `ν` rounds of degree `2ℓ + 1` and `sLog` of degree
`2ℓ + 2`, H4's `pLog` and the type point's `tLog`.

`sparseFold` is the phase's own algebra: row `s` of the skip, `Σ_r lagS(z_skip, r) · Σ_σ U_σ · Σ_k Φ(x(r), y(s), q(τ_σ, k))`
with `q(t, k)` the planes of type `t` at entry `k` (`planeAt`, the bit `k + 2^ν·(p + 2^pLog·t)` of `z`). That it is the
hidden part of the lincheck's fold at `z = R.bits`, for a table `R` whose unit slots are in range, is the assembly's
(`holo_runs_sound`, `C3/Fork/Runs.lean`), not this lemma's.
-/

namespace FlockC3.Fork

open FlockSoundness FlockSoundness.Discharge.Recursion
open Game Model Accounting
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]

/-- The skip's row `r` as a point of `{0, 1}^6`. -/
def skipBits (r : Fin 64) : Fin 6 → F := fun j => if r.val.testBit j.val then 1 else 0

/-- Plane `p` of type `t` at entry `k` of the registered table `z`: bit `k + 2^ν·(p + 2^pLog·t)`, zero past the table. -/
def planeAt (sh : SparseShape) (z : Witness sh.mReg) (t p k : ℕ) : F :=
  if h : k + 2 ^ sh.nu * (p + 2 ^ sh.pLog * t) < 2 ^ sh.mReg then ((z ⟨_, h⟩).val : F) else 0

/-- **The hidden fold the sparse phase checks** at the registered table `z`: row `s` of the skip. -/
noncomputable def sparseFold (A : Arith F K) (sh : SparseShape) (pub : SparsePub F sh) (τ : Fin (2 ^ sh.sLog) → ℕ)
    (z : Witness sh.mReg) (s : Fin 64) : F :=
  ∑ r : Fin 64, A.lagS pub.z r * ∑ σ : Fin (2 ^ sh.sLog), pub.U σ * ∑ k : Fin (2 ^ sh.nu),
    sparsePhi sh.ell pub.alpha (cat6 (skipBits r) pub.xRest) (cat6 (skipBits s) pub.yRest)
      (fun p => planeAt sh z (τ σ) p k.val)

/-! ## The count's polynomials

Each sum-check of `sparsePhase` checks a sum over a cube of the polynomial below, the check the next one hands on:
H2's rounds `g2` (the skip's rows), H3's `g3` (the entries) and `g4` (the slots); H4's are the planes' and the types'
values (`slotPlanes`, `typeVal`), each a multilinear extension. -/

namespace SparseCount

omit [Fintype F] [DecidableEq F] [Fintype K] [DecidableEq K]

/-- The planes of type `t` at a point `κ` of the entries. -/
def planeExt (sh : SparseShape) (z : Witness sh.mReg) (t : ℕ) (κ : Fin sh.nu → F) (p : ℕ) : F :=
  ∑ k : Fin (2 ^ sh.nu), eqAt κ k.val * planeAt sh z t p k.val

/-- The planes at the entries' point `rk` and the slots' point `ς`. -/
def slotPlanes (sh : SparseShape) (τ : Fin (2 ^ sh.sLog) → ℕ) (z : Witness sh.mReg) (rk : Fin sh.nu → F)
    (ς : Fin sh.sLog → F) (p : ℕ) : F :=
  ∑ σ : Fin (2 ^ sh.sLog), eqAt ς σ.val * planeExt sh z (τ σ) rk p

/-- Type `t`'s value at the entries' point `rk` and the planes' point `ζp`. -/
def typeVal (sh : SparseShape) (z : Witness sh.mReg) (rk : Fin sh.nu → F) (ζp : Fin sh.pLog → F) (t : ℕ) : F :=
  ∑ p : Fin (2 ^ sh.pLog), eqAt ζp p.val * planeExt sh z t rk p.val

/-- The weight of type `t` at the slots' point `ς` (the verdict's `wt`). -/
def slotWeight (sh : SparseShape) (τ : Fin (2 ^ sh.sLog) → ℕ) (ς : Fin sh.sLog → F) (t : ℕ) : F :=
  ∑ σ : Fin (2 ^ sh.sLog), if τ σ = t then eqAt ς σ.val else 0

/-- `Φ` at the skip's rows `x` and columns `y`. -/
def phiAt (sh : SparseShape) (pub : SparsePub F sh) (y x : Fin 6 → F) (q : ℕ → F) : F :=
  sparsePhi sh.ell pub.alpha (cat6 x pub.xRest) (cat6 y pub.yRest) q

/-- The fold at rows `x` and columns `y`, before the skip's weights. -/
def psi (sh : SparseShape) (pub : SparsePub F sh) (τ : Fin (2 ^ sh.sLog) → ℕ) (z : Witness sh.mReg)
    (x y : Fin 6 → F) : F :=
  ∑ σ : Fin (2 ^ sh.sLog), pub.U σ * ∑ k : Fin (2 ^ sh.nu), phiAt sh pub y x fun p => planeAt sh z (τ σ) p k.val

/-- The skip's weights at rows `x` (the verdict's `what`). -/
def lagAt (A : Arith F K) {sh : SparseShape} (pub : SparsePub F sh) (x : Fin 6 → F) : F :=
  ∑ s : Fin 64, eqAt x s.val * A.lagS pub.z s

/-- The slots' weights at `ς` (the verdict's `ur`). -/
def uAt {sh : SparseShape} (pub : SparsePub F sh) (ς : Fin sh.sLog → F) : F :=
  ∑ σ : Fin (2 ^ sh.sLog), eqAt ς σ.val * pub.U σ

/-- H2's polynomial, in the skip's rows. -/
def g2 (A : Arith F K) (sh : SparseShape) (pub : SparsePub F sh) (τ : Fin (2 ^ sh.sLog) → ℕ) (z : Witness sh.mReg)
    (ζc x : Fin 6 → F) : F :=
  lagAt A pub x * psi sh pub τ z x ζc

/-- H3's polynomial in the entries. -/
def g3 (sh : SparseShape) (pub : SparsePub F sh) (τ : Fin (2 ^ sh.sLog) → ℕ) (z : Witness sh.mReg)
    (ζc ζr : Fin 6 → F) (κ : Fin sh.nu → F) : F :=
  ∑ σ : Fin (2 ^ sh.sLog), pub.U σ * phiAt sh pub ζc ζr (planeExt sh z (τ σ) κ)

/-- H3's polynomial in the slots. -/
def g4 (sh : SparseShape) (pub : SparsePub F sh) (τ : Fin (2 ^ sh.sLog) → ℕ) (z : Witness sh.mReg)
    (ζc ζr : Fin 6 → F) (rk : Fin sh.nu → F) (ς : Fin sh.sLog → F) : F :=
  uAt pub ς * phiAt sh pub ζc ζr (slotPlanes sh τ z rk ς)

/-! ### Degrees -/

section Degrees

variable (sh : SparseShape) (pub : SparsePub F sh) (τ : Fin (2 ^ sh.sLog) → ℕ) (z : Witness sh.mReg)

theorem planeExt_polyLe (t : ℕ) (v : Fin sh.nu → F) (j : Fin sh.nu) (p : ℕ) :
    PolyLe 1 fun x => planeExt sh z t (Function.update v j x) p :=
  PolyLe.sum _ fun k _ => (polyLe_eqAt v j k.val).mul_const _

theorem slotPlanes_polyLe (rk : Fin sh.nu → F) (v : Fin sh.sLog → F) (j : Fin sh.sLog) (p : ℕ) :
    PolyLe 1 fun x => slotPlanes sh τ z rk (Function.update v j x) p :=
  PolyLe.sum _ fun σ _ => (polyLe_eqAt v j σ.val).mul_const _

omit z in
theorem phiAt_polyLe (ζc ζr : Fin 6 → F) (q : F → ℕ → F) (hq : ∀ p, PolyLe 1 fun t => q t p) :
    PolyLe (2 * sh.ell + 1) fun t => phiAt sh pub ζc ζr (q t) := by
  refine (polyLe_sparsePhi sh.ell pub.alpha (fun _ => cat6 ζr pub.xRest) (fun _ => cat6 ζc pub.yRest) q 1
    (fun _ => 1) (fun _ => 1) (hq _) (hq _) (fun _ _ => PolyLe.const _ _) (fun i _ => hq i)
    (fun _ _ => PolyLe.const _ _) (fun i _ => hq _)).mono ?_
  rw [Finset.sum_const, Finset.card_range, smul_eq_mul]
  omega

theorem psi_degLe_x (y : Fin 6 → F) : DegLe 1 fun x => psi sh pub τ z x y := fun v j => by
  refine PolyLe.sum _ fun σ _ => PolyLe.const_mul _ (PolyLe.sum _ fun k _ => ?_)
  refine (polyLe_sparsePhi sh.ell pub.alpha (fun t => cat6 (Function.update v j t) pub.xRest)
    (fun _ => cat6 y pub.yRest) (fun _ p => planeAt sh z (τ σ) p k.val) 0 (fun i => if i = j.val then 1 else 0)
    (fun _ => 0) (PolyLe.const _ _) (PolyLe.const _ _) (fun i _ => polyLe_cat6 v j _ i)
    (fun _ _ => PolyLe.const _ _) (fun _ _ => PolyLe.const _ _) (fun _ _ => PolyLe.const _ _)).mono ?_
  simp only [add_zero, zero_add]
  exact sum_ite_le_one _ _

theorem psi_degLe_y (x : Fin 6 → F) : DegLe 1 (psi sh pub τ z x) := fun v j => by
  refine PolyLe.sum _ fun σ _ => PolyLe.const_mul _ (PolyLe.sum _ fun k _ => ?_)
  refine (polyLe_sparsePhi sh.ell pub.alpha (fun _ => cat6 x pub.xRest)
    (fun t => cat6 (Function.update v j t) pub.yRest) (fun _ p => planeAt sh z (τ σ) p k.val) 0 (fun _ => 0)
    (fun i => if i = j.val then 1 else 0) (PolyLe.const _ _) (PolyLe.const _ _) (fun _ _ => PolyLe.const _ _)
    (fun _ _ => PolyLe.const _ _) (fun i _ => polyLe_cat6 v j _ i) (fun _ _ => PolyLe.const _ _)).mono ?_
  simp only [zero_add]
  exact sum_ite_le_one _ _

theorem g2_degLe (A : Arith F K) (ζc : Fin 6 → F) : DegLe (1 + 1) (g2 A sh pub τ z ζc) := fun v j =>
  (PolyLe.sum _ fun s _ => (polyLe_eqAt v j s.val).mul_const _).mul (psi_degLe_x sh pub τ z ζc v j)

theorem g3_degLe (ζc ζr : Fin 6 → F) : DegLe (2 * sh.ell + 1) (g3 sh pub τ z ζc ζr) := fun v j =>
  PolyLe.sum _ fun σ _ => PolyLe.const_mul _ (phiAt_polyLe sh pub ζc ζr
    (fun t => planeExt sh z (τ σ) (Function.update v j t)) fun p => planeExt_polyLe sh z (τ σ) v j p)

theorem g4_degLe (ζc ζr : Fin 6 → F) (rk : Fin sh.nu → F) :
    DegLe (2 * sh.ell + 1 + 1) (g4 sh pub τ z ζc ζr rk) := fun v j =>
  ((PolyLe.sum _ fun σ _ => (polyLe_eqAt v j σ.val).mul_const _).mul (phiAt_polyLe sh pub ζc ζr
    (fun t => slotPlanes sh τ z rk (Function.update v j t)) fun p => slotPlanes_polyLe sh τ z rk v j p)).mono
    (by omega)

end Degrees

/-! ### The sums each check hands on -/

section Sums

variable [CharP F 2] (sh : SparseShape) (pub : SparsePub F sh) (τ : Fin (2 ^ sh.sLog) → ℕ) (z : Witness sh.mReg)

omit pub τ in
theorem planeExt_bitsOf (t : ℕ) (b : Fin (2 ^ sh.nu)) :
    planeExt (F := F) sh z t (bitsOf sh.nu b.val) = fun p => planeAt sh z t p b.val :=
  funext fun p => sum_eqAt_bitsOf le_rfl b.val b.isLt (fun k : Fin (2 ^ sh.nu) => planeAt sh z t p k.val)

omit pub in
theorem slotPlanes_bitsOf (rk : Fin sh.nu → F) (b : Fin (2 ^ sh.sLog)) :
    slotPlanes sh τ z rk (bitsOf sh.sLog b.val) = planeExt sh z (τ b) rk :=
  funext fun p => sum_eqAt_bitsOf le_rfl b.val b.isLt (fun σ => planeExt sh z (τ σ) rk p)

omit τ z in
theorem lagAt_bitsOf (A : Arith F K) (b : Fin 64) : lagAt A pub (bitsOf 6 b.val) = A.lagS pub.z b :=
  sum_eqAt_bitsOf (by norm_num) b.val b.isLt _

omit τ z in
theorem uAt_bitsOf (b : Fin (2 ^ sh.sLog)) : uAt pub (bitsOf sh.sLog b.val) = pub.U b :=
  sum_eqAt_bitsOf le_rfl b.val b.isLt _

/-- **H1's claim**: the hidden fold's extension at the skip's columns is H2's sum. -/
theorem fold_sum (A : Arith F K) (ζc : Fin 6 → F) :
    ∑ s : Fin 64, eqAt ζc s.val * sparseFold A sh pub τ z s = cube (g2 A sh pub τ z ζc) := by
  have hf : ∀ s : Fin 64, sparseFold A sh pub τ z s =
      ∑ r : Fin 64, A.lagS pub.z r * psi sh pub τ z (bitsOf 6 r.val) (bitsOf 6 s.val) := fun s => rfl
  simp only [hf, Finset.mul_sum]
  rw [Finset.sum_comm, cube_eq_fin (by norm_num : 64 = 2 ^ 6)]
  refine Finset.sum_congr rfl fun r _ => ?_
  rw [g2, lagAt_bitsOf, ← sum_fin_eqAt_mul (by norm_num : 64 = 2 ^ 6) _ (psi_degLe_y sh pub τ z (bitsOf 6 r.val)) ζc,
    Finset.mul_sum]
  exact Finset.sum_congr rfl fun s _ => by ring

/-- **H2's end**: `g2` at the skip's rows is the skip's weights times H3's sum. -/
theorem g2_eq (A : Arith F K) (ζc ζr : Fin 6 → F) :
    g2 A sh pub τ z ζc ζr = lagAt A pub ζr * cube (g3 sh pub τ z ζc ζr) := by
  rw [g2, cube_eq_fin rfl]
  congr 1
  simp only [g3, planeExt_bitsOf, psi, Finset.mul_sum]
  exact Finset.sum_comm

/-- **H3's entries' end**: `g3` at the entries' point is the slots' sum. -/
theorem g3_eq_cube (ζc ζr : Fin 6 → F) (rk : Fin sh.nu → F) :
    g3 sh pub τ z ζc ζr rk = cube (g4 sh pub τ z ζc ζr rk) := by
  rw [cube_eq_fin rfl, g3]
  refine Finset.sum_congr rfl fun b _ => ?_
  rw [g4, uAt_bitsOf, slotPlanes_bitsOf]

omit [CharP F 2] pub in
/-- **H4's batch**: the planes' values at `ζp` are the types' values weighted by the slots'. -/
theorem batch_split (hτ : ∀ σ, τ σ < 2 ^ sh.tLog) (rk : Fin sh.nu → F) (rs : Fin sh.sLog → F)
    (ζp : Fin sh.pLog → F) :
    ∑ p : Fin (2 ^ sh.pLog), eqAt ζp p.val * slotPlanes sh τ z rk rs p.val =
      ∑ t : Fin (2 ^ sh.tLog), slotWeight sh τ rs t.val * typeVal sh z rk ζp t.val := by
  have hL : ∀ σ : Fin (2 ^ sh.sLog), ∑ t : Fin (2 ^ sh.tLog),
      (if τ σ = t.val then eqAt rs σ.val else 0) * typeVal sh z rk ζp t.val = eqAt rs σ.val * typeVal sh z rk ζp (τ σ) :=
    fun σ => by
      rw [Finset.sum_eq_single ⟨τ σ, hτ σ⟩ (fun t _ ht => ?_) (fun h => absurd (Finset.mem_univ _) h)]
      · simp
      · rw [if_neg (fun h => ht (Fin.ext h.symm)), zero_mul]
  calc ∑ p : Fin (2 ^ sh.pLog), eqAt ζp p.val * slotPlanes sh τ z rk rs p.val
      = ∑ σ : Fin (2 ^ sh.sLog), eqAt rs σ.val * typeVal sh z rk ζp (τ σ) := by
        simp only [slotPlanes, typeVal, Finset.mul_sum]
        refine Finset.sum_comm.trans (Finset.sum_congr rfl fun σ _ => Finset.sum_congr rfl fun p _ => ?_)
        ring
    _ = ∑ σ : Fin (2 ^ sh.sLog), ∑ t : Fin (2 ^ sh.tLog),
          (if τ σ = t.val then eqAt rs σ.val else 0) * typeVal sh z rk ζp t.val :=
        Finset.sum_congr rfl fun σ _ => (hL σ).symm
    _ = _ := Finset.sum_comm.trans (Finset.sum_congr rfl fun t _ => (Finset.sum_mul _ _ _).symm)

omit pub in
/-- With no types, the slots' weight is one, as is `eqAt` at the empty point. -/
theorem slotWeight_tLog_zero (h0 : sh.tLog = 0) (hτ : ∀ σ, τ σ < 2 ^ sh.tLog) (rs : Fin sh.sLog → F)
    (ρ : Fin sh.tLog → F) (g : ℕ → F) :
    ∑ t : Fin (2 ^ sh.tLog), slotWeight sh τ rs t.val * g t.val = ∑ t : Fin (2 ^ sh.tLog), eqAt ρ t.val * g t.val := by
  refine Finset.sum_congr rfl fun t _ => ?_
  have h1 : 2 ^ sh.tLog = 1 := by rw [h0, pow_zero]
  have ht : t.val = 0 := by have := t.isLt; omega
  have hρ : eqAt ρ t.val = 1 := Finset.prod_eq_one fun j _ => absurd j.isLt (by omega)
  have hw : slotWeight sh τ rs t.val = 1 := by
    have hs : ∀ σ, τ σ = t.val := fun σ => by have := hτ σ; omega
    rw [slotWeight, Finset.sum_congr rfl fun σ _ => if_pos (hs σ), ← Finset.sum_range (fun σ => eqAt rs σ)]
    exact sum_eqAt _ rs
  rw [hρ, hw]

omit [CharP F 2] pub τ in
/-- **A registered claim's value on the table** is its types' values at its point. -/
theorem toClaim_eval (c : RegClaim F sh) :
    c.toClaim.eval z = ∑ t : Fin (2 ^ sh.tLog), eqAt c.rho t.val * typeVal sh z c.rk c.zetaP t.val := by
  set f : ℕ → F := fun a => eqAt (fun j : Fin sh.mReg => c.coord j.val) a *
    (if h : a < 2 ^ sh.mReg then ((z ⟨a, h⟩).val : F) else 0) with hf
  have h1 : c.toClaim.eval z = ∑ a ∈ Finset.range (2 ^ sh.mReg), f a := by
    rw [Finset.sum_range]
    unfold Claim.eval
    refine Finset.sum_congr rfl fun a _ => ?_
    simp only [hf, RegClaim.toClaim, dif_pos a.isLt, Fin.eta]
    rw [coord_split c a.val a.isLt]
  have h2 : 2 ^ sh.mReg = 2 ^ sh.nu * (2 ^ sh.pLog * 2 ^ sh.tLog) := by
    rw [SparseShape.mReg, pow_add, pow_add, mul_assoc]
  rw [h1, show Finset.range (2 ^ sh.mReg) = Finset.range (2 ^ sh.nu * (2 ^ sh.pLog * 2 ^ sh.tLog)) by rw [h2],
    sum_range_mul, sum_range_mul]
  simp only [Finset.sum_range]
  simp only [typeVal, planeExt, Finset.mul_sum]
  refine Finset.sum_congr rfl fun t _ => Finset.sum_congr rfl fun p _ => Finset.sum_congr rfl fun k _ => ?_
  simp only [hf, planeAt]
  rw [coord_eqAt c k.val p.val t.val k.isLt p.isLt]
  ring

end Sums

end SparseCount

namespace SparseCount

theorem sparseError_eq (A : Arith F K) (hA : A.Correct) (sh : SparseShape) :
    ENNReal.ofReal (sparseError sh) =
      ((6 : ℕ) : ℝ≥0∞) / Fintype.card F + (((6 * (1 + 1) : ℕ) : ℝ≥0∞) / Fintype.card F +
        (((sh.nu * (2 * sh.ell + 1) : ℕ) : ℝ≥0∞) / Fintype.card F +
          (((sh.sLog * (2 * sh.ell + 1 + 1) : ℕ) : ℝ≥0∞) / Fintype.card F +
            ((sh.pLog : ℝ≥0∞) / Fintype.card F + ((sh.tLog : ℝ≥0∞) / Fintype.card F + 0))))) := by
  rw [sparseError, ofReal_nat_div_qF, hA.card_F]
  simp only [add_zero, ENNReal.div_add_div_same]
  congr 1
  · push_cast
    ring
  · norm_num

/-- `sparse_sound`'s count, coin by coin. -/
theorem phase_le (A : Arith F K) (hA : A.Correct) (sh : SparseShape) (pub : SparsePub F sh)
    (τ : Fin (2 ^ sh.sLog) → ℕ) (hτ : ∀ σ, τ σ < 2 ^ sh.tLog) (z : Witness sh.mReg) (h : Fin 64 → F)
    (hne : h ≠ sparseFold A sh pub τ z) :
    value (fun o : Option (RegClaim F sh) => ∃ c, o = some c ∧ c.toClaim.Holds z) (sparsePhase A sh pub τ h) ≤
      ENNReal.ofReal (sparseError sh) := by
  have : CharP F 2 := hA.charTwo
  rw [sparseError_eq A hA sh]
  unfold sparsePhase
  -- H1: the skip's columns
  refine value_drawFin_le _ 6 (fun b => vget h b - vget (sparseFold A sh pub τ z) b)
    (exists_vget_ne (by norm_num) hne) _ _ fun ζc hζc => ?_
  rw [sum_range_vget_sub (N := 64) (by norm_num) ζc, sub_ne_zero, fold_sum] at hζc
  -- H2: the skip's rows
  refine value_degRounds_bind_le _ False 1 6 _ (g2_degLe sh pub τ z A ζc) _ (Or.inr hζc) _ _
    fun ζr c₂ hc₂ => ?_
  dsimp only
  refine value_bind_le_of_forall _ _ _ _ fun v₂ => ?_
  -- H3: the entries
  refine value_degRounds_bind_le _ (c₂ ≠ lagAt A pub ζr * v₂) (2 * sh.ell) sh.nu _ (g3_degLe sh pub τ z ζc ζr) v₂
    ?_ _ _ fun rk c₃ hc₃ => ?_
  · by_cases hE : c₂ = lagAt A pub ζr * v₂
    · refine Or.inr fun hv => (hc₂.resolve_left id) ?_
      rw [hE, hv, g2_eq]
    · exact Or.inl hE
  dsimp only
  -- H3: the slots
  refine value_degRounds_bind_le _ (c₂ ≠ lagAt A pub ζr * v₂) (2 * sh.ell + 1) sh.sLog _
    (g4_degLe sh pub τ z ζc ζr rk) c₃ (by rwa [← g3_eq_cube]) _ _ fun rs c₄ hc₄ => ?_
  dsimp only
  refine value_bind_le_of_forall _ _ _ _ fun u => ?_
  -- H4: the planes
  refine value_drawFin_le_or _ (c₂ ≠ lagAt A pub ζr * v₂ ∨ c₄ ≠ uAt pub rs * phiAt sh pub ζc ζr (vget u))
    sh.pLog (fun b => vget u b - vget (fun p : Fin (2 ^ sh.pLog) => slotPlanes sh τ z rk rs p.val) b) ?_ _ _
    fun ζp hζp => ?_
  · by_cases hE : c₂ ≠ lagAt A pub ζr * v₂ ∨ c₄ ≠ uAt pub rs * phiAt sh pub ζc ζr (vget u)
    · exact Or.inl hE
    · push Not at hE
      refine Or.inr (exists_vget_ne rfl ?_)
      rintro rfl
      refine hc₄.resolve_left (not_not.mpr hE.1) ?_
      rw [hE.2, g4]
      congr 1
      exact sparsePhi_congr _ _ _ _ fun p hp => vget_fun _ (lt_two_pow_pLog sh hp)
  rw [sum_range_vget_sub rfl ζp, sub_ne_zero] at hζp
  by_cases h0 : sh.tLog = 0
  · -- no types: the planes' batch is the claim
    rw [if_pos h0]
    refine le_trans (b := 0) ?_ zero_le
    refine value_bind_le_of_forall _ _ _ _ fun vt => ?_
    refine value_bind_le_of_forall _ _ _ 0 fun ρ => (value_pure_eq_zero ?_).le
    rintro ⟨c, hc, hhold⟩
    simp only [sparseVerdict, if_pos h0] at hc
    split_ifs at hc with hchk
    · cases hc
      unfold Claim.Holds at hhold
      rw [toClaim_eval] at hhold
      dsimp only [RegClaim.toClaim] at hhold
      rcases hζp with (hE | hE) | hS
      · exact hE hchk.1
      · exact hE hchk.2
      · rw [batch_split sh τ z hτ, slotWeight_tLog_zero sh τ h0 hτ rs ρ (typeVal sh z rk ζp)] at hS
        exact hS hhold.symm
  · -- H4: the type's point
    rw [if_neg h0]
    refine value_bind_le_of_forall _ _ _ _ fun vt => ?_
    refine value_drawFin_le_or _ ((c₂ ≠ lagAt A pub ζr * v₂ ∨ c₄ ≠ uAt pub rs * phiAt sh pub ζc ζr (vget u)) ∨
        ∑ q : Fin (2 ^ sh.pLog), eqAt ζp q.val * u q ≠ ∑ t : Fin (2 ^ sh.tLog), slotWeight sh τ rs t.val * vt t)
      sh.tLog (fun b => vget vt b - vget (fun t : Fin (2 ^ sh.tLog) => typeVal sh z rk ζp t.val) b) ?_ _ _
      fun ρ hρ => ?_
    · by_cases hE : (c₂ ≠ lagAt A pub ζr * v₂ ∨ c₄ ≠ uAt pub rs * phiAt sh pub ζc ζr (vget u)) ∨
          ∑ q : Fin (2 ^ sh.pLog), eqAt ζp q.val * u q ≠ ∑ t : Fin (2 ^ sh.tLog), slotWeight sh τ rs t.val * vt t
      · exact Or.inl hE
      · push Not at hE
        refine Or.inr (exists_vget_ne rfl ?_)
        rintro rfl
        have hS := hζp.resolve_left fun h' => h'.elim (· hE.1.1) (· hE.1.2)
        rw [batch_split sh τ z hτ] at hS
        exact hS hE.2
    rw [sum_range_vget_sub rfl ρ, sub_ne_zero] at hρ
    refine (value_pure_eq_zero ?_).le
    rintro ⟨c, hc, hhold⟩
    simp only [sparseVerdict, if_neg h0] at hc
    split_ifs at hc with hchk
    · cases hc
      unfold Claim.Holds at hhold
      rw [toClaim_eval] at hhold
      dsimp only [RegClaim.toClaim] at hhold
      rcases hρ with ((hE | hE) | hE) | hS
      · exact hE hchk.1
      · exact hE hchk.2.1
      · exact hE hchk.2.2
      · exact hS hhold.symm

end SparseCount

/-- **The sparse phase is sound**: at slots' types in range, a hidden fold `h` other than the registered table's
(`sparseFold`) yields a registered claim that holds on the table with probability at most `sparseError`. -/
theorem sparse_sound (A : Arith F K) (hA : A.Correct) (sh : SparseShape) (pub : SparsePub F sh)
    (τ : Fin (2 ^ sh.sLog) → ℕ) (hτ : ∀ σ, τ σ < 2 ^ sh.tLog) (z : Witness sh.mReg) (h : Fin 64 → F)
    (hne : h ≠ sparseFold A sh pub τ z) :
    value (fun o : Option (RegClaim F sh) => ∃ c, o = some c ∧ c.toClaim.Holds z) (sparsePhase A sh pub τ h) ≤
      ENNReal.ofReal (sparseError sh) :=
  SparseCount.phase_le A hA sh pub τ hτ z h hne

end FlockC3.Fork
