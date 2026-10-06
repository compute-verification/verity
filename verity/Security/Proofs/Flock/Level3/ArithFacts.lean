/-
The soundness lane's `Model.Arith.Correct` facts (Project store
`internal/lanes/flock-verifier/20260927T0200Z-handoff-from-flock-soundness-arith-facts.md`), each in the shape of its field,
for the executable's functions: `F = GF128`, `K = GF256`, and the concrete `pack`, `unpack`, `pinned`, `omega` and `lo`
below. The instance itself lives in the soundness package.
-/
import Proofs.Flock.Level3.Seam
import Flock.Piop
import Mathlib.Data.Fintype.Card
import Mathlib.Data.Fintype.Prod
import Mathlib.Algebra.Polynomial.Degree.Operations
import Mathlib.Algebra.Polynomial.Degree.Support

open Polynomial Flock

set_option maxRecDepth 20000

namespace FlockLevel3

/-! ### The fields: finite, of order `2^128` and `2^256`, characteristic 2 -/

def u64Fin : UInt64 ≃ Fin (2 ^ 64) := ⟨fun x => x.toFin, UInt64.ofFin, fun _ => rfl, fun _ => rfl⟩

def f128Equiv : GF128 ≃ Fin (2 ^ 64) × Fin (2 ^ 64) :=
  ⟨fun a => (GF128.val a |>.lo.toFin, GF128.val a |>.hi.toFin), fun p => GF128.of ⟨UInt64.ofFin p.1, UInt64.ofFin p.2⟩,
    fun _ => rfl, fun _ => rfl⟩

instance instFintypeGF128 : Fintype GF128 := Fintype.ofEquiv _ f128Equiv.symm
instance instDecidableEqGF128 : DecidableEq GF128 := inferInstanceAs (DecidableEq F128)

def f256Equiv : GF256 ≃ GF128 × GF128 :=
  ⟨fun a => ((a : F256).c0, (a : F256).c1), fun p => GF256.of ⟨p.1.val, p.2.val⟩, fun _ => rfl, fun _ => rfl⟩

instance instFintypeGF256 : Fintype GF256 := Fintype.ofEquiv _ f256Equiv.symm
instance instDecidableEqGF256 : DecidableEq GF256 := inferInstanceAs (DecidableEq F256)

theorem card_F : Fintype.card GF128 = 2 ^ 128 := by
  rw [Fintype.card_congr f128Equiv, Fintype.card_prod, Fintype.card_fin]; norm_num

theorem card_K : Fintype.card GF256 = 2 ^ 256 := by
  rw [Fintype.card_congr f256Equiv, Fintype.card_prod, card_F]; norm_num

theorem charTwo : CharP GF128 2 := inferInstance

theorem GF256.two_eq_zero : (2 : GF256) = 0 := by
  rw [← map_ofNat embed 2, GF128.two_eq_zero, map_zero]

instance instCharPGF256OfNatNat : CharP GF256 2 := (CharP.charP_iff_prime_eq_zero Nat.prime_two).mpr (by exact_mod_cast GF256.two_eq_zero)

theorem charTwoK : CharP GF256 2 := inferInstance

/-! ### Packing: bit `v` is the coefficient of `x^v` (§12.1) -/

/-- A GF128 element as its residue class. -/
noncomputable def toA (a : GF128) : AdjoinRoot G := toAdj a.val

theorem toA_injective : Function.Injective toA := fun _ _ h => toAdj_injective h
theorem toA_of (a : F128) : toA (GF128.of a) = toAdj a := rfl
theorem toA_add (a a' : GF128) : toA (a + a') = toA a + toA a' := equivAdj.apply_symm_apply _

/-- The bits of an element in the packing basis `x^v`. -/
noncomputable def unpack (a : GF128) : Fin 128 → ZMod 2 := fun v => (F128.poly a.val).coeff v

/-- `Σ_v b_v · x^v`. -/
noncomputable def pack (b : Fin 128 → ZMod 2) : GF128 :=
  GF128.of (equivAdj.symm (AdjoinRoot.mk G (∑ v : Fin 128, C (b v) * X ^ (v : ℕ))))

theorem sum_degree_lt (b : Fin 128 → ZMod 2) : (∑ v : Fin 128, C (b v) * X ^ (v : ℕ)).degree < 128 := by
  have := degree_sum_fin_lt b; exact_mod_cast this

theorem poly_eq_of_mk_eq {p : P} (a : F128) (hp : p.degree < 128) (h : toAdj a = AdjoinRoot.mk G p) : F128.poly a = p := by
  rw [toAdj, AdjoinRoot.mk_eq_mk] at h
  have hdeg : (F128.poly a - p).degree < G.degree := by
    rw [G_degree]; exact lt_of_le_of_lt (degree_sub_le _ _) (max_lt (F128.poly_degree_lt a) hp)
  exact sub_eq_zero.mp (eq_zero_of_dvd_of_degree_lt h hdeg)

theorem poly_pack (b : Fin 128 → ZMod 2) : F128.poly (pack b).val = ∑ v : Fin 128, C (b v) * X ^ (v : ℕ) :=
  poly_eq_of_mk_eq _ (sum_degree_lt b) (equivAdj.apply_symm_apply _)

theorem unpack_pack (b : Fin 128 → ZMod 2) : unpack (pack b) = b := by
  funext v
  rw [unpack, poly_pack, finsetSum_coeff]
  simp only [coeff_C_mul_X_pow]
  rw [Finset.sum_eq_single v (fun w _ hw => if_neg (fun h => hw (Fin.ext h.symm))) (by simp)]
  simp

theorem pack_unpack (a : GF128) : pack (unpack a) = a := by
  apply toA_injective
  show toAdj (equivAdj.symm _) = toAdj a.val
  rw [← equivAdj_apply, Equiv.apply_symm_apply, toAdj]
  congr 1
  have hd : (F128.poly a.val).natDegree < 128 := by
    have := F128.poly_degree_lt a.val
    rcases eq_or_ne (F128.poly a.val) 0 with h | h
    · rw [h]; simp
    · exact (natDegree_lt_iff_degree_lt h).mpr (by exact_mod_cast this)
  conv_rhs => rw [as_sum_range' (F128.poly a.val) 128 hd]
  rw [← Fin.sum_univ_eq_sum_range (fun i => monomial i ((F128.poly a.val).coeff i)) 128]
  refine Finset.sum_congr rfl fun v _ => ?_
  rw [unpack, C_mul_X_pow_eq_monomial]

theorem val_add (a a' : GF128) : (a + a').val = a.val + a'.val :=
  (GF128.add_eq a.val a'.val : GF128.of a.val + GF128.of a'.val = GF128.of (a.val + a'.val))

theorem unpack_add (a a' : GF128) : unpack (a + a') = unpack a + unpack a' := by
  funext v
  rw [unpack, val_add, F128.poly_add, coeff_add]; rfl

/-! ### `K = F ⊕ F·u`: a K coin is two F limbs (§13.2) -/

theorem ofLimbs_bijective : Function.Bijective fun w : GF128 × GF128 => embed w.1 + embed w.2 * u := by
  have h : (fun w : GF128 × GF128 => embed w.1 + embed w.2 * u) = fun w => f256Equiv.symm w :=
    funext fun w => (ofLimbs w.1.val w.2.val).symm
  rw [h]; exact f256Equiv.symm.bijective

/-! ### Query positions: the low bits of `lo` are balanced (§13.5) -/

/-- The low 64-bit word. -/
def lo (w : GF128) : ℕ := (GF128.val w).lo.toNat

theorem card_residue (b : ℕ) (hb : b ≤ 64) (t : ℕ) (ht : t < 2 ^ b) :
    (Finset.univ.filter fun l : Fin (2 ^ 64) => l.val % 2 ^ b = t).card = 2 ^ (64 - b) := by
  have hpow : 2 ^ 64 = 2 ^ b * 2 ^ (64 - b) := by rw [← pow_add]; congr 1; omega
  have hb0 : 0 < 2 ^ b := Nat.two_pow_pos b
  rw [← Fintype.card_fin (2 ^ (64 - b))]
  refine Finset.card_bij' (fun l _ => ⟨l.val / 2 ^ b, ?_⟩) (fun k _ => ⟨t + 2 ^ b * k.val, ?_⟩) ?_ ?_ ?_ ?_
  · rw [Nat.div_lt_iff_lt_mul hb0, mul_comm, ← hpow]; exact l.isLt
  · have := k.isLt
    calc t + 2 ^ b * k.val < 2 ^ b + 2 ^ b * k.val := by omega
      _ = 2 ^ b * (k.val + 1) := by ring
      _ ≤ 2 ^ b * 2 ^ (64 - b) := Nat.mul_le_mul_left _ this
      _ = 2 ^ 64 := hpow.symm
  · intro l _; simp
  · intro k _; simp [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt ht]
  · intro l hl
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hl
    ext; simp only; rw [← hl]; exact Nat.mod_add_div _ _
  · intro k _
    ext; simp only
    rw [Nat.add_mul_div_left _ _ hb0, Nat.div_eq_of_lt ht, zero_add]

set_option linter.constructorNameAsVariable false in
theorem lo_balanced : ∀ b ≤ 64, ∀ t < 2 ^ b,
    (Finset.univ.filter fun w : GF128 => lo w % 2 ^ b = t).card = Fintype.card GF128 / 2 ^ b := by
  intro b hb t ht
  have e : (Finset.univ.filter fun w : GF128 => lo w % 2 ^ b = t).card =
      (Finset.univ.filter fun p : Fin (2 ^ 64) × Fin (2 ^ 64) => p.1.val % 2 ^ b = t).card := by
    refine Finset.card_bij' (fun w _ => f128Equiv w) (fun p _ => f128Equiv.symm p) ?_ ?_ ?_ ?_
    · intro w hw
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hw ⊢
      exact hw
    · intro p hp
      simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hp ⊢
      exact hp
    · intro w _; exact f128Equiv.symm_apply_apply w
    · intro p _; exact f128Equiv.apply_symm_apply p
  have hprod : (Finset.univ.filter fun p : Fin (2 ^ 64) × Fin (2 ^ 64) => p.1.val % 2 ^ b = t) =
      (Finset.univ.filter fun l : Fin (2 ^ 64) => l.val % 2 ^ b = t) ×ˢ Finset.univ := by ext; simp
  rw [e, hprod, Finset.card_product, card_residue b hb t ht, Finset.card_univ, Fintype.card_fin, card_F]
  symm
  apply Nat.div_eq_of_eq_mul_left (Nat.two_pow_pos b)
  rw [← pow_add, ← pow_add, show 64 - b + 64 + b = 128 by omega]

/-! ### The Reed–Solomon domain points (§13.1) -/

/-- `ω_q`: the element whose low word is `q`. -/
def omega (q : ℕ) : GF128 := GF128.of ⟨q.toUInt64, 0⟩

theorem omega_injective : ∀ d ≤ 64, Set.InjOn omega {q | q < 2 ^ d} := by
  intro d hd q hq q' hq' h
  have hq64 : q < 2 ^ 64 := lt_of_lt_of_le hq (Nat.pow_le_pow_right (by norm_num) hd)
  have hq64' : q' < 2 ^ 64 := lt_of_lt_of_le hq' (Nat.pow_le_pow_right (by norm_num) hd)
  have := congrArg (fun w : GF128 => (GF128.val w).lo.toNat) h
  simp only [omega, GF128.of, GF128.val, UInt64.toNat_ofNat'] at this
  norm_num at hq64 hq64' this
  omega

/-! ### The pinned zerocheck coordinates (§9.2) -/

/-- The seven pinned coordinates (the executable's `Flock.pinned`). -/
def pinnedF (i : Fin 7) : GF128 := GF128.of (Flock.pinned[i.val]!)

theorem pinned_exec_ne_one : ∀ i : Fin 7, Flock.pinned[i.val]! ≠ 1 := by decide +kernel

theorem pinned_ne_one : ∀ i, pinnedF i ≠ 1 := by
  intro i h
  rw [GF128.one_eq] at h
  exact pinned_exec_ne_one i h

end FlockLevel3
