/-
Level 3, L3-F: the executable's `F128` is the ring GF(2)[X]/(G). `toAdj` sends an element to its residue class; it is a
bijection that carries `+` and `*` to the quotient ring's (the field structure, once `G` is irreducible: `GF128Field`).
-/
import Proofs.Flock.Level3.GF128
import Mathlib.RingTheory.AdjoinRoot
import Mathlib.Algebra.Ring.TransferInstance
import Mathlib.Tactic.FinCases
import Mathlib.Algebra.Field.ZMod

open Polynomial Flock

namespace FlockLevel3

instance instFactPrimeOfNatNat_flockLevel3 : Fact (Nat.Prime 2) := ⟨Nat.prime_two⟩

/-- Every polynomial with no coefficient at `k` or above is the polynomial of a number below `2^k`. -/
theorem natPoly_surj : ∀ (k : ℕ) (r : P), (∀ i, k ≤ i → r.coeff i = 0) → ∃ n, n < 2 ^ k ∧ natPoly n = r := by
  intro k
  induction k with
  | zero =>
    intro r h
    refine ⟨0, by norm_num, ?_⟩
    ext i; simp [h i (Nat.zero_le _)]
  | succ k ih =>
    intro r h
    obtain ⟨n, hn, hr⟩ := ih r.divX (fun i hi => by rw [coeff_divX]; exact h _ (by omega))
    have hb : (r.coeff 0).val < 2 := ZMod.val_lt _
    refine ⟨2 ^ 1 * n + (r.coeff 0).val, by rw [pow_succ]; omega, ?_⟩
    rw [natPoly_split _ _ 1 hb, hr, pow_one]
    have hc : natPoly (r.coeff 0).val = C (r.coeff 0) := by
      have : ∀ z : ZMod 2, natPoly z.val = C z := by
        intro z; fin_cases z
        · show natPoly 0 = C 0; simp
        · show natPoly 1 = C 1; simp [natPoly_one]
      exact this _
    rw [hc]
    exact X_mul_divX_add r

/-- The residue class of an element. -/
noncomputable def toAdj (a : F128) : AdjoinRoot G := AdjoinRoot.mk G (F128.poly a)

theorem mk_modByMonic (p : P) : AdjoinRoot.mk G (p %ₘ G) = AdjoinRoot.mk G p := by
  rw [AdjoinRoot.mk_eq_mk, modByMonic_eq_sub_mul_div]
  exact ⟨-(p /ₘ G), by ring⟩

theorem toAdj_add (a b : F128) : toAdj (a + b) = toAdj a + toAdj b := by
  simp [toAdj, F128.poly_add]

theorem toAdj_mul (a b : F128) : toAdj (a * b) = toAdj a * toAdj b := by
  simp [toAdj, F128.poly_mul, mk_modByMonic]

theorem toAdj_zero : toAdj 0 = 0 := by simp [toAdj, F128.poly_zero]
theorem toAdj_one : toAdj 1 = 1 := by simp [toAdj, F128.poly_one]

set_option maxRecDepth 20000 in
theorem toAdj_injective : Function.Injective toAdj := by
  intro a b h
  rw [toAdj, toAdj, AdjoinRoot.mk_eq_mk] at h
  have hdeg : (F128.poly a - F128.poly b).degree < G.degree := by
    rw [G_degree]
    exact lt_of_le_of_lt (degree_sub_le _ _) (max_lt (F128.poly_degree_lt a) (F128.poly_degree_lt b))
  have h0 : F128.poly a - F128.poly b = 0 := eq_zero_of_dvd_of_degree_lt h hdeg
  exact F128.poly_injective (sub_eq_zero.mp h0)

theorem toAdj_surjective : Function.Surjective toAdj := by
  intro y
  obtain ⟨p, rfl⟩ := AdjoinRoot.mk_surjective y
  have hdeg : (p %ₘ G).degree < 128 := by rw [← G_degree]; exact degree_modByMonic_lt p G_monic
  obtain ⟨n, hn, hr⟩ := natPoly_surj 128 (p %ₘ G) (fun i hi => coeff_eq_zero_of_degree_lt
    (lt_of_lt_of_le hdeg (by exact_mod_cast hi)))
  refine ⟨⟨UInt64.ofNat (n % 2 ^ 64), UInt64.ofNat (n / 2 ^ 64)⟩, ?_⟩
  have hw : w128 (UInt64.ofNat (n % 2 ^ 64)) (UInt64.ofNat (n / 2 ^ 64)) = n := by
    unfold w128
    rw [UInt64.toNat_ofNat', UInt64.toNat_ofNat', Nat.mod_mod, Nat.mod_eq_of_lt (by omega : n / 2 ^ 64 < 2 ^ 64)]
    omega
  rw [toAdj, F128.poly, hw, hr, mk_modByMonic]

noncomputable def equivAdj : F128 ≃ AdjoinRoot G := Equiv.ofBijective toAdj ⟨toAdj_injective, toAdj_surjective⟩

theorem equivAdj_apply (a : F128) : equivAdj a = toAdj a := rfl

end FlockLevel3
