import Proofs.Flock.Soundness.Discharge.Composed.Coined

/-!
# The coin facts of `CoinedOK`, from `Admits`

At a coin vector the honest prover doesn't refuse (`CoinedSetup.Admits`), a `Coined` table that reads its coins (`β`,
`ρ`, the rank check's weights and `embed` the inclusion of F128 in F256: `CoinedOK`'s coin ties) passes Lemma B's coin
checks:

* `rank_of_maskRankOk`: the rank check (`Table.Rank`) is `MaskRankOk` at the table's weights.
* `fOf_ne_zero`: `β ≠ 0` from `beta.isZero = false`.
* `wr_bijective`: `W^r` bijective from `Zk.rhoIndependent`, the determinant of `(a, b) ↦ ρ₀·a + ρ₁·b` over F128.
-/

namespace FlockSoundness.Discharge.Composed

open Flock FlockLevel3 ZkView Refine

local notation "𝔽" => FlockLevel3.GF128

/-- **The rank check from the mask slot's rank**: at weights `eq(p_k, x)`, `Table.Rank` is `MaskRankOk`. -/
theorem rank_of_maskRankOk {W Hi VL : Type} {z : Sizes} (T : Coined GF256 W Hi VL z) (pts : Fin z.nC → List F128)
    (words : Fin z.nMask → ℕ) (h : ∀ k x, T.eqw k x = fOf (eqWord (pts k) (words x))) (hM : MaskRankOk pts words) :
    T.tab.Rank := by
  have e : (fun c : Fin z.nMask → ZMod 2 => fun k => ∑ x, T.eqw k x * ZK.Table.castF (F := 𝔽) (c x)) =
      fun u => fun j => ∑ x, fOf (eqWord (pts j) (words x)) * ZK.Table.castF (F := 𝔽) (u x) := by
    funext c k
    simp only [h]
  show Function.Surjective (fun c : Fin z.nMask → ZMod 2 => fun k => ∑ x, T.eqw k x * ZK.Table.castF (F := 𝔽) (c x))
  rw [e]
  exact hM

theorem isZero_zero : (0 : F128).isZero = true := rfl

/-- **`β ≠ 0`** from the executable's test. -/
theorem fOf_ne_zero {b : F128} (h : b.isZero = false) : fOf b ≠ 0 := by
  intro h0
  have hb : b = 0 := by
    have : GF128.of b = GF128.of 0 := h0.trans GF128.zero_eq
    exact this
  rw [hb, isZero_zero] at h
  exact Bool.noConfusion h

/-- An element of GF256 from its coordinates. -/
theorem gf256_ext {x y : F256} (h0 : GF128.of x.c0 = GF128.of y.c0) (h1 : GF128.of x.c1 = GF128.of y.c1) :
    GF256.of x = GF256.of y := by
  cases x; cases y
  cases h0; cases h1
  rfl

/-- `ρ · (a, 0)`'s coordinates over F128: `(ρ₀·a, ρ₁·a)`. -/
theorem mul_ofF (ρ : F256) (a : F128) :
    GF128.of (ρ * ⟨a, 0⟩).c0 = GF128.of ρ.c0 * GF128.of a ∧
      GF128.of (ρ * ⟨a, 0⟩).c1 = GF128.of ρ.c1 * GF128.of a := by
  rw [F256.mul_def]
  refine ⟨?_, ?_⟩
  · show GF128.of (ρ.c0 * a + F256.xinv * (ρ.c1 * 0)) = _
    rw [← GF128.add_eq, ← GF128.mul_eq, ← GF128.mul_eq, ← GF128.mul_eq, ← GF128.zero_eq]
    ring
  · show GF128.of ((ρ.c0 + ρ.c1) * (a + 0) + ρ.c0 * a) = _
    rw [← GF128.add_eq, ← GF128.mul_eq, ← GF128.add_eq, ← GF128.add_eq, ← GF128.mul_eq, ← GF128.zero_eq]
    linear_combination (GF128.of ρ.c0 * GF128.of a) * GF128.two_eq_zero

/-- `x + y`'s coordinates. -/
theorem add_c (x y : F256) :
    GF128.of (x + y).c0 = GF128.of x.c0 + GF128.of y.c0 ∧ GF128.of (x + y).c1 = GF128.of x.c1 + GF128.of y.c1 := by
  rw [F256.add_def]
  exact ⟨(GF128.add_eq _ _).symm, (GF128.add_eq _ _).symm⟩

/-- **`(a, b) ↦ ρ₀·a + ρ₁·b` is a bijection `F128² → F256`** when `ρ₀, ρ₁` are independent over F128
(`ZkLigerito.rhoIndependent`: the determinant `ρ₀.c0·ρ₁.c1 + ρ₀.c1·ρ₁.c0` is nonzero). -/
theorem combo_bijective (ρ : F256 × F256) (h : Zk.rhoIndependent ρ = true) :
    Function.Bijective fun p : 𝔽 × 𝔽 => GF256.of (ρ.1 * ⟨p.1, 0⟩ + ρ.2 * ⟨p.2, 0⟩) := by
  set r00 := GF128.of ρ.1.c0
  set r01 := GF128.of ρ.1.c1
  set r10 := GF128.of ρ.2.c0
  set r11 := GF128.of ρ.2.c1
  have hD : r00 * r11 + r01 * r10 ≠ 0 := by
    have hz : (ρ.1.c0 * ρ.2.c1 + ρ.1.c1 * ρ.2.c0).isZero = false := by
      simpa [Zk.rhoIndependent] using h
    have := fOf_ne_zero hz
    simpa [fOf, ← GF128.add_eq, ← GF128.mul_eq, r00, r01, r10, r11] using this
  have hc : ∀ a b : F128,
      GF128.of (ρ.1 * ⟨a, 0⟩ + ρ.2 * ⟨b, 0⟩).c0 = r00 * GF128.of a + r10 * GF128.of b ∧
      GF128.of (ρ.1 * ⟨a, 0⟩ + ρ.2 * ⟨b, 0⟩).c1 = r01 * GF128.of a + r11 * GF128.of b := by
    intro a b
    obtain ⟨e0, e1⟩ := add_c (ρ.1 * ⟨a, 0⟩) (ρ.2 * ⟨b, 0⟩)
    obtain ⟨a0, a1⟩ := mul_ofF ρ.1 a
    obtain ⟨b0, b1⟩ := mul_ofF ρ.2 b
    exact ⟨by rw [e0, a0, b0], by rw [e1, a1, b1]⟩
  have two : (2 : 𝔽) = 0 := GF128.two_eq_zero
  refine ⟨fun p q hpq => ?_, fun y => ?_⟩
  · obtain ⟨p0, p1⟩ := hc p.1 p.2
    obtain ⟨q0, q1⟩ := hc q.1 q.2
    have h0 : r00 * GF128.of p.1 + r10 * GF128.of p.2 = r00 * GF128.of q.1 + r10 * GF128.of q.2 := by
      rw [← p0, ← q0]; exact congrArg (fun x : F256 => GF128.of x.c0) hpq
    have h1 : r01 * GF128.of p.1 + r11 * GF128.of p.2 = r01 * GF128.of q.1 + r11 * GF128.of q.2 := by
      rw [← p1, ← q1]; exact congrArg (fun x : F256 => GF128.of x.c1) hpq
    have ea : (r00 * r11 + r01 * r10) * (GF128.of p.1 - GF128.of q.1) = 0 := by
      linear_combination r11 * h0 - r10 * h1 + (r01 * r10 * (GF128.of p.1 - GF128.of q.1)) * two
    have eb : (r00 * r11 + r01 * r10) * (GF128.of p.2 - GF128.of q.2) = 0 := by
      linear_combination r00 * h1 - r01 * h0 + (r01 * r10 * (GF128.of p.2 - GF128.of q.2)) * two
    have ha := sub_eq_zero.mp ((mul_eq_zero.mp ea).resolve_left hD)
    have hb := sub_eq_zero.mp ((mul_eq_zero.mp eb).resolve_left hD)
    exact Prod.ext ha hb
  · have hDi : (r00 * r11 + r01 * r10) * (r00 * r11 + r01 * r10)⁻¹ = 1 := mul_inv_cancel₀ hD
    refine ⟨((r11 * GF128.of y.c0 + r10 * GF128.of y.c1) * (r00 * r11 + r01 * r10)⁻¹,
      (r01 * GF128.of y.c0 + r00 * GF128.of y.c1) * (r00 * r11 + r01 * r10)⁻¹), ?_⟩
    obtain ⟨e0, e1⟩ := hc ((r11 * GF128.of y.c0 + r10 * GF128.of y.c1) * (r00 * r11 + r01 * r10)⁻¹)
      ((r01 * GF128.of y.c0 + r00 * GF128.of y.c1) * (r00 * r11 + r01 * r10)⁻¹)
    show GF256.of _ = GF256.of (y : F256)
    refine gf256_ext ?_ ?_
    · rw [e0]
      show r00 * ((r11 * GF128.of y.c0 + r10 * GF128.of y.c1) * (r00 * r11 + r01 * r10)⁻¹) +
        r10 * ((r01 * GF128.of y.c0 + r00 * GF128.of y.c1) * (r00 * r11 + r01 * r10)⁻¹) = GF128.of y.c0
      linear_combination GF128.of y.c0 * hDi + (r00 * r10 * GF128.of y.c1 * (r00 * r11 + r01 * r10)⁻¹) * two
    · rw [e1]
      show r01 * ((r11 * GF128.of y.c0 + r10 * GF128.of y.c1) * (r00 * r11 + r01 * r10)⁻¹) +
        r11 * ((r01 * GF128.of y.c0 + r00 * GF128.of y.c1) * (r00 * r11 + r01 * r10)⁻¹) = GF128.of y.c1
      linear_combination GF128.of y.c1 * hDi + (r01 * r11 * GF128.of y.c0 * (r00 * r11 + r01 * r10)⁻¹) * two

/-- **`W^r` bijective** at a table reading `ρ` and the inclusion `embed`, from `Zk.rhoIndependent`. -/
theorem wr_bijective {W Hi VL : Type} {z : Sizes} (T : Coined GF256 W Hi VL z) (r : Fin 2) (ρ : F256 × F256)
    (h0 : T.ρ r 0 = GF256.of ρ.1) (h1 : T.ρ r 1 = GF256.of ρ.2)
    (he : ∀ a : F128, T.embed (fOf a) = GF256.of ⟨a, 0⟩) (h : Zk.rhoIndependent ρ = true) :
    Function.Bijective fun p : 𝔽 × 𝔽 => T.tab.Wr r p.1 p.2 := by
  have e : (fun p : 𝔽 × 𝔽 => T.tab.Wr r p.1 p.2) =
      fun p : 𝔽 × 𝔽 => GF256.of (ρ.1 * ⟨p.1, 0⟩ + ρ.2 * ⟨p.2, 0⟩) := by
    funext p
    show T.ρ r 0 * T.embed p.1 + T.ρ r 1 * T.embed p.2 = _
    rw [h0, h1, show T.embed p.1 = GF256.of ⟨p.1, 0⟩ from he p.1, show T.embed p.2 = GF256.of ⟨p.2, 0⟩ from he p.2,
      GF256.mul_eq, GF256.mul_eq, GF256.add_eq]
  rw [e]
  exact combo_bijective ρ h

/-- **The coin facts from `Admits`**: at a coin vector the honest prover doesn't refuse, a table that reads its coins
passes the rank check, has each `W^r` bijective and each `β ≠ 0`. -/
theorem coin_facts (C : CoinedSetup) (c : CoinsZK C.Rest) (hc : C.Admits c)
    (hβ : ∀ r, (C.T c).β r = fOf (c.beta r))
    (hρ : ∀ r, (C.T c).ρ r 0 = GF256.of (c.rho r).1 ∧ (C.T c).ρ r 1 = GF256.of (c.rho r).2)
    (hw : ∀ k x, (C.T c).eqw k x = fOf (eqWord (C.points c k) (C.maskWords x)))
    (he : ∀ a : F128, (C.T c).embed (fOf a) = GF256.of ⟨a, 0⟩) :
    (C.T c).tab.Rank ∧ (∀ r, Function.Bijective fun p : 𝔽 × 𝔽 => (C.T c).tab.Wr r p.1 p.2) ∧
      ∀ r, (C.T c).tab.β r ≠ 0 := by
  obtain ⟨hb, hm, hr, -⟩ := hc
  refine ⟨rank_of_maskRankOk _ _ _ hw hm, fun r => wr_bijective _ r _ (hρ r).1 (hρ r).2 he (hr r), fun r => ?_⟩
  show (C.T c).β r ≠ 0
  rw [hβ r]
  exact fOf_ne_zero (hb r)

end FlockSoundness.Discharge.Composed
