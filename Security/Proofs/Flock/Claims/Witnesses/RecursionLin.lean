import Proofs.Flock.Claims.Witnesses.Recursion
import Definitions.Claims.RecursionLin

/-!
# The linear recursion claims' witnesses

`KSLin`, `BitsLin` and `KSBoundLin` (`Definitions.Claims.RecursionLin`) fail at `KS`'s, `Bits`' and `KSBound`'s toy
records (`ksR`, `ksRAt` with `ksB`): the outer witnesses record no call, so `Fdev` and `B_I` find nothing
(`prFdev_noCalls`, `prBinner_noCalls`), and `Fdev`'s tail `β/(M+1)` is below `1/2` at `M = ⌈2β⌉` (`tail_lt_half`),
while the silent prover is always accepted on a statement with no witness and `κ < 1/2`. Their `meets` witnesses are
`KS`'s, `Bits`' and `KSBound`'s: the claims read the same decoders.
-/

namespace Claims.Witness.Recursion

open Claims Claims.Recursion

section Numbers

variable {Coin W Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- `Fdev` finds nothing where the outer witnesses record no call. -/
theorem fdev_pr_noCalls (H : Msg → Msg) (D : Recursion.Deploy Coin) (hc : ∀ x ws, D.calls x ws = [])
    (I : Recursion.Inner Coin W) (DO : Flock.Deploy Coin Col T) (decO : List (ℕ × Col) → List T) {Ω : Type}
    [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg) (M : ℕ) (hk : 0 < I.V.k (xI D x)) :
    (Fdev H D I DO decO P x M hk).Pr H = 0 := by
  unfold Finder.Pr
  rw [Pr_congr fun s => iff_false_intro ?_]
  · exact Pr_false
  · rintro ⟨u, v, h, -⟩
    simp only [Fdev, linkOut, hc, List.append_nil, findColl, reduceCtorEq] at h

/-- `B_I` finds nothing where the outer witnesses record no call. -/
theorem binner_pr_noCalls (H : Msg → Msg) (D : Recursion.Deploy Coin) (hc : ∀ x ws, D.calls x ws = [])
    (I : Recursion.Inner Coin W) (DO : Flock.Deploy Coin Col T) (decO : List (ℕ × Col) → List T) {Ω : Type}
    [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg) (hk : 0 < I.V.k (xI D x)) :
    (Binner H D I DO decO P x hk).Pr H = 0 := by
  unfold Finder.Pr
  rw [Pr_congr fun s => iff_false_intro ?_]
  · exact Pr_false
  · rintro ⟨u, v, h, -⟩
    simp only [Binner, hc, List.append_nil, findColl, reduceCtorEq] at h

theorem prFdev_noCalls (H : Msg → Msg) (D : Recursion.Deploy Coin) (hc : ∀ x ws, D.calls x ws = [])
    (I : Recursion.Inner Coin W) (DO : Flock.Deploy Coin Col T) (decO : List (ℕ × Col) → List T) {Ω : Type}
    [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg) (M : ℕ) : PrFdev H D I DO decO P x M = 0 := by
  unfold PrFdev
  split_ifs with hk
  · exact fdev_pr_noCalls H D hc I DO decO P x M hk
  · rfl

theorem prBinner_noCalls (H : Msg → Msg) (D : Recursion.Deploy Coin) (hc : ∀ x ws, D.calls x ws = [])
    (I : Recursion.Inner Coin W) (DO : Flock.Deploy Coin Col T) (decO : List (ℕ × Col) → List T) {Ω : Type}
    [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg) : PrBinner H D I DO decO P x = 0 := by
  unfold PrBinner
  split_ifs with hk
  · exact binner_pr_noCalls H D hc I DO decO P x hk
  · rfl

theorem fdev_expectedCR_noCalls (H : Msg → Msg) (D : Recursion.Deploy Coin) (hc : ∀ x ws, D.calls x ws = [])
    (I : Recursion.Inner Coin W) (DO : Flock.Deploy Coin Col T) (decO : List (ℕ × Col) → List T) {Ω : Type}
    [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg) (M : ℕ) (hk : 0 < I.V.k (xI D x)) :
    ExpectedCR H (Fdev H D I DO decO P x M hk) := by
  unfold ExpectedCR
  rw [fdev_pr_noCalls H D hc I DO decO P x M hk]
  exact div_nonneg (Ex_nonneg fun _ => Nat.cast_nonneg _) (by positivity)

theorem binner_expectedCR_noCalls (H : Msg → Msg) (D : Recursion.Deploy Coin) (hc : ∀ x ws, D.calls x ws = [])
    (I : Recursion.Inner Coin W) (DO : Flock.Deploy Coin Col T) (decO : List (ℕ × Col) → List T) {Ω : Type}
    [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg) (hk : 0 < I.V.k (xI D x)) :
    ExpectedCR H (Binner H D I DO decO P x hk) := by
  unfold ExpectedCR
  rw [binner_pr_noCalls H D hc I DO decO P x hk]
  exact div_nonneg (Ex_nonneg fun _ => Nat.cast_nonneg _) (by positivity)

/-- `Fdev`'s tail at `M = ⌈2β⌉` is below `1/2`. -/
theorem tail_lt_half (β : ℝ) : β * (1 / (((⌈2 * β⌉₊ : ℕ) : ℝ) + 1)) < 1 / 2 := by
  have hM : 2 * β ≤ ((⌈2 * β⌉₊ : ℕ) : ℝ) := Nat.le_ceil _
  have hp : (0 : ℝ) < ((⌈2 * β⌉₊ : ℕ) : ℝ) + 1 := by positivity
  rw [mul_one_div, div_lt_iff₀ hp]
  linarith

/-- **`KSLin` fails** at a record whose outer witnesses record no call, at a statement with no witness where the
silent prover is always accepted, `B` finds nothing and `κ < 1/2`. -/
theorem not_KSLin [DecidableEq Col] [Zero Col] {H : Msg → Msg} {D : Recursion.Deploy Coin}
    {I : Recursion.Inner Coin W} {DO : Flock.Deploy Coin Col T} (decO : List (ℕ × Col) → List T)
    (hdec : decOK DO decO) (x : Msg)
    (hbad : PrBad H D.V (Recursion.R D I) (Recursion.E H D I DO decO) x silentP = 1)
    (hB : PrB H D.V silentP x = 0) (hc : ∀ x ws, D.calls x ws = []) (hκ : Recursion.κ D I x < 1 / 2) :
    ¬ Recursion.KSLin H D I DO := by
  intro h
  have h1 := h decO hdec x Unit silentP ⌈2 * βdev D I x⌉₊
  rw [hbad, hB, prFdev_noCalls H D hc I DO decO silentP x, prBinner_noCalls H D hc I DO decO silentP x] at h1
  have := tail_lt_half (βdev D I x)
  simp only [mul_zero, Real.sqrt_zero, add_zero, zero_add] at h1
  linarith

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- `λ_lin = 1 > 0` at `ksR` and `accI`: `κ = 1/8`, `γ = 0`, `γ_BI = β = 1` and `c_Be = 4`. -/
theorem accI_lamLin_pos (x : Msg) : 0 < lamLin (ksR Coin) (accI Coin) x := by
  have hγBI : γBI (ksR Coin) (accI Coin) x = 1 := by simp [γBI, accI]
  have hβ : βdev (ksR Coin) (accI Coin) x = 1 := by simp [βdev, NI, accI, accV]
  have hc : (cBe (ksR Coin) x : ℝ) = 4 := by
    have : IsEmpty (Fin ((ksR Coin).nS x)) := Fin.isEmpty'
    unfold cBe
    rw [Finset.univ_eq_empty, Finset.sum_empty]
    norm_num
  have e4 : (4 : ℝ) = 2 ^ ((2 : ℕ) : ℝ) := by rw [Real.rpow_natCast]; norm_num
  have h4 : Real.logb 2 4 = 2 := by
    rw [e4, Real.logb_rpow (by norm_num) (by norm_num)]
    norm_num
  have e8 : (1 : ℝ) / (1 / 8) = 2 ^ ((3 : ℕ) : ℝ) := by rw [Real.rpow_natCast]; norm_num
  have h8 : Real.logb 2 ((1 : ℝ) / (1 / 8)) = 3 := by
    rw [e8, Real.logb_rpow (by norm_num) (by norm_num)]
    norm_num
  have hκ : (accI Coin).κ [] = 1 / 8 := rfl
  unfold lamLin
  rw [ksR_κ, ksR_γ, Real.logb_zero, hγBI, hβ, hc, one_mul, h4, hκ, h8, sub_pos]
  refine lt_min (lt_min (lt_min ?_ ?_) ?_) ?_ <;> norm_num

end Numbers

/-! ## The claims -/

section Claims

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **The recursion's `KSLin` fails** at `ksR`, `accI` and `Flock.ksD` outside. -/
theorem KSLin_fails (H : Msg → Msg) : ¬ Recursion.KSLin H (ksR Coin) (accI Coin) (Flock.ksD Coin) :=
  not_KSLin (fun _ => [()]) (Flock.listWithin_unit _ _ _) [] (ksR_prBad H _ _ [] (ksR_noR _ [])) (ksR_prB H [])
    (fun _ _ => rfl) (by rw [ksR_κ]; norm_num [accI])

/-- **The recursion's `KSBoundLin` fails** at `ksRAt`, `ksB` inside and `Flock.ksD` outside: the bound record's `κ` is
`Flock.ksD`'s. -/
theorem KSBoundLin_fails (H : Msg → Msg) : ¬ KSBoundLin H (ksRAt Coin) (ksB Coin) (Flock.ksD Coin) := by
  intro h
  have h' : Recursion.KSLin H (ksRV Coin (ksB Coin).IVStar) (Inner.ofFlock (ksB Coin).deploy fun _ => [()])
      (Flock.ksD Coin) :=
    h (fun _ => [()]) (Flock.listWithin_unit _ _ _)
  refine not_KSLin (D := ksRV Coin (ksB Coin).IVStar) (I := Inner.ofFlock (ksB Coin).deploy fun _ => [()])
    (fun _ => [()]) (Flock.listWithin_unit _ _ _) [] (ksRV_prBad H _ _ _ [] (ksRV_noR _ _ []))
    (ksRV_prB H _ []) (fun _ _ => rfl) ?_ h'
  rw [ksRV_κ]
  exact Flock.ksD_κ (Coin := Coin) []

/-- **The recursion's `BitsLin` fails** at `ksR`, `accI` and `Flock.ksD` outside, where `λ_lin = 1`. -/
theorem BitsLin_fails (H : Msg → Msg) : ¬ Recursion.BitsLin H (ksR Coin) (accI Coin) (Flock.ksD Coin) := fun h =>
  bits_bound_fails H _ _ _ (ksR_noR _ []) (accI_lamLin_pos []) <|
    h (fun _ => [()]) (Flock.listWithin_unit _ _ _) [] Unit silentP 1 le_rfl (calls_silentP_accV H [])
      (strictBirthday_B_accV H silentP []) (binner_expectedCR_noCalls H (ksR Coin) (fun _ _ => rfl) _ _ _ silentP [])
      (fun M => fdev_expectedCR_noCalls H (ksR Coin) (fun _ _ => rfl) _ _ _ silentP [] M)

end Claims

end Claims.Witness.Recursion
