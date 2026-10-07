import Proofs.Core.TC.Ampere.Arith
import Proofs.Core.TC.Proofs.Group

/-!
# One Ampere group, the relation's side

`Verity.SecurityProofs.TC.Proofs.Group` for the Ampere/Ada parameters (`AP`): a term's alignment, and every column of
the finite path. The maximum exponent (`estar_eq`) and the output state (`out_poly`) do not depend on the parameters and
are reused.
-/

namespace Verity.TC.Proofs.Ampere

open Verity.TC Verity.TC.Spec Verity.TC.Relation

/-- `relation.params(AMPERE_BF16_M16N8K16)`. -/
def AP : Params := { W := 25, rho := 1, A := 26, B := 30, K := 35, Z := -132, n := 8 }

theorem params_ampere : params AMPERE_BF16_M16N8K16 = AP := by
  have : bitLength 8 = 4 := bitLength_eq 8 4 (by decide) (by decide) (by decide)
  simp [params, AMPERE_BF16_M16N8K16, AP, this]

theorem tPow_AP (d : Int) (pw ipw : Int) (h : tPow AP d = some (pw, ipw)) :
    0 ≤ d ∧ d < 512 ∧ pw = 2 ^ (min d.toNat 26) ∧ ipw = 2 ^ (26 - min d.toNat 26) := by
  unfold tPow at h
  split at h
  · simp only [Option.some.injEq, Prod.mk.injEq, AP] at h
    exact ⟨by omega, by omega, h.1.symm, h.2.symm⟩
  · cases h

theorem term_facts (align : Align) (Estar : Int) (i : Nat) (Pprev : Int) (x : TermIn) (t : TermCols)
    (h : termHolds AP align Estar i Pprev x t) (hs : 0 ≤ x.scaled)
    (htab : i ≠ 0 → x.scaled = Pprev * 2 ^ 10 ∧ 0 ≤ Pprev) :
    0 ≤ Estar - x.e ∧ Estar - x.e < 512 ∧ t.aligned - 2 * t.signed = contrib Estar x := by
  obtain ⟨hd, hpow, hal, hsg⟩ := h
  unfold LIN at hd
  unfold LOOKUP at hpow
  unfold MUL at hsg
  have hd' : t.d = Estar - x.e := by omega
  obtain ⟨h0, h512, hpw, hipw⟩ := tPow_AP _ _ _ hpow
  rw [hd'] at h0 h512 hpw hipw
  refine ⟨h0, h512, ?_⟩
  have hal' : t.aligned = x.scaled / 2 ^ (min (Estar - x.e).toNat 26) := by
    split at hal
    · obtain ⟨h1, _, h3, h4⟩ := hal
      unfold MUL at h1 h3
      unfold RANGE at h4
      rw [hpw] at h1
      rw [hipw] at h3
      exact (div_of_rem _ _ _ _ _ 26 (Nat.min_le_right _ _) h1 h3 h4.1 h4.2).1
    · rename_i hi
      have hi0 : i ≠ 0 := fun e => hi (Or.inl e)
      obtain ⟨hsc, hP0⟩ := htab hi0
      unfold LOOKUP tShift at hal
      simp only [AP] at hal
      split at hal
      · simp only [Option.some.injEq] at hal
        rw [← hal, hsc, hd', Nat.shiftLeft_eq, Nat.shiftRight_eq_div_pow, natCast_div_pow]
        congr 1
        rw [Int.natCast_mul, Int.toNat_of_nonneg hP0]
        simp
      · cases hal
  unfold contrib
  rw [← hal', ← hsg, Int.sub_mul, Int.one_mul, Int.mul_assoc]


/-- `e_out` for the total `T` and maximum exponent `E`. -/
def eOut (T E : Int) : Int := E + bitLength T.natAbs - 25

/-- `e_fin`: `e_out`, raised to the FP32 floor. -/
def eFin (T E : Int) : Int := max (eOut T E) (-126)

/-- `magW`: `|T|` in the 25-bit window. -/
def magWOf (T : Int) : Nat := T.natAbs * 2 ^ (30 - bitLength T.natAbs) / 2 ^ 5

/-- `mag24`: the window after the subnormal shift and the rescale. -/
def mag24Of (T E : Int) : Nat := magWOf T / 2 ^ (min (eFin T E - eOut T E + 1) 35).toNat


/-- The bit length through `T_norm` and the top-bit range check. -/
theorem norm_L (m L normpow z_t normB : Int) (hm0 : 0 ≤ m) (hnorm : LOOKUP (tNorm AP) L (normpow, z_t))
    (hnB : MUL m normpow normB) (hr : RANGE (normB - (1 - z_t) * 2 ^ (AP.B - 1)) (AP.B - 1)) :
    L = bitLength m.toNat ∧ z_t = (if m = 0 then 1 else 0) ∧ normB = ((m.toNat * 2 ^ (30 - bitLength m.toNat) : Nat) : Int) := by
  unfold LOOKUP tNorm at hnorm
  unfold MUL at hnB
  unfold RANGE at hr
  split at hnorm
  · rename_i hL
    simp only [AP] at hnorm hr hL
    simp only [Option.some.injEq, Prod.mk.injEq] at hnorm
    obtain ⟨hp, hz⟩ := hnorm
    subst hp hz hnB
    obtain ⟨n, rfl⟩ : ∃ n : Nat, m = n := ⟨m.toNat, (Int.toNat_of_nonneg hm0).symm⟩
    obtain ⟨l, rfl⟩ : ∃ l : Nat, L = l := ⟨L.toNat, (Int.toNat_of_nonneg hL.1).symm⟩
    simp only [Int.toNat_natCast] at *
    have hl30 : l ≤ 30 := by omega
    have hcast : ((n : Int) * 2 ^ (30 - l)) = ((n * 2 ^ (30 - l) : Nat) : Int) := by simp
    rw [hcast] at hr ⊢
    by_cases hl0 : l = 0
    · subst hl0
      simp only [ite_true, Nat.sub_zero] at hr ⊢
      have : n * 2 ^ 30 < 2 ^ 29 := by omega
      have hn : n = 0 := by
        rcases Nat.eq_zero_or_pos n with h | h
        · exact h
        · have := Nat.mul_le_mul_right (2 ^ 30) h; omega
      subst hn
      simp [bitLength]
    · have hl0' : ¬ ((l : Int) = 0) := by omega
      simp only [hl0', ite_false] at hr
      have hlo : 2 ^ 29 ≤ n * 2 ^ (30 - l) := by omega
      have hhi : n * 2 ^ (30 - l) < 2 ^ 30 := by omega
      have e31 : 2 ^ 29 = 2 ^ (l - 1) * 2 ^ (30 - l) := by rw [← Nat.pow_add]; congr 1; omega
      have e30 : 2 ^ 30 = 2 ^ l * 2 ^ (30 - l) := by rw [← Nat.pow_add]; congr 1; omega
      have h1 : 2 ^ (l - 1) ≤ n := by
        rw [e31] at hlo; exact Nat.le_of_mul_le_mul_right hlo (Nat.two_pow_pos _)
      have h2 : n < 2 ^ l := by
        rw [e30] at hhi; exact Nat.lt_of_mul_lt_mul_right hhi
      have hb := bitLength_eq n l (by omega) h1 h2
      have hn0 : n ≠ 0 := by have := Nat.two_pow_pos (l - 1); omega
      simp [hb, hn0, hl0]
  · cases hnorm

theorem tExp_AP (idx ef ps ips ov : Int) (h : tExp AP idx = some (ef, ps, ips, ov)) :
    0 ≤ idx ∧ idx < 512 ∧ ef = max (idx - 192) (-126) ∧
      ps = 2 ^ (min (ef - (idx - 192) + 1) 35).toNat ∧ ips = 2 ^ (35 - min (ef - (idx - 192) + 1) 35).toNat ∧
      ov = if ef > 127 then 1 else 0 := by
  unfold tExp at h
  split at h
  · rename_i hi
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨h1, h2, h3, h4⟩ := h
    subst h1 h2 h3 h4
    simp only [E_OFF, FP32_MIN_E, FP32_MAX_E, AP]
    refine ⟨hi.1, hi.2, ?_, ?_, ?_, ?_⟩ <;> first | rfl | trivial | (push_cast; rfl) | simp
  · cases h

/-- Every column of the finite path, as a function of the total and `E*`. -/
structure FinitePath (g : GroupCols) : Prop where
  mag_t : g.mag_t = (g.total.natAbs : Int)
  T_lt : g.total.natAbs < 2 ^ 30
  z_t : g.z_t = if g.total = 0 then 1 else 0
  neg_t : g.total ≠ 0 → g.neg_t = if g.total < 0 then 1 else 0
  neg01 : g.neg_t = 0 ∨ g.neg_t = 1
  e_fin : g.e_fin = eFin g.total g.Estar
  ovf : g.ovf = if eFin g.total g.Estar > 127 then 1 else 0
  mag24 : g.mag24 = (mag24Of g.total g.Estar : Int)
  z_out : g.z_out = if mag24Of g.total g.Estar = 0 then 1 else 0

theorem finite_path (g : GroupCols)
    (h5 : BOOL g.neg_t) (h6 : MUL (1 - 2 * g.neg_t) g.total g.mag_t) (h7 : RANGE g.mag_t AP.B)
    (h8 : LOOKUP (tNorm AP) g.L (g.normpow, g.z_t)) (h9 : MUL g.mag_t g.normpow g.normB)
    (h10 : RANGE (g.normB - (1 - g.z_t) * 2 ^ (AP.B - 1)) (AP.B - 1))
    (h11 : LIN (g.magW * 2 ^ (AP.B - AP.W) + g.r_n - g.normB)) (h12 : RANGE g.r_n (AP.B - AP.W))
    (h13 : LIN (g.e_out - (g.Estar + g.L - AP.W)))
    (h14 : LOOKUP (tExp AP) (g.e_out + E_OFF) (g.e_fin, g.pow_s, g.invpow_s, g.ovf))
    (h15 : MUL g.mag24 g.pow_s (g.magW - g.r_s)) (h16 : RANGE g.mag24 24)
    (h17 : MUL g.r_s g.invpow_s g.rsinv) (h18 : RANGE g.rsinv AP.K) (h19 : ISZERO g.mag24 24 g.z_out) :
    FinitePath g := by
  have hb := bool_of _ h5
  unfold MUL at h6 h9 h15 h17
  unfold RANGE at h7 h12 h16 h18
  unfold LIN at h11 h13
  unfold ISZERO RANGE at h19
  simp only [AP] at h7 h11 h12 h13 h18
  have hmag : g.mag_t = (g.total.natAbs : Int) := by
    rcases hb with h | h <;> rw [h] at h6 <;> omega
  have hmt : g.mag_t.toNat = g.total.natAbs := by omega
  obtain ⟨hL, hzt, hnB⟩ := norm_L g.mag_t g.L g.normpow g.z_t g.normB (by omega) h8 (by unfold MUL; exact h9) h10
  rw [hmt] at hL hnB
  have hW : g.magW = ((magWOf g.total : Nat) : Int) := by
    unfold magWOf
    rw [natCast_div_pow, ← hnB]
    have := (Int.ediv_emod_unique (b := (2 : Int) ^ 5) (by decide)).mpr
      ⟨(by omega : g.r_n + 2 ^ 5 * g.magW = g.normB), h12.1, by simpa using h12.2⟩
    exact this.1.symm
  have heo : g.e_out = eOut g.total g.Estar := by unfold eOut; omega
  unfold LOOKUP at h14
  obtain ⟨_, _, hef, hps, hips, hov⟩ := tExp_AP _ _ _ _ _ h14
  have hsh : g.e_out + E_OFF - 192 = g.e_out := by simp [E_OFF]
  rw [hsh] at hef hps hips
  have hef' : g.e_fin = eFin g.total g.Estar := by unfold eFin; rw [← heo, hef]
  have hk : (35 - min (g.e_fin - g.e_out + 1) 35).toNat = 35 - (min (g.e_fin - g.e_out + 1) 35).toNat := by omega
  rw [hps] at h15
  rw [hips, hk] at h17
  have hq := (div_of_rem _ _ _ _ _ 35 (by omega) h15 h17 h18.1 h18.2).1
  have hm24 : g.mag24 = ((mag24Of g.total g.Estar : Nat) : Int) := by
    rw [hq, hW, ← natCast_div_pow]
    unfold mag24Of
    rw [← hef', ← heo]
  refine ⟨hmag, by omega, ?_, ?_, hb, hef', ?_, hm24, ?_⟩
  · rw [hzt, hmag]; simp
  · intro hT
    rcases hb with h | h <;> rw [h] at h6 ⊢ <;> split <;> omega
  · rw [hov, hef']
  · rw [h19.2, hm24]; simp


end Verity.TC.Proofs.Ampere
