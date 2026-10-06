import Proofs.Core.TC.Proofs.Norm

/-!
# One group, the relation's side: what the predicates force

For the Hopper parameters (`HP`), `term_facts` reads a term's alignment off its predicates, `estar_eq` the group's
maximum exponent, `finite_path` every column of the finite path, and `out_poly` the output state as a polynomial in the
group's flags. None of this looks at the semantics.
-/

namespace Verity.TC.Proofs

open Verity.TC Verity.TC.Spec Verity.TC.Relation

/-- `relation.params(HOPPER_BF16_WGMMA_K16)`. -/
def HP : Params := { W := 26, rho := 2, A := 27, B := 32, K := 36, Z := -133, n := 16 }

theorem params_hopper : params HOPPER_BF16_WGMMA_K16 = HP := by
  have : bitLength 16 = 5 := bitLength_eq 16 5 (by decide) (by decide) (by decide)
  simp [params, HOPPER_BF16_WGMMA_K16, HP, this]

theorem tPow_HP (d : Int) (pw ipw : Int) (h : tPow HP d = some (pw, ipw)) :
    0 ≤ d ∧ d < 512 ∧ pw = 2 ^ (min d.toNat 27) ∧ ipw = 2 ^ (27 - min d.toNat 27) := by
  unfold tPow at h
  split at h
  · simp only [Option.some.injEq, Prod.mk.injEq, HP] at h
    exact ⟨by omega, by omega, h.1.symm, h.2.symm⟩
  · cases h

theorem term_facts (align : Align) (Estar : Int) (i : Nat) (Pprev : Int) (x : TermIn) (t : TermCols)
    (h : termHolds HP align Estar i Pprev x t) (hs : 0 ≤ x.scaled)
    (htab : i ≠ 0 → x.scaled = Pprev * 2 ^ 11 ∧ 0 ≤ Pprev) :
    0 ≤ Estar - x.e ∧ Estar - x.e < 512 ∧ t.aligned - 2 * t.signed = contrib Estar x := by
  obtain ⟨hd, hpow, hal, hsg⟩ := h
  unfold LIN at hd
  unfold LOOKUP at hpow
  unfold MUL at hsg
  have hd' : t.d = Estar - x.e := by omega
  obtain ⟨h0, h512, hpw, hipw⟩ := tPow_HP _ _ _ hpow
  rw [hd'] at h0 h512 hpw hipw
  refine ⟨h0, h512, ?_⟩
  have hal' : t.aligned = x.scaled / 2 ^ (min (Estar - x.e).toNat 27) := by
    split at hal
    · obtain ⟨h1, _, h3, h4⟩ := hal
      unfold MUL at h1 h3
      unfold RANGE at h4
      rw [hpw] at h1
      rw [hipw] at h3
      exact (div_of_rem _ _ _ _ _ 27 (Nat.min_le_right _ _) h1 h3 h4.1 h4.2).1
    · rename_i hi
      have hi0 : i ≠ 0 := fun e => hi (Or.inl e)
      obtain ⟨hsc, hP0⟩ := htab hi0
      unfold LOOKUP tShift at hal
      simp only [HP] at hal
      split at hal
      · simp only [Option.some.injEq] at hal
        rw [← hal, hsc, hd', Nat.shiftLeft_eq, Nat.shiftRight_eq_div_pow, natCast_div_pow]
        congr 1
        rw [Int.natCast_mul, Int.toNat_of_nonneg hP0]
        simp
      · cases hal
  unfold contrib
  rw [← hal', ← hsg, Int.sub_mul, Int.one_mul, Int.mul_assoc]

/-- The product of `c` and every `E - x.e` is zero exactly when a factor is. -/
theorem chain_zero (E : Int) : ∀ (xs : List TermIn) (c : Int),
    xs.foldl (fun c x => c * (E - x.e)) c = 0 → c = 0 ∨ ∃ x ∈ xs, E = x.e
  | [], c, h => Or.inl h
  | x :: xs, c, h => by
    simp only [List.foldl_cons] at h
    rcases chain_zero E xs _ h with h | ⟨y, hy, he⟩
    · rcases Int.mul_eq_zero.mp h with h | h
      · exact Or.inl h
      · exact Or.inr ⟨x, List.mem_cons_self .., by omega⟩
    · exact Or.inr ⟨y, List.mem_cons_of_mem _ hy, he⟩

/-- The group's `E*` is the maximum of its rows' exponents and the floor. -/
theorem estar_eq (Z E : Int) (xs : List TermIn) (hge : ∀ x ∈ xs, x.e ≤ E) (hZ : ∃ x ∈ xs, Z ≤ x.e)
    (hchain : xs.foldl (fun c x => c * (E - x.e)) (E - Z) = 0) : E = maxE Z xs := by
  obtain ⟨hm1, hm2⟩ := le_maxE xs Z
  have hEZ : Z ≤ E := by obtain ⟨x, hx, h⟩ := hZ; have := hge x hx; omega
  have h1 : maxE Z xs ≤ E := by
    rcases maxE_attained xs Z with h | ⟨x, hx, h⟩
    · omega
    · rw [h]; exact hge x hx
  have h2 : E ≤ maxE Z xs := by
    rcases chain_zero E xs _ hchain with h | ⟨x, hx, h⟩
    · omega
    · rw [h]; exact hm2 x hx
  omega

/-- `e_out` for the total `T` and maximum exponent `E`. -/
def eOut (T E : Int) : Int := E + bitLength T.natAbs - 26

/-- `e_fin`: `e_out`, raised to the FP32 floor. -/
def eFin (T E : Int) : Int := max (eOut T E) (-126)

/-- `magW`: `|T|` in the 26-bit window. -/
def magWOf (T : Int) : Nat := T.natAbs * 2 ^ (32 - bitLength T.natAbs) / 2 ^ 6

/-- `mag24`: the window after the subnormal shift and the rescale. -/
def mag24Of (T E : Int) : Nat := magWOf T / 2 ^ (min (eFin T E - eOut T E + 2) 36).toNat

theorem bool_of (x : Int) (h : BOOL x) : x = 0 ∨ x = 1 := by
  unfold BOOL MUL at h
  rcases Int.mul_eq_zero.mp h with h | h
  · exact Or.inl h
  · exact Or.inr (by omega)

/-- The bit length through `T_norm` and the top-bit range check. -/
theorem norm_L (m L normpow z_t normB : Int) (hm0 : 0 ≤ m) (hnorm : LOOKUP (tNorm HP) L (normpow, z_t))
    (hnB : MUL m normpow normB) (hr : RANGE (normB - (1 - z_t) * 2 ^ (HP.B - 1)) (HP.B - 1)) :
    L = bitLength m.toNat ∧ z_t = (if m = 0 then 1 else 0) ∧ normB = ((m.toNat * 2 ^ (32 - bitLength m.toNat) : Nat) : Int) := by
  unfold LOOKUP tNorm at hnorm
  unfold MUL at hnB
  unfold RANGE at hr
  split at hnorm
  · rename_i hL
    simp only [HP] at hnorm hr hL
    simp only [Option.some.injEq, Prod.mk.injEq] at hnorm
    obtain ⟨hp, hz⟩ := hnorm
    subst hp hz hnB
    obtain ⟨n, rfl⟩ : ∃ n : Nat, m = n := ⟨m.toNat, (Int.toNat_of_nonneg hm0).symm⟩
    obtain ⟨l, rfl⟩ : ∃ l : Nat, L = l := ⟨L.toNat, (Int.toNat_of_nonneg hL.1).symm⟩
    simp only [Int.toNat_natCast] at *
    have hl32 : l ≤ 32 := by omega
    have hcast : ((n : Int) * 2 ^ (32 - l)) = ((n * 2 ^ (32 - l) : Nat) : Int) := by simp
    rw [hcast] at hr ⊢
    by_cases hl0 : l = 0
    · subst hl0
      simp only [ite_true, Nat.sub_zero] at hr ⊢
      have : n * 2 ^ 32 < 2 ^ 31 := by omega
      have hn : n = 0 := by
        rcases Nat.eq_zero_or_pos n with h | h
        · exact h
        · have := Nat.mul_le_mul_right (2 ^ 32) h; omega
      subst hn
      simp [bitLength]
    · have hl0' : ¬ ((l : Int) = 0) := by omega
      simp only [hl0', ite_false] at hr
      have hlo : 2 ^ 31 ≤ n * 2 ^ (32 - l) := by omega
      have hhi : n * 2 ^ (32 - l) < 2 ^ 32 := by omega
      have e31 : 2 ^ 31 = 2 ^ (l - 1) * 2 ^ (32 - l) := by rw [← Nat.pow_add]; congr 1; omega
      have e32 : 2 ^ 32 = 2 ^ l * 2 ^ (32 - l) := by rw [← Nat.pow_add]; congr 1; omega
      have h1 : 2 ^ (l - 1) ≤ n := by
        rw [e31] at hlo; exact Nat.le_of_mul_le_mul_right hlo (Nat.two_pow_pos _)
      have h2 : n < 2 ^ l := by
        rw [e32] at hhi; exact Nat.lt_of_mul_lt_mul_right hhi
      have hb := bitLength_eq n l (by omega) h1 h2
      have hn0 : n ≠ 0 := by have := Nat.two_pow_pos (l - 1); omega
      simp [hb, hn0, hl0]
  · cases hnorm

theorem tExp_HP (idx ef ps ips ov : Int) (h : tExp HP idx = some (ef, ps, ips, ov)) :
    0 ≤ idx ∧ idx < 512 ∧ ef = max (idx - 192) (-126) ∧
      ps = 2 ^ (min (ef - (idx - 192) + 2) 36).toNat ∧ ips = 2 ^ (36 - min (ef - (idx - 192) + 2) 36).toNat ∧
      ov = if ef > 127 then 1 else 0 := by
  unfold tExp at h
  split at h
  · rename_i hi
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨h1, h2, h3, h4⟩ := h
    subst h1 h2 h3 h4
    simp only [E_OFF, FP32_MIN_E, FP32_MAX_E, HP]
    refine ⟨hi.1, hi.2, ?_, ?_, ?_, ?_⟩ <;> first | rfl | trivial | (push_cast; rfl) | simp
  · cases h

/-- Every column of the finite path, as a function of the total and `E*`. -/
structure FinitePath (g : GroupCols) : Prop where
  mag_t : g.mag_t = (g.total.natAbs : Int)
  T_lt : g.total.natAbs < 2 ^ 32
  z_t : g.z_t = if g.total = 0 then 1 else 0
  neg_t : g.total ≠ 0 → g.neg_t = if g.total < 0 then 1 else 0
  neg01 : g.neg_t = 0 ∨ g.neg_t = 1
  e_fin : g.e_fin = eFin g.total g.Estar
  ovf : g.ovf = if eFin g.total g.Estar > 127 then 1 else 0
  mag24 : g.mag24 = (mag24Of g.total g.Estar : Int)
  z_out : g.z_out = if mag24Of g.total g.Estar = 0 then 1 else 0

theorem finite_path (g : GroupCols)
    (h5 : BOOL g.neg_t) (h6 : MUL (1 - 2 * g.neg_t) g.total g.mag_t) (h7 : RANGE g.mag_t HP.B)
    (h8 : LOOKUP (tNorm HP) g.L (g.normpow, g.z_t)) (h9 : MUL g.mag_t g.normpow g.normB)
    (h10 : RANGE (g.normB - (1 - g.z_t) * 2 ^ (HP.B - 1)) (HP.B - 1))
    (h11 : LIN (g.magW * 2 ^ (HP.B - HP.W) + g.r_n - g.normB)) (h12 : RANGE g.r_n (HP.B - HP.W))
    (h13 : LIN (g.e_out - (g.Estar + g.L - HP.W)))
    (h14 : LOOKUP (tExp HP) (g.e_out + E_OFF) (g.e_fin, g.pow_s, g.invpow_s, g.ovf))
    (h15 : MUL g.mag24 g.pow_s (g.magW - g.r_s)) (h16 : RANGE g.mag24 24)
    (h17 : MUL g.r_s g.invpow_s g.rsinv) (h18 : RANGE g.rsinv HP.K) (h19 : ISZERO g.mag24 24 g.z_out) :
    FinitePath g := by
  have hb := bool_of _ h5
  unfold MUL at h6 h9 h15 h17
  unfold RANGE at h7 h12 h16 h18
  unfold LIN at h11 h13
  unfold ISZERO RANGE at h19
  simp only [HP] at h7 h11 h12 h13 h18
  have hmag : g.mag_t = (g.total.natAbs : Int) := by
    rcases hb with h | h <;> rw [h] at h6 <;> omega
  have hmt : g.mag_t.toNat = g.total.natAbs := by omega
  obtain ⟨hL, hzt, hnB⟩ := norm_L g.mag_t g.L g.normpow g.z_t g.normB (by omega) h8 (by unfold MUL; exact h9) h10
  rw [hmt] at hL hnB
  have hW : g.magW = ((magWOf g.total : Nat) : Int) := by
    unfold magWOf
    rw [natCast_div_pow, ← hnB]
    have := (Int.ediv_emod_unique (b := (2 : Int) ^ 6) (by decide)).mpr
      ⟨(by omega : g.r_n + 2 ^ 6 * g.magW = g.normB), h12.1, by simpa using h12.2⟩
    exact this.1.symm
  have heo : g.e_out = eOut g.total g.Estar := by unfold eOut; omega
  unfold LOOKUP at h14
  obtain ⟨_, _, hef, hps, hips, hov⟩ := tExp_HP _ _ _ _ _ h14
  have hsh : g.e_out + E_OFF - 192 = g.e_out := by simp [E_OFF]
  rw [hsh] at hef hps hips
  have hef' : g.e_fin = eFin g.total g.Estar := by unfold eFin; rw [← heo, hef]
  have hk : (36 - min (g.e_fin - g.e_out + 2) 36).toNat = 36 - (min (g.e_fin - g.e_out + 2) 36).toNat := by omega
  rw [hps] at h15
  rw [hips, hk] at h17
  have hq := (div_of_rem _ _ _ _ _ 36 (by omega) h15 h17 h18.1 h18.2).1
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

/-- The output state as the relation computes it from the group's flags, `e_fin` and `mag24`. -/
def outPoly (Z N Pf Ng O zt zo nt ef m : Int) : State :=
  let ovf_eff := O * (1 - zt)
  let any_inf := Pf + Ng - Pf * Ng
  let nan_out := N + (1 - N) * (Pf * Ng)
  let inf_out := (1 - nan_out) * (any_inf + (1 - any_inf) * ovf_eff)
  let neg_inf := any_inf * Ng + (1 - any_inf) * nt
  let fin := 1 - nan_out - inf_out
  let neg_fin := nt * (1 - zo)
  let e_state := ef + zo * (Z - ef)
  ⟨nan_out, inf_out, inf_out * neg_inf + fin * neg_fin, Z + fin * (e_state - Z), fin * m⟩

theorem out_poly (Z : Int) (g : GroupCols)
    (h28 : MUL g.ovf (1 - g.z_t) g.ovf_eff)
    (h30 : LIN (g.any_inf - (g.any_pinf + g.any_ninf - g.any_pinf * g.any_ninf)))
    (h31 : MUL (1 - g.any_nan) (g.any_pinf * g.any_ninf) (g.nan_out - g.any_nan))
    (h33 : MUL (1 - g.nan_out) (g.any_inf + (1 - g.any_inf) * g.ovf_eff) g.inf_out)
    (h35 : MUL (1 - g.any_inf) g.neg_t (g.neg_inf - g.any_inf * g.any_ninf))
    (h36 : LIN (g.fin - (1 - g.nan_out - g.inf_out)))
    (h37 : MUL g.neg_t (1 - g.z_out) g.neg_fin)
    (h38 : MUL g.z_out (Z - g.e_fin) (g.e_state - g.e_fin))
    (h40 : MUL g.fin g.neg_fin (g.out.neg - g.inf_out * g.neg_inf))
    (h41 : MUL g.fin (g.e_state - Z) (g.out.e - Z))
    (h42 : MUL g.fin g.mag24 g.out.M)
    (h43 : LIN (g.out.nan - g.nan_out)) (h44 : LIN (g.out.inf - g.inf_out)) :
    g.out = outPoly Z g.any_nan g.any_pinf g.any_ninf g.ovf g.z_t g.z_out g.neg_t g.e_fin g.mag24 := by
  unfold MUL at *
  unfold LIN at *
  have e30 : g.any_inf = g.any_pinf + g.any_ninf - g.any_pinf * g.any_ninf := by omega
  have e31 : g.nan_out = g.any_nan + (1 - g.any_nan) * (g.any_pinf * g.any_ninf) := by omega
  have e35 : g.neg_inf = g.any_inf * g.any_ninf + (1 - g.any_inf) * g.neg_t := by omega
  have e36 : g.fin = 1 - g.nan_out - g.inf_out := by omega
  have e38 : g.e_state = g.e_fin + g.z_out * (Z - g.e_fin) := by omega
  generalize g.out = o at *
  cases o
  dsimp only at h40 h41 h42 h43 h44
  simp only [outPoly, State.mk.injEq]
  rw [h28, ← e30, ← e31, h33, ← e35, ← e36, h37, ← e38]
  refine ⟨by omega, by omega, by omega, by omega, h42.symm⟩

end Verity.TC.Proofs
