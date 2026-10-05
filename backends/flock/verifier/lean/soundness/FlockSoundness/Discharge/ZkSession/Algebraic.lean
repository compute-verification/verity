import FlockSoundness.Discharge.ZkSession.Level0
import FlockSoundness.OpeningPhase
import FlockSoundness.Game.Union
import FlockSoundness.Protocol.Discharge.ZkSession.Algebraic

/-!
# M1's masked algebraic phase (live/PROTOCOL.md §1, §2, §4)

Under `--zk` the prover masks every value the algebraic phase sends (`v' = v + h`, one pad each, §2): zerocheck's
`round1_ab`, `round1_c`, the residual rounds' `(G(1), G(∞))`, `final_a` and `final_b`; lincheck's rounds' `(q(1), q(∞))`
and `z_partial`; and `final_c`. The verifier replays zerocheck and lincheck on the same coins with every
message-dependent quantity an affine form in the pads (`LinF`), and each check it can no longer make becomes a row
(§4): zerocheck c_eval, zerocheck final a·b, lincheck final, ring switch ab and ring switch c. The opening runs as in the
clear on the ab and c claims' claim-check values (`openingZK`); the ring switches' `s_hat_v` are sent unmasked (§6 is
what hides them).

**Where `final_c'` comes.** The live verifier never absorbs `final_c'`, and `final_c'` is the constant of two rows, so a
prover that picks it after the inner proof's batching coins can cancel the batched constraint. Here it is received at
the end of the phase, before the inner proof (`restZK`): the protocol as Lean defines it absorbs `final_c'` before the
inner proof's label, which the live verifier must also do for this statement to describe it.

**The coupling** (VEIL Lemma 4.3): at every pad vector `h` whose product pad is the product (`h_ab = h_fa·h_fb`), the
masked rep (`repMaskedZK`) ends with a true verdict and all seven rows vanishing at `h` with probability at most the
clear rep's (`Model.repPad`) acceptance probability (`repMaskedZK_couple`). Unmasking each message at `h` turns a masked
strategy into a clear one; the forms evaluate at `h` to the clear values, and a row vanishing at `h` is the clear check it
replaces (characteristic 2).
-/

namespace FlockSoundness.Discharge.ZkSession

open Game Model Accounting
open scoped ENNReal

variable {F K : Type} [Field F] [Field K]

/-! ## Forms over `F` -/

/-- The form's value at the pads `h`. -/
def LinF.eval {np : ℕ} (h : Fin np → F) : LinF F np →ₗ[F] F where
  toFun ℓ := ∑ i, ℓ.1 i * h i + ℓ.2
  map_add' a b := by
    simp only [Prod.fst_add, Prod.snd_add, Pi.add_apply, add_mul, Finset.sum_add_distrib]
    ring
  map_smul' c a := by
    simp only [Prod.smul_fst, Prod.smul_snd, Pi.smul_apply, smul_eq_mul, RingHom.id_apply, mul_add,
      Finset.mul_sum, mul_assoc]

theorem LinF.eval_toLin {np : ℕ} (ℓ : LinF F np) (h : Fin np → F) : ℓ.toLin.eval h = LinF.eval h ℓ := rfl

section Eval

variable {np : ℕ} (h : Fin np → F)

@[simp] theorem eval_cF (x : F) : LinF.eval h (cF x) = x := by
  simp [LinF.eval, cF]

@[simp] theorem eval_pF (i : Fin np) : LinF.eval h (pF i) = h i := by
  simp only [LinF.eval, pF, LinearMap.coe_mk, AddHom.coe_mk, add_zero]
  rw [Finset.sum_eq_single i (fun j _ hj => by rw [Pi.single_eq_of_ne hj, zero_mul])
    (fun hi => absurd (Finset.mem_univ i) hi), Pi.single_eq_same, one_mul]

@[simp] theorem eval_mskF (i : Fin np) (v : F) : LinF.eval h (mskF i v) = v + h i := by
  simp [mskF]

end Eval

/-- In characteristic 2 a sum vanishes exactly when its terms are equal (§4 writes each check `lhs = rhs` as
`lhs + rhs = 0`). -/
theorem add_eq_zero_iff_c2 [CharP F 2] (x y : F) : x + y = 0 ↔ x = y := by
  rw [add_eq_zero_iff_eq_neg, CharTwo.neg_eq]

/-! ## Rows -/

/-- The event the coupling bounds: a true verdict and every row vanishing at `h`. -/
def Vanish {np n : ℕ} (h : Fin np → F) (o : (Fin n → Lin F np) × Bool) : Prop :=
  o.2 = true ∧ ∀ k, (o.1 k).eval h = 0

theorem vanish_consRow {np n : ℕ} (h : Fin np → F) (row : Lin F np) (o : (Fin n → Lin F np) × Bool) :
    Vanish h (consRow row o) ↔ row.eval h = 0 ∧ Vanish h o := by
  simp only [Vanish, consRow, Fin.forall_fin_succ, Fin.cons_zero, Fin.cons_succ]
  tauto

open Classical in
/-- A row put in front: where it does not vanish at `h`, nothing does. -/
theorem value_consRow {np n : ℕ} (h : Fin np → F) (row : Lin F np) (g : Game ((Fin n → Lin F np) × Bool)) :
    value (Vanish h) (g.map (consRow row)) ≤ if row.eval h = 0 then value (Vanish h) g else 0 := by
  rw [value_map]
  split_ifs with hr
  · exact (value_congr (fun o => by rw [vanish_consRow]; exact ⟨fun h' => h'.2, fun h' => ⟨hr, h'⟩⟩) g).le
  · rw [value_congr (q := fun _ => False) (fun o => by rw [vanish_consRow]; exact ⟨fun h' => hr h'.1, False.elim⟩),
      value_false]

/-! ## Zerocheck -/

theorem eval_residualStepL {np : ℕ} (h : Fin np → F) (req ρ : F) (claim g₁ gInf : LinF F np) :
    LinF.eval h (residualStepL req ρ claim g₁ gInf) =
      residualStep req (LinF.eval h claim) (LinF.eval h g₁) (LinF.eval h gInf) ρ := by
  simp only [residualStepL, residualStep, map_add, map_smul, smul_eq_mul]
  ring

variable [Fintype F] [DecidableEq F]

omit [DecidableEq F] in
theorem residual_couple {np : ℕ} (P : PadsA np) (h : Fin np → F) {α β : Type} (p : α → Prop) (p' : β → Prop) :
    ∀ (n j : ℕ) (req : Fin n → F) (cl : LinF F np) (c : F) (f : (Fin n → F) × LinF F np → Game α)
      (f' : (Fin n → F) × F → Game β),
    LinF.eval h cl = c → (∀ ρs cl', value p (f (ρs, cl')) ≤ value p' (f' (ρs, LinF.eval h cl'))) →
    value p ((residualRoundsZK P n j req cl).bind f) ≤ value p' ((residualRounds n req c).bind f')
  | 0, _, _, cl, c, f, f', hc, hf => by
      simp only [residualRoundsZK, residualRounds, pure_eq, bind_ret]
      rw [← hc]
      exact hf _ _
  | n + 1, j, req, cl, c, f, f', hc, hf => by
      simp only [residualRoundsZK, residualRounds, bind_eq, pure_eq, bind_assoc, bind_ret]
      refine value_recv_le fun g => ⟨(g.1 + h (P.ml j 0), g.2 + h (P.ml j 1)), ?_⟩
      refine value_bind_mono _ fun ρ => ?_
      refine residual_couple P h p p' n (j + 1) _ _ _ _ _ ?_ fun ρs cl' => hf _ _
      rw [eval_residualStepL, hc, eval_mskF, eval_mskF]

/-- The clear outputs at the pads `h`. -/
def ZcZK.at {np m : ℕ} (h : Fin np → F) (zc : ZcZK F np m) : ZcOut F m :=
  ⟨zc.z, zc.ρ, zc.r, LinF.eval h zc.va, LinF.eval h zc.vb, LinF.eval h zc.vc⟩

/-- **Zerocheck couples**: at pads with `h_ab = h_fa·h_fb`, the masked zerocheck followed by any masked continuation
that puts the row in front is bounded by the clear zerocheck followed by the clear continuation at the unmasked
outputs. -/
theorem zerocheckZK_couple [CharP F 2] (A : Arith F K) (m : ℕ) {np : ℕ} (P : PadsA np) (tr : Triple np)
    (h : Fin np → F) (htr : h tr.ab = h tr.fa * h tr.fb) {n : ℕ}
    (k : ZcZK F np m → Game ((Fin n → Lin F np) × Bool)) (k' : Option (ZcOut F m) → Game Bool)
    (hk : ∀ zc, value (Vanish h) (k zc) ≤ value (· = true) (k' (some (zc.at h)))) :
    value (Vanish h) ((zerocheckZK A m P tr).bind fun o => (k o.1).map (consRow o.2.toLin)) ≤
      value (· = true) ((zerocheck A m).bind k') := by
  simp only [zerocheckZK, zerocheck, bind_eq, pure_eq, bind_assoc, bind_ret]
  refine value_bind_mono _ fun _ => ?_
  refine value_bind_mono _ fun rs => ?_
  refine value_recv_le fun msg => ⟨(fun i => msg.1 i + h (P.r1ab i), fun i => msg.2 i + h (P.r1c i)), ?_⟩
  refine value_bind_mono _ fun z => ?_
  refine residual_couple P h _ _ _ 0 _ _ _ _ _ ?_ fun ρs cl => ?_
  · simp only [map_add, map_sum, map_smul, smul_eq_mul, eval_mskF]
  refine value_recv_le fun v => ⟨(v.1 + h tr.fa, v.2 + h tr.fb), ?_⟩
  refine (value_consRow h _ _).trans ?_
  split_ifs with hrow hc
  · have hz : ZcZK.at h (⟨z, ρs, zcVector A rs, mskF tr.fa v.1, mskF tr.fb v.2,
        ∑ i, A.lagΛ z i • mskF (P.r1c i) (msg.2 i)⟩ : ZcZK F np m) =
        ⟨z, ρs, zcVector A rs, v.1 + h tr.fa, v.2 + h tr.fb, ∑ i, A.lagΛ z i * (msg.2 i + h (P.r1c i))⟩ := by
      simp [ZcZK.at, map_sum, map_smul, smul_eq_mul]
    exact (hk _).trans (by rw [hz])
  · exfalso
    apply hc
    rw [LinF.eval_toLin] at hrow
    simp only [map_add, map_smul, smul_eq_mul, eval_cF, eval_pF] at hrow
    rw [(add_eq_zero_iff_c2 _ _).1 hrow, htr]
    ring
  · exact zero_le
  · exact zero_le

/-! ## Lincheck -/

omit [DecidableEq F] in
theorem lincheckRounds_couple {np : ℕ} (P : PadsA np) (h : Fin np → F) {α β : Type} (p : α → Prop)
    (p' : β → Prop) :
    ∀ (n j : ℕ) (cl : LinF F np) (c : F) (f : (Fin n → F) × LinF F np → Game α) (f' : (Fin n → F) × F → Game β),
    LinF.eval h cl = c → (∀ ts cl', value p (f (ts, cl')) ≤ value p' (f' (ts, LinF.eval h cl'))) →
    value p ((lincheckRoundsZK P n j cl).bind f) ≤ value p' ((lincheckRounds n c).bind f')
  | 0, _, cl, c, f, f', hc, hf => by
      simp only [lincheckRoundsZK, lincheckRounds, pure_eq, bind_ret]
      rw [← hc]
      exact hf _ _
  | n + 1, j, cl, c, f, f', hc, hf => by
      simp only [lincheckRoundsZK, lincheckRounds, bind_eq, pure_eq, bind_assoc, bind_ret]
      refine value_recv_le fun q => ⟨(q.1 + h (P.lc j 0), q.2 + h (P.lc j 1)), ?_⟩
      refine value_bind_mono _ fun t => ?_
      refine lincheckRounds_couple P h p p' n (j + 1) _ _ _ _ ?_ fun ts cl' => hf _ _
      simp only [map_add, map_smul, smul_eq_mul, eval_mskF, hc]
      ring

/-- The clear claim at the pads `h`. -/
def ClaimZK.at {np m : ℕ} (h : Fin np → F) (c : ClaimZK F np m) : Claim F m := ⟨LinF.eval h c.value, c.skip, c.x⟩

/-- **Lincheck couples**: the masked lincheck followed by any masked continuation that puts the row in front is bounded
by the clear lincheck on the unmasked zerocheck outputs, followed by the clear continuation at the unmasked claim. -/
theorem lincheckZK_couple [CharP F 2] (A : Arith F K) (S : Statement) {np : ℕ} (P : PadsA np) (zc : ZcZK F np S.m)
    (h : Fin np → F) {n : ℕ} (k : ClaimZK F np S.m → Game ((Fin n → Lin F np) × Bool))
    (k' : Option (Claim F S.m) → Game Bool)
    (hk : ∀ ab, value (Vanish h) (k ab) ≤ value (· = true) (k' (some (ab.at h)))) :
    value (Vanish h) ((lincheckZK A S P zc).bind fun o => (k o.1).map (consRow o.2.toLin)) ≤
      value (· = true) ((lincheck A S (zc.at h)).bind k') := by
  simp only [lincheckZK, lincheck, bind_eq, pure_eq, bind_assoc, bind_ret]
  refine value_bind_mono _ fun α => ?_
  refine value_bind_mono _ fun β => ?_
  refine lincheckRounds_couple P h _ _ _ 0 _ _ _ _ ?_ fun ts cl => ?_
  · simp [ZcZK.at, map_add, map_smul, smul_eq_mul]
  refine value_recv_le fun zPartial => ⟨fun s => zPartial s + h (P.zp s), ?_⟩
  refine value_bind_mono _ fun r' => ?_
  refine (value_consRow h _ _).trans ?_
  split_ifs with hrow hc
  · refine (hk _).trans (le_of_eq ?_)
    simp [ClaimZK.at, ZcZK.at, map_sum, map_smul, smul_eq_mul]
  · exfalso
    apply hc
    rw [LinF.eval_toLin] at hrow
    simp only [map_add, map_sum, map_smul, smul_eq_mul, eval_mskF] at hrow
    exact ((add_eq_zero_iff_c2 _ _).1 hrow).symm
  · exact zero_le
  · exact zero_le

/-! ## The opening -/

omit [Fintype F] in
/-- The opening's verdict on two claims with values `va`, `vc`: their claim checks hold, and the opening with the
claim checks as values accepts. -/
theorem opResult_two (A : Arith F K) {m : ℕ} (hm : 7 ≤ m) (va vc : F) (a c : Claim F m)
    (extra : List (Claim F m)) (ea ec : (Fin 128 → F) × (Fin 7 → F))
    (swE : List (Claim F m × (Fin 128 → F) × (Fin 7 → F))) (γ : Fin (extra.length + 2) → F) :
    opResult A hm (⟨va, a.skip, a.x⟩ :: ⟨vc, c.skip, c.x⟩ :: extra)
        ((⟨va, a.skip, a.x⟩, ea) :: (⟨vc, c.skip, c.x⟩, ec) :: swE) γ =
      if ∑ v, packedWeight a hm v * ea.1 v = va ∧ ∑ v, packedWeight c hm v * ec.1 v = vc then
        opResult A hm (⟨∑ v, packedWeight a hm v * ea.1 v, a.skip, a.x⟩ ::
            ⟨∑ v, packedWeight c hm v * ec.1 v, c.skip, c.x⟩ :: extra)
          ((⟨∑ v, packedWeight a hm v * ea.1 v, a.skip, a.x⟩, ea) ::
            (⟨∑ v, packedWeight c hm v * ec.1 v, c.skip, c.x⟩, ec) :: swE) γ
      else none := by
  have hpw : ∀ (v' : F) (c' : Claim F m), packedWeight (⟨v', c'.skip, c'.x⟩ : Claim F m) hm = packedWeight c' hm :=
    fun _ _ => rfl
  by_cases h1 : ∑ v, packedWeight a hm v * ea.1 v = va <;> by_cases h2 : ∑ v, packedWeight c hm v * ec.1 v = vc
  · rw [ite_eq_left ⟨h1, h2⟩, h1, h2]
  all_goals
    rw [ite_eq_right (by tauto)]
    simp only [opResult, List.all_cons, hpw, h1, h2, decide_false, Bool.false_and, Bool.and_false,
      Bool.false_eq_true, ite_false]

variable [Fintype K] [DecidableEq K]

/-- **The rest couples**: at every pad vector `h`, the rest of the masked rep ends with a true verdict and its rows
vanishing at `h` with probability at most the clear opening's on the unmasked claims, followed by the clear level 0. -/
theorem restZK_couple (A : Arith F K) (hA : A.Correct) {np e : ℕ} (P₀ : Pads0 np e) (PA : PadsA np)
    (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (t : ℕ) (c₀ : F) (r : ℕ) (C₀ : Oracle F)
    (extra : List (Claim F S.m)) (zc : ZcZK F np S.m) (ab : ClaimZK F np S.m) (h : Fin np → F)
    (hm7 : 7 ≤ S.m) (k' : Option (OpenOut F) → Game Bool)
    (hk' : ∀ op, k' (some op) = ligeritoPad A sch t c₀ e r op.T op.b C₀) :
    value (Vanish h) (restZK A P₀ PA S sch hm t c₀ r C₀ extra zc ab) ≤
      value (· = true) ((opening A hm7 (ab.at h :: cClaim A (zc.at h) :: extra)).bind k') := by
  have := hA.charTwo
  simp only [restZK, openingZK, opening_eq, List.mapM_cons, opStep, bind_eq, pure_eq, bind_assoc, bind_ret]
  refine value_recv_le fun sa => ⟨sa, ?_⟩
  refine value_bind_mono _ fun ra => ?_
  refine value_recv_le fun sc => ⟨sc, ?_⟩
  refine value_bind_mono _ fun rc => ?_
  refine value_bind_mono _ fun swE => ?_
  refine value_bind_mono _ fun γ => ?_
  have key := opResult_two A hm7 (LinF.eval h ab.value) (LinF.eval h zc.vc) ab.pub ⟨0, A.lagS zc.z, zc.r⟩ extra
    (sa, ra) (sc, rc) swE γ
  split
  · simp [value, Vanish]
  rename_i op heq
  rw [show opResult A hm7 (ClaimZK.at h ab :: cClaim A (ZcZK.at h zc) :: extra)
      ((ClaimZK.at h ab, sa, ra) :: (cClaim A (ZcZK.at h zc), sc, rc) :: swE) γ = _ from key]
  split_ifs with hc
  · erw [heq, hk']
    refine (value_bind_mono (f' := Game.ret) _ fun lv => ?_).trans
      (by rw [bind_ret_right]; exact ligeritoPadZK_couple A P₀ sch t c₀ r op.T op.b C₀ h)
    classical
    simp only [recv, bind_send, bind_ret, value_send, value_ret]
    refine iSup_le fun fc => ?_
    split_ifs with hv hq
    · exact le_rfl
    · exact absurd ⟨hv.1, eval_eq_zero_of_limbRows A hA lv.1 h fun k => by
        fin_cases k; exacts [hv.2 3, hv.2 4]⟩ hq
    · exact zero_le
    · exact le_rfl
  · refine (value_bind_le_of_forall _ _ _ 0 fun lv => ?_).trans zero_le
    classical
    simp only [recv, bind_send, bind_ret, value_send, value_ret]
    refine iSup_le fun fc => ?_
    rw [ite_eq_right fun hv => ?_]
    exfalso
    have r0 := hv.2 0
    have r1 := hv.2 1
    have r2 := hv.2 2
    simp only [Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.cons_val_two, Matrix.head_cons,
      Matrix.tail_cons, LinF.eval_toLin, map_add, eval_cF, eval_mskF] at r0 r1 r2
    rw [add_eq_zero_iff_c2] at r0 r1 r2
    exact hc ⟨r0.symm, r2 ▸ r1.symm⟩

/-! ## The masked rep -/

/-- **The masked rep couples** (VEIL Lemma 4.3): at every pad vector `h` whose product pad is the product, the masked
rep ends with a true verdict and its seven rows vanishing at `h` with probability at most the clear rep's
(`Model.repPad`) acceptance probability. -/
theorem repMaskedZK_couple (A : Arith F K) (hA : A.Correct) {np e : ℕ} (P₀ : Pads0 np e) (PA : PadsA np)
    (tr : Triple np) (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (t : ℕ) (c₀ : F) (r : ℕ) (C₀ : Oracle F)
    (extra : List (Claim F S.m)) (h : Fin np → F) (htr : h tr.ab = h tr.fa * h tr.fb) :
    value (fun o => o.2 = true ∧ ∀ k, (o.1 k).eval h = 0) (repMaskedZK A P₀ PA tr S sch hm t c₀ r C₀ extra) ≤
      value (· = true) (repPad A S sch hm t c₀ e r C₀ extra) := by
  have := hA.charTwo
  simp only [repPad, bind_eq, pure_eq]
  refine zerocheckZK_couple A S.m PA tr h htr _ _ fun zc => ?_
  refine lincheckZK_couple A S PA zc h _ _ fun ab => ?_
  exact restZK_couple A hA P₀ PA S sch hm t c₀ r C₀ extra zc ab h (by omega) _ fun _ => rfl

end FlockSoundness.Discharge.ZkSession
