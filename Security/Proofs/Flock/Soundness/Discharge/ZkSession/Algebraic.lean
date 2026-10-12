import Proofs.Flock.Soundness.Discharge.ZkSession.Level0
import Proofs.Flock.Soundness.OpeningPhase
import Proofs.Flock.Soundness.Game.Union

/-!
# C-Flock ZK's masked algebraic phase (live/PROTOCOL.md §1, §2, §4)

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

/-- An affine form in the pads `h : Fin np → F`, `Σ_i coef_i·h_i + const`, as a vector space. -/
abbrev LinF (F : Type) (np : ℕ) := (Fin np → F) × F

/-- The form's value at the pads `h`. -/
def LinF.eval {np : ℕ} (h : Fin np → F) : LinF F np →ₗ[F] F where
  toFun ℓ := ∑ i, ℓ.1 i * h i + ℓ.2
  map_add' a b := by
    simp only [Prod.fst_add, Prod.snd_add, Pi.add_apply, add_mul, Finset.sum_add_distrib]
    ring
  map_smul' c a := by
    simp only [Prod.smul_fst, Prod.smul_snd, Pi.smul_apply, smul_eq_mul, RingHom.id_apply, mul_add,
      Finset.mul_sum, mul_assoc]

/-- The row a form is. -/
def LinF.toLin {np : ℕ} (ℓ : LinF F np) : Lin F np := ⟨ℓ.1, ℓ.2⟩

theorem LinF.eval_toLin {np : ℕ} (ℓ : LinF F np) (h : Fin np → F) : ℓ.toLin.eval h = LinF.eval h ℓ := rfl

/-- A constant form. -/
def cF {np : ℕ} (x : F) : LinF F np := (0, x)

/-- Pad `i`. -/
def pF {np : ℕ} (i : Fin np) : LinF F np := (Pi.single i 1, 0)

/-- A sent value `v'` unmasked by pad `i`: `v' + h_i`. -/
def mskF {np : ℕ} (i : Fin np) (v : F) : LinF F np := cF v + pF i

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

/-- A phase's rows with a row in front. -/
def consRow {np n : ℕ} (row : Lin F np) (o : (Fin n → Lin F np) × Bool) : (Fin (n + 1) → Lin F np) × Bool :=
  (Fin.cons row o.1, o.2)

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

/-- The pads of the algebraic phase (§2): `round1_ab`, `round1_c`, residual round `j`'s `G(1), G(∞)`, `final_c`,
lincheck round `j`'s `q(1), q(∞)` and `z_partial`. `final_a`, `final_b` and `h_ab` are the triple's `fa`, `fb`, `ab`.
Soundness holds for any placement. -/
structure PadsA (np : ℕ) where
  r1ab : Fin 64 → Fin np
  r1c : Fin 64 → Fin np
  ml : ℕ → Fin 2 → Fin np
  fc : Fin np
  lc : ℕ → Fin 2 → Fin np
  zp : Fin 64 → Fin np

/-! ## Zerocheck -/

/-- `residualStep` on forms. -/
def residualStepL {np : ℕ} (req ρ : F) (claim g₁ gInf : LinF F np) : LinF F np :=
  ((1 + req)⁻¹ * (1 + ρ)) • (claim + req • g₁) + ρ • g₁ + (ρ * (1 + ρ)) • gInf

theorem eval_residualStepL {np : ℕ} (h : Fin np → F) (req ρ : F) (claim g₁ gInf : LinF F np) :
    LinF.eval h (residualStepL req ρ claim g₁ gInf) =
      residualStep req (LinF.eval h claim) (LinF.eval h g₁) (LinF.eval h gInf) ρ := by
  simp only [residualStepL, residualStep, map_add, map_smul, smul_eq_mul]
  ring

variable [Fintype F] [DecidableEq F]

/-- The masked residual rounds (Z6): round `j`'s message is unmasked by its pads. -/
def residualRoundsZK {np : ℕ} (P : PadsA np) :
    (n : ℕ) → ℕ → (Fin n → F) → LinF F np → Game ((Fin n → F) × LinF F np)
  | 0, _, _, claim => pure (Fin.elim0, claim)
  | n + 1, j, req, claim => do
      let g ← recv (F × F)
      let ρ ← draw F
      let (ρs, c) ← residualRoundsZK P n (j + 1) (Fin.tail req)
        (residualStepL (req 0) ρ claim (mskF (P.ml j 0) g.1) (mskF (P.ml j 1) g.2))
      pure (Fin.cons ρ ρs, c)

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

/-- The masked zerocheck's outputs: the coins, as in `ZcOut`, and `final_a`, `final_b`, `Pc(z)` as forms. -/
structure ZcZK (F : Type) (np m : ℕ) where
  z : F
  ρ : Fin (m - 6) → F
  r : Fin (m - 6) → F
  va : LinF F np
  vb : LinF F np
  vc : LinF F np

/-- The clear outputs at the pads `h`. -/
def ZcZK.at {np m : ℕ} (h : Fin np → F) (zc : ZcZK F np m) : ZcOut F m :=
  ⟨zc.z, zc.ρ, zc.r, LinF.eval h zc.va, LinF.eval h zc.vb, LinF.eval h zc.vc⟩

/-- **The masked zerocheck** (§1, §4): `Model.zerocheck` on masked messages. Its check `G_final = a·b` becomes the row
`G_final + a'b' + a'·h_fb + b'·h_fa + h_ab` (§4's zerocheck final a·b), which it returns. -/
def zerocheckZK (A : Arith F K) (m : ℕ) {np : ℕ} (P : PadsA np) (tr : Triple np) :
    Game (ZcZK F np m × LinF F np) := do
  let _rSkip ← draw (Fin 6 → F)
  let rs ← draw (Fin (m - 13) → F)
  let r := zcVector A rs
  let msg ← recv ((Fin 64 → F) × (Fin 64 → F))
  let z ← draw F
  let pc : LinF F np := ∑ i, A.lagΛ z i • mskF (P.r1c i) (msg.2 i)
  let claim₀ := ∑ i, A.combW z i • (mskF (P.r1ab i) (msg.1 i) + mskF (P.r1c i) (msg.2 i)) + pc
  let (ρ, claim) ← residualRoundsZK P (m - 6) 0 r claim₀
  let v ← recv (F × F)
  let row := claim + (cF (v.1 * v.2) + v.1 • pF tr.fb + v.2 • pF tr.fa + pF tr.ab)
  pure (⟨z, ρ, r, mskF tr.fa v.1, mskF tr.fb v.2, pc⟩, row)

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

/-- The masked lincheck rounds (L3): round `j`'s message is unmasked by its pads. -/
def lincheckRoundsZK {np : ℕ} (P : PadsA np) : (n : ℕ) → ℕ → LinF F np → Game ((Fin n → F) × LinF F np)
  | 0, _, claim => pure (Fin.elim0, claim)
  | n + 1, j, claim => do
      let q ← recv (F × F)
      let t ← draw F
      let q₁ := mskF (P.lc j 0) q.1
      let q₂ := mskF (P.lc j 1) q.2
      let e₀ := claim + q₁
      let c₁ := e₀ + q₁ + q₂
      let (ts, c) ← lincheckRoundsZK P n (j + 1) ((t ^ 2) • q₂ + t • c₁ + e₀)
      pure (Fin.cons t ts, c)

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

/-- A claim whose value is a form: the masked lincheck's `ab` claim. -/
structure ClaimZK (F : Type) (np m : ℕ) where
  value : LinF F np
  skip : Fin 64 → F
  x : Fin (m - 6) → F

/-- The claim's point, its value left out (the opening reads only the point). -/
def ClaimZK.pub {np m : ℕ} (c : ClaimZK F np m) : Claim F m := ⟨0, c.skip, c.x⟩

/-- The clear claim at the pads `h`. -/
def ClaimZK.at {np m : ℕ} (h : Fin np → F) (c : ClaimZK F np m) : Claim F m := ⟨LinF.eval h c.value, c.skip, c.x⟩

/-- **The masked lincheck** (§1, §4): `Model.lincheck` on masked messages, from the masked zerocheck's forms. Its check
becomes the row `running + Σ_i comb_i·(z_partial'_i + h_zp(i))` (§4's lincheck final), which it returns with the `ab`
claim, whose value is a form. -/
def lincheckZK (A : Arith F K) (S : Statement) {np : ℕ} (P : PadsA np) (zc : ZcZK F np S.m) :
    Game (ClaimZK F np S.m × LinF F np) := do
  let α ← draw F
  let β ← draw F
  let n := S.kLog - 6
  let (ts, claim) ← lincheckRoundsZK P n 0 (α • zc.va + zc.vb + cF β)
  let zPartial ← recv (Fin 64 → F)
  let r' ← draw F
  let ρIn : Fin n → F := fun j => zc.ρ ⟨j.val, by have := S.hk; omega⟩
  let folded := foldAll (List.ofFn ts) (2 ^ S.kLog) (comb A S zc.z α β ρIn)
  let zp : Fin 64 → LinF F np := fun s => mskF (P.zp s) (zPartial s)
  let x : Fin (S.m - 6) → F := fun i =>
    if h : i.val < n then ts ⟨n - 1 - i.val, by omega⟩ else zc.ρ i
  pure (⟨∑ s : Fin 64, A.lagS r' s • zp s, A.lagS r', x⟩, claim + ∑ s : Fin 64, folded s • zp s)

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

/-- **The masked opening** (§4): `Model.opening` with the `ab` and `c` claims' values replaced by their claim checks
`Σ_v w_v·s_hat_v[v]` on the prover's `s_hat_v`, as the live verifier calls it (their own values are masked). Returns the
two claim checks, for the ring-switch rows. Only the two claims' points are read. -/
def openingZK (A : Arith F K) {m : ℕ} (hm : 7 ≤ m) (a c : Claim F m) (extra : List (Claim F m)) :
    Game (Option (OpenOut F) × F × F) := do
  let ea ← opStep a
  let ec ← opStep c
  let swE ← extra.mapM opStep
  let γ ← draw (Fin (a :: c :: extra).length → F)
  let ca := ∑ v, packedWeight a hm v * ea.2.1 v
  let cc := ∑ v, packedWeight c hm v * ec.2.1 v
  let a₁ : Claim F m := ⟨ca, a.skip, a.x⟩
  let c₁ : Claim F m := ⟨cc, c.skip, c.x⟩
  pure (opResult A hm (a₁ :: c₁ :: extra) ((a₁, ea.2) :: (c₁, ec.2) :: swE) γ, ca, cc)

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

/-- **The rest of a masked rep**, after lincheck: the masked opening, level 0 masked (`ligeritoPadZK`), then
`final_c'`, received before the inner proof. Returns §4's ring switch ab, zerocheck c_eval and ring switch c rows, and
level 0's two lane-phase rows. -/
noncomputable def restZK (A : Arith F K) {np e : ℕ} (P₀ : Pads0 np e) (PA : PadsA np) (S : Statement)
    (sch : Schedule) (hm : 13 ≤ S.m) (t : ℕ) (c₀ : F) (r : ℕ) (C₀ : Oracle F) (extra : List (Claim F S.m))
    (zc : ZcZK F np S.m) (ab : ClaimZK F np S.m) : Game ((Fin 5 → Lin F np) × Bool) := do
  let o ← openingZK A (by omega) ab.pub ⟨0, A.lagS zc.z, zc.r⟩ extra
  match o.1 with
  | none => pure (fun _ => ⟨0, 0⟩, false)
  | some op => do
      let lv ← ligeritoPadZK A P₀ sch t c₀ r op.T op.b C₀
      let fc ← recv F
      pure (![(ab.value + cF o.2.1).toLin, (zc.vc + mskF PA.fc fc).toLin, (mskF PA.fc fc + cF o.2.2).toLin,
        limbRows A lv.1 0, limbRows A lv.1 1], lv.2)

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

/-- A masked rep after zerocheck: the masked lincheck, its row in front of the rest's. -/
noncomputable def afterZcZK (A : Arith F K) {np e : ℕ} (P₀ : Pads0 np e) (PA : PadsA np) (S : Statement)
    (sch : Schedule) (hm : 13 ≤ S.m) (t : ℕ) (c₀ : F) (r : ℕ) (C₀ : Oracle F) (extra : List (Claim F S.m))
    (zc : ZcZK F np S.m) : Game ((Fin 6 → Lin F np) × Bool) :=
  (lincheckZK A S PA zc).bind fun o => (restZK A P₀ PA S sch hm t c₀ r C₀ extra zc o.1).map (consRow o.2.toLin)

/-- **A masked rep** (C-Flock ZK's algebraic phase, then level 0 masked): the masked zerocheck, lincheck and opening,
level 0 (`ligeritoPadZK`) and `final_c'`. Returns its verdict and its seven rows, in order zerocheck final a·b, lincheck
final, ring switch ab, zerocheck c_eval, ring switch c, and level 0's two lane-phase rows. -/
noncomputable def repMaskedZK (A : Arith F K) {np e : ℕ} (P₀ : Pads0 np e) (PA : PadsA np) (tr : Triple np)
    (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (t : ℕ) (c₀ : F) (r : ℕ) (C₀ : Oracle F)
    (extra : List (Claim F S.m)) : Game ((Fin 7 → Lin F np) × Bool) :=
  (zerocheckZK A S.m PA tr).bind fun o =>
    (afterZcZK A P₀ PA S sch hm t c₀ r C₀ extra o.1).map (consRow o.2.toLin)

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
