import FlockSoundness.Protocol.Discharge.ZkSession.Level0

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

/-- An affine form in the pads `h : Fin np → F`, `Σ_i coef_i·h_i + const`, as a vector space. -/
abbrev LinF (F : Type) (np : ℕ) := (Fin np → F) × F

/-- The row a form is. -/
def LinF.toLin {np : ℕ} (ℓ : LinF F np) : Lin F np := ⟨ℓ.1, ℓ.2⟩

/-- A constant form. -/
def cF {np : ℕ} (x : F) : LinF F np := (0, x)

/-- Pad `i`. -/
def pF {np : ℕ} (i : Fin np) : LinF F np := (Pi.single i 1, 0)

/-- A sent value `v'` unmasked by pad `i`: `v' + h_i`. -/
def mskF {np : ℕ} (i : Fin np) (v : F) : LinF F np := cF v + pF i

section Eval

variable {np : ℕ} (h : Fin np → F)

end Eval

/-! ## Rows -/

/-- A phase's rows with a row in front. -/
def consRow {np n : ℕ} (row : Lin F np) (o : (Fin n → Lin F np) × Bool) : (Fin (n + 1) → Lin F np) × Bool :=
  (Fin.cons row o.1, o.2)

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

/-- The masked zerocheck's outputs: the coins, as in `ZcOut`, and `final_a`, `final_b`, `Pc(z)` as forms. -/
structure ZcZK (F : Type) (np m : ℕ) where
  z : F
  ρ : Fin (m - 6) → F
  r : Fin (m - 6) → F
  va : LinF F np
  vb : LinF F np
  vc : LinF F np

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

/-- A claim whose value is a form: the masked lincheck's `ab` claim. -/
structure ClaimZK (F : Type) (np m : ℕ) where
  value : LinF F np
  skip : Fin 64 → F
  x : Fin (m - 6) → F

/-- The claim's point, its value left out (the opening reads only the point). -/
def ClaimZK.pub {np m : ℕ} (c : ClaimZK F np m) : Claim F m := ⟨0, c.skip, c.x⟩

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

/-! ## The masked rep -/

/-- A masked rep after zerocheck: the masked lincheck, its row in front of the rest's. -/
noncomputable def afterZcZK (A : Arith F K) {np e : ℕ} (P₀ : Pads0 np e) (PA : PadsA np) (S : Statement)
    (sch : Schedule) (hm : 13 ≤ S.m) (t : ℕ) (c₀ : F) (r : ℕ) (C₀ : Oracle F) (extra : List (Claim F S.m))
    (zc : ZcZK F np S.m) : Game ((Fin 6 → Lin F np) × Bool) :=
  (lincheckZK A S PA zc).bind fun o => (restZK A P₀ PA S sch hm t c₀ r C₀ extra zc o.1).map (consRow o.2.toLin)

/-- **A masked rep** (M1's algebraic phase, then level 0 masked): the masked zerocheck, lincheck and opening, level 0
(`ligeritoPadZK`) and `final_c'`. Returns its verdict and its seven rows, in order zerocheck final a·b, lincheck final,
ring switch ab, zerocheck c_eval, ring switch c, and level 0's two lane-phase rows. -/
noncomputable def repMaskedZK (A : Arith F K) {np e : ℕ} (P₀ : Pads0 np e) (PA : PadsA np) (tr : Triple np)
    (S : Statement) (sch : Schedule) (hm : 13 ≤ S.m) (t : ℕ) (c₀ : F) (r : ℕ) (C₀ : Oracle F)
    (extra : List (Claim F S.m)) : Game ((Fin 7 → Lin F np) × Bool) :=
  (zerocheckZK A S.m PA tr).bind fun o =>
    (afterZcZK A P₀ PA S sch hm t c₀ r C₀ extra o.1).map (consRow o.2.toLin)

end FlockSoundness.Discharge.ZkSession
