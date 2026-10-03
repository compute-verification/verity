import FlockSoundness.Discharge.ZkExec.VerdictLane

/-!
# The masked replay's rows are the executable's

`zk_replay_rows`: `Zk.replay`'s three constraints, its `ab` value and its `c` value are the forms the model's masked
zerocheck and lincheck compute, at pads placed as the executable's layout places them.

* `forIn_foldF`: a loop that consumes one one-coin round per step and pushes that coin computes the fold of its step.
* `res_fold_rel`, `lc_fold_rel`: the residual and lincheck rounds' claims (`resOut`, `lcRoundsOut`) are the folds of the
  executable's steps.
* `prod_rel`: the product row `a'b' + a'·h_fb + b'·h_fa + h_ab`.
* `folded_rel`: the circuit's 64 entries are the model's `foldAll` of `comb`.
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL Transcript

local notation "𝔽" => FlockLevel3.GF128

/-- **A loop that pushes its rounds' coins computes the fold of its step.** -/
theorem forIn_foldF {σ : Type} (body : ℕ → σ × Array F128 → TM (ForInStep (σ × Array F128)))
    (F : ℕ → σ → F128 → σ) (R : Array Round) (k₀ : ℕ)
    (hbody : ∀ i s t b t', t.rounds = R → t.k = k₀ + i → body i s t = .ok (b, t') →
      t'.rounds = R ∧ t'.k = t.k + 1 ∧
        b = .yield (F i s.1 ((R[k₀ + i]!).coins.getD 0 0), s.2.push ((R[k₀ + i]!).coins.getD 0 0))) :
    ∀ (n a : ℕ) (s : σ × Array F128) (t t' : Flock.Transcript) (out : σ × Array F128),
      t.rounds = R → t.k = k₀ + a → (forIn (List.range' a n) s body : TM (σ × Array F128)) t = .ok (out, t') →
      out.1 = (List.range' a n).foldl (fun x i => F i x ((R[k₀ + i]!).coins.getD 0 0)) s.1
  | 0, a, s, t, t', out, _, _, h => by
      simp only [List.range'_zero, List.forIn_nil] at h
      obtain ⟨rfl, rfl⟩ := pure_ok h
      rfl
  | n + 1, a, s, t, t', out, hR, hk, h => by
      rw [forIn_range'_cons] at h
      obtain ⟨b, t₁, h₁, h₂⟩ := bind_ok h
      obtain ⟨hr, hk', rfl⟩ := hbody a s t b t₁ hR hk h₁
      rw [forIn_foldF body F R k₀ hbody n (a + 1) _ t₁ t' out hr (by omega) h₂, List.range'_succ, List.foldl_cons]

/-! ## The rounds' steps -/

/-- The replay's residual step on the running constraint, at round `j`'s message, `r[j]` and the coin `rr`. -/
def resStepE (lay : Zk.PadLayout) (rounds : Array (F128 × F128)) (r : Array F128) (j : ℕ) (running : Zk.Lin)
    (rr : F128) : Zk.Lin :=
  let g1 := Zk.Lin.masked rounds[j]!.1 (lay.ml j 0) lay.len
  let ginf := Zk.Lin.masked rounds[j]!.2 (lay.ml j 1) lay.len
  let g0 := (running.add (g1.scale r[j]!)).scale (1 + r[j]!).inv
  ((g0.scale (1 + rr)).add (g1.scale rr)).add (ginf.scale (rr * (1 + rr)))

/-- The replay's lincheck step on the running constraint, at round `j`'s message and the coin `rr`. -/
def lcStepE (lay : Zk.PadLayout) (rounds : Array (F128 × F128)) (j : ℕ) (running : Zk.Lin) (rr : F128) : Zk.Lin :=
  let e1 := Zk.Lin.masked rounds[j]!.1 (lay.lc j 0) lay.len
  let einf := Zk.Lin.masked rounds[j]!.2 (lay.lc j 1) lay.len
  let e0 := running.add e1
  let c1 := (e0.add e1).add einf
  ((einf.scale (rr * rr)).add (c1.scale rr)).add e0

section Steps

variable {np : ℕ} (P : ZkSession.PadsA np) (lay : Zk.PadLayout) (hs : lay.len = np)
  (rounds : Array (F128 × F128))
include hs

theorem res_step_rel (r : Array F128) (j : ℕ) (hml : ∀ c : Fin 2, (P.ml j c).val = lay.ml j c.val) (x : Zk.Lin)
    (cl : ZkSession.LinF 𝔽 np) (hx : LinRel cl.toLin x) (rr : F128) :
    LinRel (ZkSession.residualStepL (fOf r[j]!) (fOf rr) cl (ZkSession.mskF (P.ml j 0) (pairOf rounds[j]!).1)
      (ZkSession.mskF (P.ml j 1) (pairOf rounds[j]!).2)).toLin (resStepE lay rounds r j x rr) := by
  subst hs
  have h1 := rel_masked rounds[j]!.1 (P.ml j 0) (hml 0)
  have h2 := rel_masked rounds[j]!.2 (P.ml j 1) (hml 1)
  have e := rel_add (rel_add (rel_scale (rel_scale (rel_add hx (rel_scale h1 r[j]!)) (1 + r[j]!).inv) (1 + rr))
    (rel_scale h1 rr)) (rel_scale h2 (rr * (1 + rr)))
  have he : ZkSession.residualStepL (fOf r[j]!) (fOf rr) cl (ZkSession.mskF (P.ml j 0) (pairOf rounds[j]!).1)
      (ZkSession.mskF (P.ml j 1) (pairOf rounds[j]!).2) =
      fOf (1 + rr) • fOf (1 + r[j]!).inv • (cl + fOf r[j]! • ZkSession.mskF (P.ml j 0) (fOf rounds[j]!.1)) +
        fOf rr • ZkSession.mskF (P.ml j 0) (fOf rounds[j]!.1) +
        fOf (rr * (1 + rr)) • ZkSession.mskF (P.ml j 1) (fOf rounds[j]!.2) := by
    rw [ZkSession.residualStepL, smul_smul, fOf_inv, fOf_add, fOf_add, fOf_mul, fOf_add, fOf_one, mul_comm (1 + fOf rr)]
    rfl
  rw [he]
  exact e

theorem lc_step_rel (j : ℕ) (hlc : ∀ c : Fin 2, (P.lc j c).val = lay.lc j c.val) (x : Zk.Lin)
    (cl : ZkSession.LinF 𝔽 np) (hx : LinRel cl.toLin x) (rr : F128) :
    LinRel ((fOf rr ^ 2) • ZkSession.mskF (P.lc j 1) (pairOf rounds[j]!).2 +
        fOf rr • (cl + ZkSession.mskF (P.lc j 0) (pairOf rounds[j]!).1 + ZkSession.mskF (P.lc j 0) (pairOf rounds[j]!).1 +
          ZkSession.mskF (P.lc j 1) (pairOf rounds[j]!).2) + (cl + ZkSession.mskF (P.lc j 0) (pairOf rounds[j]!).1)).toLin
      (lcStepE lay rounds j x rr) := by
  subst hs
  have h1 := rel_masked rounds[j]!.1 (P.lc j 0) (hlc 0)
  have h2 := rel_masked rounds[j]!.2 (P.lc j 1) (hlc 1)
  have e := rel_add (rel_add (rel_scale h2 (rr * rr)) (rel_scale (rel_add (rel_add (rel_add hx h1) h1) h2) rr))
    (rel_add hx h1)
  rw [fOf_mul, ← pow_two] at e
  exact e

end Steps

/-- **The residual rounds' claim is the fold of the replay's step.** -/
theorem res_fold_rel {np : ℕ} (P : ZkSession.PadsA np) (lay : Zk.PadLayout) (hs : lay.len = np)
    (rounds : Array (F128 × F128)) (r : Array F128) (C : ℕ → Array F128) :
    ∀ (n a : ℕ) (req : Fin n → 𝔽) (x : Zk.Lin) (cl : ZkSession.LinF 𝔽 np),
      (∀ i : Fin n, req i = fOf r[a + i.val]!) → (∀ i < n, ∀ c : Fin 2, (P.ml (a + i) c).val = lay.ml (a + i) c.val) →
      LinRel cl.toLin x →
      LinRel (resOut P rounds C n a a req cl).toLin
        ((List.range' a n).foldl (fun x i => resStepE lay rounds r i x ((C i).getD 0 0)) x)
  | 0, a, req, x, cl, _, _, hx => by simpa [resOut] using hx
  | n + 1, a, req, x, cl, hreq, hml, hx => by
      rw [resOut, List.range'_succ, List.foldl_cons]
      refine res_fold_rel P lay hs rounds r C n (a + 1) (Fin.tail req) _ _ (fun i => ?_) (fun i hi c => ?_) ?_
      · rw [Fin.tail, hreq, Fin.val_succ, Nat.add_assoc, Nat.add_comm 1]
      · rw [Nat.add_assoc, Nat.add_comm 1]; exact hml (i + 1) (by omega) c
      · have e := res_step_rel P lay hs rounds r a (hml 0 (by omega)) x cl hx ((C a).getD 0 0)
        rw [hreq 0, getElem!_eq_getD (C a) 0]
        simpa using e

/-- **The lincheck rounds' claim is the fold of the replay's step.** -/
theorem lc_fold_rel {np : ℕ} (P : ZkSession.PadsA np) (lay : Zk.PadLayout) (hs : lay.len = np)
    (rounds : Array (F128 × F128)) (C : ℕ → Array F128) :
    ∀ (n a : ℕ) (x : Zk.Lin) (cl : ZkSession.LinF 𝔽 np),
      (∀ i < n, ∀ c : Fin 2, (P.lc (a + i) c).val = lay.lc (a + i) c.val) → LinRel cl.toLin x →
      LinRel (lcRoundsOut P rounds C n a a cl).toLin
        ((List.range' a n).foldl (fun x i => lcStepE lay rounds i x ((C i).getD 0 0)) x)
  | 0, a, x, cl, _, hx => by simpa [lcRoundsOut] using hx
  | n + 1, a, x, cl, hlc, hx => by
      rw [lcRoundsOut, List.range'_succ, List.foldl_cons]
      refine lc_fold_rel P lay hs rounds C n (a + 1) _ _ (fun i hi c => ?_) ?_
      · rw [Nat.add_assoc, Nat.add_comm 1]; exact hlc (i + 1) (by omega) c
      · have e := lc_step_rel P lay hs rounds a (hlc 0 (by omega)) x cl hx ((C a).getD 0 0)
        rw [getElem!_eq_getD (C a) 0]
        exact e

/-- **The product row** `a'b' + a'·h_fb + b'·h_fa + h_ab` is the replay's `prod`. -/
theorem prod_rel {np : ℕ} (tr : ZkSession.Triple np) (lay : Zk.PadLayout) (hs : lay.len = np)
    (hfa : tr.fa.val = lay.fa) (hfb : tr.fb.val = lay.fb) (hab : tr.ab.val = lay.hab) (h₁ : lay.fa ≠ lay.fb)
    (h₂ : lay.fa ≠ lay.hab) (h₃ : lay.fb ≠ lay.hab) (fA fB : F128) :
    LinRel (ZkSession.cF (fOf fA * fOf fB) + fOf fA • ZkSession.pF tr.fb + fOf fB • ZkSession.pF tr.fa +
        ZkSession.pF tr.ab : ZkSession.LinF 𝔽 np).toLin
      ⟨fA * fB, (((Array.replicate lay.len 0).set! lay.fb fA).set! lay.fa fB).set! lay.hab 1⟩ := by
  subst hs
  refine ⟨by simp, ?_, fun i => ?_⟩
  · show fOf fA * fOf fB + fOf fA * 0 + fOf fB * 0 + 0 = fOf (fA * fB)
    rw [fOf_mul]; ring
  · show (ZkSession.cF (fOf fA * fOf fB) + fOf fA • ZkSession.pF tr.fb + fOf fB • ZkSession.pF tr.fa +
        ZkSession.pF tr.ab : ZkSession.LinF 𝔽 lay.len).1 i =
      fOf ((((Array.replicate lay.len 0).set! lay.fb fA).set! lay.fa fB).set! lay.hab 1)[i.val]!
    simp only [Prod.fst_add, Prod.smul_fst, Pi.add_apply, Pi.smul_apply, ZkSession.cF, ZkSession.pF, smul_eq_mul,
      Pi.zero_apply]
    have hfa' := tr.fa.isLt
    have hfb' := tr.fb.isLt
    have hab' := tr.ab.isLt
    rw [set_get _ _ (by simp; omega) (by simp), set_get _ _ (by simp; omega) (by simp),
      set_get _ _ (by simp; omega) (by simp), repl_get _ i.isLt]
    simp only [Pi.single_apply, Fin.ext_iff, hfa, hfb, hab]
    by_cases ia : i.val = lay.hab
    · have : i.val ≠ lay.fa := fun h => h₂ (h.symm.trans ia)
      have : i.val ≠ lay.fb := fun h => h₃ (h.symm.trans ia)
      simp_all [fOf_one]
    by_cases ib : i.val = lay.fa
    · have : i.val ≠ lay.fb := fun h => h₁ (ib.symm.trans h)
      simp_all
    by_cases ic : i.val = lay.fb
    · simp_all
    simp_all [fOf_zero]

/-- **The circuit's 64 entries are the model's fold** (as in `lincheck_refines`): `CircuitFold.folded` at the
lincheck's coins is `foldAll` of the model's `comb` at the rounds' challenges. -/
theorem folded_rel (S : Model.Statement) (c : CircuitFold) (hc : FoldRealizes c S) (α β z : F128)
    (rho ts w : Array F128) (hrho : S.kLog - 6 ≤ rho.size)
    (hw : c.folded α β (skipWeights z) (rho.extract 0 (S.kLog - 6)) ts = .ok w)
    (ρM tsM : Fin (S.kLog - 6) → 𝔽) (hρ : ∀ j, ρM j = fOf rho[j.val]!) (hts : ts.size = S.kLog - 6)
    (htsM : ∀ j, tsM j = fOf ts[j.val]!) :
    ∀ i < 64, Model.foldAll (List.ofFn tsM) (2 ^ S.kLog) (Model.comb execArith S (fOf z) (fOf α) (fOf β) ρM) i =
      fOf w[i]! := by
  have hk6 : 6 ≤ S.kLog := S.hk6
  obtain ⟨v, hv, hvsz, hwv⟩ := hc.folded _ _ _ _ _ w hw
  rw [hc.pin] at hwv
  have hcomb : ∀ col < 2 ^ S.kLog, fOf (v.modify S.pin.val fun x => x + β)[col]! =
      Model.comb execArith S (fOf z) (fOf α) (fOf β) ρM col := by
    intro col hcol
    have hfr := hc.fold α (skipWeights z) (rho.extract 0 (S.kLog - 6)) v (skipWeights_size _) (by simp; omega) hv
      ⟨col, hcol⟩
    rw [FlockLevel3.modify_get! v S.pin.val col _ (by omega), Model.comb, dite_eq_left hcol]
    have hρ' : vecOf (S.kLog - 6) (rho.extract 0 (S.kLog - 6)) = ρM := by
      funext j
      rw [hρ]
      simp only [vecOf]
      rw [getElem!_pos _ _ (by simp; omega), getElem!_pos _ _ (by omega), Array.getElem_extract]
      simp
    have hsum : fOf v[col]! = ∑ i : Fin (2 ^ S.kLog), execArith.lagS (fOf z) ⟨i.val % 64, Nat.mod_lt _ (by norm_num)⟩ *
        Model.eqAt ρM (i.val / 64) * (fOf α * ((S.A₀ i ⟨col, hcol⟩).val : 𝔽) + ((S.B₀ i ⟨col, hcol⟩).val : 𝔽)) := by
      rw [hfr, hρ']
      rfl
    by_cases hp : S.pin.val = col
    · rw [ite_eq_left hp, ite_eq_left hp.symm, fOf_add, hsum]
    · rw [ite_eq_right hp, ite_eq_right (Ne.symm hp), add_zero, hsum]
  intro i hi
  have hof : List.ofFn tsM = ts.toList.map fOf := by
    refine List.ext_getElem (by simp [hts]) fun j h₁ h₂ => ?_
    rw [List.getElem_ofFn, List.getElem_map, htsM, getElem!_pos ts j (by simp at h₁; omega)]
    simp
  have hs₀sz : (v.modify S.pin.val fun x => x + β).size = 2 ^ S.kLog := by simp [hvsz]
  have hlen : ts.toList.length = S.kLog - 6 := by simp [hts]
  have hi' : i < (v.modify S.pin.val fun x => x + β).size / 2 ^ ts.toList.length := by
    rw [hs₀sz, hlen, Nat.pow_div (by omega) (by norm_num), show S.kLog - (S.kLog - 6) = 6 by omega]; exact hi
  rw [hof, hwv, ← Array.foldl_toList, foldl_halve ts.toList _ i hi', hs₀sz]
  symm
  refine foldAll_congr (ts.toList.map fOf) (2 ^ S.kLog) _ _ (fun c hc' => ?_) i ?_
  · rw [hcomb c hc']
  · rw [List.length_map, hlen, Nat.pow_div (by omega) (by norm_num), show S.kLog - (S.kLog - 6) = 6 by omega]
    exact hi

end FlockSoundness.Discharge.ZkExec
