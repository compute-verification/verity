import Proofs.Flock.Soundness.Discharge.ZkExec.VerdictLane

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

/-! ## The replay's first constraints -/

section Init

variable (lay : Zk.PadLayout) (P : ZkSession.PadsA lay.len)

theorem rel_maskedArr (v : Array F128) (pad : Fin 64 → Fin lay.len) (idx : ℕ → ℕ)
    (hpad : ∀ i : Fin 64, (pad i).val = idx i.val) (k : Fin 64) :
    LinRel (ZkSession.mskF (pad k) (vecOf 64 v k)).toLin
      ((Array.range 64).map fun i => Zk.Lin.masked v[i]! (idx i) lay.len)[k.val]! := by
  rw [getElem!_range_map _ k.isLt]
  exact rel_masked v[k.val]! (pad k) (hpad k)

/-- `P_c(z)`: the `c` half of round 1 at the output coset's weights. -/
theorem pc_rel (zcp : ZerocheckProof) (hr1c : ∀ i : Fin 64, (P.r1c i).val = lay.r1c i.val) (z : F128) :
    LinRel (∑ i, execArith.lagΛ (fOf z) i • ZkSession.mskF (P.r1c i) (vecOf 64 zcp.round1c i)).toLin
      (Zk.Lin.combo (lagrange nodesL den6Inv z)
        ((Array.range 64).map fun i => Zk.Lin.masked zcp.round1c[i]! (lay.r1c i) lay.len) lay.len) :=
  rel_combo _ _ _ (by rw [lagrange_size, nodesL_size]) (by simp)
    (rel_maskedArr lay zcp.round1c P.r1c lay.r1c hr1c)

/-- The zerocheck's first claim `P_ab(z)`: round 1 at the combined weights, plus `P_c(z)`. -/
theorem zcInit_rel (zcp : ZerocheckProof) (hr1ab : ∀ i : Fin 64, (P.r1ab i).val = lay.r1ab i.val)
    (hr1c : ∀ i : Fin 64, (P.r1c i).val = lay.r1c i.val) (z : F128) :
    LinRel (∑ i, execArith.combW (fOf z) i • (ZkSession.mskF (P.r1ab i) (vecOf 64 zcp.round1ab i) +
          ZkSession.mskF (P.r1c i) (vecOf 64 zcp.round1c i)) +
        ∑ i, execArith.lagΛ (fOf z) i • ZkSession.mskF (P.r1c i) (vecOf 64 zcp.round1c i)).toLin
      ((Zk.Lin.combo (combWeights z) ((Array.range 64).map fun i =>
          ((Array.range 64).map fun i => Zk.Lin.masked zcp.round1ab[i]! (lay.r1ab i) lay.len)[i]!.add
            ((Array.range 64).map fun i => Zk.Lin.masked zcp.round1c[i]! (lay.r1c i) lay.len)[i]!) lay.len).add
        (Zk.Lin.combo (lagrange nodesL den6Inv z)
          ((Array.range 64).map fun i => Zk.Lin.masked zcp.round1c[i]! (lay.r1c i) lay.len) lay.len)) := by
  refine rel_add (rel_combo _ _ _ (combWeights_size _) (by simp) fun k => ?_) (pc_rel lay P zcp hr1c z)
  rw [getElem!_range_map _ k.isLt]
  exact rel_add (rel_maskedArr lay zcp.round1ab P.r1ab lay.r1ab hr1ab k) (rel_maskedArr lay zcp.round1c P.r1c lay.r1c hr1c k)

end Init

/-- **The masked replay's rows** (`zk_replay_plays` with its constraints): at pads placed as the layout `lay` places
them, the replay's three constraints, its `ab` value and its `c` value are the forms the model's masked zerocheck and
lincheck compute on the same messages and coins. -/
theorem zk_replay_rows (st : Setup) (lay : Zk.PadLayout) (p : Proof) (root : ByteArray) (t t' : Flock.Transcript)
    (rp : Zk.Replayed) (S : Model.Statement) (hm : st.m = S.m) (hk : st.kLog = S.kLog) (h15 : 15 ≤ S.m)
    (hwf : RoundsWF t.rounds) (hc : FoldRealizes st.fold S) (h : Zk.replay st lay p root t = .ok (rp, t'))
    {np : ℕ} (P : ZkSession.PadsA np) (tr : ZkSession.Triple np) (hlen : lay.len = np)
    (hr1ab : ∀ i : Fin 64, (P.r1ab i).val = lay.r1ab i.val) (hr1c : ∀ i : Fin 64, (P.r1c i).val = lay.r1c i.val)
    (hml : ∀ j < S.m - 6, ∀ c : Fin 2, (P.ml j c).val = lay.ml j c.val) (hfc : P.fc.val = lay.fc)
    (hlc : ∀ j < S.kLog - 6, ∀ c : Fin 2, (P.lc j c).val = lay.lc j c.val)
    (hzp : ∀ i : Fin 64, (P.zp i).val = lay.zp i.val) (hfa : tr.fa.val = lay.fa) (hfb : tr.fb.val = lay.fb)
    (hab : tr.ab.val = lay.hab) (h₁ : lay.fa ≠ lay.fb) (h₂ : lay.fa ≠ lay.hab) (h₃ : lay.fb ≠ lay.hab) :
    t'.rounds = t.rounds ∧ t.k ≤ t'.k ∧
      ∃ (k₁ : ℕ) (zc : ZkSession.ZcZK 𝔽 np S.m) (row : ZkSession.LinF 𝔽 np) (ab : ZkSession.ClaimZK 𝔽 np S.m)
        (row' : ZkSession.LinF 𝔽 np),
        t.k ≤ k₁ ∧ k₁ ≤ t'.k ∧
        Plays fOf (ZkSession.zerocheckZK execArith S.m P tr) (zcMsgs S.m p.zerocheck) (consumed t.rounds t.k k₁)
          (zc, row) ∧
        Plays fOf (ZkSession.lincheckZK execArith S P zc) (lcMsgs S.kLog p.lincheck) (consumed t.rounds k₁ t'.k)
          (ab, row') ∧
        (∀ v : F128, ClaimRel S.m { rp.claims[0]! with value := v } ⟨fOf v, ab.skip, ab.x⟩) ∧
        (∀ v : F128, ClaimRel S.m { rp.claims[1]! with value := v } ⟨fOf v, execArith.lagS zc.z, zc.r⟩) ∧
        rp.cons.size = 3 ∧ LinRel (zc.vc + ZkSession.mskF P.fc (fOf p.zerocheck.finalC)).toLin rp.cons[0]!.2 ∧
        LinRel row.toLin rp.cons[1]!.2 ∧ LinRel row'.toLin rp.cons[2]!.2 ∧ LinRel ab.value.toLin rp.abValue ∧
        rp.cValue = Zk.Lin.masked p.zerocheck.finalC lay.fc lay.len := by
  subst hlen
  have hk6 : 6 ≤ S.kLog := S.hk6
  have hkm : S.kLog ≤ S.m := S.hk
  unfold Zk.replay at h
  dsimp only at h
  rw [hm, hk] at h
  obtain ⟨t₁, k₁, h⟩ := label_bind h
  obtain ⟨t₂, k₂, h⟩ := absorbBytes_bind h
  obtain ⟨t₃, k₃, h⟩ := absorbBytes_bind h
  obtain ⟨t₄, k₄, h⟩ := label_bind h
  obtain ⟨t₅, k₅, h⟩ := label_bind h
  obtain ⟨t₆, k₆, h⟩ := absorbBytes_bind h
  obtain ⟨lt₆, n₆, t₇, r₇, kk₇, h⟩ := sampleVec_bind (by decide) h
  obtain ⟨lt₇, n₇, t₈, r₈, kk₈, h⟩ := sampleVec_bind (by omega) h
  obtain ⟨t₉, k₉, h⟩ := absorbSlice_bind h
  obtain ⟨t₁₀, k₁₀, h⟩ := absorbSlice_bind h
  obtain ⟨lt₁₀, n₁₀, t₁₁, r₁₁, kk₁₁, h⟩ := sample_bind h
  obtain ⟨s, t₁₂, hloop, h⟩ := bind_ok h
  obtain ⟨t₁₃, k₁₃, h⟩ := absorbF_bind h
  obtain ⟨t₁₄, k₁₄, h⟩ := absorbF_bind h
  obtain ⟨t₁₅, k₁₅, h⟩ := label_bind h
  obtain ⟨lt₁₅, n₁₅, t₁₆, r₁₆, kk₁₆, h⟩ := sample_bind h
  obtain ⟨lt₁₆, n₁₆, t₁₇, r₁₇, kk₁₇, h⟩ := sample_bind h
  obtain ⟨s', t₁₈, hloop', h⟩ := bind_ok h
  rcases hf : st.fold.folded ((t₁₅.rounds[t₁₅.k]!).coins.getD 0 0) ((t₁₆.rounds[t₁₆.k]!).coins.getD 0 0)
      (skipWeights ((t₁₀.rounds[t₁₀.k]!).coins.getD 0 0)) (s.2.extract 0 (S.kLog - 6)) s'.2 with err | v
  · rw [hf] at h
    obtain ⟨_, _, h₀, _⟩ := bind_ok h
    cases h₀
  rw [hf] at h
  simp only [pure_bind] at h
  obtain ⟨hv64, h⟩ := check_bind h
  obtain ⟨t₁₉, k₁₉, h⟩ := absorbSlice_bind h
  obtain ⟨lt₁₉, n₁₉, t₂₀, r₂₀, kk₂₀, h⟩ := sample_bind h
  obtain ⟨hrp, ht'⟩ := pure_ok h
  subst hrp
  -- the loops: one one-coin round per step, its coin pushed
  rw [forIn_range_eq] at hloop hloop'
  obtain ⟨hr₁₂, hk₁₂, hn₁₂, hs⟩ := forIn_push _ (fun i st' tt b tt' hb => by
      obtain ⟨tt₁, kk₁, hb⟩ := absorbF_bind hb
      obtain ⟨tt₂, kk₂, hb⟩ := absorbF_bind hb
      obtain ⟨-, nn, tt₃, rr₃, kk₃, hb⟩ := sample_bind hb
      obtain ⟨rfl, rfl⟩ := pure_ok hb
      have e1 : tt₂.rounds = tt.rounds := by rw [kk₂.2, kk₁.2]
      have e2 : tt₂.k = tt.k := by rw [kk₂.1, kk₁.1]
      refine ⟨by rw [← e1, ← e2]; exact nn, by rw [rr₃, e1], by rw [kk₃, e2], ?_⟩
      rw [← e1, ← e2]
      exact ⟨_, rfl⟩) (S.m - 6) 0 _ t₁₁ t₁₂ s hloop
  obtain ⟨hr₁₈, hk₁₈, hn₁₈, hs'⟩ := forIn_push _ (fun i st' tt b tt' hb => by
      obtain ⟨tt₁, kk₁, hb⟩ := absorbF_bind hb
      obtain ⟨tt₂, kk₂, hb⟩ := absorbF_bind hb
      obtain ⟨-, nn, tt₃, rr₃, kk₃, hb⟩ := sample_bind hb
      obtain ⟨rfl, rfl⟩ := pure_ok hb
      have e1 : tt₂.rounds = tt.rounds := by rw [kk₂.2, kk₁.2]
      have e2 : tt₂.k = tt.k := by rw [kk₂.1, kk₁.1]
      refine ⟨by rw [← e1, ← e2]; exact nn, by rw [rr₃, e1], by rw [kk₃, e2], ?_⟩
      rw [← e1, ← e2]
      exact ⟨_, rfl⟩) (S.kLog - 6) 0 _ t₁₇ t₁₈ s' hloop'
  -- the loops' constraints: the folds of their steps
  have hfold₁ := forIn_foldF _ (fun j running rr => resStepE lay p.zerocheck.rounds
      (pinned ++ (t₇.rounds[t₇.k]!).coins) j running rr) t₁₁.rounds t₁₁.k (fun i st' tt b tt' hR hK hb => by
      obtain ⟨tt₁, kk₁, hb⟩ := absorbF_bind hb
      obtain ⟨tt₂, kk₂, hb⟩ := absorbF_bind hb
      obtain ⟨-, -, tt₃, rr₃, kk₃, hb⟩ := sample_bind hb
      obtain ⟨rfl, rfl⟩ := pure_ok hb
      have e1 : tt₂.rounds = tt.rounds := by rw [kk₂.2, kk₁.2]
      have e2 : tt₂.k = tt.k := by rw [kk₂.1, kk₁.1]
      refine ⟨by rw [rr₃, e1, hR], by rw [kk₃, e2], ?_⟩
      rw [e1, e2, hR, hK]
      rfl) (S.m - 6) 0 _ t₁₁ t₁₂ s rfl rfl hloop
  have hfold₂ := forIn_foldF _ (fun j running rr => lcStepE lay p.lincheck.rounds j running rr) t₁₇.rounds t₁₇.k
    (fun i st' tt b tt' hR hK hb => by
      obtain ⟨tt₁, kk₁, hb⟩ := absorbF_bind hb
      obtain ⟨tt₂, kk₂, hb⟩ := absorbF_bind hb
      obtain ⟨-, -, tt₃, rr₃, kk₃, hb⟩ := sample_bind hb
      obtain ⟨rfl, rfl⟩ := pure_ok hb
      have e1 : tt₂.rounds = tt.rounds := by rw [kk₂.2, kk₁.2]
      have e2 : tt₂.k = tt.k := by rw [kk₂.1, kk₁.1]
      refine ⟨by rw [rr₃, e1, hR], by rw [kk₃, e2], ?_⟩
      rw [e1, e2, hR, hK]
      rfl) (S.kLog - 6) 0 _ t₁₇ t₁₈ s' rfl rfl hloop'
  -- the rounds and the counter, in terms of `t`
  have hr₆ : t₆.rounds = t.rounds := by rw [k₆.2, k₅.2, k₄.2, k₃.2, k₂.2, k₁.2]
  have hk₆' : t₆.k = t.k := by rw [k₆.1, k₅.1, k₄.1, k₃.1, k₂.1, k₁.1]
  have hr₇ : t₇.rounds = t.rounds := by rw [r₇, hr₆]
  have hk₇ : t₇.k = t.k + 1 := by rw [kk₇, hk₆']
  have hr₁₀ : t₁₀.rounds = t.rounds := by rw [k₁₀.2, k₉.2, r₈, hr₇]
  have hk₁₀ : t₁₀.k = t.k + 2 := by rw [k₁₀.1, k₉.1, kk₈, hk₇]
  have hr₁₁ : t₁₁.rounds = t.rounds := by rw [r₁₁, hr₁₀]
  have hk₁₁ : t₁₁.k = t.k + 3 := by rw [kk₁₁, hk₁₀]
  have hr₁₅ : t₁₅.rounds = t.rounds := by rw [k₁₅.2, k₁₄.2, k₁₃.2, hr₁₂, hr₁₁]
  have hk₁₅ : t₁₅.k = t.k + 3 + (S.m - 6) := by rw [k₁₅.1, k₁₄.1, k₁₃.1, hk₁₂, hk₁₁]
  have hr₁₆ : t₁₆.rounds = t.rounds := by rw [r₁₆, hr₁₅]
  have hk₁₆ : t₁₆.k = t.k + 3 + (S.m - 6) + 1 := by rw [kk₁₆, hk₁₅]
  have hr₁₇ : t₁₇.rounds = t.rounds := by rw [r₁₇, hr₁₆]
  have hk₁₇ : t₁₇.k = t.k + 3 + (S.m - 6) + 2 := by rw [kk₁₇, hk₁₆]
  have hr₁₉ : t₁₉.rounds = t.rounds := by rw [k₁₉.2, hr₁₈, hr₁₇]
  have hk₁₉ : t₁₉.k = t.k + 3 + (S.m - 6) + 2 + (S.kLog - 6) := by rw [k₁₉.1, hk₁₈, hk₁₇]
  have hrt : t'.rounds = t.rounds := by rw [ht', r₂₀, hr₁₉]
  have hkt : t'.k = t.k + 3 + (S.m - 6) + 2 + (S.kLog - 6) + 1 := by rw [ht', kk₂₀, hk₁₉]
  set c₀ := (t₆.rounds[t₆.k]!).coins with hc₀
  set c₁ := (t₇.rounds[t₇.k]!).coins with hc₁
  set c₂ := (t₁₀.rounds[t₁₀.k]!).coins with hc₂
  set C : ℕ → Array F128 := fun i => (t₁₁.rounds[t₁₁.k + i]!).coins with hC
  set cα := (t₁₅.rounds[t₁₅.k]!).coins with hcα
  set cβ := (t₁₆.rounds[t₁₆.k]!).coins with hcβ
  set C' : ℕ → Array F128 := fun i => (t₁₇.rounds[t₁₇.k + i]!).coins with hC'
  set cr := (t₁₉.rounds[t₁₉.k]!).coins with hcr
  have hsz₀ : c₀.size = 6 := by rw [hc₀, RoundsWF.get (hr₆ ▸ hwf), n₆]
  have hsz₁ : c₁.size = S.m - 13 := by rw [hc₁, RoundsWF.get (hr₇ ▸ hwf), n₇]
  have hsz₂ : c₂.size = 1 := by rw [hc₂, RoundsWF.get (hr₁₀ ▸ hwf), n₁₀]
  have hszα : cα.size = 1 := by rw [hcα, RoundsWF.get (hr₁₅ ▸ hwf), n₁₅]
  have hszβ : cβ.size = 1 := by rw [hcβ, RoundsWF.get (hr₁₆ ▸ hwf), n₁₆]
  have hszr : cr.size = 1 := by rw [hcr, RoundsWF.get (hr₁₉ ▸ hwf), n₁₉]
  have hCsz : ∀ i ∈ List.range' 0 (S.m - 6), (C i).size = 1 := fun i hi => by
    rw [hC]; simp only; rw [RoundsWF.get (hr₁₁ ▸ hwf), hn₁₂ i (by simpa using hi)]
  have hC'sz : ∀ i ∈ List.range' 0 (S.kLog - 6), (C' i).size = 1 := fun i hi => by
    rw [hC']; simp only; rw [RoundsWF.get (hr₁₇ ▸ hwf), hn₁₈ i (by simpa using hi)]
  -- the coins consumed, split at the lincheck's first
  have hcoins₁ : consumed t.rounds t.k (t.k + 3 + (S.m - 6)) = c₀ :: c₁ :: c₂ :: (List.range' 0 (S.m - 6)).map C := by
    rw [consumed_append t.rounds (show t.k ≤ t.k + 1 by omega) (by omega),
      consumed_append t.rounds (show t.k + 1 ≤ t.k + 2 by omega) (by omega),
      consumed_append t.rounds (show t.k + 2 ≤ t.k + 3 by omega) (by omega), consumed_succ, consumed_succ,
      consumed_succ, consumed_eq_map, hc₀, hc₁, hc₂, hC, hr₆, hk₆', hr₇, hk₇, hr₁₀, hk₁₀, hr₁₁, hk₁₁]
    rfl
  have hcoins₂ : consumed t.rounds (t.k + 3 + (S.m - 6)) t'.k =
      cα :: cβ :: ((List.range' 0 (S.kLog - 6)).map C' ++ [cr]) := by
    rw [hkt, consumed_append t.rounds (show t.k + 3 + (S.m - 6) ≤ t.k + 3 + (S.m - 6) + 1 by omega) (by omega),
      consumed_append t.rounds (show t.k + 3 + (S.m - 6) + 1 ≤ t.k + 3 + (S.m - 6) + 2 by omega) (by omega),
      consumed_append t.rounds (show t.k + 3 + (S.m - 6) + 2 ≤ t.k + 3 + (S.m - 6) + 2 + (S.kLog - 6) by omega)
        (by omega), consumed_succ, consumed_succ, consumed_succ, consumed_eq_map, hcα, hcβ, hcr, hC', hr₁₅, hk₁₅,
      hr₁₆, hk₁₆, hr₁₇, hk₁₇, hr₁₉, hk₁₉]
    simp
  -- the coins the loops pushed
  obtain ⟨hssz, hsget⟩ := push_get s.2 (S.m - 6) _ hs
  obtain ⟨hs'sz, hs'get⟩ := push_get s'.2 (S.kLog - 6) _ hs'
  have hzM : fOf c₂[0]! = fOf (c₂.getD 0 0) := by rw [getElem!_eq_getD]
  have hrM : fOf cr[0]! = fOf (cr.getD 0 0) := by rw [getElem!_eq_getD]
  refine ⟨hrt, by omega, t.k + 3 + (S.m - 6), ?zc, ?row, ?ab, ?row', by omega, by omega, ?pz, ?pl, ?c0, ?c1, ?sz,
    ?r0, ?r1, ?r2, ?rab, ?rc⟩
  case pz =>
    rw [hcoins₁]
    unfold ZkSession.zerocheckZK
    refine Plays.bind' (ms₂ := zcMsgs S.m p.zerocheck) (cs₂ := c₁ :: c₂ :: (List.range' 0 (S.m - 6)).map C)
      (Plays.drawVec c₀ hsz₀ (by decide)) ?_ (by simp) (by simp)
    refine Plays.bind' (ms₂ := zcMsgs S.m p.zerocheck) (cs₂ := c₂ :: (List.range' 0 (S.m - 6)).map C)
      (Plays.drawVec c₁ hsz₁ (by omega)) ?_ (by simp) (by simp)
    refine Plays.bind' (ms₂ := ((List.range (S.m - 6)).map fun i => ⟨𝔽 × 𝔽, pairOf p.zerocheck.rounds[i]!⟩) ++
      [⟨𝔽 × 𝔽, (fOf p.zerocheck.finalA, fOf p.zerocheck.finalB)⟩]) (cs₂ := c₂ :: (List.range' 0 (S.m - 6)).map C)
      (Plays.recv _ (vecOf 64 p.zerocheck.round1ab, vecOf 64 p.zerocheck.round1c)) ?_ (by simp [zcMsgs]) (by simp)
    refine Plays.bind' (ms₂ := ((List.range (S.m - 6)).map fun i => ⟨𝔽 × 𝔽, pairOf p.zerocheck.rounds[i]!⟩) ++
      [⟨𝔽 × 𝔽, (fOf p.zerocheck.finalA, fOf p.zerocheck.finalB)⟩]) (cs₂ := (List.range' 0 (S.m - 6)).map C)
      (Plays.drawOne c₂ hsz₂) ?_ (by simp) (by simp)
    refine Plays.bind' (ms₂ := [⟨𝔽 × 𝔽, (fOf p.zerocheck.finalA, fOf p.zerocheck.finalB)⟩]) (cs₂ := [])
      (residualZK_plays P p.zerocheck.rounds C (S.m - 6) 0 0 _ _ hCsz) ?_ (by simp [List.range_eq_range']) (by simp)
    refine Plays.bind' (ms₂ := []) (cs₂ := []) (Plays.recv _ (fOf p.zerocheck.finalA, fOf p.zerocheck.finalB)) ?_
      (by simp) (by simp)
    exact Plays.pure (ι := fOf) _
  case pl =>
    rw [hcoins₂]
    unfold ZkSession.lincheckZK
    refine Plays.bind' (ms₂ := lcMsgs S.kLog p.lincheck)
      (cs₂ := cβ :: ((List.range' 0 (S.kLog - 6)).map C' ++ [cr])) (Plays.drawOne cα hszα) ?_ (by simp) (by simp)
    refine Plays.bind' (ms₂ := lcMsgs S.kLog p.lincheck) (cs₂ := (List.range' 0 (S.kLog - 6)).map C' ++ [cr])
      (Plays.drawOne cβ hszβ) ?_ (by simp) (by simp)
    refine Plays.bind' (ms₂ := [⟨Fin 64 → 𝔽, vecOf 64 p.lincheck.zPartial⟩]) (cs₂ := [cr])
      (lcRoundsZK_plays P p.lincheck.rounds C' (S.kLog - 6) 0 0 _ hC'sz) ?_
      (by simp [lcMsgs, List.range_eq_range']) (by simp)
    refine Plays.bind' (ms₂ := []) (cs₂ := [cr]) (Plays.recv _ (vecOf 64 p.lincheck.zPartial)) ?_ (by simp) (by simp)
    refine Plays.bind' (ms₂ := []) (cs₂ := []) (Plays.drawOne cr hszr) ?_ (by simp) (by simp)
    exact Plays.pure (ι := fOf) _
  case c0 =>
    intro w
    show ClaimRel S.m (⟨w, skipWeights (cr.getD 0 0), s'.2.reverse ++ s.2.extract (S.kLog - 6) (S.m - 6)⟩ :
      Flock.Claim) _
    have hxs : (s'.2.reverse ++ s.2.extract (S.kLog - 6) (S.m - 6)).size = S.m - 6 := by
      simp [hssz, hs'sz]; omega
    refine ⟨rfl, skipWeights_size _, fun _ => by simp only [hrM]; rfl, hxs, fun i => ?_⟩
    simp only
    rw [getElem!_pos _ i.val (by rw [hxs]; exact i.isLt), Array.getElem_append]
    by_cases hi : i.val < S.kLog - 6
    · have hi' : i.val < s'.2.reverse.size := by simp [hs'sz, hi]
      rw [dite_eq_left hi, dite_eq_left hi', Array.getElem_reverse, ← getElem!_pos s'.2 _ (by omega),
        hs'get _ (by omega), Nat.zero_add, getElem!_eq_getD]
      congr 3
      simp [hs'sz]
    · have hi' : ¬ i.val < s'.2.reverse.size := by simp [hs'sz, hi]
      have hidx : S.kLog - 6 + (i.val - s'.2.reverse.size) = i.val := by simp [hs'sz]; omega
      rw [dite_eq_right hi, dite_eq_right hi', Array.getElem_extract,
        ← getElem!_pos s.2 _ (by simp [hs'sz] at *; omega), hidx, hsget _ i.isLt, Nat.zero_add, getElem!_eq_getD]
  case c1 =>
    intro w
    simp only [abcClaims_one]
    exact ⟨rfl, skipWeights_size _, fun _ => by rw [hzM]; rfl,
      by rw [Array.size_append, FlockLevel3.pinned_size, hsz₁]; omega, fun i => zcVector_eq c₁ hsz₁ (by omega) i⟩
  case sz => rfl
  case r0 =>
    rw [hzM]
    show LinRel _ ((Zk.Lin.combo _ _ lay.len).add _)
    refine rel_add ?_ ?_
    · have e := pc_rel lay P p.zerocheck hr1c (c₂.getD 0 0)
      convert e using 2
    · exact rel_masked _ P.fc hfc
  case r1 =>
    rw [hzM]
    show LinRel _ (s.1.add _)
    rw [hfold₁]
    refine rel_add ?_ (prod_rel tr lay rfl hfa hfb hab h₁ h₂ h₃ _ _)
    refine res_fold_rel P lay rfl p.zerocheck.rounds (pinned ++ c₁) C (S.m - 6) 0 _ _ _
      (fun i => by rw [Nat.zero_add]; exact zcVector_eq c₁ hsz₁ (by omega) i)
      (fun i hi c => by rw [Nat.zero_add]; exact hml i hi c) ?_
    convert zcInit_rel lay P p.zerocheck hr1ab hr1c (c₂.getD 0 0) using 2
  case r2 =>
    rw [hzM, getElem!_eq_getD cα 0, getElem!_eq_getD cβ 0]
    show LinRel _ (s'.1.add _)
    rw [hfold₂]
    refine rel_add (lc_fold_rel P lay rfl p.lincheck.rounds C' (S.kLog - 6) 0 _ _
      (fun i hi c => by rw [Nat.zero_add]; exact hlc i hi c)
      (rel_addConst (rel_add (rel_scale (rel_masked p.zerocheck.finalA tr.fa hfa) (cα.getD 0 0))
        (rel_masked p.zerocheck.finalB tr.fb hfb)) (cβ.getD 0 0))) ?_
    have hcb := rel_combo v _ (fun k => ZkSession.mskF (P.zp k) (vecOf 64 p.lincheck.zPartial k)) (eq_of_beq hv64)
      (by simp) (rel_maskedArr lay p.lincheck.zPartial P.zp lay.zp hzp)
    convert hcb using 3
    rename_i k _
    congr 1
    exact folded_rel S st.fold hc _ _ _ s.2 s'.2 v (by rw [hssz]; omega) hf _ _
      (fun j => by
        show fOf (C (0 + j.val))[0]! = _
        rw [hsget j.val (by omega), Nat.zero_add, getElem!_eq_getD]
        try rfl) hs'sz (fun j => by
        rw [hs'get j.val j.isLt, Nat.zero_add, getElem!_eq_getD]
        try rfl) k k.isLt
  case rab =>
    rw [hrM]
    convert rel_combo _ _ _ (skipWeights_size _) (by simp) (rel_maskedArr lay p.lincheck.zPartial P.zp lay.zp hzp)
      using 3
    rfl
  case rc => rfl

end FlockSoundness.Discharge.ZkExec
