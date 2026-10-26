import Proofs.Flock.Bridge.Public
import Proofs.Flock.Soundness.Game.Prob

/-!
# `SampledKSLivePub`'s honest and nontrivial lemmas (LEAN-66)

* **Honest** (`sampledKSLivePub_honest`): the hypotheses of `SampledKSLivePub` on the analysis' parameters hold at the
  deployed ones: `k = ⌈2^24/e⌉` (`Bridge.kDep`), `ρ = 1/2`, `M = R_w = 2^192` and `t' = 2^80`. Its other hypotheses read
  the prover's run (the circuit file, the registration, the record's custody) or are its named assumptions (A3, and the
  finders' collision resistance `TableCRZ` at the budgets `q_F`, `q_S` and `LinkCRZC` at `t'`).
* **Nontrivial** (`sampledKSLivePub_nontrivial`): `SampledKSLivePubAll`, `SampledKSLivePub` at the verifier that
  accepts everything, is false wherever, at those parameters, the fixed extractor's values fail the claim
  (`¬ RelS`) and the finders' premises hold, with the draw's miss at most `1/4` and at most `2^40` commit strings
  (`Bridge.ioSum`). There every run is accepted, so `Pr[accept ∧ ¬ RelS] = 1`, while `deltaPub < 1`
  (`Bridge.deltaPub_deployed_lt_one`). The instance is a premise: the deployed parser (`Flock.Desc.parse`) reads the
  circuit file through `Lean.Json`'s parser, a `partial def` the kernel can't unfold, so no proof exhibits one.
-/

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

/-- **The analysis' `k` as deployed**: `⌈2^24/e⌉`, the least `k` with `rateM1 k ≤ 1/2`. -/
noncomputable def _root_.Flock.Bridge.kDep : ℕ := ⌈(2 : ℝ) ^ 24 / Real.exp 1⌉₊

theorem _root_.Flock.Bridge.kDep_bounds :
    (2 : ℝ) ^ 24 ≤ Real.exp 1 * Bridge.kDep ∧ (Bridge.kDep : ℝ) ≤ 2 ^ 24 := by
  have he : 0 < Real.exp 1 := Real.exp_pos 1
  have he1 : 1 ≤ Real.exp 1 := Real.one_le_exp (by norm_num)
  refine ⟨?_, ?_⟩
  · have h := Nat.le_ceil ((2 : ℝ) ^ 24 / Real.exp 1)
    rw [div_le_iff₀ he] at h
    unfold Bridge.kDep
    linarith [mul_comm (Real.exp 1) (⌈(2 : ℝ) ^ 24 / Real.exp 1⌉₊ : ℝ)]
  · have h2 : (2 : ℝ) ^ 24 / Real.exp 1 ≤ (2 : ℝ) ^ 24 := div_le_self (by positivity) he1
    have h : Bridge.kDep ≤ 2 ^ 24 := Nat.ceil_le.2 (h2.trans_eq (by norm_num))
    exact_mod_cast h

/-- **`SampledKSLivePub`'s parameter hypotheses hold as deployed**: `1 ≤ k`, `1 ≤ R_w`, `1 ≤ M`, `ρ < 1`,
`rateM1 k ≤ ρ` and `0 ≤ t'`, at `k = ⌈2^24/e⌉`, `R_w = M = 2^192`, `ρ = 1/2` and `t' = 2^80`. -/
theorem sampledKSLivePub_honest :
    1 ≤ Bridge.kDep ∧ 1 ≤ (2 ^ 192 : ℕ) ∧ 1 ≤ (2 ^ 192 : ℕ) ∧ (1 / 2 : ℝ) < 1 ∧
      rateM1 Bridge.kDep ≤ 1 / 2 ∧ (0 : ℝ) ≤ 2 ^ 80 := by
  obtain ⟨hk, -⟩ := Bridge.kDep_bounds
  have he : 0 < Real.exp 1 := Real.exp_pos 1
  have hk0 : (0 : ℝ) < Bridge.kDep :=
    pos_of_mul_pos_right (lt_of_lt_of_le (by norm_num) hk) he.le
  refine ⟨Nat.cast_pos.1 hk0, Nat.one_le_two_pow, Nat.one_le_two_pow, by norm_num, ?_, by positivity⟩
  unfold rateM1
  rw [div_le_iff₀ (mul_pos he hk0)]
  linarith

set_option exponentiation.threshold 1024 in
/-- **`deltaPub` is below `1` as deployed**: at `k = ⌈2^24/e⌉`, `q_F = 2^81`, `q_S = 2^80`, `t' = 2^80`,
`M = R_w = 2^192` and `ρ = 1/2`, for a draw whose miss is at most `1/4` and at most `2^40` commit strings. -/
theorem _root_.Flock.Bridge.deltaPub_deployed_lt_one {n kd K₀ W : ℕ}
    (hmiss : ((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ) ≤ 1 / 4) (hW : (W : ℝ) ≤ 2 ^ 40) :
    Bridge.deltaPub n kd K₀ Bridge.kDep (2 ^ 81) (2 ^ 80) (2 ^ 80) (2 ^ 192) (2 ^ 192) W (1 / 2) < 1 := by
  obtain ⟨-, hk⟩ := Bridge.kDep_bounds
  have hk0 : (0 : ℝ) ≤ Bridge.kDep := Nat.cast_nonneg _
  have he1 : 1 ≤ Real.exp 1 := Real.one_le_exp (by norm_num)
  have h513 : (2 : ℝ) ^ (513 : ℝ) = 2 ^ 513 := by
    rw [show (513 : ℝ) = ((513 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
  have h256 : (2 : ℝ) ^ (256 : ℝ) = 2 ^ 256 := by
    rw [show (256 : ℝ) = ((256 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
  have h203 : (2 : ℝ) ^ (-(203 : ℝ)) ≤ 1 / 2 ^ 200 := by
    rw [Real.rpow_neg (by norm_num), show (203 : ℝ) = ((203 : ℕ) : ℝ) by norm_num, Real.rpow_natCast]
    norm_num
  have hx : Real.sqrt (((2 : ℝ) ^ 81) ^ 2 / 2 ^ (513 : ℝ)) ≤ 1 / 2 ^ 170 := by
    rw [h513]
    calc Real.sqrt (((2 : ℝ) ^ 81) ^ 2 / 2 ^ 513) ≤ Real.sqrt ((1 / 2 ^ 170) ^ 2) :=
          Real.sqrt_le_sqrt (by norm_num)
      _ = 1 / 2 ^ 170 := Real.sqrt_sq (by norm_num)
  have hkx : Real.sqrt ((Bridge.kDep : ℝ) * (((2 : ℝ) ^ 81) ^ 2 / 2 ^ (513 : ℝ))) ≤ 1 / 2 ^ 150 := by
    rw [h513]
    calc Real.sqrt ((Bridge.kDep : ℝ) * (((2 : ℝ) ^ 81) ^ 2 / 2 ^ 513))
          ≤ Real.sqrt ((1 / 2 ^ 150) ^ 2) := by
            refine Real.sqrt_le_sqrt ?_
            calc (Bridge.kDep : ℝ) * (((2 : ℝ) ^ 81) ^ 2 / 2 ^ 513)
                ≤ 2 ^ 24 * (((2 : ℝ) ^ 81) ^ 2 / 2 ^ 513) := mul_le_mul_of_nonneg_right hk (by positivity)
              _ ≤ (1 / 2 ^ 150) ^ 2 := by norm_num
      _ = 1 / 2 ^ 150 := Real.sqrt_sq (by norm_num)
  have hqS : ((2 : ℝ) ^ 80) ^ 2 / 2 ^ (513 : ℝ) ≤ 1 / 2 ^ 300 := by
    rw [h513]
    norm_num
  have hks : ksClosedM1 Bridge.kDep (2 ^ 81) (2 ^ 80) ≤ 1 / 16 := by
    unfold ksClosedM1
    exact le_trans (add_le_add (add_le_add (add_le_add h203 (mul_le_mul_of_nonneg_left hx (by norm_num))) hqS) hkx)
      (by norm_num)
  have ht1 : 2 * ((2 : ℝ) ^ 80 * (1 + Bridge.kDep)) / 2 ^ (256 : ℝ) ≤ 1 / 2 ^ 150 := by
    rw [h256]
    calc 2 * ((2 : ℝ) ^ 80 * (1 + Bridge.kDep)) / 2 ^ 256 ≤ 2 * ((2 : ℝ) ^ 80 * (1 + 2 ^ 24)) / 2 ^ 256 := by
          gcongr
      _ ≤ 1 / 2 ^ 150 := by norm_num
  have hM : (2 : ℝ) ^ 192 ≤ Real.exp 1 * 2 ^ 192 := le_mul_of_one_le_left (by positivity) he1
  have hc : ((2 ^ 192 : ℕ) : ℝ) = (2 : ℝ) ^ 192 := by norm_num
  have ht2 : 1 / (Real.exp 1 * ((2 ^ 192 : ℕ) : ℝ)) ≤ 1 / 2 ^ 192 := by
    rw [hc]
    exact one_div_le_one_div_of_le (by positivity) hM
  have ht3 : (Bridge.kDep : ℝ) / (Real.exp 1 * ((2 ^ 192 : ℕ) : ℝ)) ≤ 2 ^ 24 / 2 ^ 192 := by
    rw [hc]
    exact (div_le_div_of_nonneg_left hk0 (by positivity) hM).trans (div_le_div_of_nonneg_right hk (by positivity))
  have hB0 : 0 ≤ 2 * ((2 : ℝ) ^ 80 * (1 + Bridge.kDep)) / 2 ^ (256 : ℝ) +
      1 / (Real.exp 1 * ((2 ^ 192 : ℕ) : ℝ)) + (Bridge.kDep : ℝ) / (Real.exp 1 * ((2 ^ 192 : ℕ) : ℝ)) := by
    positivity
  have hWB : (W : ℝ) * (2 * ((2 : ℝ) ^ 80 * (1 + Bridge.kDep)) / 2 ^ (256 : ℝ) +
      1 / (Real.exp 1 * ((2 ^ 192 : ℕ) : ℝ)) + (Bridge.kDep : ℝ) / (Real.exp 1 * ((2 ^ 192 : ℕ) : ℝ))) ≤
      2 ^ 40 * (1 / 2 ^ 150 + 1 / 2 ^ 192 + 2 ^ 24 / 2 ^ 192) :=
    mul_le_mul hW (by linarith) hB0 (by positivity)
  have hhalf : ∀ x : ℝ, x / (1 - 1 / 2) = 2 * x := fun x => by ring
  unfold Bridge.deltaPub
  rw [hhalf, hhalf]
  have hnum : (2 : ℝ) ^ 40 * (1 / 2 ^ 150 + 1 / 2 ^ 192 + 2 ^ 24 / 2 ^ 192) ≤ 1 / 8 := by norm_num
  linarith

open Classical in
/-- **`SampledKSLivePub` at the verifier that accepts everything**: its statement with `LiveAcceptsZK I o` replaced by
`True`. -/
def SampledKSLivePubAll : Prop :=
  ∀ (n : ℕ) (Reg : Type)
    (I : Inputs) (ht : I.tags.typed = false)
    (D : Flock.Desc) (hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D) (hU : Built D.unroll.unit)
    (pub : Public) (hpub : HmOut.loadPublic (zkTags I) D.unroll I.publicFile = .ok pub) (hpt : pub.tables = none)
    (hscope : Discharge.Layout.scopeOkH D.unroll = true)
    (_hrows : Bridge.RowsOk D.unroll)
    (_hdup : Flock.Registered.dupName D.unroll = none) (_hroots : Bridge.PortRoots.rootsColl D.unroll pub n = none)
    (desc : Claims.Msg) (qr : Lean.Json) (_hpop : Bridge.PopMatches desc qr n)
    (kd : ℕ) (_hkd : kd ≤ n) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : DrawSetupZHJ (inputsJ I 0) (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf D.unroll.g (kd / nTab I)) → Discharge.Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    (_hA3 : Assumptions.UniformRandomBytes Lb src)
    (hTags : TagsOkZHJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr kd)
    (k Rw M : ℕ) (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) (ρ : ℝ) (_hρ : ρ < 1)
    (_hρk : rateM1 k ≤ ρ) (t' : ℝ) (_ht' : 0 ≤ t') (K₀ : ℕ) (_hK₀ : K₀ ≤ n)
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    (qF qS : ℝ)
    (q : Claims.Sampling.Query),
    let L := Law.execOS (n := n) src (.subset kd) n
    let P := (keyProg D.unroll hU n).partition (Discharge.Layout.outsOf D.unroll hU n)
    let vb := (rsZHJ dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt).bindingZ
      (rsZHJ_computes dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt)
    let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJH_live I ht (Flock.Desc.parse_flat hD) (Discharge.Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ hTags
          hCust.1
          fun _ => drawOkZKHJ_of_file I ht (Flock.Desc.parse_flat hD) kd n hdiv dj hCust.2 _)) τ
    let Xr := XplurZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) k Rw
    let δ : ℝ := ((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ)
    let Cc := (keyProg D.unroll hU n).circuit (Discharge.Layout.outsOf D.unroll hU n)
    let rows : Fin Cc.N → Option Bridge.RowAt := fun g =>
      (FlockSoundness.Discharge.HiddenExec.rowOfH D.unroll hU n (Discharge.Layout.outsOf D.unroll hU n) g).map fun r =>
        ⟨r.2.1.val, r.1.val, Discharge.Layout.portRowW D.unroll r.2.1, r.2.2.val⟩
    let roots := Bridge.rootsOf D.unroll pub n
    let Cd := Bridge.commitDep Cc rows D.unroll n roots
    let κ : Fin Cc.N → Option Bool := fun g => Bridge.constsOf D.unroll hU n g
    (∀ ω j, TableCRZ execArith hs512 (planZHJ dj y₀ (L.draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) qF qS) →
    LinkCRZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) k Rw M t' →
    prob (fun _ => True ∧
        ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg Cd δ (Bridge.stmtOf P κ kd desc q roots) (Bridge.partOf P)
          (Bridge.witDep Cc Cd Claims.Sampling.sha512Msg roots Xr))
        (auditLive L Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
      ENNReal.ofReal (Bridge.deltaPub n kd K₀ k qF qS t' M Rw (Bridge.ioSum P) ρ)

open Classical in
/-- **`SampledKSLivePub`'s nontrivial lemma** (LEAN-66): `SampledKSLivePub` at the verifier that accepts everything is
false at any instance of its setting where, at the deployed parameters (`k = ⌈2^24/e⌉`, `R_w = M = 2^192`, `ρ = 1/2`,
`t' = 2^80`, `q_F = 2^81`, `q_S = 2^80`), the finders' premises hold, the fixed extractor's values fail the claim, the
draw's miss is at most `1/4` and the units read at most `2^40` commit strings: every run is accepted, so
`Pr[accept ∧ ¬ RelS] = 1`, and `deltaPub < 1` there. -/
theorem sampledKSLivePub_nontrivial :
  ∀ (n : ℕ) (Reg : Type)
    (I : Inputs) (ht : I.tags.typed = false)
    (D : Flock.Desc) (hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D) (hU : Built D.unroll.unit)
    (pub : Public) (hpub : HmOut.loadPublic (zkTags I) D.unroll I.publicFile = .ok pub) (hpt : pub.tables = none)
    (hscope : Discharge.Layout.scopeOkH D.unroll = true)
    (_hrows : Bridge.RowsOk D.unroll)
    (_hdup : Flock.Registered.dupName D.unroll = none) (_hroots : Bridge.PortRoots.rootsColl D.unroll pub n = none)
    (desc : Claims.Msg) (qr : Lean.Json) (_hpop : Bridge.PopMatches desc qr n)
    (kd : ℕ) (_hkd : kd ≤ n) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : DrawSetupZHJ (inputsJ I 0) (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf D.unroll.g (kd / nTab I)) → Discharge.Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    (_hA3 : Assumptions.UniformRandomBytes Lb src)
    (hTags : TagsOkZHJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr kd)
    (K₀ : ℕ) (_hK₀ : K₀ ≤ n)
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    (q : Claims.Sampling.Query),
    let L := Law.execOS (n := n) src (.subset kd) n
    let P := (keyProg D.unroll hU n).partition (Discharge.Layout.outsOf D.unroll hU n)
    let vb := (rsZHJ dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt).bindingZ
      (rsZHJ_computes dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt)
    let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJH_live I ht (Flock.Desc.parse_flat hD) (Discharge.Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ hTags
          hCust.1
          fun _ => drawOkZKHJ_of_file I ht (Flock.Desc.parse_flat hD) kd n hdiv dj hCust.2 _)) τ
    let Xr := XplurZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) Bridge.kDep (2 ^ 192)
    let δ : ℝ := ((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ)
    let Cc := (keyProg D.unroll hU n).circuit (Discharge.Layout.outsOf D.unroll hU n)
    let rows : Fin Cc.N → Option Bridge.RowAt := fun g =>
      (FlockSoundness.Discharge.HiddenExec.rowOfH D.unroll hU n (Discharge.Layout.outsOf D.unroll hU n) g).map fun r =>
        ⟨r.2.1.val, r.1.val, Discharge.Layout.portRowW D.unroll r.2.1, r.2.2.val⟩
    let roots := Bridge.rootsOf D.unroll pub n
    let Cd := Bridge.commitDep Cc rows D.unroll n roots
    let κ : Fin Cc.N → Option Bool := fun g => Bridge.constsOf D.unroll hU n g
    (∀ ω j, TableCRZ execArith hs512 (planZHJ dj y₀ (L.draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) (2 ^ 81) (2 ^ 80)) →
    LinkCRZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) Bridge.kDep (2 ^ 192) (2 ^ 192)
      (2 ^ 80) →
    ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg Cd δ (Bridge.stmtOf P κ kd desc q roots) (Bridge.partOf P)
      (Bridge.witDep Cc Cd Claims.Sampling.sha512Msg roots Xr) →
    δ ≤ 1 / 4 → (Bridge.ioSum P : ℝ) ≤ 2 ^ 40 →
    ¬ SampledKSLivePubAll := by
  intro n Reg I ht D hD hU pub hpub hpt hscope hrows hdup hroots desc qr hpop kd hkd hdiv dj S₀ d₀ y₀ wr dg N Ω _ _ Lb
    src hA3 hTags hCust K₀ hK₀ τ q L P vb σ Xr δ Cc rows roots Cd κ hCR hLink hbad hmiss hW hAll
  obtain ⟨hk, hRw, hM, hρ, hρk, ht'⟩ := sampledKSLivePub_honest
  have h := hAll n Reg I ht D hD hU pub hpub hpt hscope hrows hdup hroots desc qr hpop kd hkd hdiv dj S₀ d₀ y₀ wr dg N
    Ω Lb src hA3 hTags hCust Bridge.kDep (2 ^ 192) (2 ^ 192) hk hRw hM (1 / 2) hρ hρk (2 ^ 80) ht' K₀ hK₀ τ (2 ^ 81)
    (2 ^ 80) q hCR hLink
  have h1 : prob (fun _ => True ∧
      ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg Cd δ (Bridge.stmtOf P κ kd desc q roots) (Bridge.partOf P)
        (Bridge.witDep Cc Cd Claims.Sampling.sha512Msg roots Xr))
      (auditLive L Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ = 1 :=
    le_antisymm (prob_le_one _ _ _)
      ((prob_true _ _).symm.le.trans (prob_mono (fun _ _ => ⟨trivial, hbad⟩) _ _))
  have hlt := Bridge.deltaPub_deployed_lt_one (n := n) (kd := kd) (K₀ := K₀) hmiss hW
  exact not_le.2 (ENNReal.ofReal_lt_one.2 hlt) (h1.symm.le.trans h)

end Flock.Guarantees
