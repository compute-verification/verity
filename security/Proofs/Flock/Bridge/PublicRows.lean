module

public import Proofs.Flock.Bridge.PublicDeployed
public import Proofs.Flock.Bridge.QsRows
import Proofs.Flock.Bridge.KSLive

/-! `SampledKSLivePub`'s error by the rows a sample reads: at most `k_d · n_p + 2` commit strings. -/

@[expose] public section

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

open Classical in
theorem _root_.Flock.Bridge.sampledKSLive_le_deltaPub_of_qs :
    ∀ (n : ℕ) (Reg : Type)
      (I : Inputs) (ht : I.tags.typed = false)
      (D : Flock.Desc) (hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D) (hU : Built D.unroll.unit)
      (pub : Public) (hpub : HmOut.loadPublic (zkTags I) D.unroll I.publicFile = .ok pub) (hpt : pub.tables = none)
      (hscope : Discharge.Layout.scopeOkH D.unroll = true)
      (_hrows : Bridge.RowsOk D.unroll)
      (_hdup : Flock.PortRoots.dupId D.unroll pub = false) (_hroots : Bridge.PortRoots.rootsColl D.unroll pub n = none)
      (desc : Claims.Msg) (qr : Lean.Json) (_hpop : Bridge.PopMatches desc qr n)
      (kd : ℕ) (_hkd : kd ≤ n) (hdiv : kd % nTab I = 0)
      (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
      (y₀ : SampleSetupZHJ (inputsJ I 0) (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀)
      (wr : (S : Finset (Fin n)) → (R : Reg) →
        LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf D.unroll.g (kd / nTab I)) → Discharge.Exec.Transcript)
      (dg : ByteArray → ByteArray) (N : ℕ)
      (Ω : Type) [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
      (_hA3 : Assumptions.UniformRandomBytes Lb src)
      (hTags : TagsOkZHJ I)
      (hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr kd)
      (k Rw M : ℕ) (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) (ρ : ℝ) (_hρ : ρ < 1)
      (_hρk : rateM1 k ≤ ρ) (t' : ℝ) (_ht' : 0 ≤ t') (K₀ : ℕ) (_hK₀ : K₀ ≤ n)
      (τ : Strategy (auditLive (SamplingDesign.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
      (v : ℝ) (_hv : 0 ≤ v)
      (q : Claims.Sampling.Query)
      (W : ℕ),
      let L := SamplingDesign.execOS (n := n) src (.subset kd) n
      let P := (keyProg D.unroll hU n).partition (Discharge.Layout.outsOf D.unroll hU n)
      let vb := (rsZHJ dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt).bindingZ
        (rsZHJ_computes dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt)
      let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
        (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
        (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
          (runsZCJH_live I ht (Flock.Desc.parse_flat hD) (Discharge.Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ hTags
            hCust.1
            fun _ => sampleOkZKHJ_of_file I ht (Flock.Desc.parse_flat hD) kd n hdiv dj hCust.2 _)) τ
      let Xr := XplurZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) k Rw
      let δ : ℝ := ((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ)
      let Cc := (keyProg D.unroll hU n).circuit (Discharge.Layout.outsOf D.unroll hU n)
      let rows : Fin Cc.N → Option Bridge.RowAt := fun g =>
        (FlockSoundness.Discharge.HiddenExec.rowOfH D.unroll hU n (Discharge.Layout.outsOf D.unroll hU n) g).map fun r =>
          ⟨r.2.1.val, r.1.val, Discharge.Layout.portRowW D.unroll r.2.1, r.2.2.val⟩
      let roots := Bridge.rootsOf D.unroll pub n
      let Cd := Bridge.commitDep Cc rows D.unroll pub n roots
      let κ : Fin Cc.N → Option Bool := fun g => Bridge.constsOf D.unroll hU n g
      (∑ p : vb.Pos, (Fintype.card (ΩcZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb L p) : ℝ) /
          Fintype.card L.Ω ≤ (W : ℝ)) →
      (∀ ω j, FlockSoundness.Assumptions.TableCRZAt execArith hs512 (planZHJ dj y₀ (L.draw ω) (reg σ)) j
        (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) (2 * (t' + v)) (t' + v)) →
      FlockSoundness.Assumptions.LinkCRZCAt execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ)
        k Rw M (t' + v) →
      prob (fun o => LiveAcceptsZK I o ∧
          ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg Cd δ (Bridge.stmtOf P κ (.subset kd) desc q roots)
            (Bridge.partOf P κ (Bridge.nsOf desc qr)) (Bridge.witDep Cc Cd Claims.Sampling.sha512Msg roots Xr))
          (auditLive L Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
        ENNReal.ofReal (Bridge.deltaPub n kd K₀ k (2 * (t' + v)) (t' + v) (t' + v) M Rw W ρ) := by
  intro n Reg I ht D hD hU pub hpub hpt hscope hrows hdup hroots desc qr hpop kd hkd hdiv dj S₀ d₀ y₀ wr dg N Ω _ _ Lb src
    hA3 hTags hCust k Rw M hk hRw hM ρ hρ hρk t' ht' K₀ hK₀ τ v hv q W L P vb σ Xr δ Cc rows roots Cd κ hQs hCR hLink
  have hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ := fun S R j =>
    (rateZ_le_m1w _ j (planZHJ_M1W dj y₀ S R j) k).trans hρk
  refine (Flock.SecurityProofs.SampledKSLiveV2 n Reg I ht D hD hU pub hpub hpt hscope hrows hdup hroots desc qr hpop kd
    hkd hdiv dj S₀ d₀ y₀ wr dg N Ω Lb src hA3 hTags hCust k Rw M hk hRw hM ρ hρ hr t' ht' K₀ hK₀ τ v hv q hCR
    hLink).trans ?_
  have h1ρ : 0 < 1 - ρ := sub_pos.2 hρ
  have hB : 0 ≤ 2 * ((t' + v) * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw) := by
    have : 0 ≤ (t' + v) * (1 + k) := mul_nonneg (add_nonneg ht' hv) (by positivity)
    positivity
  have hmiss : ((n - K₀).choose kd : ℝ≥0∞) / (n.choose kd : ℝ≥0∞) =
      ENNReal.ofReal (((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ)) := by
    have hc : (0 : ℝ) < (n.choose kd : ℝ) := by exact_mod_cast Nat.choose_pos hkd
    rw [ENNReal.ofReal_div_of_pos hc, ENNReal.ofReal_natCast, ENNReal.ofReal_natCast]
  have hks : ksAvgStrictZ (planZHJ dj y₀) L k (2 * (t' + v)) (t' + v) (reg σ) ≤
      ENNReal.ofReal (ksClosedM1 k (2 * (t' + v)) (t' + v) / (1 - ρ)) :=
    (ksAvgStrictZ_planZHJ_le_closed dj y₀ L k (2 * (t' + v)) (t' + v) (reg σ) (hρk.trans_lt hρ)).trans
      (ENNReal.ofReal_le_ofReal
        (div_le_div_of_nonneg_left (ksClosedM1_nonneg k (2 * (t' + v)) (t' + v)) h1ρ (by linarith)))
  have hlink : ENNReal.ofReal ((∑ p : vb.Pos,
        (Fintype.card (ΩcZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb L p) : ℝ) / Fintype.card L.Ω) *
      (2 * ((t' + v) * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ)) ≤
      ENNReal.ofReal ((W : ℝ) *
        (2 * ((t' + v) * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ)) :=
    ENNReal.ofReal_le_ofReal (div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_right hQs hB) h1ρ.le)
  have hx : (0 : ℝ) ≤ ((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ) := by positivity
  have hy : 0 ≤ ksClosedM1 k (2 * (t' + v)) (t' + v) / (1 - ρ) :=
    div_nonneg (ksClosedM1_nonneg k (2 * (t' + v)) (t' + v)) h1ρ.le
  have hz : 0 ≤ (W : ℝ) *
      (2 * ((t' + v) * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ) :=
    div_nonneg (mul_nonneg (Nat.cast_nonneg _) hB) h1ρ.le
  rw [hmiss]
  refine (add_le_add (add_le_add le_rfl hks) hlink).trans (le_of_eq ?_)
  unfold Bridge.deltaPub
  rw [ENNReal.ofReal_add (add_nonneg hx hy) hz, ENNReal.ofReal_add hx hy]

open Classical in
theorem _root_.Flock.Bridge.sampledKSLivePub_rows :
    ∀ (n : ℕ) (Reg : Type)
      (I : Inputs) (ht : I.tags.typed = false)
      (D : Flock.Desc) (hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D) (hU : Built D.unroll.unit)
      (pub : Public) (hpub : HmOut.loadPublic (zkTags I) D.unroll I.publicFile = .ok pub) (hpt : pub.tables = none)
      (hscope : Discharge.Layout.scopeOkH D.unroll = true)
      (_hrows : Bridge.RowsOk D.unroll)
      (_hdup : Flock.PortRoots.dupId D.unroll pub = false) (_hroots : Bridge.PortRoots.rootsColl D.unroll pub n = none)
      (desc : Claims.Msg) (qr : Lean.Json) (_hpop : Bridge.PopMatches desc qr n)
      (kd : ℕ) (_hkd : kd ≤ n) (hdiv : kd % nTab I = 0)
      (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
      (y₀ : SampleSetupZHJ (inputsJ I 0) (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀)
      (wr : (S : Finset (Fin n)) → (R : Reg) →
        LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf D.unroll.g (kd / nTab I)) → Discharge.Exec.Transcript)
      (dg : ByteArray → ByteArray) (N : ℕ)
      (Ω : Type) [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
      (_hA3 : Assumptions.UniformRandomBytes Lb src)
      -- the byte source holds the sample's first read, and the samples that run out add at most one commit string
      (_hLb : 2 * kd * Flock.Draw.width n + 64 ≤ Lb 0)
      (_hsz : kd * (n * D.unroll.ports.size) ≤ 2 ^ 64)
      (hTags : TagsOkZHJ I)
      (hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr kd)
      (k Rw M : ℕ) (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) (ρ : ℝ) (_hρ : ρ < 1)
      (_hρk : rateM1 k ≤ ρ) (t' : ℝ) (_ht' : 0 ≤ t') (K₀ : ℕ) (_hK₀ : K₀ ≤ n)
      (τ : Strategy (auditLive (SamplingDesign.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
      (v : ℝ) (_hv : 0 ≤ v)
      (q : Claims.Sampling.Query),
      let L := SamplingDesign.execOS (n := n) src (.subset kd) n
      let P := (keyProg D.unroll hU n).partition (Discharge.Layout.outsOf D.unroll hU n)
      let vb := (rsZHJ dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt).bindingZ
        (rsZHJ_computes dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt)
      let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
        (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
        (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
          (runsZCJH_live I ht (Flock.Desc.parse_flat hD) (Discharge.Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ hTags
            hCust.1
            fun _ => sampleOkZKHJ_of_file I ht (Flock.Desc.parse_flat hD) kd n hdiv dj hCust.2 _)) τ
      let Xr := XplurZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) k Rw
      let δ : ℝ := ((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ)
      let Cc := (keyProg D.unroll hU n).circuit (Discharge.Layout.outsOf D.unroll hU n)
      let rows : Fin Cc.N → Option Bridge.RowAt := fun g =>
        (FlockSoundness.Discharge.HiddenExec.rowOfH D.unroll hU n (Discharge.Layout.outsOf D.unroll hU n) g).map fun r =>
          ⟨r.2.1.val, r.1.val, Discharge.Layout.portRowW D.unroll r.2.1, r.2.2.val⟩
      let roots := Bridge.rootsOf D.unroll pub n
      let Cd := Bridge.commitDep Cc rows D.unroll pub n roots
      let κ : Fin Cc.N → Option Bool := fun g => Bridge.constsOf D.unroll hU n g
      (∀ ω j, FlockSoundness.Assumptions.TableCRZAt execArith hs512 (planZHJ dj y₀ (L.draw ω) (reg σ)) j
        (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) (2 * (t' + v)) (t' + v)) →
      FlockSoundness.Assumptions.LinkCRZCAt execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ)
        k Rw M (t' + v) →
      prob (fun o => LiveAcceptsZK I o ∧
          ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg Cd δ (Bridge.stmtOf P κ (.subset kd) desc q roots)
            (Bridge.partOf P κ (Bridge.nsOf desc qr)) (Bridge.witDep Cc Cd Claims.Sampling.sha512Msg roots Xr))
          (auditLive L Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
        ENNReal.ofReal (Bridge.deltaPub n kd K₀ k (2 * (t' + v)) (t' + v) (t' + v) M Rw
          (kd * D.unroll.ports.size + 3) ρ) := by
  intro n Reg I ht D hD hU pub hpub hpt hscope hrows hdup hroots desc qr hpop kd hkd hdiv dj S₀ d₀ y₀ wr dg N Ω _ _ Lb src
    hA3 hLb hsz hTags hCust k Rw M hk hRw hM ρ hρ hρk t' ht' K₀ hK₀ τ v hv q L P vb σ Xr δ Cc rows roots Cd κ hCR hLink
  have hQs0 := Bridge.qs_vbOf_execOS_le (dj := dj) (y₀ := y₀) (ht := ht) (hc := Flock.Desc.parse_flat hD)
    (hscope := hscope) (hU := hU) (hpub := hpub) (hpt := hpt) (Reg := Reg) Lb src hA3 kd hkd hLb
  have hout : (kd : ℝ) * (2 ^ 64 : ℝ)⁻¹ * ((n : ℝ) * D.unroll.ports.size) ≤ 1 := by
    have hsz' : (kd : ℝ) * ((n : ℝ) * D.unroll.ports.size) ≤ 2 ^ 64 := by exact_mod_cast hsz
    have h64 : (0 : ℝ) < 2 ^ 64 := by positivity
    calc (kd : ℝ) * (2 ^ 64 : ℝ)⁻¹ * ((n : ℝ) * D.unroll.ports.size)
        = (kd : ℝ) * ((n : ℝ) * D.unroll.ports.size) * (2 ^ 64 : ℝ)⁻¹ := by ring
      _ ≤ 2 ^ 64 * (2 ^ 64 : ℝ)⁻¹ := mul_le_mul_of_nonneg_right hsz' (inv_nonneg.2 h64.le)
      _ = 1 := mul_inv_cancel₀ h64.ne'
  have hQs : ∑ p : vb.Pos, (Fintype.card (ΩcZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb L p) : ℝ) /
      Fintype.card L.Ω ≤ ((kd * D.unroll.ports.size + 3 : ℕ) : ℝ) := by
    refine hQs0.trans ((add_le_add le_rfl hout).trans (le_of_eq ?_))
    push_cast
    ring
  exact Bridge.sampledKSLive_le_deltaPub_of_qs n Reg I ht D hD hU pub hpub hpt hscope hrows hdup hroots desc qr hpop kd
    hkd hdiv dj S₀ d₀ y₀ wr dg N Ω Lb src hA3 hTags hCust k Rw M hk hRw hM ρ hρ hρk t' ht' K₀ hK₀ τ v hv q
    (kd * D.unroll.ports.size + 3) hQs hCR hLink

open Classical in
theorem _root_.Flock.Bridge.sampledKSLivePub_rows_deployed :
    ∀ (n : ℕ) (Reg : Type)
      (I : Inputs) (ht : I.tags.typed = false)
      (D : Flock.Desc) (hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D) (hU : Built D.unroll.unit)
      (pub : Public) (hpub : HmOut.loadPublic (zkTags I) D.unroll I.publicFile = .ok pub) (hpt : pub.tables = none)
      (hscope : Discharge.Layout.scopeOkH D.unroll = true)
      (_hrows : Bridge.RowsOk D.unroll)
      (_hdup : Flock.PortRoots.dupId D.unroll pub = false) (_hroots : Bridge.PortRoots.rootsColl D.unroll pub n = none)
      (desc : Claims.Msg) (qr : Lean.Json) (_hpop : Bridge.PopMatches desc qr n)
      (kd : ℕ) (_hkd : kd ≤ n) (hdiv : kd % nTab I = 0)
      (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
      (y₀ : SampleSetupZHJ (inputsJ I 0) (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀)
      (wr : (S : Finset (Fin n)) → (R : Reg) →
        LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf D.unroll.g (kd / nTab I)) → Discharge.Exec.Transcript)
      (dg : ByteArray → ByteArray) (N : ℕ)
      (Ω : Type) [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
      (_hA3 : Assumptions.UniformRandomBytes Lb src)
      (_hLb : 2 * kd * Flock.Draw.width n + 64 ≤ Lb 0)
      (_hsz : kd * (n * D.unroll.ports.size) ≤ 2 ^ 64)
      (_hW : kd * D.unroll.ports.size + 3 ≤ 2 ^ 40)
      (hTags : TagsOkZHJ I)
      (hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr kd)
      (t' : ℝ) (_ht' : 0 ≤ t') (_ht80 : t' ≤ 2 ^ 80) (K₀ : ℕ) (_hK₀ : K₀ ≤ n)
      (τ : Strategy (auditLive (SamplingDesign.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
      (v : ℝ) (_hv : 0 ≤ v) (_hv80 : v ≤ 2 ^ 80)
      (q : Claims.Sampling.Query),
      let L := SamplingDesign.execOS (n := n) src (.subset kd) n
      let P := (keyProg D.unroll hU n).partition (Discharge.Layout.outsOf D.unroll hU n)
      let vb := (rsZHJ dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt).bindingZ
        (rsZHJ_computes dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt)
      let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
        (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
        (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
          (runsZCJH_live I ht (Flock.Desc.parse_flat hD) (Discharge.Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ hTags
            hCust.1
            fun _ => sampleOkZKHJ_of_file I ht (Flock.Desc.parse_flat hD) kd n hdiv dj hCust.2 _)) τ
      let Xr := XplurZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) Bridge.kDep (2 ^ 192)
      let δ : ℝ := ((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ)
      let Cc := (keyProg D.unroll hU n).circuit (Discharge.Layout.outsOf D.unroll hU n)
      let rows : Fin Cc.N → Option Bridge.RowAt := fun g =>
        (FlockSoundness.Discharge.HiddenExec.rowOfH D.unroll hU n (Discharge.Layout.outsOf D.unroll hU n) g).map fun r =>
          ⟨r.2.1.val, r.1.val, Discharge.Layout.portRowW D.unroll r.2.1, r.2.2.val⟩
      let roots := Bridge.rootsOf D.unroll pub n
      let Cd := Bridge.commitDep Cc rows D.unroll pub n roots
      let κ : Fin Cc.N → Option Bool := fun g => Bridge.constsOf D.unroll hU n g
      (∀ ω j, FlockSoundness.Assumptions.TableCRZAt execArith hs512 (planZHJ dj y₀ (L.draw ω) (reg σ)) j
        (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) (2 * (t' + v)) (t' + v)) →
      FlockSoundness.Assumptions.LinkCRZCAt execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ)
        Bridge.kDep (2 ^ 192) (2 ^ 192) (t' + v) →
      prob (fun o => LiveAcceptsZK I o ∧
          ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg Cd δ (Bridge.stmtOf P κ (.subset kd) desc q roots)
            (Bridge.partOf P κ (Bridge.nsOf desc qr)) (Bridge.witDep Cc Cd Claims.Sampling.sha512Msg roots Xr))
          (auditLive L Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
        ENNReal.ofReal (((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ) + 2 ^ (-(100 : ℝ))) := by
  intro n Reg I ht D hD hU pub hpub hpt hscope hrows hdup hroots desc qr hpop kd hkd hdiv dj S₀ d₀ y₀ wr dg N Ω _ _ Lb src
    hA3 hLb hsz hW hTags hCust t' ht' ht80 K₀ hK₀ τ v hv hv80 q L P vb σ Xr δ Cc rows roots Cd κ hCR hLink
  have hF : 2 * (t' + v) ≤ 2 ^ 82 := by linarith
  have hS : t' + v ≤ 2 ^ 82 := by linarith
  have hT : t' + v ≤ 2 ^ 81 := by linarith
  have h0 : 0 ≤ t' + v := add_nonneg ht' hv
  refine (Bridge.sampledKSLivePub_rows n Reg I ht D hD hU pub hpub hpt hscope hrows hdup hroots desc qr hpop kd hkd hdiv
    dj S₀ d₀ y₀ wr dg N Ω Lb src hA3 hLb hsz hTags hCust Bridge.kDep (2 ^ 192) (2 ^ 192) Bridge.one_le_kDep
    Nat.one_le_two_pow Nat.one_le_two_pow (1 / 2) (by norm_num) Bridge.rateM1_kDep t' ht' K₀ hK₀ τ v hv q hCR
    hLink).trans ?_
  exact ENNReal.ofReal_le_ofReal
    (Bridge.deltaPub_le_deployed n kd K₀ (kd * D.unroll.ports.size + 3) hW (by linarith) hF h0 hS h0 hT)

end Flock.Guarantees
