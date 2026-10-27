import Proofs.Flock.Bridge.ScopeGate
import Proofs.Flock.Bridge.RefusedWitnesses

/-!
# `SampledKSLiveV4`'s nontrivial lemma (LEAN-66)

It reads `SampledKSLivePub`'s failing instance (`Bridge.toy_exceeds`, `Guarantees.sampledKSLivePub_nontrivial`) through
`SampledKSLiveV3`'s (`sampledKSLiveV3_nontrivial`), with no instance of its own.

* `SampledKSLiveV4All`: `SampledKSLiveV4` with both of its acceptances (`Bridge.LiveAcceptsScoped` and
  `Bridge.RunPopOk`) replaced by `True`.
* `sampledKSLiveV3All_of_v4All`: it implies `SampledKSLiveV3All`. At V3's files and draw, `Bridge.KsScoped` holds, so
  V4 is V3's bound on an event with the same claims relation.
* `sampledKSLiveV4_nontrivial`: so `SampledKSLiveV4All` is false at `SampledKSLiveV3`'s failing instance.
-/

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

open Classical in
/-- **`SampledKSLiveV4` at the verifiers that accept everything**: its statement with `Bridge.LiveAcceptsScoped I o`
and `Bridge.RunPopOk desc qr n` replaced by `True`. -/
def SampledKSLiveV4All : Prop :=
  ∀ (n : ℕ) (Reg : Type) (I : Inputs)
    (D : Flock.Desc) (hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D)
    (pub : Public) (hpub : HmOut.loadPublic (zkTags I) D.unroll I.publicFile = .ok pub)
    (_hn : pub.n = n)
    (desc : Claims.Msg) (qr : Lean.Json)
    (kd : ℕ) (_hkd : kd ≤ n)
    (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : DrawSetupZHJ (inputsJ I 0) (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf D.unroll.g (kd / nTab I)) → Discharge.Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    (_hA3 : Assumptions.UniformRandomBytes Lb src)
    (hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr kd)
    (k Rw M : ℕ) (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) (ρ : ℝ) (_hρ : ρ < 1)
    (_hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) (t' : ℝ) (_ht' : 0 ≤ t') (K₀ : ℕ) (_hK₀ : K₀ ≤ n)
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    (v : ℝ)
    (q : Claims.Sampling.Query),
    if hs : Bridge.KsScoped I D pub kd then
      let L := Law.execOS (n := n) src (.subset kd) n
      let P := (keyProg D.unroll hs.built n).partition (Discharge.Layout.outsOf D.unroll hs.built n)
      let vb := (rsZHJ dj y₀ hs.typed (Flock.Desc.parse_flat hD) hs.scope hs.built hpub hs.tables).bindingZ
        (rsZHJ_computes dj y₀ hs.typed (Flock.Desc.parse_flat hD) hs.scope hs.built hpub hs.tables)
      let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
        (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
        (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
          (runsZCJH_live I hs.typed (Flock.Desc.parse_flat hD)
            (Discharge.Layout.scopeOkH_spec hs.built hs.scope).2.1 dj y₀ wr _ hs.tags hCust.1
            fun _ => drawOkZKHJ_of_file I hs.typed (Flock.Desc.parse_flat hD) kd n hs.div dj hCust.2 _)) τ
      let Xr := XplurZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) k Rw
      let δ : ℝ := ((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ)
      let Cc := (keyProg D.unroll hs.built n).circuit (Discharge.Layout.outsOf D.unroll hs.built n)
      let rows : Fin Cc.N → Option Bridge.RowAt := fun g =>
        (FlockSoundness.Discharge.HiddenExec.rowOfH D.unroll hs.built n
          (Discharge.Layout.outsOf D.unroll hs.built n) g).map fun r =>
          ⟨r.2.1.val, r.1.val, Discharge.Layout.portRowW D.unroll r.2.1, r.2.2.val⟩
      let roots := Bridge.rootsOf D.unroll pub n
      let Cd := Bridge.commitDep Cc rows D.unroll n roots
      let κ : Fin Cc.N → Option Bool := fun g => Bridge.constsOf D.unroll hs.built n g
      (∀ ω j, FlockSoundness.Assumptions.TableCRZAt execArith hs512 (planZHJ dj y₀ (L.draw ω) (reg σ)) j
        (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) (2 * (t' + v)) (t' + v)) →
      FlockSoundness.Assumptions.LinkCRZCAt execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ)
        k Rw M t' →
      prob (fun _ => (True ∧ True) ∧
          ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg Cd δ (Bridge.stmtOf P κ kd desc q roots) (Bridge.partOf P)
            (Bridge.witDep Cc Cd Claims.Sampling.sha512Msg roots Xr))
          (auditLive L Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ ≤
        ((n - K₀).choose kd : ℝ≥0∞) / (n.choose kd : ℝ≥0∞) +
          ksAvgStrictZ (planZHJ dj y₀) L k (2 * (t' + v)) (t' + v) (reg σ) +
          ENNReal.ofReal ((∑ p : vb.Pos,
              (Fintype.card (ΩcZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb L p) : ℝ) / Fintype.card L.Ω) *
            (2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ))
    else
      prob (fun _ => True ∧ True)
        (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)) τ = 0

/-- **`SampledKSLiveV4All` gives `SampledKSLiveV3All`**: V3's files and draw are in `Bridge.KsScoped`, where V4's bound
is V3's, on an event with the same claims relation. -/
theorem sampledKSLiveV3All_of_v4All (h : SampledKSLiveV4All) : SampledKSLiveV3All := by
  intro n Reg I ht D hD hU pub hpub hpt hscope hn desc qr kd hkd hdiv dj S₀ d₀ y₀ wr dg N Ω _ _ Lb src hA3 hTags
    hCust k Rw M hk hRw hM ρ hρ hr t' ht' K₀ hK₀ τ v q L P vb σ Xr δ Cc rows roots Cd κ hT hL
  have hs : Bridge.KsScoped I D pub kd := ⟨ht, hU, hpt, hscope, hdiv, hTags⟩
  have h4 := h n Reg I D hD pub hpub hn desc qr kd hkd dj S₀ d₀ y₀ wr dg N Ω Lb src hA3 hCust k Rw M hk hRw hM ρ hρ hr
    t' ht' K₀ hK₀ τ v q
  split at h4
  · exact (prob_mono (fun _ ho => ⟨⟨trivial, trivial⟩, ho.2⟩) _ τ).trans (h4 hT hL)
  · exact absurd hs ‹_›

open Classical in
/-- **`SampledKSLiveV4`'s nontrivial lemma** (LEAN-66): `SampledKSLiveV4` at the verifiers that accept everything is
false at `SampledKSLivePub`'s failing instance, since it implies `SampledKSLiveV3All` (`sampledKSLiveV3All_of_v4All`),
which is false there (`sampledKSLiveV3_nontrivial`). -/
theorem sampledKSLiveV4_nontrivial :
  ∀ (n : ℕ) (Reg : Type)
    (I : Inputs) (ht : I.tags.typed = false)
    (D : Flock.Desc) (hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D) (hU : Built D.unroll.unit)
    (pub : Public) (hpub : HmOut.loadPublic (zkTags I) D.unroll I.publicFile = .ok pub) (hpt : pub.tables = none)
    (hscope : Discharge.Layout.scopeOkH D.unroll = true) (_hn : pub.n = n)
    (desc : Claims.Msg) (qr : Lean.Json)
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
    (t' v : ℝ)
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
    let Cc := (keyProg D.unroll hU n).circuit (Discharge.Layout.outsOf D.unroll hU n)
    kd = n → 1 ≤ K₀ → D.unroll.ports.size = 0 → (∃ g ∈ Cc.outputs, (Cc.op g).isInput = false) →
    Bridge.ioSum P ≤ 2 ^ 40 → 0 ≤ t' → t' ≤ 2 ^ 80 → 0 ≤ v → v ≤ 2 ^ 80 →
    (∀ ω j, FlockSoundness.Assumptions.TableCRZAt execArith hs512 (planZHJ dj y₀ (L.draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZHJ dj y₀) (cont σ ω)) (2 * (t' + v)) (t' + v)) →
    FlockSoundness.Assumptions.LinkCRZCAt execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ)
      Bridge.kDep (2 ^ 192) (2 ^ 192) t' →
    ¬ SampledKSLiveV4All := by
  intro n Reg I ht D hD hU pub hpub hpt hscope hn desc qr kd hkd hdiv dj S₀ d₀ y₀ wr dg N Ω _ _ Lb
    src hA3 hTags hCust K₀ hK₀ τ t' v q L P vb σ Cc hkdn hK1 hports hout hW ht0 ht80 hv0 hv80 hCR hLink hAll
  exact sampledKSLiveV3_nontrivial n Reg I ht D hD hU pub hpub hpt hscope hn desc qr kd hkd hdiv dj S₀ d₀ y₀ wr dg N Ω
    Lb src hA3 hTags hCust K₀ hK₀ τ t' v q hkdn hK1 hports hout hW ht0 ht80 hv0 hv80 hCR hLink
    (sampledKSLiveV3All_of_v4All hAll)

end Flock.Guarantees
