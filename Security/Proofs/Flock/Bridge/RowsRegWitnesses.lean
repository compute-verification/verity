import Proofs.Flock.Bridge.RowsReg
import Proofs.Flock.Bridge.PublicWitnesses
import Proofs.Flock.Bridge.PublicNontrivial

/-!
# `SampledKSLiveV2`'s and `SampledKSLiveV3`'s honest and nontrivial lemmas

* **Honest**: their hypotheses on the analysis' parameters are `SampledKSLiveV2Z`'s, which hold as deployed
  (`sampledKSLiveV2_honest`), and their file hypotheses hold at the deployed session (`SampledKSLiveHonest`).
* **Nontrivial** (`sampledKSLiveV2_nontrivial`, `sampledKSLiveV3_nontrivial`): each statement at the verifier that
  accepts everything (`SampledKSLiveV2All`, `SampledKSLiveV3All`) is false at the toy session of
  `Flock.Test.ToyPubFiles`, where `SampledKSLivePub`'s is (`sampledKSLivePub_nontrivial`), under the same two interim
  premises (`Bridge.Assumptions.ToyFilesParse`, `Assumptions.ToySetupPub`). V2's bound is at most Pub's
  (`sampledKSLiveV2_le_pub`), so V2's statement at the verifier that accepts everything gives Pub's
  (`pubAll_of_v2All`); V3's is refuted at the toy as Pub's is, its bound at most Pub's there.
-/

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

open Classical in
/-- **`SampledKSLiveV2` at the verifier that accepts everything**: its statement with
`Discharge.ZkReg.LiveAcceptsZKR I own o` replaced by `True`. -/
def SampledKSLiveV2All : Prop :=
  ∀ (n : ℕ) (Reg : Type)
    -- F's registered roots: registered before the statement and its coin
    (own : Array Flock.Registered.Port)
    (I : Inputs) (ht : I.tags.typed = false)
    (D : Flock.Desc) (hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D) (hU : Built D.unroll.unit)
    (pub : Public) (hpub : HmOut.loadPublic (zkTags I) D.unroll I.publicFile = .ok pub) (hpt : pub.tables = none)
    (hscope : Discharge.Layout.scopeOkH D.unroll = true)
    (_hrows : Bridge.RowsOk D.unroll)
    (_hdup : Flock.Registered.dupName D.unroll = none) (_hroots : Bridge.PortRoots.rootsColl D.unroll pub n = none)
    -- the statement's descriptor, and the registration's query, whose population is the game's n units in order
    (desc : Claims.Msg) (qr : Lean.Json) (_hpop : Bridge.PopMatches desc qr n)
    (kd : ℕ) (_hkd : kd ≤ n) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : DrawSetupZHJ (inputsJ I 0) (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀)
    -- the registrant's opening of the session's registered reads
    (rd : FlockSoundness.Registered.ReadsX
      (layShHJ (Reg := Reg) dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU (pub := pub))
      (Discharge.ZkReg.rsShHJS dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub))
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf D.unroll.g (kd / nTab I)) → Discharge.Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the verifier's operating-system bytes, and A3
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    (_hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the verify command's tags, and live-verifier
    (hTags : TagsOkZHJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr kd)
    (k Rw M : ℕ) (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) (ρ : ℝ) (_hρ : ρ < 1)
    (_hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) (K₀ : ℕ) (_hK₀ : K₀ ≤ n)
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    -- one play of the audit, in SHA-512 evaluations
    (qR : ℝ)
    -- what the sampling statement carries and `RelS` doesn't read
    (q : Claims.Sampling.Query),
    let L := Law.execOS (n := n) src (.subset kd) n
    let G := auditLive L Reg (liveOfZJ dg (planZHJ dj y₀) N wr)
    let P := (keyProg D.unroll hU n).partition (Discharge.Layout.outsOf D.unroll hU n)
    let vb := (rsZHJ dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt).bindingZ
      (rsZHJ_computes dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt)
    let wo : vb.Pos → vb.Opening := fun p => ((p, rd.wv p, rd.ws p) : vb.Opening)
    let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJH_live I ht (Flock.Desc.parse_flat hD) (Discharge.Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ hTags
          hCust.1
          fun _ => drawOkZKHJ_of_file I ht (Flock.Desc.parse_flat hD) kd n hdiv dj hCust.2 _)) τ
    let Xr := XplurZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) k Rw
    let δ : ℝ := ((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ)
    -- the deployed commitment's layout: the game's rows, read off the circuit file
    let Cc := (keyProg D.unroll hU n).circuit (Discharge.Layout.outsOf D.unroll hU n)
    let rows : Fin Cc.N → Option Bridge.RowAt := fun g =>
      (FlockSoundness.Discharge.HiddenExec.rowOfH D.unroll hU n (Discharge.Layout.outsOf D.unroll hU n) g).map fun r =>
        ⟨r.2.1.val, r.1.val, Discharge.Layout.portRowW D.unroll r.2.1, r.2.2.val⟩
    -- the registration's frame-v3 roots, over the public file's rows
    let roots := Bridge.rootsOf D.unroll pub n
    let Cd := Bridge.commitDep Cc rows D.unroll n roots
    -- the circuit's constants, public
    let κ : Fin Cc.N → Option Bool := fun g => Bridge.constsOf D.unroll hU n g
    -- the claims' values at an outcome: the registrant's row's bit at a registered read, where the registrant's leaves
    -- are the registered ones or the verifier owns the reads at the outcome's draw, and the evaluation on the extracted
    -- plurality's inputs at every other wire
    let vreg : Reg × Finset (Fin n) × Discharge.Exec.Transcript → Fin Cc.N → Bool := fun o g =>
      if (¬ rd.Off o.1 ∨ Discharge.ZkReg.OwnReadsHJ I own D.unroll pub o.2.2 rd.isRead rd.port rd.readAt) ∧
          rd.isRead (vb.pos g) then
        vb.bit (vb.val (wo (vb.pos g))) g
      else Cc.eval Xr g
    -- the registered reads' finder: it plays the audit once and outputs the registered check's pair at an off read
    let BR := Bridge.strictComp (StrictCR.oneRun G τ fun o =>
        rd.pick o.1 (Discharge.ZkReg.regPathsHJ I D.unroll pub o.2.2 rd.port)) qR
    prob (fun o => True ∧
        ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg Cd δ (Bridge.stmtOf P κ kd desc q roots) (Bridge.partOf P κ)
          (Bridge.witRows Cc Cd Claims.Sampling.sha512Msg roots (vreg o)))
        G τ ≤
      ((n - K₀).choose kd : ℝ≥0∞) / (n.choose kd : ℝ≥0∞) +
        ksAvgPrZ execArith hs512 (planZHJ dj y₀) k (reg σ) (cont σ) +
        ENNReal.ofReal ((∑ p : vb.Pos,
            (Fintype.card (ΩcZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb L p) : ℝ) / Fintype.card L.Ω *
            (linkFinderPrZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) p k Rw M +
              1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw))) / (1 - ρ)) +
        rowsPrZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb
          (fun c : vb.Pos => ¬ rd.Off (reg σ) ∧ rd.isRead c) wo k ρ (reg σ) (cont σ) +
        ENNReal.ofReal (BR.Pr Claims.Sampling.sha512Msg)

open Classical in
/-- **`SampledKSLiveV3` at the verifier that accepts everything**: its statement with
`Bridge.LiveAcceptsKSR I own o ∧ Bridge.RunPopOk desc qr n` replaced by `True`. -/
def SampledKSLiveV3All : Prop :=
  ∀ (n : ℕ) (Reg : Type)
    -- F's registered roots: registered before the statement and its coin
    (own : Array Flock.Registered.Port)
    (I : Inputs) (ht : I.tags.typed = false)
    (D : Flock.Desc) (hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D) (hU : Built D.unroll.unit)
    (pub : Public) (hpub : HmOut.loadPublic (zkTags I) D.unroll I.publicFile = .ok pub) (hpt : pub.tables = none)
    (hscope : Discharge.Layout.scopeOkH D.unroll = true)
    -- the game's units are the public file's population
    (_hn : pub.n = n)
    -- the statement's descriptor, and the registration's query
    (desc : Claims.Msg) (qr : Lean.Json)
    (kd : ℕ) (_hkd : kd ≤ n) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : DrawSetupZHJ (inputsJ I 0) (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀)
    -- the registrant's opening of the session's registered reads
    (rd : FlockSoundness.Registered.ReadsX
      (layShHJ (Reg := Reg) dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU (pub := pub))
      (Discharge.ZkReg.rsShHJS dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub))
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf D.unroll.g (kd / nTab I)) → Discharge.Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the verifier's operating-system bytes, and A3
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    (_hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the verify command's tags, and live-verifier
    (hTags : TagsOkZHJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr kd)
    (k Rw M : ℕ) (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) (ρ : ℝ) (_hρ : ρ < 1)
    (_hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ) (K₀ : ℕ) (_hK₀ : K₀ ≤ n)
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.subset kd) n) Reg (liveOfZJ dg (planZHJ dj y₀) N wr)))
    -- one play of the audit, in SHA-512 evaluations
    (qR : ℝ)
    -- what the sampling statement carries and `RelS` doesn't read
    (q : Claims.Sampling.Query),
    let L := Law.execOS (n := n) src (.subset kd) n
    let G := auditLive L Reg (liveOfZJ dg (planZHJ dj y₀) N wr)
    let P := (keyProg D.unroll hU n).partition (Discharge.Layout.outsOf D.unroll hU n)
    let vb := (rsZHJ dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt).bindingZ
      (rsZHJ_computes dj y₀ ht (Flock.Desc.parse_flat hD) hscope hU hpub hpt)
    let wo : vb.Pos → vb.Opening := fun p => ((p, rd.wv p, rd.ws p) : vb.Opening)
    let σ := simZJ execArith hs512 (planZHJ dj y₀) I wr (opnAtZHJ (planZHJ dj y₀) I dj) dg (decAtZJ (planZHJ dj y₀))
      (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ I wr _ _ _ _ _
        (runsZCJH_live I ht (Flock.Desc.parse_flat hD) (Discharge.Layout.scopeOkH_spec hU hscope).2.1 dj y₀ wr _ hTags
          hCust.1
          fun _ => drawOkZKHJ_of_file I ht (Flock.Desc.parse_flat hD) kd n hdiv dj hCust.2 _)) τ
    let Xr := XplurZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) k Rw
    let δ : ℝ := ((n - K₀).choose kd : ℝ) / (n.choose kd : ℝ)
    -- the deployed commitment's layout: the game's rows, read off the circuit file
    let Cc := (keyProg D.unroll hU n).circuit (Discharge.Layout.outsOf D.unroll hU n)
    let rows : Fin Cc.N → Option Bridge.RowAt := fun g =>
      (FlockSoundness.Discharge.HiddenExec.rowOfH D.unroll hU n (Discharge.Layout.outsOf D.unroll hU n) g).map fun r =>
        ⟨r.2.1.val, r.1.val, Discharge.Layout.portRowW D.unroll r.2.1, r.2.2.val⟩
    -- the registration's frame-v3 roots, over the public file's rows
    let roots := Bridge.rootsOf D.unroll pub n
    let Cd := Bridge.commitDep Cc rows D.unroll n roots
    -- the circuit's constants, public
    let κ : Fin Cc.N → Option Bool := fun g => Bridge.constsOf D.unroll hU n g
    -- the claims' values at an outcome, V2's
    let vreg : Reg × Finset (Fin n) × Discharge.Exec.Transcript → Fin Cc.N → Bool := fun o g =>
      if (¬ rd.Off o.1 ∨ Discharge.ZkReg.OwnReadsHJ I own D.unroll pub o.2.2 rd.isRead rd.port rd.readAt) ∧
          rd.isRead (vb.pos g) then
        vb.bit (vb.val (wo (vb.pos g))) g
      else Cc.eval Xr g
    -- the registered reads' finder, V2's
    let BR := Bridge.strictComp (StrictCR.oneRun G τ fun o =>
        rd.pick o.1 (Discharge.ZkReg.regPathsHJ I D.unroll pub o.2.2 rd.port)) qR
    prob (fun o => True ∧
        ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg Cd δ (Bridge.stmtOf P κ kd desc q roots) (Bridge.partOf P κ)
          (Bridge.witRows Cc Cd Claims.Sampling.sha512Msg roots (vreg o)))
        G τ ≤
      ((n - K₀).choose kd : ℝ≥0∞) / (n.choose kd : ℝ≥0∞) +
        ksAvgPrZ execArith hs512 (planZHJ dj y₀) k (reg σ) (cont σ) +
        ENNReal.ofReal ((∑ p : vb.Pos,
            (Fintype.card (ΩcZ execArith P (planZHJ dj y₀) (tabZHJ dj y₀) vb L p) : ℝ) / Fintype.card L.Ω *
            (linkFinderPrZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb (reg σ) (cont σ) p k Rw M +
              1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw))) / (1 - ρ)) +
        rowsPrZC execArith hs512 P (planZHJ dj y₀) (tabZHJ dj y₀) vb
          (fun c : vb.Pos => ¬ rd.Off (reg σ) ∧ rd.isRead c) wo k ρ (reg σ) (cont σ) +
        ENNReal.ofReal (BR.Pr Claims.Sampling.sha512Msg)

/-- **`SampledKSLiveV2` at the verifier that accepts everything gives `SampledKSLivePub` there**: V2's bound is at
most Pub's (`sampledKSLiveV2_le_pub`), and Pub's rate cap `rateM1 k ≤ ρ` gives V2's rate hypothesis
(`rateZ_le_m1w`, `planZHJ_M1W`). -/
theorem pubAll_of_v2All (h : SampledKSLiveV2All) : SampledKSLivePubAll := by
  intro n Reg own I ht D hD hU pub hpub hpt hscope hrows hdup hroots desc qr hpop kd hkd hdiv dj S₀ d₀ y₀ rd wr dg N Ω
    _ _ Lb src hA3 hTags hCust k Rw M hk hRw hM ρ hρ hρk t' K₀ hK₀ τ v qR q
  have hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ := fun S R j =>
    (rateZ_le_m1w _ j (planZHJ_M1W dj y₀ S R j) k).trans hρk
  exact (h n Reg own I ht D hD hU pub hpub hpt hscope hrows hdup hroots desc qr hpop kd hkd hdiv dj S₀ d₀ y₀ rd wr dg N
    Ω Lb src hA3 hTags hCust k Rw M hk hRw hM ρ hρ hr K₀ hK₀ τ qR q).trans
    (sampledKSLiveV2_le_pub n Reg own I ht D hD hU pub hpub hpt hscope kd hkd hdiv dj S₀ d₀ y₀ rd wr dg N Ω Lb src hTags
      hCust k Rw M hRw ρ hρ hρk t' K₀ τ v qR)

/-- **`SampledKSLiveV2`'s nontrivial lemma**: `SampledKSLiveV2` at the verifier that accepts everything is false at the
toy session's files and setup, the two interim premises of `sampledKSLivePub_nontrivial`, since it gives
`SampledKSLivePub` there (`pubAll_of_v2All`). -/
theorem sampledKSLiveV2_nontrivial (hf : Bridge.Assumptions.ToyFilesParse) (hs : Assumptions.ToySetupPub) :
    ¬ SampledKSLiveV2All :=
  fun h => sampledKSLivePub_nontrivial hf hs (pubAll_of_v2All h)

open Flock.Test.ToyPubFiles (circuit pub progDesc progQuery)
open Flock.Assumptions (toyPubI toyPubI_typed toyPubI_tags toyPubI_div ToyΩ toyLb toySrc toySrc_uniform)
open Flock.Bridge.ToyPub (gOut xIn)

attribute [local irreducible] Flock.Test.ToyPubFiles.circuit in
open Classical in
/-- **`SampledKSLiveV3`'s nontrivial lemma**: `SampledKSLiveV3` at the verifier that accepts everything is false at
the toy session's files and setup, as `SampledKSLivePub`'s is (`sampledKSLivePub_nontrivial`): the toy's population is
its one unit (`pub.n = 1`), the claim fails at every run, so `Pr[True ∧ ¬ RelS] = 1`, while V3's bound is at most Pub's
(`sampledKSLiveV2_le_pub`), below 1 (`Bridge.deltaPubR_lt_one`). -/
theorem sampledKSLiveV3_nontrivial (hf : Bridge.Assumptions.ToyFilesParse) (hs : Assumptions.ToySetupPub) :
    ¬ SampledKSLiveV3All := by
  intro hAll
  obtain ⟨hD0, hsc, hpub0, -, ⟨Dd, hDd, hDp⟩, ⟨dp, hdp, hidx⟩, -⟩ := hf
  have hD : Flock.Desc.parse (zkTags toyPubI) toyPubI.circuitFile toyPubI.tables = .ok (.single circuit) := hD0
  have hpub : HmOut.loadPublic (zkTags toyPubI) circuit toyPubI.publicFile = .ok pub := hpub0
  have hU : Built circuit.unit := FlockC3.VStarList.built_of_parse
    (FlockSoundness.Discharge.Hidden.parse_untyped ((zkTags_typed _).trans toyPubI_typed) (Flock.Desc.parse_flat hD)).1
  have hscope : Discharge.Layout.scopeOkH circuit = true := by
    rw [← FlockSoundness.Discharge.FailClosed.scopeOkH_eq]
    exact hsc
  have hpt : pub.tables = none := rfl
  obtain ⟨dj, S₀, d₀, y₀, wr, hrec, hCust, dg, N, τ, t', v, qR, hB⟩ := hs hD hU hpub hpt hscope
  obtain ⟨hk, hRw, hM, hρ, hρk⟩ := sampledKSLivePub_honest
  have hr : ∀ S R j, rateZ (planZHJ (Reg := Unit) dj y₀ S R) j Bridge.kDep ≤ 1 / 2 := fun S R j =>
    (rateZ_le_m1w _ j (planZHJ_M1W dj y₀ S R j) _).trans hρk
  let hc := Flock.Desc.parse_flat hD
  let rd := Bridge.ToyPub.lieReads (Reg := Unit) dj y₀ toyPubI_typed hc hscope hU hpub
  let own := Bridge.ToyPub.lieOwn (Reg := Unit) dj y₀ toyPubI_typed hc hscope hU hpub
  have h := hAll 1 Unit own toyPubI toyPubI_typed (.single circuit) hD hU pub hpub hpt hscope rfl progDesc.data.toList
    progQuery 1 le_rfl toyPubI_div dj S₀ d₀ y₀ rd wr dg N ToyΩ toyLb toySrc toySrc_uniform toyPubI_tags hCust Bridge.kDep
    (2 ^ 192) (2 ^ 192) hk hRw hM (1 / 2) hρ hr 1 le_rfl τ qR ⟨"", 0, []⟩
  have hcmp := sampledKSLiveV2_le_pub 1 Unit own toyPubI toyPubI_typed (.single circuit) hD hU pub hpub hpt hscope 1
    le_rfl toyPubI_div dj S₀ d₀ y₀ rd wr dg N ToyΩ toyLb toySrc toyPubI_tags hCust Bridge.kDep (2 ^ 192) (2 ^ 192) hRw
    (1 / 2) hρ hρk t' 1 τ v qR
  have hlt := Bridge.deltaPubR_lt_one (n := 1) (kd := 1) (K₀ := 1) one_pos le_rfl (by norm_num)
    (Bridge.ToyPub.ioSum_le hU) (mul_nonneg (mul_nonneg (sq_nonneg _) (Nat.cast_nonneg _)) (Claims.Pr_nonneg _)) hB
  have hown : ∀ t : Discharge.Exec.Transcript, recordDraw t.record = some Flock.Test.ToyPubFiles.draw →
      Discharge.ZkReg.OwnReadsHJ toyPubI own circuit pub t rd.isRead rd.port rd.readAt := fun t ht =>
    Bridge.ToyPub.ownReads_lie dj y₀ toyPubI_typed hc hscope hU hpub le_rfl le_rfl ht hDd hDp hdp hidx
      Bridge.ToyPub.toy_names
  obtain ⟨ux, kx, hx⟩ := Bridge.ToyPub.rowOf_xIn hU
  refine absurd (h.trans hcmp) (not_le.2 (Bridge.ToyPub.exceeds_of_outcomes τ (fun ω o' => ?_) hlt))
  have ho := hown _ (hrec ((Law.execOS (n := 1) toySrc (.subset 1) 1).draw ω) τ.1 o')
  refine Bridge.not_relS_row _ 1 rfl _ _ _ _ _ _ (by positivity) (Bridge.ToyPub.gOut_mem hU)
    (Bridge.ToyPub.unit_gOut hU) (Bridge.ToyPub.op_gOut hU) (Bridge.ToyPub.op_xIn hU) (Bridge.ToyPub.consts_xIn hU)
    ?_ ?_
  · beta_reduce
    split
    · exact (Bridge.ToyPub.bit_lie hx []).trans rfl
    · rename_i hn
      exact (hn ⟨Or.inr ho, ux, _, Bridge.vbOf_pos_row hx⟩).elim
  · intro d hd
    have hr := Bridge.commitDep_rows_of_opens _ _ _ _ _ _ _ _ hd
    obtain ⟨⟨ug, qg, kg⟩, hrow⟩ := Option.ne_none_iff_exists'.1 fun h => hr (Option.map_eq_none_iff.2 h)
    have hq := Bridge.ToyPub.rowOf_gOut hU hrow
    have hrow' : FlockSoundness.Discharge.HiddenExec.rowOfH circuit hU 1
        (FlockSoundness.Discharge.Layout.outsOf circuit hU 1) (gOut hU) = some (ug, ⟨qg, kg⟩) := hrow
    beta_reduce
    split
    · exact (Bridge.ToyPub.bit_lie hrow' []).trans (decide_eq_true hq)
    · rename_i hn
      exact (hn ⟨Or.inr ho, ug, qg, Bridge.vbOf_pos_row hrow⟩).elim

end Flock.Guarantees
