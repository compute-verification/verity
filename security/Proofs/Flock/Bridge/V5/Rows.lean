module

public import Proofs.Flock.Bridge.V5.KSLive
public import Proofs.Flock.Bridge.V5.QsRows

/-!
# `SampledKSLiveV5`'s error by the rows and public gates a sample reads

`SampledKSLiveV5`'s third term is `Q_s · (…) / (1 − ρ)`, `Q_s = Σ_c |Ω_c|/|Ω|` the average number of commit strings a
sample reads. At the session's value binding at public inputs a sample of `k_d` proof blocks reads at most
`k_d · r + 2` commit strings, `r = n_p + pubBits` the rows and the public gates of a proof block, unless the live
sample runs out of bytes (`qs_vbP_execSel_le`).

* `sampledKSLiveV5_rows`: `SampledKSLiveV5`'s statement with `k_d · r + 3` in place of `Q_s`, at the sampling design
  the verifier samples under (subset, stratified or work), under two more hypotheses: the verifier's byte source holds
  the stages' first reads, `firstRead law n ≤ Lb 0`, and `k_d · n · r ≤ 2^64`, so that the samples that run out add at
  most one commit string to the average.
-/

@[expose] public section

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

/-- **The sampling design the verifier samples under is on the registration's strata**: `designAt` is `drawnLaw`'s
(`lawOn_drawsUnder`) or `subset 0`, which samples no proof block. -/
theorem _root_.Flock.Bridge.lawOn_designAt {desc : Claims.Msg} {qr lj : Lean.Json} {tb : Option Lean.Json} {n : ℕ}
    (hpop : Bridge.PopMatches desc qr n) :
    Bridge.LawOn (Bridge.designAt desc qr lj tb n) (Bridge.nsOf desc qr)
      (Bridge.lawK (Bridge.designAt desc qr lj tb n)) := by
  rcases Bridge.designAt_cases desc qr lj tb n with hlaw | h0
  · exact Bridge.lawOn_drawsUnder hpop hlaw (Bridge.notBernoulli_drawsUnder hlaw)
  · rw [h0]
    exact ⟨rfl, Nat.zero_le _⟩

open Classical in
/-- **The sampled audit's knowledge soundness, live, at the verifier's public inputs, with the binding term by the rows
and public gates a sample reads.** `SampledKSLiveV5`'s setting, hypotheses and named assumptions, with two more
hypotheses: the verifier's byte source holds the stages' first reads of the sampling design it samples under,
`firstRead law n ≤ Lb 0`, and `k_d · n · r ≤ 2^64`, `r = n_p + pubBits` the rows (`D.unroll.ports.size`) and the public
gates (`Bridge.pubBits`) of a proof block. The bound is `SampledKSLiveV5`'s with `k_d · r + 3` in place of the average
number of commit strings a sample reads: a sample of `k_d` proof blocks reads their rows, their public gates, the
zero's gate and the constant's (`k_d · r + 2`), and the samples that run out of bytes, of probability at most
`k_d · 2⁻⁶⁴`, read at most `n · r` more, one more on average. -/
theorem _root_.Flock.Bridge.sampledKSLiveV5_rows :
    ∀ (n : ℕ) (Reg : Type)
      (I : Inputs) (own : Option ByteArray) (ht : I.tags.typed = false)
      (D : Flock.Desc) (hD : Flock.Desc.parseStmt (zkTags I) I.circuitFile I.tables = .ok D),
      let hU : Built D.unroll.unit := Discharge.Hidden.built_of_parseStmt hD ((zkTags_typed I).trans ht)
      ∀ (pi : HmIn.Ports) (hpi : HmIn.parse D.unroll = .ok pi)
      (pub : Public) (hpub : HmIn.loadPublic (zkTags I) D.unroll pi I.publicFile = .ok pub)
      (hscope : Discharge.Layout.scopeOkH D.unroll = true)
      (_hrows : Bridge.RowsOk D.unroll)
      (_hdup : Flock.PortRoots.dupId D.unroll pub = false)
      (desc : Claims.Msg) (qr : Lean.Json) (_hpop : Bridge.PopMatches desc qr n)
      (lj : Lean.Json) (tb : Option Lean.Json),
      let law := Bridge.designAt desc qr lj tb n
      let kd := Bridge.lawK law
      ∀ (hdiv : kd % nTab I = 0)
      (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
      (y₀ : DrawSetupZHJP (inputsJ I 0) own (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀)
      (wr : (S : Finset (Fin n)) → (R : Reg) →
        LiveOutJ (planZHJP (Reg := Reg) dj y₀ S R).length (mPtsOf D.unroll.g (kd / nTab I)) →
          Discharge.Exec.Transcript)
      (dg : ByteArray → ByteArray) (N : ℕ)
      (Ω : Type) [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
      (_hA3 : Assumptions.UniformRandomBytes Lb src)
      -- the byte source holds the stages' first reads, and the samples that run out add at most one commit string
      (_hLb : Bridge.firstRead law n ≤ Lb 0)
      (_hsz : kd * (n * (D.unroll.ports.size + Bridge.pubBits D.unroll hU pi)) ≤ 2 ^ 64)
      (hCust : Assumptions.Zk.RecordCustodyZKJPLaw I own dj (fun S R => (planZHJP (Reg := Reg) dj y₀ S R).length) wr
        fun _ => law)
      (k Rw M : ℕ) (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) (ρ : ℝ) (_hρ : ρ < 1)
      (_hr : ∀ S R j, rateZ (planZHJP (Reg := Reg) dj y₀ S R) j k ≤ ρ) (t' : ℝ) (_ht' : 0 ≤ t')
      (B : Finset ℕ)
      (τ : Strategy (auditLive (Bridge.execSel (n := n) src law n) Reg (liveOfZJ dg (planZHJP dj y₀) N wr)))
      (qF qS : ℝ)
      (q : Claims.Sampling.Query),
      if hTags : TagsOkZHJ I then
        let L := Bridge.execSel (n := n) src law n
        let P := (keyProgP D.unroll pi hU n).partition (Discharge.Layout.outsOfP D.unroll hU n pi)
        let vb := (rsZHJP dj y₀ ht (Flock.Desc.parseStmt_stmt hD) hscope hU hpi hpub).bindingZ
          (rsZHJP_computes dj y₀ ht (Flock.Desc.parseStmt_stmt hD) hscope hU hpi hpub)
        let σ := simZJA execArith hs512 (planZHJP dj y₀) (AcceptsZKP I own) wr (opnAtZHJP (planZHJP dj y₀) I own dj) dg
          (decAtZJ (planZHJP dj y₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!)
          N
          (decodesZCJH_custodyP I own ht (Flock.Desc.parseStmt_stmt hD) law kd hdiv dj y₀ wr _ hTags hCust
            (Bridge.lawSized_size (Bridge.sized_designAt desc qr lj tb n) hCust.2)) τ
        let Xr := XplurZC execArith hs512 P (planZHJP dj y₀) (tabZHJP dj y₀) vb (reg σ) (cont σ) k Rw
        let Cc := (keyProgP D.unroll pi hU n).circuit (Discharge.Layout.outsOfP D.unroll hU n pi)
        let Lm := Bridge.LeafMap.ofPublic D.unroll pub n
        let rows : Fin Cc.N → Option Bridge.RowAt := fun g =>
          (Discharge.HiddenExec.rowOfHP D.unroll hU n pi (Discharge.Layout.outsOfP D.unroll hU n pi) g).map fun r =>
            ⟨r.2.1.val, Lm.ref r.2.1.val r.1.val, Discharge.Layout.portRowW D.unroll r.2.1, r.2.2.val⟩
        let roots := Bridge.rootsOfL D.unroll Lm
        let Cd := Bridge.commitDepL Cc rows D.unroll Lm roots
        let κ : Fin Cc.N → Option Bool := fun g => Bridge.constsOfP D.unroll hU n pi pub g
        let r := Properties.Spot.oneStageOf law
        let ns := Bridge.nsOf desc qr
        let δ : ℝ := (Bridge.designOf r).Pa (Bridge.partOf P κ ns) B
        (∀ ω j, TableCRZ execArith hs512 (planZHJP dj y₀ (L.draw ω) (reg σ)) j
          (zcStrat execArith hs512 (planZHJP dj y₀) (cont σ ω)) qF qS) →
        LinkCRZC execArith hs512 P (planZHJP dj y₀) (tabZHJP dj y₀) vb (reg σ) (cont σ) k Rw M t' →
        prob (fun o => LiveAcceptsZKP I own o ∧
            ¬ Claims.Sampling.RelS Verity.sha512Msg Cd δ (Bridge.stmtOf P κ r desc q roots)
              (Bridge.partOf P κ ns) (Bridge.witDep Cc Cd Verity.sha512Msg roots Xr))
            (auditLive L Reg (liveOfZJ dg (planZHJP dj y₀) N wr)) τ ≤
          ENNReal.ofReal δ +
            ksAvgStrictZ (planZHJP dj y₀) L k qF qS (reg σ) +
            ENNReal.ofReal (((kd * (D.unroll.ports.size + Bridge.pubBits D.unroll hU pi) + 3 : ℕ) : ℝ) *
              (2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ)) +
            Claims.ecrBound Verity.sha512Msg (Bridge.rootsFinderL D.unroll Lm)
              (Bridge.rootsCostL D.unroll Lm)
      else
        prob (fun o => Bridge.LiveAcceptsKSP I own o)
          (auditLive (Bridge.execSel (n := n) src law n) Reg (liveOfZJ dg (planZHJP dj y₀) N wr)) τ = 0 := by
  intro n Reg I own ht D hD hU pi hpi pub hpub hscope hrows hdup desc qr hpop lj tb law kd hdiv dj S₀ d₀ y₀ wr dg N Ω _
    _ Lb src hA3 hLb hsz hCust k Rw M hk hRw hM ρ hρ hr t' ht' B τ qF qS q
  have h := Flock.SecurityProofs.SampledKSLiveV5 n Reg I own ht D hD pi hpi pub hpub hscope hrows hdup desc qr hpop lj
    tb hdiv dj S₀ d₀ y₀ wr dg N Ω Lb src hA3 hCust k Rw M hk hRw hM ρ hρ hr t' ht' B τ qF qS q
  split
  swap
  · next hTags =>
    rw [dite_eq_right hTags] at h
    exact h
  rename_i hTags
  rw [dite_eq_left hTags] at h
  intro L P vb σ Xr Cc Lm rows roots Cd κ r ns δ hTab hLink
  have hlawOn : Bridge.LawOn law (Bridge.nsOf desc qr) kd := Bridge.lawOn_designAt hpop
  have hQs0 := Bridge.qs_vbP_execSel_le (dj := dj) (y₀ := y₀) (ht := ht) (hc := Flock.Desc.parseStmt_stmt hD)
    (hscope := hscope) (hU := hU) (hpi := hpi) (hpub := hpub) (Reg := Reg) Lb src hA3 hlawOn (Bridge.nsOf_sum hpop) hLb
  have hout : (kd : ℝ) * (2 ^ 64 : ℝ)⁻¹ *
      ((n : ℝ) * ((D.unroll.ports.size + Bridge.pubBits D.unroll hU pi : ℕ) : ℝ)) ≤ 1 := by
    have hsz' : (kd : ℝ) * ((n : ℝ) * ((D.unroll.ports.size + Bridge.pubBits D.unroll hU pi : ℕ) : ℝ)) ≤ 2 ^ 64 := by
      exact_mod_cast hsz
    have h64 : (0 : ℝ) < 2 ^ 64 := by positivity
    calc (kd : ℝ) * (2 ^ 64 : ℝ)⁻¹ * ((n : ℝ) * ((D.unroll.ports.size + Bridge.pubBits D.unroll hU pi : ℕ) : ℝ))
        = (kd : ℝ) * ((n : ℝ) * ((D.unroll.ports.size + Bridge.pubBits D.unroll hU pi : ℕ) : ℝ)) *
            (2 ^ 64 : ℝ)⁻¹ := by ring
      _ ≤ 2 ^ 64 * (2 ^ 64 : ℝ)⁻¹ := mul_le_mul_of_nonneg_right hsz' (inv_nonneg.2 h64.le)
      _ = 1 := mul_inv_cancel₀ h64.ne'
  have hQs : ∑ p : vb.Pos, (Fintype.card (ΩcZ execArith P (planZHJP dj y₀) (tabZHJP dj y₀) vb L p) : ℝ) /
      Fintype.card L.Ω ≤ ((kd * (D.unroll.ports.size + Bridge.pubBits D.unroll hU pi) + 3 : ℕ) : ℝ) := by
    refine hQs0.trans ((add_le_add le_rfl hout).trans (le_of_eq ?_))
    push_cast
    ring
  have h1ρ : 0 < 1 - ρ := sub_pos.2 hρ
  have hB : 0 ≤ 2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw) := by
    have : 0 ≤ t' * (1 + k) := mul_nonneg ht' (by positivity)
    positivity
  refine (h hTab hLink).trans (add_le_add (add_le_add le_rfl (ENNReal.ofReal_le_ofReal ?_)) le_rfl)
  exact div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_right hQs hB) h1ρ.le

end Flock.Guarantees
