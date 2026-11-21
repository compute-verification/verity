module

public import Proofs.Flock.Bridge.V5.Statement
public import Proofs.Flock.Bridge.PopCovers

/-!
# `SampledKSLiveV5Subset`: V5 at a population the strata only cover, under the uniform subset (stated)

`SampledKSLiveV5` with `Bridge.PopCovers` in place of `Bridge.PopMatches`, at a registration whose sampling design is the
uniform subset (`Bridge.IsSubsetDesign` of `Bridge.designAt`; `All` is `subset n`): the population is the units
`0, …, n − 1` in unit order and the strata's ranges need only be a disjoint cover of them. A subset sampling design's
`P_a` reads the strata only through their sum (`Bridge.pa_partOf_subset`), so `δ` is the subset's own miss of `B`
whatever their order. Stratified and work sampling designs keep `SampledKSLiveV5`'s form.

* `Bridge.SampledKSLiveV5On`: `SampledKSLiveV5R` at the two facts about the registration's strata that its proof reads,
  in place of `PopMatches`: they sum to `n` and the sampling design is on them (`SamplingDesignOn` at `nsOf`).
* `SampledKSLiveV5Subset`: the guarantee.
-/

@[expose] public section

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game Spot.Sampling FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

open Classical in
/-- **`SampledKSLiveV5R` at its strata's facts**: V5's bound under the roots guard, with `PopMatches` replaced by what
its assembly reads of it: the registration's strata `nsOf desc qr` sum to `n`, and the sampling design the verifier
samples under is on them (`Bridge.SamplingDesignOn`). -/
def _root_.Flock.Bridge.SampledKSLiveV5On : Prop :=
  ∀ (n : ℕ) (Reg : Type)
    (I : Inputs) (own : Option ByteArray) (ht : I.tags.typed = false)
    (D : Flock.Desc) (hD : Flock.Desc.parseStmt (zkTags I) I.circuitFile I.tables = .ok D) (hU : Built D.unroll.unit)
    (pi : HmIn.Ports) (hpi : HmIn.parse D.unroll = .ok pi)
    (pub : Public) (hpub : HmIn.loadPublic (zkTags I) D.unroll pi I.publicFile = .ok pub)
    (hscope : Discharge.Layout.scopeOkH D.unroll = true)
    (_hrows : Bridge.RowsOk D.unroll)
    (_hdup : Flock.PortRoots.dupId D.unroll pub = false)
    (_hroots : Bridge.RootFixesDomainL D.unroll (Bridge.LeafMap.ofPublic D.unroll pub n))
    (desc : Hash.Msg) (qr : Lean.Json)
    (lj : Lean.Json) (tb : Option Lean.Json) (samplingDesign : Flock.Draw.SamplingDesign) (kd : ℕ)
    (hlaw : Bridge.SamplesUnder desc qr lj tb n samplingDesign kd) (hb : Bridge.NotBernoulli samplingDesign)
    -- the registration's strata sum to the population, and the sampling design is on them
    (_hns : (Bridge.nsOf desc qr).sum = n) (_hon : Bridge.SamplingDesignOn samplingDesign (Bridge.nsOf desc qr) kd)
    (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : SampleSetupZHJP (inputsJ I 0) own (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJP (Reg := Reg) dj y₀ S R).length (mPtsOf D.unroll.g (kd / nTab I)) → Discharge.Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    (_hA3 : Assumptions.UniformRandomBytes Lb src)
    (hTags : TagsOkZHJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJPLaw I own dj (fun S R => (planZHJP (Reg := Reg) dj y₀ S R).length) wr
      fun _ => samplingDesign)
    (k Rw M : ℕ) (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) (ρ : ℝ) (_hρ : ρ < 1)
    (_hr : ∀ S R j, rateZ (planZHJP (Reg := Reg) dj y₀ S R) j k ≤ ρ) (t' : ℝ) (_ht' : 0 ≤ t')
    (B : Finset ℕ)
    (τ : Strategy (auditLive (Bridge.execSel (n := n) src samplingDesign n) Reg (liveOfZJ dg (planZHJP dj y₀) N wr)))
    (qF qS : ℝ)
    (q : Claims.Sampling.Query),
    let L := Bridge.execSel (n := n) src samplingDesign n
    let P := (keyProgP D.unroll pi hU n).partition (Discharge.Layout.outsOfP D.unroll hU n pi)
    let vb := (rsZHJP dj y₀ ht (Flock.Desc.parseStmt_stmt hD) hscope hU hpi hpub).bindingZ
      (rsZHJP_computes dj y₀ ht (Flock.Desc.parseStmt_stmt hD) hscope hU hpi hpub)
    let σ := simZJA execArith hs512 (planZHJP dj y₀) (AcceptsZKP I own) wr (opnAtZHJP (planZHJP dj y₀) I own dj) dg
      (decAtZJ (planZHJP dj y₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesZCJH_custodyP I own ht (Flock.Desc.parseStmt_stmt hD) samplingDesign kd hdiv dj y₀ wr _ hTags hCust
        (Bridge.samplingDesignSized_size (Bridge.samplingDesignSized_samplesUnder hlaw hb) hCust.2)) τ
    let Xr := XplurZC execArith hs512 P (planZHJP dj y₀) (tabZHJP dj y₀) vb (reg σ) (cont σ) k Rw
    let Cc := (keyProgP D.unroll pi hU n).circuit (Discharge.Layout.outsOfP D.unroll hU n pi)
    let Lm := Bridge.LeafMap.ofPublic D.unroll pub n
    let rows : Fin Cc.N → Option Bridge.RowAt := fun g =>
      (Discharge.HiddenExec.rowOfHP D.unroll hU n pi (Discharge.Layout.outsOfP D.unroll hU n pi) g).map fun r =>
        ⟨r.2.1.val, Lm.ref r.2.1.val r.1.val, Discharge.Layout.portRowW D.unroll r.2.1, r.2.2.val⟩
    let roots := Bridge.rootsOfL D.unroll Lm
    let Cd := Bridge.commitDepL Cc rows D.unroll Lm roots
    let κ : Fin Cc.N → Option Bool := fun g => Bridge.constsOfP D.unroll hU n pi pub g
    let r := Properties.Spot.oneStageOf samplingDesign
    let ns := Bridge.nsOf desc qr
    let δ : ℝ := (Bridge.designOf r).Pa (Bridge.partOf P κ ns) B
    (∀ ω j, TableCRZ execArith hs512 (planZHJP dj y₀ (L.draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZHJP dj y₀) (cont σ ω)) qF qS) →
    LinkCRZC execArith hs512 P (planZHJP dj y₀) (tabZHJP dj y₀) vb (reg σ) (cont σ) k Rw M t' →
    prob (fun o => LiveAcceptsZKP I own o ∧
        ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg Cd δ (Bridge.stmtOf P κ r desc q roots) (Bridge.partOf P κ ns)
          (Bridge.witDep Cc Cd Claims.Sampling.sha512Msg roots Xr))
        (auditLive L Reg (liveOfZJ dg (planZHJP dj y₀) N wr)) τ ≤
      ENNReal.ofReal δ +
        ksAvgStrictZ (planZHJP dj y₀) L k qF qS (reg σ) +
        ENNReal.ofReal ((∑ p : vb.Pos,
            (Fintype.card (ΩcZ execArith P (planZHJP dj y₀) (tabZHJP dj y₀) vb L p) : ℝ) / Fintype.card L.Ω) *
          (2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ))

open Classical in
/-- **The sampled audit's knowledge soundness, live, at the verifier's public inputs, under the uniform subset at a
population the strata cover**: `SampledKSLiveV5` at a registration whose layout's population is the units `0, …, n − 1`
each once in any order (`Bridge.PopCovers`), not in order (`Bridge.PopMatches`), and whose sampling design the verifier
samples under is the uniform subset (`Bridge.IsSubsetDesign` of `Bridge.designAt`; `All` is `subset n`). `δ` is the
subset's own miss of `B`, `P_a(B)` on `partOf P κ ns`, which reads the strata `ns` only through their sum; at `subset kd`
and the first `K₀` proof blocks it is `C(n − K₀, kd)/C(n, kd)` (`Bridge.pa_subset_range`). Its named assumptions are
V5's. -/
def SampledKSLiveV5Subset : Prop :=
  ∀ (n : ℕ) (Reg : Type)
    (I : Inputs) (own : Option ByteArray) (ht : I.tags.typed = false)
    (D : Flock.Desc) (hD : Flock.Desc.parseStmt (zkTags I) I.circuitFile I.tables = .ok D),
    let hU : Built D.unroll.unit := Discharge.Hidden.built_of_parseStmt hD ((zkTags_typed I).trans ht)
    ∀ (pi : HmIn.Ports) (hpi : HmIn.parse D.unroll = .ok pi)
    (pub : Public) (hpub : HmIn.loadPublic (zkTags I) D.unroll pi I.publicFile = .ok pub)
    (hscope : Discharge.Layout.scopeOkH D.unroll = true)
    (_hrows : Bridge.RowsOk D.unroll)
    (_hdup : Flock.PortRoots.dupId D.unroll pub = false)
    -- the statement's descriptor, and the registration's query, whose strata cover the game's n proof blocks
    (desc : Hash.Msg) (qr : Lean.Json) (_hpop : Bridge.PopCovers desc qr n)
    -- the registration's sampling design object and work table, under which the verifier samples a uniform subset
    (lj : Lean.Json) (tb : Option Lean.Json) (_hsub : Bridge.IsSubsetDesign (Bridge.designAt desc qr lj tb n)),
    let samplingDesign := Bridge.designAt desc qr lj tb n
    let kd := Bridge.samplingDesignK samplingDesign
    ∀ (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : SampleSetupZHJP (inputsJ I 0) own (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJP (Reg := Reg) dj y₀ S R).length (mPtsOf D.unroll.g (kd / nTab I)) → Discharge.Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the verifier's operating-system bytes, and A3
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    (_hA3 : Assumptions.UniformRandomBytes Lb src)
    -- live-verifier at `own` and the sampling design
    (hCust : Assumptions.Zk.RecordCustodyZKJPLaw I own dj (fun S R => (planZHJP (Reg := Reg) dj y₀ S R).length) wr
      fun _ => samplingDesign)
    (k Rw M : ℕ) (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) (ρ : ℝ) (_hρ : ρ < 1)
    (_hr : ∀ S R j, rateZ (planZHJP (Reg := Reg) dj y₀ S R) j k ≤ ρ) (t' : ℝ) (_ht' : 0 ≤ t')
    -- the proof blocks whose miss is `δ`
    (B : Finset ℕ)
    (τ : Strategy (auditLive (Bridge.execSel (n := n) src samplingDesign n) Reg (liveOfZJ dg (planZHJP dj y₀) N wr)))
    (qF qS : ℝ)
    -- what the sampling statement carries and `RelS` doesn't read
    (q : Claims.Sampling.Query),
    if hTags : TagsOkZHJ I then
      let L := Bridge.execSel (n := n) src samplingDesign n
      let P := (keyProgP D.unroll pi hU n).partition (Discharge.Layout.outsOfP D.unroll hU n pi)
      let vb := (rsZHJP dj y₀ ht (Flock.Desc.parseStmt_stmt hD) hscope hU hpi hpub).bindingZ
        (rsZHJP_computes dj y₀ ht (Flock.Desc.parseStmt_stmt hD) hscope hU hpi hpub)
      let σ := simZJA execArith hs512 (planZHJP dj y₀) (AcceptsZKP I own) wr (opnAtZHJP (planZHJP dj y₀) I own dj) dg
        (decAtZJ (planZHJP dj y₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
        (decodesZCJH_custodyP I own ht (Flock.Desc.parseStmt_stmt hD) samplingDesign kd hdiv dj y₀ wr _ hTags hCust
          (Bridge.samplingDesignSized_size (Bridge.sized_designAt desc qr lj tb n) hCust.2)) τ
      let Xr := XplurZC execArith hs512 P (planZHJP dj y₀) (tabZHJP dj y₀) vb (reg σ) (cont σ) k Rw
      -- the deployed commitment's layout: the game's rows, read off the circuit file, at the public file's leaf map
      let Cc := (keyProgP D.unroll pi hU n).circuit (Discharge.Layout.outsOfP D.unroll hU n pi)
      let Lm := Bridge.LeafMap.ofPublic D.unroll pub n
      let rows : Fin Cc.N → Option Bridge.RowAt := fun g =>
        (Discharge.HiddenExec.rowOfHP D.unroll hU n pi (Discharge.Layout.outsOfP D.unroll hU n pi) g).map fun r =>
          ⟨r.2.1.val, Lm.ref r.2.1.val r.1.val, Discharge.Layout.portRowW D.unroll r.2.1, r.2.2.val⟩
      let roots := Bridge.rootsOfL D.unroll Lm
      let Cd := Bridge.commitDepL Cc rows D.unroll Lm roots
      -- the circuit's constants and every proof block's public input bits, public
      let κ : Fin Cc.N → Option Bool := fun g => Bridge.constsOfP D.unroll hU n pi pub g
      -- the sampling design's design, the registration's strata, and the sampling design's own miss of `B`
      let r := Properties.Spot.oneStageOf samplingDesign
      let ns := Bridge.nsOf desc qr
      let δ : ℝ := (Bridge.designOf r).Pa (Bridge.partOf P κ ns) B
      -- cr/sha-512 for the knowledge finders of each table the prover's session runs
      (∀ ω j, TableCRZ execArith hs512 (planZHJP dj y₀ (L.draw ω) (reg σ)) j
        (zcStrat execArith hs512 (planZHJP dj y₀) (cont σ ω)) qF qS) →
      -- ecr/sha-512 for the prover's link finders
      LinkCRZC execArith hs512 P (planZHJP dj y₀) (tabZHJP dj y₀) vb (reg σ) (cont σ) k Rw M t' →
      prob (fun o => LiveAcceptsZKP I own o ∧
          ¬ Claims.Sampling.RelS Claims.Sampling.sha512Msg Cd δ (Bridge.stmtOf P κ r desc q roots)
            (Bridge.partOf P κ ns) (Bridge.witDep Cc Cd Claims.Sampling.sha512Msg roots Xr))
          (auditLive L Reg (liveOfZJ dg (planZHJP dj y₀) N wr)) τ ≤
        ENNReal.ofReal δ +
          ksAvgStrictZ (planZHJP dj y₀) L k qF qS (reg σ) +
          ENNReal.ofReal ((∑ p : vb.Pos,
              (Fintype.card (ΩcZ execArith P (planZHJP dj y₀) (tabZHJP dj y₀) vb L p) : ℝ) / Fintype.card L.Ω) *
            (2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ)) +
          Claims.ecrBound Claims.Sampling.sha512Msg (Bridge.rootsFinderL D.unroll Lm) (Bridge.rootsCostL D.unroll Lm)
    else
      prob (fun o => Bridge.LiveAcceptsKSP I own o)
        (auditLive (Bridge.execSel (n := n) src samplingDesign n) Reg (liveOfZJ dg (planZHJP dj y₀) N wr)) τ = 0

end Flock.Guarantees
