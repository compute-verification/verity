module

public import Proofs.Flock.Bridge.LeafMap
public import Proofs.Flock.Bridge.LeafRoots
public import Proofs.Flock.Bridge.Draw
public import Proofs.Flock.Bridge.SampledDesign
public import Proofs.Flock.Soundness.Discharge.ZkHidden.CustodyJP
public import Proofs.Flock.Soundness.Discharge.Hidden.Built
public import Flock.KsScope
public import Flock.PortRoots

/-! `SampledKSLiveV5`, the sampled audit's knowledge soundness at the verifier's public inputs, stated, with
the targets its assembly takes. -/

@[expose] public section

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

/-- **The circuit's constants at public inputs**: the constant wires' (`constWiresOf`: `oneG` is `1`, `zeroG` is `0`),
and at proof block `u`'s public gate of bit `t` of public word `q` (`Layout.keyPubOf`) that bit of `u`'s public input
words in the population's public file `pub` (`HiddenExec.pubBitOf`). -/
noncomputable def _root_.Flock.Bridge.constsOfP (c : Flock.Circuit) (hU : Built c.unit) (n : ℕ) (pi : HmIn.Ports)
    (pub : Public) (g : Fin (Discharge.Layout.keyN c hU n)) : Option Bool :=
  match Bridge.constWiresOf c hU n g with
  | some b => some b
  | none => (Discharge.Layout.keyPubOf c hU n pi g).map fun x =>
      Discharge.HiddenExec.pubBitOf pi (fun (_ : Unit) => pub) () x.1 x.2.1 x.2.2

def _root_.Flock.Bridge.OpeningsCanonTargetV5 : Prop :=
  ∀ (n : ℕ) (Reg : Type)
    (I : Inputs) (own : Option ByteArray) (ht : I.tags.typed = false)
    (D : Flock.Desc) (hD : Flock.Desc.parseStmt (zkTags I) I.circuitFile I.tables = .ok D) (hU : Built D.unroll.unit)
    (pi : HmIn.Ports) (hpi : HmIn.parse D.unroll = .ok pi)
    (pub : Public) (hpub : HmIn.loadPublic (zkTags I) D.unroll pi I.publicFile = .ok pub)
    (hscope : Discharge.Layout.scopeOkH D.unroll = true)
    (kd : ℕ) (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : SampleSetupZHJP (inputsJ I 0) own (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀),
    let P := (keyProgP D.unroll pi hU n).partition (Discharge.Layout.outsOfP D.unroll hU n pi)
    let vb := (rsZHJP (Reg := Reg) dj y₀ ht (Flock.Desc.parseStmt_stmt hD) hscope hU hpi hpub).bindingZ
      (rsZHJP_computes dj y₀ ht (Flock.Desc.parseStmt_stmt hD) hscope hU hpi hpub)
    let Cc := (keyProgP D.unroll pi hU n).circuit (Discharge.Layout.outsOfP D.unroll hU n pi)
    let Lm := Bridge.LeafMap.ofPublic D.unroll pub n
    let rows : Fin Cc.N → Option Bridge.RowAt := fun g =>
      (Discharge.HiddenExec.rowOfHP D.unroll hU n pi (Discharge.Layout.outsOfP D.unroll hU n pi) g).map fun r =>
        ⟨r.2.1.val, Lm.ref r.2.1.val r.1.val, Discharge.Layout.portRowW D.unroll r.2.1, r.2.2.val⟩
    let κ : Fin Cc.N → Option Bool := fun g => Bridge.constsOfP D.unroll hU n pi pub g
    OpeningsCanon execArith P (planZHJP dj y₀) (tabZHJP dj y₀) vb
      (fun g o => Bridge.canonAt rows κ g (vb.bit (vb.val o) g) (vb.val o) o.2.2)

open Classical in
def _root_.Flock.Bridge.EndToEndHiddenV5 : Prop :=
  ∀ (n : ℕ) (Reg : Type)
    (I : Inputs) (own : Option ByteArray) (ht : I.tags.typed = false)
    (D : Flock.Desc) (hD : Flock.Desc.parseStmt (zkTags I) I.circuitFile I.tables = .ok D) (hU : Built D.unroll.unit)
    (pi : HmIn.Ports) (hpi : HmIn.parse D.unroll = .ok pi)
    (pub : Public) (hpub : HmIn.loadPublic (zkTags I) D.unroll pi I.publicFile = .ok pub)
    (hscope : Discharge.Layout.scopeOkH D.unroll = true)
    (samplingDesign : Flock.Draw.SamplingDesign) (kd : ℕ) (ns : List ℕ) (_hns : ns.sum = n) (hlaw : Bridge.SamplingDesignOn samplingDesign ns kd)
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
    (W : Finset (Fin n))
    (τ : Strategy (auditLive (Bridge.execSel (n := n) src samplingDesign n) Reg (liveOfZJ dg (planZHJP dj y₀) N wr)))
    (qF qS : ℝ),
    let L := Bridge.execSel (n := n) src samplingDesign n
    let P := (keyProgP D.unroll pi hU n).partition (Discharge.Layout.outsOfP D.unroll hU n pi)
    let vb := (rsZHJP dj y₀ ht (Flock.Desc.parseStmt_stmt hD) hscope hU hpi hpub).bindingZ
      (rsZHJP_computes dj y₀ ht (Flock.Desc.parseStmt_stmt hD) hscope hU hpi hpub)
    let σ := simZJA execArith hs512 (planZHJP dj y₀) (AcceptsZKP I own) wr (opnAtZHJP (planZHJP dj y₀) I own dj) dg
      (decAtZJ (planZHJP dj y₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesZCJH_custodyP I own ht (Flock.Desc.parseStmt_stmt hD) samplingDesign kd hdiv dj y₀ wr _ hTags hCust
        (Bridge.samplingDesignOn_size hlaw hCust.2)) τ
    let Cc := (keyProgP D.unroll pi hU n).circuit (Discharge.Layout.outsOfP D.unroll hU n pi)
    let Lm := Bridge.LeafMap.ofPublic D.unroll pub n
    let rows : Fin Cc.N → Option Bridge.RowAt := fun g =>
      (Discharge.HiddenExec.rowOfHP D.unroll hU n pi (Discharge.Layout.outsOfP D.unroll hU n pi) g).map fun r =>
        ⟨r.2.1.val, Lm.ref r.2.1.val r.1.val, Discharge.Layout.portRowW D.unroll r.2.1, r.2.2.val⟩
    let κ : Fin Cc.N → Option Bool := fun g => Bridge.constsOfP D.unroll hU n pi pub g
    (∀ ω j, TableCRZ execArith hs512 (planZHJP dj y₀ (L.draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZHJP dj y₀) (cont σ ω)) qF qS) →
    LinkCRZC execArith hs512 P (planZHJP dj y₀) (tabZHJP dj y₀) vb (reg σ) (cont σ) k Rw M t' →
    prob (fun o => LiveAcceptsZKP I own o ∧
        W ⊆ incorrectRegZc execArith P (planZHJP dj y₀) (tabZHJP dj y₀) vb
          (fun g o => Bridge.canonAt rows κ g (vb.bit (vb.val o) g) (vb.val o) o.2.2) (reg σ)
          (XplurZC execArith hs512 P (planZHJP dj y₀) (tabZHJP dj y₀) vb (reg σ) (cont σ) k Rw))
        (auditLive L Reg (liveOfZJ dg (planZHJP dj y₀) N wr)) τ ≤
      ENNReal.ofReal ((Properties.Spot.oneStageOf samplingDesign).Pa ns (W.map Fin.valEmbedding)) +
        ksAvgStrictZ (planZHJP dj y₀) L k qF qS (reg σ) +
        ENNReal.ofReal ((∑ p : vb.Pos,
            (Fintype.card (ΩcZ execArith P (planZHJP dj y₀) (tabZHJP dj y₀) vb L p) : ℝ) / Fintype.card L.Ω) *
          (2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ))

open Classical in
def _root_.Flock.Bridge.TransportsLiveV5 : Prop :=
  ∀ (n : ℕ) (Reg : Type)
    (I : Inputs) (own : Option ByteArray) (ht : I.tags.typed = false)
    (D : Flock.Desc) (hD : Flock.Desc.parseStmt (zkTags I) I.circuitFile I.tables = .ok D) (hU : Built D.unroll.unit)
    (pi : HmIn.Ports) (hpi : HmIn.parse D.unroll = .ok pi)
    (pub : Public) (hpub : HmIn.loadPublic (zkTags I) D.unroll pi I.publicFile = .ok pub)
    (hscope : Discharge.Layout.scopeOkH D.unroll = true)
    (_hrows : Bridge.RowsOk D.unroll)
    (_hdup : Flock.PortRoots.dupId D.unroll pub = false)
    (_hroots : Bridge.RootFixesDomainL D.unroll (Bridge.LeafMap.ofPublic D.unroll pub n))
    (samplingDesign : Flock.Draw.SamplingDesign) (kd : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : SampleSetupZHJP (inputsJ I 0) own (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJP (Reg := Reg) dj y₀ S R).length (mPtsOf D.unroll.g (kd / nTab I)) → Discharge.Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ) (L : SamplingDesign n)
    (hTags : TagsOkZHJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJPLaw I own dj (fun S R => (planZHJP (Reg := Reg) dj y₀ S R).length) wr
      fun _ => samplingDesign)
    (hsz : ∀ S ud, Flock.Draw.UnitSample.ofJson (dj S) = .ok ud → ud.units.size = kd)
    (τ : Strategy (auditLive L Reg (liveOfZJ dg (planZHJP dj y₀) N wr))),
    let P := (keyProgP D.unroll pi hU n).partition (Discharge.Layout.outsOfP D.unroll hU n pi)
    let vb := (rsZHJP dj y₀ ht (Flock.Desc.parseStmt_stmt hD) hscope hU hpi hpub).bindingZ
      (rsZHJP_computes dj y₀ ht (Flock.Desc.parseStmt_stmt hD) hscope hU hpi hpub)
    let σ := simZJA execArith hs512 (planZHJP dj y₀) (AcceptsZKP I own) wr (opnAtZHJP (planZHJP dj y₀) I own dj) dg
      (decAtZJ (planZHJP dj y₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesZCJH_custodyP I own ht (Flock.Desc.parseStmt_stmt hD) samplingDesign kd hdiv dj y₀ wr L hTags hCust hsz) τ
    let Cc := (keyProgP D.unroll pi hU n).circuit (Discharge.Layout.outsOfP D.unroll hU n pi)
    let Lm := Bridge.LeafMap.ofPublic D.unroll pub n
    let rows : Fin Cc.N → Option Bridge.RowAt := fun g =>
      (Discharge.HiddenExec.rowOfHP D.unroll hU n pi (Discharge.Layout.outsOfP D.unroll hU n pi) g).map fun r =>
        ⟨r.2.1.val, Lm.ref r.2.1.val r.1.val, Discharge.Layout.portRowW D.unroll r.2.1, r.2.2.val⟩
    let κ : Fin Cc.N → Option Bool := fun g => Bridge.constsOfP D.unroll hU n pi pub g
    let roots := Bridge.rootsOfL D.unroll Lm
    let Cd := Bridge.commitDepL Cc rows D.unroll Lm roots
    ∀ u, ∀ g ∈ P.io u, κ g = none → ∀ o : vb.Opening, vb.com o = vb.cs (reg σ) (vb.pos g) →
      Bridge.canonOpen rows g (vb.val o) o.2.2 →
      ∃ d : Claims.Msg, Cd.Opens Claims.Sampling.sha512Msg roots ⟨Bridge.posOf Cc g, vb.bit (vb.val o) g, d⟩ = true

open Classical in
/-- `SampledKSLiveV5` under the roots guard (`Bridge.RootFixesDomainL`), without its `ecr/sha-512` term at the
roots finder. -/
def _root_.Flock.Bridge.SampledKSLiveV5R : Prop :=
  ∀ (n : ℕ) (Reg : Type)
    (I : Inputs) (own : Option ByteArray) (ht : I.tags.typed = false)
    (D : Flock.Desc) (hD : Flock.Desc.parseStmt (zkTags I) I.circuitFile I.tables = .ok D) (hU : Built D.unroll.unit)
    (pi : HmIn.Ports) (hpi : HmIn.parse D.unroll = .ok pi)
    (pub : Public) (hpub : HmIn.loadPublic (zkTags I) D.unroll pi I.publicFile = .ok pub)
    (hscope : Discharge.Layout.scopeOkH D.unroll = true)
    (_hrows : Bridge.RowsOk D.unroll)
    (_hdup : Flock.PortRoots.dupId D.unroll pub = false)
    (_hroots : Bridge.RootFixesDomainL D.unroll (Bridge.LeafMap.ofPublic D.unroll pub n))
    (desc : Claims.Msg) (qr : Lean.Json) (_hpop : Bridge.PopMatches desc qr n)
    (lj : Lean.Json) (tb : Option Lean.Json) (samplingDesign : Flock.Draw.SamplingDesign) (kd : ℕ)
    (hlaw : Bridge.SamplesUnder desc qr lj tb n samplingDesign kd) (hb : Bridge.NotBernoulli samplingDesign) (hdiv : kd % nTab I = 0)
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
    -- what the sampling statement carries and `RelS` doesn't read
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

/-- **`verify --zk` accepts at `own`, files and transcript**: the live audit's transcript (`LiveAcceptsZKP`), and the
last two refusals of `verify --zk`'s verdict, on the circuit it checks, `ProvedScope.circuitOf`'s (a templated
statement's carries its template): the files' rows, names and roots (`Flock.PortRoots.checkOf`), and their form
(`Flock.KsScope.checkOf`). At public outputs `verify --zk` checks `HmOut.parse`'s circuit, but `KsScope.checkOf`
refuses those tags before it reads a circuit. -/
def _root_.Flock.Bridge.LiveAcceptsKSP {n : ℕ} {Reg : Type} (I : Inputs) (own : Option ByteArray)
    (o : Reg × Finset (Fin n) × Discharge.Exec.Transcript) : Prop :=
  LiveAcceptsZKP I own o ∧
    Flock.PortRoots.checkOf (Flock.ProvedScope.circuitOf (zkTags I) I.circuitFile I.tables) (zkTags I) I.publicFile =
      .ok () ∧
    ∃ count draw, Flock.KsScope.checkOf (Flock.ProvedScope.circuitOf (zkTags I) I.circuitFile I.tables) (zkTags I)
      I.publicFile count draw = .ok ()

open Classical in
/-- **`SampledKSLiveV2` at the verifier's public inputs, with or without shared rows, under the sampling design the verifier samples
under from the registration**: verity-verify-proof `--zk --public-inputs` at `own` accepts and the committed trace fixed at
registration, read as the sampling witness of the session's statement at the constants `constsOfP` (the constant, the
zero and every proof block's public input bits) and the sampling design's design (`oneStageOf samplingDesign`), is outside `RelS` at `δ`, at
the deployed commitment at the public file's leaf map, with probability at most `δ + ε_ks + ε_bind`. `δ` is not a
parameter: it is the sampling design's own probability of sampling no proof block of a set `B`, `P_a(B)` of its design on
`partOf P κ ns` at the strata `ns` of the registration's layout (`nsOf`). At `subset kd` and the first `K₀` proof blocks
it is `C(n − K₀, kd)/C(n, kd)` (`Bridge.pa_subset_range`), and under `All`, whose sampling design is `subset n`, it is `0` at any
`B` holding a proof block (`Bridge.pa_all`). Its named assumptions are V2's, with `live-verifier` at `own` and the sampling design
(`RecordCustodyZKJPLaw`). The sampling design is the one the verifier samples under from the registration's sampling design object and work table
(`Bridge.designAt`: every sample `OneStage.sampledFrom` samples is under it, `Bridge.sampledFrom_designAt`), with no hypothesis that
the registration names one: at a registration that names none, which the verifier refuses, `designAt` is `subset 0`, which
samples no proof block, so `δ` is 1. It is on its layout's strata at `PopMatches` (`Bridge.samplingDesignOn_samplesUnder`), and no
Bernoulli sampling design, though `Draw.flockSamplingDesign` parses one: `sampledFrom` refuses it (`OneStage.coveredDesign`, `Bridge.notBernoulli_designAt`). The link finders are charged `t'` per run
(`cTZC`), one run's SHA-512 evaluations, the prover's and the verifier's: V2's `t' + v`. No hypothesis is taken on the
registration's roots: two ports with one root and different domains are a SHA-512 collision that recomputing the roots finds, so the
bound carries `ecr/sha-512`'s term at that finder (`Bridge.rootsFinderL`, no coins, `Bridge.rootsCostL` SHA-512 calls),
at least 1 at a registration whose roots expose one and `rootsCostL / 2^256` at any other. Nor is one taken on the
proof block's circuit: its unit is built because the circuit file parses (`Discharge.Hidden.built_of_parseStmt`). Nor on
the `--zk` tags: at tags in `TagsOkZHJ` the bound holds, and at any others `verify --zk` refuses the files
(`Flock.KsScope.checkOf`'s `tagsWhy`), so the event that it accepts them (`Bridge.LiveAcceptsKSP`) has probability 0. -/
def SampledKSLiveV5 : Prop :=
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
    let samplingDesign := Bridge.designAt desc qr lj tb n
    let kd := Bridge.samplingDesignK samplingDesign
    ∀ (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : SampleSetupZHJP (inputsJ I 0) own (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJP (Reg := Reg) dj y₀ S R).length (mPtsOf D.unroll.g (kd / nTab I)) → Discharge.Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    (_hA3 : Assumptions.UniformRandomBytes Lb src)
    (hCust : Assumptions.Zk.RecordCustodyZKJPLaw I own dj (fun S R => (planZHJP (Reg := Reg) dj y₀ S R).length) wr
      fun _ => samplingDesign)
    (k Rw M : ℕ) (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) (ρ : ℝ) (_hρ : ρ < 1)
    (_hr : ∀ S R j, rateZ (planZHJP (Reg := Reg) dj y₀ S R) j k ≤ ρ) (t' : ℝ) (_ht' : 0 ≤ t')
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

def _root_.Flock.Bridge.AssemblyTargetV5 : Prop :=
  Bridge.TransportsLiveV5 → Bridge.FaultyRightTargetC → Bridge.EndToEndHiddenV5 → Bridge.SampledKSLiveV5R

end Flock.Guarantees
