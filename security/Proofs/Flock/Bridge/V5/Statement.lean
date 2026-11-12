module

public import Proofs.Flock.Bridge.LeafMap
public import Proofs.Flock.Bridge.LeafRoots
public import Proofs.Flock.Bridge.Draw
public import Proofs.Flock.Bridge.DrawnLaw
public import Proofs.Flock.Soundness.Discharge.ZkHidden.CustodyJP
public import Proofs.Flock.Soundness.Discharge.Hidden.Built
public import Flock.KsScope
public import Flock.PortRoots

/-!
# `SampledKSLiveV5`: the sampled audit's knowledge soundness at the verifier's public inputs (stated)

`SampledKSLiveV2` at flock-verify `--zk --public-inputs`, with or without shared rows, under the sampling design the verifier samples
under from the registration.
`own` is the verifier's own recomputation of each proof block's public input words (`HmIn.checkOwn`), and the statement
holds at every `own`:

* the circuit file's descriptor `D` (`Flock.Desc.parseStmt`): a flat circuit, or a templated statement whose circuit
  `D.unroll` is its committed circuit unrolled through its template, read everywhere as the statement circuit;
* the circuit's public input ports `pi` (`HmIn.parse`) and the population's public file loaded over them
  (`HmIn.loadPublic`), in place of `HmOut.loadPublic` and `pub.tables = none`;
* the program `Placed.keyProgP`, whose proof blocks read the gates of their keys with public inputs, its outputs
  `Layout.outsOfP`, the binding `ZkHidden.rsZHJP`, the session's tables `planZHJP` and its simulator `simZJA` at
  flock-verify's acceptance at `own` (`AcceptsZKP`, through `decodesZCJH_custodyP`);
* the sampling design `law` the verifier samples under, sampling `kd` proof blocks (`Bridge.designAt`, `kd` its `Bridge.lawK`): the
  registration's law object expanded over the strata of the registration's layout, else the sampling design it parses to
  (`Bridge.drawnLaw`: `Draw.deriveLaw` over `Partition.strata`, else `Draw.flockLaw`, as `OneStage.drawnFrom` samples,
  `Bridge.drawnFrom_drawnLaw` and `Bridge.drawnFrom_designAt`), and `subset 0` at a registration that names none; never a
  Bernoulli law, which `drawnFrom` refuses (`Bridge.notBernoulli_designAt`). `Bridge.SampledKSLiveV5R` states the bound at any `law` the verifier samples
  under (`Bridge.DrawsUnder`). At `PopMatches` it is on those strata (`Bridge.LawOn` at `Bridge.nsOf`,
  `Bridge.lawOn_drawsUnder`): `subset kd` (`All` is `subset n`), or `stratified K` or `work K` over them. The verifier
  samples under it from its own bytes (`Bridge.execSel`, `Spot.Draw.drawUnits` on one stream), and `δ` is that
  sampling design's own miss of a set `B` of proof blocks (`P_a(B)` of `designOf (oneStageOf law)` on `partOf P κ ns`), not a
  parameter;
* `live-verifier` at `own` and the sampling design (`Assumptions.Zk.RecordCustodyZKJPLaw`);
* the `--zk` tags hidden-output ones (`ZkHidden.TagsOkZHJ`) as a case, not a hypothesis: at any others `verify --zk`
  refuses (`Flock.KsScope.checkOf`), so its acceptance (`Bridge.LiveAcceptsKSP`) has probability 0;
* the constants `Bridge.constsOfP`: the constant wires `Bridge.constWiresOf`, the constant and the zero, and each proof
  block's public gates at its public input words' bits in the population's public file (`HiddenExec.pubBitOf`). The
  verifier derives the public inputs, so the statement fixes them, and a public gate is read with no opening;
* the registration at the public file's leaf map (`LeafMap.ofPublic`: one leaf per proof block, or a `shared_rows` file's
  refs), as `TransportsLiveCR` reads it.

The targets V5's assembly takes (`Bridge.AssemblyTargetV5`): `TransportsLiveV5` (`TransportsLiveCR` at these, at any
sample), `EndToEndHiddenV5` (`EndToEndHiddenC` at these, from `OpeningsCanonTargetV5`, bounding the event that a set `W`
of proof blocks is wrong by the sampling design's `P_a` of `W`), and `FaultyRightTargetC` as V2 takes it.
-/

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

/-- **A sampled proof block's openings are canonical at public inputs** (`OpeningsCanonTarget` at `keyProgP`, `rsZHJP` and
`constsOfP`): in every satisfying witness of a sampled proof block, the opening of each of its committed wires' commit strings
carries the constant at a constant wire (the constant, the zero, and the public gates at their public bits,
`HiddenExec.keyProgHP_valPinned`), and at any other is a row of its wire's row's width with a salt of `saltLen`
bytes. -/
def _root_.Flock.Bridge.OpeningsCanonTargetV5 : Prop :=
  ∀ (n : ℕ) (Reg : Type)
    (I : Inputs) (own : Option ByteArray) (ht : I.tags.typed = false)
    (D : Flock.Desc) (hD : Flock.Desc.parseStmt (zkTags I) I.circuitFile I.tables = .ok D) (hU : Built D.unroll.unit)
    (pi : HmIn.Ports) (hpi : HmIn.parse D.unroll = .ok pi)
    (pub : Public) (hpub : HmIn.loadPublic (zkTags I) D.unroll pi I.publicFile = .ok pub)
    (hscope : Discharge.Layout.scopeOkH D.unroll = true)
    (kd : ℕ) (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : DrawSetupZHJP (inputsJ I 0) own (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀),
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
/-- **`EndToEndHiddenC` at public inputs, under the registration's law**: its bound at flock-verify
`--zk --public-inputs`'s acceptance, the session at `own` (`planZHJP`, `rsZHJP`, `simZJA`), `live-verifier` at `own`
and the sampling design, the verifier's sample under `law` on strata of sizes `ns` (`Bridge.execSel`), and "wrong" the proof blocks
wrong on the registration with canonical openings at `constsOfP`: every proof block of `W` is wrong with probability at
most the sampling design's `P_a` of `W` and the extraction's terms. -/
def _root_.Flock.Bridge.EndToEndHiddenV5 : Prop :=
  ∀ (n : ℕ) (Reg : Type)
    (I : Inputs) (own : Option ByteArray) (ht : I.tags.typed = false)
    (D : Flock.Desc) (hD : Flock.Desc.parseStmt (zkTags I) I.circuitFile I.tables = .ok D) (hU : Built D.unroll.unit)
    (pi : HmIn.Ports) (hpi : HmIn.parse D.unroll = .ok pi)
    (pub : Public) (hpub : HmIn.loadPublic (zkTags I) D.unroll pi I.publicFile = .ok pub)
    (hscope : Discharge.Layout.scopeOkH D.unroll = true)
    -- the registration's law over strata of sizes `ns`, sampling `kd` proof blocks
    (law : Flock.Draw.Law) (kd : ℕ) (ns : List ℕ) (_hns : ns.sum = n) (hlaw : Bridge.LawOn law ns kd)
    (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : DrawSetupZHJP (inputsJ I 0) own (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJP (Reg := Reg) dj y₀ S R).length (mPtsOf D.unroll.g (kd / nTab I)) → Discharge.Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the verifier's operating-system bytes, and A3
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    (_hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the verify command's tags, and live-verifier at `own` and the sampling design
    (hTags : TagsOkZHJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJPLaw I own dj (fun S R => (planZHJP (Reg := Reg) dj y₀ S R).length) wr
      fun _ => law)
    -- the count curve's parameters
    (k Rw M : ℕ) (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) (ρ : ℝ) (_hρ : ρ < 1)
    (_hr : ∀ S R j, rateZ (planZHJP (Reg := Reg) dj y₀ S R) j k ≤ ρ) (t' : ℝ) (_ht' : 0 ≤ t')
    -- the proof blocks the bound is about
    (W : Finset (Fin n))
    -- the live prover
    (τ : Strategy (auditLive (Bridge.execSel (n := n) src law n) Reg (liveOfZJ dg (planZHJP dj y₀) N wr)))
    (qF qS : ℝ),
    let L := Bridge.execSel (n := n) src law n
    let P := (keyProgP D.unroll pi hU n).partition (Discharge.Layout.outsOfP D.unroll hU n pi)
    let vb := (rsZHJP dj y₀ ht (Flock.Desc.parseStmt_stmt hD) hscope hU hpi hpub).bindingZ
      (rsZHJP_computes dj y₀ ht (Flock.Desc.parseStmt_stmt hD) hscope hU hpi hpub)
    let σ := simZJA execArith hs512 (planZHJP dj y₀) (AcceptsZKP I own) wr (opnAtZHJP (planZHJP dj y₀) I own dj) dg
      (decAtZJ (planZHJP dj y₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesZCJH_custodyP I own ht (Flock.Desc.parseStmt_stmt hD) law kd hdiv dj y₀ wr _ hTags hCust
        (Bridge.lawOn_size hlaw hCust.2)) τ
    let Cc := (keyProgP D.unroll pi hU n).circuit (Discharge.Layout.outsOfP D.unroll hU n pi)
    let Lm := Bridge.LeafMap.ofPublic D.unroll pub n
    let rows : Fin Cc.N → Option Bridge.RowAt := fun g =>
      (Discharge.HiddenExec.rowOfHP D.unroll hU n pi (Discharge.Layout.outsOfP D.unroll hU n pi) g).map fun r =>
        ⟨r.2.1.val, Lm.ref r.2.1.val r.1.val, Discharge.Layout.portRowW D.unroll r.2.1, r.2.2.val⟩
    let κ : Fin Cc.N → Option Bool := fun g => Bridge.constsOfP D.unroll hU n pi pub g
    -- cr/sha-512 for the knowledge finders of each table the prover's session runs
    (∀ ω j, TableCRZ execArith hs512 (planZHJP dj y₀ (L.draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZHJP dj y₀) (cont σ ω)) qF qS) →
    -- ecr/sha-512 for the prover's link finders
    LinkCRZC execArith hs512 P (planZHJP dj y₀) (tabZHJP dj y₀) vb (reg σ) (cont σ) k Rw M t' →
    prob (fun o => LiveAcceptsZKP I own o ∧
        W ⊆ wrongRegZc execArith P (planZHJP dj y₀) (tabZHJP dj y₀) vb
          (fun g o => Bridge.canonAt rows κ g (vb.bit (vb.val o) g) (vb.val o) o.2.2) (reg σ)
          (XplurZC execArith hs512 P (planZHJP dj y₀) (tabZHJP dj y₀) vb (reg σ) (cont σ) k Rw))
        (auditLive L Reg (liveOfZJ dg (planZHJP dj y₀) N wr)) τ ≤
      ENNReal.ofReal ((Properties.Spot.oneStageOf law).Pa ns (W.map Fin.valEmbedding)) +
        ksAvgStrictZ (planZHJP dj y₀) L k qF qS (reg σ) +
        ENNReal.ofReal ((∑ p : vb.Pos,
            (Fintype.card (ΩcZ execArith P (planZHJP dj y₀) (tabZHJP dj y₀) vb L p) : ℝ) / Fintype.card L.Ω) *
          (2 * (t' * (1 + k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * M) + k / (Real.exp 1 * Rw)) / (1 - ρ))

open Classical in
/-- **The canonical registered openings open the deployed commitment at public inputs** (`TransportsLiveCR` at
`keyProgP`, `rsZHJP`, the session at `own` and `constsOfP`, at any sample `L` under a sampling design whose sample files decode to `kd`
units): at every proof block `u`, every committed wire `g` of `u` that isn't a constant, and every opening of `g`'s
registered commit string at the prover's registration that is canonical at `g`, some data opens `g`'s claims position
to the opening's bit there, under the roots of the public file's leaf map at SHA-512. -/
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
    (law : Flock.Draw.Law) (kd : ℕ) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : DrawSetupZHJP (inputsJ I 0) own (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJP (Reg := Reg) dj y₀ S R).length (mPtsOf D.unroll.g (kd / nTab I)) → Discharge.Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ) (L : Law n)
    -- the verify command's tags, and live-verifier at `own` and the sampling design
    (hTags : TagsOkZHJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJPLaw I own dj (fun S R => (planZHJP (Reg := Reg) dj y₀ S R).length) wr
      fun _ => law)
    (hsz : ∀ S ud, Flock.Draw.UnitDraw.ofJson (dj S) = .ok ud → ud.units.size = kd)
    (τ : Strategy (auditLive L Reg (liveOfZJ dg (planZHJP dj y₀) N wr))),
    let P := (keyProgP D.unroll pi hU n).partition (Discharge.Layout.outsOfP D.unroll hU n pi)
    let vb := (rsZHJP dj y₀ ht (Flock.Desc.parseStmt_stmt hD) hscope hU hpi hpub).bindingZ
      (rsZHJP_computes dj y₀ ht (Flock.Desc.parseStmt_stmt hD) hscope hU hpi hpub)
    let σ := simZJA execArith hs512 (planZHJP dj y₀) (AcceptsZKP I own) wr (opnAtZHJP (planZHJP dj y₀) I own dj) dg
      (decAtZJ (planZHJP dj y₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesZCJH_custodyP I own ht (Flock.Desc.parseStmt_stmt hD) law kd hdiv dj y₀ wr L hTags hCust hsz) τ
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
/-- **`SampledKSLiveV5` under the roots guard**: V5's bound without its `ecr/sha-512` term at the roots finder, at
registrations whose roots at the public file's leaf map fix their domains (`Bridge.RootFixesDomainL`), at any sampling
design `law` the verifier samples under from the registration (`Bridge.DrawsUnder`). V5 adds that term in place of the
guard (`Bridge.le_add_ecrBoundL`) and states the bound at `Bridge.designAt`, and `SampledKSLiveV5Accepted` reads the guard
off `verify --zk`'s acceptance. -/
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
    -- the statement's descriptor, and the registration's query, whose population is the game's n proof blocks in order
    (desc : Claims.Msg) (qr : Lean.Json) (_hpop : Bridge.PopMatches desc qr n)
    -- the registration's law object and work table, the sampling design the verifier samples under from them, sampling kd proof
    -- blocks, and that it is no Bernoulli law
    (lj : Lean.Json) (tb : Option Lean.Json) (law : Flock.Draw.Law) (kd : ℕ)
    (hlaw : Bridge.DrawsUnder desc qr lj tb n law kd) (hb : Bridge.NotBernoulli law) (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : DrawSetupZHJP (inputsJ I 0) own (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJP (Reg := Reg) dj y₀ S R).length (mPtsOf D.unroll.g (kd / nTab I)) → Discharge.Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the verifier's operating-system bytes, and A3
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    (_hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the verify command's tags, and live-verifier at `own` and the sampling design
    (hTags : TagsOkZHJ I)
    (hCust : Assumptions.Zk.RecordCustodyZKJPLaw I own dj (fun S R => (planZHJP (Reg := Reg) dj y₀ S R).length) wr
      fun _ => law)
    (k Rw M : ℕ) (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) (ρ : ℝ) (_hρ : ρ < 1)
    (_hr : ∀ S R j, rateZ (planZHJP (Reg := Reg) dj y₀ S R) j k ≤ ρ) (t' : ℝ) (_ht' : 0 ≤ t')
    -- the proof blocks whose miss is `δ`
    (B : Finset ℕ)
    (τ : Strategy (auditLive (Bridge.execSel (n := n) src law n) Reg (liveOfZJ dg (planZHJP dj y₀) N wr)))
    (qF qS : ℝ)
    -- what the sampling statement carries and `RelS` doesn't read
    (q : Claims.Sampling.Query),
    let L := Bridge.execSel (n := n) src law n
    let P := (keyProgP D.unroll pi hU n).partition (Discharge.Layout.outsOfP D.unroll hU n pi)
    let vb := (rsZHJP dj y₀ ht (Flock.Desc.parseStmt_stmt hD) hscope hU hpi hpub).bindingZ
      (rsZHJP_computes dj y₀ ht (Flock.Desc.parseStmt_stmt hD) hscope hU hpi hpub)
    let σ := simZJA execArith hs512 (planZHJP dj y₀) (AcceptsZKP I own) wr (opnAtZHJP (planZHJP dj y₀) I own dj) dg
      (decAtZJ (planZHJP dj y₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
      (decodesZCJH_custodyP I own ht (Flock.Desc.parseStmt_stmt hD) law kd hdiv dj y₀ wr _ hTags hCust
        (Bridge.lawSized_size (Bridge.lawSized_drawsUnder hlaw hb) hCust.2)) τ
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
    let r := Properties.Spot.oneStageOf law
    let ns := Bridge.nsOf desc qr
    let δ : ℝ := (Bridge.designOf r).Pa (Bridge.partOf P κ ns) B
    -- cr/sha-512 for the knowledge finders of each table the prover's session runs
    (∀ ω j, TableCRZ execArith hs512 (planZHJP dj y₀ (L.draw ω) (reg σ)) j
      (zcStrat execArith hs512 (planZHJP dj y₀) (cont σ ω)) qF qS) →
    -- ecr/sha-512 for the prover's link finders
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
under from the registration**: flock-verify `--zk --public-inputs` at `own` accepts and the committed trace fixed at
registration, read as the sampling witness of the session's statement at the constants `constsOfP` (the constant, the
zero and every proof block's public input bits) and the sampling design's design (`oneStageOf law`), is outside `RelS` at `δ`, at
the deployed commitment at the public file's leaf map, with probability at most `δ + ε_ks + ε_bind`. `δ` is not a
parameter: it is the sampling design's own probability of sampling no proof block of a set `B`, `P_a(B)` of its design on
`partOf P κ ns` at the strata `ns` of the registration's layout (`nsOf`). At `subset kd` and the first `K₀` proof blocks
it is `C(n − K₀, kd)/C(n, kd)` (`Bridge.pa_subset_range`), and under `All`, whose sampling design is `subset n`, it is `0` at any
`B` holding a proof block (`Bridge.pa_all`). Its named assumptions are V2's, with `live-verifier` at `own` and the sampling design
(`RecordCustodyZKJPLaw`). The sampling design is the one the verifier samples under from the registration's law object and work table
(`Bridge.designAt`: every sample `OneStage.drawnFrom` draws is under it, `Bridge.drawnFrom_designAt`), with no hypothesis that
the registration names one: at a registration that names none, which the verifier refuses, `designAt` is `subset 0`, which
samples no proof block, so `δ` is 1. It is on its layout's strata at `PopMatches` (`Bridge.lawOn_drawsUnder`), and no
Bernoulli law, though `Draw.flockLaw` parses one: `drawnFrom` refuses it (`OneStage.coveredDesign`, `Bridge.notBernoulli_designAt`). The link finders are charged `t'` per run
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
    -- the statement's descriptor, and the registration's query, whose population is the game's n proof blocks in order
    (desc : Claims.Msg) (qr : Lean.Json) (_hpop : Bridge.PopMatches desc qr n)
    -- the registration's law object and work table
    (lj : Lean.Json) (tb : Option Lean.Json),
    -- the sampling design the verifier samples under from them, sampling kd proof blocks
    let law := Bridge.designAt desc qr lj tb n
    let kd := Bridge.lawK law
    ∀ (hdiv : kd % nTab I = 0)
    (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : DrawSetupZHJP (inputsJ I 0) own (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀)
    (wr : (S : Finset (Fin n)) → (R : Reg) →
      LiveOutJ (planZHJP (Reg := Reg) dj y₀ S R).length (mPtsOf D.unroll.g (kd / nTab I)) → Discharge.Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the verifier's operating-system bytes, and A3
    (Ω : Type) [Fintype Ω] [Nonempty Ω] (Lb : Fin 1 → ℕ) (src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256)
    (_hA3 : Assumptions.UniformRandomBytes Lb src)
    -- live-verifier at `own` and the sampling design
    (hCust : Assumptions.Zk.RecordCustodyZKJPLaw I own dj (fun S R => (planZHJP (Reg := Reg) dj y₀ S R).length) wr
      fun _ => law)
    (k Rw M : ℕ) (_hk : 1 ≤ k) (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) (ρ : ℝ) (_hρ : ρ < 1)
    (_hr : ∀ S R j, rateZ (planZHJP (Reg := Reg) dj y₀ S R) j k ≤ ρ) (t' : ℝ) (_ht' : 0 ≤ t')
    -- the proof blocks whose miss is `δ`
    (B : Finset ℕ)
    (τ : Strategy (auditLive (Bridge.execSel (n := n) src law n) Reg (liveOfZJ dg (planZHJP dj y₀) N wr)))
    (qF qS : ℝ)
    -- what the sampling statement carries and `RelS` doesn't read
    (q : Claims.Sampling.Query),
    if hTags : TagsOkZHJ I then
      let L := Bridge.execSel (n := n) src law n
      let P := (keyProgP D.unroll pi hU n).partition (Discharge.Layout.outsOfP D.unroll hU n pi)
      let vb := (rsZHJP dj y₀ ht (Flock.Desc.parseStmt_stmt hD) hscope hU hpi hpub).bindingZ
        (rsZHJP_computes dj y₀ ht (Flock.Desc.parseStmt_stmt hD) hscope hU hpi hpub)
      let σ := simZJA execArith hs512 (planZHJP dj y₀) (AcceptsZKP I own) wr (opnAtZHJP (planZHJP dj y₀) I own dj) dg
        (decAtZJ (planZHJP dj y₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) N
        (decodesZCJH_custodyP I own ht (Flock.Desc.parseStmt_stmt hD) law kd hdiv dj y₀ wr _ hTags hCust
          (Bridge.lawSized_size (Bridge.sized_designAt desc qr lj tb n) hCust.2)) τ
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
      let r := Properties.Spot.oneStageOf law
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
        (auditLive (Bridge.execSel (n := n) src law n) Reg (liveOfZJ dg (planZHJP dj y₀) N wr)) τ = 0

/-- **V5's assembly**: canonical transport at public inputs and M1's core over `RightRegZc` turn `EndToEndHiddenV5` at
the incorrect proof blocks into `SampledKSLiveV5R`: outside `RelS` at `δ`, the incorrect proof blocks' `P_a` is at most `δ`. -/
def _root_.Flock.Bridge.AssemblyTargetV5 : Prop :=
  Bridge.TransportsLiveV5 → Bridge.FaultyRightTargetC → Bridge.EndToEndHiddenV5 → Bridge.SampledKSLiveV5R

end Flock.Guarantees
