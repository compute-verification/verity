import Proofs.Flock.Recursive.KSStatement
import Proofs.Flock.Claims.Recursion.DeployedBrings
import Proofs.Flock.Claims.Recursion.OpenRowsDeployed
import Proofs.Flock.Claims.Recursion.RecRecord
import Proofs.Flock.Bridge.CircuitOf
import VStarRecursive
import Properties.Verity.Game

/-!
# Sampled proofs' recursive configuration: the component's property and its target (ONE-9p)

Sampled proofs' recursive configuration (`VStar.Recursive`, `verity/sampled-proofs/recursive/v1`, which
`VStar.Recursive.Sampled` runs when the auditor's parameters name `"form": "recursive"`) checks, on the prover's blobs,
the recursive session that `Flock.Guarantees.RecursiveKSAt` is stated about. This module states two things.

* **The property** (`acc_of_accepts`). Take a run of the deployed recursive verifier that `RecursiveKSAt` reads
  (`recordAt`), a statement that names the component's inputs (`Names`), and blobs that show that run (`Shows`). If
  the compiled component accepts those blobs and the records it reads hold the run's live coins (`LiveCoins`), then the
  statement is dense and the run accepts. That is the event `RecursiveKSAt`'s `PrBad` counts.
  - Each V* part's record is a prover-sent blob. The model verifier checks a record's coins against its own live coins
    (`FlockDeploy.custody`), and the compiled component, which holds no live coin, can't. So `LiveCoins` is
    `live-verifier`'s content at the recursive session (`Assumptions.Zk.RecordCustodyZKJ`).
  - `live-verifier` is admissible here only until ONE-4 (PRF-200) lands. After that, the component reads the session's
    coins through ONE-4's signed device log, `LiveCoins` follows from the log's signatures, and the premise goes.
* **The target** (`sampledProofsRecursive`): sampled proofs at the recursive configuration, in
  `Vocab.Target .sampled`'s form (`Properties.Verity.Game`) with the recursive component's verdict (`RecVerdict`).
  - The game is causal (R5): the parameters, the registered circuit file and the commitment come first; then the
    device's releases, each after the bytes it answers; and the evidence last.
  - The claim is read off the commitment (R4, `Vocab.CommitHolds`), the circuit is the committed file's (R6,
    `Vocab.Registered`), and the break is `Vocab.Break`.
  - The error is the draw's miss term plus `t · 2^{-λ}`.

  It takes as hypotheses `RecursiveKSAt` at the deployed brings, the honest half (the honest run is accepted at the
  deployed statements, `HonestHalf`), and, at the session each strategy's play holds (`SessionIn`), the draw's miss,
  the named SHA-512 premises (`CRAt`) and `live-verifier` (`SessionPremises`).

The model this reads (`RecDeploy`, `Flock.Guarantees.RecursiveKSAt`) is still in the proofs tree, so the property sits
here rather than in `Properties` until C-Flock's spec is extracted.
-/

namespace Verity.Recursive

open Claims (Msg Out)
open FlockDeploy (Coin)
open RecDeploy

/-- **The deployed bound record** that `RecursiveKSAt` reads at the weights `wE` and `w`: `bdDeployed` at the
deployed reader `DeployedR.rdR` and V\*'s last message's reader `DeployedR.finalR`. -/
noncomputable def boundAt (wE w : Msg → ℝ) :
    Claims.Recursion.Bound Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace :=
  bdDeployed _root_.Flock.Readable.relation (FlockDeploy.nextR DeployedR.rdR)
    (FlockDeploy.deployPlainTrimR _root_.Flock.Readable.relation DeployedR.rdR).cols wE FlockDeploy.joinR
    rowInDeployed DeployedR.finalR w

/-- **The deployed recursive record** that `RecursiveKSAt DeployedR.rdR DeployedR.finalR OpenRows.elemR wE w
(deployedB b) aux nOf hd recRest` is stated over: `recBoundAt` at the deployed brings, with the coin server's record
`recTr aux nOf hd recRest`, at V\*'s layout `DeployedR.layR OpenRows.elemR`, whose inner view runs `IVStar`. -/
noncomputable def recordAt (wE w : Msg → ℝ) (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round)
    (nOf : Msg → ℕ → ℕ) (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8) :
    Claims.Recursion.Deploy Coin :=
  (recBoundAt _root_.Flock.Readable.relation { deployedB b with tr := recTr aux nOf hd recRest }
    (DeployedR.layR OpenRows.elemR)).record (boundAt wE w).IVStar

/-- The staged inputs of `recordAt`'s verifier: the coin server's record `recTr aux nOf hd recRest` read as
`Stage.ofMsg` reads it. -/
noncomputable def stageAt (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8) : Stage :=
  Stage.ofMsg (recTr aux nOf hd recRest)

/-- The inner and staging rounds of the recursive verifier at `x`. -/
def kIS (IV : Claims.Verifier Coin) (x : Msg) : ℕ := IV.k (xIOf x) + kS x

/-- Session `s`'s last message, as `FlockDeploy.Final`, in the run's view `v`. -/
noncomputable def finalAt (IV : Claims.Verifier Coin) (st : Stage) (x : Msg) (s : ℕ) (v : Claims.Seen) :
    Option FlockDeploy.Final :=
  ((session IV st x s v).msgs.getLast?).bind fun m => FlockDeploy.finalOf m.1

/-- The file name `flock-verify` gives a rep stream `w`'s proof (`Witness.lean`'s `{w.replace "/" "."}.bin`). -/
def repName (w : String) : String := s!"{w.replace "/" "."}.bin"

/-- **The blobs show the run.** In the run's view `v` at statement `x`, the recursive session's files that the compiled
component reads are:
* the coin server's transcript, its record of the inner rounds (`Stage.transcript`, `recTr`);
* the inner session's record and each part's public file, off the staging message (`Stage.record`, `Stage.pub`);
* and, for each part `s`, the record and the rep files of session `s`'s last message, whose statement is session `s`'s
  V\* statement (`vstarAt`), each rep file under the name its stream gives, in the stream order of that session's
  recursive setup (`setupO`). -/
def Shows (IV : Claims.Verifier Coin) (st : Stage) (x : Msg) (pr : VStar.Recursive.Proofs) (bs : Core.Blobs)
    (v : Claims.Seen) : Prop :=
  bs.get pr.transcript "rec inner transcript" = .ok (st.transcript x (v.msgs.take (kIS IV x))) ∧
  bs.get pr.record "the inner session's record" = .ok (st.record x (v.msgs.take (kIS IV x))) ∧
  ∀ s q, pr.parts[s]? = some q →
    bs.get q.pub "a part's public file" = .ok (st.pub x (v.msgs.take (kIS IV x)) s) ∧
    ∃ f, finalAt IV st x s v = some f ∧ f.stmt = vstarAt st x (v.msgs.take (kIS IV x)) s ∧
      bs.get q.record "a part's record" = .ok f.record.toUTF8 ∧
      ∀ i spec sts, setupO (vstarAt st x (v.msgs.take (kIS IV x)) s) = some (i, spec, sts) →
        ∀ k (h : k < spec.repStreams.size), ∃ d,
          (q.reps.find? (·.1 == repName spec.repStreams[k].2.2)).map (·.2) = some d ∧
          ∃ hk : k < f.proofs.size, bs.get d "session files" = .ok f.proofs[k]

/-- **The records hold the live coins** (`live-verifier` at the recursive session, `Assumptions.Zk.RecordCustodyZKJ`):
at each V\* session `s` of the run's view `v`, wherever the outer verifier decides on session `s`'s messages and coins
(`actO`), the live session the record shows is the session's link message, first coin and round messages with their
coins (`FlockDeploy.custody`). The model verifier checks this itself. The compiled component can't, since the coins are
the coin server's. It is admissible until ONE-4 (PRF-200) lands; then the component reads the coins through ONE-4's
signed device log, and this premise goes. -/
noncomputable def LiveCoins (IV : Claims.Verifier Coin) (st : Stage) (x : Msg) (v : Claims.Seen) : Prop :=
  ∀ s < nS x, ∀ i spec sts link c₁ mids f,
    actO (vstarAt st x (v.msgs.take (kIS IV x)) s) ((session IV st x s v).msgs.map Prod.fst)
        ((session IV st x s v).msgs.map fun m => outCoin m.2) =
      .decide i spec sts link c₁ mids f →
    (do FlockDeploy.custody spec (← FlockDeploy.liveOf spec f.record) link c₁ mids : Flock.V Unit) = .ok ()

/-- **The statement names the component's inputs.** `x` is a compiled statement (`decodeX`) whose parts are:
* its inner statement, which is the inner session's as `innerHeld` reads it off the blobs: the circuit, public file,
  partition, program and draw;
* its own inner shape, which is the one `innerHeld` derives;
* its registration, which is the evidence's;
* its bindings, which are every part's;
* its part `j`, which carries `p.own.parts[j]`'s circuit, partition, program, registration and rows.

Piece B of PRF-202's split owns this definition, and refines it where the model reads a field it misses. -/
def Names (p : VStar.Recursive.Params) (e : VStar.Recursive.Evidence) (bs : Core.Blobs) (x : Msg) : Prop :=
  ∃ w ri, decodeX x = some w ∧ FlockDeploy.decode w.inner = some ri ∧
    bs.get p.own.circuit "the inner circuit" = .ok ri.circuit ∧
    bs.get e.proofs.pub "the inner public file" = .ok ri.publicFile ∧
    ri.statement = VStar.Recursive.INNER_STATEMENT ∧ ri.tables = [] ∧
    ri.partition.map (FlockDeploy.jsonOf "partition") = some (.ok p.own.partition) ∧
    ri.program = some p.own.program ∧
    ri.draw = (bs.get e.proofs.draw "the inner statement's draw").toOption ∧
    (∀ h, VStar.Recursive.innerHeld p.own e.proofs bs = .ok h →
      (do Flock.RecVStar.Inner.ofJson (← FlockDeploy.jsonOf "rec-inner" w.recInner)) = .ok h.shape) ∧
    (e.proofs.registration.mapM fun d => (bs.get d "rec inner registration").toOption) = some w.registration ∧
    w.parts.length = p.own.parts.size ∧
    ∀ j (hj : j < p.own.parts.size) (q : PartX), w.parts[j]? = some q →
      FlockDeploy.jsonOf "rec-bindings" w.bindings = .ok p.own.parts[j].bindings ∧
      bs.get p.own.parts[j].circuit s!"part {j}'s circuit" = .ok q.base.circuit ∧
      q.base.partition.map (FlockDeploy.jsonOf "partition") = some (.ok p.own.parts[j].partition) ∧
      q.base.program = some p.own.parts[j].program ∧
      q.base.registered.map (FlockDeploy.jsonOf "registered") = p.own.parts[j].registered.map Except.ok ∧
      q.rows.map (FlockDeploy.jsonOf "rec-rows") = p.own.parts[j].rows.map Except.ok

/-- **ONE-9p's property: the recursive component's acceptance is the event `RecursiveKSAt` is stated over.** Suppose
the compiled recursive component accepts `(p, c, e, bs)`, `x` names its inputs (`Names`), the blobs show a run of the
deployed recursive verifier (`recordAt`) at `x` (`Shows`), and that run's records hold its live coins (`LiveCoins`).
Then `x` is dense (`RecDeploy.DenseAt`, the statements `RecursiveKSAt` covers), and the run accepts. -/
theorem acc_of_accepts (wE w : Msg → ℝ) (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (p : VStar.Recursive.Params) (c : SampledProofs.OneStage.Commitment) (e : VStar.Recursive.Evidence)
    (bs : Core.Blobs) {pub : SampledProofs.OneStage.Public} {prof : Core.Integrity}
    (hacc : VStar.Recursive.checkBlobs p c e bs = .accept pub prof)
    (x : Msg) (hnames : Names p e bs x) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω)
    (r : Fin ((recordAt wE w b aux nOf hd recRest).V.k x) → Coin)
    (hshow : Shows (boundAt wE w).IVStar (stageAt aux nOf hd recRest) x e.proofs bs
      (Claims.run FlockDeploy.sha512 P (recordAt wE w b aux nOf hd recRest).V x ω r).seen)
    (hlive : LiveCoins (boundAt wE w).IVStar (stageAt aux nOf hd recRest) x
      (Claims.run FlockDeploy.sha512 P (recordAt wE w b aux nOf hd recRest).V x ω r).seen) :
    RecDeploy.DenseAt x ∧
      (Claims.run FlockDeploy.sha512 P (recordAt wE w b aux nOf hd recRest).V x ω r).out = .acc := by
  sorry

/-! ## The target -/

open Accounting (Device)

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α]

/-- **The recursive configuration's verdict**: `Vocab.Verdict .sampled` with the recursive configuration's compiled
component in place of onsite's sampled component. The conjuncts:
* the circuit file passed the commit phase's check (`Registered`, R6);
* the parameters name the recursive form;
* sampled proofs' component accepts what the prover sent, on some blobs;
* every release its check read is in the device's log.

The blobs are content-addressed bytes sent with the evidence, last, and the event counts any blobs on which the
component accepts. -/
def RecVerdict (V : Vocab d α) (o : Verity.Outcome) : Prop :=
  V.Registered o.params o.file ∧
    (∃ j p, Core.canonicalJson o.params = .ok j ∧ VStar.Recursive.Sampled.paramsOfJson j = .ok (.inr p)) ∧
    (∃ bs, (VStar.Recursive.Sampled.protocol.answer o.params o.commitment o.evidence bs).verdict = .accept) ∧
    ∀ r ∈ V.coinsRead .sampled o.sent, r ∈ o.log

/-- **The honest half** (non-vacuity, from HONESTIV). On every dense statement where the inner relation holds, the
deployed record's honest prover is accepted except with the record's completeness error `δcRec`, at every tape and
every inner decoder. -/
def HonestHalf (wE w : Msg → ℝ) (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8) : Prop :=
  ∀ decI, Claims.Complete FlockDeploy.sha512 (recordAt wE w b aux nOf hd recRest).V
    (recordAt wE w b aux nOf hd recRest).P
    (Claims.Recursion.R (recordAt wE w b aux nOf hd recRest)
      (Claims.Recursion.Inner.ofFlock (boundAt wE w).deploy decI))
    RecDeploy.DenseAt ((recordAt wE w b aux nOf hd recRest).δc FlockDeploy.sha512)

/-- **The named SHA-512 premises at a prover**, with the run's oracle calls bounded by `t` and the level at least `λ`.
These are `BitsLin2On`'s hypotheses at `recordAt`:
* the run makes at most `t` oracle calls;
* the inner and setup calls are bounded by `n` and `v`;
* `StrictBirthday` holds at `B`, and `ExpectedCR` at the inner, device, forked-coin and first-coin finders;
* `lamLin2` is at least `λ` there. -/
def CRAt (wE w : Msg → ℝ) (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8) (t : ℕ) (lam : ℝ)
    (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω) : Prop :=
  let D := recordAt wE w b aux nOf hd recRest
  let Bd := boundAt wE w
  let H := FlockDeploy.sha512
  (∀ ω r, (Claims.run H P D.V x ω r).pLog.length + (Claims.run H P D.V x ω r).vLog.length ≤ t) ∧
  ∀ decI, Claims.Recursion.decOK Bd.deploy decI → ∀ decO,
    Claims.Recursion.decOK (deployZkRec _root_.Flock.Readable.relation) decO →
    ∃ n v : ℕ, (∀ cs ws, (D.innerCalls H x cs ws).length ≤ n) ∧
      (∀ ps, (Claims.Recursion.setupCalls H D (D.vstar x ps)).length ≤ v) ∧
      lam ≤ Claims.Recursion.lamLin2 D (Claims.Recursion.Inner.ofFlock Bd.deploy decI) FlockDeploy.KS.CpDep x n v ∧
      (∀ hk : 0 < D.V.k x, Claims.StrictBirthday H (Claims.B H D.V P x hk)) ∧
      (∀ hk : 0 < (Claims.Recursion.Inner.ofFlock Bd.deploy decI).V.k (Claims.Recursion.xI D x),
        Claims.ExpectedCR H ((Claims.Recursion.BinnerV H D (Claims.Recursion.Inner.ofFlock Bd.deploy decI)
          (deployZkRec _root_.Flock.Readable.relation) decO P x hk).charge
            (Claims.Recursion.setupLen D x + 2 * (n + v)))) ∧
      (∀ (M : ℕ) (hk : 0 < (Claims.Recursion.Inner.ofFlock Bd.deploy decI).V.k (Claims.Recursion.xI D x)),
        Claims.ExpectedCR H ((Claims.Recursion.FdevV H D (Claims.Recursion.Inner.ofFlock Bd.deploy decI)
          (deployZkRec _root_.Flock.Readable.relation) decO P x M hk).charge
            (Claims.Recursion.setupLen D x + 2 * (n + v)))) ∧
      (∀ (M : ℕ) (hW : 0 < FlockDeploy.KS.CpDep.width (Claims.Recursion.xI D x)),
        Claims.ExpectedCR H ((Claims.Recursion.FCI H D (Claims.Recursion.Inner.ofFlock Bd.deploy decI)
          (deployZkRec _root_.Flock.Readable.relation) decO FlockDeploy.KS.CpDep P x M hW).charge
            (Claims.Recursion.setupLen D x + 2 * (n + v)))) ∧
      (∀ R : ℕ, Claims.ExpectedCR H ((Claims.Recursion.F0I H D (Claims.Recursion.Inner.ofFlock Bd.deploy decI)
          (deployZkRec _root_.Flock.Readable.relation) decO P x R).charge
            (Claims.Recursion.setupLen D x +
              ((Claims.Recursion.Inner.ofFlock Bd.deploy decI).k (Claims.Recursion.xI D x) + 1) * (n + v))))

/-- **The recursive session inside the causal game** at `θ`, `F`, `c` and the strategy `σ`: the coupling H's proof
builds from ONE-4's device log (`Vocab.rounds`, `Vocab.coinsRead`).
* The device's releases before the session, its draw among them, are `ξ`, uniform on `Ξ`.
* At `ξ` the strategy plays the recursive session at the statement `x ξ` as the prover `P ξ`, against the session's
  live coins. `P ξ`'s tape in `Ω` holds every other release.
* `out ξ s` is the play's outcome at `ξ`, at the tape and the session's coins `s`. The play's outcome under the
  device's uniform releases is distributed so (`prob_le`).
* `drawOf ξ` is the units `ξ` draws.
* Wherever the recursive component accepts on blobs `bs`, the parameters and evidence it decodes name `x ξ`, and the
  blobs show the run (`names_shows`). -/
structure SessionIn (V : Vocab d α) (Dp : Claims.Recursion.Deploy Coin) (IV : Claims.Verifier Coin) (st : Stage)
    (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) where
  Ξ : Type
  Ω : Type
  [finΞ : Fintype Ξ]
  [neΞ : Nonempty Ξ]
  [finΩ : Fintype Ω]
  [neΩ : Nonempty Ω]
  x : Ξ → Msg
  P : Ξ → Claims.Prover Ω
  out : (ξ : Ξ) → Ω × (Fin (Dp.V.k (x ξ)) → Coin) → Verity.Outcome
  drawOf : Ξ → Finset (Fin (V.units F))
  prob_le : ∀ E : Verity.Outcome → Prop, Game.prob E (V.game θ F c) σ ≤
    ENNReal.ofReal (Claims.Ex fun ξ => Claims.Pr fun s : Ω × (Fin (Dp.V.k (x ξ)) → Coin) => E (out ξ s))
  names_shows : ∀ ξ s bs j p e,
    (VStar.Recursive.Sampled.protocol.answer (out ξ s).params (out ξ s).commitment (out ξ s).evidence bs).verdict =
      .accept →
    Core.canonicalJson (out ξ s).params = .ok j → VStar.Recursive.Sampled.paramsOfJson j = .ok (.inr p) →
    VStar.Recursive.Sampled.evidenceOf (out ξ s).evidence = .ok (.inr e) →
    Names p e (Core.Blobs.ofArray bs) (x ξ) ∧
      Shows IV st (x ξ) e.proofs (Core.Blobs.ofArray bs) (Claims.run FlockDeploy.sha512 (P ξ) Dp.V (x ξ) s.1 s.2).seen

attribute [instance] SessionIn.finΞ SessionIn.neΞ SessionIn.finΩ SessionIn.neΩ

/-- **The premises at a session**:
* the draw misses any `K₀` units with probability at most `δs`;
* each prover meets the named SHA-512 premises at `t` calls and level `λ` (`CRAt`);
* the records hold the run's live coins (`LiveCoins`, `live-verifier`). `live-verifier` is admissible until ONE-4
  (PRF-200) lands; then the component reads the coins through ONE-4's signed device log and this conjunct goes. -/
def SessionPremises (wE w : Msg → ℝ) (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8) (V : Vocab d α)
    (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c))
    (S : SessionIn V (recordAt wE w b aux nOf hd recRest) (boundAt wE w).IVStar (stageAt aux nOf hd recRest) θ F c σ)
    (K₀ t : ℕ) (δs lam : ℝ) : Prop :=
  (∀ U : Finset (Fin (V.units F)), K₀ ≤ U.card → Claims.Pr (fun ξ : S.Ξ => Disjoint (S.drawOf ξ) U) ≤ δs) ∧
  (∀ ξ, CRAt wE w b aux nOf hd recRest t lam (S.x ξ) S.Ω (S.P ξ)) ∧
  (∀ ξ ω r, LiveCoins (boundAt wE w).IVStar (stageAt aux nOf hd recRest) (S.x ξ)
    (Claims.run FlockDeploy.sha512 (S.P ξ) (recordAt wE w b aux nOf hd recRest).V (S.x ξ) ω r).seen)

/-- **ONE-9p's target: sampled proofs at the recursive configuration**, in `Vocab.Target .sampled`'s form over the causal
game (R4, R5, R6), with the recursive configuration's verdict (`RecVerdict`). At all parameters `θ`, circuit files `F`
and commitments `c`, and for every strategy, the probability that the recursive component accepts while sampled
proofs' claim read off the commitment fails (`¬ CommitHolds`), with no break, is at most `δs + t · 2^{-λ}`.

It takes these hypotheses:
* `RecursiveKSAt` at the deployed brings, and the honest half (`HonestHalf`, HONESTIV);
* the vocabulary's circuit is ONE-7's at the registered file (`hC`);
* `K₀` covers the claim's `ε` share (`hK₀`);
* every strategy's play holds a recursive session (`SessionIn`) whose premises hold (`hS`). The coupling is H's own
  obligation from ONE-4's log. The premises are the draw's miss, the named SHA-512 premises at the session's prover,
  and `live-verifier`, which goes once ONE-4 (PRF-200) lands. -/
theorem sampledProofsRecursive (wE w : Msg → ℝ) (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (hKS : Flock.Guarantees.RecursiveKSAt DeployedR.rdR DeployedR.finalR OpenRows.elemR wE w (deployedB b) aux nOf
      hd recRest)
    (hHon : HonestHalf wE w b aux nOf hd recRest)
    (V : Vocab d α)
    (hC : ∀ θ F, V.Registered θ F → ∃ (D : Flock.Desc) (hU : FlockSoundness.Built D.unroll.unit) (tags : Flock.Tags),
      Flock.Desc.parse tags F = .ok D ∧ V.circuitOf F = Flock.Bridge.circuitOf D hU (V.units F) ∧
        HEq (V.partsOf F) (Flock.Bridge.partsOf D hU (V.units F)))
    (K₀ : ByteArray → ByteArray → ℕ) (δs lam : ℝ) (t : ℕ)
    (hK₀ : ∀ θ F (m : ℕ), (V.certOf θ).ε * V.units F < m → K₀ θ F ≤ m)
    (hS : ∀ θ F c (σ : Game.Strategy (V.game θ F c)),
      ∃ S : SessionIn V (recordAt wE w b aux nOf hd recRest) (boundAt wE w).IVStar (stageAt aux nOf hd recRest) θ F c σ,
        SessionPremises wE w b aux nOf hd recRest V θ F c σ S (K₀ θ F) t δs lam) :
    ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)),
      Game.prob (fun o => RecVerdict V o ∧ ¬ V.CommitHolds θ F c ∧ ¬ V.Break θ F c o) (V.game θ F c) σ ≤
        ENNReal.ofReal (δs + t * (2 : ℝ) ^ (-lam)) := by
  sorry

end Verity.Recursive
