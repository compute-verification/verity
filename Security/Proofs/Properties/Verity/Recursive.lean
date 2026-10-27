import Proofs.Flock.Recursive.KSStatement
import Proofs.Flock.Claims.Recursion.DeployedBrings
import Proofs.Flock.Claims.Recursion.OpenRowsDeployed
import Proofs.Flock.Claims.Recursion.RecRecord
import Proofs.Flock.Bridge.CircuitOf
import VStarRecursive
import Properties.Verity.Game
import Proofs.Properties.Verity.Soundness
import Proofs.Properties.Verity.Recursive.Compiled
import Proofs.Properties.Verity.Recursive.Numbers
import Proofs.Properties.Verity.Recursive.Names
import Proofs.Properties.Verity.Recursive.PartStatement
import Proofs.Properties.Verity.Recursive.PartVerdict
import Proofs.Properties.Verity.Recursive.SetupChecks
import Proofs.Properties.Verity.Recursive.ScheduleRun

/-!
# Sampled proofs' recursive configuration: the component's property and its target (ONE-9p)

Sampled proofs' recursive configuration (`VStar.Recursive`, `verity/sampled-proofs/recursive/v1`, which
`VStar.Recursive.Sampled` runs when the auditor's parameters name `"form": "recursive"`) checks, on the prover's blobs,
the recursive session that `Flock.Guarantees.RecursiveKSAt` is stated about. This module states two things.

* **The property** (`acc_of_accepts`). Take a run of the deployed recursive verifier that `RecursiveKSAt` reads
  (`recordAt`), a statement that names the component's inputs (`Names`), and blobs that show that run (`Shows`), whose
  staged rows are the component's (`RowsShow`). If the compiled component accepts those blobs and the records it reads
  hold the run's live coins (`LiveCoins`), then the statement is dense and the run accepts. That is the event `RecursiveKSAt`'s `PrBad` counts.
  - Each V* part's record is a prover-sent blob. The model verifier checks a record's coins against its own live coins
    (`FlockDeploy.custody`), and the compiled component, which holds no live coin, can't. So `LiveCoins` is
    `live-verifier`'s content at the recursive session (`Assumptions.Zk.RecordCustodyZKJ`).
  - `live-verifier` is admissible here only until ONE-4 (PRF-200) lands. After that, the component reads the session's
    coins through ONE-4's signed device log, `LiveCoins` follows from the log's signatures, and the premise goes.
* **The target** (`sampledProofsRecursive`): sampled proofs at the recursive configuration, in
  `Vocab.Target .sampled`'s form (`Properties.Verity.Game`) with the recursive component's verdict (`RecVerdict`).
  - The game is causal (R5): the parameters, the registered circuit file and the commitment come first; then the
    device's releases, each after the bytes it answers; and the evidence last.
  - The claim reads one fixed extractor's transcript (R4′, `Vocab.extract`): `Dep.extract`, at each bit of each gate
    the value most often read off the witnesses `RecursiveKSAt`'s extractor returns, over the draws and the accepted
    runs that draw the gate's unit. The circuit is the committed file's (R6, `Vocab.Registered`).
  - Daniel's per-prover form: the bound holds at every strategy, with no hypothesis on it but `live-verifier`. It is
    ONE8's `condError FinderCR δ` (`Dep.err`, the shape of `Vocab.error`): where the finders built from the strategy
    meet their named SHA-512 premises (`Dep.FinderCR`: C-Flock's at each session prover, `ecr/sha-512` at each
    binding finder `Dep.bindAt` at its budget), the draw's miss plus `t' · 2^{-λ}` plus the binding term (`Dep.delta`);
    elsewhere `⊤`. Every budget is a defined function of the strategy's cost `t'` (`Vocab.cost`), with no call model.
  - Beside it, `sampledProofsRecursive_nontrivial` (an accept-everything verifier breaks the bound) and
    `sampledProofsRecursive_honest` (the hypotheses that are the deployment's choices hold at the honest one).

  It takes as hypotheses `RecursiveKSAt` at the deployed brings, the honest half (the honest run is accepted at the
  deployed statements, `HonestHalf`), the coupling at each strategy (`SessionIn`, ONE-4's projection) with the draw's
  miss, the deployment's reading of a witness (`hread`) and its collision from two (`hcoll`), and `live-verifier`.

The model this reads (`RecDeploy`, `Flock.Guarantees.RecursiveKSAt`) is still in the proofs tree, so the property sits
here rather than in `Properties` until C-Flock's spec is extracted.
-/

namespace Verity.Recursive

open scoped ENNReal
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
        f.proofs.size = spec.repStreams.size ∧ ∀ k (h : k < spec.repStreams.size), ∃ d,
          (q.reps.find? (·.1 == repName spec.repStreams[k].2.2)).map (·.2) = some d ∧
          ∃ hk : k < f.proofs.size, bs.get d "session files" = .ok f.proofs[k]

/-- **The staged rows are the component's.** V\*'s shared rows (`--rec-rows`) are no part of `x` (#4744): the prover
stages one set for every part, off the staging message (`Stage.rows`). The compiled component reads each part's rows
from its parameters, so the run's staged rows, read as JSON, are every part's. -/
def RowsShow (IV : Claims.Verifier Coin) (st : Stage) (x : Msg) (o : VStar.Recursive.Own) (v : Claims.Seen) : Prop :=
  ∀ j (hj : j < o.parts.size),
    o.parts[j].rows.map Except.ok = some (FlockDeploy.jsonOf "rec-rows" (st.rows x (v.msgs.take (kIS IV x))))

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

/-- **G, the schedule's half: the run accepts where every V\* session sets up at its staged statement and the outer
verifier accepts it there** (F3's `run_acc_sha512`, with C2's `numsOf_staged` for its numbers and `vstarAt_staged` for
its statement). -/
theorem run_acc_of_staged (IV : Claims.Verifier Coin) (st : Stage) (x : Msg) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω)
    (r : Fin ((RecDeploy.V IV st).k x) → Coin) (hn : 0 < nS x)
    (hset : ∀ s < nS x, ∃ o, setupO (staged st x
      ((Claims.run FlockDeploy.sha512 P (RecDeploy.V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) s) = some o)
    (hacc : ∀ s < nS x, ((nextO (staged st x
        ((Claims.run FlockDeploy.sha512 P (RecDeploy.V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) s)
        ((session IV st x s (Claims.run FlockDeploy.sha512 P (RecDeploy.V IV st) x ω r).seen).msgs.map Prod.fst)
        ((session IV st x s (Claims.run FlockDeploy.sha512 P (RecDeploy.V IV st) x ω r).seen).msgs.map
          fun m => outCoin m.2)).eval FlockDeploy.sha512).1 = .acc) :
    (Claims.run FlockDeploy.sha512 P (RecDeploy.V IV st) x ω r).out = .acc := by
  refine Schedule.run_acc_sha512 IV st x P ω r hn (fun s hs => ?_) (fun s hs => ?_)
  · obtain ⟨o, ho⟩ := hset s hs
    exact numsOf_staged ho
  · obtain ⟨o, ho⟩ := hset s hs
    rw [vstarAt_staged ho]
    exact hacc s hs

/-- **ONE-9p's property: the recursive component's acceptance is the event `RecursiveKSAt` is stated over.** Suppose
the compiled recursive component accepts `(p, c, e, bs)`, `x` names its inputs (`Names`), the blobs show a run of the
deployed recursive verifier (`recordAt`) at `x` (`Shows`) whose staged rows are the component's (`RowsShow`), and
that run's records hold its live coins (`LiveCoins`).
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
    (hrows : RowsShow (boundAt wE w).IVStar (stageAt aux nOf hd recRest) x p.own
      (Claims.run FlockDeploy.sha512 P (recordAt wE w b aux nOf hd recRest).V x ω r).seen)
    (hlive : LiveCoins (boundAt wE w).IVStar (stageAt aux nOf hd recRest) x
      (Claims.run FlockDeploy.sha512 P (recordAt wE w b aux nOf hd recRest).V x ω r).seen) :
    RecDeploy.DenseAt x ∧
      (Claims.run FlockDeploy.sha512 P (recordAt wE w b aux nOf hd recRest).V x ω r).out = .acc := by
  sorry

/-- **Sampled proofs' one-stage form is v2** (the bridge's carry-over lemma, soundness 02:10Z): where the parameters name
the one-stage form and `VStar.Recursive.Sampled` accepts, the evidence is one-stage too and v2
(`SampledProofs.OneStageZk.checkBlobs`) accepts the same commitment, evidence and blobs with the same public output and
profile. So what is proved of v2's acceptance (ONE3P's `acceptMeans`, ONE8's sampled proofs bound) holds of the
composed component's one-stage branch. -/
theorem sampled_oneStage_accepts {p₀ : SampledProofs.OneStageZk.Params} {c : SampledProofs.OneStage.Commitment}
    {e : VStar.Recursive.Sampled.Evidence} {bs : Core.Blobs} {pub : SampledProofs.OneStage.Public}
    {prof : Core.Integrity}
    (h : VStar.Recursive.Sampled.checkBlobs (.inl p₀) c e bs = .accept pub prof) :
    ∃ e₀, e = .inl e₀ ∧ SampledProofs.OneStageZk.checkBlobs p₀ c e₀ bs = .accept pub prof := by
  cases e with
  | inl e₀ => exact ⟨e₀, rfl, h⟩
  | inr e₁ => simp [VStar.Recursive.Sampled.checkBlobs] at h

/-- `sampled_oneStage_accepts` at the component, `VStar.Recursive.Sampled.protocol`'s blob check. -/
theorem sampled_protocol_oneStage_accepts {p₀ : SampledProofs.OneStageZk.Params}
    {c : SampledProofs.OneStage.Commitment} {e : VStar.Recursive.Sampled.Evidence} {bs : Core.Blobs}
    {pub : SampledProofs.OneStage.Public} {prof : Core.Integrity}
    (h : VStar.Recursive.Sampled.protocol.checkBlobs
      (show VStar.Recursive.Sampled.protocol.Params from Sum.inl p₀) c e bs = .accept pub prof) :
    ∃ e₀, e = .inl e₀ ∧ SampledProofs.OneStageZk.checkBlobs p₀ c e₀ bs = .accept pub prof :=
  sampled_oneStage_accepts h

/-! ## The target -/

open Accounting (Device)

variable {ι : Type} {d : Device ι} {α : Type} [Fintype α]

/-- **The recursive configuration's verdict**: `Vocab.Verdict .sampled` with the recursive configuration's compiled
component in place of onsite's sampled component. The conjuncts:
* the circuit file passed the commit phase's check (`Registered`, R6);
* NCI accepted the registration sampled proofs committed to (`RegistrationAccepts`);
* the parameters name the recursive form;
* sampled proofs' component accepts what the prover sent, on some blobs.

The blobs are content-addressed bytes sent with the evidence, last, and the event counts any blobs on which the
component accepts. -/
def RecVerdict (V : Vocab d α) (o : Verity.Outcome) : Prop :=
  V.Registered o.params o.file ∧ RegistrationAccepts o.sent ∧
    (∃ j p, Core.canonicalJson o.params = .ok j ∧ VStar.Recursive.Sampled.paramsOfJson j = .ok (.inr p)) ∧
    ∃ bs, (VStar.Recursive.Sampled.protocol.answer o.params o.commitment o.evidence bs).verdict = .accept

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

/-- **The deployment** `RecursiveKSAt` is stated at (the weights, the brings and the coin server's record), with the
inner decoder `decI` and the outer decoder `decO` that fix its extractor. -/
structure Dep where
  wE : Msg → ℝ
  w : Msg → ℝ
  b : Brings
  aux : Msg → ℕ → Flock.RecRecord.Round
  nOf : Msg → ℕ → ℕ
  hd : Msg → Flock.RecRecord.Record
  recRest : Msg → List (Msg × Out) → List UInt8
  decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace
  decO : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.ZkTrace

namespace Dep

variable (q : Dep)

/-- The deployed recursive record. -/
noncomputable def D : Claims.Recursion.Deploy Coin := recordAt q.wE q.w q.b q.aux q.nOf q.hd q.recRest

/-- The deployed bound record. -/
noncomputable def Bd : Claims.Recursion.Bound Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace := boundAt q.wE q.w

/-- The staged inputs of the deployed record's verifier. -/
noncomputable def st : Stage := stageAt q.aux q.nOf q.hd q.recRest

/-- The inner protocol, Flock at the inner decoder. -/
noncomputable def I : Claims.Recursion.Inner Coin Claims.Flock.Wit :=
  Claims.Recursion.Inner.ofFlock q.Bd.deploy q.decI

/-- The recursive relation: the statement parses and the inner relation holds at its inner statement. -/
noncomputable def R : Claims.Rel Claims.Flock.Wit := Claims.Recursion.R q.D q.I

/-- **The recursive session's fixed extractor**: `RecursiveKSAt`'s `E` at the two decoders. -/
noncomputable def E : Claims.Extractor Coin Claims.Flock.Wit q.D.V :=
  Claims.Recursion.E FlockDeploy.sha512 q.D q.I (deployZkRec _root_.Flock.Readable.relation) q.decO

/-- **The extractor's reruns at `x`**, at most (`Claims.Recursion.Efficient`'s worst-case count): `1 + waits · cap` for
the inner protocol, times `1 + Σ waits · cap` over the outer sessions. -/
noncomputable def runs (x : Msg) : ℕ :=
  (1 + q.I.waits (Claims.Recursion.xI q.D x) * q.I.cap (Claims.Recursion.xI q.D x)) *
    (1 + ∑ s : Fin (q.D.nS x), (q.D.nums x s).waits * (q.D.nums x s).cap)

end Dep

/-- The SHA-512 calls of one run: the prover's and the verifier's. -/
def runCalls (t : Claims.Trace) : ℕ := t.pLog.length + t.vLog.length

/-- **The recursive session inside the causal game** at `θ`, `F`, `c` and the strategy `σ`: the coupling of ONE-4's
device log (`Vocab.rounds`, `Vocab.coinsRead`) with the session.
* The device's releases before the session, its draw among them, are `ξ`, uniform on `Ξ`.
* At `ξ` the strategy plays the recursive session at the statement `x ξ` as the prover `P ξ`, against the session's
  live coins. `P ξ`'s tape in `Ω` holds every other release.
* `out ξ s` is the play's outcome at `ξ`, at the tape and the session's coins `s`. The play's outcome under the
  device's uniform releases is distributed so (`prob_le`).
* `drawOf ξ` is the units `ξ` draws.
* Wherever the recursive component accepts on blobs `bs`, the parameters and evidence it decodes name `x ξ`, and the
  blobs show the run and its staged rows are the component's (`names_shows`).
* Each run of the session is part of one run of `σ`, so its SHA-512 calls are at most the strategy's, `V.cost`
  (`calls_le`). -/
structure SessionIn (V : Vocab d α) (q : Dep) (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)) where
  Ξ : Type
  Ω : Type
  [finΞ : Fintype Ξ]
  [neΞ : Nonempty Ξ]
  [finΩ : Fintype Ω]
  [neΩ : Nonempty Ω]
  x : Ξ → Msg
  P : Ξ → Claims.Prover Ω
  out : (ξ : Ξ) → Ω × (Fin (q.D.V.k (x ξ)) → Coin) → Verity.Outcome
  drawOf : Ξ → Finset (Fin (V.units F))
  prob_le : ∀ E : Verity.Outcome → Prop, Game.prob E (V.game θ F c) σ ≤
    ENNReal.ofReal (Claims.Ex fun ξ => Claims.Pr fun s : Ω × (Fin (q.D.V.k (x ξ)) → Coin) => E (out ξ s))
  names_shows : ∀ ξ s bs j p e,
    (VStar.Recursive.Sampled.protocol.answer (out ξ s).params (out ξ s).commitment (out ξ s).evidence bs).verdict =
      .accept →
    Core.canonicalJson (out ξ s).params = .ok j → VStar.Recursive.Sampled.paramsOfJson j = .ok (.inr p) →
    VStar.Recursive.Sampled.evidenceOf (out ξ s).evidence = .ok (.inr e) →
    Names p e (Core.Blobs.ofArray bs) (x ξ) ∧
      Shows q.Bd.IVStar q.st (x ξ) e.proofs (Core.Blobs.ofArray bs)
        (Claims.run FlockDeploy.sha512 (P ξ) q.D.V (x ξ) s.1 s.2).seen ∧
      RowsShow q.Bd.IVStar q.st (x ξ) p.own (Claims.run FlockDeploy.sha512 (P ξ) q.D.V (x ξ) s.1 s.2).seen
  calls_le : ∀ ξ ω r, runCalls (Claims.run FlockDeploy.sha512 (P ξ) q.D.V (x ξ) ω r) ≤ V.cost θ F c σ

attribute [instance] SessionIn.finΞ SessionIn.neΞ SessionIn.finΩ SessionIn.neΩ

namespace Dep

variable (q : Dep) (V : Vocab d α) {θ F c : ByteArray} {σ : Game.Strategy (V.game θ F c)}

/-- **A run of the session with the extractor's seed** at `ξ`: the tape, the session's coins and the seed. -/
abbrev Sample (S : SessionIn V q θ F c σ) (ξ : S.Ξ) : Type :=
  S.Ω × (Fin (q.D.V.k (S.x ξ)) → Coin) × q.E.C (S.x ξ)

/-- **The weight of bit `i` of gate `g` reading `bit`** at the session `S`: the probability, over the draw and a run with
the extractor's seed, that the run accepts, `g` is a wire of a unit `ξ` draws, and the extractor returns a witness in
the recursive relation whose reading (`readX`) has `bit` there. -/
noncomputable def weight (S : SessionIn V q θ F c σ) (readX : Msg → Claims.Flock.Wit → (V.circuitOf F).Transcript)
    (g : Fin (V.circuitOf F).N) (i : ℕ) (bit : Bool) : ℝ :=
  Claims.Ex fun ξ : S.Ξ => Claims.Pr fun s : q.Sample V S ξ =>
    (Claims.run FlockDeploy.sha512 (S.P ξ) q.D.V (S.x ξ) s.1 s.2.1).out = .acc ∧
      (∃ u ∈ S.drawOf ξ, g ∈ (V.partsOf F).io u) ∧
      ∃ wt, (q.E.eval FlockDeploy.sha512 (S.P ξ) (S.x ξ) s.1 s.2.1 s.2.2).1 = some wt ∧ q.R (S.x ξ) wt ∧
        (readX (S.x ξ) wt g).getLsbD i = bit

/-- **The recursive configuration's extracted transcript** at the session `S` (R4′, `XplurZC`'s plurality at the
recursive session): each bit of each gate is the value the extractor's witnesses read there most often, over the
draws and the accepted runs that draw the gate's unit (`false` on a tie). It is one transcript per strategy, a
construction applied to `S`, never chosen per strategy. -/
noncomputable def extract (S : SessionIn V q θ F c σ) (readX : Msg → Claims.Flock.Wit → (V.circuitOf F).Transcript) :
    (V.circuitOf F).Transcript := fun g =>
  BitVec.ofNat ((V.circuitOf F).w g) (∑ i ∈ Finset.range ((V.circuitOf F).w g),
    if q.weight V S readX g i false < q.weight V S readX g i true then 2 ^ i else 0)

/-- **The binding finder at the draws `ξ₁` and `ξ₂`**: it plays the session twice, at `ξ₁` and at `ξ₂`, with
independent tapes, coins and seeds, runs the extractor on both, and outputs `collOf`'s pair from the two witnesses.
Its cost is the two runs' SHA-512 calls, the two extractions' and `collOf`'s `dColl`. -/
noncomputable def bindAt (Se : SessionIn V q θ F c σ)
    (collOf : Msg → Claims.Flock.Wit → Msg → Claims.Flock.Wit → Option (Msg × Msg)) (dColl : ℕ) (ξ₁ ξ₂ : Se.Ξ) :
    Claims.Finder where
  S := q.Sample V Se ξ₁ × q.Sample V Se ξ₂
  out s :=
    match (q.E.eval FlockDeploy.sha512 (Se.P ξ₁) (Se.x ξ₁) s.1.1 s.1.2.1 s.1.2.2).1,
      (q.E.eval FlockDeploy.sha512 (Se.P ξ₂) (Se.x ξ₂) s.2.1 s.2.2.1 s.2.2.2).1 with
    | some w₁, some w₂ => collOf (Se.x ξ₁) w₁ (Se.x ξ₂) w₂
    | _, _ => none
  cost s :=
    runCalls (Claims.run FlockDeploy.sha512 (Se.P ξ₁) q.D.V (Se.x ξ₁) s.1.1 s.1.2.1) +
      runCalls (Claims.run FlockDeploy.sha512 (Se.P ξ₂) q.D.V (Se.x ξ₂) s.2.1 s.2.2.1) +
      (q.E.eval FlockDeploy.sha512 (Se.P ξ₁) (Se.x ξ₁) s.1.1 s.1.2.1 s.1.2.2).2.1 +
      (q.E.eval FlockDeploy.sha512 (Se.P ξ₁) (Se.x ξ₁) s.1.1 s.1.2.1 s.1.2.2).2.2 +
      (q.E.eval FlockDeploy.sha512 (Se.P ξ₂) (Se.x ξ₂) s.2.1 s.2.2.1 s.2.2.2).2.1 +
      (q.E.eval FlockDeploy.sha512 (Se.P ξ₂) (Se.x ξ₂) s.2.1 s.2.2.1 s.2.2.2).2.2 + dColl

/-- **C-Flock's SHA-512 premises at one session's prover**, asked only of the finders `BitsLin2On` builds from it, at
the inner and setup call bounds `n` and `v`: `StrictBirthday` at `B`, and `ExpectedCR` at the inner, device,
forked-coin and first-coin finders at their charged costs. -/
def SessionCR (n v : ℕ) (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω) : Prop :=
  (∀ hk : 0 < q.D.V.k x, Claims.StrictBirthday FlockDeploy.sha512 (Claims.B FlockDeploy.sha512 q.D.V P x hk)) ∧
    (∀ hk : 0 < q.I.V.k (Claims.Recursion.xI q.D x),
      Claims.ExpectedCR FlockDeploy.sha512 ((Claims.Recursion.BinnerV FlockDeploy.sha512 q.D q.I
        (deployZkRec _root_.Flock.Readable.relation) q.decO P x hk).charge
          (Claims.Recursion.setupLen q.D x + 2 * (n + v)))) ∧
    (∀ (M : ℕ) (hk : 0 < q.I.V.k (Claims.Recursion.xI q.D x)),
      Claims.ExpectedCR FlockDeploy.sha512 ((Claims.Recursion.FdevV FlockDeploy.sha512 q.D q.I
        (deployZkRec _root_.Flock.Readable.relation) q.decO P x M hk).charge
          (Claims.Recursion.setupLen q.D x + 2 * (n + v)))) ∧
    (∀ (M : ℕ) (hW : 0 < FlockDeploy.KS.CpDep.width (Claims.Recursion.xI q.D x)),
      Claims.ExpectedCR FlockDeploy.sha512 ((Claims.Recursion.FCI FlockDeploy.sha512 q.D q.I
        (deployZkRec _root_.Flock.Readable.relation) q.decO FlockDeploy.KS.CpDep P x M hW).charge
          (Claims.Recursion.setupLen q.D x + 2 * (n + v)))) ∧
    (∀ R : ℕ, Claims.ExpectedCR FlockDeploy.sha512 ((Claims.Recursion.F0I FlockDeploy.sha512 q.D q.I
        (deployZkRec _root_.Flock.Readable.relation) q.decO P x R).charge
          (Claims.Recursion.setupLen q.D x + (q.I.k (Claims.Recursion.xI q.D x) + 1) * (n + v))))

/-- **The strategy's cost parameter `t'`**: its SHA-512 calls in one run (`Vocab.cost`, ONE8's `t' τ`), at least one.
Every finder budget below is a defined function of it and of public counts. -/
def tCost (_ : SessionIn V q θ F c σ) : ℕ := max 1 (V.cost θ F c σ)

/-- **The binding finder's budget at the draws `ξ₁`, `ξ₂`**, a defined function of the strategy's cost `t'`: two runs,
and two extractions of at most `runs` reruns each (every rerun a run of the session, so at most `t'` calls), plus each
extraction's rerun count and `collOf`'s `dColl`: `(2 + r₁ + r₂) · t' + r₁ + r₂ + dColl`. -/
noncomputable def bindBudget (S : SessionIn V q θ F c σ) (dColl : ℕ) (ξ₁ ξ₂ : S.Ξ) : ℕ :=
  (2 + q.runs (S.x ξ₁) + q.runs (S.x ξ₂)) * q.tCost V S + q.runs (S.x ξ₁) + q.runs (S.x ξ₂) + dColl

/-- **`FinderCR`: the named SHA-512 premises at the strategy, asked only of the finders built from it.** Each session
prover's finders meet C-Flock's premises (`SessionCR`, at the inner and setup call bounds `nI` and `vS`; `B`'s runs are
the session's, at most `t'` calls each). Each binding finder (`bindAt`) outputs a SHA-512 collision with probability at
most its budget over `2^256`: `ecr/sha-512` (`Claims.ExpectedCR`) asked at `bindBudget`, a defined function of the
strategy's cost parameter `t' = tCost` (`Vocab.cost`), as ONE8's `qF`/`qS` are of `t' τ`. Quantitative, so `⊤` is the
error only where a finder built from the strategy beats its budget. -/
def FinderCR (S : SessionIn V q θ F c σ)
    (collOf : Msg → Claims.Flock.Wit → Msg → Claims.Flock.Wit → Option (Msg × Msg)) (dColl : ℕ)
    (nI vS : Msg → ℕ) : Prop :=
  (∀ ξ, q.SessionCR (nI (S.x ξ)) (vS (S.x ξ)) (S.x ξ) S.Ω (S.P ξ)) ∧
    ∀ ξ₁ ξ₂, (q.bindAt V S collOf dColl ξ₁ ξ₂).Pr FlockDeploy.sha512 ≤
      (q.bindBudget V S dColl ξ₁ ξ₂ : ℝ) / 2 ^ (256 : ℝ)

/-- **The error's number at the strategy**, from `t'` and public parameters: the draw's miss `δs`, plus `BitsLin2On`'s
`t' · 2^{-λ}`, plus the binding term, the file's wire bits times `√(E[bindBudget] / 2^256)` over pairs of draws. -/
noncomputable def delta (S : SessionIn V q θ F c σ) (dColl : ℕ) (δs lam : ℝ) : ℝ≥0∞ :=
  ENNReal.ofReal (δs + q.tCost V S * (2 : ℝ) ^ (-lam) +
    (∑ g, (V.circuitOf F).w g : ℕ) *
      Real.sqrt (Claims.Ex (fun p : S.Ξ × S.Ξ => (q.bindBudget V S dColl p.1 p.2 : ℝ)) / 2 ^ (256 : ℝ)))

open Classical in
/-- **The error at the strategy `τ`**, ONE8's `condError FinderCR δ τ := if FinderCR τ then δ τ else ⊤` (the shape of
`Vocab.error`): `delta` where the finders built from `τ` meet `FinderCR`, and `⊤` elsewhere. -/
noncomputable def err (Sof : (τ : Game.Strategy (V.game θ F c)) → SessionIn V q θ F c τ)
    (collOf : Msg → Claims.Flock.Wit → Msg → Claims.Flock.Wit → Option (Msg × Msg)) (dColl : ℕ)
    (nI vS : Msg → ℕ) (δs lam : ℝ) (τ : Game.Strategy (V.game θ F c)) : ℝ≥0∞ :=
  if q.FinderCR V (Sof τ) collOf dColl nI vS then q.delta V (Sof τ) dColl δs lam else ⊤

end Dep

/-- **ONE-9p's target: sampled proofs at the recursive configuration**, in Daniel's per-prover form (26 Oct) over the
causal game (R4′, R5, R6), with the recursive configuration's verdict (`RecVerdict`). At all parameters `θ`, circuit
files `F` and commitments `c`, and for every strategy `σ`, the probability that the recursive component accepts while
sampled proofs' claim fails on the extractor's transcript (`¬ Holds .sampled`) is at most `Dep.err` at `σ`, ONE8's
`condError FinderCR δ σ`: where `FinderCR` holds at the finders built from `σ`,
`δs + t'(σ) · 2^{-λ} + bits · √(E[bindBudget(σ)] / 2^256)`, and `⊤` elsewhere, with `t'(σ)` the strategy's SHA-512
calls in one run (`Vocab.cost`) and every finder's budget defined from it. There is no hypothesis on the strategy but
`live-verifier`.

It takes these hypotheses:
* `RecursiveKSAt` at the deployed brings (`hKS`), and the honest half (`HonestHalf`, HONESTIV);
* the decoders meet the list-decoding spec (`hdecI`, `hdecO`);
* the inner and setup call bounds `nI`, `vS` and the level `λ` (`hn`, `hv`, `hlam`), at the dense statements;
* `K₀` covers the claim's `ε` share (`hK₀`);
* `Sof` is the coupling at each strategy (ONE-4's projection, `SessionIn`), and its draw misses any `K₀` units with
  probability at most `δs` (`hdraw`);
* the vocabulary's extractor is the recursive configuration's (`hX`, `Dep.extract` at `Sof`);
* the deployment's reading `readX` of a witness in the recursive relation is correct at, and opened by the commitment
  on, every unit the statement draws (`hread`); and `collOf`, from two such witnesses that read a wire of units drawn
  in both differently, outputs a SHA-512 collision (`hcoll`);
* the records hold the run's live coins (`LiveCoins`, `live-verifier`). `live-verifier` is admissible until ONE-4
  (PRF-200) lands; then the component reads the coins through ONE-4's signed device log and this hypothesis goes. -/
theorem sampledProofsRecursive (q : Dep)
    (hKS : Flock.Guarantees.RecursiveKSAt DeployedR.rdR DeployedR.finalR OpenRows.elemR q.wE q.w (deployedB q.b)
      q.aux q.nOf q.hd q.recRest)
    (hHon : HonestHalf q.wE q.w q.b q.aux q.nOf q.hd q.recRest)
    (hdecI : Claims.Recursion.decOK q.Bd.deploy q.decI)
    (hdecO : Claims.Recursion.decOK (deployZkRec _root_.Flock.Readable.relation) q.decO)
    (nI vS : Msg → ℕ) (lam : ℝ)
    (hn : ∀ x, RecDeploy.DenseAt x → ∀ cs ws, (q.D.innerCalls FlockDeploy.sha512 x cs ws).length ≤ nI x)
    (hv : ∀ x, RecDeploy.DenseAt x → ∀ ps,
      (Claims.Recursion.setupCalls FlockDeploy.sha512 q.D (q.D.vstar x ps)).length ≤ vS x)
    (hlam : ∀ x, RecDeploy.DenseAt x → lam ≤ Claims.Recursion.lamLin2 q.D q.I FlockDeploy.KS.CpDep x (nI x) (vS x))
    (V : Vocab d α)
    (K₀ : ByteArray → ByteArray → ℕ) (δs : ℝ)
    (hK₀ : ∀ θ F (m : ℕ), (V.certOf θ).ε * V.units F < m → K₀ θ F ≤ m)
    (Sof : ∀ θ F c (σ : Game.Strategy (V.game θ F c)), SessionIn V q θ F c σ)
    (hdraw : ∀ θ F c σ (U : Finset (Fin (V.units F))), K₀ θ F ≤ U.card →
      Claims.Pr (fun ξ : (Sof θ F c σ).Ξ => Disjoint ((Sof θ F c σ).drawOf ξ) U) ≤ δs)
    (readX : (F : ByteArray) → Msg → Claims.Flock.Wit → (V.circuitOf F).Transcript)
    (hX : ∀ θ F c σ, V.extract θ F c σ = q.extract V (Sof θ F c σ) (readX F))
    (hread : ∀ θ F c σ ξ wt, V.Registered θ F → q.R ((Sof θ F c σ).x ξ) wt → ∀ u ∈ (Sof θ F c σ).drawOf ξ,
      (V.partsOf F).Correct (readX F ((Sof θ F c σ).x ξ) wt) u ∧
        ∀ g ∈ (V.partsOf F).io u, V.Opens θ F c g (readX F ((Sof θ F c σ).x ξ) wt g))
    (collOf : ByteArray → Msg → Claims.Flock.Wit → Msg → Claims.Flock.Wit → Option (Msg × Msg)) (dColl : ℕ)
    (hcoll : ∀ θ F c σ ξ₁ ξ₂ w₁ w₂ (g : Fin (V.circuitOf F).N),
      q.R ((Sof θ F c σ).x ξ₁) w₁ → q.R ((Sof θ F c σ).x ξ₂) w₂ →
      (∃ u ∈ (Sof θ F c σ).drawOf ξ₁, g ∈ (V.partsOf F).io u) →
      (∃ u ∈ (Sof θ F c σ).drawOf ξ₂, g ∈ (V.partsOf F).io u) →
      readX F ((Sof θ F c σ).x ξ₁) w₁ g ≠ readX F ((Sof θ F c σ).x ξ₂) w₂ g →
      ∃ m m', collOf F ((Sof θ F c σ).x ξ₁) w₁ ((Sof θ F c σ).x ξ₂) w₂ = some (m, m') ∧
        Claims.Collision FlockDeploy.sha512 m m')
    (hlive : ∀ θ F c σ ξ ω r, LiveCoins q.Bd.IVStar q.st ((Sof θ F c σ).x ξ)
      (Claims.run FlockDeploy.sha512 ((Sof θ F c σ).P ξ) q.D.V ((Sof θ F c σ).x ξ) ω r).seen) :
    ∀ (θ F c : ByteArray) (σ : Game.Strategy (V.game θ F c)),
      Game.prob (fun o => RecVerdict V o ∧ ¬ V.Holds θ F c σ .sampled o) (V.game θ F c) σ ≤
        q.err V (Sof θ F c) (collOf F) dColl nI vS δs lam σ := by
  sorry

/-- **Its conclusion fails for a verifier that accepts everything**: at a strategy whose extracted transcript fails
the claim, where the finders built from it meet `FinderCR` and the error's number (`Dep.delta`) is below one. -/
theorem sampledProofsRecursive_nontrivial (q : Dep) (V : Vocab d α) (θ F c : ByteArray)
    (Sof : (τ : Game.Strategy (V.game θ F c)) → SessionIn V q θ F c τ) (σ : Game.Strategy (V.game θ F c))
    (collOf : Msg → Claims.Flock.Wit → Msg → Claims.Flock.Wit → Option (Msg × Msg)) (dColl : ℕ)
    (nI vS : Msg → ℕ) (δs lam : ℝ)
    (hbad : ¬ V.Holds θ F c σ .sampled ⟨θ, F, c, [], ByteArray.empty⟩)
    (hcr : q.FinderCR V (Sof σ) collOf dColl nI vS) (hlt : q.delta V (Sof σ) dColl δs lam < 1) :
    ¬ Game.prob (fun o => True ∧ ¬ V.Holds θ F c σ .sampled o) (V.game θ F c) σ ≤
      q.err V Sof collOf dColl nI vS δs lam σ := by
  intro h
  have h1 : Game.prob (fun o => True ∧ ¬ V.Holds θ F c σ .sampled o) (V.game θ F c) σ = 1 :=
    prob_play_eq_one (V.rounds θ) θ F c (fun o => True ∧ ¬ V.Holds θ F c σ .sampled o)
      (fun _ _ => ⟨trivial, hbad⟩) σ
  rw [Dep.err, if_pos hcr] at h
  exact lt_irrefl (1 : ℝ≥0∞) ((h1.symm.trans_le h).trans_lt hlt)

/-- **Its hypotheses hold at the honest deployment where they are choices of the deployment**: `K₀ = ⌊ε n⌋ + 1` covers
the claim's `ε` share (`hK₀`) wherever `ε n ≥ 0`; and where the extracted transcript is correct on every unit and the
commitment opens it (the honest commitment's case), the target's event is empty, so its conclusion holds at every
bound. The other hypotheses are code facts and the coupling, each its owner's: `RecursiveKS` (proofs), `HonestHalf`
(HONESTIV), `hread` and `hcoll` (PRF-202's bridge with ONE-7's `place`), `Sof` and `hdraw` (ONE-4). -/
theorem sampledProofsRecursive_honest (V : Vocab d α) :
    (∀ θ F, 0 ≤ (V.certOf θ).ε * V.units F → ∀ m : ℕ, (V.certOf θ).ε * V.units F < m →
      ⌊(V.certOf θ).ε * V.units F⌋₊ + 1 ≤ m) ∧
    ∀ θ F c (σ : Game.Strategy (V.game θ F c)), (∀ u, (V.partsOf F).Correct (V.extract θ F c σ) u) →
      (∀ g, V.Opens θ F c g (V.extract θ F c σ g)) → 0 ≤ (V.certOf θ).ε → ∀ b : ℝ≥0∞,
      Game.prob (fun o => RecVerdict V o ∧ ¬ V.Holds θ F c σ .sampled o) (V.game θ F c) σ ≤ b := by
  refine ⟨fun θ F hε m hm => Nat.succ_le_of_lt ((Nat.floor_lt hε).2 hm), ?_⟩
  intro θ F c σ hc ho hε b
  refine (Game.prob_mono (p := fun o => RecVerdict V o ∧ ¬ V.Holds θ F c σ .sampled o) (q := fun _ => False)
    (fun o ho' => ho'.2 (V.holds_sampled_of_opens θ F c σ o hc ho hε))
    _ σ).trans ?_
  exact (Game.prob_false _ σ).le.trans (zero_le _)

end Verity.Recursive
