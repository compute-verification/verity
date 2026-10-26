import Proofs.Flock.Recursive.KSStatement
import Proofs.Flock.Claims.Recursion.DeployedBrings
import Proofs.Flock.Claims.Recursion.OpenRowsDeployed
import Proofs.Flock.Claims.Recursion.RecRecord
import Proofs.Flock.Bridge.CircuitOf
import VStarRecursive

/-!
# The recursive integrity component: its property and its leg (ONE-9p)

The recursive form of integrity (`VStar.Recursive`, `verity/sampled-proofs/recursive/v1`, which
`VStar.Recursive.Integrity` runs when the auditor's parameters name `"form": "recursive"`) checks, on the prover's blobs,
the recursive session that `Flock.Guarantees.RecursiveKSAt` is stated about. This module states two things.

* **The property** (`acc_of_accepts`). Take a run of the deployed recursive verifier that `RecursiveKSAt` reads
  (`recordAt`), and blobs that show that run (`Shows`). If the compiled component accepts those blobs and the records it
  reads hold the run's live coins (`LiveCoins`), then the statement is dense and the run accepts. That is the event
  `RecursiveKSAt`'s `PrBad` counts.
  - Each V* part's record is a prover-sent blob. The model verifier checks a record's coins against its own live coins
    (`FlockDeploy.custody`), and the compiled component, which holds no live coin, can't. So `LiveCoins` is
    `live-verifier`'s content at the recursive session (`Assumptions.Zk.RecordCustodyZKJ`).
* **The leg** (`integrityLeg`), in soundness's decision-1 form:
  - commit-then-coin: the parameters and the commitment are constants of the play;
  - `∃ X`, fixed by the commitment, on ONE-7's `circuitOf` of the inner circuit;
  - the device's coins with `Break`;
  - a public error: the draw's miss term plus `t · 2^{-λ}`.

  Its play is causal in the component's rounds (`Play`): after the device's draw, an interactive prover runs against
  the session's live coins, and the evidence is what it sends after the run. A play that read every coin before
  sending its first message would forge a Flock session.

  It takes as hypotheses `RecursiveKSAt` at the deployed brings, the honest half (the honest run is accepted at the
  deployed statements, `HonestHalf`), and the named SHA-512 premises at the play's provers (`CRAt`).

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
* and, for each part `s`, the record and the rep files of session `s`'s last message, each rep file under the name its
  stream gives, in the stream order of that session's recursive setup (`setupO`). -/
def Shows (IV : Claims.Verifier Coin) (st : Stage) (x : Msg) (pr : VStar.Recursive.Proofs) (bs : Core.Blobs)
    (v : Claims.Seen) : Prop :=
  bs.get pr.transcript "rec inner transcript" = .ok (st.transcript x (v.msgs.take (kIS IV x))) ∧
  bs.get pr.record "the inner session's record" = .ok (st.record x (v.msgs.take (kIS IV x))) ∧
  ∀ s q, pr.parts[s]? = some q →
    bs.get q.pub "a part's public file" = .ok (st.pub x (v.msgs.take (kIS IV x)) s) ∧
    ∃ f, finalAt IV st x s v = some f ∧ bs.get q.record "a part's record" = .ok f.record.toUTF8 ∧
      ∀ i spec sts, setupO (vstarAt st x (v.msgs.take (kIS IV x)) s) = some (i, spec, sts) →
        ∀ k (h : k < spec.repStreams.size), ∃ d,
          (q.reps.find? (·.1 == repName spec.repStreams[k].2.2)).map (·.2) = some d ∧
          ∃ hk : k < f.proofs.size, bs.get d "session files" = .ok f.proofs[k]

/-- **The records hold the live coins** (`live-verifier` at the recursive session, `Assumptions.Zk.RecordCustodyZKJ`):
at each V\* session `s` of the run's view `v`, wherever the outer verifier decides on session `s`'s messages and coins
(`actO`), the live session the record shows is the session's link message, first coin and round messages with their
coins (`FlockDeploy.custody`). The model verifier checks this itself. The compiled component can't, since the coins are
the coin server's. -/
noncomputable def LiveCoins (IV : Claims.Verifier Coin) (st : Stage) (x : Msg) (v : Claims.Seen) : Prop :=
  ∀ s < nS x, ∀ i spec sts link c₁ mids f,
    actO (vstarAt st x (v.msgs.take (kIS IV x)) s) ((session IV st x s v).msgs.map Prod.fst)
        ((session IV st x s v).msgs.map fun m => outCoin m.2) =
      .decide i spec sts link c₁ mids f →
    (do FlockDeploy.custody spec (← FlockDeploy.liveOf spec f.record) link c₁ mids : Flock.V Unit) = .ok ()

/-- **ONE-9p's property: the recursive component's acceptance is the event `RecursiveKSAt` is stated over.** Suppose
the compiled recursive component accepts `(p, c, e, bs)`, the blobs show a run of the deployed recursive verifier
(`recordAt`) at `x` (`Shows`), and that run's records hold its live coins (`LiveCoins`). Then `x` is dense
(`RecDeploy.DenseAt`, the statements `RecursiveKSAt` covers), and the run accepts. -/
theorem acc_of_accepts (wE w : Msg → ℝ) (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (p : VStar.Recursive.Params) (c : SampledProofs.OneStage.Commitment) (e : VStar.Recursive.Evidence)
    (bs : Core.Blobs) {pub : SampledProofs.OneStage.Public} {prof : Core.Integrity}
    (hacc : VStar.Recursive.checkBlobs p c e bs = .accept pub prof)
    (x : Msg) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω)
    (r : Fin ((recordAt wE w b aux nOf hd recRest).V.k x) → Coin)
    (hshow : Shows (boundAt wE w).IVStar (stageAt aux nOf hd recRest) x e.proofs bs
      (Claims.run FlockDeploy.sha512 P (recordAt wE w b aux nOf hd recRest).V x ω r).seen)
    (hlive : LiveCoins (boundAt wE w).IVStar (stageAt aux nOf hd recRest) x
      (Claims.run FlockDeploy.sha512 P (recordAt wE w b aux nOf hd recRest).V x ω r).seen) :
    RecDeploy.DenseAt x ∧
      (Claims.run FlockDeploy.sha512 P (recordAt wE w b aux nOf hd recRest).V x ω r).out = .acc := by
  sorry

/-! ## The leg -/

/-- **A strategy at the recursive component, causal in its rounds.** The parameters and the commitment are sent before
any coin, so both are constants of the play. Then the challenger device draws `ξ`. At `ξ` the prover opens the
recursive session at the statement `stmt ξ` and runs `P ξ` against the session's live coins. The evidence and blobs it
sends are `send ξ` of that run. -/
structure Play (Ξ Ω : Type) where
  params : ByteArray
  commitment : ByteArray
  stmt : Ξ → Msg
  P : Ξ → Claims.Prover Ω
  send : Ξ → Claims.Trace → ByteArray × Array ByteArray

variable {Ξ Ω : Type}

/-- What the play sends at the draw `ξ`, after the run `t`: the evidence and the blobs. -/
def Play.sent (τ : Play Ξ Ω) (ξ : Ξ) (t : Claims.Trace) : ByteArray × Array ByteArray := τ.send ξ t

/-- The integrity component's verdict on what the play sent after the run `t` at the draw `ξ`. -/
def Play.answer (τ : Play Ξ Ω) (ξ : Ξ) (t : Claims.Trace) : Core.Answer :=
  VStar.Recursive.Integrity.protocol.answer τ.params τ.commitment (τ.sent ξ t).1 (τ.sent ξ t).2

/-- **The component accepts on the device's coins.** It accepts what the play sent, and the coins it read
(`coinsRead`, ONE-4's decoding of the device's signed log entries) are the device's `ξ`. -/
def Accepts (coinsRead : ByteArray → ByteArray → ByteArray → Array ByteArray → Option Ξ) (τ : Play Ξ Ω) (ξ : Ξ)
    (t : Claims.Trace) : Prop :=
  (τ.answer ξ t).verdict = .accept ∧ coinsRead τ.params τ.commitment (τ.sent ξ t).1 (τ.sent ξ t).2 = some ξ

/-- **The break.** The component accepts on coins other than those the device logged, which is a signature under the
device's key on a message missing from its log. -/
def Break (coinsRead : ByteArray → ByteArray → ByteArray → Array ByteArray → Option Ξ) (τ : Play Ξ Ω) (ξ : Ξ)
    (t : Claims.Trace) : Prop :=
  (τ.answer ξ t).verdict = .accept ∧ coinsRead τ.params τ.commitment (τ.sent ξ t).1 (τ.sent ξ t).2 ≠ some ξ

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

/-- **What a play sends shows its run** (`Shows`), read as the integrity component reads it: the evidence decoded at
the recursive form, and the blobs. -/
def ShowsSent (IV : Claims.Verifier Coin) (st : Stage) (x : Msg) (sent : ByteArray × Array ByteArray)
    (v : Claims.Seen) : Prop :=
  ∀ e, VStar.Recursive.Integrity.evidenceOf sent.1 = .ok (.inr e) → Shows IV st x e.proofs (Core.Blobs.ofArray sent.2) v

/-- **The parameters name the inner circuit `D`**: they decode to the recursive form, and the inner circuit file they
pin by SHA-512 reads as `D`. -/
def NamesInner (θ : ByteArray) (D : Flock.Desc) : Prop :=
  ∃ (j : Lean.Json) (p : VStar.Recursive.Params) (bytes : ByteArray) (tags : Flock.Tags),
    Core.canonicalJson θ = .ok j ∧ VStar.Recursive.Integrity.paramsOfJson j = .ok (.inr p) ∧
    Flock.hexOf (Flock.Sha512.hash bytes) = p.own.circuit ∧ Flock.Desc.parse tags bytes = .ok D

/-- **ONE-9p's leg: integrity at the recursive component, in decision 1's form.** Take a play whose parameters name
the inner circuit `D` with `n` units. Suppose:
* the device's draw misses any `K₀` units with probability at most `δs` (`hDraw`);
* every play's prover meets the named SHA-512 premises at `t` calls and level `λ` (`hCR`);
* what it sends shows its run (`hShow`), and the records hold the run's live coins (`hLive`); both are `live-verifier`:
  the session's files are the coin server's records of the live session.

Then there is a transcript `X`, fixed by the commitment, on `circuitOf D hU n`. The probability, over the draw, the
prover's tape and the session's live coins, that the component accepts on the device's coins while more than `ε · n`
units are wrong in `X`, with no break, is at most `δs + t · 2^{-λ}`.

It takes `RecursiveKSAt` at the deployed brings and the honest half as hypotheses. -/
theorem integrityLeg (wE w : Msg → ℝ) (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (hKS : Flock.Guarantees.RecursiveKSAt DeployedR.rdR DeployedR.finalR OpenRows.elemR wE w (deployedB b) aux nOf
      hd recRest)
    (hHon : HonestHalf wE w b aux nOf hd recRest)
    [Fintype Ξ] [Nonempty Ξ] [Fintype Ω] [Nonempty Ω]
    (coinsRead : ByteArray → ByteArray → ByteArray → Array ByteArray → Option Ξ)
    (D : Flock.Desc) (hU : FlockSoundness.Built D.unroll.unit) (n K₀ : ℕ) (ε δs lam : ℝ) (t : ℕ)
    (drawOf : Ξ → Finset (Fin n))
    (hK₀ : ∀ m : ℕ, ε * n < m → K₀ ≤ m)
    (hDraw : ∀ S : Finset (Fin n), K₀ ≤ S.card → Claims.Pr (fun ξ : Ξ => Disjoint (drawOf ξ) S) ≤ δs)
    (τ : Play Ξ Ω) (hθ : NamesInner τ.params D)
    (hCR : ∀ ξ, CRAt wE w b aux nOf hd recRest t lam (τ.stmt ξ) Ω (τ.P ξ))
    (hShow : ∀ ξ ω r, ShowsSent (boundAt wE w).IVStar (stageAt aux nOf hd recRest) (τ.stmt ξ)
      (τ.sent ξ (Claims.run FlockDeploy.sha512 (τ.P ξ) (recordAt wE w b aux nOf hd recRest).V (τ.stmt ξ) ω r))
      (Claims.run FlockDeploy.sha512 (τ.P ξ) (recordAt wE w b aux nOf hd recRest).V (τ.stmt ξ) ω r).seen)
    (hLive : ∀ ξ ω r, LiveCoins (boundAt wE w).IVStar (stageAt aux nOf hd recRest) (τ.stmt ξ)
      (Claims.run FlockDeploy.sha512 (τ.P ξ) (recordAt wE w b aux nOf hd recRest).V (τ.stmt ξ) ω r).seen) :
    ∃ X : ByteArray → (Flock.Bridge.circuitOf D hU n).Transcript,
      Claims.Ex (fun ξ : Ξ =>
        Claims.Pr (fun s : Ω × (Fin ((recordAt wE w b aux nOf hd recRest).V.k (τ.stmt ξ)) → Coin) =>
          Accepts coinsRead τ ξ
              (Claims.run FlockDeploy.sha512 (τ.P ξ) (recordAt wE w b aux nOf hd recRest).V (τ.stmt ξ) s.1 s.2) ∧
            ε * n < (((Flock.Bridge.partsOf D hU n).wrong (X τ.commitment)).card : ℝ) ∧
            ¬ Break coinsRead τ ξ
              (Claims.run FlockDeploy.sha512 (τ.P ξ) (recordAt wE w b aux nOf hd recRest).V (τ.stmt ξ) s.1 s.2)))
        ≤ δs + t * (2 : ℝ) ^ (-lam) := by
  sorry

end Verity.Recursive
