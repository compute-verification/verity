import Proofs.Flock.Recursive.KSStatement
import Proofs.Flock.Claims.Recursion.DeployedBrings
import Proofs.Flock.Claims.Recursion.OpenRowsDeployed
import Proofs.Flock.Claims.Recursion.RecRecord
import Proofs.Flock.Bridge.CircuitOf
import VStarRecursive
import Properties.Verity.Game
import Proofs.Properties.Verity.Soundness
import Proofs.Properties.Verity.Recursive.BaseSetup
import Proofs.Properties.Verity.Recursive.Compiled
import Proofs.Properties.Verity.Recursive.Numbers
import Proofs.Properties.Verity.Recursive.Names
import Proofs.Properties.Verity.Recursive.PartStatement
import Proofs.Properties.Verity.Recursive.PartVerdict
import Proofs.Properties.Verity.Recursive.SetupChecks
import Proofs.Properties.Verity.Recursive.ScheduleRun
import Proofs.Properties.Verity.Recursive.Refusals

/-!
# Sampled proofs' recursive configuration: the component's property and its target

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
  - `live-verifier` is admissible here only until the component reads the session's coins through the
    coin server's signed device log. Then `LiveCoins` follows from the log's signatures, and the premise goes.
* **The target** (`sampledProofsRecursive`, stated and proved in `Proofs.Properties.Verity.Recursive.Target`): sampled
  proofs at the recursive configuration, in `Vocab.Target .sampled`'s form (`Properties.Verity.Game`) with the
  recursive component's verdict (`RecVerdict`).
  - The game is causal (R5): the parameters, the registered circuit file and the commitment come first; then the
    device's releases, each after the bytes it answers; and the evidence last.
  - The claim reads one fixed extractor's transcript (R4′, `Vocab.extract`): `Dep.extract`, at each bit of each gate
    the value most often read off the witnesses `RecursiveKSAt`'s extractor returns, over the draws and the accepted
    runs that draw the gate's unit. The circuit is the committed file's (R6, `Vocab.Registered`).
  - The per-prover form: the bound holds at every strategy, with no hypothesis on it but `live-verifier`. It is
    the shared `condError FinderCR δ` (`Dep.err`, the shape of `Vocab.error`): where the finders built from the
    strategy meet their named SHA-512 premises (`Dep.FinderCR`: C-Flock's at each session prover, `ecr/sha-512` at each
    binding finder `Dep.bindAt` at its budget), the draw's miss plus `t' · 2^{-λ}` plus the binding term (`Dep.delta`);
    elsewhere `⊤`. Every budget is a defined function of the strategy's cost `t'` (`Vocab.cost`), with no call model.
  - Beside it, `sampledProofsRecursive_nontrivial` (an accept-everything verifier breaks the bound) and
    `sampledProofsRecursive_honest` (the hypotheses that are the deployment's choices hold at the honest one).

  It takes as hypotheses `RecursiveKSAt` at the deployed brings, the honest half (the honest run is accepted at the
  deployed statements, `HonestHalf`), the coupling at each strategy (`SessionIn`, the session's projection) with the
  draw's
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
* and, for each part `s`, session `s`'s link message, its first, which the record's live session is checked against
  (`FlockDeploy.linkOf`); and the record and the rep files of session `s`'s last message, whose statement is session
  `s`'s V\* statement (`vstarAt`), each rep file under the name its stream gives, in the stream order of that session's
  recursive setup (`setupO`). -/
def Shows (IV : Claims.Verifier Coin) (st : Stage) (x : Msg) (pr : VStar.Recursive.Proofs) (bs : Core.Blobs)
    (v : Claims.Seen) : Prop :=
  bs.get pr.transcript "rec inner transcript" = .ok (st.transcript x (v.msgs.take (kIS IV x))) ∧
  bs.get pr.record "the inner session's record" = .ok (st.record x (v.msgs.take (kIS IV x))) ∧
  ∀ s q, pr.parts[s]? = some q →
    bs.get q.pub "a part's public file" = .ok (st.pub x (v.msgs.take (kIS IV x)) s) ∧
    (∃ m link, (session IV st x s v).msgs.head? = some m ∧ FlockDeploy.linkOf m.1 = some link) ∧
    ∃ f, finalAt IV st x s v = some f ∧ f.stmt = vstarAt st x (v.msgs.take (kIS IV x)) s ∧
      bs.get q.record "a part's record" = .ok f.record.toUTF8 ∧
      ∀ i spec sts, setupO (vstarAt st x (v.msgs.take (kIS IV x)) s) = some (i, spec, sts) →
        f.proofs.size = spec.repStreams.size ∧ ∀ k (h : k < spec.repStreams.size), ∃ d,
          (q.reps.find? (·.1 == repName spec.repStreams[k].2.2)).map (·.2) = some d ∧
          ∃ hk : k < f.proofs.size, bs.get d "session files" = .ok f.proofs[k]

/-- **The staged rows are the component's.** V\*'s shared rows (`--rec-rows`) are no part of `x`: the prover
stages one set for every part, off the staging message (`Stage.rows`). The compiled component reads them once, from
the evidence's `rows` blob (`innerHeld`), so that blob is the run's staged rows. -/
def RowsShow (IV : Claims.Verifier Coin) (st : Stage) (x : Msg) (pr : VStar.Recursive.Proofs) (bs : Core.Blobs)
    (v : Claims.Seen) : Prop :=
  bs.get pr.rows "the staged rows" = .ok (st.rows x (v.msgs.take (kIS IV x)))

/-- **The records hold the live coins** (`live-verifier` at the recursive session, `Assumptions.Zk.RecordCustodyZKJ`):
at each V\* session `s` of the run's view `v`, wherever the outer verifier decides on session `s`'s messages and coins
(`actO`), the live session the record shows is the session's link message, first coin and round messages with their
coins (`FlockDeploy.custody`). The model verifier checks this itself. The compiled component can't, since the coins are
the coin server's. It is admissible until the component reads the coins through the coin server's signed
device log; then this premise goes. -/
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

/-- **What the compiled component refuses, as the model states it**: Flock's own checks on the inner
statement (`InnerOkAt`, the compiled `innerOk`) and files too long for their length words (its `fits` at `2^56`):
`x` with each part's public file, the inner session's record, the coin server's transcript and V\*'s staged rows, at the inner rounds
`ms`. -/
structure Refusals (x : Msg) (st : Stage) (ms : List (Msg × Out)) : Prop where
  innerOk : InnerOkAt x
  pub : ∀ s < nS x, x.length + (st.pub x ms s).size < 2 ^ 64
  record : (st.record x ms).size < 2 ^ 64
  transcript : (st.transcript x ms).size < 2 ^ 64
  rows : (st.rows x ms).size < 2 ^ 64

theorem sparse_of_dense {i : Flock.RecVStar.Inner} (h : VStar.Recursive.dense i = true) : i.sparse = none := by
  cases hs : i.sparse with
  | none => rfl
  | some s => simp [VStar.Recursive.dense, hs] at h

/-- V\*'s list has a statement at every shape it lists: its last, the top cap. -/
theorem statements_pos {sh : Flock.RecVStar.Inner} {vs : Array VStar.Stmt}
    (hvs : Flock.RecVStar.statements sh = .ok vs) : 0 < vs.size := by
  obtain ⟨_, -, _, _, _, _, A, C, -, -, -, -, -, -, -, K, n, -, -, hss⟩ := FlockC3.Reps.list_ok hvs
  rw [hss]
  simp only [Array.size_append]
  first | omega | (simp; done) | (simp <;> omega)

/-- The verifier's registered reads, as the model's readers take them off the part's `registered` object. -/
theorem regC_of_part {pr : Option Lean.Json} {registered : Option (Array Flock.Registered.Port)} {complete : Bool}
    {known : Array (String × Array (Array ℕ × ByteArray))} {regParams : Array String}
    (h : PartRegistered pr registered complete known regParams) :
    (pr.mapM VStar.Recursive.registeredOf).map (Option.map (·.1)) = .ok registered ∧
      pr.bind (fun r => (VStar.Recursive.registeredOf r).toOption.map (·.1)) = registered := by
  cases pr with
  | none =>
    simp only [PartRegistered] at h
    obtain ⟨rfl, -, -, -⟩ := h
    exact ⟨rfl, rfl⟩
  | some r =>
    simp only [PartRegistered] at h
    obtain ⟨own, ho, rfl⟩ := h
    refine ⟨?_, ?_⟩
    · rw [Part.mapM_some_ok ho]
      try rfl
    · show (VStar.Recursive.registeredOf r).toOption.map (·.1) = some own
      rw [ho]
      try rfl

/-- **An accepted compiled part sets up as the model's part** (pieces C, D1 and D2 at one part). Where the compiled part
`j`'s verdict is `.ok ()`, `x`'s inner statement and part `j` are the component's (`InnerNames`, `PartNames`), the staged
files are the blobs' (`Shows`' clauses at `j`, and `RowsShow`): the model reads part `j`'s statement (`partRaw`) as
the compiled part's inputs (`partIns`), its recursive setup passes
with the compiled session, and so does its statement check. -/
theorem part_setup_of_verdict {o : VStar.Recursive.Own} {pr : VStar.Recursive.Proofs} {bs : Core.Blobs}
    {h : VStar.Recursive.InnerHeld} (hh : VStar.Recursive.innerHeld o pr bs = .ok h) (hsp : h.shape.sparse = none)
    {w : RecX} {ri : FlockDeploy.Raw} (hI : InnerNames o pr bs w ri)
    {st : Stage} {x : Msg} {ms : List (Msg × Out)} {j : ℕ} {q : PartX}
    {po : VStar.Recursive.PartOwn} {pp : VStar.Recursive.PartProof}
    (hpv : VStar.Recursive.partVerdict h j po pp bs = .ok (.ok ())) (hP : PartNames po pp bs j q)
    (hSp : bs.get pp.pub "a part's public file" = .ok (st.pub x ms j))
    (hSr : bs.get pr.record "the inner session's record" = .ok (st.record x ms))
    (hSt : bs.get pr.transcript "rec inner transcript" = .ok (st.transcript x ms))
    (hSrows : bs.get pr.rows "the staged rows" = .ok (st.rows x ms)) :
    ∃ (tags0 : Flock.Tags) (registered : Option (Array Flock.Registered.Port)) (bindings : VStar.Bindings)
      (recordB : ByteArray) (record : String) (d : Lean.Json) (spec : Flock.SessionSpec) (sts : Array Flock.Stmt),
      Flock.Tags.find? VStar.Recursive.STATEMENT = some tags0 ∧
      po.registered.bind (fun r => (VStar.Recursive.registeredOf r).toOption.map (·.1)) = registered ∧
      bs.get pp.record s!"part {j}'s record" = .ok recordB ∧ String.fromUTF8? recordB = some record ∧
      Flock.Witness.recordDraw record = some d ∧
      (partRaw (xrOf o h w ri) st x ms j q).ins =
        .ok (Part.partIns tags0 q.base.circuit (st.pub x ms j) po registered (some d) h j bindings (some h.rows)) ∧
      (Part.partIns tags0 q.base.circuit (st.pub x ms j) po registered (some d) h j bindings (some h.rows)).setup =
        .ok (spec, sts) ∧
      (Part.partIns tags0 q.base.circuit (st.pub x ms j) po registered (some d) h j bindings (some h.rows)).base.read.check =
        .ok () ∧
      SampledProofs.OneStageZk.buildSessionOf
        (Flock.ProvedScope.circuitOf (partTags tags0 q.base.circuit) q.base.circuit [])
        (partTags tags0 q.base.circuit) q.base.circuit (st.pub x ms j) (partCoins tags0) [] 1 (some d)
        (some po.partition) (some po.program) registered = .ok (spec, sts) := by
  obtain ⟨tags0, circuitB, pubB, registered, complete, known, regParams, bindings, recordB, record, d, spec, sts,
    hT, hplain, hcB, hpB, -, hPR, hbind, hrecB, hrec, hhm, hrd, -, hbuild, -, -, -, hc, hscope⟩ :=
    partVerdict_setup hpv rfl
  obtain rfl : q.base.circuit = circuitB := Except.ok.inj (hP.circuit.symm.trans hcB)
  obtain rfl : st.pub x ms j = pubB := Except.ok.inj ((get_what hSp).symm.trans hpB)
  obtain ⟨hregC, hregB⟩ := regC_of_part hPR
  have hhm' : (Flock.Zk.tags (tags0.withCoins (Flock.Coins.tree (Flock.Zk.coinSpec (tags0.tableNames 1))))
      (Flock.Zk.maskLogOf q.base.circuit) 1).hm96Rows = true := hhm
  have hdraw := hP.draw tags0 recordB record hT hrecB hrec
  rw [if_pos hhm', hrd] at hdraw
  have hins := Part.ins_partRaw (r := xrOf o h w ri) (st := st) (x := x) (ms := ms) (s := j) hh hT hplain
    hregC hbind hI.circuit hI.pub hI.statement hI.os hI.tables hI.partition hI.draw (hI.shape h hh)
    hI.registration hP.form rfl hP.partition hP.program hP.registered hP.bindings hdraw rfl hSr hSt hSrows
  obtain ⟨-, hbset, hregs, -, hfull⟩ := baseSetup_of_build (tags0 := tags0) (circuitB := q.base.circuit)
    (pubB := st.pub x ms j) (draw := some d) (partition := some po.partition) (program := some po.program)
    (registered := registered) hbuild
    (Part.partIns tags0 q.base.circuit (st.pub x ms j) po registered (some d) h j bindings (some h.rows)).base.read
    rfl rfl rfl rfl
  have hm : SetupMatch (Part.partIns tags0 q.base.circuit (st.pub x ms j) po registered (some d) h j bindings (some h.rows))
      h j tags0 (partTags tags0 q.base.circuit) q.base.circuit (st.pub x ms j) po.partition po.program registered
      bindings (some h.rows) (some d) :=
    { zk := rfl, stmtTags := rfl, tags := Part.partIns_tags _ _ _ _ _ _ _ _ _ _, circuitFile := rfl,
      publicFile := rfl, tables := rfl, count := rfl, partition := rfl, program := rfl, publicInputs := rfl,
      readRegistered := rfl, registered := rfl, draw := rfl, inner := rfl, src := rfl, commits := rfl, link := rfl,
      part := rfl, bindings := rfl, rows := rfl }
  exact ⟨tags0, registered, bindings, recordB, record, d, spec, sts, hT, hregB, hrecB, hrec, hrd, hins,
    recIns_setup_of_compiled hm hsp hhm hbset hregs hfull hc, read_check_of_scope hm hscope, hbuild⟩

/-- **The staged statement sets up** where it reads back as itself and the model's recursive setup passes on its
reading (D2's `setupO_staged_of_compiled`, at the setup's result). -/
theorem setupO_staged_of_setup {x : Msg} {r : XR} {st : Stage} {ms : List (Msg × Out)} {s : ℕ} {p : PartX}
    {i : RecIns} {spec : Flock.SessionSpec} {sts : Array Flock.Stmt}
    (hr : readX x = some r) (hp : r.w.parts[s]? = some p)
    (hdec : decodeRec (partRaw r st x ms s p).encode = some (partRaw r st x ms s p))
    (hi : (partRaw r st x ms s p).ins = .ok i) (hzk : i.base.read.zk = true) (hsu : i.setup = .ok (spec, sts)) :
    insO (staged st x ms s) = some i ∧ setupO (staged st x ms s) = some (i, spec, sts) := by
  have hz : (partRaw r st x ms s p).base.zk = true := by
    rw [← FlockDeploy.Raw.ins_zk (RecRaw.ins_base hi)]
    exact hzk
  have hstg : staged st x ms s = (partRaw r st x ms s p).encode := by
    unfold staged
    simp only [hr, hp]
    rw [if_pos hdec]
  have hins : insO (staged st x ms s) = some i := by
    rw [hstg]
    unfold insO
    simp [hdec, hz, hi, Except.toOption]
  refine ⟨hins, ?_⟩
  unfold setupO
  simp [hins, hsu, Except.toOption]

theorem actO_eq {xO : Msg} {i : RecIns} {spec : Flock.SessionSpec} {sts : Array Flock.Stmt} (as : List Msg)
    (cs : List Coin) (h : setupO xO = some (i, spec, sts)) : actO xO as cs = actAt i.base spec sts as cs := by
  unfold actO
  rw [h]
  try rfl

/-- `actAt` decides at a last message that reads as the prover's last and a first that reads as the link message. -/
theorem actAt_decide (s : FlockDeploy.Ins) (spec : Flock.SessionSpec) (sts : Array Flock.Stmt) (a₁ : Msg)
    (as : List Msg) (c₁ : Coin) (cs : List Coin) {a : Msg} {c : Coin} {f : FlockDeploy.Final}
    {link : List ByteArray × List ByteArray} (hl : (a₁ :: as).getLast? = some a)
    (hc : (c₁ :: cs).getLast? = some c) (hf : FlockDeploy.finalOf a = some f)
    (hk : FlockDeploy.linkOf a₁ = some link) :
    actAt s spec sts (a₁ :: as) (c₁ :: cs) =
      .decide s spec sts link c₁ ((as.dropLast.map List.tail).zip cs.dropLast) f := by
  have ha' : ∀ t, a ≠ 0 :: t := by
    intro t ht
    subst ht
    first
      | (simp [FlockDeploy.finalOf] at hf; done)
      | cases hf
      | (unfold FlockDeploy.finalOf at hf; split at hf <;> simp_all)
  unfold actAt
  rw [hl, hc]
  split
  · rename_i heq _
    exact absurd (Option.some.inj heq) (ha' _)
  · rename_i a₂ heq _
    obtain rfl := Option.some.inj heq
    simp only [hf, hk]
  · rename_i heq
    cases heq

/-- **The outer verifier decides on a session** whose statement sets up, whose last message reads as the prover's last
and whose first reads as the link message. -/
theorem actO_decide_of {xO : Msg} {i : RecIns} {spec : Flock.SessionSpec} {sts : Array Flock.Stmt}
    {L : List (Msg × Out)} {f : FlockDeploy.Final} {m : Msg × Out} {link : List ByteArray × List ByteArray}
    (hset : setupO xO = some (i, spec, sts))
    (hfin : L.getLast?.bind (fun m => FlockDeploy.finalOf m.1) = some f) (hhd : L.head? = some m)
    (hlk : FlockDeploy.linkOf m.1 = some link) :
    ∃ c₁ mids, actO xO (L.map Prod.fst) (L.map fun m => outCoin m.2) = .decide i.base spec sts link c₁ mids f := by
  rcases hml : L.getLast? with _ | ml
  · rw [hml] at hfin
    cases hfin
  rw [hml] at hfin
  have hfm : FlockDeploy.finalOf ml.1 = some f := hfin
  cases L with
  | nil => simp at hhd
  | cons m₀ rest =>
    simp only [List.head?_cons, Option.some.injEq] at hhd
    rw [← hhd] at hlk
    refine ⟨outCoin m₀.2,
      ((rest.map Prod.fst).dropLast.map List.tail).zip ((rest.map fun m => outCoin m.2).dropLast), ?_⟩
    rw [actO_eq _ _ hset]
    exact actAt_decide i.base spec sts m₀.1 (rest.map Prod.fst) (outCoin m₀.2) (rest.map fun m => outCoin m.2)
      (a := ml.1) (c := outCoin ml.2)
      (by
        rw [show m₀.1 :: rest.map Prod.fst = (m₀ :: rest).map Prod.fst from rfl, List.getLast?_map, hml]
        try rfl)
      (by
        rw [show outCoin m₀.2 :: rest.map (fun m => outCoin m.2) = (m₀ :: rest).map (fun m => outCoin m.2) from rfl,
          List.getLast?_map, hml]
        try rfl)
      hfm hlk

/-- **G at each session**: where the compiled component accepts, `x` names its inputs, the blobs show the run's view
`v`, its staged rows are the component's, its records hold the live coins and the component's refusals hold (`Refusals`),
`x` is dense, it has a part, and every V\* session sets up at its staged statement and the outer verifier accepts it
there (pieces A, B, C, D2 and E). -/
theorem sessions_of_accepts (IV : Claims.Verifier Coin) (st : Stage) (x : Msg) (v : Claims.Seen)
    {p : VStar.Recursive.Params} {c : SampledProofs.OneStage.Commitment} {e : VStar.Recursive.Evidence}
    {bs : Core.Blobs} {pub : SampledProofs.OneStage.Public} {prof : Core.Integrity}
    (hacc : VStar.Recursive.checkBlobs p c e bs = .accept pub prof) (hnames : Names p e bs x)
    (hshow : Shows IV st x e.proofs bs v) (hrows : RowsShow IV st x e.proofs bs v) (hlive : LiveCoins IV st x v)
    (hrf : Refusals x st (v.msgs.take (kIS IV x))) :
    DenseAt x ∧ 0 < nS x ∧ ∀ s < nS x,
      (∃ o, setupO (staged st x (v.msgs.take (kIS IV x)) s) = some o) ∧
      ((nextO (staged st x (v.msgs.take (kIS IV x)) s) ((session IV st x s v).msgs.map Prod.fst)
        ((session IV st x s v).msgs.map fun m => outCoin m.2)).eval FlockDeploy.sha512).1 = .acc := by
  obtain ⟨h, ss, hh, hss, hdn, hsz1, hsz2, hparts, -⟩ := parts_of_accepts hacc
  have hsp := sparse_of_dense hdn
  obtain ⟨w, ri, hx, hi, hI, hl, hPN⟩ := id hnames
  have hlt : ∀ s q, w.parts[s]? = some q → s < p.own.parts.size ∧ s < e.proofs.parts.size := by
    intro s q hq
    obtain ⟨hs, -⟩ := List.getElem?_eq_some_iff.1 hq
    omega
  have hPN' : ∀ s q (hq : w.parts[s]? = some q),
      PartNames (p.own.parts[s]'(hlt s q hq).1) (e.proofs.parts[s]'(hlt s q hq).2) bs s q := fun s q hq =>
    hPN s q _ _ hq (Array.getElem?_eq_getElem (hlt s q hq).1) (Array.getElem?_eq_getElem (hlt s q hq).2)
  have hcore := fun s q (hq : w.parts[s]? = some q) =>
    part_setup_of_verdict (st := st) (x := x) (ms := v.msgs.take (kIS IV x)) hh hsp hI
      (hparts s (hlt s q hq).1 (hlt s q hq).2) (hPN' s q hq)
      (hshow.2.2 s _ (Array.getElem?_eq_getElem (hlt s q hq).2)).1 hshow.2.1 hshow.1 hrows
  have hset : PartsSetUp p.own h x := by
    intro w' ri' hx' hi' s q hq
    have ew : w' = w := Option.some.inj (hx'.symm.trans hx)
    rw [ew, hi] at hi'
    have eri : ri' = ri := (Option.some.inj hi').symm
    rw [ew, eri]
    rw [ew] at hq
    obtain ⟨_, _, _, _, _, _, _, _, -, -, -, -, -, hins, hsu, hck, -⟩ := hcore s q hq
    exact ⟨st, _, _, _, _, hins, hsu, hck⟩
  obtain ⟨w', ri', hx', hi', -, hrx, hdense⟩ := readX_of_names hnames hh hss hsz1 hrf.innerOk hset
  have ew : w' = w := Option.some.inj (hx'.symm.trans hx)
  rw [ew, hi] at hi'
  have eri : ri' = ri := (Option.some.inj hi').symm
  rw [ew, eri] at hrx
  have hnS : nS x = w.parts.length := by
    unfold RecDeploy.nS
    rw [hrx]
    try rfl
  have hpos : 0 < nS x := by
    rw [hnS, hl, hsz1]
    exact statements_pos hss
  refine ⟨hdense, hpos, fun s hs => ?_⟩
  have hsw : s < w.parts.length := by omega
  have hq : w.parts[s]? = some w.parts[s] := List.getElem?_eq_getElem hsw
  obtain ⟨tags0, registered, bindings, recordB, record, d, spec, sts, hT, hreg, hrecB, hrec, hrd, hins, hsu, -,
    hbuild⟩ := hcore s _ hq
  obtain ⟨hpubS, ⟨m, link, hhd, hlk⟩, f, hfin, hstmt, hrecf, hrep⟩ :=
    hshow.2.2 s _ (Array.getElem?_eq_getElem (hlt s _ hq).2)
  have hP := hPN' s _ hq
  have hfit := Part.partRaw_fits hrx hq hP.form (hrf.pub s hs) hrf.record hrf.transcript hrf.rows
  obtain ⟨hinsO, hsetO⟩ := setupO_staged_of_setup hrx hq (Part.decodeRec_encode_fits hfit) hins rfl hsu
  have hvs := vstarAt_staged hsetO
  rw [hvs] at hstmt hrep
  obtain ⟨hnp, hreps⟩ := hrep _ _ _ hsetO
  obtain ⟨c₁, mids, ha⟩ := actO_decide_of (L := (session IV st x s v).msgs) hsetO hfin hhd hlk
  have hl2 := hlive s hs
  rw [hvs] at hl2
  have hlv := hl2 _ _ _ _ _ _ _ ha
  have hrecord : f.record = record := by
    have e1 : f.record.toUTF8 = recordB := Except.ok.inj ((get_what hrecf).symm.trans hrecB)
    have e2 := FlockComplete.fromUTF8?_toUTF8 f.record
    rw [e1, hrec] at e2
    exact (Option.some.inj e2).symm
  refine ⟨⟨_, hsetO⟩, ?_⟩
  exact nextO_acc_of_partVerdict (hparts s (hlt s _ hq).1 (hlt s _ hq).2) ha hT hP.circuit hpubS
    (Part.partIns_tags _ _ _ _ _ _ _ _ _ _) rfl rfl rfl rfl hreg rfl
    (by unfold innerOfO; rw [hinsO]; try rfl)
    (by unfold srcOfO; rw [hinsO]; try rfl)
    (by rw [hrecord]; exact hrd.symm)
    (by rw [Part.partIns_tags]; exact hbuild)
    hstmt hrecf hreps hnp hlv

/-- `acc_of_accepts` where the component's refusals hold at the run's inner rounds (`Refusals`). -/
theorem acc_of_accepts_of (wE w : Msg → ℝ) (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (p : VStar.Recursive.Params) (c : SampledProofs.OneStage.Commitment) (e : VStar.Recursive.Evidence)
    (bs : Core.Blobs) {pub : SampledProofs.OneStage.Public} {prof : Core.Integrity}
    (hacc : VStar.Recursive.checkBlobs p c e bs = .accept pub prof)
    (x : Msg) (hnames : Names p e bs x) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω)
    (r : Fin ((recordAt wE w b aux nOf hd recRest).V.k x) → Coin)
    (hshow : Shows (boundAt wE w).IVStar (stageAt aux nOf hd recRest) x e.proofs bs
      (Claims.run FlockDeploy.sha512 P (recordAt wE w b aux nOf hd recRest).V x ω r).seen)
    (hrows : RowsShow (boundAt wE w).IVStar (stageAt aux nOf hd recRest) x e.proofs bs
      (Claims.run FlockDeploy.sha512 P (recordAt wE w b aux nOf hd recRest).V x ω r).seen)
    (hlive : LiveCoins (boundAt wE w).IVStar (stageAt aux nOf hd recRest) x
      (Claims.run FlockDeploy.sha512 P (recordAt wE w b aux nOf hd recRest).V x ω r).seen)
    (hrf : Refusals x (stageAt aux nOf hd recRest)
      ((Claims.run FlockDeploy.sha512 P (recordAt wE w b aux nOf hd recRest).V x ω r).seen.msgs.take
        (kIS (boundAt wE w).IVStar x))) :
    RecDeploy.DenseAt x ∧
      (Claims.run FlockDeploy.sha512 P (recordAt wE w b aux nOf hd recRest).V x ω r).out = .acc := by
  obtain ⟨hdense, hpos, hsess⟩ := sessions_of_accepts (boundAt wE w).IVStar (stageAt aux nOf hd recRest) x
    (Claims.run FlockDeploy.sha512 P (recordAt wE w b aux nOf hd recRest).V x ω r).seen hacc hnames hshow hrows hlive
    hrf
  exact ⟨hdense, run_acc_of_staged (boundAt wE w).IVStar (stageAt aux nOf hd recRest) x P ω r hpos
    (fun s hs => (hsess s hs).1) (fun s hs => (hsess s hs).2)⟩

/-- **The component's property: its acceptance is the event `RecursiveKSAt` is stated over.** Suppose
the compiled recursive component accepts `(p, c, e, bs)`, `x` names its inputs (`Names`), the blobs show a run of the
deployed recursive verifier (`recordAt`) at `x` (`Shows`) whose staged rows are the component's (`RowsShow`), and
that run's records hold its live coins (`LiveCoins`).
Then `x` is dense (`RecDeploy.DenseAt`, the statements `RecursiveKSAt` covers), and the run accepts.

The proof is `acc_of_accepts_of` at `Refusals`, the model's statement of what the compiled component refuses
(`innerOk`, `fits`, `statementFits`), read off its check (`RefusalsOf`). -/
theorem acc_of_accepts (wE w : Msg → ℝ) (b : Brings) (aux : Msg → ℕ → Flock.RecRecord.Round) (nOf : Msg → ℕ → ℕ)
    (hd : Msg → Flock.RecRecord.Record) (recRest : Msg → List (Msg × Out) → List UInt8)
    (p : VStar.Recursive.Params) (c : SampledProofs.OneStage.Commitment) (e : VStar.Recursive.Evidence)
    (bs : Core.Blobs) {pub : SampledProofs.OneStage.Public} {prof : Core.Integrity}
    (hacc : VStar.Recursive.checkBlobs p c e bs = .accept pub prof)
    (x : Msg) (hnames : Names p e bs x) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω)
    (r : Fin ((recordAt wE w b aux nOf hd recRest).V.k x) → Coin)
    (hshow : Shows (boundAt wE w).IVStar (stageAt aux nOf hd recRest) x e.proofs bs
      (Claims.run FlockDeploy.sha512 P (recordAt wE w b aux nOf hd recRest).V x ω r).seen)
    (hrows : RowsShow (boundAt wE w).IVStar (stageAt aux nOf hd recRest) x e.proofs bs
      (Claims.run FlockDeploy.sha512 P (recordAt wE w b aux nOf hd recRest).V x ω r).seen)
    (hlive : LiveCoins (boundAt wE w).IVStar (stageAt aux nOf hd recRest) x
      (Claims.run FlockDeploy.sha512 P (recordAt wE w b aux nOf hd recRest).V x ω r).seen) :
    RecDeploy.DenseAt x ∧
      (Claims.run FlockDeploy.sha512 P (recordAt wE w b aux nOf hd recRest).V x ω r).out = .acc :=
  acc_of_accepts_of wE w b aux nOf hd recRest p c e bs hacc x hnames P ω r hshow hrows hlive
    ⟨RefusalsOf.innerOkAt_of_compiled hacc hnames,
     RefusalsOf.pub_of_compiled hacc hnames (fun s q hq => (hshow.2.2 s q hq).1),
     RefusalsOf.record_of_compiled hacc hshow.2.1,
     RefusalsOf.transcript_of_compiled hacc hshow.1,
     RefusalsOf.rows_of_compiled hacc hrows⟩

/-- **Sampled proofs' one-stage form is v2** (the bridge's carry-over lemma, soundness 02:10Z): where the parameters name
the one-stage form and `VStar.Recursive.Sampled` accepts, the evidence is one-stage too and v2
(`SampledProofs.OneStageZk.checkBlobs`) accepts the same commitment, evidence and blobs with the same public output and
profile. So what is proved of v2's acceptance (`acceptMeans`, the sampled proofs bound) holds of the
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
def RecVerdict (V : Vocab d α) (o : Verity.Ending) : Prop :=
  V.Registered o.params o.file ∧ RegistrationAccepts o.sent ∧
    (∃ j p, Core.canonicalJson o.params = .ok j ∧ VStar.Recursive.Sampled.paramsOfJson j = .ok (.inr p)) ∧
    ∃ bs, (VStar.Recursive.Sampled.protocol.answer o.params o.commitment o.evidence bs).verdict = .accept

/-- **The honest half** (non-vacuity). On every dense statement where the inner relation holds, the
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

/-- **The recursive session inside the causal game** at `θ`, `F`, `c` and the strategy `σ`: the coupling of the
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
  out : (ξ : Ξ) → Ω × (Fin (q.D.V.k (x ξ)) → Coin) → Verity.Ending
  drawOf : Ξ → Finset (Fin (V.units F))
  prob_le : ∀ E : Verity.Ending → Prop, Game.prob E (V.game θ F c) σ ≤
    ENNReal.ofReal (Claims.Ex fun ξ => Claims.Pr fun s : Ω × (Fin (q.D.V.k (x ξ)) → Coin) => E (out ξ s))
  names_shows : ∀ ξ s bs j p e,
    (VStar.Recursive.Sampled.protocol.answer (out ξ s).params (out ξ s).commitment (out ξ s).evidence bs).verdict =
      .accept →
    Core.canonicalJson (out ξ s).params = .ok j → VStar.Recursive.Sampled.paramsOfJson j = .ok (.inr p) →
    VStar.Recursive.Sampled.evidenceOf (out ξ s).evidence = .ok (.inr e) →
    Names p e (Core.Blobs.ofArray bs) (x ξ) ∧
      Shows q.Bd.IVStar q.st (x ξ) e.proofs (Core.Blobs.ofArray bs)
        (Claims.run FlockDeploy.sha512 (P ξ) q.D.V (x ξ) s.1 s.2).seen ∧
      RowsShow q.Bd.IVStar q.st (x ξ) e.proofs (Core.Blobs.ofArray bs) (Claims.run FlockDeploy.sha512 (P ξ) q.D.V (x ξ) s.1 s.2).seen
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

/-- **The strategy's cost parameter `t'`**: its SHA-512 calls in one run (`Vocab.cost`), at least one.
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
strategy's cost parameter `t' = tCost` (`Vocab.cost`), as `qF` and `qS` are of `t' τ`. Quantitative, so `⊤` is the
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
/-- **The error at the strategy `τ`**, the shared `condError FinderCR δ τ := if FinderCR τ then δ τ else ⊤`
(the shape of `Vocab.error`): `delta` where the finders built from `τ` meet `FinderCR`, and `⊤` elsewhere. -/
noncomputable def err (Sof : (τ : Game.Strategy (V.game θ F c)) → SessionIn V q θ F c τ)
    (collOf : Msg → Claims.Flock.Wit → Msg → Claims.Flock.Wit → Option (Msg × Msg)) (dColl : ℕ)
    (nI vS : Msg → ℕ) (δs lam : ℝ) (τ : Game.Strategy (V.game θ F c)) : ℝ≥0∞ :=
  if q.FinderCR V (Sof τ) collOf dColl nI vS then q.delta V (Sof τ) dColl δs lam else ⊤

end Dep

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
bound. The other hypotheses are code facts and the coupling, each its owner's: `RecursiveKS`, `HonestHalf`,
`hread` and `hcoll` (the bridge, with `place`), `Sof` and `hdraw` (the session's projection). -/
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
  exact (Game.prob_false _ σ).le.trans zero_le

end Verity.Recursive
