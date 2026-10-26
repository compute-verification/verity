import Proofs.Properties.SampledProofs.OneStageZk
import Proofs.Flock.Claims.Sampling.Deployed
import Proofs.Flock.Claims.KS.ZkAccepts
import Proofs.Flock.Claims.Recursion.RecRecord

/-! # One-stage v2's acceptance is the sampled model's (ONE-3t, PRF-209)

Sampled proofs' one-stage audit on the proofs (`verity/sampled-proofs/one-stage/v2`, `checkBlobs`) runs
`flock-verify verify --zk` on each of the evidence's sessions, from the blobs. `acc_of_accepts` says what that
acceptance means for the run `Claims.Sampling`'s bounds count: the run of the deployed sampled verifier
(`SamplingDeploy.deploy rel cd`) at a checked unit whose session the blobs show accepts.

* `Names`: the session's statement `xS`, as the deployed Flock verifier reads its bytes (`FlockDeploy.Raw`), is
  `verify --zk` at the component's own inputs: the `verity/flock-circuit` statement, the circuit and public files the
  digests name, the lookup tables the blobs hold, one table, the record's own draw, the verifier's partition, program
  and registered file, and no other input.
* `Shows`: the session's last message is the `Final` whose statement is `xS`, whose record is the session's record
  blob, and whose proof files are the session's rep blobs, in the stream order of `xS`'s setup.
* `LiveCoins` (`live-verifier`, `Assumptions.Zk.RecordCustodyZKJ`, at the session): the session is the live one its
  record was written for. It opens with its link message, and wherever the deployed verifier decides on its messages
  and coins, the record's live session is that link, its coin and the round messages with theirs
  (`FlockDeploy.custody`). The model verifier checks this itself; the compiled component, which holds no live coin,
  can't. It is admissible until ONE-4 (PRF-200) lands, after which the component reads the coins through ONE-4's
  signed device log and this premise goes.
* `SessionRuns sha512 (deploy rel cd)`: the code fact that a checked unit's session is a Flock run deciding the
  sampled run, which `sampling_deployed` takes too. `SamplingDeploy.Code.V` is a field, so nothing else ties the
  sampled verifier's runs to Flock's.
-/

open Lean (Json)
open SampledProofs SampledProofs.OneStageZk

namespace Properties.SampledProofs.OneStageZk

open Claims (Msg Out)

/-- The last message of a session's view, as the deployed verifier's `Final`. -/
def finalIn (v : Claims.Seen) : Option FlockDeploy.Final :=
  (v.msgs.getLast?).bind fun m => FlockDeploy.finalOf m.1

/-- **The statement names the component's inputs.** `xS` decodes (`FlockDeploy.decode`) to `verify --zk` at the
`verity/flock-circuit` statement without `--coins`, on the circuit and public files the blobs hold under the digests
`p.circuit` and `pub`, the lookup tables the blobs hold (`tablesOf`), one table, the draw of `record` (`recordDraw`, as
the deployed verifier rebuilds a draw from its bytes), the verifier's own partition, program and registered file, and
no `--work`, `--work-table`, `--stratified`, `--closure` or `--public-inputs`. -/
def Names (p : Own) (pub : String) (blobs : Array ByteArray) (record : String) (xS : Msg) : Prop :=
  ∃ w : FlockDeploy.Raw, FlockDeploy.decode xS = some w ∧ w.statement = STATEMENT ∧ w.zk = true ∧
    w.osCoins = false ∧
    (Core.Blobs.ofArray blobs).get p.circuit "circuit" = .ok w.circuit ∧
    (Core.Blobs.ofArray blobs).get pub "public file" = .ok w.publicFile ∧
    w.tables = tablesOf (Core.Blobs.ofArray blobs) ∧ w.count = 1 ∧
    (w.draw.mapM (FlockDeploy.jsonOf "draw")).map (Option.map Flock.norm) = .ok (Flock.Witness.recordDraw record) ∧
    w.partition.map (FlockDeploy.jsonOf "partition") = some (.ok p.partition) ∧ w.program = some p.program ∧
    w.work = none ∧ w.workTable = none ∧ w.stratified = none ∧ w.closure = none ∧
    w.registered.map (FlockDeploy.jsonOf "registered reads") = p.registered.map .ok ∧ w.publicInputs = none

/-- **The blobs show the session.** The session view `v`'s last message is a `Final` `f` whose statement is `xS`,
which names the component's inputs at `f`'s record (`Names`); the blob under the session's `record` digest is `f`'s
record; and at `xS`'s setup, `f` carries one proof file per rep stream, each the blob the session's `reps` name for
that stream's file (`<stream with / as .>.bin`). -/
def Shows (p : Own) (pub : String) (blobs : Array ByteArray) (s : Session) (xS : Msg) (v : Claims.Seen) : Prop :=
  ∃ f, finalIn v = some f ∧ f.stmt = xS ∧ Names p pub blobs f.record xS ∧
    (Core.Blobs.ofArray blobs).get s.record s!"session {s.name}'s record" = .ok f.record.toUTF8 ∧
    ∀ i spec sts, FlockDeploy.setupOf true xS = some (i, spec, sts) →
      f.proofs.size = spec.repStreams.size ∧ ∀ k (hk : k < spec.repStreams.size), ∃ d,
        (s.reps.find? (·.1 == s!"{spec.repStreams[k].2.2.replace "/" "."}.bin")).map (·.2) = some d ∧
        ∃ hk' : k < f.proofs.size,
          (Core.Blobs.ofArray blobs).get d s!"session files: {spec.repStreams[k].2.2.replace "/" "."}.bin" =
            .ok f.proofs[k]

/-- **The record holds the session's live coins** (`live-verifier` at the session, `Assumptions.Zk.RecordCustodyZKJ`):
the session view `v` at `xS` opens with its link message (`FlockDeploy.linkOf`), and wherever the deployed verifier
decides on `v`'s messages and the coins its answers show (`RecDeploy.outCoin`), the live session the record shows is
that link, its coin and the round messages with theirs (`FlockDeploy.custody`). -/
noncomputable def LiveCoins (xS : Msg) (v : Claims.Seen) : Prop :=
  (∃ link, (v.msgs.head?).bind (fun m => FlockDeploy.linkOf m.1) = some link) ∧
    ∀ i spec sts link c₁ mids f,
      FlockDeploy.act true xS (v.msgs.map Prod.fst) (v.msgs.map fun m => RecDeploy.outCoin m.2) =
          .decide i spec sts link c₁ mids f →
        (do FlockDeploy.custody spec (← FlockDeploy.liveOf spec f.record) link c₁ mids : Flock.V Unit) = .ok ()

/-- **ONE-3t's property: one-stage v2's acceptance is the sampled model's.** Suppose one-stage v2 accepts
`(p, c, e, blobs)`, and the deployed sampled verifier (`SamplingDeploy.deploy rel cd`, whose checked units' sessions
are Flock runs, `SessionRuns`) checks the unit `u` at the coins `r`. If `u`'s session in the run's view shows one of
the evidence's sessions `s` (`Shows`) and holds its live coins (`LiveCoins`), the run accepts. -/
theorem acc_of_accepts (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (cd : SamplingDeploy.Code)
    (hruns : Claims.Sampling.SessionRuns FlockDeploy.sha512 (SamplingDeploy.deploy rel cd))
    (p : Params) (c : OneStage.Commitment) (e : Evidence) (blobs : Array ByteArray) {pub : OneStage.Public}
    {prof : Core.Integrity} (hacc : checkBlobs p c e (Core.Blobs.ofArray blobs) = .accept pub prof)
    (x : Msg) {Ω : Type} (P : Claims.Prover Ω) (ω : Ω)
    (r : Fin ((SamplingDeploy.deploy rel cd).V.k x) → FlockDeploy.Coin) (u : ℕ × ℕ)
    (hu : ∃ q : Fin (Claims.Sampling.nPU (SamplingDeploy.deploy rel cd) x),
      ((SamplingDeploy.deploy rel cd).checked x r)[q.val]? = some u)
    (s : Session) (hs : s ∈ e.sessions)
    (hshow : ∀ xS v, (SamplingDeploy.deploy rel cd).session x u
        (Claims.run FlockDeploy.sha512 P (SamplingDeploy.deploy rel cd).V x ω r).seen = some (xS, v) →
      Shows p.own e.pub blobs s xS v ∧ LiveCoins xS v) :
    (Claims.run FlockDeploy.sha512 P (SamplingDeploy.deploy rel cd).V x ω r).out = .acc := by
  sorry

end Properties.SampledProofs.OneStageZk
