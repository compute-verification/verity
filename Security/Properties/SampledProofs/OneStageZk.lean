import SampledProofs.OneStageZk

/-!
# Sampled proofs' one-stage audit on the proofs (`verity/sampled-proofs/one-stage/v2`): what its Lean verdict means

The property of `SampledProofs.OneStageZk.checkBlobs`, the check onsite's and remote's integrity component runs on the
caller's blobs (`Core.BlobProtocol.on`, so on `Core.Blobs.ofArray blobs`), stated against it as it runs, for every input
of the types its decoders produce. It takes no premise.

`AcceptMeans`: what v2 accepts on `blobs` has

* **the files the parameters and the evidence commit**: a circuit file whose SHA-512 is the auditor's committed digest
  (`proofs.circuit`), and a public file whose SHA-512 is the evidence's `proofs.public` (`Names`);
* **every session verified by flock-verify `--zk`**: at least one session, and for each of the evidence's sessions, in
  order, its record (the blob its `record` names, as UTF-8 text) and proof files on which flock-verify `--zk` accepts
  (`AcceptsZK`) on the inputs `verify --zk` reads (`inputsOf`: the statement `verity/flock-circuit`, those two files, one
  table, the lookup tables the blobs hold, and the verifier's own partition object and program), set up at the record's
  own draw, and the units the session proved (`provedUnits`: the public file's unit indices of the instances its record
  drew);
* **the proved units are the evidence's**: the sessions' proved units, in session order, sorted, are its `proved_units`;
* **the program is the partition object's**: the descriptor's digest (`programDigest`, `flock-verify`'s
  `program_sha512`) is the program the partition object names;
* **v1's check accepts the verdicts this run computed**, publishing `pub` and the profile `prof`: so
  `Properties.SampledProofs.AcceptMeans` gives the registration's R1–R7, the draw of the registered law over the
  registered population, the units proved and served as drawn with the receipt naming the commitment's digest
  (`UnitsProved`: `proved_units` is what the draw proves), and the published bounds and profile.

`AcceptsZK` is the model's `ZkExec.AcceptsZK` at `Witness.toExec I` by definition (`Witness.setupOf_eq`), the event
`SampledKSLive` and its restatements are stated over: `Proofs.Properties.SampledProofs.OneStageZk.acceptsZK_iff`.

Nothing here says where the draw came from: the sessions set up at their records' draws, which their proved units tie
to the evidence's `proved_units` and v1's `UnitsProved` to `Draw.proved` of the evidence's draw. That the draw is the
challenger's live coins is ONE-4's (`answerAt`).
-/

open Lean (Json)
open SampledProofs SampledProofs.OneStageZk

namespace Properties.SampledProofs.OneStageZk

/-- **flock-verify `--zk` accepts the session** `record` with the proof files `proofs` on the inputs `I`: the session's
statement sets up at the record's draw (`Flock.Witness.setupOf`, `Stmt.setupTables` under `--zk`'s tags and coins),
every table passes `Zk.shapeOk`, and `Flock.Zk.verify` accepts the record and the proof files. -/
def AcceptsZK (I : Flock.Witness.Inputs) (record : String) (proofs : Array ByteArray) : Prop :=
  ∃ spec sts, Flock.Witness.setupOf I record = .ok (spec, sts) ∧ sts.forM Flock.Zk.shapeOk = .ok () ∧
    Flock.Zk.verify spec (sts.map Flock.Setup.ofCircuit) record proofs = .ok ()

/-- The digest `d` names the bytes `b`: `d` is the lowercase hex of `b`'s SHA-512. -/
def Names (d : String) (b : ByteArray) : Prop := Flock.hexOf (Flock.Sha512.hash b) = d

/-- One session as v2 verified it on `I`: its record is the bytes its `record` digest names, as the UTF-8 text `record`;
flock-verify `--zk` accepts it with the proof files `proofs`; and it proved the units `units`. -/
def SessionAccepted (I : Flock.Witness.Inputs) (s : RunFiles) (record : String) (proofs : Array ByteArray)
    (units : Array Nat) : Prop :=
  (∃ b, Names s.record b ∧ String.fromUTF8? b = some record) ∧ AcceptsZK I record proofs ∧
    provedUnits I.publicFile record = .ok units

/-- What v2 accepts on `blobs`: the committed circuit and public files; for each of the evidence's sessions, in order
(`ss`, at least one), its record and proof files flock-verify `--zk` accepts on `verify --zk`'s inputs, and the units
it proved, which together are the evidence's `proved_units`; the descriptor's digest the partition object's program;
and v1's check accepting the verdicts this run computed, with what it publishes. -/
def AcceptMeans : Prop :=
  ∀ (p : Params) (c : OneStage.Commitment) (e : Evidence) (blobs : Array ByteArray) (pub : OneStage.Public)
    (prof : Core.Integrity),
    checkBlobs p c e (Core.Blobs.ofArray blobs) = .accept pub prof →
      ∃ (tags : Flock.Tags) (circuitB pubB : ByteArray) (ss : List (String × Array ByteArray × Array Nat))
        (vs : Array (String × Json)),
        Flock.Tags.find? STATEMENT = some tags ∧ Names p.own.circuit circuitB ∧ Names e.pub pubB ∧
        0 < e.sessions.size ∧ ss.length = e.sessions.size ∧
        (∀ x ∈ e.sessions.toList.zip ss,
          SessionAccepted (inputsOf tags p.own circuitB pubB (Core.Blobs.ofArray blobs)) x.1 x.2.1 x.2.2.1 x.2.2.2) ∧
        (ss.foldl (fun acc x => acc ++ x.2.2) #[]).qsort (· < ·) = e.base.provedUnits ∧
        (∃ h, programDigest p.own.program = some h ∧ (Json.str h == getD p.own.partition "program") = true) ∧
        OneStage.check p.base c { e.base with verdicts := vs } = .accept pub prof

end Properties.SampledProofs.OneStageZk
