import SampledProofs.OneStageZk

/-!
# Sampled proofs' one-stage audit on the proofs (`verity/sampled-proofs/one-stage/v2`): what its Lean verdict means

The property of `SampledProofs.OneStageZk.checkBlobs`, the check onsite's and remote's sampled proofs subprotocol runs on the
caller's blobs (`Core.BlobProtocol.on`, so on `Core.Blobs.ofArray blobs`), stated against it as it runs, for every input
of the types its decoders produce. It takes no premise.

`AcceptMeans`: what v2 accepts on `blobs` is `Accepted`, the single form's (`AcceptedOne`) or the group form's
(`AcceptedGroup`). The single form's has

* **the files the parameters and the evidence commit**: a circuit file, one of the caller's blobs, whose SHA-512 is the
  auditor's committed digest (`proofs.circuit`: the registration check, R6), and a public file, one of the blobs, whose
  SHA-512 is the evidence's `proofs.public` (`Blob`);
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

The group form's has the same for each member, on its own inputs (`MemberAccepted`: the auditor's circuit digest and
registered file for the member, the group's program and partition object, the member's public file), with **the
registration's leaf layer naming the members' public files** (`LeafLayer`), **the members' proved units each proved
once and together the evidence's `proved_units`**, and v1's check accepting the verdicts. `CircuitsHeld` reads R6 off
either: every committed circuit digest names one of the blobs.

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

/-- The bytes `b` are one of the caller's `blobs`, and the digest `d` names them. -/
def Blob (blobs : Array ByteArray) (d : String) (b : ByteArray) : Prop := b ∈ blobs ∧ Names d b

/-- One session as v2 verified it on `I`: its record is the bytes its `record` digest names, as the UTF-8 text `record`;
flock-verify `--zk` accepts it with the proof files `proofs`; and it proved the units `units`. -/
def SessionAccepted (I : Flock.Witness.Inputs) (s : Session) (record : String) (proofs : Array ByteArray)
    (units : Array Nat) : Prop :=
  (∃ b, Names s.record b ∧ String.fromUTF8? b = some record) ∧ AcceptsZK I record proofs ∧
    provedUnits I.publicFile record = .ok units

/-- What the single form accepted on `blobs`, publishing `pub` and `prof`: the committed circuit and public files; for
each of the evidence's sessions, in order (`ss`, at least one), its record and proof files flock-verify `--zk` accepts
on `verify --zk`'s inputs, and the units it proved, which together are the evidence's `proved_units`; the descriptor's
digest the partition object's program; and v1's check accepting the verdicts this run computed, with what it
publishes. -/
def AcceptedOne (p : Params) (c : OneStage.Commitment) (e : Evidence) (blobs : Array ByteArray) (pub : OneStage.Public)
    (prof : Core.Integrity) : Prop :=
  ∃ (tags : Flock.Tags) (circuitB pubB : ByteArray) (ss : List (String × Array ByteArray × Array Nat))
    (vs : Array (String × Json)),
    Flock.Tags.find? STATEMENT = some tags ∧ Blob blobs p.own.circuit circuitB ∧ Blob blobs e.pub pubB ∧
    0 < e.sessions.size ∧ ss.length = e.sessions.size ∧
    (∀ x ∈ e.sessions.toList.zip ss,
      SessionAccepted (inputsOf tags p.own circuitB pubB (Core.Blobs.ofArray blobs)) x.1 x.2.1 x.2.2.1 x.2.2.2) ∧
    (ss.foldl (fun acc x => acc ++ x.2.2) #[]).qsort (· < ·) = e.base.provedUnits ∧
    (∃ h, programDigest p.own.program = some h ∧ (Json.str h == getD p.own.partition "program") = true) ∧
    OneStage.check p.base c { e.base with verdicts := vs } = .accept pub prof

/-- One member as the group form verified it on `blobs`, at the statement's tags `tags`: its circuit file, one of the
blobs, whose SHA-512 is the member's committed digest (`o.circuit`, the auditor's); its public file, one of the blobs,
whose SHA-512 is its `public`; for each of its sessions, in order (`ss`), its record and proof files flock-verify
`--zk` accepts on `verify --zk`'s inputs at the member's own (`inputsOf tags o`: the group's program and partition
object, the member's circuit and public file), with the units it proved; and, when it has a session, the descriptor's
digest the partition object's program. -/
def MemberAccepted (tags : Flock.Tags) (o : Own) (x : Member) (blobs : Array ByteArray)
    (ss : List (String × Array ByteArray × Array Nat)) : Prop :=
  ∃ circuitB pubB, Blob blobs o.circuit circuitB ∧ Blob blobs x.pub pubB ∧ ss.length = x.sessions.size ∧
    (∀ y ∈ x.sessions.toList.zip ss,
      SessionAccepted (inputsOf tags o circuitB pubB (Core.Blobs.ofArray blobs)) y.1 y.2.1 y.2.2.1 y.2.2.2) ∧
    (0 < x.sessions.size → ∃ h, programDigest o.program = some h ∧ (Json.str h == getD o.partition "program") = true)

/-- **The registration's leaf layer names the members' public files**: the record's `leaf_layer` is the SHA-512 of the
canonical object naming each member's public file's digest by the member's name, so the public files the sessions
were verified on are the ones registered before the draw. -/
def LeafLayer (c : OneStage.Commitment) (xs : Array Member) : Prop :=
  ∃ t d, Flock.canon (Json.mkObj (xs.map fun x => (x.name, Json.str x.pub)).toList) = .ok t ∧
    (getD c.record "leaf_layer" == .str d) = true ∧ Names d t.toUTF8

/-- What the group form accepted on `blobs`, publishing `pub` and `prof`: at least one member, the evidence's members
the parameters' by name in order; the registration's leaf layer the members' public files' (`LeafLayer`); each member
accepted on its own inputs (`MemberAccepted`, `sss` its sessions' witnesses), at least one session among them; the
units the sessions proved, member by member in session order, sorted, the evidence's `proved_units`, which are
strictly increasing, so no unit is proved twice; and v1's check accepting the verdicts this run computed, with what
it publishes. -/
def AcceptedGroup (p : Params) (c : OneStage.Commitment) (e : Evidence) (blobs : Array ByteArray)
    (pub : OneStage.Public) (prof : Core.Integrity) : Prop :=
  ∃ (tags : Flock.Tags) (sss : List (List (String × Array ByteArray × Array Nat))) (vs : Array (String × Json)),
    Flock.Tags.find? STATEMENT = some tags ∧ 0 < p.members.size ∧
    e.members.map (·.name) = p.members.map (·.1) ∧ LeafLayer c e.members ∧ sss.length = p.members.size ∧
    (∀ y ∈ (p.members.toList.zip e.members.toList).zip sss, MemberAccepted tags y.1.1.2 y.1.2 blobs y.2) ∧
    0 < (e.members.toList.map (·.sessions.size)).sum ∧
    (sss.foldl (fun acc ss => ss.foldl (fun acc x => acc ++ x.2.2) acc) #[]).qsort (· < ·) = e.base.provedUnits ∧
    increasing e.base.provedUnits.toList = true ∧
    OneStage.check p.base c { e.base with verdicts := vs } = .accept pub prof

/-- What v2 accepted on `blobs`: the single form's, with no member in the parameters or the evidence, or the group
form's. -/
def Accepted (p : Params) (c : OneStage.Commitment) (e : Evidence) (blobs : Array ByteArray) (pub : OneStage.Public)
    (prof : Core.Integrity) : Prop :=
  (p.members = #[] ∧ e.members = #[] ∧ AcceptedOne p c e blobs pub prof) ∨ AcceptedGroup p c e blobs pub prof

/-- **The parameters' committed circuit files are blobs**: the single form's circuit digest, and each member's, names
one of the caller's blobs. -/
def CircuitsHeld (p : Params) (blobs : Array ByteArray) : Prop :=
  (p.members = #[] → ∃ F, Blob blobs p.own.circuit F) ∧ ∀ m ∈ p.members, ∃ F, Blob blobs m.2.circuit F

/-- **What v2 accepts on `blobs` is `Accepted`.** -/
def AcceptMeans : Prop :=
  ∀ (p : Params) (c : OneStage.Commitment) (e : Evidence) (blobs : Array ByteArray) (pub : OneStage.Public)
    (prof : Core.Integrity),
    checkBlobs p c e (Core.Blobs.ofArray blobs) = .accept pub prof → Accepted p c e blobs pub prof

end Properties.SampledProofs.OneStageZk
