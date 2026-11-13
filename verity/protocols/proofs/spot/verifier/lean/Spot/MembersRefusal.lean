module

public import Spot.MembersParse
meta import Spot.OneStageZk
meta import Spot.MembersParse

/-!
# The per-member form's refusal, run

`refusal PARAMS COMMITMENT EVIDENCE [BLOB ...]`, a command of the one-stage v2 executable (`ZkMain`): why the audit's
check refuses a case of the per-member form, and no verdict of an audit.

It reads a case's files as `check` does, with the parsers that read either form (`paramsEitherOfJson`,
`evidenceEitherOf`), and runs the audit's check on the blobs (`checkBlobs`) unless both are the one statement's, which
it refuses unread (`NOT_MEMBERS`): `check` decides that form. The per-member form's check accepts nothing
(`checkMembersAt_refuses`), so the command answers only why a case is refused: the reason of the first check that
fails, and `MULTI_REFUSED` exactly where every check before the refusal passed. `membersOutcome_refuses` proves it
accepts nothing and `refusal_exit` that its exit is never 0, on any files. Its line holds the reason under `refusal`
and says what it is (`REFUSAL`); it names no verdict, so nothing reads it as a verifier's. `protocol` reads
`paramsOfJson` and `evidenceOf`, which refuse the form's keys, and this command is outside `protocol`.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Spot.OneStageZk

open Lean (Json)
open Flock (V)

/-- What `refusal` says of what it prints: a refusal's reason, and no verdict of an audit. -/
def REFUSAL : String := "the per-member form's refusal, no verdict"

/-- `refusal`'s answer to a case whose parameters and evidence are both the one statement's, which it does not
check. -/
def NOT_MEMBERS : String :=
  "proofs: the parameters and the evidence are the one statement's, which check decides: the refusal's run reads " ++
    "the per-member form"

/-- **The audit's check accepts only the one statement's form**: where `checkBlobs` accepts, the parameters name no
members and the evidence no statements (`checkBlobsAt_accept`). -/
theorem checkBlobs_accept {p : Params} {c : OneStage.Commitment} {e : Evidence} {bs : Core.Blobs}
    {pub : OneStage.Public} {prof : Core.Integrity} (h : checkBlobs p c e bs = .accept pub prof) :
    p.members = none ∧ e.statements = none := by
  unfold checkBlobs Core.atCoins at h
  split at h
  · exact (checkBlobsAt_accept h).2
  · cases h

/-- **The audit's check on a case's files, read in either form**: the parameters `params` (canonical JSON,
`paramsEitherOfJson`), the registration record `commitment` and the evidence `evidence` (`evidenceEitherOf`), each
refused under its file's name as `Protocol.answer` refuses it, then `checkBlobs` on the blobs `blobs`; parameters and
evidence that are both the one statement's are refused unchecked (`NOT_MEMBERS`). -/
def membersOutcome (params commitment evidence : ByteArray) (blobs : Array ByteArray) :
    Core.Outcome OneStage.Public Core.Integrity :=
  match Core.canonicalJson params >>= paramsEitherOfJson with
  | .error why => .reject s!"params: {why}"
  | .ok p =>
    match OneStage.commitmentOf commitment with
    | .error why => .reject s!"commitment: {why}"
    | .ok c =>
      match evidenceEitherOf evidence with
      | .error why => .reject s!"evidence: {why}"
      | .ok e =>
        if p.members.isNone && e.statements.isNone then .reject NOT_MEMBERS
        else checkBlobs p c e (Core.Blobs.ofArray blobs)

/-- **`membersOutcome` accepts nothing**, on any files: the one statement's form is refused unchecked, and on any
other `checkBlobs` accepts nothing (`checkBlobs_accept`, from `checkMembersAt_refuses`). -/
theorem membersOutcome_refuses (params commitment evidence : ByteArray) (blobs : Array ByteArray)
    (pub : OneStage.Public) (prof : Core.Integrity) :
    membersOutcome params commitment evidence blobs ≠ .accept pub prof := by
  intro h
  unfold membersOutcome at h
  split at h
  · cases h
  · split at h
    · cases h
    · split at h
      · cases h
      · split at h
        · cases h
        · rename_i hn
          obtain ⟨hm, hs⟩ := checkBlobs_accept h
          simp [hm, hs] at hn

/-- `refusal`'s exit on the check's outcome, as `check`'s: 0 on accept, 1 on reject and 2 on undecided. -/
def refusalExit : Core.Outcome OneStage.Public Core.Integrity → UInt32
  | .accept _ _ => 0
  | .reject _ => 1
  | .undecided _ => 2

/-- **`refusal` never exits 0**, on any files: its exit is `refusalExit` of `membersOutcome`, which accepts nothing
(`membersOutcome_refuses`). -/
theorem refusal_exit (params commitment evidence : ByteArray) (blobs : Array ByteArray) :
    refusalExit (membersOutcome params commitment evidence blobs) ≠ 0 := by
  have h := membersOutcome_refuses params commitment evidence blobs
  cases ho : membersOutcome params commitment evidence blobs with
  | accept pub prof => exact absurd ho (h pub prof)
  | reject r => first | (show (1 : UInt32) ≠ 0; decide) | (simp [refusalExit]; done)
  | undecided r => first | (show (2 : UInt32) ≠ 0; decide) | (simp [refusalExit]; done)

/-- `refusal`'s line on the check's outcome: exactly `{"decided", "refusal", "run"}`, the reason under `refusal`
(null on an acceptance, which `membersOutcome` never gives), `decided` false where the check is undecided, and `run`
the words `REFUSAL`. -/
def refusalLine (o : Core.Outcome OneStage.Public Core.Integrity) : Json :=
  let (decided, why) : Bool × Json := match o with
    | .accept _ _ => (true, Json.null)
    | .reject r => (true, Json.str r)
    | .undecided r => (false, Json.str r)
  Json.mkObj [("decided", .bool decided), ("refusal", why), ("run", .str REFUSAL)]

/-- `membersMain`'s commands, and `refusal PARAMS COMMITMENT EVIDENCE [BLOB ...]`: one line, `refusalLine` of
`membersOutcome` on the files, exit 1, or 2 where the check is undecided, never 0 (`refusal_exit`). The line names no
verdict. -/
def refusalMain (args : List String) : IO UInt32 := do
  match args with
  | "refusal" :: params :: commitment :: evidence :: blobs =>
    let bs ← blobs.toArray.mapM fun (f : String) => IO.FS.readBinFile f
    let o := membersOutcome (← IO.FS.readBinFile params) (← IO.FS.readBinFile commitment)
      (← IO.FS.readBinFile evidence) bs
    let line := refusalLine o
    IO.println ((Flock.canon line).toOption.getD line.compress)
    return refusalExit o
  | "refusal" :: _ =>
    IO.eprintln s!"usage: refusal PARAMS COMMITMENT EVIDENCE [BLOB ...] ({REFUSAL})"
    return 64
  | _ => membersMain args

private def refusalOf (o : Core.Outcome OneStage.Public Core.Integrity) : String := (refusalLine o).compress

#guard refusalOf (.reject "proofs: x") ==
  "{\"decided\":true,\"refusal\":\"proofs: x\",\"run\":\"the per-member form's refusal, no verdict\"}"
#guard refusalOf (.undecided "u") ==
  "{\"decided\":false,\"refusal\":\"u\",\"run\":\"the per-member form's refusal, no verdict\"}"
#guard match paramsOfJson (Json.mkObj []) with
  | .error why => refusalOf (membersOutcome "{}".toUTF8 .empty .empty #[]) == refusalOf (.reject s!"params: {why}")
  | .ok _ => false
#guard refusalExit (membersOutcome "{}".toUTF8 .empty .empty #[]) == 1
#guard refusalExit (.undecided "u") == 2

end Spot.OneStageZk
