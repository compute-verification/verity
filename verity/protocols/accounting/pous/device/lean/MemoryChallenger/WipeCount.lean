module

public import MemoryChallenger.WipeProtocol
meta import MemoryChallenger.WipeProtocol

/-!
# The wipe's count, as a composition holds it

`WipeProtocol.check` reads an egress count off its evidence and certifies nothing on it, since no network certifier
signs it. This module is the wipe's half of the step that holds that count to the network certifier's count over the
span between the session's two marks: what a composition's tie reads off the wipe's case, and the check that publishes
on the count once a composition holds it.

* `named`: the start mark the signed message names, the 64 bytes each of the session's two marks opens to
  (`StreamedWipe.startMessage`, `StreamedWipe.endMessage`), and the evidence's count, in whole MiB
  (`StreamedWipe.egressUnits`' unit). A composition holds each to what the count over a span between two marks names of
  its own case. The wipe names no device, link or key.
* `checkCounted`: `check`, and on its acceptance a rejection of an evidence that gives no count (`NO_COUNT`) and of a
  session that names no start mark (`NO_MARK`), else `certify` at the evidence's count in `uncounted`'s place. Its
  profile names the bound's gaps and not the count as one (`countedProfileJson`).

`checkCounted` holds the count to nothing, as `check` does: outside a composition that ties `named` to the network
certifier's count it certifies on what the evidence states. So no command line serves it under the accepted schemes.
`mkCounted` builds the protocol for the composition that holds the tie, and `countedTest` is the same under the test
signers' schemes alone, for the difftest (`wipe-count-test`).

`wipe_period.py`'s `named` and `outcome_counted` are the Python mirror, held to this by `tests/test_wipe_count.py`.
-/

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace MemoryChallenger.WipeProtocol

open Lean (Json)
open Flock

/-- The id of the wipe's component whose count a composition holds. -/
def COUNTED_ID : String := "verity/wipe/counted/v0"

def NO_COUNT : String :=
  "the evidence gives no egress count: a counted wipe is held to the network certifier's count over its span"

def NO_MARK : String :=
  "the session names no start mark: the memory challenger held no start certificate, so no count is of its span"

/-- What a wipe's case says of its span: the start mark its signed message names, the 64 bytes each of its two marks
opens to, and the egress its evidence gives, in whole MiB. -/
structure Named where
  startMark : ByteArray
  startMessage : ByteArray
  endMessage : ByteArray
  egressMib : Int

/-- `Named` of a case's commitment and evidence; `none` while the session is unsigned, when the evidence gives no
count, or when the signed session has no end message. -/
def named (c : StreamedWipe.Commitment) (e : Evidence) : Option Named :=
  match e.signed, e.egressMib with
  | some s, some n =>
    match StreamedWipe.endMessage s with
    | .ok m => some ⟨s.startMark, StreamedWipe.startMessage c, m, n⟩
    | .error _ => none
  | _, _ => none

/-- `check` with the count held by a composition: its refusals, then `NO_COUNT` and `NO_MARK`, and on acceptance
`certify` at the evidence's count. -/
def checkCounted (schemes : Schemes) (p : Params) (c : StreamedWipe.Commitment) (e : Evidence) :
    Core.Outcome Public (StreamedWipe.Params × Int) :=
  match check schemes p c e with
  | .accept _ _ =>
    match named c e with
    | none => .reject NO_COUNT
    | some n =>
      if n.startMark.size != StreamedWipe.MARK_BYTES then .reject NO_MARK
      else .accept (certify c.params n.egressMib) (c.params, n.egressMib)
  | o => o

/-- `profileJson` of a counted wipe: its gaps are the bound's (`gaps`) alone, the count being no gap once held. -/
def countedProfileJson (x : StreamedWipe.Params × Int) : Json :=
  (profileJson x).setObjVal! "gaps" (.arr ((gaps x.1 x.2).map Json.str).toArray)

/-- The wipe's component with its count held, for the composition that holds the tie: `mkProtocol`'s formats and
digest, `checkCounted` and `countedProfileJson`. -/
def mkCounted (id : String) (schemes : Schemes) : Core.Protocol where
  id := id
  Params := Params
  Commitment := StreamedWipe.Commitment
  Evidence := Evidence
  Public := Public
  Profile := StreamedWipe.Params × Int
  params := decodeParams
  commitment := decodeCommitment
  evidence := decodeEvidence
  digest := digest
  check := checkCounted schemes
  publicJson := Public.toJson
  profileJson := countedProfileJson

/-- `mkCounted` under the test signers' schemes alone, its id `test/verity/wipe/counted/v0`. -/
def countedTest : Core.Protocol := mkCounted (TEST_PREFIX ++ COUNTED_ID) TEST_SCHEMES

private def guardParams : StreamedWipe.Params := ⟨16, 3, 1, 1000, #[("gpu", 16 * StreamedWipe.LABEL_BYTES)]⟩

private def guardCommitment : StreamedWipe.Commitment := ⟨"n0".toUTF8, ⟨#[0]⟩, guardParams⟩

private def guardMark : ByteArray := ⟨Array.replicate StreamedWipe.MARK_BYTES 7⟩

/-- A closed session of no batches under no key: `check` rejects it, and `named` reads it. -/
private def guardSigned : StreamedWipe.Signed :=
  ⟨guardCommitment, StreamedWipe.COMPLETE, guardMark, 0, 0, #[], "", ByteArray.empty, ByteArray.empty⟩

#guard (named guardCommitment ⟨some guardSigned, some 3⟩).map (fun n =>
    (n.startMark == guardMark, n.startMessage == StreamedWipe.startMessage guardCommitment, n.endMessage.size,
      n.egressMib)) == some (true, true, 64, 3)
#guard (named guardCommitment ⟨some guardSigned, none⟩).isNone
#guard (named guardCommitment ⟨none, some 3⟩).isNone
-- `check`'s own refusals stand: an unsigned session is undecided, and one under no accepted scheme rejected
#guard match checkCounted TEST_SCHEMES ⟨"n0".toUTF8, ⟨#[0]⟩, #[]⟩ guardCommitment ⟨none, some 3⟩ with
  | .undecided _ => true
  | _ => false
#guard match checkCounted TEST_SCHEMES ⟨"n0".toUTF8, ⟨#[0]⟩, #[]⟩ guardCommitment ⟨some guardSigned, some 3⟩ with
  | .reject r => r != NO_COUNT && r != NO_MARK
  | _ => false
-- the counted profile names no gap for a count the bound covers, where `check`'s names the count
#guard match (countedProfileJson (guardParams, 0)).getObjVal? "gaps", (profileJson (guardParams, 0)).getObjVal? "gaps" with
  | .ok (.arr a), .ok (.arr b) => a.isEmpty && b.size == 1
  | _, _ => false
#guard countedTest.id == "test/verity/wipe/counted/v0"

end MemoryChallenger.WipeProtocol
