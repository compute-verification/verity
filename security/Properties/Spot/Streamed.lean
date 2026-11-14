module

public import Spot.OneStageZkStream

/-!
# v2's streamed check decides what its check decides on the blobs fed (`Spot.OneStageZk.Streamed.stream`)

The one-stage audit's v2 (`Spot.OneStageZk`) checks blobs it holds whole (`OneStageZk.checkBlobs`), and also blobs it
reads one at a time (`OneStageZk.Streamed.stream`, a `Core.Stream`): before the first blob it names their order by
SHA-512, it steps on each as it comes and keeps of it only what the check reads later, and it refuses at the first blob
out of its place (`Core.Stream.run`). `OneStageZk.Streamed.protocol`'s `stream` and `checkBlobs` are those two.

* `StreamIsCheck`: on every list of blobs, the streamed run is a rejection, or it is the answer of `checkBlobs` on the
  blobs fed: an accept with the same public outputs and profile, or a rejection or an undecided answer with the same
  reason.

It takes no premise and no hypothesis: the run is a function of the parameters, the commitment, the evidence and the
blobs fed. The loop that reads the files and feeds them (`Core.Stream.files`) is not in it.
-/

@[expose] public section

open Spot

namespace Properties.Spot

/-- **The streamed check refuses, or decides what `checkBlobs` decides on the blobs fed**: for every parameters,
commitment, evidence and list of blobs, in whatever order the blobs come, the run of v2's stream on them
(`Core.Stream.run`, a refusal read as a rejection) is a rejection, or it is `OneStageZk.checkBlobs`' answer on those
blobs, each under its SHA-512 (`Core.Blobs.ofArray`). So the stream accepts only what `checkBlobs` accepts, with its
public outputs and profile. -/
def StreamIsCheck : Prop :=
  ∀ (p : OneStageZk.Params) (c : OneStage.Commitment) (e : OneStageZk.Evidence) (bs : List ByteArray),
    (∃ r, (OneStageZk.Streamed.stream p c e).run .reject bs = .reject r) ∨
      (OneStageZk.Streamed.stream p c e).run .reject bs = OneStageZk.checkBlobs p c e (Core.Blobs.ofArray bs.toArray)

/-- Parameters of the per-member form that name no member, and no file of the verifier's own. -/
def streamedParams : OneStageZk.Params :=
  { base :=
      { verifier :=
          { program := default, query := .null, law := .null, workTable := none, scheme := "", ports := #[],
            leafLayer := none, anchors := #[], log := #[], partition := none },
        delta := ⟨0, 1⟩ },
    own := { programJ := .null, program := .empty, partition := .null, circuit := "", pub := "", registered := none },
    members := some #[] }

/-- A commitment of no record. -/
def streamedCommitment : OneStage.Commitment := { record := .null, digest := .empty }

/-- Evidence of the one statement's form, with no session: it names no statement, so with `streamedParams` it is of
neither form, and `checkBlobs` rejects it on any blobs. -/
def streamedEvidence : OneStageZk.Evidence :=
  { base :=
      { receipt := .null, draw := .null, provedUnits := #[], instances := 0, statement := .null, verdicts := #[],
        sessions := #[], served := none, sessionRoots := none },
    sessions := #[],
    statements := none }

/-- A streamed run that accepts every feed, with nothing sampled. -/
def acceptAllRun (_ : List ByteArray) : Core.Outcome OneStage.Public Core.Integrity :=
  .accept { drawn := #[], proved := #[], bound := 0, perTemplate := none }
    { programs := #[], prescribed := #[], linked := false, covered := #[], levels := #[], delta := ⟨0, 1⟩,
      drawn := #[] }

end Properties.Spot
