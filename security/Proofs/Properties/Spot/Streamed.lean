module

public import Properties.Spot.Streamed
public import Spot.OneStageZkStreamKept

/-! # Proof that v2's streamed check decides what its check decides on the blobs fed (`Properties.Spot.Streamed`)

`streamIsCheck` is `Core.StreamProtocol.run_eq` at v2's stream protocol, whose finish on a feed in the stream's order is
`checkBlobs` on the blobs fed (`Spot.OneStageZk.Streamed.streamSound`).

`streamIsCheck_honest` is its honest lemma: at parameters of the per-member form and evidence of the one statement's
form (`streamedParams`, `streamedEvidence`), whose stream plans no blob, the run on no blob is not the stream's own
refusal of a feed: it is `checkBlobs`' answer on no blob. `streamIsCheck_nontrivial` is its nontrivial lemma: there
`checkBlobs` accepts nothing, the two being of two forms, so a run that accepts every feed is neither a rejection nor
`checkBlobs`' answer.
-/

@[expose] public section

open Spot

namespace Properties.Spot.SecurityProofs

/-- **v2's streamed check refuses, or decides what `checkBlobs` decides on the blobs fed** (`StreamIsCheck`). -/
theorem streamIsCheck : StreamIsCheck := fun p c e bs =>
  Core.StreamProtocol.run_eq OneStageZk.Streamed.protocol OneStageZk.Streamed.streamSound p c e bs

/-- A stream whose plan names no blob, run on no blob, is its finish on its first state. -/
private theorem run_nil {Out : Type} (s : Core.Stream Out) (refuse : String → Out) (hp : s.plan = .ok #[]) :
    s.run refuse [] = s.finish (s.steps []) := by
  unfold Core.Stream.run
  rw [hp]
  first
    | rfl
    | simp [Core.Stream.close, Core.Stream.start, Core.Stream.steps]

/-- The witnesses' stream is the one that keeps every blob, on a plan that names none. -/
private theorem streamed_keepAll :
    OneStageZk.Streamed.stream streamedParams streamedCommitment streamedEvidence =
      OneStageZk.Streamed.keepAll streamedParams streamedCommitment streamedEvidence (pure #[]) := rfl

/-- At the witnesses' parameters and evidence, which are of two forms, `checkBlobs` accepts no blobs. -/
private theorem streamed_not_accept (bs : Core.Blobs) (pub : OneStage.Public) (prof : Core.Integrity) :
    OneStageZk.checkBlobs streamedParams streamedCommitment streamedEvidence bs ≠ .accept pub prof := by
  intro h
  unfold OneStageZk.checkBlobs Core.atCoins at h
  split at h
  · rcases OneStageZk.checkBlobsAt_accept h with ⟨hm, -, -⟩ | ⟨_, sts, -, hs, -⟩
    · exact nomatch (show some (#[] : Array OneStageZk.Member) = none from hm)
    · exact nomatch (show (none : Option (Array OneStageZk.StatementFiles)) = some sts from hs)
  · cases h

/-- **`StreamIsCheck`'s honest lemma**: at the witnesses' parameters and evidence, the streamed run on no blob is
`checkBlobs`' answer on no blob. -/
theorem streamIsCheck_honest :
    (OneStageZk.Streamed.stream streamedParams streamedCommitment streamedEvidence).run .reject [] =
      OneStageZk.checkBlobs streamedParams streamedCommitment streamedEvidence
        (Core.Blobs.ofArray ([] : List ByteArray).toArray) := by
  have hp : (OneStageZk.Streamed.keepAll streamedParams streamedCommitment streamedEvidence (pure #[])).plan =
      .ok #[] := rfl
  rw [streamed_keepAll, run_nil _ _ hp]
  exact OneStageZk.Streamed.keepAll_sound _ _ _ _ []

/-- **`StreamIsCheck`'s nontrivial lemma**: at the witnesses' parameters and evidence, a run that accepts every feed is
neither a rejection nor `checkBlobs`' answer on the blobs fed, so it does not have `StreamIsCheck`'s conclusion. -/
theorem streamIsCheck_nontrivial :
    ¬ ((∃ r, acceptAllRun [] = .reject r) ∨
        acceptAllRun [] =
          OneStageZk.checkBlobs streamedParams streamedCommitment streamedEvidence
            (Core.Blobs.ofArray ([] : List ByteArray).toArray)) := by
  rintro (⟨r, h⟩ | h)
  · first
      | cases h
      | simp [acceptAllRun] at h
  · exact streamed_not_accept _ _ _ h.symm

end Properties.Spot.SecurityProofs
