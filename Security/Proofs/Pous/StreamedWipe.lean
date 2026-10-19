import Proofs.Pous.StreamedWipe.Count
import Proofs.Pous.StreamedWipe.Bound
import Proofs.Pous.StreamedWipe.Deployed

/-!
# The streamed-label wipe

* `StreamedWipe/Count.lean`: the counts, sampling (`PousStreamedWipe.sampling_count`) and compression
  (`PousStreamedWipe.compression_count`), and the winning runs (`PousStreamedWipe.win_count`).
* `StreamedWipe/Bound.lean`: the ideal bound (`PousStreamedWipe.win_ideal_le`) and `StreamedWipe`, with the PRF's
  advantage (`PousStreamedWipe.streamed_wipe`).
* `StreamedWipe/Deployed.lean`: `StreamedWipeMeets2600GB`, at 2.6 TB and 74,000 challenges
  (`PousStreamedWipe.streamed_wipe_meets_2600GB`).

None of them uses a named assumption.
-/
