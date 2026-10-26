import Proofs.Pous.StreamedWipe.Count
import Proofs.Pous.StreamedWipe.Bound
import Proofs.Pous.StreamedWipe.Deployed
import Proofs.Pous.StreamedWipe.Family
import Proofs.Pous.StreamedWipe.Honest

/-!
# The streamed-label wipe

* `StreamedWipe/Count.lean`: the counts, sampling (`PousStreamedWipe.sampling_count`) and compression
  (`PousStreamedWipe.compression_count`), and the winning runs (`PousStreamedWipe.win_count`).
* `StreamedWipe/Bound.lean`: the ideal bound (`PousStreamedWipe.win_ideal_le`) and `StreamedWipe`, with the PRF's
  advantage (`PousStreamedWipe.streamed_wipe`).
* `StreamedWipe/Deployed.lean`: `StreamedWipeMeets2600GB`, at 2.6 TB and 74,000 challenges
  (`PousStreamedWipe.streamed_wipe_meets_2600GB`).
* `StreamedWipe/Family.lean`: `StreamedWipeMeetsFamily`, `⌈M/19⌉ + E` bits at `2^-128` with 2152 reads for every fill
  of a segment or more (`PousStreamedWipe.streamed_wipe_meets_family`), and its instances at 96 GiB and 141 GB
  (`StreamedWipeMeets96GiB`, `StreamedWipeMeets141GB`).
* `StreamedWipe/Honest.lean`: the honest server, which keeps every label, is accepted on every run
  (`PousStreamedWipe.honest_accepts`).

None of them uses a named assumption.
-/
