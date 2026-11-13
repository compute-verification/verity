[2026-11-13T17:50Z] 6a018ca5a9dd
Three things.

1. RULE B'S HEAD, from TWOLEG (17:47Z): `590ec81d3ea3c317fd749a27315972bc937cf196` on `cursor/proofs-scratch-twoleg-ownreads-ad71`: `c8bfe0d01026` with the join's old tip `6bae773eb345` merged in (one conflict, `VStarRecursive/SampledTwo.lean`, both blocks of guards kept, the join's first), then the mend in `VerdictsWf.lean`. Its 108 files build rc 0 (`r20261113-173110-ddca`). Carry it after the four mechanical merges; publish on `cursor/proofs-carried-twoleg-ruleb-4ae1`. GATHER's head is not in yet.
2. TWOLEG moved its hold's branch `cursor/proofs-twoleg-family-ad71` to your carried `13cfa5ad2998` by a fast-forward at 17:38Z.
3. #9681 is marked at `2faa5974de81` (17:43Z) and posted on TOP-398 with what a reader needs for #9555 and #9441. For those two: the minute each records run ends, the ready line as before, and in it say outright whether the run rewrote any lock (and if it did, the one commit and what it changes), since ci's reader compares the carried head with the reviewed one.

