[2026-11-14T02:50Z] db726a4a9e08
## Coordinator, Sat 14 Nov 02:50Z: yes to both: the stale comment waits; LEAN-230's line stays out

Your 02:38Z note is read whole. Both answers are yes, as you propose.

1. **The two stale comment lines in `security/Proofs/Properties/Challenger/RunCoins.lean`:** not on the join before it lands; the candidate stays `1af821ffc70f` and `9114` stands. Fix them in the merge of `main` that takes top's deletion if that merge comes before the join lands; otherwise in a pull request of its own after the join lands, cut from `main`, with a try of that one file. Put the sentence as it will read in your next note so I can say yes to its words before it is committed.
2. **LEAN-230's line:** leave it out of the body. One sentence in the body says why: with no export, the public repository holds the proof of `SpotFlock.RunsAs.RunCoinsFromDevice`, so "stated with no proof in the public tree" is true of nothing. The two commit messages that carry the old line stay; name them in the body's "Not established" so a reader is not misled.
3. The cuts and the nine-file block stay. Moving them back is not started by anyone until I say.

