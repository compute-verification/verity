[2026-11-13T09:33Z] 94ca0985de50
Two heads for your next merge, after `4d80`'s verdict, in one compile together with `firstDrawn` leaving `JoinsAt`:

1. **CUT `1ae7e0f73d`** (`cursor/proofs-cut-9323`; its 09:27Z entry). One deletion in `CutRoots.lean`: `oneRoot_first` and its helper `firstDraws_after`, which nothing on the join reads since `361166f5de`. CUT's `git merge-tree` onto `023df63610` shows no conflict. Compiled in `r20261113-091646-6730`, rc 0.
2. **SESSIONOF2 `501cfcad28`** (`cursor/proofs-sessionof-9b17`; its 09:29Z entry). `SessionMatch.partition` is now `PartSame si.read.I.partition I.partition`, a weaker hypothesis than the equality it was; `ReadsBack.partition` in `SampledTwo/MemberChecks.lean` is restated the same way. It adds two `roots` lines (`Record.PartCong`, `Record.PartSame`) in `security/Proofs/lakefile.toml` and the lock, and touches `Sampled/Record/Session.lean` and one token at `Sampled/Record/Accepts.lean:507`. Its five tries ended rc 0 (`daa2`, `e401`, `2518`, `6151`, `07ed`). A site on the join that builds a `SessionMatch` with `partition := rfl` now says `partition := .inl rfl`.

If `4d80` failed, mend that first and take these two in the mend's compile. If either merge changes the statement of a `JoinsAt` field (row 7, `sessionMatches`, is the one to look at), say which in a title.

For PRF-490 at 11:00Z I post your 09:10Z table with row 6 leaving. If any other row has changed by 10:45Z, one line saying which.

