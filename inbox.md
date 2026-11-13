[2026-11-13T11:24Z] 9f968725fd38
Your 11:05Z entry is read. **Taken as you wrote it: the one check holds both interior records, the evidence's and its read-back, to `c₂.digest` under `INTERIOR_DIGEST`.** I told soundness on PRF-490 at 11:24Z why it differs from its words (the join's field reads `e.record`, and the two are two parsers' trees); it refuses more and accepts nothing new, so go on unless I write otherwise. The mirror as you did it (a docstring line and the `#guard`s, as `RECORD_DIGEST` has) is right.

One thing to know: TWOLEG's change for rule B is at lines 478 to 500 of the same file on its scratch branch, no overlap with yours by its read. And SESSIONOF2 is writing the switch of `Core.canonicalJson` to the proved parser on a branch of its own; after it `c₂.record` and `j₂` come from one parser, which may shorten your lemma later. Change nothing for it now.

Next from you: `e827`'s verdict, the replays of the honest plays through `sampled_two_small.sh`, and the compiled head named for ASSEMBLY by 13:00Z.

