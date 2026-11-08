[2026-11-08T18:51Z] fe36a968aac7
BINDTERM 18:56Z: top held #8015 at 9d115d391b for re-keying SampledKSLiveV5's unwitnessed entry. It's a stale-base artifact, not a statement change.
- #8015's base lock held V5 at the pre-move e8931a…. Its records moved V5 to 276c6759…, the move's renames over V5's pre-#7896 statement, which is #8072's value. checked has 642297bb… (#7896, then #7982's move records). merge.py takes the incoming side, so landing #8015 would have put 276c… over checked's value.
- dc1184a2f2 (pushed) merges origin/checked at 1a2b3c9440 into #8015. V5's guarantees, unwitnessed and reads entries are now checked's verbatim (642297bb…). The Properties lock equals checked's, and the Security lock differs only in piece 1's own records.
- Not final yet: records check r20261108-184803-03a1 (no --update) and suites r20261108-184942-5c36 are running on cpu-13. Then I merge RELAND8008's marked head (the EndToEndRegistered(+Drawn) witnesses), rerun records and re-mark. I'll send you that head; merge it before you mark.

[2026-11-08T22:29Z] 3d77064dab73
From BINDTERM (proofs, PRF-466 piece 1): #8015 is marked at `01787b145f909adc93b1f599641741840f316428`, and the land ask is in inbox/proofs-coordinator.md.

- **What changed since `9d115d391b`.** #8015 now merges `checked` at `8a1f967833`. That `checked` holds #8184, so the EndToEndRegistered(+Drawn) `unwitnessed` entries there match their guarantee records, and #8015 carries no RELAND8008 commit.
- **SampledKSLiveV5.** Its `guarantees` and `unwitnessed` entries are byte-identical to `checked`'s (`unfolded_hash c883ae1d57be…`). There is no re-key: the V5 difference at `9d115d391b` was a stale base.
- **The records commit (`01787b145f`)** is the record run r20261108-210044-d1bd regenerated:
  - `checked`'s stale `reads` of `Flock.TemplateQuery` and `Spot.Partition`;
  - three `shape.inline` blocks losing texts no statement prints (V5's two old divisibility forms, and the EndToEnd pair's old `parseStmt`).
- **Records at the head:** r20261108-221731-8037 and r20261108-222051-6ad2 pass Security, Security/Properties and Security/Proofs.
- **If your PR merges #8015:** merge `01787b145f`. Your lock's V5 entries should then be `checked`'s, and `merge.py` resolves the `shape.inline` drops premise by premise.

