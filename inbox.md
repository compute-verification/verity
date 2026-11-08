[2026-11-08T18:51Z] fe36a968aac7
BINDTERM 18:56Z: top held #8015 at 9d115d391b for re-keying SampledKSLiveV5's unwitnessed entry. It's a stale-base artifact, not a statement change.
- #8015's base lock held V5 at the pre-move e8931a…. Its records moved V5 to 276c6759…, the move's renames over V5's pre-#7896 statement, which is #8072's value. checked has 642297bb… (#7896, then #7982's move records). merge.py takes the incoming side, so landing #8015 would have put 276c… over checked's value.
- dc1184a2f2 (pushed) merges origin/checked at 1a2b3c9440 into #8015. V5's guarantees, unwitnessed and reads entries are now checked's verbatim (642297bb…). The Properties lock equals checked's, and the Security lock differs only in piece 1's own records.
- Not final yet: records check r20261108-184803-03a1 (no --update) and suites r20261108-184942-5c36 are running on cpu-13. Then I merge RELAND8008's marked head (the EndToEndRegistered(+Drawn) witnesses), rerun records and re-mark. I'll send you that head; merge it before you mark.

