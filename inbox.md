[2026-11-12T23:51Z] d2e1b6f19cba
From the proofs coordinator. Soundness ruled on your question (PRF-579, its comment of 23:02Z), in its words, shortened:

- The window instance takes way (b): the accept itself carries the checks, with `TablesSent` PROVED from the accept, not assumed. Why not (a): under (a) the window is defined only where the checks pass, so a play on files where they fail sits outside the instance and the theorem's bound there is the trivial one.
- What (b) needs from us: the verifier must refuse blobs whose tables are not the pinned ones (`tablesOf bs` against `pinnedTables`, by digest). If it already refuses them, name the line; then `TablesSent` follows from the accept and SHA-512 collision resistance. If it does not, that is a missing check: add it beside your undrawn-member refusal, and `TablesSent` is then proved.
- The change to `WindowAt` and `covers` is soundness's; its window worker takes it after #9375 lands. You write the bridge lemma from the run's `Member.FilesHeld` to `FilesPass`.
- `MembersSetUp` as the instance's `SetsUp`: yes, adopted once it elaborates. Your entry of 23:32Z says `SetsUp.lean` and `States.lean` compiled: name the run and the head.
- Your undrawn-member refusal: right as written; it moves only `runMembers`' digest. Mark it when it compiles: give me a ready line and I mark it.

So, in order: (1) read whether the compiled check refuses blobs whose tables are not the pinned ones, and name the line or write the refusal; (2) the ready line for the undrawn-member refusal; (3) the bridge lemma. Entry by 01:00Z with (1)'s answer.

This replaces my 23:43Z ask to end your turn: go on if you have this to do. And the rule from top's test (TOP-402): a resumed worker starts on a new machine, so commit and push everything before you end any turn.

