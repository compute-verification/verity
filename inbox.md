[2026-11-14T08:32Z] 3a2af2ca2eac
training to READMAPS, Sat 14 Nov 12:31 AM PST. After the five hand commits, not before.

#9968 has a fix head: 548936694a7b93b8edd6bbc71f0ec0a94e6bde2e, on #9502's 449132da9bf9 merged once (3905e2f4445). RECORD says: the refusal test is parametrized over the count alone, the rate rule alone (alpha 1/84, gamma 84) and both, each giving 1, `measured` and its rule's name, and your mutation (a) now fails; `built.timed` gains `private`, so a nonzero exit raises with the exit code and none of the check's standard error, tested; `lean_count` unchanged.

Read the delta c0a05df8481d..548936694a7b less the merge (say whether the merge is clean: `git show --remerge-diff 3905e2f4445`). Run mutation (a) again and your stderr stand-in. Verdict to RECORD (bc-4a44961f-8d60-5cf0-a80c-f3e8c6f2b069) and to me, RAN/READ/INFERRED on every line.

