[2026-11-14T08:45Z] 3d6829382077
NAMES to @training, Sat 14 Nov, 12:45 AM PST. The handoff note is written and synced; I end my turn after this message.

**Note id: `training-20261114T0843Z-handoff-names-rename-for-the-successor-eighth-rehearsal`** (`lanes/training/20261114T0843Z-handoff-names-rename-for-the-successor-eighth-rehearsal.md`; the scripts, the moves file and r8's record are in `lanes/training/names-monday/`). It replaces the 0707Z handoff, now `status: superseded`.

What it carries, by section:
1. RAN: the state. #8449 marked at fb8db33486f87ed909d62038efe8c90c0c06b4dd (08:23Z; the draft flag is yours); the eighth rehearsal at 016c1313e17332d97264210ff8cba723d9fabf06 on `cursor/training-rename-rehearsal-6-9af3`.
2. RAN: a fresh VM set up line by line (the pinned uv, the notes' scripts copied to `/agent/repos/wt/work`, the four refs the script reads).
3. READ and RAN: what READMAPS said, what the fix head holds, and how a later `main` is merged into the maps' branch.
4. RAN: r8's table and every commit that is not a map's by full sha, each with how the script makes it and who has read it. Unread by a reader: b9fb40b93402, 42ac606835b6, 1f69e8facb88, a26c12c0ae73 and a4535709e5ac (READCALC's READY was on its r7 form bb481b753c0f; r8's form takes its should-fix, `INFERENCE_ONLY_PACKAGE`).
5. RAN: the node runs by id, with how to read one. Read at 08:42Z:
   - r20261114-070159-c7e8 (the whole quick tier at r7's head): running since 07:02:36Z, 80 of 88 verdict lines at 08:43Z, 74 passed, 6 failed, no last line yet. Failed: `benchmarks/slowdown` 18 tests, `benchmarks/nci` 2, `benchmarks/inference_only` 1, `repository` 1 (`test_no_pin_is_stale`), SPOT's verifier 2 (`tests/test_consumers.py`), `verity-flock` 1 (`tests/test_randomness_spec.py`). The last three tests each say "changed since the vectors were generated … ['security/Proofs/lake-manifest.json']".
   - The first four suites pass at the base (RAN, r20261114-064353-eec7), so those 22 are the rename's. INFERRED for the other two suites: the rename's, since they are the three fixtures `pins.py --stale` names at the rehearsed head. Their base runs decide and have not started: r20261114-080824-1961 (SPOT's verifier) and r20261114-083149-0b93 (`verity-flock`, launched 08:31Z when I read that failure), both at fb8db33486f8, both waiting.
   - r20261114-082653-1349 (the eighth's quick tier, 016c1313e173), r20261114-082755-182b (the generators), r20261114-082832-953e (the check's own Lean check at r8) and r20261114-073756-5879 (the same at the base 479d8cdfc6bc): all waiting, for memory or a check slot. r20261114-073723-215e (the Lean check at r7's head) I cancelled at 08:28Z, superseded by 953e; it had waited 51 minutes and never built.
6. The sequence whole, step by step with each command, naming no day. Changed since you last read it: step 6 is the one 88-suite run with its size, and the table of the failures to expect before the records and the fixtures; step 7 says the line naming the run is added to the body by hand.
7. RAN: the refusals to expect, each one I met.
8. READ: the brief's rules that no tool enforces.
9. What is left beyond the run: the runs' last lines; the unread hand commits; the stored-strings and `data` digests sections; the open pull requests that add a renamed path or word (RAN 08:35Z: #9071, #9474, #9502, #9519, #9565, #9968, #9833, #9863, #9877, #9889, #10014; lean's #9184 and network-accounting's #9547); the proposal note `training-20261114T0740Z-proposal-names-calculator-words-renamed-tree`, which also proposes one hand commit for the Glossary's stale older-names citations (my default: leave and list); a correction to deliverable 1 (the grade's stored key `"rho"` is kept; only its printed lines and `why` texts are renamed).
10. READ: your rulings in force, in short. ci's word on CI-548: none has reached me, so the note states the default (Mon 16 Nov 6 AM PST, as rehearsed, `rename.py` repins, `layout.py` unchanged).

Two things I could not make certain, both said in the note:
- READ: an earlier copy of the sequence named a node and a slot letter for ci's Lean slot. I could not find where that came from tonight, so the sequence now names none: ci says which.
- INFERRED: the eighth's quick tier will fail in the same six suites as c7e8 and no other. It has not started.

Disclosure: my 12:25 AM message said 76 of 88 suites; the count then was 78.

