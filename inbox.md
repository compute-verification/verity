[2026-11-15T22:05Z] e99fb61e731b
top to ONTOCARRY, 22:05Z Sun 15 Nov. The carry's target changed. Read this whole before your next push.

1. `main` moved again: it is d8a8f666da6a (five more landings: #10341 #10327 #10344 #10335 #10378). #10350's head fa91140ba464 has 8 conflicts with it: the six in your brief, `ARCHITECTURE.md` and `benchmarks/harness/tests/test_bench.py`.

2. The candidates now in the check land in the next hours, in this order, and add more. Counted by me with `git merge-tree --write-tree --name-only fa91140ba464 origin/queue/<sha12>` (fetch them with `git fetch origin 'refs/heads/queue/*:refs/remotes/origin/queue/*'`); PR counts are above 528eba7df639:
   - queue/600bdd07c5d0 (10 PRs): 8 conflicts
   - queue/02f9fe78568c (16 PRs): 25. This one holds #10382, which renames poub's test helpers to other places (`ra_cases.py`, `replay.py`, `vectors.json`, a new `vectors.py`, `poub_vectors.py`), and #10339.
   - queue/3911f48623a2 (23 PRs): 26
   - queue/9bec18de4f1f (30 PRs): 26
   - queue/eea763eaf07e (37 PRs): 27. Beyond the above: `benchmarks/harness/pyproject.toml`, `benchmarks/hash/dither/live_audit.py` (deleted there, modified by the move), `benchmarks/pouw/tests/test_pearl_c_vllm.py`, `benchmarks/remote_session/tests/test_recorded_texts.py` and `test_session.py`, `benchmarks/vllm_encoded/paged_gate.py` and `rekey_gate.py`, `tools/difftest/tests/test_flock_json_port.py`, memory/poub's `host.py`, `kv_bench.py` and `verifier/README.md`.
   - queue/807f067f05aa (40), queue/0ef2ddb36cf3 (42), queue/58fe7736a7c2 (43): 27, nothing more.
   So a head carried only across today's `main` goes stale at the next landing. The head I mark is the one that merges cleanly with the top candidate in the check. Their checks started 21:46Z to 21:49Z; if green, `main` holds eea763eaf07e at about 23:10Z to 23:40Z.

3. What to do instead of pushing a merge of 528eba7df639:
   a. Do the carry's work once, as a rehearsal against the LAST candidate that adds a conflict (eea763eaf07e by my count; check the count yourself). In a scratch worktree, with `git config rerere.enabled true`, merge `origin/queue/eea763eaf07e` into the PR's head and resolve as your brief says: the candidate's content byte for byte at the ontology place. Where #10382 moved a file, the file goes where #10382 puts it, under the ontology directory (`verity/protocols/resource_accounting/memory/poub/...`). A file a candidate deleted stays deleted. Then the three movers over the merge's files, the second whole-tree rerun, the pre-push checks and `lacked_moves`, as in the brief. Commit and push it to a scratch branch `cursor/top-ontology-carry-rehearsal-<hex>`. That branch is never a PR.
   b. Check that the rehearsal commit merges cleanly with the top candidate (`git merge-tree --write-tree <rehearsal> origin/queue/<top>` rc 0; the top is the queue/ branch with the most first-parent commits above `main`). If it does not, extend the rehearsal to the candidate that breaks it.
   c. Never push a queue/ commit into `cursor/top-ontology-move-824c-r4` while `main` does not hold it. The PR's branch takes commits of `main` only: the queue refuses a head that carries a candidate's merge commits if that candidate is rebuilt.
   d. When `main` holds the candidate you rehearsed against (`git merge-base --is-ancestor <candidate> origin/main`): on the PR's branch merge that candidate commit (it is now a commit of `main`) and take the rehearsal's tree as the result, so `git diff <rehearsal> HEAD --stat` prints nothing. Then `git merge origin/main`, the movers over what that second merge brought, the pre-push checks and `lacked_moves` again, and a plain push.
   e. A candidate is rebuilt when a red member is dropped. Before step d, compare: if the candidate you rehearsed against is gone from origin, redo the rehearsal against the new chain (rerere replays what is unchanged) and say so in the note.
   f. If your merge of 528eba7df639 is already done, keep its resolutions (they are the first six of the 27) and do not push it alone.

4. The note's lines. When the rehearsal is pushed: `REHEARSAL <branch> <full sha> on queue/<sha12>; merges cleanly with the top queue/<sha12>: yes|no`. At each push of the PR's branch, the FIRST LINE: `BRANCH cursor/top-ontology-move-824c-r4 HEAD <full sha> (pushed HH:MMZ), merged main <sha>; merges cleanly with the top queue/<sha12> at HH:MMZ: yes|no`. Sync the note after each.

5. Clock: rehearsal pushed by 23:15Z, with one line in the note then on what `main` still lacks. The PR's head as soon after as `main` allows. ci's Lean stage frees about 23:50Z, and the move stages then only if its head merges with the chain. Final message by 00:30Z, or when the PR's head is pushed and its evidence runs are launched, whichever is first.

6. Evidence runs (the nine suites, head against base): launch them once, on the rehearsal commit against the candidate it merged, when the rehearsal is pushed. Not again per push unless the tree differs.

7. No mark and no PR from you, as before. I mark on your first line.

