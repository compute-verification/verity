[2026-11-14T15:46Z] 7ab16ad68969
RECORD, Sat 14 Nov 7:46 AM PST. **#10012 is marked at 5be997485a91e22dbaf9dfd4fbd446ec1a95aa82, and #9968's cure head is 6ea6039e29d3ac07314bab3ebde7a8f311f3f010, with its one quick run r20261114-154622-0004 launched.**
- #10012: RAN. r20261114-082712-ffac is done, rc 0, tree_sha 5be997485a91, and its last line reads "3 suites: 3 passed (0 cached), 0 failed". The queue's preflight onto main a116c5614312 refused nothing. After it lands, I launch the row's command once from main and send the id.
- #9968: one commit, 6ea6039e29d, holds both cures in `benchmarks/inference_only/pyproject.toml` and `tools/verity/check/tests/test_after_lean.py`, beyond the reader's words. RAN on this VM: `test_after_lean.py` 3 passed, `test_served_record.py` 24 passed. The body says both changes. The run is `--changed 6953e38e55a --quick` at 6ea6039e29d.

[2026-11-14T15:47Z] 372b5be7d977
NAMES2, Sat 14 Nov 7:46 AM PST. Resumed on a fresh VM; set up again; nothing pushed.

**Recommendation: rebuild the chain on today's `main` (case A) on a new branch, not one merge: `main` changed 122 files under `security/`, both locks included, so run 1's records match no tree that takes today's `main`, and run 1 repeats either way. I start the local rebuild now, which pushes nothing; I push it and open a new draft only on your word, since #10102's branch cannot take it without a force-push.**

(1) Run 1, r20261114-092815-f711: done rc 0, tree c0791fcafb2b.
- RAN, last line: `lean-records: OUTCOME {"outcome": "regenerated", ...}`. It regenerated `security` (record sha256 f94fc6ba…) and `security/Properties` (3dee7d2e…); `security/Proofs` passed. So run 2 is not needed.
- RAN: I fetched both records from the store. After mapping the chain's locks through `lean-moves.json`, every guarantee's name matches 1:1 (none only-old or only-new). Only hashes differ (`type_hash`, `unfolded_hash`, `digest`). Seven signatures differ, in whitespace alone (longer names rewrap). The tool classed all 109 and 24 lines as "the same statements, printed differently, rehashed, moved or lifted". No statement changed.

(2) What `main` at a116c5614312 does to the chain.
- RAN: a trial merge of `main` into 86227c4e conflicts in three files:
  - `benchmarks/harness/verity_bench/overhead_records.py`
  - `tools/research/src/research/store/moves.json`
  - `benchmarks/rho/tests/test_lean_names.py`, a test `main` added inside `benchmarks/rho/`. Git places it under `slowdown/`, but no map has run over its names. Anything else `main` added since f28bdad keeps the old names the same way.
- RAN: `main` changed 122 files under `security/` (Pouw, Hash, Claims…), both locks and the Properties lakefile.
- INFERRED: a merge would need the maps re-run as an uncertified catch-up commit, plus run 1 again. That path was never rehearsed. A rebuild is the rehearsed script with `BASE=` today's `main` and no `MAPS=`.
- Cost of the rebuild: about 30 min on the VM (the script, then step 3's reading). These repeat: run 1 (its records), run 3 and step 6. None of those last two had run yet. The rebuild's hand commits are the same, the README cure included. If `main` changed a line one of them edits, the script stops, and I redo that edit by hand.

(3) RAN: one move landed on `main` since f28bdad24636 that the chain lacks: `8469d5dcc16`, "Move by tools/move/registration_map.toml" (`verity/hardware/registration.py` to `verity/protocols/accounting/registration/verifier/registration.py`). The chain adds no line naming the old path and renames no file the move edits. A rebuild on today's `main` contains it, so no `restack.py` is needed.

(4) Next three steps, each with its run:
1. The rebuild: `rehearse.sh`, case A, on that hour's `main` as BASE. Branch `cursor/training-rename-2-a32b`, scratch `cursor/training-rename-moves-2-a32b`, then step 3's checks. VM only, no run. Push and a new draft PR on your word. #10102 is yours to close or retitle.
2. Records run 1 again on the new scratch branch (`--records security --update --moved`), one run. Last time: 4.5 h wall, mostly waiting for memory.
3. Commit the records, then run 3 (`pins.py --regen`, the three fixtures), then step 6's quick tier at the final head. One run each. Run 3 does not need the records. Question, recommended "yes": may run 3 go beside run 1 instead of after it, to save a few hours?

Other runs (RAN, last lines):
- 0b93 (Flock at the base): "1484 passed, 30 skipped", so `verity-flock`'s one failure is the rename's. 1961 (SPOT at the base): "916 passed", so SPOT's two are the rename's.
- 1349 (r8's quick tier): failed rc 1, "88 suites: 82 passed (1 cached), 6 failed". Exactly the table's six suites and 25 tests, the same tests as c7e8.
- 182b: "DATA: 15 same, 0 moved, 0 missing, of 15" (the fourth such run). 953e (the check's package at r8): "AUDIT: PASS", 1730 declarations, 17 modules, 0 guarantees, as 5879 at the base.
- **No failure outside the table, and none not on the base.**

Hour on `main` (INFERRED): about Sun 15 Nov 2 AM PST if the runs wait for memory as last night, by 11 PM Sat if they start at once. It waits on run 1's place on the cluster, the reader's READY on the rebuilt chain, and ci's Lean stage.

Events: none addressed to @training reached me this turn.

[2026-11-14T15:47Z] 5cb058020e7c
ROWS, Sat 14 Nov 7:46 AM PST

**#10069 is at 2a2b599263a9bacb7e8b41e0c02aec40bc0350f2, with both of READSTEP's SHOULD FIX done; its quick run r20261114-154556-ee6d is running, and I mark when it ends rc 0 at this head.**

- RAN: one commit, `test_library.py:111` now says "the four runs". It is the only change from f4c55b04966f.
- RAN: the body is restored with `research queue edit`. It has an `Issue: none: ...` line again, and the queue's `named_issue` on it returns `none: ...`. It also has a "Not checked" paragraph with the datacenter's-year selector NOTE, and points verification at the new head and run.
- RAN: the previous quick run r20261114-093256-a75b is done, rc 0, `tree_sha` f4c55b04966f. Its last line, read by me: "7 suites: 7 passed (0 cached), 0 failed".
- RAN (`research run-times`, the `run_s` key):
  - The circuit_properties timing run r20261114-084145-ff06 took 470 s, rc 0, at main 9a520e76c0, uv sync included.
  - The library run r20261114-084119-33f2 took 2.5 s, rc 0, and its stdout holds only `rc 0`. INFERRED: it traced nothing at that commit, so it gives no time for the library. The first run of the row's command from main will give one.
- READ, events: READSTEP's delta-read report of #10069 reached me through the inbox hook. I acted on none of it beyond your two SHOULD FIX.

