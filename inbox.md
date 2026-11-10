[2026-11-10T16:17Z] 578a01fd126b
TIMEPAIR, Tue 10 Nov 8:20 AM PST. **Window holds.** The rehearsal across the two hosts worked whole [run: serve r20261110-160259-c494 on cpu-10, observe r20261110-160306-012b on cpu-8, no pool, START 8:05:00, 300 buckets a case]:
- All 9 cases left their record on both sides, none with a problem; the guard passes (planted 1.000, ignored 0.000); `pair_score.py` scored them from the two fetched runs.
- Route and address: the observer on cpu-8 reached cpu-10 on 47811 in every case. Both clocks follow PHC0, RMS offset under 2 µs [run: `chronyc tracking` in each host.txt].
- Its rows are a rehearsal's (150 scored a row), no result: write as is 0.037 [0.003, 0.104].
- One fault, fixed: `window.sh` ended rc 2 on its last `ls` though every case ran. It now exits 0 only when every case left its record; a 8-minute second rehearsal checks that now (START 8:22).
- How cpu-10 is A: its address goes to both sides as a sent file; no address is in `window.sh`, host.txt, Linear or the notes.
- Next: launch both about 9:45 with START 10:00:00 AM PST, then 10:40 with START 10:55:00 (`--timeout 50m`, ends 11:26).

[2026-11-10T16:23Z] 982cb19f5a65
READMEIFACE, NET-307, Tue 10 Nov 8:22 AM PST. Done for this turn. No PR. The branch is pushed, the note synced, one comment added on NET-307.
1. Head `94410a90516e` on `cursor/net-307-readme-interface-5cb1`: one more commit on f70287cb54, three READMEs, 13 lines in and 12 out. The prover's and datacenter's are untouched; `python -m datacenter.demo` and `## Overview` stay. [run: `git ls-remote --heads origin cursor/net-307-readme-interface-5cb1`; `git show --stat HEAD`]
2. Item 9: the policy's README holds 55 sites, 55 allowed, as on the staging (it was 63). Its section names `NciCheck` and `nci-check` once each and no difftest. Two repeats left the older text: "its check, `NciCheck`, decides" and `(NciCheck.Protocol)` after the protocol's id. The line-112 fix removed no site. [run: `tools/net307/ratchet_sites.py`] The gate [run: `ratchet_gate.py 41d5514d76806351783e0d91a3ef5f94ce63658f HEAD`, exit 0]: `passed: no lane-text site is new in the 5 file(s) it changes; no retired-term site is new in the 5 file(s) it changes`.
3. One thing you did not rule on: the gate also failed f70287cb54 on the verifier's README, 29 -> 30, the module `warden` in the section. [run: the gate on f70287cb54, exit 1] I applied your rule there: the section keeps the name, and "(`warden`)" left one older sentence; the module's own bullet still has it. It is 29 again. Say if you want it otherwise.
4. Your difftest bullet is changed in two places. No test runs a difftest executable: the Lean check runs each one's main from the lock's `runs`, and the tests hold that data to its generator, so the bullet says that. [read: `tools/verity/lean/check.py`, `tests`; run: `git grep -nE 'subprocess|lake'` over those tests, none] And it has no "six", since the issue bars counts. My first commit's device README said the tests "run the executables": that was my error, corrected here.
5. Item 11: both lines fixed, each path checked first. [run: `git ls-tree 41d5514d76`] No other dead path in the five: 169 mentions, 0 dead [run: `five_checks.py`]; the 26 tokens a stricter scan flags are protocol ids or file names beside their directory [run: `strict_paths.py`; read each].
6. Item 2's checks on the new head: the three tests, 30 passed, no finding in the five; all five under the cap; `tests/test_repository.py` and the retired-names lint, 20 passed, naming none of the five. [run: the `pytest` lines in the note]
7. Newer staging: `cursor/top-layout-staged-eb0b` at d833bd4b05, 7:23 AM PST. Both commits cherry-pick onto it without conflict, and the five files come out identical to my head's. [run: `git cherry-pick -n` in a scratch worktree, removed since; nothing pushed] Not done: no executable built or run, and `main` not looked at. Note: `20261110T1600Z-report-net307-readme-interface`.

[2026-11-10T16:28Z] d937c5151074
TIMEPAIR, Tue 10 Nov 8:28 AM PST. **The second rehearsal passes: both sides rc 0, 9 of 9 records each, scored** [run: serve r20261110-161722-417d on cpu-10, observe r20261110-161729-a30f on cpu-8, no pool]. The window holds, as sent.
- Correction to my last message: it was sent at 8:17 AM PST, not 8:20, and this rehearsal's START was 8:19:00, not 8:22. I wrote those two times ahead of the clock.
- Your 8:10 message on the layout is read: I cross the branch myself after the second run is scored (TOP-394's two commands, `--dry-run --json`, then `--push-to` a new branch), tell you its name, and run the suites again there.
- The premise restated as a rate is written in my note, section "The premise as a rate" (note `20261110T1554Z-state-net296-timepair`, synced).

[2026-11-10T16:38Z] b401879c09d9
SCANLEAN, NET-304, 8:38 AM PST. A red that is #8539's own, for proofs; and what I did on your 8:15 message.

- **`verity-pouw-benchmarks` is red at #8539's own head `b3a5cc07e5`, with none of my commits: "1228 passed, 9 failed", all nine in `benchmarks/pouw/tests/test_served_zk.py`** [run `r20261110-162209-e040`, vy-nebius-cpu-4, tree `b3a5cc07e5`]. The same nine fail at my `8791251ea3` [run `r20261110-154950-ebff`]. Locally one of them ends `Refused: lean-python-disagree: registration` [run on my VM]. Likely cause [read, not run]: `benchmarks/pouw/served_zk/of_record.py` L169-190 holds Lean's registration piece equal to Python's `{"codes": …}` (`_same`), and since `8425afa44e` (in #8539, not on `main`) the piece also returns `fanout` under a template query. My branch touches nothing under `benchmarks/pouw` or `verity/core/protocols/spot`.
- So my suites run cannot be all green while that stands: at `8791251ea3` it is 50 of 51, the two network suites passing with their slow tier (85 and 522 passed).
- Added one negative on the real R8 path (test only, `33f2261026`): a root sharing two run Inputs, registered with both tied; Python's registration check lets both through, its verdict names both, the answer `window_answer` builds from it is refused by the mirror's `tied_input_wired`. It does not go through Lean's answer (PRF-504).
- Two docstrings now say the `need` reads the names the answer carries and that nothing checks the answer carries them all (`8e64478637`, not pushed yet: I push with the lock); no proofs' file touched.
- Lock run 1 `r20261110-154914-51ef` still building the Proofs tree (48 min in). I merge `b3a5cc07e5` when it ends. Inbox read; #8539 holds no `Layout-move:` commit.

