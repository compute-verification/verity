[2026-11-13T20:24Z] 252d0d4ffd7c
SCANLEAN to the lead, Fri 13 Nov 12:26 PM PST. **One thing to know before you mark #9664: the carry left four hand-written sections of `security/Properties/lean-audit.json` in the order from before the moves, so the pull request's diff of that lock is 1,100 lines where it was 114, and landing it puts `main`'s lock back in that order. The data is right. I change nothing.**

- **What differs from `main`'s lock at `b5a49641edb6`** [ran: both locks loaded and compared section by section]: as data, only this change's own entries (3 guarantees, 1 witnesses entry, 1 `shape.properties` entry, 2 new `reads` groups and 2 changed). As text, the order of `roots` (15 of 98 positions), `meaning` (10 of 135), `witnesses` (65 of 70 keys) and `layers` (its keys).
- **Where it comes from** [ran]: `main` reordered those sections at the moves (its order at `5eb1231b2558` is not its order at `1e4d5c993c08`), and the carried branch keeps the branch's order from before (`2bc10c6a6d36` and the carried head agree on every common key). It came with the carry, at `a9afd2cbe5e5`, not with the records commit.
- **On landing** [ran: a trial merge onto `main` at `f479876a8758`, nothing written to a branch]: clean, 21 files, and the merged lock is in the branch's order. I do not know whether the queue's merge uses the lock's driver where mine did not.
- **The records run passes with this order**, so the check takes either. I did not look at whether any tool reads the order of `layers` or `roots` as meaning something.
- **#8708's `e6cf178b538b` has the same** (1,092 lines; the same four sections), and I expect every branch the tool carried has [argued, not looked]. It is the tool's, so ci's to say whether it matters.

[2026-11-13T20:26Z] ad054e38de7d
TERMGB to network-accounting (lead), Fri 13 Nov 12:26 PM PST. **The writer's pull request is open, unmarked: #9811, head `03d7028d3234317e3e7dc5a09305520254566f50`, base `main`, linked to NET-403.**

- Branch `cursor/na-registrations-writer-b4e6`. It holds #9501's carried head `b2e51a3184bb`; against that head it is its own 8 files, +819 −83 [ran].
- **Its run at that head** [ran, `r20261113-180539-7169`, `suites.py --changed d119240998be --quick`]: exit 1. 61 suites, 60 passed (1 cached), 1 failed: `verity-bench`, 619 passed and 8 failed, the eight of #9501's body, the same by name as in #9501's run of record. `verity-network-traces` 754 passed: 746 at #9501's head and this branch's eight.
- The pins step at that head: exit 0, no pin stale [ran, `r20261113-180619-ff68`].
- Its hand merges, as sent: the base `92b5727aca40` (one file, `benchmarks/network_traces/connection_rows_run.py`, one place: this branch's two constants kept, the comment in the base's words) and #9501's carried head (one file, `benchmarks/network_traces/stop_verdicts.py`, two places, each side's lines kept).
- Still open from my last line: #9329 is red on its own test at `10b60872b98f`; its mend is committed here and running on a node, not pushed.

[2026-11-13T20:29Z] f882b4c17e76
ONSITEMAIN, Fri 13 Nov 12:29 PM PST. Your 11:26 AM line is read. A's and B's second runs are in, and the test that failed once is found.

- **#9670 (A)** `de84698df4801d5bbc407119a08abaed16a4a0a9` [rev-parse, origin]: the two suites alone, no `TMPDIR` set by the job, `r20261113-190602-3d15`, exit 0: 2 suites, 2 passed; `verity-vllm` 17,152 passed, `research` 5,009 passed [ran, pool]. All thirteen pass. With the first run's 59 (`r20261113-170813-b455`), every suite the change touches has passed at this head. The body is up with both runs.
- **#9677 (B)** `ae22778f58508840f3b0ac6beeecc685d37c362c`: first run `r20261113-180028-02a4` ended exit 1, 61 suites, 59 passed, the same thirteen; the two suites alone `r20261113-194001-2660`, exit 0, 2 of 2 [ran, pool]. Body up.
- **#9678 (L)** and **#9679 (I1)**: first runs ended exit 1 on the same thirteen and nothing else (`r20261113-181431-fabd`, `r20261113-183712-7c78`: 61 suites, 59 passed each; pins and records changed no path). Their two-suite runs are launched, once each: `r20261113-201645-ef6e` (L), `r20261113-202850-e025` (I1). H's and NET-411's first runs are still going; I launch theirs as each ends.
- **#9319** `b82b94e92007539b86cf625b86ff11f76e963d83`: its runs are in (`r20261113-175646-88ac` 6 of 6; `r20261113-183812-1c92` 4 of 4; the toy `r20261113-182751-cab5` exit 0). Its body has the heading for proofs (`benchmarks/one_stage/a5.py`: `audit`'s `session` +7 −3, `session_ended` +12, `before` on `NegativeWorkers` +10 −4) and the eight callers by owner with what each would take. Neither of the two suites is in its changed set, so it has no second run.
- **The test that failed once: the cause is found and reproduced** [ran]. `served_verdicts.sh` starts the certifier's chain in the background beside V3c's warm and audit (`v3c_proved`), so where the chain's line falls among theirs is the scheduler's; the test compared two runs' lines whole. On my VM, 160 runs of the script as the test runs it, four abreast: 159 wrote `CERTIFIER-CHAIN` before `V3C-WARM`, one after; steps and limits the same in all. Held by hand until V3c's period line, the chain's line falls last every time and the old comparison fails on it, in the lines alone.
  - The fix, the test file only (+29 −6): the chain's own step, limit and line are left out of the comparison, each run holding exactly one of each; the rest is compared in order as before; and a fourth run with the chain held is added, so the order that failed by chance is in the test every time. It waits on the line's presence, no sleep and no clock.
  - Branch `cursor/na-kept-cache-test-order-84f9` at `21196b7d766198695f80f89536f89f5d0597cb6a`, off `main` at `f479876a8758`, pushed, **no pull request yet**: I open it after NET-411, as you said. Its suites [ran, pool]: `r20261113-194801-d1b4`, exit 0, 4 of 4.
  - **Question:** which issue does its `Refs` line name? I would write `Refs NET-331`; `research queue open --issue` links the pull request to that issue, and I post nothing on NET-331 otherwise. Say if it is another.

Least sure: that no other test compares two runs' lines whole; I searched the benchmarks' tests by pattern and found none.

[2026-11-13T20:33Z] 7ddf6be73247
POLICYROWS to network-accounting (lead), Fri 13 Nov 12:33 PM PST. **#9547's second run is green, and its body is set.** The head is still `dd6555ae618a` [ran: `git ls-remote`]; I pushed nothing to it.

- **Run `r20261113-190212-7d81`, at `dd6555ae618a`: exit 0** [ran; I read its outputs]. `adapters/vllm`: 17,154 passed, 0 failed, 479 skipped. `tools/research`: 5,009 passed, 0 failed, 3 skipped. No kept pass reused; the pool's own temp directory; `git status` empty after it.
- **#9547's verdict:** nothing rewritten, by the pins or by the records run (outcome `pass`); 58 of 60 suites passed in the first run, `r20261113-165816-4f26`; the two others passed in the second.
- **The body** [ran: `research queue edit`, read back, equal] carries the carried head, the tool's word, both runs, the 13 tests and the cause, and what the carry changed file by file. The runs before the carry stay, said to be before it, with today's paths. No path or name the moves retired is left in it.
- Still running: `r20261113-201050-6104`, #9644's two suites, launched 12:11 PM PST. My return comes when it has ended and #9644's body is set.

