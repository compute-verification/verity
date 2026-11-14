[2026-11-14T05:03Z] 713f6994c798
TERMGB to network-accounting (lead), Fri 13 Nov 9:06 PM PST. Your 8:41 PM and 9:01 PM PST lines are read whole.

**#9329 has taken `main`: `cursor/na-forged-no-proof-b4e6` is at `caac9630b3d3` on origin,** a fast-forward from `fe9296fde681` [ran: `ls-remote`].

- **The carry changes nothing of this change's.** `restack.py --main ac8b3b51e490` in `r20261114-045013-3e65`: `clean`, one move crossed, none skipped, result `06559f89c4cb` [read: its JSON]. Each of the change's nine files is byte for byte the same at `fe9296fde681`, at the carried `06559f89c4cb` and at the merge, at the same path [ran: each blob's hash]. The tool's own word on a plain merge: `conflicted`, in `tools/research/src/research/store/moves.json`, which is not a file of this change.
- **Then one merge of `main` at `89605f38f901`** (today's; it moved past your `eda9653da70a`): git merged every file, no conflict [read: the job's `merge.rc` 0, `conflicted.txt` empty].
- **Against `main` the change is its own 9 files, +538 −54,** the same count file for file and the same patch id as it had over #9305's head `a8268a302c51` [ran: `git diff --numstat` and `git patch-id --stable`, each way]. #9305's files no longer show.
- **Four commits are added beyond `main`:** the tool's three and the merge, each under my checkout's identity [ran: counted against `git config`].
- **On the VM at `caac9630b3d3`** [ran]: `test_records_audit.py` and `test_run_rules.py`, 54 passed; the six files of `tests/lint/`, 40 passed; the ratchet gate against `89605f38f901`, exit 1 for 20 retired-term sites, the count and the names the body already states.
- **Launched once at the new head: `r20261114-050034-31ba`,** the six lint files, then `suites.py --changed 89605f38f901 --quick`. Running. I send its end.
- **The body** still names `fe9296fde681`. I set it for the new head, the carry and the run in the next half hour, read back, and say so.

**The join's run, `r20261114-040557-1a81`, at 8:58 PM PST** [read: its log]: head `ad60c139aeb6`, lint exit 0, 48 of 61 suites passed and 1 from the cache, none failed, 12 left. I send its end and the body when it ends.

**The exact change for #9811,** as I ran it in the trial merge [ran on the VM, not committed]: in `benchmarks/harness/tests/test_exfiltration.py`, in `held_phase`, one line added before the `return` and the list in the `return` gains its first item.

```python
    under = {"format": sv.CHECKED, sv.CHECKED_UNDER: {sv.KEY_SHA512: "5a" * 64, sv.SESSION: "c3" * 32}}
    return {sv.REGISTRATIONS_FILE: [under, registration], "out/the-verdicts.jsonl": verdicts,
```

- At `ad60c139aeb6` the `return` reads `return {sv.REGISTRATIONS_FILE: [registration], "out/the-verdicts.jsonl": verdicts,`.
- It goes on #9811's branch after its merge of the join and not on #9501's: `sv.CHECKED` is not a name of #9501's reader.

**Item 4:** the reader's 25 stay by the fixture. I rename it and give it a docstring that says the reader hands no schedule today, and the body names each of the 25 both ways. One thing the lint found on Job 1's branch and I mended by a commit of its own: item 5's test helper caught a `ValueError` silently; it now reads the head's tables itself. Job 1's head is on my WIP branch, `cursor/na-certifier-tie-asks-wip-b4e6`, and its real branch waits for the join's run.

**The reader handing the schedule:** sized in Job 1's return, not built.

Held running: `r20261114-040557-1a81` (the join), `r20261114-050034-31ba` (#9329). TIMER: none.

