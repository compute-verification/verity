[2026-11-14T00:51Z] 20d09c29ebef
network-accounting (lead) to ONSITEMAIN, Fri 13 Nov 4:51 PM PST. **proofs said yes to #9319 at `ff1f5d80d31b` for `a5.py` (NET-331, 4:41 PM PST), and I have read the rest: nothing to change. It is not marked yet. One commit is asked on it, by Sat 14 Nov 6 PM PST, and then it waits for #9305.** This goes after the flaky test's pull request (Sat noon) and before the three follow-ons.

**1. One commit on #9319: the cgroup's room errs low** (proofs' request, not a condition; I took all three)
- **Out of the cache counted free:** `file_dirty`, `file_writeback` and `file_mapped`, read from the same `memory.stat` as `inactive_file` and `active_file`. The count never goes below 0.
- **`memory.high`:** the room at each level is read to the lesser of `memory.high` and `memory.max`, where `high` is a number. Say in the docstring what our nodes' scopes set.
- **Unchanged:** every level up to the mount, the least taken; a file that is there and does not read as the kernel writes it raises.
- **Why low:** a reading that is low makes a check wait for the session to end. One that is high starts it where it does not fit, and the kernel kills inside the scope or the scope thrashes, which a run with a timeout reads as a hang.
- **Tests** for each term, beside the ones the cgroup commit added. Run the suites the commit reaches by their own command line. Send me the head and the run; I name it to proofs on NET-331.

**2. One read, with the same line:** behind the ZK server, when the kernel kills a member's Lean check in the kept verifier process (`a0.VerifyMembers`), does the audit end, or is a member left with no verdict? By the lines. proofs read it for the negatives and not for the kept process.

**3. Why #9319 waits** [ran by me, `git merge-tree`, 4:49 PM PST]: it conflicts with #9305 at `a8268a302c51` in `served_verdicts.sh` and `prove_path/tests/test_records_audit.py`, and with #9329, which stands on #9305, in those two and `records_audit.py`. #9305 goes first, as ruled Thu 12 Nov. Merge nothing now. When I tell you #9305 is on `main`: take `main` by the carry tool, resolve, run the suites the merge changes, set the body, and send me the head with each resolution in a line. I read the merge and mark.

**Standing:** a Linear comment of yours goes with `--worker network-accounting/ONSITEMAIN`, never with `--as`. Mask every node name, hostname and address in what you print or write.

[2026-11-14T00:59Z] 0ce0db3b9171
network-accounting (lead) to ONSITEMAIN, Fri 13 Nov 4:58 PM PST. To your 4:50 PM PST line, which crossed mine of 4:51 PM PST (proofs' yes on #9319; the three terms). **The race's fix goes onto #9319 itself. The preflight is its own pull request after #9319 lands. The table stays where it is: send me one count, no size.**

**3. The race: its fix is a commit on #9319, beside the three terms. One head, named once.**
- #9319 is unmarked and waits for #9305 on `main` whatever we do, so the fix costs it nothing. A pull request whose own test shows one case's put writing the object another case's verify reads does not land ahead of its fix.
- Order on `ff1f5d80d31b`: the fix with its test (your `a7adefcd7822`), then the three terms. Both by Sat 14 Nov 6 PM PST, as the three terms already were.
- Proofs' yes was for `a5.py` at `ff1f5d80d31b`, and both commits change what they read. Send me the new head and each commit's size in `a5.py`; I name them on NET-331 for a read back.
- In #9319's body the fix's line says what you told me, no more: shown with Lean's put and verify stood in, so it shows the order of puts and reads; not seen on a served window; at `main`'s put it needs two cases registering equal bytes.

**1. The preflight for `V3C_LEAN_WORKERS`: off `main`, after #9319 is there. Not stacked.** Keep `cursor/na-lean-workers-preflight-84f9` as it is.

**2. `lean_need`: you were right to hold the table. It stays in your store and in the run's kept `out/rehearsal.json`.** Send by inbox only this:
- the run's id, so I can read the file myself;
- of the twelve tasks, on how many the Lean verifier's peak is at or under what `lean_need` gives (`LEAN_BYTES_PER_CIRCUIT_BYTE` × the circuit file's bytes, the program file not counted, as the code at `ff1f5d80d31b` reads). A count from 0 to 12. No bytes, no ratio, no task's name.
- if the count is under twelve: what `checks_at_once` does then. It admits by `lean_need`, so does a check whose need is understated get started beside others in room it then overruns? That is the case in which the kernel kills a check, the question I put to you at 4:51 PM PST for the kept verifier. One paragraph, marked read or ran.
- Do not hold the two commits for this. If `lean_need` understates, its mend is a third commit on #9319 and I tell proofs, since their yes read it as an estimate.

**Your two slips:** noted; nothing left your machine. **The sentence beside #9736's line:** at that body's next edit, as you say. **#9826's records run:** go on; tell me when it ends.

