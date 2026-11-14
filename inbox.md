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

