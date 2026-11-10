[2026-11-10T18:29Z] 1f33cd969d25
DIGESTLOCK, NET-301, Tue 10 Nov 10:29 AM PST. **Done: the four steps, on `main` as it is. No PR is open; it waits on your word.**
1. Head `4b3f7d6966` on `cursor/net301-definition-digest-c520`, pushed: `main` `e7b03e4a6c` plus the move, eight files, +166 −106 [run: `git diff --stat`]. Main merged twice, no conflict either time [run].
2. Records, `r20261110-155215-4f28`: `OUTCOME regenerated` [run]. Committed `b065f88a53`, +12 −7 in each Security lock: the five under `reads["NciCheck.DefinitionDigest"]`, `auditOf` rehashed, nothing else [run: key-by-key compare]. A and B left out as you ruled.
3. Check on the committed records, `r20261110-170246-bfc0` at the head: `OUTCOME pass`, all five packages, nci without `"format": 1` [run]. Your condition holds.
4. Suites, `r20261110-170640-5b1b` at the head: "2 suites: 2 passed (0 cached), 0 failed" (network 83, network-traces 519, none skipped) and "54 suites: 54 passed (5 cached), 0 failed" for `--changed e7b03e4a6c --quick` [run].
5. The pins' readers passed unchanged [read: the suites' passed lists]: `test_slot_entry_pins.py`, `test_network_period_protocol.py` (25), `test_scan_zk.py`, `test_chain_record.py`, `test_certifier_piece.py`. `slotEntryPins`' two digests are main's [read].
6. Composition in full, `r20261110-164512-02e4`: "27 passed in 4562 s", but at `7873a54b80`, before the records commit and the second merge; at the head only its quick tier (19 passed) [run].
7. The DM a landing sends: "definition `NetworkCheck.Period.auditOf` changed" under Security/Properties and Security, nothing else [run: `spec_alert.message`].
8. PR body: last section of `note:20261110T1555Z-state-net301-digestlock`, with your one line on A and B. Left: open the PR; the full Lean check (no kernel replay ran, fast path only).
9. Not looked at: the merge-tree reads against the lane's other open branches; whether the retired-terms ratchet warns after landing.
10. Least sure: why the update's review counts 628 and 387 "read under another name" and not five; I did not work it out [argued only that the line changes no statement].

[2026-11-10T18:33Z] a00c13c9fe62
TIMEPAIR, Tue 10 Nov 10:32 AM PST. **The window's first run is scored: both sides rc 0, 9 of 9 cases, guard passes, no case with a problem. No lateness I can lay on the observer.** [run: serve r20261110-174418-af43 on cpu-10, observe r20261110-174425-4be1 on cpu-8, no idle hold (driver `none`), 1,000 scored a row]
- Bits an interval, each set's larger row: as is 0.024 [0.010, 0.043] (burst; write 0.0006 [0, 0.006]; control 0.000 [0, 0.019]); with the waker 0.008 [0.001, 0.020] (burst); cores awake 0.002 [0, 0.009] (write).
- As is misses the target across two hosts too: its lower bound is 25 times 4.17e-4. The other two sets' bounds reach 0, but their upper bounds are 22 and 47 times 4.17e-4: 1,000 scored cannot show the target.
- **Lateness is all on A, and only in the awake set:** 26 late server writes and 87 late pacer releases in its write case, 2 and 1 in its burst case; none in as is or the waker set. Both counters are A's processes against A's clock. That set spins an idle process on each of A's 64 cores (A's load 0.27 at the start, 63 at the end).
- **The observer has no lateness counter.** Its receive stamp is its own process's `time.monotonic_ns()` after `recv`, not the kernel's, so cpu-8's load can widen what it sees: spread between quartiles 23 to 46 µs at the observer where the pacer's own write ends spread 4 to 13 µs. I cannot split that between A's send path, the network and B.
- Load: A 0.00 at launch, 0.27 at START, 63 at the end; B 105, 44, 17 [run: each host.txt].
- The second run launches about 10:40 with START 10:55:00, `--timeout 50m`.

[2026-11-10T18:33Z] a4dd3899fd39
TIMEPAIR, Tue 10 Nov 10:33 AM PST. **Correction to my message of 10:33 on the first run, its second bullet. The figures in its first bullet are right.**
- Wrong: "the other two sets' bounds reach 0". Only the awake set's does (0.002 [0, 0.009]). **The waker set's lower bound is 0.0013, 3 times 4.17e-4: with the waker the burst row still misses the target**, as as is does (lower bound 0.010, 25 times).
- The upper bounds I gave in the wrong order: the waker's is 47 times 4.17e-4, the awake set's 22 times.

