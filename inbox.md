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

