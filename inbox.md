[2026-11-11T01:55Z] 876ac3858302
ONSITEMAIN, NET-327, Tue 10 Nov 5:55 PM PST. I wait on one run: `r20261111-015022-82a7` (running).

**(c): now 68 files, still seven items; none decided by the base's new commits.** Two files moved from (b) to (c), both ours, both item 7: `Security/Definitions/NetworkCertifier/RecordRows.lean` and `Security/Proofs/NetworkCertifier/RecordRows.lean`.
- Base: `fxQueueSched` moved to the definitions file, value as it was (`cells := ![6, 2, 2]`). [read: that file:117 at `eb56c217f831`]
- Staging: keeps it in the proofs file with `cells := ![5, 1, 2]`. [read: that file:268 at `00d79b22e4d2`]

**Rows that changed** [git merge-tree: the same 105 conflict, the same 156 clean; sort by my script on the VM]:

| group | (b) | (c) |
|---|---|---|
| 7 | 5 → 4 | 9 → 10 |
| 8 | 3 → 2 | 17 → 18 |
| all | 22 → 20 | 66 → 68 |

**Nothing became carriable.**

**Fixed by the base inside item 7, not deciding it:** the witnesses' instances live in `Security/Definitions/NetworkCertifier/Witness.lean` (lines 42 to 157); `WitnessToys.lean` holds no `def`. [read] Staging adds its new instances in `WitnessToys.lean`. Item 5: the base's one line in `tests/test_chain_record.py` chooses between main's test and the side's driver; staging and the base agree there. [read]

**Merged:** `eb56c217f831f53084645527c9fa90181582edce` into `cursor/net-onsite-port-e83a`, merge commit `c090b05057e7`, pushed. No conflict: none of the base's 22 files is one of the six carried. No reorganization in it; `main` not merged. [read]

**`b840`, at `5d82c83ee4b7`, before the new base** [run, exit 1; its `out/` read by me]: as you read it. Both builds rc 0; `NetworkCertifier.Rows`, `NetworkCheck.RowsAudit`, `RowsMain` and `network-rows:exe` built; network 88 passed; certifier 1142 passed, 2 failed, 1 skipped. Neither failure reads a carried file: both are raised in Python at `compile_slice_v6.py:1099`, the call of `ands`. No fix made.

**`82a7`, at `c090b05057e7`, after** [run, not ended]: the same two builds and two suites.

**The base's failures gone:** in my two suites I expect none gone and none new: circuits' cure is not in `eb56c217f831`. Your 12 are in `network_traces` and `network_e2e/encodings`, which my run does not run; I read only that their four test files are among the base's 22. [read] Say if you want those two suites run at this head.

**Least sure:** the two moved files rest on my script's rule; I read the one declaration they meet at, not the rest.

**Note:** `lanes/network-accounting/20261111T0118Z-state-net327-onsiteport.md`, updated.

