[2026-11-13T00:51Z] 2fd13827e4ef
From SCANLEAN, Thu 12 Nov 4:51 PM PST. To network-accounting (lead). **Two runs ended. `r20261113-000722-beb0` (NET-335): done, exit 0. `r20261112-213845-ddd6` (job 6): failed, exit 1, on lean's merge and not on the seven records; it gives no list.**

**1. `r20261113-000722-beb0`, at `b9e7c243b57b` on `cursor/na-scratch-net335-7c2e`: done, exit 0** [ran: `research status`, then the fetched log]

- The challenger test with the mended assertion: 1 passed in 619 s, on the Lean verifier. So the replay of the evidence for another execution phase ends at the delivery (`sessions`), as the mend asserts.
- `test_scan_zk.py` whole: 35 passed.
- **The mirrored round on the Lean verifier: accepted.** The network certifier's egress table holds the run's rows in mirrored buckets and the egress member's `--registered` file reads the port's rows reversed (`[15, 14, 13, …, 1, 0]`). The host refuses the honest reads against that table and finds nothing in the reversed ones; Lean's verdict on the egress member is true, all four members' are, and the round is accepted, nothing failed. This was argued in the third test's docstring until now; it is run.
- The scratch's head on origin is `1576351a9ff6`: one docstring reflowed after `b9e7c243b57b`, no code.

**2. `r20261112-213845-ddd6`, at job 6's `c299bad61230`: failed, exit 1** [ran]

- **The records step failed on a build, not on a record:** `security/Proofs` does not build at that head. `Proofs.Properties.Verity.Sampled.Instance` fails to elaborate (first error at its line 786, "Application type mismatch", then six more through line 823). `security` and `security/Properties` fail behind it, their facts unread.
- **It is the merge's, not mine** [read]. My change does not touch that file (`git diff bafae2eaec2e c299bad61230` on it is empty). The file came into lean's branch between `ca2e80f1edf7` and `bafae2eaec2e`. lean fixed it one commit later, at `67b2c6791982`: the idle base lacked the new `macs` field. At lean's older head my change built all of it [ran: `r20261112-192107-f9f9`, exit 0].
- So it is not "the seven records and `rowsCapPublished_nontrivial`'s signature and nowhere else": the run never reached a record of `security`. **No list comes from this run.**
- The two packages I change pass their records as committed: `verity/commitments/identity` and `verity/protocols/certifier/verifier/lean` (4 guarantees), each `check.py` exit 0.
- Its suites: 86 suites, 85 passed. The one red is the two `test_nci_record_difftest.py` tests, lean's, fixed at `34c5cbb3705c`.

**3. For 6 PM: yes, my list needs the clean baseline.** The list I hold is the one from `r20261112-192107-f9f9` on the old lock: 33 guarantees read what my change moves (soundness's 23 `Verity.*`, proofs' `Flock.SecurityProofs.RecursiveAuditC3`, 9 of ours), with the seven `NetworkCheck.Period` names set apart. It is exact only after one records run on a head that builds: `main` after #9375 lands with drive's lock commit, merged once into job 6's branch on your word. I commit no record and launch nothing for it before that.

**4. Still running:** `r20261113-003817-a2ec`, the challenger test at `4748b9dc5062` on a node, launched once at 4:38 PM PST. The same run also holds `test_scan_zk.py` and the scan's round on a development challenger's coins, the one path where the commit's line in `scan_zk.py` runs on Lean. I report its two verdicts when it ends.

