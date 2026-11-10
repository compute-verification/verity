[2026-11-10T04:24Z] cb8abc1042a2
From the network-accounting coordinator, Mon 9 Nov 8:35 PM PST. Four things for the staging (NET-199). None needs an answer before your next report.

**1. PRF-484 is answered** (proofs, Mon 7:36 PM PST; top's tree on TOP-394): `main` keeps the remote branch's per-member form of the audit. Monday's shape can't accept on `main` before a proof covers it. So the onsite scan service and period check read the per-member form: cut the three staged changes that waited on this to it. Soundness has three questions open on it; if one touches a staged file, say which in your report.

**2. A break the merge won't report, from the clock-openings split** (PUBLICCERT, its comment on NET-283 of Mon 8:18 PM PST; branch `cursor/net283-clock-openings-8f35` at 4c577caaf5, no PR yet). The split moves the clock openings out of `NetworkCheck/Period.lean` (lines 1284 to 1398) into a new `NetworkCheck/Clocks.lean`, and `RecordsMain.lean` then imports `NetworkCheck.Clocks`, not `NetworkCheck.Period`. Staging's `RecordsMain.lean` also serves `("circuit", NetworkCheck.Circuit.circuitFn)` and reaches that module through `import NetworkCheck.Period`. git merges the two cleanly, and the merged file names a module it does not import. And `NetworkCheck.Circuit` imports `NciCheck.Desc`, so with the `circuit` port in `network-records` that executable needs the inference-only policy's check again, which the public export withholds.
   - In your report: where the `circuit` port can be served so `network-records` stays free of `NciCheck` (the period executable, or one of its own), and which staged PR carries it. Change nothing in staging for it until I answer.
   - Also in the split: `keysOf`, `exactly` and `hexAt` lose `private`, and `clockTag` moves with `clockDigest`.

**3. The scan's Lean proofs are parked until the onsite demo is on `main`** (Daniel's re-cut). SCANCOUNT ports no `Walk`, `Causal` or `Fits` this round, and `PeriodLeakRegistered` is not restated: PR 7 and PR 8 carry the onsite branch's Lean as it stands. SCANCOUNT's `scanOf` PR (the two refusals) still lands on main before PR 3.

**4. Two corrections to my earlier notes to you.** I wrote that "the onsite statements read no checked input": loose. NET-285's statements read the scan's two ties. And the later statement at the scan is the scans' instance of main's `PeriodLeakChecked` with no registered input (soundness's ruling), not a restated `PeriodLeakRegistered`.

