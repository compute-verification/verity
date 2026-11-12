[2026-11-12T11:21Z] 860557a9a732
BOUNDTIE, NET-349, #9181 at `ad421f210f89073ab84e1eabb980d591b5a7f309`, unmarked, 3:22 AM PST Thu 12 Nov. **55 of the 56 suites have a pass at this head. The circuit-check suite is still running, well inside its limits.** Each status below read by me at 11:20 UTC [ran: `research fetch --all`, each run's `stdout.log`].

- `r20261112-103233-2f27`: done, exit 0, "3 suites: 3 passed (0 cached), 0 failed". `verity/definitions` 950 passed, 2 skipped; `verity-numerical` 641 passed, 7 skipped; `verity-core-evaluate` 67 passed, 1 skipped.
- `r20261112-103244-4dd6`: done, exit 0, "3 suites: 3 passed (0 cached), 0 failed". `verity/circuits` 1,125 passed; `verity/commitments` 592 passed; `verity/commitments/windows` 95 passed, 1 skipped.
- `r20261112-103207-83b5`: the circuit-check suite, **running**. It waited 148 s for a slot and took one at about 10:34 UTC, so it is 46 minutes in. Its `pyproject.toml` says a pass can take up to an hour and gives it 7,200 s [read]; the slot's limit is 12,000 s from taking the slot, which is about 5:54 AM PST. So its own limit ends it by about 4:35 AM PST, 80 minutes before the slot's [argued]. Its log holds no verdict line yet.

**The 56:** 49 passed in `r20261112-070459-b6c4`, 6 passed in the two runs above, 1 is out. No test has failed at this head in any of them.

- I have not merged `main` in and will not.
- The body's Runs section says the above, with its `Goal` and `Refs` lines kept.
- I read the last run once more in 40 minutes and send its verdict. If it is near either limit then, I say so before it is cancelled, and I do not launch it again.

[2026-11-12T11:29Z] f03d2ad70e98
DIGESTMOVE, Thu 12 Nov 3:29 AM PST. r20261112-072115-5b9f was killed by its limit at 3:28 AM PST (cancelled rc 143, STAGE_TIMEOUT: 14400 s from taking its slot) [ran]. Not relaunched; you decide.
- Before the kill, security's --update pass ended: "AUDIT ... security: PASS 27860 declarations in 738 modules; ... 1147 guarantees", "check.py --update exit 0 in 14230 s", 0 failures in its report; security/Proofs PASS 201401 in 3973 again. A lean-fast/v1 result, no kernel replay.
- security/Properties had only begun (objects dropped, no AUDIT line): no verdict for it, and no result.json for the run.
- The regenerated security record the run kept (lean-audit/record/security/lean-audit.json, on scratch 426e7457d3e9 = lean's de28a691126b + #8686's head + the close) says what my last message said: NetworkCheck.Period group 1798837538f7, 228 definitions, auditOf 01d0626ec0cd; NciCheck.DefinitionDigest group dc40fe808b88, the move's five records as #8686 has them. I committed it nowhere: lean's tip regenerated that lock since, so the lock run after lean's merge writes it.
- Now: closing comment on NET-301, my note synced, then I end. TIMER: none.

[2026-11-12T11:31Z] ce3e3e3ce7ee
NETSEATS to the network-accounting coordinator, Thu 12 Nov 3:33 AM PST. **Job C is done: read only, nothing built.** Note `note:network-accounting/20261112T1126Z-report-key-life-three-ways-and-device-clock`, synced and upstream.

- **Recommendation: (c), a registered root on the device (the tree's `lms.Identity`) that certifies the keys that sign.** It is the only one of the three where a change of signing key adds no registration entry, so the head stays. [argued]
- **The one thing to fix before the device's pull request is written:** the count, the sessions started and the run's lock go in one device state keyed by the *registered* key, a file apart from the signing key's `LeafState`. Then the key's choice can follow the device's pull request. If the count goes into `LeafState`, as I sketched for (ii), (c) has to move it out again. [argued]
- **Keys a week, a month and a year for one certifier, at 3.2 s and then at 60 s** [arithmetic on read constants]:
  - today (five leaves a phase, H = 20): 1, 4, 47 and 1, 1, 3;
  - (a): H = 22 gives 1, 1, 12 and 1, 1, 1; H = 24 gives 1, 1, 3 at 3.2 s; H = 26 gives 1, 1, 1;
  - (b), one leaf a phase: 1, 1, 10 and 1, 1, 1;
  - (c): registered keys 1, 1, 1 at both; signing keys as today.
- **(a)** [read]: the certifier's checks take the key's parameters from their JSON, so no Lean of the check moves and `lms.DEPLOYED` and `Lms.deployed` stay. A signature grows 64 B a level. But making a key is 4, 16 and 64 times today's at H = 22, 24 and 26, the tree is held whole (0.54, 2.15 and 8.59 GB), and `LeafState` re-reads its whole journal at every signature, so H = 24 or more needs `lms.py` and the key's loader rewritten. It fits (ii)'s state. No run's record I opened holds a key's making time, so I give none.
- **(b)** [read; argued]: it moves the certificate's shape in `certificate.py`, both checks' versions, five Lean files of the check, the Lean issuer, and the lock's reads of `NetworkCheck.Certificate`, `Check` and `Period`: the most trusted Lean of the three. It leaves 10 keys a year at 3.2 s. It is worth having later for bytes: a certifier's 1.71 TB of signatures a year at 3.2 s becomes 0.34 TB. It fits (ii)'s state.
- **(c)** [read; argued]: `Identity`, `PartyKey`, `Lms.Hss` and `Attest.checkKey` exist, and only tests use them. The Site registers the root's 80 bytes under a new scheme id that names both levels. With the signing key's certificate carried once in the evidence, a certificate carries and costs nothing more; an answer costs 34,836 B and one LMS check more for each signing key in it. It changes (ii)'s state as the second line says. About 300 lines of Python and 130 of Lean. Open: what `deployment` its party-key message binds, which must not move with the head.
- **The device's pull request as the one change** [argued sizes]: about 260 lines of Python, 250 of tests and 110 of Lean. The Lean is the Lean device's mirror, which the gate does not run, so it can follow. The files, and what the tests inject, are a table in the note.
- **The clock is not one argument** [read]. The Warden's clock is injected, but `precise_sleep` waits on `loop.time()`, the pacer's process reads `time.monotonic()` six times against a `due` on the Warden's clock, and `serve` sets the origin from `time.monotonic()` and passes the Warden no clock. `warden.py:391` is the client's shim, not the device; the device's is `:500`.
- **On the raw clock a tick can run early** [argued]: no sleep runs on the raw clock, and `run()` ticks after one wait whatever the clock then reads. The pull request needs the tick to wait again, with a test; nothing tests an early tick today.
- **No host model for the certifier and no clock tolerance is in the tree** [read by search]: six cards on `main` now, none the certifier's. The tree names no path for the developer to set the device's rate; what time source the host follows is outside it.
- **A rational step works** [ran]: try `r20261112-111326-0840`, exit 0, on a scratch commit never pushed. A step of 9999/10,000,000 s gives 59,994 ticks for 60,000 steps and 3,199 for 3,200, rounded down.
- **Not established:** nothing is measured; the line counts are estimates; I did not read the proofs that name `signs`.
- **Slip:** one search printed unmasked showed a node's name in my own run; it was sent nowhere.
- `main` moved to `33866d93e66e` while I read. Of the files I read only the hardware cards changed, and I read that diff. #9197 is still open at `368902c12484`.

