[2026-11-12T11:08Z] d972189b8588
NETSEATS, from the network-accounting coordinator, Thu 12 Nov 3:08 AM PST. **One read more in job C, same date (6:00 AM PST): the device's clock.** soundness answered on MEM-279 at 3:05 AM PST.

- **Its answers:** (1) one session at a time under a key: yes as `trusted-host`, once the code holds the signer's state from before the first tick, with a negative test. (2) "A device millisecond is a millisecond": no. State the clock's rate within ±ρ of real time, ρ from the host's data sheet (crystal tolerance and its temperature range, the larger reading), and count cover as span · (1 − ρ). ρ's source goes in the Site's entry for the device.
- **What I answered (3:08 AM PST):** the check keeps publishing device milliseconds exactly; the account's step states (1 − ρ)/1000 s. And: the deadlines are on `time.monotonic` (`warden.py` 391, 500), which on Linux a time daemon slews, so my intent is that the device reads the raw clock. I confirm or correct that by 6:30 AM PST from your read.
- **Read, and put in the note as its own section:**
  1. Every clock the device and its serve runner read (`warden.py`, `warden_serve.py`, `certifier_serve.py`, `auditor.py`'s gate): which call, and what each deadline, sleep and fault test is measured on. Is the clock injected everywhere, so one argument changes it?
  2. What reading `CLOCK_MONOTONIC_RAW` changes: the sleeps (a sleep runs on the steered clock), the slack test, the tests that inject a clock. Is there any path where the developer's side could set the device's rate today (a time daemon on the certifier's host in the deployment files, a clock the adapter hands over)?
  3. Does the tree name a host model for the network certifier, so ρ has a data sheet? If not, say so and where one would be named (`verity/hardware/`, the Site's entry).
  4. In the account's form on proofs' branch, can a step's length be a rational (9999 over 10,000,000)? One `#eval` in your scratch if cheap; do not push to the draft branch.

