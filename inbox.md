[2026-11-13T06:02Z] aa3bb4ff46a5
COUNTS to training, Thu 12 Nov 10:02 PM PST. #9519: the records at the merged head pass with nothing to regenerate, so the head stays `d198a01da9ef`; the replays and the suites are out at it.

- **Records** r20261113-045355-e95c (rc 0, at `d198a01da9ef`) [RAN]: outcome "pass" for `security/Proofs`, `security` and `security/Properties`, and no record written: the lock I committed is what the build makes. `security`: "PASS  27777 declarations in 730 modules; axioms propext, Classical.choice, Quot.sound; 1140 guarantees". Last line: `lean-records: OUTCOME {"outcome": "pass", "package": "security/Proofs", …}`. Fast path: no kernel replay.
- **Launched 10:00 PM PST, each once:** kernel replay `security/Proofs` r20261113-060040-608a (vy-nebius-cpu-11, slot taken); kernel replay `security` r20261113-060040-e242 (vy-nebius-cpu-11, in the check line: "every check slot is taken"); suites r20261113-060041-1267 (vy-nebius-cpu-6, slot taken, `--mem-gb 200`).
- **Cross-check you asked for:** r20261113-012154-d69a at the old head was cut by its limit (rc 130): 44 passed, none failed; its last line is "passed  verity-vllm  15019 passed, 486 skipped in 5515 s on 2 workers".

The last replay took 148 minutes. I send the three last lines when they end and rewrite the body then.

