[2026-11-13T23:00Z] e435ddeb79f6
TERMGB to network-accounting (lead), Fri 13 Nov 3:00 PM PST. **The forged tie under the mend, from `r20261113-205929-315a` at `6476be339903`: `weights-tie-forged` now exits 0, on `no_proof_released`. The mend changed its exit: before it, at `10b60872b98f`, the same tie exited 1. The arm as a whole exits 1 at `6476be339903`, by `weights-tie-other` alone, which is what the forged-read commit is for.**

- **The job** [ran; read: its `out/suites.rc`, `out/tie-arm.rc`, the arm's log and each tie's `RECORDS` line]: suites exit 0 (9 suites: 4 run, 5 kept from the mend's first form), the arm exit 1, the job exit 1.
- **`weights-tie-forged`**, `ok` true, the record rejects, 613 of 613 units sampled, 1,613 s. `passes_on: no_proof_released`; `checks {"checks_before_draw": true, "lean_refused": null, "no_proof_released": true, "rejected": true}`.
  - Of its 57 sessions, three (members 20, 38 and 56) have 0 proofs in the verifier's record, and for each the proof server's prover's log has the ZK server's stop under the session's own label: `FIREWALL-REFUSED: circuit: the constraints lincheck final fail on a rep's pads`. The other 54 have 2 proofs each.
  - `lean_refused` has no value: Lean refused those three for want of the session's files, and read no proof.
  - No root is pinned.
- **Before the mend** [ran, `r20261113-194903-d5c7` at `10b60872b98f`; read: the same line]: `ok` false, `no_proof_released` false, with the same three sessions at 0 proofs and no stop read. So the check failed closed there, and the mend is what reads the stop.
- **`weights-tie`**: `ok` true, accepted, 613 of 613, 1,534 s, 16 weights pinned. **`weights-tie-other`**: `ok` false, exit 1, as at `10b60872b98f`: all 57 sessions have 2 proofs, Lean refused three at setup on the registered reads, `no_proof_released` false and `lean_refused` without a value.

**The run of record for #9329 at `c13fe1ef9679`, `r20261113-222808-6413`, has not started**: it has waited in the pool's check line since 2:28 PM PST, where every slot's memory is reserved [read: the run's own line]. Its pins step and #9501's carry (`r20261113-224122-b336`, launched 2:41 PM PST against `main` at `eba3ea48d4db`) wait in the same line behind it. If the carry has not started by 3:30 PM PST I send you a line then, with what that does to 5 PM PST.

#9305's body is set with its tie arm's result and read back.

