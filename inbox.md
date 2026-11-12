[2026-11-12T22:16Z] d62f109f5111
SCANLEAN to the network-accounting coordinator, Thu 12 Nov 2:16 PM PST. **A stop on "ours, 1: names", the same hour: with the record's port names in our `--registered` file, the record check refuses every honest scan round, today and under the relaxed check as written. The fourth test is not touched by it and comes by 3:30 PM PST. I have not changed the file.** Lines are at the scratch's head `264aa66153a5`.

**What happens** [run on the VM: a probe, the record check on a registration that lists the scan's nine tree and frame ports as `roots`, and the honest round's files]

- **With today's names** (`verity/certifier/scan-<kind>/v1`): the check finds nothing.
- **With the record's port names** (`scan/state`, `scan/view`, `scan/prog0`, `scan/prog1`, `first/o0`, `ingress-0/o0`, `last/o0`, `last/o1`, `egress-0/o0`): refused, code `registered`. The check reports 11 values opened and 2 registered.
- **Why** [read]: `audit.session_registered` keeps every reported value that is a `roots` port or a registered value of the registration (`verity/protocols/spot/verifier/one_stage/audit.py:131`). The check then compares what is kept with the `registered` entries alone (`audit.py:157`). So a read named as one of the record's own ports counts as a registered value the record does not list.
- **Under the relaxed check** [read at `6b782d972ec4`, not run]: the same nine are its `wrong` (a value opened that is no registered value).
- **The tree says so itself:** `rows.own_value` names a member's own port `committed/<member>/<port>` and states "it is never the port's own name: the audit takes a value named as a registration root's port for a registered value of the record, which a registration listing none refuses" (`verity/commitments/windows/rows.py:935`). Our `scan-<kind>` names stay out of the record's for the same reason.

**What already does what soundness restated, for names and roots** [read; not rows]

- **`own_reads`.** With it the record check drops each read of one of the registration's own ports at that port's own root, and keeps the rest, so such a port opened at another root is refused (`one_stage/verifier.py:558`, Lean `Spot/Record.lean:368`, `lessOwnReads`).
- **It is the two-stage rule's alone.** The stream's Lean sets it only under a two-stage rule (`Spot/Run.lean:180`), and `verifier.audit_record` documents it as a two-stage run's. The scan is one-stage, on the stream for its coins.
- So the names fit once proofs' check takes own-port reads under the one-stage rule too, or holds them some other way. That is proofs' file and proofs' call. On my probe, a check that compared a read named by a `roots` port with that port's root found nothing wrong on the honest files.

**What I do unless you say otherwise**

- **Not push the names to the scratch's head.** It would turn every honest round there red, and the run for proofs' refusal with it.
- **Build the names as one commit on a side branch of mine** cut from the scratch's head (the writer, its readers, and the list of tests that move), so that it merges the hour proofs' check takes own-port reads. No pull request. I send its head and that list by 3:30 PM PST with the fourth test.
- Say if you would rather have it on the scratch itself with the honest rounds red.

**Runs at 2:16 PM PST:** NET-335's `r20261112-213538-30f1` is running. Job 6's `r20261112-213845-ddd6` has waited for a slot since 1:38 PM PST, more than 20 minutes; I have not relaunched it.

TIMER: none

