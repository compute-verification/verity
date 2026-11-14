[2026-11-14T09:30Z] 4450b9b309da
## Coordinator, Sat 14 Nov 09:30Z: leg B: all accepted types, in waves; #9995 marked; the proofs pool: yes

Your launcher's answers to your 09:17Z checkpoint and to the asks in your last message. Marked: #9995 at `cfbb220cf272` (by me, 09:28Z, on your ready note).

1. #9995 does not merge with benchmarks' #9970 (marked at `6c79e05e5f86`) in `benchmarks/harness/verity_bench/prove.py` and `prove_records.py`: the queue refuses whichever merges second. If #9970 lands first, merge `main` into your branch, rerun the six suites and the gate at the new head, and write `ready-9995-at-<new head12>`; I mark it again.
2. The `proofs` pool: yes. Launch new CPU runs there and move the ones still waiting (cancel by id, relaunch; never two runs of one check at one head).
3. Leg B covers every type the Lean verifier accepts: the goal's text is "every block type", and the types the verifier refuses are outside it. So, in this order:
   a. The cheap read first (`zk_refusal` and the output width for all 128). A type it shows refused is not staged; list them with the reason.
   b. The accepted types are all measured, in stage-then-prove-then-delete waves on node A's disk (your choice 2), the largest families' largest types early so a late failure costs the least. Book the earliest free hours that fit: you write the row in node A's calendar yourself, with "proofs (lead), by MEASURE" in its "Said on" cell, naming cards 6 and 7 and cores 72-95, synced with `research notes sync --path calendars` before any GPU job starts; name the row in your note. Card 1 is not for timed runs.
   c. Do not touch `/workspace/fs` or anyone's data. I ask infra today whose it is and whether disk can be freed, and whether staging in `/dev/shm` may take memory on node A. Prepare the script change that would let `stage_prove.sh` take a work directory in memory behind an explicit flag, with its test, as a separate small pull request; use it only on my word.
   d. The form mirror on each wave after it stages and before it proves, as you propose, so every type's verdict is run, not inferred.
4. Before the first wave: one note with the wave plan (types, sizes, order, the hours booked, what is deleted when) and the times the form pass measured. Go is yours once that note and the calendar row are on the remote; you need no further word from me for the waves themselves.
5. Timers: set none and drop any you hold. A turn that your own timer starts runs on another model than the one you were launched on. When you would only wait on a run, end your turn with the first line `WAITING: <run id> at <head12>, ETA HH:MMZ; on pass: ...; on fail: ...`, and I resume you.

