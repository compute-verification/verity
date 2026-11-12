[2026-11-12T06:50Z] e98838b0bb17
BENCHMEM, from the lead of memory-accounting, Wed 11 Nov 10:50 PM PST (06:50Z). Two additions to order 12. Act only on what names you; never run `mem-hourly`.

**1. Part B's names are decided (benchmarks, MEM-279 10:42 PM PST). Use these, not my fallback.**

- The shared last words are five: `metric`, `verified`, `capacity`, `free`, `error`. Each measurement's `inputs` names the protocol's own key it was read from (for us the verifier's record's keys: say which key each came from).
- compute-accounting's #9161 (`gh pr diff 9161`) is the first runner on the writer. Read how it names a measurement with a reading (its `capacity` and `free`, one each at two clocks) and follow the same shape for our two readings (the data sheet's and the measured), unless ours cannot. It also adds an optional `why` to `kit.measurement()`, required when the value is `None`. #9161 is not on `main` yet: do not depend on `why`; if you need a value with no number, leave the measurement out and say so in the README paragraph.
- `kit.SHARED` (the names as constants) lands by Fri noon PST. Until then the literal words are right.
- Still to say in the result: the capacity's source is the verifier's own `timing.py`, not the registered hardware.

**2. One more answer in part A's comment on MEM-227, for kernels (HASH-120, their question to us, due Thu noon PST). Arithmetic only, from word 3's record and `timing.py`'s own functions. Run nothing on the node.**

- kernels asks what a pass at 1.03 of a plain read is worth to the share, against 1.02. Today's pass is 1.061 of the read on eight cards [kernels' screen, 700 GiB].
- Give the printed share (the data sheet's reading, and the measured) on word 3's store at each of four pass lengths: today's 46.26 ms, and that pass scaled to 1.03, 1.02 and 1.00 of the plain read it implies. Give each twice: with the session's wait for its coins as it ran, and with one challenge a minute.
- Say which terms you held fixed and which move, and mark each figure [ran] with the script's path. One table, eight rows.
- If part A's comment is already posted, post this as a second comment on MEM-227.

