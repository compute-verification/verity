[2026-11-14T07:29Z] d997cea9ce50
WALKER, Fri 13 Nov, 11:30 PM PST.

**Two sessions of me are running on this VM at once since the 07:25Z reboot. I hold your 10:54 PM brief and the 11:05 PM ruling, so I propose this session does A and the other does C; please tell the other one (it wrote inbox37, asking for the brief).**

- RAN: the hook's event log shows two generations of my conversation running shell commands in the same minute, 2266a680 (this one) and 4d808059 (the one that wrote inbox37). That one has no 10:54 PM brief.
- RAN, before the reboot, at A in scratch, all lost with the VM, none pushed: M1, M3, S2, S3, N1 and N2 committed locally, and `test_work.py` passed 115 of 115.
- RAN, M2 as ruled, also before the reboot: the second point's lanes, beside the first's, in one walk.
  - The M2 probe is refused (`SpecificCancelStep_v1`, w, rows 0 and 1).
  - A specific-alone NaN and a specific-alone read past an unrun scan are both refused at the second point.
  - `test_work.py` passed 118 of 118 with `readings.json` unchanged, and the honest small model passes `-k point_evaluates`, with 23 distinct and 23 specific-alone claims.
- RAN, why the honest model was first refused at the second point (node 122, `x`, rows 2 and 8):
  - First cause: the state inputs drawn at full float range blew up θ after one AdamW step, and the softmax went one-hot.
  - Fix: drawn as θ's are (about 2^-3), but positive.
  - Second cause: tokens drawn independently collide modulo the table's 64 rows, which breaks `P_BATCH`.
  - Fix: each lane's tokens are a permutation of the least table's rows, repeated.
- Lanes, not a second walk (RAN, small model): the walk takes 0.1 s either way, and peak memory is 882 MB against 876 MB. At s-t20 the walk's value arrays double, and the s-t20 run will report that memory.
- My recommendation, which I follow unless you say otherwise:
  - I rebuild A in my own worktree and push it to a backup branch at each step, so a reboot loses nothing.
  - I send you A's full sha the minute its fix head is pushed.
  - The other session takes C in its worktree once A's head is up: the merge, S1, M-C1 and M-C2, and the trailer lines.
  - I left it the brief and this split in a file on the VM.
- INFERRED: noon holds for A. Sun 6 PM holds if s-t20 starts by early afternoon Sat.

