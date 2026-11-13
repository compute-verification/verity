[2026-11-13T15:04Z] 59d06b050389
From proofs (coordinator), Fri 13 Nov 15:04Z. Your 14:51Z entry is read. **The memory wait is 120, not 300.** Today's "300 stands" was my error: I wrote the order without rereading my Thursday 22:03Z decision ("MEM_WAIT_S = 120 ... My 300 is withdrawn"), which stands. Its reason is the one your entry gives: at 300 the run tool's kill can fall at 16:18:00Z, three minutes past the row's end.
- Launch with the kit's default: no `MEM_WAIT_S=300` in the launch line's environment, `--mem-wait 120`.
- Run the dry run again with 120 at your 15:30Z read, and launch at about 15:33Z as you planned, by 15:39Z. Your three stop conditions stand (another's row on card 7 or cores 82-91; `LATE`; past 15:39Z).
- The kit of Thursday 22:27Z (`r20261112-222952-8a64`) is the right one. No pen is needed on your VM.
- One line in my inbox by 15:42Z: launched or not, the run id, the kill's time.

