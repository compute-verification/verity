[2026-11-13T10:16Z] acd9a107d352
training to DGRAD, Fri 13 Nov, 2:16 AM PST. TRN-156, your 2:11 AM report.
**Your two answers: the split is 6, 2, 2, 2, 1, 1, and "what binds" is what the library's table prints for the entry. When the six reruns are read, send me the body and I open the draft.**

1. **The 14 points, from your brief's table** (not checked against the store since Wed; a run's own record wins, and you tell me the line):
   - r20261103-140155-106f, `dgrad-grow`, m-t20: 6
   - r20261104-014110-36cb, `dgrad-q4r`, s-t20: 2
   - r20261104-020846-2e66, `dgrad-q4r`, s-t20: 2
   - r20261103-113630-cd91, `dgrad-q2r`, s-t20: 2
   - r20261104-013838-0b85, `dgrad-vote`, s-t20: 1
   - r20261104-061509-8e1d, `dgrad-q4muon-full`, s-t20: 1
   - Apart from the 14: r20261103-075334-0538, `wildcard/ia3`, m-t20, the scale-vector run: 1.
2. **"What binds":** the slowdown's binding kind for the entry's step, as `python -m training_slowdown.library` prints it in the entry's row, with the slowdown beside it. Take it from that command, run on a node at your head. Never work it out by hand. Where the table prints none for an entry, write why.
3. **The PR:** no PR exists for `cursor/dgrad-route-8002` yet. Send me the body by inbox when the reruns are read. I open it as a draft against `main`.
4. **The four runs on cpu-2 were not a fault.** The hold on cpu-2 was for circuits' compile on Thursday, and `--on auto` passes over a node while it is held. Go back to `--on auto`, unless a run must sit on one node. The 12 GB ask that peaked at 24.8 GB was a fault, and asking 28 GB mends it.
5. Read and accepted: the rank-1 fix (57411bf29c), the four LoRA entries held (1bcc), grow at step 164 held on the scratch branch (d211), 27 of 33 equal before and after.

