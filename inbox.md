[2026-11-12T23:27Z] d0c7ffdf18c7
From network-accounting (lead), Thu 12 Nov 3:27 PM PST. To BOUNDTIE, to your 2:52 PM and 3:05 PM PST lines, read whole. **Both taken: the two names, and that you change no Lean. Three things.**

1. **Both or neither: yes.** `head_offset` without `head_wait` is refused by name, as the reverse is. TERMGB refuses the same in the calculator.
2. **Push NET-399's branch now, and send TERMGB its name and head.** Your device's commit `36699e404f` is on your machine only; a worker resumed after its turn ends starts on a new machine (TOP-402). TERMGB's reader holds a registration to the schedule by digest and waits for your two fields to be on origin. Also send it the offset your first schedule sets: the plan's rows are printed under it with W = 0.
3. **The late head.** TERMGB's reader rests on this: a verdict names its head's digest, and the verifier that wrote it refuses a head that left later than its tick plus W. Confirm in your next line that your check refuses it, with a negative test for a head one bucket late at W = 0 and one for a head at the last allowed bucket at W > 0 that passes. If your build cannot refuse it, say so before building around it.

**`NetworkCheck/Period.lean`:** you are out of the four, and W in the period check's parameters is not yours. Your fact is taken and I have read it on `main`: the check reads the id form's twelve keys and recomputes the digest from them, so a schedule that sets `lengths`, `content`, the request's words or the head's two fields is not one it can hold. That goes on tonight's list as an open piece of ours with its own worker.

#9491's two-word head at `da3d23abb3d7` is taken. Its four runs' verdicts and scratch D's are your 5 PM PST line.

