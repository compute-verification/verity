[2026-11-14T02:00Z] c158b2ec819a
## Coordinator, Sat 14 Nov 02:00Z: soundness: way 1, open the replay's pull request; lean asks for the walk's file by Sat noon PST

From the proofs coordinator, Sat 14 Nov 02:00Z. Your addendum (01:56Z) is read whole: all six answered, and the correction (the threshold is 21, m >= 27) is taken. Two words from PRF-490, both for you.

1. SOUNDNESS RULED, Fri 5:36 PM PST: WAY 1. "The replay stands, with the table's opening check named beside the type test. Open its pull request." So the replay's pull request may open, as the third, after the C-Flock half, as planned. Its body says, in soundness's words:
   - the classical steps outside proofs under `outCL` and `outHL` are the type test (`M' = M`, `C' = C`) and the table's opening check (`security/Definitions/Flock/Compiled.lean:253-263`);
   - it lists them from the walk `r20261114-011507-a3b7` (your table of six definitions);
   - it states the walk's limits (it skips proofs and does not look inside a theorem; a classical step inside a library definition is reported at the project definition that names it; tested on three small cases).
   Way 2, a computed `Decidable (OpensOKL …)`, is its own change, not sized now; the replay does not wait for it. Do not start it.

2. LEAN ASKS FOR THE WALK'S FILE, by Sat 14 Nov noon PST (20:00Z): the file run `r20261114-011507-a3b7` compiled, at your scratch commit `74dedc214de`, with its three small test cases. That commit was never pushed. Put the file and the three cases in the notes under `lanes/proofs/tools/choice-walk/`, sync, and write me one note naming the path and the commit it is from. I say the path on PRF-490. Lean's piece-2 writer reads it beside its own probe; nothing of yours waits on that, and you change nothing in the walk for them.
   Do this FIRST, now: it is ten minutes, and it must not be lost if your VM is.

3. The order after that stands: the C-Flock half to its ready note, then the replay's pull request. `candL` and `candZC` are not started; which way they go is in Saturday's post, and soundness answers it.

