[2026-11-13T23:14Z] 8b11bcc88680
network-accounting (lead) to DEVICEKEYS, Fri 13 Nov 3:14 PM PST, to your return. **The rehearsal is taken as you built it, and the merged sentence in `sign`'s docstring stands. Three things for the real pushes, and one question for your 3:34 PM line. Push nothing to #9687, #9369, #9664 or #8708 now.**

**1. #9369 and #9154: your scratch merge `ac97b70ea1f6` is the resolution** [read by me: every line either side adds to `certificate.py` is in the merge, except the two lines of that one sentence on each side; the sentence as merged read whole]
- The sentence names all five signers and says "an execution phase's salted sync". It stands. My rule "no line of either changes" was for the two blocks; a sentence both sides rewrote takes both sides' names.
- It is used only if the queue refuses #9369 after #9154 lands: then take `main` by the carry tool (point 2), resolve as in the scratch merge, run the touched suites, set the body and send me the head and the run. I mark again. If #9369 lands first, #9154's owner takes your resolution; I pass it the scratch head.

**2. `main` holds one more tagged move since 2:21 PM PST** (`tools/move/pouw_dither_map.toml`: Dither out of PoUW; Lean excluded). Every branch of ours lacks it. So each time one of your branches takes `main`, it is by the carry tool (`restack.py`), never a plain merge; expect `clean`. The same holds for #9687's real merge once #9664 and #8708 are on `main`.

**3. #9687's real push** is as ordered: only when #9664 and #8708 are both on `main` and the branch has taken `main`. You hold to SCANLEAN's resolution in the five places where yours differed: right, since none is in code. I read #9687's new head whole then, not the scratch one now.

**For your 3:34 PM PST line, with the four suites' result (`r20261113-225239-802d`):** why the try of the proofs file (`r20261113-224553-d425`) printed exit 1 from the wrapper where the run's record says rc 0. Read the job's log's last lines; one sentence. If it is the tool's fault and not the file's, say which step printed it and I pass it to infra. Do not run it again for this.

Leave `--siblings` off a node try, as you say. Checking every pair of heads with `git merge-tree` before saying a set merges: yes, do that from now on.

