[2026-11-11T01:39Z] c843f536164a
From grade (TRN-89), by reading only, no run. Your question of 5:08 PM has an answer, and it is none of the three as asked.

Carried state (308,510,720 fresh, 229,376 stored) and the gradient's cut (308,396,032 fresh, 154,255,360 stored) differ for one cause: the PROGRAM'S TEXT at Muon's momentum buffer, met by a rule of the capture's flow.
- The stored program (islands/shallow/train.py, blob 4ee640b0) writes its fp32 buffer as `b.copy_(b.float().mul_(m).add_(p.grad))`. For an fp32 buffer `b.float()` is `b`, so the mul_ and add_ are already in place and the call is `b.copy_(b)`.
- The flow reads an overwriting `copy_` of a whole storage as reading nothing of its target (structure.py, `_OVERWRITE`, the line `if not (overwrite and a in mutated and self._whole(a))`). It tests the argument's node, not its position, so a source that is the target is skipped too: the buffer's new value has no predecessor.
- So in te_core 4.12.11's trace the buffer is rewritten from nothing each step. It is not carried (state is AdamW's moments on the 3,584 gains alone, 229,376), and the gradient first meets a carried value at `p.add_(u)`, past Newton-Schulz's bf16 cast: 16 x 9,633,792 + 32 x 3,584 = 154,255,360.
- This tree's route writes `buf = st["buf"].mul_(m).add_(p.grad)`, no copy_: the buffer is carried (state + 32 x 9,633,792) and the gradient meets it at fp32 (32 x 9,637,376 = 308,396,032).
- The stored program's own count of what it carries (`v1_state`: fp32 gradient, fp32 Muon buffer, AdamW on gains) gives 308,396,032 and 308,510,720: the fresh figures, to the bit.

Not device dtype: the buffer is fp32 on either device, and the stored trace was itself recorded on a CPU node (te_core's recorder keeps the program's bf16 autocast on CPU fakes; this route turns autocast off off-cuda, which is why a is 16,384 not 8,192 and u(1) 308,396,032 not 154,255,360). Not mid_run: the figures are equal with and without it (I ran both). The token count (65,536 against 32,768 a step) is the job's path: the stored record's counter counted one micro-batch of a step's two.

A GPU job at this head: no, by reading. State stays 308,510,720 and the gradient's cut 308,396,032 (two bf16 partials cost 32 bits an element as well). a, u(1) and the gate counts would move to bf16's.

What this means for the figures of record: 2,298.49 and 114.92 rest on a trace that leaves Muon's buffer out. They are beside rho and never in S, so S does not move. I change nothing: the flow rule is in this tree too (structure.py line 505) and fixing it (skip by position, not by node) is a capture change you would have to ask for. All of this goes in the hand-off note. The PR opens next; its figures are unchanged.

