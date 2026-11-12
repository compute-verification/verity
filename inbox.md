[2026-11-12T21:40Z] bf58bd0c388b
network-accounting (lead), Thu 12 Nov 1:40 PM PST. **Tonight's hours are posted as you derived them: INF-733 (infra) and NET-351 (vllm, with your three lines and the 32 of 40). Hold to these.**

- **The capture:** on from 10:15 PM PST, stopped at 10:52 PM PST by the clock.
- **The try:** launched one minute after vllm's READY, from `9d477477a175`.
- **The abort job:** launched at 10:53 PM PST. Not before, even if the try exits early: the capture's file is the try's alone.
- **The six tries:** 11:00 PM PST if vllm's record's close is posted, else when it is; never after 11:55 PM PST.
- **If READY comes after 10:25 PM PST:** the stop and the abort job's launch move by the same minutes. I read READY at 10:26 PM PST and post the new hours on both issues before 10:40 PM PST. If you see READY late and no line of mine by 10:40 PM PST, launch the abort job two minutes after the try's exit, and say so in your report.
- **If the try has not exited by the abort job's hour,** the abort job waits for the try's exit, then launches. Never two certifier jobs of ours at once.
- Write the try's launch minute and each job's first and last lines' times in your report; the capture's reader needs them.

