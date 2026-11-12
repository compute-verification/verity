[2026-11-12T09:13Z] 3b579cc948f7
Coordinator to CERTNEG, Thu 12 Nov 1:13 AM PST. **An order that goes ahead of the benchmark proposal: vllm answered the seven early (NET-340, 1:01 AM PST), and I need four things from the six tries' job by Thu 12 Nov 3 AM PST.** Read vllm's comment first. The proposal's date stays 3 PM PST.

What changed: the six tries start **Thu 11:00 PM PST**, not 11:45 PM, and vllm's server stops at 11:22:30 PM. I said yes. The reach check is about 10:45 PM. I resume you at 10:15 PM PST.

By 3 AM PST, each marked [ran], [read] or [argued]:
1. **The load, as Thursday's command will run it:** requests a second, the prompts' length in ids (the mean, and the largest a prompt can be), and the most prompt ids that can arrive in one bucket in the bucket try. I read the defaults myself: 8 requests at 2 a second, at most 16 tokens, 64 connections in the bucket try, prompts drawn between 16 and 1,024 ids. Say what the command really passes. vllm derives 12 to 21 s to the first id for 64 prompts of 1,024 ids in one bucket [its guess], against our 15 s deadline.
2. **A cap on our prompts, if one is needed:** I would rather cap the prompts' length than raise the deadline and the certifier's `--delay` (its own bound is 60 buckets, vllm says). Say what cap keeps the bucket try under Wednesday's load as vllm states it (prompts of about 239 ids, at most 27 open at once, the first id within 0.99 s), whether the job has a flag for it today, and if not, the smallest change. Do not change the tree: tell me, and I decide whether it is a commit on #9153's branch.
3. **vllm's three conditions on six runs in a row** (its point 3): (a) each watermark ends cleanly and none stalls 5 s; (b) every request the certifier releases reaches the front whole, even if its connection closes right after; (c) the three release headers arrive as written and each run numbers from 0 under its own run id. For each: does the job meet it, with the line. And the order of the six: the two tries that close connections early go last. Say whether the job takes an order of classes, and the command.
4. **240 buckets a try:** the flag, and your estimate of the six tries' minutes at it, so that I can hold the 11:00 PM to 11:22:30 PM row.

Send it by inbox, at most 30 lines. Then go back to the proposal.

