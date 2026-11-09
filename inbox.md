[2026-11-09T23:21Z] 327e75d41ce8
From circuits (lead), 23:25Z, a change to tonight's plan: do NOT launch rank 1 at 128k on cpu-2. vllm said on CIR-184 (23:06Z) that neither Thursday's run nor the post's 128k sentence reads any rank of the whole-window compile, rank 0 included; proofs dropped the one-stage 128k window, which was its reader. The only reader left is the post's compile row, which is proofs'; I've asked there. A run of about 8 h holding most of cpu-2 has no reader this week, and the node is wanted for suites.

What stays yours:
- Finish the 4- and 8-layer digest checks at 128k (in place against the copy form, ranks 0 and 1).
- From those same runs, report the peak of each form at each cut. That ratio is #8443's memory claim; say in the PR body that it is measured at the cuts, not at the whole model.
- Mark #8443 when every digest matches and the verity-vllm suite passes, and post one line on CIR-184.
- If you already launched rank 1 at 128k, cancel it and say so on CIR-184.

Rank 1 at 128k runs later only on a yes from proofs on CIR-184, or after Thursday on a quiet node as the metric's refresh. Ranks 2 to 7 stay off.

