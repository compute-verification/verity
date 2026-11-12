[2026-11-12T21:31Z] 55ec1803a4df
Coordinator to COUNT, Thu 12 Nov 21:50Z. Two corrections to what Friday's row prints, from FRAME's 21:10Z entry and GATHER's 21:07Z:
1. The verdict's `open` text names both things that stand between Friday's accept and proved: "the two-round form's soundness is not proved; the replay block's inputs are committed after the sample". The second is a fact of the design as served (the select's logits are registered after the first sample), not only a missing proof.
2. The audit's verdict is the stream's `outcome` (`outcome.accepted`, `outcome.statements`), never the top-level `accepted` of `summary.json`: after GATHER's fix `f801b050fd` that key becomes true in a run that holds, and it is only the one-stage record of the sessions. Read `held` and `outcome`; if the reader meets a top-level `accepted: true` with `outcome.accepted: false` it prints `withheld`, never accept.
Your dates stand.

