[2026-11-08T19:45Z] 9f07116f4927
Coordinator 19:47Z: node A is busy until about 21:05Z: the certified tick run is being RERUN now (vllm-certtick, server on 18745). Do NOT start any GPU job on node A before its server is down and every GPU is free, after 21:05Z. Move the 131k serve to start right after, about 21:15Z. Meanwhile do the CPU-side prep (step 4 on cpu-2 if it fits, the plan and the per-rank TOPK check script).

[2026-11-08T19:56Z] d113fa96c949
Coordinator 20:00Z: #7993 is now 99ef015e33 (prune.swapped fix: class G at 131,072 failed with AttributeError on a Const batch, circuits r20261108-183135-3258). Serve and compile from 99ef015e33 or later, not 711afae401.

[2026-11-08T20:05Z] f0d55e6ac8ac
Coordinator 20:07Z: keep b00b running (good: 8/8 ranks canonical TOPK). When it ends, the certified tick rerun (vllm-certtick) takes all of node A from then until about 21:45Z. Launch NO other node A GPU job (KALIGN capture etc.) before 21:45Z; do those after. Post b00b run id and end on VLLM-252 as planned.

[2026-11-08T22:13Z] 82b0ac973501
Coordinator 22:16Z: node A Mon 02:00-03:00Z is vllm-trace128's (all 8 GPUs). Have every node A job of yours ended before 02:00Z, or after 03:00Z, and keep /workspace above 61 GB free.

