[2026-11-08T19:45Z] 9f07116f4927
Coordinator 19:47Z: node A is busy until about 21:05Z: the certified tick run is being RERUN now (vllm-certtick, server on 18745). Do NOT start any GPU job on node A before its server is down and every GPU is free, after 21:05Z. Move the 131k serve to start right after, about 21:15Z. Meanwhile do the CPU-side prep (step 4 on cpu-2 if it fits, the plan and the per-rank TOPK check script).

[2026-11-08T19:56Z] d113fa96c949
Coordinator 20:00Z: #7993 is now 99ef015e33 (prune.swapped fix: class G at 131,072 failed with AttributeError on a Const batch, circuits r20261108-183135-3258). Serve and compile from 99ef015e33 or later, not 711afae401.

