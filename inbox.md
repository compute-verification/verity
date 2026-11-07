[2026-11-07T21:25Z] a67e29a5a3ac
Proofs coordinator 21:25Z: when your GPU run r20261107-202812-0892 ends and releases node A's glm53-sessions hold (planned 21:33Z), append "STATIONARY END <time>" to inbox/proofs-coordinator.md right away: vllm's 128k serve is waiting for it. Do not extend past 21:35Z.

[2026-11-07T21:30Z] 7050bc5c2081
Proofs coordinator 21:35Z: thanks for the report. Node A is booked solid until about 02:40Z (128k serve, replay, vllm's tick sweep), and the overlap measurement (k = 1, 2, 4, 8 provers against one verifier) is a core lever. So create your own 1x H200 RunPod pod now: `research pods create` with --max-hours 2 under the proofs budget line, named vy-spot-overlap. Stage on CPU while it boots, run the overlap sweep, and let the pod stop itself. Post the pod id and results in inbox/proofs-coordinator.md. If RunPod has no 1x H200 stock, say so and take node A 02:40Z to 03:20Z instead.

[2026-11-07T22:01Z] af6a50275b59
Proofs coordinator 22:05Z, add to your pod run (after overlap, before the cache-on run if time is short): a PLAIN arm (flock-circuit prove without --zk, FC_COINS=tree) on the same types at k = 1, beside dev. HILLCLIMB found plain covers SPOT's one-table stratified sessions; the H200 share --zk costs per type is the number Daniel needs to decide whether SPOT runs plain. HILLCLIMB's zk_cost.py plain arm is @ 13373d71a1 on its branch.

