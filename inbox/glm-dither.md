# To bc-ae7e573d (vllm-glm-dither), from the vllm coordinator, 19:35Z 23 Oct

Path A is granted: circuits gave vllm vy-glm53-eu2 (8xH200) on their vy-glm53- budget line until 23:45Z. Switch to the full GLM53_INT8 v3 model at TP8 on vy-glm53-eu2 now. Keep path B (truncated on vy-nebius-2) only as a fallback if eu2 won't come up or the checkpoint copy can't finish by about 22:00Z.

Circuits' three conditions, all binding:
1. When you start the node, set its /etc/research/deadline to 23:45Z (never later than 00:00Z), so vm_stop's idle and deadline rules stop it if your run ends early.
2. Keep at least 300 GB of its /workspace free the whole time, so circuits can stage B8 stochastic afterwards. If the 756.5 GB checkpoint plus 300 GB doesn't fit, stop and tell the coordinator before copying. Don't delete anything that's on the node already.
3. When you're done with it, say so in your reply: is it stopped, or left up for circuits?

If an 8xH200 comes up for circuits before 23:45Z, they take that one and don't touch eu2. Nothing changes for you.

## 19:40Z, from infra
- Two idle rules watch eu2, infra's relay and vm_stop on the pod. Each stops the node after 2 h with no GPU use, and a held lease counts only within 2 h of the last GPU use. If copying the 756.5 GB checkpoint from vy-nebius-cpu-1 would take longer than 2 h before the GPUs start, tell the coordinator (in a checkpoint note or by ending your turn) before you start the copy, and infra will raise eu2's idle window.
- eu2 has had no capacity since 22 Oct. If the start is refused, end your turn saying so; infra's retry loop can try it. Meanwhile, carry on with path B on vy-nebius-2.

## 20:05Z, from kernels' DH200 run (r20261023-195535-8920, on an H200 at GLM53_INT8 v3 shapes)
- H1's domain is K >= 384 with 512 | K. At TP8, the routed experts' w2 and the shared expert's down projection are 6144 x 256 (K = 256), and H1 refuses them. On eu2, serve the experts with expert parallelism (vLLM `enable_expert_parallel`), which makes routed w2 6144 x 2048, inside the domain. Report any call still outside the domain (likely the shared expert's down projection under TP8) as a finding with its shape. Don't drop it silently.
- `integrations/vllm/verity_vllm/engine/int8_gemm_configs.py` has no sm_90 rows for GLM, so pick() falls back to vLLM's tiles. That's fine for exactness; leave it.
- On vy-nebius-2 at TP4, K = 512 for those projections, so the truncated path is inside the domain.
- Kernels may also want an 8xH200 for a served-step gamma. vllm holds eu2 until 23:45Z; if kernels asks to ride along, the coordinator will tell you here.

## 20:08Z, an ask from kernels for the eu2 run (secondary to the tie check)
Once GLM53_INT8 serves on eu2 at TP8 with expert parallelism, record decode TPOT at T64 (64 requests in flight) twice: (1) exact int8 with no PoUW (no hooks, no Commit); (2) with H1's hooks (Commit plus z/y records, receipts off). Same prompts and output length for both, and say what they were. These are the two baselines Dither's GLM number divides by. If time runs short, the tie check comes first.

## 20:15Z, CORRECTION to the 20:05Z note: do NOT use expert parallelism
`Int8ExpertsMethod` (`engine/int8_vllm.py`) refuses EP today: "experts not all on this rank (no expert parallelism)". Serve eu2 at plain TP8 (no `enable_expert_parallel`). At TP8 the routed experts' w2 and the shared expert's down projection are 6144 x 256 (K = 256), outside H1's domain (K >= 384, 512 | K). Daniel is deciding the fix (card 82c1e288: EP8, widen Pearl-C's small-k cap, or leave w2 uncredited). Tonight, record those calls as outside the domain: give their count and shapes per step, keep them out of the H1 window, and tie-check the rest (attention and dense projections, routed w13 at K = 6144). The TPOT ask is unchanged, at TP8.

## 21:05Z, from the coordinator: move to eu2 now
Your truncated serve r20261023-203316-7735 is waiting for 300 GB of vy-nebius-2's host memory, and nothing else contends for eu2. Start vy-glm53-eu2 now (deadline file 23:45Z, keep 300 GB of /workspace free) and run the full model there at TP8 with --no-batch-invariant. Its sparse-MLA backends on sm_90 refuse batch invariance too. Cancel the n2 serve once eu2 is serving. If eu2 is refused for capacity, keep the n2 serve and say so.

## 21:10Z, from infra
Your shards in vy-nebius-2's /var/tmp/vllm-glm-dither (39 GB) sit on its 247 GB root disk (64% used). They're fine for tonight's truncated run, but delete them when it ends (or as soon as eu2 is serving). Put anything bigger under /workspace: a full root disk takes the node's services down.
