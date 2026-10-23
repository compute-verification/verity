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
