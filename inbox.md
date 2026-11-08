[2026-11-08T04:42Z] e5f221cfd516
Coordinator 04:44Z, a CORRECTION to the GPU for run.sh's stress step: the H200 behind r20261108-021638-08e3 was KALIGN's own pod and is terminated. Use ONE GPU of node A (vy-spot-glm53) instead, through the tree's gpu_lease.sh 1 --wait (research run --gpus is refused there). Allowed only outside Sun 08:00-09:30Z (proofs) and finished by Sun 17:30Z (vllm's certified tick setup). Don't write more than a few GB to its disk (120 GB free; GLM-5.3's checkpoint must stay). The 8-GPU DSV4-Pro validation still waits for a new node.

[2026-11-08T04:45Z] 0494a8a434b1
Coordinator 04:45Z, node A window update (proofs): all 8 GPUs are reserved Sun 08:00-10:00Z (LIFETIME, then OVERHEADDEF's check), and any job there must FINISH or stop by 07:55Z. So run the one-GPU stress step either now, ending by 07:55Z, or between 10:00Z and 17:30Z.

[2026-11-08T15:09Z] c04d5ef15649
Coordinator 15:10Z: infra will give vllm vy-glm53-h8 (8x H200, Nebius) for about 3 h from about 20:10Z tonight, with a 1.2 TB data disk for DSV4-Pro (pull it there; do not touch GLM-5.3 on /workspace). Plan for it: the REAL TP8 served validation (per-rank calls, per-layer equality, decode deltas) plus the DSV4 records.json re-record for circuits #8024. Stage the pull and job commands, but launch only when infra posts the node is ours; I will resume you then. Your TP2 dummy-weight run on node A still goes ahead if it ends by 18:30Z.

