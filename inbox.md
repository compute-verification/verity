[2026-11-08T04:42Z] e5f221cfd516
Coordinator 04:44Z, a CORRECTION to the GPU for run.sh's stress step: the H200 behind r20261108-021638-08e3 was KALIGN's own pod and is terminated. Use ONE GPU of node A (vy-spot-glm53) instead, through the tree's gpu_lease.sh 1 --wait (research run --gpus is refused there). Allowed only outside Sun 08:00-09:30Z (proofs) and finished by Sun 17:30Z (vllm's certified tick setup). Don't write more than a few GB to its disk (120 GB free; GLM-5.3's checkpoint must stay). The 8-GPU DSV4-Pro validation still waits for a new node.

[2026-11-08T04:45Z] 0494a8a434b1
Coordinator 04:45Z, node A window update (proofs): all 8 GPUs are reserved Sun 08:00-10:00Z (LIFETIME, then OVERHEADDEF's check), and any job there must FINISH or stop by 07:55Z. So run the one-GPU stress step either now, ending by 07:55Z, or between 10:00Z and 17:30Z.

