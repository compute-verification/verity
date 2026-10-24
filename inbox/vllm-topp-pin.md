
## 09:25Z (coordinator): vy-nebius-1 is slot-only since 09:18Z
A new research launch on n1 must go through its GPU slot pool, or research.remote refuses it:
`env CHECK_SLOTS=/workspace/research/locks/gpu/slots python3 tools/verity/check/slot.py --by vllm -- <cmd>`
(4 slots g0-g3, cores 160-191; gpu-lease inside works as before). n3s 8 GPUs are idle: prefer `--on vy-nebius-3`
with gpu-lease for new GPU runs. n2 stops at 15:00Z.
