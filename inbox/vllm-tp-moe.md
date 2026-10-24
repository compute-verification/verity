# vllm-tp-moe inbox (coordinator, 18:05Z)

GPU: infra allocated n1 18:30-03:00Z, GPUs 0-3 vllm; you have 2 of them (gpu-lease 2 through the slot pool). n1 deadline/lease 03:00Z. If you ended your turn because n1 was going down, the coordinator resumes you.

## 18:10Z (coordinator): vy-nebius-3 is ready NOW
n3 has vLLM venv /workspace/jobs/venv312 (vllm 0.28.1rc1.dev472+gd9105ea80, torch 2.13.0+cu129) and /workspace/hf/hub with
Llama-3.2-1B, Llama-3.1-8B, verity-int8 Llama-3.1-8B-INT8 (+LMHEAD), Llama-3.2-1B-INT8, OLMoE-1B-7B-0924 and its INT8.
Its 8 GPUs are open to vllm spillover (gpu-lease as usual) until 03:00Z, so you can start GPU work there now rather than
waiting for n1 at 18:30Z. Point your launch at that venv and HF_HOME=/workspace/hf.
