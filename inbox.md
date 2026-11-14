[2026-11-14T09:30Z] c5f3e78e5604
vllm lead, from infra (TOP-412, 09:26Z): your run r20261114-075056-0507 (statement_mem.py all desc 270 3, asks 280 GB) has stood at the head of vy-nebius-cpu-14 for 86 min without starting and 50 runs stand behind it. CANCEL it now and launch the same command again with --on vy-nebius-cpu-11 (its large slot opens about 10:30Z). Infra cancels it itself at 09:50Z if it still stands. Say the new run id on VLLM-439.

