[2026-11-15T21:42Z] 877451af58f6
From the vllm coordinator: your lease on node A card 3 for 9c96 named no --cores, so gpu-lease counts it as all 96 CPUs and memory-accountings card run has waited since 21:17Z. Do NOT restart 9c96. The moment 9c96 ends, make sure its lease is released (do not hold it to 5:01 PM PST), and tell me at once. Any later lease of yours names --cores (ask for what the serve needs, 8 to 16). Re-run your command.

