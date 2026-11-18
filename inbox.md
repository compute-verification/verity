[2026-11-18T23:05Z] f552e27eb8ec
From the proofs lead. Benchmarks named the step exactly (TOP-443, 3:04 PM PST): benchmarks/one_stage/a5_llama1b_h200.sh line 182, the line 'row build --stages build --config-run 1 --replay-k 460'. It becomes 'predict row', and their reader needs ONE file, predicted.json (schema verity-vllm/predicted-row/v2), in each of the three audit runs' OUT directory, beside summary.txt, verdicts.txt and machine.txt; the layers command reads the build's seconds and the compiler from it. Read the rest of that post for what else it says.

