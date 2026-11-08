#8047: all six re-pinned rows are NEUTRAL at its merged head c314a1fca7f, checked by neutrality without `--update` (ci's review-1). #8047 is re-marked ready at c314a1fca7f by kernels.

Each run rebuilt the row with the pinned vLLM (0.28.1rc1.dev472+gd9105ea80) and compared every rank's step, workload and request Program with its pin: build rc 0, no differences, `manifests/records.json` unchanged.

| row | run | node | ranks | verdict |
| --- | --- | --- | --- | --- |
| GLM-5.3 b1 greedy | r20261108-154630-9af5 | cpu-9 | 8 (1 request shape each) | NEUTRAL |
| GLM-5.3 b8 greedy | r20261108-155521-6f08 | cpu-6 | 8 (8 request shapes each) | NEUTRAL |
| GLM-5.3 b8 stochastic | r20261108-154630-b66d | cpu-9 | 8 (8 request shapes each) | NEUTRAL |
| GLM-5.3 b32 | r20261108-154723-2b01 | cpu-2 | 8 (32 request shapes each) | NEUTRAL |
| Qwen3-0.6B RTX PRO 6000 TP1 | r20261108-154630-b7f8 | cpu-11 | 1 | NEUTRAL |
| Qwen3-14B RTX PRO 6000 TP2 | r20261108-154630-ea33 | cpu-11 | 2 | NEUTRAL |

The script is the b32 re-pin's `neutral.sh` with `--update` dropped (`neutral-check.sh` on cursor/kernels-kstruct-jobs-3818). On these nodes `pod_bootstrap.sh`'s last step, the B0 SmolLM2 checkpoint, fails: it reads the node-wide `/workspace/hf` cache, outside `HF_HOME`. That step decides no Program digest; each row's own checkpoint metadata passed. The PR body has the same table.
