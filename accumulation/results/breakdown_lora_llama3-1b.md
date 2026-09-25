### lora @ llama3-1b, tokens=4096, seq=4096 (r=16 ad=qkvogud)

| label | scope | M | N | K | copies | ops | MACs | MACs/token | share | ADW |
|---|---|---|---|---|---|---|---|---|---|---|
| wt-fwd[Wg] | LoraBlock | 4096 | 8192 | 2048 | 1 | 16 | 1.10e+12 | 2.68e+08 | 5.7% | yes |
| wt-fwd[Wu] | LoraBlock | 4096 | 8192 | 2048 | 1 | 16 | 1.10e+12 | 2.68e+08 | 5.7% | yes |
| wt-fwd[Wd] | LoraBlock | 4096 | 2048 | 8192 | 1 | 16 | 1.10e+12 | 2.68e+08 | 5.7% | yes |
| wt-fwd-recompute[Wg] | LoraBlockBwd | 4096 | 8192 | 2048 | 1 | 16 | 1.10e+12 | 2.68e+08 | 5.7% | yes |
| wt-fwd-recompute[Wu] | LoraBlockBwd | 4096 | 8192 | 2048 | 1 | 16 | 1.10e+12 | 2.68e+08 | 5.7% | yes |
| wt-fwd-recompute[Wd] | LoraBlockBwd | 4096 | 2048 | 8192 | 1 | 16 | 1.10e+12 | 2.68e+08 | 5.7% | yes |
| wt-dgrad[Wd] | LoraBlockBwd | 4096 | 8192 | 2048 | 1 | 16 | 1.10e+12 | 2.68e+08 | 5.7% | yes |
| wt-dgrad[Wg] | LoraBlockBwd | 4096 | 2048 | 8192 | 1 | 16 | 1.10e+12 | 2.68e+08 | 5.7% | yes |
| wt-dgrad[Wu] | LoraBlockBwd | 4096 | 2048 | 8192 | 1 | 16 | 1.10e+12 | 2.68e+08 | 5.7% | yes |
| attn-bwd | LoraBlockBwd/AttnSeqBwd | 4096 | 64 | 1.64e+04 | 8 | 32 | 1.10e+12 | 2.68e+08 | 5.7% | yes |
| wt-fwd[Wh] |  | 4096 | 1.28e+05 | 2048 | 1 | 1 | 1.08e+12 | 2.63e+08 | 5.6% | yes |
| wt-dgrad[Wh] |  | 4096 | 2048 | 1.28e+05 | 1 | 1 | 1.08e+12 | 2.63e+08 | 5.6% | yes |
| attn-fwd | LoraBlock/AttnSeq | 4096 | 4096 | 64 | 32 | 16 | 5.50e+11 | 1.34e+08 | 2.9% | yes |
| attn-fwd | LoraBlock/AttnSeq | 4096 | 64 | 4096 | 32 | 16 | 5.50e+11 | 1.34e+08 | 2.9% | yes |
| attn-fwd-recompute | LoraBlockBwd/AttnSeq | 4096 | 4096 | 64 | 32 | 16 | 5.50e+11 | 1.34e+08 | 2.9% | yes |
| attn-fwd-recompute | LoraBlockBwd/AttnSeq | 4096 | 64 | 4096 | 32 | 16 | 5.50e+11 | 1.34e+08 | 2.9% | yes |
| attn-fwd-recompute | LoraBlockBwd/AttnSeqBwd | 4096 | 4096 | 64 | 32 | 16 | 5.50e+11 | 1.34e+08 | 2.9% | yes |
| attn-bwd | LoraBlockBwd/AttnSeqBwd | 4096 | 4096 | 64 | 32 | 16 | 5.50e+11 | 1.34e+08 | 2.9% | yes |
| attn-bwd | LoraBlockBwd/AttnSeqBwd | 4096 | 64 | 4096 | 32 | 16 | 5.50e+11 | 1.34e+08 | 2.9% | yes |
| wt-fwd[Wq] | LoraBlock | 4096 | 2048 | 2048 | 1 | 16 | 2.75e+11 | 6.71e+07 | 1.4% | no |
| wt-fwd[Wo] | LoraBlock | 4096 | 2048 | 2048 | 1 | 16 | 2.75e+11 | 6.71e+07 | 1.4% | yes |
| wt-fwd-recompute[Wq] | LoraBlockBwd | 4096 | 2048 | 2048 | 1 | 16 | 2.75e+11 | 6.71e+07 | 1.4% | yes |
| wt-fwd-recompute[Wo] | LoraBlockBwd | 4096 | 2048 | 2048 | 1 | 16 | 2.75e+11 | 6.71e+07 | 1.4% | yes |
| wt-dgrad[Wo] | LoraBlockBwd | 4096 | 2048 | 2048 | 1 | 16 | 2.75e+11 | 6.71e+07 | 1.4% | yes |
| wt-dgrad[Wq] | LoraBlockBwd | 4096 | 2048 | 2048 | 1 | 16 | 2.75e+11 | 6.71e+07 | 1.4% | yes |
| wt-fwd[Wk] | LoraBlock | 4096 | 512 | 2048 | 1 | 16 | 6.87e+10 | 1.68e+07 | 0.4% | no |
| wt-fwd[Wv] | LoraBlock | 4096 | 512 | 2048 | 1 | 16 | 6.87e+10 | 1.68e+07 | 0.4% | no |
| wt-fwd-recompute[Wk] | LoraBlockBwd | 4096 | 512 | 2048 | 1 | 16 | 6.87e+10 | 1.68e+07 | 0.4% | yes |
| wt-fwd-recompute[Wv] | LoraBlockBwd | 4096 | 512 | 2048 | 1 | 16 | 6.87e+10 | 1.68e+07 | 0.4% | yes |
| wt-dgrad[Wk] | LoraBlockBwd | 4096 | 2048 | 512 | 1 | 16 | 6.87e+10 | 1.68e+07 | 0.4% | yes |
| wt-dgrad[Wv] | LoraBlockBwd | 4096 | 2048 | 512 | 1 | 16 | 6.87e+10 | 1.68e+07 | 0.4% | yes |
| wt-fwd[Bg] | LoraBlock | 4096 | 8192 | 16 | 1 | 16 | 8.59e+09 | 2.10e+06 | 0.0% | yes |
| wt-fwd[Bu] | LoraBlock | 4096 | 8192 | 16 | 1 | 16 | 8.59e+09 | 2.10e+06 | 0.0% | yes |
| wt-fwd[Ad] | LoraBlock | 4096 | 16 | 8192 | 1 | 16 | 8.59e+09 | 2.10e+06 | 0.0% | yes |
| wt-fwd-recompute[Bg] | LoraBlockBwd | 4096 | 8192 | 16 | 1 | 16 | 8.59e+09 | 2.10e+06 | 0.0% | yes |
| wt-fwd-recompute[Bu] | LoraBlockBwd | 4096 | 8192 | 16 | 1 | 16 | 8.59e+09 | 2.10e+06 | 0.0% | yes |
| wt-fwd-recompute[Ad] | LoraBlockBwd | 4096 | 16 | 8192 | 1 | 16 | 8.59e+09 | 2.10e+06 | 0.0% | yes |
| wt-dgrad[Ad] | LoraBlockBwd | 4096 | 8192 | 16 | 1 | 16 | 8.59e+09 | 2.10e+06 | 0.0% | yes |
| wt-wgrad[Ad] | LoraBlockBwd | 16 | 8192 | 4096 | 1 | 16 | 8.59e+09 | 2.10e+06 | 0.0% | yes |
| wt-dgrad[Bg] | LoraBlockBwd | 4096 | 16 | 8192 | 1 | 16 | 8.59e+09 | 2.10e+06 | 0.0% | yes |
| wt-wgrad[Bg] | LoraBlockBwd | 8192 | 16 | 4096 | 1 | 16 | 8.59e+09 | 2.10e+06 | 0.0% | yes |
| wt-dgrad[Bu] | LoraBlockBwd | 4096 | 16 | 8192 | 1 | 16 | 8.59e+09 | 2.10e+06 | 0.0% | yes |
| wt-wgrad[Bu] | LoraBlockBwd | 8192 | 16 | 4096 | 1 | 16 | 8.59e+09 | 2.10e+06 | 0.0% | yes |
| wt-fwd[Aq] | LoraBlock | 4096 | 16 | 2048 | 1 | 16 | 2.15e+09 | 5.24e+05 | 0.0% | yes |
| wt-fwd[Bq] | LoraBlock | 4096 | 2048 | 16 | 1 | 16 | 2.15e+09 | 5.24e+05 | 0.0% | yes |
| wt-fwd[Ak] | LoraBlock | 4096 | 16 | 2048 | 1 | 16 | 2.15e+09 | 5.24e+05 | 0.0% | yes |
| wt-fwd[Av] | LoraBlock | 4096 | 16 | 2048 | 1 | 16 | 2.15e+09 | 5.24e+05 | 0.0% | yes |
| wt-fwd[Ao] | LoraBlock | 4096 | 16 | 2048 | 1 | 16 | 2.15e+09 | 5.24e+05 | 0.0% | yes |
| wt-fwd[Bo] | LoraBlock | 4096 | 2048 | 16 | 1 | 16 | 2.15e+09 | 5.24e+05 | 0.0% | yes |
| wt-fwd[Ag] | LoraBlock | 4096 | 16 | 2048 | 1 | 16 | 2.15e+09 | 5.24e+05 | 0.0% | yes |
| wt-fwd[Au] | LoraBlock | 4096 | 16 | 2048 | 1 | 16 | 2.15e+09 | 5.24e+05 | 0.0% | yes |
| wt-fwd[Bd] | LoraBlock | 4096 | 2048 | 16 | 1 | 16 | 2.15e+09 | 5.24e+05 | 0.0% | yes |
| wt-fwd-recompute[Aq] | LoraBlockBwd | 4096 | 16 | 2048 | 1 | 16 | 2.15e+09 | 5.24e+05 | 0.0% | yes |
| wt-fwd-recompute[Bq] | LoraBlockBwd | 4096 | 2048 | 16 | 1 | 16 | 2.15e+09 | 5.24e+05 | 0.0% | yes |
| wt-fwd-recompute[Ak] | LoraBlockBwd | 4096 | 16 | 2048 | 1 | 16 | 2.15e+09 | 5.24e+05 | 0.0% | yes |
| wt-fwd-recompute[Av] | LoraBlockBwd | 4096 | 16 | 2048 | 1 | 16 | 2.15e+09 | 5.24e+05 | 0.0% | yes |
| wt-fwd-recompute[Ao] | LoraBlockBwd | 4096 | 16 | 2048 | 1 | 16 | 2.15e+09 | 5.24e+05 | 0.0% | yes |
| wt-fwd-recompute[Bo] | LoraBlockBwd | 4096 | 2048 | 16 | 1 | 16 | 2.15e+09 | 5.24e+05 | 0.0% | yes |
| wt-fwd-recompute[Ag] | LoraBlockBwd | 4096 | 16 | 2048 | 1 | 16 | 2.15e+09 | 5.24e+05 | 0.0% | yes |
| wt-fwd-recompute[Au] | LoraBlockBwd | 4096 | 16 | 2048 | 1 | 16 | 2.15e+09 | 5.24e+05 | 0.0% | yes |
| wt-fwd-recompute[Bd] | LoraBlockBwd | 4096 | 2048 | 16 | 1 | 16 | 2.15e+09 | 5.24e+05 | 0.0% | yes |
| wt-dgrad[Bd] | LoraBlockBwd | 4096 | 16 | 2048 | 1 | 16 | 2.15e+09 | 5.24e+05 | 0.0% | yes |
| wt-wgrad[Bd] | LoraBlockBwd | 2048 | 16 | 4096 | 1 | 16 | 2.15e+09 | 5.24e+05 | 0.0% | yes |
| wt-dgrad[Ag] | LoraBlockBwd | 4096 | 2048 | 16 | 1 | 16 | 2.15e+09 | 5.24e+05 | 0.0% | yes |
| wt-wgrad[Ag] | LoraBlockBwd | 16 | 2048 | 4096 | 1 | 16 | 2.15e+09 | 5.24e+05 | 0.0% | yes |
| wt-dgrad[Au] | LoraBlockBwd | 4096 | 2048 | 16 | 1 | 16 | 2.15e+09 | 5.24e+05 | 0.0% | yes |
| wt-wgrad[Au] | LoraBlockBwd | 16 | 2048 | 4096 | 1 | 16 | 2.15e+09 | 5.24e+05 | 0.0% | yes |
| wt-dgrad[Bo] | LoraBlockBwd | 4096 | 16 | 2048 | 1 | 16 | 2.15e+09 | 5.24e+05 | 0.0% | yes |
| wt-wgrad[Bo] | LoraBlockBwd | 2048 | 16 | 4096 | 1 | 16 | 2.15e+09 | 5.24e+05 | 0.0% | yes |
| wt-dgrad[Ao] | LoraBlockBwd | 4096 | 2048 | 16 | 1 | 16 | 2.15e+09 | 5.24e+05 | 0.0% | yes |
| wt-wgrad[Ao] | LoraBlockBwd | 16 | 2048 | 4096 | 1 | 16 | 2.15e+09 | 5.24e+05 | 0.0% | yes |
| wt-dgrad[Bq] | LoraBlockBwd | 4096 | 16 | 2048 | 1 | 16 | 2.15e+09 | 5.24e+05 | 0.0% | yes |
| wt-wgrad[Bq] | LoraBlockBwd | 2048 | 16 | 4096 | 1 | 16 | 2.15e+09 | 5.24e+05 | 0.0% | yes |
| wt-dgrad[Aq] | LoraBlockBwd | 4096 | 2048 | 16 | 1 | 16 | 2.15e+09 | 5.24e+05 | 0.0% | yes |
| wt-wgrad[Aq] | LoraBlockBwd | 16 | 2048 | 4096 | 1 | 16 | 2.15e+09 | 5.24e+05 | 0.0% | yes |
| wt-dgrad[Ak] | LoraBlockBwd | 4096 | 2048 | 16 | 1 | 16 | 2.15e+09 | 5.24e+05 | 0.0% | yes |
| wt-wgrad[Ak] | LoraBlockBwd | 16 | 2048 | 4096 | 1 | 16 | 2.15e+09 | 5.24e+05 | 0.0% | yes |
| wt-dgrad[Av] | LoraBlockBwd | 4096 | 2048 | 16 | 1 | 16 | 2.15e+09 | 5.24e+05 | 0.0% | yes |
| wt-wgrad[Av] | LoraBlockBwd | 16 | 2048 | 4096 | 1 | 16 | 2.15e+09 | 5.24e+05 | 0.0% | yes |
| wt-fwd[Bk] | LoraBlock | 4096 | 512 | 16 | 1 | 16 | 5.37e+08 | 1.31e+05 | 0.0% | yes |
| wt-fwd[Bv] | LoraBlock | 4096 | 512 | 16 | 1 | 16 | 5.37e+08 | 1.31e+05 | 0.0% | yes |
| wt-fwd-recompute[Bk] | LoraBlockBwd | 4096 | 512 | 16 | 1 | 16 | 5.37e+08 | 1.31e+05 | 0.0% | yes |
| wt-fwd-recompute[Bv] | LoraBlockBwd | 4096 | 512 | 16 | 1 | 16 | 5.37e+08 | 1.31e+05 | 0.0% | yes |
| wt-dgrad[Bk] | LoraBlockBwd | 4096 | 16 | 512 | 1 | 16 | 5.37e+08 | 1.31e+05 | 0.0% | yes |
| wt-wgrad[Bk] | LoraBlockBwd | 512 | 16 | 4096 | 1 | 16 | 5.37e+08 | 1.31e+05 | 0.0% | yes |
| wt-dgrad[Bv] | LoraBlockBwd | 4096 | 16 | 512 | 1 | 16 | 5.37e+08 | 1.31e+05 | 0.0% | yes |
| wt-wgrad[Bv] | LoraBlockBwd | 512 | 16 | 4096 | 1 | 16 | 5.37e+08 | 1.31e+05 | 0.0% | yes |

By class:

| class | MACs | share |
|---|---|---|
| wt-fwd | 5.11e+12 | 26.5% |
| wt-dgrad | 5.11e+12 | 26.5% |
| wt-fwd-recompute | 4.03e+12 | 21.0% |
| attn-bwd | 2.20e+12 | 11.4% |
| attn-fwd-recompute | 1.65e+12 | 8.6% |
| attn-fwd | 1.10e+12 | 5.7% |
| wt-wgrad | 4.62e+10 | 0.2% |

Work by op kind (MAC-equivalents, all ops):

| kind | work |
|---|---|
| matmul | 1.93e+13 |
| softmax | 1.80e+11 |
| mul | 9.66e+09 |
| add | 5.50e+09 |
| lossgrad | 4.20e+09 |
| rmsnorm | 2.18e+09 |
| swiglu | 2.15e+09 |
| gain | 2.77e+08 |
| embed | 1.43e+08 |
| sgdupdate | 2.25e+07 |

