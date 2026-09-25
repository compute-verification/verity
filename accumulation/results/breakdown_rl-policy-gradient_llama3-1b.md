### rl-policy-gradient @ llama3-1b, tokens=4096, seq=4096 ()

| label | scope | M | N | K | copies | ops | MACs | MACs/token | share | ADW |
|---|---|---|---|---|---|---|---|---|---|---|
| wt-fwd[Wg] | Block | 4096 | 8192 | 2048 | 1 | 16 | 1.10e+12 | 2.68e+08 | 4.8% | yes |
| wt-fwd[Wu] | Block | 4096 | 8192 | 2048 | 1 | 16 | 1.10e+12 | 2.68e+08 | 4.8% | yes |
| wt-fwd[Wd] | Block | 4096 | 2048 | 8192 | 1 | 16 | 1.10e+12 | 2.68e+08 | 4.8% | yes |
| wt-fwd-recompute[Wg] | BlockBwd | 4096 | 8192 | 2048 | 1 | 16 | 1.10e+12 | 2.68e+08 | 4.8% | yes |
| wt-fwd-recompute[Wu] | BlockBwd | 4096 | 8192 | 2048 | 1 | 16 | 1.10e+12 | 2.68e+08 | 4.8% | yes |
| wt-dgrad[Wd] | BlockBwd | 4096 | 8192 | 2048 | 1 | 16 | 1.10e+12 | 2.68e+08 | 4.8% | yes |
| wt-wgrad[Wd] | BlockBwd | 2048 | 8192 | 4096 | 1 | 16 | 1.10e+12 | 2.68e+08 | 4.8% | yes |
| wt-dgrad[Wg] | BlockBwd | 4096 | 2048 | 8192 | 1 | 16 | 1.10e+12 | 2.68e+08 | 4.8% | yes |
| wt-dgrad[Wu] | BlockBwd | 4096 | 2048 | 8192 | 1 | 16 | 1.10e+12 | 2.68e+08 | 4.8% | yes |
| wt-wgrad[Wg] | BlockBwd | 8192 | 2048 | 4096 | 1 | 16 | 1.10e+12 | 2.68e+08 | 4.8% | yes |
| wt-wgrad[Wu] | BlockBwd | 8192 | 2048 | 4096 | 1 | 16 | 1.10e+12 | 2.68e+08 | 4.8% | yes |
| attn-bwd | BlockBwd/AttnSeqBwd | 4096 | 64 | 1.64e+04 | 8 | 32 | 1.10e+12 | 2.68e+08 | 4.8% | yes |
| wt-fwd[Wh] |  | 4096 | 1.28e+05 | 2048 | 1 | 1 | 1.08e+12 | 2.63e+08 | 4.7% | yes |
| wt-dgrad[Wh] |  | 4096 | 2048 | 1.28e+05 | 1 | 1 | 1.08e+12 | 2.63e+08 | 4.7% | yes |
| wt-wgrad[Wh] |  | 1.28e+05 | 2048 | 4096 | 1 | 1 | 1.08e+12 | 2.63e+08 | 4.7% | yes |
| attn-fwd | Block/AttnSeq | 4096 | 4096 | 64 | 32 | 16 | 5.50e+11 | 1.34e+08 | 2.4% | yes |
| attn-fwd | Block/AttnSeq | 4096 | 64 | 4096 | 32 | 16 | 5.50e+11 | 1.34e+08 | 2.4% | yes |
| attn-fwd-recompute | BlockBwd/AttnSeq | 4096 | 4096 | 64 | 32 | 16 | 5.50e+11 | 1.34e+08 | 2.4% | yes |
| attn-fwd-recompute | BlockBwd/AttnSeq | 4096 | 64 | 4096 | 32 | 16 | 5.50e+11 | 1.34e+08 | 2.4% | yes |
| attn-fwd-recompute | BlockBwd/AttnSeqBwd | 4096 | 4096 | 64 | 32 | 16 | 5.50e+11 | 1.34e+08 | 2.4% | yes |
| attn-bwd | BlockBwd/AttnSeqBwd | 4096 | 4096 | 64 | 32 | 16 | 5.50e+11 | 1.34e+08 | 2.4% | yes |
| attn-bwd | BlockBwd/AttnSeqBwd | 4096 | 64 | 4096 | 32 | 16 | 5.50e+11 | 1.34e+08 | 2.4% | yes |
| wt-fwd[Wq] | Block | 4096 | 2048 | 2048 | 1 | 16 | 2.75e+11 | 6.71e+07 | 1.2% | yes |
| wt-fwd[Wo] | Block | 4096 | 2048 | 2048 | 1 | 16 | 2.75e+11 | 6.71e+07 | 1.2% | yes |
| wt-fwd-recompute[Wq] | BlockBwd | 4096 | 2048 | 2048 | 1 | 16 | 2.75e+11 | 6.71e+07 | 1.2% | yes |
| wt-fwd-recompute[Wo] | BlockBwd | 4096 | 2048 | 2048 | 1 | 16 | 2.75e+11 | 6.71e+07 | 1.2% | yes |
| wt-dgrad[Wo] | BlockBwd | 4096 | 2048 | 2048 | 1 | 16 | 2.75e+11 | 6.71e+07 | 1.2% | yes |
| wt-wgrad[Wo] | BlockBwd | 2048 | 2048 | 4096 | 1 | 16 | 2.75e+11 | 6.71e+07 | 1.2% | yes |
| wt-dgrad[Wq] | BlockBwd | 4096 | 2048 | 2048 | 1 | 16 | 2.75e+11 | 6.71e+07 | 1.2% | yes |
| wt-wgrad[Wq] | BlockBwd | 2048 | 2048 | 4096 | 1 | 16 | 2.75e+11 | 6.71e+07 | 1.2% | yes |
| wt-fwd[Wk] | Block | 4096 | 512 | 2048 | 1 | 16 | 6.87e+10 | 1.68e+07 | 0.3% | yes |
| wt-fwd[Wv] | Block | 4096 | 512 | 2048 | 1 | 16 | 6.87e+10 | 1.68e+07 | 0.3% | yes |
| wt-fwd-recompute[Wk] | BlockBwd | 4096 | 512 | 2048 | 1 | 16 | 6.87e+10 | 1.68e+07 | 0.3% | yes |
| wt-fwd-recompute[Wv] | BlockBwd | 4096 | 512 | 2048 | 1 | 16 | 6.87e+10 | 1.68e+07 | 0.3% | yes |
| wt-dgrad[Wk] | BlockBwd | 4096 | 2048 | 512 | 1 | 16 | 6.87e+10 | 1.68e+07 | 0.3% | yes |
| wt-dgrad[Wv] | BlockBwd | 4096 | 2048 | 512 | 1 | 16 | 6.87e+10 | 1.68e+07 | 0.3% | yes |
| wt-wgrad[Wk] | BlockBwd | 512 | 2048 | 4096 | 1 | 16 | 6.87e+10 | 1.68e+07 | 0.3% | yes |
| wt-wgrad[Wv] | BlockBwd | 512 | 2048 | 4096 | 1 | 16 | 6.87e+10 | 1.68e+07 | 0.3% | yes |

By class:

| class | MACs | share |
|---|---|---|
| wt-fwd | 5.06e+12 | 22.0% |
| wt-dgrad | 5.06e+12 | 22.0% |
| wt-wgrad | 5.06e+12 | 22.0% |
| wt-fwd-recompute | 2.89e+12 | 12.5% |
| attn-bwd | 2.20e+12 | 9.6% |
| attn-fwd-recompute | 1.65e+12 | 7.2% |
| attn-fwd | 1.10e+12 | 4.8% |

Work by op kind (MAC-equivalents, all ops):

| kind | work |
|---|---|
| matmul | 2.31e+13 |
| softmax | 1.80e+11 |
| mul | 9.94e+09 |
| lossgrad | 4.20e+09 |
| sgdupdate | 2.47e+09 |
| rmsnorm | 2.18e+09 |
| swiglu | 2.15e+09 |
| add | 1.07e+09 |
| colsum | 5.54e+08 |
| rowscale | 5.25e+08 |
| gain | 2.77e+08 |
| embed | 1.43e+08 |

