### local-sgd @ llama3-1b, tokens=8192, seq=4096 (H=2)

| label | scope | M | N | K | copies | ops | MACs | MACs/token | share | ADW |
|---|---|---|---|---|---|---|---|---|---|---|
| wt-wgrad[4096x8192] | BlockBwd | 4096 | 8192 | 2048 | 1 | 48 | 3.30e+12 | 4.03e+08 | 7.2% | yes |
| attn-bwd | BlockBwd/AttnSeqBwd | 4096 | 64 | 1.64e+04 | 8 | 64 | 2.20e+12 | 2.68e+08 | 4.8% | yes |
| wt-wgrad[4096x8192] | Block | 4096 | 8192 | 2048 | 1 | 32 | 2.20e+12 | 2.68e+08 | 4.8% | yes |
| wt-wgrad[4096x2048] | BlockBwd | 4096 | 2048 | 8192 | 1 | 32 | 2.20e+12 | 2.68e+08 | 4.8% | yes |
| wt-wgrad[8192x2048] | BlockBwd | 8192 | 2048 | 4096 | 1 | 32 | 2.20e+12 | 2.68e+08 | 4.8% | yes |
| attn-bwd | BlockBwd/AttnSeqBwd | 4096 | 4096 | 64 | 32 | 48 | 1.65e+12 | 2.01e+08 | 3.6% | yes |
| wt-fwd[Wg] | Block | 4096 | 8192 | 2048 | 1 | 16 | 1.10e+12 | 1.34e+08 | 2.4% | yes |
| wt-fwd[Wu] | Block | 4096 | 8192 | 2048 | 1 | 16 | 1.10e+12 | 1.34e+08 | 2.4% | yes |
| wt-fwd[Wd] | Block | 4096 | 2048 | 8192 | 1 | 16 | 1.10e+12 | 1.34e+08 | 2.4% | yes |
| wt-fwd-recompute[Wg] | BlockBwd | 4096 | 8192 | 2048 | 1 | 16 | 1.10e+12 | 1.34e+08 | 2.4% | yes |
| wt-fwd-recompute[Wu] | BlockBwd | 4096 | 8192 | 2048 | 1 | 16 | 1.10e+12 | 1.34e+08 | 2.4% | yes |
| wt-dgrad[Wd] | BlockBwd | 4096 | 8192 | 2048 | 1 | 16 | 1.10e+12 | 1.34e+08 | 2.4% | yes |
| wt-wgrad[Wd] | BlockBwd | 2048 | 8192 | 4096 | 1 | 16 | 1.10e+12 | 1.34e+08 | 2.4% | yes |
| wt-dgrad[Wg] | BlockBwd | 4096 | 2048 | 8192 | 1 | 16 | 1.10e+12 | 1.34e+08 | 2.4% | yes |
| wt-dgrad[Wu] | BlockBwd | 4096 | 2048 | 8192 | 1 | 16 | 1.10e+12 | 1.34e+08 | 2.4% | yes |
| wt-wgrad[Wg] | BlockBwd | 8192 | 2048 | 4096 | 1 | 16 | 1.10e+12 | 1.34e+08 | 2.4% | yes |
| wt-wgrad[Wu] | BlockBwd | 8192 | 2048 | 4096 | 1 | 16 | 1.10e+12 | 1.34e+08 | 2.4% | yes |
| attn-bwd | BlockBwd/AttnSeqBwd | 4096 | 64 | 4096 | 32 | 32 | 1.10e+12 | 1.34e+08 | 2.4% | yes |
| wt-wgrad[4096x2048] | Block | 4096 | 2048 | 8192 | 1 | 16 | 1.10e+12 | 1.34e+08 | 2.4% | yes |
| wt-wgrad[4096x2048] | BlockBwd | 4096 | 2048 | 2048 | 1 | 64 | 1.10e+12 | 1.34e+08 | 2.4% | yes |
| wt-wgrad[2048x8192] | BlockBwd | 2048 | 8192 | 4096 | 1 | 16 | 1.10e+12 | 1.34e+08 | 2.4% | yes |
| wt-fwd[Wh] |  | 4096 | 1.28e+05 | 2048 | 1 | 1 | 1.08e+12 | 1.31e+08 | 2.3% | yes |
| wt-dgrad[Wh] |  | 4096 | 2048 | 1.28e+05 | 1 | 1 | 1.08e+12 | 1.31e+08 | 2.3% | yes |
| wt-wgrad[Wh] |  | 1.28e+05 | 2048 | 4096 | 1 | 1 | 1.08e+12 | 1.31e+08 | 2.3% | yes |
| wt-wgrad[4096x128256] |  | 4096 | 1.28e+05 | 2048 | 1 | 1 | 1.08e+12 | 1.31e+08 | 2.3% | yes |
| wt-wgrad[4096x2048] |  | 4096 | 2048 | 1.28e+05 | 1 | 1 | 1.08e+12 | 1.31e+08 | 2.3% | yes |
| wt-wgrad[128256x2048] |  | 1.28e+05 | 2048 | 4096 | 1 | 1 | 1.08e+12 | 1.31e+08 | 2.3% | yes |
| attn-fwd | Block/AttnSeq | 4096 | 4096 | 64 | 32 | 16 | 5.50e+11 | 6.71e+07 | 1.2% | yes |
| attn-fwd | Block/AttnSeq | 4096 | 64 | 4096 | 32 | 16 | 5.50e+11 | 6.71e+07 | 1.2% | yes |
| attn-fwd-recompute | BlockBwd/AttnSeq | 4096 | 4096 | 64 | 32 | 16 | 5.50e+11 | 6.71e+07 | 1.2% | yes |
| attn-fwd-recompute | BlockBwd/AttnSeq | 4096 | 64 | 4096 | 32 | 16 | 5.50e+11 | 6.71e+07 | 1.2% | yes |
| attn-fwd-recompute | BlockBwd/AttnSeqBwd | 4096 | 4096 | 64 | 32 | 16 | 5.50e+11 | 6.71e+07 | 1.2% | yes |
| wt-wgrad[4096x2048] | Block | 4096 | 2048 | 2048 | 1 | 32 | 5.50e+11 | 6.71e+07 | 1.2% | yes |
| attn-bwd | Block/AttnSeq | 4096 | 4096 | 64 | 32 | 16 | 5.50e+11 | 6.71e+07 | 1.2% | yes |
| attn-bwd | Block/AttnSeq | 4096 | 64 | 4096 | 32 | 16 | 5.50e+11 | 6.71e+07 | 1.2% | yes |
| attn-bwd | BlockBwd/AttnSeq | 4096 | 4096 | 64 | 32 | 16 | 5.50e+11 | 6.71e+07 | 1.2% | yes |
| attn-bwd | BlockBwd/AttnSeq | 4096 | 64 | 4096 | 32 | 16 | 5.50e+11 | 6.71e+07 | 1.2% | yes |
| wt-wgrad[2048x2048] | BlockBwd | 2048 | 2048 | 4096 | 1 | 32 | 5.50e+11 | 6.71e+07 | 1.2% | yes |
| wt-fwd[Wq] | Block | 4096 | 2048 | 2048 | 1 | 16 | 2.75e+11 | 3.36e+07 | 0.6% | yes |
| wt-fwd[Wo] | Block | 4096 | 2048 | 2048 | 1 | 16 | 2.75e+11 | 3.36e+07 | 0.6% | yes |
| wt-fwd-recompute[Wq] | BlockBwd | 4096 | 2048 | 2048 | 1 | 16 | 2.75e+11 | 3.36e+07 | 0.6% | yes |
| wt-fwd-recompute[Wo] | BlockBwd | 4096 | 2048 | 2048 | 1 | 16 | 2.75e+11 | 3.36e+07 | 0.6% | yes |
| wt-dgrad[Wo] | BlockBwd | 4096 | 2048 | 2048 | 1 | 16 | 2.75e+11 | 3.36e+07 | 0.6% | yes |
| wt-wgrad[Wo] | BlockBwd | 2048 | 2048 | 4096 | 1 | 16 | 2.75e+11 | 3.36e+07 | 0.6% | yes |
| wt-dgrad[Wq] | BlockBwd | 4096 | 2048 | 2048 | 1 | 16 | 2.75e+11 | 3.36e+07 | 0.6% | yes |
| wt-wgrad[Wq] | BlockBwd | 2048 | 2048 | 4096 | 1 | 16 | 2.75e+11 | 3.36e+07 | 0.6% | yes |
| wt-wgrad[4096x512] | Block | 4096 | 512 | 2048 | 1 | 32 | 1.37e+11 | 1.68e+07 | 0.3% | yes |
| wt-wgrad[4096x512] | BlockBwd | 4096 | 512 | 2048 | 1 | 32 | 1.37e+11 | 1.68e+07 | 0.3% | yes |
| wt-wgrad[4096x2048] | BlockBwd | 4096 | 2048 | 512 | 1 | 32 | 1.37e+11 | 1.68e+07 | 0.3% | yes |
| wt-wgrad[512x2048] | BlockBwd | 512 | 2048 | 4096 | 1 | 32 | 1.37e+11 | 1.68e+07 | 0.3% | yes |
| wt-fwd[Wk] | Block | 4096 | 512 | 2048 | 1 | 16 | 6.87e+10 | 8.39e+06 | 0.1% | yes |
| wt-fwd[Wv] | Block | 4096 | 512 | 2048 | 1 | 16 | 6.87e+10 | 8.39e+06 | 0.1% | yes |
| wt-fwd-recompute[Wk] | BlockBwd | 4096 | 512 | 2048 | 1 | 16 | 6.87e+10 | 8.39e+06 | 0.1% | yes |
| wt-fwd-recompute[Wv] | BlockBwd | 4096 | 512 | 2048 | 1 | 16 | 6.87e+10 | 8.39e+06 | 0.1% | yes |
| wt-dgrad[Wk] | BlockBwd | 4096 | 2048 | 512 | 1 | 16 | 6.87e+10 | 8.39e+06 | 0.1% | yes |
| wt-dgrad[Wv] | BlockBwd | 4096 | 2048 | 512 | 1 | 16 | 6.87e+10 | 8.39e+06 | 0.1% | yes |
| wt-wgrad[Wk] | BlockBwd | 512 | 2048 | 4096 | 1 | 16 | 6.87e+10 | 8.39e+06 | 0.1% | yes |
| wt-wgrad[Wv] | BlockBwd | 512 | 2048 | 4096 | 1 | 16 | 6.87e+10 | 8.39e+06 | 0.1% | yes |

By class:

| class | MACs | share |
|---|---|---|
| wt-wgrad | 2.31e+13 | 50.2% |
| attn-bwd | 7.15e+12 | 15.5% |
| wt-fwd | 5.06e+12 | 11.0% |
| wt-dgrad | 5.06e+12 | 11.0% |
| wt-fwd-recompute | 2.89e+12 | 6.3% |
| attn-fwd-recompute | 1.65e+12 | 3.6% |
| attn-fwd | 1.10e+12 | 2.4% |

Work by op kind (MAC-equivalents, all ops):

| kind | work |
|---|---|
| matmul | 4.61e+13 |
| softmax | 3.61e+11 |
| mul | 1.99e+10 |
| lossgrad | 8.41e+09 |
| sgdupdate | 4.94e+09 |
| rmsnorm | 4.36e+09 |
| swiglu | 4.29e+09 |
| add | 2.15e+09 |
| colsum | 1.11e+09 |
| gain | 5.54e+08 |
| embed | 2.85e+08 |

