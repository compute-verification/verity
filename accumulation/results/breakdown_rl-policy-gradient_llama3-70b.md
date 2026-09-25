### rl-policy-gradient @ llama3-70b, tokens=4096, seq=4096 ()

| label | scope | M | N | K | copies | ops | MACs | MACs/token | share | ADW |
|---|---|---|---|---|---|---|---|---|---|---|
| wt-fwd[Wg] | Block | 4096 | 2.87e+04 | 8192 | 1 | 80 | 7.70e+13 | 1.88e+10 | 6.7% | yes |
| wt-fwd[Wu] | Block | 4096 | 2.87e+04 | 8192 | 1 | 80 | 7.70e+13 | 1.88e+10 | 6.7% | yes |
| wt-fwd[Wd] | Block | 4096 | 8192 | 2.87e+04 | 1 | 80 | 7.70e+13 | 1.88e+10 | 6.7% | yes |
| wt-fwd-recompute[Wg] | BlockBwd | 4096 | 2.87e+04 | 8192 | 1 | 80 | 7.70e+13 | 1.88e+10 | 6.7% | yes |
| wt-fwd-recompute[Wu] | BlockBwd | 4096 | 2.87e+04 | 8192 | 1 | 80 | 7.70e+13 | 1.88e+10 | 6.7% | yes |
| wt-dgrad[Wd] | BlockBwd | 4096 | 2.87e+04 | 8192 | 1 | 80 | 7.70e+13 | 1.88e+10 | 6.7% | yes |
| wt-wgrad[Wd] | BlockBwd | 8192 | 2.87e+04 | 4096 | 1 | 80 | 7.70e+13 | 1.88e+10 | 6.7% | yes |
| wt-dgrad[Wg] | BlockBwd | 4096 | 8192 | 2.87e+04 | 1 | 80 | 7.70e+13 | 1.88e+10 | 6.7% | yes |
| wt-dgrad[Wu] | BlockBwd | 4096 | 8192 | 2.87e+04 | 1 | 80 | 7.70e+13 | 1.88e+10 | 6.7% | yes |
| wt-wgrad[Wg] | BlockBwd | 2.87e+04 | 8192 | 4096 | 1 | 80 | 7.70e+13 | 1.88e+10 | 6.7% | yes |
| wt-wgrad[Wu] | BlockBwd | 2.87e+04 | 8192 | 4096 | 1 | 80 | 7.70e+13 | 1.88e+10 | 6.7% | yes |
| wt-fwd[Wq] | Block | 4096 | 8192 | 8192 | 1 | 80 | 2.20e+13 | 5.37e+09 | 1.9% | yes |
| wt-fwd[Wo] | Block | 4096 | 8192 | 8192 | 1 | 80 | 2.20e+13 | 5.37e+09 | 1.9% | yes |
| wt-fwd-recompute[Wq] | BlockBwd | 4096 | 8192 | 8192 | 1 | 80 | 2.20e+13 | 5.37e+09 | 1.9% | yes |
| wt-fwd-recompute[Wo] | BlockBwd | 4096 | 8192 | 8192 | 1 | 80 | 2.20e+13 | 5.37e+09 | 1.9% | yes |
| wt-dgrad[Wo] | BlockBwd | 4096 | 8192 | 8192 | 1 | 80 | 2.20e+13 | 5.37e+09 | 1.9% | yes |
| wt-wgrad[Wo] | BlockBwd | 8192 | 8192 | 4096 | 1 | 80 | 2.20e+13 | 5.37e+09 | 1.9% | yes |
| attn-bwd | BlockBwd/AttnSeqBwd | 4096 | 128 | 3.28e+04 | 8 | 160 | 2.20e+13 | 5.37e+09 | 1.9% | yes |
| wt-dgrad[Wq] | BlockBwd | 4096 | 8192 | 8192 | 1 | 80 | 2.20e+13 | 5.37e+09 | 1.9% | yes |
| wt-wgrad[Wq] | BlockBwd | 8192 | 8192 | 4096 | 1 | 80 | 2.20e+13 | 5.37e+09 | 1.9% | yes |
| attn-fwd | Block/AttnSeq | 4096 | 4096 | 128 | 64 | 80 | 1.10e+13 | 2.68e+09 | 1.0% | yes |
| attn-fwd | Block/AttnSeq | 4096 | 128 | 4096 | 64 | 80 | 1.10e+13 | 2.68e+09 | 1.0% | yes |
| attn-fwd-recompute | BlockBwd/AttnSeq | 4096 | 4096 | 128 | 64 | 80 | 1.10e+13 | 2.68e+09 | 1.0% | yes |
| attn-fwd-recompute | BlockBwd/AttnSeq | 4096 | 128 | 4096 | 64 | 80 | 1.10e+13 | 2.68e+09 | 1.0% | yes |
| attn-fwd-recompute | BlockBwd/AttnSeqBwd | 4096 | 4096 | 128 | 64 | 80 | 1.10e+13 | 2.68e+09 | 1.0% | yes |
| attn-bwd | BlockBwd/AttnSeqBwd | 4096 | 4096 | 128 | 64 | 80 | 1.10e+13 | 2.68e+09 | 1.0% | yes |
| attn-bwd | BlockBwd/AttnSeqBwd | 4096 | 128 | 4096 | 64 | 80 | 1.10e+13 | 2.68e+09 | 1.0% | yes |
| wt-fwd[Wh] |  | 4096 | 1.28e+05 | 8192 | 1 | 1 | 4.30e+12 | 1.05e+09 | 0.4% | yes |
| wt-dgrad[Wh] |  | 4096 | 8192 | 1.28e+05 | 1 | 1 | 4.30e+12 | 1.05e+09 | 0.4% | yes |
| wt-wgrad[Wh] |  | 1.28e+05 | 8192 | 4096 | 1 | 1 | 4.30e+12 | 1.05e+09 | 0.4% | yes |
| wt-fwd[Wk] | Block | 4096 | 1024 | 8192 | 1 | 80 | 2.75e+12 | 6.71e+08 | 0.2% | yes |
| wt-fwd[Wv] | Block | 4096 | 1024 | 8192 | 1 | 80 | 2.75e+12 | 6.71e+08 | 0.2% | yes |
| wt-fwd-recompute[Wk] | BlockBwd | 4096 | 1024 | 8192 | 1 | 80 | 2.75e+12 | 6.71e+08 | 0.2% | yes |
| wt-fwd-recompute[Wv] | BlockBwd | 4096 | 1024 | 8192 | 1 | 80 | 2.75e+12 | 6.71e+08 | 0.2% | yes |
| wt-dgrad[Wk] | BlockBwd | 4096 | 8192 | 1024 | 1 | 80 | 2.75e+12 | 6.71e+08 | 0.2% | yes |
| wt-dgrad[Wv] | BlockBwd | 4096 | 8192 | 1024 | 1 | 80 | 2.75e+12 | 6.71e+08 | 0.2% | yes |
| wt-wgrad[Wk] | BlockBwd | 1024 | 8192 | 4096 | 1 | 80 | 2.75e+12 | 6.71e+08 | 0.2% | yes |
| wt-wgrad[Wv] | BlockBwd | 1024 | 8192 | 4096 | 1 | 80 | 2.75e+12 | 6.71e+08 | 0.2% | yes |

By class:

| class | MACs | share |
|---|---|---|
| wt-fwd | 2.85e+14 | 24.6% |
| wt-dgrad | 2.85e+14 | 24.6% |
| wt-wgrad | 2.85e+14 | 24.6% |
| wt-fwd-recompute | 2.03e+14 | 17.6% |
| attn-bwd | 4.40e+13 | 3.8% |
| attn-fwd-recompute | 3.30e+13 | 2.9% |
| attn-fwd | 2.20e+13 | 1.9% |

Work by op kind (MAC-equivalents, all ops):

| kind | work |
|---|---|
| matmul | 1.16e+15 |
| softmax | 1.80e+12 |
| sgdupdate | 1.39e+11 |
| mul | 1.10e+11 |
| rmsnorm | 4.31e+10 |
| swiglu | 3.76e+10 |
| add | 2.15e+10 |
| colsum | 1.08e+10 |
| gain | 5.40e+09 |
| lossgrad | 4.20e+09 |
| embed | 5.70e+08 |
| rowscale | 5.25e+08 |

