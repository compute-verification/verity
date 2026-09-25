### local-sgd @ llama3-70b, tokens=8192, seq=4096 (H=2)

| label | scope | M | N | K | copies | ops | MACs | MACs/token | share | ADW |
|---|---|---|---|---|---|---|---|---|---|---|
| wt-wgrad[4096x28672] | BlockBwd | 4096 | 2.87e+04 | 8192 | 1 | 240 | 2.31e+14 | 2.82e+10 | 10.0% | yes |
| wt-wgrad[4096x28672] | Block | 4096 | 2.87e+04 | 8192 | 1 | 160 | 1.54e+14 | 1.88e+10 | 6.7% | yes |
| wt-wgrad[4096x8192] | BlockBwd | 4096 | 8192 | 2.87e+04 | 1 | 160 | 1.54e+14 | 1.88e+10 | 6.7% | yes |
| wt-wgrad[28672x8192] | BlockBwd | 2.87e+04 | 8192 | 4096 | 1 | 160 | 1.54e+14 | 1.88e+10 | 6.7% | yes |
| wt-wgrad[4096x8192] | BlockBwd | 4096 | 8192 | 8192 | 1 | 320 | 8.80e+13 | 1.07e+10 | 3.8% | yes |
| wt-fwd[Wg] | Block | 4096 | 2.87e+04 | 8192 | 1 | 80 | 7.70e+13 | 9.40e+09 | 3.3% | yes |
| wt-fwd[Wu] | Block | 4096 | 2.87e+04 | 8192 | 1 | 80 | 7.70e+13 | 9.40e+09 | 3.3% | yes |
| wt-fwd[Wd] | Block | 4096 | 8192 | 2.87e+04 | 1 | 80 | 7.70e+13 | 9.40e+09 | 3.3% | yes |
| wt-fwd-recompute[Wg] | BlockBwd | 4096 | 2.87e+04 | 8192 | 1 | 80 | 7.70e+13 | 9.40e+09 | 3.3% | yes |
| wt-fwd-recompute[Wu] | BlockBwd | 4096 | 2.87e+04 | 8192 | 1 | 80 | 7.70e+13 | 9.40e+09 | 3.3% | yes |
| wt-dgrad[Wd] | BlockBwd | 4096 | 2.87e+04 | 8192 | 1 | 80 | 7.70e+13 | 9.40e+09 | 3.3% | yes |
| wt-wgrad[Wd] | BlockBwd | 8192 | 2.87e+04 | 4096 | 1 | 80 | 7.70e+13 | 9.40e+09 | 3.3% | yes |
| wt-dgrad[Wg] | BlockBwd | 4096 | 8192 | 2.87e+04 | 1 | 80 | 7.70e+13 | 9.40e+09 | 3.3% | yes |
| wt-dgrad[Wu] | BlockBwd | 4096 | 8192 | 2.87e+04 | 1 | 80 | 7.70e+13 | 9.40e+09 | 3.3% | yes |
| wt-wgrad[Wg] | BlockBwd | 2.87e+04 | 8192 | 4096 | 1 | 80 | 7.70e+13 | 9.40e+09 | 3.3% | yes |
| wt-wgrad[Wu] | BlockBwd | 2.87e+04 | 8192 | 4096 | 1 | 80 | 7.70e+13 | 9.40e+09 | 3.3% | yes |
| wt-wgrad[4096x8192] | Block | 4096 | 8192 | 2.87e+04 | 1 | 80 | 7.70e+13 | 9.40e+09 | 3.3% | yes |
| wt-wgrad[8192x28672] | BlockBwd | 8192 | 2.87e+04 | 4096 | 1 | 80 | 7.70e+13 | 9.40e+09 | 3.3% | yes |
| attn-bwd | BlockBwd/AttnSeqBwd | 4096 | 128 | 3.28e+04 | 8 | 320 | 4.40e+13 | 5.37e+09 | 1.9% | yes |
| wt-wgrad[4096x8192] | Block | 4096 | 8192 | 8192 | 1 | 160 | 4.40e+13 | 5.37e+09 | 1.9% | yes |
| wt-wgrad[8192x8192] | BlockBwd | 8192 | 8192 | 4096 | 1 | 160 | 4.40e+13 | 5.37e+09 | 1.9% | yes |
| attn-bwd | BlockBwd/AttnSeqBwd | 4096 | 4096 | 128 | 64 | 240 | 3.30e+13 | 4.03e+09 | 1.4% | yes |
| wt-fwd[Wq] | Block | 4096 | 8192 | 8192 | 1 | 80 | 2.20e+13 | 2.68e+09 | 1.0% | yes |
| wt-fwd[Wo] | Block | 4096 | 8192 | 8192 | 1 | 80 | 2.20e+13 | 2.68e+09 | 1.0% | yes |
| wt-fwd-recompute[Wq] | BlockBwd | 4096 | 8192 | 8192 | 1 | 80 | 2.20e+13 | 2.68e+09 | 1.0% | yes |
| wt-fwd-recompute[Wo] | BlockBwd | 4096 | 8192 | 8192 | 1 | 80 | 2.20e+13 | 2.68e+09 | 1.0% | yes |
| wt-dgrad[Wo] | BlockBwd | 4096 | 8192 | 8192 | 1 | 80 | 2.20e+13 | 2.68e+09 | 1.0% | yes |
| wt-wgrad[Wo] | BlockBwd | 8192 | 8192 | 4096 | 1 | 80 | 2.20e+13 | 2.68e+09 | 1.0% | yes |
| attn-bwd | BlockBwd/AttnSeqBwd | 4096 | 128 | 4096 | 64 | 160 | 2.20e+13 | 2.68e+09 | 1.0% | yes |
| wt-dgrad[Wq] | BlockBwd | 4096 | 8192 | 8192 | 1 | 80 | 2.20e+13 | 2.68e+09 | 1.0% | yes |
| wt-wgrad[Wq] | BlockBwd | 8192 | 8192 | 4096 | 1 | 80 | 2.20e+13 | 2.68e+09 | 1.0% | yes |
| attn-fwd | Block/AttnSeq | 4096 | 4096 | 128 | 64 | 80 | 1.10e+13 | 1.34e+09 | 0.5% | yes |
| attn-fwd | Block/AttnSeq | 4096 | 128 | 4096 | 64 | 80 | 1.10e+13 | 1.34e+09 | 0.5% | yes |
| attn-fwd-recompute | BlockBwd/AttnSeq | 4096 | 4096 | 128 | 64 | 80 | 1.10e+13 | 1.34e+09 | 0.5% | yes |
| attn-fwd-recompute | BlockBwd/AttnSeq | 4096 | 128 | 4096 | 64 | 80 | 1.10e+13 | 1.34e+09 | 0.5% | yes |
| attn-fwd-recompute | BlockBwd/AttnSeqBwd | 4096 | 4096 | 128 | 64 | 80 | 1.10e+13 | 1.34e+09 | 0.5% | yes |
| attn-bwd | Block/AttnSeq | 4096 | 4096 | 128 | 64 | 80 | 1.10e+13 | 1.34e+09 | 0.5% | yes |
| attn-bwd | Block/AttnSeq | 4096 | 128 | 4096 | 64 | 80 | 1.10e+13 | 1.34e+09 | 0.5% | yes |
| attn-bwd | BlockBwd/AttnSeq | 4096 | 4096 | 128 | 64 | 80 | 1.10e+13 | 1.34e+09 | 0.5% | yes |
| attn-bwd | BlockBwd/AttnSeq | 4096 | 128 | 4096 | 64 | 80 | 1.10e+13 | 1.34e+09 | 0.5% | yes |
| wt-wgrad[4096x1024] | Block | 4096 | 1024 | 8192 | 1 | 160 | 5.50e+12 | 6.71e+08 | 0.2% | yes |
| wt-wgrad[4096x1024] | BlockBwd | 4096 | 1024 | 8192 | 1 | 160 | 5.50e+12 | 6.71e+08 | 0.2% | yes |
| wt-wgrad[4096x8192] | BlockBwd | 4096 | 8192 | 1024 | 1 | 160 | 5.50e+12 | 6.71e+08 | 0.2% | yes |
| wt-wgrad[1024x8192] | BlockBwd | 1024 | 8192 | 4096 | 1 | 160 | 5.50e+12 | 6.71e+08 | 0.2% | yes |
| wt-fwd[Wh] |  | 4096 | 1.28e+05 | 8192 | 1 | 1 | 4.30e+12 | 5.25e+08 | 0.2% | yes |
| wt-dgrad[Wh] |  | 4096 | 8192 | 1.28e+05 | 1 | 1 | 4.30e+12 | 5.25e+08 | 0.2% | yes |
| wt-wgrad[Wh] |  | 1.28e+05 | 8192 | 4096 | 1 | 1 | 4.30e+12 | 5.25e+08 | 0.2% | yes |
| wt-wgrad[4096x128256] |  | 4096 | 1.28e+05 | 8192 | 1 | 1 | 4.30e+12 | 5.25e+08 | 0.2% | yes |
| wt-wgrad[4096x8192] |  | 4096 | 8192 | 1.28e+05 | 1 | 1 | 4.30e+12 | 5.25e+08 | 0.2% | yes |
| wt-wgrad[128256x8192] |  | 1.28e+05 | 8192 | 4096 | 1 | 1 | 4.30e+12 | 5.25e+08 | 0.2% | yes |
| wt-fwd[Wk] | Block | 4096 | 1024 | 8192 | 1 | 80 | 2.75e+12 | 3.36e+08 | 0.1% | yes |
| wt-fwd[Wv] | Block | 4096 | 1024 | 8192 | 1 | 80 | 2.75e+12 | 3.36e+08 | 0.1% | yes |
| wt-fwd-recompute[Wk] | BlockBwd | 4096 | 1024 | 8192 | 1 | 80 | 2.75e+12 | 3.36e+08 | 0.1% | yes |
| wt-fwd-recompute[Wv] | BlockBwd | 4096 | 1024 | 8192 | 1 | 80 | 2.75e+12 | 3.36e+08 | 0.1% | yes |
| wt-dgrad[Wk] | BlockBwd | 4096 | 8192 | 1024 | 1 | 80 | 2.75e+12 | 3.36e+08 | 0.1% | yes |
| wt-dgrad[Wv] | BlockBwd | 4096 | 8192 | 1024 | 1 | 80 | 2.75e+12 | 3.36e+08 | 0.1% | yes |
| wt-wgrad[Wk] | BlockBwd | 1024 | 8192 | 4096 | 1 | 80 | 2.75e+12 | 3.36e+08 | 0.1% | yes |
| wt-wgrad[Wv] | BlockBwd | 1024 | 8192 | 4096 | 1 | 80 | 2.75e+12 | 3.36e+08 | 0.1% | yes |

By class:

| class | MACs | share |
|---|---|---|
| wt-wgrad | 1.34e+15 | 58.0% |
| wt-fwd | 2.85e+14 | 12.3% |
| wt-dgrad | 2.85e+14 | 12.3% |
| wt-fwd-recompute | 2.03e+14 | 8.8% |
| attn-bwd | 1.43e+14 | 6.2% |
| attn-fwd-recompute | 3.30e+13 | 1.4% |
| attn-fwd | 2.20e+13 | 1.0% |

Work by op kind (MAC-equivalents, all ops):

| kind | work |
|---|---|
| matmul | 2.31e+15 |
| softmax | 3.61e+12 |
| sgdupdate | 2.78e+11 |
| mul | 2.20e+11 |
| rmsnorm | 8.62e+10 |
| swiglu | 7.52e+10 |
| add | 4.29e+10 |
| colsum | 2.16e+10 |
| gain | 1.08e+10 |
| lossgrad | 8.41e+09 |
| embed | 1.14e+09 |

