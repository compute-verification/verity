### local-sgd @ llama3-8b, tokens=8192, seq=4096 (H=2)

| label | scope | M | N | K | copies | ops | MACs | MACs/token | share | ADW |
|---|---|---|---|---|---|---|---|---|---|---|
| wt-wgrad[4096x14336] | BlockBwd | 4096 | 1.43e+04 | 4096 | 1 | 128 | 3.08e+13 | 3.76e+09 | 11.6% | yes |
| wt-wgrad[4096x14336] | Block | 4096 | 1.43e+04 | 4096 | 1 | 64 | 1.54e+13 | 1.88e+09 | 5.8% | yes |
| wt-wgrad[4096x4096] | BlockBwd | 4096 | 4096 | 1.43e+04 | 1 | 64 | 1.54e+13 | 1.88e+09 | 5.8% | yes |
| wt-wgrad[14336x4096] | BlockBwd | 1.43e+04 | 4096 | 4096 | 1 | 64 | 1.54e+13 | 1.88e+09 | 5.8% | yes |
| wt-wgrad[4096x4096] | BlockBwd | 4096 | 4096 | 4096 | 1 | 192 | 1.32e+13 | 1.61e+09 | 5.0% | yes |
| attn-bwd | BlockBwd/AttnSeqBwd | 4096 | 128 | 1.64e+04 | 8 | 128 | 8.80e+12 | 1.07e+09 | 3.3% | yes |
| wt-fwd[Wg] | Block | 4096 | 1.43e+04 | 4096 | 1 | 32 | 7.70e+12 | 9.40e+08 | 2.9% | yes |
| wt-fwd[Wu] | Block | 4096 | 1.43e+04 | 4096 | 1 | 32 | 7.70e+12 | 9.40e+08 | 2.9% | yes |
| wt-fwd[Wd] | Block | 4096 | 4096 | 1.43e+04 | 1 | 32 | 7.70e+12 | 9.40e+08 | 2.9% | yes |
| wt-fwd-recompute[Wg] | BlockBwd | 4096 | 1.43e+04 | 4096 | 1 | 32 | 7.70e+12 | 9.40e+08 | 2.9% | yes |
| wt-fwd-recompute[Wu] | BlockBwd | 4096 | 1.43e+04 | 4096 | 1 | 32 | 7.70e+12 | 9.40e+08 | 2.9% | yes |
| wt-dgrad[Wd] | BlockBwd | 4096 | 1.43e+04 | 4096 | 1 | 32 | 7.70e+12 | 9.40e+08 | 2.9% | yes |
| wt-wgrad[Wd] | BlockBwd | 4096 | 1.43e+04 | 4096 | 1 | 32 | 7.70e+12 | 9.40e+08 | 2.9% | yes |
| wt-dgrad[Wg] | BlockBwd | 4096 | 4096 | 1.43e+04 | 1 | 32 | 7.70e+12 | 9.40e+08 | 2.9% | yes |
| wt-dgrad[Wu] | BlockBwd | 4096 | 4096 | 1.43e+04 | 1 | 32 | 7.70e+12 | 9.40e+08 | 2.9% | yes |
| wt-wgrad[Wg] | BlockBwd | 1.43e+04 | 4096 | 4096 | 1 | 32 | 7.70e+12 | 9.40e+08 | 2.9% | yes |
| wt-wgrad[Wu] | BlockBwd | 1.43e+04 | 4096 | 4096 | 1 | 32 | 7.70e+12 | 9.40e+08 | 2.9% | yes |
| wt-wgrad[4096x4096] | Block | 4096 | 4096 | 1.43e+04 | 1 | 32 | 7.70e+12 | 9.40e+08 | 2.9% | yes |
| attn-bwd | BlockBwd/AttnSeqBwd | 4096 | 4096 | 128 | 32 | 96 | 6.60e+12 | 8.05e+08 | 2.5% | yes |
| attn-bwd | BlockBwd/AttnSeqBwd | 4096 | 128 | 4096 | 32 | 64 | 4.40e+12 | 5.37e+08 | 1.7% | yes |
| wt-wgrad[4096x4096] | Block | 4096 | 4096 | 4096 | 1 | 64 | 4.40e+12 | 5.37e+08 | 1.7% | yes |
| wt-fwd[Wq] | Block | 4096 | 4096 | 4096 | 1 | 32 | 2.20e+12 | 2.68e+08 | 0.8% | yes |
| attn-fwd | Block/AttnSeq | 4096 | 4096 | 128 | 32 | 32 | 2.20e+12 | 2.68e+08 | 0.8% | yes |
| attn-fwd | Block/AttnSeq | 4096 | 128 | 4096 | 32 | 32 | 2.20e+12 | 2.68e+08 | 0.8% | yes |
| wt-fwd[Wo] | Block | 4096 | 4096 | 4096 | 1 | 32 | 2.20e+12 | 2.68e+08 | 0.8% | yes |
| wt-fwd-recompute[Wq] | BlockBwd | 4096 | 4096 | 4096 | 1 | 32 | 2.20e+12 | 2.68e+08 | 0.8% | yes |
| attn-fwd-recompute | BlockBwd/AttnSeq | 4096 | 4096 | 128 | 32 | 32 | 2.20e+12 | 2.68e+08 | 0.8% | yes |
| attn-fwd-recompute | BlockBwd/AttnSeq | 4096 | 128 | 4096 | 32 | 32 | 2.20e+12 | 2.68e+08 | 0.8% | yes |
| wt-fwd-recompute[Wo] | BlockBwd | 4096 | 4096 | 4096 | 1 | 32 | 2.20e+12 | 2.68e+08 | 0.8% | yes |
| wt-dgrad[Wo] | BlockBwd | 4096 | 4096 | 4096 | 1 | 32 | 2.20e+12 | 2.68e+08 | 0.8% | yes |
| wt-wgrad[Wo] | BlockBwd | 4096 | 4096 | 4096 | 1 | 32 | 2.20e+12 | 2.68e+08 | 0.8% | yes |
| attn-fwd-recompute | BlockBwd/AttnSeqBwd | 4096 | 4096 | 128 | 32 | 32 | 2.20e+12 | 2.68e+08 | 0.8% | yes |
| wt-dgrad[Wq] | BlockBwd | 4096 | 4096 | 4096 | 1 | 32 | 2.20e+12 | 2.68e+08 | 0.8% | yes |
| wt-wgrad[Wq] | BlockBwd | 4096 | 4096 | 4096 | 1 | 32 | 2.20e+12 | 2.68e+08 | 0.8% | yes |
| attn-bwd | Block/AttnSeq | 4096 | 4096 | 128 | 32 | 32 | 2.20e+12 | 2.68e+08 | 0.8% | yes |
| attn-bwd | Block/AttnSeq | 4096 | 128 | 4096 | 32 | 32 | 2.20e+12 | 2.68e+08 | 0.8% | yes |
| attn-bwd | BlockBwd/AttnSeq | 4096 | 4096 | 128 | 32 | 32 | 2.20e+12 | 2.68e+08 | 0.8% | yes |
| attn-bwd | BlockBwd/AttnSeq | 4096 | 128 | 4096 | 32 | 32 | 2.20e+12 | 2.68e+08 | 0.8% | yes |
| wt-fwd[Wh] |  | 4096 | 1.28e+05 | 4096 | 1 | 1 | 2.15e+12 | 2.63e+08 | 0.8% | yes |
| wt-dgrad[Wh] |  | 4096 | 4096 | 1.28e+05 | 1 | 1 | 2.15e+12 | 2.63e+08 | 0.8% | yes |
| wt-wgrad[Wh] |  | 1.28e+05 | 4096 | 4096 | 1 | 1 | 2.15e+12 | 2.63e+08 | 0.8% | yes |
| wt-wgrad[4096x128256] |  | 4096 | 1.28e+05 | 4096 | 1 | 1 | 2.15e+12 | 2.63e+08 | 0.8% | yes |
| wt-wgrad[4096x4096] |  | 4096 | 4096 | 1.28e+05 | 1 | 1 | 2.15e+12 | 2.63e+08 | 0.8% | yes |
| wt-wgrad[128256x4096] |  | 1.28e+05 | 4096 | 4096 | 1 | 1 | 2.15e+12 | 2.63e+08 | 0.8% | yes |
| wt-wgrad[4096x1024] | Block | 4096 | 1024 | 4096 | 1 | 64 | 1.10e+12 | 1.34e+08 | 0.4% | yes |
| wt-wgrad[4096x1024] | BlockBwd | 4096 | 1024 | 4096 | 1 | 64 | 1.10e+12 | 1.34e+08 | 0.4% | yes |
| wt-wgrad[4096x4096] | BlockBwd | 4096 | 4096 | 1024 | 1 | 64 | 1.10e+12 | 1.34e+08 | 0.4% | yes |
| wt-wgrad[1024x4096] | BlockBwd | 1024 | 4096 | 4096 | 1 | 64 | 1.10e+12 | 1.34e+08 | 0.4% | yes |
| wt-fwd[Wk] | Block | 4096 | 1024 | 4096 | 1 | 32 | 5.50e+11 | 6.71e+07 | 0.2% | yes |
| wt-fwd[Wv] | Block | 4096 | 1024 | 4096 | 1 | 32 | 5.50e+11 | 6.71e+07 | 0.2% | yes |
| wt-fwd-recompute[Wk] | BlockBwd | 4096 | 1024 | 4096 | 1 | 32 | 5.50e+11 | 6.71e+07 | 0.2% | yes |
| wt-fwd-recompute[Wv] | BlockBwd | 4096 | 1024 | 4096 | 1 | 32 | 5.50e+11 | 6.71e+07 | 0.2% | yes |
| wt-dgrad[Wk] | BlockBwd | 4096 | 4096 | 1024 | 1 | 32 | 5.50e+11 | 6.71e+07 | 0.2% | yes |
| wt-dgrad[Wv] | BlockBwd | 4096 | 4096 | 1024 | 1 | 32 | 5.50e+11 | 6.71e+07 | 0.2% | yes |
| wt-wgrad[Wk] | BlockBwd | 1024 | 4096 | 4096 | 1 | 32 | 5.50e+11 | 6.71e+07 | 0.2% | yes |
| wt-wgrad[Wv] | BlockBwd | 1024 | 4096 | 4096 | 1 | 32 | 5.50e+11 | 6.71e+07 | 0.2% | yes |

By class:

| class | MACs | share |
|---|---|---|
| wt-wgrad | 1.44e+14 | 54.1% |
| wt-fwd | 3.07e+13 | 11.6% |
| wt-dgrad | 3.07e+13 | 11.6% |
| attn-bwd | 2.86e+13 | 10.8% |
| wt-fwd-recompute | 2.09e+13 | 7.9% |
| attn-fwd-recompute | 6.60e+12 | 2.5% |
| attn-fwd | 4.40e+12 | 1.7% |

Work by op kind (MAC-equivalents, all ops):

| kind | work |
|---|---|
| matmul | 2.66e+14 |
| softmax | 7.22e+11 |
| mul | 4.41e+10 |
| sgdupdate | 3.00e+10 |
| rmsnorm | 1.73e+10 |
| swiglu | 1.50e+10 |
| add | 8.59e+09 |
| lossgrad | 8.41e+09 |
| colsum | 4.36e+09 |
| gain | 2.18e+09 |
| embed | 5.70e+08 |

