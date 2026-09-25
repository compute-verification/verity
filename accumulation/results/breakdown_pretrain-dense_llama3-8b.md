### pretrain-dense @ llama3-8b, tokens=4096, seq=4096 ()

| label | scope | M | N | K | copies | ops | MACs | MACs/token | share | ADW |
|---|---|---|---|---|---|---|---|---|---|---|
| wt-fwd[Wg] | Block | 4096 | 1.43e+04 | 4096 | 1 | 32 | 7.70e+12 | 1.88e+09 | 5.8% | yes |
| wt-fwd[Wu] | Block | 4096 | 1.43e+04 | 4096 | 1 | 32 | 7.70e+12 | 1.88e+09 | 5.8% | yes |
| wt-fwd[Wd] | Block | 4096 | 4096 | 1.43e+04 | 1 | 32 | 7.70e+12 | 1.88e+09 | 5.8% | yes |
| wt-fwd-recompute[Wg] | BlockBwd | 4096 | 1.43e+04 | 4096 | 1 | 32 | 7.70e+12 | 1.88e+09 | 5.8% | yes |
| wt-fwd-recompute[Wu] | BlockBwd | 4096 | 1.43e+04 | 4096 | 1 | 32 | 7.70e+12 | 1.88e+09 | 5.8% | yes |
| wt-dgrad[Wd] | BlockBwd | 4096 | 1.43e+04 | 4096 | 1 | 32 | 7.70e+12 | 1.88e+09 | 5.8% | yes |
| wt-wgrad[Wd] | BlockBwd | 4096 | 1.43e+04 | 4096 | 1 | 32 | 7.70e+12 | 1.88e+09 | 5.8% | yes |
| wt-dgrad[Wg] | BlockBwd | 4096 | 4096 | 1.43e+04 | 1 | 32 | 7.70e+12 | 1.88e+09 | 5.8% | yes |
| wt-dgrad[Wu] | BlockBwd | 4096 | 4096 | 1.43e+04 | 1 | 32 | 7.70e+12 | 1.88e+09 | 5.8% | yes |
| wt-wgrad[Wg] | BlockBwd | 1.43e+04 | 4096 | 4096 | 1 | 32 | 7.70e+12 | 1.88e+09 | 5.8% | yes |
| wt-wgrad[Wu] | BlockBwd | 1.43e+04 | 4096 | 4096 | 1 | 32 | 7.70e+12 | 1.88e+09 | 5.8% | yes |
| attn-bwd | BlockBwd/AttnSeqBwd | 4096 | 128 | 1.64e+04 | 8 | 64 | 4.40e+12 | 1.07e+09 | 3.3% | yes |
| wt-fwd[Wq] | Block | 4096 | 4096 | 4096 | 1 | 32 | 2.20e+12 | 5.37e+08 | 1.7% | yes |
| attn-fwd | Block/AttnSeq | 4096 | 4096 | 128 | 32 | 32 | 2.20e+12 | 5.37e+08 | 1.7% | yes |
| attn-fwd | Block/AttnSeq | 4096 | 128 | 4096 | 32 | 32 | 2.20e+12 | 5.37e+08 | 1.7% | yes |
| wt-fwd[Wo] | Block | 4096 | 4096 | 4096 | 1 | 32 | 2.20e+12 | 5.37e+08 | 1.7% | yes |
| wt-fwd-recompute[Wq] | BlockBwd | 4096 | 4096 | 4096 | 1 | 32 | 2.20e+12 | 5.37e+08 | 1.7% | yes |
| attn-fwd-recompute | BlockBwd/AttnSeq | 4096 | 4096 | 128 | 32 | 32 | 2.20e+12 | 5.37e+08 | 1.7% | yes |
| attn-fwd-recompute | BlockBwd/AttnSeq | 4096 | 128 | 4096 | 32 | 32 | 2.20e+12 | 5.37e+08 | 1.7% | yes |
| wt-fwd-recompute[Wo] | BlockBwd | 4096 | 4096 | 4096 | 1 | 32 | 2.20e+12 | 5.37e+08 | 1.7% | yes |
| wt-dgrad[Wo] | BlockBwd | 4096 | 4096 | 4096 | 1 | 32 | 2.20e+12 | 5.37e+08 | 1.7% | yes |
| wt-wgrad[Wo] | BlockBwd | 4096 | 4096 | 4096 | 1 | 32 | 2.20e+12 | 5.37e+08 | 1.7% | yes |
| attn-fwd-recompute | BlockBwd/AttnSeqBwd | 4096 | 4096 | 128 | 32 | 32 | 2.20e+12 | 5.37e+08 | 1.7% | yes |
| attn-bwd | BlockBwd/AttnSeqBwd | 4096 | 4096 | 128 | 32 | 32 | 2.20e+12 | 5.37e+08 | 1.7% | yes |
| attn-bwd | BlockBwd/AttnSeqBwd | 4096 | 128 | 4096 | 32 | 32 | 2.20e+12 | 5.37e+08 | 1.7% | yes |
| wt-dgrad[Wq] | BlockBwd | 4096 | 4096 | 4096 | 1 | 32 | 2.20e+12 | 5.37e+08 | 1.7% | yes |
| wt-wgrad[Wq] | BlockBwd | 4096 | 4096 | 4096 | 1 | 32 | 2.20e+12 | 5.37e+08 | 1.7% | yes |
| wt-fwd[Wh] |  | 4096 | 1.28e+05 | 4096 | 1 | 1 | 2.15e+12 | 5.25e+08 | 1.6% | yes |
| wt-dgrad[Wh] |  | 4096 | 4096 | 1.28e+05 | 1 | 1 | 2.15e+12 | 5.25e+08 | 1.6% | yes |
| wt-wgrad[Wh] |  | 1.28e+05 | 4096 | 4096 | 1 | 1 | 2.15e+12 | 5.25e+08 | 1.6% | yes |
| wt-fwd[Wk] | Block | 4096 | 1024 | 4096 | 1 | 32 | 5.50e+11 | 1.34e+08 | 0.4% | yes |
| wt-fwd[Wv] | Block | 4096 | 1024 | 4096 | 1 | 32 | 5.50e+11 | 1.34e+08 | 0.4% | yes |
| wt-fwd-recompute[Wk] | BlockBwd | 4096 | 1024 | 4096 | 1 | 32 | 5.50e+11 | 1.34e+08 | 0.4% | yes |
| wt-fwd-recompute[Wv] | BlockBwd | 4096 | 1024 | 4096 | 1 | 32 | 5.50e+11 | 1.34e+08 | 0.4% | yes |
| wt-dgrad[Wk] | BlockBwd | 4096 | 4096 | 1024 | 1 | 32 | 5.50e+11 | 1.34e+08 | 0.4% | yes |
| wt-dgrad[Wv] | BlockBwd | 4096 | 4096 | 1024 | 1 | 32 | 5.50e+11 | 1.34e+08 | 0.4% | yes |
| wt-wgrad[Wk] | BlockBwd | 1024 | 4096 | 4096 | 1 | 32 | 5.50e+11 | 1.34e+08 | 0.4% | yes |
| wt-wgrad[Wv] | BlockBwd | 1024 | 4096 | 4096 | 1 | 32 | 5.50e+11 | 1.34e+08 | 0.4% | yes |

By class:

| class | MACs | share |
|---|---|---|
| wt-fwd | 3.07e+13 | 23.1% |
| wt-dgrad | 3.07e+13 | 23.1% |
| wt-wgrad | 3.07e+13 | 23.1% |
| wt-fwd-recompute | 2.09e+13 | 15.7% |
| attn-bwd | 8.80e+12 | 6.6% |
| attn-fwd-recompute | 6.60e+12 | 5.0% |
| attn-fwd | 4.40e+12 | 3.3% |

Work by op kind (MAC-equivalents, all ops):

| kind | work |
|---|---|
| matmul | 1.33e+14 |
| softmax | 3.61e+11 |
| mul | 2.20e+10 |
| sgdupdate | 1.50e+10 |
| rmsnorm | 8.66e+09 |
| swiglu | 7.52e+09 |
| add | 4.29e+09 |
| lossgrad | 4.20e+09 |
| colsum | 2.18e+09 |
| gain | 1.09e+09 |
| embed | 2.85e+08 |

