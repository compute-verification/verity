### pretrain-dense @ llama3-405b, tokens=4096, seq=4096 ()

| label | scope | M | N | K | copies | ops | MACs | MACs/token | share | ADW |
|---|---|---|---|---|---|---|---|---|---|---|
| wt-fwd[Wg] | Block | 4096 | 5.32e+04 | 1.64e+04 | 1 | 126 | 4.50e+14 | 1.10e+11 | 7.0% | yes |
| wt-fwd[Wu] | Block | 4096 | 5.32e+04 | 1.64e+04 | 1 | 126 | 4.50e+14 | 1.10e+11 | 7.0% | yes |
| wt-fwd[Wd] | Block | 4096 | 1.64e+04 | 5.32e+04 | 1 | 126 | 4.50e+14 | 1.10e+11 | 7.0% | yes |
| wt-fwd-recompute[Wg] | BlockBwd | 4096 | 5.32e+04 | 1.64e+04 | 1 | 126 | 4.50e+14 | 1.10e+11 | 7.0% | yes |
| wt-fwd-recompute[Wu] | BlockBwd | 4096 | 5.32e+04 | 1.64e+04 | 1 | 126 | 4.50e+14 | 1.10e+11 | 7.0% | yes |
| wt-dgrad[Wd] | BlockBwd | 4096 | 5.32e+04 | 1.64e+04 | 1 | 126 | 4.50e+14 | 1.10e+11 | 7.0% | yes |
| wt-wgrad[Wd] | BlockBwd | 1.64e+04 | 5.32e+04 | 4096 | 1 | 126 | 4.50e+14 | 1.10e+11 | 7.0% | yes |
| wt-dgrad[Wg] | BlockBwd | 4096 | 1.64e+04 | 5.32e+04 | 1 | 126 | 4.50e+14 | 1.10e+11 | 7.0% | yes |
| wt-dgrad[Wu] | BlockBwd | 4096 | 1.64e+04 | 5.32e+04 | 1 | 126 | 4.50e+14 | 1.10e+11 | 7.0% | yes |
| wt-wgrad[Wg] | BlockBwd | 5.32e+04 | 1.64e+04 | 4096 | 1 | 126 | 4.50e+14 | 1.10e+11 | 7.0% | yes |
| wt-wgrad[Wu] | BlockBwd | 5.32e+04 | 1.64e+04 | 4096 | 1 | 126 | 4.50e+14 | 1.10e+11 | 7.0% | yes |
| wt-fwd[Wq] | Block | 4096 | 1.64e+04 | 1.64e+04 | 1 | 126 | 1.39e+14 | 3.38e+10 | 2.1% | yes |
| wt-fwd[Wo] | Block | 4096 | 1.64e+04 | 1.64e+04 | 1 | 126 | 1.39e+14 | 3.38e+10 | 2.1% | yes |
| wt-fwd-recompute[Wq] | BlockBwd | 4096 | 1.64e+04 | 1.64e+04 | 1 | 126 | 1.39e+14 | 3.38e+10 | 2.1% | yes |
| wt-fwd-recompute[Wo] | BlockBwd | 4096 | 1.64e+04 | 1.64e+04 | 1 | 126 | 1.39e+14 | 3.38e+10 | 2.1% | yes |
| wt-dgrad[Wo] | BlockBwd | 4096 | 1.64e+04 | 1.64e+04 | 1 | 126 | 1.39e+14 | 3.38e+10 | 2.1% | yes |
| wt-wgrad[Wo] | BlockBwd | 1.64e+04 | 1.64e+04 | 4096 | 1 | 126 | 1.39e+14 | 3.38e+10 | 2.1% | yes |
| wt-dgrad[Wq] | BlockBwd | 4096 | 1.64e+04 | 1.64e+04 | 1 | 126 | 1.39e+14 | 3.38e+10 | 2.1% | yes |
| wt-wgrad[Wq] | BlockBwd | 1.64e+04 | 1.64e+04 | 4096 | 1 | 126 | 1.39e+14 | 3.38e+10 | 2.1% | yes |
| attn-bwd | BlockBwd/AttnSeqBwd | 4096 | 128 | 6.55e+04 | 8 | 252 | 6.93e+13 | 1.69e+10 | 1.1% | yes |
| attn-fwd | Block/AttnSeq | 4096 | 4096 | 128 | 128 | 126 | 3.46e+13 | 8.46e+09 | 0.5% | yes |
| attn-fwd | Block/AttnSeq | 4096 | 128 | 4096 | 128 | 126 | 3.46e+13 | 8.46e+09 | 0.5% | yes |
| attn-fwd-recompute | BlockBwd/AttnSeq | 4096 | 4096 | 128 | 128 | 126 | 3.46e+13 | 8.46e+09 | 0.5% | yes |
| attn-fwd-recompute | BlockBwd/AttnSeq | 4096 | 128 | 4096 | 128 | 126 | 3.46e+13 | 8.46e+09 | 0.5% | yes |
| attn-fwd-recompute | BlockBwd/AttnSeqBwd | 4096 | 4096 | 128 | 128 | 126 | 3.46e+13 | 8.46e+09 | 0.5% | yes |
| attn-bwd | BlockBwd/AttnSeqBwd | 4096 | 4096 | 128 | 128 | 126 | 3.46e+13 | 8.46e+09 | 0.5% | yes |
| attn-bwd | BlockBwd/AttnSeqBwd | 4096 | 128 | 4096 | 128 | 126 | 3.46e+13 | 8.46e+09 | 0.5% | yes |
| wt-fwd[Wk] | Block | 4096 | 1024 | 1.64e+04 | 1 | 126 | 8.66e+12 | 2.11e+09 | 0.1% | yes |
| wt-fwd[Wv] | Block | 4096 | 1024 | 1.64e+04 | 1 | 126 | 8.66e+12 | 2.11e+09 | 0.1% | yes |
| wt-fwd-recompute[Wk] | BlockBwd | 4096 | 1024 | 1.64e+04 | 1 | 126 | 8.66e+12 | 2.11e+09 | 0.1% | yes |
| wt-fwd-recompute[Wv] | BlockBwd | 4096 | 1024 | 1.64e+04 | 1 | 126 | 8.66e+12 | 2.11e+09 | 0.1% | yes |
| wt-dgrad[Wk] | BlockBwd | 4096 | 1.64e+04 | 1024 | 1 | 126 | 8.66e+12 | 2.11e+09 | 0.1% | yes |
| wt-dgrad[Wv] | BlockBwd | 4096 | 1.64e+04 | 1024 | 1 | 126 | 8.66e+12 | 2.11e+09 | 0.1% | yes |
| wt-wgrad[Wk] | BlockBwd | 1024 | 1.64e+04 | 4096 | 1 | 126 | 8.66e+12 | 2.11e+09 | 0.1% | yes |
| wt-wgrad[Wv] | BlockBwd | 1024 | 1.64e+04 | 4096 | 1 | 126 | 8.66e+12 | 2.11e+09 | 0.1% | yes |
| wt-fwd[Wh] |  | 4096 | 1.28e+05 | 1.64e+04 | 1 | 1 | 8.61e+12 | 2.10e+09 | 0.1% | yes |
| wt-dgrad[Wh] |  | 4096 | 1.64e+04 | 1.28e+05 | 1 | 1 | 8.61e+12 | 2.10e+09 | 0.1% | yes |
| wt-wgrad[Wh] |  | 1.28e+05 | 1.64e+04 | 4096 | 1 | 1 | 8.61e+12 | 2.10e+09 | 0.1% | yes |

By class:

| class | MACs | share |
|---|---|---|
| wt-fwd | 1.65e+15 | 25.6% |
| wt-dgrad | 1.65e+15 | 25.6% |
| wt-wgrad | 1.65e+15 | 25.6% |
| wt-fwd-recompute | 1.19e+15 | 18.5% |
| attn-bwd | 1.39e+14 | 2.1% |
| attn-fwd-recompute | 1.04e+14 | 1.6% |
| attn-fwd | 6.93e+13 | 1.1% |

Work by op kind (MAC-equivalents, all ops):

| kind | work |
|---|---|
| matmul | 6.47e+15 |
| softmax | 5.68e+12 |
| sgdupdate | 8.08e+11 |
| mul | 3.43e+11 |
| rmsnorm | 1.36e+11 |
| swiglu | 1.10e+11 |
| add | 6.76e+10 |
| colsum | 3.40e+10 |
| gain | 1.70e+10 |
| lossgrad | 4.20e+09 |
| embed | 1.14e+09 |

