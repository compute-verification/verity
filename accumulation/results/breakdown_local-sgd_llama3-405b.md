### local-sgd @ llama3-405b, tokens=8192, seq=4096 (H=2)

| label | scope | M | N | K | copies | ops | MACs | MACs/token | share | ADW |
|---|---|---|---|---|---|---|---|---|---|---|
| wt-wgrad[4096x53248] | BlockBwd | 4096 | 5.32e+04 | 1.64e+04 | 1 | 378 | 1.35e+15 | 1.65e+11 | 10.4% | yes |
| wt-wgrad[4096x53248] | Block | 4096 | 5.32e+04 | 1.64e+04 | 1 | 252 | 9.01e+14 | 1.10e+11 | 7.0% | yes |
| wt-wgrad[4096x16384] | BlockBwd | 4096 | 1.64e+04 | 5.32e+04 | 1 | 252 | 9.01e+14 | 1.10e+11 | 7.0% | yes |
| wt-wgrad[53248x16384] | BlockBwd | 5.32e+04 | 1.64e+04 | 4096 | 1 | 252 | 9.01e+14 | 1.10e+11 | 7.0% | yes |
| wt-wgrad[4096x16384] | BlockBwd | 4096 | 1.64e+04 | 1.64e+04 | 1 | 504 | 5.54e+14 | 6.76e+10 | 4.3% | yes |
| wt-fwd[Wg] | Block | 4096 | 5.32e+04 | 1.64e+04 | 1 | 126 | 4.50e+14 | 5.50e+10 | 3.5% | yes |
| wt-fwd[Wu] | Block | 4096 | 5.32e+04 | 1.64e+04 | 1 | 126 | 4.50e+14 | 5.50e+10 | 3.5% | yes |
| wt-fwd[Wd] | Block | 4096 | 1.64e+04 | 5.32e+04 | 1 | 126 | 4.50e+14 | 5.50e+10 | 3.5% | yes |
| wt-fwd-recompute[Wg] | BlockBwd | 4096 | 5.32e+04 | 1.64e+04 | 1 | 126 | 4.50e+14 | 5.50e+10 | 3.5% | yes |
| wt-fwd-recompute[Wu] | BlockBwd | 4096 | 5.32e+04 | 1.64e+04 | 1 | 126 | 4.50e+14 | 5.50e+10 | 3.5% | yes |
| wt-dgrad[Wd] | BlockBwd | 4096 | 5.32e+04 | 1.64e+04 | 1 | 126 | 4.50e+14 | 5.50e+10 | 3.5% | yes |
| wt-wgrad[Wd] | BlockBwd | 1.64e+04 | 5.32e+04 | 4096 | 1 | 126 | 4.50e+14 | 5.50e+10 | 3.5% | yes |
| wt-dgrad[Wg] | BlockBwd | 4096 | 1.64e+04 | 5.32e+04 | 1 | 126 | 4.50e+14 | 5.50e+10 | 3.5% | yes |
| wt-dgrad[Wu] | BlockBwd | 4096 | 1.64e+04 | 5.32e+04 | 1 | 126 | 4.50e+14 | 5.50e+10 | 3.5% | yes |
| wt-wgrad[Wg] | BlockBwd | 5.32e+04 | 1.64e+04 | 4096 | 1 | 126 | 4.50e+14 | 5.50e+10 | 3.5% | yes |
| wt-wgrad[Wu] | BlockBwd | 5.32e+04 | 1.64e+04 | 4096 | 1 | 126 | 4.50e+14 | 5.50e+10 | 3.5% | yes |
| wt-wgrad[4096x16384] | Block | 4096 | 1.64e+04 | 5.32e+04 | 1 | 126 | 4.50e+14 | 5.50e+10 | 3.5% | yes |
| wt-wgrad[16384x53248] | BlockBwd | 1.64e+04 | 5.32e+04 | 4096 | 1 | 126 | 4.50e+14 | 5.50e+10 | 3.5% | yes |
| wt-wgrad[4096x16384] | Block | 4096 | 1.64e+04 | 1.64e+04 | 1 | 252 | 2.77e+14 | 3.38e+10 | 2.1% | yes |
| wt-wgrad[16384x16384] | BlockBwd | 1.64e+04 | 1.64e+04 | 4096 | 1 | 252 | 2.77e+14 | 3.38e+10 | 2.1% | yes |
| wt-fwd[Wq] | Block | 4096 | 1.64e+04 | 1.64e+04 | 1 | 126 | 1.39e+14 | 1.69e+10 | 1.1% | yes |
| wt-fwd[Wo] | Block | 4096 | 1.64e+04 | 1.64e+04 | 1 | 126 | 1.39e+14 | 1.69e+10 | 1.1% | yes |
| wt-fwd-recompute[Wq] | BlockBwd | 4096 | 1.64e+04 | 1.64e+04 | 1 | 126 | 1.39e+14 | 1.69e+10 | 1.1% | yes |
| wt-fwd-recompute[Wo] | BlockBwd | 4096 | 1.64e+04 | 1.64e+04 | 1 | 126 | 1.39e+14 | 1.69e+10 | 1.1% | yes |
| wt-dgrad[Wo] | BlockBwd | 4096 | 1.64e+04 | 1.64e+04 | 1 | 126 | 1.39e+14 | 1.69e+10 | 1.1% | yes |
| wt-wgrad[Wo] | BlockBwd | 1.64e+04 | 1.64e+04 | 4096 | 1 | 126 | 1.39e+14 | 1.69e+10 | 1.1% | yes |
| attn-bwd | BlockBwd/AttnSeqBwd | 4096 | 128 | 6.55e+04 | 8 | 504 | 1.39e+14 | 1.69e+10 | 1.1% | yes |
| wt-dgrad[Wq] | BlockBwd | 4096 | 1.64e+04 | 1.64e+04 | 1 | 126 | 1.39e+14 | 1.69e+10 | 1.1% | yes |
| wt-wgrad[Wq] | BlockBwd | 1.64e+04 | 1.64e+04 | 4096 | 1 | 126 | 1.39e+14 | 1.69e+10 | 1.1% | yes |
| attn-bwd | BlockBwd/AttnSeqBwd | 4096 | 4096 | 128 | 128 | 378 | 1.04e+14 | 1.27e+10 | 0.8% | yes |
| attn-bwd | BlockBwd/AttnSeqBwd | 4096 | 128 | 4096 | 128 | 252 | 6.93e+13 | 8.46e+09 | 0.5% | yes |
| attn-fwd | Block/AttnSeq | 4096 | 4096 | 128 | 128 | 126 | 3.46e+13 | 4.23e+09 | 0.3% | yes |
| attn-fwd | Block/AttnSeq | 4096 | 128 | 4096 | 128 | 126 | 3.46e+13 | 4.23e+09 | 0.3% | yes |
| attn-fwd-recompute | BlockBwd/AttnSeq | 4096 | 4096 | 128 | 128 | 126 | 3.46e+13 | 4.23e+09 | 0.3% | yes |
| attn-fwd-recompute | BlockBwd/AttnSeq | 4096 | 128 | 4096 | 128 | 126 | 3.46e+13 | 4.23e+09 | 0.3% | yes |
| attn-fwd-recompute | BlockBwd/AttnSeqBwd | 4096 | 4096 | 128 | 128 | 126 | 3.46e+13 | 4.23e+09 | 0.3% | yes |
| attn-bwd | Block/AttnSeq | 4096 | 4096 | 128 | 128 | 126 | 3.46e+13 | 4.23e+09 | 0.3% | yes |
| attn-bwd | Block/AttnSeq | 4096 | 128 | 4096 | 128 | 126 | 3.46e+13 | 4.23e+09 | 0.3% | yes |
| attn-bwd | BlockBwd/AttnSeq | 4096 | 4096 | 128 | 128 | 126 | 3.46e+13 | 4.23e+09 | 0.3% | yes |
| attn-bwd | BlockBwd/AttnSeq | 4096 | 128 | 4096 | 128 | 126 | 3.46e+13 | 4.23e+09 | 0.3% | yes |
| wt-wgrad[4096x1024] | Block | 4096 | 1024 | 1.64e+04 | 1 | 252 | 1.73e+13 | 2.11e+09 | 0.1% | yes |
| wt-wgrad[4096x1024] | BlockBwd | 4096 | 1024 | 1.64e+04 | 1 | 252 | 1.73e+13 | 2.11e+09 | 0.1% | yes |
| wt-wgrad[4096x16384] | BlockBwd | 4096 | 1.64e+04 | 1024 | 1 | 252 | 1.73e+13 | 2.11e+09 | 0.1% | yes |
| wt-wgrad[1024x16384] | BlockBwd | 1024 | 1.64e+04 | 4096 | 1 | 252 | 1.73e+13 | 2.11e+09 | 0.1% | yes |
| wt-fwd[Wk] | Block | 4096 | 1024 | 1.64e+04 | 1 | 126 | 8.66e+12 | 1.06e+09 | 0.1% | yes |
| wt-fwd[Wv] | Block | 4096 | 1024 | 1.64e+04 | 1 | 126 | 8.66e+12 | 1.06e+09 | 0.1% | yes |
| wt-fwd-recompute[Wk] | BlockBwd | 4096 | 1024 | 1.64e+04 | 1 | 126 | 8.66e+12 | 1.06e+09 | 0.1% | yes |
| wt-fwd-recompute[Wv] | BlockBwd | 4096 | 1024 | 1.64e+04 | 1 | 126 | 8.66e+12 | 1.06e+09 | 0.1% | yes |
| wt-dgrad[Wk] | BlockBwd | 4096 | 1.64e+04 | 1024 | 1 | 126 | 8.66e+12 | 1.06e+09 | 0.1% | yes |
| wt-dgrad[Wv] | BlockBwd | 4096 | 1.64e+04 | 1024 | 1 | 126 | 8.66e+12 | 1.06e+09 | 0.1% | yes |
| wt-wgrad[Wk] | BlockBwd | 1024 | 1.64e+04 | 4096 | 1 | 126 | 8.66e+12 | 1.06e+09 | 0.1% | yes |
| wt-wgrad[Wv] | BlockBwd | 1024 | 1.64e+04 | 4096 | 1 | 126 | 8.66e+12 | 1.06e+09 | 0.1% | yes |
| wt-fwd[Wh] |  | 4096 | 1.28e+05 | 1.64e+04 | 1 | 1 | 8.61e+12 | 1.05e+09 | 0.1% | yes |
| wt-dgrad[Wh] |  | 4096 | 1.64e+04 | 1.28e+05 | 1 | 1 | 8.61e+12 | 1.05e+09 | 0.1% | yes |
| wt-wgrad[Wh] |  | 1.28e+05 | 1.64e+04 | 4096 | 1 | 1 | 8.61e+12 | 1.05e+09 | 0.1% | yes |
| wt-wgrad[4096x128256] |  | 4096 | 1.28e+05 | 1.64e+04 | 1 | 1 | 8.61e+12 | 1.05e+09 | 0.1% | yes |
| wt-wgrad[4096x16384] |  | 4096 | 1.64e+04 | 1.28e+05 | 1 | 1 | 8.61e+12 | 1.05e+09 | 0.1% | yes |
| wt-wgrad[128256x16384] |  | 1.28e+05 | 1.64e+04 | 4096 | 1 | 1 | 8.61e+12 | 1.05e+09 | 0.1% | yes |

By class:

| class | MACs | share |
|---|---|---|
| wt-wgrad | 7.81e+15 | 60.4% |
| wt-fwd | 1.65e+15 | 12.8% |
| wt-dgrad | 1.65e+15 | 12.8% |
| wt-fwd-recompute | 1.19e+15 | 9.2% |
| attn-bwd | 4.50e+14 | 3.5% |
| attn-fwd-recompute | 1.04e+14 | 0.8% |
| attn-fwd | 6.93e+13 | 0.5% |

Work by op kind (MAC-equivalents, all ops):

| kind | work |
|---|---|
| matmul | 1.29e+16 |
| softmax | 1.14e+13 |
| sgdupdate | 1.62e+12 |
| mul | 6.85e+11 |
| rmsnorm | 2.71e+11 |
| swiglu | 2.20e+11 |
| add | 1.35e+11 |
| colsum | 6.79e+10 |
| gain | 3.40e+10 |
| lossgrad | 8.41e+09 |
| embed | 2.28e+09 |

