### inference-dense @ llama3-405b, tokens=4096, seq=4096 ()

| label | scope | M | N | K | copies | ops | MACs | MACs/token | share | ADW |
|---|---|---|---|---|---|---|---|---|---|---|
| wt-fwd[Wg] | Block | 4096 | 5.32e+04 | 1.64e+04 | 1 | 126 | 4.50e+14 | 1.10e+11 | 26.1% | no |
| wt-fwd[Wu] | Block | 4096 | 5.32e+04 | 1.64e+04 | 1 | 126 | 4.50e+14 | 1.10e+11 | 26.1% | no |
| wt-fwd[Wd] | Block | 4096 | 1.64e+04 | 5.32e+04 | 1 | 126 | 4.50e+14 | 1.10e+11 | 26.1% | no |
| wt-fwd[Wq] | Block | 4096 | 1.64e+04 | 1.64e+04 | 1 | 126 | 1.39e+14 | 3.38e+10 | 8.0% | no |
| wt-fwd[Wo] | Block | 4096 | 1.64e+04 | 1.64e+04 | 1 | 126 | 1.39e+14 | 3.38e+10 | 8.0% | no |
| attn-fwd | Block/AttnSeq | 4096 | 4096 | 128 | 128 | 126 | 3.46e+13 | 8.46e+09 | 2.0% | no |
| attn-fwd | Block/AttnSeq | 4096 | 128 | 4096 | 128 | 126 | 3.46e+13 | 8.46e+09 | 2.0% | no |
| wt-fwd[Wk] | Block | 4096 | 1024 | 1.64e+04 | 1 | 126 | 8.66e+12 | 2.11e+09 | 0.5% | no |
| wt-fwd[Wv] | Block | 4096 | 1024 | 1.64e+04 | 1 | 126 | 8.66e+12 | 2.11e+09 | 0.5% | no |
| wt-fwd[Wh] |  | 4096 | 1.28e+05 | 1.64e+04 | 1 | 1 | 8.61e+12 | 2.10e+09 | 0.5% | no |

By class:

| class | MACs | share |
|---|---|---|
| wt-fwd | 1.65e+15 | 96.0% |
| attn-fwd | 6.93e+13 | 4.0% |

Work by op kind (MAC-equivalents, all ops):

| kind | work |
|---|---|
| matmul | 1.72e+15 |
| softmax | 1.89e+12 |
| rmsnorm | 6.79e+10 |
| swiglu | 5.50e+10 |
| add | 1.69e+10 |
| embed | 1.14e+09 |

