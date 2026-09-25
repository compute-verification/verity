### inference-dense @ llama3-70b, tokens=4096, seq=4096 ()

| label | scope | M | N | K | copies | ops | MACs | MACs/token | share | ADW |
|---|---|---|---|---|---|---|---|---|---|---|
| wt-fwd[Wg] | Block | 4096 | 2.87e+04 | 8192 | 1 | 80 | 7.70e+13 | 1.88e+10 | 25.1% | no |
| wt-fwd[Wu] | Block | 4096 | 2.87e+04 | 8192 | 1 | 80 | 7.70e+13 | 1.88e+10 | 25.1% | no |
| wt-fwd[Wd] | Block | 4096 | 8192 | 2.87e+04 | 1 | 80 | 7.70e+13 | 1.88e+10 | 25.1% | no |
| wt-fwd[Wq] | Block | 4096 | 8192 | 8192 | 1 | 80 | 2.20e+13 | 5.37e+09 | 7.2% | no |
| wt-fwd[Wo] | Block | 4096 | 8192 | 8192 | 1 | 80 | 2.20e+13 | 5.37e+09 | 7.2% | no |
| attn-fwd | Block/AttnSeq | 4096 | 4096 | 128 | 64 | 80 | 1.10e+13 | 2.68e+09 | 3.6% | no |
| attn-fwd | Block/AttnSeq | 4096 | 128 | 4096 | 64 | 80 | 1.10e+13 | 2.68e+09 | 3.6% | no |
| wt-fwd[Wh] |  | 4096 | 1.28e+05 | 8192 | 1 | 1 | 4.30e+12 | 1.05e+09 | 1.4% | no |
| wt-fwd[Wk] | Block | 4096 | 1024 | 8192 | 1 | 80 | 2.75e+12 | 6.71e+08 | 0.9% | no |
| wt-fwd[Wv] | Block | 4096 | 1024 | 8192 | 1 | 80 | 2.75e+12 | 6.71e+08 | 0.9% | no |

By class:

| class | MACs | share |
|---|---|---|
| wt-fwd | 2.85e+14 | 92.8% |
| attn-fwd | 2.20e+13 | 7.2% |

Work by op kind (MAC-equivalents, all ops):

| kind | work |
|---|---|
| matmul | 3.07e+14 |
| softmax | 6.01e+11 |
| rmsnorm | 2.16e+10 |
| swiglu | 1.88e+10 |
| add | 5.37e+09 |
| embed | 5.70e+08 |

