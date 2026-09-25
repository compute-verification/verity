### forward-nonfixed @ llama3-8b, tokens=4096, seq=4096 ()

| label | scope | M | N | K | copies | ops | MACs | MACs/token | share | ADW |
|---|---|---|---|---|---|---|---|---|---|---|
| wt-fwd[Wg] | Block | 4096 | 1.43e+04 | 4096 | 1 | 32 | 7.70e+12 | 1.88e+09 | 21.9% | yes |
| wt-fwd[Wu] | Block | 4096 | 1.43e+04 | 4096 | 1 | 32 | 7.70e+12 | 1.88e+09 | 21.9% | yes |
| wt-fwd[Wd] | Block | 4096 | 4096 | 1.43e+04 | 1 | 32 | 7.70e+12 | 1.88e+09 | 21.9% | yes |
| wt-fwd[Wq] | Block | 4096 | 4096 | 4096 | 1 | 32 | 2.20e+12 | 5.37e+08 | 6.3% | yes |
| attn-fwd | Block/AttnSeq | 4096 | 4096 | 128 | 32 | 32 | 2.20e+12 | 5.37e+08 | 6.3% | yes |
| attn-fwd | Block/AttnSeq | 4096 | 128 | 4096 | 32 | 32 | 2.20e+12 | 5.37e+08 | 6.3% | yes |
| wt-fwd[Wo] | Block | 4096 | 4096 | 4096 | 1 | 32 | 2.20e+12 | 5.37e+08 | 6.3% | yes |
| wt-fwd[Wh] |  | 4096 | 1.28e+05 | 4096 | 1 | 1 | 2.15e+12 | 5.25e+08 | 6.1% | yes |
| wt-fwd[Wk] | Block | 4096 | 1024 | 4096 | 1 | 32 | 5.50e+11 | 1.34e+08 | 1.6% | yes |
| wt-fwd[Wv] | Block | 4096 | 1024 | 4096 | 1 | 32 | 5.50e+11 | 1.34e+08 | 1.6% | yes |

By class:

| class | MACs | share |
|---|---|---|
| wt-fwd | 3.07e+13 | 87.5% |
| attn-fwd | 4.40e+12 | 12.5% |

Work by op kind (MAC-equivalents, all ops):

| kind | work |
|---|---|
| matmul | 3.52e+13 |
| softmax | 1.20e+11 |
| rmsnorm | 4.36e+09 |
| swiglu | 3.76e+09 |
| add | 1.07e+09 |
| embed | 2.85e+08 |

