### forward-nonfixed @ llama3-1b, tokens=4096, seq=4096 ()

| label | scope | M | N | K | copies | ops | MACs | MACs/token | share | ADW |
|---|---|---|---|---|---|---|---|---|---|---|
| wt-fwd[Wg] | Block | 4096 | 8192 | 2048 | 1 | 16 | 1.10e+12 | 2.68e+08 | 17.8% | yes |
| wt-fwd[Wu] | Block | 4096 | 8192 | 2048 | 1 | 16 | 1.10e+12 | 2.68e+08 | 17.8% | yes |
| wt-fwd[Wd] | Block | 4096 | 2048 | 8192 | 1 | 16 | 1.10e+12 | 2.68e+08 | 17.8% | yes |
| wt-fwd[Wh] |  | 4096 | 1.28e+05 | 2048 | 1 | 1 | 1.08e+12 | 2.63e+08 | 17.5% | yes |
| attn-fwd | Block/AttnSeq | 4096 | 4096 | 64 | 32 | 16 | 5.50e+11 | 1.34e+08 | 8.9% | yes |
| attn-fwd | Block/AttnSeq | 4096 | 64 | 4096 | 32 | 16 | 5.50e+11 | 1.34e+08 | 8.9% | yes |
| wt-fwd[Wq] | Block | 4096 | 2048 | 2048 | 1 | 16 | 2.75e+11 | 6.71e+07 | 4.5% | yes |
| wt-fwd[Wo] | Block | 4096 | 2048 | 2048 | 1 | 16 | 2.75e+11 | 6.71e+07 | 4.5% | yes |
| wt-fwd[Wk] | Block | 4096 | 512 | 2048 | 1 | 16 | 6.87e+10 | 1.68e+07 | 1.1% | yes |
| wt-fwd[Wv] | Block | 4096 | 512 | 2048 | 1 | 16 | 6.87e+10 | 1.68e+07 | 1.1% | yes |

By class:

| class | MACs | share |
|---|---|---|
| wt-fwd | 5.06e+12 | 82.2% |
| attn-fwd | 1.10e+12 | 17.8% |

Work by op kind (MAC-equivalents, all ops):

| kind | work |
|---|---|
| matmul | 6.17e+12 |
| softmax | 6.01e+10 |
| rmsnorm | 1.11e+09 |
| swiglu | 1.07e+09 |
| add | 2.68e+08 |
| embed | 1.43e+08 |

