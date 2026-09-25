### es @ llama3-1b, tokens=32768, seq=4096 (pop=8)

| label | scope | M | N | K | copies | ops | MACs | MACs/token | share | ADW |
|---|---|---|---|---|---|---|---|---|---|---|
| other | Block | 4096 | 8192 | 2048 | 1 | 256 | 1.76e+13 | 5.37e+08 | 35.7% | yes |
| other | Block | 4096 | 2048 | 8192 | 1 | 128 | 8.80e+12 | 2.68e+08 | 17.8% | yes |
| other |  | 4096 | 1.28e+05 | 2048 | 1 | 8 | 8.61e+12 | 2.63e+08 | 17.5% | yes |
| other | Block | 4096 | 2048 | 2048 | 1 | 256 | 4.40e+12 | 1.34e+08 | 8.9% | yes |
| attn-fwd | Block/AttnSeq | 4096 | 4096 | 64 | 32 | 128 | 4.40e+12 | 1.34e+08 | 8.9% | yes |
| attn-fwd | Block/AttnSeq | 4096 | 64 | 4096 | 32 | 128 | 4.40e+12 | 1.34e+08 | 8.9% | yes |
| other | Block | 4096 | 512 | 2048 | 1 | 256 | 1.10e+12 | 3.36e+07 | 2.2% | yes |

By class:

| class | MACs | share |
|---|---|---|
| other | 4.05e+13 | 82.2% |
| attn-fwd | 8.80e+12 | 17.8% |

Work by op kind (MAC-equivalents, all ops):

| kind | work |
|---|---|
| matmul | 4.94e+13 |
| softmax | 4.81e+11 |
| perturb | 9.89e+10 |
| esupdate | 9.89e+10 |
| lossgrad | 3.36e+10 |
| rmsnorm | 8.86e+09 |
| swiglu | 8.59e+09 |
| colsum | 8.41e+09 |
| add | 2.15e+09 |
| embed | 1.14e+09 |
| scan:sumacc16 | 2.05e+06 |
| prim:AccRound16 | 8 |
| prim:AccWiden32 | 8 |
| prim:AccZero32 | 0 |

