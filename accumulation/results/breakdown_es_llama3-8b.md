### es @ llama3-8b, tokens=32768, seq=4096 (pop=8)

| label | scope | M | N | K | copies | ops | MACs | MACs/token | share | ADW |
|---|---|---|---|---|---|---|---|---|---|---|
| other | Block | 4096 | 1.43e+04 | 4096 | 1 | 512 | 1.23e+14 | 3.76e+09 | 43.8% | yes |
| other | Block | 4096 | 4096 | 1.43e+04 | 1 | 256 | 6.16e+13 | 1.88e+09 | 21.9% | yes |
| other | Block | 4096 | 4096 | 4096 | 1 | 512 | 3.52e+13 | 1.07e+09 | 12.5% | yes |
| attn-fwd | Block/AttnSeq | 4096 | 4096 | 128 | 32 | 256 | 1.76e+13 | 5.37e+08 | 6.3% | yes |
| attn-fwd | Block/AttnSeq | 4096 | 128 | 4096 | 32 | 256 | 1.76e+13 | 5.37e+08 | 6.3% | yes |
| other |  | 4096 | 1.28e+05 | 4096 | 1 | 8 | 1.72e+13 | 5.25e+08 | 6.1% | yes |
| other | Block | 4096 | 1024 | 4096 | 1 | 512 | 8.80e+12 | 2.68e+08 | 3.1% | yes |

By class:

| class | MACs | share |
|---|---|---|
| other | 2.46e+14 | 87.5% |
| attn-fwd | 3.52e+13 | 12.5% |

Work by op kind (MAC-equivalents, all ops):

| kind | work |
|---|---|
| matmul | 2.81e+14 |
| softmax | 9.62e+11 |
| perturb | 6.00e+11 |
| esupdate | 6.00e+11 |
| rmsnorm | 3.49e+10 |
| lossgrad | 3.36e+10 |
| swiglu | 3.01e+10 |
| add | 8.59e+09 |
| colsum | 8.41e+09 |
| embed | 2.28e+09 |
| scan:sumacc16 | 2.05e+06 |
| prim:AccRound16 | 8 |
| prim:AccWiden32 | 8 |
| prim:AccZero32 | 0 |

