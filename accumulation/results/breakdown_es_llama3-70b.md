### es @ llama3-70b, tokens=32768, seq=4096 (pop=8)

| label | scope | M | N | K | copies | ops | MACs | MACs/token | share | ADW |
|---|---|---|---|---|---|---|---|---|---|---|
| other | Block | 4096 | 2.87e+04 | 8192 | 1 | 1280 | 1.23e+15 | 3.76e+10 | 50.2% | yes |
| other | Block | 4096 | 8192 | 2.87e+04 | 1 | 640 | 6.16e+14 | 1.88e+10 | 25.1% | yes |
| other | Block | 4096 | 8192 | 8192 | 1 | 1280 | 3.52e+14 | 1.07e+10 | 14.3% | yes |
| attn-fwd | Block/AttnSeq | 4096 | 4096 | 128 | 64 | 640 | 8.80e+13 | 2.68e+09 | 3.6% | yes |
| attn-fwd | Block/AttnSeq | 4096 | 128 | 4096 | 64 | 640 | 8.80e+13 | 2.68e+09 | 3.6% | yes |
| other | Block | 4096 | 1024 | 8192 | 1 | 1280 | 4.40e+13 | 1.34e+09 | 1.8% | yes |
| other |  | 4096 | 1.28e+05 | 8192 | 1 | 8 | 3.44e+13 | 1.05e+09 | 1.4% | yes |

By class:

| class | MACs | share |
|---|---|---|
| other | 2.28e+15 | 92.8% |
| attn-fwd | 1.76e+14 | 7.2% |

Work by op kind (MAC-equivalents, all ops):

| kind | work |
|---|---|
| matmul | 2.45e+15 |
| perturb | 5.56e+12 |
| esupdate | 5.56e+12 |
| softmax | 4.81e+12 |
| rmsnorm | 1.73e+11 |
| swiglu | 1.50e+11 |
| add | 4.29e+10 |
| lossgrad | 3.36e+10 |
| colsum | 8.41e+09 |
| embed | 4.56e+09 |
| scan:sumacc16 | 2.05e+06 |
| prim:AccRound16 | 8 |
| prim:AccWiden32 | 8 |
| prim:AccZero32 | 0 |

