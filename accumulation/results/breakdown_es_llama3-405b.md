### es @ llama3-405b, tokens=32768, seq=4096 (pop=8)

| label | scope | M | N | K | copies | ops | MACs | MACs/token | share | ADW |
|---|---|---|---|---|---|---|---|---|---|---|
| other | Block | 4096 | 5.32e+04 | 1.64e+04 | 1 | 2016 | 7.20e+15 | 2.20e+11 | 52.3% | yes |
| other | Block | 4096 | 1.64e+04 | 5.32e+04 | 1 | 1008 | 3.60e+15 | 1.10e+11 | 26.1% | yes |
| other | Block | 4096 | 1.64e+04 | 1.64e+04 | 1 | 2016 | 2.22e+15 | 6.76e+10 | 16.1% | yes |
| attn-fwd | Block/AttnSeq | 4096 | 4096 | 128 | 128 | 1008 | 2.77e+14 | 8.46e+09 | 2.0% | yes |
| attn-fwd | Block/AttnSeq | 4096 | 128 | 4096 | 128 | 1008 | 2.77e+14 | 8.46e+09 | 2.0% | yes |
| other | Block | 4096 | 1024 | 1.64e+04 | 1 | 2016 | 1.39e+14 | 4.23e+09 | 1.0% | yes |
| other |  | 4096 | 1.28e+05 | 1.64e+04 | 1 | 8 | 6.89e+13 | 2.10e+09 | 0.5% | yes |

By class:

| class | MACs | share |
|---|---|---|
| other | 1.32e+16 | 96.0% |
| attn-fwd | 5.54e+14 | 4.0% |

Work by op kind (MAC-equivalents, all ops):

| kind | work |
|---|---|
| matmul | 1.38e+16 |
| perturb | 3.23e+13 |
| esupdate | 3.23e+13 |
| softmax | 1.52e+13 |
| rmsnorm | 5.43e+11 |
| swiglu | 4.40e+11 |
| add | 1.35e+11 |
| lossgrad | 3.36e+10 |
| embed | 9.13e+09 |
| colsum | 8.41e+09 |
| scan:sumacc16 | 2.05e+06 |
| prim:AccRound16 | 8 |
| prim:AccWiden32 | 8 |
| prim:AccZero32 | 0 |

