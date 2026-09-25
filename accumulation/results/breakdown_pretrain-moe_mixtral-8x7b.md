### pretrain-moe @ mixtral-8x7b, tokens=4096, seq=4096 ()

| label | scope | M | N | K | copies | ops | MACs | MACs/token | share | ADW |
|---|---|---|---|---|---|---|---|---|---|---|
| wt-fwd[Wg] | MoeMlp/ExpertMlp | 1024 | 1.43e+04 | 4096 | 8 | 32 | 1.54e+13 | 3.76e+09 | 26.7% | yes |
| wt-fwd[Wu] | MoeMlp/ExpertMlp | 1024 | 1.43e+04 | 4096 | 8 | 32 | 1.54e+13 | 3.76e+09 | 26.7% | yes |
| wt-fwd[Wd] | MoeMlp/ExpertMlp | 1024 | 4096 | 1.43e+04 | 8 | 32 | 1.54e+13 | 3.76e+09 | 26.7% | yes |
| wt-fwd[Wq] |  | 4096 | 4096 | 4096 | 1 | 32 | 2.20e+12 | 5.37e+08 | 3.8% | yes |
| attn-fwd | AttnSeq | 4096 | 4096 | 128 | 32 | 32 | 2.20e+12 | 5.37e+08 | 3.8% | yes |
| attn-fwd | AttnSeq | 4096 | 128 | 4096 | 32 | 32 | 2.20e+12 | 5.37e+08 | 3.8% | yes |
| wt-fwd[Wo] |  | 4096 | 4096 | 4096 | 1 | 32 | 2.20e+12 | 5.37e+08 | 3.8% | yes |
| wt-fwd[Wk] |  | 4096 | 1024 | 4096 | 1 | 32 | 5.50e+11 | 1.34e+08 | 1.0% | yes |
| wt-fwd[Wv] |  | 4096 | 1024 | 4096 | 1 | 32 | 5.50e+11 | 1.34e+08 | 1.0% | yes |
| wt-fwd[Wh] |  | 4096 | 3.20e+04 | 4096 | 1 | 1 | 5.37e+11 | 1.31e+08 | 0.9% | yes |
| wt-dgrad[Wh] |  | 4096 | 4096 | 3.20e+04 | 1 | 1 | 5.37e+11 | 1.31e+08 | 0.9% | yes |
| wt-wgrad[32000x4096] |  | 3.20e+04 | 4096 | 4096 | 1 | 1 | 5.37e+11 | 1.31e+08 | 0.9% | yes |
| wt-fwd[Wr] | MoeMlp | 4096 | 8 | 4096 | 1 | 32 | 4.29e+09 | 1.05e+06 | 0.0% | yes |

By class:

| class | MACs | share |
|---|---|---|
| wt-fwd | 5.22e+13 | 90.5% |
| attn-fwd | 4.40e+12 | 7.6% |
| wt-dgrad | 5.37e+11 | 0.9% |
| wt-wgrad | 5.37e+11 | 0.9% |

Work by op kind (MAC-equivalents, all ops):

| kind | work |
|---|---|
| matmul | 5.77e+13 |
| softmax | 1.20e+11 |
| swiglu | 7.52e+09 |
| rmsnorm | 4.36e+09 |
| add | 1.61e+09 |
| rowscale | 1.07e+09 |
| lossgrad | 1.05e+09 |
| embed | 2.52e+08 |
| colsum | 3.36e+07 |
| mul | 1.68e+07 |
| gain | 1.68e+07 |

