### inference-moe @ mixtral-8x7b, tokens=4096, seq=4096 ()

| label | scope | M | N | K | copies | ops | MACs | MACs/token | share | ADW |
|---|---|---|---|---|---|---|---|---|---|---|
| wt-fwd[Wg] | MoeMlp/ExpertMlp | 1024 | 1.43e+04 | 4096 | 8 | 32 | 1.54e+13 | 3.76e+09 | 27.2% | no |
| wt-fwd[Wu] | MoeMlp/ExpertMlp | 1024 | 1.43e+04 | 4096 | 8 | 32 | 1.54e+13 | 3.76e+09 | 27.2% | no |
| wt-fwd[Wd] | MoeMlp/ExpertMlp | 1024 | 4096 | 1.43e+04 | 8 | 32 | 1.54e+13 | 3.76e+09 | 27.2% | no |
| wt-fwd[Wq] |  | 4096 | 4096 | 4096 | 1 | 32 | 2.20e+12 | 5.37e+08 | 3.9% | no |
| attn-fwd | AttnSeq | 4096 | 4096 | 128 | 32 | 32 | 2.20e+12 | 5.37e+08 | 3.9% | no |
| attn-fwd | AttnSeq | 4096 | 128 | 4096 | 32 | 32 | 2.20e+12 | 5.37e+08 | 3.9% | no |
| wt-fwd[Wo] |  | 4096 | 4096 | 4096 | 1 | 32 | 2.20e+12 | 5.37e+08 | 3.9% | no |
| wt-fwd[Wk] |  | 4096 | 1024 | 4096 | 1 | 32 | 5.50e+11 | 1.34e+08 | 1.0% | no |
| wt-fwd[Wv] |  | 4096 | 1024 | 4096 | 1 | 32 | 5.50e+11 | 1.34e+08 | 1.0% | no |
| wt-fwd[Wh] |  | 4096 | 3.20e+04 | 4096 | 1 | 1 | 5.37e+11 | 1.31e+08 | 0.9% | no |
| wt-fwd[Wr] | MoeMlp | 4096 | 8 | 4096 | 1 | 32 | 4.29e+09 | 1.05e+06 | 0.0% | no |

By class:

| class | MACs | share |
|---|---|---|
| wt-fwd | 5.22e+13 | 92.2% |
| attn-fwd | 4.40e+12 | 7.8% |

Work by op kind (MAC-equivalents, all ops):

| kind | work |
|---|---|
| matmul | 5.66e+13 |
| softmax | 1.20e+11 |
| swiglu | 7.52e+09 |
| rmsnorm | 4.36e+09 |
| add | 1.61e+09 |
| rowscale | 1.07e+09 |
| embed | 2.52e+08 |

