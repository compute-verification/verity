### inference-moe @ qwen3-235b-a22b, tokens=4096, seq=4096 ()

| label | scope | M | N | K | copies | ops | MACs | MACs/token | share | ADW |
|---|---|---|---|---|---|---|---|---|---|---|
| wt-fwd[Wg] | MoeMlp/ExpertMlp | 256 | 1536 | 4096 | 128 | 94 | 1.94e+13 | 4.73e+09 | 17.0% | no |
| wt-fwd[Wu] | MoeMlp/ExpertMlp | 256 | 1536 | 4096 | 128 | 94 | 1.94e+13 | 4.73e+09 | 17.0% | no |
| wt-fwd[Wd] | MoeMlp/ExpertMlp | 256 | 4096 | 1536 | 128 | 94 | 1.94e+13 | 4.73e+09 | 17.0% | no |
| wt-fwd[Wq] |  | 4096 | 8192 | 4096 | 1 | 94 | 1.29e+13 | 3.15e+09 | 11.3% | no |
| attn-fwd | AttnSeq | 4096 | 4096 | 128 | 64 | 94 | 1.29e+13 | 3.15e+09 | 11.3% | no |
| attn-fwd | AttnSeq | 4096 | 128 | 4096 | 64 | 94 | 1.29e+13 | 3.15e+09 | 11.3% | no |
| wt-fwd[Wo] |  | 4096 | 4096 | 8192 | 1 | 94 | 1.29e+13 | 3.15e+09 | 11.3% | no |
| wt-fwd[Wh] |  | 4096 | 1.52e+05 | 4096 | 1 | 1 | 2.55e+12 | 6.22e+08 | 2.2% | no |
| wt-fwd[Wk] |  | 4096 | 512 | 4096 | 1 | 94 | 8.07e+11 | 1.97e+08 | 0.7% | no |
| wt-fwd[Wv] |  | 4096 | 512 | 4096 | 1 | 94 | 8.07e+11 | 1.97e+08 | 0.7% | no |
| wt-fwd[Wr] | MoeMlp | 4096 | 128 | 4096 | 1 | 94 | 2.02e+11 | 4.93e+07 | 0.2% | no |

By class:

| class | MACs | share |
|---|---|---|
| wt-fwd | 8.83e+13 | 77.4% |
| attn-fwd | 2.58e+13 | 22.6% |

Work by op kind (MAC-equivalents, all ops):

| kind | work |
|---|---|
| matmul | 1.14e+14 |
| softmax | 7.07e+11 |
| add | 1.42e+10 |
| rmsnorm | 1.27e+10 |
| rowscale | 1.26e+10 |
| swiglu | 9.46e+09 |
| embed | 3.02e+08 |

