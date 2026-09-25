### inference-moe @ deepseek-v3, tokens=4096, seq=4096 ()

| label | scope | M | N | K | copies | ops | MACs | MACs/token | share | ADW |
|---|---|---|---|---|---|---|---|---|---|---|
| wt-fwd[Wq] |  | 4096 | 1.64e+04 | 7168 | 1 | 58 | 2.79e+13 | 6.81e+09 | 14.1% | no |
| wt-fwd[Wo] |  | 4096 | 7168 | 1.64e+04 | 1 | 58 | 2.79e+13 | 6.81e+09 | 14.1% | no |
| wt-fwd[Wg] | MoeMlp/ExpertMlp | 128 | 2048 | 7168 | 256 | 58 | 2.79e+13 | 6.81e+09 | 14.1% | no |
| wt-fwd[Wu] | MoeMlp/ExpertMlp | 128 | 2048 | 7168 | 256 | 58 | 2.79e+13 | 6.81e+09 | 14.1% | no |
| wt-fwd[Wd] | MoeMlp/ExpertMlp | 128 | 7168 | 2048 | 256 | 58 | 2.79e+13 | 6.81e+09 | 14.1% | no |
| attn-fwd | AttnSeq | 4096 | 4096 | 128 | 128 | 58 | 1.59e+13 | 3.89e+09 | 8.0% | no |
| attn-fwd | AttnSeq | 4096 | 128 | 4096 | 128 | 58 | 1.59e+13 | 3.89e+09 | 8.0% | no |
| wt-fwd[Wh] |  | 4096 | 1.29e+05 | 7168 | 1 | 1 | 3.80e+12 | 9.27e+08 | 1.9% | no |
| wt-fwd[Wk] |  | 4096 | 2048 | 7168 | 1 | 58 | 3.49e+12 | 8.51e+08 | 1.8% | no |
| wt-fwd[Wv] |  | 4096 | 2048 | 7168 | 1 | 58 | 3.49e+12 | 8.51e+08 | 1.8% | no |
| wt-fwd[Wsg] | ExpertMlp | 4096 | 2048 | 7168 | 1 | 58 | 3.49e+12 | 8.51e+08 | 1.8% | no |
| wt-fwd[Wsu] | ExpertMlp | 4096 | 2048 | 7168 | 1 | 58 | 3.49e+12 | 8.51e+08 | 1.8% | no |
| wt-fwd[Wsd] | ExpertMlp | 4096 | 7168 | 2048 | 1 | 58 | 3.49e+12 | 8.51e+08 | 1.8% | no |
| wt-fwd[Wq] | Block | 4096 | 1.64e+04 | 7168 | 1 | 3 | 1.44e+12 | 3.52e+08 | 0.7% | no |
| wt-fwd[Wo] | Block | 4096 | 7168 | 1.64e+04 | 1 | 3 | 1.44e+12 | 3.52e+08 | 0.7% | no |
| attn-fwd | Block/AttnSeq | 4096 | 4096 | 128 | 128 | 3 | 8.25e+11 | 2.01e+08 | 0.4% | no |
| attn-fwd | Block/AttnSeq | 4096 | 128 | 4096 | 128 | 3 | 8.25e+11 | 2.01e+08 | 0.4% | no |
| wt-fwd[Wr] | MoeMlp | 4096 | 256 | 7168 | 1 | 58 | 4.36e+11 | 1.06e+08 | 0.2% | no |
| wt-fwd[Wk] | Block | 4096 | 2048 | 7168 | 1 | 3 | 1.80e+11 | 4.40e+07 | 0.1% | no |
| wt-fwd[Wv] | Block | 4096 | 2048 | 7168 | 1 | 3 | 1.80e+11 | 4.40e+07 | 0.1% | no |
| wt-fwd[Wgd] | Block | 4096 | 2048 | 7168 | 1 | 3 | 1.80e+11 | 4.40e+07 | 0.1% | no |
| wt-fwd[Wud] | Block | 4096 | 2048 | 7168 | 1 | 3 | 1.80e+11 | 4.40e+07 | 0.1% | no |
| wt-fwd[Wdd] | Block | 4096 | 7168 | 2048 | 1 | 3 | 1.80e+11 | 4.40e+07 | 0.1% | no |

By class:

| class | MACs | share |
|---|---|---|
| wt-fwd | 1.65e+14 | 83.1% |
| attn-fwd | 3.35e+13 | 16.9% |

Work by op kind (MAC-equivalents, all ops):

| kind | work |
|---|---|
| matmul | 1.99e+14 |
| softmax | 9.17e+11 |
| add | 1.72e+10 |
| rmsnorm | 1.44e+10 |
| rowscale | 1.36e+10 |
| swiglu | 8.81e+09 |
| embed | 4.99e+08 |

