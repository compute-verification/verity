### pretrain-moe @ deepseek-v3, tokens=4096, seq=4096 ()

| label | scope | M | N | K | copies | ops | MACs | MACs/token | share | ADW |
|---|---|---|---|---|---|---|---|---|---|---|
| wt-fwd[Wq] |  | 4096 | 1.64e+04 | 7168 | 1 | 58 | 2.79e+13 | 6.81e+09 | 13.5% | yes |
| wt-fwd[Wo] |  | 4096 | 7168 | 1.64e+04 | 1 | 58 | 2.79e+13 | 6.81e+09 | 13.5% | yes |
| wt-fwd[Wg] | MoeMlp/ExpertMlp | 128 | 2048 | 7168 | 256 | 58 | 2.79e+13 | 6.81e+09 | 13.5% | yes |
| wt-fwd[Wu] | MoeMlp/ExpertMlp | 128 | 2048 | 7168 | 256 | 58 | 2.79e+13 | 6.81e+09 | 13.5% | yes |
| wt-fwd[Wd] | MoeMlp/ExpertMlp | 128 | 7168 | 2048 | 256 | 58 | 2.79e+13 | 6.81e+09 | 13.5% | yes |
| attn-fwd | AttnSeq | 4096 | 4096 | 128 | 128 | 58 | 1.59e+13 | 3.89e+09 | 7.7% | yes |
| attn-fwd | AttnSeq | 4096 | 128 | 4096 | 128 | 58 | 1.59e+13 | 3.89e+09 | 7.7% | yes |
| wt-fwd[Wh] |  | 4096 | 1.29e+05 | 7168 | 1 | 1 | 3.80e+12 | 9.27e+08 | 1.8% | yes |
| wt-dgrad[Wh] |  | 4096 | 7168 | 1.29e+05 | 1 | 1 | 3.80e+12 | 9.27e+08 | 1.8% | yes |
| wt-wgrad[129280x7168] |  | 1.29e+05 | 7168 | 4096 | 1 | 1 | 3.80e+12 | 9.27e+08 | 1.8% | yes |
| wt-fwd[Wk] |  | 4096 | 2048 | 7168 | 1 | 58 | 3.49e+12 | 8.51e+08 | 1.7% | yes |
| wt-fwd[Wv] |  | 4096 | 2048 | 7168 | 1 | 58 | 3.49e+12 | 8.51e+08 | 1.7% | yes |
| wt-fwd[Wsg] | ExpertMlp | 4096 | 2048 | 7168 | 1 | 58 | 3.49e+12 | 8.51e+08 | 1.7% | yes |
| wt-fwd[Wsu] | ExpertMlp | 4096 | 2048 | 7168 | 1 | 58 | 3.49e+12 | 8.51e+08 | 1.7% | yes |
| wt-fwd[Wsd] | ExpertMlp | 4096 | 7168 | 2048 | 1 | 58 | 3.49e+12 | 8.51e+08 | 1.7% | yes |
| wt-fwd[Wq] | Block | 4096 | 1.64e+04 | 7168 | 1 | 3 | 1.44e+12 | 3.52e+08 | 0.7% | yes |
| wt-fwd[Wo] | Block | 4096 | 7168 | 1.64e+04 | 1 | 3 | 1.44e+12 | 3.52e+08 | 0.7% | yes |
| attn-fwd | Block/AttnSeq | 4096 | 4096 | 128 | 128 | 3 | 8.25e+11 | 2.01e+08 | 0.4% | yes |
| attn-fwd | Block/AttnSeq | 4096 | 128 | 4096 | 128 | 3 | 8.25e+11 | 2.01e+08 | 0.4% | yes |
| wt-fwd[Wr] | MoeMlp | 4096 | 256 | 7168 | 1 | 58 | 4.36e+11 | 1.06e+08 | 0.2% | yes |
| wt-fwd[Wk] | Block | 4096 | 2048 | 7168 | 1 | 3 | 1.80e+11 | 4.40e+07 | 0.1% | yes |
| wt-fwd[Wv] | Block | 4096 | 2048 | 7168 | 1 | 3 | 1.80e+11 | 4.40e+07 | 0.1% | yes |
| wt-fwd[Wgd] | Block | 4096 | 2048 | 7168 | 1 | 3 | 1.80e+11 | 4.40e+07 | 0.1% | yes |
| wt-fwd[Wud] | Block | 4096 | 2048 | 7168 | 1 | 3 | 1.80e+11 | 4.40e+07 | 0.1% | yes |
| wt-fwd[Wdd] | Block | 4096 | 7168 | 2048 | 1 | 3 | 1.80e+11 | 4.40e+07 | 0.1% | yes |

By class:

| class | MACs | share |
|---|---|---|
| wt-fwd | 1.65e+14 | 80.0% |
| attn-fwd | 3.35e+13 | 16.3% |
| wt-dgrad | 3.80e+12 | 1.8% |
| wt-wgrad | 3.80e+12 | 1.8% |

Work by op kind (MAC-equivalents, all ops):

| kind | work |
|---|---|
| matmul | 2.06e+14 |
| softmax | 9.17e+11 |
| add | 1.72e+10 |
| rmsnorm | 1.44e+10 |
| rowscale | 1.36e+10 |
| swiglu | 8.81e+09 |
| lossgrad | 4.24e+09 |
| embed | 4.99e+08 |
| colsum | 5.87e+07 |
| mul | 2.94e+07 |
| gain | 2.94e+07 |

