### hidden-adapter @ llama3-70b, tokens=4096, seq=4096 (r=16 ad=qkvogud)

| label | scope | M | N | K | copies | ops | MACs | MACs/token | share | ADW |
|---|---|---|---|---|---|---|---|---|---|---|
| wt-fwd[Wg] | LoraBlock | 4096 | 2.87e+04 | 8192 | 1 | 80 | 7.70e+13 | 1.88e+10 | 8.1% | yes |
| wt-fwd[Wu] | LoraBlock | 4096 | 2.87e+04 | 8192 | 1 | 80 | 7.70e+13 | 1.88e+10 | 8.1% | yes |
| wt-fwd[Wd] | LoraBlock | 4096 | 8192 | 2.87e+04 | 1 | 80 | 7.70e+13 | 1.88e+10 | 8.1% | yes |
| wt-fwd-recompute[Wg] | LoraBlockBwd | 4096 | 2.87e+04 | 8192 | 1 | 80 | 7.70e+13 | 1.88e+10 | 8.1% | yes |
| wt-fwd-recompute[Wu] | LoraBlockBwd | 4096 | 2.87e+04 | 8192 | 1 | 80 | 7.70e+13 | 1.88e+10 | 8.1% | yes |
| wt-fwd-recompute[Wd] | LoraBlockBwd | 4096 | 8192 | 2.87e+04 | 1 | 80 | 7.70e+13 | 1.88e+10 | 8.1% | yes |
| wt-dgrad[Wd] | LoraBlockBwd | 4096 | 2.87e+04 | 8192 | 1 | 80 | 7.70e+13 | 1.88e+10 | 8.1% | yes |
| wt-dgrad[Wg] | LoraBlockBwd | 4096 | 8192 | 2.87e+04 | 1 | 80 | 7.70e+13 | 1.88e+10 | 8.1% | yes |
| wt-dgrad[Wu] | LoraBlockBwd | 4096 | 8192 | 2.87e+04 | 1 | 80 | 7.70e+13 | 1.88e+10 | 8.1% | yes |
| wt-fwd[Wq] | LoraBlock | 4096 | 8192 | 8192 | 1 | 80 | 2.20e+13 | 5.37e+09 | 2.3% | no |
| wt-fwd[Wo] | LoraBlock | 4096 | 8192 | 8192 | 1 | 80 | 2.20e+13 | 5.37e+09 | 2.3% | yes |
| wt-fwd-recompute[Wq] | LoraBlockBwd | 4096 | 8192 | 8192 | 1 | 80 | 2.20e+13 | 5.37e+09 | 2.3% | yes |
| wt-fwd-recompute[Wo] | LoraBlockBwd | 4096 | 8192 | 8192 | 1 | 80 | 2.20e+13 | 5.37e+09 | 2.3% | yes |
| wt-dgrad[Wo] | LoraBlockBwd | 4096 | 8192 | 8192 | 1 | 80 | 2.20e+13 | 5.37e+09 | 2.3% | yes |
| attn-bwd | LoraBlockBwd/AttnSeqBwd | 4096 | 128 | 3.28e+04 | 8 | 160 | 2.20e+13 | 5.37e+09 | 2.3% | yes |
| wt-dgrad[Wq] | LoraBlockBwd | 4096 | 8192 | 8192 | 1 | 80 | 2.20e+13 | 5.37e+09 | 2.3% | yes |
| attn-fwd | LoraBlock/AttnSeq | 4096 | 4096 | 128 | 64 | 80 | 1.10e+13 | 2.68e+09 | 1.2% | yes |
| attn-fwd | LoraBlock/AttnSeq | 4096 | 128 | 4096 | 64 | 80 | 1.10e+13 | 2.68e+09 | 1.2% | yes |
| attn-fwd-recompute | LoraBlockBwd/AttnSeq | 4096 | 4096 | 128 | 64 | 80 | 1.10e+13 | 2.68e+09 | 1.2% | yes |
| attn-fwd-recompute | LoraBlockBwd/AttnSeq | 4096 | 128 | 4096 | 64 | 80 | 1.10e+13 | 2.68e+09 | 1.2% | yes |
| attn-fwd-recompute | LoraBlockBwd/AttnSeqBwd | 4096 | 4096 | 128 | 64 | 80 | 1.10e+13 | 2.68e+09 | 1.2% | yes |
| attn-bwd | LoraBlockBwd/AttnSeqBwd | 4096 | 4096 | 128 | 64 | 80 | 1.10e+13 | 2.68e+09 | 1.2% | yes |
| attn-bwd | LoraBlockBwd/AttnSeqBwd | 4096 | 128 | 4096 | 64 | 80 | 1.10e+13 | 2.68e+09 | 1.2% | yes |
| wt-fwd[Wh] |  | 4096 | 1.28e+05 | 8192 | 1 | 1 | 4.30e+12 | 1.05e+09 | 0.5% | yes |
| wt-dgrad[Wh] |  | 4096 | 8192 | 1.28e+05 | 1 | 1 | 4.30e+12 | 1.05e+09 | 0.5% | yes |
| wt-fwd[Wk] | LoraBlock | 4096 | 1024 | 8192 | 1 | 80 | 2.75e+12 | 6.71e+08 | 0.3% | no |
| wt-fwd[Wv] | LoraBlock | 4096 | 1024 | 8192 | 1 | 80 | 2.75e+12 | 6.71e+08 | 0.3% | no |
| wt-fwd-recompute[Wk] | LoraBlockBwd | 4096 | 1024 | 8192 | 1 | 80 | 2.75e+12 | 6.71e+08 | 0.3% | yes |
| wt-fwd-recompute[Wv] | LoraBlockBwd | 4096 | 1024 | 8192 | 1 | 80 | 2.75e+12 | 6.71e+08 | 0.3% | yes |
| wt-dgrad[Wk] | LoraBlockBwd | 4096 | 8192 | 1024 | 1 | 80 | 2.75e+12 | 6.71e+08 | 0.3% | yes |
| wt-dgrad[Wv] | LoraBlockBwd | 4096 | 8192 | 1024 | 1 | 80 | 2.75e+12 | 6.71e+08 | 0.3% | yes |
| wt-fwd[Bg] | LoraBlock | 4096 | 2.87e+04 | 16 | 1 | 80 | 1.50e+11 | 3.67e+07 | 0.0% | yes |
| wt-fwd[Bu] | LoraBlock | 4096 | 2.87e+04 | 16 | 1 | 80 | 1.50e+11 | 3.67e+07 | 0.0% | yes |
| wt-fwd[Ad] | LoraBlock | 4096 | 16 | 2.87e+04 | 1 | 80 | 1.50e+11 | 3.67e+07 | 0.0% | yes |
| wt-fwd-recompute[Bg] | LoraBlockBwd | 4096 | 2.87e+04 | 16 | 1 | 80 | 1.50e+11 | 3.67e+07 | 0.0% | yes |
| wt-fwd-recompute[Bu] | LoraBlockBwd | 4096 | 2.87e+04 | 16 | 1 | 80 | 1.50e+11 | 3.67e+07 | 0.0% | yes |
| wt-fwd-recompute[Ad] | LoraBlockBwd | 4096 | 16 | 2.87e+04 | 1 | 80 | 1.50e+11 | 3.67e+07 | 0.0% | yes |
| wt-dgrad[Ad] | LoraBlockBwd | 4096 | 2.87e+04 | 16 | 1 | 80 | 1.50e+11 | 3.67e+07 | 0.0% | yes |
| wt-wgrad[Ad] | LoraBlockBwd | 16 | 2.87e+04 | 4096 | 1 | 80 | 1.50e+11 | 3.67e+07 | 0.0% | yes |
| wt-dgrad[Bg] | LoraBlockBwd | 4096 | 16 | 2.87e+04 | 1 | 80 | 1.50e+11 | 3.67e+07 | 0.0% | yes |
| wt-wgrad[Bg] | LoraBlockBwd | 2.87e+04 | 16 | 4096 | 1 | 80 | 1.50e+11 | 3.67e+07 | 0.0% | yes |
| wt-dgrad[Bu] | LoraBlockBwd | 4096 | 16 | 2.87e+04 | 1 | 80 | 1.50e+11 | 3.67e+07 | 0.0% | yes |
| wt-wgrad[Bu] | LoraBlockBwd | 2.87e+04 | 16 | 4096 | 1 | 80 | 1.50e+11 | 3.67e+07 | 0.0% | yes |
| wt-fwd[Aq] | LoraBlock | 4096 | 16 | 8192 | 1 | 80 | 4.29e+10 | 1.05e+07 | 0.0% | yes |
| wt-fwd[Bq] | LoraBlock | 4096 | 8192 | 16 | 1 | 80 | 4.29e+10 | 1.05e+07 | 0.0% | yes |
| wt-fwd[Ak] | LoraBlock | 4096 | 16 | 8192 | 1 | 80 | 4.29e+10 | 1.05e+07 | 0.0% | yes |
| wt-fwd[Av] | LoraBlock | 4096 | 16 | 8192 | 1 | 80 | 4.29e+10 | 1.05e+07 | 0.0% | yes |
| wt-fwd[Ao] | LoraBlock | 4096 | 16 | 8192 | 1 | 80 | 4.29e+10 | 1.05e+07 | 0.0% | yes |
| wt-fwd[Bo] | LoraBlock | 4096 | 8192 | 16 | 1 | 80 | 4.29e+10 | 1.05e+07 | 0.0% | yes |
| wt-fwd[Ag] | LoraBlock | 4096 | 16 | 8192 | 1 | 80 | 4.29e+10 | 1.05e+07 | 0.0% | yes |
| wt-fwd[Au] | LoraBlock | 4096 | 16 | 8192 | 1 | 80 | 4.29e+10 | 1.05e+07 | 0.0% | yes |
| wt-fwd[Bd] | LoraBlock | 4096 | 8192 | 16 | 1 | 80 | 4.29e+10 | 1.05e+07 | 0.0% | yes |
| wt-fwd-recompute[Aq] | LoraBlockBwd | 4096 | 16 | 8192 | 1 | 80 | 4.29e+10 | 1.05e+07 | 0.0% | yes |
| wt-fwd-recompute[Bq] | LoraBlockBwd | 4096 | 8192 | 16 | 1 | 80 | 4.29e+10 | 1.05e+07 | 0.0% | yes |
| wt-fwd-recompute[Ak] | LoraBlockBwd | 4096 | 16 | 8192 | 1 | 80 | 4.29e+10 | 1.05e+07 | 0.0% | yes |
| wt-fwd-recompute[Av] | LoraBlockBwd | 4096 | 16 | 8192 | 1 | 80 | 4.29e+10 | 1.05e+07 | 0.0% | yes |
| wt-fwd-recompute[Ao] | LoraBlockBwd | 4096 | 16 | 8192 | 1 | 80 | 4.29e+10 | 1.05e+07 | 0.0% | yes |
| wt-fwd-recompute[Bo] | LoraBlockBwd | 4096 | 8192 | 16 | 1 | 80 | 4.29e+10 | 1.05e+07 | 0.0% | yes |
| wt-fwd-recompute[Ag] | LoraBlockBwd | 4096 | 16 | 8192 | 1 | 80 | 4.29e+10 | 1.05e+07 | 0.0% | yes |
| wt-fwd-recompute[Au] | LoraBlockBwd | 4096 | 16 | 8192 | 1 | 80 | 4.29e+10 | 1.05e+07 | 0.0% | yes |
| wt-fwd-recompute[Bd] | LoraBlockBwd | 4096 | 8192 | 16 | 1 | 80 | 4.29e+10 | 1.05e+07 | 0.0% | yes |
| wt-dgrad[Bd] | LoraBlockBwd | 4096 | 16 | 8192 | 1 | 80 | 4.29e+10 | 1.05e+07 | 0.0% | yes |
| wt-wgrad[Bd] | LoraBlockBwd | 8192 | 16 | 4096 | 1 | 80 | 4.29e+10 | 1.05e+07 | 0.0% | yes |
| wt-dgrad[Ag] | LoraBlockBwd | 4096 | 8192 | 16 | 1 | 80 | 4.29e+10 | 1.05e+07 | 0.0% | yes |
| wt-wgrad[Ag] | LoraBlockBwd | 16 | 8192 | 4096 | 1 | 80 | 4.29e+10 | 1.05e+07 | 0.0% | yes |
| wt-dgrad[Au] | LoraBlockBwd | 4096 | 8192 | 16 | 1 | 80 | 4.29e+10 | 1.05e+07 | 0.0% | yes |
| wt-wgrad[Au] | LoraBlockBwd | 16 | 8192 | 4096 | 1 | 80 | 4.29e+10 | 1.05e+07 | 0.0% | yes |
| wt-dgrad[Bo] | LoraBlockBwd | 4096 | 16 | 8192 | 1 | 80 | 4.29e+10 | 1.05e+07 | 0.0% | yes |
| wt-wgrad[Bo] | LoraBlockBwd | 8192 | 16 | 4096 | 1 | 80 | 4.29e+10 | 1.05e+07 | 0.0% | yes |
| wt-dgrad[Ao] | LoraBlockBwd | 4096 | 8192 | 16 | 1 | 80 | 4.29e+10 | 1.05e+07 | 0.0% | yes |
| wt-wgrad[Ao] | LoraBlockBwd | 16 | 8192 | 4096 | 1 | 80 | 4.29e+10 | 1.05e+07 | 0.0% | yes |
| wt-dgrad[Bq] | LoraBlockBwd | 4096 | 16 | 8192 | 1 | 80 | 4.29e+10 | 1.05e+07 | 0.0% | yes |
| wt-wgrad[Bq] | LoraBlockBwd | 8192 | 16 | 4096 | 1 | 80 | 4.29e+10 | 1.05e+07 | 0.0% | yes |
| wt-dgrad[Aq] | LoraBlockBwd | 4096 | 8192 | 16 | 1 | 80 | 4.29e+10 | 1.05e+07 | 0.0% | yes |
| wt-wgrad[Aq] | LoraBlockBwd | 16 | 8192 | 4096 | 1 | 80 | 4.29e+10 | 1.05e+07 | 0.0% | yes |
| wt-dgrad[Ak] | LoraBlockBwd | 4096 | 8192 | 16 | 1 | 80 | 4.29e+10 | 1.05e+07 | 0.0% | yes |
| wt-wgrad[Ak] | LoraBlockBwd | 16 | 8192 | 4096 | 1 | 80 | 4.29e+10 | 1.05e+07 | 0.0% | yes |
| wt-dgrad[Av] | LoraBlockBwd | 4096 | 8192 | 16 | 1 | 80 | 4.29e+10 | 1.05e+07 | 0.0% | yes |
| wt-wgrad[Av] | LoraBlockBwd | 16 | 8192 | 4096 | 1 | 80 | 4.29e+10 | 1.05e+07 | 0.0% | yes |
| wt-fwd[Bk] | LoraBlock | 4096 | 1024 | 16 | 1 | 80 | 5.37e+09 | 1.31e+06 | 0.0% | yes |
| wt-fwd[Bv] | LoraBlock | 4096 | 1024 | 16 | 1 | 80 | 5.37e+09 | 1.31e+06 | 0.0% | yes |
| wt-fwd-recompute[Bk] | LoraBlockBwd | 4096 | 1024 | 16 | 1 | 80 | 5.37e+09 | 1.31e+06 | 0.0% | yes |
| wt-fwd-recompute[Bv] | LoraBlockBwd | 4096 | 1024 | 16 | 1 | 80 | 5.37e+09 | 1.31e+06 | 0.0% | yes |
| wt-dgrad[Bk] | LoraBlockBwd | 4096 | 16 | 1024 | 1 | 80 | 5.37e+09 | 1.31e+06 | 0.0% | yes |
| wt-wgrad[Bk] | LoraBlockBwd | 1024 | 16 | 4096 | 1 | 80 | 5.37e+09 | 1.31e+06 | 0.0% | yes |
| wt-dgrad[Bv] | LoraBlockBwd | 4096 | 16 | 1024 | 1 | 80 | 5.37e+09 | 1.31e+06 | 0.0% | yes |
| wt-wgrad[Bv] | LoraBlockBwd | 1024 | 16 | 4096 | 1 | 80 | 5.37e+09 | 1.31e+06 | 0.0% | yes |

By class:

| class | MACs | share |
|---|---|---|
| wt-fwd | 2.86e+14 | 30.0% |
| wt-dgrad | 2.86e+14 | 30.0% |
| wt-fwd-recompute | 2.81e+14 | 29.5% |
| attn-bwd | 4.40e+13 | 4.6% |
| attn-fwd-recompute | 3.30e+13 | 3.5% |
| attn-fwd | 2.20e+13 | 2.3% |
| wt-wgrad | 8.48e+11 | 0.1% |

Work by op kind (MAC-equivalents, all ops):

| kind | work |
|---|---|
| matmul | 9.53e+14 |
| softmax | 1.80e+12 |
| mul | 1.05e+11 |
| add | 1.02e+11 |
| rmsnorm | 4.31e+10 |
| swiglu | 3.76e+10 |
| gain | 5.40e+09 |
| lossgrad | 4.20e+09 |
| embed | 5.70e+08 |
| sgdupdate | 4.14e+08 |

