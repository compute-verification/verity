### hidden-adapter @ llama3-8b, tokens=4096, seq=4096 (r=16 ad=qkvogud)

| label | scope | M | N | K | copies | ops | MACs | MACs/token | share | ADW |
|---|---|---|---|---|---|---|---|---|---|---|
| wt-fwd[Wg] | LoraBlock | 4096 | 1.43e+04 | 4096 | 1 | 32 | 7.70e+12 | 1.88e+09 | 7.0% | yes |
| wt-fwd[Wu] | LoraBlock | 4096 | 1.43e+04 | 4096 | 1 | 32 | 7.70e+12 | 1.88e+09 | 7.0% | yes |
| wt-fwd[Wd] | LoraBlock | 4096 | 4096 | 1.43e+04 | 1 | 32 | 7.70e+12 | 1.88e+09 | 7.0% | yes |
| wt-fwd-recompute[Wg] | LoraBlockBwd | 4096 | 1.43e+04 | 4096 | 1 | 32 | 7.70e+12 | 1.88e+09 | 7.0% | yes |
| wt-fwd-recompute[Wu] | LoraBlockBwd | 4096 | 1.43e+04 | 4096 | 1 | 32 | 7.70e+12 | 1.88e+09 | 7.0% | yes |
| wt-fwd-recompute[Wd] | LoraBlockBwd | 4096 | 4096 | 1.43e+04 | 1 | 32 | 7.70e+12 | 1.88e+09 | 7.0% | yes |
| wt-dgrad[Wd] | LoraBlockBwd | 4096 | 1.43e+04 | 4096 | 1 | 32 | 7.70e+12 | 1.88e+09 | 7.0% | yes |
| wt-dgrad[Wg] | LoraBlockBwd | 4096 | 4096 | 1.43e+04 | 1 | 32 | 7.70e+12 | 1.88e+09 | 7.0% | yes |
| wt-dgrad[Wu] | LoraBlockBwd | 4096 | 4096 | 1.43e+04 | 1 | 32 | 7.70e+12 | 1.88e+09 | 7.0% | yes |
| attn-bwd | LoraBlockBwd/AttnSeqBwd | 4096 | 128 | 1.64e+04 | 8 | 64 | 4.40e+12 | 1.07e+09 | 4.0% | yes |
| wt-fwd[Wq] | LoraBlock | 4096 | 4096 | 4096 | 1 | 32 | 2.20e+12 | 5.37e+08 | 2.0% | no |
| attn-fwd | LoraBlock/AttnSeq | 4096 | 4096 | 128 | 32 | 32 | 2.20e+12 | 5.37e+08 | 2.0% | yes |
| attn-fwd | LoraBlock/AttnSeq | 4096 | 128 | 4096 | 32 | 32 | 2.20e+12 | 5.37e+08 | 2.0% | yes |
| wt-fwd[Wo] | LoraBlock | 4096 | 4096 | 4096 | 1 | 32 | 2.20e+12 | 5.37e+08 | 2.0% | yes |
| wt-fwd-recompute[Wq] | LoraBlockBwd | 4096 | 4096 | 4096 | 1 | 32 | 2.20e+12 | 5.37e+08 | 2.0% | yes |
| attn-fwd-recompute | LoraBlockBwd/AttnSeq | 4096 | 4096 | 128 | 32 | 32 | 2.20e+12 | 5.37e+08 | 2.0% | yes |
| attn-fwd-recompute | LoraBlockBwd/AttnSeq | 4096 | 128 | 4096 | 32 | 32 | 2.20e+12 | 5.37e+08 | 2.0% | yes |
| wt-fwd-recompute[Wo] | LoraBlockBwd | 4096 | 4096 | 4096 | 1 | 32 | 2.20e+12 | 5.37e+08 | 2.0% | yes |
| wt-dgrad[Wo] | LoraBlockBwd | 4096 | 4096 | 4096 | 1 | 32 | 2.20e+12 | 5.37e+08 | 2.0% | yes |
| attn-fwd-recompute | LoraBlockBwd/AttnSeqBwd | 4096 | 4096 | 128 | 32 | 32 | 2.20e+12 | 5.37e+08 | 2.0% | yes |
| attn-bwd | LoraBlockBwd/AttnSeqBwd | 4096 | 4096 | 128 | 32 | 32 | 2.20e+12 | 5.37e+08 | 2.0% | yes |
| attn-bwd | LoraBlockBwd/AttnSeqBwd | 4096 | 128 | 4096 | 32 | 32 | 2.20e+12 | 5.37e+08 | 2.0% | yes |
| wt-dgrad[Wq] | LoraBlockBwd | 4096 | 4096 | 4096 | 1 | 32 | 2.20e+12 | 5.37e+08 | 2.0% | yes |
| wt-fwd[Wh] |  | 4096 | 1.28e+05 | 4096 | 1 | 1 | 2.15e+12 | 5.25e+08 | 1.9% | yes |
| wt-dgrad[Wh] |  | 4096 | 4096 | 1.28e+05 | 1 | 1 | 2.15e+12 | 5.25e+08 | 1.9% | yes |
| wt-fwd[Wk] | LoraBlock | 4096 | 1024 | 4096 | 1 | 32 | 5.50e+11 | 1.34e+08 | 0.5% | no |
| wt-fwd[Wv] | LoraBlock | 4096 | 1024 | 4096 | 1 | 32 | 5.50e+11 | 1.34e+08 | 0.5% | no |
| wt-fwd-recompute[Wk] | LoraBlockBwd | 4096 | 1024 | 4096 | 1 | 32 | 5.50e+11 | 1.34e+08 | 0.5% | yes |
| wt-fwd-recompute[Wv] | LoraBlockBwd | 4096 | 1024 | 4096 | 1 | 32 | 5.50e+11 | 1.34e+08 | 0.5% | yes |
| wt-dgrad[Wk] | LoraBlockBwd | 4096 | 4096 | 1024 | 1 | 32 | 5.50e+11 | 1.34e+08 | 0.5% | yes |
| wt-dgrad[Wv] | LoraBlockBwd | 4096 | 4096 | 1024 | 1 | 32 | 5.50e+11 | 1.34e+08 | 0.5% | yes |
| wt-fwd[Bg] | LoraBlock | 4096 | 1.43e+04 | 16 | 1 | 32 | 3.01e+10 | 7.34e+06 | 0.0% | yes |
| wt-fwd[Bu] | LoraBlock | 4096 | 1.43e+04 | 16 | 1 | 32 | 3.01e+10 | 7.34e+06 | 0.0% | yes |
| wt-fwd[Ad] | LoraBlock | 4096 | 16 | 1.43e+04 | 1 | 32 | 3.01e+10 | 7.34e+06 | 0.0% | yes |
| wt-fwd-recompute[Bg] | LoraBlockBwd | 4096 | 1.43e+04 | 16 | 1 | 32 | 3.01e+10 | 7.34e+06 | 0.0% | yes |
| wt-fwd-recompute[Bu] | LoraBlockBwd | 4096 | 1.43e+04 | 16 | 1 | 32 | 3.01e+10 | 7.34e+06 | 0.0% | yes |
| wt-fwd-recompute[Ad] | LoraBlockBwd | 4096 | 16 | 1.43e+04 | 1 | 32 | 3.01e+10 | 7.34e+06 | 0.0% | yes |
| wt-dgrad[Ad] | LoraBlockBwd | 4096 | 1.43e+04 | 16 | 1 | 32 | 3.01e+10 | 7.34e+06 | 0.0% | yes |
| wt-wgrad[Ad] | LoraBlockBwd | 16 | 1.43e+04 | 4096 | 1 | 32 | 3.01e+10 | 7.34e+06 | 0.0% | yes |
| wt-dgrad[Bg] | LoraBlockBwd | 4096 | 16 | 1.43e+04 | 1 | 32 | 3.01e+10 | 7.34e+06 | 0.0% | yes |
| wt-wgrad[Bg] | LoraBlockBwd | 1.43e+04 | 16 | 4096 | 1 | 32 | 3.01e+10 | 7.34e+06 | 0.0% | yes |
| wt-dgrad[Bu] | LoraBlockBwd | 4096 | 16 | 1.43e+04 | 1 | 32 | 3.01e+10 | 7.34e+06 | 0.0% | yes |
| wt-wgrad[Bu] | LoraBlockBwd | 1.43e+04 | 16 | 4096 | 1 | 32 | 3.01e+10 | 7.34e+06 | 0.0% | yes |
| wt-fwd[Aq] | LoraBlock | 4096 | 16 | 4096 | 1 | 32 | 8.59e+09 | 2.10e+06 | 0.0% | yes |
| wt-fwd[Bq] | LoraBlock | 4096 | 4096 | 16 | 1 | 32 | 8.59e+09 | 2.10e+06 | 0.0% | yes |
| wt-fwd[Ak] | LoraBlock | 4096 | 16 | 4096 | 1 | 32 | 8.59e+09 | 2.10e+06 | 0.0% | yes |
| wt-fwd[Av] | LoraBlock | 4096 | 16 | 4096 | 1 | 32 | 8.59e+09 | 2.10e+06 | 0.0% | yes |
| wt-fwd[Ao] | LoraBlock | 4096 | 16 | 4096 | 1 | 32 | 8.59e+09 | 2.10e+06 | 0.0% | yes |
| wt-fwd[Bo] | LoraBlock | 4096 | 4096 | 16 | 1 | 32 | 8.59e+09 | 2.10e+06 | 0.0% | yes |
| wt-fwd[Ag] | LoraBlock | 4096 | 16 | 4096 | 1 | 32 | 8.59e+09 | 2.10e+06 | 0.0% | yes |
| wt-fwd[Au] | LoraBlock | 4096 | 16 | 4096 | 1 | 32 | 8.59e+09 | 2.10e+06 | 0.0% | yes |
| wt-fwd[Bd] | LoraBlock | 4096 | 4096 | 16 | 1 | 32 | 8.59e+09 | 2.10e+06 | 0.0% | yes |
| wt-fwd-recompute[Aq] | LoraBlockBwd | 4096 | 16 | 4096 | 1 | 32 | 8.59e+09 | 2.10e+06 | 0.0% | yes |
| wt-fwd-recompute[Bq] | LoraBlockBwd | 4096 | 4096 | 16 | 1 | 32 | 8.59e+09 | 2.10e+06 | 0.0% | yes |
| wt-fwd-recompute[Ak] | LoraBlockBwd | 4096 | 16 | 4096 | 1 | 32 | 8.59e+09 | 2.10e+06 | 0.0% | yes |
| wt-fwd-recompute[Av] | LoraBlockBwd | 4096 | 16 | 4096 | 1 | 32 | 8.59e+09 | 2.10e+06 | 0.0% | yes |
| wt-fwd-recompute[Ao] | LoraBlockBwd | 4096 | 16 | 4096 | 1 | 32 | 8.59e+09 | 2.10e+06 | 0.0% | yes |
| wt-fwd-recompute[Bo] | LoraBlockBwd | 4096 | 4096 | 16 | 1 | 32 | 8.59e+09 | 2.10e+06 | 0.0% | yes |
| wt-fwd-recompute[Ag] | LoraBlockBwd | 4096 | 16 | 4096 | 1 | 32 | 8.59e+09 | 2.10e+06 | 0.0% | yes |
| wt-fwd-recompute[Au] | LoraBlockBwd | 4096 | 16 | 4096 | 1 | 32 | 8.59e+09 | 2.10e+06 | 0.0% | yes |
| wt-fwd-recompute[Bd] | LoraBlockBwd | 4096 | 4096 | 16 | 1 | 32 | 8.59e+09 | 2.10e+06 | 0.0% | yes |
| wt-dgrad[Bd] | LoraBlockBwd | 4096 | 16 | 4096 | 1 | 32 | 8.59e+09 | 2.10e+06 | 0.0% | yes |
| wt-wgrad[Bd] | LoraBlockBwd | 4096 | 16 | 4096 | 1 | 32 | 8.59e+09 | 2.10e+06 | 0.0% | yes |
| wt-dgrad[Ag] | LoraBlockBwd | 4096 | 4096 | 16 | 1 | 32 | 8.59e+09 | 2.10e+06 | 0.0% | yes |
| wt-wgrad[Ag] | LoraBlockBwd | 16 | 4096 | 4096 | 1 | 32 | 8.59e+09 | 2.10e+06 | 0.0% | yes |
| wt-dgrad[Au] | LoraBlockBwd | 4096 | 4096 | 16 | 1 | 32 | 8.59e+09 | 2.10e+06 | 0.0% | yes |
| wt-wgrad[Au] | LoraBlockBwd | 16 | 4096 | 4096 | 1 | 32 | 8.59e+09 | 2.10e+06 | 0.0% | yes |
| wt-dgrad[Bo] | LoraBlockBwd | 4096 | 16 | 4096 | 1 | 32 | 8.59e+09 | 2.10e+06 | 0.0% | yes |
| wt-wgrad[Bo] | LoraBlockBwd | 4096 | 16 | 4096 | 1 | 32 | 8.59e+09 | 2.10e+06 | 0.0% | yes |
| wt-dgrad[Ao] | LoraBlockBwd | 4096 | 4096 | 16 | 1 | 32 | 8.59e+09 | 2.10e+06 | 0.0% | yes |
| wt-wgrad[Ao] | LoraBlockBwd | 16 | 4096 | 4096 | 1 | 32 | 8.59e+09 | 2.10e+06 | 0.0% | yes |
| wt-dgrad[Bq] | LoraBlockBwd | 4096 | 16 | 4096 | 1 | 32 | 8.59e+09 | 2.10e+06 | 0.0% | yes |
| wt-wgrad[Bq] | LoraBlockBwd | 4096 | 16 | 4096 | 1 | 32 | 8.59e+09 | 2.10e+06 | 0.0% | yes |
| wt-dgrad[Aq] | LoraBlockBwd | 4096 | 4096 | 16 | 1 | 32 | 8.59e+09 | 2.10e+06 | 0.0% | yes |
| wt-wgrad[Aq] | LoraBlockBwd | 16 | 4096 | 4096 | 1 | 32 | 8.59e+09 | 2.10e+06 | 0.0% | yes |
| wt-dgrad[Ak] | LoraBlockBwd | 4096 | 4096 | 16 | 1 | 32 | 8.59e+09 | 2.10e+06 | 0.0% | yes |
| wt-wgrad[Ak] | LoraBlockBwd | 16 | 4096 | 4096 | 1 | 32 | 8.59e+09 | 2.10e+06 | 0.0% | yes |
| wt-dgrad[Av] | LoraBlockBwd | 4096 | 4096 | 16 | 1 | 32 | 8.59e+09 | 2.10e+06 | 0.0% | yes |
| wt-wgrad[Av] | LoraBlockBwd | 16 | 4096 | 4096 | 1 | 32 | 8.59e+09 | 2.10e+06 | 0.0% | yes |
| wt-fwd[Bk] | LoraBlock | 4096 | 1024 | 16 | 1 | 32 | 2.15e+09 | 5.24e+05 | 0.0% | yes |
| wt-fwd[Bv] | LoraBlock | 4096 | 1024 | 16 | 1 | 32 | 2.15e+09 | 5.24e+05 | 0.0% | yes |
| wt-fwd-recompute[Bk] | LoraBlockBwd | 4096 | 1024 | 16 | 1 | 32 | 2.15e+09 | 5.24e+05 | 0.0% | yes |
| wt-fwd-recompute[Bv] | LoraBlockBwd | 4096 | 1024 | 16 | 1 | 32 | 2.15e+09 | 5.24e+05 | 0.0% | yes |
| wt-dgrad[Bk] | LoraBlockBwd | 4096 | 16 | 1024 | 1 | 32 | 2.15e+09 | 5.24e+05 | 0.0% | yes |
| wt-wgrad[Bk] | LoraBlockBwd | 1024 | 16 | 4096 | 1 | 32 | 2.15e+09 | 5.24e+05 | 0.0% | yes |
| wt-dgrad[Bv] | LoraBlockBwd | 4096 | 16 | 1024 | 1 | 32 | 2.15e+09 | 5.24e+05 | 0.0% | yes |
| wt-wgrad[Bv] | LoraBlockBwd | 1024 | 16 | 4096 | 1 | 32 | 2.15e+09 | 5.24e+05 | 0.0% | yes |

By class:

| class | MACs | share |
|---|---|---|
| wt-fwd | 3.09e+13 | 28.0% |
| wt-dgrad | 3.09e+13 | 28.0% |
| wt-fwd-recompute | 2.88e+13 | 26.0% |
| attn-bwd | 8.80e+12 | 8.0% |
| attn-fwd-recompute | 6.60e+12 | 6.0% |
| attn-fwd | 4.40e+12 | 4.0% |
| wt-wgrad | 1.72e+11 | 0.2% |

Work by op kind (MAC-equivalents, all ops):

| kind | work |
|---|---|
| matmul | 1.11e+14 |
| softmax | 3.61e+11 |
| mul | 2.09e+10 |
| add | 2.07e+10 |
| rmsnorm | 8.66e+09 |
| swiglu | 7.52e+09 |
| lossgrad | 4.20e+09 |
| gain | 1.09e+09 |
| embed | 2.85e+08 |
| sgdupdate | 8.39e+07 |

