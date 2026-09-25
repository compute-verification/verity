### hidden-adapter @ llama3-405b, tokens=4096, seq=4096 (r=16 ad=qkvogud)

| label | scope | M | N | K | copies | ops | MACs | MACs/token | share | ADW |
|---|---|---|---|---|---|---|---|---|---|---|
| wt-fwd[Wg] | LoraBlock | 4096 | 5.32e+04 | 1.64e+04 | 1 | 126 | 4.50e+14 | 1.10e+11 | 8.5% | yes |
| wt-fwd[Wu] | LoraBlock | 4096 | 5.32e+04 | 1.64e+04 | 1 | 126 | 4.50e+14 | 1.10e+11 | 8.5% | yes |
| wt-fwd[Wd] | LoraBlock | 4096 | 1.64e+04 | 5.32e+04 | 1 | 126 | 4.50e+14 | 1.10e+11 | 8.5% | yes |
| wt-fwd-recompute[Wg] | LoraBlockBwd | 4096 | 5.32e+04 | 1.64e+04 | 1 | 126 | 4.50e+14 | 1.10e+11 | 8.5% | yes |
| wt-fwd-recompute[Wu] | LoraBlockBwd | 4096 | 5.32e+04 | 1.64e+04 | 1 | 126 | 4.50e+14 | 1.10e+11 | 8.5% | yes |
| wt-fwd-recompute[Wd] | LoraBlockBwd | 4096 | 1.64e+04 | 5.32e+04 | 1 | 126 | 4.50e+14 | 1.10e+11 | 8.5% | yes |
| wt-dgrad[Wd] | LoraBlockBwd | 4096 | 5.32e+04 | 1.64e+04 | 1 | 126 | 4.50e+14 | 1.10e+11 | 8.5% | yes |
| wt-dgrad[Wg] | LoraBlockBwd | 4096 | 1.64e+04 | 5.32e+04 | 1 | 126 | 4.50e+14 | 1.10e+11 | 8.5% | yes |
| wt-dgrad[Wu] | LoraBlockBwd | 4096 | 1.64e+04 | 5.32e+04 | 1 | 126 | 4.50e+14 | 1.10e+11 | 8.5% | yes |
| wt-fwd[Wq] | LoraBlock | 4096 | 1.64e+04 | 1.64e+04 | 1 | 126 | 1.39e+14 | 3.38e+10 | 2.6% | no |
| wt-fwd[Wo] | LoraBlock | 4096 | 1.64e+04 | 1.64e+04 | 1 | 126 | 1.39e+14 | 3.38e+10 | 2.6% | yes |
| wt-fwd-recompute[Wq] | LoraBlockBwd | 4096 | 1.64e+04 | 1.64e+04 | 1 | 126 | 1.39e+14 | 3.38e+10 | 2.6% | yes |
| wt-fwd-recompute[Wo] | LoraBlockBwd | 4096 | 1.64e+04 | 1.64e+04 | 1 | 126 | 1.39e+14 | 3.38e+10 | 2.6% | yes |
| wt-dgrad[Wo] | LoraBlockBwd | 4096 | 1.64e+04 | 1.64e+04 | 1 | 126 | 1.39e+14 | 3.38e+10 | 2.6% | yes |
| wt-dgrad[Wq] | LoraBlockBwd | 4096 | 1.64e+04 | 1.64e+04 | 1 | 126 | 1.39e+14 | 3.38e+10 | 2.6% | yes |
| attn-bwd | LoraBlockBwd/AttnSeqBwd | 4096 | 128 | 6.55e+04 | 8 | 252 | 6.93e+13 | 1.69e+10 | 1.3% | yes |
| attn-fwd | LoraBlock/AttnSeq | 4096 | 4096 | 128 | 128 | 126 | 3.46e+13 | 8.46e+09 | 0.7% | yes |
| attn-fwd | LoraBlock/AttnSeq | 4096 | 128 | 4096 | 128 | 126 | 3.46e+13 | 8.46e+09 | 0.7% | yes |
| attn-fwd-recompute | LoraBlockBwd/AttnSeq | 4096 | 4096 | 128 | 128 | 126 | 3.46e+13 | 8.46e+09 | 0.7% | yes |
| attn-fwd-recompute | LoraBlockBwd/AttnSeq | 4096 | 128 | 4096 | 128 | 126 | 3.46e+13 | 8.46e+09 | 0.7% | yes |
| attn-fwd-recompute | LoraBlockBwd/AttnSeqBwd | 4096 | 4096 | 128 | 128 | 126 | 3.46e+13 | 8.46e+09 | 0.7% | yes |
| attn-bwd | LoraBlockBwd/AttnSeqBwd | 4096 | 4096 | 128 | 128 | 126 | 3.46e+13 | 8.46e+09 | 0.7% | yes |
| attn-bwd | LoraBlockBwd/AttnSeqBwd | 4096 | 128 | 4096 | 128 | 126 | 3.46e+13 | 8.46e+09 | 0.7% | yes |
| wt-fwd[Wk] | LoraBlock | 4096 | 1024 | 1.64e+04 | 1 | 126 | 8.66e+12 | 2.11e+09 | 0.2% | no |
| wt-fwd[Wv] | LoraBlock | 4096 | 1024 | 1.64e+04 | 1 | 126 | 8.66e+12 | 2.11e+09 | 0.2% | no |
| wt-fwd-recompute[Wk] | LoraBlockBwd | 4096 | 1024 | 1.64e+04 | 1 | 126 | 8.66e+12 | 2.11e+09 | 0.2% | yes |
| wt-fwd-recompute[Wv] | LoraBlockBwd | 4096 | 1024 | 1.64e+04 | 1 | 126 | 8.66e+12 | 2.11e+09 | 0.2% | yes |
| wt-dgrad[Wk] | LoraBlockBwd | 4096 | 1.64e+04 | 1024 | 1 | 126 | 8.66e+12 | 2.11e+09 | 0.2% | yes |
| wt-dgrad[Wv] | LoraBlockBwd | 4096 | 1.64e+04 | 1024 | 1 | 126 | 8.66e+12 | 2.11e+09 | 0.2% | yes |
| wt-fwd[Wh] |  | 4096 | 1.28e+05 | 1.64e+04 | 1 | 1 | 8.61e+12 | 2.10e+09 | 0.2% | yes |
| wt-dgrad[Wh] |  | 4096 | 1.64e+04 | 1.28e+05 | 1 | 1 | 8.61e+12 | 2.10e+09 | 0.2% | yes |
| wt-fwd[Bg] | LoraBlock | 4096 | 5.32e+04 | 16 | 1 | 126 | 4.40e+11 | 1.07e+08 | 0.0% | yes |
| wt-fwd[Bu] | LoraBlock | 4096 | 5.32e+04 | 16 | 1 | 126 | 4.40e+11 | 1.07e+08 | 0.0% | yes |
| wt-fwd[Ad] | LoraBlock | 4096 | 16 | 5.32e+04 | 1 | 126 | 4.40e+11 | 1.07e+08 | 0.0% | yes |
| wt-fwd-recompute[Bg] | LoraBlockBwd | 4096 | 5.32e+04 | 16 | 1 | 126 | 4.40e+11 | 1.07e+08 | 0.0% | yes |
| wt-fwd-recompute[Bu] | LoraBlockBwd | 4096 | 5.32e+04 | 16 | 1 | 126 | 4.40e+11 | 1.07e+08 | 0.0% | yes |
| wt-fwd-recompute[Ad] | LoraBlockBwd | 4096 | 16 | 5.32e+04 | 1 | 126 | 4.40e+11 | 1.07e+08 | 0.0% | yes |
| wt-dgrad[Ad] | LoraBlockBwd | 4096 | 5.32e+04 | 16 | 1 | 126 | 4.40e+11 | 1.07e+08 | 0.0% | yes |
| wt-wgrad[Ad] | LoraBlockBwd | 16 | 5.32e+04 | 4096 | 1 | 126 | 4.40e+11 | 1.07e+08 | 0.0% | yes |
| wt-dgrad[Bg] | LoraBlockBwd | 4096 | 16 | 5.32e+04 | 1 | 126 | 4.40e+11 | 1.07e+08 | 0.0% | yes |
| wt-wgrad[Bg] | LoraBlockBwd | 5.32e+04 | 16 | 4096 | 1 | 126 | 4.40e+11 | 1.07e+08 | 0.0% | yes |
| wt-dgrad[Bu] | LoraBlockBwd | 4096 | 16 | 5.32e+04 | 1 | 126 | 4.40e+11 | 1.07e+08 | 0.0% | yes |
| wt-wgrad[Bu] | LoraBlockBwd | 5.32e+04 | 16 | 4096 | 1 | 126 | 4.40e+11 | 1.07e+08 | 0.0% | yes |
| wt-fwd[Aq] | LoraBlock | 4096 | 16 | 1.64e+04 | 1 | 126 | 1.35e+11 | 3.30e+07 | 0.0% | yes |
| wt-fwd[Bq] | LoraBlock | 4096 | 1.64e+04 | 16 | 1 | 126 | 1.35e+11 | 3.30e+07 | 0.0% | yes |
| wt-fwd[Ak] | LoraBlock | 4096 | 16 | 1.64e+04 | 1 | 126 | 1.35e+11 | 3.30e+07 | 0.0% | yes |
| wt-fwd[Av] | LoraBlock | 4096 | 16 | 1.64e+04 | 1 | 126 | 1.35e+11 | 3.30e+07 | 0.0% | yes |
| wt-fwd[Ao] | LoraBlock | 4096 | 16 | 1.64e+04 | 1 | 126 | 1.35e+11 | 3.30e+07 | 0.0% | yes |
| wt-fwd[Bo] | LoraBlock | 4096 | 1.64e+04 | 16 | 1 | 126 | 1.35e+11 | 3.30e+07 | 0.0% | yes |
| wt-fwd[Ag] | LoraBlock | 4096 | 16 | 1.64e+04 | 1 | 126 | 1.35e+11 | 3.30e+07 | 0.0% | yes |
| wt-fwd[Au] | LoraBlock | 4096 | 16 | 1.64e+04 | 1 | 126 | 1.35e+11 | 3.30e+07 | 0.0% | yes |
| wt-fwd[Bd] | LoraBlock | 4096 | 1.64e+04 | 16 | 1 | 126 | 1.35e+11 | 3.30e+07 | 0.0% | yes |
| wt-fwd-recompute[Aq] | LoraBlockBwd | 4096 | 16 | 1.64e+04 | 1 | 126 | 1.35e+11 | 3.30e+07 | 0.0% | yes |
| wt-fwd-recompute[Bq] | LoraBlockBwd | 4096 | 1.64e+04 | 16 | 1 | 126 | 1.35e+11 | 3.30e+07 | 0.0% | yes |
| wt-fwd-recompute[Ak] | LoraBlockBwd | 4096 | 16 | 1.64e+04 | 1 | 126 | 1.35e+11 | 3.30e+07 | 0.0% | yes |
| wt-fwd-recompute[Av] | LoraBlockBwd | 4096 | 16 | 1.64e+04 | 1 | 126 | 1.35e+11 | 3.30e+07 | 0.0% | yes |
| wt-fwd-recompute[Ao] | LoraBlockBwd | 4096 | 16 | 1.64e+04 | 1 | 126 | 1.35e+11 | 3.30e+07 | 0.0% | yes |
| wt-fwd-recompute[Bo] | LoraBlockBwd | 4096 | 1.64e+04 | 16 | 1 | 126 | 1.35e+11 | 3.30e+07 | 0.0% | yes |
| wt-fwd-recompute[Ag] | LoraBlockBwd | 4096 | 16 | 1.64e+04 | 1 | 126 | 1.35e+11 | 3.30e+07 | 0.0% | yes |
| wt-fwd-recompute[Au] | LoraBlockBwd | 4096 | 16 | 1.64e+04 | 1 | 126 | 1.35e+11 | 3.30e+07 | 0.0% | yes |
| wt-fwd-recompute[Bd] | LoraBlockBwd | 4096 | 1.64e+04 | 16 | 1 | 126 | 1.35e+11 | 3.30e+07 | 0.0% | yes |
| wt-dgrad[Bd] | LoraBlockBwd | 4096 | 16 | 1.64e+04 | 1 | 126 | 1.35e+11 | 3.30e+07 | 0.0% | yes |
| wt-wgrad[Bd] | LoraBlockBwd | 1.64e+04 | 16 | 4096 | 1 | 126 | 1.35e+11 | 3.30e+07 | 0.0% | yes |
| wt-dgrad[Ag] | LoraBlockBwd | 4096 | 1.64e+04 | 16 | 1 | 126 | 1.35e+11 | 3.30e+07 | 0.0% | yes |
| wt-wgrad[Ag] | LoraBlockBwd | 16 | 1.64e+04 | 4096 | 1 | 126 | 1.35e+11 | 3.30e+07 | 0.0% | yes |
| wt-dgrad[Au] | LoraBlockBwd | 4096 | 1.64e+04 | 16 | 1 | 126 | 1.35e+11 | 3.30e+07 | 0.0% | yes |
| wt-wgrad[Au] | LoraBlockBwd | 16 | 1.64e+04 | 4096 | 1 | 126 | 1.35e+11 | 3.30e+07 | 0.0% | yes |
| wt-dgrad[Bo] | LoraBlockBwd | 4096 | 16 | 1.64e+04 | 1 | 126 | 1.35e+11 | 3.30e+07 | 0.0% | yes |
| wt-wgrad[Bo] | LoraBlockBwd | 1.64e+04 | 16 | 4096 | 1 | 126 | 1.35e+11 | 3.30e+07 | 0.0% | yes |
| wt-dgrad[Ao] | LoraBlockBwd | 4096 | 1.64e+04 | 16 | 1 | 126 | 1.35e+11 | 3.30e+07 | 0.0% | yes |
| wt-wgrad[Ao] | LoraBlockBwd | 16 | 1.64e+04 | 4096 | 1 | 126 | 1.35e+11 | 3.30e+07 | 0.0% | yes |
| wt-dgrad[Bq] | LoraBlockBwd | 4096 | 16 | 1.64e+04 | 1 | 126 | 1.35e+11 | 3.30e+07 | 0.0% | yes |
| wt-wgrad[Bq] | LoraBlockBwd | 1.64e+04 | 16 | 4096 | 1 | 126 | 1.35e+11 | 3.30e+07 | 0.0% | yes |
| wt-dgrad[Aq] | LoraBlockBwd | 4096 | 1.64e+04 | 16 | 1 | 126 | 1.35e+11 | 3.30e+07 | 0.0% | yes |
| wt-wgrad[Aq] | LoraBlockBwd | 16 | 1.64e+04 | 4096 | 1 | 126 | 1.35e+11 | 3.30e+07 | 0.0% | yes |
| wt-dgrad[Ak] | LoraBlockBwd | 4096 | 1.64e+04 | 16 | 1 | 126 | 1.35e+11 | 3.30e+07 | 0.0% | yes |
| wt-wgrad[Ak] | LoraBlockBwd | 16 | 1.64e+04 | 4096 | 1 | 126 | 1.35e+11 | 3.30e+07 | 0.0% | yes |
| wt-dgrad[Av] | LoraBlockBwd | 4096 | 1.64e+04 | 16 | 1 | 126 | 1.35e+11 | 3.30e+07 | 0.0% | yes |
| wt-wgrad[Av] | LoraBlockBwd | 16 | 1.64e+04 | 4096 | 1 | 126 | 1.35e+11 | 3.30e+07 | 0.0% | yes |
| wt-fwd[Bk] | LoraBlock | 4096 | 1024 | 16 | 1 | 126 | 8.46e+09 | 2.06e+06 | 0.0% | yes |
| wt-fwd[Bv] | LoraBlock | 4096 | 1024 | 16 | 1 | 126 | 8.46e+09 | 2.06e+06 | 0.0% | yes |
| wt-fwd-recompute[Bk] | LoraBlockBwd | 4096 | 1024 | 16 | 1 | 126 | 8.46e+09 | 2.06e+06 | 0.0% | yes |
| wt-fwd-recompute[Bv] | LoraBlockBwd | 4096 | 1024 | 16 | 1 | 126 | 8.46e+09 | 2.06e+06 | 0.0% | yes |
| wt-dgrad[Bk] | LoraBlockBwd | 4096 | 16 | 1024 | 1 | 126 | 8.46e+09 | 2.06e+06 | 0.0% | yes |
| wt-wgrad[Bk] | LoraBlockBwd | 1024 | 16 | 4096 | 1 | 126 | 8.46e+09 | 2.06e+06 | 0.0% | yes |
| wt-dgrad[Bv] | LoraBlockBwd | 4096 | 16 | 1024 | 1 | 126 | 8.46e+09 | 2.06e+06 | 0.0% | yes |
| wt-wgrad[Bv] | LoraBlockBwd | 1024 | 16 | 4096 | 1 | 126 | 8.46e+09 | 2.06e+06 | 0.0% | yes |

By class:

| class | MACs | share |
|---|---|---|
| wt-fwd | 1.66e+15 | 31.4% |
| wt-dgrad | 1.66e+15 | 31.4% |
| wt-fwd-recompute | 1.65e+15 | 31.2% |
| attn-bwd | 1.39e+14 | 2.6% |
| attn-fwd-recompute | 1.04e+14 | 2.0% |
| attn-fwd | 6.93e+13 | 1.3% |
| wt-wgrad | 2.55e+12 | 0.0% |

Work by op kind (MAC-equivalents, all ops):

| kind | work |
|---|---|
| matmul | 5.28e+15 |
| softmax | 5.68e+12 |
| mul | 3.26e+11 |
| add | 3.09e+11 |
| rmsnorm | 1.36e+11 |
| swiglu | 1.10e+11 |
| gain | 1.70e+10 |
| lossgrad | 4.20e+09 |
| sgdupdate | 1.25e+09 |
| embed | 1.14e+09 |

