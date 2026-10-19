# GLM v2 tree (`cursor/glm-v2-21f3`)

The integration tree the GLM-5.3 v2 rows of record are built, served and judged from (CIR-25, G1). It is never a PR: each
change lands through its own PR, and a row of record cites this tree as `cursor/glm-v2-21f3@SHA`. Merges only, both sides
kept; never `next`.

Rows (each I256/O32, TP8, H200, eager, at its public cap):

- `glm53__fp8__h200__tp8__b1__i256__o32__mixed__greedy__eager`
- `glm53__fp8__h200__tp8__b8__i256__o32__mixed-arrivals__greedy__eager`
- `glm53__fp8__h200__tp8__b8__i256__o32__mixed-arrivals__stoch-t0.8-p0.95__eager`
- `glm53__fp8__h200__tp8__b32__i256__o32__mixed-arrivals__greedy__eager`

## Merged

Base `origin/main` @ c0ebeb8b93, then, in this order:

| PR | head | carries |
|---|---|---|
| #2729 | f2629dde38 | Boolean-first; #2738 eead9401a9, #2694 f031a376f4, #2765 dec1238773, #2828 85d2659299, #2848 aaccf5e3d7 |
| #2836 | e2ef5e7ba1 | m1-rows (B8 multi-request path); #2831 3f13f1e1f2, #2828 85d2659299 |
| #2837 | d0fc072205 | a row of record done only at its public cap |
| #2838 | 615e13bdc3 | served-default capture harnesses (bf16 v6); #2828 85d2659299 |
| #2852 | ba1aa4b2d2 | TP served windows refused unless the run root is SHA-512; #2842 9ef0bf1d46 |
| #2824 | 5b5840cf7f | the committer's rows-sha512 iterations |
| `cursor/tp-sha512-21f3` (CIR-63 part 1, no PR yet) | d86c42a576 | TP Commits at rows-sha512 under the SHA-512 TP run root (`verity/tp-run-root/v2`); #2852 ba1aa4b2d2, #2824 5b5840cf7f |
| `cursor/boolv-21f3` (CIR-47, no PR yet) | a326a9ce23 | main c0ebeb8b93 merged with #2729's head: CIR-47's Boolean versions are #2729's σ-lifts and on-demand constants, so it changes nothing past #2729 |
| `cursor/rows64-land-21f3` (no PR yet) | 1d8869d1e4 | `MoeSum_v3` (`MoeSum_v1` on bits as a batch of `MoeSum_v2{TOPK,64}`), A11's Boolean-first probes |
| `cursor/multimem-land-21f3` (no PR yet) | 7338aa92c8 | `AllReduceMultimem_v1` as a batch of 16-byte vectors (`AllReduceMultimemVector_v2`) |

ROWS64 and the multimem batch joined when ASK-61 was provisionally accepted (Daniel, 19 Oct 21:59Z, option A: public
power-of-two output classes up to 262,144 bits). Two conflicts, both sides kept:

- `integrations/vllm/verity_vllm/pipeline/serving_rows.py` `after_commit` (TPSHA512 against #2848): the row-table layout's `row_tables`, and the
  provenance's `commitment` read from the committer's scheme.
- `integrations/vllm/tests/predict/test_predict.py`, GLM-5.3 B8 top-p's LP25/T6 request (multimem against the realized-arrival pin): this tree's
  predictor gives `b632583a…`, the realized-arrival pin; the vector form does not move that Program.

## To merge when their PR opens

- The branches above re-merge at their PR's head once it opens.
- CIR-44: the two Call splits. No branch yet.
- The static MoE frontend switch for GLM's expert GEMMs. The ops landed in #2665; no switch commit is on any branch.
  GLM's FP8 MoE in this tree is already the unpermuted pair set: per (token, slot) pair, the expert id is a value operand
  and the slab is gathered inside the unit (`MoeExpertFp8Gemm(W)_v1`, the `indexed-read` construction).

## Not in v2

- T8b's prefix commits (`cursor/prefix-tp-6f04`): ASK-42 is provisionally A (the prefix length is private), so a prefix row
  in a batch is proved cold and T8b's C-public commits stay out.
