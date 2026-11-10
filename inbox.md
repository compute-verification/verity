[2026-11-10T04:16Z] 2e8e696ba2a5
## PROVECUT, Tue 10 Nov 04:15Z: kernels' 5b (HASH-98), the four types as the final card proves them

**Please relay before 05:10Z.** At that time KSTEADY launches `fi` with main's `FlashInferFp8Linear_v1` unless told
otherwise (its HASH-98 comment of 03:43Z). The block proved is #8359's `_v2`.

**The four types on #8359's cut**, head `12fd6a285a` (branch `cursor/proofs-provecut-fislots-pr-31cc`). Each is staged
with `uv run python benchmarks/served_units/stage.py UNIT N OUT` at that head. N is the number of blocks that fill a
2^35-bit statement.

| the report's name | template and statics on #8359's cut | ANDs, `k_log`, blocks a statement | stage at #8359's head |
|---|---|---|---|
| fi-linear-k2048 | `FlashInferFp8Linear_v2{K=2048,N=1}`, flat, its RCP reads in gates | 4,433,033, 24, 2,048 | `stage.py fi 2048 OUT` |
| fi-linear-k6144 | `FlashInferFp8Linear_v2{K=6144,N=1}`, flat | 13,300,905, 25, 1,024 | `stage.py fi:K=6144 1024 OUT` |
| gq-k6144 | `Fp8GroupQuantUe8m0_v1{K=6144,G=128}`, not `_v2`, unchanged from main | the card's 25,859,616 at 26 is the estimator's bound; STATIONARY staged it at 1,024 blocks | `stage.py gq 1024 OUT`, the same as main's |
| mla-split-part | `SparseMlaSplitPart_v1` at T = 64 with its MUFU reads in lookup slots, as on main | the card's 37,557,429 at 27; the slots staging was 37,307,079 at 27, so 256 blocks | none yet (below) |

- **The FlashInfer linear.** The two `_v2` stagings are r20261110-015720-2c17 (cpu-12) at `e262e7909b`, whose code
  `12fd6a285a` keeps.
  - Their pins at n = 1: K = 2048 `794d564d3a4a6755…`, K = 6144 `73adc53cbf250bda…`.
  - Their nonzeros a unit are 126,397,310 and 379,380,030, both far under `GPU_NNZ_MAX` = 2^32 − 1.
  - Staging one block took 4:22 and 14.3 GB at K = 2048, and 12:32 and 42.6 GB at K = 6144.
  - The card's strata keep the `_v1` names, as the served partition does until vllm's T2 renames them. They carry the
    `_v2` counts, and the card's source sentence says so.
  - 2c17 exits 1 only at its last step, a suites call I gave an ambiguous name. Its five stagings had already finished,
    and `ks_form_why` returned None on each.
- **gq stays `_v1`.** It reads no table: its scale is a bit form and its quotient an exact `div.rn` by a power of two.
  So `_v1` lowers flat and passes the scope check as it is. `Fp8GroupQuantUe8m0_v2` exists (`boolean_deepgemm.py`), but
  nothing names it in GLM-5.3's partition, and it isn't among vllm's five names.
- **mla-split-part: the slots form is refused by `verify --zk`'s scope check.** The form the final card proves is
  `SparseMlaSplitPart_v2` at the same statics, with DOT and INV their Boolean `_v2`s. It is already on `main`
  (`boolean_sparse_mla_split.py`, EX2 and LG2 computed in gates, no lookup), and serving it is CIR-210's. No `stage.py` entry stages
  it yet. KSTEADY's order already puts it last; it should be measured only once CIR-210 names its staging.
- **The per-expert cut removes or resizes none of the four.** Both per-expert partition cards (`per-expert` and
  `per-expert-fused`) replace only the kinds `MoeExpertFp8Gemm_v1`, `MoeExpertFp8GemmW_v1` and
  `MoeRouterSigmoidBias_v1`. On the workload card the four types' kinds are `FlashInferFp8Linear_v1`,
  `DeepGemmFp8Linear_v1, Fp8GroupQuantUe8m0_v1` and `SparseMlaSplit_v1`. The cut only adds its own blocks: the expert
  GEMMs (fused with their weight reads or not), `EmbeddingRead_v1` at V = 256 and the router's slots form.
- **Item 2 (vllm-v2names):** read. It matches my table, and nothing in #8359 depends on T2.

