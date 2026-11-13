# CIR-274 4b (DSAMOE): the joint census at 78 layers

**The run.** Run `r20261113-025656-e215` on vy-nebius-cpu-5, rc 0, `--mem-gb 32`, `--timeout 10800`. Its source is the
families head `b5c74a06836824b7af153d7591ad95a574d5fb85`. It ran two scripts in order (`run78.sh`):
- `p3_census.py`, DENSE's, copied unchanged;
- `p3_keycost.py`, mine, which times the window's phases and measures the key wires.

Both run the same window: rank 0, 43 prompt ids, 2 outputs, 78 layers, families frame + dense + sparse (`sets`
`frame,dense,sparse`, `cols` 0). Both give the same Program digest and the same window digest.

**Refusals: none.** `NW.node_window` calls `NodeWindow.circuit()` (`node_window.py:446-465`), which runs:
1. `_check_rows`;
2. `_Builder.settle` with `provable=True`, which runs `IF.check_proof_block`;
3. `IF.Circuit`'s `check_wires` and `check_inputs` (`interface.py:1317-1318` at the families head).

All of them passed at 78 layers. The census JSON has no `window_error`.

## Digests

- **Program:** `3c99109308321b3b3e0e1b0f493ac52abfef2f4f90b4f0de2bbd1f66cc44215838fef98134749c009f487785415d9168329db7e9d997f644082fc6316b9c495f`,
  6,979 nodes.
- **Window:** SHA-512 of `IF.circuit_bytes(circuit.description())`:
  `f4c1c567288f82ad92c0a09f1cdbe1dda8ad2263a616ca9c3aed99b7819b93970fc7130da2807e3c239c279c8addeb6a769934d9a0dc8ee7215ac5262f0f649c`.

## In total, beside the frame alone

REQBLOCK's figures are the frame alone at 78 layers; mine are the families head with every family's lowering.

| | families head, frame + dense + sparse | REQBLOCK, frame only |
|---|---|---|
| units | 123,537,384 | 60,701,290 |
| templates | 105 | 23 |
| wire patterns (after `coalesce`) | 945,322 (1,099,262 before) | 1,747 |
| window wall time | 594 s to build (lower 564.4 s + `circuit()` 29.6 s); 1,796.2 s with the census's own accounting | 182.8 s |
| window peak (tracemalloc) | 10.879 GiB after `circuit()`; 11.638 GiB at the census's end | 1.406 GiB |
| description | 308,737,696 bytes | |

- **The Program:** 10.5 s, peak 0.522 GiB. A second build of the same digest in the keycost script took 12.7 s, peak
  0.521 GiB.
- **Process max RSS (`/usr/bin/time -v`):** 12,203,092 kB for the census, which took 30:49 wall. 12,326,036 kB for
  keycost, which took 14:48 and includes its `tracemalloc` copies of the wires.
- **No 48 GB stop was needed.** The run's cgroup peaked at 11.8 GiB.

**`circuit()`'s phases**, measured by keycost: `settle` 6.34 s, `check_proof_block` 0.01 s, `coalesce` 9.21 s, `ordered`
3.04 s, `check_wires` 6.50 s, `check_inputs` 3.70 s.

**The description's bytes:**
- wires 218,696,597;
- registered 83,341,168, which breaks down as roots 41,691,254, openings 24,134,236, tables 11,974,729, params 5,008,401
  and reads 532,541;
- order 6,682,353;
- templates 10,105 and cuts 7,481.

The census's own accounting (`NodeWindow.census`, plus the script's `pairs`, `rows_a_unit` and `free_by_port`) costs
about 1,200 s over the build, which is why its window time is three times keycost's.

## By family

"wires" are wire patterns before `coalesce`, by the reader's family (`NodeWindow.census`). My families are in bold.

| family | units | templates | wires |
|---|---|---|---|
| AllGather | 16 | 1 | 2 |
| AllReduceMnnvl | 32 | 1 | 0 |
| AllReduceRmsNormMnnvl | 10,108 | 4 | 778 |
| **BatchedGemm** | 21,086,208 | 2 | 312 |
| **CublasSplitKGemmF32Out** | 825,600 | 1 | 75 |
| DeepGemmFp8Linear | 58,948,098 | 8 | 1,170 |
| **DsaLatentRow** | 3,432 | 1 | 156 |
| DsaRMSNorm | 30,888 | 2 | 468 |
| FlashInferFp8Linear | 1,370,886 | 8 | 1,170 |
| **Fp8GroupQuantUe8m0** | 29,700 | 2 | 300 |
| Gemm | 38,720 | 1 | 2 |
| **LlBf16DotprodGemm** | 19,200 | 1 | 75 |
| **MoeExpertFp8Gemm** | 27,139,200 | 4 | 619,800 |
| **MoeExpertFp8GemmW** | 11,404,800 | 4 | 315,000 |
| **MoeRouterSigmoidBias** | 3,300 | 1 | 150 |
| MoeSum | 105,600 | 1 | 1,200 |
| RMSNormCudaVector8 | 1,452 | 2 | 4 |
| RMSNormFusedCudaVector8 | 219,946 | 3 | 464 |
| ResidualAdd | 105,600 | 1 | 300 |
| RoPEHeadGptJ | 27,456 | 1 | 156 |
| SiluMul | 29,832 | 2 | 306 |
| **SparseMlaSplit** | 54,912 | 45 | 17,238 |
| gather (the frame's) | 2,082,398 | 15 | 140,136 |

- **Templates.** The table's templates sum to 111 against 105 in the Circuit: a template two families emit counts once
  for each.
- **Wire patterns after `coalesce`, by the reader's family** (keycost): MoeExpertFp8Gemm 619,208, MoeExpertFp8GemmW
  314,408, SparseMlaSplit 6,887, gather 2,577, DeepGemmFp8Linear 862, FlashInferFp8Linear 787, SiluMul 302, and every
  other family under 160.

My templates, with their units:
- **BatchedGemm:** `BatchedGemm_v1{B=1,K=192,N=1}` 14,057,472 and `{B=1,K=512,N=1}` 7,028,736.
- **MoeExpertFp8Gemm:**
  - `MoeExpertFp8BlockGemm_v3{K=6144,N=1}`: 13,209,600 with SWAP=true and 307,200 with SWAP=false;
  - reads `EmbeddingRead_v1{V=256,H=3072}` 13,516,800 and `{V=256,H=96}` 105,600.
- **MoeExpertFp8GemmW:**
  - `MoeExpertFp8BlockGemmW_v3{K=256,N=32}`: 4,953,600 with SWAP=true and 115,200 with SWAP=false;
  - reads `{V=256,H=4096}` 5,068,800 and `{V=256,H=4}` 1,267,200.
- **The rest:** `CublasSplitKGemmF32Out_v1{K=6144,N=1,SPLIT_K=4}` 825,600; `LlBf16DotprodGemm_v1{N=1,K=6144}` 19,200;
  `Fp8GroupQuantUe8m0_v1{K=256}` 26,400 and `{K=6144}` 3,300; `MoeRouterSigmoidBiasSlots_v1` 3,300;
  `DsaLatentRow_v1` 3,432; SparseMlaSplit's 45 templates 54,912.

## Unlowered, by family (nodes)

AllReduceMultimem 157, EmbeddingShard 2, StopCheck 1, TokenSelect 2: 162 of 6,979 nodes. None is my lane's.

## Free, by family

By the family the rows are of (`NodeWindow.census`):

| source | unit ports | words | ports |
|---|---|---|---|
| weights | 21,086,208 | 6,297,747,456 | 312 |
| AllReduceMultimem | 222,784 | 82,962,432 | 315 |
| peers | 35,182 | 7,023,296 | 1,113 |
| EmbeddingShard | 32 | 6,144 | 1 |

By the reader (from `free_list`):
- **weights:** all of it is `BatchedGemm_v1.w`, 21,086,208 unit ports on 312 ports. This is BatchedGemm's strided `w`
  (`reads.md` line 1; draft (a) in `needs-vllm.md`), in my half.
- **AllReduceMultimem**, read by the dense half's norms:
  - `RMSNormFusedApplyCudaVector8_v1.x` 213,280 and `.res` 1,376;
  - `RMSNormFusedSumSqCudaVector8_v1.x` 6,665 and `.res` 43;
  - `RMSNormApplyCudaVector8_v1.x` 1,376;
  - `RMSNormSumSqCudaVector8_v1.x` 43;
  - `RMSNormFusedCudaVector8_v3.x` 1.
- **peers:** `AllReduceResidualMnnvl_v1.parts` 34,944, `AllReduceMnnvl_v1.parts` 224, `AllGather_v1.parts` 14.
- **EmbeddingShard:** `AllReduceMnnvl_v1.parts` 32.

No other port of my templates is free.

## Families at 78 layers against the 4-layer Program

No family appears at 78 layers that the 4-layer Program lacks, and none is lost. Both Programs have the same 26 node
families. The 4-layer Program is run `r20261113-025444-67c6`, same source, 319 nodes, digest `7f36f17ae005…`. Node
counts, 4 layers → 78 layers:

| family | 4 layers | 78 layers |
|---|---|---|
| AllGather | 2 | 2 |
| AllReduceMnnvl | 1 | 1 |
| AllReduceMultimem | 9 | 157 |
| AllReduceRmsNormMnnvl | 8 | 156 |
| BatchedGemm | 16 | 312 |
| CublasSplitKGemmF32Out | 1 | 75 |
| DeepGemmFp8Linear | 20 | 390 |
| DsaLatentRow | 8 | 156 |
| DsaRMSNorm | 8 | 156 |
| EmbeddingShard | 2 | 2 |
| FlashInferFp8Linear | 20 | 390 |
| Fp8GroupQuantUe8m0 | 4 | 300 |
| Gemm | 2 | 2 |
| LlBf16DotprodGemm | 1 | 75 |
| MoeExpertFp8Gemm | 2 | 150 |
| MoeExpertFp8GemmW | 2 | 150 |
| MoeRouterSigmoidBias | 2 | 150 |
| MoeSum | 2 | 150 |
| RMSNormCudaVector8 | 2 | 2 |
| RMSNormFusedCudaVector8 | 8 | 156 |
| ResidualAdd | 2 | 150 |
| RoPEHeadGptJ | 8 | 156 |
| SiluMul | 10 | 306 |
| SparseMlaSplit | 176 | 3,432 |
| StopCheck | 1 | 1 |
| TokenSelect | 2 | 2 |

The MoE families scale by 75, the MoE layers: the 4-layer Program has one MoE layer and three dense ones.

## The experts' key wires, measured

Measured by keycost at 78 layers. The 4-layer figures are from run `r20261113-025444-67c6`, same source, for scale.

| | 78 layers | 4 layers |
|---|---|---|
| wire patterns, all | 945,322 | 13,238 |
| key wires (reader `EmbeddingRead_v1`, port `tok`) | **907,200** (96.0 %) | 12,096 |
| w2's `g` wires (reader `MoeExpertFp8BlockGemmW_v3`, port `g`) | 26,400 | 352 |
| description bytes, all | 308,737,696 | 4,417,654 |
| bytes of the key wires | **207,920,793** (67.3 %) | 2,721,147 |
| bytes of the `g` wires | 7,878,359 (2.6 %) | |
| Python memory of the key wires (`tracemalloc` over copies) | 225,142,120 B, of 235,763,488 B for all wires | 2,915,112 B, of 3,463,464 B |
| `check_wires`, all wires / without the key and `g` wires | **6.76 s / 0.67 s** | 0.09 s / 0.02 s |
| `ordered`, all / without | 3.08 s / 0.74 s | 0.05 s / 0.02 s |
| `coalesce` (all patterns; nearly all are key wires) | 9.21 s | 0.10 s |
| table declarations: tables / roots / openings | 56,700 / 226,800 / 113,400 (77.8 MB of the registered bytes) | 756 / 3,024 / 1,512 |

- **The count.** 907,200 = 75 MoE layers × 12,096 = 75 × B·TOPK summed over the read templates and both steps' nodes.
  `coalesce` merges none of them: 907,200 before and after.
- **Build time.** Of `circuit()`'s 29.6 s, the key and `g` wires take about 17.5 s: `check_wires` +6.1 s, `ordered`
  +2.3 s, and most of `coalesce`'s 9.2 s. Their share of the lowering's 564 s was not separated.
- **Memory.** About 0.21 GiB of Python objects, 2 % of the 10.9 GiB peak. The peak is the units and feeds.
- **Circuit bytes.** 208 MB, two thirds of the description.
- **The tables' declarations are next.** Three entries per (block, read template, node), 77.8 MB.

What a `Piece` with two member digits would take is in `reads.md`. It removes only `g` wires: 26,400 → 1,200 at 78
layers, so 945,322 → 920,122 patterns (2.7 % fewer, about 7.5 MB). The key wires stay at 907,200.
