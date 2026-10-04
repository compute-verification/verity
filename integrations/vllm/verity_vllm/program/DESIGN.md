# Circuits design ledger

This ledger covers the area's circuits: core's Boolean word library (`packages/verity/src/verity/ml/boolean/`), the Definitions in
core's `verity.ml` and in vLLM's `program/registry/`, and circuit-check (`tools/circuit_check/`). It has one row per Definition family
(a name, its `_vN` stripped) that has more than one registered version, then one per quarantine module, then other candidates that
were tried. Per-version conformance comes from `uv run verity-vllm conformance` (`registry/conformance.py`), not from here. That record
covers the 110 Definitions reachable from `b1` and `prims`: 94 are exact-model-tested, 9 model-untested, 6 not yet established and
1 superseded.

A Definition version is *current* or *superseded*. Current means a served Build binds it, the served vocabulary `b1-eager-v3`
(`program/frontend/rules/vocab.py`) names it, or it is the newest version of its kind. A Boolean version (one whose `word` names a word
Definition) has the status of its word. Superseded means no served Build binds it and a later version took its place. A superseded
version stays registered only so recorded Programs decode and replay. `vocabulary_for` still offers the older vocabularies
(`b1-eager`, `b1-eager-v2`), so nothing refuses a new Program that binds a superseded version yet. *(served)* marks a version that
`predict.build_program` binds for a step Program of a fixture model (`tests/predict/fixtures`) at one of four targets: A100, RTX 4090,
H100 or RTX PRO 6000, dense, TP2 or stochastic. No fixture is an FP8 checkpoint, so no FP8 row is marked. *(bits of vN)* marks the
Boolean version of vN. Evidence gives the commit that introduced each version: for a v1 carried over from veritor, that is the
migration commit 061b6561d, 3f50ae9ca or 4ea072f3e.

The families come from importing every registry module (`registry.catalog.load()`, every `verity.ml` module and every quarantine
module: 643 Definitions) and grouping ids by name.

## Definitions

| candidate | status | reason | evidence |
|---|---|---|---|
| `AddScalarBf16` | current: v1 (served), v2 (bits), v3 | v1 `aten.add(w.float(), C)` (Gemma-2's 1 + w); v2 v1 on bits with C folded; v3 v1 with f32(C) an input | v1 061b6561d; v2 a38c5b2ee; v3 72c7482d4 |
| `AddScalarF32` | current: v1 (served), v2 (bits), v3 | v1 `aten.add(x, C)`; v2 v1 on bits with C folded; v3 v1 with f32(C) an input | v1 061b6561d; v2 a38c5b2ee; v3 72c7482d4 |
| `AddWidenedBf16` | current: v1 (served), v2 (bits) | v2 is v1 on bits, the version the Boolean Program binds | v1 061b6561d; v2 a38c5b2ee |
| `AmpereBF16TcDot16` | current: v2 (served), v3 (bits of v2); retired: v1 (no longer registered) | core's k16 step; v2 is the same function as the retired v1 (`registry.prims`), v3 v2 on bits | v2 3f50ae9ca; v3 2b3670f51; v1 retired 74b7f51f1, 996f14e13 |
| `ArgmaxStep` | current: v1 (served), v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 061b6561d; v2 4534a4ee9 |
| `Attention` | current: v2, v3 (served), v4 (served), v5 (served), v6 (bits of v3), v7 (bits of v2), v8 (bits of v5), v9 (bits of v4), v10, v11, v12, v13; superseded: v1 | v1 has fixed 64-key blocks and only `b1-eager` (the B0/B1 records) binds it. v3 makes kBlockN a declared static (BN = fa2_kblock_n(D)), the accepted `b1-eager-v3` attention. v2 adds the step DOT and epilogue INV (target-keyed `attention_dot`). v4 / v5 are FA3's / FA2's per-iteration Check_inf (MASKED_FROM; served on H100 / RTX PRO 6000). v6–v9 are v3, v2, v5, v4 on bits. v10–v13 are v3, v5, v2, v4 with scale_log2 an input | v1 061b6561d; v2 061b6561d; v3 061b6561d; v4 c19bae0d1; v5 4166bb423; v6 5d65b300d; v7 a43bc6d7a; v8 e272fb466; v9 b58b6906b; v10 c54b70b5f; v11 6486ef582; v12 72c7482d4; v13 72c7482d4 |
| `AttentionHead` | current: v2, v3 (served), v4 (served), v5 (served), v6 (bits of v3), v7 (bits of v2), v8 (bits of v5), v9 (bits of v4), v10, v11, v12, v13; superseded: v1 | one head of Attention, versioned in step with it (see Attention): v1 fixed blocks; v2 DOT/INV statics; v3 BN static; v4/v5 FA3/FA2 Check_inf; v6–v9 on bits; v10–v13 scale an input | v1 061b6561d; v2 061b6561d; v3 4ea072f3e; v4 c19bae0d1; v5 4166bb423; v6 5d65b300d; v7 a43bc6d7a; v8 e272fb466; v9 b58b6906b; v10 b0c29fa17; v11 6486ef582; v12 72c7482d4; v13 72c7482d4 |
| `AttentionHeadSoftcap` | current: v1 (served), v2 (served), v3 (bits of v2), v4, v5 | one head of AttentionSoftcap, versioned in step with it | v1 061b6561d; v2 f23660d1b; v3 ca85ccb57; v4 72c7482d4; v5 72c7482d4 |
| `AttentionSoftcap` | current: v1 (served), v2 (served), v3 (bits of v2), v4, v5 | Gemma-2's soft-capped FA2 attention: v1 over Attention_v3's body (served on A100 / RTX 4090); v2 over FA2's per-iteration Check_inf (Attention_v5; served on RTX PRO 6000); v3 is v2 on bits; v4 / v5 are v1 / v2 with the two softcap words inputs. No chain for Hopper (FA3) is registered: the predictor refuses Gemma-2 on H100 | v1 061b6561d; v2 f23660d1b; v3 ca85ccb57; v4 72c7482d4; v5 72c7482d4 |
| `AttnBlock` | current: v2, v3 (served), v4 (served), v5 (served), v6 (bits of v3), v7 (bits of v2), v8 (bits of v5), v9 (bits of v4), v10, v11, v12, v13; superseded: v1 | one key block of Attention, versioned in step with it (see Attention) | v1 061b6561d; v2 061b6561d; v3 4ea072f3e; v4 c19bae0d1; v5 4166bb423; v6 5d65b300d; v7 a43bc6d7a; v8 e272fb466; v9 b58b6906b; v10 b0c29fa17; v11 6486ef582; v12 72c7482d4; v13 72c7482d4 |
| `AttnBlockSoftcap` | current: v1 (served), v2 (served), v3 (bits of v2), v4, v5 | one key block of AttentionSoftcap, versioned in step with it | v1 061b6561d; v2 f23660d1b; v3 ca85ccb57; v4 72c7482d4; v5 72c7482d4 |
| `Bf16Add` | current: v1 (served), v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 061b6561d; v2 5f25de809 |
| `Bf16AddF2fp` | current: v1 (served), v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 671d006f3; v2 93ff922af |
| `Bf16DivScalar` | current: v1 (served), v2 (bits), v3 | v1 `aten.div(x, C)` as f32(x) * RN32(1/C); v2 v1 on bits with C folded; v3 v1 with RN32(1/C) an input | v1 061b6561d; v2 9b9103b70; v3 72c7482d4 |
| `Bf16GtStrict` | current: v1 (served), v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 061b6561d; v2 140fe86f9 |
| `Bf16MulBf16` | current: v1, v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 061b6561d; v2 5f25de809 |
| `Bf16MulF32` | current: v1, v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 a88a70e21; v2 5f25de809 |
| `Bf16MulScalar` | current: v1 (served), v2 (bits), v3 | v1 `aten.mul(x, C)`; v2 v1 on bits with C folded; v3 v1 with f32(C) an input | v1 061b6561d; v2 9b9103b70; v3 72c7482d4 |
| `Bf16MulScalarTensor` | current: v1 (served), v2 (bits) | v2 is v1 on bits, the version the Boolean Program binds | v1 061b6561d; v2 a38c5b2ee |
| `Bf16Tanh` | current: v1 (served), v2 (bits) | v2 is v1 on bits, the version the Boolean Program binds | v1 061b6561d; v2 9b9103b70 |
| `Bf16ToF32` | current: v1 (served), v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 b654619dc; v2 947f374ce |
| `BitAnd` | current: v1 (served), v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 671d006f3; v2 a5069acd6 |
| `BlackwellE4m3QmmaDot32` | current: v1, v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 b84e85efe; v2 b9ec4b1ba; art:b3074aba4c9ddeef7f87454bfc23e73a5a3c5f73f1676260545115fe8589cc89 |
| `BlackwellE5m2QmmaDot32` | current: v1, v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 f8c22821f; v2 33e10236c; art:fc4bb8915dd22f22b3d0260a00a6d694750322852c5163ddb80a05948370a94f |
| `BlackwellMxf4OmmaDot64` | current: v1, v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 0f05a4dd6; v2 5b0e6545f; art:d8d4d1b602c9fde61df218c315430535fe5181882ab6b26c2b95dd81dc1b5d92 |
| `BlackwellNvf4OmmaDot64` | current: v1, v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 bef10831b; v2 5b0e6545f; art:3e02931df28cccd398de4219f9d977371a58cdc6e8addfb7ff0a6269e5481b0e |
| `DivFullRcp` | current: v1 (served), v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 061b6561d; v2 1a3fffeab |
| `DivFullScaleA` | current: v1 (served), v2, v3 (bits of v1) | v2 is v1 with every NaN result the GPU's 0x7FFFFFFF (one of the twelve `_v2` primitives); v3 v1 on bits | v1 061b6561d; v2 705969518; v3 1a3fffeab |
| `DotBf16` | current: v1 (served), v2 (served), v3 (bits) | the k16 dot of Gemm: v1 fixed step, v2 DOT static, v3 v2 on bits | v1 4ea072f3e; v2 b654619dc; v3 ac15d5abf |
| `DotE4m3` | current: v1, v2, v3 (bits of v1), v4 (bits of v2) | v1 over Hopper's E4M3 k32 step; v2 over the step DOT (sm_120's no-swap GEMM); v3 / v4 are v1 / v2 on bits | v1 c308f2420; v2 a3e89091c; v3 a6fd62c89; v4 b9ec4b1ba |
| `E4m3ToF32` | current: v1, v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 bef10831b; v2 a5069acd6 |
| `Embedding` | current: v1 (served), v2 (bits of v1) | row gather; v2 v1 on bits (one GatherBf16x{V}_v2 per hidden coordinate) | v1 671d006f3; v2 9ee9384ef |
| `F2fpBf16` | current: v1 (served), v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 b654619dc; v2 a6fd62c89 |
| `F32Add` | current: v1 (served), v2, v3 (bits of v1) | v2 is v1 with every NaN result the GPU's 0x7FFFFFFF (probe art:e5dd6ce9), beside an unchanged v1, which the served Programs bind; v3 is v1 on bits | v1 061b6561d; v2 e150bee06; v3 93ff922af |
| `F32AddFtz` | current: v1 (served), v2, v3 (bits of v1) | v2 is v1 with every NaN result the GPU's 0x7FFFFFFF (probe art:e5dd6ce9), beside an unchanged v1, which the served Programs bind; v3 is v1 on bits | v1 671d006f3; v2 e150bee06; v3 ac15d5abf |
| `F32Div` | current: v1 (served), v2, v3 (bits of v1) | v2 is v1 with every NaN result the GPU's 0x7FFFFFFF (probe art:e5dd6ce9), beside an unchanged v1, which the served Programs bind; v3 is v1 on bits | v1 061b6561d; v2 e150bee06; v3 93ff922af |
| `F32Fabs` | current: v1 (served), v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 4ad624e04; v2 1256f06fc |
| `F32Fma` | current: v1 (served), v2, v3 (bits of v1) | v2 is v1 with every NaN result the GPU's 0x7FFFFFFF (probe art:e5dd6ce9), beside an unchanged v1, which the served Programs bind; v3 is v1 on bits | v1 061b6561d; v2 e150bee06; v3 93ff922af |
| `F32FmaFtz` | current: v1, v2, v3 (bits of v1) | v2 is v1 with every NaN result the GPU's 0x7FFFFFFF (probe art:e5dd6ce9), beside an unchanged v1, which the served Programs bind; v3 is v1 on bits | v1 671d006f3; v2 e150bee06; v3 ac15d5abf |
| `F32FmaRm` | current: v1 (served), v2 | v2 is v1 with every NaN result the GPU's 0x7FFFFFFF (one of the twelve `_v2` primitives); v1 is the served one | v1 705969518; v2 705969518 |
| `F32FmaSubFtz` | current: v1 (served), v2, v3 (bits of v1) | v2 is v1 with every NaN result the GPU's 0x7FFFFFFF (probe art:e5dd6ce9), beside an unchanged v1, which the served Programs bind; v3 is v1 on bits | v1 671d006f3; v2 e150bee06; v3 ac15d5abf |
| `F32Fmaxf` | current: v1 (served), v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 5a6b7a5d6; v2 1256f06fc |
| `F32Fminf` | current: v1, v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 c1ace2b08; v2 1256f06fc |
| `F32Max` | current: v1, v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 671d006f3; v2 ac15d5abf |
| `F32Mul` | current: v1 (served), v2, v3 (bits of v1) | v2 is v1 with every NaN result the GPU's 0x7FFFFFFF (probe art:e5dd6ce9), beside an unchanged v1, which the served Programs bind; v3 is v1 on bits | v1 061b6561d; v2 e150bee06; v3 93ff922af |
| `F32MulFtz` | current: v1 (served), v2, v3 (bits of v1) | v2 is v1 with every NaN result the GPU's 0x7FFFFFFF (probe art:e5dd6ce9), beside an unchanged v1, which the served Programs bind; v3 is v1 on bits | v1 671d006f3; v2 e150bee06; v3 ac15d5abf |
| `F32Sub` | current: v1 (served), v2 | v1 the reference vocabulary's f32 subtract (`ref_prims`, reference-defined) and the MoE router's; v2 the GPU-NaN form (one of the twelve `_v2` primitives) | v1 061b6561d; v2 705969518 |
| `F32SubFtz` | current: v1, v2, v3 (bits of v1) | v2 is v1 with every NaN result the GPU's 0x7FFFFFFF (probe art:e5dd6ce9), beside an unchanged v1, which the served Programs bind; v3 is v1 on bits | v1 671d006f3; v2 e150bee06; v3 ac15d5abf |
| `F32ToE2m1Sat` | current: v1, v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 bef10831b; v2 a5069acd6 |
| `F32ToE4m3Sat` | current: v1, v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 4d053f01a; v2 1256f06fc |
| `Fa2InvSum` | current: v1 (served), v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 671d006f3; v2 2d453e7f5 |
| `Fa3InvSum` | current: v1 (served), v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 061b6561d; v2 919473266 |
| `Final` | current: v2; superseded: v1 | the final norm, LM head and token select inside Serve: v1 for Serve_v1, v2 (GEMM / NORM as statics) for Serve_v2 / v3 | v1 061b6561d; v2 061b6561d |
| `Fp8ActQuant` | current: v1, v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 a6fd62c89; v2 a6fd62c89 |
| `Fp8GroupQuant` | current: v1, v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 36e49e971; v2 a6fd62c89 |
| `Fp8GroupScale` | current: v1, v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 a6fd62c89; v2 a6fd62c89 |
| `Fp8RowAbsMax` | current: v1, v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 a6fd62c89; v2 a6fd62c89 |
| `Fp8RowScale` | current: v1, v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 a6fd62c89; v2 a6fd62c89 |
| `GeluTanhMul` | current: v1 (served), v2 (bits), v3 | v1 `_C.gelu_tanh_and_mul`; v2 v1 on bits; v3 v1 with the activation read from the registered `gelu_tanh_bf16` table (`gelu_tanh_mul_table`) | v1 061b6561d; v2 5110215c6; v3 5903f6bf8 |
| `GeluTanhMulBf16` | current: v1 (served), v2 (bits of v1), v3 | the element of GeluTanhMul: v2 v1 on bits; v3 reads the registered table | v1 061b6561d; v2 5110215c6; v3 5903f6bf8 |
| `Gemm` | current: v1 (served), v2 (served), v3 (bits) | v1 is the batch-invariant Triton matmul over AmpereBF16TcDot16 (accepted A100 / RTX 4090 records); v2 makes the k16 step DOT a static (`gemm_dot`, served on H100 / RTX PRO 6000); v3 is v2 on bits | v1 4ea072f3e; v2 b654619dc; v3 368980d6f |
| `GemmBias` | current: v1 (served), v2 (served) | the biased linear as one Definition: v1 Gemm_v1 then BiasAdd_v1 (A100); v2 with the target's step DOT (RTX PRO 6000, `gemm_bias_dot`) | v1 0011918fe; v2 3dbcc9040 |
| `GemmBiasCoordinate` | current: v1 (served), v2 (served) | one output word of GemmBias, versioned in step with it | v1 0011918fe; v2 3dbcc9040 |
| `GemmCoordinate` | current: v1 (served), v2 (served), v3 (bits) | one output word of Gemm, versioned in step with it (v1 fixed step, v2 DOT static, v3 v2 on bits) | v1 4ea072f3e; v2 b654619dc; v3 2b3670f51 |
| `GemmCoordinateE4m3` | current: v1, v2 (bits) | v2 is v1 on bits, the version the Boolean Program binds | v1 21e333a9b; v2 86a09cb74 |
| `GemmCoordinateMxf4` | current: v1, v2 (bits) | v2 is v1 on bits, the version the Boolean Program binds | v1 21e333a9b; v2 5b0e6545f |
| `GemmCoordinateNvf4` | current: v1, v2 (bits) | v2 is v1 on bits, the version the Boolean Program binds | v1 21e333a9b; v2 5b0e6545f |
| `GuardNegInfZero` | current: v1 (served), v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 671d006f3; v2 ac15d5abf |
| `GumbelNoiseLane` | current: v1 (served), v2 | v2 is v1 through NvLogf_v2 (GPU NaN word); the same word on every lane, since the noise never reaches those results | v1 061b6561d; v2 ce9305f05; art:e5dd6ce9 |
| `GumbelSelectF32` | current: v1 (served), v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 061b6561d; v2 ef647aa9b |
| `GumbelStreamKey` | current: v1 (served), v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 061b6561d; v2 ef647aa9b |
| `GumbelTopPTokenSelect` | current: v1, v2 (served) | the seeded temperature + top-p select: v1 with the split count an operand (`token_select_stochastic`); v2 at a constant S over TopPMask_v2 (`token_select_stochastic_split`); both bound | v1 91ceb79ee; v2 883359da6 |
| `HopperBF16WgmmaDot16` | current: v1 (served), v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 b654619dc; v2 b58b6906b |
| `HopperE4m3QgmmaDot32` | current: v1, v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 4d053f01a; v2 a6fd62c89 |
| `I32Add` | current: v1 (served), v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 061b6561d; v2 93ff922af |
| `LayerNormAten` | current: v1, v2 | GPT-NeoX's nn.LayerNorm (torch's vectorized CUDA kernel), defined in quarantine lane cov-ln but bound by `b1-eager-v3` (`layer_norm_aten`); v2 v1 with eps an input. Difftest on the pod not recorded as done | v1 4963b1606; v2 72c7482d4 |
| `LayerNormRef` | current: v1, v2 | reference vocabulary: v1 `layer_norm`, v2 v1 with eps an input (`layer_norm_value`); both bound there | v1 061b6561d; v2 72c7482d4 |
| `LayerPost` | current: v2; superseded: v1 | a layer's back half inside Serve: v1 for Serve_v1, v2 (GEMM / NORM as statics) for Serve_v2 / v3 | v1 061b6561d; v2 061b6561d |
| `LayerPre` | current: v2; superseded: v1 | a layer's front half inside Serve: v1 for Serve_v1, v2 (the GEMM / NORM implementations as statics) for Serve_v2 / v3 | v1 061b6561d; v2 061b6561d |
| `MeanTriton` | current: v1 (served), v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 061b6561d; v2 c120cf0f6 |
| `MoeExpertCoordinate` | current: v1 (served), v2 (served) | one output word of MoeExpertGemm, versioned in step with it | v1 7604eb59c; v2 7604eb59c |
| `MoeExpertCoordinateW` | current: v1 (served), v2 (served) | one output word of MoeExpertGemmW, versioned in step with it | v1 7604eb59c; v2 7604eb59c |
| `MoeExpertFp8BlockCoordinate` | current: v1, v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 34019133a; v2 34019133a |
| `MoeExpertFp8BlockCoordinateW` | current: v1, v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 34019133a; v2 34019133a |
| `MoeExpertFp8BlockGemm` | current: v1, v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 cb92679b3; v2 34019133a |
| `MoeExpertFp8BlockGemmW` | current: v1, v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 cb92679b3; v2 34019133a |
| `MoeExpertGemm` | current: v1 (served), v2 (served) | fused_moe_kernel's expert GEMM: v1 the Ampere / Ada k16 step; v2 the target's step DOT (`moe_expert_gemm_dot`, sm_120) | v1 7604eb59c; v2 7604eb59c |
| `MoeExpertGemmW` | current: v1 (served), v2 (served) | as MoeExpertGemm, with the routed weight multiplied into the accumulator | v1 7604eb59c; v2 7604eb59c |
| `MoeRouterProbs` | current: v1 (served), v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 fd265d7cb; v2 5e4316b92 |
| `MoeRouterSigmoidBias` | current: v1, v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 39d70c18e; v2 06abdcabc; art:a827deda |
| `MoeRouterSigmoidScore` | current: v1, v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 1cf1cffae; v2 1cf1cffae |
| `MoeRouterTopKOrdered` | current: v1 (served), v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 3c8f5f53b; v2 5e4316b92 |
| `MoeSum` | current: v1 (served), v2 (bits) | `_moe_C.moe_sum` (ascending-k f32 sum of the TOPK expert rows, one bf16 RNE); v2 is v1 on bits | v1 061b6561d; v2 9eda26ad7 |
| `MoeSumCoordinate` | current: v1 (served), v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 061b6561d; v2 9eda26ad7 |
| `MufuEx2Ftz` | current: v1 (served), v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 671d006f3; v2 1a3fffeab |
| `MufuRcpFtz` | current: v1, v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 061b6561d; v2 1a3fffeab |
| `MufuSqrtFtz` | current: v1 (served), v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 061b6561d; v2 1a3fffeab |
| `MufuTanh` | current: v1 (served), v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 061b6561d; v2 531e21b59 |
| `MulVecF32` | current: v1 (served), v2 (bits) | v2 is v1 on bits, the version the Boolean Program binds | v1 061b6561d; v2 a38c5b2ee |
| `NarrowF32ToBf16` | current: v1 (served), v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 061b6561d; v2 c120cf0f6 |
| `NvExpf` | current: v1 (served), v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 061b6561d; v2 25b42ffcb |
| `NvLogf` | current: v1 (served), v2, v3 (bits of v2) | v2 is v1 with every NaN result the GPU's 0x7FFFFFFF, for the re-baseline; v1 unchanged; v3 is v2 on bits | v1 ce9305f05; v2 ce9305f05; v3 ef647aa9b; art:e5dd6ce9 |
| `QuantLinearFp8` | current: v1, v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 d14de9b62; v2 a6fd62c89 |
| `RMSNormFusedCuda` | current: v2 (served), v3 (bits of v2), v4; superseded: v1 | v1 models `fused_add_rms_norm_kernel<bf16, 8>`, which vLLM d9105ea80 does not launch under VLLM_BATCH_INVARIANT=1 (1 word of 218,112 differs in B1; 13 of 2,340 B0 rows), so v2 models the generic kernel and the served vocabulary binds it; v1 stays registered so the B0/B1 `Serve_v1` records decode (`conformance.DIVERGENCES`). v3 is v2 on bits; v4 is v2 with eps an input | v1 061b6561d; v2 061b6561d; v3 7e07d75f0; v4 c54b70b5f; fixtures/verity-ir/rmsnorm-fused-b0/sweep-2026-09-07.json |
| `RMSNormRef` | current: v1, v2 | reference vocabulary (`reference-ref-prims`): v1 `rmsnorm`, v2 v1 with eps an input (`rmsnorm_value`); both bound there | v1 061b6561d; v2 72c7482d4 |
| `RMSNormTriton` | current: v1 (served), v2 (bits of v1), v3 | vLLM's batch-invariant Triton `_rms_norm_kernel`: v2 v1 on bits; v3 v1 with eps an input (`rmsnorm_value`) | v1 061b6561d; v2 7e07d75f0; v3 c54b70b5f |
| `RMSNormVarEpsCuda` | current: v1, v2 (bits of v1) | the fused CUDA norm with eps a word operand (boolean_norms); v2 v1 on bits | v1 04ffd8cac; v2 04ffd8cac |
| `RMSNormVarEpsTriton` | current: v1, v2 (bits of v1) | the Triton norm with eps a word operand (boolean_norms); v2 v1 on bits | v1 04ffd8cac; v2 04ffd8cac |
| `RoPE` | current: v1 (served), v2 (bits of v1) | vLLM `_C.rotary_embedding` NeoX; v2 v1 on bits. The GPT-J (is_neox=False) rotation is a separate family, RoPEGptJ_v1 / RoPEHeadGptJ_v1 (`ref_prims`, model-untested), which no served Build binds | v1 061b6561d; v2 a7908fa1d |
| `RoPEHead` | current: v1 (served), v2 (bits of v1) | one head of RoPE; v2 v1 on bits | v1 061b6561d; v2 a7908fa1d |
| `RopeOut` | current: v1 (served), v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 061b6561d; v2 a7908fa1d |
| `RopeOutAdd` | current: v1 (served), v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 061b6561d; v2 a7908fa1d |
| `RsqrtApprox` | current: v1 (served), v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 061b6561d; v2 1a3fffeab |
| `RsqrtF32` | current: v1 (served), v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 061b6561d; v2 c120cf0f6 |
| `ScaleRowBf16` | current: v1 (served), v2 (bits) | v2 is v1 on bits, the version the Boolean Program binds | v1 061b6561d; v2 a38c5b2ee |
| `ScaleRowF32` | current: v1 (served), v2 (bits) | v2 is v1 on bits, the version the Boolean Program binds | v1 061b6561d; v2 a38c5b2ee |
| `ScaledMmFp8` | current: v1, v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 061b6561d; v2 a6fd62c89 |
| `ScaledMmFp8Block` | current: v1, v2 (bits of v1), v3, v4 (bits of v3) | blockwise `_C.cutlass_scaled_mm` (per-128 scales): v1 over Hopper's QGMMA step; v3 v1 over the step DOT for targets whose blockwise step is not Hopper's (`fp8.block_gemm` with FP8_DOT; at cc 12.0 not yet compared with hardware); v2 / v4 are v1 / v3 on bits | v1 df13126fa; v2 a6fd62c89; v3 64df47829; v4 64df47829 |
| `ScaledMmFp8BlockCoordinate` | current: v1, v2 (bits of v1), v3, v4 (bits of v3) | one output word of ScaledMmFp8Block, versioned in step with it | v1 64df47829; v2 a6fd62c89; v3 64df47829; v4 64df47829 |
| `ScaledMmFp8BlockCoordinateGivenScale` | current: v1, v2 (bits of v1), v3, v4 (bits of v3) | the coordinate with the tile scales read, versioned in step with ScaledMmFp8Block | v1 a763a41ab; v2 a6fd62c89; v3 64df47829; v4 64df47829 |
| `ScaledMmFp8BlockSharedScale` | current: v1, v2 (bits of v1), v3, v4 (bits of v3) | ScaledMmFp8Block with each weight block's scale products computed once (SCALE shared-per-block), versioned in step with it | v1 a763a41ab; v2 a6fd62c89; v3 64df47829; v4 64df47829 |
| `ScaledMmFp8BlockSharedScaleTile` | current: v1, v2 (bits of v1), v3, v4 (bits of v3) | one weight block of ScaledMmFp8BlockSharedScale, versioned in step with it | v1 df13126fa; v2 a6fd62c89; v3 64df47829; v4 64df47829 |
| `ScaledMmFp8Coordinate` | current: v1, v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 a3e89091c; v2 a6fd62c89 |
| `ScaledMmFp8NoSwap` | current: v1, v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 64df47829; v2 b9ec4b1ba |
| `ScaledMmFp8NoSwapCoordinate` | current: v1, v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 a3e89091c; v2 b9ec4b1ba |
| `SelectBf16` | current: v1 (served), v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 671d006f3; v2 140fe86f9 |
| `SelectI32` | current: v1 (served), v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 671d006f3; v2 140fe86f9 |
| `Serve` | current: v4; superseded: v1, v2, v3 | whole-model composites of recorded profiles: v1 the B0/B1 records (`b1-eager`), v2 the profile records as statics and v3 the padded circuit (`b1-eager-v2`, pad-serve@1), v4 v1 with the BN static. No served Build binds any of them (`b1-eager-v3` has no Serve kind; the Build derives `Derived[...]` Programs) | v1 3f50ae9ca; v2 061b6561d; v3 061b6561d; v4 061b6561d |
| `SiluMul` | current: v2 (served), v4 (bits); superseded: v1, v3 (bits) | v2 adds the sm_120 edge words of `_C.silu_and_mul` that v1 misses (expf overflow, up = -0, gate -inf / -0, NaN 0x7FFF; red-team r20260930-092038-7dbc); `b1-eager-v3` binds v2 and only the older vocabularies keep v1. v3 / v4 are v1 / v2 on bits | v1 061b6561d; v2 73532473e; v3 98bc36342; v4 db0519cd4; r20260930-092038-7dbc; r20261002-123618-cee3 |
| `SiluMulBf16` | current: v2 (served), v4 (bits of v2); superseded: v1, v3 (bits of v1) | the element of SiluMul: v2 adds the edge words v1 misses (see SiluMul); v3 / v4 are v1 / v2 on bits | v1 671d006f3; v2 73532473e; v3 db0519cd4; v4 db0519cd4; r20260930-092038-7dbc |
| `SiluMulFp8BlockQuant` | current: v1, v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 3a0ea7c11; v2 3a0ea7c11 |
| `SiluMulFp8BlockQuantGroup` | current: v1, v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 3a0ea7c11; v2 3a0ea7c11 |
| `SiluMulFp8BlockQuantResult` | current: v1, v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 3a0ea7c11; v2 3a0ea7c11 |
| `SquareBf16` | current: v1 (served), v2 (bits) | v2 is v1 on bits, the version the Boolean Program binds | v1 061b6561d; v2 a38c5b2ee |
| `SquareF32` | current: v1 (served), v2 (bits) | v2 is v1 on bits, the version the Boolean Program binds | v1 061b6561d; v2 a38c5b2ee |
| `TanhBf16` | current: v1, v2 (bits of v1) | v2 is v1 on bits, the version the Boolean Program binds | v1 9b9103b70; v2 9b9103b70 |
| `TanhF32Rn` | current: v1 (served), v2 | v1 is float64 tanh rounded once (a stand-in for libdevice tanhf); v2 reads \|x\| in [LO, HI) from the registered `tanh_rn` table, equal to v1 there | v1 061b6561d; v2 2979cbbdf |
| `TokenSelect` | current: v1 (served), v2 (bits of v1) | greedy argmax, lowest index on ties; v2 is v1 on bits (a tree of separable ArgmaxStep_v2 steps) | v1 061b6561d; v2 4534a4ee9 |
| `TopPMask` | current: v1, v2 (served) | v1 takes the split count as an operand (any \|live\|); v2 fixes it as the static S so the keep word is word gates a partition query can cut (`token_select_stochastic_split`); both bound by `b1-eager-v3` | v1 061b6561d; v2 883359da6 |
| `AllReduce` (with `AllReduce2`) | current: AllReduce2_v1 (world 2, served under TP2), AllReduce_v2 (WORLD > 2); retired: AllReduce_v1 (no longer registered) | v1 was the ascending chain; v2 is NCCL's intra-node tree chain in topology order, measured on 4x L40S TP4 (1,056 of 1,056 all-reduces exact; 32 of them equal the ascending chain). World 2 is one commutative add, `AllReduce2_v1` | AllReduce_v1 d676f1d6e; AllReduce_v2 b33386e02 |
| `F32ToBf16Rn` / `F2fpBf16` | current: F2fpBf16_v1 (served), F2fpBf16_v2 (bits of v1), F32ToBf16Rn_v1 | two names for f32 -> bf16 that differ only on NaN: `F2fpBf16_v1` gives 0x7FFF, the measured `cvt.rn.bf16.f32` word, and B0/B1 and every served Build bind it. `F32ToBf16Rn_v1` gives 0x7FC0 (software canonical, `veritor.tc-ampere-bf16@1`), stays registered unchanged, and only the reference vocabulary binds it (`f32_to_bf16`); recorded traces hold no NaN at a conversion output (`conformance.NAN_CANONICALISATION`) | F2fpBf16_v1 3f50ae9ca; F2fpBf16_v2 a6fd62c89 |

## Quarantine modules

`program/registry/quarantine/` holds provisional Definitions pending admission (its `__init__` docstring). They are to move to `experimental/` later.

| candidate | status | reason | evidence |
|---|---|---|---|
| `quarantine/act` | experimental (re-export shim) | SiluMul_v2 / SiluMulBf16_v2 were re-derived here, then promoted to `registry.act` with bodies and ids unchanged; the module re-exports them for its difftest adapter | f6e0a1cdf; promoted 73532473e |
| `quarantine/dense` | experimental (re-export shims) | Gemma-2's unfused operations (lane dense): 19 modules, each Definition promoted to `registry.dense` and bound by `b1-eager-v3` (R15 dense, F-dense-05); the modules re-export them for their difftest adapters | 061b6561d |
| `quarantine/ov_moe` | experimental (re-export shim) | lane ov-moe's MoE Definitions, promoted to `registry.moe` (R13 MOE-01) with ids unchanged; re-exported for the ov-moe difftest adapters and the `gen_ov_moe_patterns` eager profile | 061b6561d |
| `quarantine/ln` (GeluErf_v1, GeluErfBf16_v1) | experimental | GPT-NeoX's exact-erf GELU (lane cov-ln): the bf16 element from an exhaustive measured table (65,536 words, torch 2.13 on sm_89); the difftest on the pod is not recorded as done; no vocabulary binds it | 061b6561d |
| `quarantine/ln` (LayerNormAten_v1, _v2) | experimental (bound by `b1-eager-v3`) | torch's vectorized CUDA LayerNorm (lane cov-ln); still in quarantine, yet the served vocabulary binds it (`layer_norm_aten`, `layer_norm_aten_value`); the difftest on the pod is not recorded as done | 061b6561d; LayerNormAten_v2 72c7482d4 |
| `quarantine/collective` (AllReduceSumBf16_v1, AllReduceSumBf16Elem_v1) | experimental | lane tp2's NCCL bf16 sum, R = 2 only (R > 2 refused in favour of `b1_tp2.AllReduce_v2`); at R = 2 equal to `AllReduce2_v1` on non-NaN words (ov-tp2: 114 of 114 snapshotted all-reduces); the difftest is pending. The vocabulary binds `AllReduce2_v1` at world 2 | 061b6561d |
| `quarantine/ov_sampling` (GumbelArgmaxBf16_v1, GumbelArgmaxStep_v1, GumbelArgmaxTempBf16_v1, GumbelArgmaxTempStep_v1) | experimental | Gumbel-max sampling at temperature 1 or a static T (no top-k or top-p), seed and position as statics; read only by the `gen_ov_sampling_patterns` eager profile. The served vocabulary binds `GumbelTopPTokenSelect_v1` / `_v2` (registry.sampling) instead | 061b6561d |

## Other candidates

| candidate | status | reason | evidence |
|---|---|---|---|
| padded MoE: disabled slots as strict propagation of a bot input | retired | the v1 reading of the padded fused-MoE lift; replaced by A12, in which each of the E - TOPK unselected slots is a DISABLED computation (`a_e = 0` gives `enc(bot)` whatever the operands) inside `LMoeFfnL_v1` (`registry.moe_pad` docstring) | `registry/moe_pad.py` docstring (A12) |
