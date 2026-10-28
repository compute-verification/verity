# The descriptor v1 evaluator in Lean

`verity.ml.circuits.evaluation.reference.evaluate` on a Program, as executable Lean: the Lake package `verity_evaluate`
(lib `Evaluate`, exe `verity-evaluate`). It decodes a descriptor v1 Program ([format](../../../ml/circuits/PROTOCOL.md)),
binds the run inputs, evaluates every gate in canonical order and returns the root's outputs, or refuses. The Program is read
through C-Flock's verifier (`Flock.Program`, required by path from `verity/core/service/flock`), which evaluates no primitive;
this package adds the primitives. It is code, so it requires nothing of `Security/`.

- `Evaluate/Prims.lean`: the primitives implemented, each with the signature the registry holds for it. E0 is the Boolean
  profile (§9: `And<n>_v1`, `Xor<n>_v1`, `Not_v1`), `Input<w>_v1` and `RegisteredInput<w>_v1`; E1 the constant family
  `Const<w>[0x<hex>]_v1` and the scalar primitives; E2 the FP32 and BF16 arithmetic and the casts that read no table; E3 the
  tensor-core dot steps; E4 the rest of the registry's 172 gate primitives and the lazy families `TableRead{IB}x{EW}_v1`,
  `GatherBf16x{V}_v1` and `GatherF32x{V}_v1`. Any other primitive (the lifting families, `BitAtx`, `TopPMaskWordx`, ...)
  is refused by name.
- `Evaluate/Scalar.lean`: the scalar primitives (`verity.ml.circuits.ops.scalar`, and the vLLM integration's `I32Sub`,
  `I64ToI32`, `U32And`, `U32Xor`, `BitXor`, `SelectBit`), transcribed branch for branch from the Python. Each names the
  definition of `Security/Definitions/Core/Scalar.lean` it must equal, or that Security has none yet; the equalities are
  proved in `Security/Proofs`, as the certifier's are, since code never requires `Security/`.
- `Evaluate/Fp32.lean`, `Evaluate/Casts.lean`: the FP32 and BF16 arithmetic (`verity.ml.circuits.ops.fp32`,
  `mufu.DivFullScaleA_v1`, and the vLLM integration's `F32Sub`, `F32Sqrt`, `F32MaxRef`, `F32GtStrict`, `F32FmaRm`,
  `F32CeilFtz`, `I32ToF32` and `DivFullScaleA_v2`) and the casts of `verity.ml.circuits.ops.prims`: copies of
  `Security/Definitions/Core/{Fp32,Prims}.lean` where Security defines the primitive, transcriptions of the Python where it
  does not, each naming which. The `_v1` add, sub, mul and div return numpy's NaN on x86-64 (`Evaluate.Fp32.hostNan`),
  which Security does not state.
- `Evaluate/TC/`: the seven tensor-core dot steps of `verity.ml.circuits.ops.prims` (`Dots.lean`: Ampere's and Hopper's BF16,
  Hopper's and sm_120's E4M3, sm_120's E5M2, NVFP4 and MXFP4), on copies of Security's step semantics: `Spec.lean` of
  `Security/Definitions/Core/TC/{Pipeline,Spec}.lean` (`tc_dot_total`), `Fp8Atom.lean` of `Pouw/Fp8Atom/{E4M3,Atom}.lean` and
  `Pouw/PearlC/Sm120.lean` (`tc_dot_total_e4m3`), `Fp4.lean` of `Pouw/PearlC/Fp4.lean` (`BlockScaledAlignAdd`). The E5M2 step
  and the FP4 primitives' rules outside the block-scaled model (NaN scales, non-finite accumulators, MXFP4's overflow to
  `±inf`) are transcribed from the Python; Security has none yet.
- `Evaluate/Words.lean`: PoUW's `Pouw*` word primitives and the three `Ncp*` (six copied from
  `Security/Definitions/Pouw/PearlC/RowPrims.lean`, the rest transcribed). `Evaluate/Libdevice.lean`: the sampler's
  `NvLogf`, Gumbel and `TopPNumkeep`, libdevice's FTZ steps as the host's doubles compute them, and Philox.
  `Evaluate/Mufu.lean` (with `MufuFits.lean`): the eight MUFU primitives, each a bit rule around its table held as the
  unit's interpolant. `Evaluate/Decimal.lean`: `F32ExpRn`, `F32ErfRn` and `F32TanhRn`, decimal arithmetic at 100 digits half
  to even. `Evaluate/Act.lean` (with `GeluErfTable.lean`, `act.py`'s stored `GeluErfBf16` region): SiLU, RoPE's outputs,
  tanh, the GELUs, the registered tables `gelu_tanh_bf16` and `tanh_rn`, and the lazy families. SiLU and tanh call the host
  C library's `exp` and `tanh` on doubles, as the Python's `math` does, so their words are the Python's on the same host and
  may differ on another. Security defines none of these but six of `Words`.
- `Evaluate/Walk.lean`: the gate walk (`Evaluate.evaluate`).
- `Evaluate/Serve.lean`: `Evaluate.evaluateFn`, which `verity-evaluate` serves through the difftest harness's line protocol
  (`Difftest.serve`, [harness](../../difftest/README.md)): `{"fn": "evaluate", "input": {"program": <descriptor>, "inputs":
  [arg, ...]}}` gives `{"output": [word, ...]}` or `{"reject": "<reason>"}`, with one argument per parameter of the program
  function: its words in leaf order, or `{"table": "gelu_tanh_bf16" | "tanh_rn"}` for a registered table, which the evaluator
  computes where a gate reads it (`tanh_rn`'s 2^28 words never enter the request). `library_table` and `registered_table`
  (`{"name": ...}`) give a MUFU table's SHA-512 and a registered table's SHA-256 (over its entries as little-endian u32), which
  the suite holds to the library's and to C-Flock's pins.

~~~sh
cd verity/core/primitives/evaluate && lake build      # .lake/build/bin/verity-evaluate
python tools/verity/lean/check.py verity/core/primitives/evaluate   # the Lean check, from the repository root
uv run tools/verity/check/suites.py verity-core-evaluate           # the difftest against the Python
~~~

`tests/test_difftest.py` holds it to the Python reference through `verity_difftest`, output for output and refusal for
refusal: seeded random Boolean composites over every node form, sampled by a Hypothesis strategy with their inputs and with
mutations both sides refuse; each scalar primitive on the whole grid of its operands' edge words, on sampled words, and on
words out of range; each FP32 and BF16 primitive the same way, its grid of FP edge words (signed zeros, subnormals, the
extreme normals, ties and overflows on rounding, the infinities and every NaN class) batched in one Program per row,
`Bf16ToF32` and `E4m3ToF32` on every word, and the GPU captures in `verity/ml/circuits/ops/tests/fixtures` (the FP32
probe of `gpu-nan-2026-09-28`, and the fetched `cvt.rn.bf16.f32` campaign of `tc-total-2026-09-07`); each tensor-core dot
step batched on every pair of its format's edge codes under every FP32 edge accumulator, on pairs of specials across its
groups, on every pair of edge scale bytes, on 3118 accumulators over structured operands, on sampled words and on words out of
range, its traced Boolean Definition evaluated gate by gate against it, and the captures: the RTX 4090 golden records, the
A100 and RTX 4090 special-value campaign, the H100 HGMMA and wgmma records, the RTX PRO 6000's E5M2 and MXFP4 tiles, and
autoproof's RTX 5090 NVFP4 vectors (the large ones fetched, `research data fetch-fixtures`); every constant of the
registry and the constant ids the Python reads; E4's primitives each on the grid of their operands' edge words (every word
for a one-operand bf16 primitive, and a sweep of every binade's edges for an FP32 one), on sampled words and on words out of
range, and on their captures (the L40S SiLU and RoPE slices, the H100 and L40S log probe, test_mufu's fingerprints and the
RTX 4090 tanh sample); the MUFU tables against the library's SHA-512 and the registered tables against C-Flock's SHA-256;
`TanhF32Rn_v2` and `GeluTanhMulBf16_v3` reading a table named in the request; the lazy families on their index edges;
a registered input bound like a run input; and explicit examples: the pinned
2-bit adder of `boolean_vectors.json`, Boolean Definitions traced from their Python definitions (up to the Ampere BF16 tensor-core
step), and malformed programs and inputs.
